#!/usr/bin/env python3
"""List and time NVIDIA Inference Hub models, to pick the two this app names.

`config.nvissues.astra.yaml` deliberately gives no default for the answer and
expand model ids, because on the Hub an id is entitlement-dependent -- what a
key can call differs per key and changes over time -- and a wrong id fails as a
401 or 404 that the client catches and logs as a warning rather than raising.
The catalogue page is not the source of truth; /v1/models with the deployment's
own key is. This prints that list, and then times the candidates, because the
answer model sits on the critical path of a 5-10s budget and a large hosted
model that takes 20s to first token disqualifies itself regardless of quality.

Stdlib only, so it also runs on a laptop or in the pod with nothing installed.

    export IH_API_KEY=...            # inference.nvidia.com/key-management
    python scripts/ih_models.py                    # every id the key can call
    python scripts/ih_models.py qwen nemotron      # filtered
    python scripts/ih_models.py --chat <model-id>  # latency, warm and cold
    python scripts/ih_models.py --chat <id> --no-think   # thinking disabled

Reasoning tokens are reported separately because on the Hub's Qwen ids they
are most of the bill. These models return chain-of-thought in
`reasoning_content` and the answer in `content`, and the token budget covers
both, reasoning first -- so a budget that looks generous can be consumed
entirely by thinking and return a *successful* response with empty content.
That is not a hypothetical: it is what every one of these models did at 400
tokens against a realistic prompt.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("IH_ENDPOINT", "https://inference-api.nvidia.com/v1")

# Long enough to be representative: the real answer prompt carries twenty bug
# chunks, and time-to-last-token on a two-sentence prompt says nothing about
# what a 15k-token prompt will cost. Padded rather than realistic, since the
# question here is the model's speed, not the answer's quality.
PROMPT = (
    "You are triaging NVIDIA bug reports. Given the evidence below, answer in "
    "two sentences.\n\n"
    + ("Bug 5816744: GB200 nodes report gpu_nvlink_disconnected after the "
       "1.3.6 driver update; fabric manager fails to add GPU to SM during "
       "initialisation. Workaround under test is an NVSwitch firmware "
       "reflash.\n") * 120
    + "\nQuestion: what is the common failure mode across these reports?"
)


def _key() -> str:
    for name in ("IH_API_KEY", "NVDEV_API_KEY", "NVIDIA_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    sys.exit("Set IH_API_KEY (get one at inference.nvidia.com/key-management).")


def _post(path: str, payload: dict, key: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_models(key: str, filters: list[str]) -> int:
    req = urllib.request.Request(
        f"{BASE}/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            ids = sorted(m["id"] for m in json.load(r).get("data", []))
    except urllib.error.HTTPError as e:
        sys.exit(f"{e.code} {e.reason} from {BASE}/models -- key rejected?")

    shown = [i for i in ids
             if not filters or any(f.lower() in i.lower() for f in filters)]
    for i in shown:
        print(f"  {i}")
    print(f"\n  {len(shown)} of {len(ids)} ids"
          + (f" matching {filters}" if filters else "") + f" on {BASE}")
    return 0


def chat(key: str, model: str, max_tokens: int = 2000,
         think: bool = True) -> int:
    payload = {"model": model,
               "messages": [{"role": "user", "content": PROMPT}],
               "temperature": 0.0, "max_tokens": max_tokens}
    if not think:
        # vLLM/NIM's hook for Qwen's thinking switch. Not every id honours it,
        # which is why the reasoning-token column below is the check: a model
        # that ignored the flag still reports the thinking it did.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    print(f"  model  {model}\n  prompt {len(PROMPT):,} chars  "
          f"budget {max_tokens}  thinking {'on' if think else 'off'}",
          flush=True)
    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        try:
            out = _post("/chat/completions", payload, key)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            sys.exit(f"  {e.code} {e.reason}: {body}")
        el = time.perf_counter() - t0
        usage = out.get("usage") or {}
        choice = out["choices"][0]
        msg = choice["message"]
        text = (msg.get("content") or "").strip()
        # Not billed separately by the API, so infer it: the reasoning string
        # is what the budget went to before any answer was written.
        think_chars = len(msg.get("reasoning_content") or "")
        print(f"  {label:>4}  {el:5.1f}s  "
              f"in={usage.get('prompt_tokens', '?')} "
              f"out={usage.get('completion_tokens', '?')} "
              f"think={think_chars:>5}ch  finish={choice.get('finish_reason')}"
              f"  {text[:70]!r}", flush=True)
        if not text:
            print("        ^ EMPTY ANSWER: the budget was spent before the "
                  "model wrote anything a user would read.", flush=True)
    return 0


def main() -> int:
    args = sys.argv[1:]
    key = _key()
    think = "--no-think" not in args
    args = [a for a in args if a != "--no-think"]
    budget = 2000
    if "--max-tokens" in args:
        i = args.index("--max-tokens")
        budget = int(args[i + 1])
        del args[i:i + 2]
    if args and args[0] == "--chat":
        if len(args) < 2:
            sys.exit("--chat needs a model id")
        return chat(key, args[1], max_tokens=budget, think=think)
    return list_models(key, args)


if __name__ == "__main__":
    raise SystemExit(main())
