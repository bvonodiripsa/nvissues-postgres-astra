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


def chat(key: str, model: str) -> int:
    print(f"  model  {model}\n  prompt {len(PROMPT):,} chars", flush=True)
    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        try:
            out = _post("/chat/completions",
                        {"model": model,
                         "messages": [{"role": "user", "content": PROMPT}],
                         "temperature": 0.0, "max_tokens": 400}, key)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            sys.exit(f"  {e.code} {e.reason}: {body}")
        el = time.perf_counter() - t0
        usage = out.get("usage") or {}
        text = (out["choices"][0]["message"].get("content") or "").strip()
        print(f"  {label:>4}  {el:5.1f}s  "
              f"in={usage.get('prompt_tokens', '?')} "
              f"out={usage.get('completion_tokens', '?')}  {text[:90]!r}",
              flush=True)
    return 0


def main() -> int:
    args = sys.argv[1:]
    key = _key()
    if args and args[0] == "--chat":
        if len(args) < 2:
            sys.exit("--chat needs a model id")
        return chat(key, args[1])
    return list_models(key, args)


if __name__ == "__main__":
    raise SystemExit(main())
