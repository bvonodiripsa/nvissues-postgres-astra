#!/usr/bin/env python
"""Per-role LLM clients.

Three distinct LLM jobs live in this pipeline, and they pull in opposite
directions on cost, latency and data residency:

  build   Offline triple extraction across the whole corpus. This is millions
          of tokens in one batch, so per-token API pricing is the wrong shape
          entirely -- it runs on the self-hosted H100 scale set (vLLM with
          DFlash speculative decoding), and the corpus never leaves the
          network.
  expand  Online keyword expansion. One short call sitting on the critical
          path before retrieval can finish. Latency dominates; depth is
          worthless here, so this wants the cheapest, fastest model.
  answer  The final online evaluation -- the only call whose output the user
          actually reads. This is the one role pointed at the frontier model.

Each role falls back to the top-level `llm` block, so an existing flat config
keeps working unchanged and single-endpoint deployments need no `roles:` key.
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

# Model families that reject `temperature` and `max_tokens` and want
# `max_completion_tokens` instead. Matched as prefixes against the model name.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")

_DEFAULT_TIMEOUTS = {"build": 600.0, "expand": 30.0, "answer": 180.0}


def role_config(cfg: dict, role: str) -> dict:
    """Merge `llm.roles.<role>` over the top-level `llm` block.

    An override that resolved to nothing does not shadow the base value. A role
    key is almost always written as `"${SOME_VAR:-}"`, and when that variable is
    unset the config layer hands back an empty string -- which, left in place,
    silently replaces a perfectly good top-level endpoint or key with "". That
    is how the expand role came to authenticate as "dummy" against a live Azure
    deployment: not from a wrong value, but from an absent one taking priority.
    """
    llm_cfg = dict(cfg.get("llm") or {})
    roles = llm_cfg.pop("roles", None) or {}
    overlay = {k: v for k, v in (roles.get(role) or {}).items()
               if not (isinstance(v, str) and not v.strip())}
    merged = {**llm_cfg, **overlay}
    merged["_role"] = role
    return merged


def model_name(role_cfg: dict, default: str = "") -> str:
    return str(role_cfg.get("llm_model") or role_cfg.get("model") or default)


def is_reasoning_model(model: str) -> bool:
    return (model or "").lower().startswith(_REASONING_PREFIXES)


def make_client(role_cfg: dict) -> AsyncOpenAI:
    role = role_cfg.get("_role", "answer")
    return AsyncOpenAI(
        base_url=(role_cfg.get("llm_endpoint") or role_cfg.get("endpoint")
                  or "http://localhost:8000/v1"),
        api_key=(role_cfg.get("llm_api_key") or role_cfg.get("api_key")
                 or role_cfg.get("azure_openai_key") or "dummy"),
        timeout=float(role_cfg.get("timeout") or _DEFAULT_TIMEOUTS.get(role, 120.0)),
        max_retries=int(role_cfg.get("max_retries", 3)),
    )


def completion_kwargs(role_cfg: dict, model: str, max_tokens: int | None = None) -> dict:
    """Build the provider-specific half of a chat.completions.create call.

    Reasoning-model quirks are the reason this exists: GPT-5.x rejects both
    `temperature` and `max_tokens` outright, while Qwen served by vLLM needs a
    chat-template kwarg to turn thinking mode off. Everything provider-specific
    is forwarded through `extra_body` so it works across openai-python versions
    rather than depending on named parameters the installed SDK may not have.
    """
    kwargs: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    reasoning = is_reasoning_model(model)
    effort = (role_cfg.get("reasoning") or {}).get("effort")

    if reasoning:
        if max_tokens:
            kwargs["max_completion_tokens"] = int(max_tokens)
        if effort:
            extra_body["reasoning_effort"] = str(effort)
    else:
        if max_tokens:
            kwargs["max_tokens"] = int(max_tokens)
        kwargs["temperature"] = float(role_cfg.get("temperature", 0.0))
        if "qwen" in (model or "").lower():
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if effort:
            extra_body["reasoning_effort"] = str(effort)

    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def build_role(cfg: dict, role: str, max_tokens: int | None = None):
    """Return `(client, model, completion_kwargs)` for one role."""
    rc = role_config(cfg, role)
    model = model_name(rc, "Qwen/Qwen3.5-27B" if role == "build" else "gpt-5.6-sol")
    return make_client(rc), model, completion_kwargs(rc, model, max_tokens)
