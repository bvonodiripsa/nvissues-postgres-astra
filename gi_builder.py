#!/usr/bin/env python
"""Offline Graph Index builder for food product database.

Reads food product documents from Cosmos DB, extracts structured triples via LLM,
resolves entities, and stores the resulting graph index back into Cosmos DB.

Usage:
    python gi_builder.py --config my.yaml                    # full build
    python gi_builder.py --config my.yaml --dry-run          # extract without writing
    python gi_builder.py --config my.yaml --question-driven  # build from question-relevant docs
    python gi_builder.py --config my.yaml --skip-extraction --reprocess  # re-run post-processing
    python gi_builder.py --config my.yaml --concurrency 32 --extraction-rounds 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
# Optional here, unlike upstream. api.py imports this module for load_config
# and gi_query imports it for EmbedClient; neither touches Cosmos, and the
# offline build path that does is not part of this deployment. Guarding the
# import is what lets the Astra image omit the Azure SDKs entirely rather than
# carrying ~40 MB of dependency for code the pod never reaches.
try:
    from azure.cosmos.aio import CosmosClient
    from azure.cosmos import PartitionKey
    from azure.identity.aio import AzureCliCredential
except ImportError:  # pragma: no cover - exercised only in database-free builds
    CosmosClient = None
    PartitionKey = None
    AzureCliCredential = None
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()

from llm_roles import build_role
from prompts_gi_issues import (
    INITIAL_EXTRACTION_PROMPT,
    GAP_ANALYSIS_PROMPT,
    TARGETED_EXTRACTION_PROMPT,
    NORMALIZE_PREDICATES_PROMPT,
    ENTITY_MERGE_PROMPT,
    build_gap_analysis_prompt,
    compute_predicate_coverage,
    find_uncategorized_predicates,
)

# =============================================================================
# In-process embedding (Qwen3-Embedding-0.6B)
# =============================================================================
#
# Qwen3-Embedding is a decoder-only model contrastively fine-tuned to read its
# embedding off the *last* token's hidden state (with an instruction prefix on
# the query side) -- not off a mean-pooled average of all tokens. An earlier
# version of this file mean-pooled instead, which every decoder model will
# happily do without erroring, but it throws away most of the contrastive
# fine-tuning: causal attention means early tokens never see later context, so
# averaging them back in dilutes the one token (the last one) that actually
# saw the whole input. Measured effect on this corpus: cosine similarity
# between an unrelated food doc and a query landed within ~0.08-0.15 of the
# actually-relevant doc's score (mean pooling) vs. ~0.18-0.23 (last-token +
# instruction prefix) -- enough to let irrelevant matches outrank the right
# answer. See docs/EMBEDDING_FIX.md.
_QUERY_INSTRUCTION = (
    "Instruct: Given a search query about a software bug report, retrieve "
    "relevant bug report passages that answer the query\nQuery:{text}"
)

_embed_model = None
_embed_tokenizer = None
_embed_lock = None


def _get_embed_model():
    global _embed_model, _embed_tokenizer, _embed_lock
    import threading
    if _embed_lock is None:
        _embed_lock = threading.Lock()
    if _embed_model is not None:
        return _embed_model, _embed_tokenizer
    with _embed_lock:
        if _embed_model is not None:
            return _embed_model, _embed_tokenizer
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_id = "Qwen/Qwen3-Embedding-0.6B"
        # Left padding is what makes "last token = index -1" true for every
        # row in a batch regardless of that row's own length.
        _embed_tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, padding_side="left"
        )
        _embed_model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        # Every question pays this cost before retrieval can start: ~30 ms on
        # GPU against ~200 ms on CPU. Call sites read the device off the model.
        if os.environ.get("GI_EMBED_DEVICE", "cuda") != "cpu" and torch.cuda.is_available():
            _embed_model = _embed_model.to("cuda")
        _embed_model.eval()
        return _embed_model, _embed_tokenizer


def embed_sync(text: str, dimensions: int = 1024, is_query: bool = False) -> list[float]:
    """Embed text in-process: last-token pooling + L2 normalize.

    `is_query` adds Qwen3-Embedding's recommended instruction prefix. Only set
    it for user questions -- document/description text (food docs, entity and
    triple descriptions) is embedded plain, matching how it was indexed.
    """
    import torch
    model, tokenizer = _get_embed_model()
    device = next(model.parameters()).device
    if is_query:
        text = _QUERY_INSTRUCTION.format(text=text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, -1]  # left-padded -> true last token
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
    vec = emb[0].cpu().tolist()
    return vec[:dimensions]


def embed_batch_sync(
    texts: list[str], dimensions: int = 1024, batch_size: int = 32, is_query: bool = False
) -> list[list[float]]:
    """Batch embed texts in-process. See `embed_sync` for `is_query`."""
    import torch
    model, tokenizer = _get_embed_model()
    device = next(model.parameters()).device
    if is_query:
        texts = [_QUERY_INSTRUCTION.format(text=t) for t in texts]
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, -1]
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        for j in range(emb.shape[0]):
            vec = emb[j].cpu().tolist()
            all_vecs.append(vec[:dimensions])
    return all_vecs


# =============================================================================
# Config
# =============================================================================

# `${VAR}` or `${VAR:-fallback}` anywhere in a config string. This is what lets
# the same YAML be baked into a container image while the secrets arrive as
# Container Apps secrets at run time -- the image stays publishable and a
# rotated key is a revision update rather than a rebuild.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(node):
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), node)
    return node


def load_config(path: str = "my.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = _expand_env(yaml.safe_load(f))
    cosmos = cfg.get("cosmos", {})
    cosmos["uri"] = os.getenv("COSMOS_ENDPOINT", cosmos.get("uri", ""))
    cosmos["key"] = os.getenv("COSMOS_KEY", cosmos.get("key", ""))
    return cfg


# =============================================================================
# LLM Client (local vLLM via OpenAI-compatible API)
# =============================================================================

class LLMClient:
    def __init__(self, cfg: dict):
        llm = cfg.get("llm", {})
        # New upstream schema: max_completion_tokens / llm_endpoint / llm_model /
        # llm_api_key. Keep the old names as fallbacks so existing configs work.
        max_tokens = int(llm.get("max_completion_tokens", llm.get("max_tokens", 2048)))
        # Extraction is the `build` role: it runs against the self-hosted H100
        # scale set rather than the online answer model. See llm_roles.
        self._client, self._model, self._call_kwargs = build_role(cfg, "build", max_tokens)

    async def complete(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **self._call_kwargs,
        )
        text = resp.choices[0].message.content or ""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# =============================================================================
# Embedding Client
# =============================================================================

class EmbedClient:
    """Async embedding client.

    Uses the configured HTTP embedding endpoint (Ollama / OpenAI-compatible) when
    `embedding.embed_endpoint` is set — so query/KG vectors are produced by the
    same model as the stored document vectors. Falls back to the in-process
    Qwen3-Embedding-0.6B model (requires `torch`) when no endpoint is configured.
    """

    def __init__(self, cfg: dict):
        emb = cfg.get("embedding", {})
        # New upstream schema uses `embed_dimensions`; keep `dimensions` fallback.
        self._dimensions = int(
            emb.get("embed_dimensions", emb.get("dimensions", 1024))
        )
        self._endpoint = str(emb.get("embed_endpoint") or emb.get("endpoint") or "").strip()
        self._model = str(emb.get("embed_model") or emb.get("model") or "").strip()
        self._use_http = self._endpoint.startswith("http")
        self._http = None

    def _normalize(self, emb: list) -> list[float]:
        d = self._dimensions
        vals = [float(x) for x in emb]
        if d <= 0:
            return vals
        if len(vals) > d:
            return vals[:d]
        if len(vals) < d:
            return vals + [0.0] * (d - len(vals))
        return vals

    async def _embed_http(self, text: str, is_query: bool = False) -> list[float]:
        import httpx
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60)
        if is_query:
            text = _QUERY_INSTRUCTION.format(text=text)
        resp = await self._http.post(
            self._endpoint,
            json={"model": self._model, "prompt": text},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding")
        if not isinstance(emb, list):
            raise ValueError(f"Invalid embedding response from {self._endpoint}")
        return self._normalize(emb)

    async def embed(self, text: str, is_query: bool = False) -> list[float]:
        """`is_query=True` for user questions; leave False for document/description text."""
        if self._use_http:
            return await self._embed_http(text, is_query)
        return await asyncio.to_thread(embed_sync, text, self._dimensions, is_query)

    async def embed_batch(
        self, texts: list[str], batch_size: int = 32, is_query: bool = False
    ) -> list[list[float]]:
        if self._use_http:
            sem = asyncio.Semaphore(8)

            async def _one(t: str):
                async with sem:
                    return await self._embed_http(t, is_query)

            return await asyncio.gather(*[_one(t) for t in texts])
        return await asyncio.to_thread(embed_batch_sync, texts, self._dimensions, batch_size, is_query)

    async def close(self):
        if self._http is not None:
            await self._http.aclose()
            self._http = None


# =============================================================================
# Cosmos DB helpers
# =============================================================================

COSMOS_INTERNAL_FIELDS = {"_rid", "_self", "_etag", "_attachments", "_ts"}


def _doc_to_json_str(item: dict, embedding_field: str = "e") -> str:
    """Serialize a Cosmos document to compact JSON for LLM consumption."""
    exclude = COSMOS_INTERNAL_FIELDS | {embedding_field, "embedding", "id"}
    clean = {k: v for k, v in item.items() if k not in exclude and v is not None}
    return json.dumps(clean, ensure_ascii=False, indent=1)


def _doc_has_content(item: dict, text_fields: list[str]) -> bool:
    for field in text_fields:
        val = item.get(field)
        if val and (isinstance(val, str) and val.strip()):
            return True
        if val and isinstance(val, list) and any(str(v).strip() for v in val):
            return True
    return False


async def read_all_chunks(cosmos: CosmosClient, db_name: str, sources: list[dict]) -> list[dict]:
    """Read all documents from source containers."""
    db = cosmos.get_database_client(db_name)
    all_chunks = []
    for source in sources:
        container_name = source.get("container_name", "")
        text_fields = source.get("embedding_text_fields", source.get("text_fields", []))
        emb_field = source.get("embedding_field", "e")
        container = db.get_container_client(container_name)
        print(f"  Reading from '{container_name}'...")
        count = 0
        async for item in container.read_all_items():
            if not _doc_has_content(item, text_fields):
                continue
            all_chunks.append({
                "chunk_id": item.get("id", ""),
                "source": source["id"],
                "container": container_name,
                "json_doc": _doc_to_json_str(item, emb_field),
                "raw_fields": {k: v for k, v in item.items()
                               if k not in COSMOS_INTERNAL_FIELDS and k != emb_field},
            })
            count += 1
            if count % 5000 == 0:
                print(f"    ... {count} documents read")
        print(f"    -> {count} documents from '{container_name}'")
    return all_chunks


async def read_question_relevant_chunks(
    cosmos: CosmosClient,
    db_name: str,
    sources: list[dict],
    embedder: EmbedClient,
    questions: list[dict],
    question_indices: list[int] | None,
    k_per_question: int = 30,
) -> list[dict]:
    """Retrieve only documents relevant to selected questions via vector search."""
    db = cosmos.get_database_client(db_name)

    if question_indices is None:
        selected = questions[:1] if questions else []
    else:
        selected = [questions[i] for i in question_indices if i < len(questions)]

    if not selected:
        print("  No questions selected.")
        return []

    print(f"  Selected {len(selected)} question(s):")
    for q in selected:
        print(f"    [{q.get('question_id', '?')}] {q.get('question_text', '?')[:80]}")

    all_chunks: list[dict] = []
    seen_ids: set[str] = set()

    for qi, q in enumerate(selected):
        q_text = q.get("question_text", "")
        q_id = q.get("question_id", f"q{qi}")
        print(f"\n  Question {q_id}: {q_text[:60]}...")

        q_emb = await embedder.embed(q_text, is_query=True)

        for source in sources:
            container_name = source.get("container_name", "")
            emb_field = source.get("embedding_field", "e")
            text_fields = source.get("embedding_text_fields", source.get("text_fields", []))
            container = db.get_container_client(container_name)

            sql = (
                "SELECT TOP @k c, VectorDistance(c.{emb}, @emb) AS score "
                "FROM c ORDER BY VectorDistance(c.{emb}, @emb)"
            ).replace("{emb}", emb_field)

            docs = []
            async for item in container.query_items(
                query=sql,
                parameters=[
                    {"name": "@k", "value": k_per_question},
                    {"name": "@emb", "value": q_emb},
                ],
            ):
                docs.append(item)

            new_count = 0
            for item in docs:
                doc = item.get("c") if isinstance(item.get("c"), dict) else item
                doc_id = doc.get("id", "")
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)

                if not _doc_has_content(doc, text_fields):
                    continue

                all_chunks.append({
                    "chunk_id": doc_id,
                    "source": source["id"],
                    "container": container_name,
                    "json_doc": _doc_to_json_str(doc, emb_field),
                    "raw_fields": {k: v for k, v in doc.items()
                                   if k not in COSMOS_INTERNAL_FIELDS and k != emb_field},
                })
                new_count += 1

            print(f"    {container_name}: {len(docs)} retrieved, {new_count} new unique")

    print(f"\n  Total unique documents: {len(all_chunks)}")
    return all_chunks


async def ensure_gi_containers(cosmos: CosmosClient, db_name: str, cfg: dict):
    """Ensure GI containers exist; create any that are missing.

    Containers are created with minimal autoscale throughput (max 1000 RU/s) and
    a vector-embedding policy on `/e`. Indexing is minimized to only the fields
    used in queries:
      * triples  — partitioned by `/s` (subject, for graph traversal); index
        only `/s` + the vector index on `/e`.
      * entities — only vector-searched, so index nothing except the vector on
        `/e`; the `id` field doubles as the partition key.
    On a name conflict the desired name is suffixed with 1, 2, … On an RBAC
    Forbidden error, an Azure CLI hint is printed and the error re-raised.
    """
    db = cosmos.get_database_client(db_name)
    kg = cfg.get("kg", {})
    cosmos_cfg = cfg.get("cosmos", {})
    dims = int(
        cfg.get("embedding", {}).get(
            "embed_dimensions", cfg.get("embedding", {}).get("dimensions", 1024)
        )
    )

    from azure.cosmos import ThroughputProperties
    # Minimal autoscale throughput: max 1000 RU/s (scales 100–1000).
    throughput = ThroughputProperties(auto_scale_max_throughput=1000)

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/e",
                "dataType": "float32",
                "dimensions": dims,
                "distanceFunction": "cosine",
            }
        ]
    }
    # Triples: only the partition-key path (subject) is filtered during graph
    # traversal; index just that. Configurable via kg.triples_partition_key_path
    # (default /s).
    triples_pk = kg.get("triples_partition_key_path", "/s")
    triples_index = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": f"{triples_pk}/?"}],
        "excludedPaths": [{"path": "/*"}],
        "vectorIndexes": [{"path": "/e", "type": "diskANN"}],
    }
    # Entities: only vector-searched (no scalar filter/sort). `id` (the partition
    # key) is a system property and is always indexed, so include nothing else —
    # just the vector index on /e.
    entities_index = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [],
        "excludedPaths": [{"path": "/*"}],
        "vectorIndexes": [{"path": "/e", "type": "diskANN"}],
    }

    async def _create(name: str, pk_path: str, indexing_policy: dict) -> None:
        kwargs = dict(
            id=name,
            partition_key=PartitionKey(path=pk_path),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
        )
        try:
            await db.create_container(offer_throughput=throughput, **kwargs)
        except Exception as ce:
            # Serverless accounts reject throughput settings — retry without.
            if "serverless" in str(ce).lower():
                await db.create_container(**kwargs)
            else:
                raise

    async def _ensure(desired: str, pk_path: str, indexing_policy: dict) -> str:
        container = db.get_container_client(desired)
        try:
            await container.read()
            print(f"  Container '{desired}' OK")
            return desired
        except Exception as e:
            if not ("NotFound" in str(e) or "404" in str(e)):
                print(f"  Container '{desired}' check: {e}")
                raise

        name, attempt = desired, 0
        while True:
            try:
                print(f"  Creating '{name}' (autoscale max 1000 RU/s, pk={pk_path}, vector dims={dims})...")
                await _create(name, pk_path, indexing_policy)
                print(f"  Container '{name}' created")
                return name
            except Exception as ce:
                msg = str(ce)
                if "Conflict" in msg or "409" in msg or "already exists" in msg.lower():
                    attempt += 1
                    name = f"{desired}{attempt}"
                    print(f"  Name '{desired}' conflicts; retrying as '{name}'...")
                    continue
                if "Forbidden" in msg or "403" in msg:
                    acct = cosmos_cfg.get("cosmos_account_name", "<account>")
                    rg = cosmos_cfg.get("cosmos_resource_group", "<resource-group>")
                    print(f"  ERROR: no permission to create '{name}' via the data plane.")
                    print(f"  Create it once via Azure CLI, then re-run:")
                    print(f"    az cosmosdb sql container create --account-name {acct} "
                          f"--database-name {db_name} --name {name} "
                          f"--partition-key-path {pk_path} --resource-group {rg}")
                raise

    # Resolve (and persist in-memory) the actual container names used.
    kg["triples_container"] = await _ensure(kg.get("triples_container", "triples"), triples_pk, triples_index)
    kg["entities_container"] = await _ensure(kg.get("entities_container", "entities"), "/id", entities_index)
    cfg["kg"] = kg


# =============================================================================
# Triple extraction (multi-round, decomposed)
# =============================================================================

def parse_json_array(text: str) -> list[dict]:
    """Robustly extract a JSON array from LLM output."""
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
    items = []
    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(m.group())
            if "subject" in obj and "predicate" in obj and "object" in obj:
                items.append(obj)
        except json.JSONDecodeError:
            continue
    return items


def triples_to_json(triples: list[dict]) -> str:
    """Format triples as compact JSON for prompt inclusion."""
    compact = [{"s": str(t.get("subject", "")), "p": str(t.get("predicate", "")), "o": str(t.get("object", ""))}
               for t in triples if t.get("subject") and t.get("predicate") and t.get("object")]
    return json.dumps(compact, ensure_ascii=False)


async def extract_triples_decomposed(
    json_doc: str,
    llm: LLMClient,
    rounds: int = 1,
    max_gaps: int = 3,
    config_fields: set[str] | None = None,
    coverage: dict[str, float] | None = None,
    gap_prompt: str | None = None,
) -> list[dict]:
    """Multi-round triple extraction with gap analysis.

    `gap_prompt`, if given, is used verbatim (already rendered by
    `build_gap_analysis_prompt`, e.g. with corpus-wide coverage baked in from
    a prior build) instead of re-rendering it per document.
    """
    resp = await llm.complete(INITIAL_EXTRACTION_PROMPT.format(json_doc=json_doc))
    triples = [t for t in parse_json_array(resp) if isinstance(t, dict)]
    for t in triples:
        t.setdefault("confidence", 0.9)

    if rounds > 1 and gap_prompt is None:
        gap_prompt = build_gap_analysis_prompt(config_fields, coverage)

    for rnd in range(2, rounds + 1):
        if not json_doc.strip():
            break

        gap_resp = await llm.complete(gap_prompt.format(
            json_doc=json_doc,
            existing_triples=triples_to_json(triples),
        ))

        gap_instructions: list = []
        gaps_match = re.search(r'\[[\s\S]*\]', gap_resp)
        if gaps_match:
            try:
                raw = json.loads(gaps_match.group())
                if isinstance(raw, list):
                    gap_instructions = raw
            except json.JSONDecodeError:
                pass
        if not gap_instructions:
            break

        gap_strings = []
        for g in gap_instructions[:max_gaps]:
            if isinstance(g, str):
                gap_strings.append(g)
            elif isinstance(g, dict):
                gap_strings.append(g.get("instruction", g.get("gap", str(g))))

        for gap in gap_strings:
            targeted_resp = await llm.complete(TARGETED_EXTRACTION_PROMPT.format(
                gap_instruction=gap,
                existing_triples=triples_to_json(triples),
                json_doc=json_doc,
            ))
            new_triples = [t for t in parse_json_array(targeted_resp) if isinstance(t, dict)]
            for t in new_triples:
                t.setdefault("confidence", 0.85)
                t["extraction_round"] = rnd
            triples.extend(new_triples)

    for t in triples:
        for field in ("subject", "predicate", "object"):
            t[field] = str(t.get(field, "")).strip()

    return triples


# =============================================================================
# Post-processing
# =============================================================================

def dedup_and_boost(triples: list[dict], min_confidence: float = 0.5) -> list[dict]:
    """Deduplicate triples; re-confirmed ones get boosted confidence."""
    merged: dict[str, dict] = {}
    for t in triples:
        s = str(t.get("subject", "")).strip().lower()
        p = str(t.get("predicate", "")).strip().lower()
        o = str(t.get("object", "")).strip().lower()
        if not s or not p or not o:
            continue
        key = f"{s}|{p}|{o}"
        if key in merged:
            old_conf = merged[key].get("confidence", 0.8)
            new_conf = t.get("confidence", 0.8)
            merged[key]["confidence"] = min(1.0, max(old_conf, new_conf) + 0.1)
            merged[key]["confirmations"] = merged[key].get("confirmations", 1) + 1
            for cid in t.get("source_chunks", []):
                if cid not in merged[key]["source_chunks"]:
                    merged[key]["source_chunks"].append(cid)
        else:
            merged[key] = {
                "subject": t.get("subject", "").strip(),
                "predicate": t.get("predicate", "").strip(),
                "object": t.get("object", "").strip(),
                "confidence": t.get("confidence", 0.8),
                "confirmations": 1,
                "source_chunks": list(t.get("source_chunks", [])),
            }
    return [t for t in merged.values() if t["confidence"] >= min_confidence]


async def normalize_predicates(triples: list[dict], llm: LLMClient, batch_size: int = 30, concurrency: int = 20) -> list[dict]:
    """Batch-normalize predicates to a standard vocabulary."""
    total_batches = (len(triples) + batch_size - 1) // batch_size
    batches = []
    for i in range(0, len(triples), batch_size):
        batches.append((i // batch_size + 1, triples[i:i + batch_size]))

    sem = asyncio.Semaphore(concurrency)
    done_count = 0
    lock = asyncio.Lock()

    async def _process_batch(batch_num: int, batch: list[dict]) -> list[dict]:
        nonlocal done_count
        compact = json.dumps([
            {"subject": t["subject"], "predicate": t["predicate"], "object": t["object"]}
            for t in batch
        ], ensure_ascii=False)
        async with sem:
            try:
                resp = await llm.complete(NORMALIZE_PREDICATES_PROMPT.format(triples=compact))
                normalized = parse_json_array(resp)
                if len(normalized) == len(batch):
                    for orig, norm in zip(batch, normalized):
                        orig["predicate"] = norm.get("predicate", orig["predicate"])
            except Exception as e:
                print(f"    batch {batch_num}/{total_batches}: ERROR {e}")

        async with lock:
            done_count += 1
            if done_count % 10 == 0 or done_count == total_batches:
                print(f"    {done_count}/{total_batches} batches done")
        return batch

    tasks = [_process_batch(bn, b) for bn, b in batches]
    results = await asyncio.gather(*tasks)
    all_normalized = []
    for batch_result in results:
        all_normalized.extend(batch_result)
    return all_normalized


async def resolve_entities(
    triples: list[dict],
    embedder: EmbedClient,
    llm: LLMClient,
    merge_threshold: float = 0.85,
) -> tuple[list[dict], dict[str, str]]:
    """Cluster entities by embedding similarity, then LLM-verify merges."""
    entities = sorted({t["subject"] for t in triples} | {t["object"] for t in triples})
    if len(entities) < 2:
        return triples, {}

    print(f"  Embedding {len(entities)} entities...")
    embeddings = await embedder.embed_batch(entities)
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    sim_matrix = emb_matrix @ emb_matrix.T
    candidate_pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if sim_matrix[i, j] >= merge_threshold and entities[i].lower() != entities[j].lower():
                candidate_pairs.append((entities[i], entities[j], float(sim_matrix[i, j])))

    if not candidate_pairs:
        print("  No entity merge candidates found.")
        return triples, {}

    print(f"  {len(candidate_pairs)} merge candidates, verifying with LLM...")
    mapping: dict[str, str] = {}
    sem = asyncio.Semaphore(20)

    async def _verify_batch(batch: list) -> list:
        pairs_json = json.dumps([
            {"entity_a": a, "entity_b": b, "similarity": round(s, 3)}
            for a, b, s in batch
        ])
        async with sem:
            try:
                resp = await llm.complete(ENTITY_MERGE_PROMPT.format(pairs=pairs_json))
                decisions = parse_json_array(resp)
            except Exception:
                decisions = []
        results = []
        for d in decisions:
            if not isinstance(d, dict):
                continue
            if d.get("merge"):
                canonical = d.get("canonical", d.get("entity_a", ""))
                a, b = d.get("entity_a", ""), d.get("entity_b", "")
                if canonical and a and b:
                    other = b if canonical == a else a
                    results.append((other, canonical))
        return results

    batch_tasks = []
    for i in range(0, len(candidate_pairs), 20):
        batch_tasks.append(_verify_batch(candidate_pairs[i:i + 20]))

    all_results = await asyncio.gather(*batch_tasks)
    for results in all_results:
        for other, canonical in results:
            mapping[other] = canonical

    if mapping:
        print(f"  Merging {len(mapping)} entity aliases")
        for t in triples:
            t["subject"] = mapping.get(t["subject"], t["subject"])
            t["object"] = mapping.get(t["object"], t["object"])

    return triples, mapping


# =============================================================================
# Store graph index to Cosmos DB
# =============================================================================

async def store_triples(
    cosmos: CosmosClient,
    db_name: str,
    container_name: str,
    triples: list[dict],
    embedder: EmbedClient,
    pk_field: str = "s",
):
    """Store triples with embeddings for vector search."""
    db = cosmos.get_database_client(db_name)
    container = db.get_container_client(container_name)

    print(f"  Embedding {len(triples)} triples...")
    descriptions = [f"{t['subject']} {t['predicate']} {t['object']}" for t in triples]
    embeddings = await embedder.embed_batch(descriptions)

    print(f"  Writing to '{container_name}'...")
    t0 = time.time()
    for i, (t, emb) in enumerate(zip(triples, embeddings)):
        doc = {
            "id": f"t_{i:06d}",
            "s": t["subject"],                 # partition key (/s)
            "p": t["predicate"],
            "o": t["object"],
            "f": t.get("confidence", 0.8),
            "n": t.get("confirmations", 1),
            "d": t.get("source_chunks", []),
            "e": emb,
        }
        if pk_field != "s":
            doc[pk_field] = t["subject"]
        await container.upsert_item(doc)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            print(f"    {i + 1}/{len(triples)} triples stored ({rate:.1f}/s)")
    print(f"    Done: {len(triples)} triples in '{container_name}' ({time.time() - t0:.1f}s)")


async def store_entities(
    cosmos: CosmosClient,
    db_name: str,
    container_name: str,
    triples: list[dict],
    embedder: EmbedClient,
):
    """Build entity index from triples and store with embeddings."""
    entities: dict[str, dict] = {}
    for t in triples:
        for role in ("subject", "object"):
            name = t[role]
            if name not in entities:
                entities[name] = {"name": name, "relations": [], "source_chunks": set()}
            rel_str = (f"{t['predicate']} -> {t['object']}" if role == "subject"
                       else f"{t['subject']} -> {t['predicate']}")
            entities[name]["relations"].append(rel_str)
            for cid in t.get("source_chunks", []):
                entities[name]["source_chunks"].add(cid)

    entity_list = list(entities.values())
    print(f"  Building descriptions for {len(entity_list)} entities...")
    descriptions = []
    for e in entity_list:
        rels = "; ".join(e["relations"][:15])
        descriptions.append(f"{e['name']}. Relations: {rels}")

    print(f"  Embedding {len(entity_list)} entities...")
    embeddings = await embedder.embed_batch(descriptions)

    db = cosmos.get_database_client(db_name)
    container = db.get_container_client(container_name)

    print(f"  Writing to '{container_name}'...")
    t0 = time.time()
    for i, (e, emb) in enumerate(zip(entity_list, embeddings)):
        doc = {
            "id": f"e_{i:06d}",              # doubles as the partition key (/id)
            "n": e["name"],
            "t": descriptions[i],
            "r": len(e["relations"]),
            "d": list(e["source_chunks"])[:50],
            "e": emb,
        }
        await container.upsert_item(doc)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            print(f"    {i + 1}/{len(entity_list)} entities stored ({rate:.1f}/s)")
    print(f"    Done: {len(entity_list)} entities in '{container_name}' ({time.time() - t0:.1f}s)")


# =============================================================================
# Main build pipeline
# =============================================================================

async def build_gi(args):
    cfg = load_config(args.config)
    cosmos_cfg = cfg["cosmos"]
    build_cfg = cfg.get("build", {})
    kg_cfg = cfg.get("kg", {})

    print("=" * 70)
    print(f"Food Graph Index Builder")
    print("=" * 70)
    print(f"  Cosmos DB: {cosmos_cfg['uri']}")
    print(f"  Database:  {cosmos_cfg['database_name']}")
    print(f"  LLM:       {cfg['llm']['endpoint']} ({cfg['llm']['model']})")
    _emb = cfg.get("embedding", {})
    _emb_ep = _emb.get("embed_endpoint") or _emb.get("endpoint")
    if _emb_ep and str(_emb_ep).startswith("http"):
        print(f"  Embedding: {_emb_ep} ({_emb.get('embed_model') or _emb.get('model')})")
    else:
        print(f"  Embedding: in-process Qwen3-Embedding-0.6B")
    print()

    llm = LLMClient(cfg)
    embedder = EmbedClient(cfg)

    # RBAC auth for Cosmos DB
    tenant_id = cosmos_cfg.get("tenant_id", "")
    if cosmos_cfg.get("use_rbac_auth") and tenant_id:
        cred = AzureCliCredential(tenant_id=tenant_id)
        cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)
        print(f"  Auth: RBAC (tenant {tenant_id[:8]}...)")
    else:
        cosmos = CosmosClient(cosmos_cfg["uri"], cosmos_cfg.get("key", ""))
        print(f"  Auth: Key-based")

    sources = cosmos_cfg.get("sources", [])

    # Determine chunks to process
    if args.skip_extraction:
        chunks = []
        print("(skipping extraction — loading triples from disk)\n")
    elif args.question_driven:
        qfile = args.questions_file or cfg.get("paths", {}).get("questions_file", "data/food.json")
        with open(qfile, encoding="utf-8") as f:
            questions = json.load(f)
        print(f"  Loaded {len(questions)} questions from {qfile}")

        qi_str = str(args.question_index).strip().lower()
        if qi_str == "all":
            question_indices = list(range(len(questions)))
        else:
            question_indices = [int(x.strip()) for x in qi_str.split(",") if x.strip().isdigit()]

        print(f"\nSTEP 1: Question-driven retrieval (k={args.question_k} per question)")
        t0 = time.time()
        chunks = await read_question_relevant_chunks(
            cosmos, cosmos_cfg["database_name"], sources,
            embedder, questions, question_indices, k_per_question=args.question_k,
        )
        print(f"  Total: {len(chunks)} unique documents in {time.time() - t0:.1f}s\n")
    else:
        print("STEP 1: Reading ALL source documents from Cosmos DB")
        t0 = time.time()
        chunks = await read_all_chunks(cosmos, cosmos_cfg["database_name"], sources)
        print(f"  Total: {len(chunks)} chunks in {time.time() - t0:.1f}s\n")

    if not chunks and not args.skip_extraction:
        print("No chunks found. Exiting.")
        return

    extraction_rounds = args.extraction_rounds or int(build_cfg.get("extraction_rounds", 1))
    max_gaps = int(build_cfg.get("max_gaps_per_round", 3))
    concurrency = args.concurrency or int(build_cfg.get("concurrency", 20))
    min_conf = float(build_cfg.get("min_confidence", 0.5))
    out_path = Path(cfg.get("paths", {}).get("output_root", "out_gi"))
    out_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_path / "checkpoint_raw_triples.json"

    if not args.dry_run:
        print("PREP: Ensuring GI containers exist...")
        await ensure_gi_containers(cosmos, cosmos_cfg["database_name"], cfg)
        print()

    # Check for checkpoint
    all_triples: list[dict] = []
    processed_chunk_ids: set[str] = set()

    if not args.skip_extraction and checkpoint_path.exists():
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                saved = json.load(f)
            all_triples = saved.get("triples", [])
            processed_chunk_ids = set(saved.get("processed_chunk_ids", []))
            remaining = [c for c in chunks if c["chunk_id"] not in processed_chunk_ids]
            print(f"RESUME: {len(processed_chunk_ids)} docs done, {len(all_triples)} raw triples, "
                  f"{len(remaining)} remaining\n")
            chunks = remaining
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: checkpoint {checkpoint_path.name} is corrupt/incomplete "
                  f"({type(e).__name__}); starting fresh.\n")
            all_triples = []
            processed_chunk_ids = set()

    # Coverage warm start: a prior completed build's triples.json, or (when
    # resuming) whatever this run's own checkpoint already has, gives a real
    # per-category coverage measurement to prioritize round 2 with, instead
    # of checking all 9 categories uniformly on every bug. First-ever build
    # has neither, so `coverage` stays None and behaviour is unchanged.
    coverage_source = list(all_triples)
    if not coverage_source:
        prior_path = out_path / "triples.json"
        if prior_path.exists():
            try:
                with open(prior_path, encoding="utf-8") as f:
                    coverage_source = json.load(f)
                print(f"  Coverage warm start: {len(coverage_source)} triples from a prior build "
                      f"({prior_path})")
            except (json.JSONDecodeError, OSError):
                coverage_source = []

    gap_coverage = compute_predicate_coverage(coverage_source) if coverage_source else None
    if gap_coverage:
        ranked = sorted(gap_coverage.items(), key=lambda kv: kv[1])
        print("  Predicate coverage (lowest first, drives round-2 priority):")
        for cat, frac in ranked:
            flag = " -- below threshold, will be gap-checked" if frac < 0.6 else " -- already common, skipped"
            print(f"    {cat:<15} {frac:5.1%}{flag}")
        overflow = find_uncategorized_predicates(coverage_source)
        if overflow:
            print("  Predicates outside PREDICATE_CATEGORIES (candidates for a new category):")
            for pred, n in overflow[:10]:
                print(f"    {pred:<30} seen {n} times")
        print()

    gap_prompt = build_gap_analysis_prompt(coverage=gap_coverage) if extraction_rounds > 1 else None

    if not args.skip_extraction:
        print(f"STEP 2: Extract triples (rounds={extraction_rounds}, concurrency={concurrency})")
        print(f"  Documents to process: {len(chunks)}")
        print("-" * 60)

        sem = asyncio.Semaphore(concurrency)
        t0_total = time.time()
        doc_num = len(processed_chunk_ids)
        total_docs = len(chunks) + doc_num

        async def _extract_one(chunk: dict) -> tuple[str, list[dict]]:
            async with sem:
                triples = await extract_triples_decomposed(
                    chunk["json_doc"], llm,
                    gap_prompt=gap_prompt,
                    rounds=extraction_rounds,
                    max_gaps=max_gaps,
                )
                for t in triples:
                    t.setdefault("source_chunks", [])
                    t["source_chunks"].append(chunk["chunk_id"])
                return chunk["chunk_id"], triples

        time_limit = getattr(args, "time_limit", None)
        time_limit_hit = False

        tasks = [_extract_one(c) for c in chunks]
        for coro in asyncio.as_completed(tasks):
            chunk_id, doc_triples = await coro
            doc_num += 1
            all_triples.extend(doc_triples)
            processed_chunk_ids.add(chunk_id)

            elapsed = time.time() - t0_total
            processed_now = doc_num - (total_docs - len(chunks))
            rate = processed_now / max(elapsed, 0.1)
            remaining_est = (total_docs - doc_num) / max(rate, 0.01)

            if doc_num % 10 == 0 or doc_num == total_docs:
                print(f"  {doc_num}/{total_docs} docs | "
                      f"{len(all_triples)} triples | "
                      f"{elapsed:.0f}s elapsed | ~{remaining_est:.0f}s remaining")

            if time_limit and elapsed >= time_limit:
                time_limit_hit = True
                print(f"\n  TIME LIMIT ({time_limit}s) reached. Saving checkpoint...")
                # Cancel remaining tasks
                for t in tasks:
                    if hasattr(t, 'cancel'):
                        t.cancel()
                break

            if doc_num % 50 == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "triples": all_triples,
                        "processed_chunk_ids": list(processed_chunk_ids),
                    }, f, ensure_ascii=False)

        # Final checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "triples": all_triples,
                "processed_chunk_ids": list(processed_chunk_ids),
            }, f, ensure_ascii=False)

        elapsed_total = time.time() - t0_total
        print(f"\n  Extraction complete: {len(all_triples)} raw triples from "
              f"{len(processed_chunk_ids)} docs in {elapsed_total:.1f}s\n")

        if time_limit_hit:
            print(f"  PAUSED: {len(processed_chunk_ids)} of {total_docs} docs processed.")
            print(f"  Checkpoint saved to: {checkpoint_path}")
            print(f"  Resume tomorrow with the same command (will auto-detect checkpoint).")
            if args.skip_post_processing:
                return
            # Still do post-processing on what we have so far
            print("  Running post-processing on extracted triples...\n")

        if args.skip_post_processing:
            print("STEPS 3-5: Skipped. Raw triples saved.")
            with open(out_path / "triples_raw.json", "w", encoding="utf-8") as f:
                json.dump(all_triples, f, ensure_ascii=False)
            return
    else:
        triples_path = out_path / "triples.json"
        if not triples_path.exists():
            triples_path = out_path / "checkpoint_raw_triples.json"
        if not triples_path.exists():
            print(f"ERROR: No triples file found in {out_path}")
            return
        with open(triples_path, encoding="utf-8") as f:
            data = json.load(f)
        all_triples = data if isinstance(data, list) else data.get("triples", [])
        print(f"  Loaded {len(all_triples)} triples from {triples_path}\n")

    # STEP 3: Dedup
    print("STEP 3: Deduplication + confidence boosting")
    before = len(all_triples)
    all_triples = dedup_and_boost(all_triples, min_confidence=min_conf)
    print(f"  {before} raw -> {len(all_triples)} after dedup\n")

    # STEP 4: Normalize predicates
    print("STEP 4: Predicate normalization")
    t0 = time.time()
    all_triples = await normalize_predicates(all_triples, llm, batch_size=30, concurrency=concurrency)
    all_triples = dedup_and_boost(all_triples, min_confidence=min_conf)
    print(f"  After normalization: {len(all_triples)} triples in {time.time() - t0:.1f}s\n")

    # STEP 5: Entity resolution
    print("STEP 5: Entity resolution")
    t0 = time.time()
    merge_threshold = float(build_cfg.get("entity_merge_threshold", 0.85))
    all_triples, mapping = await resolve_entities(all_triples, embedder, llm, merge_threshold)
    all_triples = dedup_and_boost(all_triples, min_confidence=min_conf)
    print(f"  After entity resolution: {len(all_triples)} triples in {time.time() - t0:.1f}s\n")

    # Save locally
    with open(out_path / "triples.json", "w", encoding="utf-8") as f:
        json.dump(all_triples, f, indent=2, ensure_ascii=False)
    if mapping:
        with open(out_path / "entity_mapping.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        print("DRY RUN complete (triples NOT written to Cosmos).")
        entities = sorted({t["subject"] for t in all_triples} | {t["object"] for t in all_triples})
        predicates = sorted({t["predicate"] for t in all_triples})
        print(f"  Triples: {len(all_triples)}")
        print(f"  Entities: {len(entities)}")
        print(f"  Predicates: {len(predicates)}")
        print(f"  Sample: {', '.join(predicates[:15])}")
        print("\nSample triples:")
        for t in all_triples[:10]:
            print(f"  ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")
        return

    # STEP 6: Store to Cosmos
    print("STEP 6: Storing final triples + entities to Cosmos DB (graph index)")
    triples_pk_field = kg_cfg.get("triples_partition_key_path", "/s").lstrip("/")
    await store_triples(
        cosmos, cosmos_cfg["database_name"],
        kg_cfg.get("triples_container", "kg_triples_food"),
        all_triples, embedder, pk_field=triples_pk_field,
    )
    await store_entities(
        cosmos, cosmos_cfg["database_name"],
        kg_cfg.get("entities_container", "kg_entities_food"),
        all_triples, embedder,
    )

    await embedder.close()
    await cosmos.close()

    entities_count = len({t["subject"] for t in all_triples} | {t["object"] for t in all_triples})
    print("\n" + "=" * 70)
    print("Graph Index build complete!")
    print(f"  Triples: {len(all_triples)}")
    print(f"  Entities: {entities_count}")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Food Graph Index")
    parser.add_argument("--config", default="my.yaml")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-post-processing", action="store_true")
    parser.add_argument("--extraction-rounds", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)

    parser.add_argument("--time-limit", type=int, default=None,
                        help="Stop extraction after this many seconds (saves checkpoint)")

    qd = parser.add_argument_group("question-driven mode")
    qd.add_argument("--question-driven", action="store_true")
    qd.add_argument("--questions-file", default=None)
    qd.add_argument("--question-index", default="all")
    qd.add_argument("--question-k", type=int, default=30)

    args = parser.parse_args()
    asyncio.run(build_gi(args))


if __name__ == "__main__":
    main()
