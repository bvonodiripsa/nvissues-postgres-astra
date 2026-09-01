"""Check whether a local Graph Index snapshot is stale relative to live Cosmos DB.

The exporter (`scripts/build_local_index.py`) writes `manifest.json` with a
`built_at` timestamp and a `counts` dict. This compares that against two cheap
live signals per container:

  - `SELECT VALUE COUNT(1) FROM c`      -- document count drifted
  - `SELECT VALUE MAX(c._ts) FROM c`    -- newest write happened after built_at

Either mismatch means the snapshot no longer reflects Cosmos. Both queries are
single aggregates, cheap even on multi-million-document containers -- this is
meant to run on every server startup and periodically in the background, not
just once during a manual audit.

    report = await check_freshness(cfg, "data/local_index")
    if report["stale"]:
        log.warning("local snapshot is stale: %s", report)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

# Optional here: this deployment reads Postgres and never opens a Cosmos
# connection, and `check_freshness` is not on any path it takes. Guarding the
# import means calling it without the SDKs raises at the call rather than at
# import, which would take the whole app down for a function it never reaches.
try:
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import AzureCliCredential
except ImportError:  # pragma: no cover - exercised only in database-free builds
    CosmosClient = None
    AzureCliCredential = None


async def _container_stats(db, name: str) -> dict[str, Any]:
    container = db.get_container_client(name)
    count = 0
    async for row in container.query_items(query="SELECT VALUE COUNT(1) FROM c"):
        count = row
    max_ts = 0
    async for row in container.query_items(query="SELECT VALUE MAX(c._ts) FROM c"):
        max_ts = row or 0
    return {"count": count, "max_ts": max_ts}


def _parse_built_at(built_at: str | None) -> float | None:
    if not built_at:
        return None
    try:
        # e.g. "2026-07-27T17:09:15-0700" -- offset-aware, comparable to
        # Cosmos's `_ts` (Unix epoch seconds, UTC) via .timestamp().
        return datetime.strptime(built_at, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        return None


async def check_freshness(cfg: dict, snapshot_path: str) -> dict[str, Any]:
    """Compare `snapshot_path/manifest.json` against live Cosmos state.

    Returns `{"stale": bool, "built_at": str|None, "containers": {...},
    "unrecorded_containers": [...]}`. `containers` has one entry per name in
    the manifest's `counts`, each with `snapshot_count`, `live_count`,
    `count_match`, and `stale_by_ts`. `unrecorded_containers` lists `.vecs.npy`
    files on disk that the manifest never recorded a count for (seen in
    practice: a two-step export left `food` un-recorded even though
    `food.vecs.npy` exists) -- a real gap in what this check can validate,
    reported rather than silently skipped.
    """
    manifest_path = os.path.join(snapshot_path, "manifest.json")
    if not os.path.exists(manifest_path):
        return {"stale": True, "reason": f"no manifest at {manifest_path}",
                 "built_at": None, "containers": {}, "unrecorded_containers": []}

    manifest = json.load(open(manifest_path))
    built_at_epoch = _parse_built_at(manifest.get("built_at"))

    cosmos_cfg = cfg["cosmos"]
    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    client = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = client.get_database_client(cosmos_cfg["database_name"])

    containers: dict[str, Any] = {}
    stale = False
    try:
        for name, snap_count in manifest.get("counts", {}).items():
            stats = await _container_stats(db, name)
            count_match = stats["count"] == snap_count
            stale_by_ts = built_at_epoch is not None and stats["max_ts"] > built_at_epoch
            containers[name] = {
                "snapshot_count": snap_count,
                "live_count": stats["count"],
                "count_match": count_match,
                "stale_by_ts": stale_by_ts,
            }
            if not count_match or stale_by_ts:
                stale = True
    finally:
        await client.close()
        await cred.close()

    unrecorded = sorted(
        f[: -len(".vecs.npy")] for f in os.listdir(snapshot_path)
        if f.endswith(".vecs.npy") and f[: -len(".vecs.npy")] not in manifest.get("counts", {})
    )

    return {
        "stale": stale,
        "built_at": manifest.get("built_at"),
        "containers": containers,
        "unrecorded_containers": unrecorded,
    }
