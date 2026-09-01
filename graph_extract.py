"""The small, exact structures Cosmos mode needs but shouldn't fetch from Cosmos.

See docs/nvissues/cosmos-architecture.md. Everything the app holds splits into
two groups by three orders of magnitude: chunk/triple/entity vectors (~60 GB
at 1M bugs, belongs in Cosmos) versus bug facets and bug-to-bug links
(0.22 GB + 0.07 GB at 1M, and the reason our answers beat nvbugspro's on
attribute and relationship questions). `LocalGraphIndex` already computes
both from its own GPU snapshot; this module is the same two outputs without
the snapshot -- a `CosmosBackend` can hold one of these as its `_ix` and the
structured filter, the link inventory, and comment-mention scanning all work
exactly as they do against `LocalGraphIndex`, with zero Cosmos round trips.

    extract = GraphExtract.load("data/extract")
    ff = FacetFilter(extract.bug_facets())
    links = extract.all_bug_links()

Deliberately does NOT implement `chunks_for_bugs`: mention scanning needs
actual chunk text, which stays in Cosmos in cosmos mode (60 GB doesn't fit
here by design) -- `CosmosBackend.chunks_for_bugs` already exists for that
and callers should use the backend, not `backend._ix`, to fetch chunks.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

FACETS_FILENAME = "facets.parquet"
LINKS_FILENAME = "links.parquet"
MANIFEST_FILENAME = "manifest.json"


class GraphExtract:
    """Bug facets + bug-to-bug links, read from two small parquet files."""

    def __init__(self, facets: list[dict[str, Any]], links: list[dict[str, Any]],
                manifest: dict[str, Any] | None = None):
        self._facets = facets
        self._links = links
        self.manifest = manifest or {}

    @classmethod
    def load(cls, path: str) -> "GraphExtract":
        facets_path = os.path.join(path, FACETS_FILENAME)
        links_path = os.path.join(path, LINKS_FILENAME)
        if not os.path.exists(facets_path):
            raise FileNotFoundError(
                f"No extract at {path!r} -- run scripts/build_extract.py first "
                f"(expected {FACETS_FILENAME} and {LINKS_FILENAME})")
        t0 = time.perf_counter()
        facets = pq.read_table(facets_path).to_pylist()
        links = pq.read_table(links_path).to_pylist() if os.path.exists(links_path) else []
        manifest_path = os.path.join(path, MANIFEST_FILENAME)
        manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
        print(f"[graph_extract] loaded {len(facets):,} bug facets, {len(links):,} links "
              f"from {path!r} in {time.perf_counter() - t0:.2f}s", flush=True)
        return cls(facets, links, manifest)

    @classmethod
    def try_load(cls, path: str) -> "GraphExtract | None":
        """`load`, but None instead of raising -- for callers where a missing
        extract should degrade to today's behaviour (no facets, no links)
        rather than fail the whole backend construction.
        """
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None

    def bug_facets(self) -> list[dict[str, Any]]:
        return self._facets

    def all_bug_links(self) -> list[dict[str, Any]]:
        return self._links


def write_extract(path: str, facets: list[dict[str, Any]], links: list[dict[str, Any]],
                  manifest: dict[str, Any] | None = None) -> None:
    """Counterpart to `GraphExtract.load`. Used by scripts/build_extract.py and
    by anything else (the eventual ingest job) that produces a fresh extract.

    A corpus this small always has facets, but a fresh or tiny corpus can
    have zero recorded links -- `pa.Table.from_pylist([])` can't infer a
    schema from nothing, so an empty `links` skips the file entirely rather
    than writing a degenerate zero-column parquet table. `GraphExtract.load`
    already treats a missing links file as `[]`.
    """
    os.makedirs(path, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(facets), os.path.join(path, FACETS_FILENAME))
    links_path = os.path.join(path, LINKS_FILENAME)
    if links:
        pq.write_table(pa.Table.from_pylist(links), links_path)
    elif os.path.exists(links_path):
        os.remove(links_path)  # stale links from a previous, non-empty build
    meta = dict(manifest or {})
    meta.setdefault("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta["bug_count"] = len(facets)
    meta["link_count"] = len(links)
    with open(os.path.join(path, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
