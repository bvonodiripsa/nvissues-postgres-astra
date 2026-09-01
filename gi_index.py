"""In-process Graph Index: GPU vector search, CSR graph traversal, BM25 keyword search.

Drop-in replacement for the Cosmos DB round trips in the retrieval pipeline.
Vectors live on the GPU (~3.8 GB in fp16 for the full corpus); payloads stay in
host memory as Arrow columns and are materialised only for the handful of rows
each query actually returns.

Search is exact — every vector is compared on every query. At this corpus size
that costs ~21 ms for the 1.59M triples, so an approximate index (CAGRA,
Vamana) would save time the pipeline cannot use while giving up recall.

    index = LocalGraphIndex("data/local_index")
    hits  = index.search_entities(q_emb, k=10)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

_TOKEN = re.compile(r"[a-z0-9]+")

# Payload column -> key expected by the prompt builders, per container.
_ENTITY_MAP = {"n": "name", "t": "description", "r": "relation_count", "d": "source_chunks"}
_TRIPLE_MAP = {"s": "subject", "p": "predicate", "o": "object", "f": "confidence",
               "d": "source_chunks", "w": "weight"}

# Chunk fields that BM25 indexes, mirroring the fullTextIndexes in my.yaml.
# `synopsis` is repeated on every chunk of a bug on purpose: a lexical hit on
# the title should surface the bug's comment windows too, not just its
# synopsis chunk. `bug_id` is here because "what is bug 6539931 about" is a
# routine triage question and a bare number is the one thing vector search
# cannot resolve -- without it the only way to that bug is another bug's
# comment thread happening to cite it.
_FT_FIELDS = ("bug_id", "text", "synopsis", "keywords", "categories", "module")

# Edges between two bugs, as opposed to a bug and one of its attribute values.
# They are the only edges a multi-hop question can be answered by walking, and
# also the rarest, so `max_per_node` protects them ahead of a bug's fifteenth
# keyword or fourth attachment.
LINK_PREDICATES = frozenset({
    "cloned_from", "duplicate_of", "has_duplicate", "see_also",
    "original_bug", "gated_by", "gating",
})

# Cheapest sections to read when all that is needed is what a linked bug *is*:
# its title and problem statement, not its comment thread. Shared by both
# backends -- `LocalGraphIndex.chunks_for_bugs` sorts by it locally,
# `CosmosBackend.chunks_for_bugs` sorts the same way after a remote fetch.
SUMMARY_SECTIONS = ("synopsis", "description", "repro")


class _Table:
    """Arrow-backed payload with GPU-resident vectors."""

    def __init__(self, path: str, name: str, torch, device: str):
        self.name = name
        vecs = np.load(os.path.join(path, f"{name}.vecs.npy"))
        self.vecs = torch.from_numpy(vecs).to(device)
        self.table = pq.read_table(os.path.join(path, f"{name}.payload.parquet"))
        self.cols = {c: self.table.column(c) for c in self.table.column_names}
        self.n = len(self.table)
        if self.n != self.vecs.shape[0]:
            raise ValueError(f"{name}: {self.n} payload rows vs {self.vecs.shape[0]} vectors")

    def row(self, i: int, mapping: dict[str, str] | None) -> dict[str, Any]:
        if mapping is None:
            return {c: self.cols[c][i].as_py() for c in self.cols}
        return {out: self.cols[src][i].as_py() for src, out in mapping.items() if src in self.cols}

    @property
    def nbytes(self) -> int:
        return self.vecs.numel() * self.vecs.element_size()


class LocalGraphIndex:
    def __init__(self, path: str = "data/local_index", device: str = "cuda",
                 enable_bm25: bool = True, verbose: bool = True):
        import torch  # deferred so importing this module stays cheap
        self._torch = torch
        self.path = path
        self.device = device if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()

        self.manifest = json.load(open(os.path.join(path, "manifest.json")))
        self.entities = _Table(path, "entities", torch, self.device)
        self.triples = _Table(path, "triples", torch, self.device)
        self.chunks = _Table(path, "chunks", torch, self.device)

        # Hub edges point at values almost every bug shares ("Code Defect",
        # "3-Functionality"). Traversing them is what turns a two-hop expansion
        # into a scan of the whole corpus, so the builder flags them and
        # `triples_for` drops them unless asked otherwise.
        self._hub = (np.asarray(self.triples.cols["h"].to_pylist(), dtype=bool)
                     if "h" in self.triples.cols else None)

        csr = np.load(os.path.join(path, "triples.csr.npz"))
        self._fwd_indptr, self._fwd_indices = csr["fwd_indptr"], csr["fwd_indices"]
        self._rev_indptr, self._rev_indices = csr["rev_indptr"], csr["rev_indices"]
        self._vocab: dict[str, int] = json.load(open(os.path.join(path, "triples.vocab.json")))

        self._chunk_by_id = {v.as_py(): i for i, v in enumerate(self.chunks.cols["id"])}

        self._bm25 = None
        if enable_bm25:
            self._build_bm25()

        if verbose:
            gb = sum(t.nbytes for t in (self.entities, self.triples, self.chunks)) / 1e9
            print(f"[gi_index] loaded in {time.perf_counter()-t0:.1f}s on {self.device}: "
                  f"{self.entities.n:,} entities, {self.triples.n:,} triples, "
                  f"{self.chunks.n:,} chunks, {gb:.2f} GB of vectors", flush=True)

    # ---------------------------------------------------------------- vectors

    def _search(self, tbl: _Table, q: Iterable[float], k: int, mapping):
        torch = self._torch
        qv = torch.as_tensor(np.asarray(q, dtype=np.float32), device=self.device)
        qv = qv / qv.norm().clamp_min(1e-12)
        # Vectors were L2-normalised at export, so a dot product is cosine
        # similarity and Cosmos' VectorDistance ordering is 1 - this.
        scores = (qv.to(tbl.vecs.dtype) @ tbl.vecs.T).float()
        k = min(k, tbl.n)
        vals, idx = torch.topk(scores, k)
        out = []
        for score, i in zip(vals.tolist(), idx.tolist()):
            row = tbl.row(i, mapping)
            # Cosmos VectorDistance with a cosine policy reports similarity,
            # highest first. Match that so downstream code sees one convention.
            row["score"] = score
            out.append(row)
        return out

    def search_entities(self, q_emb, k: int = 10):
        return self._search(self.entities, q_emb, k, _ENTITY_MAP)

    def search_triples(self, q_emb, k: int = 30):
        return self._search(self.triples, q_emb, k, _TRIPLE_MAP)

    def search_chunks(self, q_emb, k: int = 12):
        return self._search(self.chunks, q_emb, k, None)

    # ------------------------------------------------------------------ graph

    def triples_for(self, name: str, reverse: bool = False, include_hubs: bool = False,
                    max_per_node: int | None = None) -> list[dict[str, Any]]:
        """Triples touching `name`.

        Forward edges mirror what Cosmos can serve, since `s` is the partition
        key. Reverse edges are the ones a cross-partition scan makes
        impractical remotely, and they are what makes attribute-level seed
        entities reachable at all.

        Hub edges are dropped from the reverse direction only, because hub-ness
        is a property of the object and only bites when you enter through it.
        Going forward from `Bug 5771630`, `has_disposition -> Not an NV bug` is
        precisely the fact an answer needs; coming back the other way, that
        same node fans out to hundreds of unrelated bugs.

        `include_hubs` exists because that rule has one exception: when the
        query itself named the hub. "Which P0 bugs affect Diomedes" matches the
        Diomedes node directly, and enumerating its 359 edges is the answer
        rather than the noise. The caller knows which nodes came from the query
        and which were merely stumbled upon; `max_per_node` then bounds the
        fan-out so one popular node cannot crowd out the other seeds.
        """
        node = self._vocab.get(name)
        if node is None:
            return []
        rows = list(self._fwd_indices[self._fwd_indptr[node]:self._fwd_indptr[node + 1]])
        if reverse:
            rev = self._rev_indices[self._rev_indptr[node]:self._rev_indptr[node + 1]]
            if self._hub is not None and not include_hubs:
                rev = [i for i in rev if not self._hub[i]]
            rows += list(rev)
        # Forward edges come first, so a cap keeps the node's own attributes
        # and trims the fan-out rather than the other way round.
        #
        # Within that, bug-to-bug links are promoted ahead of attributes. The
        # builder emits them last, after every keyword, category and
        # attachment, so a flat cut by insertion order drops the clone edge on
        # any bug carrying more than `max_per_node` attributes and quietly
        # makes it unreachable — the one edge worth traversing lost to a
        # node's fourth attachment. The sort is stable, so forward edges still
        # precede reverse ones within each group.
        if max_per_node is not None and len(rows) > max_per_node:
            pred = self.triples.cols["p"]
            rows.sort(key=lambda i: pred[int(i)].as_py() not in LINK_PREDICATES)
            rows = rows[:max_per_node]
        return [self.triples.row(int(i), _TRIPLE_MAP) for i in rows]

    def docs_by_id(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        out = []
        for cid in ids:
            i = self._chunk_by_id.get(cid)
            if i is not None:
                out.append(self.chunks.row(i, None))
        return out

    # ------------------------------------------------------------------ facets

    # Bug-level facets are denormalised onto every chunk, so one row per bug
    # collapses the chunk table into something an attribute predicate can be
    # evaluated over exhaustively.
    _BUG_FACETS = ("bug_id", "synopsis", "module", "priority", "priority_rank",
                   "severity", "severity_rank", "disposition", "is_open",
                   "partner_area", "issue_type", "ms_status", "release",
                   "system", "customer_name", "os", "version", "modified_ts")

    # See module-level SUMMARY_SECTIONS -- kept as a class attribute too since
    # existing callers reference it as `self._SUMMARY_SECTIONS`.
    _SUMMARY_SECTIONS = SUMMARY_SECTIONS

    def chunks_for_bugs(self, bug_ids: Iterable[str], per_bug: int = 3) -> list[dict[str, Any]]:
        """Summary chunks for whole bugs, addressed by bug id rather than chunk id.

        Graph traversal reaches a linked bug by name, but its content lives in
        chunks the query never matched, so without this the model can see that
        `Bug A -[cloned_from]-> Bug B` and still have nothing to say about B.
        """
        index = getattr(self, "_chunks_by_bug", None)
        if index is None:
            index = {}
            bug_col, sec_col = self.chunks.cols["bug_id"], self.chunks.cols["section"]
            for i in range(self.chunks.n):
                index.setdefault(bug_col[i].as_py(), []).append(i)
            order = {s: r for r, s in enumerate(self._SUMMARY_SECTIONS)}
            for rows in index.values():
                rows.sort(key=lambda i: (order.get(sec_col[i].as_py(), len(order)), i))
            self._chunks_by_bug = index
        out = []
        for bug_id in dict.fromkeys(str(b) for b in bug_ids):
            for i in index.get(bug_id, [])[:per_bug]:
                out.append(self.chunks.row(i, None))
        return out

    def all_bug_links(self) -> list[dict[str, Any]]:
        """Every bug-to-bug edge in the corpus. Built once and cached.

        Traversal finds the links near a question's vector hits, which answers
        "what was this bug cloned from" but not "which clones outlived their
        original" -- that one names no bug to expand from and needs the whole
        edge set. There are a few dozen of them, so the join is exhaustive
        rather than sampled.
        """
        cached = getattr(self, "_all_bug_links", None)
        if cached is not None:
            return cached
        subj, pred, obj = (self.triples.cols["s"], self.triples.cols["p"],
                           self.triples.cols["o"])
        out = []
        for i in range(self.triples.n):
            if pred[i].as_py() not in LINK_PREDICATES:
                continue
            if str(subj[i].as_py()).startswith("Bug ") and str(obj[i].as_py()).startswith("Bug "):
                out.append(self.triples.row(i, _TRIPLE_MAP))
        self._all_bug_links = out
        return out

    def bug_facets(self) -> list[dict[str, Any]]:
        """One row per bug carrying its facets. Built once and cached."""
        cached = getattr(self, "_bug_facets", None)
        if cached is not None:
            return cached
        cols = {f: self.chunks.cols[f] for f in self._BUG_FACETS
                if f in self.chunks.cols}
        by_bug: dict[str, dict[str, Any]] = {}
        for i in range(self.chunks.n):
            bug_id = cols["bug_id"][i].as_py()
            if bug_id in by_bug:
                continue
            by_bug[bug_id] = {f: c[i].as_py() for f, c in cols.items()}
        self._bug_facets = list(by_bug.values())
        return self._bug_facets

    # ------------------------------------------------------------------ BM25

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        cols = [self.chunks.cols[f] for f in _FT_FIELDS if f in self.chunks.cols]
        docs = []
        for i in range(self.chunks.n):
            parts = []
            for c in cols:
                v = c[i].as_py()
                if v:
                    parts.append(v if isinstance(v, str) else " ".join(map(str, v)))
            docs.append(_TOKEN.findall(" ".join(parts).lower()))
        self._bm25 = BM25Okapi(docs)

    def fulltext_chunks(self, keyword: str, k: int = 10) -> list[dict[str, Any]]:
        """Local stand-in for Cosmos FullTextContains over the chunks container."""
        if self._bm25 is None:
            return []
        toks = _TOKEN.findall(keyword.lower())
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [self.chunks.row(int(i), None) for i in idx if scores[i] > 0]


# Keyed by snapshot path rather than a lone slot. As a single instance this
# loaded once and then ignored its argument, so a process could serve exactly
# one corpus: asking for a second snapshot silently returned the first, which
# is a wrong answer rather than an error. Two corpora are 77 MB together, so
# the reason to load once is the load time, not the memory.
_INSTANCES: dict[str, LocalGraphIndex] = {}


def get_index(path: str = "data/local_index", **kw) -> LocalGraphIndex:
    """One index per snapshot path, loaded once and kept for the process."""
    key = os.path.abspath(path)
    if key not in _INSTANCES:
        _INSTANCES[key] = LocalGraphIndex(path, **kw)
    return _INSTANCES[key]
