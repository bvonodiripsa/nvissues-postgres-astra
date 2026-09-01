"""Single implementation of GI retrieval, shared by the CLI and the web API.

The same five-stage pipeline previously existed in four near-identical copies
(`gi_query.answer`, and three functions in `api.py`). It now lives here once,
behind a backend interface with two implementations:

    CosmosBackend  — the original remote queries
    LocalBackend   — in-process GPU index (see gi_index.py)

Both return identically shaped dicts, so prompt construction downstream is
unchanged and the two can be compared directly.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

# Module-level import is cheap: gi_index defers torch to instantiation.
from gi_index import LINK_PREDICATES, SUMMARY_SECTIONS


@dataclass
class RetrievalResult:
    seed_entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# What a backend can actually serve. Callers used to ask with `hasattr`, which
# conflates three different situations: the backend does not implement the
# feature, it implements it but was not given the data it needs, and it tried
# and failed. All three came out as "no links found", which reads to whoever is
# looking at the answer as a fact about the corpus. A capability is declared
# once, up front, so the difference is available before anything is asked.
CAP_CHUNKS_FOR_BUGS = "chunks_for_bugs"   # summary chunks for a whole bug
CAP_FACETS = "facets"                     # bug_facets(), for the structured filter
CAP_LINKS = "links"                       # all_bug_links(), the link inventory
CAP_REVERSE_EDGES = "reverse_edges"       # triples_for(reverse=True)


class _BackendBase:
    """Shared capability declaration and degradation reporting.

    `degraded` collects one human-readable line per thing that was supposed to
    work on this request and did not. Retrieval stays best-effort -- a failed
    enrichment must not cost the whole answer -- but the failure travels with
    the result instead of being swallowed, so an answer built on less evidence
    than intended can say so.
    """

    capabilities: frozenset[str] = frozenset()

    def __init__(self):
        self.degraded: list[str] = []

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    def take_degraded(self) -> list[str]:
        out, self.degraded = self.degraded, []
        return out


# --------------------------------------------------------------------- Cosmos

_ENTITY_SQL = (
    "SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, "
    "c.d AS source_chunks, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
_TRIPLE_VEC_SQL = (
    "SELECT TOP @k c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, "
    "c.d AS source_chunks, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
# The projection is spelled out rather than SELECT *, so the 1024-float vector
# never crosses the wire. On the chunks container that is the difference
# between a ~2 KB and a ~6 KB row.
_CHUNK_FIELDS = (
    "c.id, c.bug_id, c.section, c.seq, c.text, c.synopsis, c.module, "
    "c.priority, c.priority_rank, c.severity, c.severity_rank, c.disposition, "
    "c.is_open, c.partner_area, c.issue_type, c.ms_status, c.release, "
    "c.customer_name, c.os, c.version, c.keywords, c.categories, "
    "c.has_attachments, c.modified_ts"
)
_CHUNK_VEC_SQL = (
    f"SELECT TOP @k {_CHUNK_FIELDS}, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
# FullTextContains alone is a boolean filter with no relevance order, so a
# bare `TOP @k` returns an arbitrary k of whatever matched -- not the same as
# LocalBackend's real BM25 via rank_bm25. FullTextScore is the actual BM25
# score, but Cosmos only allows it in `ORDER BY RANK` (never in WHERE or
# SELECT), so the shape is: filter with FullTextContains, then rank the
# filtered set by FullTextScore over both fields, fused with RRF.
_FT_SQL = (
    f"SELECT TOP @k {_CHUNK_FIELDS} "
    "FROM c WHERE FullTextContains(c.text, @kw) "
    "OR FullTextContains(c.synopsis, @kw) "
    "ORDER BY RANK RRF(FullTextScore(c.text, @kw), FullTextScore(c.synopsis, @kw))"
)
_STRIP = ("e", "_rid", "_self", "_etag", "_attachments", "_ts")


class CosmosBackend(_BackendBase):
    """Original behaviour: one network round trip per query.

    `extract`, if given, is a `graph_extract.GraphExtract` (or anything
    exposing `bug_facets()`/`all_bug_links()`, e.g. `LocalGraphIndex` in a
    pinch) stored as `self._ix` -- the same attribute name `LocalBackend`
    uses for its full index. `api.py`'s `_facet_filter`/`_bug_links` read
    `backend._ix` generically and don't care which backend it came from, so
    this is what turns the structured filter and the link inventory back on
    in cosmos mode without either feature depending on `LocalGraphIndex`.

    `extract=None` (the default) leaves the facet and link capabilities
    undeclared, so callers skip those features knowingly instead of reading
    an empty result as "this corpus has no links".

    Reverse edges are never offered: `WHERE c.o = @pk` is a cross-partition
    scan of the whole triples container.
    """

    def __init__(self, entities_ctr, triples_ctr, chunks_ctr, triples_pk_field: str = "s",
                extract=None):
        super().__init__()
        self._entities, self._triples, self._chunks = entities_ctr, triples_ctr, chunks_ctr
        self._pk = triples_pk_field
        self._ix = extract
        caps = {CAP_CHUNKS_FOR_BUGS}
        if extract is not None:
            if hasattr(extract, "bug_facets"):
                caps.add(CAP_FACETS)
            if hasattr(extract, "all_bug_links"):
                caps.add(CAP_LINKS)
        self.capabilities = frozenset(caps)

    async def _collect(self, ctr, query, params=None, strip=False):
        out = []
        async for doc in ctr.query_items(query=query, parameters=params or []):
            if strip:
                for k in _STRIP:
                    doc.pop(k, None)
            out.append(doc)
        return out

    async def search_entities(self, q_emb, k):
        return await self._collect(self._entities, _ENTITY_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}])

    async def search_triples(self, q_emb, k):
        return await self._collect(self._triples, _TRIPLE_VEC_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}])

    async def search_chunks(self, q_emb, k):
        return await self._collect(self._chunks, _CHUNK_VEC_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}], strip=True)

    async def triples_for(self, names, reverse=False, include_hubs=False, max_per_node=None):
        # `reverse` would need `WHERE c.o = @pk`, a cross-partition scan over
        # the whole container, so it is not offered remotely. Forward-only also
        # makes `include_hubs` moot here: hub suppression applies to reverse
        # edges, which this backend cannot serve in the first place.
        #
        # One IN-list query instead of one query per name: each hop is capped
        # at 10 names (retrieve()'s frontier width), so this is a single round
        # trip per hop rather than up to 10 -- both fewer chances to hit the
        # container's RU ceiling and less retry-backoff exposure when one does
        # get throttled. TOP can't express "N per subject" over an IN-list (no
        # PARTITION BY in Cosmos SQL), so the per-node cap that used to live in
        # the query is enforced client-side after grouping by subject instead.
        #
        # The cap must not be a plain head-cut of whatever order Cosmos
        # returned. A hub bug carries far more attribute triples
        # (has_attachment, has_keyword) than link triples, so cutting to
        # `max_per_node` unordered drops the duplicate_of/see_also edge the
        # question turned on -- reproduced live on the 1M-scale corpus (19 Aug
        # 2026: Bug 1000233's `see_also -> Bug 1000459` vanished from the
        # prompt under ~30 attribute triples on the same bug). Cosmos will not
        # run the query-side fix: `ORDER BY (c.p IN (...)) DESC, c.id ASC` and
        # `ORDER BY c.p ASC, c.id ASC` both come back "One of the input values
        # is invalid." / "does not have a corresponding composite index"
        # against this account. So each group is sorted link-predicates-first
        # client-side, matching `gi_index.LocalGraphIndex.triples_for` so both
        # backends keep the same edges. A stable sort, so relative order within
        # each of the two classes is whatever the query returned.
        names = list(dict.fromkeys(names))
        if not names:
            return []
        params = [{"name": f"@n{i}", "value": n} for i, n in enumerate(names)]
        sql = (f"SELECT c.s AS subject, c.p AS predicate, c.o AS object, "
               f"c.f AS confidence, c.d AS source_chunks, c.w AS weight "
               f"FROM c WHERE c.{self._pk} IN ({', '.join(p['name'] for p in params)})")
        rows = await self._collect(self._triples, sql, params)
        if not max_per_node:
            return rows
        by_subject: dict[str, list[dict]] = {}
        for t in rows:
            by_subject.setdefault(t.get("subject"), []).append(t)
        # Grouping is by the triple's subject, which equals the requested name
        # only while the partition key is `s`; ordering by `names` first and
        # then draining anything left keeps every group under a different
        # `triples_pk_field` too.
        out = []
        for name in list(names) + [s for s in by_subject if s not in set(names)]:
            group = by_subject.pop(name, None)
            if not group:
                continue
            group.sort(key=lambda t: t.get("predicate") not in LINK_PREDICATES)
            out.extend(group[:int(max_per_node)])
        return out

    async def docs_by_id(self, ids):
        """Fetch chunks by id, routed to the partitions that can hold them.

        The chunks container partitions on `/bug_id`, so a filter on `c.id`
        alone is a cross-partition query: the gateway fans out to every
        physical partition and merges, which at 1M bugs is the difference
        between a few targeted reads and a scan of the container per question.
        A chunk id is `{bug_id}-{section}-{seq}` -- both when built and after
        `populate_scale_db.remap_chunk`, which reassigns `id` from the new
        `bug_id` -- so the partition key is recoverable from the id itself.
        Naming it narrows to those partitions; the `c.id` filter still selects
        the rows within them.

        If a batch comes back short, that convention did not hold for some id,
        and the narrowing would silently drop evidence. So the batch is re-read
        on `c.id` alone and the discrepancy recorded on `self.degraded` rather
        than left to look like a chunk that was never found.
        """
        out = []
        ids = [str(i) for i in ids]
        for start in range(0, len(ids), 20):
            batch = ids[start:start + 20]
            id_params = [{"name": f"@i{n}", "value": i} for n, i in enumerate(batch)]
            id_list = ", ".join(p["name"] for p in id_params)
            bugs = list(dict.fromkeys(i.split("-", 1)[0] for i in batch))
            bug_params = [{"name": f"@b{n}", "value": b} for n, b in enumerate(bugs)]
            bug_list = ", ".join(p["name"] for p in bug_params)
            rows = await self._collect(
                self._chunks,
                f"SELECT {_CHUNK_FIELDS} FROM c "
                f"WHERE c.bug_id IN ({bug_list}) AND c.id IN ({id_list})",
                bug_params + id_params)
            if len(rows) < len(set(batch)):
                fallback = await self._collect(
                    self._chunks,
                    f"SELECT {_CHUNK_FIELDS} FROM c WHERE c.id IN ({id_list})",
                    id_params)
                if len(fallback) > len(rows):
                    self.degraded.append(
                        f"docs_by_id: partition narrowing lost "
                        f"{len(fallback) - len(rows)} of {len(batch)} chunks; "
                        f"chunk ids do not all start with their bug_id")
                    rows = fallback
            out += rows
        return out

    async def fulltext_chunks(self, keywords, k):
        batches = await asyncio.gather(*[
            self._collect(self._chunks, _FT_SQL, [{"name": "@k", "value": k}, {"name": "@kw", "value": kw}])
            for kw in keywords
        ], return_exceptions=True)
        return [d for b in batches if isinstance(b, list) for d in b]

    async def chunks_for_bugs(self, bug_ids, per_bug: int = 3):
        """Summary chunks for whole bugs, addressed by bug id rather than chunk id.

        Mirrors `LocalGraphIndex.chunks_for_bugs`: fetch everything on that
        bug's partition, then keep the cheapest `per_bug` sections client-side
        (Cosmos has no ORDER BY over an arbitrary section-priority list). A
        linked-bug set is capped at `max_linked_bugs` (8 by default) before it
        reaches here, so this used to be up to 8 single-partition reads fired
        together with `asyncio.gather` -- 8 round trips, and 8 independent
        chances to get 429'd. `IN (...)` over the partition key turns it into
        one query while keeping partition-key pushdown (Cosmos routes
        straight to each named bug's partition instead of scanning the whole
        container) -- `ARRAY_CONTAINS(@list, c.bug_id)` looked equivalent but
        measured far worse (14s vs 5s) because the optimizer does not give it
        the same partition-key routing; the array sits on the wrong side of
        the comparison for that pushdown to apply.
        """
        ids = list(dict.fromkeys(str(b) for b in bug_ids))
        if not ids:
            return []
        params = [{"name": f"@id{i}", "value": bid} for i, bid in enumerate(ids)]
        sql = (f"SELECT {_CHUNK_FIELDS} FROM c "
               f"WHERE c.bug_id IN ({', '.join(p['name'] for p in params)})")
        try:
            rows = await self._collect(self._chunks, sql, params, strip=True)
        except Exception as exc:
            # This enrichment is a bonus (the far end of a link gets its own
            # summary instead of just being named), not load-bearing, so one
            # failed batch degrades to nothing rather than failing the whole
            # question -- the same trade-off `return_exceptions=True` made
            # per-bug before batching turned it into a single query. It is
            # reported, though: without that, an answer that could not read the
            # linked bugs is indistinguishable from one where they had nothing
            # to add.
            self.degraded.append(
                f"chunks_for_bugs: could not read summaries for "
                f"{len(ids)} linked bug(s): {type(exc).__name__}: {exc}")
            return []
        # Section rank alone leaves ties (a long description is several chunks,
        # all section="description"), and `per_bug` then keeps an arbitrary
        # subset of them. `seq` is the chunk's position within its section, so
        # this reads a bug from its beginning rather than from the middle.
        order = {s: i for i, s in enumerate(SUMMARY_SECTIONS)}
        by_bug: dict[str, list[dict]] = {}
        for d in rows:
            by_bug.setdefault(str(d.get("bug_id")), []).append(d)
        out = []
        for bid in ids:
            group = by_bug.get(bid, [])
            group.sort(key=lambda d: (order.get(d.get("section"), len(order)),
                                      d.get("seq") or 0))
            out.extend(group[:per_bug])
        return out


# ---------------------------------------------------------------------- local


class LocalBackend(_BackendBase):
    """In-process GPU index. Same interface, no network."""

    def __init__(self, index, reverse_edges: bool = False):
        super().__init__()
        self._ix = index
        self.reverse_edges = reverse_edges
        caps = {CAP_CHUNKS_FOR_BUGS}
        for cap, method in ((CAP_FACETS, "bug_facets"), (CAP_LINKS, "all_bug_links")):
            if hasattr(index, method):
                caps.add(cap)
        if reverse_edges:
            caps.add(CAP_REVERSE_EDGES)
        self.capabilities = frozenset(caps)

    async def search_entities(self, q_emb, k):
        return self._ix.search_entities(q_emb, k)

    async def search_triples(self, q_emb, k):
        return self._ix.search_triples(q_emb, k)

    async def search_chunks(self, q_emb, k):
        return self._ix.search_chunks(q_emb, k)

    async def triples_for(self, names, reverse=None, include_hubs=False, max_per_node=None):
        rev = self.reverse_edges if reverse is None else reverse
        out = []
        for n in names:
            out += self._ix.triples_for(n, reverse=rev, include_hubs=include_hubs,
                                        max_per_node=max_per_node)
        return out

    async def docs_by_id(self, ids):
        return self._ix.docs_by_id(ids)

    async def fulltext_chunks(self, keywords, k):
        out = []
        for kw in keywords:
            out += self._ix.fulltext_chunks(kw, k)
        return out

    async def chunks_for_bugs(self, bug_ids, per_bug=3):
        return self._ix.chunks_for_bugs(bug_ids, per_bug)


# ------------------------------------------------------------------ pipeline



def backend_can(backend, capability: str) -> bool:
    """Whether `backend` declares `capability`.

    A backend that predates `capabilities` (or a test double) is taken at its
    word if it has the method: this is a narrowing of the old `hasattr` check,
    not a new requirement, so it must not turn a working backend off.
    """
    caps = getattr(backend, "capabilities", None)
    if caps is None:
        return hasattr(backend, capability)
    return capability in caps


def _linked_bug_ids(triples: list[dict]) -> list[str]:
    """Bug ids sitting at either end of a bug-to-bug link, in traversal order."""
    out: list[str] = []
    for t in triples:
        if t.get("predicate") not in LINK_PREDICATES:
            continue
        for end in (t.get("subject"), t.get("object")):
            if isinstance(end, str) and end.startswith("Bug "):
                bug_id = end.split(maxsplit=1)[1].strip()
                if bug_id.isdigit() and bug_id not in out:
                    out.append(bug_id)
    return out


async def retrieve(backend, q_emb, cfg: dict, *, keywords: list[str] | None = None,
                   anchor_bugs: list[str] | None = None) -> RetrievalResult:
    """Stages 2-4 of the GI-RAG pipeline. Embedding is the caller's job.

    `anchor_bugs` are bug ids the question named outright. They are seeded into
    the graph directly instead of being left to vector search, which ranks by
    resemblance and has no reason to rank the one bug the question actually
    named above ten that merely read like it.
    """
    q = cfg.get("query", {})
    seed_k = int(q.get("seed_entities_k", 10))
    max_hops = int(q.get("max_hops", 1))
    max_triples = int(q.get("max_triples", 40))
    max_per_node = int(q.get("max_triples_per_node", 25))
    max_source = int(q.get("max_source_chunks", 15))
    vec_k = int(q.get("vector_augment_k", 12))
    max_linked = int(q.get("max_linked_bugs", 8))
    linked_chunks = int(q.get("linked_bug_chunks", 2))
    # How many vector-matched attribute objects to walk back from. Three, not
    # ten: each one returns up to `max_triples_per_node` bugs, and a question
    # naming one attribute should not spend its budget on the four others that
    # merely ranked nearby.
    attr_seeds = int(q.get("attribute_seeds", 3))

    res = RetrievalResult()

    # --- entity search -----------------------------------------------------
    t0 = time.perf_counter()
    seed_entities = await backend.search_entities(q_emb, seed_k)
    res.timings["entity_search"] = time.perf_counter() - t0
    res.seed_entities = seed_entities
    if not seed_entities:
        return res

    # --- graph traversal + triple vector search ----------------------------
    # Vector triple search only needs the question embedding, so it runs beside
    # the PK hop walk instead of waiting for it. Same merge afterward.
    t0 = time.perf_counter()
    vec_triples_task = asyncio.create_task(backend.search_triples(q_emb, 30))
    # Anchors go first so they survive the hop-0 batch cap and are expanded
    # with hubs included, exactly as a node the question named should be.
    anchor_names = [f"Bug {b}" for b in (anchor_bugs or [])]
    names = anchor_names + [e["name"] for e in seed_entities[:10]
                            if e["name"] not in set(anchor_names)]
    visited: set[str] = set()
    pk_triples: list[dict] = []
    for hop in range(max_hops):
        batch = [n for n in names if n not in visited][:10]
        if not batch:
            break
        visited.update(batch)
        # Hop 0 is the set of entities the question itself matched, so a hub
        # among them was asked for by name and is worth expanding. Later hops
        # are nodes we merely arrived at, where the same expansion would drag
        # in hundreds of bugs that share nothing but a status field.
        new_triples = await backend.triples_for(
            batch, include_hubs=(hop == 0), max_per_node=max_per_node
        )
        pk_triples += new_triples
        # Frontier for the *next* hop is this hop's own new nodes, not the
        # hop-0 objects reused forever. Previously this only ran when
        # hop == 0, so any max_hops > 2 was silently a no-op: `names` never
        # advanced past hop 1's frontier, and by hop 2 every name in it was
        # already in `visited`, so `batch` came back empty and the loop
        # always broke there regardless of how high max_hops was set.
        #
        # It also only looked at `t["object"]`, which is wrong for anything
        # discovered through a reverse edge: with reverse_edges on, seed
        # entities are frequently *objects* (claims/tags like "post-workout",
        # "high protein snack") rather than product names, so the triples
        # found are `<product> -[has_claim]-> <seed>` -- the object *is* the
        # seed itself (already visited), and the actually-new node worth
        # exploring next is the subject (the product). Collecting both
        # endpoints and dropping whichever one is already visited handles
        # forward- and reverse-discovered triples the same way.
        # Bugs reached through a link go to the front of the frontier. The
        # frontier is only five nodes wide, and a bug has dozens of attribute
        # edges against at most a handful of link edges, so ordering by
        # whatever the set iterator yields loses the second hop of a clone
        # chain to a release name almost every time.
        ordered: list[str] = []
        linked_names: set[str] = set()
        for linked_only in (True, False):
            for t in new_triples:
                is_link = t.get("predicate") in LINK_PREDICATES
                if is_link is not linked_only:
                    continue
                for end in (t.get("subject", ""), t.get("object", "")):
                    if end and end not in visited and end not in ordered:
                        ordered.append(end)
                        if is_link:
                            linked_names.add(end)
        names = ordered[:5]

        # The triple budget is there to stop the walk wandering through
        # attribute space, not to stop it finishing a chain. Ten seeds at
        # twenty-five edges each exhaust it during hop 0 alone, so checking it
        # before the next hop made max_hops > 1 a no-op: every multi-hop
        # question was silently answered with one hop. Spend it only when
        # nothing linked is left to walk.
        if len(pk_triples) >= max_triples and not (set(names) & linked_names):
            break

    vec_triples = await vec_triples_task

    # An attribute object the question matched by vector -- "1-System Crash",
    # "Azure" -- is a node the question named, as much as a bug number is, and
    # is treated as one: walked backwards, with hubs included.
    #
    # Both halves of that matter. Backwards, because an attribute edge has no
    # materialised inverse and the bugs under an attribute are the answer to
    # "which bugs are severity 1-System Crash" -- forward from the attribute
    # there is nothing. And hubs included, because the attributes a question
    # names are exactly the popular ones: severity, priority and CSP are all
    # hub objects by in-degree, so the suppression that stops a traversal
    # stumbling into a popular node also blocks every question that asks about
    # one. Suppression still applies where it was meant to, on the objects the
    # walk merely arrived at during the hop loop above.
    #
    # This is a separate round trip rather than part of hop 0 because the vector
    # search deliberately runs beside the hop loop; folding it in would mean
    # waiting for it before walking anything.
    rev_triples: list[dict] = []
    named: list[str] = []
    for t in vec_triples:
        obj = t.get("object")
        if (isinstance(obj, str) and obj and not obj.startswith("Bug ")
                and obj not in visited and obj not in named):
            named.append(obj)
    named = named[:attr_seeds]
    if named and backend_can(backend, CAP_REVERSE_EDGES):
        rev_triples = await backend.triples_for(
            named, reverse=True, include_hubs=True, max_per_node=max_per_node)

    seen: set[str] = set()
    merged = []
    # Ordered by how much each source knew about the question before the cut is
    # applied. The seeds' own attribute edges come last: a bug asserts a couple
    # of dozen of them and they describe a bug already found, so ahead of the
    # vector and reverse hits they would spend the whole budget restating the
    # facets of bugs the answer already has.
    pk_links = [t for t in pk_triples if t.get("predicate") in LINK_PREDICATES]
    pk_attrs = [t for t in pk_triples if t.get("predicate") not in LINK_PREDICATES]
    for t in pk_links + vec_triples + rev_triples + pk_attrs:
        key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
        if key not in seen:
            seen.add(key)
            merged.append(t)
    # Link triples are found last and are the fewest, so a plain head-cut
    # discards the hop the question depended on in favour of a hop-0 keyword.
    links = [t for t in merged if t.get("predicate") in LINK_PREDICATES]
    others = [t for t in merged if t.get("predicate") not in LINK_PREDICATES]
    res.triples = (links + others)[:max_triples]
    res.timings["graph_traversal"] = time.perf_counter() - t0
    res.stats["pk_triples"] = len(pk_triples)
    res.stats["vec_triples"] = len(vec_triples)
    res.stats["rev_triples"] = len(rev_triples)

    # --- source documents --------------------------------------------------
    # Chunk vector search only needs the question embedding, so it runs beside
    # the id/linked fetches. Full-text still waits on `keywords` when provided.
    t0 = time.perf_counter()
    # A set has no order, so slicing one to `max_source` picked an arbitrary
    # subset of the evidence: str hashing is salted per process, so the same
    # question asked of two replicas could be answered from different chunks,
    # and a dropped chunk looks exactly like a chunk that was never found.
    # Accumulating in the order the evidence was ranked makes the cut fall on
    # the weakest support instead -- res.triples is already link-first (see
    # above), and seed entities come last because an entity's chunks are the
    # broadest match of the three.
    chunk_ids: dict[str, None] = {}
    for t in res.triples:
        chunk_ids.update(dict.fromkeys(t.get("source_chunks") or []))
    for e in seed_entities[:5]:
        chunk_ids.update(dict.fromkeys(e.get("source_chunks") or []))
    source_ids = list(chunk_ids)[:max_source]

    async def _no_docs() -> list[dict]:
        return []

    docs_task = (backend.docs_by_id(source_ids) if source_ids
                 else _no_docs())
    vec_chunks_task = asyncio.create_task(backend.search_chunks(q_emb, vec_k))
    ft_task = (asyncio.create_task(backend.fulltext_chunks(keywords, 10))
               if keywords else None)
    # A question that names a bug outright ("what is bug 1013122 about?") must
    # see that bug's own text, but graph seeding only helps when the bug has
    # entities/triples -- on a large corpus many bugs have none, so the anchor
    # would otherwise never reach the sources and the model answers "not in the
    # provided context". Fetch the named bugs' chunks directly, overlapping the
    # vector search, and seat them first so they survive the source cap.
    anchor_task = (asyncio.create_task(
                       backend.chunks_for_bugs(list(anchor_bugs), per_bug=4))
                   if anchor_bugs and hasattr(backend, "chunks_for_bugs")
                   else None)

    source_chunks = await docs_task
    seen_ids = {d.get("id") for d in source_chunks}

    if anchor_task is not None:
        anchor_docs = await anchor_task
        prepend = [d for d in anchor_docs if d.get("id") not in seen_ids]
        for d in prepend:
            seen_ids.add(d.get("id"))
        source_chunks = prepend + source_chunks
        res.stats["anchor_chunks"] = len(prepend)

    # A link triple names the bug at the far end but carries only the near
    # bug's chunks, so the model can see that an edge exists and still know
    # nothing about where it leads. Fetch the far end's own summary. Kept
    # after docs_by_id so we don't re-fetch bugs already in hand; still
    # overlaps the in-flight chunk vector search.
    linked = [b for b in _linked_bug_ids(res.triples) if b not in {
        str(d.get("bug_id")) for d in source_chunks}]
    if linked and backend_can(backend, CAP_CHUNKS_FOR_BUGS):
        for doc in await backend.chunks_for_bugs(linked[:max_linked], per_bug=linked_chunks):
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))
        res.stats["linked_bugs"] = len(linked[:max_linked])

    for doc in await vec_chunks_task:
        if doc.get("id") not in seen_ids:
            source_chunks.append(doc)
            seen_ids.add(doc.get("id"))

    if ft_task is not None:
        for doc in await ft_task:
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))

    res.source_chunks = source_chunks
    res.timings["source_fetch"] = time.perf_counter() - t0
    res.stats["source_ids"] = len(source_ids)
    degraded = backend.take_degraded() if hasattr(backend, "take_degraded") else []
    if degraded:
        res.stats["degraded"] = degraded
    return res
