"""Postgres implementation of the retrieval contract in `retrieval.py`.

Third backend alongside CosmosBackend and LocalBackend, serving the same seven
methods and returning identically shaped dicts, so `retrieve()` and every prompt
downstream are unchanged.

It lives in its own module rather than beside the other two because it is the
only one that needs psycopg: importing it from `retrieval` would make the driver
a hard dependency of the Cosmos and local deployments, which do not use it.

What is different here, and why it matters
------------------------------------------
The other two backends each give up something this one does not have to:

  Cosmos serves one container per concern, so a question that filters on
  attributes *and* ranks by vector distance has to over-fetch from one and
  filter in Python. Here both are columns of the same table, so the filter is
  a WHERE clause on an index and the database does the work.

  LocalBackend holds the corpus in process memory, which bounds it to what one
  machine can hold and makes every incremental load a full re-export.

The cost is a network round trip per call, against Cosmos' one and the local
index's none -- so the methods that `retrieve()` runs concurrently must actually
run concurrently. Hence a real async pool rather than a single connection: with
one connection psycopg serialises them and the "parallel" vector, lexical and
anchor fetches would queue behind each other.

Conventions worth stating once
------------------------------
`score` is cosine *similarity*, highest first -- `1 - (emb <=> q)`, since
pgvector's `<=>` is cosine distance. Cosmos' VectorDistance under a cosine
policy reports similarity and `gi_index._search` deliberately matches it, so
this is the third implementation of one convention rather than a new one.

`bug_id` is text, matching Cosmos. It is a bigint column, but callers compare it
against ids parsed out of `"Bug 123"` strings, and while they mostly `str()`
defensively, one that forgot would silently never match.

Two fields in the chunk shape are null across the main download, and that is
the corpus rather than a mapping bug -- both key names were checked against the
raw JSON. `release` comes from a "Release" custom field that the general corpus
does not carry (0 of 3,000 sampled bugs have bugCustomFieldsInfo), and
`bug.attachments` is null in all 572 sampled raw records because the bulk
endpoint does not return attachment metadata. They are real columns rather than
aliased constants because the benchmark corpora do populate them, and
`structured_filter.FACET_COLUMNS` names `release` as something a question may
filter on.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import pgconn
from gi_index import SUMMARY_SECTIONS
from retrieval import (CAP_CHUNKS_FOR_BUGS, CAP_FACETS, CAP_LINKS,
                       CAP_REVERSE_EDGES, _BackendBase)

try:                                    # pgvector >= 0.3
    from pgvector import Vector as _Vector
except ImportError:                     # pragma: no cover - older layout
    from pgvector.utils import Vector as _Vector

# The chunk projection. Spelled out rather than `c.*` for the same reason the
# Cosmos backend spells it out: `emb` is 4 KB of float32 per row and nothing
# downstream reads it, so selecting it would trade the whole point of holding
# text and vectors in one table for a much fatter result set.
_CHUNK_COLS = """
    c.chunk_id                                     as id,
    c.bug_id::text                                 as bug_id,
    c.section,
    c.seq,
    c.text,
    b.synopsis, b.module,
    b.priority, b.priority_rank, b.severity, b.severity_rank,
    b.disposition, b.is_open,
    b.partner_area, b.issue_type, b.ms_status, b.release,
    b.customer_name, b.os, b.version, b.keywords, b.categories,
    b.has_attachments,
    extract(epoch from b.modified_date)::bigint    as modified_ts
"""

# Cheapest sections that say what a bug *is*. Ordered, so "the first two chunks
# of this bug" means synopsis then description rather than whichever the
# planner returned.
_SECTION_ORDER = """
    case c.section
         when 'synopsis'    then 0
         when 'description' then 1
         when 'repro'       then 2
         else 3 end
"""

# Vector search over chunks. No `where emb is not null`: a null vector makes the
# distance null, nulls sort last under ASC, so they cannot reach the top k
# anyway -- and adding the predicate risks the planner choosing a filter+sort
# over the diskann index scan, which is the difference between milliseconds and
# a sequential scan of a million rows.
_SEARCH_CHUNKS = f"""
select {_CHUNK_COLS}, 1 - (c.emb <=> %(q)s) as score
  from chunks c
  join bugs b using (bug_id)
 order by c.emb <=> %(q)s
 limit %(k)s
"""

# Vector search over bug nodes, then their summary chunk ids.
#
# Two levels on purpose. `source_chunks` is what turns a node hit into quotable
# passages, but joining it in the same level as the ORDER BY/LIMIT invites the
# planner to build the array for every candidate row it considers rather than
# the k it returns. The CTE settles which k first; the lateral runs k times.
_SEARCH_ENTITIES = f"""
with top as (
    select e.entity_id, e.name, e.emb_text, e.degree,
           1 - (e.emb <=> %(q)s) as score
      from entities e
     order by e.emb <=> %(q)s
     limit %(k)s
)
select t.name,
       t.emb_text                        as description,
       t.degree                          as relation_count,
       coalesce(s.ids, '{{}}'::text[])   as source_chunks,
       t.score
  from top t
  left join lateral (
        select array_agg(c.chunk_id order by {_SECTION_ORDER}, c.seq) as ids
          from chunks c
         where c.bug_id = (case when t.entity_id like 'bug:%%'
                                then split_part(t.entity_id, ':', 2)::bigint end)
           and c.section = any(%(sections)s)
       ) s on true
 order by t.score desc
"""

# Edges out of the named bugs.
#
# `source_chunks` carries the *subject's* summary chunks: an edge is evidence
# about the bug it leaves, and without it a traversal hit contributes a
# relationship the prompt cannot quote anything for.
#
# `confidence` is 1.0 because these edges are not inferred. Cosmos' triples came
# from an LLM extraction pass and carried its confidence; these are nvbugspro's
# own related-bugs payload restated, so anything below 1.0 would be inventing
# doubt the source does not express.
#
# The per-node cap is applied here rather than in Python: ordering puts edges
# whose far bug we actually hold first, since an edge to a bug we can quote is
# worth more to an answer than one we can only name.
_TRIPLES_FOR = f"""
select 'Bug ' || l.src_bug_id            as subject,
       l.predicate,
       'Bug ' || l.dst_bug_id            as object,
       1.0::float                        as confidence,
       coalesce(s.ids, '{{}}'::text[])   as source_chunks,
       1.0::float                        as weight,
       l.dst_synopsis, l.dst_module, l.dst_is_closed, l.dst_is_fixed,
       l.dst_on_record
  from (
        select l.*,
               row_number() over (partition by l.src_bug_id
                                  order by l.dst_on_record desc,
                                           l.predicate, l.dst_bug_id) as rn
          from bug_links l
         where l.src_bug_id = any(%(ids)s)
       ) l
  left join lateral (
        select array_agg(c.chunk_id order by {_SECTION_ORDER}, c.seq) as ids
          from chunks c
         where c.bug_id = l.src_bug_id
           and c.section = any(%(sections)s)
       ) s on true
 where %(cap)s is null or l.rn <= %(cap)s
"""

# Attribute edges out of the named bugs: what the record says a bug *is*, as
# opposed to which other bugs it points at.
#
# `source_chunks` is read off the row rather than joined, because the builder
# already stored the subject's summary chunk ids there -- the same four chunks
# this file's `_TRIPLES_FOR` assembles with a lateral per call, settled once at
# build time for edges that outnumber link edges three to one.
#
# Hub edges are kept here regardless of `include_hubs`. Suppression exists to
# stop a traversal *arriving* through a popular node and fanning out to every
# bug beneath it; an edge leaving a bug we already hold describes that bug, and
# dropping "has severity 3-Functionality" would withhold the field rather than
# bound anything. Its 0.2 weight already says how little it distinguishes.
_ATTRS_FOR = """
select t.s                               as subject,
       t.p                               as predicate,
       t.o                               as object,
       t.confidence::float               as confidence,
       coalesce(t.source_chunks, '{}'::text[]) as source_chunks,
       t.weight::float                   as weight
  from (
        select t.*,
               row_number() over (partition by t.s
                                  order by t.weight desc, t.p, t.o) as rn
          from triples t
         where t.s = any(%(names)s)
           and not t.is_link
       ) t
 where %(cap)s is null or t.rn <= %(cap)s
"""

# The reverse direction, and the reason the attribute half of the graph is worth
# building: from an object the question matched -- "1-System Crash", "Azure" --
# back to the bugs that assert it. Without this an attribute-level seed is a
# dead end, since `bug_links` holds no attribute edges to walk at all.
#
# Here hub suppression is the whole bound. A hub object is one thousands of bugs
# point at, so walking into it returns thousands of bugs that share only a
# status field -- unless the question named that object itself, which is what
# `include_hubs` means at hop 0. `weight desc` then puts the objects that
# actually distinguish a bug first within the per-node cap.
#
# No `dst_*` columns, unlike the two queries above. Those describe the far end
# of an edge, and on a reverse edge the far end is the *subject* -- a bug -- so
# naming its synopsis `dst_synopsis` would put the far bug's fields under keys
# that mean the opposite everywhere else. Nothing needs them: the row's
# `source_chunks` are the subject bug's own summary chunks, so a reverse hit is
# already quotable without a second lookup.
# One bounded lookup per name rather than one window over all of them. The
# `row_number() over (partition by o)` this replaces had to read and sort every
# edge pointing at every requested attribute before it could number them, and
# only then throw all but 25 of each away. Walking back from the 30 attributes
# of a single bug meant sorting a few hundred thousand rows to return 476: 41.2s,
# against 4.3s here for byte-identical output. Some attributes are large -- the
# biggest non-hub object has 48,297 edges into it -- and the window's cost was
# the sum of them, where LATERAL's is the sum of what it keeps.
#
# The null cap means "no limit" and is spelled as a sentinel rather than a
# branch, so both cases plan the same way.
_ATTRS_INTO = """
select t.s                               as subject,
       t.p                               as predicate,
       t.o                               as object,
       t.confidence::float               as confidence,
       coalesce(t.source_chunks, '{}'::text[]) as source_chunks,
       t.weight::float                   as weight
  from unnest(%(names)s::text[]) as n(name)
  cross join lateral (
        select t.*
          from triples t
         where t.o = n.name
           and not t.is_link
           and (%(include_hubs)s or not t.is_hub)
         order by t.weight desc, t.s
         limit coalesce(%(cap)s::int, 2147483647)
       ) t
"""

# Vector search over edges. The sentence is the builder's and the reference's --
# "Bug 5009640 has module KMD - core" -- so a question can match an edge
# directly instead of only through one of its endpoints.
_SEARCH_TRIPLES = """
select t.s                               as subject,
       t.p                               as predicate,
       t.o                               as object,
       t.confidence::float               as confidence,
       coalesce(t.source_chunks, '{}'::text[]) as source_chunks,
       t.weight::float                   as weight,
       1 - (t.emb <=> %(q)s)             as score
  from triples t
 order by t.emb <=> %(q)s
 limit %(k)s
"""

_DOCS_BY_ID = f"""
select {_CHUNK_COLS}, null::float as score
  from chunks c
  join bugs b using (bug_id)
 where c.chunk_id = any(%(ids)s)
"""

# Lexical search. ts_rank_cd rather than ts_rank: cover density rewards hits
# that appear close together, which is what distinguishes a passage about the
# query from one that mentions its words in unrelated paragraphs.
#
# Ranking is confined to a bounded candidate set, and on the 1M corpus that is
# the difference between this query being the whole response time and being
# noise. `order by ts_rank_cd(...) limit 10` has to score every match before it
# can know the top ten, so cost tracks how common the term is rather than how
# many rows come back: "bug" matches 900,514 of 4.36M chunks and took 58.9s,
# while "nvlink" at 33,177 took 0.5s. Every question of the form "what is bug
# 6081965 about" contains the word bug, so that 59s was being paid on the
# critical path of the most ordinary question the corpus gets asked.
#
# `cand` has no ORDER BY, so the GIN scan stops once it has `cap` rows instead
# of materialising the whole match bitmap -- measured 0.68s for the same "bug"
# query, an 86x difference. Below the cap the result is exactly what it was
# before, globally ranked; above it, the top-k comes from the first `cap`
# matches the index yields rather than from all of them. That is a real loss of
# precision and an acceptable one, because a cover-density ranking spread across
# a fifth of the corpus is not carrying information anyway -- and the terms
# selective enough for the ranking to mean something are exactly the ones that
# never reach the cap.
_FULLTEXT = f"""
with q as (
    select websearch_to_tsquery('english', %(kw)s) as tsq
),
cand as (
    select c.chunk_id
      from chunks c, q
     where c.tsv @@ q.tsq
     limit %(cap)s
)
select {_CHUNK_COLS}, ts_rank_cd(c.tsv, q.tsq) as score
  from cand
  join chunks c using (chunk_id)
  join bugs b using (bug_id),
       q
 order by score desc
 limit %(k)s
"""

# Rows the lexical scan may consider before it stops looking. 3,000 is where the
# worst case measured 0.68s on the 1M corpus; the cost is roughly linear in this
# number, so it is the knob to turn if lexical recall on common terms matters
# more than the tail of the response time.
_FT_CANDIDATE_CAP = 3000

# One lateral per bug id, so each bug costs an index lookup capped at `per_bug`
# rather than the whole set being sorted and windowed. `unnest` keeps it a
# single round trip regardless of how many bugs were asked for.
_CHUNKS_FOR_BUGS = f"""
select {_CHUNK_COLS}, null::float as score
  from unnest(%(ids)s::bigint[]) as u(bug_id)
  join bugs b on b.bug_id = u.bug_id
  join lateral (
        select c.chunk_id, c.bug_id, c.section, c.seq, c.text
          from chunks c
         where c.bug_id = u.bug_id
           and c.section = any(%(sections)s)
         order by {_SECTION_ORDER}, c.seq
         limit %(per_bug)s
       ) c on true
"""

# `system` is aliased from `system_name`, the one name in gi_index._BUG_FACETS
# that differs from its column; structured_filter reads these keys by name.
_BUG_FACETS = """
select b.bug_id::text as bug_id, b.synopsis, b.module,
       b.priority, b.priority_rank, b.severity, b.severity_rank,
       b.disposition, b.is_open,
       b.partner_area, b.issue_type, b.ms_status, b.release,
       b.system_name  as system,
       b.customer_name, b.os, b.version,
       extract(epoch from b.modified_date)::bigint as modified_ts
  from bugs b
"""

_ALL_BUG_LINKS = """
select 'Bug ' || l.src_bug_id as subject,
       l.predicate,
       'Bug ' || l.dst_bug_id as object,
       1.0::float             as confidence,
       '{}'::text[]           as source_chunks,
       1.0::float             as weight
  from bug_links l
"""

_DIGITS = re.compile(r"^\s*(\d{4,})\s*$")


class PostgresBackend(_BackendBase):
    """The retrieval contract served from one relational store."""

    # All three of these are ordinary SQL here, which is the point of the store:
    # facets are columns, links are rows, and chunk text sits beside its vector.
    #
    # On CAP_REVERSE_EDGES. For bug-to-bug edges it would be redundant: ingest
    # materialises both directions -- verified on the loaded corpus, 0 of 177,950
    # edges lack an inverse and the predicate pairs are exactly symmetric (6,191
    # duplicate_of / 6,191 has_duplicate, 2,586 gating / 2,586 gated_by, 1,505
    # each way for clones) -- so a forward query already returns what a reverse
    # one would, under the inverse predicate name, and declaring it would return
    # each relationship twice and inflate `max_triples` with duplicates.
    #
    # Attribute edges are the opposite case. `Bug 5009640 has severity 1-System
    # Crash` has no materialised inverse and no useful one to materialise: the
    # reverse is not another edge but a set of thousands. Walking it is what
    # turns an attribute the question matched into the bugs underneath it, so the
    # capability is declared when a corpus actually has attribute edges -- see
    # `_probe`, which is also why this is an instance attribute now rather than
    # a class one. A database with only `bug_links` behaves exactly as before.
    capabilities = frozenset({CAP_CHUNKS_FOR_BUGS, CAP_FACETS, CAP_LINKS})

    def __init__(self, dsn: str | None = None, *, min_size: int = 2,
                 max_size: int = 8, sections: tuple[str, ...] = SUMMARY_SECTIONS,
                 reverse_edges: bool = True):
        super().__init__()
        self._dsn = dsn or pgconn.dsn()
        self._min, self._max = min_size, max_size
        self._sections = list(sections)
        self._pool = None
        self._lock = asyncio.Lock()
        # Whether to walk attribute edges backwards by default, mirroring
        # LocalBackend's constructor flag. It only takes effect on a corpus that
        # has attribute edges to walk.
        self.reverse_edges = reverse_edges
        # Set by `_probe` on first use: whether this database has a `triples`
        # table, and whether anything in it is embedded. Both are properties of
        # the corpus rather than of the code -- the 175k database predates the
        # table, and the 1M one has the edges long before it has their vectors.
        self._has_triples: bool | None = None
        self._has_triple_vecs = False
        # `_ix` is the attribute api.py reads for the structured filter and the
        # link inventory, the same name CosmosBackend and LocalBackend use.
        self._ix = PostgresExtract(self._dsn)

    # ------------------------------------------------------------------ pool

    async def _ready(self):
        """Open the pool on first use.

        Built here rather than in __init__ because an async pool binds to the
        running event loop, and backends are constructed during config parsing
        -- before there is one.
        """
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    from pgvector.psycopg import register_vector_async
                    from psycopg_pool import AsyncConnectionPool

                    async def configure(conn):
                        # Registers the vector adapters, so a Python list of
                        # floats binds as `vector` without being rendered to a
                        # 20 KB string literal per query.
                        await register_vector_async(conn)

                    # autocommit is not a detail here. Without it every
                    # `pool.connection()` block opens and commits a
                    # transaction, which on a link with a 170 ms round trip
                    # costs two extra trips per call -- measured at 544 ms for
                    # an eight-row lookup whose server-side cost is 2 ms. Every
                    # method on this backend is a single read-only statement,
                    # so there is no transaction to want: retrieval reads a
                    # snapshot and writes nothing.
                    # One worker per connection, and a deadline scaled to how
                    # many there are. Both exist because a connection to Azure
                    # Postgres from here costs 7.1s to establish: with the
                    # default three workers and the default 30s wait, a pool
                    # asked for eight cannot fill in time, and psycopg_pool
                    # responds to that timeout by *closing* the pool -- so
                    # every subsequent query fails with PoolClosed rather than
                    # merely having been slow to start. Opening them in
                    # parallel turns 8x7.1s of sequential handshakes into one.
                    self._pool = AsyncConnectionPool(
                        self._dsn, min_size=self._min, max_size=self._max,
                        kwargs={"autocommit": True},
                        num_workers=max(3, self._min),
                        configure=configure, open=False)
                    await self._pool.open(
                        wait=True, timeout=max(30.0, 15.0 * self._min))
        return self._pool

    async def close(self):
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _rows(self, sql: str, params: dict | None = None) -> list[dict]:
        """Run one statement and return dicts keyed by column name."""
        pool = await self._ready()
        async with pool.connection() as conn:
            cur = conn.cursor()
            await cur.execute(sql, params or {})
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]

    async def _probe(self) -> bool:
        """Once per process: does this corpus have an attribute graph?

        Asked of the database rather than configured, because the answer differs
        per database on one server and changes as a build progresses. Two
        separate facts: the table's existence decides whether traversal has
        attribute edges at all, and whether any row is embedded decides whether
        `search_triples` can rank them -- the 1M corpus will have 23M edges and
        no vectors, and a vector search there would order by a null distance and
        return an arbitrary k with null scores, which is worse than returning
        nothing.

        `to_regclass` rather than catching UndefinedTable: a failed statement
        aborts its transaction, and this pool runs in autocommit precisely so
        that no call has to think about that.
        """
        if self._has_triples is None:
            rows = await self._rows(
                "select to_regclass('public.triples') is not null as present")
            self._has_triples = bool(rows and rows[0]["present"])
            if self._has_triples:
                # Bounded to a prefix, because the negative answer is the
                # expensive one. `exists (select 1 ... where emb is not null)`
                # returns the instant it finds a vector, but when there are none
                # it has to prove that by reading all 16,078,856 rows -- 32s,
                # paid once, in the middle of whichever user's question happened
                # to arrive first, and looking exactly like a 46s graph
                # traversal. There is no index on `emb` to help: the build's
                # partial index covers `emb is null`, which is every row here.
                #
                # A prefix answers the question that matters. If a meaningful
                # share of the corpus is embedded, the first 50,000 rows contain
                # a vector; if they do not, `search_triples` would be ranking a
                # rounding error and the honest answer is the one this gives.
                rows = await self._rows(
                    "select exists (select 1 from "
                    "  (select emb from triples limit 50000) s "
                    " where emb is not null) as any_emb")
                self._has_triple_vecs = bool(rows and rows[0]["any_emb"])
                if self.reverse_edges:
                    self.capabilities = frozenset(
                        self.capabilities | {CAP_REVERSE_EDGES})
        return self._has_triples

    # -------------------------------------------------------------- vectors

    async def search_entities(self, q_emb, k):
        return await self._rows(_SEARCH_ENTITIES,
                                {"q": _vec(q_emb), "k": int(k),
                                 "sections": self._sections})

    async def search_triples(self, q_emb, k):
        """Rank edges by their sentence, when the corpus has embedded them.

        Cosmos embedded each extracted triple, so a question could match an edge
        directly. That was unavailable here for as long as the only edges were
        `bug_links` rows, which have no text of their own; `triples` gives them
        the reference's sentence and the vector that goes with it.

        On a corpus whose edges are not embedded this returns empty rather than
        recording a degradation, as it did when no corpus had them: nothing was
        attempted and failed, so reporting one would put a warning on every
        answer. The reach is not lost either way -- `search_entities` ranks the
        bugs and `triples_for` returns their edges, which is the same evidence
        arrived at through the node instead of the edge.
        """
        if not await self._probe() or not self._has_triple_vecs:
            return []
        return await self._rows(_SEARCH_TRIPLES, {"q": _vec(q_emb), "k": int(k)})

    async def search_chunks(self, q_emb, k):
        return await self._rows(_SEARCH_CHUNKS, {"q": _vec(q_emb), "k": int(k)})

    # ---------------------------------------------------------------- graph

    async def triples_for(self, names, reverse=None, include_hubs=False,
                          max_per_node=None):
        """Every edge touching the named nodes: links out, attributes both ways.

        A name is one of two things and is dispatched on which. `"Bug 5009640"`
        is a bug, and its edges are its links (from `bug_links`, which carries
        the far bug's synopsis and status on the row) plus what the record says
        it is. Anything else -- `"1-System Crash"`, `"Azure"`, an engineer's
        name -- is an attribute object, and the only edges it has are the bugs
        pointing at it, which is a reverse walk by definition.

        The three queries are independent, so they go together rather than in
        sequence: on a link with a 170 ms round trip, running them one after
        another would cost the traversal two extra RTTs per hop.

        `include_hubs` bounds only the reverse walk, and `max_per_node` bounds
        each query per node; see the notes on the statements themselves for why
        the two differ.
        """
        names = [str(n) for n in (names or []) if str(n).strip()]
        if not names:
            return []
        cap = int(max_per_node) if max_per_node else None
        rev = self.reverse_edges if reverse is None else bool(reverse)

        tasks = []
        ids = _bug_ids(names)
        if ids:
            tasks.append(self._rows(_TRIPLES_FOR,
                                    {"ids": ids, "cap": cap,
                                     "sections": self._sections}))
        if await self._probe():
            bugs = [n for n in names if _bug_ids([n])]
            if bugs:
                tasks.append(self._rows(_ATTRS_FOR,
                                        {"names": bugs, "cap": cap}))
            # Only non-bug names are walked backwards. A bug's inverse links are
            # already materialised, and its attribute edges point away from it
            # by construction, so a reverse query keyed on a bug would return
            # nothing while costing a round trip.
            others = [n for n in names if not _bug_ids([n])]
            if rev and others:
                tasks.append(self._rows(
                    _ATTRS_INTO, {"names": others, "cap": cap,
                                  "include_hubs": bool(include_hubs)}))
        if not tasks:
            return []
        out: list[dict] = []
        for batch in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(batch, BaseException):
                # One direction failing must not cost the others: a traversal
                # with the link edges and without the attribute edges is a
                # worse answer, not a failed request.
                self.degraded.append(f"triples_for: {type(batch).__name__}: {batch}")
                continue
            out += batch
        return out

    # ---------------------------------------------------------------- chunks

    async def docs_by_id(self, ids):
        ids = [str(i) for i in ids]
        if not ids:
            return []
        rows = await self._rows(_DOCS_BY_ID, {"ids": ids})
        # Returned in the order asked for: `retrieve()` accumulates evidence
        # ranked-best-first and cuts at `max_source`, so letting the planner's
        # order through would make the cut fall somewhere arbitrary.
        by_id = {r["id"]: r for r in rows}
        found = [by_id[i] for i in ids if i in by_id]
        if len(found) != len(ids):
            missing = len(ids) - len(found)
            self.degraded.append(
                f"{missing} of {len(ids)} source chunks not found by id")
        return found

    async def fulltext_chunks(self, keywords, k):
        """Lexical search, one query per keyword, matching LocalBackend.

        A bare bug number is handled separately. `tsv` is built from chunk text
        alone, so the number of the bug a chunk belongs to is not in its own
        lexical index -- "what is bug 6539931 about" would match only other
        bugs' comment threads citing it, and never the bug itself. The local
        backend's BM25 indexes `bug_id` as a field and does not have this gap;
        resolving the number against `bug_id` directly is the same answer by a
        cheaper route than widening a million-row tsvector.
        """
        tasks, direct = [], []
        for kw in keywords or []:
            m = _DIGITS.match(str(kw))
            if m:
                direct.append(int(m.group(1)))
            else:
                tasks.append(self._rows(_FULLTEXT, {"kw": str(kw), "k": int(k),
                                                    "cap": _FT_CANDIDATE_CAP}))
        if direct:
            tasks.append(self.chunks_for_bugs(direct, per_bug=int(k)))
        out: list[dict] = []
        for batch in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(batch, BaseException):
                # One malformed keyword must not cost the other keywords'
                # results, nor the vector search running beside this.
                self.degraded.append(f"full-text search failed: {batch}")
                continue
            out += batch
        return out

    async def chunks_for_bugs(self, bug_ids, per_bug: int = 3):
        ids = []
        for b in bug_ids or []:
            try:
                ids.append(int(str(b).strip()))
            except ValueError:
                continue
        if not ids:
            return []
        return await self._rows(_CHUNKS_FOR_BUGS,
                                {"ids": ids, "per_bug": int(per_bug),
                                 "sections": self._sections})


class PostgresExtract:
    """Corpus-wide facets and link inventory, read synchronously and cached.

    Hangs off the backend as `_ix` because that is the interface `api.py`
    already reads: `_facet_filter` calls `backend._ix.bug_facets()` and builds a
    `FacetFilter` from it, once per process, without awaiting. CosmosBackend
    puts a `graph_extract.GraphExtract` there and LocalBackend puts the whole
    index; this is the same shape backed by SQL, so both features work here with
    no change to the caller.

    Synchronous on purpose, despite the backend being async: it is read once at
    startup, not per question, so a blocking read costs one pause during warmup
    rather than holding the event loop during serving.

    Both queries return the whole corpus, which is what the in-memory contract
    asks for and is the one place this backend does more work than it needs to.
    A filter is a WHERE clause over indexed columns and a count is a count;
    pushing them down replaces these transfers entirely. Kept whole so the
    backend is a drop-in first, with the pushdown as a following change rather
    than a prerequisite -- and worth doing, because the caller was written when
    the link inventory was "a few dozen" edges and this corpus has 177,950.
    """

    def __init__(self, dsn: str):
        # A resolved libpq connection string, not a database name: psycopg is
        # called directly here rather than through pgconn.connect(), whose first
        # argument is a dbname and would quietly build a nonsense DSN out of one.
        self._dsn = dsn
        self._facets: list[dict[str, Any]] | None = None
        self._links: list[dict[str, Any]] | None = None
        self._counts: dict[str, int] | None = None

    def _fetch(self, sql: str) -> list[dict[str, Any]]:
        import psycopg

        with psycopg.connect(self._dsn, connect_timeout=30,
                             autocommit=True) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def bug_facets(self) -> list[dict[str, Any]]:
        if self._facets is None:
            self._facets = self._fetch(_BUG_FACETS)
        return self._facets

    def corpus_counts(self) -> dict[str, int]:
        """Counts for the UI header, cached for the process.

        Exists because `api.py._corpus_counts` otherwise reads `ix.chunks.n`,
        `ix.entities.n` and `ix.triples.n` -- Arrow table attributes that only a
        local snapshot has, so the header would raise AttributeError here rather
        than degrade. `triples` is the bug-to-bug edge count: these edges are
        what this store has instead of LLM-extracted triples.

        `count(*)` over a million chunks reads the whole table, which is a few
        seconds. That is why this is cached and why it is not called per
        request -- and it is exact, which a `reltuples` estimate would not be.
        """
        if self._counts is None:
            rows = self._fetch("""
                select (select count(*) from bugs)      as bugs,
                       (select count(*) from chunks)    as chunks,
                       (select count(*) from entities)  as entities,
                       (select count(*) from bug_links) as triples
            """)
            self._counts = rows[0] if rows else {}
        return self._counts

    def all_bug_links(self) -> list[dict[str, Any]]:
        """Every bug-to-bug edge.

        No LINK_PREDICATES filter, unlike the local index: there, edges share a
        table with attribute triples and have to be separated out. Here
        `bug_links` is exclusively bug-to-bug, so every row qualifies -- which
        is just as well, because the corpus carries 1,505 `has_clone` edges, a
        predicate LINK_PREDICATES does not list, so filtering by that set would
        silently drop them.
        """
        if self._links is None:
            self._links = self._fetch(_ALL_BUG_LINKS)
        return self._links


def _vec(q_emb):
    """Coerce an embedding to something that binds as `vector`.

    Three spellings reach here: a plain list from the embedding service, a numpy
    array (`tolist`) from a local model, and pgvector's own `Vector` (`to_list`)
    when a stored vector is used as the query, which is how the smoke test makes
    its expected answer knowable.

    The result is a `Vector` rather than a list because a Python list of floats
    binds as `double precision[]`, and there is no `vector <=> double
    precision[]` operator -- the query fails outright rather than falling back
    to something slower, so this is a hard requirement and not a nicety.
    """
    if isinstance(q_emb, _Vector):
        return q_emb
    # pgvector renders `vector` as the text `[1,2,3]` on a connection where its
    # adapters were never registered. Iterating that yields characters, so the
    # failure is a confusing "could not convert string to float: '['" rather
    # than anything about vectors -- worth parsing here so a caller that reads
    # a stored vector back does not have to know which adapters are loaded.
    if isinstance(q_emb, str):
        return _Vector([float(x) for x in q_emb.strip().strip("[]").split(",")])
    for attr in ("tolist", "to_list"):
        fn = getattr(q_emb, attr, None)
        if fn is not None:
            return _Vector(fn())
    return _Vector([float(x) for x in q_emb])


def _bug_ids(names) -> list[int]:
    """Bug ids from `"Bug 123"` names, order-preserving and deduplicated.

    Non-bug names are dropped rather than raising: once facet nodes exist,
    seed entities will include labels like `"Diomedes"`, which have no bug id
    and no edges in `bug_links`.
    """
    out: dict[int, None] = {}
    for n in names or []:
        s = str(n).strip()
        if s.lower().startswith("bug "):
            s = s[4:].strip()
        if s.isdigit():
            out[int(s)] = None
    return list(out)
