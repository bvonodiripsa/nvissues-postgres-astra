#!/usr/bin/env python
"""
GI-RAG API + Web UI — single graph-index + LLM backend.

Pipeline: entity/triple vector search + graph traversal + LLM keyword expansion +
semantic rerank, then a single LLM answer call (speculative decoding when the
configured model/endpoint supports it).

Config is a single YAML file (default: my.yaml; override with --config).
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openai
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gi_builder import load_config
from graph_extract import GraphExtract
from gi_query import GIQueryEngine
from gi_query import (LINKS_UNAVAILABLE, MENTION_HEADER, NO_BUG_LINKS, NO_CORPUS_STATS,
                      NO_STRUCTURED_SET, comment_mentions, render_answer_prompt,
                      render_bug_links, source_budget)
from retrieval import (CAP_CHUNKS_FOR_BUGS, CAP_FACETS, CAP_LINKS, LINK_PREDICATES,
                       backend_can, retrieve)

_ROOT = Path(__file__).parent
log = logging.getLogger("food_dflash.api")

# Without this the root logger sits at its WARNING default and every log.info
# in this module is discarded -- the keyword expansion, the structured filter's
# predicate and match count, the corpora loaded at startup. All of it looked
# like code that had never run, and diagnosing a disagreement between two
# builds meant adding logging that was already there.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BACKENDS = {
    "gi": {
        "label": "GI-RAG",
        "description": "Graph Index RAG + LLM (speculative decoding when supported)",
        "badge_color": "#059669",
    },
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Retained for API compatibility; there is a single backend now.
    backend: str = Field(default="gi")
    # Which corpus to answer from. Absent means the first one configured, so
    # every existing caller -- the benchmark runners included -- is unchanged.
    corpus: str | None = None


def _load_questions(path: str) -> list[dict]:
    qpath = _ROOT / path
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
    return []


def _load_questions_from_cfg(cfg: dict) -> list[dict]:
    paths = cfg.get("paths", {})
    # Upstream schema uses `questions_path`; keep `questions_file` fallback.
    return _load_questions(paths.get("questions_path",
                                     paths.get("questions_file",
                                               "data/nvissues_questions.json")))


def _corpora_from_cfg(cfg: dict) -> list[dict]:
    """The corpora to serve, in menu order.

    A `corpora:` list makes one deployment serve several; without it the single
    `index.snapshot_path` / `cosmos.database_name` is the only corpus.

    Cosmos-mode entries identify a corpus by `database_name` (no local snapshot
    required). Local-mode entries still require `snapshot_path` + manifest.
    """
    listed = cfg.get("corpora") or []

    # CORPORA restricts the config's list to a comma-separated subset, in the
    # order given. One config can then serve one corpus in one deployment and
    # all of them in another, instead of a near-duplicate config per corpus that
    # drifts from the original the first time a query parameter is retuned.
    #
    # It matters for startup cost, not tidiness: an engine is constructed per
    # corpus, and with `use_structured_filter` each holds one facet row per bug
    # in the process. Serving the 175,018-bug corpus alongside a 1,000-bug slice
    # means paying for the former to answer questions about the latter.
    only = [c.strip() for c in os.environ.get("CORPORA", "").split(",") if c.strip()]
    if only and listed:
        by_id = {str(e.get("id")): e for e in listed}
        unknown = [c for c in only if c not in by_id]
        if unknown:
            log.warning("CORPORA names no such corpus: %s (have %s)",
                        ", ".join(unknown), ", ".join(by_id))
        listed = [by_id[c] for c in only if c in by_id]
        if not listed:
            raise RuntimeError(
                f"CORPORA={os.environ['CORPORA']!r} matched none of the "
                f"configured corpora ({', '.join(by_id)})")
        log.info("CORPORA restricts serving to: %s",
                 ", ".join(str(e.get("id")) for e in listed))

    mode = str(cfg.get("index", {}).get("mode", "cosmos")).lower()
    cosmos_mode = mode == "cosmos"
    # Postgres corpora are identified the same way -- by a database name rather
    # than a snapshot directory -- so they take the same no-snapshot path below.
    pg_mode = mode in ("postgres", "pg")
    default_db = (cfg.get("cosmos") or {}).get("database_name") or "nvissues"
    if pg_mode:
        default_db = (str((cfg.get("index") or {}).get("dbname") or "").strip()
                      or os.environ.get("PGDATABASE") or "nvissues")
    if not listed:
        return [{
            "id": "default",
            "label": cfg.get("index", {}).get("label", "Bug corpus"),
            "snapshot_path": cfg.get("index", {}).get("snapshot_path",
                                                      "data/local_index"),
            "database_name": default_db,
            "questions_path": None,
        }]
    out = []
    for i, entry in enumerate(listed):
        cid = str(entry.get("id") or f"corpus{i}")
        label = str(entry.get("label") or cid)
        snapshot = str(entry.get("snapshot_path") or "").strip()
        database = str(entry.get("database_name") or "").strip() or default_db
        if cosmos_mode or pg_mode:
            # Neither store needs a local Graph Index snapshot.
            out.append({
                "id": cid,
                "label": label,
                "snapshot_path": snapshot or "data/local_index",
                "database_name": database,
                "questions_path": entry.get("questions_path"),
                # Corpora that share a Cosmos database with a much bigger
                # corpus (e.g. a small demo subset uploaded into the same
                # database that also holds the 1M-bug scale fill) cannot get
                # an accurate bug/chunk/entity/triple count from a plain
                # COUNT(1) over the shared containers -- that counts
                # everything in the database, not just this corpus's bugs.
                # `stats` lets the config state the true, already-published
                # counts for that subset instead of a misleading live count.
                "stats": entry.get("stats"),
                # Per-corpus override for index.extract_path (facets + link
                # inventory). Corpora sharing a database need their own
                # extract scoped to just their bugs -- a global extract_path
                # would either be the wrong corpus's facets or, worse,
                # silently shared between two corpora that shouldn't be.
                "extract_path": entry.get("extract_path"),
            })
            continue
        if not snapshot or not (_ROOT / snapshot / "manifest.json").exists():
            log.warning("Corpus %r skipped: no snapshot at %s",
                        cid, snapshot or "(unset)")
            continue
        out.append({
            "id": cid,
            "label": label,
            "snapshot_path": snapshot,
            "database_name": database,
            "questions_path": entry.get("questions_path"),
            "stats": entry.get("stats"),
            "extract_path": entry.get("extract_path"),
        })
    return out


def _merge_keywords(basic: list[str], expanded: list[str] | None) -> list[str]:
    """The question's own keywords, then the LLM's, deduplicated in order.

    This was `list(set(...))`, which made the answer depend on the process's
    string hash seed: full-text search fetches ten chunks per keyword in the
    order given, the merged passages are then cut to a character budget, so the
    order decides which passages the model is actually shown. Two processes
    serving the same corpus answered the same question differently and
    consistently -- one naming the recorded duplicate six times out of six, the
    other missing it six times out of six -- because each had its own seed and
    kept it for life. Nothing about the corpus or the question had changed.
    """
    return list(dict.fromkeys(basic[:5] + (expanded or [])[:6]))


def _resolve(corpus: str | None):
    """The (engine, questions) pair for a corpus id, defaulting to the first."""
    corpora = app.state.corpora
    entry = corpora.get(corpus or "") or corpora[app.state.default_corpus]
    return entry


# ---------------------------------------------------------------------------
# GI-RAG streaming (single Graph Index + LLM backend)
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM_PROMPT = (
    "You are an NVIDIA bug triage engineer. Answer only from the supplied bug "
    "data and never invent a bug number."
)

_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(model: str):
    """Best-effort exact tokenizer for `model`, cached per process.

    Loaded from the local HF cache only (vLLM already pulled it to serve
    `model`) -- this must never make a network call or otherwise block or
    fail an answer over a token-count estimate. Falls back to None, which
    tells the caller to estimate by character count instead.
    """
    if model in _tokenizer_cache:
        return _tokenizer_cache[model]
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as e:
        log.warning("No local tokenizer for %r, falling back to a character "
                    "estimate for prompt sizing: %s", model, e)
    _tokenizer_cache[model] = tok
    return tok


def _count_tokens(model: str, text: str) -> int:
    tok = _get_tokenizer(model)
    if tok is not None:
        try:
            return len(tok.encode(text))
        except Exception:
            pass
    # ~3.2 chars/token deliberately over-counts English/code-mixed bug text --
    # better to under-request completion tokens than to blow the context
    # window when no exact tokenizer is available.
    return int(len(text) / 3.2) + 1


def _capped_llm_kwargs(engine, system_prompt: str, user_prompt: str) -> dict:
    """`engine._llm_call_kwargs` with its token budget clamped to whatever the
    model's context window has left after this specific prompt.

    `query.max_answer_tokens` is one fixed number, but retrieval is corpus-
    and question-dependent -- a wide COMPLETE MATCHING SET or a big SOURCE
    DOCUMENTS window can push the prompt itself past ten thousand tokens, and
    vLLM (`--max-model-len`) rejects the call outright with a 400 rather than
    truncating anything, the moment prompt + requested completion tokens
    exceeds it. A shorter but real answer beats a hard failure.
    """
    kwargs = dict(engine._llm_call_kwargs)
    key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    requested = kwargs.get(key)
    if not requested:
        return kwargs
    max_model_len = int((engine._cfg.get("llm") or {}).get("max_model_len", 16384))
    prompt_tokens = (_count_tokens(engine._llm_model, system_prompt)
                     + _count_tokens(engine._llm_model, user_prompt))
    # Margin covers chat-template overhead (role markers, BOS/EOS) plus slack
    # for the character-count fallback being approximate. 64 still 400'd on a
    # corpus-wide link question once the merged scale corpus's larger link
    # inventory pushed the exact tokenizer count within ~65 tokens of
    # max_model_len -- the two counts (ours vs. the served chat template's)
    # are close but not identical, and near the ceiling that gap is what
    # decides whether the request lands under or over it.
    budget = max_model_len - prompt_tokens - 200
    # `max(256, ...)` here used to float the answer budget back up over
    # `budget` whenever the prompt alone left less than 256 tokens spare --
    # which is exactly a big-corpus prompt (crosscsp sharing nvissues-scale's
    # containers can push retrieval past 16K tokens) and exactly the case
    # this function exists to prevent a 400 for. `budget` is the hard ceiling
    # vLLM will accept; a short real answer under it beats a 400 over it.
    capped = max(1, min(int(requested), budget))
    if capped < requested:
        log.warning("Answer token budget capped %d -> %d (prompt ~%d tokens, "
                    "model max %d)", requested, capped, prompt_tokens, max_model_len)
    kwargs[key] = capped
    return kwargs


# A customer asking for a "flowchart" or a "histogram" gets one -- they should
# never have to know or type the word "mermaid" themselves. Checked in this
# order because a question can plausibly ask for both ("show the duplicate
# chain as a diagram and the severities as a histogram"); histogram wording
# is the more specific of the two and wins when both are present.
_HISTOGRAM_WORDS = re.compile(r"\b(histograms?|bar[\s-]?charts?|bar[\s-]?graphs?)\b", re.IGNORECASE)
_DIAGRAM_WORDS = re.compile(r"\b(flow[\s-]?charts?|diagrams?)\b", re.IGNORECASE)


def _with_chart_hint(question: str) -> str:
    """`question`, plus an explicit Mermaid rendering instruction if it asked
    for a chart or diagram in plain language.

    Only ever appended to the copy of the question that lands in the answer
    prompt's `{question}` slot -- retrieval, keyword expansion and the
    structured filter all still see the customer's original wording, since
    "mermaid" is off-topic vocabulary that would only dilute a bug-corpus
    vector search, not help it.
    """
    # Word-substitution, not an appended instruction paragraph: earlier
    # attempts spelled out the exact Mermaid grammar in a bolted-on
    # sentence ("Render it as a mermaid xychart-beta bar chart. Use this
    # exact syntax: ...") and that much extra instruction made the model
    # fixate on the diagram and drop its normal written answer entirely.
    # Splicing "mermaid ..." into the question in place of the plain-
    # language phrase reads as one natural request, so the model answers
    # it the same way it always does (prose, then a chart) instead of
    # treating diagram-formatting as the whole task. The two known Mermaid
    # syntax foot-guns (xychart-beta's "min --> max" range, and stray
    # `style`/color overrides) are handled defensively on the frontend
    # (normalizeMermaidSource in static/index.html) instead of here, so
    # the prompt doesn't need to carry that burden.
    if "mermaid" in question.lower():
        return question  # already explicit; do not stack a second instruction
    if _HISTOGRAM_WORDS.search(question):
        return _HISTOGRAM_WORDS.sub(lambda m: "mermaid xychart-beta " + m.group(0), question)
    if _DIAGRAM_WORDS.search(question):
        return _DIAGRAM_WORDS.sub(lambda m: "mermaid " + m.group(0), question)
    return question


_STOP_WORDS = frozenset(
    "i me my we our you your he she it they them a an the this that these those "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need dare ought to of in on at by for with about against "
    "between through during before after above below from up down out off over under "
    "again further then once here there when where why how all both each few more most "
    "other some such no nor not only own same so than too very and but or if while "
    "because until just also already always never still even much really very "
    "what which who whom whose search searching looking find finding want need "
    "please help me tell give show recommend suggest "
    # Words that name the unit of the corpus, and so appear in almost every
    # document while distinguishing none of them. "bug" is in 900,514 of the 1M
    # corpus's 4.36M chunks -- a term that matches a fifth of everything is a
    # stop word here in the same way "the" is in English, however meaningful it
    # looks. It arrives in the keyword list from the question itself ("what is
    # bug 6081965 about"), where the bug *number* beside it is the entire query.
    "bug bugs nvbug nvbugs issue issues ticket tickets defect defects "
    # The act of asking, rather than anything asked about.
    "summarise summarize describe explain list compare overview".split()
)


# Letters-only tokenisation discards exactly the terms this corpus is searched
# by: bug numbers (6539931), bugcheck codes (0x116), driver versions (593.20)
# and module filenames (nvlddmkm.sys). Those are also the terms a vector search
# handles worst, so dropping them before BM25 left them unsearchable by either
# route. Internal dots and dashes are kept whole here; gi_index splits them on
# its own, so "nvlddmkm.sys" still matches a chunk mentioning either half.
_TERM = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._+#-]*")


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful content words from a question, stripping stop words."""
    terms = []
    for raw in _TERM.findall(question.lower()):
        word = raw.strip("._-+#")
        if len(word) > 2 and word not in _STOP_WORDS:
            terms.append(word)
    return terms


async def _llm_expand_keywords(question: str, engine) -> list[str]:
    """Use the LLM to expand a question into additional bug-search terms.

    Feeds BM25, so the useful output is vocabulary the reporter would have
    typed, not synonyms of the question. Bug threads say "bugcheck 0x116",
    not "system stability incident".
    """
    try:
        resp = await engine._llm_expand.chat.completions.create(
            model=engine._llm_expand_model,
            messages=[
                {"role": "system", "content": "Extract search keywords for an NVIDIA bug tracking database "
                 "from the user question. Return ONLY a comma-separated list of 5-8 single-word or "
                 "two-word terms as they would appear in a bug report: driver and module names "
                 "(nvlddmkm, dxgkrnl), bugcheck and error codes, subsystems (TDR, DisplayPort, Vulkan), "
                 "GPU or platform names, OS builds, and symptom words (hang, crash, flicker, corruption). "
                 "No explanations."},
                {"role": "user", "content": question},
            ],
            **engine._llm_expand_kwargs,
        )
        raw = (resp.choices[0].message.content or "").strip()
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return terms
    except Exception as e:
        log.warning("LLM keyword expansion failed: %s", e)
        return []


# Two forms, because bug ids are not a fixed width. Most are 7 digits, but the
# cross-CSP corpus contains 60792 and nvbugs ids run shorter the further back
# they go, so a 7-digit-only pattern could never anchor them. A bare number is
# taken at 6-9 digits, wide enough for every id these corpora hold and still
# too wide to be a version, a year or a line count. Below that the number has
# to be introduced as a bug for it to count, which is how it is written when it
# is meant that way ("bug 60792"), and keeps "we saw 500 of these" out.
_BUG_ID = re.compile(r"\b(?:bugs?|nvbugs?)[\s#:]*(\d{4,9})\b|\b(\d{6,9})\b", re.I)


def _anchor_bugs(question: str) -> list[str]:
    """Bug ids the question names outright.

    Vector search ranks by resemblance, so a question that says "bug 6499140"
    has no guarantee of putting that bug's node in the top-k, and its graph
    edges — the clone and duplicate links a multi-hop answer turns on — are
    then never traversed at all.
    """
    found = [labelled or bare for labelled, bare in _BUG_ID.findall(question)]
    return list(dict.fromkeys(found))[:4]


async def _missing_anchor_bugs(engine, anchors: list[str]) -> list[str]:
    """Anchor bug ids the question names outright that do not exist in this
    corpus, checked directly against the data rather than left for the model
    to notice.

    A system-prompt rule saying "say so and stop if the named bug isn't in
    the context" was tried first and it worked for exactly the id under
    test, then made answers about real bugs occasionally hedge or go quiet
    too -- a rule added to the shared answer prompt shifts every answer's
    behavior, not just the one case it targets. Checking existence here and
    short-circuiting before retrieval or the LLM ever runs has no such blast
    radius: real bugs never reach this check, and it cannot destabilize an
    answer it never applies to.
    """
    if not anchors:
        return []
    found = {b["bug_id"] for b in await _bugs_by_ids(engine, anchors)}
    return [a for a in anchors if a not in found]


def _missing_bugs_message(missing: list[str]) -> str:
    if len(missing) == 1:
        return f"Bug {missing[0]} is not in the corpus."
    return "Bugs " + ", ".join(missing) + " are not in the corpus."


def _facet_filter(engine, backend):
    """The corpus facet filter, built once per process from the backend's facets.

    Exhaustive attribute questions need one row per bug for the whole corpus.
    A local snapshot has that in memory; under `index.mode: cosmos` it arrives
    via `CosmosBackend(extract=...)`, which is small enough to hold either way
    (0.35 MB of facets against 350-550 KB of chunk text per bug). A backend
    that declares neither cannot answer exhaustively at all, which is why this
    asks for the capability rather than probing for the method.
    """
    cached = getattr(engine, "_facet_filter", None)
    if cached is not None:
        return cached
    if not backend_can(backend, CAP_FACETS):
        engine._facet_filter = False
        return False
    from structured_filter import FacetFilter
    engine._facet_filter = FacetFilter(backend._ix.bug_facets())
    return engine._facet_filter


# A two-hop walk from a bug in a large duplicate cluster reaches a few dozen
# edges. Rendered ones the corpus can resolve sort first, so this cuts the
# unverifiable tail rather than the answer.
_MAX_LINK_LINES = 25

_LINK_WORDS = re.compile(
    r"\b(clon(?:e|es|ed|ing)|duplicat(?:e|es|ed)|dupe|derived|forked"
    r"|original bug|see[- ]also|related bug|linked bug)\b", re.I)


def _links_near(links: list[dict], anchors: list[str], hops: int = 2) -> list[dict]:
    """Link edges within `hops` of any anchor bug, walked over the whole set.

    Graph traversal has a triple budget to keep the context bounded, and it is
    spent on the anchor's own attributes long before it reaches the sibling two
    hops out -- which is how "what other bug was folded into the same original"
    came back naming the one duplicate whose record this corpus lacks. Link
    edges number in the hundreds, so walking them for a few named bugs costs
    nothing and does not need rationing.
    """
    frontier = {f"Bug {a}" for a in anchors}
    seen_nodes, picked, out = set(frontier), set(), []
    for _ in range(max(1, hops)):
        nxt = set()
        for i, e in enumerate(links):
            s, o = str(e.get("subject", "")), str(e.get("object", ""))
            if s not in frontier and o not in frontier:
                continue
            if i not in picked:
                picked.add(i)
                out.append(e)
            nxt |= {n for n in (s, o) if n not in seen_nodes}
        if not nxt:
            break
        seen_nodes |= nxt
        frontier = nxt
    return out


# Comment mentions are read from the anchor bug's own chunks, so the whole bug
# is wanted rather than the three summary sections a linked bug is described by.
_MENTION_CHUNKS = 200


async def _with_mentions(backend, anchors: list[str], links: list[dict], facets: dict,
                         links_text: str) -> str:
    """Append bug ids the anchor's text names but no edge records.

    Only for questions that name a bug. A corpus-wide link question is asking
    what the graph asserts, and every bug id loosely mentioned in 211 comment
    threads would bury that under candidates.

    Reads `backend.chunks_for_bugs` (async, real for both `CosmosBackend` and
    `LocalBackend`) rather than `backend._ix.chunks_for_bugs` -- the latter
    only ever existed on `LocalGraphIndex`, so this feature silently turned
    off in cosmos mode even after `_ix` gained facets and links via
    `graph_extract.GraphExtract`, which deliberately does not implement it
    (mention scanning needs chunk text, which belongs in Cosmos, not in the
    small extract).
    """
    if not backend_can(backend, CAP_CHUNKS_FOR_BUGS):
        return links_text
    recorded = {str(e.get(k, "")).removeprefix("Bug ")
                for e in links for k in ("subject", "object")}
    rows = await backend.chunks_for_bugs(anchors, per_bug=_MENTION_CHUNKS)
    block = comment_mentions(rows, anchors, recorded, facets)
    if not block:
        return links_text
    if links_text == NO_BUG_LINKS:
        # Left as "(no links traversed)" with a mention block underneath, the
        # two lines read as a contradiction. The absence is still the answer to
        # what is *recorded*, and saying both is the honest form.
        links_text = ("No bug-to-bug link is recorded for the bug in question. "
                      "That is the answer to what it is linked to.")
    return f"{links_text}\n\n{block}"


async def _bug_links(engine, backend, question: str, triples) -> str:
    """Bug-to-bug edges for the question, with the status at both ends resolved.

    A question naming a bug is answered by what traversal reached from it. One
    that names none but asks about clones or duplicates in general -- "which
    are still open though the bug they came from was fixed" -- is a join over
    every edge, and the handful traversal happened to reach is a biased sample
    of it. The corpus holds a few dozen such edges, so that case gets all of
    them and the filtering is left to the model.

    Three outcomes, kept distinct: the edges (present), `NO_BUG_LINKS` from
    `render_bug_links` (absent -- traversal ran and found none), and
    `LINKS_UNAVAILABLE` (the inventory could not be read). The last used to
    return `""`, which the prompt filled in with `NO_BUG_LINKS`, so a failure
    to read the links was quoted back to the reader as a fact about the bug.
    """
    try:
        ff = _facet_filter(engine, backend)
        facets = ff.by_id if ff else {}
        anchors = _anchor_bugs(question)
        if backend_can(backend, CAP_LINKS):
            links = backend._ix.all_bug_links()
            if anchors:
                near = _links_near(links, anchors)
                text = render_bug_links(near, LINK_PREDICATES, facets,
                                        max_lines=_MAX_LINK_LINES)
                return await _with_mentions(backend, anchors, near, facets, text)
            if _LINK_WORDS.search(question):
                return render_bug_links(links, LINK_PREDICATES, facets,
                                        both_ends_only=True)
        elif anchors or _LINK_WORDS.search(question):
            # The question is about links and the inventory is absent, so what
            # traversal reached is a biased sample of it -- say so rather than
            # present the sample as the whole.
            log.warning("Bug link inventory unavailable; backend declares %s",
                        sorted(getattr(backend, "capabilities", []) or []))
            return LINKS_UNAVAILABLE
        return render_bug_links(triples, LINK_PREDICATES, facets)
    except Exception as e:
        log.warning("Bug link rendering failed: %s", e)
        return LINKS_UNAVAILABLE


def _graph_triples(all_triples, links_text: str):
    """The graph context minus the bug-to-bug edges BUG LINKS already carries.

    Left in, the two sections state the same relationships in two notations, and
    asked which bugs are duplicates of another the model enumerated the arrow
    form -- reporting "Bug 6410402 has duplicate and cloned from Bug 6347823",
    which is two predicates read as prose, a guessed direction, and a far end
    with no record here. Telling it which section to prefer did not work; there
    is no reason to offer the choice.
    """
    if not links_text or links_text in (NO_BUG_LINKS, LINKS_UNAVAILABLE):
        return all_triples
    return [t for t in all_triples
            if t.get("predicate") not in LINK_PREDICATES
            or not (str(t.get("subject", "")).startswith("Bug ")
                    and str(t.get("object", "")).startswith("Bug "))]


def _structured_set(engine, backend, question: str, cfg: dict) -> tuple[str, dict]:
    """Render the exhaustive matching set for the prompt, plus stats to log."""
    q = cfg.get("query", {})
    if not q.get("use_structured_filter", True):
        return NO_STRUCTURED_SET, {}
    try:
        ff = _facet_filter(engine, backend)
        if not ff:
            return NO_STRUCTURED_SET, {}
        from structured_filter import render_set
        selection = ff.select(question)
        if not selection:
            return NO_STRUCTURED_SET, {"predicates": selection.describe()}
        limit = int(q.get("structured_filter_max_bugs", 120))
        return render_set(selection, limit), {
            "predicates": selection.describe(),
            "matched": len(selection.rows),
        }
    except Exception as e:  # never let filtering break an answer
        log.warning("Structured filter failed: %s", e)
        return NO_STRUCTURED_SET, {}


_RERANK_URL_SUFFIX = "dbinference.azure.com:443/inference/semanticReranking"


async def _rerank_token(engine, scope: str) -> str | None:
    """Acquire (and cache on the engine) a bearer token for the reranker service.

    `AzureCliCredential` cannot authenticate inside a headless Container App
    replica (no interactive `az login` there), so this reuses whatever
    `DefaultAzureCredential` `engine._get_cosmos()` already constructed for
    RBAC-mode Cosmos access -- a managed identity in ACA, the CLI locally --
    rather than building a second, ACA-incompatible credential of its own.
    """
    now = time.time()
    if getattr(engine, "_ranker_token", None) and now < getattr(engine, "_ranker_token_exp", 0) - 60:
        return engine._ranker_token
    try:
        await engine._get_cosmos()  # ensures engine._cred is set for RBAC configs
        cred = getattr(engine, "_cred", None)
        if cred is None:
            # Key-auth Cosmos leaves _cred unset. Prefer Azure CLI here: the VM
            # managed identity that DefaultAzureCredential would pick first does
            # not have Semantic Reranker User on this box.
            from azure.identity.aio import AzureCliCredential
            tenant = (engine._cfg.get("cosmos") or {}).get("tenant_id") or None
            cred = AzureCliCredential(tenant_id=tenant) if tenant else AzureCliCredential()
            engine._ranker_cli_cred = cred
        tok = await cred.get_token(scope)
        engine._ranker_token = tok.token
        engine._ranker_token_exp = tok.expires_on
        return tok.token
    except Exception as e:
        log.warning("Reranker token acquisition failed: %s", e)
        return None


def _keep_pinned(ranked: list[dict], docs: list[dict],
                 pinned_bugs: set[str] | None) -> list[dict]:
    """Re-seat chunks of bugs the question named that the ranker dropped.

    The ranker scores resemblance to the question, and "what is bug 1007844
    about?" reads no more like that bug's own text than like the hundreds of
    near-identical bugs around it. So on a large corpus the ranker routinely
    evicts the one bug the question actually named, and the model then answers
    that the bug is not in the provided context -- while holding its chunks.
    """
    if not pinned_bugs:
        return ranked
    kept = {d.get("id") for d in ranked}
    missing = [d for d in docs
               if str(d.get("bug_id")) in pinned_bugs and d.get("id") not in kept]
    return missing + ranked if missing else ranked


async def _semantic_rerank(engine, question: str, docs: list[dict],
                           pinned_bugs: set[str] | None = None) -> list[dict]:
    """Rerank bug sections via the Cosmos semantic-reranker HTTP endpoint (ranker.* config).

    Mirrors the upstream CombinedRetriever behaviour: every candidate is scored
    by the ranker and the top ``ranker.k_ranker`` are kept, except that chunks of
    bugs named in the question are always kept (see `_keep_pinned`). Falls back
    to the existing order when the ranker is disabled/unconfigured, on any error,
    or when there are already <= k_ranker candidates.
    """
    if not docs:
        return docs

    ranker = engine._cfg.get("ranker", {})
    account = str(ranker.get("account_name", "")).strip()
    region = str(ranker.get("region", "")).strip()
    k_ranker = int(ranker.get("k_ranker", 0) or 0)
    if not ranker.get("use_ranker", True) or not account or not region or k_ranker <= 0:
        return docs
    # Nothing to trim if we already have <= k_ranker candidates.
    if len(docs) <= k_ranker:
        return docs

    import json as _json
    doc_strings = []
    for doc in docs:
        # The synopsis is what identifies the bug and the section text is what
        # distinguishes this chunk from the bug's other chunks, so the ranker
        # needs both to tell two sections of the same bug apart.
        parts = []
        synopsis = doc.get("synopsis", "")
        if synopsis:
            parts.append(synopsis)
        module = doc.get("module", "")
        if module:
            parts.append(f"Module: {module}")
        text = doc.get("text", "")
        if text:
            parts.append(f"{doc.get('section', 'text')}: {text[:1000]}")
        doc_strings.append(" | ".join(parts) if parts else _json.dumps(doc)[:500])

    # The ranker rejects payloads containing empty strings.
    if any(not (isinstance(s, str) and s.strip()) for s in doc_strings):
        return docs

    scope = str(ranker.get("token_scope", "https://dbinference.azure.com/.default")).strip()
    token = await _rerank_token(engine, scope)
    if not token:
        return docs

    url_suffix = str(ranker.get("url_suffix", _RERANK_URL_SUFFIX)).strip()
    url = f"https://{account}.{region}.{url_suffix}"
    body = {
        "query": question,
        "documents": doc_strings,
        "return_documents": False,
        "top_k": k_ranker,
        "batch_size": int(ranker.get("batch_size", 32)),
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        import httpx
        client = getattr(engine, "_ranker_http", None)
        if client is None:
            client = httpx.AsyncClient(timeout=30)
            engine._ranker_http = client
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        scores = resp.json().get("Scores", [])
        if scores:
            ranked = [docs[s["index"]] for s in scores if s["index"] < len(docs)]
            return _keep_pinned(ranked, docs, pinned_bugs)
    except Exception as e:
        log.warning("Semantic reranker failed (falling back to vector order): %s", e)

    return docs


async def _identify_missing_containers(engine) -> list[str]:
    """Return configured Graph Index containers that don't exist (queried from Cosmos)."""
    missing: list[str] = []
    try:
        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        gi = engine._gi_cfg
        for n in (gi.get("entities_container", "entities"),
                  gi.get("triples_container", "triples"),
                  gi.get("source_container", "chunks")):
            try:
                await db.get_container_client(n).read()
            except Exception as e:
                if "NotFound" in str(e) or "404" in str(e):
                    missing.append(n)
    except Exception:
        pass
    return missing


async def _stream_dflash_sse(question: str, engine: GIQueryEngine,
                              corpus_id: str = "", corpus_stats: dict | None = None):
    """DFlash path: GI retrieval (local index or Cosmos, via retrieval.py) + real
    token-by-token LLM streaming with speculative decoding.

    Retrieval used to be duplicated here as hardcoded Cosmos queries, separate
    from `_dflash_answer`'s copy. Both now call the same `retrieve()` against
    whichever backend `index.mode` selects, so this path gets the local-index
    speedup for free and there is exactly one place that implements the
    five-stage pipeline. The LLM call is also switched from
    "wait for the full completion, then fake-chunk it" to `stream=True`, so
    time-to-first-token drops from the full generation time to roughly one
    speculative-decoding step.
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        anchors = _anchor_bugs(question)
        missing = await _missing_anchor_bugs(engine, anchors)
        if missing:
            yield _sse({"stage": "token", "text": _missing_bugs_message(missing)})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        yield _sse({"stage": "progress", "message": "Embedding question...", "_ts": _elapsed(t0)})

        t_embed = time.perf_counter()
        q_emb = await engine._embedder.embed(question, is_query=True)
        timings["embed"] = time.perf_counter() - t_embed
        yield _sse({"stage": "progress", "message": f"Embedded in {timings['embed']:.2f}s", "_ts": _elapsed(t0)})

        # Keyword expansion is an LLM call; start it now so it overlaps retrieval.
        basic_kw = _extract_keywords(question)
        kw_task = asyncio.create_task(_llm_expand_keywords(question, engine))

        yield _sse({"stage": "progress", "message": "Retrieving (entity search + graph traversal + sources)...",
                     "_ts": _elapsed(t0)})

        backend = await engine._get_backend()
        t_retr = time.perf_counter()
        result = await retrieve(backend, q_emb, engine._cfg,
                                anchor_bugs=anchors)
        timings.update(result.timings)
        seed_entities = result.seed_entities
        all_triples = result.triples
        source_chunks = result.source_chunks

        if not seed_entities:
            kw_task.cancel()
            yield _sse({"stage": "progress", "message": "No entities found.", "_ts": _elapsed(t0)})
            yield _sse({"stage": "token", "text": "No relevant entities found in the graph index."})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        entity_names = [e["name"] for e in seed_entities[:8]]
        yield _sse({"stage": "progress",
                     "message": f"Retrieved {len(seed_entities)} entities, {len(all_triples)} triples, "
                                f"{len(source_chunks)} sources in {time.perf_counter() - t_retr:.2f}s "
                                f"({result.stats.get('pk_triples', 0)} PK + {result.stats.get('vec_triples', 0)} vec): "
                                f"{', '.join(entity_names[:5])}",
                     "_ts": _elapsed(t0)})

        # --- Keyword-expanded full-text search, merged into source_chunks ---
        llm_keywords = await kw_task
        all_kw = _merge_keywords(basic_kw, llm_keywords)
        log.info("Keywords basic=%s llm=%s combined=%s", basic_kw[:5], llm_keywords, all_kw)

        t_ft = time.perf_counter()
        seen_ids = {doc.get("id") for doc in source_chunks}
        for doc in await backend.fulltext_chunks(all_kw, 10):
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))
        timings["source_fetch"] += time.perf_counter() - t_ft

        # --- Rerank ---
        t_rerank = time.perf_counter()
        pre_rerank_n = len(source_chunks)
        source_chunks = await _semantic_rerank(engine, question, source_chunks,
                                              pinned_bugs=set(anchors))
        timings["rerank"] = time.perf_counter() - t_rerank
        log.info("Semantic rerank: %d -> %d chunks in %.2fs",
                 pre_rerank_n, len(source_chunks), timings["rerank"])

        yield _sse({
            "stage": "stats",
            "seed_entities": len(seed_entities),
            "triples_found": len(all_triples),
            "source_chunks": len(source_chunks),
            "entity_names": entity_names,
            "_ts": _elapsed(t0),
        })

        # --- Build prompt + streaming LLM call ---
        yield _sse({"stage": "progress",
                     "message": f"Retrieval done in {time.perf_counter() - t0:.1f}s — calling LLM ({engine._llm_model})...",
                     "_ts": _elapsed(t0)})

        structured_text, structured_stats = _structured_set(
            engine, backend, question, engine._cfg)
        if structured_stats.get("matched"):
            yield _sse({"stage": "progress",
                        "message": f"Attribute filter {structured_stats['predicates']} "
                                   f"matches {structured_stats['matched']} bugs corpus-wide",
                        "_ts": _elapsed(t0)})

        links_text = await _bug_links(engine, backend, question, all_triples)
        graph_context = engine._build_graph_context(
            seed_entities, _graph_triples(all_triples, links_text))
        source_text = engine._build_source_text(
            source_chunks, source_budget(structured_stats.get("matched") and structured_text))
        prompt = render_answer_prompt(_with_chart_hint(question), graph_context, source_text,
                                      structured_text, links_text,
                                      await _corpus_stats_text(engine, corpus_id, corpus_stats))

        t_llm = time.perf_counter()
        first_token_at: float | None = None
        stream = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            **_capped_llm_kwargs(engine, _ANSWER_SYSTEM_PROMPT, prompt),
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                timings["ttft"] = first_token_at - t_llm
            yield _sse({"stage": "token", "text": delta})

        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0

        yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})

    except Exception as e:
        log.exception("dflash stream error: %s", e)
        msg = str(e)
        if "NotFound" in msg or "404" in msg:
            missing = await _identify_missing_containers(engine)
            if missing:
                msg = (
                    f"Cosmos DB container(s) not found in database '{engine._db_name}': "
                    f"{', '.join(missing)}. Check gi.triples_container / gi.entities_container "
                    f"in your config, or (re)build the graph index."
                )
        yield _sse({"stage": "error", "message": msg})

    yield "data: [DONE]\n\n"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    main_cfg = os.environ.get("GI_CONFIG", str(_ROOT / "my.yaml"))
    cfg_path = Path(main_cfg)
    if not cfg_path.exists():
        raise RuntimeError(
            f"Config not found: {cfg_path}. Provide --config or create my.yaml."
        )

    cfg = load_config(str(cfg_path))

    # Cosmos DB Semantic Reranker endpoint comes from config; an explicit env
    # var wins. The azure-cosmos SDK reads this env var at rerank time.
    reranker_endpoint = cfg.get("cosmos", {}).get("semantic_reranker_endpoint")
    if reranker_endpoint:
        os.environ.setdefault(
            "AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT", str(reranker_endpoint)
        )

    corpora_cfg = _corpora_from_cfg(cfg)
    if not corpora_cfg:
        raise RuntimeError("No corpus has a snapshot on disk; nothing to serve.")

    # One engine per corpus, each with its own snapshot path / Cosmos database,
    # all sharing the first one's embedding model and LLM clients.
    engines: dict[str, dict] = {}
    first: GIQueryEngine | None = None
    for entry in corpora_cfg:
        ccfg = copy.deepcopy(cfg)
        ccfg.setdefault("index", {})["snapshot_path"] = entry["snapshot_path"]
        if entry.get("database_name"):
            ccfg.setdefault("cosmos", {})["database_name"] = entry["database_name"]
            if str(cfg.get("index", {}).get("mode", "")).lower() in ("postgres", "pg"):
                ccfg.setdefault("index", {})["dbname"] = entry["database_name"]
        if entry.get("extract_path"):
            ccfg.setdefault("index", {})["extract_path"] = entry["extract_path"]
        engine = GIQueryEngine(ccfg, share_from=first)
        first = first or engine
        questions = (_load_questions(entry["questions_path"])
                     if entry["questions_path"] else _load_questions_from_cfg(cfg))
        engines[entry["id"]] = {"engine": engine, "questions": questions,
                                "label": entry["label"], "id": entry["id"],
                                "snapshot_path": entry["snapshot_path"],
                                "database_name": entry.get("database_name"),
                                "stats": entry.get("stats")}
        log.info("Corpus %s (%s): db=%s snapshot=%s, %d questions",
                 entry["id"], entry["label"],
                 entry.get("database_name"), entry["snapshot_path"], len(questions))

    engine = first
    questions = engines[corpora_cfg[0]["id"]]["questions"]

    # Pay every one-time cost here rather than inside whichever request arrives
    # first. All of it is best-effort — an unreachable Cosmos or a down vLLM must
    # not take the whole app down, so failures surface on first request instead.
    #   embedder: model load onto the GPU.
    #   llm:      vLLM JIT-compiles its attention and speculative-decode kernels
    #             on the first real inference ("FlashInfer GDN prefill is
    #             JIT-compiled; first run may take a while"), ~40s of dead air.
    #             Keyword expansion is the ask pipeline's first LLM call, so a
    #             throwaway one here compiles the same kernels.
    #   backend:  for index.mode: local, loading the snapshot (numpy -> GPU, BM25
    #             build) takes a few seconds.
    #   stats:    corpus totals are COUNT(1) over four containers.
    log.info("Warming up embedder + retrieval backend + LLM...")

    async def _warm(labels, aws):
        for label, result in zip(labels, await asyncio.gather(*aws,
                                                             return_exceptions=True)):
            if isinstance(result, Exception):
                log.warning("%s warmup failed: %s", label, result)

    # `_get_backend` has no init lock, so the corpus-count prefetch below (which
    # goes through it) has to wait for the backends rather than race them into
    # building a second Cosmos client each.
    await _warm(["embedder", "llm"] + [f"backend[{cid}]" for cid in engines],
                [engine._embedder.embed("warmup"),
                 _llm_expand_keywords("warmup", engine),
                 *(e["engine"]._get_backend() for e in engines.values())])
    await _warm([f"stats[{cid}]" for cid in engines],
                [_corpus_stats_text(e["engine"], e["id"], e.get("stats"))
                 for e in engines.values()])

    app.state.engine = engine
    app.state.questions = questions
    app.state.corpora = engines
    app.state.default_corpus = corpora_cfg[0]["id"]

    yield

    for entry in engines.values():
        await entry["engine"].close()


app = FastAPI(title="nvissues GI-RAG", version="2.0.0", lifespan=lifespan)


# Shared-secret guard, active only when RETRIEVER_TOKEN is set.
#
# Needed because of how this process is now reached. Run locally or on a VM
# inside the network, the network was the boundary and there was nothing to
# authenticate. Run on the DGX behind an outbound tunnel, the endpoint is on the
# public internet by construction -- the tunnel exists precisely to make it so --
# and the corpus behind it is NVIDIA-internal bug reports, synopses, customer
# names and comment text. An unauthenticated URL is the wrong default for that
# even while the hostname is an unguessable one.
#
# Off by default, so local runs and the in-VNet Container Apps deployment are
# unaffected and no existing workflow needs a token it did not have before. The
# check is deliberately not applied to /health: the ingress tier polls it to
# report whether the DGX is up, and a liveness probe that needs a credential is
# a liveness probe that reports the credential's state rather than the service's.
_TOKEN = os.environ.get("RETRIEVER_TOKEN", "").strip()
_OPEN_PATHS = {"/health"}


@app.middleware("http")
async def _require_token(request, call_next):
    if _TOKEN and request.url.path not in _OPEN_PATHS:
        sent = request.headers.get("authorization", "")
        prefix = "bearer "
        got = sent[len(prefix):] if sent.lower().startswith(prefix) else ""
        # Constant-time, so a wrong token cannot be narrowed down by timing the
        # rejection. Cheap insurance on a public endpoint.
        if not hmac.compare_digest(got, _TOKEN):
            log.warning("rejected unauthenticated %s %s", request.method,
                        request.url.path)
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_ROOT / "static" / "index.html")

app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/backends")
async def get_backends():
    bid, binfo = next(iter(BACKENDS.items()))
    return JSONResponse(content=[{
        "id": bid,
        "label": binfo["label"],
        "description": binfo["description"],
        "badge_color": binfo["badge_color"],
        "question_count": len(app.state.questions),
    }])

async def _corpus_counts(engine) -> dict:
    """Bug/chunk/entity/triple counts for the UI header.

    Local backends expose an in-memory index; cosmos backends query COUNT(1)
    against each container (cheap enough for a menu refresh).

    A `CosmosBackend` with a `GraphExtract` overlay also has an `_ix`, but
    that extract deliberately holds only a subset of the database's bugs
    (facets/links for the ones with real, unchanged ids -- see
    graph_extract.py) and has no `chunks`/`entities`/`triples` counts at
    all, so it must not be mistaken for a full local index here or the
    header would report a few hundred bugs for a million-bug database.
    """
    backend = await engine._get_backend()
    ix = getattr(backend, "_ix", None)
    # A backend that can count itself is asked to. The branch below reads
    # `.chunks.n` and friends, which are Arrow tables belonging to a local
    # snapshot; a store that keeps its counts in the database has no such
    # attributes and would raise AttributeError rather than degrade.
    if ix is not None and hasattr(ix, "corpus_counts"):
        return ix.corpus_counts()
    if ix is not None and not isinstance(ix, GraphExtract):
        return {
            "bugs": len(ix.bug_facets()) if hasattr(ix, "bug_facets") else None,
            "chunks": ix.chunks.n,
            "entities": ix.entities.n,
            "triples": ix.triples.n,
        }
    # Cosmos path: count docs in the four containers.
    out: dict = {"database": getattr(engine, "_db_name", None)}
    try:
        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        gi = engine._gi_cfg
        mapping = {
            "bugs": "issues",
            "chunks": gi.get("source_container", "chunks"),
            "entities": gi.get("entities_container", "entities"),
            "triples": gi.get("triples_container", "triples"),
        }
        for key, cname in mapping.items():
            n = 0
            async for row in db.get_container_client(cname).query_items(
                    "SELECT VALUE COUNT(1) FROM c"):
                n = int(row)
            out[key] = n
    except Exception as e:
        log.warning("corpus counts failed for %s: %s",
                    getattr(engine, "_db_name", "?"), e)
    return out


# Retrieval only ever surfaces a handful of bugs, so "how big is the database"
# cannot be answered from the prompt's source documents -- the model counts what
# it can see and reports that. The real totals go in as their own prompt section.
# COUNT(1) over four containers is too slow to repeat per question, so cache it.
_CORPUS_STATS_TTL = 300.0
_corpus_stats_cache: dict[str, tuple[float, str]] = {}


async def _corpus_stats_text(engine, corpus_id: str = "", stats: dict | None = None) -> str:
    # Keyed by corpus id, not database name: a corpus sharing its database
    # with a much bigger one (see `stats` override in _corpora_from_cfg)
    # would otherwise collide on the same cache entry as that bigger corpus.
    key = corpus_id or getattr(engine, "_db_name", "?")
    now = time.time()
    hit = _corpus_stats_cache.get(key)
    if hit and now - hit[0] < _CORPUS_STATS_TTL:
        return hit[1]
    if stats:
        counts = dict(stats)
    else:
        try:
            counts = await _corpus_counts(engine)
        except Exception as e:  # a missing count must not break the answer
            log.warning("corpus stats failed for %s: %s", key, e)
            return NO_CORPUS_STATS
    parts = [f"{counts[k]:,} {label}"
             for k, label in (("bugs", "bugs"), ("chunks", "text sections"),
                              ("entities", "graph entities"),
                              ("triples", "graph relationships"))
             if isinstance(counts.get(k), int)]
    if not parts:
        return NO_CORPUS_STATS
    text = ("This database holds " + ", ".join(parts)
            + ". These are the totals for the whole corpus.")
    _corpus_stats_cache[key] = (now, text)
    return text


@app.get("/v1/corpus")
async def get_corpus(corpus: str | None = None):
    """Counts for one corpus, for the UI header to state.

    One deployment serves several corpora, so a header with the numbers
    written into it is a header that is wrong most of the time -- and wrong in
    the most quotable place on the page.
    """
    entry = _resolve(corpus)
    if entry.get("stats"):
        return JSONResponse(content=dict(entry["stats"]))
    return JSONResponse(content=await _corpus_counts(entry["engine"]))


@app.get("/v1/corpora")
async def get_corpora():
    """Every corpus this deployment serves, in menu order, with its counts."""
    out = []
    for cid, entry in app.state.corpora.items():
        counts = dict(entry["stats"]) if entry.get("stats") \
            else await _corpus_counts(entry["engine"])
        out.append({
            "id": cid,
            "label": entry["label"],
            "default": cid == app.state.default_corpus,
            "question_count": len(entry["questions"]),
            **counts,
        })
    return JSONResponse(content=out)


@app.get("/v1/questions")
async def get_questions(backend: str = "gi", corpus: str | None = None):
    return JSONResponse(content=_resolve(corpus)["questions"])


# The bug list the UI shows on the left, and matches answer text against to
# highlight which bugs an answer actually cites. `issues` is the system of
# record and its `id` *is* the bug id, so this is a projection, not a join.
_BUGS_CACHE: dict[str, list[dict]] = {}


async def _bugs_by_ids(engine, ids: list[str]) -> list[dict]:
    """Look up specific bugs by id, regardless of the browse-list window.

    The left panel only downloads the first `limit` bugs, so on a large corpus
    an answer routinely cites bug ids that panel never loaded. Highlighting
    resolves cited ids against this endpoint so every cited bug can be shown,
    not just the ones that happened to fall inside the browse window.
    """
    ids = [str(i) for i in ids][:200]  # answers cite a handful; cap the IN-list
    if not ids:
        return []
    rows: list[dict] = []
    backend = await engine._get_backend()
    ix = getattr(backend, "_ix", None)
    remaining = ids
    if ix is not None and hasattr(ix, "bug_facets"):
        # bug_facets() is a list of per-bug dicts (LocalGraphIndex and
        # GraphExtract both build it that way), not a dict keyed by id.
        by_id = {str(f.get("bug_id")): f for f in ix.bug_facets()}
        found = []
        for bid in ids:
            facets = by_id.get(bid)
            if facets is not None:
                rows.append({"bug_id": bid,
                             "synopsis": str((facets or {}).get("synopsis", ""))})
                found.append(bid)
        # A `LocalGraphIndex` covers every bug, so nothing is left to look up
        # elsewhere. A `GraphExtract` overlay only covers the subset of the
        # database it was built for (see graph_extract.py), so an id it
        # doesn't recognize is not necessarily missing from the corpus --
        # it just isn't in that subset, and the cosmos query below is the
        # one that can actually say so.
        if not isinstance(ix, GraphExtract):
            return rows
        remaining = [i for i in ids if i not in found]
        if not remaining:
            return rows
    try:
        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        ctr = db.get_container_client("issues")
        params = [{"name": f"@id{i}", "value": v} for i, v in enumerate(remaining)]
        placeholders = ", ".join(p["name"] for p in params)
        query = f"SELECT c.id, c.synopsis FROM c WHERE c.id IN ({placeholders})"
        async for doc in ctr.query_items(query, parameters=params):
            rows.append({"bug_id": str(doc.get("id")),
                         "synopsis": str(doc.get("synopsis") or "")})
    except Exception as e:
        log.warning("bug id lookup failed for %s: %s",
                    getattr(engine, "_db_name", "?"), e)
    return rows


@app.get("/v1/bugs")
async def get_bugs(corpus: str | None = None, limit: int = 5000,
                   ids: str | None = None):
    entry = _resolve(corpus)
    engine = entry["engine"]

    if ids:
        wanted = [t.strip() for t in ids.split(",") if t.strip()]
        return JSONResponse(content=await _bugs_by_ids(engine, wanted))

    cache_key = f"{entry['id']}:{limit}"
    if cache_key in _BUGS_CACHE:
        return JSONResponse(content=_BUGS_CACHE[cache_key])

    rows: list[dict] = []
    backend = await engine._get_backend()
    ix = getattr(backend, "_ix", None)
    if ix is not None and not isinstance(ix, GraphExtract) and hasattr(ix, "bug_facets"):
        for facets in ix.bug_facets()[:limit]:
            rows.append({"bug_id": str(facets.get("bug_id")),
                         "synopsis": str((facets or {}).get("synopsis", ""))})
    else:
        try:
            cosmos = await engine._get_cosmos()
            db = cosmos.get_database_client(engine._db_name)
            ctr = db.get_container_client("issues")
            query = f"SELECT TOP {int(limit)} c.id, c.synopsis FROM c"
            async for doc in ctr.query_items(query):
                rows.append({"bug_id": str(doc.get("id")),
                             "synopsis": str(doc.get("synopsis") or "")})
        except Exception as e:
            log.warning("bug list failed for %s: %s",
                        getattr(engine, "_db_name", "?"), e)

    rows.sort(key=lambda r: r["bug_id"])
    _BUGS_CACHE[cache_key] = rows
    return JSONResponse(content=rows)

@app.post("/v1/ask/stream")
async def ask_stream(body: AskRequest):
    entry = _resolve(body.corpus)
    gen = _stream_dflash_sse(body.question, entry["engine"], entry["id"], entry.get("stats"))

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        # Connection: close forces the browser to open a fresh TCP connection for
        # every ask instead of pooling this one for reuse. Each ask already spends
        # seconds on retrieval + LLM, so a new handshake costs nothing by
        # comparison -- but a reused keep-alive connection that a NAT/firewall
        # silently dropped while idle looks, from the browser's side, exactly
        # like a request that never got a response: no error, just silence until
        # some OS-level retransmit timeout finally gives up. That symptom (works,
        # then randomly hangs on the *next* request) is what this heads off.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                "Connection": "close"},
    )

async def _dflash_answer(question: str, engine: GIQueryEngine,
                          corpus_id: str = "", corpus_stats: dict | None = None) -> dict:
    """Non-streaming DFlash: full GI retrieval + non-streaming LLM, returns result dict."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    anchors = _anchor_bugs(question)
    missing = await _missing_anchor_bugs(engine, anchors)
    if missing:
        timings["total"] = time.perf_counter() - t0
        return {"answer": _missing_bugs_message(missing), "timings": timings}

    q_emb = await engine._embedder.embed(question, is_query=True)
    timings["embed"] = time.perf_counter() - t0

    # Keyword expansion is an LLM call, so start it now and let it run
    # alongside retrieval; its results are only needed for the full-text merge.
    basic_kw = _extract_keywords(question)
    kw_task = asyncio.create_task(_llm_expand_keywords(question, engine))

    backend = await engine._get_backend()
    result = await retrieve(backend, q_emb, engine._cfg,
                            anchor_bugs=anchors)
    timings.update(result.timings)
    seed_entities = result.seed_entities
    all_triples = result.triples
    source_chunks = result.source_chunks

    if not seed_entities:
        kw_task.cancel()
        timings["total"] = time.perf_counter() - t0
        return {"answer": "No relevant entities found.", "timings": timings}

    llm_keywords = await kw_task
    all_kw = _merge_keywords(basic_kw, llm_keywords)
    log.info("Keywords basic=%s llm=%s combined=%s", basic_kw[:5], llm_keywords, all_kw)

    t_ft = time.perf_counter()
    retrieved_n = len(source_chunks)
    seen_ids = {doc.get("id") for doc in source_chunks}
    for doc in await backend.fulltext_chunks(all_kw, 10):
        if doc.get("id") not in seen_ids:
            source_chunks.append(doc)
            seen_ids.add(doc.get("id"))
    timings["source_fetch"] += time.perf_counter() - t_ft
    log.info("Chunks: %d retrieved, %d after full-text merge over %d keywords",
             retrieved_n, len(source_chunks), len(all_kw))

    pre_rerank_n = len(source_chunks)
    t_rerank = time.perf_counter()
    source_chunks = await _semantic_rerank(engine, question, source_chunks,
                                          pinned_bugs=set(anchors))
    timings["rerank"] = time.perf_counter() - t_rerank
    if len(source_chunks) != pre_rerank_n:
        log.info("Semantic rerank: %d -> %d chunks in %.2fs",
                 pre_rerank_n, len(source_chunks), timings["rerank"])
    else:
        log.info("Semantic rerank: unchanged (%d chunks) in %.2fs",
                 pre_rerank_n, timings["rerank"])

    t_filter = time.perf_counter()
    structured_text, structured_stats = _structured_set(engine, backend, question, engine._cfg)
    timings["structured_filter"] = time.perf_counter() - t_filter
    if structured_stats.get("matched"):
        log.info("Structured filter %s -> %d bugs",
                 structured_stats["predicates"], structured_stats["matched"])

    links_text = await _bug_links(engine, backend, question, all_triples)
    graph_context = engine._build_graph_context(
        seed_entities, _graph_triples(all_triples, links_text))
    source_text = engine._build_source_text(
        source_chunks, source_budget(structured_stats.get("matched") and structured_text))
    prompt = render_answer_prompt(_with_chart_hint(question), graph_context, source_text,
                                  structured_text, links_text,
                                  await _corpus_stats_text(engine, corpus_id, corpus_stats))
    # Section sizes and the digest of the whole thing. An answer that changed
    # and a prompt that changed are different problems, and the log could not
    # tell them apart: two builds that disagreed on the same question took an
    # afternoon to separate because nothing recorded what either was sent.
    log.info("Prompt %s: %d chars (graph %d, sources %d over %d chunks, set %d, "
             "links %d)%s", hashlib.sha1(prompt.encode()).hexdigest()[:10],
             len(prompt), len(graph_context), len(source_text), len(source_chunks),
             len(structured_text or ""), len(links_text),
             ", mention rule" if MENTION_HEADER in links_text else "")

    t_llm = time.perf_counter()
    try:
        resp = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[
                {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **_capped_llm_kwargs(engine, _ANSWER_SYSTEM_PROMPT, prompt),
        )
    except openai.BadRequestError as e:
        # `_capped_llm_kwargs` sizes the completion budget from our own token
        # estimate, which is close to but not identical to the model's own
        # count -- close enough that this is rare, but a corpus-wide question
        # with no anchor bug (every bug-to-bug link, unfiltered) can still
        # build a prompt that lands right on the boundary. A clear "too much
        # to answer" beats a raw 500 with a stack trace the UI has no way to
        # render.
        log.warning("LLM call exceeded context window for %r: %s", question, e)
        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0
        return {"answer": ("This question pulled in more corpus data than fits in one "
                           "answer. Try naming a specific bug, or narrowing the question."),
               "timings": timings}
    timings["llm"] = time.perf_counter() - t_llm
    timings["total"] = time.perf_counter() - t0

    answer = (resp.choices[0].message.content if resp.choices else "") or ""
    if not answer.strip():
        # A reasoning model spends the completion budget on reasoning before it
        # writes anything, so a budget `_capped_llm_kwargs` clamped too far
        # comes back as a *successful* response with empty content rather than
        # as an error -- and an empty string renders as a blank page, which
        # looks like the app broke rather than like the answer being cut off
        # before its first token. Say which it was, and log the numbers needed
        # to fix it: this is `llm.max_model_len` set below the model's real
        # window, or `query.max_answer_tokens` set below what reasoning costs.
        finish = getattr(resp.choices[0], "finish_reason", None) if resp.choices else None
        usage = getattr(resp, "usage", None)
        log.warning("Empty answer for %r: finish_reason=%s usage=%s budget=%s",
                    question, finish, usage,
                    _capped_llm_kwargs(engine, _ANSWER_SYSTEM_PROMPT, prompt))
        answer = ("The model returned no text for this question: its answer "
                  "budget was spent before the first word. Retrieval worked -- "
                  f"{len(source_chunks)} source chunks and {len(all_triples)} "
                  "graph edges were found. Narrow the question, or raise "
                  "`llm.max_model_len` to the answer model's real context "
                  "window.")
    return {"answer": answer, "timings": timings}


@app.post("/v1/ask")
async def ask(body: AskRequest):
    t0 = time.perf_counter()
    entry = _resolve(body.corpus)
    result = await _dflash_answer(body.question, entry["engine"], entry["id"], entry.get("stats"))
    result["corpus"] = entry["id"]
    result["http_wall_s"] = round(time.perf_counter() - t0, 4)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="nvissues GI-RAG API (single Graph Index + LLM backend)")
    parser.add_argument(
        "--config",
        default="my.yaml",
        help="Path to the YAML config (default: my.yaml).",
    )
    parser.add_argument("--host", default="localhost", help="Host to bind (default: localhost).")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080).")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")
    os.environ["GI_CONFIG"] = str(cfg_path.resolve())

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
