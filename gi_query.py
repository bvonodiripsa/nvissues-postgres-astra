#!/usr/bin/env python
"""Online GI-RAG query engine for bug reports.

Given a question:
  1. Embed the question
  2. Vector-search the entity index for seed entities
  3. Fetch connected triples (graph traversal)
  4. Fetch source docs for provenance
  5. Single LLM call with structured graph context + source text

Target: 2-4 seconds per question.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import yaml
# Optional here. `index.mode: postgres` never constructs a CosmosClient, and
# the credential is only reached by the Cosmos reranker, which this config
# disables. See gi_builder for why the Astra image omits the Azure SDKs.
try:
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import DefaultAzureCredential
except ImportError:  # pragma: no cover - exercised only in database-free builds
    CosmosClient = None
    DefaultAzureCredential = None

from prompts_gi_issues import (FALLBACK_ANSWER_PROMPT, GRAPHRAG_ANSWER_PROMPT,
                               MENTION_RULE)
from gi_builder import EmbedClient, embed_sync, load_config
from llm_roles import build_role
from retrieval import CosmosBackend, retrieve

log = logging.getLogger("food_dflash.gi_query")

# Distinguishes "extract not loaded yet" from "loaded, and there wasn't one on
# disk" -- both are falsy-ish, but only the second should be cached as final.
_UNSET = object()


# =============================================================================
# Source rendering
# =============================================================================

# Retrieval returns bug *sections*, and a single bug contributes many of them
# while repeating its synopsis and every status field on each one. Rendering
# chunk by chunk therefore spends most of the context window restating the
# same header. Grouping by bug states each bug's identity and status once and
# leaves the budget for the prose that actually differs between sections.

# Reading order of a bug report, so the model sees the problem statement
# before the triage thread that reacts to it. Anything unlisted sorts last.
_SECTION_ORDER = ("synopsis", "description", "repro", "comments", "attachments")

# Comment threads run to tens of thousands of characters on a long-lived bug.
_SECTION_CHARS = 1500

# Total budget for the rendered sources, roughly 15K tokens. Bugs are emitted
# in retrieval order (graph-derived, then vector, then lexical), so this drops
# the weakest matches rather than an arbitrary tail. Questions whose answer is
# a *set* of bugs are the reason this is a character budget and not a bug
# count: 40 one-line bugs is a better answer than 12 verbose ones.
# Cap for local vLLM --max-model-len 16384 (Azure OpenAI A/B used a larger
# window, where 60k source chars fit). Qwen tokenization of bug text is dense
# (~3 chars/token), so keep this low enough that prompt+completion fits.
# Measured: 28k source chars put the worst prompt at 13.9k tokens, leaving under
# 2.5k for the answer -- set enumerations then stopped mid-list. 24k keeps the
# worst prompt near 12k so the 2048-token answer budget always fits.
_SOURCE_CHARS = 24000


def _section_key(doc: dict) -> tuple[int, int]:
    section = str(doc.get("section", ""))
    try:
        rank = _SECTION_ORDER.index(section)
    except ValueError:
        rank = len(_SECTION_ORDER)
    return rank, int(doc.get("seq", 0) or 0)


def _join(value) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value if v)
    return str(value or "")


def _bug_header(bug_id: str, doc: dict) -> list[str]:
    lines = [f"[Bug {bug_id}] {doc.get('synopsis', '') or '(no synopsis)'}"]

    # Priority and disposition decide whether a matching bug is even an answer
    # to "do we have a problem with X", so they are never truncated away.
    status = [
        f"priority={doc.get('priority') or 'unset'}",
        f"severity={doc.get('severity') or 'unset'}",
        f"disposition={doc.get('disposition') or 'unset'}",
        f"open={'yes' if doc.get('is_open') else 'no'}",
    ]
    lines.append("  " + "  ".join(status))

    for label, value in (
        ("module", doc.get("module")),
        ("type", doc.get("issue_type")),
        ("release", doc.get("release")),
        ("customer", doc.get("customer_name")),
        ("partner_area", doc.get("partner_area")),
        ("ms_status", doc.get("ms_status")),
        ("os", doc.get("os")),
        ("version", doc.get("version")),
        ("keywords", _join(doc.get("keywords"))),
        ("categories", _join(doc.get("categories"))),
    ):
        text = _join(value)
        if text:
            lines.append(f"  {label}={text}")
    return lines


def _bug_block(bug_id: str, docs: list[dict]) -> str:
    docs.sort(key=_section_key)
    lines = _bug_header(bug_id, docs[0])

    # A bug's repro section is frequently the description verbatim, and
    # overlapping comment windows repeat their shared span. Emitting both
    # spends the budget restating text the model has already read.
    emitted: list[str] = [str(docs[0].get("synopsis", "") or "")]
    for doc in docs:
        text = str(doc.get("text", "") or "").strip()
        if not text or any(text in seen for seen in emitted):
            continue
        emitted.append(text)
        if len(text) > _SECTION_CHARS:
            text = text[:_SECTION_CHARS] + " […truncated]"
        section = doc.get("section", "text")
        label = f"{section}[{doc.get('seq', 0)}]" if section == "comments" else str(section)
        lines.append(f"  --- {label} ---")
        lines.append("  " + text.replace("\n", "\n  "))
    return "\n".join(lines)


def build_source_text(source_chunks: list[dict], max_chars: int = _SOURCE_CHARS) -> str:
    """Render retrieved bug sections as grouped, cited bug records."""
    if not source_chunks:
        return "(No source documents available)"

    by_bug: dict[str, list[dict]] = {}
    for doc in source_chunks:
        # Chunk ids are `<bug_id>-<section>-<seq>`, so the id's prefix is the
        # fallback when a projection omitted bug_id.
        bug_id = str(doc.get("bug_id") or str(doc.get("id", "?")).split("-")[0])
        by_bug.setdefault(bug_id, []).append(doc)

    blocks: list[str] = []
    used = 0
    for bug_id, docs in by_bug.items():
        block = _bug_block(bug_id, docs)
        if blocks and used + len(block) > max_chars:
            # Say so rather than letting the model assume it saw everything;
            # the answer prompt asks it to flag incomplete context.
            blocks.append(f"({len(by_bug) - len(blocks)} further retrieved bugs "
                          f"omitted for length)")
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


NO_STRUCTURED_SET = (
    "(The question states no attribute filter, so no exhaustive set applies. "
    "Answer from the graph index and source documents.)"
)

# Listing an exhaustive set costs output tokens — 66 bugs is a couple of
# thousand — and on a 16K answer model a full 60K-character context leaves too
# few to finish the list, so the enumeration stops mid-sentence. When a set is
# present the whole context is therefore capped well below the model limit and
# the sources give way, down to a floor that still carries real detail. With no
# set there is nothing to enumerate and the full budget applies as before.
_MAX_CONTEXT_CHARS = 48000
_MIN_SOURCE_CHARS = 15000


def source_budget(structured_set: str | None) -> int:
    """Characters of source text that leave the model room to list the set."""
    if not structured_set:
        return _SOURCE_CHARS
    # A small structured match (a handful of bugs, well under _SOURCE_CHARS
    # itself) should not grant *more* source room than a question with no
    # structured match at all gets -- `_MAX_CONTEXT_CHARS - len(structured_set)`
    # alone approaches 48000 as the set shrinks toward empty, which on a
    # question that also names a bug with an unusually long comment thread
    # (or a multi-hop link chain of several such bugs) built a source
    # section alone past what the rest of the prompt had room left for.
    # Capping at `_SOURCE_CHARS` keeps the no-filter case as the ceiling in
    # both directions; the floor for a large set is unchanged.
    return max(_MIN_SOURCE_CHARS, min(_SOURCE_CHARS, _MAX_CONTEXT_CHARS - len(structured_set)))


NO_BUG_LINKS = "(No bug-to-bug links were traversed for this question.)"
# The third case. `NO_BUG_LINKS` is a finding -- traversal ran and reached no
# edge -- and the model is entitled to answer "nothing is linked" from it.
# This one is not a finding: the link data could not be read at all, and the
# same sentence would then be a fabrication. Kept distinct so an unavailable
# inventory can never be quoted back as an empty one.
LINKS_UNAVAILABLE = (
    "(The bug-link inventory could not be read for this question. Do not "
    "conclude that a bug has no links: state that link information was "
    "unavailable.)")
_LINK_SYNOPSIS_CHARS = 90


# A clone edge has a direction and the direction is the whole answer: the
# question "is the bug this was cloned from also fixed" is about the original,
# not the clone. Rendered as `A --cloned_from--> B` the model has to remember
# which end the arrow makes dependent, and it does not — asked for clones that
# outlived a fixed original it reported the exact inverse. Naming the roles in
# words removes the inference: there is no arrow left to read backwards.
#
# Each entry maps a predicate to (dependent, authority, dependent_role,
# authority_role), where `dependent` is the index of the subordinate bug in
# (subject, object). `has_duplicate` points the other way to `duplicate_of`,
# so its ends are swapped rather than given their own phrasing.
_LINK_ROLES = {
    "cloned_from":  (0, 1, "clone", "original"),
    "duplicate_of": (0, 1, "duplicate", "surviving original"),
    "original_bug": (0, 1, "duplicate", "surviving original"),
    "has_duplicate": (1, 0, "duplicate", "surviving original"),
    "gated_by":     (0, 1, "blocked bug", "blocking bug"),
    "gating":       (1, 0, "blocked bug", "blocking bug"),
}
_RELATED = ("see_also",)


def _status(row: dict | None) -> str:
    if not row:
        return "status not in this corpus"
    bits = [b for b in (str(row.get("priority") or "").strip(),
                        str(row.get("disposition") or "").strip()) if b]
    state = "open" if row.get("is_open") else "closed"
    text = f"{state}" + (f", {', '.join(bits)}" if bits else "")
    # A duplicate edge is exactly the case where the two ends belong to
    # different customers -- that's the fact "is another provider tracking
    # this" turns on, and priority/disposition alone never say who filed
    # either bug. Omitting it here silently degrades every cross-customer
    # link question to "a bug exists" with no way to say whose it is.
    customer = str(row.get("customer_name") or "").strip()
    if customer:
        text += f", customer {customer}"
    return text


def _synopsis(row: dict | None) -> str:
    text = str((row or {}).get("synopsis") or "").replace("\n", " ").strip()
    if len(text) > _LINK_SYNOPSIS_CHARS:
        text = text[:_LINK_SYNOPSIS_CHARS] + "\u2026"
    return text


def _bug_id(name: str) -> str:
    return name.split(maxsplit=1)[1].strip() if str(name).startswith("Bug ") else str(name)


def normalize_links(triples: list[dict], link_predicates) -> list[tuple[str, str, str]]:
    """Collapse link triples to one (kind, dependent, authority) per bug pair.

    The same pair often carries three predicates at once -- `duplicate_of`,
    `original_bug` and `see_also` all joining 6401842 to 6401760 -- which
    listed separately reads as three findings instead of one fact.
    """
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    # `see_also` is considered last, so the pair keeps its definite relationship:
    # the two bugs are not merely "related", one is a duplicate of the other,
    # and that is what the question about the surviving original is asking.
    ordered = sorted(triples, key=lambda t: t.get("predicate") in _RELATED)
    for t in ordered:
        pred = t.get("predicate")
        if pred not in link_predicates:
            continue
        subject, obj = str(t.get("subject", "")), str(t.get("object", ""))
        if not subject.startswith("Bug ") or not obj.startswith("Bug "):
            continue
        ends = (_bug_id(subject), _bug_id(obj))
        role = _LINK_ROLES.get(pred)
        if role:
            dependent, authority = ends[role[0]], ends[role[1]]
            kind = pred
        else:
            dependent, authority = ends
            kind = "see_also"
        pair = tuple(sorted((dependent, authority)))
        if pair in seen or dependent == authority:
            continue
        seen.add(pair)
        out.append((kind, dependent, authority))
    return out


def render_bug_links(triples: list[dict], link_predicates, facets: dict,
                     both_ends_only: bool = False,
                     max_lines: int | None = None) -> str:
    """State each bug-to-bug edge in role language, with both ends resolved.

    The edge is already in the graph context, but it arrives as one line among
    a few thousand characters of triples and tens of thousands of source text,
    and a modest answer model reads straight past it -- asked whether the bug a
    fixed clone came from is also fixed, it reported that no such link exists.

    `both_ends_only` drops edges leaving the corpus. Naming an unknown bug is
    worth doing when the question asked what this one was cloned from, but a
    question that compares the two ends cannot be answered across such an edge,
    and 292 of them at 50K characters would crowd out the 15 that can.
    """
    lines, unresolved = [], []
    grouped: dict[str, list[str]] = {}
    for kind, dependent, authority in normalize_links(triples, link_predicates):
        near, far = facets.get(dependent), facets.get(authority)
        _, _, dep_role, auth_role = _LINK_ROLES.get(kind, (0, 1, "clone", "original"))
        # A bug in a duplicate cluster usually has siblings this corpus holds and
        # siblings it does not, and asked what happened to the other bug folded
        # into the same original the model named the one with no record and then
        # said it could not describe it. Merely sorting those last was not enough
        # -- the same id also appears in the graph triples, and repetition won.
        # Separated out and labelled, they stop reading as candidate answers.
        if near is None or far is None:
            missing = dependent if near is None else authority
            if kind in _RELATED:
                unresolved.append(f"- Bug {dependent} is related to Bug {authority}.")
            else:
                unresolved.append(f"- Bug {dependent} is a {dep_role} of "
                                  f"Bug {authority}.")
            unresolved[-1] += f" Bug {missing} has no record here."
            continue
        if kind in _RELATED:
            # A `see_also` edge carries no direction, so unlike the clone/
            # duplicate branch below there is no single "authority" side
            # whose synopsis obviously matters more -- a question asking
            # what the *linked* bug is about needs to hear its topic from
            # whichever side that is, and status text alone (open/closed,
            # priority) never says what a bug is actually about. Bug 5632588
            # -> Bug 6165745 named the right bug and then denied any GB300
            # firmware connection existed, because nothing here said 6165745
            # was a GB300 firmware bug.
            #
            # Only worth the extra text for a question about a named bug,
            # though: `both_ends_only` is the corpus-wide inventory with no
            # anchor, where the question is a status comparison ("still open
            # though its original was fixed") and never what either bug is
            # about -- there, adding a synopsis to every one of what can be
            # a few hundred edges is exactly the bloat that pushed a
            # no-anchor question past the model's context window.
            if both_ends_only:
                line = (f"- Bug {dependent} is related to Bug {authority}. "
                        f"Bug {dependent} is {_status(near)}; "
                        f"Bug {authority} is {_status(far)}.")
            else:
                near_text, far_text = _synopsis(near), _synopsis(far)
                line = (f"- Bug {dependent} is related to Bug {authority}. "
                        f"Bug {dependent} is {_status(near)}"
                        + (f' and reads "{near_text}"' if near_text else "") + "; "
                        f"Bug {authority} is {_status(far)}"
                        + (f' and reads "{far_text}"' if far_text else "") + ".")
        else:
            text = _synopsis(far)
            line = (f"- Bug {dependent} is a {dep_role} of Bug {authority}. "
                    f"The {dep_role} Bug {dependent} is {_status(near)}. "
                    f"The {auth_role} Bug {authority} is {_status(far)}"
                    + (f' and reads "{text}".' if text else "."))
        lines.append(line)
        if both_ends_only:
            key = ("related" if kind in _RELATED
                   else f"{bool(near.get('is_open'))}:{bool(far.get('is_open'))}")
            grouped.setdefault(key, []).append(line)
    if not lines and not unresolved:
        return NO_BUG_LINKS
    if max_lines is not None:
        lines = lines[:max_lines]
        unresolved = unresolved[:max(0, max_lines - len(lines))]
    if unresolved and not both_ends_only:
        if lines:
            lines.insert(0, "Links whose two bugs are both on record -- answer "
                            "from these:")
        lines.append("Links reaching a bug with no record here. The link is "
                     "real, but that bug's priority, status and content are "
                     "unknown, so do not offer it as the answer to what a bug "
                     "became, was folded into, or now says:")
        lines += unresolved
    if both_ends_only:
        return _grouped_links(grouped, len(lines))
    return "\n".join(lines)


# Bug ids written into a comment: "previously raised by MSFT in NVBug - 6160991".
# A keyword is required rather than matching bare 6-8 digit runs, because comment
# threads are full of part numbers, serials and firmware builds, and one wrong
# bug id in an answer costs more than every mention this finds is worth.
_MENTION_RE = re.compile(
    r"(?:nv\s*bugs?|nvbug|bugs?|dupe of|duplicate of)[\s:#\-]*(\d{6,8})", re.I)

# Most bug ids in a thread are not relationships. Over 211 bugs the pattern
# alone matches 406 times: approval bots, check-in requests, "the vfa of bug X",
# a DVS run. Requiring a word that claims a relationship nearby leaves 59, and
# spot-checking those they are all genuine -- "Closing as dupe of 5411374",
# "See also this similar request ... bug 4568959". Recall is not the goal here;
# a block of maybes would be worse than the silence it replaces.
_MENTION_CUE = re.compile(
    r"\b(dup|dupe|duplicate|same (?:issue|problem|bug|failure)"
    r"|previously (?:raised|reported|seen|filed)"
    r"|already (?:raised|reported|filed|tracked)"
    r"|tracked (?:in|as|under|by)|related (?:to|bug)|see also|known issue"
    r"|counterpart|clone[ds]?|similar (?:to|issue))\b", re.I)

# Workflow traffic that quotes a bug id without saying anything about it.
_MENTION_NOISE = re.compile(
    r"(approval request|swipat|has been updated by|changelist"
    r"|check ?in request|mail(?:ed)? to|nvbugmail)", re.I)

# Enough of the sentence to be quotable, and no more: a mention block that runs
# to paragraphs competes with the sources it is meant to annotate.
_MENTION_CHARS = 240
_MENTION_WINDOW = 140
_MENTION_BUGS = 6

NO_MENTIONS = ""

# The block's own first words, and how render_answer_prompt knows whether to
# carry the rule that governs it. One string so the two cannot drift apart.
MENTION_HEADER = "Bug ids named in the text of a bug the question asks about"


def comment_mentions(rows: list[dict], anchors: list[str], recorded: set[str],
                     facets: dict, max_bugs: int = _MENTION_BUGS) -> str:
    """Bug ids named in an anchor bug's own text but recorded as no link.

    Traversal reads edges, so a relationship that exists only as a sentence is
    invisible to it. Asked what 6165782 was linked to, the answer was "no
    recorded link" -- true of the fields, while a comment on that very bug reads
    "previously raised by MSFT in NVBug - 6160991". nvbugspro's AI search found
    it and we did not, because it reads the thread and we read the graph.

    These are candidates, never links: `recorded` ids are dropped so the block
    never restates BUG LINKS, and the wording keeps the distinction the far end
    of a real edge earns and a mention does not.
    """
    anchor_set = {str(a) for a in anchors}
    found: dict[str, tuple[str, str]] = {}
    for row in rows:
        near = str(row.get("bug_id") or "")
        text = str(row.get("text") or "").replace("\n", " ")
        for m in _MENTION_RE.finditer(text):
            bug = m.group(1)
            if bug in anchor_set or bug in recorded or bug == near or bug in found:
                continue
            context = text[max(0, m.start() - _MENTION_WINDOW):
                           m.end() + _MENTION_WINDOW]
            if _MENTION_NOISE.search(context):
                continue
            if not _MENTION_CUE.search(context):
                continue
            start = max(0, m.start() - _MENTION_CHARS // 2)
            quote = text[start:m.end() + _MENTION_CHARS // 2].strip()
            found[bug] = (near, " ".join(quote.split()))
            if len(found) >= max_bugs:
                break

    if not found:
        return NO_MENTIONS

    out = [f"{MENTION_HEADER}, where no "
           "link between them is recorded. A mention is not a link: say it is "
           "named in a comment, quote the wording, and do not call it a "
           "duplicate, clone or related bug on the strength of it."]
    for bug, (near, quote) in found.items():
        far = facets.get(bug)
        state = (f"Bug {bug} is {_status(far)} here." if far
                 else f"Bug {bug} has no record here, so its status and content "
                      f"are unknown.")
        out.append(f'- Bug {bug} is named in the text of Bug {near}: "\u2026{quote}\u2026" '
                   f"{state}")
    return "\n".join(out)


# The comparison a corpus-wide link question asks for, done here rather than
# left to the model. Given a flat list of two dozen links it read them as
# candidates to be examined one at a time, narrated each non-match, and
# concluded before reaching the third line -- which happened to be a match.
# Grouping by the comparison turns the question into picking a heading.
_LINK_GROUP_HEADINGS = (
    ("False:True", "Clone and duplicate links where the dependent bug (the "
                   "clone or duplicate) is CLOSED and the bug it came from is "
                   "still OPEN:"),
    ("True:False", "Clone and duplicate links where the dependent bug (the "
                   "clone or duplicate) is still OPEN and the bug it came from "
                   "is CLOSED:"),
    ("True:True", "Clone and duplicate links where both bugs are open:"),
    ("False:False", "Clone and duplicate links where both bugs are closed:"),
    ("related", "See-also links, which carry no clone or duplicate direction "
                "and are not evidence that either bug came from the other:"),
)


# Per-group cap on a corpus-wide, no-anchor link question. Scale's merged
# extract carries several times the link count the original driver-only
# corpus did (crosscsp's 211 bugs added on top), and this render used to put
# every resolved link's full line in every group unconditionally -- on the
# "True:True"/"both open" groups, which run into the hundreds, that alone
# pushed some prompts to or past the model's context window (a 400 from
# vLLM, not a bad answer). The groups a question like this actually needs
# ("still open though its original was fixed") tend to be the small ones;
# capping the large ones and saying so plainly costs nothing for those.
_MAX_GROUP_LINES = 40


def _grouped_links(grouped: dict[str, list[str]], total: int) -> str:
    out = [f"Every bug-to-bug link in the corpus whose two bugs are both on "
           f"record -- {total} in all, already sorted by how the status at one "
           f"end compares with the other. The group headings have done that "
           f"comparison for you: take the heading that matches the question and "
           f"give every link under it. Links elsewhere in the context reach bugs "
           f"with no record here and cannot be compared, so do not add them.",
           # Scope, stated rather than counted. A bug id can also be named in a
           # comment with no link recorded, and those are surfaced for a question
           # about one named bug. Corpus-wide they are not: measured over both
           # corpora, 17 and 18 such ids exist and all but one are bugs with no
           # record here, so listing them would offer three dozen candidates
           # whose status cannot be read against a question that compares status.
           "This inventory is of recorded links. A bug id named in comment text "
           "with no link recorded is not a link and is not listed here; almost "
           "every one of those is a bug with no record in this corpus. If the "
           "question is about one bug by number, its comment mentions are shown "
           "separately."]
    for key, heading in _LINK_GROUP_HEADINGS:
        rows = grouped.get(key) or []
        out.append("")
        # The count belongs in the heading as a claim, not appended as a tag. As
        # a trailing "(2)" it was skipped: the model gave the first of two links
        # and called it the only one.
        if not rows:
            out.append(f"{heading[:-1]} -- there are none of these.")
        elif len(rows) <= _MAX_GROUP_LINES:
            out.append(f"{heading[:-1]} -- there are exactly {len(rows)} of "
                       f"these and all {len(rows)} are listed, so if this is "
                       f"the group the question asks for, every one of them "
                       f"belongs in the answer:")
            out += rows
        else:
            out.append(f"{heading[:-1]} -- there are {len(rows)} of these; "
                       f"only the first {_MAX_GROUP_LINES} are listed below "
                       f"to keep this within budget, so give those and say "
                       f"the full count is {len(rows)} rather than presenting "
                       f"this partial list as complete:")
            out += rows[:_MAX_GROUP_LINES]
    return "\n".join(out)


NO_CORPUS_STATS = "(Corpus size unavailable.)"


def render_answer_prompt(question: str, graph_context: str, source_text: str,
                         structured_set: str | None = None,
                         bug_links: str | None = None,
                         corpus_stats: str | None = None) -> str:
    """Fill the answer template. Every caller goes through here so a new slot
    cannot be left unsubstituted in one path and leak into a prompt."""
    links = bug_links or NO_BUG_LINKS
    return (GRAPHRAG_ANSWER_PROMPT
            .replace("{mention_rule}", MENTION_RULE if MENTION_HEADER in links else "")
            .replace("{corpus_stats}", corpus_stats or NO_CORPUS_STATS)
            .replace("{graph_context}", graph_context)
            .replace("{bug_links}", links)
            .replace("{structured_set}", structured_set or NO_STRUCTURED_SET)
            .replace("{source_chunks}", source_text)
            .replace("{question}", question))


# =============================================================================
# GI Query Engine
# =============================================================================

class GIQueryEngine:
    def __init__(self, cfg: dict, *, share_from: "GIQueryEngine | None" = None):
        """One engine per corpus; `share_from` reuses another's model clients.

        Serving two corpora means two engines, because everything that is
        per-corpus -- the backend, the facet filter cached on it -- is reached
        through one. What is not per-corpus is the embedding model, and loading
        a second copy of it would double both the startup wait and the resident
        memory to hold the same weights twice.
        """
        self._cfg = cfg
        self._cosmos: CosmosClient | None = None
        self._cred: DefaultAzureCredential | None = None
        self._embedder = share_from._embedder if share_from else EmbedClient(cfg)

        llm_cfg = cfg.get("llm", {})
        # Answer token budget: project-specific query.max_answer_tokens first,
        # then upstream llm.max_completion_tokens, then a safe default.
        self._max_tokens = int(
            cfg.get("query", {}).get("max_answer_tokens")
            or llm_cfg.get("max_completion_tokens")
            or llm_cfg.get("max_tokens")
            or 1024
        )
        # Two online roles. `answer` is the final evaluation -- the only output
        # the user reads -- so it points at the frontier model. `expand` is a
        # short keyword call sitting on the critical path before retrieval can
        # finish, where latency matters and depth does not. Both fall back to
        # the top-level `llm` block when no `roles:` are configured.
        if share_from is not None:
            self._llm = share_from._llm
            self._llm_model = share_from._llm_model
            self._llm_call_kwargs = share_from._llm_call_kwargs
            self._llm_expand = share_from._llm_expand
            self._llm_expand_model = share_from._llm_expand_model
            self._llm_expand_kwargs = share_from._llm_expand_kwargs
        else:
            self._llm, self._llm_model, self._llm_call_kwargs = build_role(
                cfg, "answer", self._max_tokens
            )
            self._llm_expand, self._llm_expand_model, self._llm_expand_kwargs = build_role(
                cfg, "expand", 80
            )

        cosmos_cfg = cfg["cosmos"]
        self._db_name = cosmos_cfg["database_name"]
        self._gi_cfg = cfg.get("kg", {})
        self._triples_pk_field = self._gi_cfg.get("triples_partition_key_path", "/s").lstrip("/")
        self._query_cfg = cfg.get("query", {})
        self._backend = None
        self._extract = _UNSET
        self._freshness_task: asyncio.Task | None = None
        self._local_build_task: asyncio.Task | None = None

    async def _get_backend(self):
        """Retrieval backend selected by `index.mode` in the config.

        `local` serves the whole Graph Index from GPU memory; `cosmos` keeps
        the original remote queries. Both satisfy the same interface, so the
        rest of the pipeline is unaffected.

        If the local snapshot already exists on disk, it loads directly (a
        few seconds) -- call this once eagerly during server startup (see
        `api.py`'s `lifespan`) so that cost lands before the first real
        request, not during it. If it doesn't exist yet, this does *not*
        block on building it (`build_local_index.py`'s export took 7-10
        minutes even co-located with Cosmos): it serves from Cosmos
        immediately and kicks off the export + load as a background task,
        swapping `self._backend` to the local index the moment it's ready.
        Set `index.auto_build: false` to disable that and fail loudly
        instead, if an unattended multi-minute Cosmos export on first use is
        not wanted.
        """
        if self._backend is not None:
            return self._backend
        icfg = self._cfg.get("index", {})
        mode = str(icfg.get("mode", "cosmos")).lower()
        # Postgres needs none of the machinery below: there is no snapshot to
        # export, no freshness to compare against a second store, and no
        # separate extract to load, because facets and edges are tables in the
        # same database as the vectors. So it returns before any of it.
        if mode in ("postgres", "pg"):
            self._backend = self._make_postgres_backend(icfg)
            return self._backend
        if mode != "local":
            self._backend = await self._make_cosmos_backend()
            return self._backend

        snapshot_path = icfg.get("snapshot_path", "data/local_index")
        manifest_path = os.path.join(snapshot_path, "manifest.json")

        if os.path.exists(manifest_path):
            self._backend = self._make_local_backend(icfg)
            if bool(icfg.get("check_freshness", True)) and self._freshness_task is None:
                self._freshness_task = asyncio.create_task(self._log_freshness(snapshot_path))
            return self._backend

        if not bool(icfg.get("auto_build", True)):
            raise RuntimeError(
                f"index.mode is 'local' but no snapshot at {manifest_path} and "
                f"index.auto_build is false. Run scripts/build_local_index.py first."
            )

        log.warning("No local snapshot at %s -- serving from Cosmos while building "
                    "one in the background (this takes several minutes)", manifest_path)
        self._backend = await self._make_cosmos_backend()
        if self._local_build_task is None:
            self._local_build_task = asyncio.create_task(self._build_and_swap(icfg, snapshot_path))
        return self._backend

    async def _make_cosmos_backend(self) -> CosmosBackend:
        cosmos = await self._get_cosmos()
        db = cosmos.get_database_client(self._db_name)
        return CosmosBackend(
            db.get_container_client(self._gi_cfg.get("entities_container", "entities")),
            db.get_container_client(self._gi_cfg.get("triples_container", "triples")),
            db.get_container_client(self._gi_cfg.get("source_container", "chunks")),
            self._triples_pk_field,
            extract=self._get_extract(),
        )

    def _get_extract(self):
        """Bug facets + bug-to-bug links for cosmos mode, cached on first use.

        See graph_extract.py and docs/nvissues/cosmos-architecture.md. Missing
        on disk is not an error: it reproduces exactly today's cosmos-mode
        behaviour (structured filter and link inventory both silently off) --
        run scripts/build_extract.py to turn them on. Extract path defaults
        next to the local snapshot rather than inside it, since the extract
        is meant to outlive any one snapshot rebuild.
        """
        if self._extract is not _UNSET:
            return self._extract
        icfg = self._cfg.get("index", {})
        path = icfg.get("extract_path", "data/extract")
        from graph_extract import GraphExtract
        self._extract = GraphExtract.try_load(path)
        if self._extract is None:
            log.info("No graph extract at %s -- structured filter and link "
                     "inventory stay off in cosmos mode until "
                     "scripts/build_extract.py is run", path)
        return self._extract

    def _make_postgres_backend(self, icfg: dict):
        """One relational store for vectors, text, facets and edges.

        Credentials come from the environment (see pgconn), never from the
        config, so `config.nvissues.pg.yaml` is committable and identical on a
        laptop, on the DGX and in a Container App.

        `pool_max` bounds concurrent statements. `retrieve()` fires the vector,
        lexical and anchor fetches together, so a pool of one would serialise
        them and turn three overlapping round trips into three sequential ones.
        """
        from pg_backend import PostgresBackend
        # `dbname` is the per-corpus knob: one deployment serves several corpora
        # by pointing each engine at its own database on the same server, so the
        # credentials stay in the environment and only the database name varies.
        dsn = str(icfg.get("dsn") or "").strip()
        if not dsn and icfg.get("dbname"):
            import pgconn
            dsn = pgconn.dsn(str(icfg["dbname"]))
        return PostgresBackend(
            dsn=dsn or None,
            min_size=int(icfg.get("pool_min", 2)),
            max_size=int(icfg.get("pool_max", 8)),
            # Walking attribute edges backwards is what turns an attribute the
            # question matched into the bugs under it. It is a no-op on a
            # corpus without attribute edges, so the default matches the local
            # backend's rather than being off until someone sets it.
            reverse_edges=bool(icfg.get("reverse_edges", True)),
        )

    def _make_local_backend(self, icfg: dict):
        from gi_index import get_index
        from retrieval import LocalBackend
        index = get_index(
            icfg.get("snapshot_path", "data/local_index"),
            device=icfg.get("device", "cuda"),
            enable_bm25=bool(icfg.get("enable_bm25", True)),
        )
        return LocalBackend(index, reverse_edges=bool(icfg.get("reverse_edges", True)))

    async def _log_freshness(self, snapshot_path: str):
        """Fire-and-forget: compare the snapshot against live Cosmos and log. Never
        blocks serving and never swaps backends on its own -- staleness here is a
        signal for an operator, not an automatic rebuild trigger."""
        try:
            from snapshot_freshness import check_freshness
            report = await check_freshness(self._cfg, snapshot_path)
            if report["stale"]:
                log.warning("Local snapshot at %s is STALE vs Cosmos: %s", snapshot_path, report)
            else:
                log.info("Local snapshot at %s is fresh (built %s)", snapshot_path, report.get("built_at"))
        except Exception as e:
            log.warning("Freshness check failed (%s); continuing to serve existing snapshot", e)

    async def _build_and_swap(self, icfg: dict, snapshot_path: str):
        """Background task: export Cosmos -> snapshot, load it, then swap `self._backend`."""
        t0 = time.time()
        try:
            from scripts.build_local_index import build as build_snapshot
            await build_snapshot(self._cfg["cosmos"], snapshot_path)
            # numpy/torch/BM25 loading is blocking CPU work -- keep it off the
            # event loop so in-flight Cosmos-backed requests aren't stalled.
            backend = await asyncio.to_thread(self._make_local_backend, icfg)
            self._backend = backend
            log.info("Swapped to local Graph Index backend after %.1f min background build",
                     (time.time() - t0) / 60)
        except Exception as e:
            log.warning("Background local-index build failed (%s); staying on Cosmos", e)

    async def _get_cosmos(self) -> CosmosClient:
        """`AzureCliCredential` only ever authenticates on a machine that has run
        `az login` interactively, which a headless Container App replica
        cannot do -- RBAC mode was untestable in cosmos-mode ACA deployments
        until this was `DefaultAzureCredential`, which tries a system- or
        user-assigned managed identity first (ACA: `az containerapp identity
        assign --system-assigned`, then grant it the Cosmos DB Built-in Data
        Reader role) and falls back to the Azure CLI locally, so the same
        config works in both places. `AZURE_TENANT_ID`, if set, is picked up
        by the environment-credential and CLI fallbacks on their own; it does
        not need to be threaded through explicitly.
        """
        if self._cosmos is None:
            cosmos_cfg = self._cfg["cosmos"]
            if cosmos_cfg.get("use_rbac_auth"):
                self._cred = DefaultAzureCredential()
                self._cosmos = CosmosClient(cosmos_cfg["uri"], credential=self._cred)
            else:
                self._cosmos = CosmosClient(cosmos_cfg["uri"], cosmos_cfg.get("key", ""))
        return self._cosmos

    async def close(self):
        if self._cosmos:
            await self._cosmos.close()
        if self._cred:
            await self._cred.close()
        ranker_http = getattr(self, "_ranker_http", None)
        if ranker_http is not None:
            await ranker_http.aclose()

    async def answer(self, question: str) -> dict[str, Any]:
        """Enhanced GI-RAG pipeline: embed -> entities -> graph -> vector augment -> LLM."""
        timings: dict[str, float] = {}
        t_total = time.time()

        # Step 1: Embed question
        t0 = time.time()
        q_emb = await self._embedder.embed(question, is_query=True)
        timings["embed"] = time.time() - t0

        # Steps 2-4: entity search, graph traversal and source fetch, against
        # whichever backend index.mode selects.
        result = await retrieve(await self._get_backend(), q_emb, self._cfg)
        timings.update(result.timings)
        seed_entities = result.seed_entities
        all_triples = result.triples
        source_chunks = result.source_chunks

        if not seed_entities:
            timings["total"] = time.time() - t_total
            return {
                "answer": "No relevant entities found in the graph index.",
                "entities": [],
                "triples": [],
                "timings": timings,
            }

        # Step 5: Build prompt and call LLM
        t0 = time.time()
        graph_context = self._build_graph_context(seed_entities, all_triples)
        source_text = self._build_source_text(source_chunks)

        prompt = render_answer_prompt(question, graph_context, source_text)

        resp = await self._llm.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
            **self._llm_call_kwargs,
        )
        answer = resp.choices[0].message.content or ""
        timings["llm"] = time.time() - t0

        timings["total"] = time.time() - t_total

        return {
            "answer": answer,
            "entities_found": len(seed_entities),
            "triples_found": len(all_triples),
            "source_docs": len(source_chunks),
            "timings": timings,
        }

    def _build_graph_context(self, entities: list[dict], triples: list[dict]) -> str:
        """Format graph data for LLM prompt."""
        lines = []
        lines.append("ENTITIES:")
        for e in entities[:10]:
            lines.append(f"  - {e['name']} ({e.get('relation_count', 0)} relations)")

        lines.append("\nFACTS:")
        for t in triples:
            conf = t.get("confidence", "")
            conf_str = f" [conf={conf}]" if conf else ""
            lines.append(f"  ({t.get('subject','')}) --[{t.get('predicate','')}]--> ({t.get('object','')}){conf_str}")

        return "\n".join(lines)

    def _build_source_text(self, source_chunks: list[dict],
                           max_chars: int = _SOURCE_CHARS) -> str:
        """Format retrieved bug sections for the answer prompt."""
        return build_source_text(source_chunks, max_chars)


# =============================================================================
# CLI: run benchmark with GI query
# =============================================================================

async def run_benchmark(config_path: str, questions_path: str | None = None):
    """Run benchmark questions through GI query engine."""
    cfg = load_config(config_path)
    qfile = questions_path or cfg.get("paths", {}).get("questions_file", "data/food.json")

    with open(qfile) as f:
        questions = json.load(f)

    print(f"GI-RAG Benchmark: {len(questions)} questions")
    print(f"Config: {config_path}")
    print("=" * 60)

    engine = GIQueryEngine(cfg)

    # Warm up embedding model
    from gi_builder import embed_sync
    embed_sync("warmup")

    # Run all questions in parallel
    wall_start = time.time()

    async def _answer_one(i, q):
        q_text = q.get("question_text", "")
        q_id = q.get("question_id", f"q{i}")
        result = await engine.answer(q_text)
        return q_id, q_text, q, result

    tasks = [_answer_one(i, q) for i, q in enumerate(questions)]
    raw_results = await asyncio.gather(*tasks)
    wall_time = time.time() - wall_start

    results = []
    total_time = 0.0
    for q_id, q_text, q, result in raw_results:
        total_time += result["timings"]["total"]
        print(f"\n[{q_id}] {q_text[:70]}...")
        print(f"  Time: {result['timings']['total']:.2f}s "
              f"(embed={result['timings'].get('embed', 0):.2f}s, "
              f"entities={result['timings'].get('entity_search', 0):.2f}s, "
              f"graph={result['timings'].get('graph_traversal', 0):.2f}s, "
              f"source={result['timings'].get('source_fetch', 0):.2f}s, "
              f"llm={result['timings'].get('llm', 0):.2f}s)")
        print(f"  Found: {result.get('entities_found', 0)} entities, "
              f"{result.get('triples_found', 0)} triples, "
              f"{result.get('source_docs', 0)} source docs")
        print(f"  Answer: {result['answer'][:150]}...")

        results.append({
            "question_id": q_id,
            "question_text": q_text,
            "answer": result["answer"],
            "ground_truth": q.get("answer", ""),
            "llm_model": cfg["llm"]["model"],
            "embed_model": "Qwen/Qwen3-Embedding-0.6B",
            "mode": "gi-rag",
            "timings": result["timings"],
            "entities_found": result.get("entities_found", 0),
            "triples_found": result.get("triples_found", 0),
        })

    await engine.close()

    print("\n" + "=" * 60)
    print(f"WALL TIME: {wall_time:.1f}s for {len(questions)} questions (parallel)")
    print(f"SUM of per-question times: {total_time:.1f}s "
          f"(avg {total_time / len(questions):.1f}s/question)")
    print("=" * 60)

    # Save results
    out_dir = cfg.get("paths", {}).get("output_root", "out_gi")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_file = os.path.join(out_dir, f"gi_answers_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GI-RAG Query Engine for Food")
    parser.add_argument("--config", default="my.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--question", default=None, help="Single question to answer")
    args = parser.parse_args()

    if args.question:
        cfg = load_config(args.config)
        engine = GIQueryEngine(cfg)

        async def _single():
            result = await engine.answer(args.question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nTimings: {json.dumps(result['timings'], indent=2)}")
            await engine.close()

        asyncio.run(_single())
    else:
        asyncio.run(run_benchmark(args.config, args.questions))


if __name__ == "__main__":
    main()
