"""Exhaustive attribute filtering over the bug corpus.

Vector search retrieves what a question is *about*; it cannot enumerate what a
question *selects*. Asked which bugs sit in the Audio partner area, the
embedding path returns the dozen chunks that read most like the question and
the answer cites three of the thirty-six that actually qualify. The predicate
is exact and the corpus is small, so the honest way to answer is to evaluate
the predicate over every bug rather than to hope the top-k contains them all.

This module turns a question into facet predicates using the corpus's own
vocabulary — no extra LLM round trip, so it costs nothing the latency budget
notices — evaluates them over all bugs, and renders the complete matching set
for the prompt.

    ff = FacetFilter(bug_rows)
    sel = ff.select("Which system-crash bugs are open on Krypton?")
    sel.predicates   # {'release': {'Krypton'}, 'severity': {'1-System Crash'}, 'is_open': True}
    sel.rows         # every qualifying bug, not a top-k sample
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Columns worth filtering on. Each is a bug-level facet denormalised onto every
# chunk row, so the value is identical across a bug's sections.
FACET_COLUMNS = (
    "release", "system", "partner_area", "module", "issue_type", "ms_status",
    "customer_name", "priority", "severity", "disposition",
)

# `is_open` is a qualifier, not a subject: almost every question mentions open
# or closed, and on its own it selects two thirds of the corpus. An exhaustive
# list is only worth showing when the question named something specific.
QUALIFIER_ONLY = frozenset({"is_open"})

# Above this share of the corpus the "set" is not a selection, and listing it
# would crowd out the sources without answering anything.
MAX_CORPUS_SHARE = 0.35

# Values whose surface form is an ordinary English word. Matching these on a
# bare token turns "are there any active regressions" into a hard filter on
# ms_status=Active, which silently discards most of the corpus. Multi-word
# values are unaffected; only the single-token form is suppressed.
_GENERIC_VALUES = frozenset(
    "active open closed new other none unknown general all yes no na "
    "fixed verified pending complete completed done low high medium normal "
    "bug issue defect task request feature change error problem".split()
)

_WORD = re.compile(r"[a-z0-9]+")
# "1-System Crash" -> "System Crash"; "P0-Must have" -> "Must have".
_RANK_PREFIX = re.compile(r"^\s*(?:p\d+|\d+)\s*[-–]\s*", re.I)
_CODE_PREFIX = re.compile(r"^\s*(p\d+)\b", re.I)

_OPEN_HINT = re.compile(r"\b(open|unresolved|outstanding|still\s+active)\b", re.I)
_CLOSED_HINT = re.compile(r"\b(closed|resolved|fixed|shipped)\b", re.I)


def _surface_forms(value: str) -> set[str]:
    """The ways a facet value plausibly appears in a question.

    A severity reads as "1-System Crash" in the data and "system crash" in a
    question; a priority reads as "P0-Must have" and "P0".
    """
    value = (value or "").strip()
    if not value:
        return set()
    forms = {value.lower()}
    stripped = _RANK_PREFIX.sub("", value).strip().lower()
    if stripped:
        forms.add(stripped)
    code = _CODE_PREFIX.match(value)
    if code:
        forms.add(code.group(1).lower())
    return {f for f in forms if len(f) >= 2}


def _phrase_pattern(form: str) -> re.Pattern[str]:
    """Word-boundary match for a possibly multi-word, punctuated form."""
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-–_/]+".join(
        re.escape(tok) for tok in _WORD.findall(form)) + r"(?![a-z0-9])", re.I)


@dataclass
class Selection:
    predicates: dict[str, Any] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.predicates) and bool(self.rows)

    def describe(self) -> str:
        parts = []
        for key, val in sorted(self.predicates.items()):
            if key == "is_open":
                parts.append("state=open" if val else "state=closed")
            else:
                parts.append(f"{key}={' or '.join(sorted(val))}")
        return ", ".join(parts)


class FacetFilter:
    """Matches questions against the corpus's own facet vocabulary."""

    def __init__(self, bug_rows: Iterable[dict]):
        self.bugs: list[dict] = list(bug_rows)
        self.by_id: dict[str, dict] = {str(b.get("bug_id")): b for b in self.bugs}
        # column -> list of (compiled surface form, canonical value)
        self._vocab: dict[str, list[tuple[re.Pattern[str], str]]] = {}
        for col in FACET_COLUMNS:
            values = {str(b.get(col) or "").strip() for b in self.bugs}
            entries: list[tuple[re.Pattern[str], str]] = []
            for value in values:
                if not value:
                    continue
                for form in _surface_forms(value):
                    tokens = _WORD.findall(form)
                    if not tokens:
                        continue
                    if len(tokens) == 1 and tokens[0] in _GENERIC_VALUES:
                        continue
                    entries.append((_phrase_pattern(form), value))
            if entries:
                # Longer forms first so "system crash" wins over a bare "crash"
                # if both are values of the same column.
                entries.sort(key=lambda e: -len(e[0].pattern))
                self._vocab[col] = entries

    def parse(self, question: str) -> dict[str, Any]:
        """Facet predicates the question states explicitly."""
        predicates: dict[str, Any] = {}
        for col, entries in self._vocab.items():
            hits = {value for pattern, value in entries if pattern.search(question)}
            if hits:
                predicates[col] = hits
        if _OPEN_HINT.search(question):
            predicates["is_open"] = True
        elif _CLOSED_HINT.search(question):
            predicates["is_open"] = False
        return predicates

    def apply(self, predicates: dict[str, Any]) -> list[dict]:
        """Every bug satisfying all predicates. Values OR within a column."""
        if not predicates:
            return []
        out = []
        for bug in self.bugs:
            for col, want in predicates.items():
                if col == "is_open":
                    if bool(bug.get("is_open")) is not bool(want):
                        break
                elif str(bug.get(col) or "").strip() not in want:
                    break
            else:
                out.append(bug)
        return out

    def select(self, question: str) -> Selection:
        """Predicates and their matches, or nothing if the filter is not selective."""
        predicates = self.parse(question)
        if not set(predicates) - QUALIFIER_ONLY:
            return Selection()
        rows = self.apply(predicates)
        if self.bugs and len(rows) > MAX_CORPUS_SHARE * len(self.bugs):
            return Selection()
        return Selection(predicates=predicates, rows=rows)


def _rank(bug: dict) -> tuple:
    """Most urgent and most recently touched first."""
    return (int(bug.get("priority_rank") or 99),
            int(bug.get("severity_rank") or 99),
            -int(bug.get("modified_ts") or 0))


def render_set(selection: Selection, limit: int = 120) -> str:
    """The complete matching set, compact enough to sit alongside full sources."""
    rows = sorted(selection.rows, key=_rank)
    total = len(rows)
    shown = rows[:limit]
    head = (f"{total} bugs in the corpus match {selection.describe()}. "
            + ("Every one is listed below."
               if total <= limit
               else f"The {limit} highest-priority are listed below."))
    lines = [head, ""]
    for bug in shown:
        # Customer is labelled, and labelled even when absent. Asked for a set
        # "with its customer" the answer read "not explicitly stated in source"
        # for bugs whose facet row says Google Platform Group: the field was
        # never rendered, so the model was left hunting it in the passages,
        # which cover a handful of a 13-bug set. Saying "not recorded" where it
        # is genuinely empty -- 6 of those 13 -- is also the difference between
        # a gap in the data and a gap in the answer.
        customer = str(bug.get("customer_name") or "").strip()
        facets = " | ".join(x for x in (
            bug.get("priority"), bug.get("severity"), bug.get("disposition"),
            "open" if bug.get("is_open") else "closed",
            f"customer: {customer or 'not recorded'}",
            bug.get("release"), bug.get("partner_area"),
        ) if x)
        synopsis = str(bug.get("synopsis") or "(no synopsis)").replace("\n", " ")
        lines.append(f"- Bug {bug.get('bug_id')} | {facets} | {synopsis}")
    return "\n".join(lines)
