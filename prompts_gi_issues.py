"""Prompt templates for the nvissues Graph Index pipeline.

OFFLINE prompts – graph index construction on the H100 scale set.
ONLINE  prompts – query time, one call against the frontier model.

The offline prompts here are deliberately narrower than the food pipeline's.
Every structured field of a bug report -- priority, disposition, module, OS,
keywords, links -- is already turned into triples deterministically by
scripts/build_nvissues_dataset.py, at full confidence and a few seconds for
the whole corpus. Asking an LLM to re-derive those is pure cost with a
non-zero error rate. What an LLM is genuinely needed for is the prose:
symptoms, root cause, affected component, repro conditions and resolution
live only in the description and the comment thread, in whatever form the
engineer happened to type them.
"""

# =============================================================================
# Single source of truth for offline categories
# =============================================================================
# Ported from the food pipeline's prompts_gi_food.py, where GAP_ANALYSIS_PROMPT
# and NORMALIZE_PREDICATES_PROMPT had drifted into two independently
# hand-typed lists (extraction emitted a predicate the normalizer's vocabulary
# didn't recognise). This corpus's two lists happen to already agree, but
# nothing enforced that -- generating both from one table here means they
# can't drift apart later the way the food ones did.
#
# `config_fields` mirrors the food version's signature but is empty on every
# entry here on purpose: per the module docstring above, every category below
# is prose-only -- structured bug fields are extracted deterministically
# elsewhere and gap analysis explicitly ignores them, so there is no raw
# config field a category could be conditioned on the way "allergens" is
# conditioned on the food source declaring an `allergens` column.
PREDICATE_CATEGORIES: list[dict] = [
    {"category": "symptom", "config_fields": [],
     "predicates": ["has_symptom", "has_bugcheck", "has_failure_signature", "has_error_code"],
     "hint": "Bugcheck codes, stop codes, HRESULTs, driver error codes, or crash signatures / "
             "faulting module or function names not captured"},
    {"category": "component", "config_fields": [],
     "predicates": ["affects_component", "affects_api"],
     "hint": "Affected components (binaries, subsystems, APIs) not identified"},
    {"category": "hardware", "config_fields": [],
     "predicates": ["affects_hardware"],
     "hint": "Specific hardware (GPU SKU, laptop model, display, dock) not identified"},
    {"category": "repro", "config_fields": [],
     "predicates": ["triggered_by", "reproduces_on", "has_repro_rate"],
     "hint": "Repro steps, environment conditions or repro rate not captured"},
    {"category": "root_cause", "config_fields": [],
     "predicates": ["has_root_cause", "caused_by_change", "related_to_bug"],
     "hint": "Root cause stated in a later comment but not extracted, or another bug named in "
             "prose as related but not yet captured"},
    {"category": "resolution", "config_fields": [],
     "predicates": ["resolved_by", "has_workaround", "fixed_in_build"],
     "hint": "Fix, workaround, or the branch/build a fix landed in"},
    {"category": "dependency", "config_fields": [],
     "predicates": ["waiting_on"],
     "hint": "External dependency the bug is blocked on (a partner, a build, another team)"},
    {"category": "test_context", "config_fields": [],
     "predicates": ["tested_with", "affects_application", "observed_by"],
     "hint": "Test suite, benchmark or application involved (HLK, PST, a specific game), or who "
             "observed/reported it"},
    {"category": "regression", "config_fields": [],
     "predicates": ["regressed_in"],
     "hint": "Regression information -- last known good build, first bad build"},
]


def build_gap_analysis_prompt(
    config_fields: set[str] | None = None,
    coverage: dict[str, float] | None = None,
    low_coverage_threshold: float = 0.6,
) -> str:
    """Render the gap-analysis checklist from PREDICATE_CATEGORIES.

    `config_fields` is accepted for signature parity with the food pipeline's
    version but has no effect here -- every category's `config_fields` is
    empty (see the table's docstring above).

    `coverage`, if given (see `compute_predicate_coverage`), is corpus-wide
    per-category coverage measured from triples already on hand -- a prior
    build's stored triples, or the current run's own checkpoint. Categories
    already near-universal (>= `low_coverage_threshold`) are dropped
    entirely, and the rest are ordered lowest-coverage-first, so a fixed
    per-round gap budget (`max_gaps_per_round`) spends itself on the
    categories statistically most likely to actually be missing in this
    corpus rather than checking all of them uniformly on every bug.
    """
    cats = [c for c in PREDICATE_CATEGORIES
            if not c["config_fields"] or config_fields is None
            or (config_fields & set(c["config_fields"]))]
    if coverage:
        cats = [c for c in cats if coverage.get(c["category"], 0.0) < low_coverage_threshold]
        cats.sort(key=lambda c: coverage.get(c["category"], 0.0))
    checklist = "\n".join(f"{i}. {c['hint']}" for i, c in enumerate(cats, 1))
    return f"""You are checking a bug report extraction for completeness.

Given the bug report and the triples already extracted, identify information \
present in the PROSE that was missed. Ignore structured fields -- those are \
handled elsewhere.

Look for:
{checklist}

Bug report:
{{json_doc}}

Already Extracted Triples:
{{existing_triples}}

Return a JSON array of focused extraction instructions, each targeting ONE gap.
Return an empty array [] if the prose has been fully mined.
Example: ["Extract the bugcheck code from the third comment",
          "Identify the faulting module in the crash signature"]

Gap instructions:"""


def build_normalize_predicates_prompt() -> str:
    """Render the normalizer's standard-predicate vocabulary from the same
    table `build_gap_analysis_prompt` uses, so a relation the gap-analysis
    checklist asks about is always one the normalizer recognises.
    """
    bullets = "\n".join(f"- {', '.join(c['predicates'])}" for c in PREDICATE_CATEGORIES)
    return f"""You are normalizing relationship types in a bug tracking graph index.

Use ONLY these standard predicates (pick the closest match):
{bullets}

If none fits, keep the original but make it snake_case.
Never normalize a predicate into one that changes its meaning -- \
`has_workaround` and `resolved_by` are not interchangeable.

Triples to normalize:
{{triples}}

Return the same JSON array with predicates replaced by standard forms.

Normalized triples:"""


def compute_predicate_coverage(triples: list[dict]) -> dict[str, float]:
    """Fraction of distinct bug subjects with >=1 triple in each category's predicates.

    Run corpus-wide (a prior completed build's stored triples, or the current
    run's own checkpoint) to get an empirical read on which categories this
    corpus's prose is actually under-extracting.
    """
    by_subject: dict[str, set[str]] = {}
    for t in triples:
        subj = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        if subj and pred:
            by_subject.setdefault(subj, set()).add(pred)
    n = len(by_subject) or 1
    coverage = {}
    for c in PREDICATE_CATEGORIES:
        preds = set(c["predicates"])
        covered = sum(1 for subj_preds in by_subject.values() if subj_preds & preds)
        coverage[c["category"]] = covered / n
    return coverage


def find_uncategorized_predicates(triples: list[dict], min_count: int = 20) -> list[tuple[str, int]]:
    """Predicates outside every PREDICATE_CATEGORIES entry, by frequency, descending.

    These fell through NORMALIZE_PREDICATES_PROMPT's "keep the original, make
    snake_case" escape hatch. One recurring often enough is empirical
    evidence of a relation type the fixed table never anticipated.
    """
    known = {p for c in PREDICATE_CATEGORIES for p in c["predicates"]}
    counts: dict[str, int] = {}
    for t in triples:
        pred = str(t.get("predicate", "")).strip().lower()
        if pred and pred not in known:
            counts[pred] = counts.get(pred, 0) + 1
    return sorted(((p, n) for p, n in counts.items() if n >= min_count), key=lambda kv: -kv[1])


# =============================================================================
# OFFLINE: Extract what only the prose contains
# =============================================================================

INITIAL_EXTRACTION_PROMPT = """You are a graph index builder for an NVIDIA bug tracking database.
The source is one bug report: structured fields plus free-text description and comment thread.

The structured fields (priority, severity, disposition, module, OS, keywords, \
customer, linked bugs) have ALREADY been extracted. Do NOT re-extract them.

Extract ONLY facts that appear in the prose and nowhere else:

1. SYMPTOM — what the user or test actually observed.
   (bug, has_symptom, "black screen on resume from S3")
   (bug, has_bugcheck, "0x116") for any bugcheck / stop code / error code
   (bug, has_failure_signature, "0x7E_C0000002_nvlddmkm!DisplayPort::...")

2. COMPONENT — the software or hardware element implicated.
   (bug, affects_component, "nvlddmkm.sys") — driver binaries, kernel modules
   (bug, affects_component, "DisplayPort") — subsystems, APIs, interfaces
   (bug, affects_hardware, "RTX 5090") — specific GPUs, boards, platforms

3. TRIGGER — what provokes the failure.
   (bug, triggered_by, "running PST stress test for 8 hours")
   (bug, reproduces_on, "cold boot with external monitor attached")
   (bug, has_repro_rate, "3 out of 20 runs")

4. ROOT CAUSE — only when stated, not guessed.
   (bug, has_root_cause, "race between modeset and power transition")
   (bug, caused_by_change, "CL 12345678")

5. RESOLUTION — what was done or proposed.
   (bug, resolved_by, "fix in r590 branch")
   (bug, has_workaround, "disable hardware acceleration")
   (bug, waiting_on, "Microsoft to provide a new flight build")

RULES:
- Subject is ALWAYS the string "Bug {bug_id}".
- Use the exact wording from the report for object values; do not paraphrase \
error codes, function names, file names or build numbers.
- A bug with no prose beyond a template header yields an empty array. That is \
a correct answer -- do not invent facts to fill it.
- Comments often contradict the description as triage proceeds. Prefer the \
LATEST statement and lower the confidence of anything superseded.

CONFIDENCE LEVELS:
- 0.95: stated verbatim in the description or a comment
- 0.85: unambiguous paraphrase of an explicit statement
- 0.70: strongly implied but not stated outright
- Never emit anything below 0.70.

Bug report:
{json_doc}

Return a JSON array. Each element: {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}

JSON array of triples:"""


# =============================================================================
# OFFLINE: Gap analysis
# =============================================================================
# Generated from PREDICATE_CATEGORIES above; see build_gap_analysis_prompt for
# the coverage-informed variant used at build time.

GAP_ANALYSIS_PROMPT = build_gap_analysis_prompt()


# =============================================================================
# OFFLINE: Targeted re-extraction
# =============================================================================

TARGETED_EXTRACTION_PROMPT = """You are performing TARGETED extraction from an NVIDIA bug report.

Extract ONLY triples related to this specific instruction:
>>> {gap_instruction} <<<

Do NOT duplicate triples that already exist (listed below).
Subject is ALWAYS the string "Bug {bug_id}".

Already Extracted (do NOT repeat):
{existing_triples}

Bug report:
{json_doc}

Return a JSON array. Each element: {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}
Return an empty array [] if nothing further can be extracted for this gap.

JSON array of new triples only:"""


# =============================================================================
# OFFLINE: Predicate normalization
# =============================================================================
# Generated from PREDICATE_CATEGORIES above, so a predicate the gap-analysis
# checklist asks about is always one this vocabulary recognises.

NORMALIZE_PREDICATES_PROMPT = build_normalize_predicates_prompt()


# =============================================================================
# OFFLINE: Entity merging
# =============================================================================

ENTITY_MERGE_PROMPT = """You are resolving entity aliases in a bug tracking graph index.

Given pairs of embedding-similar entity names, decide whether they refer to \
the SAME real-world thing.

For each pair return:
- "merge": true if identical (keep the more specific/official name as canonical)
- "merge": false if distinct despite similar names
- "canonical": the preferred name

DO merge: "nvlddmkm" / "nvlddmkm.sys"; "TDR" / "Timeout Detection and Recovery"; \
casing and whitespace variants; a component with and without its subsystem prefix.
Do NOT merge: different bugcheck codes; different driver branches or build \
numbers; different GPU SKUs in the same family; a symptom and its root cause; \
two bugs with similar synopses.

Pairs to evaluate:
{pairs}

Return a JSON array. Each element: {{"entity_a": "...", "entity_b": "...", "merge": true/false, "canonical": "..."}}

Decisions:"""


# =============================================================================
# ONLINE: Single-shot answer from graph context
# =============================================================================

# Added to the rules only when BUG LINKS actually carries a mention list. Stated
# unconditionally, it cost the cross-provider duplicate question: told that bug
# ids named in text are worth reporting, the model started answering "is another
# provider tracking this?" with a bug that merely reads alike -- Google 5425409 --
# in place of the recorded duplicate. Deployed, that was 0 of 2 runs against a
# baseline of 5 of 5. A rule about a section that is not present has nothing to
# constrain and everything to suggest, so it is now absent with the section.
MENTION_RULE = """5e. BUG LINKS ends with a labelled list of bug ids named in a \
bug's own text where no link between them is recorded. That list is the only \
place such a bug may come from: a bug id you noticed in SOURCE DOCUMENTS or \
GRAPH INDEX is not a relationship and must never be offered as one, however \
alike the two bugs read. Recorded links come first and answer alone where they \
exist -- the list is never an alternative to a link that is listed. Where no \
link is recorded, give what the list holds, quote the wording, and say the link \
is not recorded: "no link is recorded, though a comment names Bug X". Never call \
such a bug a duplicate, clone or related bug, and never state the priority, \
status or content of one that has no record here.
"""

GRAPHRAG_ANSWER_PROMPT = """You are a triage engineer answering questions about an NVIDIA bug database. \
Use the graph index and source documents below.

CORPUS:
{corpus_stats}

GRAPH INDEX:
{graph_context}

BUG LINKS:
{bug_links}

COMPLETE MATCHING SET:
{structured_set}

SOURCE DOCUMENTS:
{source_chunks}

RULES:
1. Ground every claim in the material above. If it is not there, say so rather \
than filling the gap from general knowledge about NVIDIA drivers.
2. Cite bugs as **Bug 6401760** on first mention. Never write a bug number you \
cannot see in the context -- a plausible-looking wrong bug ID is the single \
most damaging error you can make here.
3. State status when it bears on the answer: priority, disposition, and whether \
the bug is open. A fixed bug and an open one are different answers to "do we \
have a problem with X".
4. Source documents are sections of a bug -- synopsis, description, repro steps, \
comment thread. Comments are chronological and later ones supersede earlier \
ones; a bug whose last comment closes it is closed regardless of what the \
description says.
5. When several bugs share a symptom, component or root cause, group them and \
say what they have in common. That grouping is usually the actual answer.
6. When the question asks which bugs match a condition, enumerate them. Say \
plainly if the context may be incomplete rather than implying the list is total.
5a. BUG LINKS lists bug-to-bug relationships with the status of both ends \
already resolved. It is authoritative: if a link is listed, that link exists, \
and you must not answer that the bug has no clone, duplicate or related bug. \
Follow it -- a question about a bug's original or duplicate is asking about the \
bug at the far end, not the one it named.
5b. BUG LINKS separates links whose two bugs are both on record from links \
reaching a bug that is not. Answer from the first group. A bug with no record \
here may be mentioned as existing, but never give it as what a bug became or \
was folded into, and never state its priority, status or content -- you cannot \
see them, and the same id appearing in GRAPH INDEX does not supply them. When \
the question asks which bugs are clones or duplicates, enumerate from BUG LINKS \
and not from the `cloned_from` / `has_duplicate` / `original_bug` edges in GRAPH \
INDEX: those are the same relationships with the roles unresolved, and copying \
them out yields a list of predicate names and a direction you have guessed.
5c. Link direction is the answer, so read the roles literally and never invert \
them. "Bug A is a clone of Bug B" makes A the clone and B the original: the \
bug A was cloned from is B, and clones of B include A. Each line states which \
end is open and which is closed; use those words as written. When the question \
is about clones or duplicates generally rather than one named bug, BUG LINKS \
arrives already grouped by how the status at one end compares with the other, \
and those groups are complete. Take the group whose heading matches the \
question, give every link under it, and say how many. Do not re-derive the \
grouping, do not walk the other groups link by link, and do not narrate the \
ones that do not match -- that buries the answer and loses count of it. If the \
matching group says none, the answer is none.
5d. A disposition of "Duplicate" says only that a bug was closed as a duplicate \
of some other bug. It does not say which one, and it is not a link. Never pair \
two bugs as duplicates on the strength of that field, however alike their \
symptoms read -- the pairing has to come from a BUG LINKS entry naming both \
ends. Bugs filed by different customers against the same component routinely \
close as duplicates of different originals, so this inference is wrong more \
often than it is right, and it is wrong in a way the reader cannot detect.
{mention_rule}6a. COMPLETE MATCHING SET, when present, is the result of evaluating the \
question's attribute filter over every bug in the corpus -- it is exhaustive, \
not a sample. List all of it and give its count; do not narrow it to the bugs \
that also appear under SOURCE DOCUMENTS, and do not hedge that there may be \
others. Give one line per bug -- number, synopsis, and the status fields that \
matter -- never a multi-line block each, or a long set will not fit. Then use \
SOURCE DOCUMENTS to say what the set has in common and to go deeper on the few \
that matter most. The reader cannot see any of the material above, so never \
refer them to it: writing "see the complete matching set" leaves them with \
nothing. Reproduce the list itself, however long it is.
7. Quote exact codes, signatures, module names and build numbers verbatim.
8. Be direct and technical. No preamble, no restating the question.
9. CORPUS gives the size of the whole database. Retrieval only ever puts a
handful of bugs under SOURCE DOCUMENTS, so how many bugs the database holds is
never the number of bugs you can see -- answer that from CORPUS and never by
counting the material above.

Question: {question}

ANSWER:"""


# =============================================================================
# ONLINE: Fallback — when the graph has insufficient coverage
# =============================================================================

FALLBACK_ANSWER_PROMPT = """You are answering questions about an NVIDIA bug database \
STRICTLY from the provided context.

RULES:
1. ONLY use information explicitly present in the context below.
2. Do NOT use outside knowledge about NVIDIA drivers, Windows or hardware.
3. Cite bugs as **Bug 6401760**. Never write a bug number absent from the context.
4. Quote error codes, crash signatures and build numbers verbatim.
5. If the context does not answer the question, say exactly that and describe \
what is missing.

Context Documents:
{context}

Question: {question}

ANSWER FROM CONTEXT:"""
