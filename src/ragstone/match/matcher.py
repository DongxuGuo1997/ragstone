"""Match consultant CVs against a client assignment brief (ROADMAP 9.1).

Pipeline (LangGraph StateGraph, same shape conventions as corrective.py):

    extract -> discover -> verify -> score
                  \\-> no_candidates (conditional: nothing retrieved)

- extract: LLM parses the free-text brief into structured requirements
  (must-have skills with OR-alternatives, overall years, language,
  domain, nice-to-haves). The matcher reads ONLY the brief — golden
  labels and persona specs stay invisible to it.
- discover: each requirement becomes its own retrieval query over the
  person-tagged chunks; hits aggregate per person_id. Retrieval finds
  CANDIDATES; it does not decide coverage.
- verify: for each shortlisted candidate, one LLM call screens the full
  CV against every must-have and returns per-requirement verdicts with
  a verbatim evidence quote. Retrieval can surface a CV that merely
  brushes a topic; the verifier is what turns hits into claims.
- score: verified coverage -> tier (strong = every must-have evidenced;
  partial = exactly one missing; weak = more). Ranking is coverage
  first, nice-to-have hits second — requirement coverage, not raw
  similarity. `full_match_exists` is computed honestly: when nobody
  covers everything, the result says so instead of overselling the
  top candidate.

Gaps are phrased as "not evidenced in the CV": a CV that never mentions
Kubernetes is not proof the person cannot use it, and the wording must
not pretend otherwise.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# How many candidates survive discovery into (LLM-priced) verification.
# Sized for a bench of ~40 people; a real multi-thousand-CV pool would
# scale this with a cheaper pre-rank stage, not by widening the LLM pass.
MAX_VERIFIED_CANDIDATES = 10
# One retry when the model returns unparseable JSON, then fail closed.
PARSE_RETRIES = 1

EXTRACT_PROMPT = """You are a staffing assistant. Extract the \
requirements from this client assignment request.

Return ONLY a JSON object of this shape:
{{"must": [
    {{"kind": "skill", "alternatives": ["<skill name>", "..."]}},
    {{"kind": "years", "min_years": <integer>}},
    {{"kind": "language", "language": "<language>"}},
    {{"kind": "domain", "domain": "<industry domain>"}}
  ],
  "nice": ["<skill name>", "..."]}}

Rules:
- Create one "skill" entry PER requirement bullet or sentence, copying
  names as written. Distinct bullets are distinct requirements — NEVER
  merge them into one entry.
- Give an entry more than one alternative ONLY when that same bullet
  explicitly offers a choice ("X or Y"): those alternatives satisfy the
  requirement interchangeably.
- Include "years", "language", or "domain" entries only when the text
  states them as requirements, not preferences.
- "nice": the meriting/optional items, one skill string each.

Assignment request:
---
{brief}
---"""

VERIFY_PROMPT = """You are screening a consultant CV against assignment \
requirements.

CV:
---
{cv}
---

Requirements (JSON list, in order):
{items}

For EACH requirement decide, from the CV text alone, whether it is
evidenced. Be strict: covered=true only when the CV explicitly supports
it; a technology the CV never mentions is NOT covered, and a related or
sibling technology does NOT count (a different variant, edition, or
generation is a different technology — e.g. AUTOSAR Classic is not
AUTOSAR Adaptive, Embedded C is not C++). Alternatives listed within
one requirement ("X or Y") are interchangeable — any one suffices. For
a years-of-experience requirement, add up the engagement periods. Quote
the single most convincing CV line as evidence (verbatim, at most 25
words); use "" when not covered.

Return ONLY a JSON array with one object per requirement, same order:
[{{"requirement": "<label>", "covered": true, "evidence": "<quote>"}}]"""


# --------------------------------------------------------------------------
# Result types. Everything a UI needs rides in MatchResult — one call,
# no follow-up queries (the dedicated staffing UI renders from this).
# --------------------------------------------------------------------------


@dataclass
class Requirement:
    """One must-have item; alternatives satisfy it interchangeably."""

    kind: str  # "skill" | "years" | "language" | "domain"
    alternatives: List[str] = field(default_factory=list)
    detail: str = ""  # years count / language / domain

    @property
    def label(self) -> str:
        if self.kind == "skill":
            return " or ".join(self.alternatives)
        if self.kind == "years":
            return f"at least {self.detail} years of experience"
        if self.kind == "language":
            return f"{self.detail} (working proficiency)"
        return f"{self.detail} domain experience"

    def queries(self) -> List[str]:
        """Retrieval queries used to DISCOVER candidates for this item."""
        if self.kind == "skill":
            return list(self.alternatives)
        if self.kind == "language":
            return [self.detail]
        if self.kind == "domain":
            return [f"{self.detail} projects"]
        return []  # years is not searchable; verification handles it

    def verify_instruction(self) -> str:
        """What the verifier is asked to check for this item."""
        if self.kind == "skill":
            return f"Hands-on experience with {self.label}"
        if self.kind == "years":
            return (
                f"At least {self.detail} years of professional experience "
                "in total (compute from the engagement dates)"
            )
        if self.kind == "language":
            return (
                f"Working proficiency in {self.detail} " "(check the languages section)"
            )
        return f"Substantial project experience in the {self.detail} domain"


@dataclass
class AssignmentRequirements:
    must: List[Requirement] = field(default_factory=list)
    nice: List[str] = field(default_factory=list)


@dataclass
class RequirementFinding:
    requirement: str  # Requirement.label
    covered: bool
    evidence: str  # verbatim CV quote ("" when not covered)


@dataclass
class CandidateAssessment:
    person_id: str
    name: str
    coverage: List[RequirementFinding]
    nice_hits: List[str]  # nice-to-haves seen during discovery
    tier: str = "weak"  # "strong" | "partial" | "weak"

    @property
    def missing(self) -> List[str]:
        return [f.requirement for f in self.coverage if not f.covered]

    def gap_statement(self) -> str:
        """Honest phrasing: absence of evidence, not evidence of absence."""
        if not self.missing:
            return ""
        return "Not evidenced in the CV: " + "; ".join(self.missing)


@dataclass
class MatchResult:
    requirements: AssignmentRequirements
    candidates: List[CandidateAssessment]  # ranked best-first
    full_match_exists: bool
    summary: str

    def shortlist(self, n: int = 5) -> List[CandidateAssessment]:
        return self.candidates[:n]


# --------------------------------------------------------------------------
# Person indexing over ingested chunks.
# --------------------------------------------------------------------------

_CV_FILE_RE = re.compile(r"^(cv\d+)_(.+)\.md$")
# Words that are filename furniture, not part of a person's name.
_CV_NOISE_WORDS = re.compile(
    r"\b(?:cv|resume|curriculum|vitae|final|latest|copy|v\d+|\d{4})\b",
    re.IGNORECASE,
)


def _person_from_file_name(file_name: str) -> Optional[Tuple[str, str]]:
    match = _CV_FILE_RE.match(file_name or "")
    if not match:
        return None
    person_id, slug = match.groups()
    return person_id, slug.replace("_", " ").title()


def _person_from_any_file_name(file_name: str) -> Optional[Tuple[str, str]]:
    """One uploaded file = one person: derive identity from the stem.

    Real CV files are named things like "John_Smith_CV.pdf" or
    "resume-anna-berg-2026.docx" — strip the furniture words and
    whatever remains is the display name; the full slug stays the id
    (collision-safe even when two people share a name with different
    furniture).
    """
    stem = Path(file_name or "").stem
    if not stem:
        return None
    person_id = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    if not person_id:
        return None
    name = _CV_NOISE_WORDS.sub(" ", stem.replace("_", " ").replace("-", " "))
    name = " ".join(name.split()).title()
    return person_id, name or person_id


def stamp_person_metadata(texts: List[Document], lenient: bool = False) -> int:
    """Tag every CV chunk with person_id/person_name from its filename.

    Runs after load_and_split() and before the retriever is built, so
    both BM25 and the vector store carry the tags on every retrieved
    Document. Returns the number of chunks stamped.

    Default (strict): only the bench convention `cvNN_name.md` counts —
    the eval must never accidentally tag a stray file as a candidate.
    lenient=True treats EVERY file as one person (uploaded real CVs:
    PDF/DOCX/MD with arbitrary names).
    """
    stamped = 0
    for doc in texts:
        file_name = doc.metadata.get("file_name") or ""
        person = _person_from_file_name(file_name)
        if person is None and lenient:
            person = _person_from_any_file_name(file_name)
        if person is None:
            continue
        doc.metadata["person_id"], doc.metadata["person_name"] = person
        stamped += 1
    return stamped


def index_person_chunks(texts: List[Document]) -> Dict[str, Dict[str, Any]]:
    """person_id -> {"name": ..., "chunks": [chunk texts in file order]}."""
    people: Dict[str, Dict[str, Any]] = {}
    for doc in texts:
        person_id = doc.metadata.get("person_id")
        if not person_id:
            continue
        entry = people.setdefault(
            person_id,
            {"name": doc.metadata.get("person_name", person_id), "chunks": []},
        )
        entry["chunks"].append(doc.page_content)
    return people


def _phrase_pattern(phrase: str) -> "re.Pattern[str]":
    """Case-insensitive exact-phrase matcher with symbol-safe edges.

    (?<![\\w+]) / (?![\\w+]) instead of plain \\b so that "Embedded C"
    does not match inside "Embedded C++", and "C++" does not match
    inside longer symbol runs.
    """
    return re.compile(rf"(?i)(?<![\w+]){re.escape(phrase)}(?![\w+])")


# --------------------------------------------------------------------------
# JSON parsing, following evals/judge.py's fail-closed conventions.
# --------------------------------------------------------------------------


def _extract_json(text: str, opener: str, closer: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start, end = cleaned.find(opener), cleaned.rfind(closer)
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON {opener}...{closer} in model output")
    return json.loads(cleaned[start : end + 1])


def parse_requirements(text: str) -> AssignmentRequirements:
    raw = _extract_json(text, "{", "}")
    must: List[Requirement] = []
    for item in raw.get("must", []):
        kind = str(item.get("kind", "skill"))
        if kind == "skill":
            alternatives = [str(a) for a in item.get("alternatives", []) if a]
            if alternatives:
                must.append(Requirement(kind="skill", alternatives=alternatives))
        elif kind == "years" and item.get("min_years") is not None:
            must.append(Requirement(kind="years", detail=str(item["min_years"])))
        elif kind == "language" and item.get("language"):
            must.append(Requirement(kind="language", detail=str(item["language"])))
        elif kind == "domain" and item.get("domain"):
            must.append(Requirement(kind="domain", detail=str(item["domain"])))
    nice = [str(s) for s in raw.get("nice", []) if s]
    if not must:
        raise ValueError("extraction produced no must-have requirements")
    return AssignmentRequirements(must=must, nice=nice)


def parse_verification(text: str, expected: int) -> List[Dict[str, Any]]:
    raw = _extract_json(text, "[", "]")
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError(
            f"verification returned {len(raw) if isinstance(raw, list) else 'non-list'}"
            f" items, expected {expected}"
        )
    return [
        {
            "covered": bool(item.get("covered")),
            "evidence": str(item.get("evidence", "") or ""),
        }
        for item in raw
    ]


# --------------------------------------------------------------------------
# The matcher.
# --------------------------------------------------------------------------


class _MatchState(TypedDict, total=False):
    brief: str
    requirements: AssignmentRequirements
    shortlist_ids: List[str]
    nice_hits: Dict[str, List[str]]
    assessments: List[CandidateAssessment]
    result: MatchResult


class Matcher:
    """Core matching logic over an LLM + retriever + person chunk index.

    Decoupled from Pipeline for testability (fakes plug straight in);
    `MatchPipeline.from_pipeline` wires the real thing.
    """

    def __init__(
        self,
        llm: Any,
        retriever: Any,
        people: Dict[str, Dict[str, Any]],
        max_candidates: int = MAX_VERIFIED_CANDIDATES,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._people = people
        self._max_candidates = max_candidates
        self._graph = self._build_graph()

    # -- LLM helpers -------------------------------------------------------

    def _invoke_json(self, prompt: str, parse: Any) -> Any:
        """Call the LLM and parse; one strict retry, then raise."""
        last_error: Optional[Exception] = None
        for attempt in range(PARSE_RETRIES + 1):
            text = self._llm.invoke(prompt).content
            try:
                return parse(text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "match: JSON parse failed (attempt %d): %s", attempt + 1, exc
                )
                prompt = prompt + "\n\nRespond with ONLY the JSON. No prose."
        raise ValueError(f"match: model output unparseable: {last_error}")

    # -- graph nodes -------------------------------------------------------

    def _extract(self, state: _MatchState) -> _MatchState:
        writer = get_stream_writer()
        requirements = self._invoke_json(
            EXTRACT_PROMPT.format(brief=state["brief"]), parse_requirements
        )
        writer(
            {
                "event": "extract",
                "must": [r.label for r in requirements.must],
                "nice": requirements.nice,
            }
        )
        return {"requirements": requirements}

    def _discover(self, state: _MatchState) -> _MatchState:
        """Per-requirement retrieval, aggregated per person.

        Ranking signal is REQUIREMENT COVERAGE BREADTH (how many distinct
        must items a person surfaced for), not raw hit counts — ten CAN
        bus chunks must not outweigh one AUTOSAR chunk.
        """
        writer = get_stream_writer()
        requirements = state["requirements"]
        must_hits: Dict[str, set] = {}
        nice_hits: Dict[str, List[str]] = {}
        rank_credit: Dict[str, float] = {}

        def _search(query: str) -> List[Document]:
            docs = self._retriever.invoke(query)
            writer({"event": "discover", "query": query, "hits": len(docs)})
            return docs

        for idx, requirement in enumerate(requirements.must):
            for query in requirement.queries():
                for rank, doc in enumerate(_search(query), start=1):
                    person = doc.metadata.get("person_id")
                    if not person:
                        continue
                    must_hits.setdefault(person, set()).add(idx)
                    rank_credit[person] = rank_credit.get(person, 0.0) + 1.0 / rank
        for skill in requirements.nice:
            for rank, doc in enumerate(_search(skill), start=1):
                person = doc.metadata.get("person_id")
                if not person:
                    continue
                hits = nice_hits.setdefault(person, [])
                if skill not in hits:
                    hits.append(skill)
                rank_credit[person] = rank_credit.get(person, 0.0) + 0.5 / rank

        # Lexical channel. Retrieval rank alone is a bad gatekeeper for
        # NAMED skills: a true match whose mentions are textually weak
        # can lose its top-k spots to near-miss profiles (measured:
        # a09/cv38 — "AUTOSAR Adaptive" CVs crowded out a safety lead
        # who lists "AUTOSAR Classic" once). If the exact phrase is in
        # the CV, the person is a discovery candidate for that item,
        # whatever the retriever ranked; verification still decides
        # coverage. Word-boundary regexes avoid the substring traps
        # ("Embedded C" inside "Embedded C++").
        cv_texts = {
            person_id: "\n".join(person["chunks"])
            for person_id, person in self._people.items()
        }
        for idx, requirement in enumerate(requirements.must):
            if requirement.kind != "skill":
                continue
            patterns = [_phrase_pattern(alt) for alt in requirement.alternatives]
            for person_id, cv_text in cv_texts.items():
                if any(pattern.search(cv_text) for pattern in patterns):
                    must_hits.setdefault(person_id, set()).add(idx)
        for skill in requirements.nice:
            pattern = _phrase_pattern(skill)
            for person_id, cv_text in cv_texts.items():
                if pattern.search(cv_text):
                    hits = nice_hits.setdefault(person_id, [])
                    if skill not in hits:
                        hits.append(skill)

        candidates = sorted(
            set(must_hits) | set(nice_hits),
            key=lambda p: (
                -len(must_hits.get(p, set())),
                -len(nice_hits.get(p, [])),
                -rank_credit.get(p, 0.0),
                p,
            ),
        )
        shortlist = [p for p in candidates if p in self._people]
        shortlist = shortlist[: self._max_candidates]
        writer({"event": "shortlist", "candidates": shortlist})
        return {"shortlist_ids": shortlist, "nice_hits": nice_hits}

    def _verify(self, state: _MatchState) -> _MatchState:
        """One screening call per candidate: every must item, full CV."""
        writer = get_stream_writer()
        requirements = state["requirements"]
        items = [
            {"requirement": r.label, "check": r.verify_instruction()}
            for r in requirements.must
        ]
        assessments: List[CandidateAssessment] = []
        for person_id in state["shortlist_ids"]:
            person = self._people[person_id]
            writer({"event": "verify", "candidate": person["name"]})
            cv_text = "\n\n".join(person["chunks"])
            try:
                verdicts = self._invoke_json(
                    VERIFY_PROMPT.format(
                        cv=cv_text, items=json.dumps(items, ensure_ascii=False)
                    ),
                    lambda text: parse_verification(text, len(items)),
                )
            except ValueError:
                # Fail closed, visibly: an unverifiable candidate must not
                # silently rank as covered.
                logger.warning("match: verification unparseable for %s", person_id)
                verdicts = [{"covered": False, "evidence": ""} for _ in items]
            coverage = [
                RequirementFinding(
                    requirement=req.label,
                    covered=verdict["covered"],
                    evidence=verdict["evidence"],
                )
                for req, verdict in zip(requirements.must, verdicts)
            ]
            assessments.append(
                CandidateAssessment(
                    person_id=person_id,
                    name=person["name"],
                    coverage=coverage,
                    nice_hits=state.get("nice_hits", {}).get(person_id, []),
                )
            )
        return {"assessments": assessments}

    def _score(self, state: _MatchState) -> _MatchState:
        tiers = {0: "strong", 1: "partial"}
        for assessment in state["assessments"]:
            assessment.tier = tiers.get(len(assessment.missing), "weak")
        ranked = sorted(
            state["assessments"],
            key=lambda a: (
                len(a.missing),
                -len(a.nice_hits),
                -sum(1 for f in a.coverage if f.covered),
                a.person_id,
            ),
        )
        full_match = any(a.tier == "strong" for a in ranked)
        summary = self._summary(ranked, full_match)
        result = MatchResult(
            requirements=state["requirements"],
            candidates=ranked,
            full_match_exists=full_match,
            summary=summary,
        )
        get_stream_writer()(
            {
                "event": "result",
                "full_match": full_match,
                "ranking": [(a.person_id, a.tier) for a in ranked],
            }
        )
        return {"result": result}

    @staticmethod
    def _summary(ranked: List[CandidateAssessment], full_match: bool) -> str:
        if not ranked:
            return (
                "No candidate in the bench surfaced for these requirements. "
                "Consider relaxing the must-haves."
            )
        strong = [a for a in ranked if a.tier == "strong"]
        if full_match:
            names = ", ".join(a.name for a in strong)
            return (
                f"{len(strong)} candidate(s) meet every must-have: {names}. "
                "Ranked by verified coverage, then meriting skills."
            )
        best = ranked[0]
        return (
            "No candidate meets every must-have — this is stated rather "
            f"than papered over. Closest: {best.name} "
            f"({best.gap_statement() or 'gap unclear'})."
        )

    def _no_candidates(self, state: _MatchState) -> _MatchState:
        result = MatchResult(
            requirements=state["requirements"],
            candidates=[],
            full_match_exists=False,
            summary=self._summary([], False),
        )
        return {"assessments": [], "result": result}

    # -- graph -------------------------------------------------------------

    def _build_graph(self) -> Any:
        graph = StateGraph(_MatchState)
        graph.add_node("extract", self._extract)
        graph.add_node("discover", self._discover)
        graph.add_node("verify", self._verify)
        graph.add_node("score", self._score)
        graph.add_node("no_candidates", self._no_candidates)
        graph.add_edge(START, "extract")
        graph.add_edge("extract", "discover")
        graph.add_conditional_edges(
            "discover",
            lambda state: "verify" if state.get("shortlist_ids") else "no_candidates",
            {"verify": "verify", "no_candidates": "no_candidates"},
        )
        graph.add_edge("verify", "score")
        graph.add_edge("score", END)
        graph.add_edge("no_candidates", END)
        return graph.compile()

    # -- public API --------------------------------------------------------

    def match(self, brief: str) -> MatchResult:
        state = self._graph.invoke({"brief": brief})
        return state["result"]

    def person_cv(self, person_id: str) -> str:
        """Full CV text for a person (chunks in file order), or ""."""
        entry = self._people.get(person_id)
        return "\n\n".join(entry["chunks"]) if entry else ""

    def stream_events(self, brief: str):
        """Yield progress dicts, then {"event": "done", "result": ...}."""
        final: Optional[MatchResult] = None
        for mode, payload in self._graph.stream(
            {"brief": brief}, stream_mode=["custom", "values"]
        ):
            if mode == "custom":
                yield payload
            elif isinstance(payload, dict) and payload.get("result") is not None:
                final = payload["result"]
        yield {"event": "done", "result": final}


class MatchPipeline:
    """Thin adapter: a ragstone Pipeline (ingested + retriever set) in,
    a Matcher out."""

    @staticmethod
    def from_pipeline(pipeline: Any) -> Matcher:
        texts = getattr(pipeline, "texts", None) or []
        people = index_person_chunks(texts)
        if not people:
            raise ValueError(
                "no person-tagged chunks found — call stamp_person_metadata"
                "(pipeline.texts) after load_and_split() and before the "
                "retriever is built"
            )
        return Matcher(
            llm=pipeline.llm_proxy.get_llm(),
            retriever=pipeline.get_retriever(),
            people=people,
        )
