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

import datetime
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
# Concurrent verification calls (independent per candidate; same bounded
# pool size the ingest enrichment uses). Sequential verification was the
# dominant match cost: ten candidates, ten serialized LLM round-trips.
VERIFY_WORKERS = 8
# One retry when the model returns unparseable JSON, then fail closed.
PARSE_RETRIES = 1
# Second-vote screening (Experiment 30, part 4): every credited quote is
# audited on its own, quotes must occur verbatim in the CV, and an item
# that fails is given one chance to be re-evidenced before it flips.
SECOND_VOTE = True
# Extraction samples per brief; >1 takes a per-item majority vote across
# samples (Experiment 30 measures whether the extra calls buy stability).
EXTRACT_SAMPLES = 1

# Programming languages a model may file under the spoken-language kind;
# they are skills. Lower-cased, matched on the whole label.
_PROGRAMMING_LANGUAGES = (
    frozenset("""c c++ c# python java kotlin rust go golang javascript typescript bash
    swift scala ruby php matlab perl sql embedded c objective-c dart lua r
    julia haskell erlang elixir groovy powershell shell ada fortran cobol
    assembly verilog vhdl systemverilog""".split())
    | frozenset({"embedded c", "objective-c"})
)

# The prompt carries literal JSON braces, so it is filled with
# str.replace("{brief}", ...) — never str.format.
EXTRACT_PROMPT = """You are a staffing assistant. Extract the \
requirements from this client assignment request, line by line.

Work through the request section by section. Sections headed like
"Requirements", "Mandatory", "Required", "Must have" or "Krav" hold
must-haves; sections headed like "Preferred", "Nice to have",
"Meriting", "Plus" or "Meriterande" hold nice-to-haves. Skip "About",
"Responsibilities" and "Personal qualities" sections entirely — soft
skills and duties are not requirements.

For EVERY sentence or bullet in a must-have or nice-to-have section,
output one entry:
{"text": "<the sentence or bullet, copied verbatim>",
 "section": "must" | "nice",
 "relation": "all" | "any",
 "items": [ ... ]}

- "relation" is "any" ONLY when the sentence offers a choice with "or"
  or "either" ("Jenkins or GitLab CI"). Everything else — lists joined
  by commas, "and", "including", "such as" — is "all": every item is
  required.
- "items": one per technology, standard, tool, ability or condition the
  sentence names, of these kinds:
  {"kind": "skill", "name": "<as written, without surrounding words>"}
    — a technology, standard, tool, protocol, programming language or
    a named ability ("hardware interfacing", "working with test rigs")
  {"kind": "years", "min_years": <integer>} — the overall experience
    requirement, once; never attach years to individual skills
  {"kind": "language", "language": "<spoken language>"} — only when the
    text requires a spoken language; never infer one from the language
    the request is written in
  {"kind": "domain", "domain": "<industry>"} — only when experience from
    an industry is stated as a requirement, not implied by the client
  {"kind": "education", "field": "<degree field as stated>"} — when a
    degree is required ("Computer Science" even if "or a related field"
    follows)
  Programming languages (C, C++, Python, Java, Embedded C, ...) are
  "skill" items, never "language". A sentence stating a degree AND
  years has two items. An item under a nice-to-have heading stays
  "nice" however strongly the prose stresses it.

Return ONLY a JSON object:
{"location": "<city or site the role is based at, if stated; else \"\">",
 "requirements": [ <entries, in order> ]}

Example (a fictional request):
  Mandatory: "Hands-on experience with widget firmware, including Alpha
  SDK and the Beta bus." ->
  {"text": "Hands-on experience with widget firmware, including Alpha SDK and the Beta bus.",
   "section": "must", "relation": "all",
   "items": [{"kind": "skill", "name": "widget firmware"},
             {"kind": "skill", "name": "Alpha SDK"},
             {"kind": "skill", "name": "Beta bus"}]}
  Mandatory: "CI experience with Gamma or Delta." ->
  {"text": "CI experience with Gamma or Delta.", "section": "must",
   "relation": "any",
   "items": [{"kind": "skill", "name": "Gamma"}, {"kind": "skill", "name": "Delta"}]}
  Preferred: "Experience in the maritime industry." ->
  {"text": "Experience in the maritime industry.", "section": "nice",
   "relation": "all", "items": [{"kind": "domain", "domain": "maritime"}]}

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

# Filled with str.replace (literal braces inside). Both prompts serve
# CAPABILITY items only; named skills are audited deterministically.
AUDIT_PROMPT = """You are auditing evidence quotes from a consultant CV \
screening.

Items (JSON list, in order):
{items}

For EACH item decide, from the QUOTE ALONE, whether it explicitly
demonstrates the ability described by "check". Be strict: the quote
must describe the person doing that kind of work; a related activity
does not count (implementing a network protocol is not hardware
interfacing; writing application code is not low-level programming).
Do not assume anything the quote does not say.

Return ONLY a JSON array, same order:
[{"requirement": "<label>", "supported": true}]"""

REPAIR_PROMPT = """You are screening a consultant CV against assignment \
requirements. The abilities below were credited on quotes that did not
demonstrate them.

CV:
---
{cv}
---

Requirements (JSON list, in order):
{items}

For EACH requirement, find the single CV line that EXPLICITLY shows the
person doing that kind of work. Copy it verbatim, at most 25 words. If
no line shows it, use "" — never substitute related work.

Return ONLY a JSON array, same order:
[{"requirement": "<label>", "evidence": "<quote or empty>"}]"""


# --------------------------------------------------------------------------
# Result types. Everything a UI needs rides in MatchResult — one call,
# no follow-up queries (the dedicated staffing UI renders from this).
# --------------------------------------------------------------------------


@dataclass
class Requirement:
    """One must-have item; alternatives satisfy it interchangeably."""

    kind: str  # "skill" | "years" | "language" | "domain" | "education"
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
        if self.kind == "education":
            return (
                f"university degree in {self.detail}"
                if self.detail
                else "university degree"
            )
        return f"{self.detail} domain experience"

    def queries(self) -> List[str]:
        """Retrieval queries used to DISCOVER candidates for this item."""
        if self.kind == "skill":
            return list(self.alternatives)
        if self.kind == "language":
            return [self.detail]
        if self.kind == "domain":
            return [f"{self.detail} projects"]
        return []  # years / education are not searchable; verification decides

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
        if self.kind == "education":
            field = self.detail or "a relevant field"
            return (
                f"A completed university degree in {field} or a related field "
                "(check the education section; a degree in a neighbouring "
                "engineering or science discipline counts as related)"
            )
        return f"Substantial project experience in the {self.detail} domain"


@dataclass
class AssignmentRequirements:
    must: List[Requirement] = field(default_factory=list)
    nice: List[str] = field(default_factory=list)
    # Where the role is based, when the brief says. Context for the
    # reader, never a must-have: people relocate and commute.
    location: str = ""


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
    location_note: str = ""  # informational; never affects the tier

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


def _item_requirement(item: Dict[str, Any]) -> Optional[Requirement]:
    """One non-skill item -> Requirement (None when malformed)."""
    kind = str(item.get("kind", ""))
    if kind == "years" and item.get("min_years") is not None:
        return Requirement(kind="years", detail=str(item["min_years"]))
    if kind == "language" and item.get("language"):
        language = str(item["language"]).strip()
        if language.lower() in _PROGRAMMING_LANGUAGES:
            return Requirement(kind="skill", alternatives=[language])
        return Requirement(kind="language", detail=language)
    if kind == "domain" and item.get("domain"):
        return Requirement(kind="domain", detail=str(item["domain"]).strip())
    if kind == "education":
        field = item.get("field") or item.get("education") or item.get("degree")
        return Requirement(kind="education", detail=str(field or "").strip())
    return None


def _parse_line_shape(raw: Dict[str, Any]) -> AssignmentRequirements:
    """The per-line schema: every requirement sentence copied verbatim,
    classified must/nice, with an explicit all/any relation. Skills
    under "any" become one OR-group; under "all" one entry each."""
    must: List[Requirement] = []
    nice: List[str] = []
    seen_must: set = set()
    seen_nice: set = set()

    def add_must(requirement: Requirement) -> None:
        key = _requirement_key(requirement)
        if key in seen_must:
            return
        seen_must.add(key)
        must.append(requirement)

    def add_nice(name: str) -> None:
        if name and name.lower() not in seen_nice:
            seen_nice.add(name.lower())
            nice.append(name)

    for entry in raw.get("requirements", []):
        if not isinstance(entry, dict):
            continue
        section = str(entry.get("section", "must")).lower()
        relation = str(entry.get("relation", "all")).lower()
        items = [i for i in entry.get("items", []) if isinstance(i, dict)]
        skills = [
            str(i.get("name", "")).strip()
            for i in items
            if str(i.get("kind", "skill")) in ("skill", "capability")
            and str(i.get("name", "")).strip()
        ]
        others = [r for r in (_item_requirement(i) for i in items) if r is not None]
        if section == "nice":
            for name in skills:
                add_nice(name)
            for requirement in others:
                if requirement.kind == "skill":
                    add_nice(requirement.alternatives[0])
                elif requirement.kind == "domain":
                    add_nice(f"{requirement.detail} industry")
                elif requirement.kind == "education":
                    add_nice(f"degree in {requirement.detail}".strip())
                elif requirement.kind == "language":
                    add_nice(f"{requirement.detail} (language)")
            continue
        if relation == "any" and len(skills) >= 2:
            add_must(Requirement(kind="skill", alternatives=skills))
        else:
            for name in skills:
                add_must(Requirement(kind="skill", alternatives=[name]))
        for requirement in others:
            if requirement.kind == "years" and any(r.kind == "years" for r in must):
                continue
            add_must(requirement)
    if not must:
        raise ValueError("extraction produced no must-have requirements")
    location = str(raw.get("location") or "").strip()
    return AssignmentRequirements(must=must, nice=nice, location=location)


def parse_requirements(text: str) -> AssignmentRequirements:
    raw = _extract_json(text, "{", "}")
    if "requirements" in raw and "must" not in raw:
        return _parse_line_shape(raw)
    must: List[Requirement] = []
    location = str(raw.get("location") or "").strip()
    for item in raw.get("must", []):
        kind = str(item.get("kind", "skill"))
        if kind in ("skill", "capability"):
            alternatives = [str(a) for a in item.get("alternatives", []) if a]
            if not alternatives and item.get("name"):
                alternatives = [str(item["name"])]
            if alternatives:
                must.append(Requirement(kind="skill", alternatives=alternatives))
        elif kind == "years" and item.get("min_years") is not None:
            must.append(Requirement(kind="years", detail=str(item["min_years"])))
        elif kind == "language" and item.get("language"):
            language = str(item["language"])
            # A model that files "C++" as a spoken language: it is a skill.
            if language.strip().lower() in _PROGRAMMING_LANGUAGES:
                must.append(Requirement(kind="skill", alternatives=[language]))
            else:
                must.append(Requirement(kind="language", detail=language))
        elif kind == "domain" and item.get("domain"):
            must.append(Requirement(kind="domain", detail=str(item["domain"])))
        elif kind == "education":
            field = item.get("field") or item.get("education") or item.get("degree")
            must.append(Requirement(kind="education", detail=str(field or "")))
        elif kind == "location":
            # Never a must; keep it as context if the top-level field is empty.
            location = location or str(item.get("location") or item.get("city") or "")
    nice = [str(s) for s in raw.get("nice", []) if s]
    if not must:
        raise ValueError("extraction produced no must-have requirements")
    return AssignmentRequirements(must=must, nice=nice, location=location)


def _requirement_key(requirement: Requirement) -> Tuple[str, Any]:
    """Canonical identity of a must item, for voting across samples."""
    if requirement.kind == "skill":
        return ("skill", frozenset(a.strip().lower() for a in requirement.alternatives))
    return (requirement.kind, requirement.detail.strip().lower())


def vote_requirements(samples: List[AssignmentRequirements]) -> AssignmentRequirements:
    """Per-item majority across extraction samples.

    A must item, a nice-to-have or a location survives when it appears
    in more than half the samples; items keep the order and wording of
    their first appearance. One sample returns as-is.
    """
    if len(samples) == 1:
        return samples[0]
    quorum = len(samples) / 2.0
    must_counts: Dict[Tuple[str, Any], int] = {}
    must_first: Dict[Tuple[str, Any], Requirement] = {}
    for sample in samples:
        for requirement in sample.must:
            key = _requirement_key(requirement)
            must_counts[key] = must_counts.get(key, 0) + 1
            must_first.setdefault(key, requirement)
    must = [must_first[k] for k, c in must_counts.items() if c > quorum]
    nice_counts: Dict[str, int] = {}
    nice_first: Dict[str, str] = {}
    for sample in samples:
        for name in sample.nice:
            nice_key = name.strip().lower()
            nice_counts[nice_key] = nice_counts.get(nice_key, 0) + 1
            nice_first.setdefault(nice_key, name)
    nice = [nice_first[k] for k, c in nice_counts.items() if c > quorum]
    location_counts: Dict[str, int] = {}
    location_first: Dict[str, str] = {}
    for sample in samples:
        if sample.location:
            loc_key = sample.location.strip().lower()
            location_counts[loc_key] = location_counts.get(loc_key, 0) + 1
            location_first.setdefault(loc_key, sample.location)
    location = ""
    if location_counts:
        best = max(location_counts, key=lambda k: location_counts[k])
        if location_counts[best] > quorum:
            location = location_first[best]
    if not must:
        raise ValueError("extraction samples agree on no must-have requirement")
    return AssignmentRequirements(must=must, nice=nice, location=location)


def location_note(wanted: str, cv_text: str) -> str:
    """Informational line: does the CV mention the assignment's city?

    Deterministic and deliberately not a gap — a CV that names another
    city is a conversation about availability, not missing evidence.
    """
    if not wanted:
        return ""
    city = re.split(r"[(,/]| or | och |\s-\s", wanted, maxsplit=1)[0].strip()
    if not city:
        return ""
    pattern = re.compile(rf"(?<!\w){re.escape(city)}(?!\w)", re.IGNORECASE)
    if pattern.search(cv_text):
        return f"Assignment location {wanted}: mentioned in the CV."
    return (
        f"Assignment location {wanted}: not mentioned in the CV — "
        "informational, not a gap."
    )


# Engagement date ranges: "2019-2022", "2021 – present", "Jan 2019 - Mar
# 2022", "2020 till 2023" (Swedish). Education entries ("MSc ..., 2014",
# "2009-2014 BSc") are excluded by line so a degree does not count as
# work; single years never match.
_MONTH = r"(?:[A-Za-z]{3,9}\.?\s+)?"
_YEAR_RANGE = re.compile(
    rf"(?<!\d){_MONTH}((?:19|20)\d{{2}})\s*(?:[-–—]|to|till)\s*{_MONTH}"
    r"((?:19|20)\d{2}|present|now|current|today|ongoing|nu|nuvarande|pågående|idag)"
    r"(?!\d)",
    re.IGNORECASE,
)
_EDUCATION_HEADING = re.compile(
    r"^(?:education|academic|studies|qualifications|utbildning|training|"
    r"certifications?|courses)\b",
    re.IGNORECASE,
)
_ANY_HEADING = re.compile(
    r"^(?:education|academic|studies|qualifications|utbildning|training|"
    r"certifications?|courses|degrees?|school|experience|work experience|"
    r"work history|work|professional experience|professional background|"
    r"employment|engagements|selected engagements|career|positions|roles|"
    r"assignments|consulting|projects|uppdrag|erfarenhet|anställningar|"
    r"profile|summary|skills|core competencies|competenc|languages|"
    r"publications|contact|other|interests|references|awards|volunteer)\b",
    re.IGNORECASE,
)
_EXPERIENCE_HEADING = re.compile(
    r"^(?:experience|work experience|work history|work|professional "
    r"experience|professional background|employment|engagements|selected "
    r"engagements|career|positions|roles|assignments|consulting|uppdrag|"
    r"erfarenhet|anställningar)\b",
    re.IGNORECASE,
)
_EDUCATION_LINE = re.compile(
    r"\b(?:b\.?sc|m\.?sc|b\.?a|m\.?a|b\.?eng|m\.?eng|b\.?tech|m\.?tech|"
    r"bachelor|master|ph\.?d|doctor|university|universitet|högskola|college|"
    r"school|institute|degree|examen|diploma|thesis|student|studies|graduated|"
    r"gpa|civilingenjör|högskoleingenjör|kandidat|magister|exchange)\b",
    re.IGNORECASE,
)


def years_of_experience(
    cv_text: str, today_year: Optional[int] = None
) -> Optional[Tuple[float, int, int, List[Tuple[int, int]]]]:
    """Years covered by the CV's engagement date ranges, as a union.

    Returns (years, first_year, last_year, spans) — spans are the merged
    intervals, so the evidence can list exactly what was counted — or
    None when the CV carries no usable date range; the caller then falls
    back to the model's reading. Overlapping engagements are not
    double-counted; an open range ends this year.

    Which ranges count: when the CV has an Experience-like heading, only
    ranges under Experience-like headings (education, certifications
    and summary timelines anywhere else are ignored, whatever their
    layout). Without such a heading, every range counts except those
    under an Education-like heading, on a degree line, or on a bare
    date line right after a degree line.
    """
    year_now = today_year or datetime.date.today().year
    lines = cv_text.splitlines()

    def _heading(line: str) -> str:
        text = line.strip().lstrip("#*-•· ").rstrip(":* ").strip()
        if text and len(text) <= 60 and _ANY_HEADING.match(text):
            return text
        return ""

    has_experience = any(
        _EXPERIENCE_HEADING.match(h) for h in map(_heading, lines) if h
    )
    intervals: List[Tuple[int, int]] = []
    in_education = False
    in_experience = False
    recent: List[str] = []  # the three lines before this one
    for line in lines:
        heading = _heading(line)
        if heading:
            in_education = bool(_EDUCATION_HEADING.match(heading))
            in_experience = bool(_EXPERIENCE_HEADING.match(heading))
            recent = [line]
            continue
        residue = _YEAR_RANGE.sub("", line)
        bare_dates = not re.search(r"[A-Za-z]{3,}", residue)
        # A bare date line belongs to whatever the previous three lines
        # describe: "Master in X / Chalmers / 2019 - 2021" puts the
        # degree two lines above the dates.
        degree_above = any(_EDUCATION_LINE.search(r) for r in recent[-3:])
        excluded = (
            in_education
            or _EDUCATION_LINE.search(line)
            or (bare_dates and degree_above)
        )
        if has_experience:
            excluded = excluded or not in_experience
        recent.append(line)
        if excluded:
            continue
        for match in _YEAR_RANGE.finditer(line):
            start = int(match.group(1))
            end_token = match.group(2)
            end = int(end_token) if end_token.isdigit() else year_now
            if end < start:
                continue
            intervals.append((start, min(end, year_now)))
    if not intervals:
        return None
    intervals.sort()
    merged: List[List[int]] = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    spans = [(a, b) for a, b in merged]
    total = float(sum(b - a for a, b in spans))
    return total, spans[0][0], spans[-1][1], spans


def years_verdict(
    requirement: Requirement, computed: Tuple[float, int, int, List[Tuple[int, int]]]
) -> Optional[Dict[str, Any]]:
    """A years verdict from arithmetic, or None when the label won't parse."""
    try:
        minimum = float(requirement.detail)
    except (TypeError, ValueError):
        return None
    total, _first, _last, spans = computed
    listed = ", ".join(f"{a}–{b}" for a, b in spans)
    return {
        "covered": total + 1e-9 >= minimum,
        "evidence": (
            f"Engagement dates {listed} add up to about {total:g} years "
            "(computed from the CV's work-history date ranges, not quoted)"
        ),
    }


def parse_audit(text: str, expected: int) -> List[bool]:
    raw = _extract_json(text, "[", "]")
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError(
            f"audit returned {len(raw) if isinstance(raw, list) else 'non-list'} items"
        )
    return [bool(item.get("supported")) for item in raw]


def parse_repair(text: str, expected: int) -> List[str]:
    raw = _extract_json(text, "[", "]")
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError(
            f"repair returned {len(raw) if isinstance(raw, list) else 'non-list'} items"
        )
    return [str(item.get("evidence", "") or "").strip() for item in raw]


def _squash(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def quote_in_cv(quote: str, cv_text: str) -> bool:
    """Is the quote verbatim in the CV? Tolerant of case, whitespace,
    punctuation and the glued tokens a DOCX conversion leaves behind —
    both sides compare on their letters and digits only."""
    core = quote.strip().strip("\"'“”‘’").strip()
    core = re.sub(r"^(\.\.\.|…)\s*|\s*(\.\.\.|…)$", "", core).strip()
    if not core:
        return False
    return _squash(core) in _squash(cv_text)


def names_any(text: str, names: List[str]) -> bool:
    """Does the text name one of the skills? Word-bounded for short names
    (C, Go, C++); letters-and-digits containment as well for names of
    four or more characters, so DOCX-glued tokens still count."""
    for name in names:
        if _phrase_pattern(name).search(text):
            return True
        squashed = _squash(name)
        if len(squashed) >= 4 and squashed in _squash(text):
            return True
    return False


def nice_named_in(cv_text: str, skill: str) -> bool:
    """Does the CV name a nice-to-have? Looser than a must-have: the
    phrase itself, its singular ("Hypervisors" ~ "Hypervisor"), or — for
    a phrase of three or more words — its first two words ("Android
    Automotive Software Development" ~ "Android Automotive apps")."""
    candidates = [skill]
    if len(skill) > 4 and skill.endswith("s") and not skill.endswith("ss"):
        candidates.append(skill[:-1])
    words = skill.split()
    if len(words) >= 3:
        candidates.append(" ".join(words[:2]))
    return any(_phrase_pattern(c).search(cv_text) for c in candidates)


_SYMBOLS = set("+#/.&")


def is_product_name(names: List[str]) -> bool:
    """Is any alternative a product-like name the CV would have to spell
    out? A single token ("Docker", "Java", "C"), or any token carrying a
    digit, a symbol, or capitals after its first letter ("PyTest",
    "CANoe", "ISO 26262", "gRPC", "C++"), or a one-letter token in a
    two-word name ("Embedded C", "MISRA C"). Everything else is a phrase
    of ordinary words — "hardware interfacing", "Python-based test
    automation", "Swedish driving license B" — that a CV can evidence
    without repeating verbatim."""
    for name in names:
        tokens = name.split()
        if len(tokens) == 1:
            return True
        for token in tokens:
            if any(ch.isdigit() or ch in _SYMBOLS for ch in token):
                return True
            for part in token.split("-"):
                letters = [ch for ch in part if ch.isalpha()]
                if len(letters) >= 3 and any(ch.isupper() for ch in letters[1:]):
                    return True
            if len(token) == 1 and token.isalpha() and len(tokens) <= 2:
                return True
    return False


def evidence_line(cv_text: str, names: List[str], max_words: int = 25) -> str:
    """The first CV line naming one of the skills, trimmed to a window of
    words around the name — verbatim evidence chosen by code."""
    for line in cv_text.splitlines():
        stripped = line.strip().lstrip("#*-•· ").strip()
        if not stripped or not names_any(stripped, names):
            continue
        words = stripped.split()
        if len(words) <= max_words:
            return stripped
        hit = next(
            (k for k, w in enumerate(words) if names_any(w, names)),
            None,
        )
        if hit is None:
            hit = next(
                (
                    k
                    for k in range(len(words))
                    if names_any(" ".join(words[k : k + 3]), names)
                ),
                0,
            )
        lo = max(0, hit - max_words // 2)
        return " ".join(words[lo : lo + max_words])
    return ""


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
        verify_workers: int = VERIFY_WORKERS,
        extract_samples: int = EXTRACT_SAMPLES,
        second_vote: bool = SECOND_VOTE,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._people = people
        self._max_candidates = max_candidates
        self._verify_workers = verify_workers
        self._extract_samples = max(1, extract_samples)
        self._second_vote = second_vote
        # Joined once: the lexical channel scans these per match and the
        # verifier reads them per candidate — at XL scale re-joining 400
        # CVs per brief was measurable waste.
        self._person_cv_text = {
            person_id: "\n\n".join(person["chunks"])
            for person_id, person in people.items()
        }
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

    def extract_requirements(self, brief: str) -> AssignmentRequirements:
        """The extraction step alone (the eval's stability probe calls it).

        With extract_samples > 1 the brief is extracted that many times
        and the samples vote per item (see vote_requirements).
        """
        prompt = EXTRACT_PROMPT.replace("{brief}", brief)
        samples = [
            self._invoke_json(prompt, parse_requirements)
            for _ in range(self._extract_samples)
        ]
        return vote_requirements(samples)

    def _extract(self, state: _MatchState) -> _MatchState:
        writer = get_stream_writer()
        requirements = self.extract_requirements(state["brief"])
        writer(
            {
                "event": "extract",
                "must": [r.label for r in requirements.must],
                "nice": requirements.nice,
                "location": requirements.location,
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
        nice_surfaced: set = set()
        rank_credit: Dict[str, float] = {}

        searched: Dict[str, List[Document]] = {}

        def _search(query: str) -> List[Document]:
            # Memoized per match: the same phrase can appear as a must
            # alternative AND a nice-to-have; one retrieval is enough.
            if query in searched:
                return searched[query]
            docs = self._retriever.invoke(query)
            searched[query] = docs
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
        # Retrieval SURFACES people for a nice-to-have; whether the CV
        # actually names it is decided lexically below. In a one-person
        # pool every query returns that person, which must not read as
        # "has every preferred item".
        for skill in requirements.nice:
            for rank, doc in enumerate(_search(skill), start=1):
                person = doc.metadata.get("person_id")
                if not person:
                    continue
                nice_surfaced.add(person)
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
        for idx, requirement in enumerate(requirements.must):
            if requirement.kind != "skill":
                continue
            patterns = [_phrase_pattern(alt) for alt in requirement.alternatives]
            for person_id, cv_text in self._person_cv_text.items():
                if any(pattern.search(cv_text) for pattern in patterns):
                    must_hits.setdefault(person_id, set()).add(idx)
        for skill in requirements.nice:
            for person_id, cv_text in self._person_cv_text.items():
                if nice_named_in(cv_text, skill):
                    hits = nice_hits.setdefault(person_id, [])
                    if skill not in hits:
                        hits.append(skill)

        candidates = sorted(
            set(must_hits) | nice_surfaced | set(nice_hits),
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
        """One screening call per candidate: every must item, full CV.

        Candidates verify CONCURRENTLY (same bounded-pool pattern as
        ingest enrichment): the calls are independent, the chat clients
        are thread-safe, and verification dominates match latency —
        sequential, ten candidates cost ten round-trips. The stream
        writer is captured on the node thread (it is ContextVar-bound;
        worker threads must not call get_stream_writer() themselves),
        and events fire from the node thread as futures land. Results
        keep shortlist order regardless of completion order.
        """
        writer = get_stream_writer()
        requirements = state["requirements"]
        items = [
            {"requirement": r.label, "check": r.verify_instruction()}
            for r in requirements.must
        ]

        def _screen(person_id: str) -> List[Dict[str, Any]]:
            cv_text = self._person_cv_text[person_id]
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
                return [{"covered": False, "evidence": ""} for _ in items]
            if not self._second_vote:
                return verdicts
            return self._audit(cv_text, requirements.must, items, verdicts)

        shortlist = state["shortlist_ids"]
        for person_id in shortlist:
            writer({"event": "verify", "candidate": self._people[person_id]["name"]})
        if self._verify_workers > 1 and len(shortlist) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=min(self._verify_workers, len(shortlist))
            ) as pool:
                all_verdicts = list(pool.map(_screen, shortlist))
        else:
            all_verdicts = [_screen(person_id) for person_id in shortlist]

        # The years requirement is arithmetic, not judgement: a verifier
        # quoting "11 years of experience" from a profile blurb credits
        # whatever the blurb claims (seen on a real CV, Experiment 30).
        # When the CV carries date ranges, the union of those ranges
        # decides; the model's reading stands only when it carries none.
        years_index = next(
            (i for i, r in enumerate(requirements.must) if r.kind == "years"), None
        )
        assessments: List[CandidateAssessment] = []
        for person_id, verdicts in zip(shortlist, all_verdicts):
            if years_index is not None:
                computed = years_of_experience(self._person_cv_text[person_id])
                if computed is not None:
                    arithmetic = years_verdict(requirements.must[years_index], computed)
                    if arithmetic is not None:
                        verdicts = list(verdicts)
                        verdicts[years_index] = arithmetic
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
                    name=self._people[person_id]["name"],
                    coverage=coverage,
                    nice_hits=state.get("nice_hits", {}).get(person_id, []),
                    location_note=location_note(
                        requirements.location, self._person_cv_text[person_id]
                    ),
                )
            )
        return {"assessments": assessments}

    def _audit(
        self,
        cv_text: str,
        must: List[Requirement],
        items: List[Dict[str, str]],
        verdicts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Second-vote screening of every credited item.

        Named skills are audited deterministically and for free: the
        credit stands only if the CV NAMES the skill (or an alternative).
        A quote that names it and occurs in the CV is fine as it is; a
        quote that does not is replaced by the first CV line that names
        the skill; no such line means the first pass credited a sibling
        ("Android" for Java, "Vector CANoe" for CAN bus) and the item
        flips. Measured: a model judging quotes flagged a quote that
        literally named the technology and returned empty repairs for
        skills the CV lists, so the model is kept out of this path.

        A phrase of ordinary words the CV never repeats ("hardware
        interfacing", "Python-based test automation", "Swedish driving
        license B") has no product name to look for, so a stricter
        quote-only judge decides, with one repair call for anything it
        rejects; a repaired quote must occur in the CV and pass the
        judge in turn. Product names the CV never names flip outright.
        An
        unparseable audit reverts to the first pass; an unparseable
        repair flips. Years, degree, language and domain are exempt:
        years is arithmetic, the rest are holistic readings a quote-only
        judge second-guesses (measured on a degree).
        """
        result = [dict(v) for v in verdicts]
        capability_idx: List[int] = []
        for i, verdict in enumerate(verdicts):
            if not verdict["covered"] or must[i].kind != "skill":
                continue
            requirement = must[i]
            quote = verdict["evidence"]
            if quote_in_cv(quote, cv_text) and names_any(
                quote, requirement.alternatives
            ):
                continue
            line = evidence_line(cv_text, requirement.alternatives)
            if line:
                result[i] = {"covered": True, "evidence": line}
                logger.info(
                    "match audit: %s re-evidenced | first: %r | line: %r",
                    requirement.label,
                    quote,
                    line,
                )
            elif not is_product_name(requirement.alternatives):
                # A phrase of ordinary words: let the judge decide.
                capability_idx.append(i)
            else:
                result[i] = {"covered": False, "evidence": ""}
                logger.info(
                    "match audit: %s FLIPPED (CV never names it) | first: %r",
                    requirement.label,
                    quote,
                )
        if not capability_idx:
            return result
        suspect = [
            i
            for i in capability_idx
            if not quote_in_cv(verdicts[i]["evidence"], cv_text)
        ]
        to_vote = [i for i in capability_idx if i not in suspect]
        if to_vote:
            payload = [{**items[i], "quote": verdicts[i]["evidence"]} for i in to_vote]
            try:
                supported = self._invoke_json(
                    AUDIT_PROMPT.replace(
                        "{items}", json.dumps(payload, ensure_ascii=False)
                    ),
                    lambda text: parse_audit(text, len(payload)),
                )
            except ValueError:
                logger.warning("match: audit unparseable; first pass stands")
                supported = [True] * len(payload)
            suspect += [i for i, ok in zip(to_vote, supported) if not ok]
        if not suspect:
            return result
        suspect.sort()
        payload = [items[i] for i in suspect]
        try:
            repaired = self._invoke_json(
                REPAIR_PROMPT.replace("{cv}", cv_text).replace(
                    "{items}", json.dumps(payload, ensure_ascii=False)
                ),
                lambda text: parse_repair(text, len(payload)),
            )
        except ValueError:
            logger.warning("match: repair unparseable; suspect items flip")
            repaired = [""] * len(payload)
        verbatim = [
            (i, quote)
            for i, quote in zip(suspect, repaired)
            if quote and quote_in_cv(quote, cv_text)
        ]
        accepted: Dict[int, str] = {}
        if verbatim:
            payload = [{**items[i], "quote": quote} for i, quote in verbatim]
            try:
                supported = self._invoke_json(
                    AUDIT_PROMPT.replace(
                        "{items}", json.dumps(payload, ensure_ascii=False)
                    ),
                    lambda text: parse_audit(text, len(payload)),
                )
            except ValueError:
                logger.warning("match: repair audit unparseable; repairs stand")
                supported = [True] * len(payload)
            accepted = {i: q for (i, q), ok in zip(verbatim, supported) if ok}
        for i, quote in zip(suspect, repaired):
            if i in accepted:
                result[i] = {"covered": True, "evidence": accepted[i]}
                logger.info(
                    "match audit: %s re-evidenced | first: %r | repaired: %r",
                    must[i].label,
                    verdicts[i]["evidence"],
                    accepted[i],
                )
            else:
                result[i] = {"covered": False, "evidence": ""}
                logger.info(
                    "match audit: %s FLIPPED | first: %r | repair: %r",
                    must[i].label,
                    verdicts[i]["evidence"],
                    quote,
                )
        return result

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
        return self._person_cv_text.get(person_id, "")

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
    def from_pipeline(pipeline: Any, **matcher_kwargs: Any) -> Matcher:
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
            **matcher_kwargs,
        )
