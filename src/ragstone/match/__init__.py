"""Staffing match: assignment brief -> ranked, evidence-backed shortlist.

ROADMAP 9.1. The matcher never sees the bench oracle or persona specs —
it reads the same free text a human staffer would: the brief and the
ingested CVs.
"""

from .matcher import (
    AssignmentRequirements,
    CandidateAssessment,
    Matcher,
    MatchPipeline,
    MatchResult,
    Requirement,
    RequirementFinding,
    parse_requirements,
    parse_verification,
    stamp_person_metadata,
)

__all__ = [
    "AssignmentRequirements",
    "CandidateAssessment",
    "Matcher",
    "MatchPipeline",
    "MatchResult",
    "Requirement",
    "RequirementFinding",
    "parse_requirements",
    "parse_verification",
    "stamp_person_metadata",
]
