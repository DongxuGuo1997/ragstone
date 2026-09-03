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
    location_note,
    parse_requirements,
    parse_verification,
    stamp_person_metadata,
    vote_requirements,
    years_of_experience,
    years_verdict,
)

__all__ = [
    "AssignmentRequirements",
    "CandidateAssessment",
    "Matcher",
    "MatchPipeline",
    "MatchResult",
    "Requirement",
    "RequirementFinding",
    "location_note",
    "parse_requirements",
    "parse_verification",
    "stamp_person_metadata",
    "vote_requirements",
    "years_of_experience",
    "years_verdict",
]
