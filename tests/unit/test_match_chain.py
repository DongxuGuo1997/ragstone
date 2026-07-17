"""
Unit tests for the staffing matcher (ROADMAP 9.1).

Everything runs on fakes: the LLM is a scripted response queue, the
retriever a canned query->documents mapping. What is locked down:
requirement parsing (incl. OR-groups), person tagging from filenames,
discovery aggregation by coverage breadth, fail-closed verification,
tier assignment, and the honesty contract — no strong candidate means
full_match_exists=False and a gap statement phrased as absence of
evidence.
"""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from ragstone.match import (
    Matcher,
    Requirement,
    parse_requirements,
    parse_verification,
    stamp_person_metadata,
)
from ragstone.match.matcher import index_person_chunks


class _FakeLLM:
    """Returns queued responses in order; records every prompt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("fake LLM ran out of responses")
        return SimpleNamespace(content=self.responses.pop(0))


class _FakeRetriever:
    """query -> docs mapping; unknown queries return nothing."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.mapping.get(query, [])


def _doc(person_id, name, text="chunk"):
    return Document(
        page_content=text,
        metadata={"person_id": person_id, "person_name": name},
    )


EXTRACTION = json.dumps(
    {
        "must": [
            {"kind": "skill", "alternatives": ["AUTOSAR Classic"]},
            {"kind": "skill", "alternatives": ["Jenkins", "GitLab CI"]},
            {"kind": "years", "min_years": 5},
        ],
        "nice": ["Vector CANoe"],
    }
)


def _verify_response(covered_flags, evidence="quoted line"):
    return json.dumps(
        [
            {
                "requirement": f"r{i}",
                "covered": flag,
                "evidence": evidence if flag else "",
            }
            for i, flag in enumerate(covered_flags)
        ]
    )


class TestParsing:
    def test_requirements_parse_with_or_group_and_kinds(self):
        text = json.dumps(
            {
                "must": [
                    {"kind": "skill", "alternatives": ["Jenkins", "GitLab CI"]},
                    {"kind": "years", "min_years": 5},
                    {"kind": "language", "language": "Swedish"},
                    {"kind": "domain", "domain": "automotive"},
                ],
                "nice": ["Helm"],
            }
        )
        parsed = parse_requirements(text)
        assert [r.kind for r in parsed.must] == ["skill", "years", "language", "domain"]
        assert parsed.must[0].alternatives == ["Jenkins", "GitLab CI"]
        assert parsed.must[0].label == "Jenkins or GitLab CI"
        assert parsed.must[1].detail == "5"
        assert parsed.nice == ["Helm"]

    def test_fenced_json_is_accepted(self):
        fenced = "```json\n" + EXTRACTION + "\n```"
        assert len(parse_requirements(fenced).must) == 3

    def test_no_musts_is_an_error(self):
        with pytest.raises(ValueError):
            parse_requirements(json.dumps({"must": [], "nice": ["x"]}))

    def test_prose_is_an_error(self):
        with pytest.raises(ValueError):
            parse_requirements("The requirements are AUTOSAR and Jenkins.")

    def test_verification_count_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            parse_verification(_verify_response([True]), expected=2)

    def test_verification_parses(self):
        verdicts = parse_verification(_verify_response([True, False]), expected=2)
        assert verdicts[0]["covered"] is True
        assert verdicts[1] == {"covered": False, "evidence": ""}


class TestRequirementBehavior:
    def test_skill_queries_are_the_alternatives(self):
        req = Requirement(kind="skill", alternatives=["Jenkins", "GitLab CI"])
        assert req.queries() == ["Jenkins", "GitLab CI"]

    def test_years_is_not_searchable_but_is_verified(self):
        req = Requirement(kind="years", detail="5")
        assert req.queries() == []
        assert "engagement dates" in req.verify_instruction()


class TestPersonTagging:
    def test_stamp_from_cv_filenames_only(self):
        docs = [
            Document(
                page_content="a", metadata={"file_name": "cv07_anders_lindqvist.md"}
            ),
            Document(page_content="b", metadata={"file_name": "notes.md"}),
            Document(page_content="c", metadata={}),
        ]
        assert stamp_person_metadata(docs) == 1
        assert docs[0].metadata["person_id"] == "cv07"
        assert docs[0].metadata["person_name"] == "Anders Lindqvist"
        assert "person_id" not in docs[1].metadata

    def test_index_groups_chunks_in_order(self):
        docs = [
            _doc("cv01", "A", "first"),
            _doc("cv02", "B", "other"),
            _doc("cv01", "A", "second"),
        ]
        people = index_person_chunks(docs)
        assert people["cv01"]["chunks"] == ["first", "second"]
        assert people["cv02"]["name"] == "B"

    def test_lenient_mode_tags_arbitrary_uploaded_filenames(self):
        docs = [
            Document(page_content="a", metadata={"file_name": "John_Smith_CV.pdf"}),
            Document(
                page_content="b",
                metadata={"file_name": "resume-anna-berg-2026.docx"},
            ),
            Document(page_content="c", metadata={"file_name": "cv07_ola_vik.md"}),
        ]
        assert stamp_person_metadata(docs, lenient=True) == 3
        assert docs[0].metadata["person_name"] == "John Smith"
        assert docs[1].metadata["person_name"] == "Anna Berg"
        # Bench convention still wins for bench-shaped names.
        assert docs[2].metadata["person_id"] == "cv07"
        # Distinct files stay distinct people.
        ids = {doc.metadata["person_id"] for doc in docs}
        assert len(ids) == 3

    def test_strict_mode_still_ignores_arbitrary_names(self):
        docs = [Document(page_content="a", metadata={"file_name": "John_Smith_CV.pdf"})]
        assert stamp_person_metadata(docs) == 0


def _matcher(llm_responses, retriever_mapping, people, **kwargs):
    return Matcher(
        llm=_FakeLLM(llm_responses),
        retriever=_FakeRetriever(retriever_mapping),
        people=people,
        **kwargs,
    )


PEOPLE = {
    "cv01": {"name": "Astrid Okafor", "chunks": ["cv one text"]},
    "cv02": {"name": "Marta Strand", "chunks": ["cv two text"]},
}


class TestMatchFlow:
    def test_strong_match_ranks_first_and_full_match_is_true(self):
        retriever = {
            "AUTOSAR Classic": [_doc("cv01", "Astrid Okafor")],
            "Jenkins": [_doc("cv01", "Astrid Okafor"), _doc("cv02", "Marta Strand")],
            "Vector CANoe": [_doc("cv01", "Astrid Okafor")],
        }
        result = _matcher(
            [
                EXTRACTION,
                _verify_response([True, True, True]),  # cv01: all musts
                _verify_response([False, True, True]),  # cv02: misses one
            ],
            retriever,
            PEOPLE,
        ).match("brief text")
        assert [a.person_id for a in result.candidates] == ["cv01", "cv02"]
        assert result.candidates[0].tier == "strong"
        assert result.candidates[1].tier == "partial"
        assert result.full_match_exists is True
        assert "Astrid Okafor" in result.summary

    def test_no_strong_match_is_stated_honestly(self):
        retriever = {"AUTOSAR Classic": [_doc("cv01", "Astrid Okafor")]}
        result = _matcher(
            [EXTRACTION, _verify_response([True, False, True])],
            retriever,
            PEOPLE,
        ).match("brief")
        assert result.full_match_exists is False
        assert "No candidate meets every must-have" in result.summary
        gap = result.candidates[0].gap_statement()
        assert gap.startswith("Not evidenced in the CV:")

    def test_nothing_retrieved_yields_empty_honest_result(self):
        result = _matcher([EXTRACTION], {}, PEOPLE).match("brief")
        assert result.candidates == []
        assert result.full_match_exists is False
        assert "No candidate" in result.summary

    def test_unparseable_verification_fails_closed(self):
        retriever = {"AUTOSAR Classic": [_doc("cv01", "Astrid Okafor")]}
        result = _matcher(
            [EXTRACTION, "not json", "still not json"],  # parse + retry
            retriever,
            PEOPLE,
        ).match("brief")
        candidate = result.candidates[0]
        assert candidate.tier == "weak"
        assert all(not f.covered for f in candidate.coverage)
        assert result.full_match_exists is False

    def test_shortlist_is_capped_before_verification(self):
        people = {
            f"cv{i:02d}": {"name": f"P{i}", "chunks": ["text"]} for i in range(1, 5)
        }
        retriever = {
            "AUTOSAR Classic": [_doc(f"cv{i:02d}", f"P{i}") for i in range(1, 5)]
        }
        matcher = _matcher(
            [EXTRACTION, _verify_response([True, True, True])],
            retriever,
            people,
            max_candidates=1,
        )
        result = matcher.match("brief")
        assert len(result.candidates) == 1

    def test_coverage_breadth_beats_hit_depth(self):
        # cv02 hits ONE requirement many times (many chunks); cv01 hits
        # two distinct requirements once each. cv01 must shortlist first.
        many = [_doc("cv02", "Marta Strand", f"chunk{i}") for i in range(4)]
        retriever = {
            "AUTOSAR Classic": [_doc("cv01", "Astrid Okafor")],
            "Jenkins": [_doc("cv01", "Astrid Okafor")] + many,
            "GitLab CI": many,
        }
        matcher = _matcher(
            [
                EXTRACTION,
                _verify_response([True, True, True]),
                _verify_response([False, True, False]),
            ],
            retriever,
            PEOPLE,
        )
        result = matcher.match("brief")
        assert [a.person_id for a in result.candidates][0] == "cv01"

    def test_lexical_channel_rescues_retrieval_crowd_out(self):
        # The retriever never surfaces cv02 for the skill query, but its
        # CV literally contains the phrase — the lexical channel must
        # make it a discovery candidate (measured failure: a09/cv38).
        extraction = json.dumps(
            {
                "must": [{"kind": "skill", "alternatives": ["AUTOSAR Classic"]}],
                "nice": [],
            }
        )
        people = {
            "cv01": {"name": "A", "chunks": ["works with AUTOSAR Classic daily"]},
            "cv02": {"name": "B", "chunks": ["deep AUTOSAR Classic platform work"]},
        }
        retriever = {"AUTOSAR Classic": [_doc("cv01", "A")]}  # cv02 never returned
        result = _matcher(
            [extraction, _verify_response([True]), _verify_response([True])],
            retriever,
            people,
        ).match("brief")
        assert {c.person_id for c in result.candidates} == {"cv01", "cv02"}

    def test_lexical_channel_respects_symbol_boundaries(self):
        # "Embedded C++" must not count as lexical evidence of
        # "Embedded C" — the substring trap the taxonomy regexes guard.
        extraction = json.dumps(
            {"must": [{"kind": "skill", "alternatives": ["Embedded C"]}], "nice": []}
        )
        people = {
            "cv01": {"name": "A", "chunks": ["modern Embedded C++ development"]},
            "cv02": {"name": "B", "chunks": ["ten years of Embedded C firmware"]},
        }
        result = _matcher(
            [extraction, _verify_response([True])],
            {},  # retriever returns nothing at all
            people,
        ).match("brief")
        assert [c.person_id for c in result.candidates] == ["cv02"]

    def test_stream_events_surface_progress_and_result(self):
        retriever = {"AUTOSAR Classic": [_doc("cv01", "Astrid Okafor")]}
        matcher = _matcher(
            [EXTRACTION, _verify_response([True, True, True])],
            retriever,
            PEOPLE,
        )
        events = list(matcher.stream_events("brief"))
        kinds = [e.get("event") for e in events]
        for expected in (
            "extract",
            "discover",
            "shortlist",
            "verify",
            "result",
            "done",
        ):
            assert expected in kinds, f"missing event {expected}"
        assert events[-1]["result"] is not None
        assert events[-1]["result"].candidates[0].person_id == "cv01"
