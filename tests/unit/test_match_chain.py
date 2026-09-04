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
    is_product_name,
    location_note,
    nice_named_in,
    parse_requirements,
    parse_verification,
    quote_in_cv,
    stamp_person_metadata,
    vote_requirements,
    years_of_experience,
    years_verdict,
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
    # verify_workers=1 keeps scripted response order deterministic; the
    # parallel path gets its own order-insensitive test below.
    kwargs.setdefault("verify_workers", 1)
    # The second-vote audit adds scripted calls; tests that exercise it
    # switch it on explicitly.
    kwargs.setdefault("second_vote", False)
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

    def test_parallel_verification_preserves_shortlist_order(self):
        # Four candidates verified concurrently; responses are identical
        # (order-insensitive by construction), and the assessments must
        # come back in shortlist order regardless of completion order.
        extraction = json.dumps(
            {
                "must": [{"kind": "skill", "alternatives": ["AUTOSAR Classic"]}],
                "nice": [],
            }
        )
        people = {
            f"cv{i:02d}": {"name": f"P{i}", "chunks": ["AUTOSAR Classic work"]}
            for i in range(1, 5)
        }
        retriever = {
            "AUTOSAR Classic": [_doc(f"cv{i:02d}", f"P{i}") for i in range(1, 5)]
        }
        matcher = _matcher(
            [extraction] + [_verify_response([True])] * 4,
            retriever,
            people,
            verify_workers=4,
        )
        result = matcher.match("brief")
        assert len(result.candidates) == 4
        assert all(c.tier == "strong" for c in result.candidates)
        # Deterministic final ranking (ties broken by person_id).
        assert [c.person_id for c in result.candidates] == [
            "cv01",
            "cv02",
            "cv03",
            "cv04",
        ]

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


# --------------------------------------------------------------------------
# Experiment 30: long-form RFQ extraction — new kinds, guards, voting.
# --------------------------------------------------------------------------


def _extraction(must, nice=(), location=""):
    return json.dumps({"must": must, "nice": list(nice), "location": location})


class TestRfqExtractionParsing:
    def test_education_kind_and_location_field(self):
        req = parse_requirements(
            _extraction(
                [
                    {"kind": "skill", "alternatives": ["Docker"]},
                    {"kind": "education", "field": "Computer Science"},
                ],
                location="Gothenburg (hybrid)",
            )
        )
        kinds = [r.kind for r in req.must]
        assert kinds == ["skill", "education"]
        assert req.must[1].label == "university degree in Computer Science"
        assert "education section" in req.must[1].verify_instruction()
        assert req.must[1].queries() == []
        assert req.location == "Gothenburg (hybrid)"

    def test_programming_language_filed_as_language_becomes_skill(self):
        req = parse_requirements(
            _extraction(
                [
                    {"kind": "language", "language": "C++"},
                    {"kind": "language", "language": "Swedish"},
                ]
            )
        )
        assert [(r.kind, r.label) for r in req.must] == [
            ("skill", "C++"),
            ("language", "Swedish (working proficiency)"),
        ]

    def test_location_never_becomes_a_must(self):
        req = parse_requirements(
            _extraction(
                [
                    {"kind": "skill", "alternatives": ["Docker"]},
                    {"kind": "location", "location": "Stockholm"},
                ]
            )
        )
        assert [r.kind for r in req.must] == ["skill"]
        assert req.location == "Stockholm"


class TestVoting:
    def _sample(self, skills, nice=(), location=""):
        return parse_requirements(
            _extraction(
                [{"kind": "skill", "alternatives": list(a)} for a in skills],
                nice=nice,
                location=location,
            )
        )

    def test_majority_keeps_agreed_items_and_drops_flukes(self):
        a = self._sample([["Docker"], ["Python"]], nice=["Helm"], location="Lund")
        b = self._sample([["Docker"], ["Python"], ["Kubernetes"]], location="Lund")
        c = self._sample([["docker"], ["Python"]], nice=["Helm"], location="Malmo")
        voted = vote_requirements([a, b, c])
        assert [r.label for r in voted.must] == ["Docker", "Python"]
        assert voted.nice == ["Helm"]
        assert voted.location == "Lund"

    def test_or_group_identity_is_order_insensitive(self):
        a = self._sample([["Jenkins", "GitLab CI"]])
        b = self._sample([["GitLab CI", "Jenkins"]])
        assert len(vote_requirements([a, b]).must) == 1

    def test_single_sample_passes_through(self):
        a = self._sample([["Docker"]])
        assert vote_requirements([a]) is a

    def test_no_agreement_fails_closed(self):
        a = self._sample([["Docker"]])
        b = self._sample([["Python"]])
        with pytest.raises(ValueError):
            vote_requirements([a, b])


class TestLocationNote:
    def test_city_in_cv_is_mentioned(self):
        note = location_note(
            "Gothenburg (hybrid)", "Senior engineer - Gothenburg, Sweden"
        )
        assert note.startswith("Assignment location Gothenburg (hybrid): mentioned")

    def test_city_absent_is_informational_not_a_gap(self):
        note = location_note("Stockholm (Kista)", "Senior engineer - Lund, Sweden")
        assert "not mentioned" in note and "not a gap" in note

    def test_no_location_no_note(self):
        assert location_note("", "anything") == ""


class TestLocationFlowsThroughTheGraph:
    def test_note_rides_on_the_assessment_and_the_extract_event(self):
        extraction = _extraction(
            [{"kind": "skill", "alternatives": ["AUTOSAR Classic"]}],
            location="Gothenburg",
        )
        verification = json.dumps(
            [{"requirement": "AUTOSAR Classic", "covered": True, "evidence": "q"}]
        )
        people = {
            "cv01": {
                "name": "Astrid Okafor",
                "chunks": ["Astrid - Gothenburg, Sweden"],
            },
            "cv02": {"name": "Marta Strand", "chunks": ["Marta - Lund, Sweden"]},
        }
        retriever = {
            "AUTOSAR Classic": [
                _doc("cv01", "Astrid Okafor"),
                _doc("cv02", "Marta Strand"),
            ]
        }
        result = _matcher(
            [extraction, verification, verification], retriever, people
        ).match("brief")
        notes = {c.person_id: c.location_note for c in result.candidates}
        assert notes["cv01"].endswith("mentioned in the CV.")
        assert "not a gap" in notes["cv02"]
        assert all(c.tier == "strong" for c in result.candidates)
        assert result.requirements.location == "Gothenburg"


def _line_extraction(entries, location=""):
    return json.dumps({"location": location, "requirements": entries})


def _skill(name):
    return {"kind": "skill", "name": name}


class TestLineShapedExtraction:
    def test_conjunction_yields_one_entry_per_item(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "Hands-on experience, including AUTOSAR Classic and CAN bus.",
                        "section": "must",
                        "relation": "all",
                        "items": [_skill("AUTOSAR Classic"), _skill("CAN bus")],
                    }
                ]
            )
        )
        assert [r.label for r in req.must] == ["AUTOSAR Classic", "CAN bus"]

    def test_choice_yields_one_or_group(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "CI experience with Jenkins or GitLab CI.",
                        "section": "must",
                        "relation": "any",
                        "items": [_skill("Jenkins"), _skill("GitLab CI")],
                    }
                ]
            )
        )
        assert [r.label for r in req.must] == ["Jenkins or GitLab CI"]

    def test_preferred_domain_stays_out_of_must(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "Programming language: Embedded C.",
                        "section": "must",
                        "relation": "all",
                        "items": [_skill("Embedded C")],
                    },
                    {
                        "text": "Experience in the automotive industry.",
                        "section": "nice",
                        "relation": "all",
                        "items": [{"kind": "domain", "domain": "automotive"}],
                    },
                    {
                        "text": "Expertise in Vector CANoe.",
                        "section": "nice",
                        "relation": "all",
                        "items": [_skill("Vector CANoe")],
                    },
                ]
            )
        )
        assert [r.kind for r in req.must] == ["skill"]
        assert req.nice == ["automotive industry", "Vector CANoe"]

    def test_degree_and_years_on_one_line_are_two_items(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "University degree in Computer Science and at least 5 years.",
                        "section": "must",
                        "relation": "all",
                        "items": [
                            {"kind": "education", "field": "Computer Science"},
                            {"kind": "years", "min_years": 5},
                        ],
                    }
                ],
                location="Gothenburg",
            )
        )
        assert [(r.kind, r.detail) for r in req.must] == [
            ("education", "Computer Science"),
            ("years", "5"),
        ]
        assert req.location == "Gothenburg"

    def test_programming_language_item_under_language_kind_is_a_skill(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "Programming languages C++, Python, Java",
                        "section": "must",
                        "relation": "all",
                        "items": [
                            {"kind": "language", "language": "C++"},
                            _skill("Python"),
                            _skill("Java"),
                        ],
                    }
                ]
            )
        )
        assert [r.label for r in req.must] == ["Python", "Java", "C++"]
        assert all(r.kind == "skill" for r in req.must)

    def test_duplicate_skill_lines_collapse(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "Docker.",
                        "section": "must",
                        "relation": "all",
                        "items": [_skill("Docker")],
                    },
                    {
                        "text": "Docker again.",
                        "section": "must",
                        "relation": "all",
                        "items": [_skill("docker")],
                    },
                ]
            )
        )
        assert len(req.must) == 1


class TestYearsArithmetic:
    def test_union_of_ranges_does_not_double_count(self):
        cv = "Lead (2019-2022)\nSenior (2021 - 2024)\nJunior (2014-2016)"
        assert years_of_experience(cv, today_year=2026)[:3] == (7.0, 2014, 2024)

    def test_open_range_ends_this_year_and_months_are_ignored(self):
        cv = "Engineer, Jan 2020 – present"
        assert years_of_experience(cv, today_year=2026)[:3] == (6.0, 2020, 2026)

    def test_swedish_range_words(self):
        cv = "Utvecklare 2018 till 2021\nKonsult 2021 – nu"
        assert years_of_experience(cv, today_year=2026)[:3] == (8.0, 2018, 2026)

    def test_education_lines_do_not_count(self):
        cv = "MSc Computer Science, Lund University, 2009-2014\nEngineer (2015-2017)"
        assert years_of_experience(cv, today_year=2026)[:3] == (2.0, 2015, 2017)

    def test_no_ranges_means_no_verdict(self):
        assert years_of_experience("11 years of experience", today_year=2026) is None

    def test_verdict_compares_against_the_requirement(self):
        req = Requirement(kind="years", detail="5")
        short = years_verdict(req, (4.0, 2022, 2026, [(2022, 2026)]))
        long = years_verdict(req, (7.0, 2019, 2026, [(2019, 2026)]))
        assert short is not None and not short["covered"]
        assert long is not None and long["covered"]
        assert "not quoted" in long["evidence"]
        assert (
            years_verdict(Requirement(kind="years", detail="five"), (7.0, 1, 2)) is None
        )


class TestYearsOverrideTheModel:
    def test_blurb_credit_is_overruled_by_dates(self):
        extraction = json.dumps(
            {
                "must": [
                    {"kind": "skill", "alternatives": ["Docker"]},
                    {"kind": "years", "min_years": 5},
                ],
                "nice": [],
            }
        )
        # The model credits both; the CV's only date range is two years.
        verification = json.dumps(
            [
                {"requirement": "Docker", "covered": True, "evidence": "Docker"},
                {
                    "requirement": "years",
                    "covered": True,
                    "evidence": "a seasoned engineer with 11 years of experience",
                },
            ]
        )
        people = {
            "cv01": {
                "name": "Astrid Okafor",
                "chunks": [
                    "Seasoned engineer with 11 years of experience. Docker.",
                    "Engineer - a client (2024-present)",
                ],
            }
        }
        result = _matcher(
            [extraction, verification],
            {"Docker": [_doc("cv01", "Astrid Okafor")]},
            people,
        ).match("brief")
        candidate = result.candidates[0]
        years = next(
            f for f in candidate.coverage if f.requirement.startswith("at least")
        )
        assert not years.covered
        assert "computed" in years.evidence
        assert candidate.tier == "partial"

    def test_without_dates_the_model_reading_stands(self):
        extraction = json.dumps(
            {"must": [{"kind": "years", "min_years": 5}], "nice": []}
        )
        chunk = "Seasoned engineer with 11 years of experience."
        people = {"cv01": {"name": "Astrid Okafor", "chunks": [chunk]}}
        # A years-only brief has nothing to search for, so discovery
        # surfaces nobody and no verification call is made: the honest
        # empty result, with no arithmetic to apply.
        result = _matcher([extraction], {}, people).match("brief")
        assert result.candidates == []
        assert years_of_experience(chunk) is None


class TestQuoteInCv:
    CV = "Profile\n- Skills: C++, Android, QNX\n- Automated testingHypervisor SOME/IP"

    def test_verbatim_and_case_whitespace_drift(self):
        assert quote_in_cv("Skills: C++, Android, QNX", self.CV)
        assert quote_in_cv("skills:  c++,\nandroid, qnx", self.CV)

    def test_glued_docx_tokens_still_match(self):
        assert quote_in_cv("Automated testing Hypervisor SOME/IP", self.CV)

    def test_absent_or_empty_is_false(self):
        assert not quote_in_cv("Java developer since 2010", self.CV)
        assert not quote_in_cv('""', self.CV)
        assert not quote_in_cv("", self.CV)


class TestYearsSectionAwareness:
    def test_education_section_dates_on_next_line_do_not_count(self):
        cv = (
            "Experience\nSoftware engineer, Acme\n2018 - present\n\n"
            "Education\nMaster in Systems, Control and Mechatronics\n"
            "Chalmers University of Technology\n2015 - 2017\n"
        )
        assert years_of_experience(cv, today_year=2026)[:3] == (8.0, 2018, 2026)

    def test_with_an_experience_heading_only_that_section_counts(self):
        cv = (
            "Summary\nBuilding software since 2015 - present.\n\n"
            "Education\nMaster in Mechatronics\n2019 - 2021\n"
            "Bachelor of Engineering\n2015 - 2019\n\n"
            "Work experience\nEngineer, Acme\n2021 - present\n"
            "Intern, Beta (2017-2018)\n"
        )
        years, first, last, spans = years_of_experience(cv, today_year=2026)
        assert (years, first, last) == (6.0, 2017, 2026)
        assert spans == [(2017, 2018), (2021, 2026)]

    def test_evidence_lists_the_spans(self):
        verdict = years_verdict(
            Requirement(kind="years", detail="5"),
            (6.0, 2017, 2026, [(2017, 2018), (2021, 2026)]),
        )
        assert verdict is not None
        assert "2017–2018, 2021–2026" in verdict["evidence"]

    def test_degree_two_lines_above_bare_dates_excludes_them(self):
        cv = (
            "Bachelor of Engineering, Automation\nXJTU\n2015 - 2019\n"
            "Engineer, Acme (2021-2026)\n"
        )
        assert years_of_experience(cv, today_year=2026)[:3] == (5.0, 2021, 2026)

    def test_work_history_heading_counts_as_experience(self):
        cv = (
            "WORK HISTORY\nEngineer, Acme\n2021 - 2026\n\n"
            "EDUCATION\nMaster of Science\nChalmers\n2019 - 2021\n"
            "Bachelor\nXJTU\n2015 - 2019\n"
        )
        years, first, last, spans = years_of_experience(cv, today_year=2026)
        assert spans == [(2021, 2026)]

    def test_degree_line_before_the_dates_excludes_them(self):
        cv = "MSc Computer Science, KTH\n2009-2014\nEngineer (2015-2017)"
        assert years_of_experience(cv, today_year=2026)[:3] == (2.0, 2015, 2017)


def _audit_people(chunks):
    return {"cv01": {"name": "Astrid Okafor", "chunks": chunks}}


AUDIT_EXTRACTION = json.dumps(
    {
        "must": [
            {"kind": "skill", "alternatives": ["C++"]},
            {"kind": "skill", "alternatives": ["Java"]},
        ],
        "nice": [],
    }
)
AUDIT_RETRIEVER = {"C++": [_doc("cv01", "Astrid Okafor")], "Java": []}
CV_LINES = [
    "Skills: C++ Android QNX Python",
    "Migrated perf-critical logic to C++ and optimized the render pipeline.",
]


def _verdicts(cpp_quote, java_quote):
    return json.dumps(
        [
            {"requirement": "C++", "covered": True, "evidence": cpp_quote},
            {"requirement": "Java", "covered": True, "evidence": java_quote},
        ]
    )


def _audit(*supported):
    return json.dumps(
        [
            {"requirement": r, "supported": ok}
            for r, ok in zip(("C++", "Java"), supported)
        ]
    )


class TestSecondVote:
    def test_sibling_credit_flips_when_the_cv_never_names_the_skill(self):
        # Java credited from the skills line, which names Android, not
        # Java. No model call is needed: the CV never names Java.
        llm = [AUDIT_EXTRACTION, _verdicts(CV_LINES[1], CV_LINES[0])]
        result = _matcher(
            llm, AUDIT_RETRIEVER, _audit_people(CV_LINES), second_vote=True
        ).match("b")
        candidate = result.candidates[0]
        assert [f.covered for f in candidate.coverage] == [True, False]
        assert candidate.tier == "partial"
        assert candidate.gap_statement() == "Not evidenced in the CV: Java"

    def test_quote_naming_the_skill_stands_untouched(self):
        llm = [AUDIT_EXTRACTION, _verdicts(CV_LINES[1], CV_LINES[0])]
        result = _matcher(
            llm, AUDIT_RETRIEVER, _audit_people(CV_LINES), second_vote=True
        ).match("b")
        cpp = result.candidates[0].coverage[0]
        assert cpp.covered and cpp.evidence == CV_LINES[1]

    def test_weak_or_hallucinated_quote_is_replaced_by_the_naming_line(self):
        # C++ credited from a line that does not name it, then from a
        # quote that is not in the CV at all: both become the CV line.
        for quote in ("Wrote unit tests with a C-family framework", "Ten years of C++"):
            llm = [AUDIT_EXTRACTION, _verdicts(quote, CV_LINES[0])]
            result = _matcher(
                llm, AUDIT_RETRIEVER, _audit_people(CV_LINES), second_vote=True
            ).match("b")
            cpp = result.candidates[0].coverage[0]
            assert cpp.covered and cpp.evidence == CV_LINES[0]

    def test_glued_docx_token_still_names_the_skill(self):
        chunks = ["Skills: C++ Android QNX Python Automated testingHypervisor"]
        extraction = json.dumps(
            {"must": [{"kind": "skill", "alternatives": ["Hypervisor"]}], "nice": []}
        )
        verdict = json.dumps(
            [
                {
                    "requirement": "Hypervisor",
                    "covered": True,
                    "evidence": "Automated testingHypervisor",
                }
            ]
        )
        result = _matcher(
            [extraction, verdict],
            {"Hypervisor": [_doc("cv01", "Astrid Okafor")]},
            _audit_people(chunks),
            second_vote=True,
        ).match("b")
        assert result.candidates[0].coverage[0].covered

    def test_switched_off_makes_no_changes(self):
        llm = [AUDIT_EXTRACTION, _verdicts(CV_LINES[1], CV_LINES[0])]
        result = _matcher(llm, AUDIT_RETRIEVER, _audit_people(CV_LINES)).match("b")
        assert all(f.covered for f in result.candidates[0].coverage)


CAPABILITY_EXTRACTION = json.dumps(
    {
        "must": [
            {"kind": "skill", "alternatives": ["C++"]},
            {"kind": "skill", "alternatives": ["hardware interfacing"]},
        ],
        "nice": [],
    }
)
CAP_RETRIEVER = {
    "C++": [_doc("cv01", "Astrid Okafor")],
    "hardware interfacing": [_doc("cv01", "Astrid Okafor")],
}
CAP_LINES = [
    "Migrated perf-critical logic to C++.",
    "Implemented SOME/IP interfaces between cluster OS domains.",
    "Brought up sensor boards and wrote register-level drivers on the bench rig.",
]


def _cap_verdicts(cap_quote):
    return json.dumps(
        [
            {"requirement": "C++", "covered": True, "evidence": CAP_LINES[0]},
            {
                "requirement": "hardware interfacing",
                "covered": True,
                "evidence": cap_quote,
            },
        ]
    )


class TestGenericSkillJudge:
    def test_judge_rejects_related_work_and_repair_finds_the_real_line(self):
        llm = [
            CAPABILITY_EXTRACTION,
            _cap_verdicts(CAP_LINES[1]),  # SOME/IP credited as hardware interfacing
            json.dumps([{"requirement": "hardware interfacing", "supported": False}]),
            json.dumps(
                [{"requirement": "hardware interfacing", "evidence": CAP_LINES[2]}]
            ),
            json.dumps([{"requirement": "hardware interfacing", "supported": True}]),
        ]
        result = _matcher(
            llm, CAP_RETRIEVER, _audit_people(CAP_LINES), second_vote=True
        ).match("b")
        cap = result.candidates[0].coverage[1]
        assert cap.covered and cap.evidence == CAP_LINES[2]

    def test_repaired_line_the_judge_rejects_flips(self):
        llm = [
            CAPABILITY_EXTRACTION,
            _cap_verdicts(CAP_LINES[1]),
            json.dumps([{"requirement": "hardware interfacing", "supported": False}]),
            json.dumps(
                [{"requirement": "hardware interfacing", "evidence": CAP_LINES[1]}]
            ),
            json.dumps([{"requirement": "hardware interfacing", "supported": False}]),
        ]
        result = _matcher(
            llm, CAP_RETRIEVER, _audit_people(CAP_LINES), second_vote=True
        ).match("b")
        assert not result.candidates[0].coverage[1].covered

    def test_empty_or_invented_repair_flips(self):
        for repair in ("", "Designed FPGA interfaces at a lab"):
            llm = [
                CAPABILITY_EXTRACTION,
                _cap_verdicts(CAP_LINES[1]),
                json.dumps(
                    [{"requirement": "hardware interfacing", "supported": False}]
                ),
                json.dumps(
                    [{"requirement": "hardware interfacing", "evidence": repair}]
                ),
            ]
            result = _matcher(
                llm, CAP_RETRIEVER, _audit_people(CAP_LINES), second_vote=True
            ).match("b")
            candidate = result.candidates[0]
            assert not candidate.coverage[1].covered
            assert candidate.tier == "partial"

    def test_supported_quote_needs_no_repair(self):
        llm = [
            CAPABILITY_EXTRACTION,
            _cap_verdicts(CAP_LINES[2]),
            json.dumps([{"requirement": "hardware interfacing", "supported": True}]),
        ]
        result = _matcher(
            llm, CAP_RETRIEVER, _audit_people(CAP_LINES), second_vote=True
        ).match("b")
        assert result.candidates[0].tier == "strong"

    def test_unparseable_audit_keeps_the_first_pass(self):
        llm = [CAPABILITY_EXTRACTION, _cap_verdicts(CAP_LINES[1]), "no", "still no"]
        result = _matcher(
            llm, CAP_RETRIEVER, _audit_people(CAP_LINES), second_vote=True
        ).match("b")
        assert result.candidates[0].tier == "strong"


class TestProductNameHeuristic:
    def test_product_names(self):
        for name in (
            "AUTOSAR Classic",
            "ISO 26262",
            "C++",
            "gRPC",
            "PyTest",
            "Vector CANoe",
            "Docker",
            "Embedded C",
            "MISRA C",
            "5G RAN",
            "SOME/IP",
            "GitLab CI",
        ):
            assert is_product_name([name]), name

    def test_phrases_of_ordinary_words(self):
        for name in (
            "hardware interfacing",
            "device drivers",
            "secure boot",
            "Python-based test automation",
            "Swedish driving license B",
            "Android Automotive Software Development",
            "AI-assisted software development tools",
            "working with test rigs",
        ):
            assert not is_product_name([name]), name

    def test_product_name_the_cv_never_names_flips_without_a_call(self):
        extraction = json.dumps(
            {"must": [{"kind": "skill", "alternatives": ["Yocto"]}], "nice": []}
        )
        verdict = json.dumps(
            [{"requirement": "Yocto", "covered": True, "evidence": CAP_LINES[0]}]
        )
        result = _matcher(
            [extraction, verdict],
            {"Yocto": [_doc("cv01", "Astrid Okafor")]},
            _audit_people(CAP_LINES),
            second_vote=True,
        ).match("b")
        assert not result.candidates[0].coverage[0].covered

    def test_capability_items_from_an_old_extractor_are_skills(self):
        req = parse_requirements(
            _line_extraction(
                [
                    {
                        "text": "Hands-on experience with hardware interfacing.",
                        "section": "must",
                        "relation": "all",
                        "items": [
                            {"kind": "capability", "name": "hardware interfacing"}
                        ],
                    }
                ]
            )
        )
        assert [(r.kind, r.label) for r in req.must] == [
            ("skill", "hardware interfacing")
        ]


class TestNiceToHaveNaming:
    CV = "Skills: Hypervisor, QNX. Developed Android Automotive apps for a head unit."

    def test_singular_and_head_phrase_count(self):
        assert nice_named_in(self.CV, "Hypervisors")
        assert nice_named_in(self.CV, "QNX")
        assert nice_named_in(self.CV, "Android Automotive Software Development")

    def test_unrelated_phrase_does_not(self):
        assert not nice_named_in(self.CV, "Yocto")
        assert not nice_named_in(self.CV, "Kubernetes Operators")


class TestTitleCasePhrasesGoToTheJudge:
    def test_python_based_test_automation_is_kept_on_a_pytest_line(self):
        extraction = json.dumps(
            {
                "must": [
                    {"kind": "skill", "alternatives": ["Python-based test automation"]}
                ],
                "nice": [],
            }
        )
        line = "Wrote GTest (C++) unit tests and PyTest (Python) component tests."
        verdict = json.dumps(
            [
                {
                    "requirement": "Python-based test automation",
                    "covered": True,
                    "evidence": line,
                }
            ]
        )
        judged = json.dumps(
            [{"requirement": "Python-based test automation", "supported": True}]
        )
        result = _matcher(
            [extraction, verdict, judged],
            {"Python-based test automation": [_doc("cv01", "Astrid Okafor")]},
            _audit_people([line]),
            second_vote=True,
        ).match("b")
        finding = result.candidates[0].coverage[0]
        assert finding.covered and finding.evidence == line


class TestNiceHitsAreLexical:
    def test_surfacing_for_a_nice_query_is_not_a_tick(self):
        # One-person pool: retrieval returns cv01 for every query. Only
        # the nice-to-have the CV actually names is ticked.
        extraction = json.dumps(
            {
                "must": [{"kind": "skill", "alternatives": ["Docker"]}],
                "nice": ["Helm", "Kubernetes"],
            }
        )
        verdict = json.dumps(
            [{"requirement": "Docker", "covered": True, "evidence": "Docker"}]
        )
        people = {
            "cv01": {"name": "Astrid Okafor", "chunks": ["Docker and Helm charts."]}
        }
        docs = [_doc("cv01", "Astrid Okafor")]
        result = _matcher(
            [extraction, verdict],
            {"Docker": docs, "Helm": docs, "Kubernetes": docs},
            people,
        ).match("b")
        assert result.candidates[0].nice_hits == ["Helm"]
