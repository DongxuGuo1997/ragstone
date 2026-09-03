"""
Extraction-level scoring for the staffing eval (a10/a11, the RFQ briefs).

The tier metrics can stay perfect while extraction is wrong, so the
harness scores the extracted must-have set directly. These tests lock
the canonical form: OR-groups compare as sets, years as integers, a
domain label that merely contains the oracle's word counts as a hit,
and a degree requirement is presence-only.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from ragstone.match import AssignmentRequirements, Requirement

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(EVALS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_staffing_eval", EVALS_DIR / "run_staffing_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    sys.path.remove(str(EVALS_DIR))


RECORD = {
    "must_have": [["Jenkins", "GitLab CI"], ["Docker"], ["Python"]],
    "min_years": 3,
    "language": "Swedish",
    "domain": "automotive",
    "degree": "Computer Science",
    "nice_to_have": ["Kubernetes"],
    "location": "Gothenburg (hybrid)",
}


class TestOracleItems:
    def test_every_dimension_is_one_item(self, harness):
        items = harness.oracle_items(RECORD)
        assert ("skill", frozenset({"jenkins", "gitlab ci"})) in items
        assert ("years", 3) in items
        assert ("language", "swedish") in items
        assert ("domain", "automotive") in items
        assert ("education",) in items
        assert len(items) == 7

    def test_absent_dimensions_are_absent(self, harness):
        record = {**RECORD, "language": None, "domain": None, "degree": None}
        items = harness.oracle_items(record)
        assert not any(i[0] in ("language", "domain", "education") for i in items)


class TestExtractedItems:
    def test_or_group_order_and_case_do_not_matter(self, harness):
        req = AssignmentRequirements(
            must=[Requirement(kind="skill", alternatives=["gitlab ci", "JENKINS"])]
        )
        assert harness.extracted_items(req) == {
            ("skill", frozenset({"jenkins", "gitlab ci"}))
        }

    def test_conjunction_collapsed_into_or_is_a_miss(self, harness):
        req = AssignmentRequirements(
            must=[Requirement(kind="skill", alternatives=["Docker", "Python"])]
        )
        oracle = harness.oracle_items(RECORD)
        assert not (harness.extracted_items(req) & oracle)

    def test_domain_label_containing_oracle_word_counts(self, harness):
        req = AssignmentRequirements(
            must=[Requirement(kind="domain", detail="automotive industry")]
        )
        assert harness.extracted_items(req, "automotive") == {("domain", "automotive")}

    def test_years_parse_to_int(self, harness):
        req = AssignmentRequirements(must=[Requirement(kind="years", detail="3")])
        assert harness.extracted_items(req) == {("years", 3)}


class TestLocationCaptured:
    def test_city_word_in_extracted_location(self, harness):
        req = AssignmentRequirements(must=[])
        req.location = "Gothenburg, hybrid"  # type: ignore[attr-defined]
        assert harness.location_captured(req, RECORD)

    def test_missing_location_attribute_is_not_captured(self, harness):
        req = AssignmentRequirements(must=[])
        assert not harness.location_captured(req, RECORD)
