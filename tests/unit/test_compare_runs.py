"""
Unit tests for the paired McNemar comparison instrument.

The exact p-values are hand-computed from the binomial definition, so a
regression here means the statistics changed, not just the formatting.
"""

import importlib.util
import json
from pathlib import Path

import pytest

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


@pytest.fixture()
def compare_runs():
    spec = importlib.util.spec_from_file_location(
        "compare_runs", EVALS_DIR / "compare_runs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dump(path, records, model="m"):
    lines = [{"_meta": {"model": model}}] + records
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def _rec(case_id, correct, faithful=True, category="factual"):
    return {
        "id": case_id,
        "category": category,
        "correct": correct,
        "faithful": faithful,
    }


class TestExactP:
    def test_hand_computed_values(self, compare_runs):
        # b=1, c=5: p = 2 * [C(6,0)+C(6,1)] / 2^6 = 14/64
        assert abs(compare_runs.mcnemar_exact_p(1, 5) - 0.21875) < 1e-12
        # b=0, c=8: p = 2 * 1/256
        assert abs(compare_runs.mcnemar_exact_p(0, 8) - 0.0078125) < 1e-12
        # symmetric
        assert compare_runs.mcnemar_exact_p(5, 1) == compare_runs.mcnemar_exact_p(1, 5)

    def test_no_discordance_is_p_one(self, compare_runs):
        assert compare_runs.mcnemar_exact_p(0, 0) == 1.0

    def test_balanced_flips_cap_at_one(self, compare_runs):
        # b=c makes the two-sided doubling overshoot; must clamp to 1.0.
        assert compare_runs.mcnemar_exact_p(3, 3) == 1.0


class TestPairing:
    def test_counts_and_flip_ids(self, compare_runs, tmp_path):
        a = compare_runs.load_dump(
            _dump(
                tmp_path / "a.jsonl",
                [
                    _rec("q1", True),
                    _rec("q2", True),
                    _rec("q3", False),
                    _rec("q4", False),
                ],
            )
        )[1]
        b = compare_runs.load_dump(
            _dump(
                tmp_path / "b.jsonl",
                [
                    _rec("q1", True),  # both pass
                    _rec("q2", False),  # A-only pass
                    _rec("q3", True),  # B-only pass
                    _rec("q4", False),  # both fail
                ],
            )
        )[1]
        result = compare_runs.compare_metric(a, b, "correct")
        assert result["paired"] == 4
        assert result["both_pass"] == 1
        assert result["both_fail"] == 1
        assert result["a_only"] == ["q2"]
        assert result["b_only"] == ["q3"]
        assert result["rate_a"] == 0.5
        assert result["rate_b"] == 0.5
        assert result["p"] == 1.0

    def test_unpaired_and_unjudged_records_are_skipped(self, compare_runs, tmp_path):
        a = compare_runs.load_dump(
            _dump(
                tmp_path / "a.jsonl",
                [_rec("q1", True), _rec("only_a", True), {"id": "q2"}],
            )
        )[1]
        b = compare_runs.load_dump(
            _dump(tmp_path / "b.jsonl", [_rec("q1", True), _rec("q2", True)])
        )[1]
        result = compare_runs.compare_metric(a, b, "correct")
        # q2 has no boolean verdict in A; only_a is unpaired.
        assert result["paired"] == 1

    def test_category_filter(self, compare_runs, tmp_path):
        a = compare_runs.load_dump(
            _dump(
                tmp_path / "a.jsonl",
                [_rec("q1", True), _rec("mt1", False, category="multi_turn")],
            )
        )[1]
        b = compare_runs.load_dump(
            _dump(
                tmp_path / "b.jsonl",
                [_rec("q1", True), _rec("mt1", True, category="multi_turn")],
            )
        )[1]
        result = compare_runs.compare_metric(a, b, "correct", category="multi_turn")
        assert result["paired"] == 1
        assert result["b_only"] == ["mt1"]

    def test_no_comparable_verdicts_returns_none(self, compare_runs, tmp_path):
        a = compare_runs.load_dump(
            _dump(tmp_path / "a.jsonl", [{"id": "q1", "hit": 1}])
        )[1]
        b = compare_runs.load_dump(
            _dump(tmp_path / "b.jsonl", [{"id": "q1", "hit": 1}])
        )[1]
        assert compare_runs.compare_metric(a, b, "correct") is None

    def test_judge_dict_verdicts_are_normalized(self, compare_runs, tmp_path):
        # Real run_eval dumps store {"verdict": "pass"/"fail", "reason": ...}.
        def rec(case_id, verdict):
            return {
                "id": case_id,
                "category": "factual",
                "correct": {"verdict": verdict, "reason": "r"},
            }

        a = compare_runs.load_dump(
            _dump(tmp_path / "a.jsonl", [rec("q1", "pass"), rec("q2", "pass")])
        )[1]
        b = compare_runs.load_dump(
            _dump(tmp_path / "b.jsonl", [rec("q1", "pass"), rec("q2", "fail")])
        )[1]
        result = compare_runs.compare_metric(a, b, "correct")
        assert result["paired"] == 2
        assert result["a_only"] == ["q2"]
        assert compare_runs.verdict_bool(None) is None
        assert compare_runs.verdict_bool({"verdict": "weird"}) is None
