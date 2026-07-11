"""
Unit tests for evals/rejudge.py (the judge-delta instrument) and the
judge's reasoning threading — no network, stub judges throughout.

rejudge.py exists to score IDENTICAL stored answers under two judges; if
it silently re-retrieved, re-generated, or mis-split categories, the
published cloud-vs-local judge delta would measure the wrong thing.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


class _RecordingJudge:
    """Scriptable judge stub: returns queued verdicts, records calls."""

    def __init__(self):
        self.correctness_calls = []
        self.faithfulness_calls = []
        self.verdict = {"verdict": "pass", "reason": "ok"}

    def judge_correctness(self, *args):
        self.correctness_calls.append(args)
        return dict(self.verdict)

    def judge_faithfulness(self, *args):
        self.faithfulness_calls.append(args)
        return dict(self.verdict)


@pytest.fixture()
def harness(monkeypatch):
    """(rejudge module, recording judge stub) with the stub installed."""
    stub = _RecordingJudge()
    monkeypatch.setitem(sys.modules, "judge", stub)
    # rejudge does `from run_eval import ...`; force a fresh import so the
    # stubbed judge module is what run_eval's late imports resolve to.
    monkeypatch.delitem(sys.modules, "run_eval", raising=False)
    spec = importlib.util.spec_from_file_location("rejudge", EVALS_DIR / "rejudge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, stub


def _record(case_id, category="single_fact", answer="an answer", **overrides):
    record = {
        "id": case_id,
        "category": category,
        "question": "q?",
        "turns": None,
        "gold_answer": "gold",
        "answer": answer,
        "context": "the stored context",
        "latency_s": 1.0,
        "tokens": 10,
        "output_tokens": 5,
        "stage_ms": {},
        "correct": {"verdict": "pass", "reason": "stored"},
        "faithful": {"verdict": "pass", "reason": "stored"},
    }
    record.update(overrides)
    return record


class TestRejudge:
    def test_metrics_split_single_vs_multi_turn(self, harness):
        rejudge, _ = harness
        results = rejudge.rejudge_records(
            [_record("s1"), _record("s2"), _record("m1", category="multi_turn")],
            "j-model",
            "openai",
            None,
        )
        metrics, sizes = rejudge.compute_metrics(results)
        assert metrics["correct_rate"] == 1.0
        assert metrics["multi_turn_correct_rate"] == 1.0
        assert sizes["correct_rate"] == 2
        assert sizes["multi_turn_correct_rate"] == 1

    def test_empty_answer_fails_closed_with_zero_judge_calls(self, harness):
        rejudge, stub = harness
        results = rejudge.rejudge_records(
            [_record("s1", answer="")], "j-model", "openai", None
        )
        assert results[0]["correct"]["verdict"] == "fail"
        assert results[0]["faithful"]["verdict"] == "fail"
        assert stub.correctness_calls == []
        assert stub.faithfulness_calls == []

    def test_stored_context_reaches_the_judge_verbatim(self, harness):
        # The instrument's entire point: never a fresh retrieval.
        rejudge, stub = harness
        rejudge.rejudge_records([_record("s1")], "j-model", "ollama", True)
        (call,) = stub.faithfulness_calls
        question, context, answer, model, provider, reasoning = call
        assert context == "the stored context"
        assert answer == "an answer"
        assert (model, provider, reasoning) == ("j-model", "ollama", True)

    def test_flips_compare_against_stored_verdicts(self, harness):
        rejudge, stub = harness
        stub.verdict = {"verdict": "fail", "reason": "this judge disagrees"}
        results = rejudge.rejudge_records([_record("s1")], "j", "openai", None)
        flips = rejudge.find_flips(results)
        assert {(f["id"], f["metric"], f["stored"], f["new"]) for f in flips} == {
            ("s1", "correct", "pass", "fail"),
            ("s1", "faithful", "pass", "fail"),
        }

    def test_parse_failures_counted_separately(self, harness):
        rejudge, stub = harness
        stub.verdict = {
            "verdict": "fail",
            "reason": "unparseable judge output: <think>...",
        }
        results = rejudge.rejudge_records([_record("s1")], "j", "ollama", None)
        assert rejudge.count_parse_failures(results) == 2
        # A genuine disagreement is NOT a parse failure.
        stub.verdict = {"verdict": "fail", "reason": "contradicts the gold answer"}
        results = rejudge.rejudge_records([_record("s2")], "j", "ollama", None)
        assert rejudge.count_parse_failures(results) == 0

    def test_load_dump_separates_meta_from_records(self, harness, tmp_path):
        rejudge, _ = harness
        path = tmp_path / "d.jsonl"
        lines = [
            json.dumps({"_meta": {"model": "m", "n_cases": 1}}),
            json.dumps(_record("s1")),
            "",  # trailing blank line must be tolerated
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        meta, records = rejudge.load_dump(str(path))
        assert meta["model"] == "m"
        assert [r["id"] for r in records] == ["s1"]


class TestJudgeReasoningThreading:
    """The real evals/judge.py must pass reasoning only when set, and only
    to the Ollama branch."""

    @pytest.fixture()
    def real_judge(self, monkeypatch):
        recorded = {}

        class _FakeChatOllama:
            def __init__(self, **kwargs):
                recorded.update(kwargs)

            def invoke(self, prompt):
                return SimpleNamespace(content='{"verdict": "pass", "reason": "r"}')

        monkeypatch.setitem(
            sys.modules,
            "langchain_ollama",
            SimpleNamespace(ChatOllama=_FakeChatOllama),
        )
        spec = importlib.util.spec_from_file_location(
            "judge_real", EVALS_DIR / "judge.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, recorded

    def test_reasoning_true_is_passed(self, real_judge):
        judge_real, recorded = real_judge
        judge_real._get_judge_llm("local-j", "ollama", reasoning=True)
        assert recorded["reasoning"] is True
        assert recorded["temperature"] == 0

    def test_unset_reasoning_is_omitted(self, real_judge):
        judge_real, recorded = real_judge
        judge_real._get_judge_llm("local-j", "ollama", reasoning=None)
        assert "reasoning" not in recorded

    def test_verdicts_flow_end_to_end(self, real_judge):
        judge_real, _ = real_judge
        verdict = judge_real.judge_correctness(
            "q?", "gold", "answer", "local-j", "ollama", True
        )
        assert verdict == {"verdict": "pass", "reason": "r"}


class TestVerdictParsing:
    """_parse_verdict must not fail an ANSWER for the JUDGE's formatting.

    Experiment 23 lost two visible-"pass" verdicts to fail-closed parsing
    when the judge quoted the answer inside its reason without escaping.
    """

    @pytest.fixture()
    def judge_real(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "langchain_ollama", SimpleNamespace(ChatOllama=object)
        )
        spec = importlib.util.spec_from_file_location(
            "judge_parse", EVALS_DIR / "judge.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_clean_json_parses(self, judge_real):
        verdict = judge_real._parse_verdict('{"verdict": "pass", "reason": "ok"}')
        assert verdict == {"verdict": "pass", "reason": "ok"}

    def test_code_fenced_json_parses(self, judge_real):
        text = '```json\n{"verdict": "fail", "reason": "wrong tier"}\n```'
        assert judge_real._parse_verdict(text)["verdict"] == "fail"

    def test_unescaped_quotes_in_reason_are_rescued(self, judge_real):
        # The Experiment 23 failure shape: valid verdict, broken JSON.
        text = (
            '{"verdict": "pass", "reason": "the answer says "10^25" '
            'which matches the "systemic risk" threshold"}'
        )
        verdict = judge_real._parse_verdict(text)
        assert verdict["verdict"] == "pass"
        assert "10^25" in verdict["reason"]

    def test_missing_verdict_still_fails_closed(self, judge_real):
        with pytest.raises(ValueError):
            judge_real._parse_verdict('{"reason": "no verdict here"}')
        with pytest.raises(ValueError):
            judge_real._parse_verdict("no json at all")

    def test_invalid_verdict_value_fails_closed(self, judge_real):
        with pytest.raises(ValueError):
            judge_real._parse_verdict('{"verdict": "maybe", "reason": "r"}')
