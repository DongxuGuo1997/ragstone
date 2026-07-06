"""
Unit tests for the eval harness's multi-turn logic (no network, no judge).

The harness itself is quality infrastructure — a bug here silently
invalidates every measurement, so its session handling and metric split
are tested with a scripted pipeline and a stub judge.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


@pytest.fixture()
def run_eval(monkeypatch):
    """Import evals/run_eval.py with a stub `judge` module installed."""
    judge_stub = SimpleNamespace(
        judge_correctness=lambda *a, **k: {"verdict": "pass", "reason": ""},
        judge_faithfulness=lambda *a, **k: {"verdict": "pass", "reason": ""},
    )
    monkeypatch.setitem(sys.modules, "judge", judge_stub)
    spec = importlib.util.spec_from_file_location("run_eval", EVALS_DIR / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ScriptedPipeline:
    """Records every ask (question, session_id) in order."""

    def __init__(self):
        self.asks = []
        self._docs = [Document(page_content="Halvorsen is a planetary geologist.")]

    def ask_question(self, question, session_id=None, use_cache=True):
        self.asks.append((question, session_id))
        return "an answer"

    def get_retriever(self):
        pipeline = self

        class _R:
            def invoke(self, q):
                return pipeline._docs

        return _R()

    def get_last_retrieved_documents(self):
        return self._docs


ARGS = SimpleNamespace(k=4, judge_model="stub", judge_provider="openai")


def _case(case_id, category, question, turns=None):
    case = {
        "id": case_id,
        "category": category,
        "question": question,
        "gold_answer": "gold",
        "must_contain": ["gold"],
    }
    if turns:
        case["turns"] = turns
    return case


class TestMultiTurnGeneration:
    def test_turns_run_in_order_in_one_session(self, run_eval):
        pipeline = _ScriptedPipeline()
        cases = [
            _case(
                "mt01",
                "multi_turn",
                "What is her background?",
                turns=["Who commanded it?", "What is her background?"],
            )
        ]

        run_eval.eval_generation(pipeline, cases, ARGS)

        assert pipeline.asks == [
            ("Who commanded it?", "eval_mt01"),
            ("What is her background?", "eval_mt01"),
        ]

    def test_single_turn_cases_unchanged(self, run_eval):
        pipeline = _ScriptedPipeline()
        run_eval.eval_generation(
            pipeline, [_case("q01", "factual", "When did it launch?")], ARGS
        )
        assert pipeline.asks == [("When did it launch?", "eval_q01")]

    def test_metrics_are_split_by_category(self, run_eval):
        pipeline = _ScriptedPipeline()
        cases = [
            _case("q01", "factual", "single?"),
            _case("mt01", "multi_turn", "final?", turns=["first?", "final?"]),
        ]

        metrics, rows, _efficiency = run_eval.eval_generation(pipeline, cases, ARGS)

        # Single-turn rates keep their historical names (baseline
        # comparability); multi-turn quality is a separate, gateable metric.
        assert metrics["correct_rate"] == 1.0
        assert metrics["multi_turn_correct_rate"] == 1.0
        assert metrics["multi_turn_faithful_rate"] == 1.0
        assert len(rows) == 2


class TestRetrievalLayerSkips:
    def test_multi_turn_and_unanswerable_are_skipped(self, run_eval):
        pipeline = _ScriptedPipeline()
        cases = [
            _case("q01", "factual", "Who is Halvorsen?"),
            _case("mt01", "multi_turn", "final?", turns=["a?", "final?"]),
            _case("q02", "unanswerable", "What is her salary?"),
        ]
        cases[0]["must_contain"] = ["halvorsen"]

        _metrics, rows = run_eval.eval_retrieval(pipeline, cases, k=4)

        assert [r["case"]["id"] for r in rows] == ["q01"]
