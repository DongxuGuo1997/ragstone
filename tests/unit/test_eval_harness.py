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
    """Session handling and metric split for scripted conversations."""

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

        # Both turns run in order and share ONE session; session ids are
        # namespaced per run so persistent checkpoint backends can't feed
        # stale history into a later invocation.
        assert [q for q, _ in pipeline.asks] == [
            "Who commanded it?",
            "What is her background?",
        ]
        sessions = {s for _, s in pipeline.asks}
        assert len(sessions) == 1
        assert sessions.pop().endswith("_mt01")

    def test_single_turn_cases_unchanged(self, run_eval):
        pipeline = _ScriptedPipeline()
        run_eval.eval_generation(
            pipeline, [_case("q01", "factual", "When did it launch?")], ARGS
        )
        [(question, session_id)] = pipeline.asks
        assert question == "When did it launch?"
        assert session_id.endswith("_q01")

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
    """Layer 1 must skip categories it cannot score fairly."""

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


class TestConfidenceIntervals:
    """CI annotation mechanizes the Experiment-9 lesson (small-n deltas)."""

    def test_ci_halfwidth_matches_binomial_formula(self, run_eval):
        # p=0.5, n=100 -> 1.96 * sqrt(0.25/100) = 0.098
        assert abs(run_eval.ci95_halfwidth(0.5, 100) - 0.098) < 1e-3
        # Degenerate rates have zero width; empty samples have none.
        assert run_eval.ci95_halfwidth(1.0, 50) == 0.0
        assert run_eval.ci95_halfwidth(0.9, 0) is None

    def test_sample_sizes_split_by_category(self, run_eval):
        retrieval_rows = [{"case": {"id": "q1"}}] * 7
        generation_rows = [
            {"case": {"category": "factual"}},
            {"case": {"category": "distractor"}},
            {"case": {"category": "multi_turn"}},
        ]
        sizes = run_eval.metric_sample_sizes(retrieval_rows, generation_rows)
        assert sizes["hit_rate"] == 7
        assert sizes["correct_rate"] == 2
        assert sizes["multi_turn_correct_rate"] == 1

    def test_format_metric_annotates_rates_but_not_mrr(self, run_eval):
        sizes = {"hit_rate": 100, "mrr": 100}
        assert run_eval.format_metric("hit_rate", 0.5, sizes) == "0.5 ±0.098 (n=100)"
        assert run_eval.format_metric("mrr", 0.82, sizes) == "0.82 (n=100)"
        # Metrics without a known n render unannotated.
        assert run_eval.format_metric("correct_rate", 0.9, sizes) == "0.9"

    def test_gate_says_whether_a_drop_exceeds_the_ci(
        self, run_eval, tmp_path, monkeypatch, capsys
    ):
        import json as _json
        from types import SimpleNamespace as _NS

        baseline_path = tmp_path / "baseline.json"
        args = _NS(
            provider="openai",
            model="m",
            judge_provider="openai",
            judge_model="m",
            k=4,
            chain_type="simple",
            mode="full",
            rerank=False,
            set="smoke",
            update_baseline=False,
        )
        key = run_eval.baseline_key(args)
        baseline_path.write_text(
            _json.dumps(
                {key: {"metrics": {"correct_rate": 0.95}, "date": "2026-01-01"}}
            )
        )
        monkeypatch.setattr(run_eval, "BASELINE_PATH", baseline_path)

        # Drop of 0.15 at n=20 (CI ±0.196 at p=0.8) -> within the CI.
        rc = run_eval.check_baseline(args, {"correct_rate": 0.8}, {"correct_rate": 20})
        assert rc == 1
        assert "within the 95% CI" in capsys.readouterr().out

        # Same drop at n=400 (CI ±0.039) -> outside the CI.
        rc = run_eval.check_baseline(args, {"correct_rate": 0.8}, {"correct_rate": 400})
        assert rc == 1
        assert "outside the 95% CI" in capsys.readouterr().out
