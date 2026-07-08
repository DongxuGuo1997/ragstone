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


def _full_args(**overrides):
    """An args namespace with every field the dump/key paths read."""
    ns = SimpleNamespace(
        provider="ollama",
        model="local-m",
        judge_provider="openai",
        judge_model="stub",
        k=4,
        chain_type="simple",
        mode="full",
        rerank=False,
        set="smoke",
        ollama_reasoning=None,
        judge_reasoning=None,
    )
    for name, value in overrides.items():
        setattr(ns, name, value)
    return ns


class TestLocalStackAdditions:
    """ROADMAP 8.0 wiring: dumps, output tokens, reasoning key suffixes."""

    def test_rows_carry_context_and_output_tokens(self, run_eval):
        pipeline = _ScriptedPipeline()
        cases = [_case("c1", "single_fact", "q?")]
        _, rows, efficiency = run_eval.eval_generation(pipeline, cases, ARGS)
        assert rows[0]["context"].startswith("Halvorsen")
        # The scripted pipeline reports no usage metadata: the tokens/s
        # figure must degrade to 0, never divide by zero.
        assert rows[0]["output_tokens"] == 0
        assert efficiency["total_output_tokens"] == 0
        assert efficiency["output_tokens_per_s"] == 0.0

    def test_dump_answers_round_trip(self, run_eval, tmp_path):
        import json as _json

        pipeline = _ScriptedPipeline()
        decline = _case("c2", "unanswerable", "unknowable?")
        decline["gold_answer"] = None
        cases = [_case("c1", "single_fact", "q?"), decline]
        _, rows, _ = run_eval.eval_generation(pipeline, cases, ARGS)

        path = tmp_path / "dumps" / "d.jsonl"
        run_eval.dump_answers(str(path), rows, _full_args(ollama_reasoning="off"))

        lines = path.read_text(encoding="utf-8").splitlines()
        meta = _json.loads(lines[0])["_meta"]
        assert meta["model"] == "local-m"
        assert meta["ollama_reasoning"] == "off"
        assert meta["n_cases"] == 2
        records = [_json.loads(line) for line in lines[1:]]
        assert records[0]["answer"] == "an answer"
        assert records[0]["context"].startswith("Halvorsen")
        # None gold answers select decline grading downstream — the dump
        # must preserve them as null, not stringify or drop them.
        assert records[1]["gold_answer"] is None

    def test_baseline_key_reasoning_suffixes_only_when_set(self, run_eval):
        plain = run_eval.baseline_key(_full_args())
        assert "reasoning" not in plain  # committed keys stay untouched
        suffixed = run_eval.baseline_key(
            _full_args(ollama_reasoning="off", judge_reasoning="on")
        )
        assert suffixed.endswith("|ollama-reasoning=off|judge-reasoning=on")
        assert suffixed.startswith(plain)

    def test_judge_reasoning_reaches_the_judge_calls(self, run_eval):
        calls = []
        judge_stub = sys.modules["judge"]
        judge_stub.judge_correctness = lambda *a: (
            calls.append(a),
            {"verdict": "pass", "reason": ""},
        )[1]
        judge_stub.judge_faithfulness = lambda *a: (
            calls.append(a),
            {"verdict": "pass", "reason": ""},
        )[1]

        args = _full_args(judge_reasoning="on")
        run_eval.eval_generation(
            _ScriptedPipeline(), [_case("c1", "single_fact", "q?")], args
        )
        # Both judge calls got the normalized True as their final argument.
        assert len(calls) == 2
        assert all(call[-1] is True for call in calls)
