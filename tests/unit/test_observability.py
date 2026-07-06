"""
Unit tests for per-request observability (no network required).

Each ask must emit exactly one structured line on ``ragstone.requests``
with a correlation id, and the line must appear on success, cache hit,
and failure alike — observability that disappears on errors is useless.
"""

import logging

import pytest

from ragstone.rag.pipeline import OpenAIPipeline
from ragstone.utils.exceptions import ChainExecutionError
from ragstone.utils.observability import estimate_cost_usd, track_request

REQUEST_LOGGER = "ragstone.requests"


def _request_records(caplog):
    return [r for r in caplog.records if r.name == REQUEST_LOGGER]


class TestTrackRequest:
    def test_emits_one_line_with_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            with track_request("s1", chain_type="simple") as metrics:
                pass

        records = _request_records(caplog)
        assert len(records) == 1
        line = records[0].getMessage()
        assert f"request={metrics.request_id}" in line
        assert "session=s1" in line
        assert "chain=simple" in line
        assert "cache_hit=False" in line
        assert "latency_ms=" in line
        assert "tokens=0" in line  # no LLM usage reported inside the block
        assert "error" not in line

    def test_line_emitted_on_error_and_exception_propagates(self, caplog):
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            with pytest.raises(ValueError):
                with track_request("s1"):
                    raise ValueError("boom")

        records = _request_records(caplog)
        assert len(records) == 1
        assert "error=ValueError" in records[0].getMessage()

    def test_request_ids_are_unique_per_call(self, caplog):
        ids = []
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            for _ in range(2):
                with track_request("s1") as metrics:
                    ids.append(metrics.request_id)
        assert ids[0] != ids[1]

    def test_cache_hit_annotation_is_logged(self, caplog):
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            with track_request("s1") as metrics:
                metrics.cache_hit = True
        assert "cache_hit=True" in _request_records(caplog)[0].getMessage()


class TestCostEstimate:
    def test_known_model_prices_input_and_output_separately(self):
        # gpt-4o-mini: $0.15/M input, $0.60/M output.
        cost = estimate_cost_usd(1_000_000, 1_000_000, "gpt-4o-mini")
        assert cost == pytest.approx(0.75)

    def test_unknown_or_local_model_returns_none(self):
        assert estimate_cost_usd(100, 100, "llama3") is None
        assert estimate_cost_usd(100, 100, None) is None


class TestPipelineRequestLogging:
    def _pipeline_with_stub_chain(self, answer="Paris."):
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        pipeline._chain_type = "simple"

        class _StubChain:
            def ask_question(self, query, session_id):
                return answer

            def stream_question(self, query, session_id):
                yield from answer.split()

        pipeline._chain = _StubChain()
        return pipeline

    def test_ask_question_emits_request_line(self, caplog):
        pipeline = self._pipeline_with_stub_chain()
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            assert pipeline.ask_question("capital?", session_id="obs1") == "Paris."

        records = _request_records(caplog)
        assert len(records) == 1
        line = records[0].getMessage()
        assert "session=obs1" in line
        assert "chain=simple" in line
        assert "cache_hit=False" in line

    def test_stream_emits_request_line_after_consumption(self, caplog):
        pipeline = self._pipeline_with_stub_chain("streamed answer here")
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            chunks = list(pipeline.ask_question_stream("capital?", session_id="obs2"))

        assert chunks == ["streamed", "answer", "here"]
        records = _request_records(caplog)
        assert len(records) == 1
        assert "session=obs2" in records[0].getMessage()

    def test_cache_hit_is_visible_in_request_line(self, caplog, monkeypatch):
        import ragstone.rag.pipeline as pl

        class _StubCache:
            def get_response(self, question, session_id):
                return "cached!"

        monkeypatch.setattr(pl, "_is_response_cache_enabled", lambda: True)
        monkeypatch.setattr(pl, "_get_query_cache", lambda: _StubCache())

        pipeline = self._pipeline_with_stub_chain()
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            assert pipeline.ask_question("capital?", session_id="obs3") == "cached!"

        assert "cache_hit=True" in _request_records(caplog)[0].getMessage()

    def test_last_metrics_exposed_after_ask(self, caplog):
        # The glass-box UI reads pipeline.last_metrics after each answer;
        # the object must be complete by the time ask_question returns.
        pipeline = self._pipeline_with_stub_chain()
        assert pipeline.last_metrics is None  # nothing asked yet

        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            pipeline.ask_question("capital?", session_id="m1")

        metrics = pipeline.last_metrics
        assert metrics is not None
        assert metrics.session_id == "m1"
        assert metrics.chain_type == "simple"
        assert metrics.cache_hit is False
        assert metrics.latency_ms >= 0
        assert metrics.error is None

    def test_last_metrics_complete_after_stream_consumed(self, caplog):
        pipeline = self._pipeline_with_stub_chain("streamed answer here")
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            list(pipeline.ask_question_stream("capital?", session_id="m2"))

        metrics = pipeline.last_metrics
        assert metrics is not None
        assert metrics.session_id == "m2"
        assert metrics.latency_ms >= 0

    def test_failure_is_visible_in_request_line(self, caplog):
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        pipeline._chain_type = "simple"

        class _FailingChain:
            def ask_question(self, query, session_id):
                raise RuntimeError("api down")

        pipeline._chain = _FailingChain()
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            with pytest.raises(ChainExecutionError):
                pipeline.ask_question("capital?", session_id="obs4")

        line = _request_records(caplog)[0].getMessage()
        assert "error=ChainExecutionError" in line
