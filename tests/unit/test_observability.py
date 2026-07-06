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
from ragstone.utils.observability import (
    estimate_cost_usd,
    record_stage,
    time_stage,
    track_request,
)

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


class TestStageTiming:
    def test_stages_accumulate_and_appear_in_log_line(self, caplog):
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            with track_request("s1") as metrics:
                record_stage("retrieval", 120)
                record_stage("retrieval", 80)  # agent mode: repeated searches
                record_stage("rephrase", 500)

        assert metrics.stage_ms == {"retrieval": 200, "rephrase": 500}
        line = _request_records(caplog)[0].getMessage()
        assert "stages=rephrase:500,retrieval:200" in line

    def test_record_stage_outside_request_is_noop(self):
        record_stage("retrieval", 100)  # must not raise or leak anywhere

    def test_time_stage_context_manager_measures(self):
        with track_request("s1") as metrics:
            with time_stage("retrieval"):
                pass
        assert "retrieval" in metrics.stage_ms
        assert metrics.stage_ms["retrieval"] >= 0

    def test_generation_ms_is_the_unattributed_remainder(self):
        with track_request("s1") as metrics:
            record_stage("rephrase", 10)
        # latency is tiny here; derived generation must never go negative.
        assert metrics.generation_ms == max(0, metrics.latency_ms - 10)

    def test_stages_recorded_inside_langgraph_nodes_reach_the_request(self):
        # The critical propagation property: graph nodes run under copied
        # contexts (ThreadPool + contextvars.copy_context), and stage
        # recordings made there must land on the outer request's metrics.
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
        from ragstone.rag.rag import RagProxy
        from ragstone.utils.full_chain import FullChain

        llm = FakeListChatModel(responses=["Paris.", "REPHRASED", "About 2M."])

        class _Proxy:
            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
        full_chain = FullChain(
            _Proxy(), RagProxy(model=llm, retriever=retriever), MemoryProxy()
        )
        full_chain.create_full_chain("simple")

        with track_request("s1") as first_turn:
            full_chain.ask_question("capital?", session_id="st1")
        assert "rephrase" not in first_turn.stage_ms  # no history yet

        with track_request("s1") as second_turn:
            full_chain.ask_question("population?", session_id="st1")
        assert "rephrase" in second_turn.stage_ms  # timed inside the node


class TestRetrievalStageCapture:
    def test_source_recording_retriever_times_retrieval(self):
        from ragstone.rag.memory import SimpleTextRetriever
        from ragstone.rag.pipeline import _SourceRecordingRetriever

        wrapped = SimpleTextRetriever.from_texts(["Paris is the capital."])
        retriever = _SourceRecordingRetriever(wrapped=wrapped, record=[])

        with track_request("s1") as metrics:
            docs = retriever.invoke("capital?")

        assert len(docs) == 1
        assert "retrieval" in metrics.stage_ms


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
        # Streaming records time-to-first-token separately from the
        # end-to-end latency (they can differ wildly; UIs show both).
        assert metrics.first_token_ms is not None
        assert metrics.first_token_ms <= max(metrics.latency_ms, 1)

    def test_non_streaming_ask_leaves_first_token_unset(self, caplog):
        pipeline = self._pipeline_with_stub_chain()
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            pipeline.ask_question("capital?", session_id="m3")
        assert pipeline.last_metrics.first_token_ms is None

    def test_progress_events_pass_through_but_are_never_cached(
        self, caplog, monkeypatch
    ):
        import ragstone.rag.pipeline as pl

        cached = {}

        class _RecordingCache:
            def get_response(self, question, session_id):
                return None

            def cache_response(self, question, response, session_id):
                cached[question] = response

        monkeypatch.setattr(pl, "_is_response_cache_enabled", lambda: True)
        monkeypatch.setattr(pl, "_get_query_cache", lambda: _RecordingCache())

        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        pipeline._chain_type = "agent"

        class _EventingChain:
            def stream_question(self, query, session_id):
                yield {"event": "search", "query": "refined"}
                yield "the "
                yield "answer"

        pipeline._chain = _EventingChain()
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER):
            received = list(pipeline.ask_question_stream("q?", session_id="ev1"))

        assert {"event": "search", "query": "refined"} in received
        assert cached == {"q?": "the answer"}  # text only, no event noise

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
