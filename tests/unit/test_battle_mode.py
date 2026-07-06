"""
Unit tests for the comparison ("battle mode") backend (no network).

Covers the two pieces the Compare tab stands on:
- Pipeline.make_chain_variant: extra chains over the SAME retriever,
  without disturbing the pipeline's main chain.
- The worker that streams a chain into a queue from a background thread,
  which is the only part of the tab that runs off the main thread.
"""

import queue

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.memory import SimpleTextRetriever
from ragstone.rag.pipeline import OpenAIPipeline
from ragstone.ui.streamlit_app import StreamlitApp
from ragstone.utils.exceptions import ChainInitializationError


class _FakeLLMProxy:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm

    def get_model_name(self):
        return "fake-model"


def _configured_pipeline(responses):
    pipeline = OpenAIPipeline(model="gpt-4o-mini")
    pipeline.LLM = _FakeLLMProxy(FakeListChatModel(responses=list(responses)))
    pipeline._retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
    return pipeline


class TestMakeChainVariant:
    def test_variant_answers_without_touching_main_chain(self):
        pipeline = _configured_pipeline(["Main.", "Variant."])
        pipeline.create_rag_chain(chain_type="simple")
        main_chain = pipeline.get_chain()

        variant = pipeline.make_chain_variant("simple")

        assert variant is not main_chain
        assert pipeline.get_chain() is main_chain  # main chain untouched
        assert pipeline._chain_type == "simple"
        assert variant.ask_question("capital?", session_id="v1") in {
            "Main.",
            "Variant.",
        }

    def test_variants_share_the_retriever(self):
        pipeline = _configured_pipeline(["A.", "B."])
        variant_a = pipeline.make_chain_variant("simple")
        variant_b = pipeline.make_chain_variant("simple")

        assert variant_a._rag.get_retriever() is pipeline._retriever
        assert variant_b._rag.get_retriever() is pipeline._retriever

    def test_variant_memory_is_isolated_from_main_chain(self):
        pipeline = _configured_pipeline(["Main.", "Variant."])
        pipeline.create_rag_chain(chain_type="simple")
        variant = pipeline.make_chain_variant("simple")

        pipeline.ask_question("capital?", session_id="shared-id")
        variant.ask_question("capital?", session_id="shared-id")

        # Same session id, but separate checkpointers: each side saw
        # exactly one turn (2 messages), not each other's history.
        for chain in (pipeline.get_chain(), variant):
            state = chain.get_chain().get_state(
                {"configurable": {"thread_id": "shared-id"}}
            )
            assert len(state.values["messages"]) == 2

    def test_variant_requires_retriever(self):
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        pipeline.LLM = _FakeLLMProxy(FakeListChatModel(responses=["x"]))
        pipeline._retriever = None
        with pytest.raises(ChainInitializationError):
            pipeline.make_chain_variant("simple")


class TestStreamChainToQueue:
    def test_streams_chunks_then_done_with_metrics(self):
        class _FakeChain:
            def stream_question(self, question, session_id):
                yield {"event": "search", "query": "refined"}
                yield "the "
                yield "answer"

        out: queue.Queue = queue.Queue()
        StreamlitApp._stream_chain_to_queue(
            _FakeChain(), "agent", "q?", "cmp-1-agent", out
        )

        items = []
        while not out.empty():
            items.append(out.get_nowait())

        kinds = [k for k, _ in items]
        assert kinds == ["chunk", "chunk", "chunk", "done"]
        assert items[0][1] == {"event": "search", "query": "refined"}
        metrics = items[-1][1]
        assert metrics.chain_type == "agent"
        assert metrics.session_id == "cmp-1-agent"
        assert metrics.latency_ms >= 0
        assert metrics.first_token_ms is not None  # captured for the column badge

    def test_error_is_reported_then_done(self):
        class _FailingChain:
            def stream_question(self, question, session_id):
                yield "partial "
                raise RuntimeError("api down")

        out: queue.Queue = queue.Queue()
        StreamlitApp._stream_chain_to_queue(
            _FailingChain(), "simple", "q?", "cmp-2-simple", out
        )

        items = []
        while not out.empty():
            items.append(out.get_nowait())

        kinds = [k for k, _ in items]
        assert kinds == ["chunk", "error", "done"]
        assert "api down" in items[1][1]
        assert items[-1][1].error == "RuntimeError"  # metrics recorded it too
