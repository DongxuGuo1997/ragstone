"""
Unit tests for robustness fixes: generated-query parsing and error
propagation from the ask paths (no network required).
"""

from typing import List

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from know_rag.rag.memory import SimpleTextRetriever
from know_rag.rag.pipeline import Pipeline
from know_rag.rag.rag import RagProxy, parse_generated_queries
from know_rag.utils.exceptions import (
    ChainExecutionError,
    ChainInitializationError,
)


class TestParseGeneratedQueries:
    def test_drops_blank_lines_and_list_markers(self):
        text = "1. What is X?\n\n2) How does Y work?\n- Z details\n   \n"
        assert parse_generated_queries(text) == [
            "What is X?",
            "How does Y work?",
            "Z details",
        ]

    def test_plain_queries_pass_through(self):
        assert parse_generated_queries("alpha\nbeta") == ["alpha", "beta"]

    def test_all_blank_gives_empty_list(self):
        assert parse_generated_queries("\n  \n") == []


class _RecordingRetriever(SimpleTextRetriever):
    """SimpleTextRetriever that records the queries it receives."""

    queries: List[str] = []

    def _get_relevant_documents(self, query, *, run_manager):
        self.queries.append(query)
        return super()._get_relevant_documents(query, run_manager=run_manager)


class TestMultiQueryFiltering:
    def test_multi_query_chain_filters_blank_and_numbered_queries(self):
        # First LLM call generates queries (with numbering and a blank
        # line), second call answers. Blank lines must never reach the
        # retriever — real embedding APIs reject empty input.
        llm = FakeListChatModel(responses=["1. alpha\n\n2. beta\n", "Paris."])
        retriever = _RecordingRetriever.from_texts(["Paris is the capital."])
        rag = RagProxy(model=llm, retriever=retriever)

        answer = rag.make_multi_query_chain().invoke("capital?")

        assert answer == "Paris."
        assert retriever.queries == ["alpha", "beta"]


class _BoomChain:
    """Stand-in FullChain whose generation always fails."""

    def ask_question(self, query, session_id):
        raise RuntimeError("connection dropped")

    def stream_question(self, query, session_id):
        yield "partial "
        raise RuntimeError("connection dropped")


class TestAskPathErrors:
    def test_ask_question_without_chain_raises(self):
        with pytest.raises(ChainInitializationError):
            Pipeline().ask_question("hi")

    def test_ask_question_stream_without_chain_raises(self):
        with pytest.raises(ChainInitializationError):
            list(Pipeline().ask_question_stream("hi"))

    def test_ask_failure_raises_chain_execution_error(self):
        pipeline = Pipeline()
        pipeline._chain = _BoomChain()
        with pytest.raises(ChainExecutionError, match="connection dropped"):
            pipeline.ask_question("q", session_id="s")

    def test_stream_failure_raises_after_partial_output(self):
        # A mid-stream failure must surface as an exception so callers can
        # tell a truncated answer from a complete one.
        pipeline = Pipeline()
        pipeline._chain = _BoomChain()
        chunks = []
        with pytest.raises(ChainExecutionError, match="connection dropped"):
            for chunk in pipeline.ask_question_stream("q", session_id="s"):
                chunks.append(chunk)
        assert chunks == ["partial "]
