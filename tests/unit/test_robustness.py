"""
Unit tests for robustness fixes: generated-query parsing and error
propagation from the ask paths (no network required).
"""

import os
from typing import List

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.models.base_model import OpenAIProxy
from ragstone.rag.memory import SimpleTextRetriever
from ragstone.rag.pipeline import Pipeline
from ragstone.rag.rag import RagProxy, parse_generated_queries
from ragstone.utils.exceptions import (
    ChainExecutionError,
    ChainInitializationError,
    LLMInitializationError,
    RetrieverInitializationError,
    ValidationError,
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


class TestSetupPathErrors:
    """Setup failures must raise at the cause, not surface at ask time."""

    def test_create_rag_chain_without_llm_raises(self):
        with pytest.raises(ChainInitializationError, match="LLM is not set"):
            Pipeline().create_rag_chain()

    def test_set_retriever_without_documents_raises(self):
        with pytest.raises(RetrieverInitializationError, match="load_and_split"):
            Pipeline()._set_retriever(embeddings=None)

    def test_set_retriever_openai_without_key_raises(self, monkeypatch):
        pytest.importorskip("langchain_openai")
        from ragstone.rag.pipeline import OpenAIPipeline

        # Construct while the key is present, then remove it: the retriever
        # setup must check the key at call time.
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set in test environment")
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RetrieverInitializationError, match="OPENAI_API_KEY"):
            pipeline.set_retriever_openai()

    def test_openai_set_llm_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LLMInitializationError, match="OPENAI_API_KEY"):
            OpenAIProxy().set_llm("gpt-4o-mini")


class _PromptCapturingFake(FakeListChatModel):
    """Records the final prompt messages each LLM call receives."""

    last_messages: List = []

    def _generate(self, messages, **kwargs):
        self.last_messages = list(messages)
        return super()._generate(messages, **kwargs)


class TestChainInputShapes:
    def test_dict_input_is_extracted_not_rendered(self):
        # A {"question": ...} input must put the question string into the
        # prompt, not the dict's repr.
        llm = _PromptCapturingFake(responses=["Paris."])
        retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
        chain = RagProxy(model=llm, retriever=retriever).make_chain()

        answer = chain.invoke({"question": "capital?"})

        assert answer == "Paris."
        prompt_text = llm.last_messages[0].content
        assert "capital?" in prompt_text
        assert "{'question'" not in prompt_text

    def test_empty_question_raises_validation_error(self):
        llm = FakeListChatModel(responses=["x"])
        retriever = SimpleTextRetriever.from_texts(["content"])
        chain = RagProxy(model=llm, retriever=retriever).make_chain()

        with pytest.raises(ValidationError):
            chain.invoke("   ")
