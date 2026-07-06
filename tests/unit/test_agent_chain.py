"""
Unit tests for the agentic RAG chain (no network required).

The agent is driven by a fake tool-calling chat model: scripted AIMessages
with tool_calls simulate the model deciding to search, and the retriever
records the queries it receives — proving the loop actually runs.
"""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
from ragstone.rag.rag import RagProxy
from ragstone.utils.exceptions import ValidationError
from ragstone.utils.full_chain import FullChain


class ToolCallingFakeModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts tool binding (returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


class QueryRecordingRetriever(SimpleTextRetriever):
    """SimpleTextRetriever that records every query it receives."""

    queries: List[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        self.queries.append(query)
        return super()._get_relevant_documents(query, run_manager=run_manager)


def _make_agent_chain(messages, retriever=None):
    llm = ToolCallingFakeModel(messages=iter(messages))
    if retriever is None:
        retriever = SimpleTextRetriever.from_texts(["Paris is the capital of France."])
    return RagProxy(model=llm, retriever=retriever).make_agent_chain(), retriever


def _search_call(query, call_id):
    return AIMessage(
        "",
        tool_calls=[
            {"name": "search_documents", "args": {"query": query}, "id": call_id}
        ],
    )


class TestAgentRagChain:
    def test_agent_searches_then_answers(self):
        retriever = QueryRecordingRetriever.from_texts(
            ["Paris is the capital of France."]
        )
        chain, _ = _make_agent_chain(
            [_search_call("France capital", "c1"), AIMessage("Paris.")],
            retriever=retriever,
        )

        answer = chain.invoke("capital?")

        assert answer == "Paris."
        # The retriever was queried with the AGENT's refined query, not the
        # raw user question — the model drove retrieval.
        assert retriever.queries == ["France capital"]

    def test_agent_can_search_multiple_times(self):
        retriever = QueryRecordingRetriever.from_texts(["Some fact."])
        chain, _ = _make_agent_chain(
            [
                _search_call("first query", "c1"),
                _search_call("refined query", "c2"),
                AIMessage("Found it on the second search."),
            ],
            retriever=retriever,
        )

        answer = chain.invoke({"question": "hard question?"})

        assert answer == "Found it on the second search."
        assert retriever.queries == ["first query", "refined query"]

    def test_stream_yields_answer_tokens(self):
        # Direct answer (no tool call): the stream must carry the answer
        # text through, chunk by chunk.
        chain, _ = _make_agent_chain([AIMessage("Paris is the capital.")])

        chunks = list(chain.stream("capital?"))

        assert len(chunks) > 1  # actually streamed, not one blob
        assert "".join(chunks) == "Paris is the capital."

    def test_invalid_input_raises_validation_error(self):
        chain, _ = _make_agent_chain([AIMessage("unused")])
        with pytest.raises(ValidationError):
            chain.invoke("   ")
        with pytest.raises(ValidationError):
            chain.invoke(123)

    def test_agent_chain_type_works_through_full_chain(self):
        # End to end through the public path: FullChain wires the agent
        # chain into the memory graph, exactly as pipeline/UI/MCP do.
        llm = ToolCallingFakeModel(messages=iter([AIMessage("Paris.")]))

        class _FakeLLMProxy:
            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(["Paris is the capital of France."])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_FakeLLMProxy(), rag, MemoryProxy())
        full_chain.create_full_chain("agent")

        assert full_chain.ask_question("capital?", session_id="s1") == "Paris."
