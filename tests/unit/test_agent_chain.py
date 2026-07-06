"""
Unit tests for the agentic RAG chain (no network required).

The agent is driven by a fake tool-calling chat model: scripted AIMessages
with tool_calls simulate the model deciding to search, and the retriever
records the queries it receives — proving the loop actually runs.
"""

import json
from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
from ragstone.rag.rag import RagProxy
from ragstone.utils.exceptions import ValidationError
from ragstone.utils.full_chain import FullChain


class ToolCallingFakeModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts tool binding (returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


class StreamingToolCallFakeModel(ToolCallingFakeModel):
    """Streams each scripted message as one chunk, tool calls included.

    GenericFakeChatModel cannot stream an empty-content tool-call message
    (it splits content into word chunks); real models stream tool calls as
    tool_call_chunks, which is what this emits.
    """

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        chunk = AIMessageChunk(
            content=message.content,
            tool_call_chunks=[
                {
                    "name": tc["name"],
                    "args": json.dumps(tc["args"]),
                    "id": tc["id"],
                    "index": i,
                    "type": "tool_call_chunk",
                }
                for i, tc in enumerate(message.tool_calls)
            ],
        )
        yield ChatGenerationChunk(message=chunk)


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

    def test_stream_with_events_yields_searches_then_answer(self):
        # The live-thinking feature: the search must surface as an event
        # BEFORE the answer text, carrying the agent's actual query.
        llm = StreamingToolCallFakeModel(
            messages=iter(
                [_search_call("solar fault codes", "c1"), AIMessage("E-42 means…")]
            )
        )
        retriever = SimpleTextRetriever.from_texts(["E-42 is overvoltage."])
        chain = RagProxy(model=llm, retriever=retriever).make_agent_chain()

        items = list(chain.stream_with_events("what is E-42?"))

        events = [i for i in items if isinstance(i, dict)]
        text = "".join(i for i in items if isinstance(i, str))
        assert events == [{"event": "search", "query": "solar fault codes"}]
        assert text == "E-42 means…"
        # And the event arrived before any answer text.
        assert isinstance(items[0], dict)

    def test_plain_stream_keeps_text_only_contract(self):
        # Runnable[..., str] consumers must never receive dicts.
        llm = StreamingToolCallFakeModel(
            messages=iter([_search_call("q", "c1"), AIMessage("answer")])
        )
        retriever = SimpleTextRetriever.from_texts(["doc"])
        chain = RagProxy(model=llm, retriever=retriever).make_agent_chain()

        chunks = list(chain.stream("question?"))

        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks) == "answer"

    def test_events_flow_through_memory_graph_but_not_into_answer(self):
        # Full path: agent chain inside the memory graph, consumed the way
        # the pipeline consumes it. Events must reach the stream, the
        # recorded answer must contain only text.
        llm = StreamingToolCallFakeModel(
            messages=iter([_search_call("nested q", "n1"), AIMessage("Paris.")])
        )

        class _FakeLLMProxy:
            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_FakeLLMProxy(), rag, MemoryProxy())
        full_chain.create_full_chain("agent")

        received = list(full_chain.stream_question("capital?", session_id="s1"))

        assert {"event": "search", "query": "nested q"} in received
        text = "".join(c for c in received if isinstance(c, str))
        assert text == "Paris."
        # The checkpointed answer must not contain event noise.
        state = full_chain.get_chain().get_state({"configurable": {"thread_id": "s1"}})
        assert state.values["answer"] == "Paris."

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
