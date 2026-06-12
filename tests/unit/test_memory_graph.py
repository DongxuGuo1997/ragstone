"""
Unit tests for the LangGraph-based conversation memory (no network required).
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
from ragstone.rag.rag import RagProxy
from ragstone.utils.full_chain import FullChain


class _CountingFakeChatModel(FakeListChatModel):
    """FakeListChatModel that counts how many times it is called."""

    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        self.calls += 1
        return super()._stream(*args, **kwargs)


class _FakeLLMProxy:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _make_full_chain(llm):
    retriever = SimpleTextRetriever.from_texts(["Paris is the capital of France."])
    rag = RagProxy(model=llm, retriever=retriever)
    full_chain = FullChain(_FakeLLMProxy(llm), rag, MemoryProxy())
    full_chain.create_full_chain("simple")
    return full_chain


class TestMemoryBehavior:
    def test_first_turn_skips_rephrase(self):
        llm = _CountingFakeChatModel(responses=["Paris."])
        full_chain = _make_full_chain(llm)

        answer = full_chain.ask_question("capital?", session_id="s1")

        assert answer == "Paris."
        assert llm.calls == 1  # no rephrase call on an empty history

    def test_second_turn_rephrases_against_history(self):
        # Call order: turn-1 answer, turn-2 rephrase, turn-2 answer.
        # The sentinel proves the rephrase call consumed response #2.
        llm = _CountingFakeChatModel(responses=["Paris.", "REPHRASED", "About 2M."])
        full_chain = _make_full_chain(llm)

        first = full_chain.ask_question("capital?", session_id="s1")
        second = full_chain.ask_question("population?", session_id="s1")

        assert first == "Paris."
        assert second == "About 2M."
        assert llm.calls == 3

    def test_history_stores_original_questions(self):
        llm = _CountingFakeChatModel(responses=["Paris.", "REPHRASED", "About 2M."])
        full_chain = _make_full_chain(llm)

        full_chain.ask_question("capital?", session_id="s1")
        full_chain.ask_question("population?", session_id="s1")

        state = full_chain.get_chain().get_state({"configurable": {"thread_id": "s1"}})
        contents = [m.content for m in state.values["messages"]]
        assert contents == ["capital?", "Paris.", "population?", "About 2M."]

    def test_sessions_are_isolated(self):
        llm = _CountingFakeChatModel(responses=["Paris.", "Paris."])
        full_chain = _make_full_chain(llm)

        full_chain.ask_question("capital?", session_id="a")
        calls_before_b = llm.calls
        full_chain.ask_question("capital?", session_id="b")

        # Session "b" saw no history, so no rephrase call was made.
        assert llm.calls == calls_before_b + 1
        state_b = full_chain.get_chain().get_state({"configurable": {"thread_id": "b"}})
        assert len(state_b.values["messages"]) == 2
