"""
Unit tests for the LangGraph-based conversation memory (no network required).
"""

import pytest
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

    def test_interpretation_exposed_after_rephrase(self):
        # The glass-box UI reads the rephrased question from graph state;
        # it must be None on the first turn and populated after a follow-up.
        llm = _CountingFakeChatModel(responses=["Paris.", "REPHRASED", "About 2M."])
        full_chain = _make_full_chain(llm)

        full_chain.ask_question("capital?", session_id="s1")
        assert full_chain.get_interpretation("s1") is None  # no rephrase yet

        full_chain.ask_question("population?", session_id="s1")
        assert full_chain.get_interpretation("s1") == "REPHRASED"

    def test_interpretation_none_for_unknown_session(self):
        llm = _CountingFakeChatModel(responses=["Paris."])
        full_chain = _make_full_chain(llm)
        assert full_chain.get_interpretation("never-used") is None

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


class TestSessionCap:
    def test_history_is_trimmed_beyond_the_cap(self, monkeypatch):
        import ragstone.rag.memory as memory_mod

        # Cap of 4 messages = 2 turns; the 3rd turn must evict the 1st.
        monkeypatch.setattr(memory_mod, "MAX_SESSION_MESSAGES", 4)
        llm = _CountingFakeChatModel(responses=["A1.", "R2", "A2.", "R3", "A3."])
        full_chain = _make_full_chain(llm)

        full_chain.ask_question("q1?", session_id="cap")
        full_chain.ask_question("q2?", session_id="cap")
        full_chain.ask_question("q3?", session_id="cap")

        state = full_chain.get_chain().get_state({"configurable": {"thread_id": "cap"}})
        contents = [m.content for m in state.values["messages"]]
        # Oldest turn (q1) evicted; the two most recent remain, in order.
        assert contents == ["q2?", "A2.", "q3?", "A3."]

    def test_short_sessions_are_untouched(self):
        llm = _CountingFakeChatModel(responses=["Paris."])
        full_chain = _make_full_chain(llm)
        full_chain.ask_question("capital?", session_id="s")
        state = full_chain.get_chain().get_state({"configurable": {"thread_id": "s"}})
        assert len(state.values["messages"]) == 2


class TestRephraseModelSelection:
    def test_none_uses_main_model(self):
        from ragstone.rag.memory import _make_rephrase_llm

        llm = _CountingFakeChatModel(responses=["x"])
        assert _make_rephrase_llm(llm, None) is llm

    def test_configured_model_builds_a_sibling(self):
        from ragstone.rag.memory import _make_rephrase_llm

        built = {}

        class _ModelTakingFake(_CountingFakeChatModel):
            def __init__(self, model=None, temperature=None, **kwargs):
                kwargs.setdefault("responses", ["y"])
                super().__init__(**kwargs)
                built["model"] = model

        main = _ModelTakingFake(model="big-model")
        rephrase = _make_rephrase_llm(main, "tiny-model")

        assert rephrase is not main
        assert built["model"] == "tiny-model"

    def test_unbuildable_model_falls_back_to_main(self):
        from ragstone.rag.memory import _make_rephrase_llm

        # FakeListChatModel requires `responses`; constructing a sibling
        # with only model= raises — the fallback must kick in silently.
        llm = _CountingFakeChatModel(responses=["x"])
        assert _make_rephrase_llm(llm, "tiny-model") is llm

    def test_sibling_inherits_the_main_models_reasoning(self):
        # ChatOllama-shaped: the thinking knob set on the main model
        # (RAGSTONE_OLLAMA_REASONING) must carry to a configured utility
        # sibling, or rephrase/grade/route quietly revert to thinking.
        from ragstone.rag.memory import _make_rephrase_llm

        built = {}

        class _OllamaShaped:
            def __init__(self, model=None, temperature=None, reasoning=None):
                self.model = model
                self.reasoning = reasoning
                built["model"] = model
                built["reasoning"] = reasoning

        main = _OllamaShaped(model="big-model", reasoning=False)
        sibling = _make_rephrase_llm(main, "tiny-model")

        assert sibling is not main
        assert built == {"model": "tiny-model", "reasoning": False}

    def test_openai_shaped_sibling_gets_no_reasoning_kwarg(self):
        # ChatOpenAI has no reasoning attribute/param: passing one would
        # raise in __init__ and trip the fallback — build must stay clean.
        from ragstone.rag.memory import _make_rephrase_llm

        built = {}

        class _OpenAIShaped:
            def __init__(self, model=None, temperature=None):
                self.model = model
                built["model"] = model

        main = _OpenAIShaped(model="big-model")
        sibling = _make_rephrase_llm(main, "tiny-model")

        assert sibling is not main
        assert built["model"] == "tiny-model"


class TestCheckpointBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            MemoryProxy(type="redis")

    def test_inmemory_alias_accepted(self):
        # Backward-compatible alias for the old default.
        assert MemoryProxy(type="InMemory")._type == "memory"

    def test_sqlite_memory_persists_across_graphs(self, tmp_path):
        # The whole point of the sqlite backend: a fresh graph pointed at the
        # same file sees a prior "process's" conversation history.
        pytest.importorskip("langgraph.checkpoint.sqlite")
        db_path = str(tmp_path / "checkpoints.sqlite")

        def build_chain():
            llm = FakeListChatModel(responses=["Paris.", "REPHRASED", "About 2M."])
            retriever = SimpleTextRetriever.from_texts(
                ["Paris is the capital of France."]
            )
            rag = RagProxy(model=llm, retriever=retriever)
            proxy = MemoryProxy(type="sqlite")
            proxy._db_path = db_path  # redirect off the default store/ path
            chain = FullChain(_FakeLLMProxy(llm), rag, proxy)
            chain.create_full_chain("simple")
            return chain

        first_chain = build_chain()
        assert first_chain.ask_question("capital?", session_id="s1") == "Paris."

        # Simulate a restart: a brand-new graph over the same SQLite file.
        second_chain = build_chain()
        state = second_chain.get_chain().get_state(
            {"configurable": {"thread_id": "s1"}}
        )
        assert [m.content for m in state.values["messages"]] == ["capital?", "Paris."]

    def test_close_releases_sqlite_connection_and_is_idempotent(self, tmp_path):
        # Without close(), a server that rebuilds chains or deletes
        # pipelines leaks one SQLite connection per built chain.
        import sqlite3

        pytest.importorskip("langgraph.checkpoint.sqlite")
        proxy = MemoryProxy(type="sqlite")
        proxy._db_path = str(tmp_path / "checkpoints.sqlite")
        proxy._make_checkpointer()
        conn = proxy._conn
        assert conn is not None

        proxy.close()
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
        proxy.close()  # safe to call again

    def test_close_is_a_noop_for_the_inmemory_backend(self):
        proxy = MemoryProxy(type="memory")
        proxy._make_checkpointer()
        proxy.close()  # nothing to release, must not raise

    def test_sqlite_missing_dependency_raises_actionable_error(self, monkeypatch):
        # If the optional package is absent, selecting sqlite must fail with a
        # clear "install the extra" message, not an opaque ImportError.
        import builtins

        from ragstone.utils.exceptions import ChainInitializationError

        real_import = builtins.__import__

        def deny_sqlite(name, *args, **kwargs):
            if name == "langgraph.checkpoint.sqlite":
                raise ImportError("no sqlite saver")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", deny_sqlite)
        with pytest.raises(ChainInitializationError, match="sqlite"):
            MemoryProxy(type="sqlite")._make_checkpointer()
