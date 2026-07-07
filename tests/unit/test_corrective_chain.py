"""
Unit tests for the Corrective RAG chain (no network required).

Every path through the graph is exercised with scripted fakes: the happy
path (grade passes first try), the correction cycle (rewrite -> retrieve
-> grade again), and the bounded refusal (attempts exhausted -> honest
"not found", no answer-model call wasted).
"""

from typing import List

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.corrective import CorrectiveRagChain
from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
from ragstone.rag.rag import RagProxy
from ragstone.utils.exceptions import ValidationError
from ragstone.utils.full_chain import FullChain


class _CountingFakeChatModel(FakeListChatModel):
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        self.calls += 1
        return super()._stream(*args, **kwargs)


class QueryRecordingRetriever(SimpleTextRetriever):
    queries: List[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        self.queries.append(query)
        return super()._get_relevant_documents(query, run_manager=run_manager)


def _chain(answer_responses, utility_responses, retriever=None, max_retries=2):
    answer_llm = _CountingFakeChatModel(responses=list(answer_responses))
    utility_llm = _CountingFakeChatModel(responses=list(utility_responses))
    if retriever is None:
        retriever = QueryRecordingRetriever.from_texts(["Paris is the capital."])
    chain = CorrectiveRagChain(
        answer_llm, retriever, utility_llm=utility_llm, max_retries=max_retries
    )
    return chain, answer_llm, utility_llm, retriever


class TestHappyPath:
    def test_relevant_first_try_answers_directly(self):
        chain, answer_llm, utility_llm, retriever = _chain(
            answer_responses=["Paris."], utility_responses=["yes"]
        )

        assert chain.invoke("capital?") == "Paris."
        assert retriever.queries == ["capital?"]  # one retrieval
        assert utility_llm.calls == 1  # one grade, no rewrite
        assert answer_llm.calls == 1

    def test_events_expose_the_loop(self):
        chain, *_ = _chain(answer_responses=["Paris."], utility_responses=["yes"])

        items = list(chain.stream_with_events("capital?"))

        events = [i["event"] for i in items if isinstance(i, dict)]
        text = "".join(i for i in items if isinstance(i, str))
        assert events == ["retrieve", "grade"]
        assert text == "Paris."

    def test_plain_stream_keeps_text_only_contract(self):
        chain, *_ = _chain(answer_responses=["Paris."], utility_responses=["yes"])
        chunks = list(chain.stream("capital?"))
        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks) == "Paris."


class TestCorrectionCycle:
    def test_poor_grade_triggers_rewrite_and_reretrieval(self):
        # Grade says no, rewrite produces a new query, second grade says
        # yes -> answered. Utility calls: grade, rewrite, grade.
        chain, answer_llm, utility_llm, retriever = _chain(
            answer_responses=["E-42 means overvoltage."],
            utility_responses=["no", "inverter fault E-42 meaning", "yes"],
        )

        answer = chain.invoke("what is E-42?")

        assert answer == "E-42 means overvoltage."
        assert retriever.queries == ["what is E-42?", "inverter fault E-42 meaning"]
        assert utility_llm.calls == 3
        assert answer_llm.calls == 1

    def test_event_sequence_shows_the_cycle(self):
        chain, *_ = _chain(
            answer_responses=["answer"],
            utility_responses=["no", "better query", "yes"],
        )

        events = [i for i in chain.stream_with_events("q?") if isinstance(i, dict)]

        assert [e["event"] for e in events] == [
            "retrieve",
            "grade",
            "rewrite",
            "retrieve",
            "grade",
        ]
        assert events[1]["relevant"] is False
        assert events[2]["query"] == "better query"
        assert events[4]["relevant"] is True


class TestBoundedRefusal:
    def test_exhausted_attempts_refuse_with_evidence(self):
        # Grader never approves: grade, rewrite, grade, rewrite, grade —
        # then refusal. The answer model is NEVER called.
        chain, answer_llm, utility_llm, retriever = _chain(
            answer_responses=["should never be used"],
            utility_responses=["no", "query-2", "no", "query-3", "no"],
            max_retries=2,
        )

        answer = chain.invoke("what is the commander's salary?")

        assert "couldn't find this in the documents" in answer
        # The refusal cites every query it tried — evidence, not vibes.
        for tried in ["what is the commander's salary?", "query-2", "query-3"]:
            assert tried in answer
        assert answer_llm.calls == 0  # no tokens wasted on a doomed answer
        assert len(retriever.queries) == 3  # initial + exactly max_retries

    def test_cycle_is_bounded_even_with_zero_retries(self):
        chain, answer_llm, _, retriever = _chain(
            answer_responses=["unused"],
            utility_responses=["no"],
            max_retries=0,
        )
        answer = chain.invoke("q?")
        assert "couldn't find" in answer
        assert len(retriever.queries) == 1
        assert answer_llm.calls == 0


class TestIntegration:
    def test_invalid_input_raises_validation_error(self):
        chain, *_ = _chain(answer_responses=["x"], utility_responses=["yes"])
        with pytest.raises(ValidationError):
            chain.invoke("   ")

    def test_corrective_chain_type_through_full_chain(self):
        # Public path: FullChain wires the corrective chain into the
        # memory graph. Model serves: grade ("yes") then streamed answer.
        llm = _CountingFakeChatModel(responses=["yes", "Paris."])

        class _Proxy:
            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_Proxy(), rag, MemoryProxy())
        full_chain.create_full_chain("corrective")

        assert full_chain.ask_question("capital?", session_id="s1") == "Paris."

    def test_events_flow_through_memory_graph(self):
        llm = _CountingFakeChatModel(responses=["yes", "Paris."])

        class _Proxy:
            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(["Paris is the capital."])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_Proxy(), rag, MemoryProxy())
        full_chain.create_full_chain("corrective")

        received = list(full_chain.stream_question("capital?", session_id="s2"))

        event_kinds = [c["event"] for c in received if isinstance(c, dict)]
        assert event_kinds == ["retrieve", "grade"]
        text = "".join(c for c in received if isinstance(c, str))
        assert text == "Paris."
        # Checkpointed answer stays clean of event noise.
        state = full_chain.get_chain().get_state({"configurable": {"thread_id": "s2"}})
        assert state.values["answer"] == "Paris."
