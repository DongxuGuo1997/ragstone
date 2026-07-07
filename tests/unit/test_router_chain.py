"""
Unit tests for query routing (chain_type="auto", no network).

The router's contract: one utility-model call decides the strategy, the
chosen chain answers, a route event precedes everything, classification
failures degrade to the simple chain (never break answering), and the
Runnable[..., str] contract holds so memory/servers/UIs work unchanged.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.router import RouterRagChain
from ragstone.utils.exceptions import ValidationError


class _RecordingChain:
    """Stands in for the simple/corrective chains; records invocations."""

    def __init__(self, name, events=()):
        self.name = name
        self.events = list(events)
        self.calls = []

    def invoke(self, question):
        self.calls.append(question)
        return f"{self.name} answer"

    def stream(self, question):
        self.calls.append(question)
        yield f"{self.name} "
        yield "answer"

    def stream_with_events(self, question):
        self.calls.append(question)
        yield from self.events
        yield f"{self.name} "
        yield "answer"


class _BoomModel(FakeListChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("classifier down")


def _router(verdict="simple", simple=None, careful=None, llm=None):
    simple = simple or _RecordingChain("simple")
    careful = careful or _RecordingChain(
        "careful", events=[{"event": "retrieve", "query": "q"}]
    )
    llm = llm or FakeListChatModel(responses=[verdict])
    return RouterRagChain(simple, careful, llm), simple, careful


class TestRouting:
    """The classifier's verdict picks the chain; exactly one chain runs."""

    def test_simple_verdict_routes_to_the_simple_chain(self):
        router, simple, careful = _router("simple")
        assert router.invoke("What is the warranty?") == "simple answer"
        assert simple.calls == ["What is the warranty?"]
        assert careful.calls == []

    def test_careful_verdict_routes_to_the_corrective_chain(self):
        router, simple, careful = _router("careful")
        assert router.invoke("Compare the MK-3 and BX-2") == "careful answer"
        assert careful.calls == ["Compare the MK-3 and BX-2"]
        assert simple.calls == []

    def test_verdict_parsing_tolerates_case_and_whitespace(self):
        router, _, careful = _router("  Careful \n")
        router.invoke("q?")
        assert careful.calls == ["q?"]


class TestFailSafe:
    """A broken router must degrade to the default chain, never raise."""

    def test_unrecognized_verdict_falls_back_to_simple(self):
        router, simple, careful = _router("both, probably?")
        assert router.invoke("q?") == "simple answer"
        assert careful.calls == []

    def test_classifier_exception_falls_back_to_simple(self):
        router, simple, careful = _router(llm=_BoomModel(responses=["unused"]))
        assert router.invoke("q?") == "simple answer"
        assert simple.calls == ["q?"]

    def test_blank_input_is_still_rejected(self):
        router, _, _ = _router()
        with pytest.raises(ValidationError):
            router.invoke("   ")


class TestStreamingContract:
    """Route event first; plain stream() stays text-only."""

    def test_stream_with_events_yields_route_first_then_delegate(self):
        router, _, _ = _router("careful")
        chunks = list(router.stream_with_events("compare?"))
        assert chunks[0] == {"event": "route", "strategy": "careful"}
        assert chunks[1] == {"event": "retrieve", "query": "q"}  # delegate's
        assert "".join(c for c in chunks if isinstance(c, str)) == "careful answer"

    def test_plain_stream_never_yields_dicts(self):
        router, _, _ = _router("careful")
        chunks = list(router.stream("compare?"))
        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks) == "careful answer"

    def test_delegates_without_events_use_plain_stream(self):
        class _PlainChain:
            def stream(self, question):
                yield "plain answer"

        router = RouterRagChain(
            _PlainChain(),
            _RecordingChain("careful"),
            FakeListChatModel(responses=["simple"]),
        )
        chunks = list(router.stream_with_events("q?"))
        assert chunks == [{"event": "route", "strategy": "simple"}, "plain answer"]


class TestFullChainIntegration:
    """chain_type="auto" through the memory graph: events flow, answer clean."""

    def test_auto_chain_streams_route_events_through_memory(self, monkeypatch):
        from ragstone.models.base_model import LLMProxy
        from ragstone.rag.memory import MemoryProxy, SimpleTextRetriever
        from ragstone.rag.rag import RagProxy
        from ragstone.utils.full_chain import FullChain

        # One fake model serves both the classifier and the answer chain
        # (FakeListChatModel cycles). Its response is not a valid verdict,
        # so this integration case deliberately exercises the fail-safe
        # route (-> simple) end to end through the memory graph.
        llm = FakeListChatModel(responses=["The warranty is 28 years."])

        class _FakeLLMProxy(LLMProxy):
            def set_llm(self, model_name=None, **kwargs):
                self._llm = llm

            def get_llm(self):
                return llm

        retriever = SimpleTextRetriever.from_texts(
            ["The Helios MK-3 warranty is 28 years."]
        )
        rag = RagProxy(model=llm, retriever=retriever)
        chain = FullChain(_FakeLLMProxy(), rag, MemoryProxy())
        chain.create_full_chain("auto")
        router = chain._rag  # noqa: SLF001 - sanity only

        collected = list(chain.stream_question("How long is it?", "s1"))
        events = [c for c in collected if isinstance(c, dict)]
        text = "".join(c for c in collected if isinstance(c, str))
        assert any(e.get("event") == "route" for e in events)
        assert "28 years" in text

        # The checkpointed answer stays free of event noise.
        state = chain.get_chain().get_state({"configurable": {"thread_id": "s1"}})
        assert "28 years" in state.values["answer"]
        assert router is not None
