"""
Unit tests for request-scoped ask state (no network).

This is the end of the single-writer caveat: two asks running
CONCURRENTLY on one pipeline must each see their own sources and
metrics. The suite also locks the ownership rule (a context from
pipeline A never leaks into pipeline B's reads) and the fallback for
retrievals outside any ask (compare-view chain variants).
"""

import threading

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragstone.rag.ask_context import (
    SourceRecordingRetriever,
    begin_ask,
    current_ask,
)
from ragstone.rag.pipeline import OpenAIPipeline


def _doc(text):
    return Document(page_content=text, metadata={"source": f"{text[:8]}.md"})


class _EchoRetriever(BaseRetriever):
    """Returns a doc naming the query, so bleed is detectable."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        return [_doc(f"docs-for:{query}")]


def _pipeline():
    pipeline = OpenAIPipeline(model="gpt-4o-mini")
    pipeline._chain_type = "simple"
    pipeline._retriever = SourceRecordingRetriever(
        wrapped=_EchoRetriever(), owner_id=id(pipeline)
    )

    class _Chain:
        """Answers by retrieving through the pipeline's recorder."""

        def __init__(self, pipeline, gate=None):
            self._pipeline = pipeline
            self._gate = gate

        def ask_question(self, query, session_id):
            self._pipeline._retriever.invoke(query)
            if self._gate is not None:
                self._gate.wait(timeout=5)  # hold both asks in flight
            return f"answer:{query}"

        def has_history(self, session_id):
            return False

    return pipeline, _Chain


class TestConcurrentAsks:
    """The headline: concurrent asks on ONE pipeline do not bleed."""

    def test_each_concurrent_ask_sees_its_own_sources_and_metrics(self):
        pipeline, chain_cls = _pipeline()
        barrier = threading.Barrier(2)
        pipeline._chain = chain_cls(pipeline, gate=barrier)
        results = {}

        def ask(question):
            answer = pipeline.ask_question(question, session_id=question)
            # Introspection reads happen in the SAME call flow as the ask
            # — the supported pattern — while the other ask is also live.
            results[question] = {
                "answer": answer,
                "docs": [d.page_content for d in pipeline.get_last_retrieved_documents()],
                "sources": pipeline.get_sources(question),
                "session": pipeline.last_metrics.session_id,
            }

        threads = [
            threading.Thread(target=ask, args=(question,))
            for question in ("alpha?", "beta?")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        for question in ("alpha?", "beta?"):
            entry = results[question]
            assert entry["answer"] == f"answer:{question}"
            assert entry["docs"] == [f"docs-for:{question}"]  # no bleed
            assert entry["sources"][0]["snippet"] == f"docs-for:{question}"
            assert entry["session"] == question  # metrics not swapped

    def test_unrelated_thread_sees_most_recently_started_ask(self):
        pipeline, chain_cls = _pipeline()
        pipeline._chain = chain_cls(pipeline)
        pipeline.ask_question("gamma?", session_id="s")

        seen = {}

        def read():
            seen["docs"] = [
                d.page_content for d in pipeline.get_last_retrieved_documents()
            ]

        thread = threading.Thread(target=read)
        thread.start()
        thread.join(timeout=5)
        assert seen["docs"] == ["docs-for:gamma?"]  # legacy semantics kept


class TestOwnership:
    """A context is only readable by the pipeline that opened it."""

    def test_stale_context_from_another_pipeline_is_ignored(self):
        pipeline_a, chain_cls = _pipeline()
        pipeline_a._chain = chain_cls(pipeline_a)
        pipeline_a.ask_question("from-a?", session_id="a")

        # Same thread, different pipeline: A's context is still on the
        # ContextVar, but B must not serve A's data as its own.
        pipeline_b, chain_cls_b = _pipeline()
        pipeline_b._chain = chain_cls_b(pipeline_b)
        assert current_ask() is not None  # A's context is indeed live
        assert pipeline_b.get_last_retrieved_documents() == []
        assert pipeline_b.last_metrics is None

    def test_recorder_ignores_a_foreign_context(self):
        recorder = SourceRecordingRetriever(wrapped=_EchoRetriever(), owner_id=1)
        begin_ask("foreign", owner_id=2)
        docs = recorder.invoke("q?")
        assert current_ask().record == []  # not recorded into the foreign ask
        assert recorder.fallback_record == docs


class TestFallback:
    """Retrievals outside any ask keep bounded, most-recent semantics."""

    def test_fallback_replaces_rather_than_grows(self):
        recorder = SourceRecordingRetriever(wrapped=_EchoRetriever(), owner_id=99)
        begin_ask("other", owner_id=1)  # foreign context -> fallback path
        recorder.invoke("first")
        recorder.invoke("second")
        assert [d.page_content for d in recorder.fallback_record] == [
            "docs-for:second"
        ]

    def test_cache_hit_leaves_an_empty_record(self):
        # A cache hit retrieves nothing; the ask's record must be empty,
        # not the previous ask's leftovers.
        pipeline, chain_cls = _pipeline()
        pipeline._chain = chain_cls(pipeline)
        pipeline.ask_question("delta?", session_id="s")
        assert pipeline.get_last_retrieved_documents() != []

        pipeline._begin_ask("fresh question with no retrieval")
        assert pipeline.get_last_retrieved_documents() == []
