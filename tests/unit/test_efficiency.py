"""
Unit tests for efficiency fixes: source reuse (no double retrieval),
vector-store reuse via corpus fingerprint, rephrase history window, and
cross-encoder caching. No network required.
"""

from typing import List
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.memory import (
    MAX_REPHRASE_HISTORY,
    MemoryProxy,
    SimpleTextRetriever,
)
from ragstone.rag.pipeline import Pipeline, _SourceRecordingRetriever
from ragstone.rag.rag import RagProxy
from ragstone.utils.full_chain import FullChain


class _CountingRetriever(SimpleTextRetriever):
    calls: int = 0

    def _get_relevant_documents(self, query, *, run_manager):
        self.calls += 1
        return super()._get_relevant_documents(query, run_manager=run_manager)


class _FakeLLMProxy:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


def _make_pipeline(llm, inner_retriever):
    pipeline = Pipeline()
    pipeline._retriever = _SourceRecordingRetriever(wrapped=inner_retriever)
    rag = RagProxy(model=llm, retriever=pipeline._retriever)
    full_chain = FullChain(_FakeLLMProxy(llm), rag, MemoryProxy())
    full_chain.create_full_chain("simple")
    pipeline._chain = full_chain
    return pipeline


class TestSourceReuse:
    def test_get_sources_reuses_docs_from_last_ask(self):
        llm = FakeListChatModel(responses=["Paris."])
        inner = _CountingRetriever(
            docs=[
                Document(
                    page_content="Paris is the capital of France.",
                    metadata={"source": "/data/france.md"},
                )
            ]
        )
        pipeline = _make_pipeline(llm, inner)

        answer = pipeline.ask_question("capital?", session_id="s")
        assert answer == "Paris."
        calls_after_ask = inner.calls

        sources = pipeline.get_sources("capital?")

        assert inner.calls == calls_after_ask  # no second retrieval
        assert sources == [
            {
                "source": "france.md",
                "snippet": "Paris is the capital of France.",
            }
        ]

    def test_get_sources_for_other_question_invokes_retriever(self):
        llm = FakeListChatModel(responses=["Paris."])
        inner = _CountingRetriever.from_texts(["Paris is the capital."])
        pipeline = _make_pipeline(llm, inner)

        pipeline.ask_question("capital?", session_id="s")
        calls_after_ask = inner.calls

        pipeline.get_sources("a different question")

        assert inner.calls == calls_after_ask + 1


class TestVectorStoreReuse:
    def test_set_retriever_skips_reembedding_unchanged_corpus(self):
        pipeline = Pipeline()
        pipeline.texts = [Document(page_content="hello", metadata={"source": "a"})]
        pipeline.vector_db = Mock()
        vs = Mock()
        vs.as_retriever.return_value = SimpleTextRetriever.from_texts(["hello"])
        pipeline.vector_db.db = vs

        pipeline._set_retriever(embeddings=None, use_ensemble=False)
        pipeline._set_retriever(embeddings=None, use_ensemble=False)

        assert pipeline.vector_db.create_db.call_count == 1

    def test_set_retriever_rebuilds_on_corpus_change(self):
        pipeline = Pipeline()
        pipeline.texts = [Document(page_content="hello", metadata={})]
        pipeline.vector_db = Mock()
        vs = Mock()
        vs.as_retriever.return_value = SimpleTextRetriever.from_texts(["hello"])
        pipeline.vector_db.db = vs

        pipeline._set_retriever(embeddings=None, use_ensemble=False)
        pipeline.texts.append(Document(page_content="world", metadata={}))
        pipeline._set_retriever(embeddings=None, use_ensemble=False)

        assert pipeline.vector_db.create_db.call_count == 2

    def test_corpus_fingerprint_is_stable_and_content_sensitive(self):
        pipeline = Pipeline()
        pipeline.texts = [Document(page_content="a", metadata={"source": "x"})]
        first = pipeline._corpus_fingerprint(None)
        assert pipeline._corpus_fingerprint(None) == first

        pipeline.texts[0].metadata["source"] = "y"
        assert pipeline._corpus_fingerprint(None) != first


class _PromptSizeFake(FakeListChatModel):
    """Records how many messages each LLM call receives."""

    sizes: List[int] = []

    def _generate(self, messages, **kwargs):
        self.sizes.append(len(messages))
        return super()._generate(messages, **kwargs)


class TestRephraseHistoryWindow:
    def test_rephrase_sees_at_most_window_messages(self):
        llm = _PromptSizeFake(responses=["x"])
        retriever = SimpleTextRetriever.from_texts(["context"])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_FakeLLMProxy(llm), rag, MemoryProxy())
        full_chain.create_full_chain("simple")

        # 8 turns -> 14 history messages before the last rephrase, well past
        # the window.
        for i in range(8):
            full_chain.ask_question(f"q{i}", session_id="w")

        # Rephrase prompt = 1 system + windowed history + 1 human question.
        assert max(llm.sizes) == MAX_REPHRASE_HISTORY + 2


class TestCrossEncoderCache:
    def test_cross_encoder_loaded_once_per_model(self):
        pytest.importorskip("sentence_transformers")
        from ragstone.rag.reranker import wrap_with_reranker

        base = SimpleTextRetriever.from_texts(["a"])
        first = wrap_with_reranker(base)
        second = wrap_with_reranker(base)

        assert first.base_compressor.model is second.base_compressor.model
