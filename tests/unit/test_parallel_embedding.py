"""
Unit tests for concurrent batch embedding (no network required).

Order preservation is the property everything depends on: FAISS pairs
vectors with texts positionally, so a single out-of-order batch would
silently attach every answer to the wrong chunks.
"""

import threading
import time

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragstone.rag.embeddings import embed_texts_parallel


class _JitteredFakeEmbeddings(Embeddings):
    """Deterministic vectors, adversarial timing.

    Earlier batches sleep LONGER, forcing completion order to invert
    submission order — exactly the scenario that corrupts a naive
    implementation.
    """

    def __init__(self):
        self.calls = 0
        self.threads = set()
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        with self._lock:
            self.calls += 1
            call_number = self.calls
            self.threads.add(threading.get_ident())
        time.sleep(0.05 / call_number)  # first submitted finishes last
        return [[float(len(t)), float(ord(t[0]))] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), float(ord(text[0]))]


def _expected(texts):
    return [[float(len(t)), float(ord(t[0]))] for t in texts]


class TestEmbedTextsParallel:
    def test_order_preserved_under_adversarial_completion_order(self):
        embeddings = _JitteredFakeEmbeddings()
        texts = [f"{chr(97 + i)}-text-{i}" for i in range(20)]

        vectors = embed_texts_parallel(embeddings, texts, batch_size=3, max_workers=4)

        assert vectors == _expected(texts)
        assert embeddings.calls == 7  # ceil(20 / 3) batches
        assert len(embeddings.threads) > 1  # actually ran concurrently

    def test_single_batch_uses_plain_path(self):
        embeddings = _JitteredFakeEmbeddings()
        texts = ["a", "b"]
        vectors = embed_texts_parallel(embeddings, texts, batch_size=10, max_workers=4)
        assert vectors == _expected(texts)
        assert embeddings.calls == 1

    def test_single_worker_stays_sequential(self):
        embeddings = _JitteredFakeEmbeddings()
        texts = ["a", "b", "c", "d"]
        vectors = embed_texts_parallel(embeddings, texts, batch_size=1, max_workers=1)
        assert vectors == _expected(texts)
        assert len(embeddings.threads) == 1

    def test_batch_failure_fails_the_whole_operation(self):
        class _FailingEmbeddings(_JitteredFakeEmbeddings):
            def embed_documents(self, texts):
                if any("poison" in t for t in texts):
                    raise RuntimeError("api rejected batch")
                return super().embed_documents(texts)

        texts = ["ok-1", "ok-2", "poison", "ok-3"]
        with pytest.raises(RuntimeError, match="api rejected batch"):
            embed_texts_parallel(
                _FailingEmbeddings(), texts, batch_size=1, max_workers=4
            )


class TestFaissParallelIngestion:
    def test_create_db_produces_searchable_index_with_metadata(self, monkeypatch):
        pytest.importorskip("faiss")
        from ragstone.config.settings import get_config
        from ragstone.rag.vector_db import FaissProxy

        # Force multiple concurrent batches even for a small corpus.
        monkeypatch.setattr(get_config().database, "batch_size", 2)
        monkeypatch.setattr(get_config().database, "embed_workers", 3)

        docs = [
            Document(page_content=f"document number {i}", metadata={"source": f"s{i}"})
            for i in range(7)
        ]
        proxy = FaissProxy()
        proxy.create_db(docs=docs, embeddings=_JitteredFakeEmbeddings())

        results = proxy.db.similarity_search("document number 3", k=7)
        assert len(results) == 7
        # Text↔metadata pairing survived the concurrent assembly.
        by_content = {d.page_content: d.metadata["source"] for d in results}
        assert by_content["document number 5"] == "s5"
