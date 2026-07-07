"""
Unit tests for the content-addressed embedding cache (no network).

The cache's contract: unchanged chunks never reach the embedding API,
vectors round-trip exactly (a cached index is bit-identical), keys
separate models, order is preserved through mixed hits/misses, and the
off switch restores the uncached path verbatim.
"""

from typing import List

import pytest
from langchain_core.embeddings import Embeddings

from ragstone.config.settings import get_config
from ragstone.rag.embedding_cache import EmbeddingCache, model_id_for
from ragstone.rag.embeddings import embed_texts_cached


class _CountingEmbeddings(Embeddings):
    """Deterministic per-text vectors; counts every embedded text."""

    def __init__(self, model: str = "fake-model"):
        self.model = model
        self.embedded: List[str] = []

    def _vector(self, text: str) -> List[float]:
        seed = float(sum(ord(c) for c in f"{self.model}:{text}"))
        return [seed, seed / 3.0, 0.123456789012345]

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    """Point the process-wide cache at a fresh file for each test."""
    import ragstone.rag.embedding_cache as ec

    path = str(tmp_path / "cache.sqlite")
    monkeypatch.setattr(get_config().database, "embed_cache_enabled", True)
    monkeypatch.setattr(get_config().database, "embed_cache_path", path)
    monkeypatch.setattr(ec, "_cache", None)
    monkeypatch.setattr(ec, "_cache_path", None)
    return path


TEXTS = ["solar warranty is 28 years", "coffee rests 10 days", "thrust is 4.2 MN"]


class TestIncrementalIngest:
    """Only changed chunks may reach the embedding API (Experiment 14)."""

    def test_second_ingest_embeds_nothing(self, cache_path):
        embeddings = _CountingEmbeddings()
        first = embed_texts_cached(embeddings, TEXTS, batch_size=2, max_workers=2)
        assert embeddings.embedded == TEXTS  # cold cache: everything embeds

        again = embed_texts_cached(embeddings, TEXTS, batch_size=2, max_workers=2)
        assert embeddings.embedded == TEXTS  # warm cache: no new API work
        # Bit-identical vectors: the rebuilt index cannot differ.
        assert again == first

    def test_only_the_changed_chunk_is_embedded(self, cache_path):
        embeddings = _CountingEmbeddings()
        embed_texts_cached(embeddings, TEXTS, batch_size=10, max_workers=1)
        embeddings.embedded.clear()

        updated = TEXTS[:2] + ["thrust is now 4.4 MN"]  # one edited doc
        vectors = embed_texts_cached(embeddings, updated, batch_size=10, max_workers=1)

        assert embeddings.embedded == ["thrust is now 4.4 MN"]
        assert len(vectors) == 3

    def test_order_is_preserved_through_mixed_hits_and_misses(self, cache_path):
        embeddings = _CountingEmbeddings()
        embed_texts_cached(embeddings, [TEXTS[1]], batch_size=10, max_workers=1)

        vectors = embed_texts_cached(embeddings, TEXTS, batch_size=10, max_workers=1)
        # Every position must hold ITS text's vector, hit or miss.
        assert vectors == [embeddings._vector(t) for t in TEXTS]

    def test_models_never_share_vectors(self, cache_path):
        a, b = _CountingEmbeddings("model-a"), _CountingEmbeddings("model-b")
        embed_texts_cached(a, TEXTS, batch_size=10, max_workers=1)
        embed_texts_cached(b, TEXTS, batch_size=10, max_workers=1)
        assert b.embedded == TEXTS  # model-b was a full miss

    def test_off_switch_bypasses_the_cache(self, cache_path, monkeypatch):
        monkeypatch.setattr(get_config().database, "embed_cache_enabled", False)
        embeddings = _CountingEmbeddings()
        embed_texts_cached(embeddings, TEXTS, batch_size=10, max_workers=1)
        embed_texts_cached(embeddings, TEXTS, batch_size=10, max_workers=1)
        assert embeddings.embedded == TEXTS * 2  # embedded twice, no cache

    def test_cache_persists_across_instances(self, cache_path):
        embeddings = _CountingEmbeddings()
        EmbeddingCache(cache_path).put_many(
            model_id_for(embeddings), TEXTS, [embeddings._vector(t) for t in TEXTS]
        )
        # A fresh instance over the same file sees the vectors (restart
        # survival — the point of a disk cache).
        fresh = EmbeddingCache(cache_path)
        found = fresh.get_many(model_id_for(embeddings), TEXTS)
        assert sorted(found) == [0, 1, 2]
        assert found[0] == embeddings._vector(TEXTS[0])  # exact round-trip


class TestModelIdentity:
    """Cache keys must separate embedding models."""

    def test_model_id_includes_class_and_model_name(self):
        embeddings = _CountingEmbeddings("text-embedding-3-small")
        assert model_id_for(embeddings) == (
            "_CountingEmbeddings:text-embedding-3-small"
        )
