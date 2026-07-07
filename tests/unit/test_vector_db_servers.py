"""
Unit tests for the server-backed vector stores (ROADMAP 4.1).

Qdrant runs in embedded local mode against tmp_path — full coverage with
no server, no Docker, no network. pgvector cannot be embedded, so its
tests run only when RAGSTONE_PG_URL points at a live Postgres (the
docker-compose.yml service, or CI's service container) and skip cleanly
otherwise. Both suites use deterministic fake embeddings so nearest-
neighbor results are exact, and both include the load-bearing assertion:
PARITY — the same corpus and embeddings must rank identically to FAISS.
"""

import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragstone.rag.vector_db import FaissProxy, create_vector_store_proxy

pytest.importorskip("qdrant_client")
pytest.importorskip("langchain_qdrant")

from ragstone.rag.vector_db_servers import PgVectorProxy, QdrantProxy  # noqa: E402


class _KeywordEmbeddings(Embeddings):
    """Deterministic embeddings: each known keyword owns one axis.

    Similarity is exact and inspectable — a query about "solar" must
    retrieve the solar document first, on every backend. Off-axis
    components are small and DISTINCT per axis, so no two similarities
    ever tie: ties would let backends legitimately order results
    differently and turn the parity test into a coin flip.
    """

    AXES = ["solar", "coffee", "rocket", "transit"]

    def _vector(self, text):
        text = text.lower()
        vec = [
            1.0 if axis in text else 0.01 * (i + 1) for i, axis in enumerate(self.AXES)
        ]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


DOCS = [
    Document(page_content="solar panel warranty is 28 years", metadata={"source": "a"}),
    Document(page_content="coffee is roasted for 12 minutes", metadata={"source": "b"}),
    Document(
        page_content="rocket thrust reaches 4.2 meganewtons", metadata={"source": "c"}
    ),
]


@pytest.fixture
def qdrant_proxy(tmp_path):
    """A proxy that always releases its embedded-mode file lock."""
    proxy = QdrantProxy(path=str(tmp_path / "qdrant"))
    yield proxy
    proxy.cleanup()


class TestQdrantProxy:
    def test_create_and_search_returns_nearest_document(self, qdrant_proxy):
        qdrant_proxy.create_db(docs=DOCS, embeddings=_KeywordEmbeddings())
        assert qdrant_proxy.is_initialized

        [top] = qdrant_proxy.find_similar("how long is the solar warranty?", k=1)
        assert "28 years" in top.page_content
        assert top.metadata["source"] == "a"  # metadata round-trips

    def test_recreate_replaces_not_appends(self, qdrant_proxy):
        # The Chroma lesson: rebuilding must never mix with the old corpus.
        embeddings = _KeywordEmbeddings()
        qdrant_proxy.create_db(docs=DOCS, embeddings=embeddings)
        qdrant_proxy.create_db(
            docs=[Document(page_content="only solar remains", metadata={})],
            embeddings=embeddings,
        )
        results = qdrant_proxy.find_similar("solar", k=4)
        assert [d.page_content for d in results] == ["only solar remains"]

    def test_retriever_interface_works_for_the_ensemble(self, qdrant_proxy):
        # The pipeline consumes .db.as_retriever(); that contract is what
        # keeps every chain backend-agnostic.
        qdrant_proxy.create_db(docs=DOCS, embeddings=_KeywordEmbeddings())
        retriever = qdrant_proxy.db.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("rocket thrust")
        assert "4.2 meganewtons" in results[0].page_content

    def test_cleanup_releases_the_local_file_lock(self, tmp_path):
        # Embedded mode locks QDRANT_PATH; a second proxy over the same
        # path must work after the first closes.
        path = str(tmp_path / "qdrant")
        first = QdrantProxy(path=path)
        first.create_db(docs=DOCS, embeddings=_KeywordEmbeddings())
        first.cleanup()

        second = QdrantProxy(path=path)
        second.create_db(docs=DOCS, embeddings=_KeywordEmbeddings())
        assert second.is_initialized
        second.cleanup()

    def test_factory_dispatches_qdrant(self, tmp_path, monkeypatch):
        from ragstone.config.settings import get_config

        monkeypatch.setattr(get_config().database, "qdrant_path", str(tmp_path / "q"))
        proxy = create_vector_store_proxy("qdrant")
        assert proxy.type == "Qdrant"

    def test_parity_with_faiss(self, qdrant_proxy):
        """The acceptance criterion: identical ranking to FAISS at equal k."""
        embeddings = _KeywordEmbeddings()

        faiss = FaissProxy()
        faiss.create_db(docs=DOCS, embeddings=embeddings)
        qdrant_proxy.create_db(docs=DOCS, embeddings=embeddings)

        for query in ["solar warranty", "roasted coffee", "rocket", "unrelated"]:
            faiss_top = [d.page_content for d in faiss.find_similar(query, k=3)]
            qdrant_top = [d.page_content for d in qdrant_proxy.find_similar(query, k=3)]
            assert qdrant_top == faiss_top, f"ranking diverged for {query!r}"


PG_URL = os.getenv("RAGSTONE_PG_URL")


@pytest.mark.skipif(
    not PG_URL, reason="RAGSTONE_PG_URL not set (start the compose postgres)"
)
class TestPgVectorProxy:
    def test_create_search_parity_and_replace(self):
        embeddings = _KeywordEmbeddings()
        proxy = PgVectorProxy(connection=PG_URL)
        try:
            proxy.create_db(docs=DOCS, embeddings=embeddings)
            assert proxy.is_initialized

            [top] = proxy.find_similar("how long is the solar warranty?", k=1)
            assert "28 years" in top.page_content
            assert top.metadata["source"] == "a"

            # Parity with FAISS on identical inputs.
            faiss = FaissProxy()
            faiss.create_db(docs=DOCS, embeddings=embeddings)
            for query in ["solar warranty", "roasted coffee", "rocket"]:
                faiss_top = [d.page_content for d in faiss.find_similar(query, k=3)]
                pg_top = [d.page_content for d in proxy.find_similar(query, k=3)]
                assert pg_top == faiss_top, f"ranking diverged for {query!r}"

            # Rebuild replaces, never appends.
            proxy.create_db(
                docs=[Document(page_content="only solar remains", metadata={})],
                embeddings=embeddings,
            )
            results = proxy.find_similar("solar", k=4)
            assert [d.page_content for d in results] == ["only solar remains"]
        finally:
            proxy.cleanup()

    def test_missing_connection_is_a_clear_error(self, monkeypatch):
        from ragstone.config.settings import get_config
        from ragstone.utils.exceptions import VectorStoreInitializationError

        monkeypatch.setattr(get_config().database, "pg_url", None)
        with pytest.raises(VectorStoreInitializationError, match="RAGSTONE_PG_URL"):
            PgVectorProxy()
