"""Server-backed vector stores: Qdrant and pgvector (ROADMAP 4.1).

Both implement the same `VectorStoreProxy` ABC as FAISS/Chroma, so the
pipeline, ensemble retriever, chains, and eval harness are untouched —
`VECTOR_STORE_TYPE=qdrant|pgvector` is the whole switch. Retrieval
quality must be identical to FAISS at equal k (Experiment 13 gates it);
what these buy is durability, metadata filtering, and a path to
multi-process access.

Qdrant runs **embedded by default** (`QDRANT_PATH`, no server, no Docker
— fully testable offline); set `QDRANT_URL` to talk to a real server
with the identical code path. pgvector always needs a Postgres with the
pgvector extension (`RAGSTONE_PG_URL`; docker-compose.yml provides one).

Collection naming follows the Chroma lesson: unique per proxy instance
by default so pipelines can never mix or append into each other's
corpora; set `RAGSTONE_COLLECTION` to pin a stable name when one
deployment owns the store and wants the index to survive restarts.

Both extras are optional installs:

    pip install "ragstone[qdrant]"      # qdrant-client + langchain-qdrant
    pip install "ragstone[pgvector]"    # langchain-postgres + psycopg
"""

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

from ..config.settings import get_config
from ..utils.exceptions import (
    VectorStoreInitializationError,
    VectorStoreOperationError,
)
from .embeddings import embed_texts_cached, make_openai_embeddings
from .vector_db import VectorStoreProxy

logger = logging.getLogger(__name__)


def _default_embeddings():
    """OpenAI embeddings from config — the same fallback FaissProxy uses."""
    return make_openai_embeddings()


def _collection_name(prefix: str) -> str:
    """Configured stable name, or a unique per-instance one (isolation)."""
    configured = get_config().database.collection_name
    return configured or f"{prefix}_{uuid.uuid4().hex[:8]}"


# Embedded Qdrant clients are shared per path and live for the process:
# local mode holds an EXCLUSIVE filesystem lock on the storage folder, so
# a client per proxy would crash the SECOND pipeline on the same path.
# Collections stay isolated per proxy; only the client handle is shared.
_local_clients: Dict[str, Any] = {}
_local_clients_lock = threading.Lock()


def _shared_local_client(path: str):
    from qdrant_client import QdrantClient

    with _local_clients_lock:
        if path not in _local_clients:
            _local_clients[path] = QdrantClient(path=path)
        return _local_clients[path]


class QdrantProxy(VectorStoreProxy):
    """Qdrant vector store: embedded local mode, or a server via QDRANT_URL.

    Ingestion reuses the cached parallel embedder (precomputed vectors are
    upserted through the raw client), then wraps the collection in
    LangChain's `QdrantVectorStore` for the retriever interface.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        db_cfg = get_config().database
        self._path = path or db_cfg.qdrant_path
        self._url = url or db_cfg.qdrant_url
        self._collection = collection_name or _collection_name("ragstone")
        # Pin state is captured NOW: cleanup() must not re-read config,
        # or a config reload between init and cleanup could drop a pinned
        # deployment index (or leak an unpinned scratch collection).
        self._owns_collection = (
            collection_name is None and db_cfg.collection_name is None
        )
        self._client: Optional[Any] = None
        self._db: Optional[Any] = None
        self._is_initialized = False
        mode = f"server {self._url}" if self._url else f"embedded {self._path}"
        logger.info(f"QdrantProxy initialized ({mode})")

    @property
    def type(self) -> str:
        return "Qdrant"

    @property
    def db(self):
        return self._db

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized and self._db is not None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ImportError(
                    "The Qdrant backend requires the 'qdrant' extra. "
                    "Install it with: pip install 'ragstone[qdrant]'"
                ) from exc
            if self._url:
                self._client = QdrantClient(
                    url=self._url, api_key=os.getenv("QDRANT_API_KEY")
                )
            else:
                # Embedded mode: persists to disk, no server process. The
                # client is SHARED per path (exclusive folder lock) and
                # lives for the process — cleanup() must not close it.
                self._client = _shared_local_client(self._path)
        return self._client

    def create_db(
        self, docs: List, embeddings: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """Embed documents concurrently and (re)build the collection.

        Rebuilding REPLACES the collection — appending into a previous
        run's data silently mixes corpora (the Chroma lesson).
        """
        try:
            self._validate_documents(docs)
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import models

            if embeddings is None:
                embeddings = _default_embeddings()

            db_cfg = get_config().database
            texts = [doc.page_content for doc in docs]
            vectors = embed_texts_cached(
                embeddings,
                texts,
                batch_size=db_cfg.batch_size,
                max_workers=db_cfg.embed_workers,
            )

            client = self._get_client()
            if client.collection_exists(self._collection):
                client.delete_collection(self._collection)
            client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    # Cosine: OpenAI embeddings are unit-normalized, so
                    # rankings match FAISS's L2 ordering exactly.
                    distance=models.Distance.COSINE,
                ),
            )
            points = [
                models.PointStruct(
                    id=i,
                    vector=vector,
                    payload={"page_content": text, "metadata": doc.metadata},
                )
                for i, (text, vector, doc) in enumerate(zip(texts, vectors, docs))
            ]
            client.upsert(collection_name=self._collection, points=points)

            self._db = QdrantVectorStore(
                client=client,
                collection_name=self._collection,
                embedding=embeddings,
                content_payload_key="page_content",
                metadata_payload_key="metadata",
            )
            self._is_initialized = True
            logger.info(
                f"Qdrant collection '{self._collection}' built with "
                f"{len(docs)} documents"
            )
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}", exc_info=True)
            raise VectorStoreInitializationError(
                f"Qdrant database creation failed: {e}"
            ) from e

    def find_similar(self, query: str, k: int = 4, **kwargs: Any) -> List:
        self._validate_query(query)
        db = self._db
        if db is None or not self._is_initialized:
            raise VectorStoreOperationError("Qdrant store is not initialized")
        try:
            return db.similarity_search(query, k=k)
        except Exception as e:
            raise VectorStoreOperationError(f"Qdrant search failed: {e}") from e

    def cleanup(self) -> None:
        """Drop this proxy's collection (if it owns it) and detach.

        Pinned (configured) collection names are preserved — they are the
        deployment's persistent index, not this instance's scratch space.
        Embedded-mode clients are shared per path and stay open for the
        process (closing would yank the handle from other pipelines);
        server-mode clients are per-proxy and are closed here.
        """
        try:
            if self._client is not None:
                if self._owns_collection and self._client.collection_exists(
                    self._collection
                ):
                    self._client.delete_collection(self._collection)
                if self._url:  # per-proxy server client; local ones are shared
                    self._client.close()
                self._client = None
            self._db = None
            self._is_initialized = False
            logger.info("QdrantProxy cleanup completed")
        except Exception as e:
            logger.error(f"Error during QdrantProxy cleanup: {e}")


class PgVectorProxy(VectorStoreProxy):
    """pgvector store: RAG on the Postgres you already run.

    Requires a reachable Postgres with the pgvector extension
    (`RAGSTONE_PG_URL`, e.g.
    postgresql+psycopg://ragstone:ragstone@localhost:5432/ragstone —
    docker-compose.yml provides one). Ingestion reuses
    the cached parallel embedder via PGVector.add_embeddings.
    """

    def __init__(
        self,
        connection: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        db_cfg = get_config().database
        self._connection = connection or db_cfg.pg_url
        if not self._connection:
            raise VectorStoreInitializationError(
                "pgvector requires a connection string: set RAGSTONE_PG_URL"
            )
        self._collection = collection_name or _collection_name("ragstone")
        # Captured at init for the same reason as QdrantProxy: cleanup()
        # must not change its drop/preserve decision on a config reload.
        self._owns_collection = (
            collection_name is None and db_cfg.collection_name is None
        )
        self._db: Optional[Any] = None
        self._is_initialized = False
        logger.info(f"PgVectorProxy initialized (collection {self._collection})")

    @property
    def type(self) -> str:
        return "PGVector"

    @property
    def db(self):
        return self._db

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized and self._db is not None

    def create_db(
        self, docs: List, embeddings: Optional[Any] = None, **kwargs: Any
    ) -> None:
        """Embed documents concurrently and (re)build the collection.

        pre_delete_collection ensures rebuilds replace rather than append
        (the Chroma lesson).
        """
        try:
            self._validate_documents(docs)
            try:
                from langchain_postgres import PGVector
            except ImportError as exc:
                raise ImportError(
                    "The pgvector backend requires the 'pgvector' extra. "
                    "Install it with: pip install 'ragstone[pgvector]'"
                ) from exc

            if embeddings is None:
                embeddings = _default_embeddings()

            db_cfg = get_config().database
            texts = [doc.page_content for doc in docs]
            vectors = embed_texts_cached(
                embeddings,
                texts,
                batch_size=db_cfg.batch_size,
                max_workers=db_cfg.embed_workers,
            )

            self._db = PGVector(
                embeddings=embeddings,
                connection=self._connection,
                collection_name=self._collection,
                use_jsonb=True,
                pre_delete_collection=True,
            )
            self._db.add_embeddings(
                texts=texts,
                embeddings=vectors,
                metadatas=[doc.metadata for doc in docs],
            )
            self._is_initialized = True
            logger.info(
                f"pgvector collection '{self._collection}' built with "
                f"{len(docs)} documents"
            )
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Failed to create pgvector collection: {e}", exc_info=True)
            raise VectorStoreInitializationError(
                f"pgvector database creation failed: {e}"
            ) from e

    def find_similar(self, query: str, k: int = 4, **kwargs: Any) -> List:
        self._validate_query(query)
        db = self._db
        if db is None or not self._is_initialized:
            raise VectorStoreOperationError("pgvector store is not initialized")
        try:
            return db.similarity_search(query, k=k)
        except Exception as e:
            raise VectorStoreOperationError(f"pgvector search failed: {e}") from e

    def cleanup(self) -> None:
        """Drop this proxy's collection (unless pinned) and release the store."""
        try:
            if self._db is not None and self._owns_collection:
                self._db.delete_collection()
            self._db = None
            self._is_initialized = False
            logger.info("PgVectorProxy cleanup completed")
        except Exception as e:
            logger.error(f"Error during PgVectorProxy cleanup: {e}")
