import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional

from ..config.settings import get_config
from ..utils import (
    VectorStoreInitializationError,
    VectorStoreOperationError,
)

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_CHROMA_PERSIST_DIR = "store/chroma_db"
DEFAULT_FAISS_INDEX_NAME = "faiss_index"
DEFAULT_SIMILARITY_K = 4
MAX_QUERY_LENGTH = 10000  # Prevent excessive query lengths

# Import cache to avoid repeated imports
_import_cache = {}


def _get_cached_import(module_name: str, attr_name: str = None):
    """Get cached import or import and cache it."""
    cache_key = f"{module_name}.{attr_name}" if attr_name else module_name

    if cache_key not in _import_cache:
        try:
            if attr_name:
                module = __import__(module_name, fromlist=[attr_name])
                _import_cache[cache_key] = getattr(module, attr_name)
            else:
                _import_cache[cache_key] = __import__(module_name)
        except ImportError as e:
            logger.error(f"Failed to import {cache_key}: {e}")
            raise

    return _import_cache[cache_key]


def _lazy_import_langchain_docs():
    """Lazy import for LangChain documents."""
    return _get_cached_import("langchain_core.documents", "Document")


def _lazy_import_openai_embeddings():
    """Lazy import for LangChain OpenAI embeddings."""
    return _get_cached_import("langchain_openai", "OpenAIEmbeddings")


def _lazy_import_chroma():
    """Lazy import for ChromaDB."""
    chroma_module = _get_cached_import("langchain_chroma", "Chroma")
    chromadb_module = _get_cached_import("chromadb")
    return chroma_module, chromadb_module


def _lazy_import_faiss():
    """Lazy import for FAISS."""
    return _get_cached_import("langchain_community.vectorstores", "FAISS")


def _lazy_import_numpy():
    """Lazy import for numpy."""
    return _get_cached_import("numpy")


class VectorStoreProxy(ABC):
    """
    Abstract base class for vector store proxies with enhanced interface.
    """

    @property
    @abstractmethod
    def type(self) -> str:
        """Get the vector store type identifier."""
        pass

    @property
    @abstractmethod
    def db(self):
        """Get the underlying database instance."""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        pass

    @abstractmethod
    def create_db(self, docs: List, embeddings: Optional = None, **kwargs) -> None:
        """Create the vector database from documents."""
        pass

    @abstractmethod
    def find_similar(self, query: str, k: int = DEFAULT_SIMILARITY_K, **kwargs) -> List:
        """Find similar documents to the query."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources and connections."""
        pass

    def _validate_query(self, query: str) -> None:
        """
        Validate search query.

        Args:
            query: Search query string.

        Raises:
            ValueError: If query is invalid.
        """
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long. Maximum length: {MAX_QUERY_LENGTH}")

    def _validate_documents(self, docs: List) -> None:
        """
        Validate document list.

        Args:
            docs: List of documents to validate.

        Raises:
            ValueError: If documents are invalid.
        """
        if not isinstance(docs, list):
            raise ValueError("Documents must be provided as a list")

        if not docs:
            raise ValueError("Document list cannot be empty")

        # Import Document class only when needed for validation
        Document = _lazy_import_langchain_docs()

        for i, doc in enumerate(docs):
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Document at index {i} is not a valid Document instance"
                )

            if not hasattr(doc, "page_content") or not doc.page_content:
                raise ValueError(
                    f"Document at index {i} has empty or missing page_content"
                )


class ChromaProxy(VectorStoreProxy):
    """
    Proxy for ChromaDB vector store with lazy loading and connection management.
    """

    def __init__(self, persist_directory: str = DEFAULT_CHROMA_PERSIST_DIR) -> None:
        """Initialize ChromaProxy with lazy loading."""
        self.persist_directory = persist_directory
        self._db = None
        self._client = None
        self._is_initialized = False

        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(
            f"ChromaProxy initialized with persist_directory: {persist_directory}"
        )

    @property
    def type(self) -> str:
        """Get the vector store type identifier."""
        return "Chroma"

    @property
    def db(self):
        """Get the underlying Chroma database instance."""
        return self._db

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self._is_initialized and self._db is not None

    @contextmanager
    def _ensure_client(self):
        """Context manager to ensure ChromaDB client is available."""
        if self._client is None:
            Chroma, chromadb = _lazy_import_chroma()
            self._client = chromadb.PersistentClient(path=self.persist_directory)

        try:
            yield self._client
        except Exception as e:
            logger.error(f"ChromaDB client error: {e}")
            raise

    def create_db(
        self,
        docs: List,
        collection_name: str = "default_chroma_collection",
        embeddings: Optional = None,
    ) -> None:
        """
        Create ChromaDB database from documents with lazy loading.

        Args:
            docs: List of Document objects to add to the database.
            collection_name: Name for the Chroma collection.
            embeddings: Embeddings instance to use.

        Raises:
            VectorStoreInitializationError: If database creation fails.
        """
        try:
            # Validate inputs
            self._validate_documents(docs)

            # Lazy import
            Chroma, chromadb = _lazy_import_chroma()

            # Initialize embeddings if not provided
            if embeddings is None:
                try:
                    logger.info("Using standard LangChain OpenAI embeddings for Chroma")
                    OpenAIEmbeddings = _lazy_import_openai_embeddings()
                    # Use clean configuration to avoid organization issues
                    embeddings = OpenAIEmbeddings(
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        openai_organization=None,
                        model=get_config().llm.openai_embedding_model,
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize default embeddings: {e}")
                    raise VectorStoreInitializationError(
                        f"Failed to initialize default embeddings: {e}"
                    ) from e

            # Use context manager for client
            with self._ensure_client() as client:
                try:
                    # Reuse our PersistentClient instead of passing
                    # persist_directory — otherwise LangChain opens a second
                    # client on the same sqlite directory.
                    self._db = Chroma.from_documents(
                        documents=docs,
                        embedding=embeddings,
                        collection_name=collection_name,
                        client=client,
                    )

                    self._is_initialized = True
                    logger.info(
                        f"Successfully created ChromaDB with {len(docs)} documents in collection '{collection_name}'"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to create Chroma database: {e}", exc_info=True
                    )
                    raise VectorStoreInitializationError(
                        f"ChromaDB creation failed: {e}"
                    ) from e

        except Exception as e:
            logger.error(f"ChromaProxy.create_db failed: {e}", exc_info=True)
            self._is_initialized = False
            raise

    def find_similar(
        self,
        query: str,
        k: int = DEFAULT_SIMILARITY_K,
        collection_name: Optional[str] = None,
    ) -> List:
        """
        Find similar documents to the query.

        Args:
            query: Search query string.
            k: Number of similar documents to return.
            collection_name: Optional collection name to search in.

        Returns:
            List of similar Document objects.

        Raises:
            VectorStoreOperationError: If similarity search fails.
        """
        try:
            # Validate query
            self._validate_query(query)

            if not self.is_initialized:
                raise VectorStoreOperationError(
                    "ChromaDB not initialized. Call create_db() first."
                )

            # Perform similarity search
            results = self._db.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query")
            return results

        except Exception as e:
            logger.error(f"ChromaDB similarity search failed: {e}", exc_info=True)
            raise VectorStoreOperationError(f"Similarity search failed: {e}") from e

    def get_collection(self, collection_name: str):
        """
        Get a specific collection from ChromaDB.

        Args:
            collection_name: Name of the collection to retrieve.

        Returns:
            Collection object or None if not found.
        """
        try:
            with self._ensure_client() as client:
                try:
                    return client.get_collection(name=collection_name)
                except Exception:
                    logger.warning(f"Collection '{collection_name}' not found")
                    return None
        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}': {e}")
            return None

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from ChromaDB.

        Args:
            collection_name: Name of the collection to delete.
        """
        try:
            with self._ensure_client() as client:
                try:
                    client.delete_collection(name=collection_name)
                    logger.info(f"Successfully deleted collection '{collection_name}'")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete collection '{collection_name}': {e}"
                    )

                # Reset state if we deleted the current collection
                if (
                    hasattr(self._db, "_collection")
                    and self._db._collection.name == collection_name
                ):
                    self._db = None
                    self._is_initialized = False

        except Exception as e:
            logger.error(f"Error during collection deletion: {e}")

    def cleanup(self) -> None:
        """Clean up ChromaDB resources."""
        try:
            if self._client:
                # ChromaDB client cleanup
                self._client = None
                logger.info("ChromaDB client cleaned up")

            self._db = None
            self._is_initialized = False
            logger.info("ChromaProxy cleanup completed")

        except Exception as e:
            logger.error(f"Error during ChromaProxy cleanup: {e}")


class FaissProxy(VectorStoreProxy):
    """
    Proxy for FAISS vector store with lazy loading and optimizations.
    """

    def __init__(self) -> None:
        """Initialize FaissProxy with lazy loading."""
        self._db = None
        self._is_initialized = False
        logger.info("FaissProxy initialized")

    @property
    def type(self) -> str:
        """Get the vector store type identifier."""
        return "FAISS"

    @property
    def db(self):
        """Get the underlying FAISS database instance."""
        return self._db

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self._is_initialized and self._db is not None

    def create_db(self, docs: List, embeddings: Optional = None, **kwargs) -> None:
        """
        Create FAISS database from documents with lazy loading.

        Args:
            docs: List of Document objects to add to the database.
            embeddings: Embeddings instance to use.
            **kwargs: Additional arguments for FAISS.

        Raises:
            VectorStoreInitializationError: If database creation fails.
        """
        try:
            # Validate inputs
            self._validate_documents(docs)

            # Lazy import FAISS
            FAISS = _lazy_import_faiss()

            # Initialize embeddings if not provided
            if embeddings is None:
                try:
                    logger.info("Using standard LangChain OpenAI embeddings for FAISS")
                    OpenAIEmbeddings = _lazy_import_openai_embeddings()
                    # Use clean configuration to avoid organization issues
                    embeddings = OpenAIEmbeddings(
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        openai_organization=None,
                        model=get_config().llm.openai_embedding_model,
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize default embeddings: {e}")
                    raise VectorStoreInitializationError(
                        f"Failed to initialize default embeddings: {e}"
                    ) from e

            # Create FAISS vector store with progress logging
            logger.info(f"Creating FAISS vector store from {len(docs)} documents...")
            self._db = FAISS.from_documents(
                documents=docs, embedding=embeddings, **kwargs
            )
            self._is_initialized = True
            logger.info(
                f"Successfully created FAISS database with {len(docs)} documents"
            )

        except Exception as e:
            logger.error(f"FaissProxy.create_db failed: {e}", exc_info=True)
            self._is_initialized = False
            raise VectorStoreInitializationError(f"FAISS creation failed: {e}") from e

    def find_similar(
        self, query: str, k: int = DEFAULT_SIMILARITY_K, use_mmr: bool = False, **kwargs
    ) -> List:
        """
        Find similar documents to the query using FAISS.

        Args:
            query: Search query string.
            k: Number of similar documents to return.
            use_mmr: Whether to use Maximum Marginal Relevance for diversity.
            **kwargs: Additional arguments for similarity search.

        Returns:
            List of similar Document objects.

        Raises:
            VectorStoreOperationError: If similarity search fails.
        """
        try:
            # Validate query
            self._validate_query(query)

            if not self.is_initialized:
                raise VectorStoreOperationError(
                    "FAISS database not initialized. Call create_db() first."
                )

            # Choose search method based on use_mmr flag
            if use_mmr:
                results = self._db.max_marginal_relevance_search(query, k=k, **kwargs)
                logger.info(f"Found {len(results)} diverse documents using MMR")
            else:
                results = self._db.similarity_search(query, k=k, **kwargs)
                logger.info(
                    f"Found {len(results)} similar documents using standard search"
                )

            return results

        except Exception as e:
            logger.error(f"FAISS similarity search failed: {e}", exc_info=True)
            raise VectorStoreOperationError(f"Similarity search failed: {e}") from e

    def save_local(
        self, folder_path: str, index_name: str = DEFAULT_FAISS_INDEX_NAME
    ) -> None:
        """
        Save FAISS index to local storage.

        Args:
            folder_path: Directory to save the index.
            index_name: Name for the index files.

        Raises:
            VectorStoreOperationError: If save operation fails.
        """
        try:
            if not self.is_initialized:
                raise VectorStoreOperationError(
                    "FAISS database not initialized. Cannot save."
                )

            # Ensure directory exists
            os.makedirs(folder_path, exist_ok=True)

            # Save FAISS index
            self._db.save_local(folder_path, index_name)
            logger.info(f"Successfully saved FAISS index to {folder_path}/{index_name}")

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}", exc_info=True)
            raise VectorStoreOperationError(f"Save operation failed: {e}") from e

    def load_local(
        self,
        folder_path: str,
        embeddings: Optional = None,
        index_name: str = DEFAULT_FAISS_INDEX_NAME,
        allow_dangerous_deserialization: bool = False,
    ) -> None:
        """
        Load FAISS index from local storage with lazy imports.

        Args:
            folder_path: Directory containing the index.
            embeddings: Embeddings instance to use.
            index_name: Name of the index files.
            allow_dangerous_deserialization: Whether to allow pickle deserialization.

        Raises:
            VectorStoreOperationError: If load operation fails.
        """
        try:
            # Check if index files exist
            index_path = os.path.join(folder_path, f"{index_name}.faiss")
            pkl_path = os.path.join(folder_path, f"{index_name}.pkl")

            if not os.path.exists(index_path) or not os.path.exists(pkl_path):
                raise VectorStoreOperationError(
                    f"FAISS index files not found in {folder_path}"
                )

            # Lazy import FAISS
            FAISS = _lazy_import_faiss()

            # Initialize embeddings if not provided
            if embeddings is None:
                try:
                    logger.warning(
                        "Using standard LangChain OpenAI embeddings for loading FAISS. This MUST match the embeddings used at save time."
                    )
                    OpenAIEmbeddings = _lazy_import_openai_embeddings()
                    # Use clean configuration to avoid organization issues
                    embeddings = OpenAIEmbeddings(
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        openai_organization=None,
                        model=get_config().llm.openai_embedding_model,
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize default embeddings: {e}")
                    raise VectorStoreOperationError(
                        f"Failed to initialize default embeddings: {e}"
                    ) from e

            # Load FAISS index
            self._db = FAISS.load_local(
                folder_path,
                embeddings,
                index_name,
                allow_dangerous_deserialization=allow_dangerous_deserialization,
            )
            self._is_initialized = True
            logger.info(
                f"Successfully loaded FAISS index from {folder_path}/{index_name}"
            )

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
            self._is_initialized = False
            raise VectorStoreOperationError(f"Load operation failed: {e}") from e

    def cleanup(self) -> None:
        """Clean up FAISS resources."""
        try:
            self._db = None
            self._is_initialized = False
            logger.info("FaissProxy cleanup completed")

        except Exception as e:
            logger.error(f"Error during FaissProxy cleanup: {e}")


def create_vector_store_proxy(store_type: str, **kwargs) -> VectorStoreProxy:
    """
    Factory function to create vector store proxies with lazy loading.

    Args:
        store_type: Type of vector store ("chroma" or "faiss").
        **kwargs: Additional arguments for the vector store.

    Returns:
        VectorStoreProxy: Configured vector store proxy.

    Raises:
        ValueError: If store_type is unsupported.
    """
    store_type_lower = store_type.lower()

    if store_type_lower == "chroma":
        return ChromaProxy(**kwargs)
    elif store_type_lower == "faiss":
        return FaissProxy(**kwargs)
    else:
        supported_types = ["chroma", "faiss"]
        raise ValueError(
            f"Unsupported vector store type: {store_type}. Supported types: {supported_types}"
        )
