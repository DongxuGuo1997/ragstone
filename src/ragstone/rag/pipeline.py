import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from ..config.settings import get_config
from ..models.base_model import OllamaProxy, OpenAIProxy
from ..utils.exceptions import ChainExecutionError, ChainInitializationError
from ..utils.full_chain import FullChain
from .memory import MemoryProxy
from .rag import RagProxy
from .splitter import split_documents

# Configure logging
logger = logging.getLogger(__name__)

# Import cache for heavy LangChain dependencies
_pipeline_cache = {}


@dataclass
class CacheStats:
    """Counters for the response cache."""

    hits: int = 0
    misses: int = 0

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_queries
        return self.hits / total if total > 0 else 0.0


class QueryResultCache:
    """Exact-match response cache with TTL and LRU eviction.

    Entries are keyed by (question, context_hash) so answers never leak
    across sessions or document sets. Deliberately simple: earlier versions
    had normalized and semantic-similarity tiers, but those could return a
    cached answer for a materially different question, and the semantic
    tier spent embedding API calls just to probe the cache. Exact matching
    is cheap, thread-safe, and can't be wrong.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[Tuple[str, str], Tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    @staticmethod
    def _key(question: str, context_hash: str = "") -> Tuple[str, str]:
        return question.strip(), context_hash or ""

    def get_response(self, question: str, context_hash: str = "") -> Optional[str]:
        """Return the cached response for this exact question, or None."""
        key = self._key(question, context_hash)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            timestamp, response = entry
            if time.time() - timestamp > self.ttl_seconds:
                del self._entries[key]
                self.stats.misses += 1
                return None
            self._entries.move_to_end(key)  # refresh LRU position
            self.stats.hits += 1
            return response

    def cache_response(
        self, question: str, response: str, context_hash: str = ""
    ) -> None:
        """Store a response, evicting least-recently-used entries if full."""
        key = self._key(question, context_hash)
        with self._lock:
            self._entries[key] = (time.time(), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache size and hit/miss counters."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": f"{self.stats.hit_rate:.1%}",
            }

    def clear_cache(self) -> None:
        """Drop all cached responses."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        if count:
            logger.info(f"Cleared {count} cached responses")


# Global cache instance
_query_cache = None
_query_cache_lock = threading.Lock()


def _get_query_cache() -> QueryResultCache:
    """Get or create the global response cache using configuration."""
    global _query_cache
    if _query_cache is None:
        with _query_cache_lock:
            if _query_cache is None:
                from ..config.settings import get_config

                config = get_config()
                _query_cache = QueryResultCache(
                    max_size=config.cache.response_cache_size,
                    ttl_seconds=config.cache.response_cache_ttl,
                )
    return _query_cache


def _is_response_cache_enabled() -> bool:
    """Check if the response cache is enabled in configuration."""
    from ..config.settings import get_config

    return get_config().cache.enable_response_cache


def _get_cached_pipeline_import(import_type: str):
    """Get cached pipeline import or import and cache it."""
    if import_type not in _pipeline_cache:
        try:
            if import_type == "document":
                from langchain_core.documents import Document

                _pipeline_cache[import_type] = Document
            elif import_type == "ensemble_retriever":
                from langchain_classic.retrievers import EnsembleRetriever

                _pipeline_cache[import_type] = EnsembleRetriever
            elif import_type == "bm25_retriever":
                from langchain_community.retrievers import BM25Retriever

                _pipeline_cache[import_type] = BM25Retriever
            elif import_type == "ollama_embeddings":
                from langchain_ollama import OllamaEmbeddings

                _pipeline_cache[import_type] = OllamaEmbeddings
            elif import_type == "openai_embeddings":
                from langchain_openai import OpenAIEmbeddings

                _pipeline_cache[import_type] = OpenAIEmbeddings
            elif import_type == "local_loader":
                from .loader import LocalLoader

                _pipeline_cache[import_type] = LocalLoader
            elif import_type == "optimized_local_loader":
                from .loader import OptimizedLocalLoader

                _pipeline_cache[import_type] = OptimizedLocalLoader
            elif import_type == "remote_loader":
                from .loader import RemoteLoader

                _pipeline_cache[import_type] = RemoteLoader
            elif import_type == "vector_db_proxy":
                from .vector_db import create_vector_store_proxy

                _pipeline_cache[import_type] = create_vector_store_proxy
            else:
                raise ValueError(f"Unknown import type: {import_type}")
        except ImportError as e:
            logger.error(f"Failed to import {import_type}: {e}")
            raise

    return _pipeline_cache[import_type]


# Embedding selection is expensive (probes several Ollama models with real
# requests); remember the working choice per LLM model for the process.
_smart_embeddings_cache: Dict[str, Any] = {}


class _SourceRecordingRetriever(BaseRetriever):
    """Wraps the final retriever and records the documents it returns.

    Lets get_sources() reuse the documents retrieved while answering instead
    of paying for a second retrieval (query embedding + ensemble + reranker)
    per question. The record is cleared at the start of each ask.
    """

    wrapped: BaseRetriever
    record: List = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List:
        docs = self.wrapped.invoke(query)
        self.record.extend(docs)
        return docs


class Pipeline:
    """Base pipeline class with lazy loading and optimized performance."""

    def __init__(
        self,
        loader_name: str = "local",
        vector_store_type: Optional[str] = None,
        optimize_loading: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the pipeline with loaders and vector store.

        Args:
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        # Lazy import loaders
        if optimize_loading:
            LocalLoader = _get_cached_pipeline_import("optimized_local_loader")
            self.local_loader = LocalLoader(
                name="optimized_local", max_workers=max_workers, enable_cache=True
            )
            logger.info(f"🚀 Using OptimizedLocalLoader with {max_workers} workers")
        else:
            LocalLoader = _get_cached_pipeline_import("local_loader")
            self.local_loader = LocalLoader(name="local")
            logger.info("Using standard LocalLoader")

        RemoteLoader = _get_cached_pipeline_import("remote_loader")
        create_vector_store_proxy = _get_cached_pipeline_import("vector_db_proxy")

        self.remote_loader = RemoteLoader(name="remote")

        # Use factory function to create vector store
        if vector_store_type is None:
            vector_store_type = get_config().database.default_type

        self.vector_db = create_vector_store_proxy(vector_store_type)

        self.texts: Optional[List] = None
        self._retriever: Optional = None
        self._chain: Optional[FullChain] = None
        self.LLM: Optional = None
        self._last_question: Optional[str] = None
        self._vector_db_fingerprint: Optional[str] = None

        logger.info(
            f"Pipeline initialized with loader: {loader_name}, vector store: {vector_store_type or 'default'}, optimization: {optimize_loading}"
        )

    def load_and_split(
        self,
        data_dir: str = "data",
        uploaded_files: Optional[List] = None,
        page_urls: Optional[List[str]] = None,
        wiki_query: Optional[str] = None,
    ) -> Optional[List]:
        """
        Load and split documents from various sources. Sets self.texts.

        Args:
            data_dir (str): The directory to load documents from. Defaults to "data".
            uploaded_files (Optional[List]): A list of uploaded files.
            page_urls (Optional[List[str]]): URLs of pages to load documents from.
            wiki_query (Optional[str]): A Wikipedia query to load documents from.

        Returns:
            Optional[List]: The list of split documents, or None if no documents were processed.
        """
        docs: List = []
        logger.info(
            f"Starting document loading. data_dir='{data_dir}', uploaded_files={'yes' if uploaded_files else 'no'}, page_urls={'yes' if page_urls else 'no'}, wiki_query='{wiki_query if wiki_query else 'no'}'"
        )

        # Ensure loaders are reset or handle multiple calls appropriately if needed
        # For this example, assuming they load fresh each time `load` is called.
        self.local_loader.load(data_dir=data_dir, uploaded_files=uploaded_files)
        loaded_local_docs = self.local_loader.get_documents()
        if loaded_local_docs:
            docs.extend(loaded_local_docs)
            logger.info(f"Loaded {len(loaded_local_docs)} documents from local loader.")

        self.remote_loader.load(page_urls=page_urls, wiki_query=wiki_query)
        loaded_remote_docs = self.remote_loader.get_documents()
        if loaded_remote_docs:
            docs.extend(loaded_remote_docs)
            logger.info(
                f"Loaded {len(loaded_remote_docs)} documents from remote loader."
            )

        if not docs:
            logger.warning("No documents were loaded from any source.")
            self.texts = None
            return None

        self.texts = split_documents(docs)
        if self.texts:
            logger.info(
                f"Successfully split {len(docs)} source documents into {len(self.texts)} chunks."
            )
        else:
            logger.warning("Document splitting resulted in no text chunks.")
        return self.texts

    def _set_retriever(
        self,
        embeddings: Optional = None,
        use_ensemble: bool = True,
        use_reranker: bool = False,
    ) -> None:
        """
        Set the retriever for the pipeline with lazy loading and timeout protection.

        Args:
            embeddings (Optional): The embeddings to use for the vector store.
            use_ensemble (bool): Whether to use an ensemble retriever (BM25 + vector store). Defaults to True.
            use_reranker (bool): Whether to add a cross-encoder reranking
                stage (requires the `rerank` extra). Defaults to False.
        """
        if not self.texts:
            logger.error("Cannot set retriever: No texts have been loaded and split.")
            return

        logger.info(
            f"Setting retriever with timeout protection. Using ensemble: {use_ensemble}."
        )

        # With a reranker, the first stage fetches a wide candidate pool and
        # the cross-encoder narrows it back down to similarity_k.
        final_k = get_config().database.similarity_k
        if use_reranker:
            from .reranker import RERANK_CANDIDATES

            stage_one_k = RERANK_CANDIDATES
        else:
            stage_one_k = final_k

        # Add timeout protection for vector store creation
        try:
            # Re-embedding the corpus is the expensive step (one API call per
            # chunk). Skip it when the corpus and embedding model are
            # unchanged — e.g. toggling the reranker or rebuilding the chain.
            fingerprint = self._corpus_fingerprint(embeddings)
            if (
                fingerprint == self._vector_db_fingerprint
                and self.vector_db.db is not None
            ):
                logger.info(
                    "Corpus and embeddings unchanged; reusing existing vector store."
                )
            else:
                logger.info("Creating vector store database...")
                self.vector_db.create_db(docs=self.texts, embeddings=embeddings)
                self._vector_db_fingerprint = fingerprint

            vs = self.vector_db.db  # Use the property .db
            if vs is None:
                logger.error("Failed to create or access vector store database.")
                return

            logger.info("Vector store database created successfully.")
            vs_retriever = vs.as_retriever(search_kwargs={"k": stage_one_k})

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

        if use_ensemble:
            try:
                # Lazy import Document and BM25Retriever
                Document = _get_cached_pipeline_import("document")
                BM25Retriever = _get_cached_pipeline_import("bm25_retriever")
                EnsembleRetriever = _get_cached_pipeline_import("ensemble_retriever")

                docs_for_bm25 = [
                    doc
                    for doc in self.texts
                    if isinstance(doc, Document) and doc.page_content
                ]
                if not docs_for_bm25:
                    logger.warning(
                        "No page content found in documents for BM25Retriever. Using vector store retriever only."
                    )
                    self._retriever = vs_retriever
                else:
                    # from_documents (not from_texts) so chunks keep their
                    # source metadata for citations.
                    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
                    bm25_retriever.k = stage_one_k
                    self._retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vs_retriever], weights=[0.4, 0.6]
                    )
                    logger.info(
                        "Ensemble retriever created with BM25 and vector store retriever."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to create BM25Retriever or EnsembleRetriever: {e}. Falling back to vector store retriever only.",
                    exc_info=True,
                )
                self._retriever = vs_retriever
        else:
            self._retriever = vs_retriever
            logger.info("Vector store retriever created.")

        if use_reranker and self._retriever is not None:
            from .reranker import wrap_with_reranker

            self._retriever = wrap_with_reranker(self._retriever, top_k=final_k)

        if self._retriever is not None:
            # Outermost wrapper: records final retrieved docs so get_sources
            # can reuse them without a second retrieval.
            self._retriever = _SourceRecordingRetriever(wrapped=self._retriever)

        # A different retriever (new corpus, toggled ensemble/reranker) can
        # change answers — cached responses from the old setup are stale.
        _get_query_cache().clear_cache()

    def _corpus_fingerprint(self, embeddings) -> str:
        """Hash of the loaded corpus plus the embedding configuration.

        Identical fingerprints mean the existing vector store can be reused.
        """
        h = hashlib.sha256()
        for doc in self.texts or []:
            h.update(getattr(doc, "page_content", str(doc)).encode())
            h.update(b"\x00")
            h.update(str(sorted(getattr(doc, "metadata", {}).items())).encode())
            h.update(b"\x01")
        h.update(
            f"{type(embeddings).__name__}:{getattr(embeddings, 'model', '')}".encode()
        )
        return h.hexdigest()

    def get_chain(self) -> Optional[FullChain]:
        """Returns the created RAG chain, if any."""
        if not self._chain:
            logger.warning("Attempted to get chain, but it has not been created yet.")
        return self._chain

    def get_retriever(self):
        """Return the configured retriever, or None if not set."""
        return self._retriever

    def _begin_ask(self, question: str) -> None:
        """Reset per-question state so get_sources reflects this ask."""
        self._last_question = question
        if isinstance(self._retriever, _SourceRecordingRetriever):
            self._retriever.record.clear()

    def get_sources(self, question: str, k: int = 4) -> List[Dict[str, str]]:
        """
        Return the source chunks behind an answer, for displaying citations.

        If `question` is the one most recently asked, the documents recorded
        during that ask are reused — no second retrieval (and no extra
        embedding API call). Otherwise the retriever is invoked directly.

        Args:
            question (str): The question to retrieve sources for.
            k (int): Maximum number of sources to return. Defaults to 4.

        Returns:
            List[Dict[str, str]]: One entry per chunk, each with a "source"
            (file name or URL, "unknown" if unavailable) and a "snippet".
        """
        if not self._retriever:
            logger.warning("Cannot get sources: retriever is not set.")
            return []

        if (
            question == self._last_question
            and isinstance(self._retriever, _SourceRecordingRetriever)
            and self._retriever.record
        ):
            docs = list(self._retriever.record)
        else:
            try:
                docs = self._retriever.invoke(question)
            except Exception as e:
                logger.error(f"Failed to retrieve sources: {e}", exc_info=True)
                return []

        sources = []
        seen = set()
        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            # Show just the file name for local paths
            if "/" in source and "://" not in source:
                source = source.rsplit("/", 1)[-1]
            snippet = doc.page_content[:300].strip()
            key = (source, snippet[:80])
            if key in seen:
                continue
            seen.add(key)
            sources.append({"source": source, "snippet": snippet})
            if len(sources) >= k:
                break
        return sources

    def create_rag_chain(self, chain_type: str = "simple") -> None:
        """
        Creates the RAG chain using the configured LLM and retriever.

        Args:
            chain_type (str): The type of RAG chain to create (e.g., "simple", "multi_query", "fusion"). Defaults to "simple".
        """
        if not self.LLM:
            logger.error("Cannot create RAG chain: LLM is not set.")
            return
        if not self._retriever:
            logger.error("Cannot create RAG chain: Retriever is not set.")
            return

        logger.info(f"Creating RAG chain of type: {chain_type}")
        llm_instance = self.LLM.get_llm()
        if not llm_instance:
            logger.error(
                "Cannot create RAG chain: Failed to get LLM instance from proxy."
            )
            return

        rag_proxy = RagProxy(model=llm_instance, retriever=self._retriever)
        memory_proxy = MemoryProxy()  # Assuming default initialization is fine

        self._chain = FullChain(
            llm_proxy=self.LLM, rag_proxy=rag_proxy, memory_proxy=memory_proxy
        )
        try:
            self._chain.create_full_chain(chain_type=chain_type)
            logger.info(f"Successfully created RAG chain of type: {chain_type}")
        except Exception as e:
            logger.error(
                f"Error creating RAG chain of type '{chain_type}': {e}", exc_info=True
            )
            self._chain = None  # Ensure chain is None if creation fails

    def ask_question(
        self, question: str, session_id: str = "default", use_cache: bool = True
    ) -> Optional[str]:
        """
        Asks a question to the RAG chain, with optional response caching.

        Args:
            question (str): The question to ask.
            session_id (str): The session ID for memory. Defaults to "default".
            use_cache (bool): Whether to use the response cache. Defaults to True.

        Returns:
            str: The answer from the RAG chain.

        Raises:
            ChainInitializationError: If the RAG chain has not been created.
            ChainExecutionError: If answer generation fails (network, API,
                model errors). The original exception is attached.
        """
        if not self._chain:
            raise ChainInitializationError(
                "RAG chain is not created. Load documents, set a retriever, "
                "and call create_rag_chain() first."
            )

        self._begin_ask(question)
        logger.info(
            f"Asking question (session: {session_id}): '{question[:100]}{'...' if len(question) > 100 else ''}'"
        )

        cache_enabled = _is_response_cache_enabled()
        if use_cache and cache_enabled:
            cached_response = _get_query_cache().get_response(question, session_id)
            if cached_response:
                logger.info("⚡ Cache hit! Returning cached response.")
                return cached_response

        # Cache miss - generate new response
        try:
            start_time = time.time()
            response = self._chain.ask_question(query=question, session_id=session_id)

            if response and use_cache and cache_enabled:
                # Cache the successful response
                _get_query_cache().cache_response(question, response, session_id)

                generation_time = time.time() - start_time
                logger.info(
                    f"✅ Generated and cached response in {generation_time:.2f}s"
                )

                # Log cache statistics periodically
                if _get_query_cache().stats.total_queries % 10 == 0:
                    stats = _get_query_cache().get_stats()
                    logger.info(
                        f"📊 Cache stats: {stats['hit_rate']} hit rate "
                        f"({stats['hits']} hits / {stats['misses']} misses)"
                    )
            elif response and not cache_enabled:
                generation_time = time.time() - start_time
                logger.info(
                    f"✅ Generated response in {generation_time:.2f}s (cache disabled)"
                )

            logger.info("Received response from RAG chain.")
            return response

        except Exception as e:
            logger.error(f"Error during ask_question: {e}", exc_info=True)
            raise ChainExecutionError(
                f"Failed to generate a response: {e}", original_exception=e
            ) from e

    def ask_question_stream(
        self, question: str, session_id: str = "default", use_cache: bool = True
    ):
        """
        Ask a question and yield the answer incrementally as text chunks.

        Behaves like :meth:`ask_question`, but streams. On a cache hit the
        full cached answer is yielded as a single chunk; otherwise chunks are
        yielded as the LLM produces them and the complete answer is cached at
        the end.

        Args:
            question (str): The question to ask.
            session_id (str): The session ID for memory.
            use_cache (bool): Whether to use the response cache.

        Yields:
            str: Successive chunks of the answer.

        Raises:
            ChainInitializationError: If the RAG chain has not been created.
            ChainExecutionError: If generation fails, including mid-stream —
                callers must treat an exception after partial output as an
                incomplete answer, not silence.
        """
        if not self._chain:
            raise ChainInitializationError(
                "RAG chain is not created. Load documents, set a retriever, "
                "and call create_rag_chain() first."
            )

        self._begin_ask(question)
        cache_enabled = _is_response_cache_enabled()
        if use_cache and cache_enabled:
            cached_response = _get_query_cache().get_response(question, session_id)
            if cached_response:
                logger.info("⚡ Cache hit! Streaming cached response.")
                yield cached_response
                return

        try:
            parts = []
            for chunk in self._chain.stream_question(
                query=question, session_id=session_id
            ):
                parts.append(chunk)
                yield chunk
            response = "".join(parts)
            if response and use_cache and cache_enabled:
                _get_query_cache().cache_response(question, response, session_id)
        except Exception as e:
            logger.error(f"Error during ask_question_stream: {e}", exc_info=True)
            raise ChainExecutionError(
                f"Failed to generate a response: {e}", original_exception=e
            ) from e

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive caching performance statistics.

        Returns:
            Dict with cache performance metrics including hit rates, time saved, and cost savings.
        """
        if not _is_response_cache_enabled():
            return {
                "cache_enabled": False,
                "message": "Response cache is disabled in configuration",
            }

        stats = _get_query_cache().get_stats()
        stats["cache_enabled"] = True
        return stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        _get_query_cache().clear_cache()


class OpenAIPipeline(Pipeline):
    """OpenAI-based pipeline with lazy loading and performance optimizations."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        loader_name: str = "local",
        vector_store_type: Optional[str] = None,
        optimize_loading: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the OpenAIPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to "gpt-3.5-turbo".
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        super().__init__(
            loader_name=loader_name,
            vector_store_type=vector_store_type,
            optimize_loading=optimize_loading,
            max_workers=max_workers,
        )

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for OpenAIPipeline")

        # Initialize OpenAI LLM with lazy loading
        self.LLM = OpenAIProxy()
        self.LLM.set_llm(model_name=model)
        logger.info(f"OpenAIPipeline initialized with model: {model}")

    def set_retriever_openai(
        self, use_ensemble: bool = True, use_reranker: bool = False
    ) -> None:
        """
        Set the retriever for OpenAI pipeline with OpenAI embeddings.
        This method ALWAYS uses OpenAI embeddings, bypassing any Ollama preferences.
        Includes timeout handling to prevent hanging.

        Args:
            use_ensemble (bool): Whether to use ensemble retriever. Defaults to True.
            use_reranker (bool): Add a cross-encoder reranking stage
                (requires the `rerank` extra). Defaults to False.
        """
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found. Cannot create embeddings.")
            return

        logger.info("OpenAI Pipeline: Creating embeddings with timeout protection...")

        try:
            # Use standard LangChain OpenAI embeddings with clean configuration
            OpenAIEmbeddings = _get_cached_pipeline_import("openai_embeddings")

            # Create embeddings with explicit parameters and timeout handling
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                # Explicitly exclude organization to prevent "your_org_id_here" error
                openai_organization=None,
                model=get_config().llm.openai_embedding_model,
                # Add timeout and retry settings
                request_timeout=30,  # 30 second timeout
                max_retries=2,  # Retry up to 2 times
            )

            # No probe call here: auth/connectivity failures surface on the
            # first real embedding request moments later, so a paid "test"
            # embed per setup buys nothing.
            logger.info(
                f"✅ OpenAI embeddings created successfully (model: {embeddings.model})"
            )
            self._set_retriever(
                embeddings=embeddings,
                use_ensemble=use_ensemble,
                use_reranker=use_reranker,
            )

        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI embeddings: {e}")
            raise


class OllamaPipeline(Pipeline):
    """Ollama-based pipeline with lazy loading and performance optimizations."""

    def __init__(
        self,
        model: str = "llama3",
        loader_name: str = "local",
        vector_store_type: Optional[str] = None,
        optimize_loading: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the OllamaPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to "llama3".
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        super().__init__(
            loader_name=loader_name,
            vector_store_type=vector_store_type,
            optimize_loading=optimize_loading,
            max_workers=max_workers,
        )

        # Initialize Ollama LLM with lazy loading
        self.LLM = OllamaProxy()
        self.LLM.set_llm(model_name=model)
        logger.info(f"OllamaPipeline initialized with model: {model}")

    def set_retriever_ollama(
        self, use_ensemble: bool = True, use_reranker: bool = False
    ) -> None:
        """
        Set the retriever for Ollama pipeline with smart embedding fallback.
        Tries Ollama embeddings first, falls back to OpenAI embeddings if unavailable.

        Args:
            use_ensemble (bool): Whether to use ensemble retriever. Defaults to True.
            use_reranker (bool): Add a cross-encoder reranking stage
                (requires the `rerank` extra). Defaults to False.
        """
        embeddings = self._get_smart_embeddings()
        if embeddings:
            self._set_retriever(
                embeddings=embeddings,
                use_ensemble=use_ensemble,
                use_reranker=use_reranker,
            )
        else:
            logger.error("Failed to create any embeddings. Cannot set retriever.")

    def _get_smart_embeddings(self):
        """
        Get embeddings with intelligent model-aware selection, memoized.

        The model probing (several Ollama round trips) runs once per LLM
        model per process; later retriever setups reuse the result.

        Returns:
            Embeddings instance or None if all attempts fail.
        """
        cache_key = self.LLM.get_model_name() if self.LLM else "default"
        if cache_key in _smart_embeddings_cache:
            logger.info(f"Reusing embeddings selected earlier for '{cache_key}'")
            return _smart_embeddings_cache[cache_key]

        embeddings = self._select_smart_embeddings()
        if embeddings is not None:
            _smart_embeddings_cache[cache_key] = embeddings
        return embeddings

    def _select_smart_embeddings(self):
        """Probe embedding options in speed order (uncached)."""
        from ..config.settings import get_config

        config = get_config()

        # Try Ollama embeddings first if preferred
        if config.llm.prefer_ollama_embeddings:

            # STRATEGY 1: Try dedicated embedding models first (MUCH FASTER!)
            logger.info("🚀 Trying dedicated embedding models for optimal speed...")
            dedicated_models = [
                "nomic-embed-text:latest",
                "nomic-embed-text",
                "all-minilm:latest",
                "all-minilm",
                "mxbai-embed-large:latest",
                "mxbai-embed-large",
            ]

            for model in dedicated_models:
                try:
                    logger.info(f"⚡ Testing fast embedding model: {model}")
                    embeddings = self._try_embedding_model(model)
                    if embeddings:
                        logger.info(
                            f"✅ SUCCESS: Using fast embedding model '{model}'!"
                        )
                        return embeddings
                except Exception as e:
                    logger.debug(f"Embedding model '{model}' not available: {e}")
                    continue

            # STRATEGY 2: Try model-specific preferences
            llm_model = self.LLM.get_model_name() if self.LLM else None
            if llm_model:
                logger.info(
                    f"🔄 Trying model-specific embedding preferences for {llm_model}..."
                )
                embedding_models = self._get_embedding_models_for_llm(llm_model, config)

                # Filter to only available models if auto-detection is enabled
                if config.llm.auto_detect_available_models:
                    available_models = self._get_available_ollama_models()
                    embedding_models = [
                        model for model in embedding_models if model in available_models
                    ]
                    if embedding_models:
                        logger.info(
                            f"Found {len(embedding_models)} available embedding models for {llm_model}: {embedding_models}"
                        )
                    else:
                        logger.warning(
                            f"No embedding models available for {llm_model}, using fallback list"
                        )
                        embedding_models = self._get_embedding_models_for_llm(
                            llm_model, config
                        )

                ollama_embeddings = self._try_ollama_embeddings(embedding_models)
                if ollama_embeddings:
                    return ollama_embeddings

            # STRATEGY 3: Try LLM model directly as last resort (SLOWEST!)
            if llm_model:
                logger.warning(
                    f"⚠️ Trying LLM model '{llm_model}' directly as embedding model (will be SLOW)"
                )
                ollama_embeddings = self._try_direct_llm_embeddings(llm_model)
                if ollama_embeddings:
                    logger.warning(
                        f"⚠️ Using LLM model '{llm_model}' for embeddings - this will be slow!"
                    )
                    return ollama_embeddings

            logger.warning(
                "All Ollama embedding strategies failed, falling back to OpenAI embeddings"
            )

        # Fallback to OpenAI embeddings
        if os.getenv("OPENAI_API_KEY"):
            try:
                logger.info("Using standard LangChain OpenAI embeddings as fallback")
                OpenAIEmbeddings = _get_cached_pipeline_import("openai_embeddings")
                # Use clean configuration to avoid organization issues
                return OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_organization=None,
                    model=get_config().llm.openai_embedding_model,
                )
            except Exception as e:
                logger.error(f"Failed to create OpenAI embeddings: {e}")
        else:
            logger.error("No OpenAI API key available for fallback embeddings")

        return None

    def _try_embedding_model(self, model_name: str):
        """
        Try to use a dedicated embedding model.

        Args:
            model_name: The embedding model name to try

        Returns:
            OllamaEmbeddings instance or None if it fails
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")

        try:
            logger.debug(
                f"🚀 Creating OllamaEmbeddings with embedding model: {model_name}"
            )
            embeddings = OllamaEmbeddings(model=model_name)

            # Test the embeddings with a simple query to verify it works
            test_result = embeddings.embed_query("test")
            if test_result:
                dimensions = len(test_result)
                logger.info(
                    f"✅ Embedding model '{model_name}' works! Dimensions: {dimensions}"
                )
                return embeddings

        except Exception as e:
            logger.debug(f"❌ Embedding model '{model_name}' failed: {e}")

        return None

    def _try_direct_llm_embeddings(self, model_name: str):
        """
        Try to use the LLM model directly as an embedding model.
        This is the new simplified approach based on LangChain's capability.

        Args:
            model_name: The LLM model name to try as embedding model

        Returns:
            OllamaEmbeddings instance or None if it fails
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")

        try:
            logger.info(f"🚀 Creating OllamaEmbeddings with LLM model: {model_name}")
            embeddings = OllamaEmbeddings(model=model_name)

            # Test the embeddings with a simple query to verify it works
            test_result = embeddings.embed_query("test")
            if test_result:
                dimensions = len(test_result)
                logger.info(
                    f"✅ LLM model '{model_name}' works as embedding model! Dimensions: {dimensions}"
                )
                return embeddings

        except Exception as e:
            logger.warning(
                f"❌ LLM model '{model_name}' failed as embedding model: {e}"
            )

        return None

    def _get_embedding_models_for_llm(
        self, llm_model: Optional[str], config
    ) -> List[str]:
        """
        Get embedding models based on the LLM model selected.

        Args:
            llm_model: The LLM model name (e.g., "llama3", "phi4")
            config: Configuration object

        Returns:
            List of embedding models to try, in order of preference
        """
        if not llm_model:
            return config.llm.model_embedding_preferences.get("ollama_default", [])

        # Normalize model name (remove version suffixes for matching)
        base_model = llm_model.split(":")[0]  # "deepseek-r1:8b" -> "deepseek-r1"

        # Try exact match first
        if llm_model in config.llm.model_embedding_preferences:
            logger.info(
                f"Using embedding preferences for exact model match: {llm_model}"
            )
            return config.llm.model_embedding_preferences[llm_model]

        # Try base model match
        if base_model in config.llm.model_embedding_preferences:
            logger.info(
                f"Using embedding preferences for base model match: {base_model}"
            )
            return config.llm.model_embedding_preferences[base_model]

        # Fallback to default
        logger.info(f"No specific embedding preferences for {llm_model}, using default")
        return config.llm.model_embedding_preferences.get("ollama_default", [])

    def _get_available_ollama_models(self) -> List[str]:
        """
        Query Ollama to get list of available models.

        Returns:
            List of available model names
        """
        try:
            import requests

            from ..config.settings import get_config

            config = get_config()

            response = requests.get(f"{config.api.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]
                logger.debug(f"Available Ollama models: {available_models}")
                return available_models
            else:
                logger.warning(f"Failed to query Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Could not detect available Ollama models: {e}")
            return []

    def _try_ollama_embeddings(self, embedding_models: List[str]):
        """
        Try to create Ollama embeddings with multiple model options.

        Args:
            embedding_models: List of embedding models to try.

        Returns:
            OllamaEmbeddings instance or None if all models fail.
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")

        for model in embedding_models:
            try:
                logger.info(f"Attempting to use Ollama embedding model: {model}")
                embeddings = OllamaEmbeddings(model=model)

                # Test the embeddings with a simple query to verify the model works
                test_result = embeddings.embed_query("test")
                if test_result:
                    logger.info(f"Successfully using Ollama embedding model: {model}")
                    return embeddings

            except Exception as e:
                logger.warning(f"Ollama embedding model '{model}' failed: {e}")
                continue

        logger.warning("All Ollama embedding models failed")
        return None
