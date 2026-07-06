import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from ..config.settings import get_config
from ..models.base_model import OllamaProxy, OpenAIProxy
from ..utils.exceptions import (
    ChainExecutionError,
    ChainInitializationError,
    RetrieverInitializationError,
)
from ..utils.full_chain import FullChain
from ..utils.observability import RequestMetrics, track_request
from .cache import QueryResultCache  # noqa: F401  re-exported; tests import here
from .cache import get_query_cache as _get_query_cache
from .cache import is_response_cache_enabled as _is_response_cache_enabled
from .embeddings import get_smart_embeddings, make_openai_embeddings
from .memory import MemoryProxy
from .rag import RagProxy, validate_question
from .splitter import split_documents

logger = logging.getLogger(__name__)

# Cache of heavy lazy imports (Document, retrievers, loaders, vector stores).
_pipeline_cache: Dict[str, Any] = {}


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
            logger.info(f"Using OptimizedLocalLoader with {max_workers} workers")
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
        self._retriever: Optional[Any] = None
        self._chain: Optional[FullChain] = None
        self._chain_type: Optional[str] = None
        self.LLM: Optional[Any] = None
        self._last_question: Optional[str] = None
        self._last_metrics: Optional[RequestMetrics] = None
        self._suggested_questions: Optional[Tuple[str, List[str]]] = None
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
        embeddings: Optional[Any] = None,
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
            raise RetrieverInitializationError(
                "Cannot set retriever: no documents have been loaded. "
                "Call load_and_split() with at least one data source first."
            )

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
        """Validate the question and reset per-question state.

        Validation runs here — before the cache lookup and before any
        retrieval/LLM call — so malformed or oversized input is rejected
        without spending anything.

        Raises:
            ValidationError: If the question is empty, not a string, or
                exceeds the configured length cap.
        """
        validate_question(question)
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

    def _build_chain(self, chain_type: str) -> FullChain:
        """Build a FullChain of the given type over the CURRENT retriever.

        Shared by create_rag_chain (the pipeline's main chain) and
        make_chain_variant (extra chains for comparison UIs) — the
        expensive work (embedding the corpus) happened when the retriever
        was set, so building a chain is cheap.

        Raises:
            ChainInitializationError: If the LLM or retriever is not set
                up, or if chain construction fails.
        """
        if not self.LLM or not self.LLM.get_llm():
            raise ChainInitializationError(
                "Cannot create RAG chain: LLM is not set. Construct the "
                "pipeline with a model (or call LLM.set_llm) first.",
                chain_type=chain_type,
            )
        if not self._retriever:
            raise ChainInitializationError(
                "Cannot create RAG chain: retriever is not set. Call "
                "set_retriever_openai() or set_retriever_ollama() first.",
                chain_type=chain_type,
            )

        logger.info(f"Creating RAG chain of type: {chain_type}")
        rag_proxy = RagProxy(model=self.LLM.get_llm(), retriever=self._retriever)
        memory_proxy = MemoryProxy()

        chain = FullChain(
            llm_proxy=self.LLM, rag_proxy=rag_proxy, memory_proxy=memory_proxy
        )
        try:
            chain.create_full_chain(chain_type=chain_type)
        except Exception as e:
            raise ChainInitializationError(
                f"Error creating RAG chain of type '{chain_type}': {e}",
                chain_type=chain_type,
                original_exception=e,
            ) from e
        return chain

    def create_rag_chain(self, chain_type: str = "simple") -> None:
        """
        Creates the RAG chain using the configured LLM and retriever.

        Args:
            chain_type (str): The type of RAG chain to create ("simple", "multi_query", "fusion", or "agent"). Defaults to "simple".

        Raises:
            ChainInitializationError: If the LLM or retriever is not set up,
                or if chain construction fails.
        """
        try:
            chain = self._build_chain(chain_type)
        except ChainInitializationError:
            self._chain = None
            raise
        self._chain = chain
        self._chain_type = chain_type  # recorded in per-request log lines
        logger.info(f"Successfully created RAG chain of type: {chain_type}")

    def make_chain_variant(self, chain_type: str) -> FullChain:
        """Build an ADDITIONAL chain of a different type over the same
        retriever and vector store — no re-embedding, and the pipeline's
        main chain is left untouched. Comparison UIs use this to run two
        techniques against the identical corpus.

        Args:
            chain_type (str): "simple", "multi_query", "fusion", or "agent".

        Returns:
            FullChain: An independent chain with its own conversation
            memory (sessions do not leak between variants).

        Raises:
            ChainInitializationError: If the LLM or retriever is not set
                up, or if chain construction fails.
        """
        return self._build_chain(chain_type)

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

        with track_request(session_id, chain_type=self._chain_type) as metrics:
            # The metrics object is mutable and completed when the context
            # closes, so exposing it now is safe: by the time a caller reads
            # last_metrics (after this method returns), it is fully filled.
            self._last_metrics = metrics
            cache_enabled = _is_response_cache_enabled()
            if use_cache and cache_enabled:
                cached_response = _get_query_cache().get_response(question, session_id)
                if cached_response:
                    logger.info("Cache hit! Returning cached response.")
                    metrics.cache_hit = True
                    return cached_response

            # Cache miss - generate new response
            try:
                start_time = time.time()
                response = self._chain.ask_question(
                    query=question, session_id=session_id
                )

                if response and use_cache and cache_enabled:
                    # Cache the successful response
                    _get_query_cache().cache_response(question, response, session_id)

                    generation_time = time.time() - start_time
                    logger.info(
                        f"Generated and cached response in {generation_time:.2f}s"
                    )

                    # Log cache statistics periodically
                    if _get_query_cache().stats.total_queries % 10 == 0:
                        stats = _get_query_cache().get_stats()
                        logger.info(
                            f"Cache stats: {stats['hit_rate']} hit rate "
                            f"({stats['hits']} hits / {stats['misses']} misses)"
                        )
                elif response and not cache_enabled:
                    generation_time = time.time() - start_time
                    logger.info(
                        f"Generated response in {generation_time:.2f}s (cache disabled)"
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
            str: Successive chunks of the answer text.
            dict: Progress events (e.g. {"event": "search", "query": ...}
                from the agent chain's live searches). Events are not part
                of the answer text and are never cached — consumers that
                only want text can skip non-str chunks.

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
        with track_request(session_id, chain_type=self._chain_type) as metrics:
            self._last_metrics = metrics  # completed when the stream ends
            cache_enabled = _is_response_cache_enabled()
            if use_cache and cache_enabled:
                cached_response = _get_query_cache().get_response(question, session_id)
                if cached_response:
                    logger.info("Cache hit! Streaming cached response.")
                    metrics.cache_hit = True
                    yield cached_response
                    return

            try:
                parts = []
                for chunk in self._chain.stream_question(
                    query=question, session_id=session_id
                ):
                    # Progress events pass through to the consumer but are
                    # not part of the answer text (and must not be cached).
                    if isinstance(chunk, str):
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

    def suggest_questions(self, n: int = 3) -> List[str]:
        """Generate starter questions the loaded corpus can answer.

        Costs one LLM call per corpus; the result is cached against a hash
        of the chunk contents, so repeated calls (UI reruns, retriever
        rebuilds over the same documents) are free. Returns an empty list
        when no documents or no LLM are available — callers can simply
        hide the feature.

        Args:
            n (int): How many questions to generate. Defaults to 3.
        """
        if not self.texts or not self.LLM or not self.LLM.get_llm():
            return []

        # Sample chunks spread across the corpus, not just the first file.
        step = max(1, len(self.texts) // 4)
        excerpts = [doc.page_content[:500] for doc in self.texts[::step][:4]]
        corpus_key = hashlib.sha256("\x00".join(excerpts).encode("utf-8")).hexdigest()
        if self._suggested_questions and self._suggested_questions[0] == corpus_key:
            return self._suggested_questions[1][:n]

        prompt = (
            "Here are excerpts from a document collection:\n\n"
            + "\n---\n".join(excerpts)
            + f"\n\nGenerate {n} short, diverse questions that this "
            "collection can answer. Cover different excerpts where "
            "possible. One question per line, no numbering."
        )
        try:
            from langchain_core.output_parsers import StrOutputParser

            from .rag import parse_generated_queries

            text = (self.LLM.get_llm() | StrOutputParser()).invoke(prompt)
            questions = parse_generated_queries(text)[:n]
        except Exception as e:
            # Suggestions are a convenience — never let them break the app.
            logger.warning(f"Could not generate suggested questions: {e}")
            return []

        self._suggested_questions = (corpus_key, questions)
        return questions

    @property
    def last_metrics(self) -> Optional[RequestMetrics]:
        """Metrics of the most recent ask (latency, tokens, cache hit).

        Complete once the ask returns (or, for streaming, once the stream
        is fully consumed). None before the first ask.
        """
        return self._last_metrics

    def get_last_interpretation(self, session_id: str) -> Optional[str]:
        """The standalone question the rephrase step produced for the most
        recent turn of a session, or None (first turn / unknown session).

        Reads checkpointed graph state; makes no LLM call. Lets UIs show
        how a follow-up like "what about its population?" was understood.
        """
        if self._chain is None:
            return None
        return self._chain.get_interpretation(session_id)

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
        model: str = "gpt-4o-mini",
        loader_name: str = "local",
        vector_store_type: Optional[str] = None,
        optimize_loading: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the OpenAIPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to "gpt-4o-mini".
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
            raise RetrieverInitializationError(
                "OPENAI_API_KEY not found — cannot create OpenAI embeddings. "
                "Set it in your .env file, or use set_retriever_ollama() for "
                "a fully local setup."
            )

        logger.info("Creating OpenAI embeddings...")
        embeddings = make_openai_embeddings()
        self._set_retriever(
            embeddings=embeddings,
            use_ensemble=use_ensemble,
            use_reranker=use_reranker,
        )


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
        embeddings = get_smart_embeddings(
            self.LLM.get_model_name() if self.LLM else None
        )
        if not embeddings:
            raise RetrieverInitializationError(
                "Failed to create any embeddings: no Ollama embedding model "
                "responded and no OPENAI_API_KEY is set for fallback. "
                "Is Ollama running? (ollama serve)"
            )
        self._set_retriever(
            embeddings=embeddings,
            use_ensemble=use_ensemble,
            use_reranker=use_reranker,
        )
