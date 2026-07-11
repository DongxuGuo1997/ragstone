"""Embedding-model selection for the RAG pipeline.

OpenAI embeddings are straightforward. For Ollama, this module probes the
dedicated embedding models a local install typically carries (nomic et al.)
and falls back to OpenAI if none responds. It deliberately does NOT fall
back to embedding with the chat LLM itself: that "works" mechanically but
produces drastically worse retrieval — a clear error beats a pipeline that
silently degrades (every measured run, incl. Experiment 21, used a
dedicated embedder).
"""

import logging
import os
from typing import Any, List, Optional

from ..config.settings import get_config

logger = logging.getLogger(__name__)

# Dedicated embedding models, fastest first.
_DEDICATED_MODELS = [
    "nomic-embed-text:latest",
    "nomic-embed-text",
    "all-minilm:latest",
    "all-minilm",
    "mxbai-embed-large:latest",
    "mxbai-embed-large",
]

# Probing costs real round trips; the choice doesn't depend on the chat
# model, so one process-wide memo covers every retriever setup.
_selected_embeddings: Optional[Any] = None


def _openai_embeddings_cls() -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings


def _ollama_embeddings_cls() -> Any:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings


def embed_texts_parallel(
    embeddings: Any,
    texts: List[str],
    batch_size: int,
    max_workers: int,
) -> List[List[float]]:
    """Embed texts in concurrent batches, preserving input order.

    Embedding a large corpus is network-bound: the provider slices it into
    batches but sends them one HTTP request at a time. Issuing batches from
    a small thread pool overlaps those round-trips (threads are the right
    tool — the GIL is released while waiting on the network), which is the
    dominant ingestion cost for big corpora.

    A single batch (or max_workers <= 1) falls through to a plain
    embed_documents call. Any batch failure fails the whole operation —
    a partially embedded corpus must never be indexed silently.
    """
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    if len(batches) <= 1 or max_workers <= 1:
        return embeddings.embed_documents(texts)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(
        f"Embedding {len(texts)} texts in {len(batches)} batches "
        f"({max_workers} workers)"
    )
    results: List[Optional[List[List[float]]]] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        future_to_index = {
            executor.submit(embeddings.embed_documents, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()  # raises on failure

    return [vector for batch in results if batch for vector in batch]


def embed_texts_cached(
    embeddings: Any,
    texts: List[str],
    batch_size: int,
    max_workers: int,
) -> List[List[float]]:
    """Embed texts, serving unchanged ones from the content-addressed cache.

    This is what makes re-ingestion incremental (Experiment 14): only
    cache misses — new or edited chunks — reach the embedding API; their
    vectors are computed via :func:`embed_texts_parallel` and written
    back. Cached vectors round-trip exactly, so the resulting index is
    bit-identical to an uncached build. With the cache disabled
    (RAGSTONE_EMBED_CACHE=off) this is embed_texts_parallel verbatim.
    """
    from .embedding_cache import get_embedding_cache, model_id_for

    cache = get_embedding_cache()
    if cache is None:
        return embed_texts_parallel(embeddings, texts, batch_size, max_workers)

    model_id = model_id_for(embeddings)
    cached = cache.get_many(model_id, texts)
    miss_indices = [i for i in range(len(texts)) if i not in cached]
    logger.info(
        f"Embedding cache: {len(cached)} hits, {len(miss_indices)} misses "
        f"({model_id})"
    )
    if miss_indices:
        miss_texts = [texts[i] for i in miss_indices]
        miss_vectors = embed_texts_parallel(
            embeddings, miss_texts, batch_size, max_workers
        )
        cache.put_many(model_id, miss_texts, miss_vectors)
        for index, vector in zip(miss_indices, miss_vectors):
            cached[index] = vector

    return [cached[i] for i in range(len(texts))]


def make_openai_embeddings() -> Any:
    """Construct OpenAI embeddings from configuration.

    The model comes from ``config.llm.openai_embedding_model``; organization
    is pinned to None so a stray ``OPENAI_ORG_ID`` can't break auth.
    """
    OpenAIEmbeddings = _openai_embeddings_cls()
    llm_cfg = get_config().llm
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=None,
        model=llm_cfg.openai_embedding_model,
        request_timeout=llm_cfg.timeout,
        max_retries=llm_cfg.max_retries,
    )


def _probe_ollama_model(model_name: str) -> Optional[Any]:
    """Build OllamaEmbeddings for a model and verify it actually embeds.

    Returns the working embeddings, or None if the model is unavailable or
    the test embedding fails.
    """
    try:
        embeddings = _ollama_embeddings_cls()(model=model_name)
        if embeddings.embed_query("test"):
            logger.info(f"Using Ollama embedding model: {model_name}")
            return embeddings
    except Exception as e:
        logger.debug(f"Ollama embedding model '{model_name}' unavailable: {e}")
    return None


def get_smart_embeddings() -> Optional[Any]:
    """Select embeddings (Ollama-first, OpenAI fallback), memoized."""
    global _selected_embeddings
    if _selected_embeddings is not None:
        logger.info("Reusing embeddings selected earlier this process")
        return _selected_embeddings

    embeddings = _select_smart_embeddings()
    if embeddings is not None:
        _selected_embeddings = embeddings
    return embeddings


def _select_smart_embeddings() -> Optional[Any]:
    """Probe dedicated Ollama embedders, then the OpenAI fallback (uncached)."""
    config = get_config()

    if config.llm.prefer_ollama_embeddings:
        for model in _DEDICATED_MODELS:
            embeddings = _probe_ollama_model(model)
            if embeddings:
                return embeddings
        logger.warning(
            "No dedicated Ollama embedding model responded "
            "(fix: `ollama pull nomic-embed-text`); trying OpenAI."
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            return make_openai_embeddings()
        except Exception as e:
            logger.error(f"Failed to create OpenAI fallback embeddings: {e}")
    else:
        logger.error("No OPENAI_API_KEY available for fallback embeddings.")

    return None
