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

from langchain_core.embeddings import Embeddings

from ..config.settings import get_config

logger = logging.getLogger(__name__)

# Dedicated embedding models, probed in order. embeddinggemma leads by
# measurement (Experiment 25): on real legal text it matched the cloud
# embedder (hit 0.80/MRR 0.65 vs nomic's 0.56/0.47) and improved the
# fictional smoke slice (1.0/0.939) — end-to-end, the local stack's
# correctness rose +14.3pp from this swap alone. nomic remains the
# fallback for installs that don't have it.
_DEDICATED_MODELS = [
    "embeddinggemma:latest",
    "embeddinggemma",
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


# Per-family retrieval task conventions, from each model's card. Embedding
# models are trained with these markers; omitting them degrades silently —
# Experiment 22 measured bare nomic ranking a needle chunk 5th behind
# three contributor name-lists, 1st with prefixes. A model with no entry
# (bge-m3) genuinely needs none. Each convention is validated empirically
# (prefix on/off A-B on the retrieval slices, Experiment 25) — the table
# records model-card contracts, the harness checks they actually help.
_TASK_CONVENTIONS = {
    "nomic-embed-text": ("search_document: {t}", "search_query: {q}"),
    "mxbai-embed-large": (
        "{t}",
        "Represent this sentence for searching relevant passages: {q}",
    ),
    "snowflake-arctic-embed": ("{t}", "query: {q}"),
    # embeddinggemma deliberately has NO entry: its documented templates
    # measured HARMFUL through Ollama (hit 0.76/MRR 0.62 templated vs
    # 0.80/0.65 bare — Experiment 25's A-B; the modelfile template is a
    # passthrough, so the cause is uncertain). The A-B knob exists for
    # exactly this: conventions are hypotheses until the slice votes.
    "qwen3-embedding": (
        "{t}",
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: {q}",
    ),
}


class TaskPrefixedEmbeddings(Embeddings):
    """An Ollama embedder with the task templates its model card requires.

    Subclasses the langchain Embeddings ABC — vector stores isinstance-
    check it (FAISS treats anything else as a bare callable). The class
    name plus the model attribute form the embedding-cache and corpus-
    fingerprint identity, so prefixed vectors can never collide with
    vectors from the bare embedder; templates are keyed 1:1 by model
    family, so one identity never spans two conventions.
    """

    def __init__(self, inner: Any, doc_template: str, query_template: str) -> None:
        self._inner = inner
        self._doc_template = doc_template
        self._query_template = query_template
        self.model = inner.model  # cache/fingerprint identity

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(
            [self._doc_template.replace("{t}", t) for t in texts]
        )

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(self._query_template.replace("{q}", text))


def _task_convention_for(model_name: str):
    """The (doc, query) templates for a model, or None if it needs none."""
    for family, templates in _TASK_CONVENTIONS.items():
        if model_name.startswith(family):
            return templates
    return None


def _probe_ollama_model(model_name: str) -> Optional[Any]:
    """Build OllamaEmbeddings for a model and verify it actually embeds.

    Returns the working embeddings (task-prefixed per the model's
    convention), or None if the model is unavailable or the test
    embedding fails. RAGSTONE_EMBED_TASK_PREFIXES=off disables wrapping —
    the A-B knob that lets the harness validate each convention instead
    of trusting the table.
    """
    try:
        embeddings: Any = _ollama_embeddings_cls()(model=model_name)
        convention = _task_convention_for(model_name)
        prefixes_on = (
            os.getenv("RAGSTONE_EMBED_TASK_PREFIXES", "on").strip().lower() != "off"
        )
        if convention is not None and prefixes_on:
            embeddings = TaskPrefixedEmbeddings(embeddings, *convention)
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

    # An explicitly pinned embedder is an operator decision: honor it or
    # fail LOUD. Falling back to a different model here would silently
    # serve different retrieval than the operator chose (and with a
    # persisted index, different vectors than the corpus was built with).
    pinned = config.llm.ollama_embed_model
    if pinned:
        embeddings = _probe_ollama_model(pinned)
        if embeddings:
            return embeddings
        logger.error(
            f"RAGSTONE_OLLAMA_EMBED_MODEL={pinned!r} did not respond "
            f"(fix: `ollama pull {pinned}`); refusing to substitute "
            "another embedder."
        )
        return None

    if config.llm.prefer_ollama_embeddings:
        for model in _DEDICATED_MODELS:
            embeddings = _probe_ollama_model(model)
            if embeddings:
                return embeddings
        if config.profile == "local":
            # No-egress profile: an OpenAI fallback here would be exactly
            # the silent cloud call the profile exists to make impossible.
            logger.error(
                "No dedicated Ollama embedding model responded and "
                "RAGSTONE_PROFILE=local forbids the OpenAI fallback "
                "(fix: `ollama pull nomic-embed-text`)."
            )
            return None
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
