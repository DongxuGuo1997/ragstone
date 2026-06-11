"""
Optional cross-encoder reranking stage.

Two-stage retrieval: a fast first stage (vector / BM25 ensemble) fetches a
wide candidate pool, then a local cross-encoder rescores each
(question, chunk) pair and keeps only the best. Requires the `rerank`
extra::

    pip install -e ".[rerank]"

The cross-encoder runs fully locally (no API calls), so it works in
offline/Ollama mode too. The first use downloads the model (~80 MB).
"""

import logging

logger = logging.getLogger(__name__)

# Small, widely-used cross-encoder; good quality/latency balance on CPU.
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many first-stage candidates the reranker scores.
RERANK_CANDIDATES = 20

_INSTALL_HINT = (
    "Reranking requires the 'rerank' extra. "
    'Install it with: pip install -e ".[rerank]"'
)


def wrap_with_reranker(
    retriever, top_k: int = 4, model_name: str = DEFAULT_CROSS_ENCODER
):
    """
    Wrap a retriever with a cross-encoder reranking stage.

    Args:
        retriever: The first-stage retriever (should return a wide candidate
            pool — see RERANK_CANDIDATES).
        top_k: Number of chunks to keep after reranking.
        model_name: HuggingFace cross-encoder model name.

    Returns:
        A retriever that reranks the base retriever's results.

    Raises:
        ImportError: If the `rerank` extra (sentence-transformers) is not
            installed.
    """
    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e

    logger.info(f"Reranker enabled: {model_name} (keeping top {top_k})")
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=top_k),
        base_retriever=retriever,
    )
