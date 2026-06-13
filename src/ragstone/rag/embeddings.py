"""Embedding-model selection for the RAG pipeline.

OpenAI embeddings are straightforward. Ollama embedding selection is not:
Ollama serves many models and the right one depends on what the user has
installed, so this module probes options in speed order — dedicated
embedding models first, then model-specific preferences, then the LLM model
itself — and memoizes the working choice per model for the process.
"""

import logging
import os
from typing import Any, List, Optional

from ..config.settings import get_config

logger = logging.getLogger(__name__)

# Dedicated embedding models, fastest first. Tried before anything else.
_DEDICATED_MODELS = [
    "nomic-embed-text:latest",
    "nomic-embed-text",
    "all-minilm:latest",
    "all-minilm",
    "mxbai-embed-large:latest",
    "mxbai-embed-large",
]

# Probing Ollama models costs real round trips; remember the working choice
# per LLM model for the process so later retriever setups reuse it.
_smart_embeddings_cache: dict = {}


def _openai_embeddings_cls() -> Any:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings


def _ollama_embeddings_cls() -> Any:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings


def make_openai_embeddings() -> Any:
    """Construct OpenAI embeddings from configuration.

    The model comes from ``config.llm.openai_embedding_model``; organization
    is pinned to None so a stray ``OPENAI_ORG_ID`` can't break auth.
    """
    OpenAIEmbeddings = _openai_embeddings_cls()
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=None,
        model=get_config().llm.openai_embedding_model,
        request_timeout=30,
        max_retries=2,
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


def get_smart_embeddings(llm_model_name: Optional[str]) -> Optional[Any]:
    """Select embeddings (Ollama-first, OpenAI fallback), memoized per model."""
    cache_key = llm_model_name or "default"
    if cache_key in _smart_embeddings_cache:
        logger.info(f"Reusing embeddings selected earlier for '{cache_key}'")
        return _smart_embeddings_cache[cache_key]

    embeddings = _select_smart_embeddings(llm_model_name)
    if embeddings is not None:
        _smart_embeddings_cache[cache_key] = embeddings
    return embeddings


def _select_smart_embeddings(llm_model: Optional[str]) -> Optional[Any]:
    """Probe embedding options in speed order (uncached)."""
    config = get_config()

    if config.llm.prefer_ollama_embeddings:
        # 1. Dedicated embedding models — fastest, try first.
        for model in _DEDICATED_MODELS:
            embeddings = _probe_ollama_model(model)
            if embeddings:
                return embeddings

        # 2. Model-specific preferences for the chosen LLM.
        if llm_model:
            preferred = _embedding_models_for_llm(llm_model, config)
            if config.llm.auto_detect_available_models:
                available = _available_ollama_models()
                preferred = [m for m in preferred if m in available] or preferred
            for model in preferred:
                embeddings = _probe_ollama_model(model)
                if embeddings:
                    return embeddings

        # 3. The LLM model itself, as a last resort (slow).
        if llm_model:
            logger.warning(
                f"Trying LLM model '{llm_model}' directly as an embedding "
                "model — this will be slow."
            )
            embeddings = _probe_ollama_model(llm_model)
            if embeddings:
                return embeddings

        logger.warning(
            "All Ollama embedding strategies failed; falling back to OpenAI."
        )

    # Fallback: OpenAI embeddings.
    if os.getenv("OPENAI_API_KEY"):
        try:
            return make_openai_embeddings()
        except Exception as e:
            logger.error(f"Failed to create OpenAI fallback embeddings: {e}")
    else:
        logger.error("No OPENAI_API_KEY available for fallback embeddings.")

    return None


def _embedding_models_for_llm(llm_model: Optional[str], config: Any) -> List[str]:
    """Embedding models to try for a given LLM, in preference order."""
    prefs = config.llm.model_embedding_preferences
    if not llm_model:
        return prefs.get("ollama_default", [])

    base_model = llm_model.split(":")[0]  # "deepseek-r1:8b" -> "deepseek-r1"
    if llm_model in prefs:
        return prefs[llm_model]
    if base_model in prefs:
        return prefs[base_model]
    return prefs.get("ollama_default", [])


def _available_ollama_models() -> List[str]:
    """Query the Ollama server for installed models (empty list on failure)."""
    try:
        import requests

        base_url = get_config().api.ollama_base_url
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
        logger.warning(f"Failed to query Ollama models: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not detect available Ollama models: {e}")
    return []
