"""Provider-bound pipelines: the OpenAI and Ollama subclasses.

The base Pipeline (pipeline.py) is provider-neutral; these subclasses
bind an LLM proxy and an embedding strategy, and `build_pipeline` is the
single home of the provider dispatch and default-model choice — the REST
API, MCP server, and terminal chat all construct pipelines through it
(tests stub it as their seam).
"""

import logging
import os
from typing import Optional

from ..models.base_model import OllamaProxy, OpenAIProxy
from ..utils.exceptions import RetrieverInitializationError
from .embeddings import get_smart_embeddings, make_openai_embeddings
from .pipeline import Pipeline

logger = logging.getLogger(__name__)


# Per-provider default models — the ONE place they are defined; the REST
# API, MCP server, and terminal chat all resolve defaults through here.
# The Ollama default is decided by measurement (Experiment 21): qwen3.5:9b
# was the quality/latency balance point of the measured tier menu — the
# previous default, llama3, was never measured at all.
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "ollama": "qwen3.5:9b"}


def build_pipeline(provider: str, model: Optional[str] = None) -> Pipeline:
    """Construct the right pipeline subclass for a provider name.

    The single home of the provider dispatch: every server and UI builds
    pipelines through this function (tests stub it as their seam).

    Args:
        provider: "openai" or "ollama" (case-insensitive).
        model: Model name; defaults to the provider's entry in
            DEFAULT_MODELS.

    Raises:
        ValueError: If the provider is not recognized.
    """
    provider = provider.strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of "
            f"{sorted(DEFAULT_MODELS)}."
        )
    resolved_model = model or DEFAULT_MODELS[provider]
    if provider == "openai":
        return OpenAIPipeline(model=resolved_model)
    return OllamaPipeline(model=resolved_model)


class OpenAIPipeline(Pipeline):
    """OpenAI-based pipeline with lazy loading and performance optimizations."""

    provider = "openai"

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
        self.llm_proxy = OpenAIProxy()
        self.llm_proxy.set_llm(model_name=model)
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

    def setup_retriever(
        self, use_ensemble: bool = True, use_reranker: bool = False
    ) -> None:
        """Provider-neutral alias for set_retriever_openai (see Pipeline)."""
        self.set_retriever_openai(use_ensemble=use_ensemble, use_reranker=use_reranker)


class OllamaPipeline(Pipeline):
    """Ollama-based pipeline with lazy loading and performance optimizations."""

    provider = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODELS["ollama"],
        loader_name: str = "local",
        vector_store_type: Optional[str] = None,
        optimize_loading: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the OllamaPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to DEFAULT_MODELS["ollama"].
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
        self.llm_proxy = OllamaProxy()
        self.llm_proxy.set_llm(model_name=model)
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
        embeddings = get_smart_embeddings()
        if not embeddings:
            raise RetrieverInitializationError(
                "No embedding backend available: no dedicated Ollama "
                "embedding model responded (fix: `ollama pull "
                "nomic-embed-text`, and check `ollama serve` is running) "
                "and no OPENAI_API_KEY is set for fallback."
            )
        self._set_retriever(
            embeddings=embeddings,
            use_ensemble=use_ensemble,
            use_reranker=use_reranker,
        )

    def setup_retriever(
        self, use_ensemble: bool = True, use_reranker: bool = False
    ) -> None:
        """Provider-neutral alias for set_retriever_ollama (see Pipeline)."""
        self.set_retriever_ollama(use_ensemble=use_ensemble, use_reranker=use_reranker)
