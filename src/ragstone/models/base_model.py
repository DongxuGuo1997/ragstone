import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..config.settings import get_config
from ..utils.exceptions import LLMInitializationError

logger = logging.getLogger(__name__)

# Import cache for heavy dependencies
_model_cache: dict = {}


def _get_cached_llm_import(provider: str):
    """Get cached LLM import or import and cache it."""
    if provider not in _model_cache:
        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI

                _model_cache[provider] = ChatOpenAI
            elif provider == "ollama":
                from langchain_ollama import ChatOllama

                _model_cache[provider] = ChatOllama
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except ImportError as e:
            logger.error(f"Failed to import {provider} LLM: {e}")
            raise

    return _model_cache[provider]


class LLMProxy(ABC):
    """Abstract base class for LLM proxies with lazy loading."""

    def __init__(self):
        self._llm: Optional[Any] = None
        self._model_name: Optional[str] = None
        logger.info(f"{self.__class__.__name__} initialized.")

    @abstractmethod
    def set_llm(self, model_name: str, **kwargs: Any) -> Optional[Any]:
        """
        Abstract method to set and configure the LLM instance.

        Args:
            model_name (str): The name of the model to use.
            **kwargs: Additional keyword arguments for LLM configuration.

        Returns:
            Optional: The configured LLM instance, or None if setup fails.
        """
        pass

    @abstractmethod
    def get_llm(self) -> Optional[Any]:
        """
        Abstract method to get the LLM instance.

        Returns:
            Optional: The LLM instance, or None if not set.
        """
        pass

    def get_model_name(self) -> Optional[str]:
        """
        Get the name of the configured LLM model.

        Returns:
            Optional[str]: The name of the model, or None if not set.
        """
        return self._model_name


class OpenAIProxy(LLMProxy):
    """Proxy class for OpenAI LLM with lazy loading."""

    def __init__(self):
        super().__init__()
        # Lazy-loaded LLM instance
        self._llm: Optional[Any] = None

    def get_llm(self) -> Optional[Any]:
        """Get the OpenAI LLM instance."""
        if not self._llm:
            logger.warning("OpenAI LLM instance requested but not set.")
        return self._llm

    def set_llm(
        self, model_name: str, temperature: float = 0.0, **kwargs: Any
    ) -> Optional[Any]:
        """
        Set and configure the OpenAI LLM instance with lazy loading.

        Args:
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-4o-mini").
            temperature (float): The temperature setting for the LLM. Defaults to 0.0.
            **kwargs: Additional keyword arguments for ChatOpenAI.

        Returns:
            The configured ChatOpenAI instance.

        Raises:
            LLMInitializationError: If the API key is missing or setup fails.
        """
        logger.info(
            f"Attempting to set OpenAI LLM to model: {model_name}, temperature: {temperature}"
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMInitializationError(
                "OPENAI_API_KEY environment variable not found. Set it in your "
                ".env file or environment to use OpenAI models.",
                model_name=model_name,
                provider="openai",
            )

        try:
            # Lazy import ChatOpenAI only when needed
            ChatOpenAI = _get_cached_llm_import("openai")

            # Resilience defaults from config: bounded retries on transient
            # API errors and a request timeout so a hung call cannot hang
            # the request forever. Caller kwargs still win.
            llm_cfg = get_config().llm
            kwargs.setdefault("max_retries", llm_cfg.max_retries)
            kwargs.setdefault("request_timeout", llm_cfg.timeout)

            self._llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,  # Explicitly pass API key
                **kwargs,
            )
            self._model_name = model_name
            logger.info(f"Successfully set OpenAI LLM to model: {self._model_name}")
            return self._llm
        except Exception as e:
            self._llm = None
            self._model_name = None
            raise LLMInitializationError(
                f"Failed to initialize ChatOpenAI with model {model_name}: {e}",
                model_name=model_name,
                provider="openai",
                original_exception=e,
            ) from e


class OllamaProxy(LLMProxy):
    """Proxy class for Ollama LLM with lazy loading."""

    def __init__(self):
        super().__init__()
        # Lazy-loaded LLM instance
        self._llm: Optional[Any] = None

    def get_llm(self) -> Optional[Any]:
        """Get the Ollama LLM instance."""
        if not self._llm:
            logger.warning("Ollama LLM instance requested but not set.")
        return self._llm

    def set_llm(self, model_name: str = "qwen3.5:9b", **kwargs: Any) -> Optional[Any]:
        """
        Set and configure the Ollama LLM instance with lazy loading.

        The default mirrors DEFAULT_MODELS["ollama"] (providers.py), which
        is decided by measurement — Experiment 21.

        Args:
            model_name (str): The name of the Ollama model to use
                (e.g., "qwen3.5:9b").
            **kwargs: Additional keyword arguments for ChatOllama.

        Returns:
            The configured ChatOllama instance.

        Raises:
            LLMInitializationError: If Ollama setup fails.
        """
        logger.info(f"Attempting to set Ollama LLM to model: {model_name}")
        try:
            # Lazy import ChatOllama only when needed
            ChatOllama = _get_cached_llm_import("ollama")

            config = get_config()
            kwargs.setdefault("base_url", config.api.ollama_base_url)
            # ChatOllama has no retry/timeout params of its own; the request
            # timeout goes to its underlying httpx client instead. Retries
            # are less relevant for a local server, so none are forced here.
            kwargs.setdefault("client_kwargs", {"timeout": config.llm.timeout})
            # Thinking control (RAGSTONE_OLLAMA_REASONING): applied here in
            # the proxy so every surface — REST API, MCP, eval, chat —
            # honors it. None means the knob is absent entirely and the
            # model's own default stands; setdefault keeps caller kwargs
            # winning, same as base_url above.
            if config.llm.ollama_reasoning is not None:
                kwargs.setdefault("reasoning", config.llm.ollama_reasoning)
            self._llm = ChatOllama(
                model=model_name, **kwargs
            )  # 'model' is the correct param for ChatOllama
            self._model_name = model_name
            logger.info(f"Successfully set Ollama LLM to model: {self._model_name}")
            return self._llm
        except Exception as e:
            self._llm = None
            self._model_name = None
            raise LLMInitializationError(
                f"Failed to initialize ChatOllama with model {model_name}: {e}. "
                "Is Ollama running? (ollama serve)",
                model_name=model_name,
                provider="ollama",
                original_exception=e,
            ) from e
