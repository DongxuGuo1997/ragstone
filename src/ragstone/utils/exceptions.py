"""
Custom exceptions for Ragstone.

One root (PipelineError) with a small family per subsystem. The root is a
load-bearing contract: the MCP server's _safe_error and the REST API's
exception handler treat any PipelineError message as user-safe to surface,
and anything else as internal (logged, but replaced with a generic message).
ValidationError is the single class the API maps to HTTP 422.

Every class here is raised somewhere in the codebase — when adding a new
one, add the raise site with it.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """
    Base exception class for all pipeline-related errors.

    This is the root exception that all other custom exceptions inherit from.
    It provides additional context for structured handling.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize the PipelineError.

        Args:
            message: Human-readable error message.
            error_code: Optional error code for programmatic handling.
            context: Optional dictionary with additional context information.
            original_exception: Optional original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
        self.original_exception = original_exception
        # Deliberately no logging here: whether a raised exception is an
        # error is the catcher's call, not the constructor's. Logging on
        # construction caused double logging and ERROR noise for failures
        # that were handled gracefully.


class ConfigurationError(PipelineError):
    """Raised when there are configuration-related issues."""

    def __init__(self, message: str, config_section: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if config_section:
            context["config_section"] = config_section
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class LLMError(PipelineError):
    """Base class for LLM-related errors."""

    pass


class LLMInitializationError(LLMError):
    """Raised when LLM initialization fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if model_name:
            context["model_name"] = model_name
        if provider:
            context["provider"] = provider
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class VectorStoreError(PipelineError):
    """Base class for vector store-related errors."""

    pass


class VectorStoreInitializationError(VectorStoreError):
    """Raised when vector store initialization fails."""

    def __init__(self, message: str, store_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if store_type:
            context["store_type"] = store_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class VectorStoreOperationError(VectorStoreError):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class DocumentLoadingError(PipelineError):
    """Raised when document loading fails."""

    pass


class RetrievalError(PipelineError):
    """Base class for retrieval-related errors."""

    pass


class RetrieverInitializationError(RetrievalError):
    """Raised when retriever initialization fails."""

    def __init__(self, message: str, retriever_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if retriever_type:
            context["retriever_type"] = retriever_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ChainError(PipelineError):
    """Base class for chain-related errors."""

    pass


class ChainInitializationError(ChainError):
    """Raised when chain initialization fails."""

    def __init__(self, message: str, chain_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if chain_type:
            context["chain_type"] = chain_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ChainExecutionError(ChainError):
    """Raised when chain execution fails."""

    def __init__(self, message: str, step: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if step:
            context["step"] = step
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ValidationError(PipelineError):
    """Raised when input validation fails. Maps to HTTP 422 in the API."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)[:100]  # Truncate long values
        kwargs["context"] = context
        super().__init__(message, **kwargs)
