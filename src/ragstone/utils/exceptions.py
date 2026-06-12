"""
Custom exceptions for the Ragstone.

This module defines a hierarchy of custom exceptions to provide better error
handling and debugging capabilities throughout the application.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """
    Base exception class for all pipeline-related errors.

    This is the root exception that all other custom exceptions inherit from.
    It provides additional context and logging capabilities.
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

        # Log the error when it's created
        self._log_error()

    def _log_error(self) -> None:
        """Log the error with appropriate details."""
        log_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            log_msg += f" | Context: {self.context}"
        if self.original_exception:
            log_msg += f" | Original: {self.original_exception}"

        logger.error(log_msg, exc_info=self.original_exception is not None)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary for serialization.

        Returns:
            Dictionary representation of the exception.
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            ),
        }


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


class LLMConnectionError(LLMError):
    """Raised when LLM connection fails."""

    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if endpoint:
            context["endpoint"] = endpoint
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        context = kwargs.get("context", {})
        if retry_after:
            context["retry_after"] = retry_after
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


class EmbeddingError(VectorStoreError):
    """Raised when embedding operations fail."""

    def __init__(self, message: str, text_count: Optional[int] = None, **kwargs):
        context = kwargs.get("context", {})
        if text_count:
            context["text_count"] = text_count
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class DocumentLoadingError(PipelineError):
    """Base class for document loading errors."""

    pass


class FileLoadingError(DocumentLoadingError):
    """Raised when file loading fails."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if file_path:
            context["file_path"] = file_path
        if file_type:
            context["file_type"] = file_type
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class RemoteLoadingError(DocumentLoadingError):
    """Raised when remote content loading fails."""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if url:
            context["url"] = url
        if status_code:
            context["status_code"] = status_code
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class DocumentProcessingError(DocumentLoadingError):
    """Raised when document processing fails."""

    def __init__(
        self,
        message: str,
        document_count: Optional[int] = None,
        processing_step: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if document_count:
            context["document_count"] = document_count
        if processing_step:
            context["processing_step"] = processing_step
        kwargs["context"] = context
        super().__init__(message, **kwargs)


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


class SearchError(RetrievalError):
    """Raised when search operations fail."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        k: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if query:
            context["query"] = query[:100] + "..." if len(query) > 100 else query
        if k:
            context["k"] = k
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


class ConversationMemoryError(PipelineError):
    """Raised when conversation memory operations fail.

    Named to avoid shadowing the built-in MemoryError.
    """

    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if session_id:
            context["session_id"] = session_id
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ValidationError(PipelineError):
    """Raised when input validation fails."""

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


class APIError(PipelineError):
    """Base class for external API errors."""

    pass


class APIConnectionError(APIError):
    """Raised when API connection fails."""

    def __init__(
        self,
        message: str,
        api_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if api_name:
            context["api_name"] = api_name
        if endpoint:
            context["endpoint"] = endpoint
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, message: str, api_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if api_name:
            context["api_name"] = api_name
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class UIError(PipelineError):
    """Base class for UI-related errors."""

    pass


class FileUploadError(UIError):
    """Raised when file upload fails."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if filename:
            context["filename"] = filename
        if file_size:
            context["file_size"] = file_size
        kwargs["context"] = context
        super().__init__(message, **kwargs)


# Exception hierarchy mapping for easy lookup
EXCEPTION_HIERARCHY = {
    "pipeline": PipelineError,
    "configuration": ConfigurationError,
    "llm": {
        "base": LLMError,
        "initialization": LLMInitializationError,
        "connection": LLMConnectionError,
        "rate_limit": LLMRateLimitError,
    },
    "vector_store": {
        "base": VectorStoreError,
        "initialization": VectorStoreInitializationError,
        "operation": VectorStoreOperationError,
        "embedding": EmbeddingError,
    },
    "document_loading": {
        "base": DocumentLoadingError,
        "file": FileLoadingError,
        "remote": RemoteLoadingError,
        "processing": DocumentProcessingError,
    },
    "retrieval": {
        "base": RetrievalError,
        "initialization": RetrieverInitializationError,
        "search": SearchError,
    },
    "chain": {
        "base": ChainError,
        "initialization": ChainInitializationError,
        "execution": ChainExecutionError,
    },
    "memory": ConversationMemoryError,
    "validation": ValidationError,
    "api": {
        "base": APIError,
        "connection": APIConnectionError,
        "authentication": APIAuthenticationError,
    },
    "ui": {
        "base": UIError,
        "file_upload": FileUploadError,
    },
}


def get_exception_class(error_type: str) -> type:
    """
    Get exception class by type string.

    Args:
        error_type: String identifier for the exception type.

    Returns:
        Exception class corresponding to the error type.

    Raises:
        ValueError: If error_type is not found.
    """

    def _find_in_hierarchy(hierarchy: Dict, path: List[str]) -> type:
        current = hierarchy
        for part in path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError(f"Exception type not found: {error_type}")
        return current

    path = error_type.split(".")
    try:
        return _find_in_hierarchy(EXCEPTION_HIERARCHY, path)
    except (KeyError, TypeError):
        raise ValueError(f"Exception type not found: {error_type}")


def handle_exception(
    func_name: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise_as: Optional[type] = None,
) -> None:
    """
    Generic exception handler that logs and optionally re-raises exceptions.

    Args:
        func_name: Name of the function where the exception occurred.
        exception: The original exception.
        context: Additional context information.
        reraise_as: Exception class to re-raise as (if different from original).

    Raises:
        The original exception or the specified reraise_as exception.
    """
    error_context = context or {}
    error_context["function"] = func_name

    if isinstance(exception, PipelineError):
        # Already a pipeline error, just add context
        exception.context.update(error_context)
        raise exception
    else:
        # Convert to pipeline error
        if reraise_as and issubclass(reraise_as, PipelineError):
            raise reraise_as(
                message=str(exception),
                context=error_context,
                original_exception=exception,
            )
        else:
            raise PipelineError(
                message=f"Unexpected error in {func_name}: {exception}",
                context=error_context,
                original_exception=exception,
            )
