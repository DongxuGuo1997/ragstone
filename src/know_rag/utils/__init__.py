"""
Utility functions and helpers for the LangChain RAG pipeline.

This module contains exceptions, chain utilities, and common helpers.
"""

# Import common utilities
from .common import (
    batch_process,
    create_cache_key,
    ensure_directory_exists,
    format_exception_context,
    get_file_size_mb,
    get_timestamp,
    merge_dicts,
    normalize_whitespace,
    retry_with_backoff,
    safe_execute,
    sanitize_input,
    truncate_text,
    validate_file_path,
)

# Import all exceptions
from .exceptions import (
    APIAuthenticationError,
    APIConnectionError,
    APIError,
    ChainError,
    ChainExecutionError,
    ChainInitializationError,
    ConfigurationError,
    ConversationMemoryError,
    DocumentLoadingError,
    DocumentProcessingError,
    EmbeddingError,
    FileLoadingError,
    FileUploadError,
    LLMError,
    LLMInitializationError,
    PipelineError,
    RemoteLoadingError,
    RetrievalError,
    RetrieverInitializationError,
    SearchError,
    UIError,
    ValidationError,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorStoreOperationError,
)

__all__ = [
    # Exceptions
    "PipelineError",
    "ConfigurationError",
    "LLMError",
    "LLMInitializationError",
    "VectorStoreError",
    "VectorStoreInitializationError",
    "VectorStoreOperationError",
    "EmbeddingError",
    "DocumentLoadingError",
    "FileLoadingError",
    "RemoteLoadingError",
    "DocumentProcessingError",
    "RetrievalError",
    "RetrieverInitializationError",
    "SearchError",
    "ChainError",
    "ChainInitializationError",
    "ChainExecutionError",
    "ConversationMemoryError",
    "ValidationError",
    "APIError",
    "APIConnectionError",
    "APIAuthenticationError",
    "UIError",
    "FileUploadError",
    # Common utilities
    "safe_execute",
    "format_exception_context",
    "validate_file_path",
    "sanitize_input",
    "create_cache_key",
    "retry_with_backoff",
    "get_file_size_mb",
    "ensure_directory_exists",
    "truncate_text",
    "merge_dicts",
    "batch_process",
    "normalize_whitespace",
    "get_timestamp",
]
