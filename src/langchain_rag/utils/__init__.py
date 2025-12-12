"""
Utility functions and helpers for the LangChain RAG pipeline.

This module contains exceptions, chain utilities, and common helpers.
"""

# Import all exceptions
from .exceptions import (
    PipelineError,
    ConfigurationError,
    LLMError,
    LLMInitializationError,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorStoreOperationError,
    EmbeddingError,
    DocumentLoadingError,
    FileLoadingError,
    RemoteLoadingError,
    DocumentProcessingError,
    RetrievalError,
    RetrieverInitializationError,
    SearchError,
    ChainError,
    ChainInitializationError,
    ChainExecutionError,
    MemoryError,
    ValidationError,
    APIError,
    APIConnectionError,
    APIAuthenticationError,
    UIError,
    FileUploadError,
)

# Import common utilities
from .common import (
    safe_execute,
    format_exception_context,
    validate_file_path,
    sanitize_input,
    create_cache_key,
    retry_with_backoff,
    get_file_size_mb,
    ensure_directory_exists,
    truncate_text,
    merge_dicts,
    batch_process,
    normalize_whitespace,
    get_timestamp,
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
    "MemoryError",
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