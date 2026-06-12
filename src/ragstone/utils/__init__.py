"""
Utility functions and helpers for the RAG pipeline.

This module contains the exception hierarchy and chain utilities.
"""

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
]
