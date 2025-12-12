"""
LangChain RAG Pipeline - A production-ready RAG system with multiple LLM backends.

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline
with support for OpenAI and Ollama models, multiple vector stores, and MCP server integration.
"""

__version__ = "2.0.0"
__author__ = "LangChain RAG Team"
__email__ = "team@langchain-rag.com"
__license__ = "MIT"

# RAG imports (consolidated from former core and services)
from .rag.pipeline import Pipeline, OpenAIPipeline, OllamaPipeline
from .rag.vector_db import (
    VectorStoreProxy,
    ChromaProxy,
    FaissProxy,
    create_vector_store_proxy
)

# Configuration and utilities
from .config.settings import get_config, Config
from .utils.exceptions import (
    PipelineError,
    LLMInitializationError,
    VectorStoreError,
    ConfigurationError
)

__all__ = [
    # Core classes
    "Pipeline",
    "OpenAIPipeline", 
    "OllamaPipeline",
    
    # Configuration
    "get_config",
    "Config",
    
    # Vector stores
    "VectorStoreProxy",
    "ChromaProxy",
    "FaissProxy",
    "create_vector_store_proxy",
    
    # Exceptions
    "PipelineError",
    "LLMInitializationError", 
    "VectorStoreError",
    "ConfigurationError",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "PACKAGE_INFO",
    "get_package_info",
]

# Package metadata
PACKAGE_INFO = {
    "name": "langchain-rag-pipeline",
    "version": __version__,
    "description": "Production-ready RAG pipeline with multiple LLM backends",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.9",
    "features": [
        "Multiple LLM backends (OpenAI, Ollama)",
        "Multiple vector stores (FAISS, ChromaDB)",
        "MCP server integration", 
        "Streamlit UI",
        "Docker deployment",
        "Production-ready logging and monitoring"
    ]
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO 