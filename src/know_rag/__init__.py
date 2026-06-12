"""
Know-RAG - A RAG system with multiple LLM backends.

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline
with support for OpenAI and Ollama models, multiple vector stores, and MCP server integration.
"""

import os
import sys

# On macOS, faiss-cpu and torch (installed via the optional `rerank` extra)
# each bundle their own copy of the OpenMP runtime, and loading both aborts
# the process ("OMP: Error #15"). This is the standard coexistence
# workaround; it must be set before both libraries are loaded.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "2.0.0"
__author__ = "Dongxu Guo"
__email__ = "ericguohit@outlook.com"
__license__ = "MIT"

# Configuration and utilities
from .config.settings import Config, get_config

# RAG imports (consolidated from former core and services)
from .rag.pipeline import OllamaPipeline, OpenAIPipeline, Pipeline
from .rag.vector_db import (
    ChromaProxy,
    FaissProxy,
    VectorStoreProxy,
    create_vector_store_proxy,
)
from .utils.exceptions import (
    ConfigurationError,
    LLMInitializationError,
    PipelineError,
    VectorStoreError,
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
    "name": "know-rag",
    "version": __version__,
    "description": "RAG pipeline with multiple LLM backends",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.9",
    "features": [
        "Multiple LLM backends (OpenAI, Ollama)",
        "Multiple vector stores (FAISS, ChromaDB)",
        "MCP server integration",
        "Streamlit UI",
        "Configurable logging",
    ],
}


def get_package_info():
    """Get package information."""
    return PACKAGE_INFO
