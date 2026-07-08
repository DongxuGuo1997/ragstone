"""
Ragstone - A RAG system with multiple LLM backends.

This package provides a complete RAG (Retrieval-Augmented Generation) pipeline
with support for OpenAI and Ollama models, multiple vector stores, and MCP server integration.
"""

import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

# On macOS, faiss-cpu and torch (installed via the optional `rerank` extra)
# each bundle their own copy of the OpenMP runtime, and loading both aborts
# the process ("OMP: Error #15"). This is the standard coexistence
# workaround; it must be set before both libraries are loaded — which the
# lazy imports below guarantee, since nothing heavy loads at package import.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "2.1.0"
__author__ = "Dongxu Guo"
__email__ = "ericguohit@outlook.com"
__license__ = "MIT"

# The public names and where they live. Resolved lazily (PEP 562): importing
# the package costs milliseconds instead of seconds, because LangChain, faiss,
# and friends only load when a name that needs them is first touched. A
# config-only consumer (MCP client shim, docs tooling) never pays for them.
_LAZY_IMPORTS = {
    # Configuration
    "Config": ".config.settings",
    "get_config": ".config.settings",
    # Core classes
    "Pipeline": ".rag.pipeline",
    "OpenAIPipeline": ".rag.pipeline",
    "OllamaPipeline": ".rag.pipeline",
    "build_pipeline": ".rag.pipeline",
    # Vector stores
    "VectorStoreProxy": ".rag.vector_db",
    "ChromaProxy": ".rag.vector_db",
    "FaissProxy": ".rag.vector_db",
    "create_vector_store_proxy": ".rag.vector_db",
    # Exceptions
    "PipelineError": ".utils.exceptions",
    "LLMInitializationError": ".utils.exceptions",
    "VectorStoreError": ".utils.exceptions",
    "ConfigurationError": ".utils.exceptions",
}

if TYPE_CHECKING:  # give type checkers and IDEs the real symbols
    from .config.settings import Config, get_config
    from .rag.pipeline import (
        OllamaPipeline,
        OpenAIPipeline,
        Pipeline,
        build_pipeline,
    )
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


def __getattr__(name: str) -> Any:
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value  # cache so later accesses skip this hook
    return value


def __dir__() -> "list[str]":
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


__all__ = [
    # Core classes
    "Pipeline",
    "OpenAIPipeline",
    "OllamaPipeline",
    "build_pipeline",
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
    "name": "ragstone",
    "version": __version__,
    "description": "RAG pipeline with multiple LLM backends",
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.10",
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
