"""
RAG module for the LangChain RAG pipeline.

This module contains all the RAG-related functionality including pipelines,
vector stores, document loaders, splitters, memory, and RAG operations.
"""

# Agentic RAG (LLM-driven retrieval loop)
from .agent import AgentRagChain

# Corrective RAG (self-grading retrieval with bounded retry)
from .corrective import CorrectiveRagChain

# Document loading and processing (from former services)
from .loader import LocalLoader, RemoteLoader
from .memory import MemoryProxy, SimpleTextRetriever

# Pipeline classes (from former core)
from .pipeline import OllamaPipeline, OpenAIPipeline, Pipeline

# RAG operations (from former services)
from .rag import RagProxy
from .splitter import split_documents

# Vector database and embeddings (from former services)
from .vector_db import (
    ChromaProxy,
    FaissProxy,
    VectorStoreProxy,
    create_vector_store_proxy,
)

__all__ = [
    # Pipeline classes
    "Pipeline",
    "OpenAIPipeline",
    "OllamaPipeline",
    # Memory and retrieval
    "MemoryProxy",
    "SimpleTextRetriever",
    # Vector databases
    "VectorStoreProxy",
    "ChromaProxy",
    "FaissProxy",
    "create_vector_store_proxy",
    # Document loading and processing
    "LocalLoader",
    "RemoteLoader",
    "split_documents",
    # RAG operations
    "RagProxy",
    "AgentRagChain",
    "CorrectiveRagChain",
]
