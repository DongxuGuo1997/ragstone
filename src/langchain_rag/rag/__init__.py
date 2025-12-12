"""
RAG module for the LangChain RAG pipeline.

This module contains all the RAG-related functionality including pipelines,
vector stores, document loaders, splitters, memory, and RAG operations.
"""

# Pipeline classes (from former core)
from .pipeline import Pipeline, OpenAIPipeline, OllamaPipeline
from .memory import MemoryProxy, SimpleTextRetriever
from .ensemble import ensemble_retriever_from_docs

# Vector database and embeddings (from former services)
from .vector_db import (
    VectorStoreProxy,
    ChromaProxy,
    FaissProxy,
    create_vector_store_proxy
)

# Document loading and processing (from former services)
from .loader import LocalLoader, RemoteLoader
from .splitter import split_documents

# RAG operations (from former services)
from .rag import RagProxy

__all__ = [
    # Pipeline classes
    "Pipeline",
    "OpenAIPipeline",
    "OllamaPipeline",
    
    # Memory and retrieval
    "MemoryProxy",
    "SimpleTextRetriever",
    "ensemble_retriever_from_docs",
    
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
] 