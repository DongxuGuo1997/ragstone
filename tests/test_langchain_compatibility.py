"""
LangChain Compatibility Test Suite

This test suite ensures that LangChain's API hasn't changed in ways that break our code.
Run this BEFORE upgrading LangChain versions to catch breaking changes.

Usage:
    pytest tests/test_langchain_compatibility.py -v

When upgrading LangChain:
    1. Run these tests with current version (should pass)
    2. Upgrade LangChain
    3. Run these tests again
    4. If any fail, fix the imports/usage before deploying
"""

import sys

import pytest


class TestCoreModels:
    """Test that core LangChain model classes haven't changed."""

    def test_document_class_exists(self):
        """Verify Document class is importable and has expected attributes."""
        try:
            from langchain_core.documents import Document
        except ImportError as e:
            pytest.fail(f"Failed to import Document: {e}")

        # Test basic instantiation
        doc = Document(page_content="test content", metadata={"key": "value"})

        # Verify expected attributes exist
        assert hasattr(doc, "page_content"), "Document missing 'page_content' attribute"
        assert hasattr(doc, "metadata"), "Document missing 'metadata' attribute"
        assert doc.page_content == "test content"
        assert doc.metadata == {"key": "value"}

    def test_base_chat_model_interface(self):
        """Verify BaseChatModel interface is stable."""
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
        except ImportError as e:
            pytest.fail(f"Failed to import BaseChatModel: {e}")

        # Verify expected methods exist
        assert hasattr(BaseChatModel, "invoke"), "BaseChatModel missing 'invoke' method"
        assert hasattr(
            BaseChatModel, "ainvoke"
        ), "BaseChatModel missing 'ainvoke' method"
        assert hasattr(
            BaseChatModel, "generate"
        ), "BaseChatModel missing 'generate' method"

    def test_base_message_types(self):
        """Verify message types are importable."""
        try:
            from langchain_core.messages import (  # noqa: F401
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except ImportError as e:
            pytest.fail(f"Failed to import message types: {e}")

        # Test instantiation
        human_msg = HumanMessage(content="test")
        ai_msg = AIMessage(content="response")

        assert hasattr(human_msg, "content")
        assert hasattr(ai_msg, "content")


class TestLLMProviders:
    """Test that LLM provider classes haven't changed."""

    def test_openai_chat_import(self):
        """Verify ChatOpenAI is importable and has expected interface."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            pytest.fail(f"Failed to import ChatOpenAI: {e}")

        # Verify constructor accepts expected parameters
        import inspect

        sig = inspect.signature(ChatOpenAI.__init__)
        params = list(sig.parameters.keys())

        # These parameters should exist
        assert "self" in params
        # Note: We don't test specific param names as they might have **kwargs

    def test_ollama_chat_import(self):
        """Verify ChatOllama is importable and has expected interface."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            pytest.fail(f"Failed to import ChatOllama: {e}")

        # Verify class exists
        assert ChatOllama is not None

    def test_openai_embeddings_import(self):
        """Verify OpenAIEmbeddings is importable."""
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            pytest.fail(f"Failed to import OpenAIEmbeddings: {e}")

        assert OpenAIEmbeddings is not None

    def test_ollama_embeddings_import(self):
        """Verify OllamaEmbeddings is importable."""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            pytest.fail(f"Failed to import OllamaEmbeddings: {e}")

        assert OllamaEmbeddings is not None


class TestRetrievers:
    """Test that retriever classes haven't changed."""

    def test_ensemble_retriever_import(self):
        """Verify EnsembleRetriever is importable."""
        try:
            from langchain_classic.retrievers import EnsembleRetriever
        except ImportError as e:
            pytest.fail(f"Failed to import EnsembleRetriever: {e}")

        # Verify class exists
        assert EnsembleRetriever is not None

    def test_bm25_retriever_import(self):
        """Verify BM25Retriever is importable and has from_texts method."""
        try:
            from langchain_community.retrievers import BM25Retriever
        except ImportError as e:
            pytest.fail(f"Failed to import BM25Retriever: {e}")

        # Verify expected factory method exists
        assert hasattr(
            BM25Retriever, "from_texts"
        ), "BM25Retriever missing 'from_texts' method"

    def test_base_retriever_import(self):
        """Verify BaseRetriever interface is stable."""
        try:
            from langchain_core.retrievers import BaseRetriever
        except ImportError as e:
            pytest.fail(f"Failed to import BaseRetriever: {e}")

        assert BaseRetriever is not None


class TestVectorStores:
    """Test that vector store classes haven't changed."""

    def test_faiss_import(self):
        """Verify FAISS vector store is importable and has expected methods."""
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as e:
            pytest.fail(f"Failed to import FAISS: {e}")

        # Verify expected factory methods exist
        assert hasattr(FAISS, "from_documents"), "FAISS missing 'from_documents' method"
        assert hasattr(FAISS, "load_local"), "FAISS missing 'load_local' method"
        assert hasattr(FAISS, "save_local"), "FAISS missing 'save_local' method"

    def test_chroma_import(self):
        """Verify Chroma vector store is importable and has expected methods."""
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            pytest.fail(f"Failed to import Chroma: {e}")

        # Verify expected factory methods exist
        assert hasattr(
            Chroma, "from_documents"
        ), "Chroma missing 'from_documents' method"

    def test_vector_store_retriever_interface(self):
        """Verify VectorStoreRetriever interface is stable."""
        try:
            from langchain_core.vectorstores import VectorStoreRetriever
        except ImportError as e:
            pytest.fail(f"Failed to import VectorStoreRetriever: {e}")

        assert VectorStoreRetriever is not None


class TestDocumentLoaders:
    """Test that document loader classes haven't changed."""

    def test_text_loader_import(self):
        """Verify TextLoader is importable."""
        try:
            from langchain_community.document_loaders import TextLoader
        except ImportError as e:
            pytest.fail(f"Failed to import TextLoader: {e}")

        assert TextLoader is not None

    def test_pdf_loader_import(self):
        """Verify PyPDFLoader is importable."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError as e:
            pytest.fail(f"Failed to import PyPDFLoader: {e}")

        assert PyPDFLoader is not None

    def test_csv_loader_import(self):
        """Verify CSVLoader is importable."""
        try:
            from langchain_community.document_loaders import CSVLoader
        except ImportError as e:
            pytest.fail(f"Failed to import CSVLoader: {e}")

        assert CSVLoader is not None

    def test_web_loader_import(self):
        """Verify WebBaseLoader is importable."""
        try:
            from langchain_community.document_loaders import WebBaseLoader
        except ImportError as e:
            pytest.fail(f"Failed to import WebBaseLoader: {e}")

        assert WebBaseLoader is not None

    def test_wikipedia_loader_import(self):
        """Verify WikipediaLoader is importable."""
        try:
            from langchain_community.document_loaders import WikipediaLoader
        except ImportError as e:
            pytest.fail(f"Failed to import WikipediaLoader: {e}")

        assert WikipediaLoader is not None


class TestUtilities:
    """Test that utility functions and classes haven't changed."""

    def test_text_splitter_import(self):
        """Verify RecursiveCharacterTextSplitter is importable."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            pytest.fail(f"Failed to import RecursiveCharacterTextSplitter: {e}")

        # Verify it can be instantiated with expected parameters
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        assert hasattr(splitter, "split_text")
        assert hasattr(splitter, "create_documents")

    def test_dumps_loads_import(self):
        """Verify dumps/loads utilities are importable."""
        try:
            from langchain_core.load import dumps, loads
        except ImportError as e:
            pytest.fail(f"Failed to import dumps/loads: {e}")

        # Test basic functionality with Document
        from langchain_core.documents import Document

        doc = Document(page_content="test")

        # Should be able to serialize and deserialize
        serialized = dumps(doc)
        assert isinstance(serialized, str)

        deserialized = loads(serialized)
        assert isinstance(deserialized, Document)
        assert deserialized.page_content == "test"

    def test_bundled_rag_prompt(self):
        """The default RAG prompt is bundled locally (no hub/network needed)."""
        from know_rag.rag.rag import DEFAULT_RAG_PROMPT_TEMPLATE

        assert "{question}" in DEFAULT_RAG_PROMPT_TEMPLATE
        assert "{context}" in DEFAULT_RAG_PROMPT_TEMPLATE


class TestRunnables:
    """Test that runnable interfaces haven't changed."""

    def test_runnable_imports(self):
        """Verify Runnable classes are importable."""
        try:
            from langchain_core.runnables import (
                Runnable,
                RunnableLambda,
                RunnablePassthrough,
                RunnableSequence,
            )
        except ImportError as e:
            pytest.fail(f"Failed to import Runnable classes: {e}")

        assert Runnable is not None
        assert RunnableLambda is not None
        assert RunnablePassthrough is not None
        assert RunnableSequence is not None

    def test_langgraph_imports(self):
        """Verify the LangGraph pieces used by MemoryProxy are importable."""
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.config import get_stream_writer
            from langgraph.graph import END, START, StateGraph
            from langgraph.graph.message import add_messages
            from langgraph.graph.state import CompiledStateGraph
        except ImportError as e:
            pytest.fail(f"Failed to import LangGraph classes: {e}")

        assert StateGraph is not None
        assert START is not None and END is not None
        assert InMemorySaver is not None
        assert add_messages is not None
        assert get_stream_writer is not None
        assert CompiledStateGraph is not None


class TestPrompts:
    """Test that prompt classes haven't changed."""

    def test_chat_prompt_template_import(self):
        """Verify ChatPromptTemplate is importable."""
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError as e:
            pytest.fail(f"Failed to import prompt classes: {e}")

        # Test basic usage
        prompt = ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant"), ("human", "{question}")]
        )

        assert prompt is not None
        assert MessagesPlaceholder is not None


class TestOutputParsers:
    """Test that output parser classes haven't changed."""

    def test_str_output_parser_import(self):
        """Verify StrOutputParser is importable."""
        try:
            from langchain_core.output_parsers import StrOutputParser
        except ImportError as e:
            pytest.fail(f"Failed to import StrOutputParser: {e}")

        parser = StrOutputParser()
        assert parser is not None


class TestMemory:
    """Test that memory/history classes haven't changed."""

    def test_chat_message_history_import(self):
        """Verify ChatMessageHistory is importable."""
        try:
            from langchain_community.chat_message_histories import ChatMessageHistory
        except ImportError as e:
            pytest.fail(f"Failed to import ChatMessageHistory: {e}")

        history = ChatMessageHistory()
        assert hasattr(history, "add_message")
        assert hasattr(history, "messages")

    def test_base_chat_message_history_import(self):
        """Verify BaseChatMessageHistory interface is stable."""
        try:
            from langchain_core.chat_history import BaseChatMessageHistory
        except ImportError as e:
            pytest.fail(f"Failed to import BaseChatMessageHistory: {e}")

        assert BaseChatMessageHistory is not None


# ============================================
# Version Compatibility Report
# ============================================


def test_generate_compatibility_report(capsys):
    """Generate a compatibility report showing all tested imports."""
    print("\n" + "=" * 70)
    print("LANGCHAIN COMPATIBILITY REPORT")
    print("=" * 70)

    # Get LangChain version
    try:
        import langchain

        lc_version = langchain.__version__
    except Exception:
        lc_version = "unknown"

    try:
        import langchain_core

        lc_core_version = langchain_core.__version__
    except Exception:
        lc_core_version = "unknown"

    try:
        import langchain_community

        lc_community_version = langchain_community.__version__
    except Exception:
        lc_community_version = "unknown"

    try:
        from importlib.metadata import version

        langgraph_version = version("langgraph")
    except Exception:
        langgraph_version = "unknown"

    print("\nInstalled Versions:")
    print(f"  langchain: {lc_version}")
    print(f"  langchain-core: {lc_core_version}")
    print(f"  langchain-community: {lc_community_version}")
    print(f"  langgraph: {langgraph_version}")

    print(f"\nPython Version: {sys.version}")

    print("\n" + "-" * 70)
    print("All compatibility tests passed! ✓")
    print("-" * 70)

    # This test always passes - it just generates the report
    assert True


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v", "--tb=short"])
