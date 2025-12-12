"""
Pytest configuration and shared fixtures for LangChain RAG Pipeline tests.
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator, Dict, Any, List

# Set test environment variables
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-testing"
os.environ["VECTOR_STORE_TYPE"] = "faiss"
os.environ["MCP_LOG_LEVEL"] = "DEBUG"
os.environ["ENVIRONMENT"] = "testing"

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "vector_store_type": "faiss",
        "test_data_dir": "tests/data",
        "temp_store_dir": "tests/temp_store",
        "api_key": "sk-test-dummy-key-for-testing"
    }

@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_documents() -> List[str]:
    """Provide sample document content for testing."""
    return [
        "This is a test document about artificial intelligence and machine learning.",
        "The LangChain framework provides tools for building AI applications.",
        "Vector databases are used for similarity search in RAG systems.",
        "Python is a popular programming language for data science and AI."
    ]

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock response from OpenAI API for testing purposes."
                }
            }
        ]
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables automatically for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-for-testing")
    monkeypatch.setenv("VECTOR_STORE_TYPE", "faiss")
    monkeypatch.setenv("MCP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENVIRONMENT", "testing") 