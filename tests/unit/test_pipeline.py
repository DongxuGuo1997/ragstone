"""
Unit tests for the pipeline module.
"""

from unittest.mock import Mock, patch

import pytest

from ragstone.rag.pipeline import OllamaPipeline, OpenAIPipeline, Pipeline


class TestPipeline:
    """Test suite for the base Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with default parameters."""
        pipeline = Pipeline()
        assert pipeline is not None
        assert pipeline.texts is None
        assert pipeline._retriever is None
        assert pipeline._chain is None

    def test_pipeline_initialization_with_vector_store_type(self):
        """Test pipeline initialization with specific vector store type."""
        pipeline = Pipeline(vector_store_type="faiss")
        assert pipeline is not None
        assert pipeline.vector_db is not None
        assert pipeline.vector_db.type == "FAISS"

    def test_pipeline_initialization_with_chroma(self):
        """Test pipeline initialization with ChromaDB."""
        pipeline = Pipeline(vector_store_type="chroma")
        assert pipeline is not None
        assert pipeline.vector_db is not None
        assert pipeline.vector_db.type == "Chroma"

    def test_pipeline_invalid_vector_store_type(self):
        """Test pipeline initialization with invalid vector store type."""
        with pytest.raises(ValueError, match="Unsupported vector store type"):
            Pipeline(vector_store_type="invalid_type")


class TestOpenAIPipeline:
    """Test suite for the OpenAIPipeline class."""

    def test_openai_pipeline_initialization(self):
        """Test OpenAI pipeline initialization."""
        pipeline = OpenAIPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "llm_proxy")
        assert pipeline.llm_proxy is not None

    def test_openai_pipeline_with_custom_model(self):
        """Test OpenAI pipeline with custom model."""
        pipeline = OpenAIPipeline(model="gpt-4")
        assert pipeline is not None
        # Note: In real implementation, we'd verify the model is set correctly

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_pipeline_missing_api_key(self):
        """Test OpenAI pipeline fails without API key."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            OpenAIPipeline()


class TestOllamaPipeline:
    """Test suite for the OllamaPipeline class."""

    def test_ollama_pipeline_initialization(self):
        """Test Ollama pipeline initialization."""
        pipeline = OllamaPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, "llm_proxy")
        assert pipeline.llm_proxy is not None

    def test_ollama_pipeline_with_custom_model(self):
        """Test Ollama pipeline with custom model."""
        pipeline = OllamaPipeline(model="llama3")
        assert pipeline is not None
        # Note: In real implementation, we'd verify the model is set correctly


class TestPipelineDocumentLoading:
    """Test suite for document loading functionality."""

    def test_load_and_split_no_sources(self, tmp_path):
        """Test load_and_split with no data sources.

        Points at an empty tmp dir: the default "data/" may legitimately
        hold documents on a dev machine (e.g. for the compose stack), and
        this test is about the no-sources behavior, not the working tree.
        """
        pipeline = Pipeline()
        result = pipeline.load_and_split(data_dir=str(tmp_path))
        assert result is None
        assert pipeline.texts is None

    def test_load_and_split_with_data_dir(self):
        """Test load_and_split with data directory."""
        from langchain_core.documents import Document

        pipeline = Pipeline()

        # Replace the pipeline's loader instances (they are created lazily in
        # __init__, so patching the classes after construction has no effect).
        # Real Documents, not Mocks: the enrichment step reads page_content
        # and metadata for every chunk.
        pipeline.local_loader = Mock()
        pipeline.local_loader.get_documents.return_value = [
            Document(page_content="test content", metadata={"source": "t.md"})
        ]
        pipeline.remote_loader = Mock()
        pipeline.remote_loader.get_documents.return_value = []

        with patch("ragstone.rag.pipeline.split_documents") as mock_split:
            mock_split.return_value = [
                Document(page_content="split content", metadata={"source": "t.md"})
            ]
            result = pipeline.load_and_split(data_dir="test_data")

            assert result is not None
            assert len(result) == 1
            assert pipeline.texts is not None
            # The default chunk-context mode stamps the document identity
            # into the chunk text (Experiment 12).
            assert "split content" in result[0].page_content
            assert "t.md" in result[0].page_content
