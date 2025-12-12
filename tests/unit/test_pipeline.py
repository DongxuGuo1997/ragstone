"""
Unit tests for the pipeline module.
"""

import pytest
from unittest.mock import Mock, patch
from langchain_rag.rag.pipeline import Pipeline, OpenAIPipeline, OllamaPipeline
from langchain_rag.utils.exceptions import PipelineError, LLMInitializationError


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
        assert hasattr(pipeline, 'LLM')
        assert pipeline.LLM is not None
    
    def test_openai_pipeline_with_custom_model(self):
        """Test OpenAI pipeline with custom model."""
        pipeline = OpenAIPipeline(model="gpt-4")
        assert pipeline is not None
        # Note: In real implementation, we'd verify the model is set correctly
    
    @patch.dict('os.environ', {}, clear=True)
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
        assert hasattr(pipeline, 'LLM')
        assert pipeline.LLM is not None
    
    def test_ollama_pipeline_with_custom_model(self):
        """Test Ollama pipeline with custom model."""
        pipeline = OllamaPipeline(model="llama3")
        assert pipeline is not None
        # Note: In real implementation, we'd verify the model is set correctly


class TestPipelineDocumentLoading:
    """Test suite for document loading functionality."""
    
    def test_load_and_split_no_sources(self):
        """Test load_and_split with no data sources."""
        pipeline = Pipeline()
        result = pipeline.load_and_split()
        assert result is None
        assert pipeline.texts is None
    
    @patch('langchain_rag.rag.loader.LocalLoader')
    @patch('langchain_rag.rag.loader.RemoteLoader')
    def test_load_and_split_with_data_dir(self, mock_remote_loader, mock_local_loader):
        """Test load_and_split with data directory."""
        # Mock loader behavior
        mock_local_instance = Mock()
        mock_local_instance.get_documents.return_value = [Mock(page_content="test content")]
        mock_local_loader.return_value = mock_local_instance
        
        mock_remote_instance = Mock()
        mock_remote_instance.get_documents.return_value = []
        mock_remote_loader.return_value = mock_remote_instance
        
        pipeline = Pipeline()
        
        with patch('langchain_rag.rag.splitter.split_documents') as mock_split:
            mock_split.return_value = [Mock(page_content="split content")]
            result = pipeline.load_and_split(data_dir="test_data")
            
            assert result is not None
            assert len(result) == 1
            assert pipeline.texts is not None 