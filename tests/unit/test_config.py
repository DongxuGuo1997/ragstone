"""
Unit tests for the configuration module.
"""

import json
import tempfile
from pathlib import Path

import pytest

from langchain_rag.config.settings import (
    Config,
    DatabaseConfig,
    LLMConfig,
    get_config,
    load_config,
)
from langchain_rag.utils.exceptions import ConfigurationError


class TestDatabaseConfig:
    """Test suite for DatabaseConfig."""

    def test_database_config_defaults(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.default_type == "faiss"
        assert config.chroma_persist_dir == "store/chroma_db"
        assert config.faiss_index_name == "faiss_index"
        assert config.batch_size == 100
        assert config.similarity_k == 4

    def test_database_config_validation_valid_type(self):
        """Test database config validation with valid type."""
        config = DatabaseConfig(default_type="chroma")
        assert config.default_type == "chroma"

    def test_database_config_validation_invalid_type(self):
        """Test database config validation with invalid type."""
        with pytest.raises(ConfigurationError):
            DatabaseConfig(default_type="invalid_type")


class TestLLMConfig:
    """Test suite for LLMConfig."""

    def test_llm_config_defaults(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert "gpt-3.5-turbo" in config.openai_models
        assert "llama3" in config.ollama_models
        assert config.default_temperature == 0.0
        assert config.max_retries == 3
        assert config.timeout == 60


class TestConfig:
    """Test suite for main Config class."""

    def test_config_initialization(self):
        """Test config initialization with defaults."""
        config = Config()
        assert config.database is not None
        assert config.llm is not None
        assert config.loader is not None
        assert config.api is not None
        assert config.logging is not None
        assert config.ui is not None

    def test_config_to_dict(self):
        """Test config conversion to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "database" in config_dict
        assert "llm" in config_dict
        assert "loader" in config_dict
        assert "api" in config_dict
        assert "logging" in config_dict
        assert "ui" in config_dict

    def test_config_to_file(self):
        """Test config saving to file."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.to_file(temp_path)
            assert temp_path.exists()

            # Verify file content
            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            assert isinstance(saved_data, dict)
            assert "database" in saved_data

        finally:
            temp_path.unlink(missing_ok=True)

    def test_config_from_file(self):
        """Test config loading from file."""
        test_config = {
            "database": {"default_type": "chroma", "batch_size": 50},
            "llm": {"default_temperature": 0.5},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_config, f)
            temp_path = Path(f.name)

        try:
            config = Config.from_file(temp_path)
            assert config.database.default_type == "chroma"
            assert config.database.batch_size == 50
            assert config.llm.default_temperature == 0.5

        finally:
            temp_path.unlink(missing_ok=True)


class TestConfigFunctions:
    """Test suite for configuration utility functions."""

    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert isinstance(config, Config)
        assert config.database is not None

    def test_load_config_default(self):
        """Test load_config with default parameters."""
        config = load_config()
        assert isinstance(config, Config)

    def test_load_config_nonexistent_file(self):
        """Test load_config with non-existent file."""
        config = load_config("nonexistent_config.json")
        assert isinstance(config, Config)
        # Should return default config when file doesn't exist
