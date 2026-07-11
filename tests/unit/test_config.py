"""
Unit tests for the configuration module.
"""

import pytest

from ragstone.config.settings import (
    Config,
    DatabaseConfig,
    LLMConfig,
    get_config,
)
from ragstone.utils.exceptions import ConfigurationError


class TestDatabaseConfig:
    """Test suite for DatabaseConfig."""

    def test_database_config_defaults(self):
        """Test default database configuration."""
        config = DatabaseConfig()
        assert config.default_type == "faiss"
        assert config.chroma_persist_dir == "store/chroma_db"
        assert config.faiss_index_name == "faiss_index"
        assert config.batch_size == 500  # texts per embedding request
        assert config.embed_workers == 4  # concurrent embedding batches
        assert config.similarity_k == 4  # k=6 degraded end-to-end quality (Exp 10)

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
        assert "gpt-4o-mini" in config.openai_models
        assert "llama3" in config.ollama_models
        assert config.default_temperature == 0.0
        assert config.max_retries == 3
        assert config.timeout == 60


class TestOllamaReasoningConfig:
    """RAGSTONE_OLLAMA_REASONING is tri-state: unset must stay distinct
    from off (None = don't pass the knob, preserve the model default)."""

    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("RAGSTONE_OLLAMA_REASONING", raising=False)
        assert LLMConfig().ollama_reasoning is None

    @pytest.mark.parametrize("raw", ["on", "true", "1", "yes", " ON "])
    def test_on_variants(self, monkeypatch, raw):
        monkeypatch.setenv("RAGSTONE_OLLAMA_REASONING", raw)
        assert LLMConfig().ollama_reasoning is True

    @pytest.mark.parametrize("raw", ["off", "false", "0", "no", " Off "])
    def test_off_variants(self, monkeypatch, raw):
        monkeypatch.setenv("RAGSTONE_OLLAMA_REASONING", raw)
        assert LLMConfig().ollama_reasoning is False

    def test_empty_is_none(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_OLLAMA_REASONING", "  ")
        assert LLMConfig().ollama_reasoning is None

    def test_garbage_raises(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_OLLAMA_REASONING", "maybe")
        with pytest.raises(ConfigurationError):
            LLMConfig()


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


class TestConfigFunctions:
    """Test suite for configuration utility functions."""

    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        assert isinstance(config, Config)
        assert config.database is not None


class TestLocalProfileValidation:
    """RAGSTONE_PROFILE=local must fail CLOSED at boot: a no-egress
    deployment with a non-loopback endpoint must refuse to start."""

    def test_unset_profile_is_default(self, monkeypatch):
        monkeypatch.delenv("RAGSTONE_PROFILE", raising=False)
        assert Config().profile == ""

    def test_unknown_profile_raises(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_PROFILE", "hybrid")
        with pytest.raises(ConfigurationError, match="RAGSTONE_PROFILE"):
            Config()

    def test_local_profile_accepts_loopback_defaults(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_PROFILE", "local")
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert Config().profile == "local"

    def test_langsmith_tracing_is_refused(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_PROFILE", "local")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        with pytest.raises(ConfigurationError, match="LangSmith"):
            Config()

    def test_non_loopback_otel_endpoint_is_refused(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_PROFILE", "local")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.corp:4318")
        with pytest.raises(ConfigurationError, match="loopback OTLP"):
            Config()

    def test_loopback_otel_endpoint_is_accepted(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_PROFILE", "local")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
        assert Config().profile == "local"

    @pytest.mark.parametrize(
        "var, value",
        [
            ("OLLAMA_BASE_URL", "http://gpu-box.internal:11434"),
            ("QDRANT_URL", "http://qdrant.corp:6333"),
            ("RAGSTONE_PG_URL", "postgresql+psycopg://u:p@db.corp:5432/rag"),
        ],
    )
    def test_non_loopback_endpoints_are_refused(self, monkeypatch, var, value):
        monkeypatch.setenv("RAGSTONE_PROFILE", "local")
        monkeypatch.setenv(var, value)
        with pytest.raises(ConfigurationError, match="loopback"):
            Config()

    def test_hostnames_are_not_resolved(self, monkeypatch):
        # A name that HAPPENS to resolve to loopback is still refused:
        # the validator must not do DNS, and rebinding makes name-based
        # trust worthless (same stance as the SSRF guard).
        from ragstone.config.settings import _is_loopback_url

        assert _is_loopback_url("http://localhost:11434")
        assert _is_loopback_url("http://127.0.0.1:8000")
        assert _is_loopback_url("postgresql+psycopg://u:p@127.0.0.1/db")
        assert not _is_loopback_url("http://my-own-loopback-alias:11434")
        assert not _is_loopback_url("http://10.0.0.5:11434")


class TestConfigRobustness:
    """Fixes for config bugs found in review: debug override, env
    normalization, and Ollama-only deployments."""

    def test_explicit_debug_false_is_respected(self):
        config = Config(environment="development", debug=False)
        assert config.debug is False

    def test_debug_derived_from_environment_when_unset(self):
        assert Config(environment="development").debug is True
        assert Config(environment="production").debug is False

    def test_vector_store_type_env_is_normalized(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_TYPE", " FAISS ")
        assert DatabaseConfig().default_type == "faiss"

    def test_invalid_vector_store_type_names_valid_values(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_TYPE", "chromadb")
        with pytest.raises(ConfigurationError, match="faiss"):
            DatabaseConfig()
