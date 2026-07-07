"""
Configuration management for the Ragstone.

This module provides centralized configuration management with environment variable
support, validation, and type safety.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from ..utils.exceptions import ConfigurationError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for vector databases."""

    default_type: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss").strip().lower()
    )
    chroma_persist_dir: str = "store/chroma_db"
    faiss_index_name: str = "faiss_index"
    # Qdrant backend: embedded local mode by default (no server needed);
    # set QDRANT_URL to talk to a real Qdrant server instead — same code
    # path, one env var (docker-compose.yml provides one).
    qdrant_path: str = field(
        default_factory=lambda: os.getenv("QDRANT_PATH", "store/qdrant")
    )
    qdrant_url: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL") or None
    )
    # Stable collection/table name for deployments that want the index to
    # survive restarts and be shared across processes. Unset (default) =
    # a unique name per pipeline, which is the safe multi-pipeline choice.
    collection_name: Optional[str] = field(
        default_factory=lambda: os.getenv("RAGSTONE_COLLECTION") or None
    )
    # pgvector backend: connection string to a Postgres with the pgvector
    # extension (docker-compose.yml provides one).
    pg_url: Optional[str] = field(
        default_factory=lambda: os.getenv("RAGSTONE_PG_URL") or None
    )
    # Texts per embedding request during ingestion; batches are issued
    # concurrently by embed_workers threads (see rag/embeddings.py).
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("RAGSTONE_EMBED_BATCH_SIZE", "500"))
    )
    embed_workers: int = field(
        default_factory=lambda: int(os.getenv("RAGSTONE_EMBED_WORKERS", "4"))
    )
    # Retrieval depth. k=6 looked better on the retrieval-slice metric
    # (hit rate 0.955 -> 0.980) but degraded END-TO-END quality at n=224:
    # the two extra chunks are mostly distractor text that dilutes the
    # prompt (faithfulness -2pp, multi-turn faithfulness -23pp, +46%
    # tokens). Kept at 4 — see Experiment 10.
    similarity_k: int = 4
    max_query_length: int = 10000
    # BM25's share in the ensemble retriever; the vector store gets the
    # rest. The default was validated by the eval harness (EXPERIMENTS.md).
    ensemble_bm25_weight: float = field(
        default_factory=lambda: float(os.getenv("RAGSTONE_BM25_WEIGHT", "0.4"))
    )

    def __post_init__(self):
        """Validate database configuration."""
        if self.default_type not in ["faiss", "chroma", "qdrant", "pgvector"]:
            raise ConfigurationError(
                f"Invalid database type: {self.default_type!r}. "
                "Valid values are 'faiss', 'chroma', 'qdrant', and "
                "'pgvector' (set via VECTOR_STORE_TYPE or config)."
            )
        if self.default_type == "pgvector" and not self.pg_url:
            raise ConfigurationError(
                "VECTOR_STORE_TYPE=pgvector requires RAGSTONE_PG_URL "
                "(e.g. postgresql+psycopg://user:pass@localhost:5432/ragstone)"
            )
        if self.batch_size <= 0:
            raise ConfigurationError("Batch size must be positive")
        if self.embed_workers <= 0:
            raise ConfigurationError("Embed workers must be positive")
        if self.similarity_k <= 0:
            raise ConfigurationError("Similarity k must be positive")
        if not (0.0 <= self.ensemble_bm25_weight <= 1.0):
            raise ConfigurationError("Ensemble BM25 weight must be within [0, 1]")


@dataclass
class LLMConfig:
    """Configuration for language models."""

    openai_models: List[str] = field(
        default_factory=lambda: ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
    )
    ollama_models: List[str] = field(
        default_factory=lambda: ["llama3", "phi4", "deepseek-r1:8b"]
    )
    # Embedding models configuration - model-aware selection
    # NOTE: Thanks to user discovery, we now try the LLM model directly first!
    # These complex mappings below are now just FALLBACKS if the direct approach fails.
    # Example: OllamaEmbeddings(model="llama3") works directly!
    model_embedding_preferences: Dict[str, List[str]] = field(
        default_factory=lambda: {
            # Ollama LLM models and their preferred embedding models (FALLBACK only)
            "llama3": ["nomic-embed-text", "all-minilm", "mxbai-embed-large"],
            "llama3.1": ["nomic-embed-text", "all-minilm", "mxbai-embed-large"],
            "phi4": ["nomic-embed-text", "all-minilm"],
            "deepseek-r1:8b": ["nomic-embed-text", "mxbai-embed-large"],
            "deepseek-r1:14b": ["nomic-embed-text", "mxbai-embed-large"],
            "deepseek-r1:32b": ["mxbai-embed-large", "nomic-embed-text"],
            "gemma": ["all-minilm", "nomic-embed-text"],
            "gemma3:12b": ["mxbai-embed-large", "nomic-embed-text"],
            # Fallback for any Ollama model
            "ollama_default": ["nomic-embed-text", "all-minilm", "mxbai-embed-large"],
        }
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    prefer_ollama_embeddings: bool = True  # Try Ollama first, fallback to OpenAI
    auto_detect_available_models: bool = True  # Detect available Ollama models
    default_temperature: float = 0.0
    # Applied to every LLM/embedding client (see models/base_model.py and
    # rag/embeddings.py): bounded retries on transient API errors, and a
    # request timeout so a hung call cannot hang the request forever.
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("RAGSTONE_LLM_MAX_RETRIES", "3"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("RAGSTONE_LLM_TIMEOUT", "60"))
    )
    # Upper bound on question length, checked before any API call is made.
    max_question_chars: int = field(
        default_factory=lambda: int(os.getenv("RAGSTONE_MAX_QUESTION_CHARS", "4000"))
    )
    # Model for the follow-up rephrase step. Rephrasing is a trivial task
    # that sits on the critical path (it runs before retrieval can start),
    # so a fast, cheap model here directly cuts follow-up latency. None
    # (the default) uses the main answer model.
    rephrase_model: Optional[str] = field(
        default_factory=lambda: os.getenv("RAGSTONE_REPHRASE_MODEL") or None
    )

    def __post_init__(self):
        """Validate LLM configuration."""
        if not (0.0 <= self.default_temperature <= 2.0):
            raise ConfigurationError("Temperature must be between 0.0 and 2.0")
        if self.max_retries < 0:
            raise ConfigurationError("Max retries must be non-negative")
        if self.timeout <= 0:
            raise ConfigurationError("Timeout must be positive")
        if self.max_question_chars <= 0:
            raise ConfigurationError("Max question chars must be positive")


@dataclass
class LoaderConfig:
    """Configuration for document loaders."""

    default_data_dir: str = "data"
    supported_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".csv", ".pdf", ".docx", ".md"]
    )
    max_file_size_mb: int = 100
    enable_ocr: bool = False
    # When set, document ingestion is confined to this directory tree —
    # recommended for MCP/API deployments, where clients choose data_dir.
    allowed_data_root: Optional[str] = field(
        default_factory=lambda: os.getenv("RAGSTONE_DATA_ROOT") or None
    )
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # Contextual chunk enrichment: "off", "source" (prepend the document
    # identity — free), or "llm" (prepend a generated situating sentence —
    # one utility-model call per chunk at ingest). Default "source":
    # measured at n=224 it lifted hit_rate +1.5pp, MRR +2.0, and
    # faithfulness +2.9pp for +5.7% tokens — retrieval and end-to-end
    # moved TOGETHER, unlike the k=6 trap. See Experiment 12.
    chunk_context: str = field(
        default_factory=lambda: (
            os.getenv("RAGSTONE_CHUNK_CONTEXT", "source").strip().lower()
        )
    )

    def __post_init__(self):
        """Validate loader configuration."""
        if self.max_file_size_mb <= 0:
            raise ConfigurationError("Max file size must be positive")
        if self.chunk_size <= 0:
            raise ConfigurationError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ConfigurationError("Chunk overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("Chunk overlap must be less than chunk size")
        if self.chunk_context not in ("off", "source", "llm"):
            raise ConfigurationError(
                f"chunk_context must be 'off', 'source', or 'llm', "
                f"got {self.chunk_context!r}"
            )


@dataclass
class APIConfig:
    """Configuration for external APIs."""

    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None
    openai_base_url: Optional[str] = None
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    def __post_init__(self):
        """Load and validate API configuration from environment."""
        # Load from environment if not provided
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_org_id is None:
            self.openai_org_id = os.getenv("OPENAI_ORG_ID")
        if self.openai_base_url is None:
            self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        # Validate OpenAI API key format if provided
        if self.openai_api_key and not self.openai_api_key.startswith("sk-"):
            logger.warning("OpenAI API key doesn't start with 'sk-', might be invalid")


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = False
    log_file_path: str = "logs/application.log"
    max_log_size_mb: int = 10
    backup_count: int = 5

    def __post_init__(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.level}")

        if self.log_to_file:
            # Ensure log directory exists
            log_path = Path(self.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class UIConfig:
    """Configuration for the user interface."""

    title: str = "Ragstone"
    page_icon: str = ""
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme_primary_color: str = "#FF6B6B"
    theme_background_color: str = "#FFFFFF"
    max_upload_size_mb: int = 200

    def __post_init__(self):
        """Validate UI configuration."""
        if self.layout not in ["centered", "wide"]:
            raise ConfigurationError("Layout must be 'centered' or 'wide'")
        if self.initial_sidebar_state not in ["auto", "expanded", "collapsed"]:
            raise ConfigurationError("Invalid sidebar state")


@dataclass
class CacheConfig:
    """Configuration for the response cache (exact-match, per session)."""

    enable_response_cache: bool = False
    response_cache_size: int = 100  # Max number of cached responses
    response_cache_ttl: int = 3600  # 1 hour in seconds


@dataclass
class MemoryConfig:
    """Configuration for conversation memory persistence.

    The default in-memory checkpointer keeps conversation history only for
    the life of the process. Set the backend to "sqlite" (and install the
    `sqlite` extra) to persist history across restarts.
    """

    checkpoint_backend: str = field(
        default_factory=lambda: os.getenv("RAGSTONE_CHECKPOINT_BACKEND", "memory")
        .strip()
        .lower()
    )
    checkpoint_db_path: str = field(
        default_factory=lambda: os.getenv(
            "RAGSTONE_CHECKPOINT_DB", "store/checkpoints.sqlite"
        )
    )

    def __post_init__(self):
        """Validate memory configuration."""
        valid = {"memory", "sqlite"}
        if self.checkpoint_backend not in valid:
            raise ConfigurationError(
                f"Invalid checkpoint backend: {self.checkpoint_backend!r}. "
                "Valid values are 'memory' and 'sqlite' "
                "(set via RAGSTONE_CHECKPOINT_BACKEND or config)."
            )


@dataclass
class Config:
    """Main configuration class that aggregates all configuration sections."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Global settings
    environment: str = "development"
    # None means "derive from environment"; an explicit value (e.g. from a
    # config file) is always respected.
    debug: Optional[bool] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.debug is None:
            self.debug = self.environment.lower() == "development"

        # Adjust logging level based on debug mode
        if self.debug and self.logging.level == "INFO":
            self.logging.level = "DEBUG"

        logger.info(f"Configuration initialized for {self.environment} environment")

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Config instance loaded from file.

        Raises:
            ConfigurationError: If file cannot be loaded or parsed.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Create config instances from dictionaries
            database_config = DatabaseConfig(**config_data.get("database", {}))
            llm_config = LLMConfig(**config_data.get("llm", {}))
            loader_config = LoaderConfig(**config_data.get("loader", {}))
            api_config = APIConfig(**config_data.get("api", {}))
            logging_config = LoggingConfig(**config_data.get("logging", {}))
            ui_config = UIConfig(**config_data.get("ui", {}))
            cache_config = CacheConfig(**config_data.get("cache", {}))
            memory_config = MemoryConfig(**config_data.get("memory", {}))

            # Create main config
            main_config = config_data.get("main", {})
            return cls(
                database=database_config,
                llm=llm_config,
                loader=loader_config,
                api=api_config,
                logging=logging_config,
                ui=ui_config,
                cache=cache_config,
                memory=memory_config,
                environment=main_config.get("environment", "development"),
                debug=main_config.get("debug"),
            )

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Sensitive values (API keys) are excluded.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "database": {
                "default_type": self.database.default_type,
                "chroma_persist_dir": self.database.chroma_persist_dir,
                "faiss_index_name": self.database.faiss_index_name,
                "qdrant_path": self.database.qdrant_path,
                "qdrant_url": self.database.qdrant_url,
                "collection_name": self.database.collection_name,
                # pg_url is omitted: connection strings embed credentials.
                "batch_size": self.database.batch_size,
                "embed_workers": self.database.embed_workers,
                "similarity_k": self.database.similarity_k,
                "ensemble_bm25_weight": self.database.ensemble_bm25_weight,
                "max_query_length": self.database.max_query_length,
            },
            "llm": {
                "openai_models": self.llm.openai_models,
                "ollama_models": self.llm.ollama_models,
                "model_embedding_preferences": self.llm.model_embedding_preferences,
                "openai_embedding_model": self.llm.openai_embedding_model,
                "prefer_ollama_embeddings": self.llm.prefer_ollama_embeddings,
                "auto_detect_available_models": self.llm.auto_detect_available_models,
                "default_temperature": self.llm.default_temperature,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
                "max_question_chars": self.llm.max_question_chars,
                "rephrase_model": self.llm.rephrase_model,
            },
            "loader": {
                "default_data_dir": self.loader.default_data_dir,
                "supported_extensions": self.loader.supported_extensions,
                "max_file_size_mb": self.loader.max_file_size_mb,
                "enable_ocr": self.loader.enable_ocr,
                "chunk_size": self.loader.chunk_size,
                "chunk_overlap": self.loader.chunk_overlap,
                "chunk_context": self.loader.chunk_context,
                "allowed_data_root": self.loader.allowed_data_root,
            },
            "api": {
                "ollama_base_url": self.api.ollama_base_url,
                # Note: Don't save sensitive API keys to file
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "date_format": self.logging.date_format,
                "log_to_file": self.logging.log_to_file,
                "log_file_path": self.logging.log_file_path,
                "max_log_size_mb": self.logging.max_log_size_mb,
                "backup_count": self.logging.backup_count,
            },
            "ui": {
                "title": self.ui.title,
                "page_icon": self.ui.page_icon,
                "layout": self.ui.layout,
                "initial_sidebar_state": self.ui.initial_sidebar_state,
                "theme_primary_color": self.ui.theme_primary_color,
                "theme_background_color": self.ui.theme_background_color,
                "max_upload_size_mb": self.ui.max_upload_size_mb,
            },
            "cache": {
                "enable_response_cache": self.cache.enable_response_cache,
                "response_cache_size": self.cache.response_cache_size,
                "response_cache_ttl": self.cache.response_cache_ttl,
            },
            "memory": {
                "checkpoint_backend": self.memory.checkpoint_backend,
                "checkpoint_db_path": self.memory.checkpoint_db_path,
            },
            "main": {
                "environment": self.environment,
                "debug": self.debug,
            },
        }

    def to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config_path: Path where to save the configuration file.
        """
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def validate(self) -> List[str]:
        """
        Validate the entire configuration.

        Returns:
            List of validation error messages. Empty list if all valid.
        """
        errors = []

        # A missing OpenAI key is not an error: Ollama-only deployments are
        # fully supported. The key is checked when an OpenAI pipeline is
        # actually constructed.
        if not self.api.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set — OpenAI pipelines will be "
                "unavailable (Ollama still works)."
            )

        # A missing data directory is not an error either: documents can
        # come from uploads, URLs, or Wikipedia.
        data_path = Path(self.loader.default_data_dir)
        if not data_path.exists():
            logger.warning(f"Default data directory does not exist: {data_path}")

        # Check vector store directory can be created
        try:
            Path(self.database.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"Cannot create vector store directory: {e}")

        return errors

    def setup_logging(self) -> None:
        """Set up logging based on configuration."""
        import logging.handlers

        # Get numeric level
        numeric_level = getattr(logging, self.logging.level.upper())

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(
            self.logging.format, datefmt=self.logging.date_format
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler if enabled
        if self.logging.log_to_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.log_file_path,
                maxBytes=self.logging.max_log_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count,
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logger.info(
            f"Logging configured: level={self.logging.level}, file={self.logging.log_to_file}"
        )


# Global configuration instance, created on first use. Deferring this means
# a bad environment variable surfaces as a clear ConfigurationError from the
# first get_config() call instead of crashing `import ragstone` itself.
config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        The global Config instance.
    """
    global config
    if config is None:
        config = Config()
    return config


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from file or use default.

    Args:
        config_path: Optional path to configuration file.

    Returns:
        Loaded configuration instance.
    """
    global config

    if config_path and Path(config_path).exists():
        config = Config.from_file(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    else:
        config = Config()
        logger.info("Using default configuration")

    # Validate configuration
    errors = config.validate()
    if errors:
        error_msg = "Configuration validation errors:\n" + "\n".join(
            f"- {error}" for error in errors
        )
        logger.error(error_msg)
        if not config.debug:  # Only raise in production
            raise ConfigurationError(error_msg)
        else:
            logger.warning("Continuing with invalid configuration in debug mode")

    # Setup logging
    config.setup_logging()

    return config
