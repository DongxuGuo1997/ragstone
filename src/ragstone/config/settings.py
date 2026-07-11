"""
Configuration management for the Ragstone.

This module provides centralized configuration management with environment variable
support, validation, and type safety.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from ..utils.exceptions import ConfigurationError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def _tri_state_env(name: str) -> Optional[bool]:
    """Parse an on/off env var where UNSET is a distinct, meaningful state.

    None means "don't pass the knob at all" — the consumer preserves the
    underlying library's default — which a plain boolean cannot express.
    """
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("on", "true", "1", "yes"):
        return True
    if raw in ("off", "false", "0", "no"):
        return False
    raise ConfigurationError(f"{name} must be 'on' or 'off' (or unset), got {raw!r}")


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
    # Content-addressed embedding cache (Experiment 14): re-ingesting a
    # corpus embeds only changed chunks. Exact-match by model+text, so it
    # cannot alter retrieval results — disable only for benchmarking.
    embed_cache_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("RAGSTONE_EMBED_CACHE", "on").strip().lower() != "off"
        )
    )
    embed_cache_path: str = field(
        default_factory=lambda: os.getenv(
            "RAGSTONE_EMBED_CACHE_PATH", "store/embedding_cache.sqlite"
        )
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
        default_factory=lambda: [
            "qwen3.5:9b",
            "qwen3.6:35b",
            "gemma4:e4b",
            "gemma4:31b",
            "llama3",
        ]
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    prefer_ollama_embeddings: bool = True  # Try Ollama first, fallback to OpenAI
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
    # Model for the utility steps (rephrase/grade/rewrite/route), which
    # sit on the latency-critical path. None (the default) uses the main
    # answer model. A nano-tier model here is -40% rephrase latency but
    # degenerates on challenge turns ("are you sure?") — it echoes the
    # previous answer instead of forming a question, and prompt hardening
    # does not fix it (Experiment 18). Set only if your traffic is
    # standalone questions and pronoun follow-ups.
    rephrase_model: Optional[str] = field(
        default_factory=lambda: os.getenv("RAGSTONE_REPHRASE_MODEL") or None
    )
    # Thinking control for Ollama models (RAGSTONE_OLLAMA_REASONING).
    # Unset (None) keeps each model's own default. Measured July 2026 on
    # qwen3:14b: default thinking cost 91 output tokens / 4.7 s for a
    # one-word answer vs 3 tokens / 0.2 s with reasoning off — for RAG
    # answering this knob is the local latency/token switch.
    ollama_reasoning: Optional[bool] = field(
        default_factory=lambda: _tri_state_env("RAGSTONE_OLLAMA_REASONING")
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
    # Document-metadata cards: one LLM-extracted chunk per document
    # (title/authors/date, verbatim from the document head) indexed
    # alongside the content chunks. Exists because content retrieval
    # cannot answer "who wrote this?" — author blocks never rank for
    # "created/wrote" phrasing, and references sections are decoys that
    # do (Experiment 19). One utility-model call per document at ingest;
    # extraction failures skip the card, never break ingestion.
    metadata_cards: bool = field(
        default_factory=lambda: (
            os.getenv("RAGSTONE_METADATA_CARDS", "on").strip().lower()
            not in ("off", "false", "0", "no")
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
