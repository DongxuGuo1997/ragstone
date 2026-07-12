"""Boot-time environment checks: refuse to serve half-working (ROADMAP 5.11).

get_config() already fails fast on malformed VALUES — bad types, ranges,
profile invariants. What it cannot see is the ENVIRONMENT those values
point at: a vector-store backend whose extra was never installed, a
Qdrant/Postgres/Ollama endpoint nobody is listening on, a pinned model
that was never pulled, a provider with no credentials. Today each of
those surfaces as a stack trace on the first unlucky request — possibly
hours after boot, possibly mid-demo.

run_boot_checks() probes all of it up front and reports EVERY problem in
one message, each with the command that fixes it. It runs in the server
entry points (ragstone-api, ragstone-mcp), not in create_app(), so test
suites and embedders that build the app directly are never coupled to a
live environment. RAGSTONE_BOOT_CHECKS=off skips it entirely.

Probes only ever touch endpoints the operator configured (Ollama,
Qdrant, Postgres) — never any external service, so the checks are safe
under the no-egress profile.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils.exceptions import ConfigurationError
from .settings import Config, get_config

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 3


def _import_ok(module_name: str) -> bool:
    """True when the module imports — the 'is the extra installed' probe."""
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False


def _http_reachable(url: str) -> bool:
    """True when SOMETHING answers HTTP at the URL.

    Reachability, not health: any HTTP response — even a 401 or 404 —
    proves a listener; only connection refusal/timeout proves absence.
    """
    try:
        with urllib.request.urlopen(url, timeout=_PROBE_TIMEOUT_S):
            return True
    except urllib.error.HTTPError:
        return True  # an HTTP error IS a response from a live server
    except Exception:
        return False


def _ollama_tags(base_url: str) -> Optional[List[str]]:
    """Model names the Ollama server has pulled, or None if it's not up."""
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/api/tags", timeout=_PROBE_TIMEOUT_S
        ) as response:
            payload = json.load(response)
        return [model.get("name", "") for model in payload.get("models", [])]
    except Exception:
        return None


def _pg_connect_ok(pg_url: str) -> Tuple[bool, str]:
    """Attempt a Postgres connection; (ok, error text when not)."""
    try:
        import psycopg

        # The config URL is SQLAlchemy-form (postgresql+psycopg://...);
        # psycopg itself wants the plain scheme.
        dsn = pg_url.replace("postgresql+psycopg://", "postgresql://", 1)
        psycopg.connect(dsn, connect_timeout=_PROBE_TIMEOUT_S).close()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _llm_path_problems(config: Config, tags: Optional[List[str]]) -> List[str]:
    """At least one answer path must exist before the server takes traffic."""
    ollama_up = tags is not None
    if config.profile == "local":
        if not ollama_up:
            return [
                f"RAGSTONE_PROFILE=local but Ollama did not respond at "
                f"{config.api.ollama_base_url} — the local profile has no "
                f"cloud fallback. Fix: start it (`ollama serve`)."
            ]
        return []
    if not ollama_up and not os.getenv("OPENAI_API_KEY"):
        return [
            f"No answer path is available: OPENAI_API_KEY is not set and "
            f"Ollama did not respond at {config.api.ollama_base_url}. "
            f"Fix: set OPENAI_API_KEY or start Ollama (`ollama serve`)."
        ]
    return []


def _model_problems(config: Config, tags: Optional[List[str]]) -> List[str]:
    """Pinned/required models must actually be pulled, not just configured."""
    problems = []
    pinned = config.llm.ollama_embed_model
    if pinned and tags is not None and pinned not in tags:
        # A pinned embedder is honored or fails loud (rag/embeddings.py);
        # catching it at boot beats catching it on the first ingest.
        problems.append(
            f"RAGSTONE_OLLAMA_EMBED_MODEL={pinned!r} is not among the "
            f"models Ollama has pulled. Fix: `ollama pull {pinned}`."
        )
    if config.profile == "local" and tags is not None:
        from ..rag.embeddings import _DEDICATED_MODELS

        if not any(model in tags for model in _DEDICATED_MODELS):
            problems.append(
                "RAGSTONE_PROFILE=local but no dedicated embedding model "
                "is pulled, and the profile forbids the OpenAI fallback — "
                "ingestion would fail. Fix: `ollama pull embeddinggemma`."
            )
    return problems


def _store_problems(config: Config) -> List[str]:
    """The selected vector-store backend must be installed and reachable."""
    backend = config.database.default_type
    extras = {
        "qdrant": ("qdrant_client", "qdrant"),
        "pgvector": ("langchain_postgres", "pgvector"),
    }
    if backend in extras:
        module_name, extra = extras[backend]
        if not _import_ok(module_name):
            return [
                f"VECTOR_STORE_TYPE={backend} but the '{extra}' extra is "
                f'not installed. Fix: pip install -e ".[{extra}]".'
            ]
    if backend == "qdrant" and config.database.qdrant_url:
        if not _http_reachable(config.database.qdrant_url):
            return [
                f"QDRANT_URL={config.database.qdrant_url} did not respond. "
                f"Fix: start Qdrant (`docker compose up -d qdrant`) or "
                f"unset QDRANT_URL to use embedded mode."
            ]
    if backend == "pgvector":
        ok, error = _pg_connect_ok(config.database.pg_url or "")
        if not ok:
            return [
                f"VECTOR_STORE_TYPE=pgvector but Postgres at RAGSTONE_PG_URL "
                f"refused the connection ({error}). Fix: start it "
                f"(`docker compose up -d postgres`) and check the URL."
            ]
    if backend == "chroma" and not _import_ok("chromadb"):
        return [
            "VECTOR_STORE_TYPE=chroma but chromadb is not importable. Fix: pip install chromadb."
        ]
    if backend == "faiss" and not _import_ok("faiss"):
        return [
            "VECTOR_STORE_TYPE=faiss but faiss is not importable. Fix: pip install faiss-cpu."
        ]
    return []


def _checkpoint_problems(config: Config) -> List[str]:
    """Persistent memory needs its extra before the first conversation."""
    if config.memory.checkpoint_backend == "sqlite" and not _import_ok(
        "langgraph.checkpoint.sqlite"
    ):
        return [
            "RAGSTONE_CHECKPOINT_BACKEND=sqlite but the 'sqlite' extra is "
            'not installed. Fix: pip install -e ".[sqlite]".'
        ]
    return []


def _misc_problems(config: Config) -> List[str]:
    """Data-root confinement and observability wiring."""
    problems = []
    data_root = config.loader.allowed_data_root
    if data_root and not Path(data_root).is_dir():
        problems.append(
            f"RAGSTONE_DATA_ROOT={data_root!r} does not exist or is not a "
            f"directory. Fix: create it, or unset the variable."
        )
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and not _import_ok(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    ):
        problems.append(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the 'otel' extra is not "
            "installed — the server would run with tracing silently absent. "
            'Fix: pip install -e ".[otel]", or unset the endpoint.'
        )
    return problems


def boot_check_problems(config: Config) -> List[str]:
    """Every environment problem found, each with its fix. Empty = healthy."""
    tags = _ollama_tags(config.api.ollama_base_url)
    return [
        *_llm_path_problems(config, tags),
        *_model_problems(config, tags),
        *_store_problems(config),
        *_checkpoint_problems(config),
        *_misc_problems(config),
    ]


def run_boot_checks(config: Optional[Config] = None) -> None:
    """Probe the environment; raise with ALL problems or return quietly.

    Called from the server entry points. RAGSTONE_BOOT_CHECKS=off skips
    everything — the escape hatch for environments the probes misjudge.
    """
    if os.getenv("RAGSTONE_BOOT_CHECKS", "on").strip().lower() == "off":
        logger.info("Boot checks skipped (RAGSTONE_BOOT_CHECKS=off)")
        return
    problems = boot_check_problems(config or get_config())
    if problems:
        for problem in problems:
            logger.error("Boot check failed: %s", problem)
        raise ConfigurationError(
            "Refusing to start with a half-working environment "
            f"({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems)
        )
    logger.info("Boot checks passed")
