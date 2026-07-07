"""Content-addressed embedding cache (ROADMAP 1.6 + 4.3).

Embedding is the expensive step of ingestion: one API call per batch,
re-paid in full even when a single document changed. This cache keys each
chunk's vector by SHA-256 of (embedding model, exact chunk text), so
re-ingesting a corpus embeds only what actually changed — incremental
indexing without per-backend index-mutation bookkeeping: the index is
rebuilt from vectors, and vectors are only computed for cache misses.

Correctness properties:
- Exact-match by construction: a different model or a single changed
  character is a different key. There is nothing fuzzy to be wrong about.
- Vectors round-trip exactly (float64 packing), so a cached ingest builds
  a bit-identical index — retrieval metrics cannot move (Experiment 14).
- Enrichment happens BEFORE embedding, so enriched text is what gets
  keyed; toggling RAGSTONE_CHUNK_CONTEXT changes keys, as it must.

Storage is a single SQLite file (RAGSTONE_EMBED_CACHE_PATH, default
store/embedding_cache.sqlite); disable with RAGSTONE_EMBED_CACHE=off.
"""

import hashlib
import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Serialize writers within this process; SQLite serializes across
# processes itself. Reads are cheap and also funneled here for simplicity.
_LOCK = threading.Lock()


def model_id_for(embeddings: Any) -> str:
    """Stable identity of the embedding model for cache keys.

    Same convention as the pipeline's corpus fingerprint: the provider
    class plus its model name — two models must never share vectors.
    """
    model = getattr(embeddings, "model", None) or getattr(embeddings, "model_name", "")
    return f"{type(embeddings).__name__}:{model}"


def _key(model_id: str, text: str) -> str:
    payload = f"{model_id}\x00{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pack(vector: List[float]) -> bytes:
    return struct.pack(f"{len(vector)}d", *vector)


def _unpack(blob: bytes) -> List[float]:
    return list(struct.unpack(f"{len(blob) // 8}d", blob))


class EmbeddingCache:
    """SQLite-backed vector cache; safe for concurrent pipelines."""

    def __init__(self, path: str) -> None:
        self._path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get_many(self, model_id: str, texts: List[str]) -> Dict[int, List[float]]:
        """Cached vectors by input index, for the texts that have one."""
        keys = [_key(model_id, text) for text in texts]
        found: Dict[int, List[float]] = {}
        with _LOCK, self._connect() as conn:
            for i in range(0, len(keys), 500):  # stay under host param limits
                batch = keys[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT key, vector FROM embeddings "  # noqa: S608
                    f"WHERE key IN ({placeholders})",
                    batch,
                ).fetchall()
                by_key = {key: blob for key, blob in rows}
                for j, key in enumerate(batch):
                    if key in by_key:
                        found[i + j] = _unpack(by_key[key])
        return found

    def put_many(
        self, model_id: str, texts: List[str], vectors: List[List[float]]
    ) -> None:
        with _LOCK, self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                [
                    (_key(model_id, text), _pack(vector))
                    for text, vector in zip(texts, vectors)
                ],
            )


_cache: Optional[EmbeddingCache] = None
_cache_path: Optional[str] = None


def get_embedding_cache() -> Optional[EmbeddingCache]:
    """Process-wide cache instance, or None when disabled by config."""
    from ..config.settings import get_config

    db_cfg = get_config().database
    if not db_cfg.embed_cache_enabled:
        return None
    global _cache, _cache_path
    if _cache is None or _cache_path != db_cfg.embed_cache_path:
        _cache = EmbeddingCache(db_cfg.embed_cache_path)
        _cache_path = db_cfg.embed_cache_path
    return _cache
