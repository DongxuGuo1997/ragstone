"""Thread-safe, process-wide pipeline registry — with persistence.

Shared by the MCP server and the REST API: both serve tools/requests
concurrently (async handlers offloading blocking work to worker threads),
so registry reads and writes must be atomic. Critical sections cover only
the dict access — never a network or embedding call — so the lock adds no
meaningful contention.

Callers keep the returned pipeline reference for the whole operation; a
concurrent delete therefore cannot pull the object out from under an
in-flight call, it only removes it from the registry.

Persistence (ROADMAP 5.7): a configured pipeline used to die with the
process, forcing clients to re-ingest after every restart. Now each
successful retriever setup writes a manifest (provider, model, chain and
retriever config, corpus fingerprint) plus the enriched chunks, and
``get_or_restore_pipeline`` rebuilds lazily on the first request for an
unknown id. Restoration makes NO LLM calls — enrichment and metadata
cards are already baked into the persisted chunks — and re-embedding is
served from the content-addressed embedding cache, so a warm restore
costs no API spend. RAGSTONE_REGISTRY_PERSIST=off disables all of it;
RAGSTONE_REGISTRY_DIR moves the manifest directory (default
store/registry). Manifest filenames are the sha256 of the pipeline id:
ids are client-supplied strings and must never touch filesystem paths.
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..rag.pipeline import Pipeline

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1

_pipelines: Dict[str, Pipeline] = {}
_pipelines_lock = threading.Lock()
# Serializes restores so N concurrent requests for the same cold id do
# one rebuild, not N. Never held together with _pipelines_lock.
_restore_lock = threading.Lock()


def get_pipeline(pipeline_id: str) -> Optional[Pipeline]:
    """Return the pipeline for an id, or None."""
    with _pipelines_lock:
        return _pipelines.get(pipeline_id)


def put_pipeline(pipeline_id: str, pipeline: Pipeline) -> None:
    """Register (or replace) a pipeline under an id."""
    with _pipelines_lock:
        _pipelines[pipeline_id] = pipeline


def pop_pipeline(pipeline_id: str) -> Optional[Pipeline]:
    """Atomically remove and return a pipeline, or None if absent."""
    with _pipelines_lock:
        return _pipelines.pop(pipeline_id, None)


def snapshot_pipelines() -> List[Tuple[str, Pipeline]]:
    """A consistent (id, pipeline) list for read-only iteration."""
    with _pipelines_lock:
        return list(_pipelines.items())


def clear_pipelines() -> None:
    """Remove every pipeline (used by tests and shutdown paths)."""
    with _pipelines_lock:
        _pipelines.clear()


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def _registry_dir() -> Optional[Path]:
    """The manifest directory, or None when persistence is off."""
    flag = os.getenv("RAGSTONE_REGISTRY_PERSIST", "on").strip().lower()
    if flag in ("off", "false", "0", "no"):
        return None
    return Path(os.getenv("RAGSTONE_REGISTRY_DIR", "store/registry"))


def _manifest_paths(pipeline_id: str) -> Tuple[Path, Path]:
    base = _registry_dir()
    assert base is not None  # callers check _registry_dir() first
    digest = hashlib.sha256(pipeline_id.encode("utf-8")).hexdigest()[:24]
    return base / f"{digest}.json", base / f"{digest}.chunks.jsonl"


def persist_pipeline(pipeline_id: str, pipeline: Pipeline) -> bool:
    """Write the manifest + chunks for a configured pipeline.

    Best-effort by contract: called after a successful retriever setup,
    and a persistence failure must never fail the request that just
    succeeded — it only costs a re-ingest after the next restart.
    """
    base = _registry_dir()
    if base is None:
        return False
    try:
        texts = pipeline.texts or []
        manifest = {
            "version": _MANIFEST_VERSION,
            "pipeline_id": pipeline_id,
            "provider": getattr(pipeline, "provider", None),
            "model": (
                pipeline.llm_proxy.get_model_name() if pipeline.llm_proxy else None
            ),
            "chain_type": getattr(pipeline, "_chain_type", None) or "simple",
            "retriever": {
                "use_ensemble": getattr(pipeline, "_use_ensemble", True),
                "use_reranker": getattr(pipeline, "_use_reranker", False),
            },
            "fingerprint": getattr(pipeline, "_vector_db_fingerprint", None),
            "chunk_count": len(texts),
        }
        if not manifest["provider"] or not manifest["model"] or not texts:
            logger.warning(
                f"Pipeline {pipeline_id!r} not persistable "
                "(missing provider/model/chunks); skipping."
            )
            return False
        base.mkdir(parents=True, exist_ok=True)
        manifest_path, chunks_path = _manifest_paths(pipeline_id)
        with open(f"{chunks_path}.tmp", "w", encoding="utf-8") as fh:
            for doc in texts:
                fh.write(
                    json.dumps(
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        },
                        ensure_ascii=False,
                        default=str,  # loader metadata can carry odd types
                    )
                    + "\n"
                )
        os.replace(f"{chunks_path}.tmp", chunks_path)
        with open(f"{manifest_path}.tmp", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        os.replace(f"{manifest_path}.tmp", manifest_path)  # atomic: no torn reads
        logger.info(
            f"Persisted pipeline {pipeline_id!r} ({len(texts)} chunks) "
            f"to {manifest_path}"
        )
        return True
    except Exception as exc:
        logger.warning(f"Could not persist pipeline {pipeline_id!r}: {exc}")
        return False


def restore_pipeline(pipeline_id: str) -> Optional[Pipeline]:
    """Rebuild a pipeline from its manifest, register it, and return it.

    None when persistence is off, no manifest exists, or the rebuild
    fails (logged) — callers treat all three as "not found". No LLM
    calls happen here; embeddings come from the cache when warm.
    """
    base = _registry_dir()
    if base is None:
        return None
    manifest_path, chunks_path = _manifest_paths(pipeline_id)
    if not manifest_path.exists():
        return None
    with _restore_lock:
        # Another request may have finished the same restore while this
        # one waited on the lock.
        existing = get_pipeline(pipeline_id)
        if existing is not None:
            return existing
        try:
            from langchain_core.documents import Document

            from ..rag.pipeline import build_pipeline

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != _MANIFEST_VERSION:
                logger.warning(
                    f"Manifest for {pipeline_id!r} has unsupported version "
                    f"{manifest.get('version')}; ignoring."
                )
                return None
            chunks = []
            with open(chunks_path, encoding="utf-8") as fh:
                for line in fh:
                    record = json.loads(line)
                    chunks.append(
                        Document(
                            page_content=record["page_content"],
                            metadata=record.get("metadata", {}),
                        )
                    )
            if len(chunks) != manifest["chunk_count"]:
                raise ValueError(
                    f"chunk file has {len(chunks)} chunks, "
                    f"manifest expects {manifest['chunk_count']}"
                )
            pipeline = build_pipeline(manifest["provider"], manifest["model"])
            pipeline.texts = chunks
            retriever = manifest.get("retriever", {})
            pipeline.setup_retriever(
                use_ensemble=retriever.get("use_ensemble", True),
                use_reranker=retriever.get("use_reranker", False),
            )
            pipeline.create_rag_chain(chain_type=manifest["chain_type"])
            recorded = manifest.get("fingerprint")
            rebuilt = getattr(pipeline, "_vector_db_fingerprint", None)
            if recorded and rebuilt and recorded != rebuilt:
                # Not fatal — the store was just rebuilt from the same
                # chunks, so serving is correct — but it means the
                # embedding model changed since the manifest was written.
                logger.warning(
                    f"Pipeline {pipeline_id!r} fingerprint changed on restore "
                    "(embedding config differs from ingest time)."
                )
            put_pipeline(pipeline_id, pipeline)
            logger.info(
                f"Restored pipeline {pipeline_id!r} from manifest "
                f"({len(chunks)} chunks, chain={manifest['chain_type']})"
            )
            return pipeline
        except Exception as exc:
            logger.error(f"Could not restore pipeline {pipeline_id!r}: {exc}")
            return None


def get_or_restore_pipeline(pipeline_id: str) -> Optional[Pipeline]:
    """The registered pipeline, or a lazy restore from its manifest.

    Restoration embeds (cache-served when warm) — call from a worker
    thread, not an event loop.
    """
    return get_pipeline(pipeline_id) or restore_pipeline(pipeline_id)


def delete_persisted(pipeline_id: str) -> None:
    """Remove a pipeline's manifest and chunks (missing files are fine).

    DELETE means delete: the persisted corpus must not resurrect a
    pipeline the client explicitly removed.
    """
    if _registry_dir() is None:
        return
    for path in _manifest_paths(pipeline_id):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - permissions etc.
            logger.warning(f"Could not delete {path}: {exc}")


def list_persisted() -> List[str]:
    """Pipeline ids with a manifest on disk (loaded or not)."""
    base = _registry_dir()
    if base is None or not base.is_dir():
        return []
    ids = []
    for manifest_path in sorted(base.glob("*.json")):
        try:
            ids.append(
                json.loads(manifest_path.read_text(encoding="utf-8"))["pipeline_id"]
            )
        except Exception as exc:
            logger.warning(f"Unreadable manifest {manifest_path}: {exc}")
    return ids
