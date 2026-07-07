"""Thread-safe, process-wide pipeline registry.

Shared by the MCP server and the REST API: both serve tools/requests
concurrently (async handlers offloading blocking work to worker threads),
so registry reads and writes must be atomic. Critical sections cover only
the dict access — never a network or embedding call — so the lock adds no
meaningful contention.

Callers keep the returned pipeline reference for the whole operation; a
concurrent delete therefore cannot pull the object out from under an
in-flight call, it only removes it from the registry.
"""

import threading
from typing import Dict, List, Optional, Tuple

from ..rag.pipeline import Pipeline

_pipelines: Dict[str, Pipeline] = {}
_pipelines_lock = threading.Lock()


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
