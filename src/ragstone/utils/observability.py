"""Per-request observability for the ask paths.

Every ask_question / ask_question_stream call emits exactly one structured
log line on the ``ragstone.requests`` logger when it finishes:

    request=1f2e3d4c session=web-42 chain=simple cache_hit=False \
latency_ms=1440 tokens=1031

- ``request`` is a per-call correlation id, so all log lines from one call
  can be tied together.
- ``tokens`` is aggregated across every LLM call the request needed (the
  rephrase step, agent-mode searches, the answer) via LangChain's
  usage-metadata callback; models that report no usage contribute 0.
- ``cache_hit`` / ``error`` let latency numbers be segmented honestly:
  a cache hit in 2 ms and a failure after a 60 s timeout are not answers.

Operational aggregation (p95 latency, cost per session) is left to the log
processor — this module's job is to make sure the data exists.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

from langchain_core.callbacks import get_usage_metadata_callback

logger = logging.getLogger("ragstone.requests")

# The metrics object for the request currently executing on this thread of
# control. Contextvars propagate into LangGraph node execution (nodes run
# under contextvars.copy_context), so stages recorded inside graph nodes
# land on the right request — the same mechanism the token-usage callback
# relies on.
_current_metrics: ContextVar[Optional["RequestMetrics"]] = ContextVar(
    "ragstone_request_metrics", default=None
)


def record_stage(stage: str, elapsed_ms: int) -> None:
    """Attribute elapsed time to a named stage of the current request.

    A no-op when no request is being tracked (e.g. direct chain use in
    tests or scripts), so instrumented components never need guards.
    Repeated stages accumulate — e.g. the agent chain's multiple
    retrievals sum into one "retrieval" figure.
    """
    metrics = _current_metrics.get()
    if metrics is not None:
        metrics.stage_ms[stage] = metrics.stage_ms.get(stage, 0) + elapsed_ms


@contextmanager
def time_stage(stage: str) -> Iterator[None]:
    """Context manager form of record_stage for wrapping a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        record_stage(stage, int((time.perf_counter() - start) * 1000))


@dataclass
class RequestMetrics:
    """Mutable record for one ask; fields are filled in as the call runs."""

    request_id: str
    session_id: str
    chain_type: Optional[str] = None
    cache_hit: bool = False
    error: Optional[str] = None
    latency_ms: int = 0
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Streaming only: ms until the first answer token reached the caller.
    # latency_ms is end-to-end (rephrase + retrieval + full generation),
    # which can read much larger than the answer FELT — show both.
    first_token_ms: Optional[int] = None
    # Per-stage decomposition recorded via record_stage(), e.g.
    # {"rephrase": 520, "retrieval": 180}. Generation is derived: whatever
    # latency the named stages don't account for.
    stage_ms: Dict[str, int] = field(default_factory=dict)

    @property
    def generation_ms(self) -> int:
        """Latency not attributed to a named stage (LLM generation plus
        graph/checkpoint overhead). Derived rather than measured so stages
        can never double-count."""
        return max(0, self.latency_ms - sum(self.stage_ms.values()))


# Approximate USD prices per million tokens (input, output). Used only for
# the demo UI's cost estimate — update as OpenAI prices change; an unknown
# model yields None and the UI omits the estimate rather than guessing.
_PRICES_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, model: Optional[str]
) -> Optional[float]:
    """Rough USD cost of a request, or None for unknown/local models."""
    if not model or model not in _PRICES_PER_MTOK:
        return None
    price_in, price_out = _PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@contextmanager
def track_request(
    session_id: str, chain_type: Optional[str] = None
) -> Iterator[RequestMetrics]:
    """Track one request and emit the structured log line when it ends.

    Usage:
        with track_request(session_id, chain_type) as metrics:
            ...
            metrics.cache_hit = True  # callers annotate as they learn more

    The log line is emitted even when the body raises; the exception class
    name is recorded and the exception propagates unchanged.
    """
    metrics = RequestMetrics(
        request_id=uuid.uuid4().hex[:8],
        session_id=session_id,
        chain_type=chain_type,
    )
    start = time.perf_counter()
    usage_ctx = get_usage_metadata_callback()
    usage_cb = usage_ctx.__enter__()
    metrics_token = _current_metrics.set(metrics)
    try:
        yield metrics
    except Exception as exc:
        metrics.error = type(exc).__name__
        raise
    finally:
        # Record and log FIRST: when the enclosing generator is stepped
        # across different contexts (e.g. a server pumping each chunk
        # through a fresh thread), the ContextVar teardown below can fail —
        # the log line must not be lost with it.
        metrics.latency_ms = int((time.perf_counter() - start) * 1000)
        usages = usage_cb.usage_metadata.values()
        metrics.tokens = sum(u.get("total_tokens", 0) for u in usages)
        metrics.input_tokens = sum(u.get("input_tokens", 0) for u in usages)
        metrics.output_tokens = sum(u.get("output_tokens", 0) for u in usages)
        stages = ",".join(f"{k}:{v}" for k, v in sorted(metrics.stage_ms.items()))
        logger.info(
            "request=%s session=%s chain=%s cache_hit=%s latency_ms=%d tokens=%d%s%s",
            metrics.request_id,
            metrics.session_id,
            metrics.chain_type,
            metrics.cache_hit,
            metrics.latency_ms,
            metrics.tokens,
            f" stages={stages}" if stages else "",
            f" error={metrics.error}" if metrics.error else "",
        )
        # ContextVar.reset() raises ValueError if __enter__ ran in a
        # different context (the token belongs to that context). Fall back
        # to clearing the var so stale metrics can't leak into whatever
        # runs next in THIS context.
        try:
            _current_metrics.reset(metrics_token)
        except ValueError:
            _current_metrics.set(None)
        try:
            usage_ctx.__exit__(None, None, None)
        except ValueError:
            pass  # same cross-context teardown; the callback data is read
