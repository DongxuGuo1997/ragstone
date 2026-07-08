"""Prometheus exporter for the per-request metrics (ROADMAP 5.2).

Subscribes to the observability module's request observers and turns each
completed RequestMetrics into Prometheus series, so a scrape of the API's
``/metrics`` endpoint answers the operational questions directly:

- traffic and outcomes: ``ragstone_requests_total{chain,cache,error}``
  (cache hit rate and error rate are ratios of this one counter)
- latency: ``ragstone_request_duration_seconds{chain}`` and, for streams,
  ``ragstone_first_token_seconds`` (time the answer FELT like it took)
- where the time went: ``ragstone_stage_duration_seconds{stage}`` for
  rephrase/retrieval plus the derived generation remainder
- spend: ``ragstone_tokens_total{direction}``

Only aggregates leave this module — never questions, answers, session ids,
or document content. Expose /metrics on internal networks only; it is
unauthenticated (scrapers don't carry app keys), like the probes.

prometheus-client ships with the ``api`` extra; without it installed,
``init_metrics()`` reports False and the endpoint answers 503. Collectors
live in a private registry (not prometheus_client's global default), so
repeated app construction in tests can't double-register. Single-process
servers only (uvicorn's default); multiprocess gunicorn would need
prometheus_client's multiprocess mode.
"""

import logging
from typing import Optional, Tuple

from ragstone.utils.observability import RequestMetrics, add_request_observer

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _PROMETHEUS_AVAILABLE = False

# LLM requests span four orders of magnitude: a semantic-cache hit lands
# in milliseconds, an agent-mode chain can take a minute. Default buckets
# top out at 10s, which would flatten exactly the tail we care about.
_DURATION_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120)


class PrometheusExporter:
    """Owns the collectors and translates RequestMetrics into samples."""

    def __init__(self, registry: Optional["CollectorRegistry"] = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.requests = Counter(
            "ragstone_requests_total",
            "Completed ask requests by chain, cache outcome, and error class.",
            ["chain", "cache", "error"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "ragstone_request_duration_seconds",
            "End-to-end ask latency (rephrase + retrieval + generation).",
            ["chain"],
            registry=self.registry,
            buckets=_DURATION_BUCKETS,
        )
        self.first_token = Histogram(
            "ragstone_first_token_seconds",
            "Streaming only: time until the first answer token.",
            registry=self.registry,
            buckets=_DURATION_BUCKETS,
        )
        self.stage_duration = Histogram(
            "ragstone_stage_duration_seconds",
            "Per-stage latency; 'generation' is the unattributed remainder.",
            ["stage"],
            registry=self.registry,
            buckets=_DURATION_BUCKETS,
        )
        self.tokens = Counter(
            "ragstone_tokens_total",
            "LLM tokens consumed, split by direction.",
            ["direction"],
            registry=self.registry,
        )

    def observe(self, metrics: RequestMetrics) -> None:
        chain = metrics.chain_type or "none"
        self.requests.labels(
            chain=chain,
            cache="hit" if metrics.cache_hit else "miss",
            error=metrics.error or "none",
        ).inc()
        self.duration.labels(chain=chain).observe(metrics.latency_ms / 1000)
        if metrics.first_token_ms is not None:
            self.first_token.observe(metrics.first_token_ms / 1000)
        if metrics.input_tokens:
            self.tokens.labels(direction="input").inc(metrics.input_tokens)
        if metrics.output_tokens:
            self.tokens.labels(direction="output").inc(metrics.output_tokens)
        # Cache hits never ran the stages; folding their ~2ms "generation"
        # remainder into the histogram would fake a fast-LLM mode.
        if not metrics.cache_hit:
            for stage, elapsed_ms in metrics.stage_ms.items():
                self.stage_duration.labels(stage=stage).observe(elapsed_ms / 1000)
            self.stage_duration.labels(stage="generation").observe(
                metrics.generation_ms / 1000
            )


_exporter: Optional[PrometheusExporter] = None


def init_metrics() -> bool:
    """Idempotently wire the exporter to the request observers.

    Returns False when prometheus-client is not installed, so callers can
    surface "metrics unavailable" rather than break.
    """
    global _exporter
    if not _PROMETHEUS_AVAILABLE:
        return False
    if _exporter is None:
        _exporter = PrometheusExporter()
        add_request_observer(_exporter.observe)
    return True


def render_metrics() -> Optional[Tuple[bytes, str]]:
    """The exposition payload and its content type, or None if disabled."""
    if _exporter is None:
        return None
    return generate_latest(_exporter.registry), CONTENT_TYPE_LATEST
