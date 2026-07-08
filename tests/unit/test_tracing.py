"""
Unit tests for the OpenTelemetry span mapping (in-memory exporter).

What is under test: each tracked request becomes a `ragstone.ask` root
span carrying the request's outcome, measured stages become retroactive
child spans with honest durations, and the whole thing works without any
OTel context attach/detach (the streaming path tears contexts apart).
"""

import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from ragstone.utils.observability import (  # noqa: E402
    record_stage,
    track_request,
    use_request_id,
)

# set_tracer_provider is a global one-shot in OTel; install one provider
# with an in-memory exporter for the whole test process and clear the
# exporter between tests. The module tracer in observability.py is a
# ProxyTracer, so it picks this provider up even though it was imported
# first — the same late-binding a real deployment relies on.
_EXPORTER = InMemorySpanExporter()


def _install_provider_once():
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(provider)


@pytest.fixture(autouse=True)
def spans():
    _install_provider_once()
    _EXPORTER.clear()
    yield _EXPORTER


def _span(name):
    matches = [s for s in _EXPORTER.get_finished_spans() if s.name == name]
    assert matches, f"no span named {name}"
    return matches[0]


class TestRequestSpan:
    def test_ask_becomes_a_root_span_with_outcome_attributes(self, spans):
        with use_request_id("trace-1"):
            with track_request("s1", chain_type="simple") as metrics:
                metrics.cache_hit = True

        root = _span("ragstone.ask")
        assert root.parent is None
        assert root.attributes["ragstone.request_id"] == "trace-1"
        assert root.attributes["ragstone.session_id"] == "s1"
        assert root.attributes["ragstone.chain_type"] == "simple"
        assert root.attributes["ragstone.cache_hit"] is True

    def test_error_marks_the_span_status(self, spans):
        with pytest.raises(ValueError):
            with track_request("s1"):
                raise ValueError("boom")

        root = _span("ragstone.ask")
        assert root.status.status_code == StatusCode.ERROR
        assert "ValueError" in root.status.description

    def test_generation_is_an_attribute_not_a_span(self, spans):
        # Generation is derived (latency minus measured stages); a span
        # for it would fabricate an interval nothing measured.
        with track_request("s1") as metrics:
            record_stage("retrieval", 5)

        names = [s.name for s in spans.get_finished_spans()]
        assert "ragstone.generation" not in names
        root = _span("ragstone.ask")
        assert root.attributes["ragstone.generation_ms"] == metrics.generation_ms


class TestStageSpans:
    def test_measured_stages_become_children_with_honest_durations(self, spans):
        with track_request("s1", chain_type="simple"):
            record_stage("rephrase", 400)
            record_stage("retrieval", 120)

        root = _span("ragstone.ask")
        for stage, expected_ms in (("rephrase", 400), ("retrieval", 120)):
            child = _span(f"ragstone.{stage}")
            assert child.parent.span_id == root.context.span_id
            assert child.parent.trace_id == root.context.trace_id
            duration_ms = (child.end_time - child.start_time) / 1_000_000
            assert duration_ms == pytest.approx(expected_ms, abs=1)

    def test_stage_outside_a_request_creates_no_span(self, spans):
        record_stage("retrieval", 100)
        assert spans.get_finished_spans() == ()
