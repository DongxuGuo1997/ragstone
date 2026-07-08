"""
Unit tests for the Prometheus exporter (no server, no network).

The exporter is fed RequestMetrics objects directly; what is under test
is the translation into series: outcome labels, unit conversion, the
derived generation stage, and the cache-hit exclusion.
"""

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry  # noqa: E402

from ragstone.utils.metrics import PrometheusExporter  # noqa: E402
from ragstone.utils.observability import RequestMetrics  # noqa: E402


def _metrics(**overrides):
    base = dict(
        request_id="r1",
        session_id="s1",
        chain_type="simple",
        latency_ms=1500,
        tokens=300,
        input_tokens=200,
        output_tokens=100,
        stage_ms={"rephrase": 400, "retrieval": 100},
    )
    base.update(overrides)
    return RequestMetrics(**base)


@pytest.fixture
def exporter():
    return PrometheusExporter(CollectorRegistry())


def _value(exporter, name, **labels):
    return exporter.registry.get_sample_value(name, labels or None)


class TestRequestCounter:
    def test_outcomes_become_labels(self, exporter):
        exporter.observe(_metrics())
        exporter.observe(_metrics(cache_hit=True, stage_ms={}))
        exporter.observe(_metrics(error="ChainExecutionError"))

        assert (
            _value(
                exporter,
                "ragstone_requests_total",
                chain="simple",
                cache="miss",
                error="none",
            )
            == 1
        )
        assert (
            _value(
                exporter,
                "ragstone_requests_total",
                chain="simple",
                cache="hit",
                error="none",
            )
            == 1
        )
        assert (
            _value(
                exporter,
                "ragstone_requests_total",
                chain="simple",
                cache="miss",
                error="ChainExecutionError",
            )
            == 1
        )

    def test_missing_chain_type_is_labeled_none(self, exporter):
        exporter.observe(_metrics(chain_type=None))
        assert (
            _value(
                exporter,
                "ragstone_requests_total",
                chain="none",
                cache="miss",
                error="none",
            )
            == 1
        )


class TestDurations:
    def test_latency_is_recorded_in_seconds(self, exporter):
        exporter.observe(_metrics(latency_ms=1500))
        assert _value(
            exporter, "ragstone_request_duration_seconds_sum", chain="simple"
        ) == pytest.approx(1.5)

    def test_stages_and_derived_generation_are_recorded(self, exporter):
        # 1500ms total - 400 rephrase - 100 retrieval = 1000ms generation.
        exporter.observe(_metrics())
        sums = {
            stage: _value(exporter, "ragstone_stage_duration_seconds_sum", stage=stage)
            for stage in ("rephrase", "retrieval", "generation")
        }
        assert sums == {
            "rephrase": pytest.approx(0.4),
            "retrieval": pytest.approx(0.1),
            "generation": pytest.approx(1.0),
        }

    def test_cache_hits_are_excluded_from_stage_histograms(self, exporter):
        # A 2ms cache hit never ran the LLM; folding its "generation"
        # remainder into the histogram would fake a fast-LLM mode.
        exporter.observe(_metrics(cache_hit=True, latency_ms=2, stage_ms={}))
        assert (
            _value(
                exporter, "ragstone_stage_duration_seconds_count", stage="generation"
            )
            is None
        )

    def test_first_token_only_for_streams(self, exporter):
        exporter.observe(_metrics())  # non-streaming: first_token_ms is None
        assert _value(exporter, "ragstone_first_token_seconds_count") == 0

        exporter.observe(_metrics(first_token_ms=250))
        assert _value(exporter, "ragstone_first_token_seconds_sum") == pytest.approx(
            0.25
        )


class TestTokenCounters:
    def test_directions_accumulate_separately(self, exporter):
        exporter.observe(_metrics(input_tokens=200, output_tokens=100))
        exporter.observe(_metrics(input_tokens=50, output_tokens=25))
        assert _value(exporter, "ragstone_tokens_total", direction="input") == 250
        assert _value(exporter, "ragstone_tokens_total", direction="output") == 125
