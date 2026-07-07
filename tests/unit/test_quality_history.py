"""
Unit tests for the quality-history chart (pure functions, no git).

The chart is a showcase artifact generated from committed baselines; its
extraction must tolerate schema drift (early revisions lack multi-turn
metrics), the dedupe must keep only real movements, and the SVG must be
self-contained.
"""

import importlib.util
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"

spec = importlib.util.spec_from_file_location(
    "quality_history", EVALS_DIR / "quality_history.py"
)
qh = importlib.util.module_from_spec(spec)
sys.modules["quality_history"] = qh
spec.loader.exec_module(qh)


def _snapshot(short_hash, date, subject, metrics):
    return (
        short_hash,
        date,
        subject,
        {qh.SMOKE_FULL_KEY: {"metrics": metrics, "date": date}},
    )


SNAPSHOTS = [
    _snapshot("aaa1111", "2026-06-11", "initial", {"hit_rate": 0.971, "mrr": 0.902}),
    _snapshot(
        "bbb2222",
        "2026-06-12",
        "embeddings",
        {"hit_rate": 1.0, "mrr": 0.895, "correct_rate": 0.947},
    ),
    _snapshot(
        "ccc3333",
        "2026-07-01",
        "unrelated re-record",
        {"hit_rate": 1.0, "mrr": 0.895, "correct_rate": 0.947},
    ),
    _snapshot(
        "ddd4444",
        "2026-07-07",
        "enrichment",
        {"hit_rate": 1.0, "mrr": 0.931, "correct_rate": 0.974},
    ),
]


class TestExtraction:
    """Series extraction must survive early-revision schema drift."""

    def test_points_carry_only_present_metrics(self):
        points = qh.extract_series(SNAPSHOTS)
        assert len(points) == 4
        assert "correct_rate" not in points[0]  # absent in the old schema
        assert points[1]["correct_rate"] == 0.947

    def test_snapshots_without_the_key_are_skipped(self):
        snapshots = SNAPSHOTS + [("eee5555", "2026-07-08", "other key only", {})]
        assert len(qh.extract_series(snapshots)) == 4


class TestDedupe:
    """Only real metric movements should survive the default view."""

    def test_unchanged_runs_collapse_to_the_first_occurrence(self):
        points = qh.dedupe_unchanged(qh.extract_series(SNAPSHOTS))
        assert [p["hash"] for p in points] == ["aaa1111", "bbb2222", "ddd4444"]


class TestRendering:
    """Table and SVG must be self-contained and complete."""

    def test_table_marks_missing_metrics(self):
        table = qh.render_table(qh.extract_series(SNAPSHOTS)[:1])
        assert "0.971" in table
        assert "—" in table  # metrics absent in old revisions

    def test_svg_contains_every_metric_series_and_no_external_refs(self):
        points = qh.extract_series(SNAPSHOTS)
        svg = qh.render_svg(points)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        for color in qh.COLORS.values():
            assert color in svg  # each series is drawn (or legended)
        assert "http" not in svg.replace("http://www.w3.org", "")  # self-contained
        assert "aaa1111" in svg  # commit labels present
