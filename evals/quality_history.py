#!/usr/bin/env python3
"""Chart the project's measured quality over its own git history.

evals/baseline.json is committed at every quality change, which makes the
repository self-documenting: walking that file's history yields the real
trajectory of hit rate, MRR, correctness, and faithfulness across every
recorded decision — the embedding upgrade, the enrichment win, each
re-baseline. One honest chart, generated from commits, no hand-typed
numbers.

    python evals/quality_history.py            # table + docs/quality_history.svg

Zero dependencies: the SVG is hand-rolled. Regenerate after recording new
baselines; the committed SVG is embedded in the README.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EVALS_DIR = Path(__file__).parent
REPO_ROOT = EVALS_DIR.parent
SVG_PATH = REPO_ROOT / "docs" / "quality_history.svg"

# The smoke full-mode key has had this exact format since the first
# committed baseline, which is what makes the series comparable.
SMOKE_FULL_KEY = (
    "openai:gpt-4o-mini|judge:openai:gpt-4o-mini|k=4|chain=simple|mode=full"
)
METRICS = ["hit_rate", "mrr", "correct_rate", "faithful_rate"]
COLORS = {
    "hit_rate": "#2563eb",
    "mrr": "#7c3aed",
    "correct_rate": "#059669",
    "faithful_rate": "#d97706",
}

# A snapshot is one committed state of baseline.json.
Snapshot = Tuple[str, str, str, dict]  # (short_hash, date, subject, parsed_json)


def read_history() -> List[Snapshot]:
    """Every committed revision of baseline.json, oldest first."""
    log = subprocess.run(
        [
            "git",
            "log",
            "--format=%h|%ad|%s",
            "--date=short",
            "--reverse",
            "--",
            "evals/baseline.json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout
    snapshots: List[Snapshot] = []
    for line in log.splitlines():
        short_hash, date, subject = line.split("|", 2)
        blob = subprocess.run(
            ["git", "show", f"{short_hash}:evals/baseline.json"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        ).stdout
        snapshots.append((short_hash, date, subject, json.loads(blob)))
    return snapshots


def extract_series(
    snapshots: List[Snapshot], key: str = SMOKE_FULL_KEY
) -> List[Dict[str, object]]:
    """One point per snapshot that contains the tracked baseline key.

    Metrics absent in early revisions (e.g. multi_turn_*) simply don't
    appear in a point — the chart handles ragged series.
    """
    points = []
    for short_hash, date, subject, data in snapshots:
        entry = data.get(key)
        if not entry:
            continue
        metrics = entry.get("metrics", {})
        point: Dict[str, object] = {
            "hash": short_hash,
            "date": date,
            "subject": subject,
        }
        for name in METRICS:
            if name in metrics:
                point[name] = float(metrics[name])
        points.append(point)
    return points


def dedupe_unchanged(points: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Keep only points whose tracked metrics differ from the previous one.

    Many commits re-record other tiers' baselines without moving the smoke
    metrics; a run of identical rows hides the actual story. --all shows
    every committed revision instead.
    """
    kept: List[Dict[str, object]] = []
    previous: Optional[Tuple] = None
    for point in points:
        signature = tuple(point.get(name) for name in METRICS)
        if signature != previous:
            kept.append(point)
            previous = signature
    return kept


def render_table(points: List[Dict[str, object]]) -> str:
    """Markdown table of the series, for terminals and docs."""
    header = "| commit | date | " + " | ".join(METRICS) + " | change |"
    rule = "|---" * (len(METRICS) + 3) + "|"
    rows = [header, rule]
    for point in points:
        cells = [str(point.get(name, "—")) for name in METRICS]
        subject = str(point["subject"])
        if len(subject) > 48:
            subject = subject[:45] + "..."
        rows.append(
            f"| {point['hash']} | {point['date']} | "
            + " | ".join(cells)
            + f" | {subject} |"
        )
    return "\n".join(rows)


def _polyline(
    values: List[Optional[float]],
    xs: List[float],
    y_of,
    color: str,
) -> str:
    """SVG for one metric: a line through known points, dots with titles."""
    known = [(x, v) for x, v in zip(xs, values) if v is not None]
    if not known:
        return ""
    path = " ".join(f"{x:.1f},{y_of(v):.1f}" for x, v in known)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y_of(v):.1f}" r="3.5" fill="{color}">'
        f"<title>{v}</title></circle>"
        for x, v in known
    )
    return (
        f'<polyline points="{path}" fill="none" stroke="{color}" '
        f'stroke-width="2"/>{dots}'
    )


def render_svg(points: List[Dict[str, object]]) -> str:
    """A self-contained SVG line chart of the series (no dependencies)."""
    width, height = 860, 380
    left, right, top, bottom = 56, 24, 40, 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    y_min, y_max = 0.7, 1.0

    def y_of(value: float) -> float:
        clamped = max(y_min, min(y_max, value))
        return top + plot_h * (1 - (clamped - y_min) / (y_max - y_min))

    n = max(len(points), 1)
    xs = [left + plot_w * (i / max(n - 1, 1)) for i in range(n)]

    grid = []
    for tick in (0.7, 0.8, 0.9, 1.0):
        y = y_of(tick)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6b7280">{tick:.1f}</text>'
        )

    labels = []
    for x, point in zip(xs, points):
        labels.append(
            f'<text x="{x:.1f}" y="{height - bottom + 16}" text-anchor="middle" '
            f'font-size="10" fill="#6b7280">{point["date"]}</text>'
            f'<text x="{x:.1f}" y="{height - bottom + 30}" text-anchor="middle" '
            f'font-size="9" fill="#9ca3af">{point["hash"]}</text>'
        )

    lines = [
        _polyline([point.get(name) for point in points], xs, y_of, COLORS[name])
        for name in METRICS
    ]

    legend = []
    for i, name in enumerate(METRICS):
        lx = left + i * 150
        legend.append(
            f'<rect x="{lx}" y="12" width="10" height="10" fill="{COLORS[name]}"/>'
            f'<text x="{lx + 16}" y="21" font-size="11" fill="#374151">{name}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, sans-serif">'
        f'<rect width="{width}" height="{height}" fill="white"/>'
        f'<text x="{left}" y="{height - 10}" font-size="10" fill="#9ca3af">'
        f"smoke set, chain=simple, k=4 — generated from git history of "
        f"evals/baseline.json</text>"
        + "".join(grid)
        + "".join(labels)
        + "".join(lines)
        + "".join(legend)
        + "</svg>"
    )


def main() -> int:
    show_all = "--all" in sys.argv
    snapshots = read_history()
    points = extract_series(snapshots)
    if not show_all:
        points = dedupe_unchanged(points)
    if not points:
        print("No baseline history found for the tracked key.")
        return 1
    print(render_table(points))
    SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(render_svg(points), encoding="utf-8")
    print(f"\nChart written to {SVG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
