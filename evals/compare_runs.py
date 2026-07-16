#!/usr/bin/env python3
"""Paired significance test between two eval runs (McNemar, exact).

Two runs over the SAME golden set are paired data: every case has a
verdict under configuration A and under configuration B. Comparing
aggregate rates ("0.947 vs 0.974") ignores that pairing and invites
eyeball verdicts over overlapping CIs. The honest question is about the
DISCORDANT cases only — how often did A pass where B failed (b) versus
B pass where A failed (c)? Under the null (no real difference), flips
go either way with p=0.5; the exact two-sided binomial on min(b, c)
of b+c is McNemar's test in its small-sample form — right for eval
sizes where the chi-square approximation lies.

This is an INSTRUMENT, not a gate (always exits 0): it turns "fusion
looked 2.7pp better" into "b=1, c=3, p=0.625 — noise" or "b=0, c=9,
p=0.004 — real", and it lists the flipped case ids because this
repository reads its flipped cases instead of trusting any single
number.

Usage (dumps come from run_eval.py --dump-answers):

    python evals/compare_runs.py A.jsonl B.jsonl
    python evals/compare_runs.py A.jsonl B.jsonl --metrics correct
    python evals/compare_runs.py A.jsonl B.jsonl --category multi_turn
"""

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_METRICS = ("correct", "faithful")


def load_dump(path: Path):
    """Return (meta, {case_id: record}) for a --dump-answers file."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        sys.exit(f"error: {path} is empty")
    meta, records = {}, lines
    if "_meta" in lines[0]:
        meta, records = lines[0]["_meta"], lines[1:]
    by_id = {record["id"]: record for record in records if "id" in record}
    if not by_id:
        sys.exit(f"error: {path} has no case records with ids")
    return meta, by_id


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5**n
    return min(1.0, 2.0 * tail)


def verdict_bool(value):
    """Normalize a dump verdict: bool, or a judge dict, or None.

    run_eval dumps store the full judge output ({"verdict": "pass",
    "reason": ...}); rejudge and synthetic fixtures may store plain
    booleans. Anything else (null, missing, unparseable) is unjudged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        verdict = str(value.get("verdict", "")).strip().lower()
        if verdict in ("pass", "fail"):
            return verdict == "pass"
    return None


def compare_metric(a_by_id, b_by_id, metric: str, category=None):
    """Pair the runs on one metric; returns None if never comparable."""
    both_pass = both_fail = 0
    a_only, b_only = [], []  # flipped case ids — the things to go read
    a_pass = b_pass = paired = 0
    for case_id in sorted(set(a_by_id) & set(b_by_id)):
        rec_a, rec_b = a_by_id[case_id], b_by_id[case_id]
        if category and rec_a.get("category") != category:
            continue
        va = verdict_bool(rec_a.get(metric))
        vb = verdict_bool(rec_b.get(metric))
        if va is None or vb is None:
            continue  # unjudged (retrieval-only dumps, null verdicts)
        paired += 1
        a_pass += va
        b_pass += vb
        if va and vb:
            both_pass += 1
        elif va and not vb:
            a_only.append(case_id)
        elif vb and not va:
            b_only.append(case_id)
        else:
            both_fail += 1
    if paired == 0:
        return None
    return {
        "paired": paired,
        "rate_a": a_pass / paired,
        "rate_b": b_pass / paired,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "a_only": a_only,
        "b_only": b_only,
        "p": mcnemar_exact_p(len(a_only), len(b_only)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_a", type=Path)
    parser.add_argument("dump_b", type=Path)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="comma-separated record fields to compare (default: correct,faithful)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="restrict to one case category (e.g. multi_turn)",
    )
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    args = parser.parse_args()

    meta_a, a_by_id = load_dump(args.dump_a)
    meta_b, b_by_id = load_dump(args.dump_b)
    label_a = args.label_a or meta_a.get("model") or args.dump_a.name
    label_b = args.label_b or meta_b.get("model") or args.dump_b.name

    shared = set(a_by_id) & set(b_by_id)
    only_a, only_b = len(a_by_id) - len(shared), len(b_by_id) - len(shared)
    print(f"A = {label_a} ({args.dump_a.name})")
    print(f"B = {label_b} ({args.dump_b.name})")
    print(f"Paired cases: {len(shared)}", end="")
    if only_a or only_b:
        print(f"  (UNPAIRED ignored: {only_a} only in A, {only_b} only in B)")
    else:
        print()

    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        result = compare_metric(a_by_id, b_by_id, metric, args.category)
        header = f"\n{metric}" + (f" [{args.category}]" if args.category else "")
        if result is None:
            print(f"{header}: no paired boolean verdicts — skipped")
            continue
        b_count, c_count = len(result["a_only"]), len(result["b_only"])
        print(
            f"{header}: A {result['rate_a']:.3f} vs B {result['rate_b']:.3f} "
            f"(n={result['paired']}, Δ={result['rate_b'] - result['rate_a']:+.3f})"
        )
        print(
            f"  concordant: {result['both_pass']} both-pass, "
            f"{result['both_fail']} both-fail; discordant: "
            f"A-only-pass={b_count}, B-only-pass={c_count}"
        )
        print(f"  McNemar exact p = {result['p']:.4f}", end="")
        if result["p"] < 0.05:
            print("  -> difference unlikely to be noise (p < 0.05)")
        else:
            print("  -> not significant at 0.05; treat as noise until n grows")
        if result["a_only"]:
            print(f"  flips A-only-pass: {', '.join(result['a_only'])}")
        if result["b_only"]:
            print(f"  flips B-only-pass: {', '.join(result['b_only'])}")
        print("  (read the flipped cases before believing either direction)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
