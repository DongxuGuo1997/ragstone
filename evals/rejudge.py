#!/usr/bin/env python3
"""Re-judge STORED answers with a different judge — the judge-delta instrument.

Experiment 11 swapped judges but re-generated the answers, so its deltas
bundled judge disagreement with run-to-run generation variance — an upper
bound, as the experiment itself recorded. This tool closes that caveat:
it loads a ``--dump-answers`` JSONL from run_eval.py and re-scores
correctness + faithfulness on the SAME answers against the SAME stored
contexts (never a fresh retrieval), so the difference between two rejudge
runs is judge disagreement and nothing else.

    python evals/rejudge.py evals/dumps/smoke_local.jsonl \\
        --judge-provider openai --judge-model gpt-4o-mini
    python evals/rejudge.py evals/dumps/smoke_local.jsonl \\
        --judge-provider ollama --judge-model gemma4:31b --judge-reasoning on

Parse failures (a judge that cannot emit the JSON verdict) are counted
separately from disagreements: both lower the scores — fail-closed is the
harness contract — but they are different conversations. Always exits 0:
an instrument, not a gate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

EVALS_DIR = Path(__file__).parent
sys.path.insert(0, str(EVALS_DIR.parent / "src"))
sys.path.insert(0, str(EVALS_DIR))

from dotenv import load_dotenv  # noqa: E402

# run_eval gets OPENAI_API_KEY implicitly (building a pipeline imports the
# project settings, which load .env); this tool builds no pipeline, so the
# cloud-judge arm must load it explicitly.
load_dotenv()

import judge  # noqa: E402
from run_eval import format_metric, with_retries  # noqa: E402

_FAIL_CLOSED = {"verdict": "fail", "reason": "pipeline returned no answer"}


def load_dump(path: str) -> tuple:
    """The (_meta dict or None, list of case records) from a dump file."""
    meta = None
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if "_meta" in record:
                meta = record["_meta"]
            else:
                records.append(record)
    return meta, records


def rejudge_records(
    records: List[dict],
    judge_model: str,
    judge_provider: str,
    judge_reasoning: Optional[bool],
) -> List[dict]:
    """Score every stored answer; empty answers fail closed, zero calls."""
    results = []
    for record in records:
        if not record["answer"]:
            correct, faithful = dict(_FAIL_CLOSED), dict(_FAIL_CLOSED)
        else:
            correct = with_retries(
                judge.judge_correctness,
                record["question"],
                record.get("gold_answer"),
                record["answer"],
                judge_model,
                judge_provider,
                judge_reasoning,
            )
            faithful = with_retries(
                judge.judge_faithfulness,
                record["question"],
                record["context"],
                record["answer"],
                judge_model,
                judge_provider,
                judge_reasoning,
            )
        results.append({"record": record, "correct": correct, "faithful": faithful})
        print(
            f"  [{record['id']}] correct={correct['verdict']:<4} "
            f"faithful={faithful['verdict']:<4} {record['question'][:50]}"
        )
    return results


def compute_metrics(results: List[dict]) -> tuple:
    """(metrics, sizes) with the same single/multi_turn split as run_eval."""

    def _rates(subset: List[dict], prefix: str = "") -> Dict[str, float]:
        n = len(subset)
        if not n:
            return {}
        return {
            f"{prefix}correct_rate": round(
                sum(r["correct"]["verdict"] == "pass" for r in subset) / n, 3
            ),
            f"{prefix}faithful_rate": round(
                sum(r["faithful"]["verdict"] == "pass" for r in subset) / n, 3
            ),
        }

    single = [r for r in results if r["record"]["category"] != "multi_turn"]
    multi = [r for r in results if r["record"]["category"] == "multi_turn"]
    metrics: Dict[str, float] = {}
    metrics.update(_rates(single))
    metrics.update(_rates(multi, prefix="multi_turn_"))
    sizes = {
        "correct_rate": len(single),
        "faithful_rate": len(single),
        "multi_turn_correct_rate": len(multi),
        "multi_turn_faithful_rate": len(multi),
    }
    return metrics, sizes


def find_flips(results: List[dict]) -> List[dict]:
    """Cases where this judge disagrees with the verdict stored in the dump."""
    flips = []
    for r in results:
        for metric in ("correct", "faithful"):
            stored = r["record"][metric]["verdict"]
            fresh = r[metric]["verdict"]
            if stored != fresh:
                flips.append(
                    {
                        "id": r["record"]["id"],
                        "metric": metric,
                        "stored": stored,
                        "new": fresh,
                        "reason": r[metric]["reason"],
                    }
                )
    return flips


def count_parse_failures(results: List[dict]) -> int:
    """Verdicts that are fail-closed artifacts of unparseable judge output."""
    return sum(
        r[metric]["reason"].startswith("unparseable judge output")
        for r in results
        for metric in ("correct", "faithful")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="answers JSONL from run_eval.py --dump-answers")
    parser.add_argument("--judge-provider", choices=["openai", "ollama"], required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-reasoning",
        choices=["on", "off"],
        default=None,
        help="thinking control for an Ollama judge (unset = model default)",
    )
    args = parser.parse_args()
    judge_reasoning = {"on": True, "off": False}.get(args.judge_reasoning)

    meta, records = load_dump(args.dump)
    if meta:
        print(
            f"Dump: {meta.get('provider')}:{meta.get('model')} on "
            f"set={meta.get('set')} ({meta.get('date')}), originally judged by "
            f"{meta.get('judge_provider')}:{meta.get('judge_model')}"
        )
    print(
        f"Re-judging {len(records)} stored answers with "
        f"{args.judge_provider}:{args.judge_model}"
        + (f" (reasoning={args.judge_reasoning})" if args.judge_reasoning else "")
    )

    results = rejudge_records(
        records, args.judge_model, args.judge_provider, judge_reasoning
    )
    metrics, sizes = compute_metrics(results)
    flips = find_flips(results)
    parse_failures = count_parse_failures(results)

    print("\nScores under this judge (95% binomial CI):")
    for name, value in metrics.items():
        print(f"  {name}: {format_metric(name, value, sizes)}")

    if flips:
        print(f"\nVerdict flips vs the dump's stored judge ({len(flips)}):")
        print("| id | metric | stored -> new | this judge's reason |")
        print("|---|---|---|---|")
        for f in flips:
            print(
                f"| {f['id']} | {f['metric']} | {f['stored']} -> {f['new']} "
                f"| {f['reason'][:80]} |"
            )
    else:
        print("\nNo verdict flips vs the dump's stored judge.")

    print(
        f"\nParse failures (fail-closed, not disagreement): {parse_failures} "
        f"of {2 * len(records)} judged verdicts"
    )
    print(
        "\nJSON: "
        + json.dumps(
            {
                "judge": f"{args.judge_provider}:{args.judge_model}",
                "judge_reasoning": args.judge_reasoning,
                "metrics": metrics,
                "flips": len(flips),
                "parse_failures": parse_failures,
                "n_cases": len(records),
            }
        )
    )
    return 0  # instrument, not gate


if __name__ == "__main__":
    sys.exit(main())
