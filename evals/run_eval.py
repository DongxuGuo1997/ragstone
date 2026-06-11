#!/usr/bin/env python3
"""Two-layer evaluation harness for the RAG pipeline.

Layer 1 (retrieval) is deterministic and free: did the right chunks come
back for each golden question? Layer 2 (generation) uses an LLM judge and
costs a few cents: was the final answer correct and grounded?

Run from the repository root (so the project's .env is picked up):

    python evals/run_eval.py --mode retrieval      # Layer 1 only, fast
    python evals/run_eval.py                       # both layers
    python evals/run_eval.py --update-baseline     # accept current scores

The run fails (exit 1) if any metric drops more than TOLERANCE below the
committed baseline in evals/baseline.json.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

EVALS_DIR = Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus"
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
BASELINE_PATH = EVALS_DIR / "baseline.json"
REPORT_PATH = EVALS_DIR / "report.md"

# How far a metric may drop below the baseline before the run fails.
TOLERANCE = 0.05

sys.path.insert(0, str(EVALS_DIR.parent / "src"))  # allow running without install
sys.path.insert(0, str(EVALS_DIR))  # for `import judge`


def load_cases() -> list:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pipeline(args):
    from langchain_rag import OllamaPipeline, OpenAIPipeline

    if args.provider == "openai":
        pipeline = OpenAIPipeline(model=args.model)
    else:
        pipeline = OllamaPipeline(model=args.model)

    texts = pipeline.load_and_split(data_dir=str(CORPUS_DIR))
    if not texts:
        sys.exit(f"error: no documents loaded from {CORPUS_DIR}")
    print(f"Loaded corpus: {len(texts)} chunks from {CORPUS_DIR}")

    if args.provider == "openai":
        pipeline.set_retriever_openai(use_ensemble=True, use_reranker=args.rerank)
    else:
        pipeline.set_retriever_ollama(use_ensemble=True, use_reranker=args.rerank)
    if pipeline.get_retriever() is None:
        sys.exit("error: retriever was not created (check API key / Ollama server)")

    if args.mode == "full":
        pipeline.create_rag_chain(chain_type=args.chain_type)
        if pipeline.get_chain() is None:
            sys.exit("error: RAG chain was not created")
    return pipeline


def find_hit_rank(case, docs):
    """Rank (1-based) of the first chunk satisfying must_contain, else None.

    Multi-hop facts may legitimately live in different chunks, so for that
    category the union of all top-k chunks is searched (rank reported as 1).
    """
    needles = [m.lower() for m in case["must_contain"]]
    if case["category"] == "multi_hop":
        union = " ".join(d.page_content.lower() for d in docs)
        return 1 if all(n in union for n in needles) else None
    for i, doc in enumerate(docs):
        text = doc.page_content.lower()
        if all(n in text for n in needles):
            return i + 1
    return None


def eval_retrieval(pipeline, cases, k):
    retriever = pipeline.get_retriever()
    rows = []
    for case in cases:
        if case["category"] == "unanswerable":
            continue
        docs = retriever.invoke(case["question"])[:k]
        rank = find_hit_rank(case, docs)
        rows.append({"case": case, "rank": rank})
        marker = f"hit@{rank}" if rank else "MISS"
        print(f"  [{case['id']}] {marker:<7} {case['question'][:60]}")
    hits = [r for r in rows if r["rank"] is not None]
    metrics = {
        "hit_rate": round(len(hits) / len(rows), 3) if rows else 0.0,
        "mrr": round(sum(1 / r["rank"] for r in hits) / len(rows), 3) if rows else 0.0,
    }
    return metrics, rows


def eval_generation(pipeline, cases, args):
    import judge

    retriever = pipeline.get_retriever()
    rows = []
    for case in cases:
        question = case["question"]
        answer = pipeline.ask_question(
            question, session_id=f"eval_{case['id']}", use_cache=False
        )
        if not answer:
            rows.append(
                {
                    "case": case,
                    "answer": "",
                    "correct": {
                        "verdict": "fail",
                        "reason": "pipeline returned no answer",
                    },
                    "faithful": {
                        "verdict": "fail",
                        "reason": "pipeline returned no answer",
                    },
                }
            )
            print(f"  [{case['id']}] NO ANSWER")
            continue

        context = "\n\n".join(
            d.page_content for d in retriever.invoke(question)[: args.k]
        )
        correct = judge.judge_correctness(
            question, case["gold_answer"], answer, args.judge_model, args.judge_provider
        )
        faithful = judge.judge_faithfulness(
            question, context, answer, args.judge_model, args.judge_provider
        )
        rows.append(
            {"case": case, "answer": answer, "correct": correct, "faithful": faithful}
        )
        print(
            f"  [{case['id']}] correct={correct['verdict']:<4} "
            f"faithful={faithful['verdict']:<4} {question[:50]}"
        )
    n = len(rows)
    metrics = {
        "correct_rate": (
            round(sum(r["correct"]["verdict"] == "pass" for r in rows) / n, 3)
            if n
            else 0.0
        ),
        "faithful_rate": (
            round(sum(r["faithful"]["verdict"] == "pass" for r in rows) / n, 3)
            if n
            else 0.0
        ),
    }
    return metrics, rows


def write_report(args, metrics, retrieval_rows, generation_rows):
    lines = [
        "# Evaluation Report",
        "",
        f"- date: {date.today().isoformat()}",
        f"- pipeline: {args.provider}:{args.model}, chain={args.chain_type}, k={args.k}",
        f"- judge: {args.judge_provider}:{args.judge_model}",
        "",
        "## Scores",
        "",
    ]
    lines += [f"- **{name}**: {value}" for name, value in metrics.items()]
    lines += [
        "",
        "## Retrieval (Layer 1)",
        "",
        "| id | rank | question |",
        "|---|---|---|",
    ]
    for r in retrieval_rows:
        rank = r["rank"] if r["rank"] else "**MISS**"
        lines.append(f"| {r['case']['id']} | {rank} | {r['case']['question']} |")
    if generation_rows:
        lines += [
            "",
            "## Generation (Layer 2)",
            "",
            "| id | correct | faithful | question |",
            "|---|---|---|---|",
        ]
        for r in generation_rows:
            lines.append(
                f"| {r['case']['id']} | {r['correct']['verdict']} "
                f"| {r['faithful']['verdict']} | {r['case']['question']} |"
            )
        failures = [
            r
            for r in generation_rows
            if "fail" in (r["correct"]["verdict"], r["faithful"]["verdict"])
        ]
        if failures:
            lines += ["", "### Failure details", ""]
            for r in failures:
                lines += [
                    f"**{r['case']['id']}** — {r['case']['question']}",
                    f"- answer: {r['answer'][:300]}",
                    f"- correct: {r['correct']['verdict']} ({r['correct']['reason']})",
                    f"- faithful: {r['faithful']['verdict']} ({r['faithful']['reason']})",
                    "",
                ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


def baseline_key(args) -> str:
    return (
        f"{args.provider}:{args.model}|judge:{args.judge_provider}:{args.judge_model}"
        f"|k={args.k}|chain={args.chain_type}|mode={args.mode}"
        + ("|rerank" if args.rerank else "")
    )


def check_baseline(args, metrics) -> int:
    key = baseline_key(args)
    baseline = {}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.update_baseline:
        baseline[key] = {"metrics": metrics, "date": date.today().isoformat()}
        BASELINE_PATH.write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Baseline updated for: {key}")
        return 0

    if key not in baseline:
        print(f"No baseline for this configuration ({key}).")
        print("Run with --update-baseline to record one.")
        return 0

    failed = False
    for name, value in metrics.items():
        base = baseline[key]["metrics"].get(name)
        if base is None:
            continue
        if value < base - TOLERANCE:
            print(
                f"REGRESSION: {name} = {value} (baseline {base}, tolerance {TOLERANCE})"
            )
            failed = True
        else:
            print(f"ok: {name} = {value} (baseline {base})")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["retrieval", "full"], default="full")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument(
        "--judge-provider", choices=["openai", "ollama"], default="openai"
    )
    parser.add_argument("--k", type=int, default=4, help="retrieval depth")
    parser.add_argument("--chain-type", default="simple")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="enable the cross-encoder reranker (requires the rerank extra)",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--no-baseline-check", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    print(f"Loaded {len(cases)} golden cases")
    pipeline = build_pipeline(args)

    print("\nLayer 1: retrieval")
    metrics, retrieval_rows = eval_retrieval(pipeline, cases, args.k)

    generation_rows = []
    if args.mode == "full":
        print("\nLayer 2: generation (LLM judge)")
        gen_metrics, generation_rows = eval_generation(pipeline, cases, args)
        metrics.update(gen_metrics)

    print("\nScores:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    write_report(args, metrics, retrieval_rows, generation_rows)

    if args.no_baseline_check and not args.update_baseline:
        return 0
    return check_baseline(args, metrics)


if __name__ == "__main__":
    sys.exit(main())
