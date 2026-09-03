#!/usr/bin/env python3
"""Staffing-match eval: golden assignments vs the match chain (ROADMAP 9.1).

Scores the matcher against `golden_staffing.jsonl`, whose expected tiers
are true by construction (the bench oracle over persona specs, see
generate_staffing.py). The matcher itself sees only what a human staffer
would: the brief text and the ingested CVs — never the oracle.

Metrics (gated against baseline.json like run_eval.py):
- strong_recall_at_5: micro-rate of oracle-strong candidates appearing
  in the system's top 5. The bench guarantees these people exist and
  their CVs literally contain the required skills — a miss here is a
  matcher bug, not noise.
- full_match_accuracy: per assignment, `full_match_exists` must equal
  "the oracle has at least one strong candidate". Assignment a08 is
  deliberately unsatisfiable; claiming a full match there is the
  overselling failure this metric exists to catch.
- ordering_clean_rate: assignments where no out-of-tier candidate
  (neither oracle-strong nor oracle-partial) ranks above an
  oracle-strong one.
- gap_alignment (informational): for surfaced oracle-partial
  candidates, the verifier should find exactly one missing must-have,
  and it should be the oracle's one.
- extract_must_recall / extract_must_precision: the extracted must-have
  set against the oracle's (skill OR-groups as sets, years, language,
  domain, degree). The tier metrics can stay perfect while extraction
  is wrong — a pool where everyone has the dropped skill hides the
  drop — so extraction is scored directly. Motivated by a real RFQ
  (a10/a11 reproduce its traps).
- extract_nice_recall, extract_location_rate (informational): the
  nice-to-have list, and whether the brief's location was captured.
- extract_stability (with --extract-repeats N): share of briefs whose
  must-have set is identical across N extra extraction samples.

Ingest notes: metadata cards are forced OFF (per-run LLM ingest calls
would make the measured corpus nondeterministic); retrieval depth
defaults to k=12 here — discovery over 40 people wants a wider net than
QA's k=4 — and is recorded in the baseline key.

Usage (from the repository root):
    python evals/run_staffing_eval.py                     # gate
    python evals/run_staffing_eval.py --update-baseline   # record
    python evals/run_staffing_eval.py --verbose           # per-assignment
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

EVALS_DIR = Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus_staffing"
GOLDEN_PATH = EVALS_DIR / "golden_staffing.jsonl"

# The XL population (generate_staffing.py --scale N): same assignments,
# ~10x the people. It measures a different property — discovery at
# scale with the verification budget held constant — so it lives under
# its own --set, corpus, golden, and baseline keys.
SETS = {
    "staffing": (CORPUS_DIR, GOLDEN_PATH),
    "staffing_xl": (
        EVALS_DIR / "corpus_staffing_xl",
        EVALS_DIR / "golden_staffing_xl.jsonl",
    ),
}

sys.path.insert(0, str(EVALS_DIR.parent / "src"))
sys.path.insert(0, str(EVALS_DIR))

from run_eval import (  # noqa: E402  (path bootstrap above)
    BASELINE_PATH,
    TOLERANCE,
    ci95_halfwidth,
    data_fingerprint,
    with_retries,
)

SHORTLIST_N = 5
GATED_METRICS = (
    "strong_recall_at_5",
    "full_match_accuracy",
    "ordering_clean_rate",
    "extract_must_recall",
    "extract_must_precision",
)


def load_assignments() -> list:
    if not GOLDEN_PATH.exists():
        sys.exit(
            f"error: {GOLDEN_PATH} not found. "
            "Generate it with: python evals/generate_staffing.py"
        )
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_matcher(args):
    from ragstone import OllamaPipeline, OpenAIPipeline
    from ragstone.config.settings import get_config
    from ragstone.match import MatchPipeline, stamp_person_metadata
    from ragstone.rag.providers import DEFAULT_MODELS

    if args.model is None:
        args.model = DEFAULT_MODELS[args.provider]
    config = get_config()
    # Discovery over 40 people wants a wider net than QA's k=4; recorded
    # in the baseline key so runs stay comparable.
    config.database.similarity_k = args.k
    # Cards are one LLM call per CV at ingest: nondeterministic corpus
    # text inside a gated measurement. Off.
    config.loader.metadata_cards = False
    # Thinking control for local models: a reasoning-by-default answerer
    # spends ~28 min/assignment thinking through 11 extraction and
    # verification calls (measured, first local pilot) for no gated-
    # metric gain. Recorded in the baseline key when passed.
    if getattr(args, "ollama_reasoning", None) is not None:
        config.llm.ollama_reasoning = args.ollama_reasoning == "on"

    if args.provider == "openai":
        pipeline = OpenAIPipeline(model=args.model)
    else:
        pipeline = OllamaPipeline(model=args.model)

    texts = pipeline.load_and_split(data_dir=str(CORPUS_DIR))
    if not texts:
        sys.exit(f"error: no documents loaded from {CORPUS_DIR}")
    stamped = stamp_person_metadata(texts)
    people = {doc.metadata.get("person_id") for doc in texts} - {None}
    print(
        f"Loaded corpus: {len(texts)} chunks, {len(people)} people "
        f"({stamped} chunks tagged)"
    )

    if args.provider == "openai":
        pipeline.set_retriever_openai(use_ensemble=True, use_reranker=False)
    else:
        pipeline.set_retriever_ollama(use_ensemble=True, use_reranker=False)
    if pipeline.get_retriever() is None:
        sys.exit("error: retriever was not created (check API key / Ollama)")
    return MatchPipeline.from_pipeline(pipeline)


_STOPWORDS = frozenset(
    "a an at ard and least of or the in with experience years year "
    "domain language working proficiency".split()
)


def _tokens(label: str) -> set:
    return {
        word
        for word in re.findall(r"[a-z0-9+#]+", label.lower())
        if word not in _STOPWORDS
    }


def gaps_align(system_missing: list, oracle_missing: str) -> bool:
    """One missing item, and it's the oracle's one (wording-tolerant)."""
    if len(system_missing) != 1:
        return False
    return bool(_tokens(system_missing[0]) & _tokens(oracle_missing))


# Language names as a Swedish brief states them (a09 says "svenska"); the
# extractor copies the brief's word and the oracle uses the English one.
_LANGUAGE_ALIASES = {
    "svenska": "swedish",
    "engelska": "english",
    "finska": "finnish",
    "norska": "norwegian",
    "danska": "danish",
    "tyska": "german",
}


def _norm(label: str) -> str:
    return " ".join(str(label).lower().split())


def _norm_language(label: str) -> str:
    word = _norm(label)
    return _LANGUAGE_ALIASES.get(word, word)


def oracle_items(record: dict) -> set:
    """The must-have set the extractor should produce, in canonical form."""
    items = {
        ("skill", frozenset(_norm(s) for s in group)) for group in record["must_have"]
    }
    if record.get("min_years"):
        items.add(("years", int(record["min_years"])))
    if record.get("language"):
        items.add(("language", _norm_language(record["language"])))
    if record.get("domain"):
        items.add(("domain", _norm(record["domain"])))
    if record.get("degree"):
        items.add(("education",))
    return items


def extracted_items(requirements, oracle_domain: str = "") -> set:
    """The extractor's must-have set in the same canonical form.

    A domain label counts as the oracle's when it contains the oracle's
    domain word ("automotive industry" ~ "automotive"); years must parse.
    """
    items = set()
    for r in requirements.must:
        if r.kind == "skill":
            items.add(("skill", frozenset(_norm(a) for a in r.alternatives)))
        elif r.kind == "years":
            try:
                items.add(("years", int(float(r.detail))))
            except ValueError:
                items.add(("years", r.detail))
        elif r.kind == "language":
            items.add(("language", _norm_language(r.detail)))
        elif r.kind == "domain":
            detail = _norm(r.detail)
            if oracle_domain and _norm(oracle_domain) in detail:
                detail = _norm(oracle_domain)
            items.add(("domain", detail))
        elif r.kind == "education":
            items.add(("education",))
        else:
            items.add((r.kind, _norm(r.detail)))
    return items


def location_captured(requirements, record: dict) -> bool:
    """True when the extracted location names the brief's city."""
    want = _norm(record.get("location") or "").split(" ")[0]
    got = _norm(getattr(requirements, "location", "") or "")
    return bool(want) and want in got


def evaluate(matcher, assignments, verbose=False, extract_repeats=0):
    counters = {
        "strong_total": 0,
        "strong_found": 0,
        "full_match_correct": 0,
        "ordering_assignments": 0,
        "ordering_clean": 0,
        "partial_surfaced": 0,
        "gap_aligned": 0,
        "extract_oracle": 0,
        "extract_hit": 0,
        "extract_extracted": 0,
        "nice_oracle": 0,
        "nice_hit": 0,
        "location_total": 0,
        "location_hit": 0,
        "stability_total": 0,
        "stability_stable": 0,
    }
    latencies = []

    for record in assignments:
        strong_ids = [e["id"] for e in record["expected_strong"]]
        partial = {e["id"]: e["missing"] for e in record["expected_partial"]}

        start = time.time()
        result = with_retries(matcher.match, record["brief"])
        latencies.append(time.time() - start)

        ranked_ids = [c.person_id for c in result.candidates]
        top = ranked_ids[:SHORTLIST_N]

        # Denominator caps at the shortlist size: with a 25-strong XL
        # pool, "all 5 of the top 5 are oracle-strong" is a perfect
        # score, not 5/25. For pools <= 5 this is provably the old
        # recall (same intersection numerator, same denominator).
        counters["strong_total"] += min(SHORTLIST_N, len(strong_ids))
        counters["strong_found"] += sum(1 for pid in strong_ids if pid in top)

        oracle_has_strong = bool(strong_ids)
        if result.full_match_exists == oracle_has_strong:
            counters["full_match_correct"] += 1

        if strong_ids:
            counters["ordering_assignments"] += 1
            in_tier = set(strong_ids) | set(partial)
            worst_strong = max(
                (ranked_ids.index(pid) for pid in strong_ids if pid in ranked_ids),
                default=len(ranked_ids),
            )
            clean = not any(pid not in in_tier for pid in ranked_ids[:worst_strong])
            counters["ordering_clean"] += int(clean)
        else:
            clean = True  # a08: ordering has no strong anchor to violate

        for candidate in result.candidates:
            if candidate.person_id in partial:
                counters["partial_surfaced"] += 1
                counters["gap_aligned"] += int(
                    gaps_align(candidate.missing, partial[candidate.person_id])
                )

        oracle = oracle_items(record)
        extracted = extracted_items(result.requirements, record.get("domain") or "")
        counters["extract_oracle"] += len(oracle)
        counters["extract_extracted"] += len(extracted)
        counters["extract_hit"] += len(oracle & extracted)
        oracle_nice = {_norm(n) for n in record.get("nice_to_have", [])}
        got_nice = {_norm(n) for n in result.requirements.nice}
        counters["nice_oracle"] += len(oracle_nice)
        counters["nice_hit"] += len(oracle_nice & got_nice)
        if record.get("location"):
            counters["location_total"] += 1
            counters["location_hit"] += int(
                location_captured(result.requirements, record)
            )
        if extract_repeats:
            samples = [
                extracted_items(
                    with_retries(matcher.extract_requirements, record["brief"]),
                    record.get("domain") or "",
                )
                for _ in range(extract_repeats)
            ]
            counters["stability_total"] += 1
            counters["stability_stable"] += int(all(x == samples[0] for x in samples))

        if verbose:
            print(f"\n  {record['id']}: {record['title']}")
            print(
                "    extracted musts: "
                + "; ".join(r.label for r in result.requirements.must)
            )
            missed = sorted(str(i) for i in oracle - extracted)
            extra = sorted(str(i) for i in extracted - oracle)
            if missed or extra:
                print(f"    extraction: missed={missed or '-'} extra={extra or '-'}")
            else:
                print("    extraction: exact")
            print(
                f"    expected strong: {strong_ids or '-'} | "
                f"full_match={result.full_match_exists} "
                f"(correct={result.full_match_exists == oracle_has_strong}) "
                f"ordering_clean={clean}"
            )
            for candidate in result.candidates[:SHORTLIST_N]:
                marker = (
                    "S"
                    if candidate.person_id in strong_ids
                    else "P" if candidate.person_id in partial else "-"
                )
                print(
                    f"    [{marker}] {candidate.person_id} {candidate.name}: "
                    f"{candidate.tier}"
                    + (
                        f" (missing: {'; '.join(candidate.missing)})"
                        if candidate.missing
                        else ""
                    )
                )

    return counters, latencies


def baseline_key(args) -> str:
    model = args.model or "gpt-4o-mini"
    set_name = getattr(args, "set", "staffing")
    key = f"{args.provider}:{model}|k={args.k}|chain=match|set={set_name}"
    # Suffix only when the flag was passed, like run_eval.py — the
    # committed cloud key stays untouched.
    if getattr(args, "ollama_reasoning", None):
        key += f"|ollama-reasoning={args.ollama_reasoning}"
    return key


def check_baseline(scores: dict, key: str) -> int:
    if not BASELINE_PATH.exists():
        print("No baseline.json; skipping gate.")
        return 0
    baselines = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    base = baselines.get(key)
    if base is None:
        print(f"No baseline for key '{key}'; run --update-baseline to record.")
        return 0
    recorded_sha = base.get("data_sha")
    fingerprint = data_fingerprint(GOLDEN_PATH, [CORPUS_DIR])
    if recorded_sha and recorded_sha != fingerprint:
        print(
            f"DATA MISMATCH: the staffing golden set or CV corpus changed "
            f"since this baseline was recorded (data_sha {recorded_sha} -> "
            f"{fingerprint}). Re-record with --update-baseline if "
            "intentional."
        )
        return 1
    recorded = base.get("metrics", {})
    failures = []
    for metric in GATED_METRICS:
        if metric in recorded and scores[metric] < recorded[metric] - TOLERANCE:
            failures.append(
                f"{metric}: {scores[metric]:.3f} < baseline "
                f"{recorded[metric]:.3f} - {TOLERANCE}"
            )
    if failures:
        print("\nBASELINE GATE FAILED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"\nBaseline gate passed ({key}).")
    return 0


def update_baseline(scores: dict, key: str) -> None:
    """Add/replace ONE key, house schema, no reordering of the others.

    baseline.json is a committed artifact: rewriting it sorted (or with a
    different entry shape) churns every key in the diff and breaks the
    'committed baselines stay byte-identical' rule. Insertion order is
    preserved by dict round-trip; only this run's key changes.
    """
    baselines = {}
    if BASELINE_PATH.exists():
        baselines = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    baselines[key] = {
        "metrics": {metric: round(scores[metric], 3) for metric in GATED_METRICS},
        "date": time.strftime("%Y-%m-%d"),
        "data_sha": data_fingerprint(GOLDEN_PATH, [CORPUS_DIR]),
    }
    BASELINE_PATH.write_text(
        json.dumps(baselines, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Baseline updated: {key} -> {baselines[key]}")


def main() -> int:
    global CORPUS_DIR, GOLDEN_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument(
        "--set",
        choices=sorted(SETS),
        default="staffing",
        help="staffing = the 40-person bench; staffing_xl = the scale test",
    )
    parser.add_argument(
        "--ollama-reasoning",
        choices=["on", "off"],
        default=None,
        help="thinking control for local models (unset = model default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="first N assignments (pilot timing; never gated)",
    )
    parser.add_argument(
        "--extract-repeats",
        type=int,
        default=0,
        help="N extra extraction samples per brief for extract_stability (0 = off)",
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--no-baseline-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    CORPUS_DIR, GOLDEN_PATH = SETS[args.set]
    if args.limit is not None and args.update_baseline:
        sys.exit("error: --limit runs are partial; refusing --update-baseline")

    assignments = load_assignments()
    if args.limit is not None:
        assignments = assignments[: args.limit]
    matcher = build_matcher(args)

    print(f"Matching {len(assignments)} assignments...")
    counters, latencies = evaluate(
        matcher,
        assignments,
        verbose=args.verbose,
        extract_repeats=args.extract_repeats,
    )

    n_assign = len(assignments)
    scores = {
        "strong_recall_at_5": (
            counters["strong_found"] / counters["strong_total"]
            if counters["strong_total"]
            else 0.0
        ),
        "full_match_accuracy": counters["full_match_correct"] / n_assign,
        "ordering_clean_rate": (
            counters["ordering_clean"] / counters["ordering_assignments"]
            if counters["ordering_assignments"]
            else 0.0
        ),
        "gap_alignment": (
            counters["gap_aligned"] / counters["partial_surfaced"]
            if counters["partial_surfaced"]
            else 0.0
        ),
        "extract_must_recall": (
            counters["extract_hit"] / counters["extract_oracle"]
            if counters["extract_oracle"]
            else 0.0
        ),
        "extract_must_precision": (
            counters["extract_hit"] / counters["extract_extracted"]
            if counters["extract_extracted"]
            else 0.0
        ),
        "extract_nice_recall": (
            counters["nice_hit"] / counters["nice_oracle"]
            if counters["nice_oracle"]
            else 0.0
        ),
        "extract_location_rate": (
            counters["location_hit"] / counters["location_total"]
            if counters["location_total"]
            else 0.0
        ),
    }
    if args.extract_repeats:
        scores["extract_stability"] = (
            counters["stability_stable"] / counters["stability_total"]
            if counters["stability_total"]
            else 0.0
        )

    print("\nScores (rate metrics show the 95% binomial CI):")
    sizes = {
        "strong_recall_at_5": counters["strong_total"],
        "full_match_accuracy": n_assign,
        "ordering_clean_rate": counters["ordering_assignments"],
        "gap_alignment": counters["partial_surfaced"],
        "extract_must_recall": counters["extract_oracle"],
        "extract_must_precision": counters["extract_extracted"],
        "extract_nice_recall": counters["nice_oracle"],
        "extract_location_rate": counters["location_total"],
        "extract_stability": counters["stability_total"],
    }
    for name, value in scores.items():
        n = sizes[name]
        half = ci95_halfwidth(value, n) if n else 0.0
        suffix = " (informational)" if name not in GATED_METRICS else ""
        print(f"  {name}: {value:.3f} ±{half:.3f} (n={n}){suffix}")
    print("Efficiency (informational):")
    print(f"  avg_match_latency_s: {sum(latencies) / len(latencies):.1f}")

    if args.limit is not None:
        print("\nPilot run (--limit): no gating, no baseline.")
        return 0
    key = baseline_key(args)
    if args.update_baseline:
        update_baseline(scores, key)
        return 0
    if args.no_baseline_check:
        return 0
    return check_baseline(scores, key)


if __name__ == "__main__":
    sys.exit(main())
