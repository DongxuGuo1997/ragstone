# Experiments

This is the experiment log behind Ragstone's defaults. Every choice in the
pipeline — embedding model, whether to rerank, how many chunks to retrieve,
which chain type — was decided by measurement against the bundled evaluation
harness, not by intuition. This file records the hypothesis, method, result,
and decision for each, so the defaults are auditable and reproducible.

All numbers below come from `evals/run_eval.py` against the bundled corpus.
Re-run any row yourself with the command shown.

## Method (why these numbers mean something)

- **Corpus** (`evals/corpus/`): 6 short documents about *fictional* subjects
  (space missions, a city transit system, a coffee roaster, two solar-panel
  guides). Fictional facts matter: the model cannot answer from pretraining,
  so a wrong answer cleanly indicates a retrieval or grounding failure rather
  than memorized knowledge, and hallucination is detectable. The corpus
  deliberately includes near-duplicate pairs (Aurora-7 / Aurora-9, the
  Helios / Borealis solar guides) as *distractors* — easily-confused
  documents that punish imprecise retrieval.
- **Golden set** (`evals/golden.jsonl`): 38 hand-written cases — 24 factual,
  5 distractor, 4 multi-hop, 3 unanswerable, 2 paraphrase.
- **Layer 1 — retrieval** (deterministic, free): `hit_rate@k` (did any
  retrieved chunk contain all required strings) and **MRR** (mean reciprocal
  rank of the first hit). Unanswerable cases are excluded.
- **Layer 2 — generation** (LLM-as-judge, `gpt-4o-mini`, costs cents):
  **correctness** vs the gold answer and **faithfulness** to the retrieved
  context (a hallucination check). Judge prompts are in `evals/judge.py`, in
  the open.
- **Regression gate**: every run is compared to `evals/baseline.json` and
  fails if any metric drops more than 0.05 below baseline — so quality
  regressions surface as failed runs, not silent drift.

Single corpus, 35 scorable retrieval cases — treat sub-0.01 differences as
noise, not signal. The point is direction and magnitude, not decimal places.

---

## Experiment 1 — Embedding model: `ada-002` → `text-embedding-3-small`

**Hypothesis.** The newer, ~5× cheaper `text-embedding-3-small` is at least
as good as the legacy `text-embedding-ada-002` for retrieval.

**Method.** Layer 1, ensemble (BM25 + vector), `k=4`, no reranker.
```bash
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002 python evals/run_eval.py --mode retrieval --k 4
python evals/run_eval.py --mode retrieval --k 4   # default is text-embedding-3-small
```

| Embedding model            | hit_rate@4 | MRR   |
|----------------------------|-----------:|------:|
| text-embedding-ada-002     | 0.971      | 0.902 |
| **text-embedding-3-small** | **1.000**  | 0.895 |

**Result — and the honest reading.** The upgrade is a **coverage** win, not a
ranking win. `3-small` retrieves the gold chunk for *every* answerable case
(hit_rate 0.971 → 1.0 — it recovers the one case `ada-002` missed in the
top 4), while mean rank is essentially flat (MRR 0.902 → 0.895, inside the
noise floor for this corpus). A naive "newer model = better" claim would be
wrong here; the gain is recall, not ordering.

**Decision.** Adopt `text-embedding-3-small`. It is cheaper *and* improves
coverage — and you cannot rerank a chunk you never retrieved, so coverage is
the property that gates everything downstream. Ranking is handled separately
(Experiment 2).

---

## Experiment 2 — Cross-encoder reranking

**Hypothesis.** A second-stage cross-encoder that rescores a wide candidate
pool will improve ranking on the distractor cases, where first-stage
similarity confuses near-duplicate documents.

**Method.** Layer 1, ensemble fetches 20 candidates, then
`cross-encoder/ms-marco-MiniLM-L-6-v2` rescores and keeps the top 4.
```bash
python evals/run_eval.py --mode retrieval --k 4            # no rerank
python evals/run_eval.py --mode retrieval --k 4 --rerank   # requires the `rerank` extra
```

| Retrieval (k=4)        | hit_rate@4 | MRR       |
|------------------------|-----------:|----------:|
| ensemble               | 1.000      | 0.895     |
| **ensemble + rerank**  | 1.000      | **0.964** |

**Result.** MRR 0.895 → 0.964 at unchanged coverage — the **single largest
ranking lever** measured, and it lands exactly where expected: the gold chunk
moves up to rank 1 on the distractor cases. Cost is a ~80 MB local model and
per-query CPU scoring of 20 pairs — **no API calls**, so it works in
fully-offline / Ollama mode too.

**Decision.** Ship as an opt-in extra (`pip install -e ".[rerank]"`), off by
default. It is the recommended setting for precision-sensitive use, but
keeping it optional keeps the base install light and CPU-only deployments
fast. Defaulting it *on* would impose a model download and per-query latency
that not every use case wants — a knob, not a mandate.

---

## Experiment 3 — Retrieval depth `k` (the context-budget knob)

**Hypothesis.** `k` trades retrieval coverage against the number of context
tokens shipped to the LLM on every query. There should be a knee: the
smallest `k` that still finds every answer.

**Method.** Layer 1, ensemble, `text-embedding-3-small`, varying `k`.
```bash
python evals/run_eval.py --mode retrieval --k 2
python evals/run_eval.py --mode retrieval --k 4
python evals/run_eval.py --mode retrieval --k 6
```

| k   | hit_rate@k | MRR   | notes                                  |
|-----|-----------:|------:|----------------------------------------|
| 2   | 0.857      | 0.814 | some answers live in the 3rd–4th chunk |
| 4   | 1.000      | 0.895 | full coverage                          |
| 6   | 1.000      | 0.895 | no gain over k=4                       |

**Result.** The knee is at **k=4**. Below it, coverage collapses (k=2 misses
~14% of answers whose evidence sits just outside the cut). Above it, coverage
is already saturated — k=6 finds nothing new and simply pays for ~50% more
context tokens on every single query.

**Decision.** Default `k=4`: the minimum depth that achieves full coverage on
this corpus. This is the highest-leverage *cost* knob in the pipeline — every
extra chunk is extra input tokens on every query — so spending it without a
measured retrieval benefit is pure waste.

---

## Experiment 4 — Chain type: does query expansion justify its cost?

**Hypothesis.** `multi_query` (generate paraphrases of the question, retrieve
each, union the results) and `fusion` (generate queries, then reciprocal-rank-
fuse the result lists) should beat plain single-query RAG — the classic
argument is that one phrasing under-retrieves, so more phrasings help.

**Method.** Layer 2 (full), `k=4`, `gpt-4o-mini` for both answering and
judging, three chain types, same corpus and golden set.
```bash
python evals/run_eval.py --mode full --chain-type simple
python evals/run_eval.py --mode full --chain-type multi_query
python evals/run_eval.py --mode full --chain-type fusion
```

| Chain type   | correct_rate | faithful_rate | pipeline cost per question        |
|--------------|-------------:|--------------:|-----------------------------------|
| **simple**   | 0.947        | 0.947         | 1 LLM call, 1 retrieval           |
| multi_query  | 0.947        | 0.974         | ~2 LLM calls, ~3 retrievals       |
| fusion       | 0.974        | 0.921         | ~2 LLM calls, ~4 retrievals       |

**Result.** There is **no meaningful difference**. With 38 cases, one case is
~2.6 points — and every gap in the table above is exactly one case (simple
↔ fusion: one more correct; simple ↔ multi_query: one more faithful; and
fusion *loses* one on faithfulness). That is noise, not signal. Meanwhile
`multi_query` and `fusion` each add a query-generation LLM call and fan
retrieval out across 3–4 generated queries — roughly **2× the pipeline's LLM
calls and 3–4× its retrievals per question** — for zero measured gain.

Why: query expansion earns its cost when a single phrasing *under-retrieves*
— sprawling, multi-faceted, or vaguely-worded questions. This corpus's
questions are well-specified and largely single-hop, so there is little
phrasing ambiguity for expansion to repair. (Layer 1 retrieval metrics are
identical across all three rows by construction: Layer 1 probes the base
retriever with the literal question, while expansion happens *inside* the
generation chain — so its effect shows up only in Layer 2, which is what
actually reaches the user.)

**Decision.** Default to **simple**. `multi_query` and `fusion` stay available
via `--chain-type` for corpora where phrasing is genuinely ambiguous or
evidence is scattered across documents — but defaulting to them *here* would
double cost and latency for no measured quality improvement. Choosing the
plainer option *because the data says the sophisticated one doesn't pay* is
the entire point of having the harness.

---

## Defaults, decided by the numbers above

| Choice            | Default                      | Decided by   | Why                                            |
|-------------------|------------------------------|--------------|------------------------------------------------|
| Embedding model   | `text-embedding-3-small`     | Experiment 1 | full coverage, ~5× cheaper than ada-002        |
| Retrieval         | ensemble (BM25 + vector)     | —            | lexical + semantic recall                      |
| Retrieval depth   | `k=4`                        | Experiment 3 | the coverage knee — minimum k for hit_rate 1.0 |
| Reranking         | optional (`[rerank]` extra)  | Experiment 2 | biggest ranking lever (MRR +0.07), but a knob  |
| Chain type        | `simple`                     | Experiment 4 | expansion adds cost, not quality, on this data |

Every one of these will be re-examined the moment the corpus changes — which
is the point: the harness makes "should this default change?" a measurable
question instead of an argument.

