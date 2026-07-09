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
- **Golden set** (`evals/golden.jsonl`): 43 hand-written cases — 24 factual,
  5 distractor, 4 multi-hop, 3 unanswerable, 2 paraphrase, 5 multi-turn.
  (The multi-turn cases were added when conversation quality became a gated
  metric; experiments that predate them ran on the original 38 and say so.)
- **Layer 1 — retrieval** (deterministic, free): `hit_rate@k` (did any
  retrieved chunk contain all required strings) and **MRR** (mean reciprocal
  rank of the first hit). Unanswerable and multi-turn cases are excluded.
  One honest caveat: multi-hop cases count as a hit when all needles appear
  anywhere in the top-k *union*, and are pinned at rank 1 — so read MRR as
  slightly optimistic on the multi-hop slice.
- **Layer 2 — generation** (LLM-as-judge, `gpt-4o-mini`, costs cents):
  **correctness** vs the gold answer and **faithfulness** to the retrieved
  context (a hallucination check). Judge prompts are in `evals/judge.py`, in
  the open. The judge and the answering model share a model family
  (gpt-4o-mini grading gpt-4o-mini), so absolute scores may carry some
  self-preference inflation; every comparison in this file uses the same
  judge on both sides, so the *deltas* are the meaningful signal.
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

## Experiment 5 — Chunk size (the splitting knob nobody re-examines)

**Hypothesis:** the `chunk_size=1000 / chunk_overlap=200` defaults were
inherited, not chosen — smaller chunks might sharpen retrieval precision.

**Method:** retrieval layer only (deterministic, no judge), sweeping
size/overlap with `python evals/run_eval.py --mode retrieval --chunk-size N
--chunk-overlap M --no-baseline-check`.

| chunk_size / overlap | chunks | hit_rate | MRR   |
|----------------------|--------|----------|-------|
| 500 / 100            | 36     | 0.914    | 0.788 |
| 800 / 160            | 22     | 0.943    | 0.850 |
| **1000 / 200**       | **16** | **1.0**  | **0.895** |
| 1500 / 300           | 12     | 1.0      | 0.914 |

**Decision.** Keep **1000/200**. The hypothesis was wrong — smaller chunks
*hurt* on this corpus (facts get split away from their subjects, and BM25
loses term co-occurrence). 1500's MRR edge is a single-case rank shift on a
12-chunk corpus (k=4 retrieves a third of it — easy mode) and costs ~50%
more context tokens per answer. Re-run this sweep on any real corpus; the
knee will move with document structure.

---

## Experiment 6 — Ensemble weights (BM25 vs vector)

**Hypothesis:** the `[0.4, 0.6]` BM25/vector split was a guess; maybe the
lexical side deserves more weight.

**Method:** retrieval layer, `--bm25-weight W --no-baseline-check`
(vector weight is `1 − W`).

| BM25 weight | hit_rate | MRR   |
|-------------|----------|-------|
| 0.2         | 1.0      | 0.886 |
| **0.4**     | **1.0**  | **0.898** |
| 0.6         | 0.914    | 0.852 |
| 0.8         | 0.914    | 0.821 |

**Decision.** Keep **0.4**. The guess survives measurement — and the sweep
shows the failure direction clearly: leaning on BM25 costs coverage
(hit_rate drops when lexical matching outvotes semantics on paraphrased
questions). Now configurable via `RAGSTONE_BM25_WEIGHT` for per-corpus
tuning.

---

## Experiment 7 — Concurrent embedding ingestion

**Hypothesis:** corpus embedding — the dominant big-corpus ingestion cost —
is network-bound and serial: the provider slices the corpus into batches
but sends them one HTTP request at a time. Issuing batches from a small
thread pool should overlap the round-trips.

**Method:** 4,000 synthetic chunks, real API (`text-embedding-3-small`),
sequential `embed_documents` vs `embed_texts_parallel` (batch 500, 4
workers). Retrieval eval re-run on the new index path as the quality gate.

| Ingestion            | Wall time | hit_rate | MRR   |
|----------------------|-----------|----------|-------|
| sequential           | 8.7 s     | 1.0      | 0.898 |
| **parallel (4×500)** | **2.8 s** | 1.0      | 0.898 |

**Decision: parallel by default** (`RAGSTONE_EMBED_BATCH_SIZE=500`,
`RAGSTONE_EMBED_WORKERS=4`). **3.1× faster** ingestion with vectors — and
therefore retrieval metrics — identical to the sequential path. Order
preservation is unit-tested under adversarial completion order, and any
failed batch fails the whole ingestion: a partially embedded corpus is
never indexed silently. The win grows with corpus size (it's pure
round-trip overlap) and is bounded only by provider rate limits.

---

## Experiment 8 — Corrective RAG: does self-correction pay?

**Hypothesis:** grading retrieved passages before answering — and
rewriting the query when they fail the grade — should recover questions
whose phrasing misses the corpus wording, and make "I don't know" an
evidence-based verdict instead of a model mood.

**Method:** `--chain-type corrective` vs the `simple` baseline, full
judged eval. The corrective chain is a LangGraph with a conditional
grade → (answer | rewrite→retrieve cycle | refuse) topology; grading and
rewriting run on the utility model.

| | simple | agent (Exp 4 follow-up) | **corrective** |
|---|---|---|---|
| correct_rate | 0.947 | 0.947 | **0.974** |
| faithful_rate | 0.947–0.974 | 0.974 | 0.974 |
| multi_turn_correct | 1.0 | — | 1.0 |
| avg latency | ~1.5 s | 2.7 s | 2.8 s |
| total tokens | ~45 k | ~57 k | 94 k |

**Decision (as of n=43).** The first technique in this repo that appeared
to *buy* correctness: q31 (the Violet Line fare — the one case every
other chain failed) passed under corrective, because the rewrite found
phrasing the first retrieval missed. The price: ~1.9× latency and ~2.1×
tokens.

> **Superseded by Experiment 9.** The +0.027 here is exactly one flipped
> case at n=43. On the 224-case set the win does not replicate — the
> quality difference lands inside sampling noise while the 2× cost
> remains. Kept unedited above as a worked example of why sample size
> gates conclusions.

---

## Experiment 9 — Scaling the golden set: does the corrective win replicate?

**Hypothesis (meta):** at n=43, one flipped case moves correct_rate by
~2.6 points, so Experiment 8's verdict ("corrective buys +0.027") rests
on a single question. A larger set should either confirm it or expose it
as sampling noise.

**Method:** the golden set was scaled to **224 cases** (`--set large`)
over an extended 16-document fictional corpus with engineered distractors
(a third solar panel one digit away from the other two, sibling missions,
a rival company). Generated cases are validated mechanically: every
`must_contain` needle must appear verbatim in its claimed source document
(11 generator hallucinations were auto-rejected). The smoke set and its
6-document corpus are untouched, so all prior baselines stay comparable.

First, the harder corpus recalibrates everything:

| simple | smoke (n=43) | large (n=224) |
|---|---|---|
| hit_rate | 1.0 | 0.955 |
| MRR | 0.895 | 0.828 |
| multi_turn_correct | 1.0 (n=5) | 0.769 (n=13) |

Retrieval misses are dominated by the engineered distractors (Helios
torque losing to Corona/Borealis torque chunks), and the multi-turn 1.0
was small-n flattery — comparative follow-ups ("how does it compare to
its predecessor?") emerge as a real failure class.

The rematch, at ~0.5 pp resolution:

| n=224 | simple | corrective |
|---|---|---|
| correct_rate | **0.943** | 0.934 |
| faithful_rate | 0.910 | 0.929 |
| multi_turn_correct | 0.769 | 0.769 |
| avg latency | **1.33 s** | 2.28 s |
| total tokens | **242 k** | 486 k |

**Decision.** Experiment 8's conclusion **does not replicate**: at n=224
the correctness difference (−0.9 pp) and faithfulness difference
(+1.9 pp) are both inside the ±3 pp binomial noise band, while the 2×
cost is not noise. `corrective` is re-classified from "buys correctness"
to "no measured quality gain on this corpus, at double the cost" — and
`simple` keeps the default with a stronger mandate than before.

The meta-lesson is the real result: **the n=43 verdict was wrong in a
way only a bigger sample could reveal.** Design-option comparisons now
run on `--set large` by policy; the smoke set remains what it always
was — a fast per-commit regression gate, not an instrument for verdicts.

---

## Experiment 10 — The retrieval-slice trap, and two prompt fixes measured

**Question.** Two things needed settling after Experiment 9's failure
taxonomy: (a) the large-set `--k` sweep suggested `k=6` beats `k=4` on hit
rate (0.955 → 0.980) — should the default change? (b) two targeted prompt
fixes — the rephrase prompt now forces comparative follow-ups to name
every compared entity, and the answer prompt forbids inventing identifiers
not in the context — do they help at scale?

**Method.** Three full runs on `--set large`, `chain=simple`:
v1 (baseline, k=4, old prompts), v2 (k=6 + new prompts), v3 (k=4 + new
prompts — isolating the prompts from the k change).

| n=224 | v1 (k=4, old prompts) | v2 (k=6) | v3 (k=4, new prompts) |
|---|---|---|---|
| hit_rate | 0.955 | 0.940 | 0.955 |
| correct_rate | 0.943 | 0.938 | 0.929 |
| faithful_rate | 0.910 | 0.905 | 0.915 |
| multi_turn_correct | 0.769 | 0.692 | **0.846** |
| multi_turn_faithful | 0.923 | 0.692 | 0.846 |
| total tokens | 242 k | **352 k (+46%)** | 246 k |

**The k=6 regression is the headline.** The sweep that recommended k=6
only re-scored the *evaluation slice* — which chunks land in the top-k of
a standalone retrieval call. Setting `similarity_k=6` changes something
else entirely: what the **generator actually reads**. Two extra chunks of
engineered distractors (Corona K-7 specs next to Helios questions,
Aurora-11 next to Aurora-9) diluted the context; the model started
blending entities, multi-turn faithfulness collapsed 0.923 → 0.692, and
every answer paid +46% context tokens. `k=4` was reverted the same day,
with the reasoning recorded on the config field itself.

**The prompt-fix verdict, honestly stated.** At identical retrieval
(v1 vs v3, both k=4): single-turn correctness moved −1.4 pp (three cases,
inside the ±3 pp noise band) and faithfulness +0.5 pp — consistent with
the no-invented-identifiers clause trading a little eagerness for
grounding, and costing nothing. The multi-turn columns need more care
than the table suggests, because n=13 makes every case worth 7.7 pp:
correctness rose one case (0.769 → 0.846) and faithfulness *fell* one
case (0.923 → 0.846). Worse for the tidy narrative: the case that flipped
to pass (`gmt184`, "What system is it compatible with?") is not a
comparative question at all — its gold answer was corrected in the same
commit (it had been vaguer than a correct system answer), so its flip is
a *scoring* fix, not a prompt effect, and the v1 row is not a clean A/B
against v3. What the prompt changes can honestly claim: no measurable
harm at n=224, a plausible mechanism, and the smoke-set multi-turn lift
(0.8 → 1.0 — itself one case at n=5). They ship because they are
principled and free, not because n=13 proves them.

**Decision.** `similarity_k` stays **4**; both prompt changes ship; the
large-simple baseline is re-recorded. The transferable lesson — now a
standing rule for this repo: **a knob that changes what the generator
reads must be judged end-to-end, never by a retrieval-slice metric.** The
slice metric answers "did the needle land in top-k?"; it is silent about
what the other k−1 chunks do to the answer. (And its corollary from the
gmt184 confound: **never edit the golden set and the system under test in
the same measured comparison.**)

### Postscript — v4 baselines on the fixed harness

A code-quality audit then found two measurement defects in the harness
itself, both fixed before the current baselines were recorded:

1. **Faithfulness was judged against the wrong context for non-simple
   chains.** Single-turn cases re-retrieved with the raw question and
   judged against that — but a corrective answer may be grounded in a
   *rewritten*-query retrieval the judge never saw. The judge now sees
   exactly the documents the pipeline used (all of them, deduplicated).
   This bias ran AGAINST corrective in Experiments 8–9.
2. **Two "unanswerable" cases were answerable** (`g172` warranty, `g173`
   fare — both stated in the corpus), so they failed correct answers and
   rewarded false refusals. Converted to factual cases with verified
   needles. Eval sessions are also namespaced per run now, so persistent
   checkpoint backends cannot leak history between runs.

Both chains re-measured on the fixed harness and corrected golden set
(identical prompts, k=4):

| n=224, v4 | simple | corrective |
|---|---|---|
| correct_rate | 0.943 | 0.948 |
| faithful_rate | 0.938 | **0.957** |
| multi_turn_correct | 0.846 | 0.846 |
| multi_turn_faithful | 0.846 | 0.923 |
| avg latency | **1.57 s** | 2.34 s |
| total tokens | **246 k** | 493 k |

Read fairly: with the anti-corrective judging bias removed, corrective's
faithfulness edge (+1.9 pp, ~4 cases) looks more consistent than
Experiment 9 suggested — its grade-then-answer loop really does refuse or
re-ground some answers `simple` gets slightly wrong. But +1.9 pp is still
at the edge of the ±3 pp noise band, the correctness gap (+0.5 pp) is one
case, and the 2× token cost is exact. The Experiment-9 verdict stands:
**no quality gain large enough to buy at double cost; `simple` stays the
default** — now with the caveat that corrective's *faithfulness* value on
adversarial corpora is plausible rather than refuted. The meta-lesson
compounds: the harness is part of the system under test; audit it with
the same rigor as the pipeline.

---

## Experiment 11 — Bounding the answerer==judge bias

**Question.** Every judged score in this file was graded by the same
model that wrote the answers (gpt-4o-mini). LLM judges are known to
favor their own outputs — so how much of our 0.94-ish is self-flattery?

**Method.** Re-run the large set (simple chain, k=4, identical config)
with `--judge-model gpt-4.1-mini` — a different model generation from the
same provider — and compare against the self-judged v4 baseline. One
caveat stated up front: the pipeline re-generates answers on each run, so
the deltas below bundle judge disagreement with run-to-run generation
variance; they are an upper bound on the judge effect, not a pure
isolation (re-judging *stored* answers would isolate it — noted in the
roadmap).

| n=224, same system | judged by gpt-4o-mini (self) | judged by gpt-4.1-mini |
|---|---|---|
| correct_rate | 0.943 ±0.031 | 0.919 ±0.037 |
| faithful_rate | 0.938 ±0.033 | 0.934 ±0.034 |
| multi_turn_correct | 0.846 | 0.846 |
| multi_turn_faithful | 0.846 | 1.000 |

**Result.** The cross-judge scores land **2.4 pp lower on correctness**
and 0.4 pp lower on faithfulness — directionally consistent with
self-preference, small in magnitude, and inside the overlapping CIs. The
multi-turn faithfulness *rose* two cases under the stricter judge (n=13
noise, and the opposite direction of self-preference). No conclusion in
this file flips under the alternate judge.

**Decision.** Absolute judged scores should be read with a ~2 pp
self-preference haircut in mind; comparative deltas (all decisions in
this file) are unaffected since both sides share a judge. The
cross-judge column stays reproducible via
`--judge-model gpt-4.1-mini --no-baseline-check`. A different-*provider*
judge remains the stronger version of this check (roadmap 3.1 follow-up).

---

## Experiment 12 — Contextual chunk enrichment (document identity in the chunk)

**Hypothesis.** The dominant failure cluster at n=224 is engineered
distractors: "clean every 4 months with a microfiber cloth" embeds almost
identically whether it came from the Helios or the Corona manual. A chunk
that *carries its document identity in its own text* ("[Source document:
corona_solar_guide.md]") should disambiguate at retrieval time — and,
because the line also reaches the generator, at answer time too.

**Method.** `RAGSTONE_CHUNK_CONTEXT=source` prepends the identity line to
every chunk before embedding/BM25 indexing (free, deterministic; an
"llm" mode generates a richer situating sentence per chunk at ~1 utility
call each). One large-set run vs the v4 baseline, identical everything
else.

| n=224, simple | baseline (off) | source enrichment |
|---|---|---|
| hit_rate | 0.955 ±0.029 | **0.970 ±0.024** |
| mrr | 0.824 | **0.844** |
| correct_rate | 0.943 ±0.031 | 0.948 ±0.030 |
| faithful_rate | 0.938 ±0.033 | **0.967 ±0.024** |
| multi_turn_faithful | 0.846 | 0.923 |
| total tokens | 246 k | 260 k (+5.7%) |
| avg latency | 1.57 s | 1.56 s |

**Result.** Every metric moved in the predicted direction at once:
retrieval misses fell 9 → 6, MRR rose 2 points, faithfulness gained
2.9 pp. Each individual delta sits at or inside its CI edge — but four
independent metrics agreeing with the mechanism is stronger evidence
than any one of them, and it is the *contrast with Experiment 10* that
matters: k=6 bought the retrieval slice by poisoning generation, while
enrichment improved both layers together, because it adds *identity*
rather than *volume* to the context. Cost: +5.7% tokens, zero latency.

**Decision.** `chunk_context=source` becomes the **default**; baselines
re-recorded. The "llm" mode stays opt-in for corpora whose filenames
carry no meaning (ours encode the entity, so the cheap mode captures
most of the value — test the expensive mode before paying for it).

---

## Experiment 13 — Vector-store backends: parity before features

**Question.** Qdrant and pgvector backends were added behind the same
`VectorStoreProxy` ABC as FAISS (ROADMAP 4.1) for durability, filtering,
and multi-process access. The acceptance rule was set before writing any
code: **retrieval metrics must be identical to FAISS at equal k** — a
store that changes what gets retrieved isn't a backend, it's a different
system.

**Method.** The large-set retrieval slice (202 scorable cases, real
embeddings, k=4, enrichment on) once per backend via `--vector-store`;
identical corpus, chunks, and ensemble on every run. Qdrant ran in
embedded local mode; pgvector against the compose Postgres. A unit-level
parity test (deterministic embeddings, tie-free by construction) guards
the same property offline on every CI run.

| n=202, k=4 | FAISS (in-process) | Qdrant (embedded) | pgvector (server) |
|---|---|---|---|
| hit_rate | 0.970 | 0.970 | 0.970 |
| mrr | 0.844 | 0.844 | 0.844 |
| wall time (ingest + 202 queries) | 36.5 s | 37.1 s | 34.8 s |

**Result.** Bit-for-bit metric parity, and wall times within ±5% —
dominated by the embedding API on all three, which is the honest reading:
at 231 chunks, backend performance differences are noise. What the
backends buy is *operational*: the index survives restarts, lives outside
the process, and (Qdrant server mode / pgvector) can be shared — the
properties that start mattering exactly where this corpus ends.

**Decision.** FAISS stays the default (zero setup, right for demos and
evals). `VECTOR_STORE_TYPE=qdrant|pgvector` are supported first-class:
parity is enforced by test, both reuse the parallel-embedding ingest, and
`docker compose up` provides both servers. One deliberate naming rule
carried over from the Chroma incident: collections are unique per
pipeline unless `RAGSTONE_COLLECTION` pins a stable name.

---

## Experiment 14 — Incremental ingestion via the embedding cache

**Question.** Ingestion re-embeds the whole corpus even when one document
changed. ROADMAP 1.6 asked for incremental indexing; the implementation
chose a content-addressed embedding cache (SQLite, keyed by SHA-256 of
model + exact chunk text) over per-backend index mutation — one
mechanism, every backend, no ID/deletion bookkeeping. Does it deliver,
and can it move retrieval results?

**Method.** Time three ingests of the merged eval corpus (16 docs, 63
chunks, real embeddings, fresh cache file): cold cache, warm cache
(nothing changed), and one edited document. Correctness is not measured
but *proven*: vectors round-trip through the cache in float64 exactly
(unit-tested), so a warm-cache index is bit-identical to a cold build.

| ingest (63 chunks) | wall time | vs cold |
|---|---|---|
| cold cache | 0.80 s | 1× |
| warm cache, unchanged corpus | 0.03 s | **30×** |
| one document edited | 0.34 s | 2.4× |

**Result.** Embedding API work now scales with *changed* chunks, not
corpus size: the unchanged-corpus case is a pure index rebuild from
cached vectors. This corpus is small enough that even cold ingest is
sub-second — the ratios are the result, and they compound with corpus
size since the cold cost is linear in chunks while the warm cost stays
near-constant plus the edited delta.

**Decision.** On by default (`RAGSTONE_EMBED_CACHE=off` to benchmark).
Honest scope note: the index itself is still rebuilt each ingest —
cheap up to tens of thousands of chunks; true index mutation
(add/delete points in place) is the follow-on that matters beyond that,
and stays on the roadmap.

---

## Experiment 15 — Query routing: the gate was set first, and it failed

**Hypothesis.** Corrective RAG's faithfulness edge (Experiment 12:
0.986 vs 0.967) lives on confusion-prone questions; a cheap classifier
routing only *those* to the corrective chain should capture most of the
edge at a fraction of the 2× cost. **Acceptance gate, registered before
the run:** quality ≥ simple with total tokens ≤ 1.2× simple.

**Method.** `chain_type="auto"`: one utility-model call classifies each
question (*simple* = direct single-entity lookup; *careful* =
comparisons, multi-entity, possibly-unanswerable) and delegates.
Classifier failures route to simple. Full large-set run vs the simple
and corrective baselines, identical everything else.

| n=224 | simple | corrective | **auto** |
|---|---|---|---|
| correct_rate | 0.948 ±0.030 | 0.943 ±0.031 | 0.943 ±0.031 |
| faithful_rate | 0.967 ±0.024 | 0.986 ±0.016 | 0.976 ±0.021 |
| multi_turn_faithful | 0.923 | 0.923 | **1.000** |
| avg latency | **1.56 s** | 2.34 s | 2.41 s |
| total tokens | **260 k** | 493 k | 338 k |

**Result.** Routing does what it says: faithfulness lands between simple
and corrective (+0.9 pp over simple), multi-turn faithfulness reaches
1.0 (n=13 — one case better than either parent), and the smoke set came
back a clean sweep (correct 1.0, faithful 1.0 at n=38). But the cost
gate **fails**: 338 k tokens is **1.30×** simple — the classifier sent
roughly a quarter of questions down the 2× path, and the routing call
itself (767 ms average on the default model; a cheap
`RAGSTONE_REPHRASE_MODEL` would cut that substantially) taxes every
question, pushing latency to 1.5× simple.

**Decision.** `simple` stays the default; **auto ships opt-in**, exactly
like corrective — baselined under its own keys so users who choose it
get regression protection. The pre-registered gate failing and the
feature shipping anyway *as an option* is not a contradiction: the gate
decided the *default*, and the measurements tell users precisely what
the option buys (+1–2 pp faithfulness where it matters) and costs (+30%
tokens, +0.85 s). Future tuning that could flip the verdict — a stricter
classifier, the cheap router model by default, routing only multi-entity
questions — is parked in ROADMAP 1.5 rather than iterated blindly here.

### Postscript (15b) — the tuning avenue, tried once and closed

The two parked tweaks were measured as one pre-registered package
(attribution between them was not the question; the gate was): the
`gpt-4.1-nano` utility model plus a stricter classifier ("careful ONLY
for clear comparisons or likely-absent facts; when unsure, choose
simple"). Same gate, one run:

| n=224 | simple | auto (15a) | auto tuned (15b) |
|---|---|---|---|
| faithful_rate | 0.967 | 0.976 | **0.981 ±0.018** |
| correct_rate | 0.948 | 0.943 | 0.943 |
| total tokens | 260 k | 338 k (1.30×) | 324 k (**1.25×**) |
| avg latency | 1.56 s | 2.41 s | 2.04 s |
| route stage | — | 767 ms | 561 ms |

**Verdict: the gate fails again** — closer (1.25× vs the 1.20× bar), but
routing a meaningful share of questions down a 2× path arithmetically
cannot get much cheaper than this. The tuned variant strictly dominates
the original (better faithfulness, fewer tokens, faster), so it IS the
shipped opt-in implementation, recorded as the `chain=auto` baselines
with the recommended `RAGSTONE_REPHRASE_MODEL=gpt-4.1-nano` config.
Multi-turn faithfulness swung 1.0 → 0.846 between runs — one case at
n=13, the small-slice noise this file keeps warning about. The tuning
avenue is now closed with data: `auto` is for deployments that value
faithfulness over cost, and the numbers to make that call are above.

*Postscript (2026-07):* the nano utility model was briefly promoted to
per-provider default on a green smoke gate — and reverted the same day
when a live transcript showed it degenerating on challenge turns the
eval corpus never covered. The full story is Experiment 18.

---

## Experiment 16 — Load and scale: measuring the claims that were only argued

**Question.** Two performance claims had never been measured: the API's
thread model under concurrent load, and the retrieval stack beyond the
231-chunk eval corpus. Benchmarks: `evals/bench_concurrency.py` (real
server, real pipeline; a cached phase for server mechanics and an
uncached wave for end-to-end concurrency) and `evals/bench_scale.py`
(synthetic 1536-dim vectors — the embedding API's cost is linear and
known; the unknowns were OUR code).

**Concurrency (cap = 8).** Cache-hit requests measure the server itself:

| workers | ok | 429 | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| 1 | 12/12 | 0 | 1.7 | — |
| 8 | 96/96 | 0 | 4.8 | 7.6 |
| 16 | 184/192 | 8 | 8.0 | 11.4 |
| 32 | 310/384 | 74 | 17.1 | 21.6 |

Server overhead is single-digit milliseconds and degrades gracefully.
Under REAL generation load, p50 stayed flat (~2.2–2.6 s) from 1 to 32
workers while the semaphore refused overload instantly — hard
backpressure, no queue collapse, exactly as designed. Eight truly
concurrent generations completed in 2.0–2.7 s wall against 10–15 s
sequential: a 3.8–7.4× payoff (the spread is OpenAI latency variance).
The sync-core + worker-thread model is vindicated at its cap; the async
rewrite (ROADMAP 4.2) stays unjustified by data.

**Scale** (macOS, `OMP_NUM_THREADS=1` — see below; Linux needs no flag):

| chunks | FAISS p50 | Qdrant embedded p50 | BM25 p50 | FAISS build |
|---|---|---|---|---|
| 1k | 0.3 ms | 1.8 ms | 0.2 ms | 0.2 s |
| 10k | 1.1 ms | 17.7 ms | 2.8 ms | 1.8 s |
| 100k | 9.5 ms | 169.6 ms | **70.1 ms** | 18.5 s |

Three verdicts: (1) flat FAISS is linear and comfortably fine to 100k —
no ANN index needed at this scale; (2) embedded Qdrant is a small-corpus
convenience, not a scale path — at 100k it is 18× slower than FAISS and
qdrant-client itself warns above 20k points (server mode, one env var
away, is the scale path); (3) the sleeper: **BM25 becomes the ensemble's
bottleneck at 100k** (70 ms vs 9.5 ms vector) — rank-bm25's pure-Python
scoring is linear in corpus size, so the hybrid retriever's latency
story at scale is a lexical problem, not a vector one.

**The crash the benchmark earned.** On macOS, repeated FAISS searches on
the 100k index segfaulted within ten queries — bisected to faiss-cpu's
OpenMP parallel search (single queries fine, 10k fine, the LangChain
wrapper irrelevant). `OMP_NUM_THREADS=1` eliminates it at no practical
cost (9.5 ms p50 single-threaded), and the identical test inside the
Linux container passes with default threading — macOS-specific, likely
the documented risk of the `KMP_DUPLICATE_LIB_OK` coexistence
workaround. README documents the flag for large macOS indexes; Docker
deployments are unaffected.

**What load testing found beyond performance.** The benchmark exposed
two real API design flaws: the shared default `session_id` meant every
stateless client contributed to ONE conversation (follow-up rephrasing
could reinterpret your question against a stranger's history) — the API
is now stateless by default, generating a fresh session per request
unless a client opts in; and the response cache's session-scoped key,
made redundant by the history gate, was blocking all cross-client cache
hits — the scope is now corpus+chain only. Load tests find design bugs,
not just slow paths.

---

## Experiment 17 — Answer self-check: built, measured harmful, deleted

**Hypothesis.** Corrective RAG verifies retrieval before answering; a
post-generation check should be able to verify the ANSWER — one
utility-model call comparing its claims against the context actually
used, appending a visible caveat for unsupported ones. Target:
faithfulness, the metric with measured headroom. Pre-registered before
the run: the token gate (≤1.15×) was already unpassable by arithmetic
(the checker re-reads the context), so the question was whether
faithfulness gains ≥ +1.5 pp — enough for the opt-in to exist at all.

| n=224 | simple | with self-check |
|---|---|---|
| correct_rate | 0.948 ±0.030 | **0.910 ±0.039** |
| faithful_rate | 0.967 ±0.024 | **0.934 ±0.034** |
| total tokens | 260 k | 509 k (1.96×) |
| avg latency | 1.56 s | 2.02 s (+771 ms check) |

**Result: it made everything worse** — correctness fell 3.8 pp (outside
the CI: a real regression, not noise), faithfulness fell 3.3 pp, at
double the tokens. The failure reports show the mechanism, and it
generalizes beyond this corpus:

1. **The checker's precision is the ceiling.** A same-class model
   verifying full contexts produces false positives — it flagged claims
   that were true and present (the fare handbook's children's/seniors'
   prices) as unsupported.
2. **Caveats anchor the reader against the answer.** Once an answer
   disclaims its own (true) statements, the judge — and any human —
   follows the self-doubt: "the answer itself admits these claims are
   unsupported" → failed for correctness AND faithfulness. A caveat is
   itself a factual claim ("X is not in the documents"), and when false
   it is a hallucination appended to a correct answer.

**Decision: deleted, same day, under the kept-though-rejected policy's
rule 3** (ARCHITECTURE.md): this is not a corpus-conditional trade-off
like corrective or routing — the mechanism is defective at its core
unless the checker is substantially more accurate than the generator,
which a same-family utility model is not. Corrective RAG remains the
right way to buy faithfulness: it verifies retrieval and answers from
*better evidence*, instead of second-guessing finished text with a
weaker model. The transferable lesson: **output-side verification needs
a verifier meaningfully stronger than the generator, or it subtracts
value; input-side verification (grade-then-retry) degrades gracefully
because its failure mode is a wasted retrieval, not a poisoned answer.**
The first measured deletion — the policy binding its own author, one
experiment after it was written.

---

## Experiment 18 — The nano rephrase default: promoted on a green gate, falsified by a live transcript

**Background.** The nano utility model (`RAGSTONE_REPHRASE_MODEL=
gpt-4.1-nano`) had been measured at −40 % rephrase latency with
identical eval quality (Experiments 10/15b) and was promoted from
recommendation to per-provider default, gated by a smoke run that held
every baseline exactly. The same day, a real session against a PDF the
corpus had never seen produced this transcript:

> user: who created deepseek → "I don't know"
> user: **are you sure?** → *interpreted as:* "The context does not
> specify who created the DeepSeek-R1 and DeepSeek-V3 models."
> user: **who are the actors to this paper?** → *interpreted as:* "The
> context does not specify the authors or contributors…"

The rephraser was echoing the previous ANSWER as the "standalone
question", and the echoed refusal then became the retrieval query.

**Method.** Replayed the exact conversation history through the
rephrase chain, A/B across models, 2 trials per turn (temperature 0;
outputs were deterministic). Then repeated with a hardened prompt
(challenge turns must restate the question the answer responded to;
output must always be a single standalone question). Retrieval quality
of the candidate rephrasings was verified against the actual PDF index.

| turn | gpt-4o-mini | gpt-4.1-nano |
|---|---|---|
| "are you sure?" (old prompt) | echoes answer | echoes answer |
| "who are the actors to this paper?" (old prompt) | **correct question** | echoes answer |
| "are you sure?" (hardened prompt) | **correct question** | still degenerates |
| "who are the actors…" (hardened prompt) | **correct question** | still degenerates |
| pronoun/comparative controls (hardened) | correct | correct |

The correct rephrasing ("Who are the authors of the paper…?") retrieves
the PDF's contributor-list chunks at ranks 1–2 — the user would have
gotten their answer. The echo retrieves noise.

**Why the gate missed it.** Both golden sets contained ZERO
challenge-style follow-ups — all multi-turn cases were pronoun
substitutions ("what about its population?"), which nano handles
perfectly. The gate was green because the corpus was blind to the turn
type. Experiment 9's lesson, third appearance: the eval set defines
what you can see.

**Decision.** Three actions, in dependency order. (1) The nano default
is **reverted** — prompt hardening fixes the main model but not nano;
that is a capability floor, and a conversational turn as common as
"are you sure?" cannot be a known-broken default. The env var remains
for non-conversational traffic, where the −40 % is real. (2) The
hardened rephrase prompt **ships** — it measurably fixes the main
model on both failing turns with clean controls, and held every smoke
baseline. (3) Three challenge-turn cases join the smoke golden set
(mt06–mt08) so this blind spot stays covered; extending the large set
follows with the ROADMAP 3.0 multi-turn expansion. The transferable
lesson: **a cheap model earns a default only on eval coverage of the
turn types it will actually face — and a green gate over a blind
corpus is consent, not evidence.**

---

## Experiment 19 — Document-metadata cards: the references-section decoy

**Question.** The Experiment 18 transcript exposed a second failure:
"who created deepseek" refused (the author block never ranked), and
"who are the papers' authors" was answered with **authors of a paper
this one merely cites** — the references section is the most
author-dense text in an academic PDF, so it wins every "who
wrote/created" query, and the generator cannot tell a bibliography
entry from an author block. Can one extracted metadata chunk per
document fix the class?

**Method.** Three measurements, in registration order. (1) A new corpus
report with an author block AND a references decoy, plus three metadata
golden cases — measured on the system WITHOUT cards first (before-state:
all three pass; at 46 chunks the header ranks trivially, so the golden
set regresses the capability but cannot reproduce the decoy). (2) The
feature: one utility-model call per document over its head, extracting
`Title/Authors/Date/Type` verbatim ("not stated" for absent fields — a
hallucinated card would be authoritative false evidence), indexed as one
labeled chunk; failures skip the card. (3) The discriminating probe: the
real 79-chunk DeepSeek-R1 PDF, same three-turn session that failed live,
cards off vs on.

**Probe results (79-chunk PDF).**

| turn | cards off | cards on |
|---|---|---|
| "who created deepseek" | ten alphabet-tail contributor names presented as "the creators" | **"created by DeepSeek-AI"** |
| "who are the papers' authors" | same tail names (an earlier live run: authors of a *cited* paper) | card in the slice; tail names still blended in |
| "can you say they created deepseek?" | **"Yes"** — misattribution confirmed | **"No"** — corrects itself, cites the real core contributors from Appendix A |

The card extracted cleanly: `Authors: DeepSeek-AI`, title verbatim,
`Date: not stated`, references decoy excluded.

**Golden set (fixed corpus, before vs after):** MRR 0.936 → 0.952,
every other metric identical — the card is pure ranking upside there.
Two honest footnotes, both present in before AND after runs (so not
card-attributable): mt05's chronic judge noise (verdict contradicted
itself: "3.4 newtons instead of the correct 3.4 newtons"), and mt01,
where the new report's author block ("Dr. Vasquez… Dr. Lindqvist")
outranks the Halvorsen background chunk for person-background queries
and the judge misreads the noisier slice — the needle chunk IS retrieved
(rank 3, verified by direct probe) and the answer is faithful, but the
metric records the judge's verdict, so the baseline carries 0.875 with
this note rather than a re-rolled number.

**Decision.** Cards ship as a **default** (CI-gated by md01–md03 plus
the existing baselines; `RAGSTONE_METADATA_CARDS=off` to disable). Cost:
one utility call and one extra chunk per document at ingest. Residual,
recorded: contributor-list chunks can still outrank the card for
"authors" phrasing and get blended into answers (genuine contributors,
wrong framing), and a card is itself one new distractor-shaped chunk for
person queries. The transferable lesson: **document-level questions need
document-level evidence — content chunks answer "what does it say",
never reliably "what is this thing"; and the decoy is structural (a
references section exists in every paper), so the fix must be too.**

---

## Experiment 20 — Enterprise overhead: pricing the Arc 1 machinery

**Question.** Arc 1 put real machinery on every request: request-id and
audit middleware, named-key auth with sliding-window quotas, a
Prometheus observer, no-op OTel spans, and a registry lookup that can
lazily restore a pipeline from disk. Each piece was argued cheap at
review time; none was priced. Did the API get slower?

**Method** (`evals/bench_overhead.py`). The same load against two source
trees: HEAD and the actual pre-Arc-1 commit (`e8b3772`, via a git
worktree) — a real before/after, not an emulated baseline. The pipeline
is a stub that answers instantly, so every microsecond measured is
server mechanics; the server runs in-process and is hit over loopback
with keep-alive sessions, request/audit log lines written to a real
file. Run order A/B/A (new, old, new): the two "new" runs agreed within
a few microseconds on every phase, so the deltas are not machine drift.
No API key spent.

| phase (p50) | pre-Arc-1 | Arc 1 | delta |
|---|---|---|---|
| GET /health | 0.347 ms | 0.358 ms | **+11 µs** |
| POST /ask, sequential | 0.460 ms | 0.552 ms | **+92 µs** |
| POST /ask, 16 workers | 6.1 ms | 7.0 ms | +0.9 ms |
| stub throughput at saturation | 2,475 rps | 2,164 rps | −12.5% |

Startup to first healthy probe: ~62 ms, unchanged.

**Attribution.** Microbenchmarks account for essentially the whole
sequential delta, so nothing unexplained is hiding in the stack:

| feature | per call |
|---|---|
| worker-thread hop for the registry lookup (5.7) | 67 µs |
| request instrumentation (usage callback, no-op span, log line) | 9 µs |
| Prometheus observer on top | +4–6 µs |
| request-id + audit middleware, response header | ~11 µs |
| named-key auth, full constant-time scan | 0.2 µs (1 key) → 1.6 µs (25) |
| quota check (sliding window) | 0.25 µs |
| warm-path cost of persistence (`get_or_restore` vs `get`) | +0.02 µs |
| `persist_pipeline`, 1,000 chunks (configure-time, one-off) | 1.9 ms |
| `list_persisted` per manifest (each `/ready` probe) | 30 µs |

**Verdict.** The full enterprise stack costs **~0.1 ms per request**. A
real answer takes 1.5–10 s of LLM time, so the overhead is ≤0.007% of
the cheapest uncached answer; even a semantic-cache hit (~2 ms,
Experiment 16) pays only ~5%. The −12.5% throughput at synthetic
saturation is the same +0.1 ms amplified by queueing — and irrelevant at
the deployment envelope: with the `/ask` cap at 8 and generations taking
seconds, real traffic tops out around 4 rps, ~500× below the measured
2,164 rps mechanics floor. Auth cost is linear in key count by design
(the full scan is what keeps timing constant) and stays under 2 µs at 25
keys.

**Decision.** No optimization. The one candidate — checking the
in-memory registry synchronously before paying the 67 µs worker-thread
hop that guards lazy restore — is recorded here, not taken: it would
complicate the restore path to reclaim 0.004% of a real request. The
transferable lesson: **plumbing is priced in microseconds and answers in
seconds — measure the ratio before "optimizing" infrastructure, and
benchmark against the real old commit (a worktree costs one command),
not a hand-stripped imitation of it.**

---

## Experiment 21 — The local stack, measured: tiers, tokens/s, and a judge across the trust boundary

**Question.** Every number in this file was a cloud-model verdict; the
local path (Ollama + nomic-embed-text + local cross-encoder reranker)
"ran" but was never measured — the missing baseline for the local-first
strategy (ROADMAP 8.0). Three questions at once: what does all-local
cost in quality, what does each hardware tier buy, and can a **local
judge** be trusted to produce the scores a no-egress deployment needs?

**Method.** Four answerers — two light, two heavy, freshly pulled (July
2026) — each through the full smoke set (49 cases: 41 single-turn incl.
unanswerables, 8 multi-turn) on an M4 Max / 128 GB via Ollama 0.24:
nomic-embed-text embeddings (auto-probed), ensemble, k=4, judged by the
same cloud judge as the committed baselines (gpt-4o-mini). Thinking
**disabled** via the new `RAGSTONE_OLLAMA_REASONING` knob — probes showed
qwen3.5:9b spending 162 output tokens and 7.7 s on a one-word answer that
takes 3 tokens / 0.3 s with reasoning off; that knob is the difference
between honest local latency and measuring a model's inner monologue.
Runs were staged (per-model probe → 5-case timing pilot → full run,
strictly sequential with cooldowns) and answers were dumped
(`--dump-answers`) for the judge experiment below.

| local answerer (tier) | correct | faithful | multi-turn c/f | s/ask | out tok/s |
|---|---:|---:|---|---:|---:|
| gemma4:e4b — edge, 8B eff-4B | 0.951 | 0.976 | **0.625 / 0.75** | **1.5** | **32.6** |
| qwen3.5:9b — workstation, dense | **0.976** | 0.951 | 1.0 / 1.0 | 3.4 | 19.0 |
| qwen3.6:35b — server, MoE 36B | 0.951 | 0.951 | 1.0 / 1.0 | 2.4 | 18.8 |
| gemma4:31b — server, dense 31B | **1.000** | 0.951 | 1.0 / 1.0 | 8.0 | 3.3 |
| *cloud: gpt-4o-mini (baseline)* | 0.951 | 1.000 | 0.875 / 0.875 | ~1.4 | — |

(n=41 single / 8 multi-turn; every gap between columns is 0–2 cases —
CIs overlap throughout. tokens/s is pipeline throughput: output tokens ÷
ask wall-clock.)

**Retrieval (Layer 1).** nomic-embed-text + ensemble lands at hit
0.947–0.974 / MRR ~0.89 (the spread across runs is real: each model
generates its own metadata cards, so Layer 1 varies by ±1 case). The
**local reranker repairs all of it: hit 1.0 / MRR 1.0** — the same
"biggest ranking lever" verdict as Experiment 2, now fully offline.

**Findings.**
1. **The local stack does not lose on this corpus.** Every tier ≥ 0.951
   correct; local multi-turn beats the cloud baseline's 0.875 (noise-level
   margins, but the sign is not what "local = worse" predicts). The
   honest sales line changes from "loses only X points" to "measured at
   parity on our corpus — bring yours" (the 3.0 second corpus remains
   the test that generalizes this).
2. **The tier table is the product.** Edge (gemma4:e4b) matches the big
   models single-turn at 33 tok/s but **collapses on the conversational
   slice** (5/8 correct) — Experiment 18's lesson (gate utility steps on
   the turn types they face) resurfacing at model-tier level: don't ship
   edge models for chat. Heavy-dense (gemma4:31b) buys the only perfect
   correctness (41/41) at 8 s/ask and 3.3 tok/s — prefill through 31B
   dense weights dominates RAG asks (~2 k-token contexts). Heavy-MoE
   (qwen3.6:35b, `qwen35moe`) delivers workstation-class latency at
   server scale. The 9B is the balance point.
3. **A 31B local judge is usable across the trust boundary.** The dump
   made re-judging STORED answers possible — closing Experiment 11's
   recorded caveat that regenerating answers made judge deltas an upper
   bound. Control first: gpt-4o-mini re-judging its own stored verdicts
   flipped **1/98** (q04, fail→pass) — the judge-noise floor.
   gemma4:31b on the identical answers: **4/98 flips, 0/98 parse
   failures**, scores within 2.5 pp (correct) / 4.9 pp (faithful) of the
   cloud judge — and its q04 verdict agreed with the cloud judge's own
   second look. Residual true disagreement: ~3 verdicts, split both
   directions (stricter on answer completeness, more lenient on
   grounding). A no-egress deployment can score itself.

**Operational notes** (each a real deployment lesson): Ollama loaded
every model at its FULL declared context (262 k for gemma4:31b → 47 GB
resident, and heavier attention than any RAG prompt needs) — capping
`num_ctx` is the obvious dense-tier latency lever, unmeasured here;
query embedding shares the Ollama server with generation, so retrieval
latency rose 43 → 570 ms while a 19 GB model was loading — fine at this
scale, a real "one inference server" caveat at load; the local judge arm
took 39 min for 98 verdicts (~24 s each — long faithfulness contexts
through dense 31B), so budget local judging in minutes-per-hundred, not
seconds.

**Caveats, recorded.** Answerers ran at Ollama's default temperature
(OllamaProxy deliberately doesn't force 0 — a quality-affecting change
left for its own gated pass); ensemble weights and k were tuned on
OpenAI embeddings (Experiments 3/6) and inherited unmodified — the
retrieval dip IS the unretuned number; the judge is 31B cross-family
(vs the answerer's qwen), not the 70B-class the roadmap sketched;
single corpus, n=49, ceiling effects — tier ordering is trustworthy,
decimal places are not.

**Decision.** The local path graduates from "runs" to **measured**:
baselines committed under `ollama:` keys, `make eval-local` added,
thinking-off is the documented serving posture
(`RAGSTONE_OLLAMA_REASONING=off`). The dated `llama3` Ollama default is
now contradicted by evidence — promoting qwen3.5:9b is the follow-up
(8.2), gated on this table. The transferable lesson: **local quality is
not one number, it's a tier curve — and the conversational slice is
where cheap tiers quietly break, exactly where single-turn evals can't
see.**

---

## Defaults, decided by the numbers above

| Choice            | Default                      | Decided by   | Why                                            |
|-------------------|------------------------------|--------------|------------------------------------------------|
| Embedding model   | `text-embedding-3-small`     | Experiment 1 | full coverage, ~5× cheaper than ada-002        |
| Retrieval         | ensemble (BM25 + vector)     | —            | lexical + semantic recall                      |
| Retrieval depth   | `k=4`                        | Experiment 3 | the coverage knee — minimum k for hit_rate 1.0 |
| Reranking         | optional (`[rerank]` extra)  | Experiment 2 | biggest ranking lever (MRR +0.07), but a knob  |
| Chain type        | `simple`                     | Experiment 4 | expansion adds cost, not quality, on this data |
| Chunking          | `1000 / 200`                 | Experiment 5 | smaller chunks split facts from their subjects |
| Ensemble weights  | BM25 `0.4` / vector `0.6`    | Experiment 6 | more BM25 costs coverage on paraphrases        |
| Ingestion         | parallel batches (4 × 500)   | Experiment 7 | 3.1× faster embedding, identical vectors       |
| Chunk context     | `source` identity line       | Experiment 12 | hit +1.5pp, faithful +2.9pp, +5.7% tokens     |

Every one of these will be re-examined the moment the corpus changes — which
is the point: the harness makes "should this default change?" a measurable
question instead of an argument.

