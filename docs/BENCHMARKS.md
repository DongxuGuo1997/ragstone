# Benchmarks and measured decisions

Every default in ragstone was chosen by measurement, and several features
were rejected for default status by their own experiments and kept as
labeled opt-ins. This page collects the numbers: the verdicts at a glance,
the support-tier policy they produced, the cloud and local benchmark
tables, the pipeline-versus-agent comparison and the staffing matcher's
gate. The full hypothesis → method → decision log, including the
experiments that were wrong, is [EXPERIMENTS.md](../EXPERIMENTS.md); how
the eval harness itself is built is in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#how-we-know-it-works).

## Measured, at a glance

Questions this codebase answered with its own eval harness instead of
opinion:

| Question | Verdict |
|---|---|
| Does an agent loop beat the fixed pipeline? | Identical quality, **1.8× latency, 1.45× tokens** → the pipeline stays default |
| Does self-correcting retrieval (CRAG) pay? | Looked like +0.027 at n=43; **failed to replicate at n=224** — quality within noise, 2× cost is not → opt-in, honestly labeled |
| Was the small eval set lying to us? | Yes, once: scaling 43 → 224 cases reversed a conclusion (Exp 9) — design verdicts now require `--set large` |
| Where do follow-up seconds go? | **926 ms** in one rephrase LLM call → prompt + model fix: multi-turn quality **0.8 → 1.0** on the 5-case smoke slice, rephrase **−40%** |
| Are smaller chunks sharper? | No — hit rate **drops** 1.0 → 0.914 at 500 chars |
| Is concurrent embedding safe? | **3.1× faster** ingestion, identical vectors and retrieval metrics |
| Does chunk enrichment beat more context? | Document identity in the chunk: hit **+1.5pp**, faithfulness **+2.9pp** at +5.7% tokens — now the default; k=6's extra volume had *hurt* |
| Can a router capture self-correction's edge cheaply? | Quality held (faithfulness up to **0.981** tuned), but **1.25× tokens** still fails the pre-set gate after tuning → `auto` ships opt-in, `simple` stays default (Exp 15/15b) |
| Does the thread model survive real load? | p50 flat to **32 concurrent clients**, instant 429s beyond the cap, **~7× payoff** on parallel generation; the load test also caught two API design bugs (Exp 16) |
| Can a nano model run the rephrase step? | −40% latency and a green gate — then a live transcript showed it echoing answers on "are you sure?" turns the eval set never covered. **Reverted same day**, prompt hardened, blind spot added to the golden set (Exp 18) |
| Can RAG answer "who wrote this paper?" | Not from content chunks — the references section decoys every authorship query. One extracted **metadata card** per document: misattribution eliminated on a 79-chunk PDF, MRR +1.6 on the golden set (Exp 19) |
| Can the local stack match the cloud path? | After a 622 MB embedder swap and temperature parity: smoke correct **0.951–1.0 / faithful 1.0**, multi-turn 1.0 — at the cloud reference; on real regulations, local ≈ cloud (Exp 21/25/26) |
| Does reasoning mode fix weak retrieval? | No — identical correctness at **33× latency**; a 274→622 MB embedder swap fixed what thinking couldn't (Exp 25) |
| Can CV↔assignment matching be *measured*? | Ground truth by construction: strong-candidate recall@5 **1.0**, no-full-match honesty **1.0** — and the pilot's imperfect scores caught two real matcher bugs before any human read a transcript (Exp 27) |

How it works, stage by stage: [docs/HOW_IT_WORKS.md](HOW_IT_WORKS.md)
· Full methods and numbers: [EXPERIMENTS.md](../EXPERIMENTS.md) · Design
reasoning and trade-offs: [ARCHITECTURE.md](../ARCHITECTURE.md) · What's
next, with acceptance criteria: [ROADMAP.md](../ROADMAP.md)

Because baselines are committed with every quality change, the repo can
chart its own measured trajectory — generated from git history, no
hand-typed numbers (`python evals/quality_history.py`):

![Measured quality over the project's git history](quality_history.svg)

## Support tiers: what's guaranteed, what's optional, and when to enable it

A deliberate policy, not an accident: several features below were
**rejected for default status by their own experiments** and kept anyway
— as measured options for the corpora where the trade-off flips. Every
default is CI-gated; every opt-in is baselined and carries an explicit
*enable-when* condition; combinations off this list are best-effort.

| Option | Tier | Measured niche | Enable when |
|---|---|---|---|
| `simple` chain, k=4, ensemble, enrichment, embed cache | **default** (CI-gated) | the measured optimum on the eval corpus | — |
| document metadata cards | **default** (CI-gated) | "who wrote this?" answered from the document's own header; references-decoy misattribution eliminated (Exp 19) | disable via `RAGSTONE_METADATA_CARDS=off` if your corpus has no metadata questions |
| `corrective` chain | opt-in | faithfulness 0.986 vs 0.967; refuses with evidence instead of hallucinating (Exp 8/9/12) | a wrong answer costs more than a refusal. The hit<0.9 trigger looked validated at n=27 but shrank to noise at n=68 (Exp 23→24) — directionally supported, unproven |
| `auto` routing | opt-in | corrective's edge on flagged questions at 1.25× instead of 2× (Exp 15/15b) | you want corrective's insurance without paying it on every lookup |
| `agent` chain | opt-in | none on this corpus — identical quality at 1.8× latency | first-shot retrieval fails often enough that re-searching pays; measure it on YOUR corpus |
| `multi_query` | opt-in | none measured (Exp 4; multi-turn faithfulness DROPS on the regulatory corpus, Exp 24) | rarely — measure on your corpus first |
| `fusion` | opt-in | **best chain on the regulatory corpus**: correct +10.7pp, faithful 0.982 (Exp 24) at 2.3× tokens | your corpus has near-duplicate or tiered passages (fine schedules, versioned clauses, recitals mirroring articles) |
| `[rerank]` cross-encoder | opt-in | biggest ranking lever: MRR 0.93 → 1.0 (smoke) | ranking precision matters and ~80 MB local model + latency is acceptable |
| `RAGSTONE_CHUNK_CONTEXT=llm` | opt-in | untested beyond `source` mode | document names carry no meaning, so the free identity line can't disambiguate |
| `qdrant` / `pgvector` stores | opt-in (operational) | FAISS parity by test (Exp 13) | you need persistence, server-mode sharing, or the Postgres you already run |
| `sqlite` memory | opt-in (operational) | — | conversations must survive restarts |
| `chroma` store | **legacy** | none — the embedded-persistent niche is Qdrant-local's, which has parity tests Chroma lacks | migrating from an existing Chroma deployment only |

The verdicts above began as single-corpus results; the second, real
corpus (EU regulations, Experiments 23–24, n=68) has now put them on
trial with statistical power: enrichment and chunk-size held, fusion
FLIPPED (near-duplicate passages are its measured niche), multi_query
stayed rejected, and corrective's brief n=27 validation was walked
back at n=68 — the harness catching its own newest claim, twice proving
that small-slice verdicts don't survive scale (the Experiment 9
lesson).

## How the harness runs

The repo ships a small, readable RAG evaluation harness (`evals/`) that runs
against the real pipeline with a bundled corpus of fictional-fact documents
and a hand-written golden dataset:

- **Layer 1 — retrieval** (deterministic, no judge): hit rate and MRR for
  whether the right chunks come back.
- **Layer 2 — generation** (LLM-as-judge, costs a few cents): answer
  correctness vs. the gold answer, and faithfulness to the retrieved context
  (hallucination check). The judge prompts are in `evals/judge.py`, in the
  open.

```bash
make eval-retrieval   # Layer 1 only — fast, free
make eval             # both layers — needs OPENAI_API_KEY

# Free, fully local run (requires Ollama; scores are not comparable
# across different judge models):
python evals/run_eval.py --provider ollama --model qwen3.5:9b \
    --ollama-reasoning off --judge-provider ollama --judge-model gemma4:31b

# Measure the effect of reranking (requires the rerank extra)
python evals/run_eval.py --mode retrieval --rerank
```

Each run is compared against `evals/baseline.json` and **fails if any metric
drops more than 0.05 below baseline** — so quality regressions show up as
failed runs, not silent drift. After an intentional behavior change, accept
new scores with `python evals/run_eval.py --update-baseline` and commit the
file. A per-case report with judge reasons is written to `evals/report.md`.

## Benchmark results (cloud)

Every default in Ragstone was chosen by measurement, not intuition. The
numbers below come from the harness above on the bundled corpus (the
38-case revision of the golden set current at the time; today's set has
43 cases — see [EXPERIMENTS.md](../EXPERIMENTS.md) for the full
hypothesis → method → decision log). Single corpus — read these as
direction and magnitude, not decimal places.

**Retrieval (Layer 1, deterministic).** Reranking is the largest ranking
lever; `text-embedding-3-small` gives full coverage at ~5× lower cost than
ada-002; `k=4` is the coverage knee (k=2 loses answers, k=6 adds nothing).

| Configuration (k=4)                       | hit_rate | MRR   |
|-------------------------------------------|---------:|------:|
| ada-002, ensemble                         | 0.971    | 0.902 |
| text-embedding-3-small, ensemble          | 1.000    | 0.895 |
| text-embedding-3-small, ensemble + rerank | 1.000    | **0.964** |

(The local equivalents are in
["The local stack, measured"](#the-local-stack-measured) below.)

**Generation (Layer 2, LLM-judged).** The headline finding is a *negative*
one, and it drives the default: multi-query and fusion add ~2× the LLM calls
and 3–4× the retrievals per question, but deliver **no measurable correctness
gain** over plain RAG on this corpus (every gap below is a single case out of
38 — noise). So `simple` is the default; the others stay available for corpora
where question phrasing is genuinely ambiguous.

| Chain type   | correct_rate | faithful_rate | relative cost        |
|--------------|-------------:|--------------:|----------------------|
| **simple**   | 0.947        | 0.947         | 1× (baseline)        |
| multi_query  | 0.947        | 0.974         | ~2× calls, ~3× reads |
| fusion       | 0.974        | 0.921         | ~2× calls, ~4× reads |

## The local stack, measured

Everything above also runs fully offline — Ollama models,
`embeddinggemma` embeddings (the probed local default since
Experiment 25), optionally the local cross-encoder reranker — and as
of July 2026 that path is **measured, not just supported**
(Experiments 21/25/26; current 49-case smoke set, thinking disabled via
`RAGSTONE_OLLAMA_REASONING=off`, scored by the same cloud judge as the
cloud baseline, M4 Max / 128 GB).

**Local retrieval** (k=4): the probed default is now
`embeddinggemma` (622 MB, promoted by Experiment 25): smoke hit **1.0 /
MRR 0.939**, and on the real regulatory corpus it matches the cloud
embedder (hit 0.80 vs 0.78) where the previous default trailed badly
(0.56). Adding the local reranker on the fictional corpus lands **hit
1.0 / MRR 1.0**.

**Local generation**, by hardware tier (Experiment 21 — measured under
the previous `nomic-embed-text` embedder; the tier *shape* is the
durable signal):

| Local answerer (tier)         | correct | faithful | multi-turn c/f | s/ask | out tok/s |
|-------------------------------|--------:|---------:|----------------|------:|----------:|
| gemma4:e4b (edge)             | 0.951   | 0.976    | **0.63 / 0.75** | 1.5   | 33        |
| qwen3.5:9b (workstation)      | 0.976   | 0.951    | 1.0 / 1.0      | 3.4   | 19        |
| qwen3.6:35b (server, MoE)     | 0.951   | 0.951    | 1.0 / 1.0      | 2.4   | 19        |
| gemma4:31b (server, dense)    | 1.000   | 0.951    | 1.0 / 1.0      | 8.0   | 3.3       |
| *cloud: gpt-4o-mini*          | 0.951   | 1.000    | 0.88 / 0.88    | ~1.4  | —         |

Margins are 0–2 cases (CIs overlap); the durable signal is the tier
shape: the edge model matches the others single-turn but **breaks on
conversation**, and the heavy-dense model buys the last correctness
point at 4× the latency. A local 31B judge re-scoring the same stored
answers agreed with the cloud judge within 2.5–4.9 pp with zero format
failures — a no-egress deployment can run this harness end to end.
Run it yourself: `make eval-local`.

**The current local default, re-measured (July 2026)** — qwen3.5:9b +
`embeddinggemma` + temperature 0, thinking off: correct **0.951** /
faithful **1.0** / multi-turn **1.0 / 1.0** (the committed baseline);
the temperature-default arm of the same paired gate reached correct
**1.0 / faithful 1.0**. Either way the local stack sits at the cloud
reference (gpt-4o-mini: 0.951 / 1.0) on this set — the two arms differ
by judge strictness on extra correct detail, dissected case by case in
Experiment 26.

And "no-egress" is an enforced invariant, not a promise:
`RAGSTONE_PROFILE=local` refuses cloud providers, remote document
sources, and phone-home tracing, validates every endpoint as loopback
at boot, and is regression-tested by a socket-intercepting test over
the full ingest-and-ask path (`tests/integration/test_no_egress.py`).
Data-flow diagrams per deployment mode: [SECURITY.md](../SECURITY.md).

## Agent mode: pipeline vs. agent, measured

Ragstone's default is a fixed pipeline — retrieve once, answer once. The
`agent` chain type is the counterpoint: the LLM gets tools — the retriever
as `search_documents`, plus an exact `calculate` tool (LLMs retrieve
numbers well and multiply them badly; the calculator is a strict
arithmetic-only AST evaluator, never an `eval()`) — and drives the loop
itself, searching again with a refined query when the first results don't
answer the question (built on LangChain 1.x `create_agent`). Adding the
calculator was gated the usual way: a before/after smoke pair measured
identical quality (correct 0.951, faithful 1.0) with the tool in the menu,
and answers to comparison questions started including correctly computed
deltas ("longer by 3 years") instead of leaving arithmetic to the reader.

```python
pipeline.create_rag_chain(chain_type="agent")
```

Which is better? Don't guess — measure. The eval harness reports quality
(correctness, faithfulness) **and** efficiency (latency, tokens) per chain
type on the same corpus:

```bash
python evals/run_eval.py --chain-type simple
python evals/run_eval.py --chain-type agent
```

Compare the two runs in `evals/report.md`. On the bundled eval corpus
(gpt-4o-mini; measured on the earlier 38-question revision of the golden
set — the current set has 43):

| | simple | agent |
|---|---|---|
| correct rate | 0.947 | 0.947 |
| faithful rate | 0.974 | 0.974 |
| avg latency | 1.4 s | 2.7 s |
| total tokens | 39k | 57k |

Identical quality, 1.8× the latency, 1.45× the tokens: when first-shot
retrieval is already good, the agent's ability to re-search buys nothing —
it only pays. That is why the fixed pipeline is the default. On a corpus
where retrieval misses more often, the trade-off can flip; the harness
lets you find out for yours instead of guessing. The same comparison runs
fully offline — local-model quality and latency per hardware tier are
measured in ["The local stack, measured"](#the-local-stack-measured)
above.

## The staffing matcher, measured

A complete vertical use case built on the engine (ROADMAP 9): paste a
client assignment request, get an evidence-backed candidate shortlist
from a consultant-CV pool.

```bash
make run-match-ui                    # dedicated UI (or: ragstone-match)
python evals/run_staffing_eval.py    # the measured gate
```

How it works, stage by stage — with the rules, the worked example
and the measurements — is [docs/CV_MATCHING.md](CV_MATCHING.md).
Presenting it live? The rehearsed act-by-act runbook is
[docs/guides/STAFFING_DEMO_SCRIPT.md](guides/STAFFING_DEMO_SCRIPT.md).

- **Every claim cites the CV.** The brief is parsed into structured
  requirements (OR-alternatives preserved); candidates are discovered by
  per-requirement hybrid retrieval over person-tagged chunks, then
  verified requirement-by-requirement with verbatim CV quotes. Gaps are
  reported as *"not evidenced in the CV"* — absence of evidence, not
  evidence of absence.
- **Honesty is a gated metric.** The bench — 40 synthetic Nordic
  consultant CVs and 11 assignment briefs (one in Swedish, two in the
  long-form RFQ shape real requests arrive in) with ground truth true
  by construction — includes one deliberately unsatisfiable assignment,
  and scores the extracted requirement list directly (Experiment 30).
  Measured: strong-candidate recall@5 **1.0**, no-full-match honesty
  **1.0**, ranking cleanliness **1.0** (Experiment 27).
- **It scales, measured.** The same assignments over a 400-consultant
  pool (10× the bench): top-5 precision **0.950** with honesty and
  ranking discipline intact. The first run scored 0.925; its two
  imperfections were attributed, not hidden (one bench fix, one
  verifier lever), and closing the verifier lever with a code-level
  second vote produced the current number (Experiments 29 and 30).
  Verification runs concurrently on the cloud path: ~10 s per
  assignment.
- **Privacy is structural — and measured.** CVs are personal data under
  the GDPR; with the Ollama provider the entire match runs locally, and
  the full eval under the enforced no-egress profile
  (qwen3.5:9b + embeddinggemma) scores **identically to the cloud
  stack** on every gated metric. You can also upload real CVs
  (PDF/DOCX) straight into the UI — they never leave the machine. The
  repo ships only synthetic CVs, and the tool is framed as
  human-in-the-loop decision support, never automated selection.
