# Ragstone Architecture & Design Decisions

This document explains *why* Ragstone is built the way it is. The code
shows what; this shows the reasoning — including the alternatives that
were measured and rejected, and the ones deliberately not built.

## System overview

One core pipeline, four entry points:

```
        ┌─────────────┐  ┌───────────┐  ┌────────────┐  ┌─────────────┐
        │ Streamlit UI │  │ CLI chat  │  │ MCP server │  │  REST API   │
        │  (humans)    │  │  (dev)    │  │  (agents)  │  │   (apps)    │
        └──────┬───────┘  └─────┬─────┘  └─────┬──────┘  └──────┬──────┘
               │                │              │                │
               │                │        ┌─────┴────────────────┴─────┐
               │                │        │  shared pipeline registry  │
               │                │        │  (thread-safe, locked)     │
               │                │        └─────────────┬──────────────┘
               ▼                ▼                      ▼
        ┌────────────────────────────────────────────────────────────┐
        │                        Pipeline                            │
        │                                                            │
        │  load ──► split ──► embed (parallel) ──► index             │
        │                                    FAISS/Chroma + BM25     │
        │                                                            │
        │  ask ──► memory graph (LangGraph) ──► RAG chain ──► answer │
        │            │ rephrase │ answer │        simple /           │
        │            └── checkpointed ───┘        multi_query /      │
        │                per session              fusion / agent     │
        └────────────────────────────────────────────────────────────┘
                     │                          │
              ┌──────┴──────┐            ┌──────┴──────┐
              │ observability│            │ eval harness│
              │ per-request  │            │ 2 layers,   │
              │ logs+stages  │            │ CI-gated    │
              └─────────────┘            └─────────────┘
```

The design bet: **RAG is the primitive agents stand on, not the thing
agents replaced.** So the default path is a fixed, deterministic pipeline
(one retrieval, one answer call), and everything more sophisticated —
query expansion, agent loops — is an opt-in *whose cost is measured
against that baseline*.

## The ask path, end to end

`pipeline.ask_question(question, session_id)`:

1. **Guards** (`validate_question`): type, non-empty, length cap —
   rejected *before* the cache lookup or any API spend.
2. **Response cache** (opt-in): exact-match LRU keyed by
   (question, context). Deliberately not semantic — a semantic cache was
   built, measured, and removed: similarity thresholds either leaked
   wrong answers or never hit.
3. **Memory graph** (LangGraph): see below.
4. **RAG chain**: retrieve → prompt → generate, streaming end to end.

Every request emits one structured log line (`ragstone.requests`) with a
correlation id, latency, token usage, cache flag, and a per-stage
decomposition — on success, cache hit, *and* failure.

## Conversation memory: a 2-node LangGraph

```
             ┌──────────── has history? ────────────┐
   question ─┤ no                                yes├─► rephrase ─┐
             │                                      │  (timed,    │
             ▼                                      │   cheap LLM)│
           answer ◄─────────────────────────────────┴─────────────┘
        (streams via get_stream_writer, checkpointed per thread_id)
```

Decisions worth defending:

- **Why a graph at all?** LangChain 1.x deprecated
  `RunnableWithMessageHistory`; LangGraph checkpointing is the supported
  persistence model. The graph also gives us conditional routing (skip
  the rephrase LLM call entirely on first turns) for free.
- **Why `stream_mode="custom"` and not `"messages"`?** Measured during
  the migration: multi-query chains make several LLM calls per answer
  node, and `"messages"` mode leaks the query-generation tokens into the
  answer stream. A `get_stream_writer()` in the answer node forwards
  exactly the final answer tokens — and is a no-op under `.invoke()`, so
  one node implementation serves both paths.
- **History stores the ORIGINAL question, not the rephrased one** —
  the user's words are the conversation; the rephrase is an internal
  retrieval artifact (exposed to the UI as "interpreted as…").
- **The rephrase step is on the critical path** (it runs before retrieval
  can start). Instrumentation showed it costing ~926 ms per follow-up —
  the single largest fixable latency in the system. Two measured fixes:
  an entity-substitution prompt (took multi-turn correctness 0.8 → 1.0)
  and an optional cheaper model (`RAGSTONE_REPHRASE_MODEL`, −40 % rephrase
  time) — briefly promoted to default, then reverted when a live
  transcript showed nano-tier models degenerating on challenge turns
  (Experiment 18).
- **State is bounded**: sessions trim to 40 messages via `RemoveMessage`;
  the rephrase window is capped at 10. Durable checkpointing (SQLite) is
  an extra, not a default.

## Agent mode: the measured counterpoint

`chain_type="agent"` hands the retriever to the LLM as a tool
(LangChain 1.x `create_agent`) and lets it drive the loop: search, read,
refine, answer. It conforms to the same Runnable contract as the fixed
chains, so memory, caching, UI, MCP, and the API all work unchanged; its
searches surface live as typed stream events (`{"event": "search", ...}`)
that flow through the memory graph to the UI and out of the REST API as
named SSE events.

The result on the bundled corpus: **identical quality, 1.8× the latency,
1.45× the tokens.** That number — not a philosophy — is why the fixed
pipeline is the default. On a corpus where first-shot retrieval misses
more often, the trade can flip; the harness exists so you find out on
*your* corpus instead of guessing.

## Evaluation: the actual quality system

Two layers, both gated against a committed baseline (CI fails on a drop
> 0.05):

- **Layer 1 — retrieval** (deterministic, free): hit rate and MRR over a
  golden set with per-case `must_contain` needles; distractor and
  multi-hop categories included.
- **Layer 2 — generation** (LLM judge, cents): correctness against gold
  answers and faithfulness against the retrieved context, plus per-case
  latency/token/stage measurements.

Two design points that matter:

- **Multi-turn cases are scripted conversations** run in one session; the
  final turn is judged against the documents the pipeline *actually*
  retrieved post-rephrase. This is the only place the rephrase step is
  exercised — and its first run caught a real cross-product retrieval bug
  the same day it was written. Multi-turn metrics are named separately
  (`multi_turn_correct_rate`) so historical single-turn baselines stay
  comparable forever.
- **Every default is an experiment** (EXPERIMENTS.md): embedding model,
  k, reranking, chain type, chunk size, ensemble weights, parallel
  ingestion. Several hypotheses lost — smaller chunks *hurt* retrieval,
  query expansion doesn't pay on this corpus — and the defaults record
  that.

## Observability: contextvars all the way down

`track_request()` wraps every ask and owns three mechanisms, all
contextvar-based (which LangGraph propagates into node execution — the
same property the token-usage callback depends on):

- **Token accounting** across every LLM call in the request (rephrase,
  agent searches, answer) via `get_usage_metadata_callback`.
- **Stage decomposition** via `record_stage()`: the rephrase node and the
  retriever wrapper time themselves; generation is *derived* (latency
  minus named stages) so stages can never double-count.
- **One log line per request**, emitted even on failure — the failure
  case is when observability matters most.

## Serving: the boring parts done properly

- **Concurrency model**: the core is synchronous; servers offload to
  worker threads (`anyio.to_thread`). The workload is network-bound, so
  threads give full overlap (GIL released during HTTP) without an async
  rewrite. Corpus embedding — the big-input cost — runs in concurrent
  batches (measured 3.1× faster, identical vectors, order preservation
  tested adversarially).
- **Shared registry**: one locked pipeline registry serves both MCP and
  REST; deletes pop atomically so in-flight requests keep a stable
  reference.
- **Backpressure**: the API caps concurrent `/ask` requests with a
  non-blocking semaphore — 429 immediately instead of an unbounded queue.
- **Error taxonomy**: typed `PipelineError`s carry user-safe messages and
  map to precise HTTP codes; anything untyped becomes a generic 500 with
  details only in the server log. The MCP server applies the same policy.
- **Resilience**: retries and timeouts are wired into every LLM and
  embedding client from config; input length is capped pre-spend.

## Things deliberately not built (or built only when the conditions changed)

| Decision | Why |
|---|---|
| Async core — not built | Thread offload covers a network-bound workload; Experiment 16 measured p50 flat to 32 concurrent clients, so the rewrite stays unjustified by data |
| Semantic response cache — built, measured, removed | Thresholds either leak wrong answers or never fire; exact-match only, twice affirmed |
| Speculative retrieval overlap — not built | Instrumentation showed retrieval is ~150 ms; the complexity would chase the wrong 15 % |
| Docker — removed, then reinstated as optional | Removed while it was deploy-manifest baggage; brought back when the pgvector/Qdrant server stores gave it a real job. Development and tests never require it |
| pgvector/Qdrant — deferred ("Phase D"), then built | Deferred while FAISS covered the need; built with a parity gate (Experiment 13) when server-backed storage was commissioned |

The pattern in every row: **the eval harness and instrumentation get to
veto engineering enthusiasm** — and un-veto it when measured conditions
change. That discipline, not any individual feature, is the architecture.

## Kept-though-rejected: the opt-in policy

Several features were rejected for *default* status by their own
pre-registered gates (corrective RAG, query routing) and kept anyway —
deliberately. The verdicts are corpus-conditional: "self-correction
isn't worth 2× here" is a statement about a corpus whose retrieval
already hits 0.97, not a law. This repo's policy:

1. **Defaults are CI-gated.** What ships on by default earned it on the
   numbers, and a regression fails the build.
2. **Opt-ins must own a niche and a revisit condition.** Each one is
   baselined, honestly labeled with what it buys and costs, and carries
   an explicit *enable-when* (see the README's support-tier table). An
   option without a documented niche doesn't get to stay.
3. **Niche duplicates get consolidated.** Two features occupying one
   measured niche is redundancy, not learning — the weaker one's lesson
   moves to EXPERIMENTS.md and the code goes.

For a learning-focused project the living code is the artifact — a
cyclic corrective graph you can step through teaches more than a
paragraph saying one existed. The policy keeps that value while capping
its cost: nothing stays without a condition that says when it would
win.
