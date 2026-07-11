# Changelog

Notable changes to Ragstone. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Session TTL + right to erasure** (ROADMAP 5.10):
  `RAGSTONE_SESSION_TTL` expires idle conversation sessions lazily on
  the ask path; `DELETE /pipelines/{id}/sessions/{sid}` (and an MCP
  `delete_session` tool) erases a session's history on request,
  idempotently and audited. SECURITY.md now answers "where does user
  text live and when does it die" per store. Also hardened the eval
  judge's verdict parser: a clear pass/fail is rescued from invalid
  JSON (unescaped quotes in the judge's reason) instead of
  fail-closing the answer for the judge's formatting — the bug that
  cost two verdicts in Experiment 23.

- **Second evaluation corpus, v1** (Experiment 23 / ROADMAP 3.0): the
  GDPR + EU AI Act from EUR-Lex as `--set regulatory` (1,331 chunks, 27
  hand-written cases with grep-verified needles). First cross-corpus
  verdict trial: enrichment held; **corrective's enable-when condition
  validated** the first time its trigger actually occurred (hit <0.9 →
  faithfulness +8.7pp, multi-turn +50pp at 2.2× tokens); rerank's value
  split by embedder; Experiment 21's local-parity claim found its
  boundary (nomic trails on legal jargon, local generation holds, the
  local reranker closes most of the gap). New measured failure class:
  fine-tier confusion from near-duplicate numeric schedules.

- **Provable no-egress mode** (ROADMAP 8.1): `RAGSTONE_PROFILE=local`
  refuses cloud providers, the OpenAI embedding fallback, remote
  document sources, and phone-home tracing; restricts the reranker to
  its local model cache; and validates every configured endpoint as
  loopback at boot, fail closed. The invariant is regression-tested by
  a socket-intercepting test over the full ingest-and-ask path
  (CI-safe fake-model tier + live Ollama tier). Data-flow diagrams per
  deployment mode in SECURITY.md.

- **The local stack, measured** (Experiment 21 / ROADMAP 8.0): a
  four-model Ollama answerer matrix (edge → workstation → server tiers)
  through the full eval harness — parity with the cloud baseline on the
  bundled corpus, local rerank retrieval at 1.0/1.0, and a local-judge
  delta scored on identical stored answers (4/98 flips, 0 parse
  failures). New: `RAGSTONE_OLLAMA_REASONING` thinking control,
  `run_eval.py --ollama-reasoning/--judge-reasoning/--dump-answers/--limit`,
  `evals/rejudge.py`, `make eval-local`, committed `ollama:` baselines.

- **Evidence highlighting** (citations v1): source snippets in the
  Streamlit trace panel and the CLI's `/sources` now highlight the exact
  words the answer reuses — post-hoc answer-to-source alignment with
  exact character offsets, no prompt or generation change.

- **Load and scale benchmarks** (Experiment 16): `bench_concurrency.py`
  and `bench_scale.py` measure the previously argued claims — server
  overhead 1.7ms p50 on cache hits, flat p50 to 32 concurrent clients
  with clean 429 backpressure, ~7x parallel-generation payoff; FAISS
  ~10ms at 100k chunks, BM25 the ensemble bottleneck at scale, embedded
  Qdrant for small corpora only. On macOS, large FAISS indexes need
  OMP_NUM_THREADS=1 (libomp instability, bisected and documented;
  Linux/Docker unaffected).

### Fixed
- **Single-document retrieval** (Experiment 22, found live): two stacked
  bugs made document-level queries ("what is this paper?") on a lone
  uploaded document retrieve contributor name-lists and the TOC instead
  of content. The chunk-enrichment identity prefix is now skipped when a
  corpus has one source document (nothing to disambiguate — the prefix
  dominated content-empty chunks' embeddings and zeroed the document
  name's BM25 IDF), and nomic-embed-text now gets the
  `search_query:`/`search_document:` task prefixes its model card
  requires (`NomicTaskEmbeddings`, own cache namespace). single_doc MRR
  0.594→0.750 (OpenAI) / 0.656→0.719 (nomic); smoke and rerank
  baselines held exactly on both stacks.

### Removed
- **Overengineering audit pass**: the parallel `config.json` loading
  system (env vars are the one config surface; `load_config`/`from_file`/
  `to_dict`/`to_file`/`validate` and `config.example.json` deleted), the
  unused stages of Ollama embedding selection (per-LLM preference tables
  and the embed-with-the-chat-LLM last resort — a missing dedicated
  embedder is now a clear error instead of silently degraded retrieval),
  two duplicate "list installed Ollama models" implementations (one
  shared helper in `utils/ollama.py` now), the `mcp/connection_test.py`
  debug script, and unused `UIConfig` theme fields.
- **Answer self-check** (added and deleted within this cycle): Experiment
  17 measured it harming both correctness (-3.8pp, outside the CI) and
  faithfulness (-3.3pp) at 2x tokens - an imperfect checker's caveats
  poison correct answers by disclaiming true claims. The first measured
  deletion under the kept-though-rejected policy; the lesson lives in
  EXPERIMENTS.md.

### Changed
- **The REST API is stateless by default**: `session_id` now defaults to
  a fresh per-request session instead of a shared "api_session" — the
  shared default accumulated one conversation across ALL clients, letting
  follow-up rephrasing reinterpret a question against a stranger's
  history. Pass a session_id explicitly to opt into conversation memory;
  the response is unchanged in shape and returns the session used.
- **Response-cache scope is corpus+chain (session removed)**: only
  history-free turns ever touch the cache, so identical questions on the
  same corpus and chain now share entries across clients — the
  session-scoped key was blocking every cross-client hit.
- **Router tuned and its default-status question closed** (Experiment
  15b): stricter classifier + the cheap utility model reaches
  faithfulness 0.981 at 1.25x tokens — still over the pre-registered
  1.2x gate, so `chain_type="auto"` remains opt-in (recommended with
  `RAGSTONE_REPHRASE_MODEL=gpt-4.1-nano`); the tuned variant replaces
  the original as it dominates on every axis.

## [2.1.0] — 2026-07-07

### Added
- **Server-backed vector stores** (ROADMAP 4.1): `VECTOR_STORE_TYPE=qdrant`
  (embedded local mode by default, `QDRANT_URL` for a server — same code
  path) and `VECTOR_STORE_TYPE=pgvector` (`RAGSTONE_PG_URL`), behind the
  existing `VectorStoreProxy` ABC as optional extras `[qdrant]` /
  `[pgvector]`. Retrieval parity with FAISS is enforced by test and
  measured on the full eval set (Experiment 13: identical hit_rate/MRR).
- **Docker deployment option** (reinstated): `Dockerfile` for the API
  plus `docker compose up` for API + Qdrant server + Postgres/pgvector
  with healthchecks and persistent volumes. Local development remains
  venv-based; no secrets are baked into images.
- **Contextual chunk enrichment** (`RAGSTONE_CHUNK_CONTEXT`, Experiment
  12): document identity prepended to every chunk before indexing.
  Measured at n=224: hit_rate +1.5pp, MRR +2.0, faithfulness +2.9pp for
  +5.7% tokens — now the default (`source`); `llm` mode opt-in.
- **Rich terminal chat** (`ragstone-chat`): streamed answers with live
  progress events, glass-box trace (latency, tokens, cost, stage
  breakdown), and /sources /trace /chain /compare /cache /new commands.
- **Eval harness statistics**: rate metrics carry 95% binomial CIs and
  sample sizes; the baseline gate says whether a drop is outside the CI.
  Judge self-preference bounded by a cross-model judge run (Experiment
  11: −2.4pp correctness under gpt-4.1-mini; no conclusion flips).
- **Incremental ingestion** (`RAGSTONE_EMBED_CACHE`, on by default): a
  content-addressed embedding cache keyed by model + exact chunk text —
  re-ingesting a corpus embeds only new or edited chunks, across all
  vector-store backends, with bit-identical vectors (float64
  round-trip), so retrieval results cannot change.
- **Query routing** (`chain_type="auto"`, opt-in): a cheap utility-model
  classifier sends direct lookups to the simple chain and
  confusion-prone questions to the corrective chain; fail-safe to
  simple; the decision streams as a route event. Measured in Experiment
  15: faithfulness improves (+0.9pp, multi-turn 1.0) but 1.3x tokens
  failed the pre-registered default gate — simple stays the default.
- **Quality-history chart** (`evals/quality_history.py`): the project's
  measured quality trajectory, generated from the git history of its
  committed baselines — table plus dependency-free SVG, embedded in the
  README.
- **Agent mode** (`chain_type="agent"`): tool-calling retrieval loop via
  LangGraph `create_agent`, measured head-to-head against the fixed
  pipeline (Experiments, README).
- **Corrective RAG** (`chain_type="corrective"`): retrieve → grade →
  rewrite cycle with bounded retries and evidence-based refusal that
  spends zero answer tokens.
- **Large golden-set tier** (`evals/golden_large.jsonl`, 224 cases over a
  16-document corpus with engineered distractors) behind `--set large`;
  the generator validates every retrieval needle verbatim and rejects
  hallucinated cases. Smoke tier (43 cases) remains the CI gate.
- **REST API** (`ragstone-api`): pipeline lifecycle over HTTP with SSE
  streaming, API-key auth (constant-time compare), a non-blocking
  concurrency cap (429 at capacity), and liveness/readiness probes.
- **Observability**: one structured log line per request
  (`ragstone.requests`), per-stage timing (rephrase/retrieval/generation),
  token and first-token accounting via contextvars, surfaced in the UI as
  a glass-box trace with cost estimates.
- **Security guards**: `RAGSTONE_DATA_ROOT` ingestion containment, SSRF
  validation of `page_urls` (scheme + resolved-address class checks,
  redirects not followed), question length caps before any API spend.
  See SECURITY.md.
- **Multi-turn evaluation**: scripted conversations in both golden tiers
  with separately gated `multi_turn_*` metrics; faithfulness is judged
  against the documents the pipeline actually used.
- **Parallel embedding ingestion**: order-preserving, fail-loud batched
  embedding (measured 3.1× faster) with `RAGSTONE_EMBED_BATCH_SIZE` /
  `RAGSTONE_EMBED_WORKERS`.
- **Conversation memory backends**: in-process (default) or persistent
  SQLite via `RAGSTONE_CHECKPOINT_BACKEND=sqlite`; sessions capped at 40
  messages with reducer-based trimming.
- **Cheap utility model** for rephrase/grade/rewrite steps via
  `RAGSTONE_REPHRASE_MODEL`.

### Changed
- Response cache keys now encode the corpus fingerprint and chain type,
  and caching skips conversational follow-ups entirely — a cached answer
  can no longer leak across document sets or bypass the rephrase step.
- Streaming over the REST API runs the pipeline on a dedicated worker
  thread (correct token metrics under SSE) and frames multi-line chunks
  per the SSE spec.
- Chroma collections are unique per pipeline instance; rebuilding
  replaces a collection instead of appending to it.
- `similarity_k` stays 4: k=6 improved the retrieval-slice metric but
  degraded end-to-end answer quality at n=224 (Experiment 10).
- Exception hierarchy pruned to the classes actually raised; the
  `Pipeline.LLM` attribute is now `Pipeline.llm_proxy`.

### Fixed
- Two "unanswerable" eval cases whose answers were in the corpus (they
  penalized correct answers and rewarded false refusals).
- Eval sessions are namespaced per run, so persistent checkpoint backends
  cannot feed stale history into a later run's multi-turn metrics.
- SQLite conversation-memory connections are closed when chains are
  rebuilt or pipelines deleted.

## [2.0.0] — 2026-06

- Renamed to **ragstone**; packaging moved to pyproject-only.
- Migrated to LangChain 1.x / LangGraph: conversation memory is a
  2-node StateGraph (rephrase → answer) with per-session checkpointing,
  replacing the deprecated `RunnableWithMessageHistory`.
- Embeddings upgraded `text-embedding-ada-002` → `text-embedding-3-small`
  (measured: full retrieval coverage at ~5× lower cost).
- Two-layer evaluation harness (deterministic retrieval slice + LLM
  judge) with committed baselines gating CI.
- Hybrid retrieval (BM25 + vector ensemble) with optional cross-encoder
  reranking (`[rerank]` extra).

## [1.0.0]

- Initial RAG pipeline: local/web/Wikipedia loaders, FAISS/Chroma vector
  stores, OpenAI and Ollama backends, Streamlit chat UI, MCP server for
  editor/agent integration.
