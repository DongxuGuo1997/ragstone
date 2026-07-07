# Changelog

Notable changes to Ragstone. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
