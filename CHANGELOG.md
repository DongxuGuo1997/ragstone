# Changelog

Notable changes to Ragstone. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Every quality-affecting entry below carries the experiment that
measured it; the numbers live in [EXPERIMENTS.md](EXPERIMENTS.md).

### Added
- **CV↔assignment matcher** (`ragstone.match`, `ragstone-match` UI): a
  LangGraph graph that extracts structured requirements from a brief,
  discovers candidates by per-requirement retrieval over person-tagged
  chunks, verifies each with verbatim CV quotes and ranks by verified
  coverage with an honest `full_match_exists`. Gated at 1.0 on
  strong-candidate recall, no-full-match honesty and ordering
  (Experiment 27). Docs: `docs/CV_MATCHING.md`.
- **Staffing bench**: 40 synthetic consultant CVs and 11 briefs with
  ground truth true by construction, one brief unsatisfiable, one in
  Swedish, two in long-form RFQ shape; a 400-consultant scale variant
  (`--scale 10`). Extraction is scored directly
  (`extract_must_recall` / `extract_must_precision`, gated at 0.984).
  Experiments 27, 29, 30.
- **Matcher hardening from a real request** (Experiment 30): per-line
  extraction that copies each requirement sentence verbatim before
  classifying it; `education` and `location` requirement kinds; years
  of experience computed from the CV's date ranges instead of quoted;
  a code-level second vote that keeps a credited skill only if the CV
  names it (top-5 precision at scale 0.925 → 0.950); preferred items
  ticked per candidate from the CV's own words. `--second-vote off`
  keeps the previous path.
- **Staffing match fully local**: the eval under the no-egress profile
  (qwen3.5:9b + embeddinggemma) scores identically to the cloud stack
  on every gated metric. Real-CV upload (PDF/DOCX/MD/TXT) in the UI
  stays on the machine.
- **Second evaluation corpus**: GDPR and the EU AI Act as
  `--set regulatory` (68 cases). One verdict flipped (fusion is the
  best chain on near-duplicate passages, correct +10.7pp), one was
  walked back (corrective's trigger shrank to noise at n=68), three
  held. Experiments 23 and 24.
- **The local stack, measured**: four Ollama answerers through the
  full harness at parity with the cloud baseline (Experiment 21);
  embeddinggemma promoted to the local embedding default after it
  matched the cloud embedder on legal text (Experiment 25); a local
  31B judge re-scored the cloud's stored answers at 88% agreement
  (Experiment 28). `RAGSTONE_OLLAMA_REASONING`,
  `RAGSTONE_OLLAMA_EMBED_MODEL`, `make eval-local`, `evals/rejudge.py`.
- **Provable no-egress mode** (`RAGSTONE_PROFILE=local`): cloud
  providers, remote sources and tracing refused, every endpoint
  validated as loopback at boot, regression-tested by a
  socket-intercepting test. Data-flow modes in `SECURITY.md`.
- **Operability**: request ids, OpenTelemetry spans and a Prometheus
  `/metrics` endpoint; named API keys with per-key rate limits, an
  audit log and `GET /usage`; registry persistence so pipelines
  survive restarts without re-ingest; graceful shutdown; boot-time
  config checks that list every problem with its fix; pip-audit with
  an expiring allowlist and a weekly Trivy image scan.
- **Governability**: the REST surface frozen as `/v1` with RFC 7807
  error bodies (unprefixed paths stay as deprecated aliases);
  `RAGSTONE_SESSION_TTL` and `DELETE /pipelines/{id}/sessions/{sid}`
  for erasure; `SECURITY.md` answers where user text lives and when
  it dies.
- **Evidence highlighting**: source snippets mark the exact words the
  answer reuses, computed after generation with character offsets.
- **Document metadata cards**: one extracted title/authors/date chunk
  per document, so "who wrote this?" retrieves the author block
  instead of the references section (Experiment 19; default on).
- **Paired significance test** (`evals/compare_runs.py`): exact
  McNemar between two answer dumps over the same golden set.
- **Load and scale benchmarks** (Experiment 16): flat p50 to 32
  concurrent clients with clean 429s; FAISS ~10 ms at 100k chunks;
  the macOS `OMP_NUM_THREADS=1` requirement for large indexes.
- **Docs for readers, not only developers**: `docs/HOW_IT_WORKS.md`,
  `docs/CV_MATCHING.md`, `docs/BENCHMARKS.md`, `docs/DEPLOYMENT.md`,
  two demo runbooks, issue forms, a PR template and a code of conduct.

### Changed
- **README slimmed for the open-source front door**: benchmarks and the
  support-tier policy moved to `docs/BENCHMARKS.md`, operations to
  `docs/DEPLOYMENT.md`; the README opens with where the project came
  from. The changelog was compressed to this form, the roadmap trimmed
  to open items, and three stale guides removed or folded.
- **Eval integrity**: baselines carry a `data_sha` fingerprint of their
  golden set and corpus, so editing measured data fails loudly; the
  judge is pinned to a dated snapshot.
- **Entry points are tested**: all five console scripts and both
  Streamlit UIs boot in the suite.
- **Router tuned, still opt-in** (Experiment 15b): faithfulness 0.981
  at 1.25× tokens, over the pre-registered 1.2× gate.
- **The REST API is stateless by default**: `session_id` is per
  request unless a client passes one; the response cache is scoped by
  corpus and chain, never by session.
- **Local verify runs one candidate at a time on Ollama**, with
  `RAGSTONE_MATCH_VERIFY_WORKERS` for servers configured with more
  slots (see Fixed).

### Fixed
- **Local staffing match timed out mid-verify** (2026-09-04): eight
  concurrent screens against a one-slot Ollama waited past the read
  timeout. Serialized on Ollama; the local a01 match now completes in
  84 s.
- **Single-document retrieval** (Experiment 22): the enrichment prefix
  is skipped for one-document corpora and nomic-embed-text gets its
  required task prefixes. single_doc MRR 0.594 → 0.750.
- **Judge verdict parser** rescues a clear pass/fail from invalid JSON
  instead of failing the answer for the judge's formatting.

### Removed
- **Answer self-check** (Experiment 17): measured harmful to both
  correctness and faithfulness at 2× tokens, deleted the same cycle.
- **Overengineering pass**: the parallel `config.json` system, unused
  Ollama embedding fallbacks, duplicate model-listing helpers, a debug
  script and unused UI config fields.
- **`.env.example` keys nothing read** (`LOG_LEVEL`, `LOG_FORMAT`,
  `APP_NAME`, `APP_VERSION`, `DEBUG`, `OLLAMA_MODEL`).

### Security
- chromadb server-side CVEs (PYSEC-2026-311, CVE-2026-45830/-45831/
  -45833) allowlisted until 2026-11-01 with the exposure analysis on
  record: ragstone never runs the chroma HTTP server. No fixed release
  exists; the plan at expiry is to retire the legacy chroma store.
- pre-commit gained `detect-private-key`.

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
