# Deploying ragstone

The [README](../README.md) covers the quick start. This page is the
operator's reference for everything optional: the extras, reranking,
durable memory, the REST API, the Docker stack, the vector-store
backends and observability. The threat model, the deployment checklist
and the data-flow diagrams per deployment mode are in
[SECURITY.md](../SECURITY.md).

## Optional extras

| Extra | Adds | Enable when |
|---|---|---|
| `rerank` | a local cross-encoder second retrieval stage (sentence-transformers, pulls PyTorch) | ranking precision matters and an ~80 MB model plus its latency is acceptable |
| `sqlite` | conversation memory that survives restarts | conversations must outlive the process |
| `api` | the REST server (FastAPI, uvicorn, Prometheus client) | applications need HTTP access |
| `qdrant` | the Qdrant vector store, embedded or server | persistence without a server, or scale by pointing at one |
| `pgvector` | the pgvector store | RAG on the Postgres you already run |
| `otel` | OpenTelemetry SDK and OTLP/HTTP exporter | you run a trace collector |
| `all` | every runtime extra above | — |

```bash
pip install -e ".[api,qdrant]"      # pick what you need
pip install -e ".[all]"             # everything
```

## Reranking

Two-stage retrieval: the ensemble retriever fetches a wide candidate pool,
then a local cross-encoder rescores each (question, chunk) pair and keeps
only the best matches. This noticeably improves precision when documents
contain similar, easily-confused facts — on the bundled eval set it lifts
retrieval MRR from 0.90 to 0.96 and fixes the hardest distractor case.

```bash
pip install -e ".[rerank]"   # pulls sentence-transformers (~PyTorch)
```

```python
pipeline.set_retriever_openai(use_ensemble=True, use_reranker=True)
```

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB,
downloaded on first use) runs fully locally — it works in offline/Ollama
mode too. In the Streamlit UI, enable it under Advanced Settings.

## Durable conversation memory

By default, conversation history lives in memory and is lost when the
process exits. To persist it across restarts, install the `sqlite` extra
and switch the checkpoint backend:

```bash
pip install -e ".[sqlite]"   # pulls langgraph-checkpoint-sqlite
export RAGSTONE_CHECKPOINT_BACKEND=sqlite
export RAGSTONE_CHECKPOINT_DB=store/checkpoints.sqlite  # optional, this is the default
```

Each conversation is checkpointed per `session_id`, so after a restart a
session picks up exactly where it left off — follow-up questions still
resolve references against the earlier turns.

## REST API

Serve the pipeline over HTTP for applications (the MCP server covers
agents; this covers everything else):

```bash
pip install -e ".[api]"
ragstone-api          # binds 127.0.0.1:8000
```

```bash
curl -X POST localhost:8000/pipelines -H 'content-type: application/json' \
  -d '{"pipeline_id": "docs", "model": "gpt-4o-mini"}'
curl -X POST localhost:8000/pipelines/docs/documents -d '{"data_dir": "data"}' \
  -H 'content-type: application/json'
curl -X POST localhost:8000/pipelines/docs/retriever -d '{"chain_type": "simple"}' \
  -H 'content-type: application/json'
curl -X POST localhost:8000/pipelines/docs/ask -H 'content-type: application/json' \
  -d '{"question": "What is the main architecture?"}'
```

Production behaviors built in:

- **Streaming**: pass `"stream": true` to `/ask` for Server-Sent Events.
- **Restart survival**: configuring a retriever persists the pipeline
  (manifest + enriched chunks, `RAGSTONE_REGISTRY_DIR`); after a restart
  the first request for its id restores it lazily — no re-ingest, no LLM
  calls, embeddings from the cache. `DELETE` removes the persisted state
  too, and `RAGSTONE_REGISTRY_PERSIST=off` keeps everything in memory.
- **Graceful shutdown**: on SIGTERM, in-flight requests drain to
  completion before pipelines close (tested with kill -TERM under load).
- **Request correlation**: every response carries an `X-Request-ID`
  (yours, if you send a well-formed one), and the per-request server log
  line uses the same id — one string traces a request end to end.
- **Probes**: `GET /health` (liveness) and `GET /ready` (a pipeline with a
  RAG chain exists) for orchestrators; both stay unauthenticated.
- **Metrics**: `GET /metrics` exposes Prometheus series — request counts
  by chain/cache/error, latency histograms (end-to-end, per stage, and
  time-to-first-token), and token totals. Aggregates only, never content;
  unauthenticated like the probes, so expose it on internal networks.
- **Tracing**: `pip install "ragstone[otel]"` and set
  `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. a local Jaeger); each ask becomes a
  `ragstone.ask` span — request id, session, cache/token/error outcome —
  with children for the measured rephrase/retrieval stages.
- **Auth**: named keys with per-key rate limits —
  `RAGSTONE_API_KEYS=alice:key:60,batch:key` (requests/minute optional;
  legacy single `RAGSTONE_API_KEY` still works). Clients send theirs as
  `X-API-Key`; over-limit requests get `429` + `Retry-After`.
- **Audit + usage**: every gated request leaves an append-only
  `ragstone.audit` line (key name, method, path, status, request id —
  never content), and `GET /usage` reports per-key request counts.
- **Session lifecycle**: `DELETE /pipelines/{id}/sessions/{sid}` erases
  a conversation's history on request (right to erasure), and
  `RAGSTONE_SESSION_TTL` expires idle sessions — the full "where does
  user text live" retention table is in [SECURITY.md](../SECURITY.md).
- **Backpressure**: at most `RAGSTONE_API_MAX_CONCURRENCY` (default 8)
  simultaneous `/ask` requests; beyond that the server answers `429`
  immediately instead of queueing until it collapses.
- **Safe errors**: typed pipeline errors map to precise status codes with
  user-safe messages; anything unexpected is a generic `500` with details
  only in the server log.
- **Versioned contract**: `/v1/...` is the frozen surface (RFC 7807
  `application/problem+json` error bodies carrying the request id);
  unprefixed paths keep working as deprecated aliases and say so via an
  RFC 8594 `Deprecation` header.

Interactive docs at `localhost:8000/docs` (FastAPI's built-in Swagger UI).
Bind address/port via `RAGSTONE_API_HOST` / `RAGSTONE_API_PORT`.

## Docker deployment

Local development never needs Docker — the venv flow and the embedded
vector stores are primary. For a server-shaped deployment, the compose
stack runs the API next to a real Qdrant server and a Postgres with
pgvector:

```bash
OPENAI_API_KEY=sk-... docker compose up --build
# API on :8000, Qdrant on :6333, Postgres on :5432
```

Documents go in `./data` (mounted read-only; `RAGSTONE_DATA_ROOT` confines
ingestion to it). The API defaults to the Qdrant server here; switch with
`VECTOR_STORE_TYPE=pgvector` (or `faiss`) — same code path, no rebuild.
The image bakes in no secrets: keys come from the environment at run
time, and `.dockerignore` excludes `.env`.

### Vector store backends

| `VECTOR_STORE_TYPE` | Where it runs | Extra | When |
|---|---|---|---|
| `faiss` (default) | in-process, per session | — | demos, evals, notebooks |
| `qdrant` | embedded (`QDRANT_PATH`) or server (`QDRANT_URL`) | `[qdrant]` | persistence without a server; scale by pointing at one |
| `pgvector` | your Postgres (`RAGSTONE_PG_URL`) | `[pgvector]` | the database you already run |
| `chroma` | embedded, persistent dir | — | legacy persistent option |

Retrieval quality is backend-independent by construction and by test —
a parity test asserts identical rankings to FAISS, and Experiment 13
measures it on the full eval set. Scale behavior is measured too
(Experiment 16): FAISS holds ~10 ms at 100k chunks; embedded Qdrant is
for small corpora (its own client warns above 20k points — use server
mode); and on macOS, large FAISS indexes (~100k vectors) need
`OMP_NUM_THREADS=1` to avoid a libomp instability (Linux/Docker
unaffected). Set `RAGSTONE_COLLECTION` to pin a
stable collection name when one deployment owns the store (unset =
unique per pipeline, the safe multi-tenant default).

### Vector stores from code

```python
from ragstone.rag.vector_db import create_vector_store_proxy

# Create Chroma vector store
chroma_store = create_vector_store_proxy(
    "chroma",
    persist_directory="custom_chroma_db"
)

# Create FAISS vector store
faiss_store = create_vector_store_proxy("faiss")
```

## Observability

Every `ask_question` / `ask_question_stream` call emits one structured log
line on the `ragstone.requests` logger:

```
request=1f2e3d4c session=web-42 chain=simple cache_hit=False latency_ms=1440 tokens=1031
```

`request` is a correlation id for tying together all log lines from one
call; `tokens` aggregates every LLM call the request needed (rephrase,
agent searches, answer), which makes per-request cost visible. Point your
log processor at these lines for p95 latency and cost-per-session — the
data is guaranteed to exist.

For deep tracing (every prompt, retrieval, and token), Ragstone works with
LangSmith out of the box — LangChain honors these env vars automatically:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=ragstone  # optional
```

No code changes needed; unset `LANGSMITH_TRACING` and the overhead is gone.

LangSmith traces carry every prompt, and a prompt contains the user's
question and the retrieved document text — they leave your machine for
LangSmith's cloud. Keep tracing off for confidential corpora;
`RAGSTONE_PROFILE=local` refuses it at boot.
