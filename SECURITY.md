# Security Model

What Ragstone defends against, how, and — just as important — what it
does not. Written for anyone deploying the MCP server or REST API, where
inputs arrive from clients you may not fully control.

## Trust boundaries

```
 client (MCP tool call / HTTP request / UI input)
   │  question, session_id ──► length + type guards, pre-spend
   │  data_dir ─────────────► RAGSTONE_DATA_ROOT containment check
   │  page_urls ────────────► scheme + private-address (SSRF) guard
   ▼
 pipeline ──► retrieved documents ──► answer prompt
                   ▲
                   └── UNTRUSTED: documents are data, not instructions
```

## Defenses in place

| Threat | Defense | Where |
|---|---|---|
| Local file exfiltration via `data_dir` (a client points ingestion at `~/.ssh` and reads it back through answers) | `RAGSTONE_DATA_ROOT` confines ingestion to one directory tree; `..`-traversal is resolved before the containment check | `utils/security.py`, enforced in `load_and_split` |
| SSRF via `page_urls` (fetching internal services or cloud metadata endpoints) | http/https only; hosts resolving to private, loopback, link-local, or reserved addresses are rejected; the fetchers do not follow redirects (a 3xx to an internal host would bypass the check) | `utils/security.py`, `rag/loader.py` |
| Prompt injection via retrieved documents | The answer prompt delimits context and instructs the model to treat it strictly as data; an on-demand real-model check (`evals/injection_check.py`) verifies compliance | `rag/rag.py` |
| Resource exhaustion via oversized questions | Length cap rejected before any API spend (`RAGSTONE_MAX_QUESTION_CHARS`) | `rag/rag.py` |
| Unbounded concurrent load on the API | Non-blocking semaphore on `/ask` → immediate 429 | `api/server.py` |
| API key brute-forcing via response timing | Constant-time comparison (`hmac.compare_digest`) per key, full scan — timing reveals neither a matching prefix nor which key matched | `api/keys.py` |
| One client exhausting the service (or your LLM budget) | Named keys with per-key sliding-window rate limits → `429` + `Retry-After`; runtime revocation fails closed (revoking the last key locks the API, never opens it) | `api/keys.py`, `api/server.py` |
| "Who did what?" unanswerable after an incident | Append-only `ragstone.audit` line per gated request: key name, method, path, status, request id — never question or document content | `api/server.py` |
| Internal detail leakage through error messages | Typed errors carry user-safe messages; anything untyped becomes a generic message, details only in server logs | MCP `_safe_error`, API exception handler |
| Hung upstream calls holding resources forever | Timeouts + bounded retries on every LLM/embedding client | `models/`, `rag/embeddings.py` |
| Known CVEs riding in via dependencies | `pip-audit` on every CI run over the resolved dependency set; exceptions live in a time-boxed allowlist that fails the build on expiry. Trivy scans the container image (fixable HIGH/CRITICAL) on image changes and weekly | `.github/workflows/ci.yml`, `image-scan.yml`, `.github/audit-allowlist.txt` |
| "Nothing leaves the building" as a claim instead of an invariant | `RAGSTONE_PROFILE=local`: cloud providers refused, no OpenAI embedding fallback, remote document sources refused, reranker restricted to its local model cache, phone-home tracing refused, and every configured endpoint validated as loopback at boot (fail closed). Regression-tested by a socket-intercepting test over the full ingest-and-ask path | `config/settings.py`, `tests/integration/test_no_egress.py` |

## Deployment modes and data flow

Where user text can travel, per mode — the answer a DPIA asks for.

**Strict local (`RAGSTONE_PROFILE=local`)** — nothing leaves the host:

```
 documents (local files only) ─► loader ─► chunks ─► Ollama embeddings ─► FAISS/BM25 (in-process)
 question ─► Ollama (localhost) ─► answer            Ollama (localhost) ◄─ metadata cards, rephrase
 CI-enforced: socket guard fails the build on any non-loopback connection
 refused at boot/runtime: OpenAI, page_urls/wiki_query, LangSmith,
 non-loopback OTLP/Qdrant/Postgres, HuggingFace downloads (cache only)
```

**Local with cloud eval** — serving is strict-local; the eval harness
runs separately with a cloud judge (`--judge-provider openai`). Only
eval-corpus questions, answers, and retrieved chunks reach the judge —
never live user traffic. Run evals against synthetic or cleared corpora
if even that is too much.

**Hybrid (default, no profile)** — questions, retrieved chunks, and
conversation history go to the configured LLM provider (OpenAI by
default); embeddings likewise. Documents go to OpenAI at ingest only as
chunk-embedding inputs. LangSmith/OTLP send traces only if you enable
them. This is the mode the cloud baselines measure.

## Deployment checklist

```bash
RAGSTONE_API_KEYS=web:<random-32+chars>:120,batch:<random-32+chars>
                                      # named keys, per-key rpm optional
                                      # (single RAGSTONE_API_KEY also works)
RAGSTONE_DATA_ROOT=/srv/ragstone/data # confine ingestion
RAGSTONE_API_HOST=127.0.0.1           # default; put a TLS proxy in front
RAGSTONE_MAX_QUESTION_CHARS=4000
RAGSTONE_API_MAX_CONCURRENCY=8
```

## Known limitations (deliberately not claimed)

- **Prompt injection is mitigated, not solved.** Instructed-model
  defenses reduce but cannot eliminate injection; a sufficiently crafted
  document may still influence answers. Do not ingest adversarial
  corpora into a pipeline whose answers feed automated actions.
- **The SSRF guard is baseline.** It checks DNS at validation time; DNS
  rebinding between check and fetch is out of scope. Egress-restrict the
  host for defense in depth.
- **Rate limits are per process.** The sliding windows live in server
  memory: honest for one server, per-worker once you scale horizontally
  (a shared limiter is the ROADMAP 5.13 stateless-workers item).
- **Registry persistence writes corpus chunks to disk** (under
  `RAGSTONE_REGISTRY_DIR`) so pipelines survive restarts. `DELETE
  /pipelines/{id}` removes them; set `RAGSTONE_REGISTRY_PERSIST=off` if
  corpus text must never touch disk. The full "where does user text
  live" answer (sessions, caches, logs) is the ROADMAP 5.10 item.
- **No output filtering.** Answers are returned as generated; add a
  moderation layer if your deployment requires one.
- **The Streamlit UI has no auth** — it is a local demo surface, not a
  deployment target.

## Reporting

This is a demonstration project. Open a GitHub issue for security
concerns; there is no embargo process.
