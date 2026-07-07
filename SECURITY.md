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
| API key brute-forcing via response timing | Constant-time comparison (`hmac.compare_digest`) | `api/server.py` |
| Internal detail leakage through error messages | Typed errors carry user-safe messages; anything untyped becomes a generic message, details only in server logs | MCP `_safe_error`, API exception handler |
| Hung upstream calls holding resources forever | Timeouts + bounded retries on every LLM/embedding client | `models/`, `rag/embeddings.py` |

## Deployment checklist

```bash
RAGSTONE_API_KEY=<random-32+ chars>   # API auth (X-API-Key header)
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
- **Single shared API key.** No per-client identity, quotas, or audit
  attribution — front with a gateway if you need them.
- **No output filtering.** Answers are returned as generated; add a
  moderation layer if your deployment requires one.
- **The Streamlit UI has no auth** — it is a local demo surface, not a
  deployment target.

## Reporting

This is a demonstration project. Open a GitHub issue for security
concerns; there is no embargo process.
