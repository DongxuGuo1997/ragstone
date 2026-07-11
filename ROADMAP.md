# Roadmap

Candidate improvements, written for a future contributor — human or AI
agent — to evaluate and implement. Nothing here is committed work; each
item states **why** it matters, **what** to build, **how to measure it**,
and a rough **effort** (S ≈ hours, M ≈ days, L ≈ a week+).

Two standing rules inherited from this repo's history (see
EXPERIMENTS.md):

1. **Every quality-affecting change is eval-gated.** An item without a
   measurable acceptance criterion is not ready to implement. Design
   verdicts run on `--set large` (n=224); the smoke set is a regression
   gate, not an instrument.
2. **Never change the golden set and the system under test in the same
   comparison** (the Experiment-10 lesson).

---

## 1. Retrieval quality

### 1.1 Contextual chunk enrichment — DELIVERED (Experiment 12; `source` mode is the default)
Prepend a short LLM-generated context line to each chunk before embedding
("This chunk is from the Corona K-7 installation guide, section:
warranty"), so chunks carry their document identity into the vector
space. This directly targets our measured failure mode: engineered
distractors winning on bare-chunk similarity (Helios vs Corona cleaning
schedules — the largest failure cluster at n=224).
*Measure:* large-set `hit_rate`/`mrr` on the distractor slice, plus
end-to-end correct/faithful (remember Experiment 10: slice wins can be
end-to-end losses). One-time indexing cost ~1 LLM call per chunk.

### 1.2 Semantic / structure-aware chunking — M
`RecursiveCharacterTextSplitter` cuts mid-topic. Markdown-header-aware
splitting (or embedding-similarity breakpoints) keeps facts attached to
their subjects — Experiment 5 showed smaller chunks *hurt* precisely
because they split facts from subjects; smarter boundaries attack the
same problem without shrinking.
*Measure:* Experiment-5 protocol re-run: chunking variants × large set.

### 1.3 Parent-document retrieval — S/M
Embed small chunks (precise matching) but hand the generator their
enclosing section (fuller context). Cheap to build with the existing
splitter metadata.
*Measure:* end-to-end faithful/correct at fixed token budget; token cost
per answer must not exceed the k=6 regression we reverted.

### 1.4 Graph-flavored retrieval (LightRAG-style) — L, research-grade
For corpora where multi-hop questions dominate, extract an entity/relation
graph at ingest and retrieve along edges (dual-level: entities + concepts).
Microsoft's GraphRAG showed the idea; LightRAG showed it doesn't need
expensive community detection. Our multi-hop slice is small (8 cases) —
grow it first or the verdict will be noise.
*Measure:* multi-hop slice at n≥30, plus indexing cost comparison.
*Refs:* github.com/HKUDS/LightRAG, microsoft/graphrag.

### 1.5 Query routing — SHIPPED opt-in; tuning avenue CLOSED (Experiments 15/15b: tuned variant reaches faithful 0.981 at 1.25x tokens — still over the 1.2x default gate)
Classify each question (factual lookup / comparative / multi-hop /
unanswerable-looking) with the cheap utility model and route: simple chain
for lookups, corrective for likely-miss queries, agent for multi-hop.
Experiments 8–9 showed corrective costs 2× with no *average* gain — but
its refusal behavior is valuable on the right slice. Routing spends the
2× only where it might pay.
*Measure:* end-to-end large-set quality at ≤1.2× simple-chain tokens.

### 1.6 Incremental indexing / freshness — DELIVERED via the embedding cache (Experiment 14)
`load_and_split` rebuilds the world. Index per-document with content
hashes (the corpus fingerprint machinery already exists) so adding one
document embeds one document; delete/update likewise.
*Measure:* ingest time on a 15-of-16-docs-unchanged corpus; retrieval
metrics unchanged.

## 2. Agentic capabilities

### 2.1 Deep-research mode — L
The current agent does search→answer with ≤3 searches. A plan-first agent
(decompose question → search per sub-question → synthesize with citations)
is the natural showcase upgrade, and LangGraph makes the plan/execute
cycle explicit. Cap tokens per request; stream the plan as progress
events (the event side-channel already supports it).
*Measure:* a new multi-hop-heavy question set; compare vs agent mode on
quality AND token cost. Expect it to lose on simple lookups — that's what
routing (1.5) is for.

### 2.2 Answer self-check pass — CLOSED: built, measured harmful, deleted (Experiment 17: correctness −3.8pp outside the CI; caveats from an imperfect checker poison correct answers)
After generation, one cheap-model pass: "does every claim in this answer
appear in the context?" — flag or strip unsupported claims before
returning. This is corrective RAG applied to the *output* side, and it
targets faithfulness, our weaker metric (0.938 vs 0.943 at n=224).
*Measure:* large-set faithful_rate; latency budget +1 utility call.

### 2.3 Tool surface expansion for agent mode — S each
Calculator (numeric comparisons appear in the corpus questions), corpus
metadata tool ("what documents do you have?"), and a date/staleness tool.
Each is a `@tool` function away with the existing streaming events.
*Measure:* smoke additions per tool; no regression elsewhere.

## 3. Evaluation science

### 3.0 Second evaluation corpus — FIRST REAL CORPUS DELIVERED (July 2026, Experiment 23); expansion open
Seventeen experiments rest on one fictional corpus whose retrieval is
near-saturated (hit 0.97, smoke faithfulness 1.0). Every verdict — and
every opt-in's *enable-when* condition in the README's support-tier
table — is a single-corpus result. A second, materially different corpus
(real documents: long PDFs, tables, domain jargon; or a public QA set
adapted to the needle format) would validate or overturn the existing
conclusions wholesale, which is worth more than any new experiment on
the current corpus. The kept-though-rejected policy (ARCHITECTURE.md)
makes this the binding priority: optionality without a validated trigger
condition is just a shelf.
*Measure:* re-run Experiments 4/8/12/15 on the new corpus; publish which
verdicts held and which flipped. Fold the n>=40 multi-turn expansion
(3.3's sibling) into the new set so the conversation slice stops
generating one-case noise. The expansion must include challenge turns
("are you sure?") and document-metadata questions ("who wrote this?") —
Experiment 18 proved the corpus was blind to the first class, and the
same live transcript exposed the second: author blocks are extracted
and chunked but never rank for "who created X" phrasing in either
retrieval leg. The metadata-card fix was built and
measured the same week (Experiment 19, now a CI-gated default); the
second corpus should stress it with more document formats.

**Regulatory corpus delivered (July 2026, Experiment 23).** GDPR + the
EU AI Act from EUR-Lex (`--set regulatory`, 1,331 chunks, 27
grep-verified cases): enrichment's verdict HELD, **corrective's
enable-when condition was VALIDATED** (first corpus with hit <0.9;
faithfulness +8.7pp, multi-turn +50pp at 2.2× tokens), rerank's verdict
split by embedder (decisive for nomic, marginal for openai), and
Experiment 21's local-parity claim found its boundary — nomic trails
badly on legal jargon (hit 0.55 vs 0.80) while local generation holds;
the local reranker closes most of the gap. New failure class: fine-tier
confusion from near-duplicate numeric schedules (funds 1.2/1.3).
Remaining for full 3.0: re-run Experiments 4/5/15 here, expand
multi-turn beyond n=4, the 3.3 hardening pass, and a judge-parser fix
(two verdicts lost to unescaped-quote JSON, kept frozen mid-experiment).

**First slice delivered (July 2026), from a live failure.** A single
uploaded research PDF answered "what is deepseek" from three contributor
name-lists and a table of contents: the chunk-enrichment prefix
(Experiment 12's multi-doc win) dominates the embeddings of
content-empty chunks — making them nearest neighbors for any query that
names the document — and puts the document's name in every chunk, zeroing
its BM25 IDF. Diagnosis reproduced and attributed (vector leg ranks the
name lists 1–4; metadata card 14th of 80). Now measured: `--set
single_doc` (one paper-shaped fictional document with those decoy
structures engineered in, 9 document-level cases). Honest baselines
committed: retrieval MRR **0.51–0.66** vs 0.95+ on smoke; gpt-4o-mini
correct **0.667** (misses name-meaning, authorship, and fabricates on
the unanswerable) while qwen3.6:35b sweeps 9/9.
**Fixed (Experiment 22, July 2026):** two stacked bugs — the enrichment
prefix inverts on single-doc corpora (now skipped when there is one
source document) and nomic-embed-text was missing its required task
prefixes (now applied via `NomicTaskEmbeddings`). MRR 0.594→0.750
(OpenAI) / 0.656→0.719 (nomic); every protection gate held exactly;
the live PDF's "what is deepseek" now retrieves the card and abstract
instead of name lists. Remaining, documented: gpt-4o-mini's
generation-side weakness on document-level answers, and sd05's
over-specified gold answer (3.3).

### 3.1 Cross-family judge — DELIVERED same-provider (Experiment 11); different-provider judge still open
The judge and the answering model are both gpt-4o-mini; self-preference
inflation is a known LLM-as-judge bias, and it is now disclosed in
EXPERIMENTS.md but not bounded. Run one full large-set pass with a
different-family judge (e.g. a Claude or Gemini model) and report both
columns; keep the cross-family judge as an option (`--judge-provider`
already exists).
*Measure:* the delta between judges IS the result — publish it.

### 3.2 Confidence intervals in reports — DELIVERED
Every score in `report.md` should carry its binomial 95% CI
(`±1.96·√(p(1−p)/n)`), and the baseline gate should annotate whether a
drop is outside the CI, not just outside TOLERANCE. This mechanizes the
Experiment-9 lesson instead of relying on discipline.
*Measure:* n/a (tooling); acceptance = CIs shown per metric with n.

### 3.3 Golden set v3: human-verified, hardened — M
The generator's trust-nothing pipeline caught needle hallucinations but
not answerable-marked-unanswerable cases (g172/g173, found only by
review). Add: (a) a human sign-off column, (b) ≥2 needles or one
entity-name needle for distractor/factual cases (single short needles
like "2.75" can hit the wrong chunk), (c) an unanswerable audit protocol
(search the corpus for each candidate's key nouns before accepting),
(d) a gold-answer strictness review — single_doc sd05 demands a
contributor name where "the Meridian Institute" is a correct authorship
answer, so the judge fails good answers (Experiment 22).
*Measure:* audit trail committed with the set.

### 3.4 Reference-free metrics alongside judged ones — M
RAGAS-style context precision/recall complement our gold-needle hit rate
and would catch retrieval quality drift on questions without needles.
*Measure:* correlation report vs existing metrics on the large set before
trusting them for gates.
*Refs:* github.com/explodinggradients/ragas.

### 3.5 True-rank MRR for multi-hop — S
Multi-hop cases are pinned at rank 1 by fiat (disclosed in
EXPERIMENTS.md). Report the max needle rank instead; re-record baselines
once (mrr will drop honestly).

## 4. Performance and scale

### 4.1 pgvector / Qdrant backend — DELIVERED, both (Experiment 13; compose stack included)
FAISS is in-process and rebuilt per session; a server-backed store gives
persistence, metadata filtering, and multi-process access. The
`VectorStoreProxy` ABC is the seam — implement `PgVectorProxy` behind it.
Deferred once already (Phase D) as not needed for the demo; becomes
first-priority the moment the corpus outgrows memory or two processes
need one index.
*Measure:* retrieval metrics identical to FAISS at equal k; ingest and
query latency published.

### 4.2 Async pipeline core — L
Everything is sync + worker threads today (a defensible choice — the
servers stay responsive). Native `ainvoke`/`astream` through the graph
would cut thread overhead and let the API scale past the semaphore cap.
Do it only with a load test proving the thread model is the bottleneck.
*Measure:* p95 latency at 8/32/64 concurrent asks, before vs after.

### 4.3 Embedding cache & quantization — cache DELIVERED (with 1.6); quantization still open
Cache embeddings by content hash (survives re-ingest of unchanged docs;
pairs with 1.6). For large corpora, int8/binary quantization halves memory
at small recall cost.
*Measure:* recall delta on large set; memory footprint.

### 4.4 Model routing for answers — M
Route easy questions (high retrieval score margin, single entity) to a
cheaper answer model, hard ones to the strong model. The stage-timing and
cost instrumentation to verify this already exists.
*Measure:* quality flat on large set, cost −30%+ or don't ship.

## 5. Enterprise grade — the committed next arc

The quality work above is what makes Ragstone worth running; this
section is what a company needs to actually run it. Three arcs, in
dependency order — each is a coherent deliverable with a demo at the
end, not a grab bag. (Agreed as the next major work after the July 2026
engine-hardening arc.)

### Arc 1 — Operable: you can see it, secure it, and restart it

#### 5.1 Request-scoped introspection state — DELIVERED (July 2026)
Per-ask state now lives on an ownership-checked AskContext (contextvar),
ending the single-writer caveat; concurrent asks are regression-tested
for cross-request bleed.

#### 5.2 Distributed tracing: X-Request-ID + OpenTelemetry + /metrics — DELIVERED (July 2026)
All three parts shipped: the API adopts a well-formed client
`X-Request-ID` (mints one otherwise), stamps it on every response, and
`track_request` logs the same id; `/metrics` exposes Prometheus series
(requests by chain/cache/error, latency histograms end-to-end/per-stage/
first-token, token totals) via an exception-isolated observer hook; each
ask becomes a `ragstone.ask` OTel span with retroactive stage children —
explicit timestamps, no context attach, so the streaming path can't tear
it. The `[otel]` extra + `OTEL_EXPORTER_OTLP_ENDPOINT` wire the OTLP
exporter. Evidence: span topology pinned by in-memory-exporter tests;
live server smoke shows adopted ids and real collectors on `/metrics`.
Still open (nice-to-have): a committed Grafana dashboard JSON.

#### 5.3 Named API keys, per-key quotas, audit log — DELIVERED (July 2026)
`RAGSTONE_API_KEYS=name:key[:rpm],...` (legacy single key still works as
"default"): per-key sliding-window limits → 429 + Retry-After, runtime
revocation that fails closed (revoking the last key locks the API — the
first implementation silently disabled auth instead, caught by the
integration test), an append-only `ragstone.audit` line per gated
request (key name, method, path, status, request id — never content),
and `GET /usage` for per-key attribution. Measure met: integration
tests cover quota enforcement, revocation, audit attribution, and the
usage report. Limits are per process until 5.13.

#### 5.7 Registry persistence + graceful shutdown — DELIVERED (July 2026)
Retriever setup persists a manifest (provider/model/chain/retriever
config, fingerprint) plus the ENRICHED chunks; the first request for an
unknown id restores lazily — no re-ingest, no LLM calls (enrichment and
metadata cards are baked into the persisted text), embeddings from the
cache. Wired into both the REST API and the MCP server; DELETE removes
the manifest (nothing resurrects); `RAGSTONE_REGISTRY_PERSIST=off` for
no-disk deployments; manifest filenames are id hashes (client-supplied
ids never touch paths). Both measures met in tests: kill -TERM under
load completes the in-flight request with 200 (real uvicorn subprocess),
and a cleared registry serves the same corpus id from its manifest.

#### 5.8 Supply-chain CI: pip-audit + image scan — DELIVERED (July 2026)
`pip-audit` runs per push/PR over the resolved dependency set; allowlist
entries carry expiry dates and the job fails when one lapses (or lacks
one), so exceptions are time-boxed by construction. Trivy scans the
container image (fixable HIGH/CRITICAL) on Dockerfile/pyproject changes
and weekly — new CVEs appear against unchanged images. Measure met
locally: a deliberately vulnerable pin exits non-zero.

### Arc 2 — Governable: contracts, errors, and the data lifecycle

#### 5.9 API versioning (/v1) + RFC 7807 error bodies — S/M
Freeze today's REST surface as `/v1`; every error becomes a
`application/problem+json` body with type/title/detail/instance and the
request id. Contract tests pin the schema.

#### 5.10 Session TTL and deletion — DELIVERED (July 2026)
`RAGSTONE_SESSION_TTL` expires idle sessions (lazy sweep on the ask
path, same pattern as the response cache; injectable clock in tests);
`MemoryProxy.delete_session` erases a thread from the checkpointer via
`delete_thread`, surfaced as `DELETE /pipelines/{id}/sessions/{sid}`
(idempotent, audited, worker-thread offloaded) and an MCP
`delete_session` tool. SECURITY.md gained the "where user text lives,
and when it dies" retention table — per store, with the erasure
caveats stated honestly (lazy sweeps track activity per process;
pre-restart sqlite threads need explicit DELETE; logs carry ids, never
content, and are not retro-edited). The deletion test proves the turn
AFTER erasure behaves as a first turn (no rephrase against ghost
history). `/v1` path prefixes land with 5.9.

#### 5.11 Boot-time config validation, fail-fast — S
Validate the full config at startup (store reachable, model available,
key present for the chosen provider) and refuse to boot half-working,
with an actionable message per failure.

#### 5.4 SSRF: per-hop redirect validation — S *(existing item, fits here)*
#### 5.5 Output moderation hook — S *(existing item, fits here)*
#### 5.6 Session persistence & multi-user UI — M *(existing item, fits here)*

### Arc 3 — Multi-tenant and scale

#### 5.12 Document ACLs / per-tenant corpora — L
Tenant-scoped collections (the qdrant/pgvector backends already isolate
by collection name) plus per-key corpus visibility. Retrieval must
enforce the filter INSIDE the store query, not post-filter — post-
filtering leaks existence and breaks k.
*Measure:* an adversarial test suite: no query, citation, or metrics
line from tenant A ever contains tenant B content.

#### 5.13 Horizontal scale: stateless workers — M/L
With server-mode vector stores (4.1), the sqlite checkpointer is the
last per-process state. Swap it for the Postgres checkpointer, and any
number of API workers can sit behind a load balancer.
*Measure:* Experiment 16's harness re-run against 2 and 4 workers;
session continuity across workers.

#### 5.14 Backup/restore runbook — S/M
Documented, tested restore of: vector store, embedding cache, sessions,
baselines. A quarterly-restore CI job is the difference between a
backup and a hope.

## 6. Product surface

### 6.1 Citation offsets — M
`get_sources` returns chunks; returning character offsets per claim
enables click-to-highlight in the UI — the single most convincing
glass-box feature in demos.
*Measure:* offsets verified against source docs in tests.

### 6.2 Feedback loop → golden candidates — M
A thumbs-down in the UI writes (question, answer, sources, session) to a
review queue; accepted items become golden-set candidates (through the
3.3 audit protocol — rule 2 applies). This closes the loop between usage
and evaluation, which is the part most RAG demos never show.

### 6.3 Multi-corpus workspaces — M/L
The registry already isolates pipelines; expose it in the UI (corpus
picker, per-corpus chat) and the story upgrades from "a pipeline" to "a
platform". Cache scoping by corpus fingerprint (July 2026) already makes
this safe.

## 7. Code health (small, steady)

- **Unify logging style** to lazy `%`-formatting repo-wide (observability
  and memory already do it; ~66 f-string log calls elsewhere). S.
- **Unify construction verbs**: `create_` vs `make_` vs `build_` for
  chains/embeddings — pick one, alias the old names one release. S.
- **Class docstrings for the ~39 undocumented test classes** — one line
  each stating the invariant the class locks. S.
- **Deduplicate API/MCP pipeline setup** (provider dispatch + retriever
  configure closures are near-identical in both servers) into a shared
  helper module. S/M.
- **langchain-community exit plan**: BM25, the web/wiki loaders, and
  the FAISS wrapper all import from langchain-community, which upstream
  is sunsetting — and none have standalone package homes yet (checked
  2026-07). The version pin holds a working line and the compatibility
  canary suite is the tripwire; when standalone packages appear, migrate
  import-by-import with the retrieval eval proving metric parity. S/M,
  blocked on upstream.
- **mypy --strict ratchet**: the codebase is clean under current
  settings; ratchet per-module strictness the same way the original
  mypy debt was paid down. M, background.

---

## 8. Local-first and private deployment — the second strategic path

Many clients value privacy and control above raw quality: regulated
industries, public sector, and anyone whose documents cannot leave
their network. Ragstone already runs fully offline (Ollama + FAISS +
local reranker + local embeddings) — but "runs" is not a claim this
repo makes without a number attached. The honest current state: **the
entire eval harness has only ever measured the OpenAI path.** Every
verdict in EXPERIMENTS.md is a cloud-model verdict. That gap is the
first item below, and everything else keys off it.

The strategic frame: the cloud path competes on quality, where a
wrapper adds little; the local path competes on **measured quality per
unit of privacy**, where the eval harness IS the product. Nobody buys
"local RAG" — they buy "local RAG that we proved loses only X points
on your documents, with a CI-enforced guarantee that nothing leaves
the building."

### 8.0 Measure the local stack — DELIVERED (July 2026, Experiment 21)
A 2-light/2-heavy answerer matrix (gemma4:e4b, qwen3.5:9b, qwen3.6:35b,
gemma4:31b) through the full smoke harness on nomic-embed-text + the
local reranker: parity with the cloud baseline on this corpus (every
tier ≥ 0.951 correct; local rerank retrieval 1.0/1.0), with the durable
finding in the tier SHAPE — the edge model breaks on the multi-turn
slice (5/8), heavy-dense buys 41/41 correctness at 8 s/ask. Judge delta
done the strong way Experiment 11 flagged: `evals/rejudge.py` re-scores
STORED answers (`--dump-answers`), cloud self-noise floor 1/98 flips,
local gemma4:31b judge 4/98 with 0 parse failures, within 2.5–4.9 pp.
tokens/s + latency recorded per tier; `RAGSTONE_OLLAMA_REASONING`
shipped (thinking off = the measured serving posture); local baselines
committed under `ollama:` keys; `make eval-local`. Judge was 31B
cross-family, not 70B-class — rerun with a 70B judge folds into 8.2.

### 8.1 Provable no-egress mode — DELIVERED (July 2026)
`RAGSTONE_PROFILE=local` fails CLOSED: cloud providers refused at the
single provider-dispatch chokepoint, no OpenAI embedding fallback,
remote document sources (page_urls/wiki_query) refused, the reranker
restricted to its local HuggingFace cache, LangSmith tracing refused,
and every configured endpoint (Ollama/Qdrant/Postgres/OTLP) validated
as loopback AT BOOT — by name, no DNS, same rebinding stance as the
SSRF guard. The part that sells it:
`tests/integration/test_no_egress.py` intercepts socket.connect across
the full ingest-and-ask path and fails on any non-loopback destination
— a CI-safe fake-model tier (real loaders/enrichment/indexing/chain)
plus a live tier that ran the real qwen3.5:9b + nomic path clean.
Data-flow modes (strict-local / local-with-cloud-eval / hybrid)
documented in SECURITY.md. Honest limitation recorded: the guard sees
connections, not libc DNS lookups.

### 8.2 A measured local model menu, in tiers — M
`llama3` as the sole local default is dated. Curate and MEASURE three
tiers — edge (3-4B class), workstation (7-14B), server (70B+ / MoE) —
plus one reasoning-distill model to answer whether thinking models
close the local corrective/agent gap. The default per tier is decided
by the harness, like every other default in this repo.
*Measure:* eval columns per tier; the size-vs-quality knee is the
deliverable clients ask for ("what is the smallest model we can
defend?").

### 8.3 Local utility model + constrained decoding — S/M
Rephrase/grade/route on a small local sibling is the local analogue of
Experiment 18 — same latency win, same risk, so it ships only behind
the challenge-turn slice. For the router and grader, stop
prompt-and-praying: local serving supports constrained decoding
(JSON schema / grammars), which turns "usually valid" classifier output
into "always valid" — cheap reliability the cloud path never needed.
*Measure:* challenge-turn slice green on the local utility model;
router output validity 100% by construction.

### 8.4 Multilingual (Nordic) embeddings and eval slice — M
Local deployments here mean Swedish/Norwegian/Danish documents, and
embedding quality off-English varies wildly. Add a multilingual local
embedding option (bge-m3 class) and a Nordic-language golden slice
(cross-lingual too: English question over a Swedish document — the real
office pattern).
*Measure:* hit rate/MRR on the Nordic slice, per embedding model. This
single table is a consulting differentiator; nobody publishes it.

### 8.5 Local serving beyond Ollama: the concurrency story — M/L
Ollama is the right dev loop; a department is not one user. Add a
vLLM/llama.cpp-server backend option (OpenAI-compatible endpoints —
the pipeline already speaks that dialect) and re-run Experiment 16's
load harness locally, where continuous batching changes the math
entirely.
*Measure:* capacity planning table — model size x GPU x concurrent
users x p50/p99 — the number one question in a deployment pre-study.

### 8.6 Hardware guidance, measured — S
The same eval + tokens/s on three real profiles: consumer GPU,
Apple-silicon unified memory (where Ollama shines), and CPU-only
small-corpus. Kills the "do we need an A100?" conversation with data.

### 8.7 Eval-on-your-data: the harness as a consulting product — M
Package the offline harness (8.0 + 8.1) to run inside a client network
against THEIR documents with a local judge: golden-set generation
tooling exists, baselines are per-configuration, nothing leaves the
building. The engagement deliverable is the measured report — which
corpus-specific verdicts flipped, which defaults to change — i.e.,
ROADMAP 3.0 executed on the client's corpus, as a product.
*Measure:* one dry run end-to-end on a fresh machine with no API keys.

### 8.8 Privacy-tiered hybrid routing — M, after 8.0–8.2
Some clients want local-for-sensitive, cloud-for-generic. A
sensitivity route (per-corpus or per-request tag, not a classifier
guess in v1) that pins tagged corpora to the local path, with the
no-egress test asserting the pin holds under every chain type.
*Measure:* adversarial tests — no tagged-corpus token in any outbound
request, including embeddings, rephrase, and eval calls.

## Inspiration / references

Surveyed July 2026 while drafting this document:

- RAGFlow (agentic RAG engine): https://github.com/infiniflow/ragflow
- LightRAG (graph RAG without community detection): https://github.com/HKUDS/LightRAG
- Microsoft GraphRAG / LazyGraphRAG: https://github.com/microsoft/graphrag
- RAGAS (reference-free RAG evaluation): https://github.com/explodinggradients/ragas
- Framework landscape: https://www.firecrawl.dev/blog/best-open-source-rag-frameworks
- Technique comparisons: https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide
- Evaluation practice (judge bias, golden sets): https://www.braintrust.dev/articles/best-rag-evaluation-tools,
  https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026/
