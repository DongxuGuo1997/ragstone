# Roadmap

Open work, written for a future contributor — human or AI agent — to
evaluate and implement. Nothing here is committed; each item states
**why** it matters, **what** to build, **how to measure it**, and a
rough **effort** (S ≈ hours, M ≈ days, L ≈ a week+).

Two standing rules inherited from this repo's history (see
[EXPERIMENTS.md](EXPERIMENTS.md)):

1. **Every quality-affecting change is eval-gated.** An item without a
   measurable acceptance criterion is not ready to implement. Design
   verdicts run on `--set large` (n=224) or the regulatory set; the
   smoke set is a regression gate, not an instrument.
2. **Never change the golden set and the system under test in the same
   comparison** (the Experiment-10 lesson).

## Delivered

Items that started on this roadmap and shipped, one line each; the
numbers are kept because code comments and tests cite them. The
measurements are in EXPERIMENTS.md; the mechanisms in
`docs/HOW_IT_WORKS.md` and `docs/CV_MATCHING.md`.

| Item | Outcome |
|---|---|
| 1.1 Contextual chunk enrichment | Default (`source` mode); hit +1.5pp, faithful +2.9pp — Experiment 12 |
| 1.5 Query routing (`auto`) | Shipped opt-in; tuning closed at 1.25× tokens vs a 1.2× gate — Experiments 15/15b |
| 1.6 Incremental indexing | Content-addressed embedding cache — Experiment 14 |
| 2.2 Answer self-check | Built, measured harmful, deleted — Experiment 17 |
| 2.3 Agent tool surface: calculator | Exact arithmetic tool, quality-neutral — Experiment 26 |
| 3.0 Second evaluation corpus | EU regulations (68 cases) and a single-document set; one verdict flipped, one walked back — Experiments 22–24 |
| 3.1 Cross-family judge | Same-provider bound (Experiment 11), then a local 31B judge over identical stored answers (Experiment 28) |
| 3.2 Confidence intervals in reports | Binomial CI per rate metric; the gate says whether a drop is inside it |
| 4.1 pgvector and Qdrant backends | Parity with FAISS by test and measurement — Experiment 13 |
| 5.1 / 5.2 Request-scoped introspection, tracing, `/metrics` | `AskContext`, OpenTelemetry spans, Prometheus series |
| 5.3 Named API keys, quotas, audit log | Per-key limits, fail-closed revocation, `GET /usage` |
| 5.7 Registry persistence, graceful shutdown | Restart without re-ingest; in-flight requests drain on SIGTERM |
| 5.8 Supply-chain CI | pip-audit with an expiring allowlist; weekly Trivy image scan |
| 5.9 `/v1` and RFC 7807 errors | Frozen contract, deprecated aliases, problem+json everywhere |
| 5.10 Session TTL and erasure | `RAGSTONE_SESSION_TTL`, `DELETE .../sessions/{sid}`, retention table in SECURITY.md |
| 5.11 Boot-time config validation | Every problem reported at once, each with its fix |
| 6.1 Citation offsets | Post-hoc evidence highlighting, no prompt change |
| 8.0 / 8.2 The local stack, measured | Four answerer tiers at cloud parity — Experiment 21; embeddinggemma default — Experiment 25 |
| 8.1 Provable no-egress mode | `RAGSTONE_PROFILE=local`, socket-intercepting regression test |
| 9.0 / 9.1 / 9.3 / 9.4 Staffing bench, matcher, UI, scale test | Gated at 1.0 on the 11-brief bench; 0.950 top-5 precision at 400 consultants — Experiments 27, 29, 30 |
| 9.2 Strengths and weaknesses with citations | Every credited claim carries a CV quote checked against the CV; gaps read "not evidenced in the CV" — Experiment 30 |

---

## 1. Retrieval quality

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
per answer must not exceed the k=6 regression that was reverted
(Experiment 10).

### 1.4 Graph-flavored retrieval (LightRAG-style) — L, research-grade
For corpora where multi-hop questions dominate, extract an entity and
relation graph at ingest and retrieve along edges. The multi-hop slice
is small (8 cases on the large set, 6 on the regulatory set) — grow it
first or the verdict will be noise.
*Measure:* multi-hop slice at n≥30, plus indexing cost comparison.
*Refs:* github.com/HKUDS/LightRAG, microsoft/graphrag.

## 2. Agentic capabilities

### 2.1 Deep-research mode — L
The current agent does search→answer with ≤3 searches. A plan-first agent
(decompose question → search per sub-question → synthesize with citations)
is the natural upgrade, and LangGraph makes the plan/execute cycle
explicit. Cap tokens per request; stream the plan as progress events
(the event side-channel already supports it).
*Measure:* a new multi-hop-heavy question set; compare vs agent mode on
quality AND token cost. Expect it to lose on simple lookups — that is
what routing is for.

### 2.3 More agent tools — S each
A corpus metadata tool ("what documents do you have?") and a
date/staleness tool. Each is a `@tool` function away with the existing
streaming events; the calculator set the pattern (Experiment 26).
*Measure:* smoke additions per tool; no regression elsewhere.

## 3. Evaluation science

### 3.0 More corpora, and the verdict matrix re-run on them — M
Two corpora have been measured (a fictional universe and EU
regulations). On the regulatory set, Experiments 4, 5 and 15 have not
been re-run, the multi-turn slice is small, and the metadata-card
default has been stressed on few document formats. A third,
materially different corpus (long PDFs with tables, a public QA set
adapted to the needle format) would test which verdicts are
corpus-conditional.
*Measure:* the chain and chunking matrix re-run per corpus; publish
which verdicts held and which flipped.

### 3.3 Golden set v3: human-verified, hardened — M
The generator's trust-nothing pipeline caught needle hallucinations but
not answerable-marked-unanswerable cases (found only by review). Add:
(a) a human sign-off column, (b) ≥2 needles or one entity-name needle
for distractor/factual cases, (c) an unanswerable audit protocol
(search the corpus for each candidate's key nouns before accepting),
(d) a gold-answer strictness review — single_doc sd05 demands a
contributor name where the institute is a correct authorship answer,
so the judge fails good answers (Experiment 22).
*Measure:* audit trail committed with the set.

### 3.4 Reference-free metrics alongside judged ones — M
RAGAS-style context precision/recall complement the gold-needle hit
rate and would catch retrieval drift on questions without needles.
*Measure:* correlation report vs existing metrics on the large set before
trusting them for gates.
*Refs:* github.com/explodinggradients/ragas.

### 3.5 True-rank MRR for multi-hop — S
Multi-hop cases are pinned at rank 1 by fiat (disclosed in
EXPERIMENTS.md). Report the max needle rank instead; re-record baselines
once (MRR will drop honestly).

## 4. Performance and scale

### 4.2 Async pipeline core — L
Everything is sync + worker threads today, and Experiment 16 measured
p50 flat to 32 concurrent clients. Native `ainvoke`/`astream` through
the graph would cut thread overhead and let the API scale past the
semaphore cap. Do it only with a load test proving the thread model is
the bottleneck.
*Measure:* p95 latency at 8/32/64 concurrent asks, before vs after.

### 4.3 Embedding quantization — S/M
For large corpora, int8 or binary quantization halves memory at small
recall cost. The embedding cache is in place; this is the storage half.
*Measure:* recall delta on the large set; memory footprint.

### 4.4 Model routing for answers — M
Route easy questions (high retrieval score margin, single entity) to a
cheaper answer model, hard ones to the strong model. The stage-timing and
cost instrumentation to verify this already exists.
*Measure:* quality flat on the large set, cost −30%+ or don't ship.

## 5. Enterprise grade

The operable and governable arcs shipped (see Delivered). What remains
is the multi-tenant arc and three smaller items.

### 5.4 SSRF: per-hop redirect validation — S
The fetchers do not follow redirects; validating each hop would let
them, safely.

### 5.5 Output moderation hook — S
Answers are returned as generated; a pluggable post-generation filter
for deployments that need one.

### 5.6 Session persistence and multi-user UI — M
The Streamlit UI is single-user with in-process sessions; a login and
per-user session store would make it a deployment surface rather than
a demo.

### 5.12 Document ACLs / per-tenant corpora — L
Tenant-scoped collections (the qdrant/pgvector backends already isolate
by collection name) plus per-key corpus visibility. Retrieval must
enforce the filter INSIDE the store query, not post-filter —
post-filtering leaks existence and breaks k.
*Measure:* an adversarial test suite: no query, citation, or metrics
line from tenant A ever contains tenant B content.

### 5.13 Horizontal scale: stateless workers — M/L
With server-mode vector stores, the sqlite checkpointer and the
per-process rate limiter are the last per-process state. Swap the
checkpointer for Postgres and share the limiter, and any number of API
workers can sit behind a load balancer.
*Measure:* Experiment 16's harness re-run against 2 and 4 workers;
session continuity across workers.

### 5.14 Backup/restore runbook — S/M
Documented, tested restore of: vector store, embedding cache, sessions,
baselines. A quarterly-restore CI job is the difference between a
backup and a hope.

## 6. Product surface

### 6.2 Feedback loop → golden candidates — M
A thumbs-down in the UI writes (question, answer, sources, session) to a
review queue; accepted items become golden-set candidates (through the
3.3 audit protocol — rule 2 applies). This closes the loop between usage
and evaluation, which is the part most RAG demos never show.

### 6.3 Multi-corpus workspaces — M/L
The registry already isolates pipelines; expose it in the UI (corpus
picker, per-corpus chat). Cache scoping by corpus fingerprint already
makes this safe.

## 7. Code health (small, steady)

- **Unify logging style** to lazy `%`-formatting repo-wide. S.
- **Unify construction verbs**: `create_` vs `make_` vs `build_` for
  chains/embeddings — pick one, alias the old names one release. S.
- **Deduplicate API/MCP pipeline setup** (provider dispatch and
  retriever-configure closures are near-identical in both servers) into
  a shared helper. S/M.
- **langchain-community exit plan**: BM25, the web/wiki loaders and the
  FAISS wrapper import from langchain-community, which upstream is
  sunsetting with no standalone homes yet (checked 2026-07). The pin
  holds a working line and `tests/test_langchain_compatibility.py` is
  the tripwire; migrate import by import with the retrieval eval
  proving parity. S/M, blocked on upstream.
- **Retire the chroma store**: legacy tier, four unfixed upstream CVEs
  in a server ragstone never runs, allowlisted until 2026-11-01. The
  embedded-persistent niche is Qdrant-local's, which has the parity
  tests. S.
- **mypy --strict ratchet**: clean under current settings; ratchet
  per-module strictness. M, background.

## 8. Local-first and private deployment

The local path competes on **measured quality per unit of privacy**:
the eval harness is the product. The measuring is done (Experiments
21, 25, 28; the no-egress profile) — what remains is breadth.

### 8.3 Local utility model + constrained decoding — S/M
Rephrase/grade/route on a small local sibling is the local analogue of
Experiment 18 — same latency win, same risk, so it ships only behind the
challenge-turn slice. Local serving supports constrained decoding (JSON
schema / grammars), which turns "usually valid" classifier output into
"always valid".
*Measure:* challenge-turn slice green on the local utility model;
router output validity 100% by construction.

### 8.4 Multilingual (Nordic) embeddings and eval slice — M
Local deployments in the Nordics mean Swedish/Norwegian/Danish
documents, and embedding quality off-English varies wildly. Add a
multilingual local embedding option (bge-m3 class) and a Nordic-language
golden slice, cross-lingual too (English question over a Swedish
document). The staffing bench's Swedish brief is a first data point.
*Measure:* hit rate/MRR on the Nordic slice, per embedding model.

### 8.5 Local serving beyond Ollama: the concurrency story — M/L
Ollama is the right dev loop; a department is not one user. Add a
vLLM/llama.cpp-server backend option (OpenAI-compatible endpoints — the
pipeline already speaks that dialect) and re-run Experiment 16's load
harness locally, where continuous batching changes the math. The
matcher's `RAGSTONE_MATCH_VERIFY_WORKERS` knob exists for exactly this.
*Measure:* capacity table — model size × GPU × concurrent users ×
p50/p99.

### 8.6 Hardware guidance, measured — S
The same eval and tokens/s on three real profiles: consumer GPU,
Apple-silicon unified memory, and CPU-only small-corpus.

### 8.7 Eval-on-your-data: the harness as a product — M
Package the offline harness to run inside a client network against
THEIR documents with a local judge: golden-set generation tooling
exists, baselines are per-configuration, nothing leaves the building.
The deliverable is the measured report — which corpus-specific verdicts
flipped, which defaults to change.
*Measure:* one dry run end-to-end on a fresh machine with no API keys.

### 8.8 Privacy-tiered hybrid routing — M
Some deployments want local-for-sensitive, cloud-for-generic. A
sensitivity route (per-corpus or per-request tag, not a classifier
guess in v1) that pins tagged corpora to the local path, with the
no-egress test asserting the pin holds under every chain type.
*Measure:* adversarial tests — no tagged-corpus token in any outbound
request, including embeddings, rephrase, and eval calls.

## 9. Staffing match — the showcase case

Match consultant CVs against a client assignment request: shortlist the
best candidates and show each one's strengths and gaps with CV-cited
evidence. CVs are personal data under the GDPR, so "no CV leaves the
machine" turns the local path from a benchmark into an argument.
Candidate–job matching brushes the EU AI Act's high-risk employment
category: this is human-in-the-loop decision support with evidence-cited
claims, never automated selection. The bench, the matcher, the UI and
the scale test are delivered (see Delivered; `docs/CV_MATCHING.md`).

### 9.5 Human agreement on real data — M
Every matcher number so far comes from synthetic CVs with ground truth
by construction, plus one real CV and one real request. The number a
staffing manager will believe is agreement with their own ranking.
*Measure:* a handful of anonymised real CVs and briefs, ranked blind by
a staffing manager; report rank correlation and the disagreements
case by case. Data stays private; only the metric is published.

### 9.6 Local baseline on the current bench — S
The local-stack staffing baseline was recorded on the 9-brief data and
predates the two RFQ-shaped briefs. Re-record on the 11-brief bench
(about 1.5 min per brief on Apple silicon; pilot with `--limit 1`
first) before the next local claim.

## Inspiration / references

- RAGFlow (agentic RAG engine): https://github.com/infiniflow/ragflow
- LightRAG (graph RAG without community detection): https://github.com/HKUDS/LightRAG
- Microsoft GraphRAG / LazyGraphRAG: https://github.com/microsoft/graphrag
- RAGAS (reference-free RAG evaluation): https://github.com/explodinggradients/ragas
- Framework landscape: https://www.firecrawl.dev/blog/best-open-source-rag-frameworks
- Technique comparisons: https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide
- Evaluation practice (judge bias, golden sets): https://www.braintrust.dev/articles/best-rag-evaluation-tools,
  https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026/
