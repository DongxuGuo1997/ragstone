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

### 1.5 Query routing — SHIPPED opt-in (Experiment 15: quality held, 1.3x tokens failed the default gate; tuning ideas below still open)
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

### 2.2 Answer self-check pass — S/M
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
(search the corpus for each candidate's key nouns before accepting).
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

## 5. Production hardening

### 5.1 Request-scoped introspection state — M
`get_sources` / `last_metrics` / `get_last_retrieved_documents` are
single-writer per pipeline (documented in the `Pipeline` docstring). Move
per-ask state (the recording retriever's list, last question/metrics)
into a request context object returned by the ask, or contextvars.
Removes the one concurrency caveat left after the July 2026 review.
*Measure:* a concurrent-asks test asserting no cross-request bleed.

### 5.2 OpenTelemetry export — M
`ragstone.requests` log lines carry request id, latency, tokens, stages —
map them to OTel spans (retrieval span, rephrase span, generation span)
so any APM can ingest them. Alternative: a Langfuse callback for
LLM-native tracing.
*Measure:* trace visible end-to-end in a local Jaeger/Langfuse.

### 5.3 Per-client API keys and quotas — M
One shared key today (documented limitation). A keyed table with
per-client rate limits and usage attribution unlocks multi-team demos.
*Measure:* integration tests for quota enforcement; usage report per key.

### 5.4 SSRF: per-hop validation — S
Redirects are now refused outright; the friendlier version follows them
manually, re-validating each `Location` against `validate_page_url`
(bounded hops). DNS pinning (resolve once, connect to the validated IP)
would close the remaining rebinding TOCTOU.
*Measure:* unit tests with a redirecting test server.

### 5.5 Output moderation hook — S
An optional post-generation callback slot (regex/PII scrub or a
moderation model) before answers leave the API. Currently a documented
non-goal; make it a pluggable seam instead.

### 5.6 Session persistence & multi-user UI — M
The Streamlit cache toggle mutates process-global config (single-user
assumption, noted in code); sessions die with the process unless sqlite
is enabled. A proper multi-user story needs per-session config and a
session browser backed by the checkpointer.

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
