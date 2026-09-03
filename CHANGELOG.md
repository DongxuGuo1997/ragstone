# Changelog

Notable changes to Ragstone. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Eval integrity hardening**: baselines now record a `data_sha`
  fingerprint of their golden set + corpus and every gate compares it —
  editing measured data fails loudly instead of silently invalidating
  old numbers (pre-fingerprint entries keep gating on metrics alone).
  The LLM judge is pinned to a dated snapshot
  (`gpt-4o-mini-2024-07-18`, verified live) so provider-side alias
  moves can't masquerade as regressions; baseline keys keep the alias
  and answerer models stay deliberately unpinned.
- **Entry-point coverage**: all five console scripts are import-tested
  against pyproject, and both Streamlit UIs boot headless in the suite
  (server up, health endpoint answering) — a broken first command now
  fails CI.
- **Local embedding default: embeddinggemma** (Experiment 25 / ROADMAP
  8.2): a 622 MB model that matches the cloud embedder on real legal
  text (hit 0.80/MRR 0.65 vs nomic's 0.56/0.47) and improves the smoke
  slice (1.0/0.939); end-to-end local correctness rose +7–14pp from the
  swap alone. New machinery: a per-family task-convention table
  (validated empirically — embeddinggemma's documented templates
  measured HARMFUL and are deliberately not applied),
  `RAGSTONE_OLLAMA_EMBED_MODEL` (pin an embedder, fail-loud), and
  `RAGSTONE_EMBED_TASK_PREFIXES` (A-B knob). Also measured: thinking
  mode bought zero correctness at 33× latency — reasoning does not fix
  retrieval ambiguity.

### Added
- **Second-vote screening of credited evidence** (Experiment 30, part
  4). A credited skill now stands only if the CV names it: a quote that
  names the skill and occurs in the CV is kept, a weak or invented quote
  is replaced by the first CV line naming the skill, and a product name
  the CV never mentions flips to "not evidenced" — no model call. A
  generic phrase the CV never names ("hardware interfacing") goes to a
  stricter quote-only judge with one repair call whose answer must
  occur in the CV. Years, degree, language and domain are exempt. On
  the 400-consultant scale test top-5 precision rises 0.925 → 0.950
  with the Experiment-29 residual (CAN bus credited from a Vector CANoe
  line) now caught; the 11-brief gates are unchanged. The years
  arithmetic now treats an Education heading as a block and a bare
  date line after a degree as the degree's, so a Master's no longer
  counts as work. `--second-vote off` keeps the previous path.
- **Phrases are judged, product names are looked up.** The second
  vote now decides "product name or phrase" by shape rather than by
  capital letters: a single token, or a token with a digit, a symbol or
  internal capitals (PyTest, CANoe, ISO 26262, C++) must be named by
  the CV; a phrase of ordinary words ("Python-based test automation",
  "Swedish driving license B", "Android Automotive Software
  Development") goes to the quote-only judge, and a repaired quote is
  judged once more before it is accepted. On a real request this
  turned four false "not evidenced" verdicts — Pytest-based test
  automation, Android Automotive, AI-assisted tools, a driving licence
  — into credits with the CV line that shows them. Gates unchanged.
- **Requirements shown as read, preferred items ticked per candidate.**
  The staffing UI now shows what the brief was read as — must-haves,
  preferred items and location — above the shortlist, and each
  candidate lists every preferred item with a tick or a cross instead
  of a caption naming only the hits. Preferred items match more
  leniently than must-haves (singular form, or the first two words of
  a long phrase: "Hypervisors" ~ "Hypervisor", "Android Automotive
  Software Development" ~ "Android Automotive apps"). Years of
  experience count only ranges under an Experience-like heading when
  the CV has one, and the evidence lists the exact spans counted.
- **Years of experience by arithmetic** (Experiment 30, part 3). The
  matcher sums the CV's engagement date ranges (union of intervals,
  education lines excluded, open ranges end this year) and lets that
  decide the years requirement; the evidence says it was computed, not
  quoted. The model's reading is used only for a CV with no date range.
  Gates unchanged on the bench; a real CV whose profile blurb overstates
  its years can no longer pass on the blurb.
- **The matcher's extractor reads RFQs** (Experiment 30, part 2). The
  extraction schema is now per line: the model copies every sentence of
  a must/preferred section verbatim, classifies it, and states whether
  its items are all required or alternatives; the parser builds the
  requirement list from that. Two new requirement kinds — `education`
  (a degree, verified against the CV) and `location` (context on the
  requirements line and an informational note per candidate, never a
  gap) — plus guards that keep programming languages out of the
  spoken-language kind and preferred items out of must. Measured on the
  11-brief bench: strong-candidate recall back to **1.0**, extraction
  recall and precision **0.984** (both now gated), stability 1.0 across
  three samples; the real RFQ that motivated the work extracts
  identically three times out of three. Optional per-item majority
  voting across N extraction samples is available (`extract_samples`),
  default 1.
- **Long-form RFQ briefs in the staffing bench** (a10, a11) and
  extraction-level eval metrics (Experiment 30, part 1). Two
  hand-authored briefs in the shape real requests arrive in — sectioned
  prose, "including X and Y" conjunctions, comma-list conjunctions, a
  degree sentence, the location only in prose, a preferred item that
  must not be promoted — with the same ground-truth-by-construction
  oracle. `run_staffing_eval.py` now scores the extracted must-have set
  directly (`extract_must_recall` / `extract_must_precision`, plus
  nice-to-have recall, location capture and `--extract-repeats N`
  stability). Measured before any fix: the current extractor drops
  strong-candidate recall from 1.0 to 0.903 on the new bench — the
  real-RFQ failure, reproduced and on record.
- **Engine mechanism doc** (`docs/HOW_IT_WORKS.md`): how ragstone
  works end to end for users and reviewers — the ingest path (load,
  split, enrich, metadata cards, embed, index) and the ask path
  (guards, cache, rephrase, hybrid retrieval, optional rerank, the
  strict answer prompt, post-hoc evidence highlighting, the
  per-request receipt), the chain types with their measured verdicts
  and enable-when conditions, the four front doors, the two-layer
  eval gate with committed results, privacy posture, and known
  limits. Linked from the README's measured-at-a-glance footer.
- **CV-matching mechanism doc** (`docs/CV_MATCHING.md`): how the
  matcher works stage by stage — ingestion and person tagging,
  extract, discover (coverage-breadth ranking, exact-phrase channel,
  the ten-candidate cap), verify (the strictness rules, fail-closed
  parsing), score (tiers, full-match honesty, gap wording) — with a
  worked example, the metric definitions and results, the privacy
  posture, and the known limits. Written for users and reviewers,
  not only developers; linked from the README.
- **Staffing demo runbook** (`docs/guides/STAFFING_DEMO_SCRIPT.md`):
  a rehearsed six-act script for presenting the CV↔assignment matcher
  live — cloud-paced acts, a "bring your own DOCX" upload act, and a
  pre-started local finale (rehearsal measured the 9B at 25.8
  min/assignment on battery — quality-identical to cloud — so the
  local act is staged as a reveal, not a wait). Every timing and
  claim in the script traces to a measured run.
- **Open-source front door**: bug-report and feature-request issue
  forms (the latter asks how the change would be measured), a PR
  template carrying the two eval-gating rules from CONTRIBUTING.md,
  a Contributor Covenant 2.1 code of conduct, and CI/license/Python
  badges on the README.
- **Staffing scale test** (ROADMAP 9.4 / Experiment 29): the same nine
  assignments over a 400-consultant population (`--scale 10`
  generator; the committed 40-person bench proven byte-identical and
  untouched). Top-5 precision 0.925 with honesty and ordering perfect;
  the first run's misses exposed and fixed the one unchecked
  prose-label dimension (domain, now a mechanical Requirements-section
  check), and the residual three slots are attributed verifier
  leniency — the named next lever. Matcher verification now runs
  concurrently (metric-neutral, 5–8× faster: ~10 s/assignment; the
  demo eval completes in 90 s).
- **Staffing match, fully local and measured** (ROADMAP 9.3 delivered):
  the 9-assignment staffing eval on qwen3.5:9b + embeddinggemma under
  the enforced no-egress profile scores identically to the cloud stack
  — strong_recall@5 1.0, honesty 1.0, ordering 1.0. New baseline key
  `ollama:qwen3.5:9b|k=12|chain=match|set=staffing|ollama-reasoning=off`.
  "No CV leaves the machine" is now a measured claim.
- **Cross-provider judge audit** (Experiment 28 / ROADMAP 3.1
  delivered): gemma4:31b — different provider, different family, fully
  local — re-judged the regulatory dump's identical stored answers.
  88% verdict agreement, 0 parse failures in 136; correctness +3.6pp
  under the local judge (the cloud judge is conservative, not
  self-flattering), faithfulness −5.4pp (stricter grounding, including
  one genuine catch). Local judging costs hours vs cents: it is the
  periodic audit, not the per-commit gate.
- **Cross-lingual staffing case + lexical discovery channel**: the
  bench gains a09, an entirely Swedish assignment brief ("Krav" /
  "Meriterande") against the English CVs. Its first run caught a real
  matcher bug — retrieval-rank discovery let near-miss profiles crowd
  out a true match whose skill mentions were textually weak — fixed by
  a lexical channel: an exact skill phrase in a CV makes the person a
  discovery candidate regardless of retrieval rank (boundary-safe;
  verification still decides coverage). The 9-assignment bench measures
  1.0 on all gated metrics; the golden-set fingerprint gate fired on
  the bench change exactly as designed and the baseline was
  consciously re-recorded.
- **Real-CV upload in the staffing UI**: drop PDF/DOCX/MD/TXT files
  into the sidebar and the match runs against them instead of the
  bundled bench — one file per person, names derived from filenames
  ("John_Smith_CV.pdf" → John Smith), content-addressed ingest caching.
  Uploads stay on the machine; with the Ollama provider the entire
  match is local. The eval's strict `cvNN_` tagging is unchanged.
- **Paired significance testing** (`evals/compare_runs.py`): exact
  McNemar test between two `--dump-answers` runs over the same golden
  set — reports concordant/discordant counts, the exact two-sided
  p-value, and the flipped case ids (this repo reads its flips). Run
  live on the temperature pair, it formalizes Experiment 26's
  judgment: b=2/c=1, p=1.0 — noise. An instrument, never a gate.
- **Staffing match chain + dedicated UI** (ROADMAP 9.1/9.3): the
  CV↔assignment matcher (`ragstone.match`) — a LangGraph graph that
  extracts structured requirements from a free-text brief, discovers
  candidates by per-requirement retrieval over person-tagged chunks,
  verifies coverage per candidate with verbatim evidence quotes, and
  ranks by verified coverage with an honest `full_match_exists` (gaps
  read "not evidenced in the CV"). Measured on the constructed bench:
  strong_recall@5 1.0, full_match_accuracy 1.0 (incl. the deliberately
  unsatisfiable assignment), ordering_clean_rate 1.0
  (`evals/run_staffing_eval.py`, gated). Ships with a separate
  one-command staffing UI (`make run-match-ui` / `ragstone-match`):
  bundled example briefs, live progress, coverage tables, and CV
  evidence highlighting via the citations aligner.
- **Staffing-match bench** (ROADMAP 9.0): 40 synthetic consultant CVs
  (embedded automotive, telecom, cloud, DevOps) plus 8 client
  assignment briefs for the CV↔assignment matching showcase, with
  ground truth true by construction — personas are structured specs,
  expected match tiers come from a mechanical oracle, and rendered
  prose is regex-verified against the skill taxonomy (every spec skill
  mentioned, none leaked, OR-requirements phrased as alternatives).
  One assignment is deliberately unsatisfiable, so honest "no full
  match" reporting is measurable. Also serves as a contamination
  control: no model has ever seen these documents.
- **Second corpus at full power** (Experiment 24 / ROADMAP 3.0 v2):
  the regulatory golden set expanded 27 → 68 machine-audited cases
  (multi-turn ×3) and the full chain matrix re-ran with statistical
  power. One flip: fusion is the best chain on this corpus (correct
  +10.7pp, faithful 0.982 at 2.3× tokens; niche = near-duplicate or
  tiered passages). One walk-back: corrective's n=27 "enable-when
  validated" shrank to noise at n=68 — the support-tier table is
  corrected in both directions. Holds: multi_query rejected, chunk-500
  hurts, the local nomic embedding gap is real (hit 0.56 vs 0.78).

- **API versioning + RFC 7807 errors** (ROADMAP 5.9): the REST surface
  is frozen as `/v1`; unprefixed paths keep working as deprecated
  aliases marked with an RFC 8594 `Deprecation` header. Every error is
  now `application/problem+json` (type/title/status/detail/instance +
  request id); `detail` remains top-level, so existing clients keep
  parsing. Typed pipeline errors carry `urn:ragstone:problem:<Class>`
  so clients can branch without parsing prose.

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
