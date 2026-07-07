# Contributing to Ragstone

Thanks for your interest. This project holds itself to two rules that
shape every contribution, so they come first:

1. **Every quality-affecting change is eval-gated.** If your change can
   alter what gets retrieved or generated, it needs a measurement — the
   smoke set for regressions, the large set for design verdicts. "It
   should be better" is a hypothesis, not a result. See
   [EXPERIMENTS.md](EXPERIMENTS.md) for fifteen worked examples,
   including the ones where we were wrong.
2. **Never change the golden set and the system under test in the same
   measured comparison.** If your PR edits eval cases, re-record
   baselines in that same PR and say why; don't mix it with behavior
   changes you're trying to measure.

## Getting set up

```bash
git clone https://github.com/DongxuGuo1997/ragstone.git
cd ragstone
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,sqlite,api,qdrant]"   # pgvector needs a server
pytest tests/ -q                             # ~270 tests, no network
```

Copy `.env.example` to `.env` for anything that talks to OpenAI (the
paid eval layers, the live demo). The unit suite never needs a key.

## Before you open a PR

All four must be green — CI runs the same set:

```bash
black src/ tests/ && isort src/ tests/
flake8 src/ tests/ evals/
mypy src/ragstone          # zero suppressions is the standard
pytest tests/ -q
```

If your change touches retrieval or generation behavior:

```bash
python evals/run_eval.py --mode retrieval    # free-ish; gates hit/mrr
python evals/run_eval.py                     # smoke full, ~5 cents
```

A metric dropping more than 0.05 below `evals/baseline.json` fails the
run. If the drop is intended (you changed a default for measured
reasons), re-record with `--update-baseline` in the same PR and link the
measurement.

## Code standards (the ones reviewers actually check)

- **Tests assert behavior, not lines.** Prefer counting fakes (how many
  LLM calls happened), event-sequence assertions, and parity checks over
  smoke assertions. Look at `tests/unit/test_corrective_chain.py` or
  `test_vector_db_servers.py` for the house style.
- **Comments say why, not what.** The codebase's best comments carry the
  constraint or the measurement that justifies the line (see the
  `similarity_k` comment in `config/settings.py`). Match that.
- **Fail loud for correctness, degrade for capability.** A silently
  wrong index is worse than a crash; a missing optional feature should
  fall back with a warning. Know which kind your failure is.
- **Exact-match only for anything cache-like.** Fuzzy matching near
  correctness has been removed from this codebase twice; don't bring it
  back without a measurement.
- Every exception class must have a raise site; typed `PipelineError`
  messages are user-safe by contract (both servers surface them).

## Adding common things

**A chain type**: factory method on `RagProxy` (lazy-import if your
module imports from `rag.py`) → `match` arm in
`FullChain.create_full_chain` → API `Literal` → MCP `known_chain_types`
→ CLI `CHAIN_TYPES` → eval `--chain-type` choices → tests modeled on the
corrective suite → one large-set run before claiming anything.

**A vector-store backend**: subclass `VectorStoreProxy` in
`vector_db_servers.py`; `create_db` must use `embed_texts_cached` and
replace (never append) on rebuild; `cleanup` releases clients and drops
unpinned collections. Add the optional extra, offline tests with
deterministic embeddings, and the FAISS parity test — identical rankings
are the acceptance bar (Experiment 13).

**Golden cases**: needles must be verbatim substrings of the gold-source
document (`generate_cases.py::validate` enforces); unanswerable cases
need a manual corpus-absence check — one that is actually answerable
scores correct answers as failures.

## Scope notes

Docker is an optional deployment path; nothing in development or testing
may require it. New runtime dependencies need a strong case — much of
this project's identity is what it does *without* adding them.

## Reporting issues

Open a GitHub issue. For anything security-shaped, see
[SECURITY.md](SECURITY.md) — this is a demonstration project with no
embargo process.
