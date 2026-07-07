# Demo Script (live demos and the README GIF)

A 3-minute tour that shows the machinery, not just a chatbot. Works as a
live walkthrough for an audience or as the storyboard for a recorded GIF.

## Setup (before the audience arrives)

```bash
source venv/bin/activate
export RAGSTONE_REPHRASE_MODEL=gpt-4.1-nano   # faster follow-ups
streamlit run src/ragstone/ui/streamlit_app.py
```

In the sidebar: Online mode, `gpt-4o-mini`, load the bundled eval corpus
by pointing the data directory at `evals/corpus` (6 fictional documents:
two space missions, two solar-panel manuals, a transit system, a coffee
company — chosen so questions have plausible *wrong* answers to retrieve).

## Act 1 — A grounded answer, opened up (45 s)

Ask: **"What does inverter fault code E-42 mean?"**

- Answer streams in.
- Open **"🔍 How this answer was made"**: sources with snippets, and the
  badge — `⚡ first token · ⏱ total · 🔤 tokens · 💰 cost · ⛓ chain`.
- Talking point: *"Every answer carries its own receipt — sources, cost,
  and where the time went."*

## Act 2 — Memory that shows its work (30 s)

Ask: **"What torque do the Borealis BX-2 bolts need?"**
then follow up: **"And how often should the panels be cleaned?"**

- The glass-box panel shows **"Interpreted your question as: …Borealis
  BX-2 panels…"**.
- Talking point: *"The follow-up said just 'the panels' — two manuals in
  the corpus have cleaning schedules. The rephrase step resolved it to the
  right product. Our multi-turn eval caught this exact failure before a
  user ever could, and now gates it in CI."*

## Act 3 — Watch an agent think (30 s)

Sidebar → chain type **agent** (rebuild). Ask:
**"Which mission has more engine thrust, Aurora-7 or Aurora-9?"**

- `🔍 Searching: "…"` status lines appear live before the answer streams.
- Talking point: *"Same corpus, but now the model drives retrieval — you
  can see each search it chooses to run."*

## Act 4 — The verdict, measured (45 s)

Switch to the **⚔️ Compare** view. Left `simple`, right `agent`. Ask:
**"How long is the Helios MK-3 warranty?"**

- Two columns stream side by side; badges land; the verdict line prints
  ("simple answered N× faster…").
- Talking point: *"This is the whole thesis: don't argue about agents,
  measure them. On this corpus the agent buys nothing and costs double —
  so the simple pipeline is the default. On your corpus? Run it and see."*

## Extended cut (when you have ten minutes, not three)

### Act 5 — Self-correction refuses honestly (45 s)

Sidebar → chain type **corrective** (rebuild). Ask something the corpus
cannot answer: **"What is the price of the Helios MK-3?"**

- Watch the status lines: `📥 Retrieving…` → `⚖️ Graded results:
  irrelevant ✗` → `✏️ Rewriting query…` → a refusal that *cites what it
  searched for* instead of hallucinating a price.
- Talking point: *"The grade-and-retry loop spends zero answer tokens on
  a doomed question. And when we measured it at n=224, self-correction's
  edge was faithfulness, not correctness — so it's an option, not the
  default. The router (chain type `auto`) makes that call per question;
  its own cost gate kept it opt-in. Every one of those sentences is a
  committed measurement."*

### Act 6 — The same engine in a terminal (30 s)

```bash
ragstone-chat --provider openai --model gpt-4o-mini --data-dir evals/corpus
```

Ask a question, then `/sources`, then `/compare corrective <question>`.

- Talking point: *"Same pipeline, same glass-box trace, zero extra
  dependencies — the streaming events and metrics are side channels, so
  every interface gets them for free."*

### Act 7 — The closer: the repo charts its own history (20 s)

Show the README's quality chart (or run
`python evals/quality_history.py`).

- Talking point: *"Baselines are committed with every quality change, so
  this chart is generated from git history — the embedding upgrade, the
  enrichment win, nothing hand-typed. That's the whole engineering
  culture in one image."*

### If they ask "does it deploy?"

```bash
OPENAI_API_KEY=... docker compose up   # API + Qdrant server + Postgres/pgvector
```

Same code path as the laptop demo — `VECTOR_STORE_TYPE` is the whole
switch, and retrieval parity across backends is enforced by test
(Experiment 13).

## Recording the GIF (macOS)

- [Kap](https://getkap.co/) (free) or QuickTime + [gifski](https://gif.ski/).
- Record at 1280×800; target ≤ 30 s for the README (Acts 3 + 4 are the
  most legible on mute); export ≤ 10 MB.
- Save to `docs/assets/demo.gif` and embed near the top of the README:
  `![Ragstone demo](docs/assets/demo.gif)`
