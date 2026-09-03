# Staffing-Match Demo Script (the manager showcase)

A ~6-minute live tour of the CV↔assignment matcher: evidence-cited
shortlists, measured honesty, real files in, and a fully local finale.
Cloud (gpt-4o-mini) carries the pace; the local run is pre-started and
revealed at the end — its measured cost on a laptop is 10–20 minutes
per assignment (Experiment 27), which is stage death live but a strong
story told right.

All timings below were rehearsed on 2026-08-21 (see the log at the
bottom). Numbers in talking points are committed measurements.

## Setup

**T−40 min — machine.** Plug into AC power (battery throttles the GPU:
the same local match measured 25.8 min on battery vs the 10–20 min AC
band). Do Not Disturb on. `ollama list` must show `qwen3.5:9b` and
`embeddinggemma`.

**T−35 min — the local runner (terminal A):**

```bash
source venv/bin/activate
RAGSTONE_OLLAMA_REASONING=off streamlit run src/ragstone/ui/staffing_app.py \
  --server.port 8502 --server.fileWatcherType none
```

Sidebar → provider **ollama** (model defaults to `qwen3.5:9b`), pick
**"a01: Senior embedded developer, truck ECU platform"**, click **Find
candidates**, leave the tab running. The env var matters: without it
the model "thinks" and blows far past 20 minutes. Started 35 minutes
out, it finishes before or during the demo — either outcome plays
(see the finale).

**T−10 min — the main stage (terminal B):**

```bash
streamlit run src/ragstone/ui/staffing_app.py \
  --server.port 8501 --server.fileWatcherType none   # the tab the audience watches
```

Sidebar: provider **openai**, model `gpt-4o-mini`, and leave the **CV
directory** field untouched — it auto-fills with the repo's
`evals/corpus_staffing` (wait for **"40 consultants indexed"**).
Pre-warm: run a01 once (~15 s cold). This warms every cache and leaves a completed
shortlist on screen — your fallback exhibit if anything breaks later.

**Two launch rules (learned in the 2026-08-27 rehearsal).**
`--server.fileWatcherType none` is not optional on either terminal:
without it Streamlit's file watcher introspects every installed module
at startup and prints a screenful of harmless `ModuleNotFoundError: No
module named 'torchvision'` tracebacks (transformers probing for image
processors) — cosmetic, but a projector-killer and easy to mistake for
a crash. And never type in the sidebar **CV directory** field: an
accidental edit shows up as `Data directory '…' not found or is not a
directory` followed by "No documents were loaded" — reload the browser
tab and the default comes back.

**T−5 min — props and terminals.** Check `~/Desktop/ragstone-demo/`
has the five DOCX CVs (regeneration command in the appendix). Keep a
third terminal (C) ready with
`python evals/run_staffing_eval.py --verbose` typed but not run.

## Act 1 — From brief to shortlist, with receipts (90 s)

Dropdown → **a01: Senior embedded developer, truck ECU platform** (a
Swedish truck OEM brief — seven requirements including AUTOSAR
Classic, ISO 26262, 5+ years, automotive domain). **Find candidates.**

- The event stream narrates the graph live: extract → discover →
  verify, one line per candidate (~11 s total).
- Shortlist lands: **Astrid Okafor** and **Anders Nielsen** on top as
  full matches.
- Open Astrid: the coverage table shows every requirement with a
  **verbatim CV quote**; open the CV expander — the quoted evidence is
  highlighted in the source.
- Talking point: *"Nothing here is a vibe. Every claim is a quote from
  the CV, and you can click through to check it — because a staffing
  decision needs receipts, not scores."*

## Act 2 — Strengths AND weaknesses (45 s)

Scroll down the same shortlist to the partial matches (Aino Lindqvist,
Maja Sjoberg…). Each carries a gap warning naming exactly what's
missing — *"Not evidenced in the CV: ISO 26262."*

- Talking point: *"The phrasing is deliberate: absence of evidence,
  not evidence of absence. The tool proposes, names the gap, and the
  staffing manager decides — decision support, never automated
  selection."*

## Act 3 — It refuses to oversell (45 s)

Dropdown → **a08: Edge platform engineer, secure telecom workloads**
— deliberately impossible (nobody in the pool combines 5G, Kubernetes,
and secure boot). **Find candidates.**

- The result says **no full match exists**; the nearest candidates all
  show the same honest gap ("secure boot").
- Talking point: *"This assignment is unsatisfiable by construction,
  and honesty here is a gated metric — if a code change ever makes the
  system claim a full match on this brief, the build fails. We measure
  the temptation to oversell."*

**Stage cue:** switch to terminal C and hit Enter on the eval now; it
runs ~100 s in the background and will be done for the close.

## Act 4 — Swedish in, same rigor (30 s)

Dropdown → **a09: Inbyggd mjukvaruutvecklare, hyttelektronik**. **Find
candidates.**

- A Swedish-language brief, same structured extraction ("Svenska
  (working proficiency)" becomes a requirement), five full matches.
- Talking point: *"Brief language and CV language are independent —
  requirements are structured, not string-matched."*

## Act 5 — Real files in (60 s)

Drag all five DOCX files from `~/Desktop/ragstone-demo/` into the
uploader (say it plainly: uploads replace the bundled pool — this is
"match against your own files"). Re-select **a01**. **Find
candidates.**

- Ingest is ~1–3 s; the match lands in ~10 s as a clean ladder:
  **Henrik Bakke** and **Ida Sandberg** fully evidenced, **Ingrid
  Eriksson** (no MISRA C) and **Magnus Chen** (no ISO 26262) partial
  with distinct named gaps, **Sofia Dahl** weak.
- Talking point: *"Word documents straight from disk — parsed,
  person-tagged from the filename, cited like everything else. Your
  real CV database is an ingestion path, not a rewrite."*

## Act 6 — Finale: it never left the laptop (60 s)

Switch to the **port 8502 tab**: the same a01 brief, matched by
`qwen3.5:9b` entirely on this machine — it has been running since
before the meeting.

- If finished: the same two names on top, same evidence discipline.
  If still verifying: the live event stream is its own show — *"it is
  verifying a candidate on this laptop's GPU right now"* — and the
  measured parity numbers carry the claim.
- Talking points: *"CVs are personal data under the GDPR. With the
  local provider the entire pipeline — embeddings, retrieval,
  verification — runs on this machine, under a profile that refuses
  network egress by construction. And we measured it: the local stack
  scores identical 1.0s on every gated metric. The honest cost is
  time — 10–20 minutes on a laptop today. That's a hardware knob, not
  a quality gap."*

## The close — measured, not vibes (30 s)

Terminal C has finished by now. Scroll the per-assignment lines, land
on the scores:

```
strong_recall_at_5:   1.000   full_match_accuracy: 1.000
ordering_clean_rate:  1.000   (~10 s per assignment)
```

- Talking point: *"The bench's ground truth is true by construction —
  CVs are rendered from persona specs, so we know every right answer.
  Eleven assignments — two of them long-form RFQs in the shape real
  requests arrive in — three gated metrics, all at 1.0. And at 10× the
  pool — 400 consultants — top-5 precision holds at 0.925, with both
  imperfections attributed, not hidden. Every claim in this demo is a
  committed measurement in the repo."*

## If something breaks

- **OpenAI hiccup mid-act**: the pre-warmed a01 shortlist from setup
  is still on screen — narrate from it; retry the click once. Don't
  switch the main tab to ollama live (10–20 min pacing).
- **Local tab crashed / not started**: tell the story with numbers —
  the parity run is committed (identical 1.0s, Experiment 27) — and
  show the enforced no-egress profile in `.env.example`.
- **No network at all**: the local tab still works, and the rehearsal
  eval printout in the repo history backs every number.
- **"Data directory … not found" / "No documents were loaded"**: the
  CV-directory field got edited. Reload the tab; don't retype the path
  live.
- **Wall of `torchvision` tracebacks in a terminal**: harmless watcher
  noise from a launch without `--server.fileWatcherType none`. The app
  still works — ignore it, or relaunch with the flag.

## Q&A ammunition

- **"Can it run on our real CV database?"** Ingestion takes
  PDF/DOCX/MD/TXT with automatic person tagging (Act 5 was literal
  Word files). Scale is measured: 400 consultants, 0.925 top-5
  precision, ~10 s per assignment.
- **"What does it cost?"** Cloud: a few cents per matched assignment
  (one extraction + up to ten verification calls on gpt-4o-mini).
  Local: zero marginal cost, 10–20 laptop-minutes; server-class local
  hardware brings that to minutes.
- **"How do we know it's right?"** Ground truth by construction, three
  gated metrics, honesty itself gated (Act 3), CI fails on regression,
  and the scale test's two misses are root-caused in EXPERIMENTS.md
  (Experiment 29).
- **"GDPR?"** Local mode keeps CVs on the machine under an enforced
  no-egress profile, at measured quality parity. The repo ships only
  synthetic CVs. Framing is human-in-the-loop decision support — a
  shortlist with cited evidence and named gaps, never automated
  selection.
- **"What would production take?"** The engine already has a REST API
  and an MCP server; the matcher is the same pipeline. The named next
  quality lever is verifier second-vote screening (Experiment 29);
  after that, an ATS/CV-store connector and auth.

## Appendix

**Regenerate the demo props** (five XL-bench personas — two a01-strong,
two partial with distinct gaps, one out-of-domain — as DOCX):
markdown → HTML → `textutil -convert docx`; the XL sources are
`evals/corpus_staffing_xl/cv371_henrik_bakke.md`, `cv041_ida_sandberg`,
`cv039_ingrid_eriksson`, `cv007_magnus_chen`, `cv001_sofia_dahl`.
Name files `Firstname_Lastname_CV.docx` — the lenient tagger derives
the person from the filename. Marta Rahman (cv377) was deliberately
dropped: her one gap (CAN bus) sat in the verifier-leniency class
Experiment 29 attributed, so her live verdict contradicted the oracle.
Since Experiment 30 (part 4) the second vote catches exactly that
class; she can return to the props after one rehearsal confirms it.

**Rehearsal log (2026-08-21, M4 Max).** Cloud a01 match 10.9 s; full
9-assignment eval 1:42, gates 1.0/1.0/1.0. Upload act: 5-DOCX ingest
0.8 s warm, match 7.6 s, ladder exactly as scripted. Local a01
(qwen3.5:9b, reasoning off, **on battery**): 25.8 min,
quality-identical (both strong found, every gap correct,
gap_alignment 1.0) — hence the pre-started staging and the AC
requirement.

**Rehearsal log (2026-08-27, presenter's own run).** A launch without
the watcher flag produced the torchvision traceback flood, and an
accidental edit of the CV-directory field produced `Data directory …
not found`. Both are now covered by the launch rules in Setup.
