# Staffing-Match Demo Script

A five-minute tour of the CV↔assignment matcher on the bundled bench:
evidence-cited shortlists, measured honesty, your own files in, and a
fully local run at the end. Every number in the talking points is a
committed measurement (Experiments 27, 29 and 30 in
[EXPERIMENTS.md](../../EXPERIMENTS.md)); the mechanism is in
[CV_MATCHING.md](../CV_MATCHING.md).

## Setup

```bash
source venv/bin/activate
streamlit run src/ragstone/ui/staffing_app.py --server.fileWatcherType none
```

In the sidebar pick provider **openai** and model `gpt-4o-mini`, and
leave the **CV directory** field alone: it fills in with the repo's
`evals/corpus_staffing` and the badge reads **40 consultants indexed**.
Run one match before the audience arrives so every cache is warm and a
finished shortlist is already on screen.

Two launch tips. `--server.fileWatcherType none` keeps Streamlit's
file watcher from printing a harmless wall of `torchvision` tracebacks
at startup. And if the CV-directory field ever gets edited, the app
reports "No documents were loaded"; reload the browser tab and the
default comes back.

For the local finale, also `ollama pull qwen3.5:9b` and
`ollama pull embeddinggemma`.

## 1. From brief to shortlist, with receipts

Choose **a01: Senior embedded developer, truck ECU platform** and click
**Find candidates**. The event stream narrates the graph live: extract,
discover, then one verify line per candidate, about ten seconds in
all. Two full matches land on top. Open one: the coverage table shows
every requirement with a verbatim CV quote, and the CV expander
highlights each quote in the source.

*Nothing here is a score. Every claim is a quote from the CV, and you
can click through to check it.*

## 2. Strengths and weaknesses

Scroll to the partial matches. Each carries a gap that names exactly
what is missing: "Not evidenced in the CV: ISO 26262."

*Absence of evidence, not evidence of absence. The tool proposes and
names the gap; the staffing manager decides.*

## 3. It refuses to oversell

Choose **a08: Edge platform engineer, secure telecom workloads**. This
brief is unsatisfiable by construction: nobody in the pool combines 5G,
Kubernetes and secure boot. The result says no full match exists, and
the nearest candidates all show the same honest gap.

*Honesty is a gated metric. If a change ever makes the system claim a
full match on this brief, the build fails.*

## 4. Swedish in, same rigor

Choose **a09: Inbyggd mjukvaruutvecklare, hyttelektronik**. A
Swedish-language brief against English CVs gets the same structured
extraction, including a spoken-language requirement, and the same
citations.

## 5. Your own files

Drop a few PDF or DOCX CVs into the sidebar uploader, one file per
person, named `Firstname_Lastname_CV.docx` so the person is read from
the filename. Uploads replace the bundled pool. Re-run a01. Ingest
takes a second or two; the match lands in about ten.

To make props from the bench, convert a few files from
`evals/corpus_staffing_xl/` (markdown → HTML → `textutil -convert docx`
on macOS, or any converter). Pick two personas strong on a01, two
partial with different gaps, one from another domain, and the shortlist
tells a clean story.

## 6. Finale: it never left the machine

Switch the provider to **ollama** (the model defaults to `qwen3.5:9b`)
and run a01 again with thinking off:

```bash
RAGSTONE_OLLAMA_REASONING=off streamlit run src/ragstone/ui/staffing_app.py \
  --server.port 8502 --server.fileWatcherType none
```

On an Apple-silicon laptop the match completes in about a minute and a
half and puts the same two names on top with the same evidence. If the
room cannot wait, start this tab before the session and reveal it here.

*CVs are personal data. With the local provider the entire pipeline
runs on this machine under a profile that refuses network egress, and
the local stack scores the same 1.0 on every gated metric. The cost is
time, and time is a hardware knob.*

## The close: measured, not vibes

```bash
python evals/run_staffing_eval.py --verbose      # about 100 s
```

```
strong_recall_at_5:   1.000   full_match_accuracy: 1.000
ordering_clean_rate:  1.000
```

*Ground truth is true by construction, so every right answer is known.
Eleven briefs, two of them long-form RFQs, three gated metrics at 1.0.
At ten times the pool, 400 consultants, top-5 precision holds at 0.95
and every miss is root-caused in the experiment log.*

## If something breaks

- **Cloud hiccup**: the warm shortlist from setup is still on screen;
  narrate from it and retry once.
- **`httpx.ReadTimeout` on the local run**: a stock Ollama serves one
  request at a time and the matcher already verifies one candidate at
  a time there. On a slow machine raise `RAGSTONE_LLM_TIMEOUT`; do not
  raise `RAGSTONE_MATCH_VERIFY_WORKERS` against a stock server.
- **No network at all**: the local tab still works.

## Questions this usually raises

- **Can it run on a real CV database?** Ingestion takes PDF, DOCX,
  Markdown and text with person tagging from the filename; step 5 is
  literally Word files. Scale is measured at 400 consultants.
- **What does it cost?** Cloud: a few cents per brief. Local: nothing
  per match, about ninety laptop-seconds.
- **How do we know it is right?** Ground truth by construction, gated
  metrics including honesty, CI fails on regression, and the misses at
  scale are root-caused rather than hidden.
- **GDPR?** Local mode keeps CVs on the machine under an enforced
  no-egress profile at measured parity. The repo ships only synthetic
  CVs, and the tool is human-in-the-loop decision support, never
  automated selection.
