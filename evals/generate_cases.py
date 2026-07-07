#!/usr/bin/env python3
"""Generate the LARGE golden set: extended fictional corpus + ~200 cases.

Three integrity rules, enforced here rather than trusted:

1. The corpus stays FICTIONAL. New documents extend the invented universe
   (missions, products, companies) so no model can answer from
   pretraining. Never real-world material.
2. Needles are verified PROGRAMMATICALLY. Every case's `must_contain`
   strings must appear verbatim (case-insensitive) in the claimed source
   document, or the case is rejected. The retrieval layer is therefore
   judge-free and immune to generator bias.
3. Distractors are ENGINEERED. New entities deliberately mirror existing
   ones (a third solar panel, a sibling mission, a rival company) so the
   large set is harder, not just bigger.

Generated artifacts (new corpus docs + golden_large.jsonl) are committed
to git and reviewable like any hand-written golden set. Known limitation,
stated openly: generator and judge share a model family, so generation
bias cannot be fully excluded — the programmatic needle check bounds it
for retrieval metrics, not for judged ones.

Usage (from the repository root, costs ~$0.10 one-time):

    python evals/generate_cases.py            # docs if missing, then cases
    python evals/generate_cases.py --docs-only
"""

import argparse
import json
import re
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).parent
CORPUS_DIR = EVALS_DIR / "corpus"  # original docs (smoke gate — untouched)
EXTENDED_DIR = EVALS_DIR / "corpus_extended"  # generated docs (large set)
SMOKE_PATH = EVALS_DIR / "golden.jsonl"
LARGE_PATH = EVALS_DIR / "golden_large.jsonl"

sys.path.insert(0, str(EVALS_DIR.parent / "src"))

# New fictional documents: (filename, topic brief, distractor engineering)
NEW_DOCS = [
    (
        "corona_solar_guide.md",
        "Owner's maintenance guide for the fictional Corona K-7 solar panel",
        "Give it cleaning intervals, bolt torques, warranty length, and fault "
        "codes that are plausibly confusable with — but different from — the "
        "Helios MK-3 (28-year warranty, 18 Nm bolts, clean every 6 months, "
        "code E-42) and Borealis BX-2 (14 Nm bolts, clean every 12 months).",
    ),
    (
        "helios_mk4_announcement.md",
        "Product announcement for the fictional Helios MK-4 panel",
        "Same brand family as the Helios MK-3 but different numbers: new "
        "wattage, efficiency, warranty, and weight. Same-brand confusion is "
        "the trap.",
    ),
    (
        "aurora11_mission.md",
        "Overview of the fictional crewed Aurora-11 Venus-flyby mission",
        "Sibling of Aurora-7 (launched 2031, 4.2 N thrust, 224-day journey, "
        "commander Dr. Ingrid Halvorsen) and Aurora-9 (2034, 7.6 N, robotic). "
        "Different launch year, thrust, crew, and commander.",
    ),
    (
        "woomera_spaceport.md",
        "Facilities guide for the fictional Woomera spaceflight complex",
        "Mention that Aurora missions launch here; add pad names, a crew "
        "quarantine duration, and visitor rules with specific numbers.",
    ),
    (
        "ember_roastworks.md",
        "Company profile of fictional Ember Roastworks, rival of Solstice "
        "Coffee Roasters",
        "Different CEO name, founding year, and store count than Solstice — "
        "rival-company confusion is the trap.",
    ),
    (
        "meridian_green_line.md",
        "Plan for the fictional Meridian Transit Green Line extension",
        "The existing corpus has a Violet Line with fares and opening dates. "
        "Give the Green Line different fares, stations, and opening dates.",
    ),
    (
        "suncore_x10_manual.md",
        "Service manual for the fictional SunCore X10 inverter",
        "Successor of the SunCore X9 (fault codes E-42 overvoltage, E-17). "
        "New fault codes and thresholds that could be confused with the X9's.",
    ),
    (
        "tanami_facility.md",
        "Procedures document for the fictional Tanami Curation Facility",
        "It stores Aurora-7 samples (850 grams from Site Juniper). Add "
        "storage temperatures, staff counts, and access rules with numbers.",
    ),
    (
        "vela_drive_bulletin.md",
        "Maintenance bulletin for the fictional Vela Drive ion engine",
        "Used by Aurora-7 (4.2 N) and Aurora-9 (7.6 N twin cluster). Add "
        "service intervals, xenon consumption rates, and part numbers.",
    ),
    (
        "meridian_fare_handbook.md",
        "Fare policy handbook for the fictional Meridian Transit system",
        "Concession rules, day-cap amounts, and fine amounts distinct from "
        "the single-trip fares mentioned in the existing transit document.",
    ),
]

DOC_PROMPT = """You are extending a FICTIONAL test corpus for evaluating a
retrieval system. Style example from the existing corpus:

---
{example}
---

Write a NEW document: {topic}.

Requirements:
- Entirely fictional; invent all names, numbers, and specifications.
- 400-600 words of markdown with a # title and 3-5 ## sections.
- Dense with SPECIFIC facts: numbers, dates, names, model codes.
- Distractor engineering: {distractors}
- Do not contradict these established facts: Aurora-7 launched March 14,
  2031, commanded by Dr. Ingrid Halvorsen, 4.2 newtons thrust, 224-day
  journey, 850 grams collected; Aurora-9 launched June 2, 2034, 7.6
  newtons; Helios MK-3: 460 W, 28-year warranty, 18 Nm bolts, clean every
  6 months; Borealis BX-2: 14 Nm bolts, clean every 12 months.

Return ONLY the markdown document."""

CASES_PROMPT = """You are writing evaluation cases for a retrieval QA system.

Document ({filename}):
---
{document}
---

Generate exactly {n_factual} "factual" and {n_paraphrase} "paraphrase"
cases as a JSON array. Each case:
{{"question": ..., "gold_answer": ..., "must_contain": [...], "category": ...}}

Rules:
- "factual": a direct question answerable from ONE specific fact here.
- "paraphrase": asks about a fact using completely different wording than
  the document (synonyms, reordered phrasing) — never reuse the document's
  key phrasing in the question.
- must_contain: 1-2 SHORT strings copied VERBATIM from the document that
  the answering chunk must contain (numbers, names, codes). Exact
  substrings — they are checked mechanically.
- gold_answer: one short sentence that is SPECIFIC — name the exact
  entities, numbers, or model codes from the document. A vague gold
  answer ("a compatible inverter system") fails correct system answers.
Return ONLY the JSON array."""

CROSS_DOC_PROMPT = """You are writing evaluation cases for a retrieval QA
system whose corpus contains these fictional documents:

{summaries}

Generate a JSON array with exactly:
- {n_distractor} "distractor" cases: the question targets a fact in ONE
  document while a confusingly similar fact exists in another (e.g. the
  other panel's torque, the sibling mission's thrust). Set "gold_source"
  to the correct document's filename; must_contain = verbatim strings
  from that document.
- {n_multihop} "multi_hop" cases: answering requires combining facts from
  TWO documents (e.g. comparing two products' warranties). must_contain =
  one verbatim string from EACH document involved.
- {n_unanswerable} "unanswerable" cases: plausible questions about these
  entities whose answers appear in NO document (e.g. a person's age, a
  product's price when none is stated). gold_answer = null,
  must_contain = [].
- {n_multiturn} "multi_turn" cases: {{"turns": [q1, q2], ...}} where q1
  asks about an entity and q2 is a follow-up using a pronoun or vague
  reference ("its", "her", "the panels") that is unresolvable without q1.
  The case's "question" field = q2; gold_answer answers q2; must_contain
  = verbatim strings from the gold_source document.

Every case: {{"question": ..., "gold_answer": ..., "must_contain": [...],
"category": ..., "gold_source": "<filename>"}} (plus "turns" for
multi_turn). Return ONLY the JSON array."""


def _llm(model: str):
    from dotenv import load_dotenv

    load_dotenv()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0.3)


def _parse_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in response: {text[:200]}")
    return json.loads(match.group(0))


def generate_docs(llm) -> None:
    EXTENDED_DIR.mkdir(exist_ok=True)
    example = (CORPUS_DIR / "helios_solar_guide.md").read_text(encoding="utf-8")
    for filename, topic, distractors in NEW_DOCS:
        path = EXTENDED_DIR / filename
        if path.exists():
            print(f"  corpus: {filename} exists, skipping")
            continue
        prompt = DOC_PROMPT.format(
            example=example[:2500], topic=topic, distractors=distractors
        )
        doc = llm.invoke(prompt).content
        # Models sometimes wrap output in a ``` fence despite instructions;
        # fence lines are noise in retrieval chunks, so drop them.
        doc = "\n".join(
            line for line in doc.splitlines() if not line.strip().startswith("```")
        )
        path.write_text(doc.strip() + "\n", encoding="utf-8")
        print(f"  corpus: wrote {filename} ({len(doc.split())} words)")


def _corpus_paths() -> list:
    return sorted(CORPUS_DIR.glob("*.md")) + sorted(EXTENDED_DIR.glob("*.md"))


def _corpus_texts() -> dict:
    return {p.name: p.read_text(encoding="utf-8").lower() for p in _corpus_paths()}


def validate(case: dict, corpus: dict) -> str | None:
    """Return a rejection reason, or None if the case is sound."""
    category = case.get("category")
    needles = [n.lower() for n in case.get("must_contain", [])]
    source = case.get("gold_source")

    if category == "unanswerable":
        if needles or case.get("gold_answer"):
            return "unanswerable case must have no needles and null answer"
        return None
    if not needles:
        return "no needles"
    if category == "multi_hop":
        for needle in needles:
            if not any(needle in text for text in corpus.values()):
                return f"multi_hop needle not in any doc: {needle!r}"
        return None
    if source not in corpus:
        return f"unknown gold_source: {source!r}"
    for needle in needles:
        if needle not in corpus[source]:
            return f"needle not in {source}: {needle!r}"
    if category == "multi_turn" and len(case.get("turns", [])) < 2:
        return "multi_turn case needs >= 2 turns"
    return None


def generate_cases(llm) -> list:
    corpus = _corpus_texts()
    cases, rejected = [], []

    # Per-document factual + paraphrase cases.
    paths = {p.name: p for p in _corpus_paths()}
    for filename in sorted(corpus):
        raw = paths[filename].read_text(encoding="utf-8")
        response = llm.invoke(
            CASES_PROMPT.format(
                filename=filename, document=raw, n_factual=7, n_paraphrase=3
            )
        ).content
        for case in _parse_json_array(response):
            case["gold_source"] = filename
            reason = validate(case, corpus)
            (cases if reason is None else rejected).append(
                case if reason is None else (case, reason)
            )
        print(f"  cases: {filename} -> {len(cases)} accepted so far")

    # Cross-document distractor / multi-hop / unanswerable / multi-turn.
    summaries = "\n".join(
        f"- {name}: {text[:300]}..." for name, text in _corpus_texts().items()
    )
    response = llm.invoke(
        CROSS_DOC_PROMPT.format(
            summaries=summaries,
            n_distractor=12,
            n_multihop=8,
            n_unanswerable=8,
            n_multiturn=10,
        )
    ).content
    for case in _parse_json_array(response):
        reason = validate(case, corpus)
        (cases if reason is None else rejected).append(
            case if reason is None else (case, reason)
        )

    for case, reason in rejected:
        print(f"  REJECTED [{case.get('category')}]: {reason}")
    print(f"  validation: {len(cases)} accepted, {len(rejected)} rejected")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--docs-only", action="store_true")
    args = parser.parse_args()

    llm = _llm(args.model)
    print("Extending fictional corpus...")
    generate_docs(llm)
    if args.docs_only:
        return 0

    print("Generating cases...")
    generated = generate_cases(llm)

    # The large set = curated smoke cases + validated generated cases.
    with open(SMOKE_PATH, encoding="utf-8") as f:
        smoke = [json.loads(line) for line in f if line.strip()]
    seen = {c["question"].strip().lower() for c in smoke}
    merged = list(smoke)
    for i, case in enumerate(generated):
        if case["question"].strip().lower() in seen:
            continue
        seen.add(case["question"].strip().lower())
        prefix = "gmt" if case.get("category") == "multi_turn" else "g"
        case["id"] = f"{prefix}{i:03d}"
        merged.append(case)

    with open(LARGE_PATH, "w", encoding="utf-8") as f:
        for case in merged:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    from collections import Counter

    counts = Counter(c["category"] for c in merged)
    print(f"\nWrote {len(merged)} cases to {LARGE_PATH.name}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
