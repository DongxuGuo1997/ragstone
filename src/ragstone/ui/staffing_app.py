"""
Ragstone Staffing Match — the dedicated CV-matching UI (ROADMAP 9.3).

A separate app, deliberately not part of the chat UI: the staffing flow
is "paste an assignment request, read a shortlist", not a conversation.
Launch:

    make run-match-ui             # or:
    ragstone-match                # console script
    streamlit run src/ragstone/ui/staffing_app.py

The app ingests a CV directory once per (provider, model, k, dir)
combination, runs the match chain (ragstone.match), and renders the
evidence: per-candidate requirement coverage with verbatim CV quotes,
gaps phrased as "not evidenced in the CV", and the honest no-full-match
banner when nobody covers everything. With the Ollama provider no CV
text leaves the machine — the GDPR argument, live.
"""

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Allow `streamlit run src/ragstone/ui/staffing_app.py` without install
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from ragstone.config.settings import get_config  # noqa: E402
from ragstone.rag.citations import (  # noqa: E402
    find_supporting_spans,
    highlight_spans,
)
from ragstone.rag.providers import DEFAULT_MODELS  # noqa: E402

DEFAULT_CV_DIR = str(Path(__file__).resolve().parents[3] / "evals" / "corpus_staffing")
EXAMPLES_PATH = Path(__file__).resolve().parents[3] / "evals" / "golden_staffing.jsonl"
TIER_BADGES = {"strong": "🟢 strong", "partial": "🟡 partial", "weak": "⚪ weak"}
MARK_OPEN = '<mark style="background-color:#fde68a;color:inherit;">'
MARK_CLOSE = "</mark>"


def load_example_briefs(path: Path = EXAMPLES_PATH) -> Dict[str, str]:
    """Bundled demo briefs (title -> text); empty dict when absent."""
    if not path.exists():
        return {}
    briefs: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            briefs[f"{record['id']}: {record['title']}"] = record["brief"]
    return briefs


def highlight_evidence(cv_text: str, quotes: List[str]) -> str:
    """HTML of the CV with every evidence quote highlighted.

    Reuses the citations aligner (same machinery as the chat UI's source
    highlighting): spans are found per quote, merged, then wrapped.
    """
    spans: List[Tuple[int, int]] = []
    for quote in quotes:
        if quote:
            spans.extend(find_supporting_spans(quote, cv_text))
    if not spans:
        return html.escape(cv_text)
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return highlight_spans(cv_text, merged, MARK_OPEN, MARK_CLOSE, escape=html.escape)


def persist_uploads(files) -> str:
    """Write uploaded CVs into a content-addressed temp directory.

    The directory name carries a hash of every (filename, bytes) pair,
    so the cached matcher rebuilds exactly when the upload set changes
    — and two sessions uploading identical files share one ingest.
    """
    import hashlib
    import tempfile

    digest = hashlib.sha256()
    for uploaded in files:
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(uploaded.getvalue())
    target = (
        Path(tempfile.gettempdir()) / f"ragstone_cv_uploads_{digest.hexdigest()[:12]}"
    )
    target.mkdir(exist_ok=True)
    for uploaded in files:
        (target / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
    return str(target)


@st.cache_resource(show_spinner=False)
def build_matcher(
    provider: str, model: str, cv_dir: str, k: int, lenient: bool = False
):
    """Ingest the CV directory and build the match chain (once per key)."""
    from ragstone.match import MatchPipeline, stamp_person_metadata
    from ragstone.rag.pipeline import OllamaPipeline, OpenAIPipeline

    config = get_config()
    config.database.similarity_k = k
    # Cards would spend one LLM call per CV at ingest; the matcher's
    # verification stage reads full CVs anyway.
    config.loader.metadata_cards = False

    # Duck-typed across the two provider pipelines (set_retriever_* are
    # provider-specific methods), hence the Any.
    pipeline: Any = (
        OpenAIPipeline(model=model)
        if provider == "openai"
        else OllamaPipeline(model=model)
    )
    texts = pipeline.load_and_split(data_dir=cv_dir)
    if not texts:
        raise ValueError(
            f"No documents found in {cv_dir}. Point the sidebar at a "
            "directory of CV files (the bundled bench lives in "
            "evals/corpus_staffing; generate it with "
            "`python evals/generate_staffing.py`) — or upload CVs."
        )
    stamped = stamp_person_metadata(texts, lenient=lenient)
    if not stamped:
        raise ValueError(
            f"No CV files recognized in {cv_dir} — bundled-bench files "
            "must look like cv01_firstname_lastname.md; uploaded files "
            "count one-per-person automatically."
        )
    if provider == "openai":
        pipeline.set_retriever_openai(use_ensemble=True, use_reranker=False)
    else:
        pipeline.set_retriever_ollama(use_ensemble=True, use_reranker=False)
    if pipeline.get_retriever() is None:
        raise ValueError(
            "Retriever could not be created. For OpenAI: is OPENAI_API_KEY "
            "set? For Ollama: is the server running (`ollama serve`)?"
        )
    matcher = MatchPipeline.from_pipeline(pipeline)
    people = {
        doc.metadata["person_id"] for doc in texts if doc.metadata.get("person_id")
    }
    return matcher, len(people)


def _event_line(event: dict) -> Optional[str]:
    kind = event.get("event")
    if kind == "extract":
        line = "📋 Requirements: " + "; ".join(event.get("must", []))
        if event.get("location"):
            line += f" · based in {event['location']}"
        return line
    if kind == "discover":
        return f'🔎 Searching: "{event.get("query", "")}"'
    if kind == "shortlist":
        return f"👥 Verifying {len(event.get('candidates', []))} candidates"
    if kind == "verify":
        return f"⚖️ Screening {event.get('candidate', '')}"
    if kind == "result":
        return "✅ Ranking complete"
    return None


def _render_candidate(candidate, people_chunks: Dict[str, str]) -> None:
    badge = TIER_BADGES.get(candidate.tier, candidate.tier)
    with st.container(border=True):
        st.markdown(f"**{candidate.name}** — {badge}")
        rows = []
        for finding in candidate.coverage:
            rows.append(
                {
                    "requirement": finding.requirement,
                    "evidenced": "✅" if finding.covered else "❌",
                    "evidence (verbatim from CV)": finding.evidence or "—",
                }
            )
        st.table(rows)
        if candidate.missing:
            st.warning(candidate.gap_statement(), icon="⚠️")
        if candidate.nice_hits:
            st.caption("Meriting: " + ", ".join(candidate.nice_hits))
        if getattr(candidate, "location_note", ""):
            st.caption(candidate.location_note)
        cv_text = people_chunks.get(candidate.person_id, "")
        if cv_text:
            with st.expander(f"CV — {candidate.name} (evidence highlighted)"):
                quotes = [f.evidence for f in candidate.coverage if f.covered]
                st.markdown(
                    highlight_evidence(cv_text, quotes),
                    unsafe_allow_html=True,
                )


def render() -> None:
    st.set_page_config(
        page_title="Ragstone Staffing Match",
        page_icon="🎯",
        layout="wide",
    )
    st.title("🎯 Staffing match")
    st.caption(
        "Paste a client assignment request; get an evidence-backed "
        "shortlist from the consultant CV pool. Decision support with "
        "a human in the loop — every claim cites the CV, and gaps are "
        "reported as *not evidenced*, never as *cannot do*."
    )

    with st.sidebar:
        st.header("Setup")
        provider = st.radio(
            "Provider",
            ["openai", "ollama"],
            help=(
                "Ollama runs the entire match locally — no CV text "
                "leaves this machine."
            ),
        )
        model = st.text_input("Model", value=DEFAULT_MODELS[provider])
        cv_dir = st.text_input("CV directory", value=DEFAULT_CV_DIR)
        uploads = st.file_uploader(
            "…or upload real CVs (one file per person)",
            type=["pdf", "docx", "md", "txt"],
            accept_multiple_files=True,
            help=(
                "Uploaded files replace the directory above. They are "
                "written only to this machine's temp dir; with the "
                "Ollama provider nothing leaves the machine at all."
            ),
        )
        lenient = False
        if uploads:
            cv_dir = persist_uploads(uploads)
            lenient = True
            st.caption(f"Matching against {len(uploads)} uploaded CV(s).")
        k = st.slider(
            "Retrieval depth (k)",
            min_value=4,
            max_value=20,
            value=12,
            help="Chunks fetched per requirement query during discovery.",
        )
        if provider == "ollama":
            st.info("🔒 Local mode: CVs never leave this machine.")

    try:
        with st.spinner("Ingesting CVs and building the matcher…"):
            matcher, n_people = build_matcher(provider, model, cv_dir, k, lenient)
    except Exception as exc:  # surface the fix, not a traceback
        st.error(str(exc))
        st.stop()
        return
    st.sidebar.success(f"{n_people} consultants indexed")

    examples = load_example_briefs()
    brief = ""
    if examples:
        choice = st.selectbox(
            "Try a bundled assignment request…",
            ["(paste your own below)"] + list(examples),
        )
        if choice in examples:
            brief = examples[choice]
    brief = st.text_area(
        "Assignment request",
        value=brief,
        height=260,
        placeholder="Paste the client's assignment request here…",
    )

    if not st.button("Find candidates", type="primary"):
        return
    if not brief.strip():
        st.warning("Paste an assignment request first.")
        return

    result = None
    with st.status("Matching…", expanded=True) as status:
        for event in matcher.stream_events(brief):
            if event.get("event") == "done":
                result = event.get("result")
                break
            line = _event_line(event)
            if line:
                st.write(line)
        status.update(label="Match complete", state="complete")

    if result is None:
        st.error("The match run returned no result — see logs.")
        return

    if result.full_match_exists:
        st.success(result.summary, icon="✅")
    else:
        st.warning(result.summary, icon="⚠️")

    people_chunks = {
        candidate.person_id: matcher.person_cv(candidate.person_id)
        for candidate in result.candidates
    }
    top = result.shortlist(5)
    st.subheader(f"Shortlist ({len(top)} of {len(result.candidates)} assessed)")
    for candidate in top:
        _render_candidate(candidate, people_chunks)
    rest = result.candidates[5:]
    if rest:
        with st.expander(f"{len(rest)} more assessed candidates"):
            for candidate in rest:
                _render_candidate(candidate, people_chunks)


def main() -> None:
    """Console entry (`ragstone-match`): hand this file to streamlit."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())


if __name__ == "__main__":
    render()
