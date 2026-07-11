"""Contextual chunk enrichment (ROADMAP 1.1).

A chunk embedded in isolation loses its document identity — "clean every
4 months with a microfiber cloth" embeds almost identically whether it
came from the Helios or the Corona manual, which is exactly the confusion
the large-set distractor cases measure. Prepending a context line puts
that identity into the text that gets embedded, BM25-indexed, and read by
the generator.

Two modes (RAGSTONE_CHUNK_CONTEXT):
- "source": prepend the document title/filename. Deterministic, free.
- "llm":    prepend a one-line, LLM-written situating sentence (the
            "contextual retrieval" approach) — one utility-model call per
            chunk at ingest time, generated concurrently.

The identity prefix exists to tell documents apart, so a SINGLE-document
corpus gets no prefix: there is nothing to disambiguate, and the prefix
is actively harmful there (Experiment 22, found live) — it dominates the
embeddings of content-empty chunks (contributor lists and TOCs become
the nearest neighbors of any query naming the document) and puts the
document's name in every chunk, zeroing its BM25 IDF.

Enrichment happens once at ingest, after splitting and before indexing,
so both retrieval paths and the corpus fingerprint see the enriched text.
A failed LLM call falls back to source mode for that chunk — enrichment
must never break ingestion.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

METADATA_CARD_PROMPT = """You are indexing a document for search. Here is
the beginning of the document (file name: {source}):

{document_head}

Extract the document's own metadata into exactly this format, one line
per field:

Title: <the document's title>
Authors: <the people or organization who WROTE THIS document>
Date: <publication or writing date>
Type: <e.g. research paper, manual, report, article>

Rules: copy names and titles VERBATIM from the text. Only report authors
of THIS document — never names that appear in citations, references, or
related-work mentions. If a field is not stated in this excerpt, write
exactly: not stated. Reply with ONLY the four lines."""

CONTEXT_PROMPT = """You are indexing a document for search. Here is the
full document:

{document}

Here is one chunk from it:

{chunk}

Write ONE short sentence situating this chunk within the document (which
document/product/entity it belongs to and what it covers), so a search
engine can tell it apart from similar chunks in other documents. Reply
with ONLY the sentence."""

# Ingest-time concurrency for llm mode; embedding uses its own pool.
_LLM_WORKERS = 8

# How much of the parent document the llm mode shows the model: the lead
# of a document names its subject, which is what disambiguation needs.
_DOC_CONTEXT_CHARS = 4000


def _source_label(doc: Document) -> str:
    """Human-readable document identity from metadata."""
    title = doc.metadata.get("title")
    if title:
        return str(title)
    source = str(doc.metadata.get("source", "unknown"))
    return Path(source).name if "://" not in source else source


def _prefixed(doc: Document, context_line: str) -> Document:
    return Document(
        page_content=f"[{context_line}]\n{doc.page_content}",
        metadata=dict(doc.metadata),
    )


def _single_source(chunks: List[Document]) -> bool:
    """True when every chunk comes from the same source document.

    Counted on the ``source`` metadata key (the same identity the loaders
    stamp and metadata cards key on), not the display label — two files
    that happen to share a title are still two documents.
    """
    return len({str(c.metadata.get("source", "")) for c in chunks}) <= 1


def enrich_chunks(
    chunks: List[Document],
    source_docs: Optional[List[Document]] = None,
    mode: str = "off",
    llm: Optional[Any] = None,
) -> List[Document]:
    """Return chunks with a context line prepended, per the chosen mode.

    Args:
        chunks: Split documents about to be indexed.
        source_docs: The unsplit documents (needed for "llm" mode to show
            the model the full document; ignored for "source" mode).
        mode: "off" (identity), "source", or "llm".
        llm: Chat model for "llm" mode; without one, falls back to
            "source" mode with a warning.

    Returns:
        A new list of Documents; metadata is preserved untouched.
    """
    if mode == "off" or not chunks:
        return chunks

    # One source document -> identity prefixes are skipped everywhere in
    # this function: nothing to disambiguate, measured harm (module
    # docstring; the single_doc eval slice pins the numbers). LLM-written
    # situating lines still run — they carry content, not just identity.
    single = _single_source(chunks)

    if mode == "source":
        if single:
            logger.info(
                "Chunk context: single-document corpus — identity prefix "
                "skipped (nothing to disambiguate)."
            )
            return chunks
        return [_prefixed(c, f"Source document: {_source_label(c)}") for c in chunks]

    if mode != "llm":
        raise ValueError(f"Unknown chunk-context mode: {mode!r}")

    if llm is None:
        logger.warning(
            "chunk_context='llm' but no LLM available at ingest; "
            "falling back to source mode."
        )
        return enrich_chunks(chunks, source_docs, mode="source")

    # Full-document text by source, so each chunk's call can show the
    # model where the chunk came from (truncated: the lead of a document
    # names its subject, which is what disambiguation needs).
    doc_text: Dict[str, str] = {}
    for doc in source_docs or []:
        key = str(doc.metadata.get("source", ""))
        doc_text.setdefault(key, doc.page_content[:_DOC_CONTEXT_CHARS])

    chain = ChatPromptTemplate.from_template(CONTEXT_PROMPT) | llm | StrOutputParser()

    def _source_fallback(chunk: Document) -> Document:
        # Same single-document rule as source mode: a failed or impossible
        # LLM line must not degrade into exactly the prefix that harms.
        if single:
            return chunk
        return _prefixed(chunk, f"Source document: {_source_label(chunk)}")

    def _context_for(chunk: Document) -> Document:
        document = doc_text.get(str(chunk.metadata.get("source", "")))
        if not document:
            return _source_fallback(chunk)
        try:
            line = chain.invoke(
                {"document": document, "chunk": chunk.page_content}
            ).strip()
            if not line:
                raise ValueError("empty context line")
            return _prefixed(chunk, line)
        except Exception as exc:  # per-chunk fallback: never break ingest
            logger.warning(f"Chunk context generation failed ({exc}); using source")
            return _source_fallback(chunk)

    with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as pool:
        enriched = list(pool.map(_context_for, chunks))
    # Honest accounting: chunks without a matching source doc (or whose
    # LLM call failed) fell back to a source label — or, single-document
    # corpora, to no prefix at all. The log must not claim LLM lines that
    # were never generated.
    fallbacks = sum(
        1
        for original, doc in zip(chunks, enriched)
        if doc.page_content == original.page_content
        or doc.page_content.startswith("[Source document:")
    )
    logger.info(
        f"Chunk context: {len(enriched) - fallbacks} LLM lines, "
        f"{fallbacks} source-label fallbacks"
    )
    return enriched


def build_metadata_cards(
    source_docs: List[Document], llm: Optional[Any]
) -> List[Document]:
    """One card Document per source document: title/authors/date/type.

    Content retrieval cannot answer document-level questions ("who wrote
    this?", "what is this paper?"): author blocks never rank for
    "created/wrote" phrasing in either retrieval leg, while references
    sections — the most author-dense text in any academic document — do,
    and get cited as authorship evidence (Experiment 19). The card puts
    labeled metadata into the index, so those queries have a chunk that
    matches lexically ("Authors:") and semantically, and the generator
    has authoritative evidence that outranks the bibliography decoy.

    Extraction is one utility-model call per DOCUMENT (not per chunk),
    over the document head, with a verbatim-only prompt: a hallucinated
    card would become authoritative false evidence, so anything not
    stated in the head is reported as "not stated". Any failure skips
    that document's card — cards must never break ingestion.
    """
    if llm is None:
        logger.warning("Metadata cards requested but no LLM at ingest; skipping.")
        return []

    # One card per source, keyed on the same identity the loaders stamp.
    heads: Dict[str, Document] = {}
    for doc in source_docs:
        key = str(doc.metadata.get("source", ""))
        heads.setdefault(key, doc)

    chain = (
        ChatPromptTemplate.from_template(METADATA_CARD_PROMPT) | llm | StrOutputParser()
    )

    def _card_for(item: Any) -> Optional[Document]:
        source, doc = item
        try:
            fields = chain.invoke(
                {
                    "source": _source_label(doc),
                    "document_head": doc.page_content[:_DOC_CONTEXT_CHARS],
                }
            ).strip()
            if not fields:
                raise ValueError("empty card")
            return Document(
                page_content=(
                    f"[Document metadata] Source: {_source_label(doc)}\n{fields}"
                ),
                metadata={**doc.metadata, "metadata_card": True},
            )
        except Exception as exc:  # per-document fallback: never break ingest
            logger.warning(f"Metadata card failed for {source!r} ({exc}); skipping")
            return None

    with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as pool:
        cards = [c for c in pool.map(_card_for, heads.items()) if c is not None]
    logger.info(
        f"Metadata cards: {len(cards)} built, {len(heads) - len(cards)} skipped"
    )
    return cards
