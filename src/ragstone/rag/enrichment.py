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

    if mode == "source":
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
        doc_text.setdefault(key, doc.page_content[:4000])

    chain = ChatPromptTemplate.from_template(CONTEXT_PROMPT) | llm | StrOutputParser()

    def _context_for(chunk: Document) -> Document:
        document = doc_text.get(str(chunk.metadata.get("source", "")))
        if not document:
            return _prefixed(chunk, f"Source document: {_source_label(chunk)}")
        try:
            line = chain.invoke(
                {"document": document, "chunk": chunk.page_content}
            ).strip()
            if not line:
                raise ValueError("empty context line")
            return _prefixed(chunk, line)
        except Exception as exc:  # per-chunk fallback: never break ingest
            logger.warning("Chunk context generation failed (%s); using source", exc)
            return _prefixed(chunk, f"Source document: {_source_label(chunk)}")

    with ThreadPoolExecutor(max_workers=_LLM_WORKERS) as pool:
        enriched = list(pool.map(_context_for, chunks))
    logger.info("Enriched %d chunks with LLM context lines", len(enriched))
    return enriched
