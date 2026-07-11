"""
Unit tests for contextual chunk enrichment (ROADMAP 1.1, no network).

The enrichment step runs once at ingest and feeds everything downstream
(embedding, BM25, the generator's context), so its invariants are: modes
behave as documented, metadata survives untouched, an LLM failure
degrades safely instead of breaking ingestion, and — Experiment 22 — a
single-document corpus gets NO identity prefix (there is nothing to
disambiguate, and the prefix measurably poisons document-level retrieval
there).
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragstone.rag.enrichment import enrich_chunks


def _chunk(text, source="corona_solar_guide.md", title=None):
    metadata = {"source": source}
    if title:
        metadata["title"] = title
    return Document(page_content=text, metadata=metadata)


def _two_doc_chunks():
    """A minimal multi-document corpus: prefixes must apply."""
    return [
        _chunk("clean every 4 months"),
        _chunk("torque is 18 Nm", source="helios_manual.md"),
    ]


class TestOffMode:
    """mode=off must be a strict identity."""

    def test_off_is_identity(self):
        chunks = [_chunk("torque is 18 Nm")]
        assert enrich_chunks(chunks, mode="off") is chunks

    def test_empty_input_is_identity(self):
        assert enrich_chunks([], mode="source") == []


class TestSourceMode:
    """Deterministic document-identity prefixes (the Experiment-12 default)."""

    def test_prefixes_document_identity(self):
        enriched = enrich_chunks(_two_doc_chunks(), mode="source")
        assert enriched[0].page_content == (
            "[Source document: corona_solar_guide.md]\nclean every 4 months"
        )
        assert enriched[1].page_content == (
            "[Source document: helios_manual.md]\ntorque is 18 Nm"
        )

    def test_title_metadata_wins_over_filename(self):
        [enriched, _] = enrich_chunks(
            [
                _chunk("text", title="Corona K-7 Guide"),
                _chunk("other", source="helios_manual.md"),
            ],
            mode="source",
        )
        assert "[Source document: Corona K-7 Guide]" in enriched.page_content

    def test_metadata_is_preserved_and_originals_untouched(self):
        originals = _two_doc_chunks()
        enriched = enrich_chunks(originals, mode="source")
        assert enriched[0].metadata == originals[0].metadata
        assert originals[0].page_content == "clean every 4 months"  # not mutated

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="chunk-context mode"):
            enrich_chunks([_chunk("text")], mode="semantic")


class TestSingleDocumentCorpus:
    """One source document -> no identity prefix (Experiment 22).

    The prefix exists to tell documents apart; with one document it only
    poisons retrieval — it dominates the embeddings of content-empty
    chunks (name lists, TOCs) and zeroes the document name's BM25 IDF.
    """

    def test_source_mode_skips_the_prefix(self):
        chunks = [_chunk("abstract text"), _chunk("contributor name list")]
        assert enrich_chunks(chunks, mode="source") is chunks

    def test_distinctness_is_counted_on_source_not_title(self):
        # Two FILES sharing a display title are still two documents.
        chunks = [
            _chunk("a", source="v1/report.md", title="Report"),
            _chunk("b", source="v2/report.md", title="Report"),
        ]
        enriched = enrich_chunks(chunks, mode="source")
        assert all(c.page_content.startswith("[Source document:") for c in enriched)

    def test_llm_mode_still_generates_content_lines(self):
        # LLM lines carry content, not just identity — they stay on.
        llm = FakeListChatModel(responses=["Covers the cleaning schedule."])
        docs = [_chunk("Full document text.")]
        [enriched] = enrich_chunks(
            [_chunk("clean every 4 months")], source_docs=docs, mode="llm", llm=llm
        )
        assert enriched.page_content.startswith("[Covers the cleaning schedule.]")

    def test_llm_failure_falls_back_to_no_prefix(self):
        # The fallback must not reintroduce exactly the prefix that harms.
        class _BoomModel(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("api down")

        docs = [_chunk("Full document text.")]
        [enriched] = enrich_chunks(
            [_chunk("text")],
            source_docs=docs,
            mode="llm",
            llm=_BoomModel(responses=["x"]),
        )
        assert enriched.page_content == "text"

    def test_missing_llm_falls_back_to_no_prefix(self):
        [enriched] = enrich_chunks([_chunk("text")], mode="llm", llm=None)
        assert enriched.page_content == "text"


class TestLlmMode:
    """Generated context lines, with per-chunk fallback on any failure."""

    def test_prepends_generated_context_line(self):
        llm = FakeListChatModel(
            responses=["This chunk covers the Corona K-7 cleaning schedule."] * 2
        )
        docs = [
            _chunk("Full Corona K-7 document text."),
            _chunk("Full Helios text.", source="helios_manual.md"),
        ]
        enriched = enrich_chunks(
            [_chunk("clean every 4 months"), _chunk("t", source="helios_manual.md")],
            source_docs=docs,
            mode="llm",
            llm=llm,
        )
        assert enriched[0].page_content.startswith(
            "[This chunk covers the Corona K-7 cleaning schedule.]\n"
        )
        assert enriched[0].page_content.endswith("clean every 4 months")

    def test_missing_llm_falls_back_to_source_mode(self):
        enriched = enrich_chunks(_two_doc_chunks(), mode="llm", llm=None)
        assert "[Source document: corona_solar_guide.md]" in enriched[0].page_content

    def test_llm_failure_falls_back_per_chunk(self):
        class _BoomModel(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("api down")

        llm = _BoomModel(responses=["unused"])
        docs = [
            _chunk("Full document text."),
            _chunk("Other doc.", source="helios_manual.md"),
        ]
        enriched = enrich_chunks(
            [_chunk("text"), _chunk("t2", source="helios_manual.md")],
            source_docs=docs,
            mode="llm",
            llm=llm,
        )
        # Ingestion survives; multi-doc chunks still carry their identity.
        assert "[Source document: corona_solar_guide.md]" in enriched[0].page_content

    def test_chunk_without_matching_source_doc_uses_source_mode(self):
        llm = FakeListChatModel(responses=["unused"] * 2)
        enriched = enrich_chunks(
            [
                _chunk("text", source="orphan.md"),
                _chunk("known text", source="known.md"),
            ],
            source_docs=[_chunk("other doc", source="known.md")],
            mode="llm",
            llm=llm,
        )
        assert "[Source document: orphan.md]" in enriched[0].page_content


class TestMetadataCards:
    """One extracted card per document; failures skip, never break ingest."""

    def test_one_card_per_document_with_labeled_fields(self):
        from ragstone.rag.enrichment import build_metadata_cards

        llm = FakeListChatModel(
            responses=[
                "Title: Corona Installation Guide\nAuthors: Corona Corp\n"
                "Date: 2023\nType: manual"
            ]
        )
        docs = [
            _chunk("Corona Installation Guide...", source="corona_solar_guide.md"),
            # A second chunk of the SAME document must not get its own card.
            _chunk("More of the same doc", source="corona_solar_guide.md"),
        ]
        cards = build_metadata_cards(docs, llm)

        assert len(cards) == 1
        card = cards[0]
        assert card.page_content.startswith(
            "[Document metadata] Source: corona_solar_guide.md"
        )
        assert "Authors: Corona Corp" in card.page_content
        assert card.metadata["metadata_card"] is True
        assert card.metadata["source"] == "corona_solar_guide.md"

    def test_no_llm_yields_no_cards(self):
        from ragstone.rag.enrichment import build_metadata_cards

        assert build_metadata_cards([_chunk("text")], llm=None) == []

    def test_extraction_failure_skips_the_card_not_the_ingest(self):
        from ragstone.rag.enrichment import build_metadata_cards

        class _BoomModel(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("api down")

        cards = build_metadata_cards(
            [_chunk("text")], llm=_BoomModel(responses=["unused"])
        )
        assert cards == []

    def test_cards_built_per_distinct_source(self):
        from ragstone.rag.enrichment import build_metadata_cards

        llm = FakeListChatModel(
            responses=[
                "Title: A\nAuthors: not stated\nDate: not stated\nType: report",
                "Title: B\nAuthors: not stated\nDate: not stated\nType: report",
            ]
        )
        docs = [
            _chunk("doc a", source="a.md"),
            _chunk("doc b", source="b.md"),
        ]
        cards = build_metadata_cards(docs, llm)
        assert sorted(c.metadata["source"] for c in cards) == ["a.md", "b.md"]
