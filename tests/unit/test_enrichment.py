"""
Unit tests for contextual chunk enrichment (ROADMAP 1.1, no network).

The enrichment step runs once at ingest and feeds everything downstream
(embedding, BM25, the generator's context), so its invariants are: modes
behave as documented, metadata survives untouched, and an LLM failure
degrades to source mode instead of breaking ingestion.
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


class TestOffMode:
    def test_off_is_identity(self):
        chunks = [_chunk("torque is 18 Nm")]
        assert enrich_chunks(chunks, mode="off") is chunks

    def test_empty_input_is_identity(self):
        assert enrich_chunks([], mode="source") == []


class TestSourceMode:
    def test_prefixes_document_identity(self):
        [enriched] = enrich_chunks([_chunk("clean every 4 months")], mode="source")
        assert enriched.page_content == (
            "[Source document: corona_solar_guide.md]\nclean every 4 months"
        )

    def test_title_metadata_wins_over_filename(self):
        [enriched] = enrich_chunks(
            [_chunk("text", title="Corona K-7 Guide")], mode="source"
        )
        assert "[Source document: Corona K-7 Guide]" in enriched.page_content

    def test_metadata_is_preserved_and_originals_untouched(self):
        original = _chunk("text")
        [enriched] = enrich_chunks([original], mode="source")
        assert enriched.metadata == original.metadata
        assert original.page_content == "text"  # input not mutated

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="chunk-context mode"):
            enrich_chunks([_chunk("text")], mode="semantic")


class TestLlmMode:
    def test_prepends_generated_context_line(self):
        llm = FakeListChatModel(
            responses=["This chunk covers the Corona K-7 cleaning schedule."]
        )
        docs = [_chunk("Full Corona K-7 document text.")]
        [enriched] = enrich_chunks(
            [_chunk("clean every 4 months")], source_docs=docs, mode="llm", llm=llm
        )
        assert enriched.page_content.startswith(
            "[This chunk covers the Corona K-7 cleaning schedule.]\n"
        )
        assert enriched.page_content.endswith("clean every 4 months")

    def test_missing_llm_falls_back_to_source_mode(self):
        [enriched] = enrich_chunks([_chunk("text")], mode="llm", llm=None)
        assert "[Source document: corona_solar_guide.md]" in enriched.page_content

    def test_llm_failure_falls_back_per_chunk(self):
        class _BoomModel(FakeListChatModel):
            def _generate(self, *args, **kwargs):
                raise RuntimeError("api down")

        llm = _BoomModel(responses=["unused"])
        docs = [_chunk("Full document text.")]
        [enriched] = enrich_chunks(
            [_chunk("text")], source_docs=docs, mode="llm", llm=llm
        )
        # Ingestion survives; the chunk still carries its identity.
        assert "[Source document: corona_solar_guide.md]" in enriched.page_content

    def test_chunk_without_matching_source_doc_uses_source_mode(self):
        llm = FakeListChatModel(responses=["unused"])
        [enriched] = enrich_chunks(
            [_chunk("text", source="orphan.md")],
            source_docs=[_chunk("other doc", source="known.md")],
            mode="llm",
            llm=llm,
        )
        assert "[Source document: orphan.md]" in enriched.page_content
