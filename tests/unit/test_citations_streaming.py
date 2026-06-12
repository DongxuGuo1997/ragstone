"""
Unit tests for source citations and streaming (no network required).
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from know_rag.rag.memory import MemoryProxy, SimpleTextRetriever
from know_rag.rag.pipeline import Pipeline
from know_rag.rag.rag import RagProxy
from know_rag.rag.splitter import split_documents
from know_rag.utils.full_chain import FullChain


class TestSplitterMetadata:
    """Chunks must remember which document they came from."""

    def test_split_preserves_source_metadata(self):
        docs = [
            Document(page_content="alpha " * 400, metadata={"source": "a.md"}),
            Document(page_content="beta " * 400, metadata={"source": "b.md"}),
        ]
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 2
        assert all(c.metadata.get("source") in {"a.md", "b.md"} for c in chunks)
        assert {c.metadata["source"] for c in chunks} == {"a.md", "b.md"}

    def test_split_plain_strings_still_works(self):
        chunks = split_documents(["hello world"])
        assert len(chunks) == 1
        assert chunks[0].page_content == "hello world"


class TestGetSources:
    """Pipeline.get_sources returns citation-ready entries."""

    def test_sources_use_metadata_and_trim_paths(self):
        pipeline = Pipeline()
        pipeline._retriever = SimpleTextRetriever(
            docs=[
                Document(
                    page_content="some fact",
                    metadata={"source": "/data/docs/guide.md"},
                )
            ]
        )
        sources = pipeline.get_sources("anything", k=2)
        assert sources == [{"source": "guide.md", "snippet": "some fact"}]

    def test_sources_without_metadata_are_unknown(self):
        pipeline = Pipeline()
        pipeline._retriever = SimpleTextRetriever.from_texts(["some content"])
        sources = pipeline.get_sources("anything")
        assert sources[0]["source"] == "unknown"

    def test_sources_with_no_retriever(self):
        pipeline = Pipeline()
        assert pipeline.get_sources("anything") == []


class _FakeLLMProxy:
    def __init__(self, llm):
        self._llm = llm

    def get_llm(self):
        return self._llm


class TestStreaming:
    """The full chain streams the answer as text chunks."""

    def test_stream_question_yields_full_answer(self):
        llm = FakeListChatModel(responses=["Paris."])
        retriever = SimpleTextRetriever.from_texts(["Paris is the capital of France."])
        rag = RagProxy(model=llm, retriever=retriever)
        full_chain = FullChain(_FakeLLMProxy(llm), rag, MemoryProxy())
        full_chain.create_full_chain("simple")

        chunks = list(full_chain.stream_question("capital?", session_id="s1"))
        assert "".join(chunks) == "Paris."
