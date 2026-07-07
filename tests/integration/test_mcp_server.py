"""
End-to-end tests for the FastMCP server (no network required).

A real MCP client session talks to the real server over an in-memory
transport, exercising the full tool lifecycle: list tools, create a
pipeline, load documents, configure retrieval, ask a question, inspect,
and delete. The pipeline classes are stubbed, so what is under test is
the MCP layer itself — tool registration, argument passing, worker-thread
offloading, registry handling, and response text.
"""

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ragstone.mcp import mcp_server_fastmcp as srv
from ragstone.utils import registry


class _StubPipeline:
    """Mimics the Pipeline surface the MCP tools touch."""

    def __init__(self, model="stub-model"):
        self.model = model
        self.texts = None
        self._chain = None
        self.vector_db = None
        self.llm_proxy = self  # the tools call pipeline.llm_proxy.get_model_name()

    def get_model_name(self):
        return self.model

    def load_and_split(self, data_dir=None, page_urls=None, wiki_query=None):
        self.texts = [f"chunk-{i}" for i in range(3)]
        return self.texts

    def create_rag_chain(self, chain_type="simple"):
        self._chain = object()

    def get_chain(self):
        return self._chain

    def ask_question(self, question, session_id=None):
        return f"stub answer to: {question}"


class _StubOpenAIPipeline(_StubPipeline):
    def set_retriever_openai(self, use_ensemble=True, use_reranker=False):
        self.retriever_kind = ("openai", use_ensemble, use_reranker)


class _StubOllamaPipeline(_StubPipeline):
    def set_retriever_ollama(self, use_ensemble=True, use_reranker=False):
        self.retriever_kind = ("ollama", use_ensemble, use_reranker)


@pytest.fixture(autouse=True)
def stub_pipelines(monkeypatch):
    """Swap in stub pipeline classes and start/end with an empty registry."""
    monkeypatch.setattr(srv, "OpenAIPipeline", _StubOpenAIPipeline)
    monkeypatch.setattr(srv, "OllamaPipeline", _StubOllamaPipeline)
    registry.clear_pipelines()
    yield
    registry.clear_pipelines()


async def _call(session, name, args=None):
    """Call a tool and return its text response, asserting protocol success."""
    result = await session.call_tool(name, args or {})
    assert not result.isError
    return result.content[0].text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_lifecycle_over_mcp():
    async with create_connected_server_and_client_session(srv.mcp) as session:
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert {
            "create_openai_pipeline",
            "create_ollama_pipeline",
            "load_documents",
            "setup_retriever",
            "ask_question",
            "list_pipelines",
            "get_pipeline_info",
            "delete_pipeline",
        } <= names

        text = await _call(
            session,
            "create_openai_pipeline",
            {"model": "gpt-4o-mini", "pipeline_id": "p1"},
        )
        assert "created successfully" in text

        text = await _call(session, "load_documents", {"pipeline_id": "p1"})
        assert "3 document chunks" in text

        text = await _call(
            session, "setup_retriever", {"pipeline_id": "p1", "chain_type": "simple"}
        )
        assert "simple RAG chain" in text

        text = await _call(
            session, "ask_question", {"question": "hello?", "pipeline_id": "p1"}
        )
        assert "stub answer to: hello?" in text

        text = await _call(session, "list_pipelines")
        assert "p1" in text
        assert "gpt-4o-mini" in text

        text = await _call(session, "get_pipeline_info", {"pipeline_id": "p1"})
        assert "RAG chain configured:** Yes" in text

        text = await _call(session, "delete_pipeline", {"pipeline_id": "p1"})
        assert "deleted successfully" in text

        text = await _call(session, "list_pipelines")
        assert "No pipelines" in text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_pipeline_type_is_reported():
    async with create_connected_server_and_client_session(srv.mcp) as session:
        await _call(
            session,
            "create_ollama_pipeline",
            {"model": "llama3", "pipeline_id": "local"},
        )
        text = await _call(session, "get_pipeline_info", {"pipeline_id": "local"})
        assert "Type:** Ollama" in text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_operations_on_missing_pipeline_return_guidance():
    async with create_connected_server_and_client_session(srv.mcp) as session:
        for tool, args in [
            ("load_documents", {"pipeline_id": "ghost"}),
            ("setup_retriever", {"pipeline_id": "ghost"}),
            ("ask_question", {"question": "q", "pipeline_id": "ghost"}),
            ("get_pipeline_info", {"pipeline_id": "ghost"}),
            ("delete_pipeline", {"pipeline_id": "ghost"}),
        ]:
            text = await _call(session, tool, args)
            assert "not found" in text, f"{tool} should report a missing pipeline"
