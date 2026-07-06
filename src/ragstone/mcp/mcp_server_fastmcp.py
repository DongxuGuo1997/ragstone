#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server Implementation using FastMCP

This module provides a FastMCP-based server for the LangChain RAG pipeline,
offering a clean API for RAG operations with proper error handling.
"""

import logging
from functools import partial

from anyio import to_thread
from mcp.server.fastmcp import FastMCP

from ragstone.config.settings import get_config
from ragstone.rag.pipeline import OllamaPipeline, OpenAIPipeline
from ragstone.utils.exceptions import PipelineError

# The registry (shared with the REST API) is thread-safe: tools run
# concurrently on the server's event loop and hand long operations to
# worker threads via anyio, so registry access must be atomic.
from ragstone.utils.registry import get_pipeline as _get_pipeline
from ragstone.utils.registry import pop_pipeline as _pop_pipeline
from ragstone.utils.registry import put_pipeline as _put_pipeline
from ragstone.utils.registry import snapshot_pipelines as _snapshot_pipelines

# Initialize configuration
config = get_config()
logger = logging.getLogger(__name__)


def _safe_error(action: str, exc: Exception) -> str:
    """Build the error text returned to the MCP client.

    The full exception (with traceback) goes to the server log. Typed
    pipeline errors have messages written for end users, so they are
    echoed back; any other exception could leak paths or internals and is
    replaced with a generic message.
    """
    logger.error(f"{action} failed: {exc}", exc_info=True)
    if isinstance(exc, PipelineError):
        return f"{action} failed: {exc}"
    return (
        f"{action} failed with an internal error ({type(exc).__name__}). "
        "See the server logs for details."
    )


# Initialize FastMCP server
mcp = FastMCP("Ragstone")


@mcp.tool()
def create_openai_pipeline(
    model: str = "gpt-4o-mini", pipeline_id: str = "default_openai"
) -> str:
    """Create an OpenAI-based RAG pipeline.

    Args:
        model: OpenAI model to use (e.g., gpt-4o-mini, gpt-4o, gpt-4.1)
        pipeline_id: Unique identifier for this pipeline

    Returns:
        Success message with pipeline details
    """
    try:
        pipeline = OpenAIPipeline(model=model)
        _put_pipeline(pipeline_id, pipeline)
        return (
            f"OpenAI pipeline '{pipeline_id}' created successfully with model {model}"
        )
    except Exception as e:
        return _safe_error("Creating OpenAI pipeline", e)


@mcp.tool()
def create_ollama_pipeline(
    model: str = "llama3", pipeline_id: str = "default_ollama"
) -> str:
    """Create an Ollama-based RAG pipeline.

    Args:
        model: Ollama model to use (e.g., llama3, phi4, deepseek-r1:8b)
        pipeline_id: Unique identifier for this pipeline

    Returns:
        Success message with pipeline details
    """
    try:
        pipeline = OllamaPipeline(model=model)
        _put_pipeline(pipeline_id, pipeline)
        return (
            f"Ollama pipeline '{pipeline_id}' created successfully with model {model}"
        )
    except Exception as e:
        return _safe_error("Creating Ollama pipeline", e)


@mcp.tool()
async def load_documents(
    pipeline_id: str, data_dir: str = "data", page_urls: str = "", wiki_query: str = ""
) -> str:
    """Load and split documents into the specified pipeline.

    Args:
        pipeline_id: Pipeline identifier to load documents into
        data_dir: Directory containing local documents
        page_urls: Comma-separated URLs to scrape (optional)
        wiki_query: Wikipedia search query (optional)

    Returns:
        Success message with document count
    """
    pipeline = _get_pipeline(pipeline_id)
    if pipeline is None:
        return f"Pipeline '{pipeline_id}' not found. Create it first."

    try:
        # Parse URLs if provided
        urls = (
            [url.strip() for url in page_urls.split(",") if url.strip()]
            if page_urls
            else None
        )

        # Load and split documents. Run in a worker thread: this scrapes the
        # web and embeds the corpus, and a sync tool would block the server's
        # event loop (pings, other tool calls) for the whole duration.
        texts = await to_thread.run_sync(
            partial(
                pipeline.load_and_split,
                data_dir=data_dir,
                page_urls=urls,
                wiki_query=wiki_query if wiki_query else None,
            )
        )

        if texts:
            doc_count = len(texts)
            return f"Loaded and split {doc_count} document chunks into pipeline '{pipeline_id}'"
        else:
            return "No documents were loaded. Check your data sources."

    except Exception as e:
        return _safe_error("Loading documents", e)


@mcp.tool()
async def setup_retriever(
    pipeline_id: str,
    use_ensemble: bool = True,
    chain_type: str = "simple",
    use_reranker: bool = False,
) -> str:
    """Set up the retriever and RAG chain for the specified pipeline.

    Args:
        pipeline_id: Pipeline identifier to set up
        use_ensemble: Whether to use ensemble retriever (BM25 + Vector)
        chain_type: Type of RAG chain (simple, multi_query, fusion, agent)
        use_reranker: Add a cross-encoder reranking stage (requires the rerank extra)

    Returns:
        Success message with configuration details
    """
    pipeline = _get_pipeline(pipeline_id)
    if pipeline is None:
        return f"Pipeline '{pipeline_id}' not found. Create it first."

    try:

        def _configure():
            # Embeds the corpus and may download the cross-encoder model —
            # too slow to run on the server's event loop.
            if isinstance(pipeline, OpenAIPipeline):
                pipeline.set_retriever_openai(
                    use_ensemble=use_ensemble, use_reranker=use_reranker
                )
            elif isinstance(pipeline, OllamaPipeline):
                pipeline.set_retriever_ollama(
                    use_ensemble=use_ensemble, use_reranker=use_reranker
                )
            pipeline.create_rag_chain(chain_type=chain_type)

        await to_thread.run_sync(_configure)

        retriever_type = "Ensemble (BM25 + Vector)" if use_ensemble else "Vector only"
        if use_reranker:
            retriever_type += " + Reranker"
        return f"Pipeline '{pipeline_id}' configured with {retriever_type} retriever and {chain_type} RAG chain"

    except Exception as e:
        return _safe_error("Setting up the retriever", e)


@mcp.tool()
async def ask_question(
    question: str, pipeline_id: str = "default_openai", session_id: str = "mcp_session"
) -> str:
    """Ask a question to the RAG pipeline and get an answer.

    Args:
        question: Question to ask about the loaded documents
        pipeline_id: Pipeline identifier to query
        session_id: Session ID for conversation memory

    Returns:
        Answer from the RAG pipeline
    """
    pipeline = _get_pipeline(pipeline_id)
    if pipeline is None:
        return f"Pipeline '{pipeline_id}' not found. Create it first."

    try:
        # Full retrieval + LLM round trip — run off the event loop.
        response = await to_thread.run_sync(
            partial(pipeline.ask_question, question, session_id=session_id)
        )

        if response:
            return f"**Answer:** {response}"
        else:
            return "Sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        return _safe_error("Answering the question", e)


@mcp.tool()
def list_pipelines() -> str:
    """List all available pipelines and their status.

    Returns:
        Formatted list of all pipelines with their details
    """
    pipelines = _snapshot_pipelines()
    if not pipelines:
        return "No pipelines created yet. Use create_openai_pipeline or create_ollama_pipeline first."

    result = "**Available Pipelines:**\n\n"
    for pipeline_id, pipeline in pipelines:
        pipeline_type = "OpenAI" if isinstance(pipeline, OpenAIPipeline) else "Ollama"
        model = pipeline.LLM.get_model_name() if pipeline.LLM else "Not set"
        has_docs = "Yes" if pipeline.texts else "No"
        has_chain = "Yes" if pipeline.get_chain() else "No"

        result += f"**{pipeline_id}:**\n"
        result += f"  - Type: {pipeline_type}\n"
        result += f"  - Model: {model}\n"
        result += f"  - Documents loaded: {has_docs}\n"
        result += f"  - RAG chain ready: {has_chain}\n\n"

    return result


@mcp.tool()
def get_pipeline_info(pipeline_id: str) -> str:
    """Get detailed information about a specific pipeline.

    Args:
        pipeline_id: Pipeline identifier to get info for

    Returns:
        Detailed information about the pipeline
    """
    pipeline = _get_pipeline(pipeline_id)
    if pipeline is None:
        return f"Pipeline '{pipeline_id}' not found."

    pipeline_type = "OpenAI" if isinstance(pipeline, OpenAIPipeline) else "Ollama"
    model = pipeline.LLM.get_model_name() if pipeline.LLM else "Not set"
    doc_count = len(pipeline.texts) if pipeline.texts else 0
    has_chain = pipeline.get_chain() is not None

    result = f"**Pipeline Info: {pipeline_id}**\n\n"
    result += f"**Type:** {pipeline_type}\n"
    result += f"**Model:** {model}\n"
    result += f"**Document chunks:** {doc_count}\n"
    result += f"**RAG chain configured:** {'Yes' if has_chain else 'No'}\n"
    result += "**Configuration:**\n"
    result += f"  - Database type: {config.database.default_type}\n"
    result += f"  - Batch size: {config.database.batch_size}\n"
    result += f"  - Similarity K: {config.database.similarity_k}\n"

    return result


@mcp.tool()
def delete_pipeline(pipeline_id: str) -> str:
    """Delete a pipeline and free up resources.

    Args:
        pipeline_id: Pipeline identifier to delete

    Returns:
        Success or error message
    """
    # Remove atomically first so no other tool can look it up mid-cleanup.
    pipeline = _pop_pipeline(pipeline_id)
    if pipeline is None:
        return f"Pipeline '{pipeline_id}' not found."

    try:
        # Clean up resources if available
        if hasattr(pipeline, "vector_db") and pipeline.vector_db:
            pipeline.vector_db.cleanup()

        return f"Pipeline '{pipeline_id}' deleted successfully"

    except Exception as e:
        return _safe_error("Deleting the pipeline", e)


def main():
    """Entry point for the MCP server (used by the ragstone-mcp console script)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting Ragstone FastMCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
