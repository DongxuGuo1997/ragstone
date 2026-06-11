#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server Implementation using FastMCP

This module provides a FastMCP-based server for the LangChain RAG pipeline,
offering a clean API for RAG operations with proper error handling.
"""

import logging
from functools import partial
from typing import Dict, Union

from anyio import to_thread
from mcp.server.fastmcp import FastMCP

from langchain_rag.config.settings import get_config
from langchain_rag.rag.pipeline import OllamaPipeline, OpenAIPipeline

# Initialize configuration
config = get_config()
logger = logging.getLogger(__name__)

# Global pipeline storage
_pipelines: Dict[str, Union[OpenAIPipeline, OllamaPipeline]] = {}

# Initialize FastMCP server
mcp = FastMCP("LangChain RAG Pipeline")


@mcp.tool()
def create_openai_pipeline(
    model: str = "gpt-3.5-turbo", pipeline_id: str = "default_openai"
) -> str:
    """Create an OpenAI-based RAG pipeline.

    Args:
        model: OpenAI model to use (e.g., gpt-3.5-turbo, gpt-4, gpt-4o-mini)
        pipeline_id: Unique identifier for this pipeline

    Returns:
        Success message with pipeline details
    """
    try:
        pipeline = OpenAIPipeline(model=model)
        _pipelines[pipeline_id] = pipeline
        return f"✅ OpenAI pipeline '{pipeline_id}' created successfully with model {model}"
    except Exception as e:
        logger.error(f"Failed to create OpenAI pipeline: {e}")
        return f"❌ Failed to create OpenAI pipeline: {str(e)}"


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
        _pipelines[pipeline_id] = pipeline
        return f"✅ Ollama pipeline '{pipeline_id}' created successfully with model {model}"
    except Exception as e:
        logger.error(f"Failed to create Ollama pipeline: {e}")
        return f"❌ Failed to create Ollama pipeline: {str(e)}"


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
    if pipeline_id not in _pipelines:
        return f"❌ Pipeline '{pipeline_id}' not found. Create it first."

    try:
        pipeline = _pipelines[pipeline_id]

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
            return f"✅ Loaded and split {doc_count} document chunks into pipeline '{pipeline_id}'"
        else:
            return "⚠️ No documents were loaded. Check your data sources."

    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        return f"❌ Failed to load documents: {str(e)}"


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
        chain_type: Type of RAG chain (simple, multi_query, fusion)
        use_reranker: Add a cross-encoder reranking stage (requires the rerank extra)

    Returns:
        Success message with configuration details
    """
    if pipeline_id not in _pipelines:
        return f"❌ Pipeline '{pipeline_id}' not found. Create it first."

    try:
        pipeline = _pipelines[pipeline_id]

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
        return f"✅ Pipeline '{pipeline_id}' configured with {retriever_type} retriever and {chain_type} RAG chain"

    except Exception as e:
        logger.error(f"Failed to setup retriever: {e}")
        return f"❌ Failed to setup retriever: {str(e)}"


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
    if pipeline_id not in _pipelines:
        return f"❌ Pipeline '{pipeline_id}' not found. Create it first."

    try:
        pipeline = _pipelines[pipeline_id]

        # Full retrieval + LLM round trip — run off the event loop.
        response = await to_thread.run_sync(
            partial(pipeline.ask_question, question, session_id=session_id)
        )

        if response:
            return f"🤖 **Answer:** {response}"
        else:
            return "❌ Sorry, I couldn't generate a response. Please try again."

    except Exception as e:
        logger.error(f"Failed to ask question: {e}")
        return f"❌ Error generating response: {str(e)}"


@mcp.tool()
def list_pipelines() -> str:
    """List all available pipelines and their status.

    Returns:
        Formatted list of all pipelines with their details
    """
    if not _pipelines:
        return "📝 No pipelines created yet. Use create_openai_pipeline or create_ollama_pipeline first."

    result = "📋 **Available Pipelines:**\n\n"
    for pipeline_id, pipeline in _pipelines.items():
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
    if pipeline_id not in _pipelines:
        return f"❌ Pipeline '{pipeline_id}' not found."

    pipeline = _pipelines[pipeline_id]
    pipeline_type = "OpenAI" if isinstance(pipeline, OpenAIPipeline) else "Ollama"
    model = pipeline.LLM.get_model_name() if pipeline.LLM else "Not set"
    doc_count = len(pipeline.texts) if pipeline.texts else 0
    has_chain = pipeline.get_chain() is not None

    result = f"📊 **Pipeline Info: {pipeline_id}**\n\n"
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
    if pipeline_id not in _pipelines:
        return f"❌ Pipeline '{pipeline_id}' not found."

    try:
        # Clean up resources if available
        pipeline = _pipelines[pipeline_id]
        if hasattr(pipeline, "vector_db") and pipeline.vector_db:
            pipeline.vector_db.cleanup()

        del _pipelines[pipeline_id]
        return f"✅ Pipeline '{pipeline_id}' deleted successfully"

    except Exception as e:
        logger.error(f"Failed to delete pipeline: {e}")
        return f"❌ Failed to delete pipeline: {str(e)}"


def main():
    """Entry point for the MCP server (used by the rag-mcp-server console script)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting LangChain RAG Pipeline FastMCP Server...")
    mcp.run()


if __name__ == "__main__":
    main()
