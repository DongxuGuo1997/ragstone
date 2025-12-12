#!/usr/bin/env python3
"""
MCP Server for LangChain RAG Pipeline

This MCP server exposes the RAG pipeline functionality as tools that can be
used in VS Code with Cursor or other MCP-compatible clients.
"""

import sys
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union, Sequence
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    Resource,
    ListResourcesRequest,
    ReadResourceRequest,
    ReadResourceResult,
    Prompt,
    ListPromptsRequest,
    GetPromptRequest,
    GetPromptResult,
    PromptMessage,
)

from src.pipeline import OpenAIPipeline, OllamaPipeline
from src.config import load_config, get_config
from src.exceptions import PipelineError, LLMInitializationError

# Initialize configuration
config = load_config()
logger = logging.getLogger(__name__)

# Global pipeline storage
_pipelines: Dict[str, Union[OpenAIPipeline, OllamaPipeline]] = {}

# Initialize MCP server
server = Server("langchain-rag-pipeline")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for the RAG pipeline."""
    return [
        Tool(
            name="create_openai_pipeline",
            description="Create an OpenAI-based RAG pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "OpenAI model to use (e.g., gpt-3.5-turbo, gpt-4, gpt-4o-mini)",
                        "default": "gpt-3.5-turbo"
                    },
                    "pipeline_id": {
                        "type": "string", 
                        "description": "Unique identifier for this pipeline",
                        "default": "default_openai"
                    }
                }
            }
        ),
        Tool(
            name="create_ollama_pipeline",
            description="Create an Ollama-based RAG pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Ollama model to use (e.g., llama3, phi4, deepseek-r1:8b)",
                        "default": "llama3"
                    },
                    "pipeline_id": {
                        "type": "string",
                        "description": "Unique identifier for this pipeline", 
                        "default": "default_ollama"
                    }
                }
            }
        ),
        Tool(
            name="load_documents",
            description="Load and split documents into the specified pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline identifier to load documents into",
                        "default": "default_openai"
                    },
                    "data_dir": {
                        "type": "string",
                        "description": "Directory containing local documents",
                        "default": "data"
                    },
                    "page_urls": {
                        "type": "string",
                        "description": "Comma-separated URLs to scrape (optional)"
                    },
                    "wiki_query": {
                        "type": "string",
                        "description": "Wikipedia search query (optional)"
                    }
                },
                "required": ["pipeline_id"]
            }
        ),
        Tool(
            name="setup_retriever",
            description="Set up the retriever and RAG chain for the specified pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline identifier to set up",
                        "default": "default_openai"
                    },
                    "use_ensemble": {
                        "type": "boolean",
                        "description": "Whether to use ensemble retriever (BM25 + Vector)",
                        "default": True
                    },
                    "chain_type": {
                        "type": "string",
                        "description": "Type of RAG chain (simple, multi_query, fusion)",
                        "enum": ["simple", "multi_query", "fusion"],
                        "default": "simple"
                    }
                },
                "required": ["pipeline_id"]
            }
        ),
        Tool(
            name="ask_question",
            description="Ask a question to the RAG pipeline and get an answer",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask about the loaded documents"
                    },
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline identifier to query",
                        "default": "default_openai"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation memory",
                        "default": "mcp_session"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="list_pipelines",
            description="List all available pipelines and their status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_pipeline_info",
            description="Get detailed information about a specific pipeline",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline identifier to get info for"
                    }
                },
                "required": ["pipeline_id"]
            }
        ),
        Tool(
            name="delete_pipeline",
            description="Delete a pipeline and free up resources",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline identifier to delete"
                    }
                },
                "required": ["pipeline_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for the RAG pipeline."""
    
    try:
        if name == "create_openai_pipeline":
            model = arguments.get("model", "gpt-3.5-turbo")
            pipeline_id = arguments.get("pipeline_id", "default_openai")
            
            try:
                pipeline = OpenAIPipeline(model=model)
                _pipelines[pipeline_id] = pipeline
                return [TextContent(
                    type="text",
                    text=f"✅ OpenAI pipeline '{pipeline_id}' created successfully with model {model}"
                )]
            except Exception as e:
                logger.error(f"Failed to create OpenAI pipeline: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Failed to create OpenAI pipeline: {str(e)}"
                )]

        elif name == "create_ollama_pipeline":
            model = arguments.get("model", "llama3")
            pipeline_id = arguments.get("pipeline_id", "default_ollama")
            
            try:
                pipeline = OllamaPipeline(model=model)
                _pipelines[pipeline_id] = pipeline
                return [TextContent(
                    type="text",
                    text=f"✅ Ollama pipeline '{pipeline_id}' created successfully with model {model}"
                )]
            except Exception as e:
                logger.error(f"Failed to create Ollama pipeline: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Failed to create Ollama pipeline: {str(e)}"
                )]

        elif name == "load_documents":
            pipeline_id = arguments.get("pipeline_id", "default_openai")
            data_dir = arguments.get("data_dir", "data")
            page_urls = arguments.get("page_urls")
            wiki_query = arguments.get("wiki_query")
            
            if pipeline_id not in _pipelines:
                return [TextContent(
                    type="text",
                    text=f"❌ Pipeline '{pipeline_id}' not found. Create it first."
                )]
            
            try:
                pipeline = _pipelines[pipeline_id]
                
                # Parse URLs if provided
                urls = [url.strip() for url in page_urls.split(",")] if page_urls else None
                
                # Load and split documents
                texts = pipeline.load_and_split(
                    data_dir=data_dir,
                    page_urls=urls,
                    wiki_query=wiki_query
                )
                
                if texts:
                    doc_count = len(texts)
                    return [TextContent(
                        type="text",
                        text=f"✅ Loaded and split {doc_count} document chunks into pipeline '{pipeline_id}'"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"⚠️ No documents were loaded. Check your data sources."
                    )]
                    
            except Exception as e:
                logger.error(f"Failed to load documents: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Failed to load documents: {str(e)}"
                )]

        elif name == "setup_retriever":
            pipeline_id = arguments.get("pipeline_id", "default_openai")
            use_ensemble = arguments.get("use_ensemble", True)
            chain_type = arguments.get("chain_type", "simple")
            
            if pipeline_id not in _pipelines:
                return [TextContent(
                    type="text",
                    text=f"❌ Pipeline '{pipeline_id}' not found. Create it first."
                )]
            
            try:
                pipeline = _pipelines[pipeline_id]
                
                # Set up retriever based on pipeline type
                if isinstance(pipeline, OpenAIPipeline):
                    pipeline.set_retriever_openai(use_ensemble=use_ensemble)
                elif isinstance(pipeline, OllamaPipeline):
                    pipeline.set_retriever_ollama(use_ensemble=use_ensemble)
                
                # Create RAG chain
                pipeline.create_rag_chain(chain_type=chain_type)
                
                retriever_type = "Ensemble (BM25 + Vector)" if use_ensemble else "Vector only"
                return [TextContent(
                    type="text",
                    text=f"✅ Pipeline '{pipeline_id}' configured with {retriever_type} retriever and {chain_type} RAG chain"
                )]
                
            except Exception as e:
                logger.error(f"Failed to setup retriever: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Failed to setup retriever: {str(e)}"
                )]

        elif name == "ask_question":
            question = arguments.get("question")
            pipeline_id = arguments.get("pipeline_id", "default_openai")
            session_id = arguments.get("session_id", "mcp_session")
            
            if pipeline_id not in _pipelines:
                return [TextContent(
                    type="text",
                    text=f"❌ Pipeline '{pipeline_id}' not found. Create it first."
                )]
            
            try:
                pipeline = _pipelines[pipeline_id]
                
                # Ask the question
                response = pipeline.ask_question(question, session_id=session_id)
                
                if response:
                    return [TextContent(
                        type="text",
                        text=f"🤖 **Answer:** {response}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Sorry, I couldn't generate a response. Please try again."
                    )]
                    
            except Exception as e:
                logger.error(f"Failed to ask question: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Error generating response: {str(e)}"
                )]

        elif name == "list_pipelines":
            if not _pipelines:
                return [TextContent(
                    type="text",
                    text="📝 No pipelines created yet. Use create_openai_pipeline or create_ollama_pipeline first."
                )]
            
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
            
            return [TextContent(type="text", text=result)]

        elif name == "get_pipeline_info":
            pipeline_id = arguments.get("pipeline_id")
            
            if pipeline_id not in _pipelines:
                return [TextContent(
                    type="text",
                    text=f"❌ Pipeline '{pipeline_id}' not found."
                )]
            
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
            result += f"**Configuration:**\n"
            result += f"  - Database type: {config.database.default_type}\n"
            result += f"  - Batch size: {config.database.batch_size}\n"
            result += f"  - Similarity K: {config.database.similarity_k}\n"
            
            return [TextContent(type="text", text=result)]

        elif name == "delete_pipeline":
            pipeline_id = arguments.get("pipeline_id")
            
            if pipeline_id not in _pipelines:
                return [TextContent(
                    type="text",
                    text=f"❌ Pipeline '{pipeline_id}' not found."
                )]
            
            try:
                # Clean up resources if available
                pipeline = _pipelines[pipeline_id]
                if hasattr(pipeline, 'vector_db') and pipeline.vector_db:
                    pipeline.vector_db.cleanup()
                
                del _pipelines[pipeline_id]
                return [TextContent(
                    type="text",
                    text=f"✅ Pipeline '{pipeline_id}' deleted successfully"
                )]
                
            except Exception as e:
                logger.error(f"Failed to delete pipeline: {e}")
                return [TextContent(
                    type="text",
                    text=f"❌ Failed to delete pipeline: {str(e)}"
                )]

        else:
            return [TextContent(
                type="text",
                text=f"❌ Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Error in tool call {name}: {e}")
        return [TextContent(
            type="text",
            text=f"❌ Error executing {name}: {str(e)}"
        )]

@server.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="config://rag",
            name="RAG Pipeline Configuration",
            description="Current configuration for the RAG pipeline",
            mimeType="text/plain"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    if uri == "config://rag":
        return f"""RAG Pipeline Configuration:

Database Settings:
- Default type: {config.database.default_type}
- Batch size: {config.database.batch_size}
- Similarity K: {config.database.similarity_k}

LLM Settings:
- OpenAI models: {', '.join(config.llm.openai_models)}
- Ollama models: {', '.join(config.llm.ollama_models)}
- Default temperature: {config.llm.default_temperature}

Loader Settings:
- Data directory: {config.loader.default_data_dir}
- Supported extensions: {', '.join(config.loader.supported_extensions)}
- Max file size: {config.loader.max_file_size_mb}MB
- Chunk size: {config.loader.chunk_size}
- Chunk overlap: {config.loader.chunk_overlap}
"""
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_prompts()
async def list_prompts() -> List[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="rag_assistant",
            description="System prompt for RAG assistant specialized in a topic",
            arguments=[
                {
                    "name": "topic",
                    "description": "The topic or domain to specialize in",
                    "required": False
                }
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: Dict[str, str]) -> GetPromptResult:
    """Get a prompt by name."""
    if name == "rag_assistant":
        topic = arguments.get("topic", "general")
        
        prompt_text = f"""You are an AI assistant with access to a RAG (Retrieval-Augmented Generation) pipeline. 
You can help users by:

1. Creating and managing RAG pipelines (OpenAI or Ollama-based)
2. Loading documents from various sources (local files, URLs, Wikipedia)
3. Setting up retrievers with different strategies
4. Answering questions based on loaded documents

Your specialty is: {topic}

When helping users:
- Always suggest creating a pipeline first if none exists
- Recommend loading relevant documents before asking questions
- Explain the different RAG techniques available (simple, multi_query, fusion)
- Provide clear, helpful responses based on the retrieved context

Use the available tools to help users build and interact with their RAG systems effectively."""

        return GetPromptResult(
            description=f"RAG assistant specialized in {topic}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt_text)
                )
            ]
        )
    else:
        raise ValueError(f"Unknown prompt: {name}")

async def main():
    """Main entry point for the MCP server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting LangChain RAG Pipeline MCP Server...")
    
    # Run the MCP server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream, 
            InitializationOptions(
                server_name="langchain-rag-pipeline",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main()) 