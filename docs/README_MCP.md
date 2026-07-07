# Ragstone MCP Server

This project provides a Model Context Protocol (MCP) server that exposes your LangChain RAG pipeline as tools that can be used in VS Code with Cursor or other MCP-compatible clients.

## 🚀 Features

- **Multiple LLM Support**: OpenAI and Ollama-based pipelines
- **Document Loading**: Local files, URLs, and Wikipedia
- **Advanced RAG**: Simple, multi-query, fusion, agent, and corrective strategies
- **Ensemble Retrieval**: BM25 + Vector search combination
- **Memory Management**: Conversation history with session support
- **Pipeline Management**: Create, configure, and manage multiple pipelines

## 📋 Prerequisites

1. **Python Environment**: Python 3.10+ with the project dependencies installed
2. **MCP-Compatible Client**: VS Code with Cursor, Claude Desktop, or other MCP clients
3. **LLM Setup**: 
   - For OpenAI: Set `OPENAI_API_KEY` environment variable
   - For Ollama: Have Ollama running locally with models installed

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
# Install the MCP Python SDK (installed with the package)
pip install mcp

# Install other project dependencies
pip install -e .
```

### 2. Configure Your MCP Client

#### For VS Code with Cursor:

1. Open VS Code/Cursor settings
2. Search for "MCP" or "Model Context Protocol"
3. Add the server configuration:

```json
{
  "mcpServers": {
    "ragstone": {
      "command": "python",
      "args": ["ragstone_mcp_server.py"],
      "cwd": "/path/to/ragstone",
      "env": {
        "PYTHONPATH": "/path/to/ragstone/src",
        "OPENAI_API_KEY": "your-openai-api-key-if-using-openai"
      }
    }
  }
}
```

#### For Claude Desktop:

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragstone": {
      "command": "python",
      "args": ["/path/to/ragstone/ragstone_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/ragstone/src"
      }
    }
  }
}
```

### 3. Test the Server

```bash
# Test that the server starts without errors
python ragstone_mcp_server.py
```

You should see logging output indicating the server is initializing successfully.

## 🔧 Available Tools

### Pipeline Management

#### `create_openai_pipeline`
Create an OpenAI-based RAG pipeline.
- **model**: OpenAI model (e.g., "gpt-4o-mini", "gpt-4o", "gpt-4.1")
- **pipeline_id**: Unique identifier for the pipeline

#### `create_ollama_pipeline`
Create an Ollama-based RAG pipeline.
- **model**: Ollama model (e.g., "llama3", "phi4", "deepseek-r1:8b")
- **pipeline_id**: Unique identifier for the pipeline

#### `list_pipelines`
List all available pipelines and their status.

#### `get_pipeline_info`
Get detailed information about a specific pipeline.
- **pipeline_id**: Pipeline to get info for

#### `delete_pipeline`
Delete a pipeline and free up resources.
- **pipeline_id**: Pipeline to delete

### Document Management

#### `load_documents`
Load and split documents into a pipeline.
- **pipeline_id**: Target pipeline
- **data_dir**: Directory with local documents (default: "data")
- **page_urls**: Comma-separated URLs to scrape (optional)
- **wiki_query**: Wikipedia search query (optional)

#### `setup_retriever`
Configure the retriever and RAG chain.
- **pipeline_id**: Target pipeline
- **use_ensemble**: Use BM25 + Vector ensemble (default: true)
- **chain_type**: RAG strategy ("simple", "multi_query", "fusion", "agent", "corrective")

### Question Answering

#### `ask_question`
Ask questions about your loaded documents.
- **question**: Your question
- **pipeline_id**: Pipeline to query
- **session_id**: Session for conversation memory

## 📚 Usage Examples

### Basic Workflow

1. **Create a Pipeline**:
   ```
   Use the create_openai_pipeline tool with model "gpt-4o-mini"
   ```

2. **Load Documents**:
   ```
   Use load_documents with your data directory or URLs
   ```

3. **Setup Retriever**:
   ```
   Use setup_retriever with ensemble=true and chain_type="multi_query"
   ```

4. **Ask Questions**:
   ```
   Use ask_question to query your documents
   ```

### Advanced Usage

#### Multiple Specialized Pipelines
```
1. Create "legal_docs" pipeline with gpt-4 for legal documents
2. Create "tech_docs" pipeline with llama3 for technical documentation
3. Load different document sets into each pipeline
4. Use appropriate pipeline for domain-specific questions
```

#### Different RAG Strategies
- **Simple**: Basic retrieval and generation
- **Multi-query**: Generates multiple queries for better retrieval
- **Fusion**: Combines multiple retrieval strategies

## 🔍 Resources & Prompts

### Resources
- **config://rag**: Get current RAG pipeline configuration

### Prompts
- **rag_assistant**: System prompt for RAG assistant specialized in a topic

## 🐛 Troubleshooting

### Common Issues

1. **KqueueSelector Error (macOS)**:
   - This version uses the official MCP SDK to avoid this issue
   - If you still encounter it, try setting `PYTHONIOENCODING=utf-8`

2. **Import Errors**:
   - Ensure `PYTHONPATH` includes the `src` directory
   - Check that all dependencies are installed: `pip install -e .`

3. **LLM Connection Issues**:
   - **OpenAI**: Verify `OPENAI_API_KEY` is set correctly
   - **Ollama**: Ensure Ollama is running and models are installed

4. **Document Loading Issues**:
   - Check file permissions in the data directory
   - Verify URLs are accessible
   - Ensure supported file formats (see config.example.json)

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export MCP_LOG_LEVEL=DEBUG
```

### Testing Without MCP Client

You can test the server functionality directly:

```python
import sys
sys.path.insert(0, 'src')

from ragstone.rag.pipeline import OpenAIPipeline
from ragstone.utils.registry import put_pipeline, get_pipeline

# Test pipeline creation (the MCP server and REST API share this registry)
pipeline = OpenAIPipeline(model="gpt-4o-mini")
put_pipeline("test", pipeline)
assert get_pipeline("test") is pipeline
print("Pipeline registered successfully")
```

## 🔧 Configuration

The server uses the same configuration system as the main application:
- `config.example.json`: Configuration template (copy to create your own)
- Environment variables for API keys
- Logging configuration in `src/ragstone/config/settings.py`

## 🚀 Integration with Development Workflow

### Code Documentation
Use the RAG pipeline to:
- Query your codebase documentation
- Get explanations of complex code sections
- Find usage examples and patterns

### Research Assistant
- Load research papers and documentation
- Ask questions about technical concepts
- Compare different approaches and methodologies

### Project Knowledge Base
- Load project wikis, READMEs, and documentation
- Query historical decisions and architecture docs
- Get context-aware answers about your project

## 📝 Example MCP Client Interactions

### In VS Code/Cursor:
```
@ragstone create_openai_pipeline with model gpt-4 and pipeline_id "my_docs"

@ragstone load_documents with pipeline_id "my_docs" and data_dir "docs"

@ragstone ask_question "What is the main architecture pattern used in this project?" with pipeline_id "my_docs"
```

### In Claude Desktop:
The tools will appear in Claude's interface and can be invoked naturally through conversation.

## 🔐 Security Considerations

- API keys are handled through environment variables
- No sensitive data is logged by default
- Pipeline isolation prevents cross-contamination
- Resource cleanup prevents memory leaks

## 🤝 Contributing

To extend the MCP server:
1. Add new tools with the `@mcp.tool()` decorator in `src/ragstone/mcp/mcp_server_fastmcp.py`
2. Implement the tool function body
3. Update documentation and examples
4. Test with different MCP clients

## 📄 License

This MCP server implementation follows the same license as the main Ragstone project. 