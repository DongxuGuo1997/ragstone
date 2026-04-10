# LangChain RAG Pipeline

> **Note**: This is an internal proof-of-concept project, not intended for production use.

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain, supporting multiple LLM providers and a Streamlit web interface.

## Features

### Multiple LLM Providers
- **OpenAI**: GPT-3.5 Turbo, GPT-4, GPT-4o Mini
- **Ollama**: Llama3, Phi4, DeepSeek-R1, and other local models

### Flexible Data Sources
- **Local Files**: PDF, TXT, CSV, DOCX, Markdown
- **Web Pages**: Automatic scraping and content extraction
- **Wikipedia**: Search and load articles automatically
- **File Upload**: Drag-and-drop interface for documents

### RAG Techniques
- **Simple RAG**: Standard retrieval-augmented generation
- **Multi-Query RAG**: Generates multiple queries for retrieval
- **Fusion RAG**: Uses reciprocal rank fusion
- **Ensemble Retrieval**: Combines BM25 and vector similarity

### Vector Store Support
- **FAISS**: Fast similarity search with local storage
- **Chroma**: Persistent vector database with advanced features
- **Custom Embeddings**: OpenAI embeddings with fallback options

### Configuration Management
- **Environment-based**: Automatic configuration from `.env` files
- **JSON Configuration**: Structured configuration files
- **Validation**: Comprehensive configuration validation
- **Flexible Settings**: Database, LLM, UI, and API configurations

### Other Features
- Custom exception hierarchy
- Configurable logging
- Chat session memory
- Basic input validation

### MCP Integration (Model Context Protocol)

The RAG pipeline can be exposed as MCP tools for use in VS Code/Cursor and other MCP-compatible clients.

### **Quick Start**

1. **Start the MCP Server:**
   ```bash
   # Using the startup script (recommended)
   ./start_mcp_server.sh
   
   # Or directly
   python mcp_rag_server_fastmcp.py
   ```

2. **Configure Cursor/VS Code:**
   - See [docs/README_MCP.md](docs/README_MCP.md) for detailed setup instructions

### **Available MCP Tools**

- `create_openai_pipeline` - Create OpenAI-based pipeline
- `create_ollama_pipeline` - Create Ollama-based pipeline  
- `load_documents` - Load documents from various sources
- `setup_retriever` - Configure retrieval strategy
- `ask_question` - Query your documents
- `list_pipelines` - Show all pipelines
- `get_pipeline_info` - Get pipeline details
- `delete_pipeline` - Remove pipeline

### **Example Usage in Cursor**

```
# Create a pipeline
@langchain-rag-pipeline create_openai_pipeline with model "gpt-4" and pipeline_id "my_project"

# Load documents
@langchain-rag-pipeline load_documents with pipeline_id "my_project" and data_dir "docs"

# Setup advanced retrieval
@langchain-rag-pipeline setup_retriever with pipeline_id "my_project" and use_ensemble true

# Ask questions
@langchain-rag-pipeline ask_question "What is the main architecture pattern?" with pipeline_id "my_project"
```

For complete setup instructions, see [docs/README_MCP.md](docs/README_MCP.md).

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for online models)
- Ollama installed (for offline models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DongxuGuo1997/langchain-rag-pipeline.git
   cd langchain-rag-pipeline
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run run.py
   ```

## MCP Server Integration

Expose the RAG pipeline as MCP tools for use in VS Code/Cursor.

### Quick MCP Setup

1. **Start the MCP server**
   ```bash
   ./start_mcp_server.sh
   # Or manually: python mcp_rag_server.py
   ```

2. **Configure VS Code/Cursor**
   
   Add to your MCP configuration:
   ```json
   {
     "mcpServers": {
       "langchain-rag-pipeline": {
         "command": "python",
         "args": ["mcp_rag_server.py"],
         "cwd": "/path/to/your/langchain-rag-pipeline",
         "env": {
           "PYTHONPATH": "/path/to/your/langchain-rag-pipeline/src"
         }
       }
     }
   }
   ```

3. **Use in your IDE**
   ```
   @langchain-rag-pipeline create_openai_pipeline with model "gpt-4"
   @langchain-rag-pipeline load_documents with data_dir "docs"
   @langchain-rag-pipeline ask_question "What is the main architecture?"
   ```

### MCP Features

- 8 tools for pipeline management, document loading, and Q&A
- Support for multiple pipelines
- Multi-query, fusion, and ensemble retrieval options
- Session memory for conversations

**Full MCP Documentation**: See [docs/README_MCP.md](docs/README_MCP.md) for complete setup and usage guide.

## Project Structure

```
langchain-rag-pipeline/
├── src/langchain_rag/           # Core package
│   ├── config/                  # Configuration management
│   │   └── settings.py          # Dataclass-based settings
│   ├── models/                  # LLM abstraction layer
│   │   └── base_model.py        # OpenAI/Ollama proxy classes
│   ├── rag/                     # Core RAG implementation
│   │   ├── pipeline.py          # Main orchestration
│   │   ├── rag.py               # RAG chain creation
│   │   ├── loader.py            # Document loading utilities
│   │   ├── vector_db.py         # Vector store implementations
│   │   ├── memory.py            # Conversation memory
│   │   ├── ensemble.py          # Ensemble retrieval
│   │   └── splitter.py          # Document chunking
│   ├── ui/                      # User interfaces
│   │   ├── streamlit_app.py     # Web interface
│   │   ├── chat_interface.py    # CLI chat
│   │   └── app.py               # Application framework
│   ├── mcp/                     # MCP server integration
│   │   ├── mcp_server.py        # Standard MCP server
│   │   └── mcp_server_fastmcp.py
│   └── utils/                   # Utilities and helpers
│       ├── exceptions.py        # Custom exception hierarchy
│       ├── common.py            # Helper functions
│       └── full_chain.py        # Complete chain integration
├── tests/                       # Test suite
├── docker/                      # Docker deployment files
├── docs/                        # Documentation
├── examples/                    # Example code
├── scripts/                     # Utility scripts
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Configuration

Configuration options:

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_ID=your_org_id  # Optional
OLLAMA_BASE_URL=http://localhost:11434
```

### Configuration File (config.json)
```json
{
  "database": {
    "default_type": "faiss",
    "chroma_persist_dir": "store/chroma_db",
    "batch_size": 100,
    "similarity_k": 4
  },
  "llm": {
    "openai_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"],
    "ollama_models": ["llama3", "phi4", "deepseek-r1:8b"],
    "default_temperature": 0.0
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_file_path": "logs/application.log"
  }
}
```

### Docker Configuration

For Docker deployments, **FAISS is now the default vector store** (changed from ChromaDB). This provides:
- Better performance in containerized environments
- No external dependencies required
- Lower memory footprint
- Faster startup times

#### Environment Variables

```bash
# Vector store selection (defaults to faiss in Docker)
VECTOR_STORE_TYPE=faiss

# Alternative: Use ChromaDB
VECTOR_STORE_TYPE=chroma

# API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Logging
MCP_LOG_LEVEL=INFO
```

#### Quick Docker Setup

```bash
# Start with FAISS (default)
docker-compose up -d rag-pipeline

# Or use ChromaDB
echo "VECTOR_STORE_TYPE=chroma" >> .env
docker-compose --profile chromadb up -d
```

See **[docs/guides/DOCKER_GUIDE.md](docs/guides/DOCKER_GUIDE.md)** for complete Docker setup instructions.

## Usage

### Web Interface

1. **Start the application**
   ```bash
   streamlit run run.py
   ```

2. **Configure your pipeline**
   - Select online (OpenAI) or offline (Ollama) mode
   - Choose your preferred model
   - Upload documents or provide URLs
   - Adjust advanced settings if needed

3. **Build and chat**
   - Click "Build Pipeline" to initialize
   - Start asking questions about your documents

### Programmatic Usage

```python
from src import OpenAIPipeline, get_config

# Load configuration
config = get_config()

# Create pipeline
pipeline = OpenAIPipeline(model="gpt-3.5-turbo")

# Load documents
pipeline.load_and_split(
    data_dir="data",
    page_urls=["https://example.com"],
    wiki_query="artificial intelligence"
)

# Set up retriever and chain
pipeline.set_retriever_openai(use_ensemble=True)
pipeline.create_rag_chain(chain_type="multi_query")

# Ask questions
response = pipeline.ask_question("What is artificial intelligence?")
print(response)
```

## Advanced Features

### Custom Configuration

```python
from src.config import Config, DatabaseConfig, LLMConfig

# Create custom configuration
config = Config(
    database=DatabaseConfig(
        default_type="chroma",
        batch_size=50
    ),
    llm=LLMConfig(
        default_temperature=0.7
    )
)

# Save configuration
config.to_file("my_config.json")
```

### Error Handling

```python
from src.exceptions import PipelineError, LLMInitializationError

try:
    pipeline = OpenAIPipeline(model="gpt-4")
    pipeline.load_and_split(data_dir="invalid_dir")
except LLMInitializationError as e:
    print(f"LLM Error: {e.message}")
    print(f"Context: {e.context}")
except PipelineError as e:
    print(f"Pipeline Error: {e.message}")
```

### Custom Vector Stores

```python
from src.vector_db import create_vector_store_proxy

# Create Chroma vector store
chroma_store = create_vector_store_proxy(
    "chroma", 
    persist_directory="custom_chroma_db"
)

# Create FAISS vector store
faiss_store = create_vector_store_proxy("faiss")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Logging

The application includes basic logging:

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Potential issues
- **ERROR**: Error conditions
- **CRITICAL**: Critical failures

### Log Configuration
```bash
# Run with debug logging
streamlit run run.py --log DEBUG

# Run with file logging enabled
streamlit run run.py
```

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Input Validation**: All user inputs are validated
- **Safe Deserialization**: FAISS loading uses safe defaults
- **Error Information**: Sensitive information is filtered from logs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Chroma](https://github.com/chroma-core/chroma) for vector database
- [OpenAI](https://openai.com/) for language models
- [Ollama](https://ollama.ai/) for local language models

