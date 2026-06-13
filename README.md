# Ragstone

**A foundation stone for grounded Q&A applications.**

Ragstone is an efficient Retrieval-Augmented Generation (RAG) engine for
smaller applications: one LLM call per answer, fully local operation with
Ollama if you want it, and measurable answer quality via a built-in
evaluation harness. Use it as a Python library, a Streamlit app, a CLI
chat, or as an MCP tool that puts your documents in reach of any agent
(Claude, Cursor, ...). Built with LangChain 1.x and LangGraph.

## Why Ragstone?

RAG didn't get replaced by agents — it became the primitive they stand on.
Ragstone (a real building stone) leans into being that foundation:

- **Efficient by design** — a fixed, deterministic pipeline: one retrieval,
  one LLM call, predictable latency and cost. No agent loops unless you
  build them on top.
- **Local-first** — runs fully offline with Ollama, FAISS, and a local
  cross-encoder reranker. No API key required.
- **Measured, not vibes** — ships an eval harness (retrieval hit rate/MRR
  plus an LLM judge for correctness and faithfulness) that fails the build
  if quality regresses against the committed baseline.
- **A tool for agents** — the bundled MCP server makes your documents a
  first-class tool for Claude, Cursor, or any MCP-compatible client. In an
  agentic world, the agent is the customer; Ragstone is the ground it
  stands on.

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
- **Cross-Encoder Reranking** (optional): Two-stage retrieval for higher precision

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

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for online models)
- Ollama installed (for offline models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DongxuGuo1997/ragstone.git
   cd ragstone
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package**
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run src/ragstone/ui/streamlit_app.py
   # Or: make run-streamlit
   ```

## MCP Server Integration (Model Context Protocol)

Expose the RAG pipeline as MCP tools for use in VS Code/Cursor and other MCP-compatible clients.

### Quick MCP Setup

1. **Start the MCP server**
   ```bash
   # Using the startup script
   ./scripts/start_mcp_server.sh

   # Or the console script (after pip install -e .)
   ragstone-mcp

   # Or directly
   python ragstone_mcp_server.py
   ```

2. **Configure VS Code/Cursor**

   Add to your MCP configuration:
   ```json
   {
     "mcpServers": {
       "ragstone": {
         "command": "python",
         "args": ["ragstone_mcp_server.py"],
         "cwd": "/path/to/ragstone"
       }
     }
   }
   ```

3. **Use in your IDE**
   ```
   @ragstone create_openai_pipeline with model "gpt-4"
   @ragstone load_documents with data_dir "docs"
   @ragstone ask_question "What is the main architecture?"
   ```

### Available MCP Tools

- `create_openai_pipeline` - Create OpenAI-based pipeline
- `create_ollama_pipeline` - Create Ollama-based pipeline
- `load_documents` - Load documents from various sources
- `setup_retriever` - Configure retrieval strategy
- `ask_question` - Query your documents
- `list_pipelines` - Show all pipelines
- `get_pipeline_info` - Get pipeline details
- `delete_pipeline` - Remove pipeline

**Full MCP Documentation**: See [docs/README_MCP.md](docs/README_MCP.md) for complete setup and usage guide.

## Project Structure

```
ragstone/
├── src/ragstone/           # Core package
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
│   │   └── splitter.py          # Document chunking
│   ├── ui/                      # User interfaces
│   │   ├── streamlit_app.py     # Web interface
│   │   └── chat_interface.py    # CLI chat
│   ├── mcp/                     # MCP server integration
│   │   └── mcp_server_fastmcp.py
│   └── utils/                   # Utilities and helpers
│       ├── exceptions.py        # Custom exception hierarchy
│       └── full_chain.py        # Complete chain integration
├── tests/                       # Test suite
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── pyproject.toml               # Project configuration & dependencies
├── .env.example                 # Environment template
└── README.md                    # This file
```

## Configuration

Configuration options:

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_ID=your_org_id  # Optional
OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # Optional, this is the default
OLLAMA_BASE_URL=http://localhost:11434
VECTOR_STORE_TYPE=faiss  # or chroma
```

> **Note:** Persisted vector stores (FAISS indices, Chroma collections) must be
> rebuilt if the embedding model changes — embeddings from different models are
> not compatible.

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

## Usage

### Web Interface

1. **Start the application**
   ```bash
   streamlit run src/ragstone/ui/streamlit_app.py
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
from ragstone import OpenAIPipeline, get_config

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
from ragstone.config.settings import Config, DatabaseConfig, LLMConfig

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
from ragstone import OpenAIPipeline
from ragstone.utils import PipelineError, LLMInitializationError

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
from ragstone.rag.vector_db import create_vector_store_proxy

# Create Chroma vector store
chroma_store = create_vector_store_proxy(
    "chroma",
    persist_directory="custom_chroma_db"
)

# Create FAISS vector store
faiss_store = create_vector_store_proxy("faiss")
```

### Reranking (optional)

Two-stage retrieval: the ensemble retriever fetches a wide candidate pool,
then a local cross-encoder rescores each (question, chunk) pair and keeps
only the best matches. This noticeably improves precision when documents
contain similar, easily-confused facts — on the bundled eval set it lifts
retrieval MRR from 0.90 to 0.96 and fixes the hardest distractor case.

```bash
pip install -e ".[rerank]"   # pulls sentence-transformers (~PyTorch)
```

```python
pipeline.set_retriever_openai(use_ensemble=True, use_reranker=True)
```

The cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80 MB,
downloaded on first use) runs fully locally — it works in offline/Ollama
mode too. In the Streamlit UI, enable it under Advanced Settings.

## Testing

Run the test suite:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ragstone --cov-report=html
```

## Evaluation

The repo ships a small, readable RAG evaluation harness (`evals/`) that runs
against the real pipeline with a bundled corpus of fictional-fact documents
and a hand-written golden dataset:

- **Layer 1 — retrieval** (deterministic, no judge): hit rate and MRR for
  whether the right chunks come back.
- **Layer 2 — generation** (LLM-as-judge, costs a few cents): answer
  correctness vs. the gold answer, and faithfulness to the retrieved context
  (hallucination check). The judge prompts are in `evals/judge.py`, in the
  open.

```bash
make eval-retrieval   # Layer 1 only — fast, free
make eval             # both layers — needs OPENAI_API_KEY

# Free, fully local run (requires Ollama; scores are not comparable
# across different judge models):
python evals/run_eval.py --provider ollama --model llama3 \
    --judge-provider ollama --judge-model llama3

# Measure the effect of reranking (requires the rerank extra)
python evals/run_eval.py --mode retrieval --rerank
```

Each run is compared against `evals/baseline.json` and **fails if any metric
drops more than 0.05 below baseline** — so quality regressions show up as
failed runs, not silent drift. After an intentional behavior change, accept
new scores with `python evals/run_eval.py --update-baseline` and commit the
file. A per-case report with judge reasons is written to `evals/report.md`.

### Benchmark Results

Every default in Ragstone was chosen by measurement, not intuition. The
numbers below come from the harness above on the bundled corpus (38 cases);
the full hypothesis → method → decision log is in
[EXPERIMENTS.md](EXPERIMENTS.md). Single corpus — read these as direction and
magnitude, not decimal places.

**Retrieval (Layer 1, deterministic).** Reranking is the largest ranking
lever; `text-embedding-3-small` gives full coverage at ~5× lower cost than
ada-002; `k=4` is the coverage knee (k=2 loses answers, k=6 adds nothing).

| Configuration (k=4)                       | hit_rate | MRR   |
|-------------------------------------------|---------:|------:|
| ada-002, ensemble                         | 0.971    | 0.902 |
| text-embedding-3-small, ensemble          | 1.000    | 0.895 |
| text-embedding-3-small, ensemble + rerank | 1.000    | **0.964** |

**Generation (Layer 2, LLM-judged).** The headline finding is a *negative*
one, and it drives the default: multi-query and fusion add ~2× the LLM calls
and 3–4× the retrievals per question, but deliver **no measurable correctness
gain** over plain RAG on this corpus (every gap below is a single case out of
38 — noise). So `simple` is the default; the others stay available for corpora
where question phrasing is genuinely ambiguous.

| Chain type   | correct_rate | faithful_rate | relative cost        |
|--------------|-------------:|--------------:|----------------------|
| **simple**   | 0.947        | 0.947         | 1× (baseline)        |
| multi_query  | 0.947        | 0.974         | ~2× calls, ~3× reads |
| fusion       | 0.974        | 0.921         | ~2× calls, ~4× reads |

## Logging

The application includes basic logging with the standard levels (DEBUG, INFO, WARNING, ERROR, CRITICAL). Configure the level and file output via the `logging` section of `config.json` (see Configuration above).

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Input Validation**: All user inputs are validated
- **Safe Deserialization**: FAISS loading uses safe defaults
- **Error Information**: Sensitive information is filtered from logs

## Contributing

Contributions are welcome! To get started:

1. Fork the repository and create a feature branch
2. Set up the development environment: `./scripts/setup_environment.sh` (or `make install-dev`)
3. Make your changes, keeping `make lint` and `make test` green
4. Open a pull request with a clear description of the change

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/DongxuGuo1997/ragstone/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Chroma](https://github.com/chroma-core/chroma) for vector database
- [OpenAI](https://openai.com/) for language models
- [Ollama](https://ollama.ai/) for local language models
