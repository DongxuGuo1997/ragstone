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
  one LLM call, predictable latency and cost. An optional agent mode exists
  so you can measure exactly what an agent loop buys you (see below).
- **Local-first** — runs fully offline with Ollama, FAISS, and a local
  cross-encoder reranker. No API key required.
- **Measured, not vibes** — ships an eval harness (retrieval hit rate/MRR
  plus an LLM judge for correctness and faithfulness) that fails the build
  if quality regresses against the committed baseline.
- **A tool for agents** — the bundled MCP server makes your documents a
  first-class tool for Claude, Cursor, or any MCP-compatible client. In an
  agentic world, the agent is the customer; Ragstone is the ground it
  stands on.

## Measured, at a glance

Questions this codebase answered with its own eval harness instead of
opinion:

| Question | Verdict |
|---|---|
| Does an agent loop beat the fixed pipeline? | Identical quality, **1.8× latency, 1.45× tokens** → the pipeline stays default |
| Does self-correcting retrieval (CRAG) pay? | Looked like +0.027 at n=43; **failed to replicate at n=224** — quality within noise, 2× cost is not → opt-in, honestly labeled |
| Was the small eval set lying to us? | Yes, once: scaling 43 → 224 cases reversed a conclusion (Exp 9) — design verdicts now require `--set large` |
| Where do follow-up seconds go? | **926 ms** in one rephrase LLM call → prompt + model fix: multi-turn quality **0.8 → 1.0** on the 5-case smoke slice, rephrase **−40%** |
| Are smaller chunks sharper? | No — hit rate **drops** 1.0 → 0.914 at 500 chars |
| Is concurrent embedding safe? | **3.1× faster** ingestion, identical vectors and retrieval metrics |
| Does chunk enrichment beat more context? | Document identity in the chunk: hit **+1.5pp**, faithfulness **+2.9pp** at +5.7% tokens — now the default; k=6's extra volume had *hurt* |
| Can a router capture self-correction's edge cheaply? | Quality held (faithfulness up to **0.981** tuned), but **1.25× tokens** still fails the pre-set gate after tuning → `auto` ships opt-in, `simple` stays default (Exp 15/15b) |

Full methods and numbers: [EXPERIMENTS.md](EXPERIMENTS.md) · Design
reasoning and trade-offs: [ARCHITECTURE.md](ARCHITECTURE.md) · What's
next, with acceptance criteria: [ROADMAP.md](ROADMAP.md)

Because baselines are committed with every quality change, the repo can
chart its own measured trajectory — generated from git history, no
hand-typed numbers (`python evals/quality_history.py`):

![Measured quality over the project's git history](docs/quality_history.svg)

## Features

### Multiple LLM Providers
- **OpenAI**: GPT-4o Mini, GPT-4o, GPT-4.1 (any chat model by name)
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
- **Agent Mode**: LLM-driven retrieval loop, for measured comparison against the fixed pipeline
- **Corrective RAG**: retrieval grades itself, rewrites failed queries, and refuses with evidence — its apparent quality win at n=43 failed to replicate at n=224 (Experiment 9), which is the point of measuring
- **Query Routing** (`auto`, opt-in): a cheap classifier sends each question to simple or corrective — kept opt-in because its own cost gate said so (Experiments 15/15b)
- **Evidence Highlighting**: source snippets mark the exact words the answer reuses — post-hoc alignment, no prompt changes

### Vector Store Support
- **FAISS** (default): fast in-process similarity search
- **Qdrant**: embedded local mode or a server via one env var (`[qdrant]` extra)
- **pgvector**: RAG on the Postgres you already run (`[pgvector]` extra)
- **Chroma**: embedded persistent store
- Retrieval parity across backends is enforced by test (Experiment 13);
  ingestion is incremental everywhere via the embedding cache
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
   @ragstone create_openai_pipeline with model "gpt-4o-mini"
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
│   │   ├── pipeline.py          # Orchestration (load → retrieve → answer)
│   │   ├── rag.py               # RAG chain creation
│   │   ├── cache.py             # Exact-match response cache
│   │   ├── embeddings.py        # OpenAI / Ollama embedding selection
│   │   ├── loader.py            # Document loading utilities
│   │   ├── vector_db.py         # Vector store implementations
│   │   ├── memory.py            # Conversation memory (LangGraph)
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
RAGSTONE_CHECKPOINT_BACKEND=memory  # or sqlite (needs the sqlite extra)
RAGSTONE_CHECKPOINT_DB=store/checkpoints.sqlite  # used by the sqlite backend
RAGSTONE_LLM_MAX_RETRIES=3  # retries on transient LLM/embedding API errors
RAGSTONE_LLM_TIMEOUT=60  # per-request timeout in seconds
RAGSTONE_MAX_QUESTION_CHARS=4000  # questions above this are rejected pre-API
RAGSTONE_REPHRASE_MODEL=gpt-4.1-nano  # optional: faster follow-up rephrasing
RAGSTONE_CHUNK_CONTEXT=source  # document identity in chunks (Exp 12); off/llm
RAGSTONE_EMBED_CACHE=on  # re-ingesting embeds only changed chunks (Exp 14)
RAGSTONE_EMBED_CACHE_PATH=store/embedding_cache.sqlite
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
    "openai_models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
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
pipeline = OpenAIPipeline(model="gpt-4o-mini")

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

### Agent mode: pipeline vs. agent, measured

Ragstone's default is a fixed pipeline — retrieve once, answer once. The
`agent` chain type is the counterpoint: the LLM gets the retriever as a
`search_documents` tool and drives the loop itself, searching again with a
refined query when the first results don't answer the question (built on
LangChain 1.x `create_agent`).

```python
pipeline.create_rag_chain(chain_type="agent")
```

Which is better? Don't guess — measure. The eval harness reports quality
(correctness, faithfulness) **and** efficiency (latency, tokens) per chain
type on the same corpus:

```bash
python evals/run_eval.py --chain-type simple
python evals/run_eval.py --chain-type agent
```

Compare the two runs in `evals/report.md`. On the bundled eval corpus
(gpt-4o-mini; measured on the earlier 38-question revision of the golden
set — the current set has 43):

| | simple | agent |
|---|---|---|
| correct rate | 0.947 | 0.947 |
| faithful rate | 0.974 | 0.974 |
| avg latency | 1.4 s | 2.7 s |
| total tokens | 39k | 57k |

Identical quality, 1.8× the latency, 1.45× the tokens: when first-shot
retrieval is already good, the agent's ability to re-search buys nothing —
it only pays. That is why the fixed pipeline is the default. On a corpus
where retrieval misses more often, the trade-off can flip; the harness
lets you find out for yours instead of guessing.

### Durable conversation memory (optional)

By default, conversation history lives in memory and is lost when the
process exits. To persist it across restarts, install the `sqlite` extra
and switch the checkpoint backend:

```bash
pip install -e ".[sqlite]"   # pulls langgraph-checkpoint-sqlite
export RAGSTONE_CHECKPOINT_BACKEND=sqlite
export RAGSTONE_CHECKPOINT_DB=store/checkpoints.sqlite  # optional, this is the default
```

Each conversation is checkpointed per `session_id`, so after a restart a
session picks up exactly where it left off — follow-up questions still
resolve references against the earlier turns.

### REST API (optional)

Serve the pipeline over HTTP for applications (the MCP server covers
agents; this covers everything else):

```bash
pip install -e ".[api]"
ragstone-api          # binds 127.0.0.1:8000
```

```bash
curl -X POST localhost:8000/pipelines -H 'content-type: application/json' \
  -d '{"pipeline_id": "docs", "model": "gpt-4o-mini"}'
curl -X POST localhost:8000/pipelines/docs/documents -d '{"data_dir": "data"}' \
  -H 'content-type: application/json'
curl -X POST localhost:8000/pipelines/docs/retriever -d '{"chain_type": "simple"}' \
  -H 'content-type: application/json'
curl -X POST localhost:8000/pipelines/docs/ask -H 'content-type: application/json' \
  -d '{"question": "What is the main architecture?"}'
```

Production behaviors built in:

- **Streaming**: pass `"stream": true` to `/ask` for Server-Sent Events.
- **Probes**: `GET /health` (liveness) and `GET /ready` (a pipeline with a
  RAG chain exists) for orchestrators; both stay unauthenticated.
- **Auth**: set `RAGSTONE_API_KEY` and clients must send it as `X-API-Key`.
- **Backpressure**: at most `RAGSTONE_API_MAX_CONCURRENCY` (default 8)
  simultaneous `/ask` requests; beyond that the server answers `429`
  immediately instead of queueing until it collapses.
- **Safe errors**: typed pipeline errors map to precise status codes with
  user-safe messages; anything unexpected is a generic `500` with details
  only in the server log.

Interactive docs at `localhost:8000/docs` (FastAPI's built-in Swagger UI).
Bind address/port via `RAGSTONE_API_HOST` / `RAGSTONE_API_PORT`.

### Docker deployment (optional)

Local development never needs Docker — the venv flow and the embedded
vector stores are primary. For a server-shaped deployment, the compose
stack runs the API next to a real Qdrant server and a Postgres with
pgvector:

```bash
OPENAI_API_KEY=sk-... docker compose up --build
# API on :8000, Qdrant on :6333, Postgres on :5432
```

Documents go in `./data` (mounted read-only; `RAGSTONE_DATA_ROOT` confines
ingestion to it). The API defaults to the Qdrant server here; switch with
`VECTOR_STORE_TYPE=pgvector` (or `faiss`) — same code path, no rebuild.
The image bakes in no secrets: keys come from the environment at run
time, and `.dockerignore` excludes `.env`.

#### Vector store backends

| `VECTOR_STORE_TYPE` | Where it runs | Extra | When |
|---|---|---|---|
| `faiss` (default) | in-process, per session | — | demos, evals, notebooks |
| `qdrant` | embedded (`QDRANT_PATH`) or server (`QDRANT_URL`) | `[qdrant]` | persistence without a server; scale by pointing at one |
| `pgvector` | your Postgres (`RAGSTONE_PG_URL`) | `[pgvector]` | the database you already run |
| `chroma` | embedded, persistent dir | — | legacy persistent option |

Retrieval quality is backend-independent by construction and by test —
a parity test asserts identical rankings to FAISS, and Experiment 13
measures it on the full eval set. Set `RAGSTONE_COLLECTION` to pin a
stable collection name when one deployment owns the store (unset =
unique per pipeline, the safe multi-tenant default).

### Observability

Every `ask_question` / `ask_question_stream` call emits one structured log
line on the `ragstone.requests` logger:

```
request=1f2e3d4c session=web-42 chain=simple cache_hit=False latency_ms=1440 tokens=1031
```

`request` is a correlation id for tying together all log lines from one
call; `tokens` aggregates every LLM call the request needed (rephrase,
agent searches, answer), which makes per-request cost visible. Point your
log processor at these lines for p95 latency and cost-per-session — the
data is guaranteed to exist.

For deep tracing (every prompt, retrieval, and token), Ragstone works with
LangSmith out of the box — LangChain honors these env vars automatically:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=ragstone  # optional
```

No code changes needed; unset `LANGSMITH_TRACING` and the overhead is gone.

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
numbers below come from the harness above on the bundled corpus (the
38-case revision of the golden set current at the time; today's set has
43 cases — see [EXPERIMENTS.md](EXPERIMENTS.md) for the full
hypothesis → method → decision log). Single corpus — read these as
direction and magnitude, not decimal places.

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
