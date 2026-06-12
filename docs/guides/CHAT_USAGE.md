# 💬 How to Use Your Know-RAG in Chat

You have several ways to directly chat with your RAG pipeline:

## 🎯 Method 1: MCP Tools in Cursor Chat (Recommended)

Use the MCP tools directly in Cursor's chat interface:

```
@know-rag create_ollama_pipeline model="llama3" pipeline_id="my-chat"
@know-rag load_documents pipeline_id="my-chat" data_dir="data"
@know-rag setup_retriever pipeline_id="my-chat" use_ensemble=true chain_type="simple"
@know-rag ask_question question="What is this project about?" pipeline_id="my-chat"
```

**Advantages:**
- ✅ Integrated with Cursor
- ✅ No additional setup needed
- ✅ Full MCP functionality
- ✅ Session management

## 🖥️ Method 2: Command Line Chat Interface

Run the interactive command-line chat:

```bash
python -m know_rag.ui.chat_interface
# or, after `pip install`:
know-rag-chat
```

**Features:**
- 🦙 Choose Ollama or OpenAI models
- 💬 Interactive chat loop
- 📚 Automatic document loading
- 🔍 Ensemble retriever setup
- 💾 Session memory

**Usage:**
1. Choose pipeline type (ollama/openai)
2. Select your model
3. Start chatting!
4. Type 'quit' to exit

## 🔧 Method 3: Direct Python Integration

Use the pipeline directly in your own Python code:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from know_rag.rag.pipeline import OllamaPipeline

# Initialize pipeline
pipeline = OllamaPipeline(model="llama3")

# Load documents
texts = pipeline.load_and_split(data_dir="data")

# Setup retriever
pipeline.set_retriever_ollama(use_ensemble=True)
pipeline.create_rag_chain(chain_type="simple")

# Ask questions
response = pipeline.ask_question("What is this project about?")
print(response)
```

## 🚀 Quick Start Examples

### Using MCP in Cursor:
```
@know-rag create_ollama_pipeline
@know-rag load_documents data_dir="data"
@know-rag setup_retriever use_ensemble=true
@know-rag ask_question question="Explain the main features"
```

### Using Command Line:
```bash
python -m know_rag.ui.chat_interface
# Choose: ollama
# Model: llama3
# Then start asking questions!
```

## 🛠️ Configuration Options

### Pipeline Types:
- **Ollama**: Local models (llama3, phi4, deepseek-r1:8b)
- **OpenAI**: Cloud models (gpt-3.5-turbo, gpt-4, gpt-4o-mini)

### Retriever Options:
- **Ensemble**: BM25 + Vector search (recommended)
- **Vector**: Semantic search only
- **BM25**: Keyword search only

### Chain Types:
- **Simple**: Basic RAG chain
- **Multi-query**: Multiple query variations
- **Fusion**: Advanced query fusion

## 📝 Tips for Best Results

1. **Document Quality**: Add relevant documents to `data/` directory
2. **Question Style**: Ask specific, clear questions
3. **Context**: Reference previous conversations using session IDs
4. **Model Choice**: Use appropriate models for your use case
5. **Retriever**: Ensemble retriever usually gives best results

## 🔍 Troubleshooting

### Common Issues:

**No documents loaded:**
- Check `data/` directory has files
- Install missing dependencies: `pip install markdown`

**Ollama connection failed:**
- Start Ollama: `ollama serve`
- Check model availability: `ollama list`

**OpenAI API errors:**
- Set API key: `export OPENAI_API_KEY=your_key`
- Check API quota and billing

**MCP tools not working:**
- Restart MCP server: `./scripts/start_mcp_server.sh`
- Check `.cursor/mcp.json` configuration
- Verify Cursor MCP integration

## 🎯 Choose Your Method

- **For Cursor users**: Use MCP tools (Method 1)
- **For quick testing**: Use command line (Method 2)  
- **For custom integration**: Use direct Python (Method 3)

Happy chatting with your documents! 🚀 