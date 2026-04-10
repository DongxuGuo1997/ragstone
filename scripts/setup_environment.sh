#!/bin/bash
set -e

# LangChain RAG Pipeline - Environment Setup Script
# This script sets up the development environment for the project

echo "Setting up LangChain RAG Pipeline Development Environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "Python version $python_version is compatible"
else
    echo "Error: Python 3.9+ is required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e ".[dev]"

# Install pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data logs store vs_data

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << 'EOF'
# LangChain RAG Pipeline Environment Configuration

# API Keys (required for functionality)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector Store Configuration
VECTOR_STORE_TYPE=faiss

# Logging Configuration
MCP_LOG_LEVEL=INFO

# Model Configuration
DEFAULT_OPENAI_MODEL=gpt-3.5-turbo
DEFAULT_OLLAMA_MODEL=llama3

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Environment
ENVIRONMENT=development
DEBUG=true

# Remember to add your actual API keys before running the application!
EOF
    echo "Note: Please edit .env file and add your API keys"
fi

# Run basic tests to verify installation
echo "Running basic verification tests..."
python -c "
import sys
try:
    from langchain_rag.rag.pipeline import Pipeline
    from langchain_rag.config.settings import get_config
    from langchain_rag.rag.vector_db import create_vector_store_proxy
    print('Pipeline import successful')

    # Test configuration loading
    config = get_config()
    print(f'Configuration loaded (vector store: {config.database.default_type})')

    # Test vector store creation
    vector_store = create_vector_store_proxy('faiss')
    print('Vector store creation successful')

    print('All basic tests passed!')

except Exception as e:
    print(f'Verification failed: {e}')
    sys.exit(1)
"

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Run the application:"
echo "   - Streamlit UI: ./scripts/run_app.sh"
echo "   - MCP Server: ./scripts/start_mcp_server.sh"
