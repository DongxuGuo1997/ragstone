#!/bin/bash

# LangChain RAG Pipeline - Streamlit App Runner
# Usage:
#   ./scripts/run_app.sh          # Run default UI (streamlit_app.py)
#   ./scripts/run_app.sh --clean  # Run alternative UI (app.py)
#   ./scripts/run_app.sh --help   # Show help

set -e

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean    Run with app.py (alternative UI)"
    echo "  --help     Show this help message"
    echo ""
    echo "Press Ctrl+C to stop the app"
}

# Default UI file
UI_FILE="src/langchain_rag/ui/streamlit_app.py"

# Parse arguments
case "$1" in
    --clean)
        UI_FILE="src/langchain_rag/ui/app.py"
        echo "Starting LangChain RAG Streamlit App (clean mode)..."
        ;;
    --help)
        show_help
        exit 0
        ;;
    "")
        echo "Starting LangChain RAG Streamlit App..."
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

echo "UI: $UI_FILE"
echo "Press Ctrl+C to stop"
echo ""

streamlit run "$UI_FILE" --logger.level=DEBUG
