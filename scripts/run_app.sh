#!/bin/bash

# LangChain RAG Pipeline - Streamlit App Runner

set -e

echo "Starting LangChain RAG Streamlit App..."
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/langchain_rag/ui/streamlit_app.py --logger.level=DEBUG
