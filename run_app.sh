#!/bin/bash

echo "🚀 Starting LangChain RAG Streamlit App..."
echo "📱 The app will open in your browser automatically"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

streamlit run src/langchain_rag/ui/streamlit_app.py --logger.level=DEBUG