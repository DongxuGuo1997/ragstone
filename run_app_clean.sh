#!/bin/bash

echo "🚀 Starting LangChain RAG Streamlit App (Clean Mode)..."
echo "📱 The app will open in your browser automatically"
echo "🛑 Press Ctrl+C to stop the app"
echo "ℹ️  Reduced startup messages for cleaner output"
echo ""

#streamlit run src/langchain_rag/ui/streamlit_app.py --server.runOnSave false --server.fileWatcherType none
streamlit run src/langchain_rag/ui/app.py --logger.level=DEBUG