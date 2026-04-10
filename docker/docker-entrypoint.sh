#!/bin/bash
set -e

# Default values
MODE=${MODE:-mcp}
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3

    echo "Waiting for $service to be ready..."
    while ! nc -z "$host" "$port" 2>/dev/null; do
        sleep 1
    done
    echo "$service is ready"
}

# Wait for dependent services if they're configured
if [ -n "$CHROMA_HOST" ] && [ -n "$CHROMA_PORT" ]; then
    wait_for_service "$CHROMA_HOST" "$CHROMA_PORT" "ChromaDB"
fi

if [ -n "$REDIS_URL" ]; then
    REDIS_HOST=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d: -f1)
    REDIS_PORT=$(echo "$REDIS_URL" | sed 's|redis://||' | cut -d: -f2)
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"
fi

# Initialize data directories
mkdir -p /app/data /app/store /app/vs_data /app/logs

echo "Starting LangChain RAG Pipeline in $MODE mode..."

case "$MODE" in
    "mcp")
        echo "Starting MCP Server..."
        exec python mcp_rag_server_fastmcp.py
        ;;
    "streamlit")
        echo "Starting Streamlit UI..."
        exec streamlit run run.py --server.address="$HOST" --server.port="$PORT"
        ;;
    "chat")
        echo "Starting Chat Interface..."
        exec python chat_interface.py
        ;;
    "test")
        echo "Running Connection Test..."
        exec python mcp_connection_test.py
        ;;
    "shell")
        echo "Starting Interactive Shell..."
        exec /bin/bash
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: mcp, streamlit, chat, test, shell"
        exit 1
        ;;
esac
