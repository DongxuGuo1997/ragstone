#!/bin/bash

# Ragstone MCP Server Startup Script (FastMCP)

echo "Starting Ragstone MCP Server (FastMCP)..."

# Check if we're in the project directory
if [ ! -f "src/ragstone/mcp/mcp_server_fastmcp.py" ]; then
    echo "Error: src/ragstone/mcp/mcp_server_fastmcp.py not found. Please run from project root."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if MCP SDK is available
echo "Checking dependencies..."
if ! python -c "from mcp.server.fastmcp import FastMCP" 2>/dev/null; then
    echo "Error: MCP SDK not found. Please install with: pip install -e ."
    exit 1
fi

echo "MCP SDK available"

# Set environment variables for better identification
export USER_AGENT="Cursor-MCP-RAG-Pipeline/2.0"

echo "Starting FastMCP server..."
echo "Server will run until interrupted (Ctrl+C)"
echo "Configure your MCP client to connect to this server"
echo ""
echo "Cursor Configuration:"
echo "  File: .cursor/settings.json"
echo "  Server: ragstone"
echo ""

# Run the FastMCP server
python -m ragstone.mcp.mcp_server_fastmcp
