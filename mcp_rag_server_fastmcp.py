#!/usr/bin/env python3
"""
MCP Server Entry Point for LangChain RAG Pipeline

Convenience launcher for MCP clients (Cursor, VS Code, etc.) configured with a
file path. Equivalent to running the installed `rag-mcp-server` console script.
"""

import sys
from pathlib import Path

try:
    from langchain_rag.mcp.mcp_server_fastmcp import main
except ImportError:
    # Fall back to the in-repo sources when the package is not installed.
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    try:
        from langchain_rag.mcp.mcp_server_fastmcp import main
    except ImportError as e:
        print(f"Error importing MCP server: {e}")
        print("Install the package first: pip install -e .")
        sys.exit(1)

if __name__ == "__main__":
    main()
