#!/usr/bin/env python3
"""
MCP Server Entry Point for LangChain RAG Pipeline

This is a convenience entry point that imports and runs the actual MCP server
from the src directory. This makes it easier to configure in Cursor and other
MCP clients.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run the actual MCP server
try:
    # Set up logging first
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LangChain RAG Pipeline FastMCP Server (Entry Point)...")
    
    # Import the FastMCP server
    from langchain_rag.mcp.mcp_server_fastmcp import mcp
    
    if __name__ == "__main__":
        # Run the FastMCP server
        mcp.run()
        
except ImportError as e:
    print(f"❌ Error importing MCP server: {e}")
    print("Make sure you're running from the project root and all dependencies are installed.")
    print("Required packages: fastmcp, langchain_rag")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running MCP server: {e}")
    sys.exit(1) 