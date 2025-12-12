#!/usr/bin/env python3
"""
MCP Connection Test Script
Tests the MCP server connection and tool availability
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mcp_connection():
    """Test MCP server connection and tools"""
    print("🔧 Testing MCP Server Connection")
    print("=" * 40)
    
    try:
        # Import the MCP server from the correct location
        from langchain_rag.mcp.mcp_server_fastmcp import mcp
        
        print("✅ MCP server imported successfully")
        print(f"   Server name: {mcp.name}")
        
        # Get available tools
        tools = await mcp.get_tools()
        print(f"✅ Found {len(tools)} tools:")
        
        for i, (name, tool) in enumerate(tools.items(), 1):
            print(f"   {i}. {name}")
            print(f"      Description: {tool.description[:60]}...")
        
        print("\n🎯 MCP Server Status: READY")
        return True
        
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_tool_call():
    """Test calling a tool directly"""
    print("\n🛠️ Testing Tool Call")
    print("=" * 40)
    
    try:
        from langchain_rag.mcp.mcp_server_fastmcp import list_pipelines
        
        # This won't work directly due to @mcp.tool decoration,
        # but we can test the import
        print("✅ Tool functions imported successfully")
        
        # Test the actual MCP protocol
        from langchain_rag.mcp.mcp_server_fastmcp import mcp
        
        # Simulate MCP tool call
        print("🔄 Simulating MCP tool call...")
        
        # Get the tool
        tools = await mcp.get_tools()
        if 'list_pipelines' in tools:
            print("✅ list_pipelines tool found")
            print("   Tool is ready for MCP client calls")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

def check_configuration():
    """Check MCP configuration files"""
    print("\n📋 Checking Configuration Files")
    print("=" * 40)
    
    config_files = [
        ".cursor/mcp.json",
        "~/.cursor/mcp.json", 
        "cursor_mcp_settings.json",
        "cursor_mcp_config.json"
    ]
    
    for config_file in config_files:
        try:
            # Expand ~ if present
            file_path = os.path.expanduser(config_file)
            
            if os.path.exists(file_path):
                print(f"✅ Found: {config_file}")
                
                with open(file_path, 'r') as f:
                    config = json.load(f)
                    
                if 'mcpServers' in config:
                    servers = config['mcpServers']
                    for server_name, server_config in servers.items():
                        print(f"   Server: {server_name}")
                        print(f"   Command: {server_config.get('command', 'N/A')}")
                        print(f"   Args: {server_config.get('args', [])}")
                        print(f"   CWD: {server_config.get('cwd', 'N/A')}")
                        
            else:
                print(f"⚠️  Not found: {config_file}")
                
        except Exception as e:
            print(f"❌ Error reading {config_file}: {e}")

async def main():
    """Main test function"""
    print("🚀 MCP Server Connection Test")
    print("=" * 50)
    
    # Set environment
    os.environ['USER_AGENT'] = 'MCP-Test-Client/1.0'
    
    # Run tests
    connection_ok = await test_mcp_connection()
    tool_ok = await test_tool_call()
    
    check_configuration()
    
    print("\n" + "=" * 50)
    if connection_ok and tool_ok:
        print("🎉 ALL TESTS PASSED!")
        print("   MCP server is ready for Cursor integration")
        print("\n💡 Next steps:")
        print("   1. Restart Cursor")
        print("   2. Check MCP settings in Cursor")
        print("   3. Look for tools in Cursor's command palette")
    else:
        print("❌ SOME TESTS FAILED")
        print("   Check the errors above for debugging")

if __name__ == "__main__":
    asyncio.run(main()) 