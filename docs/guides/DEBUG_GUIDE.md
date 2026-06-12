# 🔧 Complete Debug & Troubleshooting Guide
## Know-RAG MCP Server for Cursor

---

## 📋 **Quick Diagnosis Checklist**

Before diving into detailed troubleshooting, run these quick checks:

```bash
# 1. Check if MCP server is running
ps aux | grep mcp_rag_server

# 2. Test server connection
python -m know_rag.mcp.connection_test

# 3. Verify configuration exists
ls -la ~/.cursor/mcp.json .cursor/mcp.json

# 4. Check Python environment
which python && python --version

# 5. Test basic functionality
python -c "from mcp.server.fastmcp import FastMCP; print('✅ MCP OK')"
```

---

## 🚨 **Common Issues & Solutions**

### **Issue 1: "0 tools enabled" in Cursor**

**Symptoms:**
- Cursor shows "know-rag" but "0 tools enabled"
- `@know-rag` doesn't work in chat

**Solutions (in order of likelihood):**

#### **🔄 Solution A: Complete Cursor Restart**
```bash
# 1. Quit Cursor completely (Cmd+Q on Mac)
# 2. Wait 5 seconds
# 3. Restart Cursor
# 4. Try: @know-rag list_pipelines
```

#### **🔧 Solution B: Fix MCP Configuration**
```bash
# Create/update global config
cat > ~/.cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "know-rag": {
      "command": "/path/to/know-rag/venv/bin/python",
      "args": ["/path/to/know-rag/know_rag_mcp_server.py"],
      "cwd": "/path/to/know-rag",
      "env": {
        "PYTHONPATH": "/path/to/know-rag/src",
        "USER_AGENT": "Cursor-MCP-Client/1.0"
      }
    }
  }
}
EOF

# Remove conflicting project configs
rm -f .cursor/mcp.json
```

#### **🐍 Solution C: Fix Python Environment**
```bash
# Use absolute path to Python in venv
which python  # Should show: /path/to/know-rag/venv/bin/python

# If not, activate venv first
source venv/bin/activate
```

### **Issue 2: MCP Server Won't Start**

**Symptoms:**
- Server crashes immediately
- Import errors in logs
- Connection test fails

**Solutions:**

#### **📦 Fix Dependencies**
```bash
# Reinstall all requirements
pip install -e .

# Check specific MCP packages
pip list | grep mcp
pip install mcp

# Verify FastMCP installation
python -c "from mcp.server.fastmcp import FastMCP; print('✅ FastMCP OK')"
```

#### **🔍 Debug Server Startup**
```bash
# Run server with debug output
export MCP_LOG_LEVEL=DEBUG
python know_rag_mcp_server.py

# Check for specific errors
python know_rag_mcp_server.py 2>&1 | head -20
```

#### **🌍 Fix Environment Variables**
```bash
# Set required environment variables
export PYTHONPATH="$(pwd)/src"
export USER_AGENT="Cursor-MCP-Client/1.0"

# Test with environment
env PYTHONPATH="$(pwd)/src" python know_rag_mcp_server.py
```

### **Issue 3: Tools Not Appearing in Cursor**

**Symptoms:**
- Server runs but tools don't show in Cursor
- No `@know-rag` autocomplete

**Solutions:**

#### **⚙️ Check Cursor MCP Settings**
1. Open Cursor Settings (Cmd+,)
2. Search for "MCP" 
3. Look for "Model Context Protocol" section
4. Verify "know-rag" is listed and enabled
5. If not listed, add manually with the configuration above

#### **🧹 Clear Cursor Cache**
```bash
# Clear Cursor cache and logs
rm -rf ~/.cursor/logs/
rm -rf ~/.cursor/CachedData/

# Restart Cursor after clearing cache
```

#### **🔄 Refresh MCP Connection**
1. In Cursor Settings → Extensions → MCP
2. Disable "know-rag" server
3. Re-enable it
4. Restart Cursor

### **Issue 4: Configuration File Conflicts**

**Symptoms:**
- Multiple config files exist
- Inconsistent behavior
- Tools work sometimes but not others

**Solution:**
```bash
# Clean up all config files
rm -f .cursor/mcp.json
rm -f cursor_mcp_*.json
rm -f mcp-config.json

# Use ONLY global config
# Edit ~/.cursor/mcp.json with correct settings (see Solution B above)

# Verify only one config exists
find . -name "*mcp*.json" -type f
ls -la ~/.cursor/mcp.json
```

---

## 🔍 **Detailed Troubleshooting Steps**

### **Step 1: Verify Server Functionality**

```bash
# Test 1: Basic server start
python know_rag_mcp_server.py
# Should output: "Starting Know-RAG FastMCP Server..."

# Test 2: Connection test
python -m know_rag.mcp.connection_test
# Should show: "✅ All 8 tools found and working correctly!"

# Test 3: Manual tool test
python -c "
from know_rag_mcp_server import mcp
print('FastMCP server loaded:', mcp.name)
"
```

### **Step 2: Verify Configuration**

```bash
# Check config file syntax
python -c "import json, os; json.load(open(os.path.expanduser('~/.cursor/mcp.json')))"

# Verify paths exist
ls -la /path/to/know-rag/venv/bin/python
ls -la /path/to/know-rag/know_rag_mcp_server.py
ls -la /path/to/know-rag/src/
```

### **Step 3: Test Cursor Integration**

```bash
# In Cursor, try these commands one by one:
@know-rag list_pipelines
# Expected: "📝 No pipelines created yet..."

@know-rag create_openai_pipeline model="gpt-3.5-turbo" pipeline_id="test"
# Expected: Pipeline creation success message

@know-rag list_pipelines
# Expected: Shows "test" pipeline
```

---

## 🛠️ **Advanced Debugging**

### **Debug Server Communication**

```bash
# Run server with maximum verbosity
export MCP_LOG_LEVEL=DEBUG
export PYTHONPATH="$(pwd)/src"
python know_rag_mcp_server.py --verbose 2>&1 | tee server_debug.log

# Check server logs
tail -f server_debug.log
```

### **Debug Cursor Logs**

```bash
# Find Cursor log files
find ~/.cursor -name "*.log" -type f

# Monitor Cursor logs for MCP errors
tail -f ~/.cursor/logs/main.log | grep -i mcp
```

### **Test with Minimal Configuration**

Create a minimal test configuration:

```bash
cat > test_mcp.json << 'EOF'
{
  "mcpServers": {
    "test-server": {
      "command": "python",
      "args": ["-c", "from mcp.server.fastmcp import FastMCP; app=FastMCP('test'); app.run()"],
      "cwd": "/path/to/know-rag"
    }
  }
}
EOF
```

---

## 📊 **Expected Behavior When Working**

### **Server Startup**
```
Starting Know-RAG FastMCP Server...
Server initialized with 8 tools:
- create_openai_pipeline
- create_ollama_pipeline  
- load_documents
- setup_retriever
- ask_question
- list_pipelines
- get_pipeline_info
- delete_pipeline
```

### **Cursor Integration**
- `@know-rag` appears in autocomplete
- All 8 tools are available
- Commands execute without errors
- Pipeline operations work correctly

### **Tool Test Results**
```bash
python -m know_rag.mcp.connection_test
```
Should output:
```
🔍 Testing MCP Server Connection...
✅ Server started successfully
✅ Found 8 tools (expected 8)
✅ All tools have descriptions
✅ All expected tools present
✅ create_openai_pipeline tool working
✅ All 8 tools found and working correctly!
```

---

## 🚀 **Complete Reset Procedure**

If all else fails, follow this complete reset:

```bash
# 1. Stop everything
pkill -f mcp_rag_server
pkill -f Cursor

# 2. Clean up all configs
rm -f .cursor/mcp.json
rm -f ~/.cursor/mcp.json
rm -f cursor_mcp_*.json
rm -f mcp-config.json

# 3. Clear caches
rm -rf ~/.cursor/logs/
rm -rf ~/.cursor/CachedData/

# 4. Reinstall dependencies
pip install -e .

# 5. Create fresh global config
cat > ~/.cursor/mcp.json << 'EOF'
{
  "mcpServers": {
    "know-rag": {
      "command": "/path/to/know-rag/venv/bin/python",
      "args": ["/path/to/know-rag/know_rag_mcp_server.py"],
      "cwd": "/path/to/know-rag",
      "env": {
        "PYTHONPATH": "/path/to/know-rag/src",
        "USER_AGENT": "Cursor-MCP-Client/1.0"
      }
    }
  }
}
EOF

# 6. Test server
python -m know_rag.mcp.connection_test

# 7. Start Cursor and test
# Try: @know-rag list_pipelines
```

---

## 🎯 **Quick Reference**

### **Available Tools**
1. `create_openai_pipeline` - Create OpenAI-based RAG pipeline
2. `create_ollama_pipeline` - Create Ollama-based RAG pipeline  
3. `load_documents` - Load documents from files/URLs/Wikipedia
4. `setup_retriever` - Configure retriever and RAG chain
5. `ask_question` - Query the RAG pipeline
6. `list_pipelines` - Show all available pipelines
7. `get_pipeline_info` - Get detailed pipeline information
8. `delete_pipeline` - Remove a pipeline

### **Essential Commands**
```bash
# Test server
python -m know_rag.mcp.connection_test

# Start server manually
python know_rag_mcp_server.py

# Check processes
ps aux | grep mcp_rag_server

# View config
cat ~/.cursor/mcp.json

# Test in Cursor
@know-rag list_pipelines
```

### **Key File Locations**
- **Global Config**: `~/.cursor/mcp.json` ✅ (Use this)
- **Project Config**: `.cursor/mcp.json` ❌ (Remove this)
- **Server Script**: `know_rag_mcp_server.py`
- **Test Script**: `src/know_rag/mcp/connection_test.py`
- **Startup Script**: `scripts/start_mcp_server.sh`

---

## 💡 **Pro Tips**

1. **Always use absolute paths** in MCP configuration
2. **Keep only one config file** (global preferred)
3. **Restart Cursor completely** after config changes
4. **Test server manually** before expecting Cursor integration
5. **Check logs** when things don't work as expected
6. **Use the connection test** to verify everything works

---

## 🆘 **Still Having Issues?**

If you've followed all steps and still have problems:

1. **Check Cursor version** - Ensure you have MCP support
2. **Try a different project** - Test with a minimal setup
3. **Check system requirements** - Verify Python version compatibility
4. **Review Cursor's MCP documentation** - For version-specific requirements
5. **Test with other MCP servers** - Verify Cursor's MCP functionality

---

**Last Updated:** December 2024  
**Status:** ✅ Comprehensive troubleshooting guide - covers all known issues 