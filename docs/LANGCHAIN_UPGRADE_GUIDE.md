# LangChain Upgrade Guide

This guide provides a safe procedure for upgrading LangChain dependencies without breaking your application.

## 🔒 **Current Stable Versions**

Your application is tested and working with:
- `langchain`: 0.3.x
- `langchain-core`: 0.3.x
- `langchain-community`: 0.3.x
- `langchain-openai`: 0.3.x
- `langchain-ollama`: 0.3.x

**Last Compatibility Test**: See test run in `tests/test_langchain_compatibility.py`

---

## 🚨 **Before You Upgrade**

### **Why Version Pinning Matters**

LangChain is actively developed and occasionally introduces breaking changes. Our version pinning strategy:

```python
"langchain>=0.3.0,<0.4.0"  # ✅ Allows: 0.3.1, 0.3.24, etc.
                            # ❌ Blocks: 0.4.0, 1.0.0, etc.
```

This ensures you get:
- ✅ **Bug fixes** (patch versions like 0.3.1 → 0.3.2)
- ✅ **Security updates**
- ❌ **No breaking API changes** (minor version bumps like 0.3.x → 0.4.x are blocked)

---

## 📋 **Safe Upgrade Procedure**

### **Step 1: Check for Available Updates**

```bash
# Activate your virtual environment
source venv/bin/activate

# Check what updates are available (doesn't install)
pip list --outdated | grep langchain
```

**Example output:**
```
langchain      0.3.24  0.3.30  wheel
langchain-core 0.3.55  0.3.60  wheel
```

### **Step 2: Run Compatibility Tests (CURRENT VERSION)**

**Before upgrading**, verify tests pass with current versions:

```bash
# Run compatibility tests
pytest tests/test_langchain_compatibility.py -v

# Should output:
# ======================== 28 passed, 1 warning in 0.96s =========================
```

✅ **All tests should pass** - this is your baseline.

### **Step 3: Review Release Notes**

Check LangChain release notes for breaking changes:

```bash
# Visit release notes
open https://github.com/langchain-ai/langchain/releases
```

**Look for**:
- ⚠️ **"BREAKING CHANGE"** labels
- ⚠️ **Deprecated API** warnings
- ⚠️ **Migration guides**

**Common breaking changes to watch for:**
- Import path changes (e.g., `langchain.x` → `langchain_core.x`)
- Method signature changes
- Removed deprecated methods
- New required parameters

### **Step 4: Create a Test Branch**

```bash
# Create a branch for testing the upgrade
git checkout -b test/langchain-upgrade

# Or if not using git, make a backup
cp -r . ../langchain-rag-pipeline-backup
```

### **Step 5: Upgrade to Latest PATCH Version**

Upgrade within the same minor version (safest):

```bash
# Upgrade to latest 0.3.x versions
pip install --upgrade 'langchain>=0.3.0,<0.4.0' \
                      'langchain-core>=0.3.0,<0.4.0' \
                      'langchain-community>=0.3.0,<0.4.0' \
                      'langchain-openai>=0.3.0,<0.4.0' \
                      'langchain-ollama>=0.3.0,<0.4.0'
```

### **Step 6: Run Compatibility Tests (NEW VERSION)**

```bash
# Run compatibility tests with new versions
pytest tests/test_langchain_compatibility.py -v
```

**Expected outcomes:**

#### ✅ **All tests pass** → Safe to proceed
```bash
======================== 28 passed in 0.96s =========================
```

#### ⚠️ **Some tests fail** → Breaking changes detected!

**Example failure:**
```
FAILED tests/test_langchain_compatibility.py::test_document_class_exists
E   ModuleNotFoundError: No module named 'langchain.docstore.document'
```

**Action**: Check release notes for migration path.

### **Step 7: Run Your Application Tests**

```bash
# Run your unit tests
pytest tests/unit/ -v

# Run your integration tests
pytest tests/integration/ -v

# Start your application and test manually
streamlit run src/langchain_rag/ui/streamlit_app.py
```

### **Step 8: Test MCP Server**

```bash
# Test MCP server
python mcp_rag_server_fastmcp.py

# In another terminal, test the tools
# (Your test procedure here)
```

### **Step 9: Update Version Pin (if successful)**

If everything works, update `pyproject.toml`:

```toml
# Before:
"langchain>=0.3.0,<0.4.0",

# After (if upgrading to 0.3.30):
# Update the comment with new tested version
# Last compatibility test: 2025-01-XX (run pytest tests/test_langchain_compatibility.py)
"langchain>=0.3.0,<0.4.0",  # Tested up to 0.3.30
```

### **Step 10: Commit and Deploy**

```bash
# Freeze exact versions for reproducibility

# Commit changes
git add pyproject.toml
git commit -m "chore: upgrade LangChain to 0.3.30 (tested)"

# Merge to main
git checkout main
git merge test/langchain-upgrade
```

---

## 🚀 **Upgrading to MAJOR/MINOR Versions**

When upgrading to a new minor version (e.g., 0.3.x → 0.4.x), **more caution is needed**:

### **Step 1: Research Migration Path**

```bash
# Check official migration guide
open https://python.langchain.com/docs/versions/migration
```

### **Step 2: Update Version Pin**

```toml
[project]
dependencies = [
    # Temporarily allow new version for testing
    "langchain>=0.3.0,<0.5.0",  # Testing 0.4.x upgrade
]
```

### **Step 3: Run Compatibility Tests**

```bash
pytest tests/test_langchain_compatibility.py -v
```

**If tests fail**, you'll see exactly which imports broke:

```
FAILED test_document_class_exists
E   ModuleNotFoundError: No module named 'langchain.docstore.document'
```

### **Step 4: Fix Import Paths**

**Example fix for import path change:**

```python
# OLD (0.3.x):
from langchain.docstore.document import Document

# NEW (0.4.x - hypothetical):
from langchain_core.documents import Document
```

**Update your code**:
1. Fix the import in your compatibility test
2. Fix the import in your actual code
3. Re-run tests

### **Step 5: Update Compatibility Tests**

When imports change, update the test:

```python
# tests/test_langchain_compatibility.py

def test_document_class_exists(self):
    """Verify Document class is importable."""
    try:
        # UPDATED import path for 0.4.x
        from langchain_core.documents import Document  # ← Changed
    except ImportError as e:
        pytest.fail(f"Failed to import Document: {e}")

    # Rest of test unchanged...
```

### **Step 6: Test Thoroughly**

- Run ALL tests (compatibility + unit + integration)
- Test manually with Streamlit UI
- Test MCP server
- Test all RAG chain types

### **Step 7: Document Changes**

Update this guide with breaking changes:

```markdown
## Breaking Changes Log

### LangChain 0.3.x → 0.4.x

- **Import change**: `langchain.docstore.document.Document` → `langchain_core.documents.Document`
- **Action taken**: Updated imports in `loader.py`, `splitter.py`, `ensemble.py`
- **Test status**: ✅ All compatibility tests passing
```

---

## 🛠️ **Troubleshooting**

### **Issue: Import fails after upgrade**

```python
ModuleNotFoundError: No module named 'langchain.xxx'
```

**Solution**:
1. Check release notes for import path changes
2. Update import in compatibility test first
3. Update import in your code
4. Run tests to verify

### **Issue: Method signature changed**

```python
TypeError: __init__() got an unexpected keyword argument 'xxx'
```

**Solution**:
1. Check release notes for parameter changes
2. Update compatibility test to verify new signature
3. Update your code to use new parameters
4. Add test case for backward compatibility if needed

### **Issue: Deprecated method**

```python
DeprecationWarning: xxx is deprecated, use yyy instead
```

**Solution**:
1. Update to use new method (follow warning guidance)
2. Test thoroughly
3. Add migration note to this guide

---

## 📊 **Compatibility Test Coverage**

Our test suite covers these critical imports:

### **Core Models** (5 tests)
- ✅ `Document` class
- ✅ `BaseChatModel` interface
- ✅ Message types (`HumanMessage`, `AIMessage`)

### **LLM Providers** (4 tests)
- ✅ `ChatOpenAI`
- ✅ `ChatOllama`
- ✅ `OpenAIEmbeddings`
- ✅ `OllamaEmbeddings`

### **Retrievers** (3 tests)
- ✅ `EnsembleRetriever`
- ✅ `BM25Retriever`
- ✅ `BaseRetriever`

### **Vector Stores** (3 tests)
- ✅ `FAISS`
- ✅ `Chroma`
- ✅ `VectorStoreRetriever`

### **Document Loaders** (5 tests)
- ✅ `TextLoader`
- ✅ `PyPDFLoader`
- ✅ `CSVLoader`
- ✅ `WebBaseLoader`
- ✅ `WikipediaLoader`

### **Utilities** (3 tests)
- ✅ `RecursiveCharacterTextSplitter`
- ✅ `dumps`/`loads` serialization
- ✅ `langchain.hub`

### **Runnables** (2 tests)
- ✅ `Runnable` interfaces
- ✅ `RunnableWithMessageHistory`

### **Prompts & Parsers** (2 tests)
- ✅ `ChatPromptTemplate`
- ✅ `StrOutputParser`

### **Memory** (2 tests)
- ✅ `ChatMessageHistory`
- ✅ `BaseChatMessageHistory`

**Total**: 28 compatibility tests

---

## 🎯 **Best Practices**

1. ✅ **Always run compatibility tests before upgrading**
2. ✅ **Test on a branch/backup first**
3. ✅ **Read release notes thoroughly**
4. ✅ **Upgrade patch versions frequently (low risk)**
5. ✅ **Upgrade minor versions cautiously (medium risk)**
6. ✅ **Upgrade major versions rarely (high risk)**
7. ✅ **Document any breaking changes you encounter**
8. ✅ **Keep pyproject.toml as the single source of dependency truth**

---

## 📅 **Upgrade Schedule**

**Recommended schedule:**

| Type | Example | Frequency | Risk | Procedure |
|------|---------|-----------|------|-----------|
| **Patch** | 0.3.24 → 0.3.30 | Monthly | 🟢 Low | Steps 1-10 |
| **Minor** | 0.3.x → 0.4.x | Quarterly | 🟡 Medium | Full migration guide |
| **Major** | 0.x → 1.x | Yearly | 🔴 High | Extensive testing |

---

## 🔗 **Resources**

- **LangChain Releases**: https://github.com/langchain-ai/langchain/releases
- **Migration Guides**: https://python.langchain.com/docs/versions/migration
- **Breaking Changes**: https://github.com/langchain-ai/langchain/issues?q=label%3A%22breaking+change%22
- **Our Compatibility Tests**: `tests/test_langchain_compatibility.py`
- **Our Issue Tracker**: (Your GitHub issues URL)

---

## 📝 **Changelog**

### 2025-01-XX - Initial Version
- Created upgrade guide
- Pinned LangChain to 0.3.x
- Added 28 compatibility tests
- All tests passing ✅
