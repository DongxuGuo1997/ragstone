#!/bin/bash
#
# LangChain Compatibility Checker
#
# This script checks if your LangChain installation is compatible with your codebase.
# Run this BEFORE upgrading LangChain to catch potential breaking changes.
#
# Usage:
#   ./scripts/check_langchain_compatibility.sh
#
# Exit codes:
#   0 - All tests passed, safe to use current LangChain version
#   1 - Some tests failed, breaking changes detected
#

set -e  # Exit on error

echo "========================================"
echo "LangChain Compatibility Check"
echo "========================================"
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest pytest-cov -q
fi

# Display current LangChain versions
echo "Current LangChain Versions:"
python -c "
try:
    import langchain
    print(f'  langchain: {langchain.__version__}')
except: pass

try:
    import langchain_core
    print(f'  langchain-core: {langchain_core.__version__}')
except: pass

try:
    import langchain_community
    print(f'  langchain-community: {langchain_community.__version__}')
except: pass

try:
    import langchain_openai
    print(f'  langchain-openai: {langchain_openai.__version__}')
except: pass

try:
    import langchain_ollama
    print(f'  langchain-ollama: {langchain_ollama.__version__}')
except: pass
" || echo "  Could not detect versions"

echo ""

# Run compatibility tests
echo "Running compatibility tests..."
echo ""

if pytest tests/test_langchain_compatibility.py -v --tb=short; then
    echo ""
    echo "========================================"
    echo "ALL COMPATIBILITY TESTS PASSED"
    echo "========================================"
    echo ""
    echo "Your LangChain installation is COMPATIBLE with your codebase."
    echo "It's safe to use this version."
    echo ""
    exit 0
else
    echo ""
    echo "========================================"
    echo "COMPATIBILITY TESTS FAILED"
    echo "========================================"
    echo ""
    echo "Breaking changes detected!"
    echo ""
    echo "Possible causes:"
    echo "  1. LangChain API has changed"
    echo "  2. Import paths have moved"
    echo "  3. Required parameters have changed"
    echo ""
    echo "Recommended actions:"
    echo "  1. Check LangChain release notes:"
    echo "     https://github.com/langchain-ai/langchain/releases"
    echo "  2. Review the upgrade notes in CONTRIBUTING.md (Upgrading LangChain)"
    echo "  3. Fix failing tests in tests/test_langchain_compatibility.py"
    echo "  4. Update your code to match new API"
    echo ""
    exit 1
fi
