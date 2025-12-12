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

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LangChain Compatibility Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the project root
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}pytest not found. Installing...${NC}"
    pip install pytest pytest-cov -q
fi

# Display current LangChain versions
echo -e "${BLUE}Current LangChain Versions:${NC}"
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
" || echo -e "${RED}  Could not detect versions${NC}"

echo ""

# Run compatibility tests
echo -e "${YELLOW}Running compatibility tests...${NC}"
echo ""

if pytest tests/test_langchain_compatibility.py -v --tb=short; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ ALL COMPATIBILITY TESTS PASSED${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Your LangChain installation is ${GREEN}COMPATIBLE${NC} with your codebase."
    echo "It's safe to use this version in production."
    echo ""
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ COMPATIBILITY TESTS FAILED${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Breaking changes detected!${NC}"
    echo ""
    echo "Possible causes:"
    echo "  1. LangChain API has changed"
    echo "  2. Import paths have moved"
    echo "  3. Required parameters have changed"
    echo ""
    echo "Recommended actions:"
    echo "  1. Check LangChain release notes:"
    echo "     https://github.com/langchain-ai/langchain/releases"
    echo "  2. Review the upgrade guide:"
    echo "     docs/LANGCHAIN_UPGRADE_GUIDE.md"
    echo "  3. Fix failing tests in tests/test_langchain_compatibility.py"
    echo "  4. Update your code to match new API"
    echo ""
    exit 1
fi
