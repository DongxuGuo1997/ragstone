.PHONY: help install install-dev test test-cov eval eval-retrieval lint format clean build dev-setup run-streamlit run-mcp-server run-chat check prepare-release

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install package with development dependencies"
	@echo "  test          Run test suite"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run code linting"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=ragstone --cov-report=html --cov-report=term

eval:
	python evals/run_eval.py --mode full

eval-retrieval:
	python evals/run_eval.py --mode retrieval

lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

# Development shortcuts
dev-setup: clean install-dev
	@echo "Development environment ready!"

run-streamlit:
	streamlit run src/ragstone/ui/streamlit_app.py

run-mcp-server:
	python -m ragstone.mcp.mcp_server_fastmcp

run-chat:
	python -m ragstone.ui.chat_interface

# Quality checks
check: lint test
	@echo "All quality checks passed!"

# Release preparation
prepare-release: clean format lint test build
	@echo "Package ready for release!" 