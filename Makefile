.PHONY: help install install-dev test test-cov lint format clean build docker-build docker-run docs serve-docs

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
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  docs          Build documentation"
	@echo "  serve-docs    Serve documentation locally"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/langchain_rag --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

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

docker-build:
	docker build -f docker/Dockerfile -t langchain-rag-pipeline .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

docs:
	mkdocs build

serve-docs:
	mkdocs serve

# Development shortcuts
dev-setup: clean install-dev
	@echo "Development environment ready!"

run-streamlit:
	streamlit run src/langchain_rag/ui/streamlit_app.py

run-mcp-server:
	python src/langchain_rag/mcp/mcp_server_fastmcp.py

run-chat:
	python src/langchain_rag/ui/chat_interface.py

# Docker development
docker-dev:
	docker-compose -f docker/docker-compose.yml --profile dev up -d

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Quality checks
check: lint test
	@echo "All quality checks passed!"

# Release preparation
prepare-release: clean format lint test build
	@echo "Package ready for release!" 