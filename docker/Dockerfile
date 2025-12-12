# Multi-stage build for LangChain RAG Pipeline
FROM python:3.11-slim as builder

# Set build arguments
ARG POETRY_VERSION=1.7.1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set poetry configuration
RUN poetry config virtualenvs.create false

# Copy dependency files
WORKDIR /app
COPY requirements.txt .
COPY pyproject.toml* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    USER_AGENT="Docker-MCP-Server/2.0" \
    MCP_LOG_LEVEL=INFO \
    VECTOR_STORE_TYPE=faiss

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code with new structure
COPY src/ ./src/
COPY configs/ ./configs/
COPY requirements.txt ./
COPY pyproject.toml ./
COPY setup.py ./
COPY README.md ./

# Install package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data store vs_data logs \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from langchain_rag.rag.pipeline import Pipeline; print('OK')" || exit 1

# Expose port for MCP server
EXPOSE 8000

# Default command using new package structure
CMD ["python", "-m", "langchain_rag.api.mcp_server_fastmcp"] 