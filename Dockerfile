# Ragstone API image (optional deployment path — local dev stays venv-based).
#
#   docker build -t ragstone-api .
#   docker compose up          # full stack: API + Qdrant + Postgres/pgvector
#
# The image contains no secrets: OPENAI_API_KEY and friends come from the
# environment at run time (see docker-compose.yml). .env is excluded by
# .dockerignore and must never be baked in.

FROM python:3.12-slim

# faiss needs libgomp at runtime; everything else ships as wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so code edits don't bust the pip cache.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,qdrant,pgvector]"

# Non-root: the API needs no privileges, and store/ must stay writable.
RUN useradd --create-home ragstone \
    && mkdir -p /app/store /app/data \
    && chown -R ragstone:ragstone /app
USER ragstone

# Containers must bind all interfaces; the host publishes the port.
ENV RAGSTONE_API_HOST=0.0.0.0 \
    RAGSTONE_API_PORT=8000

EXPOSE 8000

# Liveness only — readiness (/ready) needs documents loaded, which is a
# client action, so it belongs to the orchestrator's checks, not this one.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["ragstone-api"]
