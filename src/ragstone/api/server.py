"""REST API for the RAG pipeline, built on FastAPI.

The production entry point (`ragstone-api` console script): the same
pipeline lifecycle the MCP server exposes to agents, served over HTTP for
applications, plus liveness/readiness endpoints for orchestrators.

Production behaviors:
- Blocking pipeline work runs in worker threads, never on the event loop.
- Optional API-key auth: set RAGSTONE_API_KEY and clients must send it in
  the X-API-Key header (health/readiness stay open for probes).
- A concurrency cap on /ask: when all slots are busy the server answers
  429 immediately instead of queueing until it collapses.
- Typed pipeline errors map to precise status codes with user-safe
  messages; anything else is a generic 500 (details go to the server log).

Run it:

    pip install "ragstone[api]"
    ragstone-api                       # binds 127.0.0.1:8000 by default
"""

import hmac
import json
import logging
import os
import threading
from typing import Iterator, List, Literal, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "The REST API requires the 'api' extra. "
        "Install it with: pip install 'ragstone[api]'"
    ) from exc

from anyio import to_thread

from ragstone.config.settings import get_config
from ragstone.rag.pipeline import OllamaPipeline, OpenAIPipeline
from ragstone.utils.exceptions import PipelineError, ValidationError
from ragstone.utils.registry import (
    Pipeline,
    get_pipeline,
    pop_pipeline,
    put_pipeline,
    snapshot_pipelines,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Request/response schemas
# --------------------------------------------------------------------------


class CreatePipelineRequest(BaseModel):
    provider: Literal["openai", "ollama"] = "openai"
    model: Optional[str] = Field(
        default=None,
        description="Model name; defaults to gpt-4o-mini (openai) or llama3 (ollama).",
    )
    pipeline_id: str = "default"


class LoadDocumentsRequest(BaseModel):
    data_dir: str = "data"
    page_urls: List[str] = Field(default_factory=list)
    wiki_query: Optional[str] = None


class SetupRetrieverRequest(BaseModel):
    use_ensemble: bool = True
    chain_type: Literal["simple", "multi_query", "fusion", "agent", "corrective"] = (
        "simple"
    )
    use_reranker: bool = False


class AskRequest(BaseModel):
    question: str
    session_id: str = "api_session"
    stream: bool = False
    use_cache: bool = True


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(
    *,
    api_key: Optional[str] = None,
    max_concurrency: Optional[int] = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        api_key: Require this key in the X-API-Key header. Defaults to the
            RAGSTONE_API_KEY env var; empty means no auth (development).
        max_concurrency: Simultaneous /ask requests before the server
            answers 429. Defaults to RAGSTONE_API_MAX_CONCURRENCY (8).
    """
    key = api_key if api_key is not None else os.getenv("RAGSTONE_API_KEY", "")
    cap = (
        max_concurrency
        if max_concurrency is not None
        else int(os.getenv("RAGSTONE_API_MAX_CONCURRENCY", "8"))
    )
    # Bounded, non-blocking: a full server refuses new /ask work instead of
    # queueing it behind an unbounded backlog.
    ask_slots = threading.BoundedSemaphore(cap) if cap > 0 else None

    app = FastAPI(title="Ragstone", version="2.0.0")
    app.state.ask_slots = ask_slots  # exposed for tests/inspection

    def _require_key(request: Request) -> None:
        # compare_digest: constant-time comparison, so response timing
        # cannot be used to guess the key byte by byte.
        provided = request.headers.get("x-api-key") or ""
        if key and not hmac.compare_digest(provided, key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.exception_handler(PipelineError)
    async def _pipeline_error_handler(request: Request, exc: PipelineError):
        # Typed pipeline errors carry user-safe messages (the same contract
        # the MCP server relies on). Anything untyped falls through to
        # FastAPI's generic 500 handler, which reveals nothing.
        logger.error(f"{request.url.path} failed: {exc}", exc_info=True)
        status = 422 if isinstance(exc, ValidationError) else 500
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    # -- probes (unauthenticated: orchestrators don't carry API keys) ------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        pipelines = snapshot_pipelines()
        ready_ids = [pid for pid, p in pipelines if p.get_chain() is not None]
        if not ready_ids:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "detail": "No pipeline with a RAG chain."},
            )
        return {"ready": True, "pipelines": ready_ids}

    # -- pipeline lifecycle -------------------------------------------------

    @app.post("/pipelines", status_code=201)
    async def create_pipeline(body: CreatePipelineRequest, request: Request):
        _require_key(request)
        pipeline: Pipeline
        if body.provider == "openai":
            model = body.model or "gpt-4o-mini"
            pipeline = OpenAIPipeline(model=model)
        else:
            model = body.model or "llama3"
            pipeline = OllamaPipeline(model=model)
        put_pipeline(body.pipeline_id, pipeline)
        return {
            "pipeline_id": body.pipeline_id,
            "provider": body.provider,
            "model": model,
        }

    @app.get("/pipelines")
    async def list_pipelines(request: Request):
        _require_key(request)
        result = []
        for pipeline_id, pipeline in snapshot_pipelines():
            result.append(
                {
                    "pipeline_id": pipeline_id,
                    "provider": (
                        "openai" if isinstance(pipeline, OpenAIPipeline) else "ollama"
                    ),
                    "model": (pipeline.LLM.get_model_name() if pipeline.LLM else None),
                    "documents_loaded": bool(pipeline.texts),
                    "chain_ready": pipeline.get_chain() is not None,
                }
            )
        return {"pipelines": result}

    @app.delete("/pipelines/{pipeline_id}")
    async def delete_pipeline(pipeline_id: str, request: Request):
        _require_key(request)
        # Pop atomically first so no other request can look it up mid-cleanup.
        pipeline = pop_pipeline(pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        if getattr(pipeline, "vector_db", None):
            pipeline.vector_db.cleanup()
        return {"deleted": pipeline_id}

    @app.post("/pipelines/{pipeline_id}/documents")
    async def load_documents(
        pipeline_id: str, body: LoadDocumentsRequest, request: Request
    ):
        _require_key(request)
        pipeline = _get_or_404(pipeline_id)
        # Scrapes and embeds — far too slow for the event loop.
        texts = await to_thread.run_sync(
            lambda: pipeline.load_and_split(
                data_dir=body.data_dir,
                page_urls=body.page_urls or None,
                wiki_query=body.wiki_query,
            )
        )
        return {"pipeline_id": pipeline_id, "chunks": len(texts) if texts else 0}

    @app.post("/pipelines/{pipeline_id}/retriever")
    async def setup_retriever(
        pipeline_id: str, body: SetupRetrieverRequest, request: Request
    ):
        _require_key(request)
        pipeline = _get_or_404(pipeline_id)

        def _configure():
            if isinstance(pipeline, OpenAIPipeline):
                pipeline.set_retriever_openai(
                    use_ensemble=body.use_ensemble, use_reranker=body.use_reranker
                )
            else:
                pipeline.set_retriever_ollama(
                    use_ensemble=body.use_ensemble, use_reranker=body.use_reranker
                )
            pipeline.create_rag_chain(chain_type=body.chain_type)

        await to_thread.run_sync(_configure)
        return {
            "pipeline_id": pipeline_id,
            "chain_type": body.chain_type,
            "use_ensemble": body.use_ensemble,
            "use_reranker": body.use_reranker,
        }

    # -- asking -------------------------------------------------------------

    @app.post("/pipelines/{pipeline_id}/ask")
    async def ask(pipeline_id: str, body: AskRequest, request: Request):
        _require_key(request)
        pipeline = _get_or_404(pipeline_id)

        if ask_slots is not None and not ask_slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429,
                detail="Server is at capacity; retry shortly.",
            )
        try:
            if body.stream:
                # The generator owns the slot and releases it when the
                # stream finishes (or the client disconnects).
                return StreamingResponse(
                    _sse_stream(pipeline, body, ask_slots),
                    media_type="text/event-stream",
                )
            answer = await to_thread.run_sync(
                lambda: pipeline.ask_question(
                    body.question,
                    session_id=body.session_id,
                    use_cache=body.use_cache,
                )
            )
            return {"answer": answer, "session_id": body.session_id}
        finally:
            if not body.stream and ask_slots is not None:
                ask_slots.release()

    def _get_or_404(pipeline_id: str):
        pipeline = get_pipeline(pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return pipeline

    return app


def _sse_stream(pipeline, body: AskRequest, ask_slots) -> Iterator[str]:
    """Yield the answer as Server-Sent Events; Starlette iterates this sync
    generator in a worker thread, keeping the event loop free."""
    try:
        for chunk in pipeline.ask_question_stream(
            body.question, session_id=body.session_id, use_cache=body.use_cache
        ):
            if isinstance(chunk, dict):
                # Progress events (e.g. the agent chain's live searches)
                # become named SSE events, distinct from answer data.
                name = chunk.get("event", "progress")
                yield f"event: {name}\ndata: {json.dumps(chunk)}\n\n"
            else:
                yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except PipelineError as exc:
        # Mid-stream failures can't change the status code anymore; emit a
        # terminal error event so clients can distinguish truncation.
        logger.error(f"Stream failed: {exc}", exc_info=True)
        yield f"event: error\ndata: {exc}\n\n"
    finally:
        if ask_slots is not None:
            ask_slots.release()


def main() -> None:
    """Entry point for the ragstone-api console script."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    get_config()  # fail fast on invalid configuration
    host = os.getenv("RAGSTONE_API_HOST", "127.0.0.1")
    port = int(os.getenv("RAGSTONE_API_PORT", "8000"))
    logger.info(f"Starting Ragstone API on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
