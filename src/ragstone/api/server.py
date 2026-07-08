"""REST API for the RAG pipeline, built on FastAPI.

The production entry point (`ragstone-api` console script): the same
pipeline lifecycle the MCP server exposes to agents, served over HTTP for
applications, plus liveness/readiness endpoints for orchestrators.

Production behaviors:
- Blocking pipeline work runs in worker threads, never on the event loop.
- Every response carries an X-Request-ID (the client's own, if it sent a
  well-formed one) and the pipeline's ragstone.requests log line uses the
  same id, so one string traces a request from client to server log.
- Optional API-key auth: named keys with per-key rate limits via
  RAGSTONE_API_KEYS (legacy single RAGSTONE_API_KEY still works), sent in
  the X-API-Key header (health/readiness stay open for probes). Every
  gated request leaves an append-only audit line — key name, method,
  path, status, request id; never content — and GET /usage reports
  per-key attribution.
- A concurrency cap on /ask: when all slots are busy the server answers
  429 immediately instead of queueing until it collapses.
- Typed pipeline errors map to precise status codes with user-safe
  messages; anything else is a generic 500 (details go to the server log).

Run it:

    pip install "ragstone[api]"
    ragstone-api                       # binds 127.0.0.1:8000 by default
"""

import json
import logging
import os
import queue
import re
import threading
import uuid
from typing import Iterator, List, Literal, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without extra
    raise ImportError(
        "The REST API requires the 'api' extra. "
        "Install it with: pip install 'ragstone[api]'"
    ) from exc

from anyio import to_thread

from ragstone import __version__
from ragstone.api.keys import ApiKeyRecord, ApiKeyStore
from ragstone.config.settings import get_config
from ragstone.rag.pipeline import DEFAULT_MODELS, build_pipeline
from ragstone.utils.exceptions import PipelineError, ValidationError
from ragstone.utils.metrics import init_metrics, render_metrics
from ragstone.utils.observability import current_request_id, use_request_id
from ragstone.utils.registry import (
    get_pipeline,
    pop_pipeline,
    put_pipeline,
    snapshot_pipelines,
)

logger = logging.getLogger(__name__)

# One line per gated request — who, what, outcome — and never content.
# A separate logger so operators can route/retain it independently
# (audit trails usually outlive request logs).
audit_logger = logging.getLogger("ragstone.audit")

# A client-supplied X-Request-ID is adopted only if it looks like an id:
# the value goes into log lines and back out in a response header, so a
# hostile value must not be able to inject either (no whitespace or
# control characters, bounded length). Anything else gets a minted id.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")


class _RequestIDMiddleware:
    """Adopt or mint the request id; stamp it on every response.

    Pure ASGI rather than BaseHTTPMiddleware: the ambient id must be set
    in the same context that runs the endpoint, so it propagates into the
    worker threads anyio spawns (contextvars copy across) and is readable
    by exception handlers. BaseHTTPMiddleware runs downstream in a
    separate task and has documented contextvar-propagation caveats.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        provided = ""
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                provided = value.decode("latin-1")
                break
        request_id = (
            provided if _REQUEST_ID_RE.fullmatch(provided) else uuid.uuid4().hex
        )

        async def send_with_id(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        with use_request_id(request_id):
            await self.app(scope, receive, send_with_id)


# Probes and docs would drown the trail in scrape noise.
_AUDIT_EXEMPT = {"/health", "/ready", "/metrics", "/docs", "/openapi.json"}


class _AuditMiddleware:
    """Append-only audit trail (ROADMAP 5.3): one line per gated request.

    Who = the API key NAME (resolved by auth into request.state), what =
    method + path (the path carries the pipeline/corpus id), when = the
    log timestamp, outcome = status code, plus the request id — and
    never question or document content. Emitted when the response starts,
    so denied requests (401/429) are audited with the same machinery.

    Registered before (= inside) the request-id middleware, so
    current_request_id() is in scope here.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in _AUDIT_EXEMPT:
            await self.app(scope, receive, send)
            return

        async def send_with_audit(message) -> None:
            if message["type"] == "http.response.start":
                who = scope.get("state", {}).get("api_key_name", "anonymous")
                audit_logger.info(
                    "key=%s method=%s path=%s status=%d request=%s",
                    who,
                    scope["method"],
                    scope["path"],
                    message["status"],
                    current_request_id(),
                )
            await send(message)

        await self.app(scope, receive, send_with_audit)


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
    chain_type: Literal[
        "simple", "multi_query", "fusion", "agent", "corrective", "auto"
    ] = "simple"
    use_reranker: bool = False


class AskRequest(BaseModel):
    question: str
    # None = stateless: each request gets a fresh session. The old shared
    # default ("api_session") quietly accumulated ONE conversation across
    # every client: follow-up rephrasing could reinterpret your question
    # against someone else's history, and the response cache was
    # permanently disabled after the first request (history-bearing turns
    # bypass it by design). Pass a session_id explicitly to opt into
    # conversation memory.
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Conversation id for follow-up questions; omit for stateless "
            "one-shot asks (fresh session per request)."
        ),
    )
    stream: bool = False
    use_cache: bool = True


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(
    *,
    api_key: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    key_store: Optional[ApiKeyStore] = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        api_key: Back-compat single key for the X-API-Key header (becomes
            the name "default"). None reads the environment
            (RAGSTONE_API_KEYS + RAGSTONE_API_KEY); empty string means no
            auth (development).
        max_concurrency: Simultaneous /ask requests before the server
            answers 429. Defaults to RAGSTONE_API_MAX_CONCURRENCY (8).
        key_store: Prebuilt ApiKeyStore; overrides api_key and the
            environment (tests, embedders).
    """
    if key_store is not None:
        keys = key_store
    elif api_key is not None:
        keys = ApiKeyStore(
            [ApiKeyRecord(name="default", key=api_key)] if api_key else []
        )
    else:
        keys = ApiKeyStore.from_env()  # fail fast on malformed entries
    cap = (
        max_concurrency
        if max_concurrency is not None
        else int(os.getenv("RAGSTONE_API_MAX_CONCURRENCY", "8"))
    )
    if cap < 0:
        # A negative cap is a misconfiguration, not a request to disable
        # the limit — refusing beats silently removing the protection.
        raise ValueError(
            f"max_concurrency must be >= 0 (0 disables the cap), got {cap}"
        )
    # Bounded, non-blocking: a full server refuses new /ask work instead of
    # queueing it behind an unbounded backlog.
    ask_slots = threading.BoundedSemaphore(cap) if cap > 0 else None

    app = FastAPI(title="Ragstone", version=__version__)
    # Registration order = innermost first: audit runs inside the
    # request-id scope, so its lines carry the same id the client holds.
    app.add_middleware(_AuditMiddleware)
    app.add_middleware(_RequestIDMiddleware)
    app.state.ask_slots = ask_slots  # exposed for tests/inspection
    app.state.api_keys = keys  # exposed for tests/runtime revocation

    def _require_key(request: Request) -> None:
        """Authenticate, attribute, and rate-limit the request."""
        if not keys.enabled:
            return
        # The store compares with hmac.compare_digest per key, full scan:
        # timing reveals neither a matching prefix nor which key matched.
        name = keys.authenticate(request.headers.get("x-api-key") or "")
        if name is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        # Attribute BEFORE the quota check so 429s are audited by name.
        request.state.api_key_name = name
        if not keys.check_and_count(name):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this API key; retry shortly.",
                headers={"Retry-After": "60"},
            )

    @app.exception_handler(PipelineError)
    async def _pipeline_error_handler(request: Request, exc: PipelineError):
        # Typed pipeline errors carry user-safe messages (the same contract
        # the MCP server relies on). Anything untyped falls through to
        # FastAPI's generic 500 handler, which reveals nothing.
        request_id = current_request_id()
        logger.error(
            f"{request.url.path} failed (request={request_id}): {exc}",
            exc_info=True,
        )
        status = 422 if isinstance(exc, ValidationError) else 500
        content = {"detail": str(exc)}
        if request_id:
            # The same id is in the X-Request-ID header; repeating it in
            # the body puts it where clients actually log error payloads.
            content["request_id"] = request_id
        return JSONResponse(status_code=status, content=content)

    # -- probes (unauthenticated: orchestrators don't carry API keys) ------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Registered once per process (init is idempotent); like the probes,
    # /metrics is unauthenticated — scrapers don't carry app keys, and the
    # payload is aggregates only, never question or document content.
    metrics_enabled = init_metrics()

    @app.get("/metrics")
    async def metrics():
        if not metrics_enabled:
            raise HTTPException(
                status_code=503,
                detail="Metrics need prometheus-client: pip install 'ragstone[api]'",
            )
        payload = render_metrics()
        assert payload is not None  # metrics_enabled implies an exporter
        data, content_type = payload
        return Response(content=data, media_type=content_type)

    @app.get("/usage")
    async def usage(request: Request):
        # Per-key attribution for operators: names, counts, and limits —
        # key material never appears. Authenticated (and counted) like
        # any other gated request.
        _require_key(request)
        return {"keys": keys.usage_report()}

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
        pipeline = build_pipeline(body.provider, body.model)
        put_pipeline(body.pipeline_id, pipeline)
        return {
            "pipeline_id": body.pipeline_id,
            "provider": body.provider,
            "model": body.model or DEFAULT_MODELS[body.provider],
        }

    @app.get("/pipelines")
    async def list_pipelines(request: Request):
        _require_key(request)
        result = []
        for pipeline_id, pipeline in snapshot_pipelines():
            result.append(
                {
                    "pipeline_id": pipeline_id,
                    "provider": getattr(pipeline, "provider", "unknown"),
                    "model": (
                        pipeline.llm_proxy.get_model_name()
                        if pipeline.llm_proxy
                        else None
                    ),
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
        if hasattr(pipeline, "close"):
            pipeline.close()  # memory backend + vector store
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
            pipeline.setup_retriever(
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
        # Stateless by default: a fresh session per request unless the
        # client opted into conversation memory with an explicit id.
        session_id = body.session_id or f"api_{uuid.uuid4().hex[:12]}"
        try:
            if body.stream:
                # The stream's worker thread owns the slot and releases it
                # exactly once, when the pipeline call finishes.
                return StreamingResponse(
                    _sse_stream(
                        pipeline, body, ask_slots, session_id, current_request_id()
                    ),
                    media_type="text/event-stream",
                )
            answer = await to_thread.run_sync(
                lambda: pipeline.ask_question(
                    body.question,
                    session_id=session_id,
                    use_cache=body.use_cache,
                )
            )
            return {"answer": answer, "session_id": session_id}
        finally:
            if not body.stream and ask_slots is not None:
                ask_slots.release()

    def _get_or_404(pipeline_id: str):
        pipeline = get_pipeline(pipeline_id)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        return pipeline

    return app


def _sse_data(text: str) -> str:
    """Frame text as an SSE data payload, newline-safe.

    A literal "\\n" inside a single `data:` line splits the payload: the
    continuation lines lack the `data:` prefix, so spec-compliant clients
    drop them. Emit one `data:` line per physical line instead — clients
    rejoin them with newlines per the SSE spec.
    """
    return "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"


def _sse_stream(
    pipeline,
    body: AskRequest,
    ask_slots,
    session_id: str,
    request_id: Optional[str] = None,
) -> Iterator[str]:
    """Yield the answer as Server-Sent Events.

    The pipeline generator runs on ONE dedicated worker thread and hands
    chunks over via a queue. This matters: Starlette pumps a sync response
    generator through a fresh copied context per chunk, which would tear
    the pipeline's context-local instrumentation (track_request, the token
    usage callback) apart — enter and exit must share a context. The worker
    owns the concurrency slot and releases it exactly once, when the
    pipeline call actually finishes. The request id crosses the thread
    boundary explicitly — contextvars don't follow manual threads.
    """
    handoff: "queue.Queue[Tuple[str, object]]" = queue.Queue()
    cancelled = threading.Event()

    def _produce() -> None:
        try:
            with use_request_id(request_id):
                for chunk in pipeline.ask_question_stream(
                    body.question, session_id=session_id, use_cache=body.use_cache
                ):
                    if cancelled.is_set():
                        # Client is gone: stop at the next chunk boundary
                        # so an abandoned request doesn't pin a concurrency
                        # slot (and burn tokens) for a full generation.
                        logger.info("Client disconnected; abandoning stream")
                        break
                    handoff.put(("chunk", chunk))
            handoff.put(("done", None))
        except PipelineError as exc:
            # Mid-stream failures can't change the status code anymore;
            # emit a terminal error event so clients can distinguish a
            # real error from truncation.
            logger.error(f"Stream failed: {exc}", exc_info=True)
            handoff.put(("error", str(exc)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Stream failed unexpectedly: {exc}", exc_info=True)
            handoff.put(("error", "internal error"))
        finally:
            if ask_slots is not None:
                ask_slots.release()

    worker = threading.Thread(target=_produce, daemon=True, name="sse-ask")
    try:
        worker.start()
    except Exception:  # pragma: no cover - thread creation failure
        if ask_slots is not None:
            ask_slots.release()
        raise

    try:
        while True:
            kind, payload = handoff.get()
            if kind == "chunk":
                if isinstance(payload, dict):
                    # Progress events (e.g. the agent chain's live searches)
                    # become named SSE events, distinct from answer data.
                    name = payload.get("event", "progress")
                    yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"
                else:
                    yield _sse_data(str(payload))
            elif kind == "error":
                yield f"event: error\n{_sse_data(str(payload))}"
                return
            else:  # done
                yield "data: [DONE]\n\n"
                return
    finally:
        # Runs on normal completion AND on GeneratorExit when the client
        # disconnects mid-stream — the worker checks this flag per chunk.
        cancelled.set()


def _maybe_setup_otel() -> None:
    """Wire the OTLP exporter when the standard OTel env vars ask for it.

    Instrumentation (spans in observability.py) is always present but
    non-recording until a tracer provider exists; this is the only place
    that creates one. No endpoint configured -> tracing stays off.
    """
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the otel extra is not "
            "installed; tracing disabled. pip install 'ragstone[otel]'"
        )
        return
    # Resource.create and OTLPSpanExporter read the standard OTEL_* env
    # vars themselves (endpoint, headers, OTEL_SERVICE_NAME).
    provider = TracerProvider(resource=Resource.create({"service.name": "ragstone"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled (OTLP/HTTP)")


def main() -> None:
    """Entry point for the ragstone-api console script."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    get_config()  # fail fast on invalid configuration
    _maybe_setup_otel()
    host = os.getenv("RAGSTONE_API_HOST", "127.0.0.1")
    port = int(os.getenv("RAGSTONE_API_PORT", "8000"))
    logger.info(f"Starting Ragstone API on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
