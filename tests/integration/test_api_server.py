"""
End-to-end tests for the REST API (no network required).

A real HTTP client (Starlette TestClient) talks to the real FastAPI app;
pipelines are stubbed, so what is under test is the serving layer itself:
routing, auth, readiness, SSE streaming, the concurrency cap, and error
mapping.
"""

import re

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import ragstone.api.server as server  # noqa: E402
from ragstone.utils import registry  # noqa: E402
from ragstone.utils.exceptions import (  # noqa: E402
    ChainExecutionError,
    ValidationError,
)
from ragstone.utils.observability import current_request_id  # noqa: E402


class _StubPipeline:
    def __init__(self, model="stub-model"):
        self.model = model
        self.texts = None
        self._chain = None
        self._chain_type = None
        self.vector_db = None
        self.llm_proxy = self
        self.closed = False

    def get_model_name(self):
        return self.model

    def load_and_split(self, data_dir=None, page_urls=None, wiki_query=None):
        from langchain_core.documents import Document

        # Real Documents, like the real pipeline: the persistence layer
        # serializes page_content + metadata.
        self.texts = [
            Document(page_content="chunk-0", metadata={"source": "stub"}),
            Document(page_content="chunk-1", metadata={"source": "stub"}),
        ]
        return self.texts

    def setup_retriever(self, use_ensemble=True, use_reranker=False):
        self._use_ensemble = use_ensemble
        self._use_reranker = use_reranker

    def create_rag_chain(self, chain_type="simple"):
        self._chain = object()
        self._chain_type = chain_type

    def get_chain(self):
        return self._chain

    def ask_question(self, question, session_id=None, use_cache=True):
        if session_id:
            self.sessions = getattr(self, "sessions", set())
            self.sessions.add(session_id)
        return f"answer to: {question}"

    def ask_question_stream(self, question, session_id=None, use_cache=True):
        yield from ["streamed ", "answer"]

    def delete_session(self, session_id):
        sessions = getattr(self, "sessions", set())
        existed = session_id in sessions
        sessions.discard(session_id)
        return existed

    def close(self):
        self.closed = True


class _StubOpenAIPipeline(_StubPipeline):
    provider = "openai"


class _StubOllamaPipeline(_StubPipeline):
    provider = "ollama"


def _stub_build_pipeline(provider, model=None):
    cls = _StubOpenAIPipeline if provider == "openai" else _StubOllamaPipeline
    return cls(model=model or "stub-model")


@pytest.fixture(autouse=True)
def stub_pipelines(monkeypatch):
    monkeypatch.setattr(server, "build_pipeline", _stub_build_pipeline)
    registry.clear_pipelines()
    yield
    registry.clear_pipelines()


@pytest.fixture
def client():
    return TestClient(server.create_app(api_key="", max_concurrency=2))


def _create_ready_pipeline(client, pipeline_id="p1"):
    client.post("/pipelines", json={"pipeline_id": pipeline_id})
    client.post(f"/pipelines/{pipeline_id}/documents", json={})
    client.post(f"/pipelines/{pipeline_id}/retriever", json={})


class TestLifecycle:
    def test_health_is_always_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_ready_reflects_chain_state(self, client):
        assert client.get("/ready").status_code == 503  # nothing configured
        _create_ready_pipeline(client)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"ready": True, "pipelines": ["p1"]}

    def test_full_lifecycle(self, client):
        response = client.post(
            "/pipelines", json={"pipeline_id": "p1", "model": "gpt-4o-mini"}
        )
        assert response.status_code == 201
        assert response.json()["model"] == "gpt-4o-mini"

        response = client.post("/pipelines/p1/documents", json={})
        assert response.json() == {"pipeline_id": "p1", "chunks": 2}

        response = client.post("/pipelines/p1/retriever", json={"chain_type": "simple"})
        assert response.json()["chain_type"] == "simple"

        response = client.post("/pipelines/p1/ask", json={"question": "capital?"})
        assert response.json()["answer"] == "answer to: capital?"

        listed = client.get("/pipelines").json()["pipelines"]
        assert listed[0]["pipeline_id"] == "p1"
        assert listed[0]["chain_ready"] is True

        assert client.delete("/pipelines/p1").json() == {"deleted": "p1"}
        assert client.get("/pipelines").json() == {"pipelines": []}

    def test_unknown_pipeline_is_404(self, client):
        for method, path, body in [
            ("post", "/pipelines/ghost/documents", {}),
            ("post", "/pipelines/ghost/retriever", {}),
            ("post", "/pipelines/ghost/ask", {"question": "q"}),
            ("delete", "/pipelines/ghost", None),
        ]:
            response = getattr(client, method)(
                path, **({"json": body} if body is not None else {})
            )
            assert response.status_code == 404, path


class TestStreaming:
    def test_sse_stream_yields_chunks_and_done(self, client):
        _create_ready_pipeline(client)
        response = client.post(
            "/pipelines/p1/ask", json={"question": "capital?", "stream": True}
        )
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: streamed \n\n" in response.text
        assert "data: answer\n\n" in response.text
        assert response.text.endswith("data: [DONE]\n\n")

    def test_progress_events_become_named_sse_events(self, client):
        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _eventing_stream(question, session_id=None, use_cache=True):
            yield {"event": "search", "query": "refined query"}
            yield "answer text"

        pipeline.ask_question_stream = _eventing_stream
        response = client.post(
            "/pipelines/p1/ask", json={"question": "q", "stream": True}
        )
        assert 'event: search\ndata: {"event": "search", "query": "refined query"}' in (
            response.text
        )
        assert "data: answer text\n\n" in response.text
        assert response.text.endswith("data: [DONE]\n\n")

    def test_stream_releases_concurrency_slot(self, client):
        _create_ready_pipeline(client)
        for _ in range(5):  # more requests than the cap of 2
            response = client.post(
                "/pipelines/p1/ask", json={"question": "q", "stream": True}
            )
            assert response.status_code == 200

    def test_streamed_request_with_real_instrumentation_completes(self, client):
        # Regression: the pipeline wraps streaming in track_request, whose
        # contextvar enter/exit must share a context. Under Starlette each
        # response chunk used to be pumped through a fresh copied context,
        # so teardown raised ValueError and [DONE] (plus the metrics log
        # line) was lost. The stub here drives the REAL track_request.
        from ragstone.utils.observability import track_request

        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _instrumented_stream(question, session_id=None, use_cache=True):
            with track_request(session_id or "s", chain_type="simple"):
                yield "instrumented "
                yield "answer"

        pipeline.ask_question_stream = _instrumented_stream
        response = client.post(
            "/pipelines/p1/ask", json={"question": "q", "stream": True}
        )
        assert "data: instrumented \n\n" in response.text
        assert response.text.endswith("data: [DONE]\n\n")

    def test_multiline_chunks_are_framed_per_sse_line(self, client):
        # A literal newline inside one `data:` line makes spec-compliant
        # clients drop the continuation; each physical line must get its
        # own `data:` prefix (clients rejoin them with newlines).
        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _multiline_stream(question, session_id=None, use_cache=True):
            yield "- first\n- second"

        pipeline.ask_question_stream = _multiline_stream
        response = client.post(
            "/pipelines/p1/ask", json={"question": "q", "stream": True}
        )
        assert "data: - first\ndata: - second\n\n" in response.text
        assert response.text.endswith("data: [DONE]\n\n")


class TestMetricsEndpoint:
    """/metrics must expose real request telemetry: a request that runs
    track_request shows up in the scrape with its labels."""

    def test_scrape_reflects_an_instrumented_request(self, client):
        pytest.importorskip("prometheus_client")
        from ragstone.utils.observability import track_request

        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _instrumented(question, session_id=None, use_cache=True):
            with track_request(session_id or "s", chain_type="simple"):
                return "ok"

        pipeline.ask_question = _instrumented
        assert client.post("/pipelines/p1/ask", json={"question": "q"}).status_code == (
            200
        )

        scrape = client.get("/metrics")
        assert scrape.status_code == 200
        assert scrape.headers["content-type"].startswith("text/plain")
        assert (
            'ragstone_requests_total{cache="miss",chain="simple",error="none"}'
            in scrape.text
        )
        assert "ragstone_request_duration_seconds_bucket" in scrape.text

    def test_metrics_stays_open_without_key(self):
        pytest.importorskip("prometheus_client")
        client = TestClient(server.create_app(api_key="sekret"))
        assert client.get("/metrics").status_code == 200


class TestRequestId:
    """One string must trace a request from client to server log: every
    response carries X-Request-ID, and the pipeline (where track_request
    logs it) sees the same value — across the thread pool for plain asks
    and the manual worker thread for SSE."""

    def test_every_response_carries_a_minted_id(self, client):
        response = client.get("/health")
        assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])

    def test_client_id_is_echoed_and_reaches_the_pipeline(self, client):
        _create_ready_pipeline(client)
        seen = {}
        pipeline = registry.get_pipeline("p1")

        def _record(question, session_id=None, use_cache=True):
            seen["request_id"] = current_request_id()
            return "ok"

        pipeline.ask_question = _record
        response = client.post(
            "/pipelines/p1/ask",
            json={"question": "q"},
            headers={"X-Request-ID": "trace-42"},
        )
        assert response.headers["x-request-id"] == "trace-42"
        assert seen["request_id"] == "trace-42"

    def test_sse_worker_thread_sees_the_adopted_id(self, client):
        # The SSE producer runs on a manually created thread; the id must
        # cross that boundary explicitly, not rely on contextvars.
        _create_ready_pipeline(client)
        seen = {}
        pipeline = registry.get_pipeline("p1")

        def _recording_stream(question, session_id=None, use_cache=True):
            seen["request_id"] = current_request_id()
            yield "chunk"

        pipeline.ask_question_stream = _recording_stream
        response = client.post(
            "/pipelines/p1/ask",
            json={"question": "q", "stream": True},
            headers={"X-Request-ID": "trace-sse"},
        )
        assert response.headers["x-request-id"] == "trace-sse"
        assert response.text.endswith("data: [DONE]\n\n")
        assert seen["request_id"] == "trace-sse"

    def test_malformed_client_ids_are_replaced_not_echoed(self, client):
        # The id lands in log lines and a response header; whitespace,
        # over-length, or exotic characters must never round-trip.
        for bad in ("has space", "x" * 129, "semi;colon"):
            response = client.get("/health", headers={"X-Request-ID": bad})
            echoed = response.headers["x-request-id"]
            assert echoed != bad
            assert re.fullmatch(r"[0-9a-f]{32}", echoed)
        # Non-ASCII header bytes (a real client can send latin-1) must be
        # replaced too; httpx only sends them pre-encoded.
        response = client.get(
            "/health", headers={b"x-request-id": "ünïcode".encode("latin-1")}
        )
        assert re.fullmatch(r"[0-9a-f]{32}", response.headers["x-request-id"])

    def test_error_responses_carry_the_id_in_header_and_body(self, client):
        from ragstone.utils.exceptions import ChainExecutionError

        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _fail(question, session_id=None, use_cache=True):
            raise ChainExecutionError("boom")

        pipeline.ask_question = _fail
        response = client.post(
            "/pipelines/p1/ask",
            json={"question": "q"},
            headers={"X-Request-ID": "trace-err"},
        )
        assert response.status_code == 500
        assert response.headers["x-request-id"] == "trace-err"
        assert response.json()["request_id"] == "trace-err"

    def test_framework_error_responses_carry_the_header(self, client):
        # 404s (and 401/429) come from exception handlers, not endpoints —
        # the middleware must stamp those too.
        response = client.delete("/pipelines/ghost")
        assert response.status_code == 404
        assert "x-request-id" in response.headers


class TestAuth:
    def test_requests_without_key_are_401(self):
        client = TestClient(server.create_app(api_key="sekret"))
        assert client.get("/pipelines").status_code == 401
        assert client.post("/pipelines", json={}).status_code == 401

    def test_requests_with_key_pass(self):
        client = TestClient(server.create_app(api_key="sekret"))
        response = client.get("/pipelines", headers={"X-API-Key": "sekret"})
        assert response.status_code == 200

    def test_probes_stay_open_without_key(self):
        client = TestClient(server.create_app(api_key="sekret"))
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503  # open, just not ready


class TestSessionDeletion:
    """DELETE /pipelines/{id}/sessions/{sid} — the right-to-erasure
    endpoint (ROADMAP 5.10)."""

    def test_delete_reports_existence_and_is_idempotent(self, client):
        _create_ready_pipeline(client)
        client.post(
            "/pipelines/p1/ask", json={"question": "q?", "session_id": "erase-me"}
        )

        first = client.delete("/pipelines/p1/sessions/erase-me")
        assert first.status_code == 200
        assert first.json() == {
            "pipeline_id": "p1",
            "session_id": "erase-me",
            "deleted": True,
        }
        # Second delete: still 200, honestly reports nothing existed.
        second = client.delete("/pipelines/p1/sessions/erase-me")
        assert second.status_code == 200
        assert second.json()["deleted"] is False

    def test_unknown_pipeline_is_404(self, client):
        response = client.delete("/pipelines/ghost/sessions/s1")
        assert response.status_code == 404

    def test_requires_api_key_when_auth_enabled(self):
        client = TestClient(server.create_app(api_key="k1", max_concurrency=2))
        response = client.delete("/pipelines/p1/sessions/s1")
        assert response.status_code == 401


class TestPersistence:
    """ROADMAP 5.7 through the HTTP surface: a configured pipeline
    survives a 'restart' (registry cleared, manifests remain) without
    re-ingestion, and DELETE removes the manifest with the pipeline."""

    @pytest.fixture
    def persist_dir(self, monkeypatch, tmp_path):
        import ragstone.rag.pipeline as pipeline_module

        monkeypatch.setenv("RAGSTONE_REGISTRY_PERSIST", "on")
        monkeypatch.setenv("RAGSTONE_REGISTRY_DIR", str(tmp_path))
        # Restore builds through the real factory's import site.
        monkeypatch.setattr(pipeline_module, "build_pipeline", _stub_build_pipeline)
        return tmp_path

    def test_pipeline_survives_restart_without_reingest(self, client, persist_dir):
        _create_ready_pipeline(client)
        registry.clear_pipelines()  # the "restart"

        response = client.post("/pipelines/p1/ask", json={"question": "q"})
        assert response.status_code == 200  # lazily restored, no re-ingest
        assert response.json()["answer"] == "answer to: q"
        restored = registry.get_pipeline("p1")
        assert [d.page_content for d in restored.texts] == ["chunk-0", "chunk-1"]
        assert restored._chain_type == "simple"

    def test_restorable_pipelines_show_in_probes_and_listing(self, client, persist_dir):
        _create_ready_pipeline(client)
        registry.clear_pipelines()

        ready = client.get("/ready")
        assert ready.status_code == 200  # restorable = ready to serve
        assert ready.json()["restorable"] == ["p1"]
        listed = client.get("/pipelines").json()["pipelines"]
        assert listed == [{"pipeline_id": "p1", "state": "persisted"}]

    def test_delete_means_delete_across_restarts(self, client, persist_dir):
        _create_ready_pipeline(client)
        assert client.delete("/pipelines/p1").status_code == 200
        registry.clear_pipelines()
        # The manifest is gone too: nothing resurrects.
        response = client.post("/pipelines/p1/ask", json={"question": "q"})
        assert response.status_code == 404

    def test_delete_reaches_persisted_but_unloaded_pipelines(self, client, persist_dir):
        _create_ready_pipeline(client)
        registry.clear_pipelines()
        assert client.delete("/pipelines/p1").json() == {"deleted": "p1"}
        assert client.get("/pipelines").json() == {"pipelines": []}


class TestGracefulShutdown:
    def test_lifespan_shutdown_closes_pipelines(self):
        # TestClient's context manager drives the lifespan: on exit,
        # uvicorn-equivalent shutdown runs and pipelines must close.
        app = server.create_app(api_key="", max_concurrency=2)
        with TestClient(app) as client:
            _create_ready_pipeline(client)
            pipeline = registry.get_pipeline("p1")
            assert pipeline.closed is False
        assert pipeline.closed is True


class TestNamedKeys:
    """ROADMAP 5.3: named keys, per-key quotas, runtime revocation, the
    audit trail, and per-key usage — through the real HTTP surface."""

    def _client(self, *records):
        from ragstone.api.keys import ApiKeyRecord, ApiKeyStore

        store = ApiKeyStore([ApiKeyRecord(**r) for r in records])
        return TestClient(server.create_app(key_store=store, max_concurrency=2))

    def test_named_keys_authenticate_independently(self):
        client = self._client(
            {"name": "alice", "key": "sk-a"}, {"name": "bob", "key": "sk-b"}
        )
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-a"}).status_code == 200
        )
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-b"}).status_code == 200
        )
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-x"}).status_code == 401
        )

    def test_per_key_quota_returns_429_with_retry_after(self):
        client = self._client(
            {"name": "alice", "key": "sk-a", "rpm": 2}, {"name": "bob", "key": "sk-b"}
        )
        for _ in range(2):
            assert (
                client.get("/pipelines", headers={"X-API-Key": "sk-a"}).status_code
                == 200
            )
        throttled = client.get("/pipelines", headers={"X-API-Key": "sk-a"})
        assert throttled.status_code == 429
        assert throttled.headers["retry-after"] == "60"
        # Alice's limit is alice's problem — bob is unaffected.
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-b"}).status_code == 200
        )

    def test_runtime_revocation(self):
        client = self._client({"name": "alice", "key": "sk-a"})
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-a"}).status_code == 200
        )
        assert client.app.state.api_keys.revoke("alice") is True
        assert (
            client.get("/pipelines", headers={"X-API-Key": "sk-a"}).status_code == 401
        )

    def test_audit_trail_attributes_requests(self, caplog):
        import logging

        client = self._client({"name": "alice", "key": "sk-a", "rpm": 1})
        with caplog.at_level(logging.INFO, logger="ragstone.audit"):
            ok = client.get(
                "/pipelines",
                headers={"X-API-Key": "sk-a", "X-Request-ID": "audit-1"},
            )
            client.get("/pipelines", headers={"X-API-Key": "sk-a"})  # 429
            client.get("/pipelines", headers={"X-API-Key": "sk-wrong"})  # 401
            client.get("/health")  # exempt: probes never hit the trail

        assert ok.status_code == 200
        lines = [r.getMessage() for r in caplog.records if r.name == "ragstone.audit"]
        assert len(lines) == 3
        assert "key=alice method=GET path=/pipelines status=200 request=audit-1" in (
            lines[0]
        )
        assert "key=alice" in lines[1] and "status=429" in lines[1]
        assert "key=anonymous" in lines[2] and "status=401" in lines[2]
        assert not any("/health" in line for line in lines)

    def test_usage_endpoint_reports_attribution_without_key_material(self):
        client = self._client(
            {"name": "alice", "key": "sk-secret-a", "rpm": 60},
            {"name": "bob", "key": "sk-secret-b"},
        )
        for _ in range(3):
            client.get("/pipelines", headers={"X-API-Key": "sk-secret-a"})
        response = client.get("/usage", headers={"X-API-Key": "sk-secret-b"})
        assert response.status_code == 200
        report = {r["name"]: r for r in response.json()["keys"]}
        assert report["alice"]["requests_total"] == 3
        assert report["alice"]["rpm_limit"] == 60
        assert report["bob"]["requests_total"] >= 1  # the /usage call itself
        assert "sk-secret" not in response.text

    def test_usage_requires_a_key_when_auth_is_on(self):
        client = self._client({"name": "alice", "key": "sk-a"})
        assert client.get("/usage").status_code == 401


class TestRobustness:
    def test_full_capacity_returns_429(self, client):
        _create_ready_pipeline(client)
        slots = client.app.state.ask_slots
        slots.acquire()
        slots.acquire()  # occupy both slots
        try:
            response = client.post("/pipelines/p1/ask", json={"question": "q"})
            assert response.status_code == 429
        finally:
            slots.release()
            slots.release()

    def test_validation_error_maps_to_422(self, client):
        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _reject(question, session_id=None, use_cache=True):
            raise ValidationError("Question is too long")

        pipeline.ask_question = _reject
        response = client.post("/pipelines/p1/ask", json={"question": "q"})
        assert response.status_code == 422
        assert "too long" in response.json()["detail"]

    def test_pipeline_error_maps_to_500_with_safe_detail(self, client):
        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _fail(question, session_id=None, use_cache=True):
            raise ChainExecutionError("Failed to generate a response")

        pipeline.ask_question = _fail
        response = client.post("/pipelines/p1/ask", json={"question": "q"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to generate a response"

    def test_slot_released_after_error(self, client):
        # A failing request must not leak its concurrency slot.
        _create_ready_pipeline(client)
        pipeline = registry.get_pipeline("p1")

        def _fail(question, session_id=None, use_cache=True):
            raise ChainExecutionError("boom")

        pipeline.ask_question = _fail
        for _ in range(5):  # would exhaust the cap of 2 if slots leaked
            assert (
                client.post("/pipelines/p1/ask", json={"question": "q"}).status_code
                == 500
            )

    def test_non_ascii_api_key_header_is_401_not_500(self):
        # A real client can send latin-1 header bytes >127, which Starlette
        # decodes into a non-ASCII str; hmac.compare_digest raises TypeError
        # on non-ASCII str operands, so the comparison must run on bytes for
        # garbage headers to reject cleanly instead of erroring.
        app_client = TestClient(server.create_app(api_key="secret"))
        response = app_client.get(
            "/pipelines", headers={b"x-api-key": "s\xe9cret".encode("latin-1")}
        )
        assert response.status_code == 401

    def test_negative_concurrency_cap_is_rejected(self):
        # A negative cap is a misconfiguration; silently disabling the
        # limit would remove the 429 protection without a trace.
        with pytest.raises(ValueError, match="max_concurrency"):
            server.create_app(api_key="", max_concurrency=-1)
