"""
Unit tests for the resilience layer (no network required).

Covers the three production guards:
1. Retry/timeout config actually reaches the LLM and embedding clients.
2. Oversized/malformed questions are rejected before any API spend.
3. MCP error responses never echo internals from untyped exceptions.
"""

import pytest

from ragstone.config.settings import get_config
from ragstone.mcp.mcp_server_fastmcp import _safe_error
from ragstone.models.base_model import OllamaProxy, OpenAIProxy
from ragstone.rag.embeddings import make_openai_embeddings
from ragstone.rag.rag import extract_question, validate_question
from ragstone.utils.exceptions import (
    ChainExecutionError,
    PipelineError,
    ValidationError,
)


class TestClientResilienceWiring:
    def test_openai_llm_gets_config_retries_and_timeout(self):
        cfg = get_config().llm
        llm = OpenAIProxy().set_llm("gpt-4o-mini")
        assert llm.max_retries == cfg.max_retries
        assert llm.request_timeout == cfg.timeout

    def test_caller_kwargs_override_config_defaults(self):
        llm = OpenAIProxy().set_llm("gpt-4o-mini", max_retries=7, request_timeout=5)
        assert llm.max_retries == 7
        assert llm.request_timeout == 5

    def test_ollama_llm_gets_httpx_timeout(self):
        cfg = get_config().llm
        llm = OllamaProxy().set_llm("llama3")
        assert llm.client_kwargs == {"timeout": cfg.timeout}

    def test_openai_embeddings_get_config_retries_and_timeout(self):
        cfg = get_config().llm
        embeddings = make_openai_embeddings()
        assert embeddings.max_retries == cfg.max_retries
        assert embeddings.request_timeout == cfg.timeout


class TestQuestionGuards:
    def test_oversized_question_rejected(self, monkeypatch):
        monkeypatch.setattr(get_config().llm, "max_question_chars", 100)
        with pytest.raises(ValidationError, match="too long"):
            validate_question("x" * 101)

    def test_question_at_limit_accepted(self, monkeypatch):
        monkeypatch.setattr(get_config().llm, "max_question_chars", 100)
        assert validate_question("x" * 100) == "x" * 100

    def test_empty_and_non_string_rejected(self):
        with pytest.raises(ValidationError):
            validate_question("   ")
        with pytest.raises(ValidationError):
            validate_question(42)

    def test_extract_question_enforces_length_cap(self, monkeypatch):
        # The cap applies on every chain input shape, not just the pipeline
        # entry point.
        monkeypatch.setattr(get_config().llm, "max_question_chars", 10)
        with pytest.raises(ValidationError, match="too long"):
            extract_question({"question": "x" * 11})

    def test_pipeline_rejects_oversized_question_before_any_call(self, monkeypatch):
        # ask_question must raise from validation, before touching the
        # cache or the chain — the chain here would explode if reached.
        from ragstone.rag.pipeline import OpenAIPipeline

        monkeypatch.setattr(get_config().llm, "max_question_chars", 50)
        pipeline = OpenAIPipeline(model="gpt-4o-mini")

        class _ExplodingChain:
            def ask_question(self, **kwargs):
                raise AssertionError("chain must not be reached")

        pipeline._chain = _ExplodingChain()
        with pytest.raises(ValidationError, match="too long"):
            pipeline.ask_question("x" * 51, session_id="s")


class TestSafeMcpErrors:
    def test_typed_pipeline_error_message_passes_through(self):
        exc = ChainExecutionError("The model refused politely.")
        text = _safe_error("Answering the question", exc)
        assert "The model refused politely." in text

    def test_untyped_exception_is_not_echoed(self):
        exc = OSError("/home/user/.secrets/openai.key not readable")
        text = _safe_error("Loading documents", exc)
        assert "/home/user" not in text  # no internals leaked
        assert "OSError" in text  # but the class name aids debugging
        assert "server logs" in text

    def test_pipeline_error_subclasses_are_trusted(self):
        assert isinstance(ChainExecutionError("x"), PipelineError)
