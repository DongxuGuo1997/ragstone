"""The no-egress profile's regression test (ROADMAP 8.1).

RAGSTONE_PROFILE=local claims nothing leaves the machine. A claim like
that belongs in CI, not a sales deck: these tests intercept every socket
connection the process attempts across the FULL ingest-and-ask path and
fail on any destination that is not loopback.

Two tiers:
- The fake-model tier always runs (CI has no Ollama): real loaders, real
  enrichment, real FAISS + BM25 indexing, real chain execution — only
  the LLM and embedding vectors are faked. Any hidden outbound call
  (telemetry, a HuggingFace download, an OpenAI fallback) trips the
  guard regardless of the fakes.
- The live tier runs where an Ollama server is listening on localhost
  and proves the same invariant with real models end to end.

Honest limitation: the guard sees connect() calls, not libc DNS lookups
(getaddrinfo). A pure DNS exfiltration channel would evade it; any
actual TCP/UDP connection does not.
"""

import socket
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import ragstone.config.settings as settings_module
from ragstone.config.settings import get_config
from ragstone.utils.exceptions import ConfigurationError, ValidationError

EVAL_CORPUS = "evals/corpus"


def _is_loopback_host(host: str) -> bool:
    return host == "localhost" or host.startswith("127.") or host == "::1"


@contextmanager
def forbid_external_sockets():
    """Fail the test on any socket connection to a non-loopback address.

    Patches socket.socket.connect — the choke point under requests,
    httpx, urllib, and every LangChain HTTP client. Unix-domain sockets
    (str/bytes addresses) are local by definition and allowed.
    """
    real_connect = socket.socket.connect
    attempts = []

    def guarded(sock, address, *args, **kwargs):
        if isinstance(address, tuple) and not _is_loopback_host(str(address[0])):
            attempts.append(address)
            raise AssertionError(f"egress attempt to {address!r}")
        return real_connect(sock, address, *args, **kwargs)

    with patch.object(socket.socket, "connect", guarded):
        yield attempts


@pytest.fixture()
def local_profile(monkeypatch):
    """Activate RAGSTONE_PROFILE=local on a FRESH config singleton."""
    monkeypatch.setenv("RAGSTONE_PROFILE", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.setattr(settings_module, "config", None)
    yield
    monkeypatch.setattr(settings_module, "config", None)


class _FakeProxy:
    """LLM proxy seam: a scripted chat model, no server."""

    def __init__(self):
        self._llm = FakeListChatModel(
            responses=["Title: T\nAuthors: A\nDate: D\nType: report", "an answer"]
        )

    def get_llm(self):
        return self._llm

    def get_model_name(self):
        return "fake-local-model"


class TestNoEgressInvariant:
    def test_full_ingest_and_ask_path_stays_on_loopback(self, local_profile):
        """The flagship check: load -> split -> enrich -> cards -> embed ->
        FAISS + BM25 -> chain -> answer, with the socket guard armed the
        entire time. Vectors and generations are faked; every other layer
        (loaders, enrichment, indexing, memory, observability) is real."""
        from ragstone.rag.pipeline import Pipeline

        with forbid_external_sockets():
            pipeline = Pipeline()
            pipeline.llm_proxy = _FakeProxy()
            texts = pipeline.load_and_split(data_dir=EVAL_CORPUS)
            assert texts, "eval corpus must load"
            pipeline._set_retriever(
                embeddings=DeterministicFakeEmbedding(size=64), use_ensemble=True
            )
            pipeline.create_rag_chain(chain_type="simple")
            answer = pipeline.ask_question(
                "What is the Violet Line?", session_id="no-egress", use_cache=False
            )
            assert answer

    def test_live_local_stack_stays_on_loopback(self, local_profile):
        """Same invariant, real models: requires Ollama on localhost."""
        try:
            with socket.create_connection(("127.0.0.1", 11434), timeout=1):
                pass
        except OSError:
            pytest.skip("Ollama not listening on localhost:11434")

        from ragstone.rag.providers import build_pipeline

        get_config().llm.ollama_reasoning = False  # measured serving posture
        with forbid_external_sockets():
            pipeline = build_pipeline("ollama")
            texts = pipeline.load_and_split(data_dir="evals/corpus_single_doc")
            assert texts
            pipeline.setup_retriever(use_ensemble=True)
            pipeline.create_rag_chain(chain_type="simple")
            answer = pipeline.ask_question(
                "What is KestrelNet?", session_id="no-egress-live", use_cache=False
            )
            assert answer


class TestLocalProfileEnforcement:
    """Each egress door is individually shut, with a clear error."""

    def test_cloud_provider_is_refused(self, local_profile):
        from ragstone.rag.providers import build_pipeline

        with pytest.raises(ConfigurationError, match="forbids the 'openai'"):
            build_pipeline("openai")

    def test_remote_document_sources_are_refused(self, local_profile):
        from ragstone.rag.pipeline import Pipeline

        with pytest.raises(ValidationError, match="remote document sources"):
            Pipeline().load_and_split(page_urls=["https://example.com"])
        with pytest.raises(ValidationError, match="remote document sources"):
            Pipeline().load_and_split(wiki_query="anything")

    def test_openai_embedding_fallback_is_disabled(self, local_profile, monkeypatch):
        # Even WITH a key in the environment, a failed Ollama probe must
        # not fall through to a cloud embedding call.
        import ragstone.rag.embeddings as embeddings_module

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", lambda name: None)
        assert embeddings_module._select_smart_embeddings() is None
