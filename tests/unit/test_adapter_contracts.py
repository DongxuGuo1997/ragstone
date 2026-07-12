"""
Contract tests for the adapter layer: loader, vector_db, embeddings.

These modules wrap third-party I/O (file loaders, FAISS/Chroma, embedding
APIs), so their tests pin the CONTRACTS the rest of the engine relies on
— error wrapping, fallback order, ordering guarantees, security-relevant
request kwargs — using fakes only. No network, no API keys.
"""

import io
from types import SimpleNamespace
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import ragstone.rag.embeddings as embeddings_module
from ragstone.rag import loader as loader_module
from ragstone.rag.embeddings import (
    _select_smart_embeddings,
    embed_texts_cached,
    embed_texts_parallel,
    get_smart_embeddings,
)
from ragstone.rag.loader import LocalLoader, RemoteLoader
from ragstone.rag.vector_db import (
    ChromaProxy,
    FaissProxy,
    create_vector_store_proxy,
)
from ragstone.utils.exceptions import (
    VectorStoreInitializationError,
    VectorStoreOperationError,
)

# ---------------------------------------------------------------------------
# Loader: uploaded files (the Streamlit path)
# ---------------------------------------------------------------------------


class _FakeUpload:
    """Duck-types a Streamlit UploadedFile: .name plus .getvalue()."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class _ExplodingUpload(_FakeUpload):
    def getvalue(self) -> bytes:
        raise IOError("stream went away")


class TestUploadedFiles:
    def test_txt_upload_becomes_document_with_source_metadata(self):
        docs = LocalLoader()._process_uploaded_files(
            [_FakeUpload("notes.txt", b"Paris is the capital of France.")]
        )
        assert len(docs) == 1
        assert "Paris" in docs[0].page_content
        assert docs[0].metadata["source"] == "notes.txt"
        assert docs[0].metadata["title"] == "notes"

    def test_unsupported_extension_falls_back_to_plain_text(self):
        docs = LocalLoader()._process_uploaded_files(
            [_FakeUpload("server.log", b"line one\nline two")]
        )
        assert len(docs) == 1
        assert docs[0].page_content == "line one\nline two"
        assert docs[0].metadata["source"] == "server.log"

    def test_undecodable_unsupported_file_is_skipped_not_fatal(self):
        docs = LocalLoader()._process_uploaded_files(
            [_FakeUpload("blob.bin", b"\xff\xfe\x00\x01 not utf-8 \x80")]
        )
        assert docs == []

    def test_non_streamlit_objects_are_skipped_gracefully(self):
        docs = LocalLoader()._process_uploaded_files(
            ["a plain string", io.BytesIO(b"raw"), None]
        )
        assert docs == []

    def test_one_bad_upload_does_not_lose_the_others(self):
        docs = LocalLoader()._process_uploaded_files(
            [
                _ExplodingUpload("broken.txt", b""),
                _FakeUpload("good.txt", b"still here"),
            ]
        )
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "good.txt"


class TestLocalDirectoryLoad:
    def test_loads_supported_files_from_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha document")
        (tmp_path / "b.md").write_text("# beta document")
        local = LocalLoader()
        local.load(data_dir=str(tmp_path))
        contents = sorted(d.page_content for d in local.get_documents())
        assert contents == ["# beta document", "alpha document"]

        local.clear_documents()
        assert local.get_documents() == []

    def test_missing_directory_is_not_fatal(self, tmp_path):
        local = LocalLoader()
        local.load(data_dir=str(tmp_path / "does-not-exist"))
        assert local.get_documents() == []


# ---------------------------------------------------------------------------
# Loader: remote sources (fakes injected through the loader cache)
# ---------------------------------------------------------------------------


class _FakeWebLoader:
    """Records constructor kwargs; returns canned documents."""

    last_kwargs: dict = {}

    def __init__(self, urls, **kwargs):
        type(self).last_kwargs = {"urls": urls, **kwargs}

    def load(self):
        return [Document(page_content="web page text", metadata={"source": "web"})]


class _FailingLoader:
    def __init__(self, *args, **kwargs):
        pass

    def load(self):
        raise ConnectionError("network unreachable")


class _FakeHtml2Text:
    def transform_documents(self, docs):
        return [
            Document(page_content=f"text:{d.page_content}", metadata=d.metadata)
            for d in docs
        ]


class TestRemoteLoader:
    def test_basic_web_load_pins_timeout_and_blocks_redirects(self, monkeypatch):
        # The SSRF guard validates the ORIGINAL host, so the fetch must not
        # follow redirects — this locks that contract at the request level.
        monkeypatch.setitem(loader_module._loader_cache, "web", _FakeWebLoader)
        remote = RemoteLoader()
        remote.load(page_urls=["https://example.com"], scrape_pages=False)

        assert [d.page_content for d in remote.get_documents()] == ["web page text"]
        kwargs = _FakeWebLoader.last_kwargs
        assert kwargs["requests_kwargs"] == {"timeout": 30, "allow_redirects": False}
        assert kwargs["continue_on_failure"] is True

    def test_scrape_path_transforms_html_and_blocks_redirects(self, monkeypatch):
        class _FakeHtmlLoader(_FakeWebLoader):
            last_kwargs: dict = {}

            def load(self):
                return [Document(page_content="<b>hi</b>", metadata={})]

        monkeypatch.setitem(loader_module._loader_cache, "html", _FakeHtmlLoader)
        monkeypatch.setitem(
            loader_module._loader_cache, "html_transformer", _FakeHtml2Text
        )
        remote = RemoteLoader()
        remote.load(page_urls=["https://example.com"], scrape_pages=True)

        assert [d.page_content for d in remote.get_documents()] == ["text:<b>hi</b>"]
        kwargs = _FakeHtmlLoader.last_kwargs
        assert kwargs["requests_kwargs"]["allow_redirects"] is False
        assert kwargs["ignore_load_errors"] is True

    def test_wikipedia_query_and_doc_cap_reach_the_loader(self, monkeypatch):
        seen = {}

        class _FakeWikiLoader:
            def __init__(self, query, load_max_docs):
                seen.update(query=query, load_max_docs=load_max_docs)

            def load(self):
                return [Document(page_content="wiki text", metadata={})]

        monkeypatch.setitem(loader_module._loader_cache, "wiki", _FakeWikiLoader)
        remote = RemoteLoader()
        remote.load(wiki_query="Paris", max_wiki_docs=3)

        assert seen == {"query": "Paris", "load_max_docs": 3}
        assert len(remote.get_documents()) == 1

    def test_remote_failure_yields_empty_documents_not_an_exception(self, monkeypatch):
        monkeypatch.setitem(loader_module._loader_cache, "web", _FailingLoader)
        monkeypatch.setitem(loader_module._loader_cache, "wiki", _FailingLoader)
        remote = RemoteLoader()
        remote.load(
            page_urls=["https://example.com"], wiki_query="x", scrape_pages=False
        )
        assert remote.get_documents() == []


# ---------------------------------------------------------------------------
# Vector stores: FAISS/Chroma round trips and error wrapping (fake embeddings)
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_embed_cache(monkeypatch):
    """Neutralize the on-disk embedding cache for isolated store tests."""
    import ragstone.rag.embedding_cache as cache_module

    monkeypatch.setattr(cache_module, "get_embedding_cache", lambda: None)


def _docs() -> List[Document]:
    return [
        Document(page_content="Paris is the capital of France.", metadata={"n": 1}),
        Document(page_content="Berlin is the capital of Germany.", metadata={"n": 2}),
        Document(page_content="The Alps are a mountain range.", metadata={"n": 3}),
    ]


class TestFaissProxyContract:
    def test_create_and_find_round_trip(self, no_embed_cache):
        proxy = FaissProxy()
        proxy.create_db(_docs(), embeddings=DeterministicFakeEmbedding(size=32))
        assert proxy.is_initialized

        results = proxy.find_similar("Paris is the capital of France.", k=2)
        assert len(results) == 2
        # Deterministic fake embeddings: an exact text match is its own
        # nearest neighbor.
        assert results[0].page_content == "Paris is the capital of France."

        proxy.cleanup()
        assert not proxy.is_initialized

    def test_find_before_create_raises_operation_error(self):
        with pytest.raises(VectorStoreOperationError):
            FaissProxy().find_similar("anything")

    def test_empty_documents_raise_initialization_error(self):
        with pytest.raises(VectorStoreInitializationError):
            FaissProxy().create_db([], embeddings=DeterministicFakeEmbedding(size=8))

    def test_blank_query_raises_operation_error(self, no_embed_cache):
        proxy = FaissProxy()
        proxy.create_db(_docs(), embeddings=DeterministicFakeEmbedding(size=32))
        with pytest.raises(VectorStoreOperationError):
            proxy.find_similar("   ")

    def test_save_and_load_round_trip(self, no_embed_cache, tmp_path):
        embeddings = DeterministicFakeEmbedding(size=32)
        saver = FaissProxy()
        saver.create_db(_docs(), embeddings=embeddings)
        saver.save_local(str(tmp_path))

        loader = FaissProxy()
        loader.load_local(
            str(tmp_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        assert loader.is_initialized
        top = loader.find_similar("Berlin is the capital of Germany.", k=1)
        assert top[0].page_content == "Berlin is the capital of Germany."
        assert top[0].metadata["n"] == 2  # metadata survives persistence

    def test_save_before_create_raises(self, tmp_path):
        with pytest.raises(VectorStoreOperationError):
            FaissProxy().save_local(str(tmp_path))

    def test_load_from_empty_folder_raises(self, tmp_path):
        proxy = FaissProxy()
        with pytest.raises(VectorStoreOperationError, match="not found"):
            proxy.load_local(str(tmp_path))
        assert not proxy.is_initialized


class TestChromaProxyContract:
    def test_create_and_find_round_trip(self, tmp_path):
        pytest.importorskip("chromadb")
        proxy = ChromaProxy(persist_directory=str(tmp_path))
        proxy.create_db(
            _docs(),
            collection_name="contract_test",
            embeddings=DeterministicFakeEmbedding(size=32),
        )
        assert proxy.is_initialized

        results = proxy.find_similar("The Alps are a mountain range.", k=1)
        assert results[0].page_content == "The Alps are a mountain range."
        proxy.cleanup()
        assert not proxy.is_initialized

    def test_rebuild_replaces_the_collection_instead_of_appending(self, tmp_path):
        pytest.importorskip("chromadb")
        embeddings = DeterministicFakeEmbedding(size=32)
        proxy = ChromaProxy(persist_directory=str(tmp_path))
        proxy.create_db(_docs(), collection_name="rebuild", embeddings=embeddings)
        proxy.create_db(_docs(), collection_name="rebuild", embeddings=embeddings)

        # Ask for more results than documents: an appended (doubled)
        # collection would return duplicates.
        results = proxy.find_similar("capital", k=6)
        assert len(results) == 3


class TestVectorStoreFactory:
    def test_known_types_build_the_right_proxy(self):
        assert isinstance(create_vector_store_proxy("faiss"), FaissProxy)
        assert isinstance(create_vector_store_proxy("FAISS"), FaissProxy)
        assert isinstance(create_vector_store_proxy("chroma"), ChromaProxy)

    def test_unknown_type_raises_with_supported_list(self):
        with pytest.raises(ValueError, match="qdrant"):
            create_vector_store_proxy("pinecone")


# ---------------------------------------------------------------------------
# Embeddings: parallel batching, cache composition, smart selection
# ---------------------------------------------------------------------------


def _vec(text: str) -> List[float]:
    return [float(len(text)), float(sum(text.encode()) % 97)]


class _RecordingEmbeddings:
    """Deterministic text->vector map that records embed_documents calls."""

    def __init__(self, fail_on: str = ""):
        self.calls: List[List[str]] = []
        self._fail_on = fail_on

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.calls.append(list(texts))
        if self._fail_on and self._fail_on in texts:
            raise RuntimeError("provider rejected batch")
        return [_vec(t) for t in texts]


class TestEmbedTextsParallel:
    def test_order_is_preserved_across_concurrent_batches(self):
        texts = [f"text number {i}" for i in range(7)]
        provider = _RecordingEmbeddings()
        vectors = embed_texts_parallel(provider, texts, batch_size=2, max_workers=4)

        assert vectors == [_vec(t) for t in texts]  # exact input order
        assert len(provider.calls) == 4  # ceil(7 / 2) batches issued

    def test_single_worker_bypasses_the_thread_pool(self):
        texts = ["a", "b", "c"]
        provider = _RecordingEmbeddings()
        embed_texts_parallel(provider, texts, batch_size=1, max_workers=1)
        assert provider.calls == [texts]  # one call, all texts

    def test_any_batch_failure_fails_the_whole_operation(self):
        # A partially embedded corpus must never be indexed silently.
        texts = ["good one", "bad apple", "good two"]
        provider = _RecordingEmbeddings(fail_on="bad apple")
        with pytest.raises(RuntimeError, match="rejected"):
            embed_texts_parallel(provider, texts, batch_size=1, max_workers=3)


class _FakeCache:
    """In-memory stand-in for the SQLite embedding cache."""

    def __init__(self, known: dict):
        self._known = known
        self.put_calls: List[List[str]] = []

    def get_many(self, model_id, texts):
        return {i: self._known[t] for i, t in enumerate(texts) if t in self._known}

    def put_many(self, model_id, texts, vectors):
        self.put_calls.append(list(texts))
        self._known.update(zip(texts, vectors))


class TestEmbedTextsCached:
    def test_disabled_cache_is_parallel_verbatim(self, monkeypatch):
        import ragstone.rag.embedding_cache as cache_module

        monkeypatch.setattr(cache_module, "get_embedding_cache", lambda: None)
        provider = _RecordingEmbeddings()
        texts = ["x", "y"]
        assert embed_texts_cached(provider, texts, 10, 1) == [_vec(t) for t in texts]
        assert provider.calls == [texts]

    def test_only_misses_reach_the_provider_and_order_holds(self, monkeypatch):
        import ragstone.rag.embedding_cache as cache_module

        texts = ["cached a", "fresh b", "cached c", "fresh d"]
        cache = _FakeCache({"cached a": [1.0], "cached c": [3.0]})
        monkeypatch.setattr(cache_module, "get_embedding_cache", lambda: cache)

        provider = _RecordingEmbeddings()
        vectors = embed_texts_cached(provider, texts, 10, 1)

        assert provider.calls == [["fresh b", "fresh d"]]  # misses only
        assert vectors == [[1.0], _vec("fresh b"), [3.0], _vec("fresh d")]
        assert cache.put_calls == [["fresh b", "fresh d"]]  # written back

    def test_full_hit_never_touches_the_provider(self, monkeypatch):
        import ragstone.rag.embedding_cache as cache_module

        cache = _FakeCache({"a": [1.0], "b": [2.0]})
        monkeypatch.setattr(cache_module, "get_embedding_cache", lambda: cache)
        provider = _RecordingEmbeddings()

        assert embed_texts_cached(provider, ["a", "b"], 10, 4) == [[1.0], [2.0]]
        assert provider.calls == []


def _stub_config(prefer_ollama: bool):
    return SimpleNamespace(
        llm=SimpleNamespace(
            prefer_ollama_embeddings=prefer_ollama, ollama_embed_model=None
        ),
        profile="",
    )


class _RecordingOllamaEmbeddings:
    """Stands in for OllamaEmbeddings; records what reaches the server."""

    def __init__(self, model=None):
        self.model = model
        self.document_calls: list = []
        self.query_calls: list = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [[0.1] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.1]


class TestTaskPrefixes:
    """Embedding models have per-family task conventions from their model
    cards; the wrapper applies them (Experiment 22 — bare nomic ranked
    contributor name-lists as near-universal nearest neighbors)."""

    def test_nomic_prefixes_both_sides(self):
        inner = _RecordingOllamaEmbeddings(model="nomic-embed-text:latest")
        wrapped = embeddings_module.TaskPrefixedEmbeddings(
            inner, *embeddings_module._task_convention_for("nomic-embed-text:latest")
        )
        wrapped.embed_documents(["chunk one", "chunk two"])
        wrapped.embed_query("what is kestrelnet?")
        assert inner.document_calls == [
            ["search_document: chunk one", "search_document: chunk two"]
        ]
        assert inner.query_calls == ["search_query: what is kestrelnet?"]

    def test_mxbai_prefixes_queries_only(self):
        inner = _RecordingOllamaEmbeddings(model="mxbai-embed-large")
        wrapped = embeddings_module.TaskPrefixedEmbeddings(
            inner, *embeddings_module._task_convention_for("mxbai-embed-large")
        )
        wrapped.embed_documents(["a chunk"])
        wrapped.embed_query("a query")
        assert inner.document_calls == [["a chunk"]]  # docs stay bare
        assert inner.query_calls == [
            "Represent this sentence for searching relevant passages: a query"
        ]

    def test_embeddinggemma_has_no_convention_by_measurement(self):
        # Its documented templates measured HARMFUL through Ollama
        # (Experiment 25's A-B: 0.76/0.62 templated vs 0.80/0.65 bare),
        # so the default probe must yield the BARE embedder.
        assert embeddings_module._task_convention_for("embeddinggemma") is None

    def test_probe_wraps_known_families_but_not_others(self, monkeypatch):
        monkeypatch.setattr(
            embeddings_module,
            "_ollama_embeddings_cls",
            lambda: _RecordingOllamaEmbeddings,
        )
        for family_model in ("nomic-embed-text:latest", "mxbai-embed-large"):
            probed = embeddings_module._probe_ollama_model(family_model)
            assert isinstance(probed, embeddings_module.TaskPrefixedEmbeddings)
        # bge-m3 / embeddinggemma need no convention: bare embedder.
        for bare_model in ("bge-m3", "embeddinggemma"):
            assert isinstance(
                embeddings_module._probe_ollama_model(bare_model),
                _RecordingOllamaEmbeddings,
            )

    def test_prefixes_can_be_disabled_for_ab_measurement(self, monkeypatch):
        monkeypatch.setattr(
            embeddings_module,
            "_ollama_embeddings_cls",
            lambda: _RecordingOllamaEmbeddings,
        )
        monkeypatch.setenv("RAGSTONE_EMBED_TASK_PREFIXES", "off")
        probed = embeddings_module._probe_ollama_model("nomic-embed-text:latest")
        assert isinstance(probed, _RecordingOllamaEmbeddings)

    def test_wrapper_gets_its_own_cache_namespace(self):
        # The cache and corpus fingerprint key on class name + model:
        # prefixed vectors must never collide with bare-embedder vectors.
        from ragstone.rag.embedding_cache import model_id_for

        inner = _RecordingOllamaEmbeddings(model="nomic-embed-text:latest")
        wrapped = embeddings_module.TaskPrefixedEmbeddings(inner, "{t}", "{q}")
        assert model_id_for(wrapped) != model_id_for(inner)
        assert "nomic-embed-text:latest" in model_id_for(wrapped)


class TestPinnedEmbedder:
    """RAGSTONE_OLLAMA_EMBED_MODEL: honor the pin or fail loud — never
    substitute a different embedder than the operator chose."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        monkeypatch.setattr(embeddings_module, "_selected_embeddings", None)

    def _config(self, pinned):
        return SimpleNamespace(
            llm=SimpleNamespace(
                prefer_ollama_embeddings=True, ollama_embed_model=pinned
            ),
            profile="",
        )

    def test_pinned_model_is_probed_directly(self, monkeypatch):
        winner = object()
        probed = []

        def fake_probe(name):
            probed.append(name)
            return winner

        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", fake_probe)
        monkeypatch.setattr(
            embeddings_module, "get_config", lambda: self._config("mxbai-embed-large")
        )
        assert embeddings_module._select_smart_embeddings() is winner
        assert probed == ["mxbai-embed-large"]

    def test_unavailable_pin_fails_loud_not_fallback(self, monkeypatch):
        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", lambda name: None)
        monkeypatch.setattr(
            embeddings_module, "get_config", lambda: self._config("ghost-model")
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        # Even with an OpenAI key available, the pin must not silently
        # substitute: None -> the caller raises a clear error.
        assert embeddings_module._select_smart_embeddings() is None


class TestSmartEmbeddingSelection:
    @pytest.fixture(autouse=True)
    def _fresh_memo(self, monkeypatch):
        monkeypatch.setattr(embeddings_module, "_selected_embeddings", None)

    def test_dedicated_embedding_models_are_probed_first(self, monkeypatch):
        probed = []
        winner = object()

        def fake_probe(model_name):
            probed.append(model_name)
            return winner if model_name == "embeddinggemma:latest" else None

        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", fake_probe)
        monkeypatch.setattr(embeddings_module, "get_config", lambda: _stub_config(True))
        assert _select_smart_embeddings() is winner
        # embeddinggemma leads the probe order by measurement (Exp 25).
        assert probed == ["embeddinggemma:latest"]  # stopped at first hit

    def test_probe_failure_falls_back_to_openai(self, monkeypatch):
        sentinel = object()
        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", lambda name: None)
        monkeypatch.setattr(embeddings_module, "get_config", lambda: _stub_config(True))
        monkeypatch.setattr(
            embeddings_module, "make_openai_embeddings", lambda: sentinel
        )
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
        assert _select_smart_embeddings() is sentinel

    def test_no_working_option_returns_none(self, monkeypatch):
        # No dedicated embedder + no OpenAI key must be a clean None (the
        # caller raises a clear error) — NOT a silent fallback to embedding
        # with the chat LLM, which the old cascade did.
        monkeypatch.setattr(embeddings_module, "_probe_ollama_model", lambda name: None)
        monkeypatch.setattr(embeddings_module, "get_config", lambda: _stub_config(True))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _select_smart_embeddings() is None

    def test_selection_is_memoized(self, monkeypatch):
        calls = []

        def fake_select():
            calls.append(1)
            return object()

        monkeypatch.setattr(embeddings_module, "_select_smart_embeddings", fake_select)
        first = get_smart_embeddings()
        second = get_smart_embeddings()
        assert first is second
        assert calls == [1]  # probing cost paid once per process

    def test_failed_selection_is_not_memoized(self, monkeypatch):
        monkeypatch.setattr(embeddings_module, "_select_smart_embeddings", lambda: None)
        assert get_smart_embeddings() is None
        assert embeddings_module._selected_embeddings is None  # retried next time
