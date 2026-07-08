"""
Unit tests for registry persistence (ROADMAP 5.7) — no network, no
embeddings: the pipeline factory is faked, so what is under test is the
manifest/chunks round trip, the restore call sequence, and the failure
modes (corrupt files must mean "not found", never a crash).
"""

import json

import pytest
from langchain_core.documents import Document

import ragstone.rag.pipeline as pipeline_module
from ragstone.utils import registry


@pytest.fixture(autouse=True)
def persist_to_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGSTONE_REGISTRY_PERSIST", "on")
    monkeypatch.setenv("RAGSTONE_REGISTRY_DIR", str(tmp_path / "registry"))
    registry.clear_pipelines()
    yield tmp_path / "registry"
    registry.clear_pipelines()


class _ConfiguredPipeline:
    """Stands in for a pipeline that finished retriever setup."""

    provider = "openai"

    def __init__(self, texts):
        self.texts = texts
        self.llm_proxy = self
        self._chain_type = "fusion"
        self._use_ensemble = False
        self._use_reranker = True
        self._vector_db_fingerprint = "fp-123"

    def get_model_name(self):
        return "gpt-4o-mini"


class _RestoredPipeline:
    """Records the restore sequence the registry is supposed to drive."""

    def __init__(self, provider, model):
        self.provider = provider
        self.model = model
        self.texts = None
        self.retriever_args = None
        self.chain_type = None
        self._vector_db_fingerprint = "fp-123"

    def setup_retriever(self, use_ensemble=True, use_reranker=False):
        assert self.texts, "chunks must be set before the retriever"
        self.retriever_args = (use_ensemble, use_reranker)

    def create_rag_chain(self, chain_type="simple"):
        assert self.retriever_args, "retriever must be set before the chain"
        self.chain_type = chain_type


def _chunks():
    return [
        Document(page_content="Paris is the capital.", metadata={"source": "a.md"}),
        Document(
            page_content="[Document metadata] Source: a.md\nTitle: X",
            metadata={"source": "a.md", "metadata_card": True},
        ),
    ]


def _persist(pipeline_id="p1", texts=None):
    ok = registry.persist_pipeline(
        pipeline_id, _ConfiguredPipeline(texts if texts is not None else _chunks())
    )
    assert ok is True
    return pipeline_id


class TestRoundTrip:
    def test_restore_rebuilds_chunks_config_and_registers(self, monkeypatch):
        _persist("p1")
        built = {}
        monkeypatch.setattr(
            pipeline_module,
            "build_pipeline",
            lambda provider, model: built.setdefault(
                "p", _RestoredPipeline(provider, model)
            ),
        )

        restored = registry.get_or_restore_pipeline("p1")

        assert restored is built["p"]
        assert restored.provider == "openai"
        assert restored.model == "gpt-4o-mini"
        # Chunks round-trip exactly — enrichment and metadata cards are
        # baked into the persisted text, so restore makes no LLM calls.
        assert [d.page_content for d in restored.texts] == [
            d.page_content for d in _chunks()
        ]
        assert restored.texts[1].metadata["metadata_card"] is True
        assert restored.retriever_args == (False, True)
        assert restored.chain_type == "fusion"
        # Registered: the next lookup is a plain dict hit, no rebuild.
        assert registry.get_pipeline("p1") is restored

    def test_persisted_ids_are_listed(self):
        _persist("p1")
        _persist("p2")
        assert sorted(registry.list_persisted()) == ["p1", "p2"]

    def test_hostile_pipeline_id_cannot_escape_the_registry_dir(
        self, persist_to_tmp, monkeypatch
    ):
        # Ids are client-supplied; filenames are hashes, so traversal
        # characters are inert and the id still round-trips.
        evil = "../../outside/etc"
        _persist(evil)
        files = list(persist_to_tmp.parent.rglob("*"))
        assert all(persist_to_tmp in f.parents or f == persist_to_tmp for f in files)
        assert registry.list_persisted() == [evil]

    def test_delete_persisted_removes_both_files(self, persist_to_tmp):
        _persist("p1")
        assert len(list(persist_to_tmp.iterdir())) == 2  # manifest + chunks
        registry.delete_persisted("p1")
        assert list(persist_to_tmp.iterdir()) == []
        registry.delete_persisted("p1")  # idempotent

    def test_odd_metadata_types_serialize(self, monkeypatch):
        from pathlib import Path

        chunk = Document(page_content="x", metadata={"path": Path("/tmp/a")})
        _persist("p1", texts=[chunk])
        monkeypatch.setattr(
            pipeline_module,
            "build_pipeline",
            lambda provider, model: _RestoredPipeline(provider, model),
        )
        restored = registry.restore_pipeline("p1")
        assert restored.texts[0].metadata["path"] == "/tmp/a"  # stringified


class TestFailureModes:
    def test_persist_off_disables_everything(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_REGISTRY_PERSIST", "off")
        assert registry.persist_pipeline("p1", _ConfiguredPipeline(_chunks())) is False
        assert registry.restore_pipeline("p1") is None
        assert registry.list_persisted() == []

    def test_unconfigured_pipeline_is_not_persistable(self):
        assert registry.persist_pipeline("p1", _ConfiguredPipeline([])) is False

    def test_missing_manifest_is_none(self):
        assert registry.restore_pipeline("ghost") is None

    def test_corrupt_chunks_file_means_not_found(self, persist_to_tmp, monkeypatch):
        _persist("p1")
        for chunks_file in persist_to_tmp.glob("*.chunks.jsonl"):
            chunks_file.write_text('{"page_content": "only one\n')  # torn write
        monkeypatch.setattr(
            pipeline_module,
            "build_pipeline",
            lambda provider, model: _RestoredPipeline(provider, model),
        )
        assert registry.restore_pipeline("p1") is None  # logged, not raised

    def test_chunk_count_mismatch_means_not_found(self, persist_to_tmp, monkeypatch):
        _persist("p1")
        for manifest_file in persist_to_tmp.glob("*.json"):
            manifest = json.loads(manifest_file.read_text())
            manifest["chunk_count"] = 99
            manifest_file.write_text(json.dumps(manifest))
        monkeypatch.setattr(
            pipeline_module,
            "build_pipeline",
            lambda provider, model: _RestoredPipeline(provider, model),
        )
        assert registry.restore_pipeline("p1") is None

    def test_unsupported_manifest_version_is_ignored(self, persist_to_tmp):
        _persist("p1")
        for manifest_file in persist_to_tmp.glob("*.json"):
            manifest = json.loads(manifest_file.read_text())
            manifest["version"] = 999
            manifest_file.write_text(json.dumps(manifest))
        assert registry.restore_pipeline("p1") is None

    def test_restore_failure_never_breaks_plain_lookup(self):
        # get_or_restore on an id with no manifest behaves exactly like
        # the old get_pipeline: plain None.
        assert registry.get_or_restore_pipeline("never-existed") is None
