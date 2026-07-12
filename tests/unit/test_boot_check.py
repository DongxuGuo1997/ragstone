"""Boot-time environment checks (ROADMAP 5.11).

The invariant these tests lock: run_boot_checks reports EVERY problem in
one shot, each message carries its fix, a healthy environment boots
quietly, and RAGSTONE_BOOT_CHECKS=off really skips everything. Probes
are stubbed — no test dials a network or imports an optional extra.
"""

from types import SimpleNamespace

import pytest

from ragstone.config import boot_check
from ragstone.config.boot_check import (
    boot_check_problems,
    run_boot_checks,
)
from ragstone.utils.exceptions import ConfigurationError


def _config(
    profile="",
    ollama_embed_model=None,
    default_type="faiss",
    qdrant_url=None,
    pg_url=None,
    checkpoint_backend="memory",
    allowed_data_root=None,
):
    return SimpleNamespace(
        profile=profile,
        llm=SimpleNamespace(ollama_embed_model=ollama_embed_model),
        api=SimpleNamespace(ollama_base_url="http://localhost:11434"),
        database=SimpleNamespace(
            default_type=default_type, qdrant_url=qdrant_url, pg_url=pg_url
        ),
        memory=SimpleNamespace(checkpoint_backend=checkpoint_backend),
        loader=SimpleNamespace(allowed_data_root=allowed_data_root),
    )


@pytest.fixture
def healthy(monkeypatch):
    """A fully healthy environment; individual tests break one piece."""
    monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: ["embeddinggemma"])
    monkeypatch.setattr(boot_check, "_import_ok", lambda name: True)
    monkeypatch.setattr(boot_check, "_http_reachable", lambda url: True)
    monkeypatch.setattr(boot_check, "_pg_connect_ok", lambda url: (True, ""))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("RAGSTONE_BOOT_CHECKS", raising=False)


class TestLLMPath:
    def test_ollama_up_suffices_without_key(self, healthy):
        assert boot_check_problems(_config()) == []

    def test_key_suffices_without_ollama(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        assert boot_check_problems(_config()) == []

    def test_no_path_at_all_is_reported(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: None)
        problems = boot_check_problems(_config())
        assert len(problems) == 1
        assert "OPENAI_API_KEY" in problems[0] and "ollama serve" in problems[0]

    def test_local_profile_requires_ollama_even_with_key(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: None)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        problems = boot_check_problems(_config(profile="local"))
        assert any("no cloud fallback" in p for p in problems)


class TestModels:
    def test_missing_pinned_embedder_names_the_pull(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: ["qwen3.5:9b"])
        problems = boot_check_problems(_config(ollama_embed_model="embeddinggemma"))
        assert any("ollama pull embeddinggemma" in p for p in problems)

    def test_present_pinned_embedder_passes(self, healthy):
        assert boot_check_problems(_config(ollama_embed_model="embeddinggemma")) == []

    def test_pin_with_ollama_down_reports_llm_path_not_models(
        self, healthy, monkeypatch
    ):
        # With no tags list there is nothing to check the pin against; the
        # LLM-path problem already covers "Ollama is down".
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: None)
        problems = boot_check_problems(_config(ollama_embed_model="embeddinggemma"))
        assert len(problems) == 1 and "OPENAI_API_KEY" in problems[0]

    def test_local_profile_without_any_embedder_is_reported(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: ["qwen3.5:9b"])
        problems = boot_check_problems(_config(profile="local"))
        assert any("ollama pull embeddinggemma" in p for p in problems)


class TestStores:
    def test_missing_qdrant_extra(self, healthy, monkeypatch):
        monkeypatch.setattr(
            boot_check, "_import_ok", lambda name: name != "qdrant_client"
        )
        problems = boot_check_problems(_config(default_type="qdrant"))
        assert any(".[qdrant]" in p for p in problems)

    def test_unreachable_qdrant_server(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_http_reachable", lambda url: False)
        problems = boot_check_problems(
            _config(default_type="qdrant", qdrant_url="http://localhost:6333")
        )
        assert any("QDRANT_URL" in p and "embedded mode" in p for p in problems)

    def test_embedded_qdrant_needs_no_server(self, healthy, monkeypatch):
        monkeypatch.setattr(boot_check, "_http_reachable", lambda url: False)
        assert boot_check_problems(_config(default_type="qdrant")) == []

    def test_postgres_refusing_connection(self, healthy, monkeypatch):
        monkeypatch.setattr(
            boot_check, "_pg_connect_ok", lambda url: (False, "refused")
        )
        problems = boot_check_problems(
            _config(default_type="pgvector", pg_url="postgresql+psycopg://x")
        )
        assert any("refused" in p and "RAGSTONE_PG_URL" in p for p in problems)


class TestCheckpointAndMisc:
    def test_sqlite_backend_without_extra(self, healthy, monkeypatch):
        monkeypatch.setattr(
            boot_check,
            "_import_ok",
            lambda name: name != "langgraph.checkpoint.sqlite",
        )
        problems = boot_check_problems(_config(checkpoint_backend="sqlite"))
        assert any(".[sqlite]" in p for p in problems)

    def test_missing_data_root(self, healthy):
        problems = boot_check_problems(
            _config(allowed_data_root="/nonexistent/ragstone-data")
        )
        assert any("RAGSTONE_DATA_ROOT" in p for p in problems)

    def test_existing_data_root_passes(self, healthy, tmp_path):
        assert boot_check_problems(_config(allowed_data_root=str(tmp_path))) == []

    def test_otel_endpoint_without_extra(self, healthy, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        monkeypatch.setattr(
            boot_check, "_import_ok", lambda name: "opentelemetry" not in name
        )
        problems = boot_check_problems(_config())
        assert any(".[otel]" in p for p in problems)


class TestRunBootChecks:
    def test_reports_all_problems_at_once(self, healthy, monkeypatch):
        """One boot attempt surfaces every fix, not just the first."""
        monkeypatch.setattr(boot_check, "_ollama_tags", lambda url: None)
        monkeypatch.setattr(
            boot_check,
            "_import_ok",
            lambda name: name != "langgraph.checkpoint.sqlite",
        )
        config = _config(
            checkpoint_backend="sqlite",
            allowed_data_root="/nonexistent/ragstone-data",
        )
        with pytest.raises(ConfigurationError) as excinfo:
            run_boot_checks(config)
        message = str(excinfo.value)
        assert "3 problem(s)" in message
        for fix in ("ollama serve", ".[sqlite]", "RAGSTONE_DATA_ROOT"):
            assert fix in message

    def test_healthy_environment_boots(self, healthy):
        run_boot_checks(_config())  # must not raise

    def test_escape_hatch_skips_probes(self, healthy, monkeypatch):
        monkeypatch.setenv("RAGSTONE_BOOT_CHECKS", "off")

        def explode(url):
            raise AssertionError("probe ran despite RAGSTONE_BOOT_CHECKS=off")

        monkeypatch.setattr(boot_check, "_ollama_tags", explode)
        run_boot_checks(_config())  # must not raise, must not probe
