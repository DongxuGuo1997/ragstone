"""
Unit tests for the LLM proxies' construction kwargs.

OllamaProxy.set_llm is where the RAGSTONE_OLLAMA_REASONING knob meets
ChatOllama; nothing else covered this seam, and a silently dropped
reasoning kwarg would make every "local latency" measurement dishonest
(a thinking model quietly reasons through each retrieval question).
"""

import pytest

from ragstone.config.settings import get_config
from ragstone.models import base_model


class _RecordingChatOllama:
    """Stands in for ChatOllama; records the constructor kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


@pytest.fixture()
def recording_ollama(monkeypatch):
    monkeypatch.setitem(base_model._model_cache, "ollama", _RecordingChatOllama)
    _RecordingChatOllama.last_kwargs = {}
    return _RecordingChatOllama


class TestOllamaReasoningPassthrough:
    def test_unset_config_omits_the_kwarg_entirely(self, recording_ollama, monkeypatch):
        # None must mean "knob absent" — not reasoning=None — so the
        # model's own default stands on any langchain-ollama version.
        monkeypatch.setattr(get_config().llm, "ollama_reasoning", None)
        base_model.OllamaProxy().set_llm(model_name="m")
        assert "reasoning" not in recording_ollama.last_kwargs

    @pytest.mark.parametrize("value", [True, False])
    def test_configured_value_is_passed(self, recording_ollama, monkeypatch, value):
        monkeypatch.setattr(get_config().llm, "ollama_reasoning", value)
        base_model.OllamaProxy().set_llm(model_name="m")
        assert recording_ollama.last_kwargs["reasoning"] is value

    def test_caller_kwarg_beats_config(self, recording_ollama, monkeypatch):
        monkeypatch.setattr(get_config().llm, "ollama_reasoning", True)
        base_model.OllamaProxy().set_llm(model_name="m", reasoning=False)
        assert recording_ollama.last_kwargs["reasoning"] is False

    def test_existing_defaults_unaffected(self, recording_ollama, monkeypatch):
        # The reasoning knob must not disturb the endpoint/timeout wiring.
        monkeypatch.setattr(get_config().llm, "ollama_reasoning", False)
        base_model.OllamaProxy().set_llm(model_name="m")
        kwargs = recording_ollama.last_kwargs
        assert kwargs["base_url"] == get_config().api.ollama_base_url
        assert kwargs["client_kwargs"] == {"timeout": get_config().llm.timeout}


class TestOllamaTemperatureParity:
    """OllamaProxy defaults temperature like the OpenAI proxy does.

    Before July 2026 only the OpenAI path defaulted to 0.0; Ollama
    models silently ran at their own sampling defaults (~0.6-0.8), so
    "the same pipeline, locally" answered with more randomness than the
    cloud path it was measured against.
    """

    def test_temperature_defaults_to_config_value(self, recording_ollama):
        base_model.OllamaProxy().set_llm(model_name="m")
        expected = get_config().llm.default_temperature
        assert recording_ollama.last_kwargs["temperature"] == expected == 0.0

    def test_caller_temperature_beats_the_default(self, recording_ollama):
        # The UI slider passes temperature explicitly; it must still win.
        base_model.OllamaProxy().set_llm(model_name="m", temperature=0.7)
        assert recording_ollama.last_kwargs["temperature"] == 0.7
