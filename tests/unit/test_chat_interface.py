"""
Unit tests for the terminal chat (no network, no real pipeline).

The REPL's value is its command surface and rendering helpers; both are
driven here with a stub pipeline injected through the constructor.
"""

from types import SimpleNamespace

from ragstone.ui.chat_interface import (
    ChatInterface,
    Style,
    format_event_line,
    format_trace,
    parse_command,
)


def _metrics(**overrides):
    values = dict(
        first_token_ms=320,
        latency_ms=1420,
        cache_hit=False,
        tokens=1183,
        input_tokens=1000,
        output_tokens=183,
        chain_type="simple",
        stage_ms={"rephrase": 210, "retrieval": 89},
        generation_ms=1121,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _StubPipeline:
    def __init__(self):
        self.chain_calls = []
        self.last_metrics = _metrics()

    def ask_question_stream(self, question, session_id=None, use_cache=True):
        yield {"event": "search", "query": "refined"}
        yield "streamed "
        yield "answer"

    def get_last_interpretation(self, session_id):
        return None

    def get_sources(self, question):
        return [
            {"source": "helios_solar_guide.md", "snippet": "warranty is 25 years"},
            {"source": "helios_solar_guide.md", "snippet": "torque spec 12 Nm"},
        ]

    def create_rag_chain(self, chain_type="simple"):
        self.chain_calls.append(chain_type)


def _chat(pipeline=None):
    return ChatInterface(
        pipeline=pipeline or _StubPipeline(), style=Style(enabled=False)
    )


class TestHelpers:
    """Pure rendering/parsing helpers: no pipeline, no terminal."""

    def test_parse_command_splits_slash_commands(self):
        assert parse_command("/chain corrective") == ("chain", "corrective")
        assert parse_command("/help") == ("help", "")
        assert parse_command("plain question?") == ("", "plain question?")

    def test_event_lines_match_streamlit_semantics(self):
        assert format_event_line({"event": "search", "query": "q"}) == "searching: q"
        assert "not relevant" in format_event_line(
            {"event": "grade", "relevant": False}
        )
        assert format_event_line({"event": "unknown"}) is None

    def test_trace_line_contains_stages_cost_and_chain(self):
        trace = format_trace(_metrics(), model="gpt-4o-mini")
        assert "first token 320 ms" in trace
        assert "total 1.42 s" in trace
        assert "1183 tokens" in trace
        assert "~$" in trace  # gpt-4o-mini has a price entry
        assert "chain=simple" in trace
        assert "rephrase 210 ms > retrieval 89 ms > generation 1121 ms" in trace

    def test_cache_hit_trace_skips_tokens_and_stages(self):
        trace = format_trace(_metrics(cache_hit=True, tokens=0))
        assert "cache hit" in trace
        assert "tokens" not in trace
        assert "generation" not in trace

    def test_style_disabled_passes_text_through(self):
        style = Style(enabled=False)
        assert style.dim("x") == "x"
        assert Style(enabled=True).bold("x") == "\033[1mx\033[0m"


class TestCommands:
    """The slash-command surface, driven with an injected stub pipeline."""

    def test_ask_streams_events_answer_trace_and_sources(self, capsys):
        chat = _chat()
        chat.ask("what is the warranty?")
        out = capsys.readouterr().out
        assert "[searching: refined]" in out
        assert "streamed answer" in out
        assert "chain=simple" in out  # the trace line
        assert "sources: helios_solar_guide.md" in out  # deduplicated

    def test_chain_command_switches_and_validates(self, capsys):
        pipeline = _StubPipeline()
        chat = _chat(pipeline)
        assert chat.handle("/chain corrective") is True
        assert pipeline.chain_calls == ["corrective"]
        assert chat.chain_type == "corrective"

        chat.handle("/chain nonsense")
        assert pipeline.chain_calls == ["corrective"]  # unchanged
        assert "Unknown chain type" in capsys.readouterr().out

    def test_sources_command_prints_snippets(self, capsys):
        chat = _chat()
        chat.ask("q?")
        capsys.readouterr()
        chat.handle("/sources")
        out = capsys.readouterr().out
        assert "warranty is 25 years" in out
        assert "torque spec 12 Nm" in out

    def test_new_resets_the_session(self):
        chat = _chat()
        chat.ask("q?")
        first_session = chat.session_id
        chat.handle("/new")
        assert chat.session_id != first_session
        assert chat._last_question is None

    def test_cache_command_toggles_config(self, capsys):
        chat = _chat()
        original = chat.config.cache.enable_response_cache
        try:
            chat.handle("/cache on")
            assert chat.config.cache.enable_response_cache is True
            chat.handle("/cache off")
            assert chat.config.cache.enable_response_cache is False
            chat.handle("/cache maybe")
            assert "Usage" in capsys.readouterr().out
        finally:  # the config object is process-global
            chat.config.cache.enable_response_cache = original

    def test_exit_commands_end_the_loop(self):
        chat = _chat()
        assert chat.handle("/exit") is False
        assert chat.handle("/quit") is False
        assert chat.handle("/help") is True

    def test_unknown_command_is_reported_not_fatal(self, capsys):
        chat = _chat()
        assert chat.handle("/frobnicate") is True
        assert "Unknown command /frobnicate" in capsys.readouterr().out
