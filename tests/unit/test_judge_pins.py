"""
The judge is the measuring instrument; its model must not float.

resolve_judge_model pins known OpenAI aliases to dated snapshots at the
wire level while baseline keys keep the alias. These tests lock the
resolution rules; the snapshot's live existence was verified when the
pin was introduced (2026-07-17) and any future 404 will fail every
judged eval loudly at the first call.
"""

import importlib.util
from pathlib import Path

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


def _judge():
    spec = importlib.util.spec_from_file_location("judge", EVALS_DIR / "judge.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestJudgeModelPins:
    def test_openai_alias_resolves_to_dated_snapshot(self):
        judge = _judge()
        resolved = judge.resolve_judge_model("gpt-4o-mini", "openai")
        assert resolved == judge.JUDGE_MODEL_PINS["gpt-4o-mini"]
        assert resolved != "gpt-4o-mini"  # actually pinned
        assert "gpt-4o-mini-" in resolved  # a dated snapshot of the alias

    def test_unpinned_models_pass_through(self):
        judge = _judge()
        assert judge.resolve_judge_model("gpt-4o", "openai") == "gpt-4o"

    def test_ollama_judges_are_never_rewritten(self):
        # Local judge names (gemma4:31b etc.) must reach ChatOllama as
        # given, even if one ever collided with a pinned alias.
        judge = _judge()
        assert judge.resolve_judge_model("gpt-4o-mini", "ollama") == "gpt-4o-mini"

    def test_pins_map_aliases_to_their_own_family(self):
        judge = _judge()
        for alias, snapshot in judge.JUDGE_MODEL_PINS.items():
            assert snapshot.startswith(alias), (alias, snapshot)
