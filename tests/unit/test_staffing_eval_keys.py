"""
Baseline-key format for the staffing eval.

Same contract as run_eval's reasoning suffixes: the suffix appears ONLY
when the flag was passed, so the committed cloud key never churns.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

EVALS_DIR = Path(__file__).parent.parent.parent / "evals"


def _module():
    # run_staffing_eval imports from run_eval at module load; both live
    # in evals/, which needs to be importable here.
    sys.path.insert(0, str(EVALS_DIR))
    spec = importlib.util.spec_from_file_location(
        "run_staffing_eval", EVALS_DIR / "run_staffing_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    ns = SimpleNamespace(provider="openai", model="gpt-4o-mini", k=12)
    for name, value in overrides.items():
        setattr(ns, name, value)
    return ns


class TestStaffingBaselineKey:
    def test_cloud_key_unchanged_without_flag(self):
        key = _module().baseline_key(_args())
        assert key == "openai:gpt-4o-mini|k=12|chain=match|set=staffing"
        assert "reasoning" not in key

    def test_reasoning_suffix_only_when_passed(self):
        module = _module()
        local = _args(provider="ollama", model="qwen3.5:9b", ollama_reasoning="off")
        assert module.baseline_key(local) == (
            "ollama:qwen3.5:9b|k=12|chain=match|set=staffing" "|ollama-reasoning=off"
        )
        unset = _args(provider="ollama", model="qwen3.5:9b")
        assert "reasoning" not in module.baseline_key(unset)
