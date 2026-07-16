"""
Every advertised entry point must import — and the UIs must boot.

The package ships five console scripts (pyproject [project.scripts]).
A stale module path or import-time crash in any of them is a broken
first command for a new user, and nothing else in the suite would
notice: console scripts resolve at install time, not test time. The
import test parses pyproject directly so a sixth script is covered the
day it is added.

The two Streamlit apps additionally get a real headless boot: server
process starts, /_stcore/health answers "ok". That catches what an
import test cannot (CLI wiring, server startup), and stays hermetic —
neither app touches an LLM provider until a user acts, so no API key
or Ollama server is needed.
"""

import importlib
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

UI_APPS = [
    REPO_ROOT / "src" / "ragstone" / "ui" / "streamlit_app.py",
    REPO_ROOT / "src" / "ragstone" / "ui" / "staffing_app.py",
]


def _console_scripts() -> dict:
    # Regex rather than tomllib: the repo supports Python 3.10, where
    # tomllib does not exist, and adding a TOML dependency for one
    # test block is not worth it.
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"\[project\.scripts\]\n(.*?)(?:\n\[|\Z)", text, re.DOTALL)
    assert match, "pyproject.toml has no [project.scripts] block"
    return dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', match.group(1), re.MULTILINE))


class TestConsoleScripts:
    def test_scripts_are_declared(self):
        scripts = _console_scripts()
        assert set(scripts) == {
            "ragstone",
            "ragstone-chat",
            "ragstone-api",
            "ragstone-mcp",
            "ragstone-match",
        }

    @pytest.mark.parametrize("name,target", sorted(_console_scripts().items()))
    def test_target_imports_and_is_callable(self, name, target):
        module_name, attr = target.split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), f"{name} -> {target}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.slow
@pytest.mark.parametrize("app_path", UI_APPS, ids=lambda p: p.stem)
def test_streamlit_app_boots_headless(app_path):
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    health = f"http://127.0.0.1:{port}/_stcore/health"
    try:
        deadline = time.time() + 30
        last_error = "no response"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health, timeout=1) as response:
                    if response.read().decode().strip() == "ok":
                        return  # booted
            except Exception as exc:  # not up yet
                last_error = str(exc)
            if proc.poll() is not None:
                break  # process died — fail with its output
            time.sleep(0.5)
        output = ""
        if proc.poll() is not None and proc.stdout is not None:
            output = proc.stdout.read().decode(errors="ignore")[-2000:]
        pytest.fail(
            f"{app_path.name} did not become healthy within 30s "
            f"(last error: {last_error}). Process output:\n{output}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
