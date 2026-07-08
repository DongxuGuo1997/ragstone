"""
kill -TERM under load — the ROADMAP 5.7 measure, against a real uvicorn
process: a request in flight when SIGTERM lands must complete with 200
(zero dropped in-flight requests), and the process must then exit
cleanly. The pipeline is a stub that sleeps, so what is under test is
the serving stack's drain behavior, not the LLM.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

_SERVER_SCRIPT = """
import os, time
import uvicorn
import ragstone.api.server as server
import ragstone.utils.registry as registry

class _SlowPipeline:
    provider = "openai"
    texts = ["chunk"]
    llm_proxy = None
    def get_chain(self):
        return object()
    def ask_question(self, question, session_id=None, use_cache=True):
        time.sleep(2.0)  # long enough for SIGTERM to land mid-request
        return "slow answer"
    def close(self):
        pass

registry.put_pipeline("p1", _SlowPipeline())
uvicorn.run(
    server.create_app(api_key="", max_concurrency=4),
    host="127.0.0.1",
    port=int(os.environ["RAGSTONE_TEST_PORT"]),
    log_level="warning",
)
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_sigterm_drains_in_flight_requests():
    port = _free_port()
    env = {
        **os.environ,
        "RAGSTONE_TEST_PORT": str(port),
        "RAGSTONE_REGISTRY_PERSIST": "off",
    }
    proc = subprocess.Popen([sys.executable, "-c", _SERVER_SCRIPT], env=env)
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        else:
            pytest.fail("server never became ready")

        result = {}

        def _slow_ask():
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/pipelines/p1/ask",
                data=b'{"question": "q"}',
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result["status"] = response.status
                result["body"] = response.read().decode()

        asker = threading.Thread(target=_slow_ask)
        asker.start()
        time.sleep(0.5)  # the 2s ask is now in flight
        proc.send_signal(signal.SIGTERM)
        asker.join(timeout=30)

        assert result.get("status") == 200, "in-flight request was dropped"
        assert "slow answer" in result.get("body", "")
        # Uvicorn drains, restores the default handler, then RE-RAISES
        # the captured SIGTERM (proper unix semantics) — so "clean exit
        # after drain" is 0 or death-by-SIGTERM, never a hang or error.
        assert proc.wait(timeout=15) in (0, -signal.SIGTERM)
    finally:
        if proc.poll() is None:
            proc.kill()
