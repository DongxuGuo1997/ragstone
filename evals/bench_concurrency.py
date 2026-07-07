#!/usr/bin/env python3
"""Concurrency benchmark: does the API's thread model hold to its cap?

The server's design claims — worker-thread offload keeps the event loop
free, the semaphore refuses (429) instead of queueing, per-request
metrics stay isolated — were all reviewed and unit-tested, but p95 under
real concurrent load was never measured. This does that, cheaply:

- The server runs in-process (uvicorn in a thread) with a REAL pipeline
  over the bundled eval corpus.
- A warm phase asks WARM_QUESTIONS distinct questions once (real LLM
  calls, ~2 cents total); the load phase then hammers those same
  questions with the response cache ON, so it measures SERVER mechanics
  (routing, auth, threads, cache) rather than OpenAI's latency.
- One small uncached wave at the semaphore cap measures true end-to-end
  concurrency (unique questions, real generation).

    python evals/bench_concurrency.py            # needs OPENAI_API_KEY

Results feed Experiment 16 in EXPERIMENTS.md.
"""

import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import requests  # noqa: E402
import uvicorn  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CORPUS = str(Path(__file__).parent / "corpus")
CAP = 8  # the server's default RAGSTONE_API_MAX_CONCURRENCY

WARM_QUESTIONS = [
    "How long is the Helios MK-3 warranty?",
    "What torque do the MK-3 mounting bolts require?",
    "What does inverter fault code E-42 mean?",
    "How often should Borealis BX-2 panels be cleaned?",
    "Who commanded the Aurora-7 mission?",
    "What is the Aurora-9 mission's target?",
    "How long do Solstice beans rest after roasting?",
    "What is the Violet Line?",
]


def start_server() -> uvicorn.Server:
    from ragstone.api.server import create_app
    from ragstone.config.settings import get_config

    # The cached load phase measures SERVER mechanics; the response cache
    # is off by default, so enable it explicitly (in-process config —
    # the server runs in this process).
    get_config().cache.enable_response_cache = True

    config = uvicorn.Config(
        create_app(api_key="", max_concurrency=CAP),
        host="127.0.0.1",
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        try:
            if requests.get(f"{BASE}/health", timeout=1).ok:
                return server
        except requests.ConnectionError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def setup_pipeline() -> None:
    requests.post(
        f"{BASE}/pipelines", json={"provider": "openai", "pipeline_id": "bench"}
    ).raise_for_status()
    requests.post(
        f"{BASE}/pipelines/bench/documents", json={"data_dir": CORPUS}
    ).raise_for_status()
    requests.post(
        f"{BASE}/pipelines/bench/retriever", json={"chain_type": "simple"}
    ).raise_for_status()


def ask(question: str, use_cache: bool = True) -> Tuple[float, int]:
    start = time.perf_counter()
    response = requests.post(
        f"{BASE}/pipelines/bench/ask",
        json={"question": question, "use_cache": use_cache},
        timeout=120,
    )
    return (time.perf_counter() - start) * 1000, response.status_code


def load_phase(workers: int, requests_total: int) -> dict:
    latencies: List[float] = []
    refused = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(ask, WARM_QUESTIONS[i % len(WARM_QUESTIONS)])
            for i in range(requests_total)
        ]
        for future in futures:
            ms, status = future.result()
            if status == 429:
                refused += 1
            elif status == 200:
                latencies.append(ms)
    return {
        "workers": workers,
        "ok": len(latencies),
        "refused_429": refused,
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_ms": (
            round(statistics.quantiles(latencies, n=20)[18], 1)
            if len(latencies) >= 20
            else None
        ),
    }


def main() -> int:
    server = start_server()
    try:
        print("Setting up pipeline (loads + embeds the eval corpus)...")
        setup_pipeline()

        print("Warm phase: real generation for each distinct question")
        for question in WARM_QUESTIONS:
            ms, status = ask(question)
            print(f"  {status} {ms:7.0f} ms  {question[:50]}")

        print("\nCached load (server mechanics):")
        print("| workers | requests | ok | 429 | p50 (ms) | p95 (ms) |")
        print("|---|---|---|---|---|---|")
        for workers in (1, 8, 16, 32):
            result = load_phase(workers, requests_total=workers * 12)
            print(
                f"| {result['workers']} | {workers * 12} | {result['ok']} "
                f"| {result['refused_429']} | {result['p50_ms']} "
                f"| {result['p95_ms']} |"
            )

        print("\nUncached wave at the cap (true end-to-end concurrency):")
        unique = [
            f"What is the warranty period, asked variant {i}?" for i in range(CAP)
        ]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CAP) as pool:
            results = list(pool.map(lambda q: ask(q, use_cache=False), unique))
        wall = time.perf_counter() - start
        oks = [ms for ms, status in results if status == 200]
        print(
            f"  {len(oks)}/{CAP} ok, wall {wall:.1f}s, "
            f"p50 {statistics.median(oks):.0f} ms, max {max(oks):.0f} ms"
        )
        sequential_estimate = sum(oks) / 1000
        print(
            f"  concurrency payoff: ~{sequential_estimate / wall:.1f}x vs "
            f"sequential (sum of latencies {sequential_estimate:.1f}s)"
        )
    finally:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
