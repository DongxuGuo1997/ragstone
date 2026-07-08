#!/usr/bin/env python3
"""Overhead benchmark: what do the enterprise features cost per request?

Arc 1 (ROADMAP 5.2/5.3/5.7) put real machinery on the request path:
request-id + audit middleware, named-key auth with quotas, a Prometheus
observer, no-op OTel spans, and a registry lookup that can lazily
restore. Each piece was designed to be cheap — this measures whether the
sum actually is, by comparing the SAME load against two source trees:

    python evals/bench_overhead.py --label new
    python evals/bench_overhead.py --label old --src <pre-arc1-worktree>/src

The pipeline is a stub that answers instantly, so every microsecond
measured is server mechanics, not LLM time. No OPENAI_API_KEY needed.
The server runs in-process (uvicorn in a thread) and the load generator
hits it over loopback HTTP with keep-alive sessions — absolute numbers
are same-process loopback, so read the old-vs-new DELTA, not the totals.

A microbenchmark section (new tree only) attributes the delta to the
individual features. Results feed Experiment 20 in EXPERIMENTS.md.
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

API_KEY = "bench-key-0123456789abcdef0123456789abcdef"
PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"


def _percentiles(samples_ms: List[float]) -> Dict[str, float]:
    cuts = statistics.quantiles(samples_ms, n=100)
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 3),
        "p50_ms": round(statistics.median(samples_ms), 3),
        "p95_ms": round(cuts[94], 3),
        "p99_ms": round(cuts[98], 3),
    }


class StubPipeline:
    """Duck-typed pipeline whose ask is instant: overhead-only requests."""

    provider = "openai"
    llm_proxy = None
    _chain_type = "simple"

    def __init__(self) -> None:
        self.texts: list = []

    def get_chain(self):
        return object()

    def ask_question(
        self, question: str, session_id: Optional[str] = None, use_cache: bool = True
    ) -> str:
        return "stub answer"

    def close(self) -> None:
        pass


def start_server() -> "object":
    import uvicorn

    from ragstone.api.server import create_app
    from ragstone.utils.registry import put_pipeline

    put_pipeline("bench", StubPipeline())
    config = uvicorn.Config(
        # Cap high enough that the semaphore never refuses: 429s would
        # measure the limiter, not the per-request overhead.
        create_app(api_key=API_KEY, max_concurrency=64),
        host="127.0.0.1",
        port=PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    start = time.perf_counter()
    thread.start()
    import requests

    for _ in range(200):
        try:
            if requests.get(f"{BASE}/health", timeout=1).ok:
                print(
                    f"startup to first healthy probe: "
                    f"{(time.perf_counter() - start) * 1000:.0f} ms"
                )
                return server
        except requests.ConnectionError:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def timed_get(session, path: str) -> Tuple[float, int]:
    start = time.perf_counter()
    response = session.get(f"{BASE}{path}", timeout=10)
    return (time.perf_counter() - start) * 1000, response.status_code


def timed_ask(session) -> Tuple[float, int]:
    start = time.perf_counter()
    response = session.post(
        f"{BASE}/pipelines/bench/ask",
        json={"question": "bench?", "use_cache": False},
        headers={"X-API-Key": API_KEY},
        timeout=10,
    )
    return (time.perf_counter() - start) * 1000, response.status_code


def sequential_phase(make_session, fn, warmup: int, n: int) -> Dict[str, float]:
    session = make_session()
    for _ in range(warmup):
        fn(session)
    samples = []
    for _ in range(n):
        ms, status = fn(session)
        assert status == 200, f"unexpected status {status}"
        samples.append(ms)
    return _percentiles(samples)


def concurrent_phase(make_session, workers: int, n: int) -> Dict[str, float]:
    local = threading.local()

    def worker_ask() -> Tuple[float, int]:
        if not hasattr(local, "session"):
            local.session = make_session()
        return timed_ask(local.session)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _: worker_ask(), range(workers * 10)))  # warm
        start = time.perf_counter()
        results = list(pool.map(lambda _: worker_ask(), range(n)))
        wall = time.perf_counter() - start
    oks = [ms for ms, status in results if status == 200]
    assert len(oks) == n, f"{n - len(oks)} requests failed"
    stats = _percentiles(oks)
    stats["rps"] = round(n / wall)
    return stats


# --------------------------------------------------------------------------
# Microbenchmarks (new tree only): attribute the delta per feature.
# --------------------------------------------------------------------------


def _per_call_us(fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start) / iterations * 1_000_000


def microbench() -> Dict[str, float]:
    results: Dict[str, float] = {}

    try:
        from ragstone.api.keys import ApiKeyRecord, ApiKeyStore
    except ImportError:
        print("(old tree: no keys.py — skipping microbenchmarks)")
        return results

    from langchain_core.documents import Document

    from ragstone.utils.observability import _observers, track_request
    from ragstone.utils.registry import (
        get_or_restore_pipeline,
        get_pipeline,
        list_persisted,
        persist_pipeline,
    )

    for n_keys in (1, 5, 25):
        store = ApiKeyStore(
            [ApiKeyRecord(name=f"k{i}", key=f"key-{i}" * 8) for i in range(n_keys)]
        )
        target = f"key-{n_keys - 1}" * 8
        results[f"authenticate_{n_keys}_keys_us"] = round(
            _per_call_us(lambda: store.authenticate(target), 20_000), 2
        )

    quota_store = ApiKeyStore([ApiKeyRecord(name="q", key="k" * 32, rpm=10**9)])
    results["check_and_count_us"] = round(
        _per_call_us(lambda: quota_store.check_and_count("q"), 20_000), 2
    )

    # track_request: full enter/exit including the usage callback, the
    # non-recording OTel span, the log line, and (with/without) the
    # Prometheus observer — the per-ask instrumentation cost.
    def one_tracked_request() -> None:
        with track_request("bench-session", "simple"):
            pass

    results["track_request_with_observer_us"] = round(
        _per_call_us(one_tracked_request, 5_000), 1
    )
    saved = list(_observers)
    _observers.clear()
    results["track_request_no_observer_us"] = round(
        _per_call_us(one_tracked_request, 5_000), 1
    )
    _observers.extend(saved)

    # The extra worker-thread hop _get_or_404 pays so a lazy restore can
    # never block the event loop.
    from anyio import to_thread

    async def measure_hop() -> float:
        start = time.perf_counter()
        for _ in range(2_000):
            await to_thread.run_sync(lambda: None)
        return (time.perf_counter() - start) / 2_000 * 1_000_000

    results["to_thread_hop_us"] = round(asyncio.run(measure_hop()), 1)

    # Registry: the warm ask-path lookup, with and without the restore
    # wrapper (persistence must cost nothing once the pipeline is loaded),
    # plus the one-off config-time persist and the per-/ready listing.
    from ragstone.utils import registry

    registry.put_pipeline("bench-micro", StubPipeline())
    results["get_pipeline_us"] = round(
        _per_call_us(lambda: get_pipeline("bench-micro"), 100_000), 3
    )
    results["get_or_restore_warm_us"] = round(
        _per_call_us(lambda: get_or_restore_pipeline("bench-micro"), 100_000), 3
    )

    with tempfile.TemporaryDirectory() as tmp:
        import os

        os.environ["RAGSTONE_REGISTRY_PERSIST"] = "on"
        os.environ["RAGSTONE_REGISTRY_DIR"] = tmp
        stub = StubPipeline()
        stub.texts = [
            Document(page_content=f"chunk {i} " + "x" * 200, metadata={"i": i})
            for i in range(1_000)
        ]

        class _NamedLLM:
            def get_model_name(self) -> str:
                return "gpt-4o-mini"

        stub.llm_proxy = _NamedLLM()
        start = time.perf_counter()
        assert persist_pipeline("bench-micro-persist", stub)
        results["persist_1000_chunks_ms"] = round(
            (time.perf_counter() - start) * 1000, 1
        )
        results["list_persisted_1_manifest_us"] = round(
            _per_call_us(lambda: list_persisted(), 2_000), 1
        )
        os.environ["RAGSTONE_REGISTRY_PERSIST"] = "off"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(Path(__file__).parent.parent / "src"))
    parser.add_argument("--label", default="current")
    parser.add_argument("--n", type=int, default=2_000)
    args = parser.parse_args()

    sys.path.insert(0, args.src)
    import os

    # Deterministic posture: tracing off (spans non-recording, the default)
    # and persistence off for the serving phases — the warm ask path never
    # touches it either way; the microbench section prices it directly.
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    os.environ["RAGSTONE_REGISTRY_PERSIST"] = "off"

    # Production-like logging cost: request/audit lines are formatted and
    # written (to a scratch file), not silently dropped for lack of handler.
    log_path = Path(tempfile.gettempdir()) / f"bench_overhead_{args.label}.log"
    logging.basicConfig(level=logging.INFO, filename=str(log_path), filemode="w")

    import ragstone

    print(f"tree under test: {ragstone.__file__}")

    import requests

    server = start_server()
    try:
        health = sequential_phase(
            requests.Session, lambda s: timed_get(s, "/health"), 100, args.n
        )
        ask_seq = sequential_phase(requests.Session, timed_ask, 100, args.n)
        ask_conc = concurrent_phase(requests.Session, workers=16, n=args.n * 2)

        print(f"\n| phase ({args.label}) | mean | p50 | p95 | p99 | rps |")
        print("|---|---|---|---|---|---|")
        for name, stats in (
            ("GET /health x" + str(args.n), health),
            ("POST /ask x" + str(args.n), ask_seq),
            ("POST /ask, 16 workers", ask_conc),
        ):
            print(
                f"| {name} | {stats['mean_ms']} | {stats['p50_ms']} "
                f"| {stats['p95_ms']} | {stats['p99_ms']} "
                f"| {stats.get('rps', '-')} |"
            )

        micro = microbench()
        if micro:
            print("\nMicrobenchmarks (per call):")
            for key, value in micro.items():
                print(f"  {key:40s} {value}")

        summary = {
            "label": args.label,
            "health": health,
            "ask_seq": ask_seq,
            "ask_conc": ask_conc,
            "micro": micro,
        }
        print("\nJSON: " + json.dumps(summary))
    finally:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
