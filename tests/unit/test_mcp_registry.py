"""
Unit tests for the thread-safe MCP pipeline registry (no network required).

The FastMCP tools run concurrently on the server's event loop and hand long
operations to worker threads, so registry access must be atomic. These tests
exercise the accessor helpers directly with real threads.
"""

import threading

import pytest

from ragstone.mcp import mcp_server_fastmcp as srv


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts and ends with an empty registry."""
    with srv._pipelines_lock:
        srv._pipelines.clear()
    yield
    with srv._pipelines_lock:
        srv._pipelines.clear()


def test_put_get_pop_roundtrip():
    sentinel = object()
    srv._put_pipeline("x", sentinel)
    assert srv._get_pipeline("x") is sentinel
    assert srv._pop_pipeline("x") is sentinel
    assert srv._get_pipeline("x") is None
    assert srv._pop_pipeline("x") is None  # popping again is a safe no-op


def test_concurrent_pop_returns_object_to_exactly_one_thread():
    # The core race the lock fixes: two tools must never both believe they
    # own a pipeline. A concurrent pop must hand the object to one winner.
    sentinel = object()
    srv._put_pipeline("dup", sentinel)

    winners = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()  # maximize contention on the pop
        popped = srv._pop_pipeline("dup")
        if popped is not None:
            winners.append(popped)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert winners == [sentinel]  # exactly one thread won
    assert srv._get_pipeline("dup") is None


def test_concurrent_put_pop_leaves_registry_consistent():
    # Hammer the registry from many threads; it must never raise (e.g. a
    # "dictionary changed size during iteration") and must end empty.
    def churn(idx):
        key = f"p{idx}"
        for _ in range(200):
            srv._put_pipeline(key, object())
            srv._snapshot_pipelines()  # read side, would raise if unlocked
            srv._pop_pipeline(key)

    threads = [threading.Thread(target=churn, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert srv._snapshot_pipelines() == []
