"""
Unit tests for the thread-safe pipeline registry (no network required).

The registry is shared by the MCP server and the REST API; both run
handlers concurrently and hand long operations to worker threads, so
registry access must be atomic. These tests exercise the accessors
directly with real threads.
"""

import threading

import pytest

from ragstone.utils import registry


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts and ends with an empty registry."""
    registry.clear_pipelines()
    yield
    registry.clear_pipelines()


def test_put_get_pop_roundtrip():
    sentinel = object()
    registry.put_pipeline("x", sentinel)
    assert registry.get_pipeline("x") is sentinel
    assert registry.pop_pipeline("x") is sentinel
    assert registry.get_pipeline("x") is None
    assert registry.pop_pipeline("x") is None  # popping again is a safe no-op


def test_concurrent_pop_returns_object_to_exactly_one_thread():
    # The core race the lock fixes: two callers must never both believe
    # they own a pipeline. A concurrent pop must hand the object to one
    # winner.
    sentinel = object()
    registry.put_pipeline("dup", sentinel)

    winners = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()  # maximize contention on the pop
        popped = registry.pop_pipeline("dup")
        if popped is not None:
            winners.append(popped)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert winners == [sentinel]  # exactly one thread won
    assert registry.get_pipeline("dup") is None


def test_concurrent_put_pop_leaves_registry_consistent():
    # Hammer the registry from many threads; it must never raise (e.g. a
    # "dictionary changed size during iteration") and must end empty.
    def churn(idx):
        key = f"p{idx}"
        for _ in range(200):
            registry.put_pipeline(key, object())
            registry.snapshot_pipelines()  # read side, would raise if unlocked
            registry.pop_pipeline(key)

    threads = [threading.Thread(target=churn, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert registry.snapshot_pipelines() == []


def test_mcp_server_uses_the_shared_registry():
    # The MCP module's accessors must be the registry's, not a copy —
    # otherwise the REST API and MCP server would see different pipelines.
    from ragstone.mcp import mcp_server_fastmcp as srv

    sentinel = object()
    registry.put_pipeline("shared", sentinel)
    assert srv._get_pipeline("shared") is sentinel
