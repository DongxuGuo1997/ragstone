"""
Unit tests for the exact-match response cache (no network required).
"""

from ragstone.rag.pipeline import QueryResultCache


class TestQueryResultCache:
    def test_exact_hit_and_session_scoping(self):
        cache = QueryResultCache(max_size=10, ttl_seconds=60)
        cache.cache_response("q", "answer", "session-a")

        assert cache.get_response("q", "session-a") == "answer"
        assert cache.get_response("q", "session-b") is None
        assert cache.get_response("other", "session-a") is None

    def test_different_questions_never_collide(self):
        # The removed "normalized" tier collapsed these onto one key and
        # served one answer for both — exact matching must not.
        cache = QueryResultCache()
        cache.cache_response("training on GPUs", "answer-on", "s")

        assert cache.get_response("training with GPUs", "s") is None

    def test_expired_entries_are_not_served(self):
        cache = QueryResultCache(ttl_seconds=1)
        cache.cache_response("q", "a", "s")
        key = cache._key("q", "s")
        timestamp, response = cache._entries[key]
        cache._entries[key] = (timestamp - 5, response)

        assert cache.get_response("q", "s") is None
        assert key not in cache._entries

    def test_eviction_is_lru_not_fifo(self):
        cache = QueryResultCache(max_size=2)
        cache.cache_response("q1", "a1", "s")
        cache.cache_response("q2", "a2", "s")
        assert cache.get_response("q1", "s") == "a1"  # refresh q1

        cache.cache_response("q3", "a3", "s")  # must evict q2, not q1

        assert cache.get_response("q1", "s") == "a1"
        assert cache.get_response("q2", "s") is None
        assert cache.get_response("q3", "s") == "a3"

    def test_stats_track_hits_and_misses(self):
        cache = QueryResultCache()
        cache.cache_response("q", "a", "s")
        cache.get_response("q", "s")
        cache.get_response("nope", "s")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["hit_rate"] == "50.0%"

    def test_clear_cache(self):
        cache = QueryResultCache()
        cache.cache_response("q", "a", "s")
        cache.clear_cache()

        assert cache.get_response("q", "s") is None
