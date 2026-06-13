"""Exact-match response cache for the RAG pipeline.

A deliberately simple, thread-safe cache keyed by (question, context_hash).
Earlier versions had normalized and semantic-similarity tiers; those could
return a cached answer for a materially different question and the semantic
tier spent embedding API calls just to probe the cache. Exact matching is
cheap and can't be wrong — see EXPERIMENTS.md for the reasoning.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Counters for the response cache."""

    hits: int = 0
    misses: int = 0

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_queries
        return self.hits / total if total > 0 else 0.0


class QueryResultCache:
    """Exact-match response cache with TTL and LRU eviction.

    Entries are keyed by (question, context_hash) so answers never leak
    across sessions or document sets.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[Tuple[str, str], Tuple[float, str]]" = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    @staticmethod
    def _key(question: str, context_hash: str = "") -> Tuple[str, str]:
        return question.strip(), context_hash or ""

    def get_response(self, question: str, context_hash: str = "") -> Optional[str]:
        """Return the cached response for this exact question, or None."""
        key = self._key(question, context_hash)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            timestamp, response = entry
            if time.time() - timestamp > self.ttl_seconds:
                del self._entries[key]
                self.stats.misses += 1
                return None
            self._entries.move_to_end(key)  # refresh LRU position
            self.stats.hits += 1
            return response

    def cache_response(
        self, question: str, response: str, context_hash: str = ""
    ) -> None:
        """Store a response, evicting least-recently-used entries if full."""
        key = self._key(question, context_hash)
        with self._lock:
            self._entries[key] = (time.time(), response)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache size and hit/miss counters."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": f"{self.stats.hit_rate:.1%}",
            }

    def clear_cache(self) -> None:
        """Drop all cached responses."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        if count:
            logger.info(f"Cleared {count} cached responses")


# Global cache instance, created on first use.
_query_cache: Optional[QueryResultCache] = None
_query_cache_lock = threading.Lock()


def get_query_cache() -> QueryResultCache:
    """Get or create the global response cache using configuration."""
    global _query_cache
    if _query_cache is None:
        with _query_cache_lock:
            if _query_cache is None:
                from ..config.settings import get_config

                config = get_config()
                _query_cache = QueryResultCache(
                    max_size=config.cache.response_cache_size,
                    ttl_seconds=config.cache.response_cache_ttl,
                )
    return _query_cache


def is_response_cache_enabled() -> bool:
    """Check if the response cache is enabled in configuration."""
    from ..config.settings import get_config

    return get_config().cache.enable_response_cache
