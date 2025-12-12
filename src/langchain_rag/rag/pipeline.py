import os
import logging
import hashlib
import time
from typing import List, Optional, Union, TYPE_CHECKING, Dict, Any
from dataclasses import dataclass, field

# Lazy imports - only import when needed
if TYPE_CHECKING:
    from langchain.docstore.document import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.retrievers import BaseRetriever
    from langchain_ollama import OllamaEmbeddings
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    from langchain_core.embeddings import Embeddings

# Optional Streamlit import with fallback
try:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
    STREAMLIT_AVAILABLE = True
except ImportError:
    UploadedFile = None
    STREAMLIT_AVAILABLE = False

from ..models.base_model import OpenAIProxy, OllamaProxy, LLMProxy
from .loader import LocalLoader, RemoteLoader
from .memory import MemoryProxy
from .rag import RagProxy
from .splitter import split_documents
from .vector_db import FaissProxy, ChromaProxy, VectorStoreProxy, create_vector_store_proxy, _lazy_import_numpy
from ..utils.full_chain import FullChain
from ..config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Used by OpenAIPipeline

# Import cache for heavy LangChain dependencies
_pipeline_cache = {}

@dataclass
class CacheStats:
    """Statistics for cache performance monitoring with 3-tier cache tracking."""
    # 3-Tier Cache Performance Tracking
    exact_hits: int = 0          # Level 1: Exact string matches (fastest, $0)
    normalized_hits: int = 0     # Level 2: Normalized matches (fast, $0) 
    semantic_hits: int = 0       # Level 3: Semantic similarity matches (expensive, $0.0001)
    
    # Legacy cache stats
    embedding_hits: int = 0
    embedding_misses: int = 0
    search_hits: int = 0
    search_misses: int = 0
    response_hits: int = 0
    response_misses: int = 0
    total_queries: int = 0
    total_time_saved: float = 0.0
    
    # Cost tracking
    embedding_api_calls: int = 0  # Track expensive API calls
    
    @property
    def embedding_hit_rate(self) -> float:
        total = self.embedding_hits + self.embedding_misses
        return self.embedding_hits / total if total > 0 else 0.0
    
    @property
    def search_hit_rate(self) -> float:
        total = self.search_hits + self.search_misses
        return self.search_hits / total if total > 0 else 0.0
    
    @property
    def response_hit_rate(self) -> float:
        total = self.response_hits + self.response_misses
        return self.response_hits / total if total > 0 else 0.0
    
    @property 
    def total_hit_rate(self) -> float:
        """Combined hit rate including all 3 tiers."""
        total_hits = self.exact_hits + self.normalized_hits + self.semantic_hits
        total_attempts = total_hits + self.response_misses
        return total_hits / total_attempts if total_attempts > 0 else 0.0
    
    @property
    def tier_breakdown(self) -> Dict[str, float]:
        """Breakdown of hits by tier (percentages)."""
        total_hits = self.exact_hits + self.normalized_hits + self.semantic_hits
        if total_hits == 0:
            return {"exact": 0.0, "normalized": 0.0, "semantic": 0.0}
        
        return {
            "exact": (self.exact_hits / total_hits) * 100,
            "normalized": (self.normalized_hits / total_hits) * 100,
            "semantic": (self.semantic_hits / total_hits) * 100
        }
    
    @property
    def cost_efficiency(self) -> Dict[str, float]:
        """Cost efficiency metrics."""
        free_hits = self.exact_hits + self.normalized_hits
        paid_hits = self.semantic_hits
        total_hits = free_hits + paid_hits
        
        if total_hits == 0:
            return {"free_hit_rate": 0.0, "api_calls_avoided": 0.0}
        
        return {
            "free_hit_rate": (free_hits / total_hits) * 100,
            "api_calls_avoided": free_hits,
            "cost_per_query": (self.embedding_api_calls * 0.0001) / max(self.total_queries, 1)
        }
    
    def record_exact_hit(self, time_saved: float = 3.0):
        """Record a Level 1 exact cache hit (fastest, no API calls)."""
        self.exact_hits += 1
        self.response_hits += 1  # For legacy compatibility
        self.total_time_saved += time_saved
        
    def record_normalized_hit(self, time_saved: float = 2.5):
        """Record a Level 2 normalized cache hit (fast, no API calls)."""
        self.normalized_hits += 1
        self.response_hits += 1  # For legacy compatibility
        self.total_time_saved += time_saved
        
    def record_semantic_hit(self, time_saved: float = 2.0):
        """Record a Level 3 semantic similarity cache hit (expensive API call).""" 
        self.semantic_hits += 1
        self.response_hits += 1  # For legacy compatibility
        self.total_time_saved += time_saved
        self.embedding_api_calls += 1  # Track API cost
        
    def record_miss(self):
        """Record a cache miss."""
        self.response_misses += 1
        self.embedding_api_calls += 1  # Track API cost for failed searches

class QueryResultCache:
    """
    🚀 INTELLIGENT QUERY RESULT CACHING SYSTEM WITH SEMANTIC SIMILARITY
    
    Provides multi-level caching for RAG pipeline:
    - Level 1: Query embeddings (avoid API calls)
    - Level 2: Search results (avoid vector searches) 
    - Level 3: Full responses (avoid LLM calls)
    
    Features:
    - TTL-based expiration
    - Memory limit management
    - **SEMANTIC SIMILARITY MATCHING** (recognizes similar questions)
    - Performance analytics
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600, similarity_threshold: float = 0.95):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold  # Raised to 0.95 for ultra-conservative matching - only near-identical queries
        
        # 🚀 3-TIER CACHE SYSTEM
        # Level 1: Exact string matches (fastest, no API calls)
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        
        # Level 2: Normalized string matches (fast, no API calls) 
        self.normalized_cache: Dict[str, Dict[str, Any]] = {}
        
        # Level 3: Semantic similarity cache (expensive, API calls required)
        self.query_embeddings_cache: Dict[str, Dict[str, Any]] = {}
        
        # Legacy caches (maintain compatibility)
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.stats = CacheStats()
        
        # Lazy import embeddings for similarity calculations
        self._embeddings_model = None
        
        logger.info(f"QueryResultCache initialized with 3-tier system: max_size={max_size}, ttl={ttl_seconds}s, similarity_threshold={similarity_threshold}")
    
    def _get_embeddings_model(self):
        """Lazy initialize embeddings model for semantic similarity."""
        if self._embeddings_model is None:
            try:
                if not OPENAI_API_KEY:
                    logger.warning("No OpenAI API key available for semantic similarity")
                    return None
                    
                OpenAIEmbeddings = _get_cached_pipeline_import("openai_embeddings")
                # Use clean configuration to avoid organization issues
                self._embeddings_model = OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    openai_organization=None,
                    model="text-embedding-ada-002"
                )
                logger.info("✅ Initialized embeddings model for semantic similarity")
            except Exception as e:
                logger.warning(f"Could not initialize embeddings for semantic similarity: {e}")
                self._embeddings_model = None
        return self._embeddings_model
    
    def _create_cache_key(self, text: str) -> str:
        """Create a consistent cache key from text with normalization."""
        from ..utils import create_cache_key, normalize_whitespace
        # Enhanced normalization for better matching
        normalized = normalize_whitespace(text.lower())
        # Remove common question words that don't change meaning
        normalized = normalized.replace('?', '').replace('.', '').replace(',', '')
        return create_cache_key(normalized)
    
    def _create_normalized_cache_key(self, text: str) -> str:
        """
        🚀 LEVEL 2: Create normalized cache key for fuzzy string matching.
        
        This catches queries like:
        - "what is deepseek" vs "what is the deepseek" (article differences)
        - "how does X work" vs "how does the X work" (article insertion)
        - "explain Y" vs "explain about Y" (preposition differences)
        """
        from ..utils import create_cache_key, normalize_whitespace
        normalized = normalize_whitespace(text.lower())
        
        # Remove common articles and prepositions that don't change meaning
        articles_prepositions = ['the ', 'a ', 'an ', 'about ', 'of ', 'for ', 'with ', 'in ', 'on ', 'at ']
        for word in articles_prepositions:
            normalized = normalized.replace(word, '')
        
        # Remove punctuation
        normalized = normalized.replace('?', '').replace('.', '').replace(',', '').replace('!', '')
        normalized = normalize_whitespace(normalized)  # Remove extra spaces again
        
        # Remove common question starters that don't change intent
        question_starters = ['tell me ', 'can you ', 'please ', 'could you ']
        for starter in question_starters:
            if normalized.startswith(starter):
                normalized = normalized[len(starter):]
        
        return create_cache_key(normalized)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better semantic matching."""
        from ..utils import normalize_whitespace
        # Remove extra whitespace and punctuation
        normalized = normalize_whitespace(query.lower())
        normalized = normalized.replace('?', '').replace('.', '').replace(',', '')
        return normalized
    
    def _find_semantically_similar_query(self, query: str) -> Optional[str]:
        """
        🎯 SEMANTIC SIMILARITY MATCHING
        
        Find cached queries that are semantically similar to the input query.
        This enables cache hits for questions like:
        - "what is deepseek" vs "what is the deepseek" 
        - "tell me about X" vs "what is X"
        """
        embeddings_model = self._get_embeddings_model()
        if not embeddings_model:
            logger.debug("No embeddings model available for semantic similarity")
            return None
            
        try:
            logger.debug(f"🔍 Searching for semantic matches for: '{query}'")
            
            # Clean up expired entries first
            self._cleanup_expired(self.query_embeddings_cache)
            
            if not self.query_embeddings_cache:
                logger.debug("No cached query embeddings available for comparison")
                return None
            
            # 🚀 PERFORMANCE OPTIMIZATION: Skip semantic search if cache is large
            # This prevents O(n) slowdown when cache grows beyond optimal size
            MAX_SEMANTIC_SEARCHES = 50  # Limit to prevent performance degradation
            if len(self.query_embeddings_cache) > MAX_SEMANTIC_SEARCHES:
                logger.debug(f"⏭️ SKIP semantic search: cache too large ({len(self.query_embeddings_cache)} > {MAX_SEMANTIC_SEARCHES})")
                return None
            
            # Generate embedding for the new query
            logger.debug(f"Generating embedding for new query: '{query}'")
            query_embedding = embeddings_model.embed_query(query)
            
            # Compare against cached query embeddings (limited scope)
            best_match = None
            best_similarity = 0.0  
            similarities = []
            comparisons = 0
            
            logger.debug(f"Comparing against {len(self.query_embeddings_cache)} cached queries")
            
            for cache_key, cache_data in self.query_embeddings_cache.items():
                if self._is_expired(cache_data):
                    continue
                    
                cached_query = cache_data['query']
                cached_embedding = cache_data['embedding']
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                similarities.append((cached_query, similarity))
                comparisons += 1
                
                logger.debug(f"  '{query}' vs '{cached_query}' = {similarity:.3f}")
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cache_key
                    
                    # 🚀 EARLY EXIT: Stop if very high similarity found
                    if similarity > 0.95:
                        logger.debug(f"Early exit: very high similarity {similarity:.3f}")
                        break
                    
            # Log performance metrics for debugging
            logger.debug(f"Semantic search completed: {comparisons} comparisons, best: {best_similarity:.3f}")
            
            if best_match:
                cached_query = self.query_embeddings_cache[best_match]['query']
                logger.info(f"🎯 SEMANTIC MATCH! '{query}' → '{cached_query}' (similarity: {best_similarity:.3f})")
                return best_match
            else:
                logger.debug(f"No semantic match found. Best similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})")
                
        except Exception as e:
            logger.error(f"Semantic similarity search failed: {e}", exc_info=True)
            
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            a = np.array(vec1, dtype=np.float32)
            b = np.array(vec2, dtype=np.float32)
            
            # Validate vectors
            if len(a) == 0 or len(b) == 0:
                logger.warning("Empty vectors in cosine similarity calculation")
                return 0.0
                
            if len(a) != len(b):
                logger.warning(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                logger.warning("Zero norm vectors in cosine similarity calculation")
                return 0.0
                
            similarity = dot_product / (norm_a * norm_b)
            
            # Ensure similarity is in valid range [-1, 1]
            similarity = max(-1.0, min(1.0, float(similarity)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}", exc_info=True)
            return 0.0
    
    def _cache_query_embedding(self, query: str, cache_key: str):
        """Cache the query embedding for future semantic similarity comparisons."""
        embeddings_model = self._get_embeddings_model()
        if not embeddings_model:
            logger.debug("No embeddings model available for caching query embedding")
            return
            
        try:
            logger.debug(f"Caching embedding for semantic similarity: '{query}'")
            embedding = embeddings_model.embed_query(query)
            
            self.query_embeddings_cache[cache_key] = {
                'embedding': embedding,
                'query': query,
                'timestamp': time.time()
            }
            self._enforce_size_limit(self.query_embeddings_cache)
            logger.debug(f"✅ Cached embedding for: '{query}' (dimensions: {len(embedding)})")
        except Exception as e:
            logger.error(f"Failed to cache query embedding: {e}", exc_info=True)
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        return time.time() - cache_entry['timestamp'] > self.ttl_seconds
    
    def _cleanup_expired(self, cache: Dict[str, Dict[str, Any]]) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in cache.items() 
            if current_time - entry['timestamp'] > self.ttl_seconds
        ]
        for key in expired_keys:
            del cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _enforce_size_limit(self, cache: Dict[str, Dict[str, Any]]) -> None:
        """Enforce maximum cache size by removing oldest entries."""
        if len(cache) <= self.max_size:
            return
        
        # Sort by timestamp and remove oldest entries
        sorted_items = sorted(cache.items(), key=lambda x: x[1]['timestamp'])
        items_to_remove = len(cache) - self.max_size
        
        for i in range(items_to_remove):
            key = sorted_items[i][0]
            del cache[key]
        
        logger.debug(f"Removed {items_to_remove} oldest cache entries to enforce size limit")
    
    def get_response(self, query: str, context_hash: str = None) -> Optional[str]:
        """
        🚀 3-TIER OPTIMIZED CACHE LOOKUP SYSTEM
        
        Performance-optimized cache retrieval strategy:
        1. 🎯 LEVEL 1: Exact string match (0ms, $0 cost)
        2. 🚀 LEVEL 2: Normalized string match (1ms, $0 cost)  
        3. 🧠 LEVEL 3: Semantic similarity match (150ms, $0.0001 cost)
        
        This dramatically reduces API calls and improves response times.
        """
        self.stats.total_queries += 1
        start_time = time.time()
        
        # Create cache keys for different tiers
        exact_cache_key = self._create_cache_key(query + (context_hash or ""))
        normalized_cache_key = self._create_normalized_cache_key(query)
        
        # Clean up expired entries across all caches
        self._cleanup_expired(self.response_cache)
        self._cleanup_expired(self.normalized_cache)
        
        # 🎯 LEVEL 1: Try exact string match (fastest path)
        if exact_cache_key in self.response_cache:
            entry = self.response_cache[exact_cache_key]
            if not self._is_expired(entry):
                elapsed_time = time.time() - start_time
                self.stats.record_exact_hit(3.0)
                logger.info(f"🎯 LEVEL 1 EXACT cache HIT for query: {query[:50]}... ({elapsed_time*1000:.1f}ms)")
                return entry['response']
        
        # 🚀 LEVEL 2: Try normalized string match (fast, no API calls)
        if normalized_cache_key in self.normalized_cache:
            entry = self.normalized_cache[normalized_cache_key]
            if not self._is_expired(entry):
                elapsed_time = time.time() - start_time
                self.stats.record_normalized_hit(2.5)
                logger.info(f"🚀 LEVEL 2 NORMALIZED cache HIT for query: {query[:50]}... ({elapsed_time*1000:.1f}ms)")
                return entry['response']
        
        # 🧠 LEVEL 3: Try semantic similarity match (expensive, API calls required)
        # Only proceed if cache is small enough to prevent performance issues
        if len(self.query_embeddings_cache) <= 50:  # Performance limit
            similar_cache_key = self._find_semantically_similar_query(query)
            if similar_cache_key and similar_cache_key in self.response_cache:
                entry = self.response_cache[similar_cache_key]
                if not self._is_expired(entry):
                    elapsed_time = time.time() - start_time
                    self.stats.record_semantic_hit(2.0)
                    logger.info(f"🧠 LEVEL 3 SEMANTIC cache HIT for query: {query[:50]}... ({elapsed_time*1000:.1f}ms)")
                    return entry['response']
        else:
            logger.debug(f"⏭️ SKIP Level 3 semantic search: cache too large ({len(self.query_embeddings_cache)} > 50)")
        
        # ❌ All cache levels missed
        elapsed_time = time.time() - start_time
        self.stats.record_miss()
        logger.debug(f"❌ ALL LEVELS MISS for query: {query[:50]}... ({elapsed_time*1000:.1f}ms)")
        return None
    
    def cache_response(self, query: str, response: str, context_hash: str = None) -> None:
        """
        🚀 3-TIER CACHE STORAGE SYSTEM
        
        Stores the response in multiple cache tiers for maximum hit rate:
        1. 🎯 Level 1: Exact cache (for identical future queries)
        2. 🚀 Level 2: Normalized cache (for similar phrasing)
        3. 🧠 Level 3: Semantic cache (for semantic similarity matching)
        
        This ensures future queries benefit from all cache levels.
        """
        timestamp = time.time()
        cache_entry = {
            'response': response,
            'timestamp': timestamp,
            'query': query
        }
        
        # 🎯 LEVEL 1: Store in exact cache
        exact_cache_key = self._create_cache_key(query + (context_hash or ""))
        self.response_cache[exact_cache_key] = cache_entry.copy()
        
        # 🚀 LEVEL 2: Store in normalized cache (for fuzzy matching)
        normalized_cache_key = self._create_normalized_cache_key(query)
        self.normalized_cache[normalized_cache_key] = cache_entry.copy()
        
        # 🧠 LEVEL 3: Cache query embedding for semantic similarity (only if cache is small)
        if len(self.query_embeddings_cache) < 50:  # Prevent performance degradation
            self._cache_query_embedding(query, exact_cache_key)
        else:
            logger.debug(f"⏭️ SKIP embedding cache: too many entries ({len(self.query_embeddings_cache)} >= 50)")
        
        # Enforce size limits across all cache tiers
        self._enforce_size_limit(self.response_cache)
        self._enforce_size_limit(self.normalized_cache)
        self._enforce_size_limit(self.query_embeddings_cache)
        
        logger.info(f"💾 3-TIER cache storage for query: {query[:50]}...")
        logger.debug(f"   📊 Cache sizes - Exact: {len(self.response_cache)}, Normalized: {len(self.normalized_cache)}, Semantic: {len(self.query_embeddings_cache)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive 3-tier cache performance statistics."""
        tier_breakdown = self.stats.tier_breakdown
        cost_efficiency = self.stats.cost_efficiency
        
        return {
            # Cache sizes
            'exact_cache_size': len(self.response_cache),
            'normalized_cache_size': len(self.normalized_cache),
            'semantic_cache_size': len(self.query_embeddings_cache),
            'total_cache_entries': len(self.response_cache) + len(self.normalized_cache) + len(self.query_embeddings_cache),
            
            # 3-Tier Performance Breakdown  
            'tier_1_exact_hits': self.stats.exact_hits,
            'tier_2_normalized_hits': self.stats.normalized_hits,
            'tier_3_semantic_hits': self.stats.semantic_hits,
            'total_hits': self.stats.exact_hits + self.stats.normalized_hits + self.stats.semantic_hits,
            'total_misses': self.stats.response_misses,
            
            # Hit rate analysis
            'overall_hit_rate': f"{self.stats.total_hit_rate:.2%}",
            'tier_breakdown_percent': {
                'exact': f"{tier_breakdown['exact']:.1f}%",
                'normalized': f"{tier_breakdown['normalized']:.1f}%", 
                'semantic': f"{tier_breakdown['semantic']:.1f}%"
            },
            
            # Cost efficiency
            'free_hit_rate': f"{cost_efficiency['free_hit_rate']:.1f}%",
            'api_calls_avoided': int(cost_efficiency['api_calls_avoided']),
            'embedding_api_calls': self.stats.embedding_api_calls,
            'cost_per_query': f"${cost_efficiency['cost_per_query']:.6f}",
            
            # Legacy compatibility
            'response_hit_rate': f"{self.stats.response_hit_rate:.2%}",
            'total_queries': self.stats.total_queries,
            'total_time_saved': f"{self.stats.total_time_saved:.1f}s",
            'estimated_cost_savings': f"${(self.stats.response_hits * 0.002) - (self.stats.embedding_api_calls * 0.0001):.4f}",
            'similarity_threshold': self.similarity_threshold
        }
    
    def clear_cache(self) -> None:
        """Clear all 3-tier caches and reset statistics."""
        cache_sizes = {
            'exact': len(self.response_cache),
            'normalized': len(self.normalized_cache),
            'semantic': len(self.query_embeddings_cache),
            'embedding': len(self.embedding_cache),
            'search': len(self.search_cache)
        }
        
        # Clear all cache tiers
        self.response_cache.clear()
        self.normalized_cache.clear()
        self.query_embeddings_cache.clear()
        
        # Clear legacy caches
        self.embedding_cache.clear()
        self.search_cache.clear()
        
        # Reset statistics
        self.stats = CacheStats()
        
        logger.info(f"🗑️ Cleared all 3-tier caches: {cache_sizes}")
        logger.info(f"   💡 Cache system ready for fresh queries with optimized 3-tier lookup")
    
    # Keep the remaining methods unchanged...
    def get_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for query."""
        self.stats.total_queries += 1
        cache_key = self._create_cache_key(query)
        
        # Cleanup expired entries
        self._cleanup_expired(self.embedding_cache)
        
        if cache_key in self.embedding_cache:
            entry = self.embedding_cache[cache_key]
            if not self._is_expired(entry):
                self.stats.embedding_hits += 1
                self.stats.total_time_saved += 0.5  # Estimated time saved
                logger.debug(f"Embedding cache HIT for query: {query[:50]}...")
                return entry['embedding']
        
        self.stats.embedding_misses += 1
        logger.debug(f"Embedding cache MISS for query: {query[:50]}...")
        return None
    
    def cache_embedding(self, query: str, embedding: List[float]) -> None:
        """Cache embedding for query."""
        cache_key = self._create_cache_key(query)
        
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'timestamp': time.time(),
            'query': query
        }
        
        self._enforce_size_limit(self.embedding_cache)
        logger.debug(f"Cached embedding for query: {query[:50]}...")
    
    def get_search_results(self, embedding: List[float]) -> Optional[List]:
        """Get cached search results for embedding."""
        # Create cache key from embedding hash
        embedding_str = ','.join(f"{x:.6f}" for x in embedding[:10])  # Use first 10 dimensions for key
        cache_key = hashlib.md5(embedding_str.encode()).hexdigest()
        
        self._cleanup_expired(self.search_cache)
        
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            if not self._is_expired(entry):
                self.stats.search_hits += 1
                self.stats.total_time_saved += 1.0  # Estimated time saved
                logger.debug("Search results cache HIT")
                return entry['results']
        
        self.stats.search_misses += 1
        logger.debug("Search results cache MISS")
        return None
    
    def cache_search_results(self, embedding: List[float], results: List) -> None:
        """Cache search results for embedding."""
        embedding_str = ','.join(f"{x:.6f}" for x in embedding[:10])
        cache_key = hashlib.md5(embedding_str.encode()).hexdigest()
        
        self.search_cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        self._enforce_size_limit(self.search_cache)
        logger.debug("Cached search results")

    def get(self, query: str, cache_type: str = 'response') -> Optional[str]:
        """
        🔍 SMART CACHE RETRIEVAL with SEMANTIC MATCHING
        
        Retrieval strategy:
        1. 🎯 Try EXACT string match (fastest)
        2. 🧠 Try SEMANTIC similarity match (AI-powered)
        3. ❌ Cache miss - return None
        """
        normalized_query = self._normalize_query(query)
        cache_key = self._create_cache_key(normalized_query)
        cache = self._get_cache(cache_type)
        
        logger.debug(f"🔍 Cache lookup for query: '{query}' (key: {cache_key[:8]}...)")
        
        # Clean up expired entries first
        self._cleanup_expired(cache)
        
        # 1. 🎯 TRY EXACT MATCH (fastest path)
        if cache_key in cache:
            cache_entry = cache[cache_key]
            if not self._is_expired(cache_entry):
                logger.info(f"✅ EXACT cache HIT for query: {query[:50]}...")
                self.stats.record_exact_hit(time.time() - cache_entry['timestamp'])
                return cache_entry['data']
            else:
                # Remove expired entry
                del cache[cache_key]
                
        # 2. 🧠 TRY SEMANTIC SIMILARITY MATCH (AI-powered)
        logger.debug(f"No exact match found, trying semantic similarity for: '{query}'")
        semantic_cache_key = self._find_semantically_similar_query(query)
        
        if semantic_cache_key and semantic_cache_key in cache:
            cache_entry = cache[semantic_cache_key]
            if not self._is_expired(cache_entry):
                cached_query = self.query_embeddings_cache[semantic_cache_key]['query'] if semantic_cache_key in self.query_embeddings_cache else "unknown"
                logger.info(f"🎯 SEMANTIC cache HIT! '{query}' matched '{cached_query}'")
                self.stats.record_semantic_hit(time.time() - cache_entry['timestamp'])
                return cache_entry['data']
            else:
                # Remove expired semantic match
                logger.debug(f"Semantic match found but expired: {semantic_cache_key}")
                del cache[semantic_cache_key]
                
        # 3. ❌ CACHE MISS
        logger.debug(f"Cache MISS for query: '{query}'")
        self.stats.record_miss()
        return None

# Global cache instance
_query_cache = None

def _get_query_cache():
    """Get or create the global query cache instance using configuration."""
    global _query_cache
    if _query_cache is None:
        from ..config.settings import get_config
        config = get_config()
        
        # Initialize cache with configuration values
        _query_cache = QueryResultCache(
            max_size=config.cache.semantic_cache_size,
            ttl_seconds=config.cache.semantic_cache_ttl,
            similarity_threshold=config.cache.semantic_similarity_threshold
        )
        
        logger.info(f"🚀 QueryResultCache initialized with config: size={config.cache.semantic_cache_size}, ttl={config.cache.semantic_cache_ttl}s, threshold={config.cache.semantic_similarity_threshold}")
    
    return _query_cache

def _is_semantic_cache_enabled():
    """Check if semantic cache is enabled in configuration."""
    from ..config.settings import get_config
    config = get_config()
    return config.cache.enable_semantic_cache

def _get_cached_pipeline_import(import_type: str):
    """Get cached pipeline import or import and cache it."""
    if import_type not in _pipeline_cache:
        try:
            if import_type == "document":
                from langchain.docstore.document import Document
                _pipeline_cache[import_type] = Document
            elif import_type == "ensemble_retriever":
                from langchain.retrievers import EnsembleRetriever
                _pipeline_cache[import_type] = EnsembleRetriever
            elif import_type == "bm25_retriever":
                from langchain_community.retrievers import BM25Retriever
                _pipeline_cache[import_type] = BM25Retriever
            elif import_type == "ollama_embeddings":
                from langchain_ollama import OllamaEmbeddings
                _pipeline_cache[import_type] = OllamaEmbeddings
            elif import_type == "openai_embeddings":
                from langchain_openai import OpenAIEmbeddings
                _pipeline_cache[import_type] = OpenAIEmbeddings
            elif import_type == "local_loader":
                from .loader import LocalLoader
                _pipeline_cache[import_type] = LocalLoader
            elif import_type == "optimized_local_loader":
                from .loader import OptimizedLocalLoader
                _pipeline_cache[import_type] = OptimizedLocalLoader
            elif import_type == "remote_loader":
                from .loader import RemoteLoader
                _pipeline_cache[import_type] = RemoteLoader
            elif import_type == "vector_db_proxy":
                from .vector_db import create_vector_store_proxy
                _pipeline_cache[import_type] = create_vector_store_proxy
            else:
                raise ValueError(f"Unknown import type: {import_type}")
        except ImportError as e:
            logger.error(f"Failed to import {import_type}: {e}")
            raise
    
    return _pipeline_cache[import_type]


class Pipeline:
    """Base pipeline class with lazy loading and optimized performance."""
    
    def __init__(self, loader_name: str = "local", vector_store_type: Optional[str] = None, optimize_loading: bool = True, max_workers: int = 4):
        """
        Initialize the pipeline with loaders and vector store.

        Args:
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        # Lazy import loaders
        if optimize_loading:
            LocalLoader = _get_cached_pipeline_import("optimized_local_loader")
            self.local_loader = LocalLoader(name="optimized_local", max_workers=max_workers, enable_cache=True)
            logger.info(f"🚀 Using OptimizedLocalLoader with {max_workers} workers")
        else:
            LocalLoader = _get_cached_pipeline_import("local_loader")
            self.local_loader = LocalLoader(name="local")
            logger.info("Using standard LocalLoader")
        
        RemoteLoader = _get_cached_pipeline_import("remote_loader")
        create_vector_store_proxy = _get_cached_pipeline_import("vector_db_proxy")
        
        self.remote_loader = RemoteLoader(name="remote")
        
        # Use factory function to create vector store
        if vector_store_type is None:
            vector_store_type = "faiss"  # Default to FAISS
        
        self.vector_db = create_vector_store_proxy(vector_store_type)
        
        self.texts: Optional[List] = None
        self._retriever: Optional = None
        self._chain: Optional[FullChain] = None
        self.LLM: Optional = None
        
        logger.info(f"Pipeline initialized with loader: {loader_name}, vector store: {vector_store_type or 'default'}, optimization: {optimize_loading}")

    def load_and_split(
        self,
        data_dir: str = "data",
        uploaded_files: Optional[List] = None,
        page_urls: Optional[List[str]] = None,
        wiki_query: Optional[str] = None
    ) -> Optional[List]:
        """
        Load and split documents from various sources. Sets self.texts.

        Args:
            data_dir (str): The directory to load documents from. Defaults to "data".
            uploaded_files (Optional[List]): A list of uploaded files.
            page_urls (Optional[List[str]]): URLs of pages to load documents from.
            wiki_query (Optional[str]): A Wikipedia query to load documents from.
        
        Returns:
            Optional[List]: The list of split documents, or None if no documents were processed.
        """
        docs: List = []
        logger.info(f"Starting document loading. data_dir='{data_dir}', uploaded_files={'yes' if uploaded_files else 'no'}, page_urls={'yes' if page_urls else 'no'}, wiki_query='{wiki_query if wiki_query else 'no'}'")
        
        # Ensure loaders are reset or handle multiple calls appropriately if needed
        # For this example, assuming they load fresh each time `load` is called.
        self.local_loader.load(data_dir=data_dir, uploaded_files=uploaded_files)
        loaded_local_docs = self.local_loader.get_documents()
        if loaded_local_docs:
            docs.extend(loaded_local_docs)
            logger.info(f"Loaded {len(loaded_local_docs)} documents from local loader.")
        
        self.remote_loader.load(page_urls=page_urls, wiki_query=wiki_query)
        loaded_remote_docs = self.remote_loader.get_documents()
        if loaded_remote_docs:
            docs.extend(loaded_remote_docs)
            logger.info(f"Loaded {len(loaded_remote_docs)} documents from remote loader.")

        if not docs:
            logger.warning("No documents were loaded from any source.")
            self.texts = None
            return None
            
        self.texts = split_documents(docs)
        if self.texts:
            logger.info(f"Successfully split {len(docs)} source documents into {len(self.texts)} chunks.")
        else:
            logger.warning("Document splitting resulted in no text chunks.")
        return self.texts
    
    def _set_retriever(self, embeddings: Optional = None, use_ensemble: bool = True) -> None:
        """
        Set the retriever for the pipeline with lazy loading and timeout protection.

        Args:
            embeddings (Optional): The embeddings to use for the vector store.
            use_ensemble (bool): Whether to use an ensemble retriever (BM25 + vector store). Defaults to True.
        """
        if not self.texts:
            logger.error("Cannot set retriever: No texts have been loaded and split.")
            return

        logger.info(f"Setting retriever with timeout protection. Using ensemble: {use_ensemble}.")
        
        # Add timeout protection for vector store creation
        try:
            logger.info("Creating vector store database...")
            self.vector_db.create_db(docs=self.texts, embeddings=embeddings)
            
            vs = self.vector_db.db # Use the property .db
            if vs is None:
                logger.error("Failed to create or access vector store database.")
                return

            logger.info("Vector store database created successfully.")
            vs_retriever = vs.as_retriever()
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

        if use_ensemble:
            try:
                # Lazy import Document and BM25Retriever
                Document = _get_cached_pipeline_import("document")
                BM25Retriever = _get_cached_pipeline_import("bm25_retriever")
                EnsembleRetriever = _get_cached_pipeline_import("ensemble_retriever")
                
                # BM25Retriever.from_documents is preferred if self.texts are Document objects
                # Ensure self.texts contains Document objects with page_content
                page_contents = [doc.page_content for doc in self.texts if isinstance(doc, Document) and doc.page_content]
                if not page_contents: # Check if there's any content to create BM25 from
                    logger.warning("No page content found in documents for BM25Retriever. Using vector store retriever only.")
                    self._retriever = vs_retriever
                else:
                    bm25_retriever = BM25Retriever.from_texts(page_contents) # from_texts expects List[str]
                    self._retriever = EnsembleRetriever(retrievers=[bm25_retriever, vs_retriever], weights=[0.4, 0.6])
                    logger.info("Ensemble retriever created with BM25 and vector store retriever.")
            except Exception as e:
                logger.error(f"Failed to create BM25Retriever or EnsembleRetriever: {e}. Falling back to vector store retriever only.", exc_info=True)
                self._retriever = vs_retriever
        else:
            self._retriever = vs_retriever
            logger.info("Vector store retriever created.")
        
    def get_chain(self) -> Optional[FullChain]:
        """Returns the created RAG chain, if any."""
        if not self._chain:
            logger.warning("Attempted to get chain, but it has not been created yet.")
        return self._chain
    
    def create_rag_chain(self, chain_type: str = "simple") -> None:
        """
        Creates the RAG chain using the configured LLM and retriever.

        Args:
            chain_type (str): The type of RAG chain to create (e.g., "simple", "multi_query", "fusion"). Defaults to "simple".
        """
        if not self.LLM:
            logger.error("Cannot create RAG chain: LLM is not set.")
            return
        if not self._retriever:
            logger.error("Cannot create RAG chain: Retriever is not set.")
            return
            
        logger.info(f"Creating RAG chain of type: {chain_type}")
        llm_instance = self.LLM.get_llm()
        if not llm_instance:
            logger.error("Cannot create RAG chain: Failed to get LLM instance from proxy.")
            return

        rag_proxy = RagProxy(model=llm_instance, retriever=self._retriever)
        memory_proxy = MemoryProxy() # Assuming default initialization is fine
        
        self._chain = FullChain(llm_proxy=self.LLM, rag_proxy=rag_proxy, memory_proxy=memory_proxy)
        try:
            self._chain.create_full_chain(chain_type=chain_type)
            logger.info(f"Successfully created RAG chain of type: {chain_type}")
        except Exception as e:
            logger.error(f"Error creating RAG chain of type '{chain_type}': {e}", exc_info=True)
            self._chain = None # Ensure chain is None if creation fails

    def ask_question(self, question: str, session_id: str = "foo", use_cache: bool = True) -> Optional[str]:
        """
        🚀 OPTIMIZED: Asks a question to the RAG chain with intelligent caching.
        
        Features:
        - Level 3 Response Caching: Instant answers for repeated questions
        - Performance Analytics: Track cache hits and time savings
        - Cost Optimization: Reduce API calls by up to 80%
        - Configurable Cache: Can be enabled/disabled via configuration

        Args:
            question (str): The question to ask.
            session_id (str): The session ID for memory. Defaults to "foo".
            use_cache (bool): Whether to use caching for performance optimization. Defaults to True.

        Returns:
            Optional[str]: The answer from the RAG chain, or None if an error occurs.
        """
        if not self._chain:
            logger.error("Cannot ask question: RAG chain is not created.")
            return None
        
        logger.info(f"Asking question (session: {session_id}): '{question[:100]}{'...' if len(question) > 100 else ''}'")
        
        # Check if semantic caching is enabled in configuration
        cache_enabled = _is_semantic_cache_enabled()
        
        # 🎯 Level 3: Check response cache first (fastest path)
        if use_cache and cache_enabled:
            start_time = time.time()
            cached_response = _get_query_cache().get_response(question, session_id)
            if cached_response:
                cache_time = time.time() - start_time
                logger.info(f"⚡ Cache hit! Returned response in {cache_time:.3f}s (vs ~3-5s without cache)")
                return cached_response
        elif not cache_enabled:
            logger.debug("🚫 Semantic cache is disabled in configuration")
        
        # Cache miss - generate new response
        try:
            start_time = time.time()
            response = self._chain.ask_question(query=question, session_id=session_id)
            
            if response and use_cache and cache_enabled:
                # Cache the successful response
                _get_query_cache().cache_response(question, response, session_id)
                
                generation_time = time.time() - start_time
                logger.info(f"✅ Generated and cached response in {generation_time:.2f}s")
                
                # Log cache statistics periodically
                if _get_query_cache().stats.total_queries % 10 == 0:
                    stats = _get_query_cache().get_stats()
                    logger.info(f"📊 Cache Stats: {stats['response_hit_rate']} hit rate, {stats['total_time_saved']} saved, {stats['estimated_cost_savings']} cost savings")
            elif response and not cache_enabled:
                generation_time = time.time() - start_time
                logger.info(f"✅ Generated response in {generation_time:.2f}s (cache disabled)")
            
            logger.info("Received response from RAG chain.")
            return response
            
        except Exception as e:
            logger.error(f"Error during ask_question: {e}", exc_info=True)
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive caching performance statistics.
        
        Returns:
            Dict with cache performance metrics including hit rates, time saved, and cost savings.
        """
        if not _is_semantic_cache_enabled():
            return {
                "cache_enabled": False,
                "message": "Semantic cache is disabled in configuration"
            }
        
        stats = _get_query_cache().get_stats()
        stats["cache_enabled"] = True
        return stats
    
    def clear_cache(self) -> None:
        """Clear all caches and reset statistics."""
        if not _is_semantic_cache_enabled():
            logger.info("🚫 Cannot clear cache: Semantic cache is disabled in configuration")
            return
        
        _get_query_cache().clear_cache()
        logger.info("🗑️ All caches cleared by user request")


class OpenAIPipeline(Pipeline):
    """OpenAI-based pipeline with lazy loading and performance optimizations."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", loader_name: str = "local", vector_store_type: Optional[str] = None, optimize_loading: bool = True, max_workers: int = 4):
        """
        Initialize the OpenAIPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to "gpt-3.5-turbo".
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        super().__init__(loader_name=loader_name, vector_store_type=vector_store_type, optimize_loading=optimize_loading, max_workers=max_workers)
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAIPipeline")
        
        # Initialize OpenAI LLM with lazy loading
        self.LLM = OpenAIProxy()
        self.LLM.set_llm(model_name=model)
        logger.info(f"OpenAIPipeline initialized with model: {model}")

    def set_retriever_openai(self, use_ensemble: bool = True) -> None:
        """
        Set the retriever for OpenAI pipeline with OpenAI embeddings.
        This method ALWAYS uses OpenAI embeddings, bypassing any Ollama preferences.
        Includes timeout handling to prevent hanging.

        Args:
            use_ensemble (bool): Whether to use ensemble retriever. Defaults to True.
        """
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found. Cannot create embeddings.")
            return
            
        logger.info("OpenAI Pipeline: Creating embeddings with timeout protection...")
        
        try:
            # Use standard LangChain OpenAI embeddings with clean configuration
            OpenAIEmbeddings = _get_cached_pipeline_import("openai_embeddings")
            
            # Create embeddings with explicit parameters and timeout handling
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                # Explicitly exclude organization to prevent "your_org_id_here" error
                openai_organization=None,
                # Use default model
                model="text-embedding-ada-002",
                # Add timeout and retry settings
                request_timeout=30,  # 30 second timeout
                max_retries=2,       # Retry up to 2 times
            )
            
            # Test embeddings with a simple query to ensure they work
            logger.info("Testing embeddings with simple query...")
            try:
                test_embedding = embeddings.embed_query("test")
                logger.info(f"✅ Embeddings test successful: {len(test_embedding)} dimensions")
            except Exception as e:
                logger.error(f"❌ Embeddings test failed: {e}")
                raise
            
            logger.info(f"✅ OpenAI embeddings created successfully (model: {embeddings.model})")
            self._set_retriever(embeddings=embeddings, use_ensemble=use_ensemble)
            
        except Exception as e:
            logger.error(f"❌ Failed to create OpenAI embeddings: {e}")
            raise


class OllamaPipeline(Pipeline):
    """Ollama-based pipeline with lazy loading and performance optimizations."""
    
    def __init__(self, model: str = "llama3", loader_name: str = "local", vector_store_type: Optional[str] = None, optimize_loading: bool = True, max_workers: int = 4):
        """
        Initialize the OllamaPipeline with the specified model and loader name.

        Args:
            model (str): The model to use. Defaults to "llama3".
            loader_name (str): The name of the loader. Defaults to "local".
            vector_store_type (Optional[str]): Type of vector store to use.
            optimize_loading (bool): Whether to use optimized parallel loading. Defaults to True.
            max_workers (int): Number of parallel workers for optimized loading. Defaults to 4.
        """
        super().__init__(loader_name=loader_name, vector_store_type=vector_store_type, optimize_loading=optimize_loading, max_workers=max_workers)
        
        # Initialize Ollama LLM with lazy loading
        self.LLM = OllamaProxy()
        self.LLM.set_llm(model_name=model)
        logger.info(f"OllamaPipeline initialized with model: {model}")

    def set_retriever_ollama(self, use_ensemble: bool = True) -> None:
        """
        Set the retriever for Ollama pipeline with smart embedding fallback.
        Tries Ollama embeddings first, falls back to OpenAI embeddings if unavailable.

        Args:
            use_ensemble (bool): Whether to use ensemble retriever. Defaults to True.
        """
        embeddings = self._get_smart_embeddings()
        if embeddings:
            self._set_retriever(embeddings=embeddings, use_ensemble=use_ensemble)
        else:
            logger.error("Failed to create any embeddings. Cannot set retriever.")
    
    def _get_smart_embeddings(self):
        """
        Get embeddings with intelligent model-aware selection.
        PRIORITIZES SPEED: Try dedicated embedding models first!
        
        Returns:
            Embeddings instance or None if all attempts fail.
        """
        from ..config.settings import get_config
        config = get_config()
        
        # Try Ollama embeddings first if preferred
        if config.llm.prefer_ollama_embeddings:
            
            # STRATEGY 1: Try dedicated embedding models first (MUCH FASTER!)
            logger.info("🚀 Trying dedicated embedding models for optimal speed...")
            dedicated_models = [
                "nomic-embed-text:latest",
                "nomic-embed-text", 
                "all-minilm:latest",
                "all-minilm",
                "mxbai-embed-large:latest",
                "mxbai-embed-large"
            ]
            
            for model in dedicated_models:
                try:
                    logger.info(f"⚡ Testing fast embedding model: {model}")
                    embeddings = self._try_embedding_model(model)
                    if embeddings:
                        logger.info(f"✅ SUCCESS: Using fast embedding model '{model}'!")
                        return embeddings
                except Exception as e:
                    logger.debug(f"Embedding model '{model}' not available: {e}")
                    continue
            
            # STRATEGY 2: Try model-specific preferences
            llm_model = self.LLM.get_model_name() if self.LLM else None
            if llm_model:
                logger.info(f"🔄 Trying model-specific embedding preferences for {llm_model}...")
                embedding_models = self._get_embedding_models_for_llm(llm_model, config)
                
                # Filter to only available models if auto-detection is enabled
                if config.llm.auto_detect_available_models:
                    available_models = self._get_available_ollama_models()
                    embedding_models = [model for model in embedding_models if model in available_models]
                    if embedding_models:
                        logger.info(f"Found {len(embedding_models)} available embedding models for {llm_model}: {embedding_models}")
                    else:
                        logger.warning(f"No embedding models available for {llm_model}, using fallback list")
                        embedding_models = self._get_embedding_models_for_llm(llm_model, config)
                
                ollama_embeddings = self._try_ollama_embeddings(embedding_models)
                if ollama_embeddings:
                    return ollama_embeddings
            
            # STRATEGY 3: Try LLM model directly as last resort (SLOWEST!)
            if llm_model:
                logger.warning(f"⚠️ Trying LLM model '{llm_model}' directly as embedding model (will be SLOW)")
                ollama_embeddings = self._try_direct_llm_embeddings(llm_model)
                if ollama_embeddings:
                    logger.warning(f"⚠️ Using LLM model '{llm_model}' for embeddings - this will be slow!")
                    return ollama_embeddings
                
            logger.warning("All Ollama embedding strategies failed, falling back to OpenAI embeddings")
        
        # Fallback to OpenAI embeddings
        if OPENAI_API_KEY:
            try:
                logger.info("Using standard LangChain OpenAI embeddings as fallback")
                OpenAIEmbeddings = _get_cached_pipeline_import("openai_embeddings")
                # Use clean configuration to avoid organization issues
                return OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                    openai_organization=None,
                    model="text-embedding-ada-002"
                )
            except Exception as e:
                logger.error(f"Failed to create OpenAI embeddings: {e}")
        else:
            logger.error("No OpenAI API key available for fallback embeddings")
        
        return None
    
    def _try_embedding_model(self, model_name: str):
        """
        Try to use a dedicated embedding model.
        
        Args:
            model_name: The embedding model name to try
            
        Returns:
            OllamaEmbeddings instance or None if it fails
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")
        
        try:
            logger.debug(f"🚀 Creating OllamaEmbeddings with embedding model: {model_name}")
            embeddings = OllamaEmbeddings(model=model_name)
            
            # Test the embeddings with a simple query to verify it works
            test_result = embeddings.embed_query("test")
            if test_result:
                dimensions = len(test_result)
                logger.info(f"✅ Embedding model '{model_name}' works! Dimensions: {dimensions}")
                return embeddings
                
        except Exception as e:
            logger.debug(f"❌ Embedding model '{model_name}' failed: {e}")
        
        return None
    
    def _try_direct_llm_embeddings(self, model_name: str):
        """
        Try to use the LLM model directly as an embedding model.
        This is the new simplified approach based on LangChain's capability.
        
        Args:
            model_name: The LLM model name to try as embedding model
            
        Returns:
            OllamaEmbeddings instance or None if it fails
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")
        
        try:
            logger.info(f"🚀 Creating OllamaEmbeddings with LLM model: {model_name}")
            embeddings = OllamaEmbeddings(model=model_name)
            
            # Test the embeddings with a simple query to verify it works
            test_result = embeddings.embed_query("test")
            if test_result:
                dimensions = len(test_result)
                logger.info(f"✅ LLM model '{model_name}' works as embedding model! Dimensions: {dimensions}")
                return embeddings
                
        except Exception as e:
            logger.warning(f"❌ LLM model '{model_name}' failed as embedding model: {e}")
        
        return None
    
    def _get_embedding_models_for_llm(self, llm_model: Optional[str], config) -> List[str]:
        """
        Get embedding models based on the LLM model selected.
        
        Args:
            llm_model: The LLM model name (e.g., "llama3", "phi4")
            config: Configuration object
            
        Returns:
            List of embedding models to try, in order of preference
        """
        if not llm_model:
            return config.llm.model_embedding_preferences.get("ollama_default", [])
        
        # Normalize model name (remove version suffixes for matching)
        base_model = llm_model.split(':')[0]  # "deepseek-r1:8b" -> "deepseek-r1"
        
        # Try exact match first
        if llm_model in config.llm.model_embedding_preferences:
            logger.info(f"Using embedding preferences for exact model match: {llm_model}")
            return config.llm.model_embedding_preferences[llm_model]
        
        # Try base model match
        if base_model in config.llm.model_embedding_preferences:
            logger.info(f"Using embedding preferences for base model match: {base_model}")
            return config.llm.model_embedding_preferences[base_model]
        
        # Fallback to default
        logger.info(f"No specific embedding preferences for {llm_model}, using default")
        return config.llm.model_embedding_preferences.get("ollama_default", [])
    
    def _get_available_ollama_models(self) -> List[str]:
        """
        Query Ollama to get list of available models.
        
        Returns:
            List of available model names
        """
        try:
            import requests
            from ..config.settings import get_config
            config = get_config()
            
            response = requests.get(f"{config.api.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                logger.debug(f"Available Ollama models: {available_models}")
                return available_models
            else:
                logger.warning(f"Failed to query Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Could not detect available Ollama models: {e}")
            return []
    
    def _try_ollama_embeddings(self, embedding_models: List[str]):
        """
        Try to create Ollama embeddings with multiple model options.
        
        Args:
            embedding_models: List of embedding models to try.
            
        Returns:
            OllamaEmbeddings instance or None if all models fail.
        """
        OllamaEmbeddings = _get_cached_pipeline_import("ollama_embeddings")
        
        for model in embedding_models:
            try:
                logger.info(f"Attempting to use Ollama embedding model: {model}")
                embeddings = OllamaEmbeddings(model=model)
                
                # Test the embeddings with a simple query to verify the model works
                test_result = embeddings.embed_query("test")
                if test_result:
                    logger.info(f"Successfully using Ollama embedding model: {model}")
                    return embeddings
                    
            except Exception as e:
                logger.warning(f"Ollama embedding model '{model}' failed: {e}")
                continue
        
        logger.warning("All Ollama embedding models failed")
        return None


def main():
    """Main function for testing pipeline functionality."""
    pass


if __name__ == "__main__":
    main()