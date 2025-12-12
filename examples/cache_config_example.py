#!/usr/bin/env python3
"""
Example: Using the Cache Configuration System

This example demonstrates how to:
1. Enable/disable the semantic cache system
2. Configure cache parameters
3. Monitor cache performance
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_rag.config.settings import get_config
from langchain_rag.rag.pipeline import OpenAIPipeline


def example_cache_configuration():
    """Example showing how to configure the cache system."""
    print("🎛️ Cache Configuration Example")
    print("=" * 40)
    
    # Get configuration
    config = get_config()
    
    # Example 1: Enable cache with default settings
    print("\n1️⃣ Enable Cache (Default Settings):")
    config.cache.enable_semantic_cache = True
    print(f"   ✅ Cache enabled: {config.cache.enable_semantic_cache}")
    print(f"   ✅ Cache size: {config.cache.semantic_cache_size}")
    print(f"   ✅ Cache TTL: {config.cache.semantic_cache_ttl} seconds")
    print(f"   ✅ Similarity threshold: {config.cache.semantic_similarity_threshold}")
    
    # Example 2: Disable cache
    print("\n2️⃣ Disable Cache:")
    config.cache.enable_semantic_cache = False
    print(f"   ❌ Cache enabled: {config.cache.enable_semantic_cache}")
    print("   💡 Benefits lost: No fast responses, higher costs")
    
    # Example 3: Enable cache with custom settings
    print("\n3️⃣ Enable Cache with Custom Settings:")
    config.cache.enable_semantic_cache = True
    config.cache.semantic_cache_size = 200          # Larger cache
    config.cache.semantic_cache_ttl = 7200          # 2 hours
    config.cache.semantic_similarity_threshold = 0.90  # More relaxed matching
    
    print(f"   ✅ Cache enabled: {config.cache.enable_semantic_cache}")
    print(f"   ✅ Cache size: {config.cache.semantic_cache_size} entries")
    print(f"   ✅ Cache TTL: {config.cache.semantic_cache_ttl} seconds")
    print(f"   ✅ Similarity threshold: {config.cache.semantic_similarity_threshold}")
    
    # Example 4: Conservative vs Relaxed matching
    print("\n4️⃣ Threshold Comparison:")
    
    thresholds = [
        (0.98, "Ultra Conservative - Only near-identical queries"),
        (0.95, "Conservative - Very similar queries (default)"),
        (0.90, "Moderate - Similar meaning queries"),
        (0.85, "Relaxed - Related topic queries")
    ]
    
    for threshold, description in thresholds:
        config.cache.semantic_similarity_threshold = threshold
        print(f"   🎯 {threshold}: {description}")
    
    # Example 5: Configuration for different use cases
    print("\n5️⃣ Configuration Recommendations:")
    
    use_cases = [
        ("High Accuracy", {"size": 100, "ttl": 3600, "threshold": 0.95}),
        ("High Performance", {"size": 500, "ttl": 7200, "threshold": 0.90}),
        ("Development/Testing", {"size": 50, "ttl": 1800, "threshold": 0.85}),
        ("Production", {"size": 200, "ttl": 3600, "threshold": 0.95})
    ]
    
    for use_case, settings in use_cases:
        print(f"   📋 {use_case}:")
        print(f"      Size: {settings['size']} | TTL: {settings['ttl']}s | Threshold: {settings['threshold']}")
    
    print("\n🎉 Cache Configuration Complete!")
    print("   💡 Adjust settings based on your use case")
    print("   💡 Higher threshold = more conservative matching")
    print("   💡 Lower threshold = more cache hits but potential false positives")


def example_cache_usage():
    """Example showing how to use the cache system in practice."""
    print("\n🚀 Cache Usage Example")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ No OpenAI API key found. Set OPENAI_API_KEY to run this example.")
        return
    
    # Configure cache
    config = get_config()
    config.cache.enable_semantic_cache = True
    config.cache.semantic_cache_size = 50
    config.cache.semantic_similarity_threshold = 0.92
    
    print(f"✅ Cache configured: size={config.cache.semantic_cache_size}, threshold={config.cache.semantic_similarity_threshold}")
    
    try:
        # Create pipeline
        pipeline = OpenAIPipeline(model="gpt-3.5-turbo")
        
        # Check cache status
        stats = pipeline.get_cache_stats()
        print(f"✅ Cache status: {stats.get('cache_enabled', 'Unknown')}")
        
        # Show cache benefits
        print("\n📊 Expected Cache Benefits:")
        print("   ⚡ 25x faster responses for repeated queries")
        print("   💰 Up to 80% cost savings")
        print("   🧠 Semantic similarity matching")
        print("   📈 Improved user experience")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    example_cache_configuration()
    example_cache_usage() 