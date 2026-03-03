"""
Tests for Compilation Cache - Phase 6.2.

Tests caching functionality for LLM call optimization.
"""

import pytest
import time
from unittest.mock import Mock, patch

from core.cache import (
    CompilationCache,
    StagedCache,
    CacheStats,
    get_cache,
    reset_cache,
)


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_default_values(self):
        """CacheStats should have zero defaults."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_hit_rate_empty(self):
        """Hit rate should be 0 when no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Hit rate should calculate correctly."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hit_rate_all_hits(self):
        """Hit rate should be 1.0 when all hits."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate == 1.0

    def test_to_dict(self):
        """CacheStats.to_dict() should export correctly."""
        stats = CacheStats(hits=5, misses=5, evictions=2)
        d = stats.to_dict()
        assert d["hits"] == 5
        assert d["misses"] == 5
        assert d["evictions"] == 2
        assert d["hit_rate"] == 0.5


class TestCompilationCache:
    """Test CompilationCache class."""

    def test_init_defaults(self):
        """Cache should initialize with sensible defaults."""
        cache = CompilationCache()
        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600
        assert cache.enabled is True
        assert cache.size() == 0

    def test_init_custom_values(self):
        """Cache should accept custom initialization values."""
        cache = CompilationCache(max_size=50, ttl_seconds=1800, enabled=False)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 1800
        assert cache.enabled is False

    def test_make_key_deterministic(self):
        """Same input should produce same key."""
        cache = CompilationCache()
        key1 = cache.make_key("Build a user system", {"provider": "grok"})
        key2 = cache.make_key("Build a user system", {"provider": "grok"})
        assert key1 == key2

    def test_make_key_different_text(self):
        """Different text should produce different key."""
        cache = CompilationCache()
        key1 = cache.make_key("Build a user system", {})
        key2 = cache.make_key("Build a product system", {})
        assert key1 != key2

    def test_make_key_different_config(self):
        """Different config should produce different key."""
        cache = CompilationCache()
        key1 = cache.make_key("Build a system", {"provider": "grok"})
        key2 = cache.make_key("Build a system", {"provider": "claude"})
        assert key1 != key2

    def test_make_key_normalizes_whitespace(self):
        """Key should normalize whitespace."""
        cache = CompilationCache()
        key1 = cache.make_key("Build a system", {})
        key2 = cache.make_key("  Build a system  ", {})
        key3 = cache.make_key("BUILD A SYSTEM", {})  # Case insensitive
        assert key1 == key2
        assert key1 == key3

    def test_set_and_get(self):
        """Should store and retrieve values."""
        cache = CompilationCache()
        cache.set("key1", {"data": "value"})
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_get_missing_key(self):
        """Should return None for missing key."""
        cache = CompilationCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_get_increments_hits(self):
        """Successful get should increment hits."""
        cache = CompilationCache()
        cache.set("key1", "value")
        cache.get("key1")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 0

    def test_get_missing_increments_misses(self):
        """Failed get should increment misses."""
        cache = CompilationCache()
        cache.get("missing")
        assert cache.stats.hits == 0
        assert cache.stats.misses == 1

    def test_ttl_expiration(self):
        """Entries should expire after TTL."""
        cache = CompilationCache(ttl_seconds=1)
        cache.set("key1", "value")

        # Should exist initially
        assert cache.get("key1") == "value"

        # Wait for TTL
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Should evict oldest entry when at capacity."""
        cache = CompilationCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Add fourth - should evict key1
        cache.set("key4", "value4")

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
        assert cache.stats.evictions == 1

    def test_lru_access_order(self):
        """Accessing entry should move it to end (most recent)."""
        cache = CompilationCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 - moves it to end
        cache.get("key1")

        # Add key4 - should evict key2 (now oldest)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still exists
        assert cache.get("key2") is None       # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self):
        """Updating existing key should not increase size."""
        cache = CompilationCache(max_size=3)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key1", "updated")  # Update

        assert cache.size() == 2
        assert cache.get("key1") == "updated"

    def test_disabled_cache_returns_none(self):
        """Disabled cache should always return None."""
        cache = CompilationCache(enabled=False)
        cache.set("key1", "value")
        assert cache.get("key1") is None
        assert cache.stats.misses == 1

    def test_disabled_cache_does_not_store(self):
        """Disabled cache should not store values."""
        cache = CompilationCache(enabled=False)
        cache.set("key1", "value")
        assert cache.size() == 0

    def test_invalidate_existing_key(self):
        """Should remove specific entry."""
        cache = CompilationCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        result = cache.invalidate("key1")

        assert result is True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.size() == 1

    def test_invalidate_missing_key(self):
        """Invalidating missing key should return False."""
        cache = CompilationCache()
        result = cache.invalidate("missing")
        assert result is False

    def test_clear(self):
        """Clear should remove all entries."""
        cache = CompilationCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        count = cache.clear()

        assert count == 3
        assert cache.size() == 0

    def test_reset_stats(self):
        """Reset stats should zero all counters."""
        cache = CompilationCache()
        cache.set("key1", "value")
        cache.get("key1")
        cache.get("missing")

        cache.reset_stats()

        assert cache.stats.hits == 0
        assert cache.stats.misses == 0


class TestStagedCache:
    """Test StagedCache multi-stage cache."""

    def test_init_creates_separate_caches(self):
        """Should create separate caches for each stage."""
        cache = StagedCache()

        assert isinstance(cache.intent, CompilationCache)
        assert isinstance(cache.persona, CompilationCache)
        assert isinstance(cache.blueprint, CompilationCache)

    def test_init_custom_sizes(self):
        """Should accept custom sizes for each cache."""
        cache = StagedCache(
            intent_size=50,
            persona_size=25,
            blueprint_size=10
        )

        assert cache.intent.max_size == 50
        assert cache.persona.max_size == 25
        assert cache.blueprint.max_size == 10

    def test_independent_caches(self):
        """Each stage cache should be independent."""
        cache = StagedCache()

        cache.intent.set("key1", "intent_value")
        cache.persona.set("key1", "persona_value")
        cache.blueprint.set("key1", "blueprint_value")

        assert cache.intent.get("key1") == "intent_value"
        assert cache.persona.get("key1") == "persona_value"
        assert cache.blueprint.get("key1") == "blueprint_value"

    def test_enabled_property(self):
        """Should track enabled state."""
        cache = StagedCache(enabled=True)
        assert cache.enabled is True

        cache.enabled = False
        assert cache.intent.enabled is False
        assert cache.persona.enabled is False
        assert cache.blueprint.enabled is False

    def test_clear_all(self):
        """clear_all should clear all caches."""
        cache = StagedCache()

        cache.intent.set("k1", "v1")
        cache.intent.set("k2", "v2")
        cache.persona.set("k1", "v1")
        cache.blueprint.set("k1", "v1")

        result = cache.clear_all()

        assert result["intent"] == 2
        assert result["persona"] == 1
        assert result["blueprint"] == 1
        assert cache.total_size() == 0

    def test_stats(self):
        """stats should aggregate from all caches."""
        cache = StagedCache()

        cache.intent.set("k1", "v1")
        cache.intent.get("k1")
        cache.persona.get("missing")

        stats = cache.stats()

        assert stats["intent"]["hits"] == 1
        assert stats["persona"]["misses"] == 1

    def test_total_size(self):
        """total_size should sum all caches."""
        cache = StagedCache()

        cache.intent.set("k1", "v1")
        cache.intent.set("k2", "v2")
        cache.persona.set("k1", "v1")
        cache.blueprint.set("k1", "v1")

        assert cache.total_size() == 4


class TestGlobalCache:
    """Test global cache functions."""

    def test_get_cache_returns_singleton(self):
        """get_cache should return same instance."""
        reset_cache()  # Start fresh

        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

    def test_reset_cache(self):
        """reset_cache should clear global instance."""
        cache1 = get_cache()
        cache1.intent.set("key", "value")

        reset_cache()

        cache2 = get_cache()
        assert cache2 is not cache1
        assert cache2.intent.get("key") is None


class TestCacheIntegration:
    """Integration tests for cache with engine."""

    def test_cache_with_compile_workflow(self):
        """Cache should work in typical compile workflow."""
        cache = CompilationCache()

        # Simulate Intent caching
        description = "Build a user management system"
        config = {"provider": "grok", "model": "grok-3"}

        key = cache.make_key(description, config)

        # First call - miss
        result = cache.get(key)
        assert result is None

        # Store result
        intent = {"entities": ["User", "Role"], "processes": ["authenticate"]}
        cache.set(key, intent)

        # Second call - hit
        cached = cache.get(key)
        assert cached == intent

        # Stats
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_staged_cache_workflow(self):
        """Staged cache should handle typical workflow."""
        cache = StagedCache()

        description = "Build a product catalog"
        config = {"provider": "grok"}

        # Intent stage
        intent_key = cache.intent.make_key(description, config)
        cache.intent.set(intent_key, {"entities": ["Product", "Category"]})

        # Persona stage (uses intent result)
        import json
        intent_json = json.dumps({"entities": ["Product", "Category"]})
        persona_key = cache.persona.make_key(intent_json, config)
        cache.persona.set(persona_key, [{"name": "DomainExpert"}])

        # Verify both stages cached
        assert cache.intent.get(intent_key) is not None
        assert cache.persona.get(persona_key) is not None
        assert cache.total_size() == 2
