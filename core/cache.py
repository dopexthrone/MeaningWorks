"""
Motherlabs Compilation Cache - LRU cache for compilation stages.

Phase 6.2: Enterprise Scale & Reliability

Provides caching for expensive LLM calls to reduce latency and costs
for repeated compilations with similar inputs.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export stats as dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class CompilationCache:
    """
    LRU cache for compilation stages.

    Caches Intent extraction and Persona generation results to avoid
    redundant LLM calls for similar inputs.

    Example:
        cache = CompilationCache(max_size=100, ttl_seconds=3600)

        # Check cache
        key = cache.make_key("Build a user system", {"provider": "grok"})
        cached = cache.get(key)

        if cached is None:
            result = expensive_llm_call()
            cache.set(key, result)
        else:
            result = cached
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize compilation cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled (allows disabling for testing)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return self._stats

    def make_key(self, text: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key from input text and configuration.

        Args:
            text: Input text to hash
            config: Optional configuration dict (provider, model, etc.)

        Returns:
            16-character hex hash key
        """
        config = config or {}
        # Normalize text (strip whitespace, lowercase)
        normalized_text = text.strip().lower()
        # Create deterministic string from config
        config_str = json.dumps(config, sort_keys=True)
        # Combine and hash
        content = f"{normalized_text}:{config_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if present and not expired, None otherwise
        """
        if not self.enabled:
            self._stats.misses += 1
            return None

        if key not in self._cache:
            self._stats.misses += 1
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            self._stats.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._stats.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        # If key exists, update it
        if key in self._cache:
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        # Add new entry
        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> bool:
        """
        Remove specific entry from cache.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def size(self) -> int:
        """Return current number of cached entries."""
        return len(self._cache)

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = CacheStats()


class StagedCache:
    """
    Multi-stage cache for different compilation phases.

    Provides separate caches for:
    - Intent extraction
    - Persona generation
    - Full blueprint (optional)

    Example:
        cache = StagedCache()

        # Cache intent
        intent_key = cache.intent.make_key(description, config)
        intent = cache.intent.get(intent_key)

        # Cache personas
        persona_key = cache.persona.make_key(intent_json, config)
        personas = cache.persona.get(persona_key)
    """

    def __init__(
        self,
        intent_size: int = 100,
        persona_size: int = 50,
        blueprint_size: int = 25,
        ttl_seconds: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize staged cache.

        Args:
            intent_size: Max entries for intent cache
            persona_size: Max entries for persona cache
            blueprint_size: Max entries for full blueprint cache
            ttl_seconds: TTL for all caches
            enabled: Whether caching is enabled
        """
        self.intent = CompilationCache(
            max_size=intent_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled
        )
        self.persona = CompilationCache(
            max_size=persona_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled
        )
        self.blueprint = CompilationCache(
            max_size=blueprint_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled
        )

    @property
    def enabled(self) -> bool:
        """Check if any cache is enabled."""
        return self.intent.enabled or self.persona.enabled or self.blueprint.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable all caches."""
        self.intent.enabled = value
        self.persona.enabled = value
        self.blueprint.enabled = value

    def clear_all(self) -> Dict[str, int]:
        """
        Clear all caches.

        Returns:
            Dict with count of entries cleared per cache
        """
        return {
            "intent": self.intent.clear(),
            "persona": self.persona.clear(),
            "blueprint": self.blueprint.clear(),
        }

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.

        Returns:
            Dict with stats per cache
        """
        return {
            "intent": self.intent.stats.to_dict(),
            "persona": self.persona.stats.to_dict(),
            "blueprint": self.blueprint.stats.to_dict(),
        }

    def total_size(self) -> int:
        """Return total entries across all caches."""
        return self.intent.size() + self.persona.size() + self.blueprint.size()


# Global cache instance (can be replaced for testing)
_global_cache: Optional[StagedCache] = None


def get_cache() -> StagedCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = StagedCache()
    return _global_cache


def reset_cache() -> None:
    """Reset global cache instance."""
    global _global_cache
    _global_cache = None
