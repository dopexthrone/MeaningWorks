"""
Tests for motherlabs_platform.auth — KeyStore + RateLimiter.

~25 tests covering:
- KeyStore CRUD (create, validate, list, revoke, spend, usage logging)
- RateLimiter sliding window behavior
- Edge cases (revoked keys, budget exceeded, hash collisions)
"""

import os
import tempfile
import time

import pytest

from motherlabs_platform.auth import (
    KeyStore,
    RateLimiter,
    APIKeyRecord,
    KeyValidationResult,
    _hash_key,
)


@pytest.fixture
def db_path(tmp_path):
    """Temporary database path."""
    return str(tmp_path / "test_auth.db")


@pytest.fixture
def store(db_path):
    """Fresh KeyStore instance."""
    return KeyStore(db_path=db_path)


@pytest.fixture
def limiter(store):
    """RateLimiter with a short window for testing."""
    return RateLimiter(store, window_seconds=10)


# =============================================================================
# KEY HASHING
# =============================================================================

class TestHashKey:
    def test_deterministic(self):
        assert _hash_key("abc") == _hash_key("abc")

    def test_different_keys_different_hashes(self):
        assert _hash_key("key1") != _hash_key("key2")

    def test_returns_hex_string(self):
        h = _hash_key("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# KEYSTORE — CREATE
# =============================================================================

class TestKeyStoreCreate:
    def test_create_returns_id_and_key(self, store):
        key_id, raw_key = store.create_key("test-key")
        assert isinstance(key_id, str)
        assert isinstance(raw_key, str)
        assert len(raw_key) == 64  # secrets.token_hex(32)

    def test_create_with_custom_limits(self, store):
        key_id, _ = store.create_key("custom", rate_limit=200, budget=100.0)
        keys = store.list_keys()
        record = [k for k in keys if k.id == key_id][0]
        assert record.rate_limit_per_hour == 200
        assert record.budget_usd == 100.0

    def test_create_multiple_keys(self, store):
        store.create_key("key1")
        store.create_key("key2")
        store.create_key("key3")
        assert len(store.list_keys()) == 3


# =============================================================================
# KEYSTORE — VALIDATE
# =============================================================================

class TestKeyStoreValidate:
    def test_validate_valid_key(self, store):
        _, raw_key = store.create_key("valid-key")
        result = store.validate_key(raw_key)
        assert result.valid is True
        assert result.key_name == "valid-key"
        assert result.is_active is True

    def test_validate_invalid_key(self, store):
        result = store.validate_key("nonexistent-key")
        assert result.valid is False
        assert "Invalid" in result.reason

    def test_validate_revoked_key(self, store):
        key_id, raw_key = store.create_key("revoked-key")
        store.revoke_key(key_id)
        result = store.validate_key(raw_key)
        assert result.valid is False
        assert "revoked" in result.reason

    def test_validate_over_budget(self, store):
        key_id, raw_key = store.create_key("budget-key", budget=10.0)
        store.record_spend(key_id, 10.0)
        result = store.validate_key(raw_key)
        assert result.valid is False
        assert "Budget" in result.reason

    def test_validate_under_budget(self, store):
        key_id, raw_key = store.create_key("budget-key", budget=10.0)
        store.record_spend(key_id, 5.0)
        result = store.validate_key(raw_key)
        assert result.valid is True
        assert result.spent_usd == 5.0


# =============================================================================
# KEYSTORE — LIST
# =============================================================================

class TestKeyStoreList:
    def test_list_empty(self, store):
        assert store.list_keys() == []

    def test_list_returns_records(self, store):
        store.create_key("key-a")
        store.create_key("key-b")
        keys = store.list_keys()
        assert len(keys) == 2
        assert all(isinstance(k, APIKeyRecord) for k in keys)

    def test_list_shows_revoked(self, store):
        key_id, _ = store.create_key("revoked")
        store.revoke_key(key_id)
        keys = store.list_keys()
        assert len(keys) == 1
        assert keys[0].is_active is False


# =============================================================================
# KEYSTORE — REVOKE
# =============================================================================

class TestKeyStoreRevoke:
    def test_revoke_existing_key(self, store):
        key_id, _ = store.create_key("to-revoke")
        assert store.revoke_key(key_id) is True

    def test_revoke_nonexistent_key(self, store):
        assert store.revoke_key("fake-id") is False

    def test_revoke_is_idempotent(self, store):
        key_id, _ = store.create_key("to-revoke")
        store.revoke_key(key_id)
        # Second revoke still returns True (row exists, just already inactive)
        assert store.revoke_key(key_id) is True


# =============================================================================
# KEYSTORE — SPEND
# =============================================================================

class TestKeyStoreSpend:
    def test_record_spend_increments(self, store):
        key_id, raw_key = store.create_key("spend-key", budget=100.0)
        store.record_spend(key_id, 10.0)
        store.record_spend(key_id, 5.0)
        result = store.validate_key(raw_key)
        assert result.spent_usd == 15.0


# =============================================================================
# KEYSTORE — USAGE LOG
# =============================================================================

class TestKeyStoreUsageLog:
    def test_log_usage(self, store):
        key_id, _ = store.create_key("log-key")
        store.log_usage(key_id, domain="software", duration=1.5, cost=0.05)
        # Verify via count
        count = store.get_usage_count_since(key_id, "2000-01-01T00:00:00Z")
        assert count == 1

    def test_log_multiple_usage(self, store):
        key_id, _ = store.create_key("log-key")
        for _ in range(5):
            store.log_usage(key_id, domain="software")
        count = store.get_usage_count_since(key_id, "2000-01-01T00:00:00Z")
        assert count == 5


# =============================================================================
# RATE LIMITER
# =============================================================================

class TestRateLimiter:
    def test_allows_within_limit(self, limiter):
        allowed, remaining, _ = limiter.check_rate_limit("key1", 10)
        assert allowed is True
        assert remaining == 9

    def test_blocks_at_limit(self, limiter):
        for _ in range(5):
            limiter.check_rate_limit("key2", 5)
        allowed, remaining, _ = limiter.check_rate_limit("key2", 5)
        assert allowed is False
        assert remaining == 0

    def test_remaining_decreases(self, limiter):
        _, r1, _ = limiter.check_rate_limit("key3", 10)
        _, r2, _ = limiter.check_rate_limit("key3", 10)
        _, r3, _ = limiter.check_rate_limit("key3", 10)
        assert r1 == 9
        assert r2 == 8
        assert r3 == 7

    def test_different_keys_independent(self, limiter):
        for _ in range(5):
            limiter.check_rate_limit("key-a", 5)
        # key-a is exhausted
        allowed_a, _, _ = limiter.check_rate_limit("key-a", 5)
        assert allowed_a is False
        # key-b should be fine
        allowed_b, _, _ = limiter.check_rate_limit("key-b", 5)
        assert allowed_b is True

    def test_reset_after_window(self, store):
        """With a 1-second window, limits should reset quickly."""
        limiter = RateLimiter(store, window_seconds=1)
        for _ in range(3):
            limiter.check_rate_limit("key-reset", 3)
        allowed, _, _ = limiter.check_rate_limit("key-reset", 3)
        assert allowed is False

        time.sleep(1.1)
        allowed, remaining, _ = limiter.check_rate_limit("key-reset", 3)
        assert allowed is True
        assert remaining == 2

    def test_reset_timestamp_is_future(self, limiter):
        _, _, reset_ts = limiter.check_rate_limit("key-ts", 10)
        assert reset_ts > time.time()
