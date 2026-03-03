"""Tests for core/llm.py — RouteTier and tier-based LLM routing."""

import pytest
from unittest.mock import MagicMock, patch

from core.llm import (
    RouteTier,
    BaseLLMClient,
    MockClient,
    FailoverClient,
)


# ---------------------------------------------------------------------------
# RouteTier enum
# ---------------------------------------------------------------------------

class TestRouteTier:
    def test_values(self):
        assert RouteTier.CRITICAL.value == "critical"
        assert RouteTier.STANDARD.value == "standard"
        assert RouteTier.LOCAL.value == "local"

    def test_all_three_tiers(self):
        assert len(RouteTier) == 3


# ---------------------------------------------------------------------------
# MockClient tier compatibility
# ---------------------------------------------------------------------------

class TestMockClientTier:
    def test_complete_accepts_tier(self):
        client = MockClient()
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert "[Mock response" in result

    def test_complete_with_system_accepts_tier(self):
        client = MockClient()
        result = client.complete_with_system(
            "system", "user",
            tier=RouteTier.LOCAL,
        )
        assert "[Mock response" in result

    def test_complete_without_tier(self):
        client = MockClient()
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
        )
        assert "[Mock response" in result


# ---------------------------------------------------------------------------
# FailoverClient — basic tier_map functionality
# ---------------------------------------------------------------------------

class TestFailoverTierMap:
    def _make_mock_provider(self, name, response):
        m = MockClient()
        m._name = name
        # Override complete to track calls and return specific response
        call_log = []
        original_complete = m.complete

        def tracked_complete(*args, **kwargs):
            call_log.append(name)
            return f"[{name}]"

        m.complete = tracked_complete
        m._call_log = call_log
        return m

    def test_no_tier_map_uses_default_order(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        client = FailoverClient([p0, p1])

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p0]"  # first provider used (no tier_map)

    def test_tier_map_reorders_providers(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},  # prefer p1 for critical
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p1]"  # p1 tried first for CRITICAL

    def test_tier_map_falls_back_on_failure(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        # Make p1 fail
        p1.complete = MagicMock(side_effect=Exception("p1 failed"))

        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p0]"  # fell back to p0

    def test_different_tiers_different_providers(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        p2 = self._make_mock_provider("p2", "r2")

        client = FailoverClient(
            [p0, p1, p2],
            tier_map={
                RouteTier.CRITICAL: [2],   # p2 for critical
                RouteTier.LOCAL: [0],       # p0 for local
            },
        )

        critical_result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert critical_result == "[p2]"

        local_result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.LOCAL,
        )
        assert local_result == "[p0]"

    def test_tier_none_uses_default_order(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=None,
        )
        assert result == "[p0]"  # default order, no tier applied

    def test_unmapped_tier_uses_default_order(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.STANDARD,  # not in tier_map
        )
        assert result == "[p0]"

    def test_tier_map_with_invalid_index_ignored(self):
        p0 = self._make_mock_provider("p0", "r0")
        client = FailoverClient(
            [p0],
            tier_map={RouteTier.CRITICAL: [99]},  # out of range
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p0]"  # falls back to p0

    def test_tier_map_multiple_preferred(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")
        p2 = self._make_mock_provider("p2", "r2")
        # Make p1 fail
        p1.complete = MagicMock(side_effect=Exception("p1 failed"))

        client = FailoverClient(
            [p0, p1, p2],
            tier_map={RouteTier.CRITICAL: [1, 2]},  # try p1 then p2
        )

        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tier=RouteTier.CRITICAL,
        )
        assert result == "[p2]"  # p1 failed, p2 succeeded

    def test_complete_with_system_passes_tier(self):
        p0 = self._make_mock_provider("p0", "r0")
        p1 = self._make_mock_provider("p1", "r1")

        client = FailoverClient(
            [p0, p1],
            tier_map={RouteTier.CRITICAL: [1]},
        )

        # complete_with_system calls complete internally
        # but doesn't pass tier through by default — verify it doesn't break
        result = client.complete_with_system("system", "user")
        assert "[p0]" in result or "[Mock" in result


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_failover_without_tier_map(self):
        """FailoverClient with no tier_map works exactly as before."""
        client = FailoverClient([MockClient()])
        result = client.complete(
            messages=[{"role": "user", "content": "test"}],
        )
        assert "[Mock response" in result

    def test_failover_init_accepts_no_tier_map(self):
        """Old-style initialization still works."""
        client = FailoverClient([MockClient()])
        assert client.tier_map is None
