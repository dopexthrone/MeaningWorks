"""
Tests for mother/delegation.py and mother/peer_handlers.py.

Covers: DelegationResult, DelegationRouter, peer selection,
request handlers (compile_request, build_request, tool_offer).
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from mother.delegation import DelegationResult, DelegationRouter
from mother.wormhole import Wormhole, WormholeMessage, WormholeConnection


class TestDelegationResult:
    def test_frozen(self):
        r = DelegationResult(success=True, peer_id="peer1", task_type="compile")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = DelegationResult(success=True, peer_id="p1", task_type="compile")
        assert r.peer_name == ""
        assert r.result_data == {}
        assert r.cost_usd == 0.0

    def test_result_data_never_none(self):
        r = DelegationResult(success=True, peer_id="p1", task_type="compile", result_data=None)
        assert r.result_data == {}


class TestDelegationRouter:
    def test_choose_peer_no_candidates(self):
        """No peers connected → returns None."""
        wormhole = MagicMock()
        wormhole.connections = {}
        router = DelegationRouter(wormhole)

        peer_id = router.choose_peer_for_task("compile")
        assert peer_id is None

    def test_choose_peer_prefer_local(self):
        """prefer_local=True → returns None even if peers available."""
        wormhole = MagicMock()
        wormhole.connections = {
            "peer1": WormholeConnection(
                peer_id="peer1",
                peer_name="Test",
                capabilities=["compile"],
            ),
        }
        router = DelegationRouter(wormhole)

        peer_id = router.choose_peer_for_task("compile", prefer_local=True)
        assert peer_id is None

    def test_choose_peer_single_candidate(self):
        """One peer with capability → selected."""
        wormhole = MagicMock()
        wormhole.connections = {
            "peer1": WormholeConnection(
                peer_id="peer1",
                peer_name="Ubuntu",
                capabilities=["compile", "build"],
            ),
        }
        router = DelegationRouter(wormhole)

        peer_id = router.choose_peer_for_task("compile")
        assert peer_id == "peer1"

    def test_choose_peer_performance_based(self):
        """Multiple peers → chooses one with better success rate."""
        wormhole = MagicMock()
        wormhole.connections = {
            "peer1": WormholeConnection(peer_id="peer1", capabilities=["compile"]),
            "peer2": WormholeConnection(peer_id="peer2", capabilities=["compile"]),
        }
        router = DelegationRouter(wormhole)

        # Record stats: peer1 has better performance
        router._record_delegation("peer1", "compile", success=True, duration=5.0)
        router._record_delegation("peer1", "compile", success=True, duration=5.0)
        router._record_delegation("peer2", "compile", success=True, duration=15.0)
        router._record_delegation("peer2", "compile", success=False, duration=10.0)

        peer_id = router.choose_peer_for_task("compile")
        assert peer_id == "peer1"

    def test_record_delegation_tracks_stats(self):
        """Performance tracking accumulates correctly."""
        wormhole = MagicMock()
        wormhole.connections = {}
        router = DelegationRouter(wormhole)

        router._record_delegation("peer1", "compile", success=True, duration=5.0)
        router._record_delegation("peer1", "compile", success=False, duration=10.0)

        stats = router.get_peer_stats("peer1")
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert stats["total_duration"] == 15.0
        assert stats["by_task"]["compile"]["count"] == 2
        assert stats["by_task"]["compile"]["successes"] == 1


@pytest.mark.slow
class TestPeerHandlers:
    """Handlers require full bridge context. Marked slow for integration."""

    def test_handler_imports(self):
        """Verify handlers can be imported."""
        from mother.peer_handlers import (
            handle_compile_request,
            handle_build_request,
            handle_tool_offer,
            create_message_router,
        )
        assert callable(handle_compile_request)
        assert callable(handle_build_request)
        assert callable(handle_tool_offer)
        assert callable(create_message_router)
