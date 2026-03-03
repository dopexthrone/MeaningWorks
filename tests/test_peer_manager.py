"""
Tests for PeerManager — peer lifecycle, heartbeat, tool sync.

All tests use mocked dependencies — no actual network needed.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass


# --- Helper fixtures ---

def _make_mock_store(peers=None):
    """Create a mocked InstanceIdentityStore."""
    store = MagicMock()
    store.list_peers.return_value = peers or []
    store.get_or_create_self.return_value = MagicMock(
        instance_id="self-id",
        name="Ubuntu Mother",
        api_endpoint="http://localhost:8000",
    )
    store.register_peer = MagicMock()
    store.get_peer.return_value = None
    store.remove_peer.return_value = True
    return store


def _make_peer_record(instance_id="peer-1", name="Mac Mother", endpoint="http://100.1.2.3:8000"):
    from motherlabs_platform.instance_identity import InstanceRecord
    return InstanceRecord(
        instance_id=instance_id,
        name=name,
        created_at="2026-01-01T00:00:00",
        api_endpoint=endpoint,
        is_self=False,
    )


def _make_reachable_status(instance_id="peer-1"):
    from motherlabs_platform.peer_client import PeerStatus
    return PeerStatus(
        instance_id=instance_id,
        reachable=True,
        latency_ms=15.0,
        last_checked="2026-02-15T10:00:00",
    )


def _make_unreachable_status(instance_id="peer-1"):
    from motherlabs_platform.peer_client import PeerStatus
    return PeerStatus(
        instance_id=instance_id,
        reachable=False,
        latency_ms=0.0,
        last_checked="2026-02-15T10:00:00",
        error="Connection refused",
    )


# --- Start/Stop Lifecycle ---

class TestPeerManagerLifecycle:
    """Test PeerManager start/stop."""

    def test_start_stop(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            await manager.start()
            assert manager.is_running
            assert manager._heartbeat_task is not None

            await manager.stop()
            assert not manager.is_running
        asyncio.run(_run())

    def test_start_loads_known_peers(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            record = _make_peer_record()
            store = _make_mock_store(peers=[record])
            manager = PeerManager(store, heartbeat_interval=300.0)

            # Mock the health check so it doesn't actually connect
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status()):
                await manager.start()

            assert len(manager._peers) == 1
            assert "peer-1" in manager._peers

            await manager.stop()
        asyncio.run(_run())

    def test_double_start_is_noop(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            await manager.start()
            task1 = manager._heartbeat_task
            await manager.start()
            task2 = manager._heartbeat_task
            assert task1 is task2

            await manager.stop()
        asyncio.run(_run())


# --- Add/Remove Peer ---

class TestPeerManagerAddRemove:
    """Test adding and removing peers."""

    def test_add_peer_reachable(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock,
                            return_value=True):
                    status = await manager.add_peer("peer-1", "Mac Mother", "http://100.1.2.3:8000")

            assert status.reachable is True
            assert "peer-1" in manager._peers
            assert manager._peers["peer-1"].is_online is True
            store.register_peer.assert_called_once()
        asyncio.run(_run())

    def test_add_peer_unreachable(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status()):
                status = await manager.add_peer("peer-1", "Mac Mother", "http://100.1.2.3:8000")

            assert status.reachable is False
            assert manager._peers["peer-1"].is_online is False
        asyncio.run(_run())

    def test_remove_peer(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock, return_value=True):
                    await manager.add_peer("peer-1", "Mac Mother", "http://100.1.2.3:8000")

            with patch("motherlabs_platform.peer_client.PeerClient.close",
                        new_callable=AsyncMock):
                result = await manager.remove_peer("peer-1")

            assert result is True
            assert "peer-1" not in manager._peers
        asyncio.run(_run())

    def test_remove_nonexistent_peer(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            result = await manager.remove_peer("no-such-peer")
            assert result is True  # store.remove_peer returns True
        asyncio.run(_run())


# --- Online Peers ---

class TestPeerManagerOnline:
    """Test get_online_peers filtering."""

    def test_get_online_peers(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            # Add one reachable, one not
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status("p1")):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock, return_value=True):
                    await manager.add_peer("p1", "Mac", "http://100.1.2.3:8000")

            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status("p2")):
                await manager.add_peer("p2", "Windows", "http://100.1.2.4:8000")

            online = manager.get_online_peers()
            assert len(online) == 1
            assert online[0].record.name == "Mac"
        asyncio.run(_run())


# --- Heartbeat ---

class TestPeerManagerHeartbeat:
    """Test heartbeat loop marking peers offline."""

    def test_peer_marked_offline_after_max_failures(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager, PeerState
            from motherlabs_platform.peer_client import PeerClient, PeerStatus

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0, max_failures=2)

            # Add a peer that was online
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock, return_value=True):
                    await manager.add_peer("peer-1", "Mac", "http://100.1.2.3:8000")

            assert manager._peers["peer-1"].is_online is True

            # Simulate two health check failures
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status()):
                await manager._check_all_peers()
                assert manager._peers["peer-1"].consecutive_failures == 1
                assert manager._peers["peer-1"].is_online is True  # Not yet at max

                await manager._check_all_peers()
                assert manager._peers["peer-1"].consecutive_failures == 2
                assert manager._peers["peer-1"].is_online is False  # Now offline
        asyncio.run(_run())

    def test_peer_comes_back_online(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0, max_failures=1)

            # Add and immediately mark offline
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status()):
                await manager.add_peer("peer-1", "Mac", "http://100.1.2.3:8000")

            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_unreachable_status()):
                await manager._check_all_peers()

            assert manager._peers["peer-1"].is_online is False

            # Peer comes back
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                await manager._check_all_peers()

            assert manager._peers["peer-1"].is_online is True
            assert manager._peers["peer-1"].consecutive_failures == 0
        asyncio.run(_run())


# --- Tool Sync ---

class TestPeerManagerToolSync:
    """Test tool digest sync."""

    def test_sync_finds_new_tools(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            # Add online peer
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock, return_value=True):
                    await manager.add_peer("peer-1", "Mac", "http://100.1.2.3:8000")

            # Mock remote tools and local registry
            with patch("motherlabs_platform.peer_client.PeerClient.list_tools",
                        new_callable=AsyncMock,
                        return_value=[
                            {"package_id": "remote-1", "name": "weather"},
                            {"package_id": "local-1", "name": "counter"},
                        ]):
                mock_registry = MagicMock()
                mock_tool = MagicMock()
                mock_tool.package_id = "local-1"
                mock_registry.list_tools.return_value = [mock_tool]

                with patch("motherlabs_platform.tool_registry.get_tool_registry",
                            return_value=mock_registry):
                    new_tools = await manager.sync_tool_digests("peer-1")

            assert len(new_tools) == 1
            assert new_tools[0]["package_id"] == "remote-1"
        asyncio.run(_run())

    def test_sync_offline_peer_returns_empty(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0)

            # No peers — sync returns empty
            result = await manager.sync_tool_digests("nonexistent")
            assert result == []
        asyncio.run(_run())


# --- Graceful Degradation ---

class TestPeerManagerGraceful:
    """Test that failures don't crash the manager."""

    def test_health_check_exception_handled(self):
        async def _run():
            from motherlabs_platform.peer_manager import PeerManager

            store = _make_mock_store()
            manager = PeerManager(store, heartbeat_interval=300.0, max_failures=1)

            # Add peer
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        return_value=_make_reachable_status()):
                with patch("motherlabs_platform.peer_client.PeerClient.register_self",
                            new_callable=AsyncMock, return_value=True):
                    await manager.add_peer("peer-1", "Mac", "http://100.1.2.3:8000")

            # Health check raises exception
            with patch("motherlabs_platform.peer_client.PeerClient.health_check",
                        new_callable=AsyncMock,
                        side_effect=Exception("Network error")):
                # Should not raise
                await manager._check_all_peers()

            # Peer should have incremented failures
            assert manager._peers["peer-1"].consecutive_failures == 1
        asyncio.run(_run())
