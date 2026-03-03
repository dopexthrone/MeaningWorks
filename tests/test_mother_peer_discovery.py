"""
Tests for mother/peer_discovery.py -- LEAF module.

Covers: PeerInfo frozen dataclass, PeerRegistry persistence,
add_peer, list_peers, update_seen, remove_peer.

Zeroconf/mDNS tests are skipped if zeroconf not installed.
"""

import json
import time
from pathlib import Path

import pytest

from mother.peer_discovery import PeerInfo, PeerRegistry


class TestPeerInfo:
    def test_frozen(self):
        p = PeerInfo(instance_id="test123", name="Test")
        with pytest.raises(AttributeError):
            p.name = "Changed"

    def test_defaults(self):
        p = PeerInfo(instance_id="abc")
        assert p.name == "Unknown"
        assert p.host == ""
        assert p.port == 0


class TestPeerRegistry:
    def test_add_peer(self, tmp_path):
        storage = tmp_path / "peers.json"
        registry = PeerRegistry(storage_path=storage)

        peer = registry.add_peer(
            instance_id="peer1",
            name="Mother-Ubuntu",
            host="192.168.1.100",
            port=8765,
            version="1.0",
            capabilities=["compile", "build"],
        )

        assert peer.instance_id == "peer1"
        assert peer.name == "Mother-Ubuntu"
        assert peer.host == "192.168.1.100"
        assert peer.port == 8765

    def test_persistence(self, tmp_path):
        storage = tmp_path / "peers.json"

        # Add peer and save
        registry1 = PeerRegistry(storage_path=storage)
        registry1.add_peer("peer1", "Test", "127.0.0.1", 8765)

        # Load in new instance
        registry2 = PeerRegistry(storage_path=storage)
        peer = registry2.get_peer("peer1")

        assert peer is not None
        assert peer.name == "Test"
        assert peer.host == "127.0.0.1"

    def test_list_peers(self, tmp_path):
        registry = PeerRegistry(storage_path=tmp_path / "peers.json")
        registry.add_peer("peer1", "First", "127.0.0.1", 8765)
        time.sleep(0.1)
        registry.add_peer("peer2", "Second", "127.0.0.2", 8766)

        peers = registry.list_peers()
        assert len(peers) == 2
        # Most recent first
        assert peers[0].instance_id == "peer2"

    def test_active_only_filter(self, tmp_path):
        registry = PeerRegistry(storage_path=tmp_path / "peers.json")

        # Add old peer
        old_peer = PeerInfo(
            instance_id="old",
            name="Old",
            host="1.1.1.1",
            port=8765,
            last_seen=time.time() - 400,  # 400s ago
        )
        registry.peers["old"] = old_peer
        registry._save()

        # Add recent peer
        registry.add_peer("recent", "Recent", "2.2.2.2", 8765)

        all_peers = registry.list_peers(active_only=False)
        assert len(all_peers) == 2

        active_peers = registry.list_peers(active_only=True, timeout=300.0)
        assert len(active_peers) == 1
        assert active_peers[0].instance_id == "recent"

    def test_update_seen(self, tmp_path):
        registry = PeerRegistry(storage_path=tmp_path / "peers.json")
        registry.add_peer("peer1", "Test", "127.0.0.1", 8765)

        initial_time = registry.get_peer("peer1").last_seen
        time.sleep(0.1)
        registry.update_seen("peer1")
        updated_time = registry.get_peer("peer1").last_seen

        assert updated_time > initial_time

    def test_remove_peer(self, tmp_path):
        registry = PeerRegistry(storage_path=tmp_path / "peers.json")
        registry.add_peer("peer1", "Test", "127.0.0.1", 8765)

        assert registry.get_peer("peer1") is not None
        removed = registry.remove_peer("peer1")
        assert removed is True
        assert registry.get_peer("peer1") is None

    def test_remove_nonexistent(self, tmp_path):
        registry = PeerRegistry(storage_path=tmp_path / "peers.json")
        removed = registry.remove_peer("nonexistent")
        assert removed is False


class TestZeroconfIntegration:
    """Tests requiring zeroconf library. Skipped if not installed."""

    @pytest.mark.skipif(
        not pytest.importorskip("zeroconf", reason="zeroconf not installed"),
        reason="zeroconf not available",
    )
    def test_start_discovery_returns_count(self, tmp_path):
        """start_discovery runs without crashing (no actual services to find)."""
        from mother.peer_discovery import start_discovery

        registry = PeerRegistry(storage_path=tmp_path / "peers.json")
        count = start_discovery(registry, timeout=0.5)
        assert isinstance(count, int)
        assert count >= 0
