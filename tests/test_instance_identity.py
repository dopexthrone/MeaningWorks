"""Tests for motherlabs_platform/instance_identity.py — Instance identity store."""

import pytest

from motherlabs_platform.instance_identity import (
    InstanceRecord,
    TrustGraphDigest,
    InstanceIdentityStore,
    build_trust_graph_digest,
    serialize_trust_graph_digest,
)
from core.tool_package import ToolDigest


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_instance.db")


@pytest.fixture
def store(tmp_db):
    return InstanceIdentityStore(db_path=tmp_db)


class MockToolRegistry:
    """Mock tool registry for testing trust graph digest."""

    def __init__(self, tools=None):
        self._tools = tools or []

    def list_tools(self):
        return self._tools


# =============================================================================
# SELF IDENTITY
# =============================================================================

class TestSelfIdentity:
    def test_create_self(self, store):
        identity = store.get_or_create_self("test-instance")
        assert identity.is_self is True
        assert identity.name == "test-instance"
        assert len(identity.instance_id) == 16
        assert identity.created_at != ""

    def test_idempotent(self, store):
        id1 = store.get_or_create_self("test")
        id2 = store.get_or_create_self("different-name")
        # Second call should return same ID
        assert id1.instance_id == id2.instance_id

    def test_frozen(self, store):
        identity = store.get_or_create_self()
        with pytest.raises(AttributeError):
            identity.name = "changed"


# =============================================================================
# PEER MANAGEMENT
# =============================================================================

class TestPeerManagement:
    def test_register_peer(self, store):
        store.register_peer("peer001", "Alice", "http://alice.example.com:8000")
        peers = store.list_peers()
        assert len(peers) == 1
        assert peers[0].instance_id == "peer001"
        assert peers[0].name == "Alice"
        assert peers[0].api_endpoint == "http://alice.example.com:8000"
        assert peers[0].is_self is False

    def test_get_peer(self, store):
        store.register_peer("peer001", "Alice", "http://alice:8000")
        peer = store.get_peer("peer001")
        assert peer is not None
        assert peer.name == "Alice"

    def test_get_nonexistent_peer(self, store):
        assert store.get_peer("nonexistent") is None

    def test_list_peers_empty(self, store):
        assert store.list_peers() == []

    def test_list_peers_excludes_self(self, store):
        store.get_or_create_self("self")
        store.register_peer("peer001", "Alice", "http://alice:8000")
        peers = store.list_peers()
        assert len(peers) == 1
        assert peers[0].is_self is False

    def test_remove_peer(self, store):
        store.register_peer("peer001", "Alice", "http://alice:8000")
        assert store.remove_peer("peer001") is True
        assert store.get_peer("peer001") is None

    def test_remove_nonexistent(self, store):
        assert store.remove_peer("nonexistent") is False

    def test_update_peer(self, store):
        store.register_peer("peer001", "Alice", "http://old:8000")
        store.register_peer("peer001", "Alice Updated", "http://new:8000")
        peer = store.get_peer("peer001")
        assert peer.name == "Alice Updated"
        assert peer.api_endpoint == "http://new:8000"

    def test_multiple_peers(self, store):
        store.register_peer("peer001", "Alice", "http://a:8000")
        store.register_peer("peer002", "Bob", "http://b:8000")
        peers = store.list_peers()
        assert len(peers) == 2


# =============================================================================
# TRUST GRAPH DIGEST
# =============================================================================

class TestTrustGraphDigest:
    def test_empty_registry(self):
        registry = MockToolRegistry()
        digest = build_trust_graph_digest("inst001", "test", registry)
        assert digest.tool_count == 0
        assert digest.verified_tool_count == 0
        assert digest.domain_counts == {}
        assert digest.avg_trust_score == 0.0

    def test_with_tools(self):
        tools = [
            ToolDigest("p1", "Tool1", "software", "fp1", 80.0, "verified", 3, 2, "inst001", "2026-01-01"),
            ToolDigest("p2", "Tool2", "software", "fp2", 60.0, "partial", 2, 1, "inst001", "2026-01-01"),
            ToolDigest("p3", "Tool3", "process", "fp3", 90.0, "verified", 5, 4, "inst001", "2026-01-01"),
        ]
        registry = MockToolRegistry(tools)
        digest = build_trust_graph_digest("inst001", "test", registry)
        assert digest.tool_count == 3
        assert digest.verified_tool_count == 2
        assert digest.domain_counts == {"software": 2, "process": 1}
        assert abs(digest.avg_trust_score - 76.7) < 0.1

    def test_frozen(self):
        registry = MockToolRegistry()
        digest = build_trust_graph_digest("inst001", "test", registry)
        with pytest.raises(AttributeError):
            digest.tool_count = 99

    def test_serialize(self):
        registry = MockToolRegistry()
        digest = build_trust_graph_digest("inst001", "test", registry)
        data = serialize_trust_graph_digest(digest)
        assert data["instance_id"] == "inst001"
        assert data["instance_name"] == "test"
        assert isinstance(data["domain_counts"], dict)
