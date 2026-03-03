"""Tests for Build 5 genome wiring — successor spawning, reputation, network mapping."""

import pytest

from mother.appendage_builder import BuildSpec, generate_build_prompt
from mother.peer_discovery import PeerInfo, PeerRegistry


class TestSuccessorSpawning:
    """#17: Child appendages inherit parent context."""

    def test_build_prompt_includes_constraints(self):
        """BuildSpec.constraints are included in the generated prompt."""
        spec = BuildSpec(
            name="test-agent",
            description="Test agent",
            capability_gap="Testing",
            capabilities=("test",),
            constraints="PARENT SYSTEM:\nCodebase: 91 files\n\nRECENT BUILDS:\n[ok] test",
        )
        prompt = generate_build_prompt(spec)
        assert "PARENT SYSTEM" in prompt
        assert "91 files" in prompt
        assert "RECENT BUILDS" in prompt

    def test_build_prompt_without_constraints(self):
        """BuildSpec without constraints still generates valid prompt."""
        spec = BuildSpec(
            name="test-agent",
            description="Test agent",
            capability_gap="Testing",
            capabilities=("test",),
        )
        prompt = generate_build_prompt(spec)
        assert "test-agent" in prompt
        assert "ADDITIONAL CONSTRAINTS" not in prompt

    def test_build_spec_frozen(self):
        """BuildSpec is a frozen dataclass."""
        spec = BuildSpec(name="a", description="b", capability_gap="c")
        with pytest.raises(AttributeError):
            spec.name = "changed"


class TestNetworkMapping:
    """#41: Peer registry maps known instances."""

    def test_add_and_list_peers(self, tmp_path):
        """Peers persist and can be listed."""
        storage = tmp_path / "peers.json"
        registry = PeerRegistry(storage_path=storage)
        registry.add_peer(
            instance_id="peer-001",
            name="TestMother",
            host="192.168.1.10",
            port=8080,
            version="1.0",
            capabilities=["compile", "build"],
        )
        peers = registry.list_peers()
        assert len(peers) == 1
        assert peers[0].instance_id == "peer-001"
        assert "compile" in peers[0].capabilities

    def test_peer_persistence(self, tmp_path):
        """Peers survive registry restart."""
        storage = tmp_path / "peers.json"
        reg1 = PeerRegistry(storage_path=storage)
        reg1.add_peer(instance_id="peer-002", name="Persistent", host="10.0.0.1", port=8080)
        # New registry from same file
        reg2 = PeerRegistry(storage_path=storage)
        peers = reg2.list_peers()
        assert any(p.instance_id == "peer-002" for p in peers)

    def test_peer_info_fields(self):
        """PeerInfo has all required fields."""
        peer = PeerInfo(
            instance_id="p1",
            name="Test",
            host="localhost",
            port=9000,
            trust_score=85.0,
            capabilities=["search"],
        )
        assert peer.trust_score == 85.0
        assert peer.capabilities == ["search"]


class TestReputationAwareness:
    """#39: Compile success rate feeds into confidence/posture."""

    def test_confidence_from_success_rate(self):
        """Verify senses compute confidence from compile stats."""
        from mother.senses import SenseObservations, compute_senses
        obs = SenseObservations(
            compile_count=10,
            compile_success_count=8,
            messages_this_session=5,
        )
        sv = compute_senses(obs)
        # Confidence should be positive when success rate is high
        assert sv.confidence > 0.0

    def test_low_success_rate_lowers_confidence(self):
        """Low success rate should reduce confidence."""
        from mother.senses import SenseObservations, compute_senses
        obs_good = SenseObservations(
            compile_count=10,
            compile_success_count=9,
            messages_this_session=5,
        )
        obs_bad = SenseObservations(
            compile_count=10,
            compile_success_count=2,
            messages_this_session=5,
        )
        sv_good = compute_senses(obs_good)
        sv_bad = compute_senses(obs_bad)
        assert sv_good.confidence >= sv_bad.confidence
