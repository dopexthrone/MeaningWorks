"""
Phase 10.8: Conflict Resolution Tests

Tests for:
- Structural conflicts (entity vs process) resolved by reframing
- Non-structural conflicts left unresolved
- Resolution text includes both perspectives
- Resolved conflicts marked correctly in SharedState
- Digest surfaces resolution text for synthesis
"""

import pytest
from unittest.mock import Mock
from core.protocol import SharedState, Message, MessageType
from core.engine import MotherlabsEngine
from core.llm import MockClient
from core.digest import build_dialogue_digest


def make_engine():
    """Create engine with mock client for testing."""
    return MotherlabsEngine(llm_client=MockClient())


class TestStructuralConflictResolution:
    """Structural disagreements should be resolved by reframing."""

    def test_entity_vs_process_resolved(self):
        """Entity says entity, Process says process → resolved as boundary."""
        engine = make_engine()
        state = SharedState()
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="ConflictOracle: structural disagreement (entity vs process)",
            positions={
                "Entity": "ConflictOracle as entity/structure",
                "Process": "ConflictOracle as process/flow"
            }
        )
        resolved = engine._resolve_conflicts(state)
        assert resolved == 1
        assert state.conflicts[0]["resolved"] is True
        assert "Boundary component" in state.conflicts[0]["resolution"]
        assert "ConflictOracle" in state.conflicts[0]["resolution"]

    def test_process_vs_entity_resolved(self):
        """Process says process, Entity says entity → also resolved."""
        engine = make_engine()
        state = SharedState()
        state.add_conflict(
            agent_a="Process",
            agent_b="Entity",
            topic="Agent: structural disagreement (process vs entity)",
            positions={
                "Process": "Agent as process/flow",
                "Entity": "Agent as entity/structure"
            }
        )
        resolved = engine._resolve_conflicts(state)
        assert resolved == 1
        assert state.conflicts[0]["resolved"] is True

    def test_resolution_includes_both_views(self):
        """Resolution text should reference both entity and process perspectives."""
        engine = make_engine()
        state = SharedState()
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="ThresholdConfig: structural disagreement (entity vs process)",
            positions={
                "Entity": "ThresholdConfig as entity/structure",
                "Process": "ThresholdConfig as process/flow"
            }
        )
        engine._resolve_conflicts(state)
        resolution = state.conflicts[0]["resolution"]
        assert "structural properties" in resolution
        assert "behavioral properties" in resolution

    def test_multiple_structural_conflicts(self):
        """Multiple structural conflicts all resolved."""
        engine = make_engine()
        state = SharedState()
        for comp in ["ConflictOracle", "Agent", "ConstraintBoundary"]:
            state.add_conflict(
                agent_a="Entity",
                agent_b="Process",
                topic=f"{comp}: structural disagreement (entity vs process)",
                positions={
                    "Entity": f"{comp} as entity/structure",
                    "Process": f"{comp} as process/flow"
                }
            )
        resolved = engine._resolve_conflicts(state)
        assert resolved == 3
        assert all(c["resolved"] for c in state.conflicts)


class TestNonStructuralConflictsPreserved:
    """Semantic/explicit conflicts should NOT be resolved."""

    def test_explicit_conflict_not_resolved(self):
        """CONFLICT: line from agent stays unresolved."""
        engine = make_engine()
        state = SharedState()
        state.add_conflict(
            agent_a="Process",
            agent_b="other",
            topic="SharedState is a transition hub not just storage",
            positions={
                "Process": "SharedState as active participant",
                "Entity": "SharedState as repository"
            }
        )
        resolved = engine._resolve_conflicts(state)
        assert resolved == 0
        assert state.conflicts[0]["resolved"] is False

    def test_mixed_conflicts(self):
        """Only structural conflicts resolved, semantic ones preserved."""
        engine = make_engine()
        state = SharedState()
        # Structural — should resolve
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="Oracle: structural disagreement (entity vs process)",
            positions={"Entity": "Oracle as entity", "Process": "Oracle as process"}
        )
        # Semantic — should not resolve
        state.add_conflict(
            agent_a="Process",
            agent_b="other",
            topic="Flows require structural foundations",
            positions={"Process": "flows first", "Entity": "entities first"}
        )
        resolved = engine._resolve_conflicts(state)
        assert resolved == 1
        assert state.conflicts[0]["resolved"] is True
        assert state.conflicts[1]["resolved"] is False

    def test_already_resolved_skipped(self):
        """Already-resolved conflicts not re-processed."""
        engine = make_engine()
        state = SharedState()
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="X: structural disagreement (entity vs process)",
            positions={"Entity": "X as entity", "Process": "X as process"}
        )
        state.resolve_conflict(0, "Already resolved manually")
        resolved = engine._resolve_conflicts(state)
        assert resolved == 0
        assert state.conflicts[0]["resolution"] == "Already resolved manually"


class TestDigestIncludesResolution:
    """Verify resolved conflicts appear in digest for synthesis."""

    def test_resolved_conflict_shows_in_digest(self):
        """Resolved conflict should appear as [RESOLVED] with resolution text."""
        state = SharedState()
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="ConflictOracle: structural disagreement (entity vs process)",
            positions={
                "Entity": "ConflictOracle as entity/structure",
                "Process": "ConflictOracle as process/flow"
            }
        )
        state.resolve_conflict(0, "Boundary component: include both aspects")

        digest = build_dialogue_digest(state)
        assert "RESOLVED" in digest
        assert "Boundary component" in digest

    def test_unresolved_still_in_digest(self):
        """Unresolved conflicts still appear as [UNRESOLVED]."""
        state = SharedState()
        state.add_conflict(
            agent_a="Process",
            agent_b="Entity",
            topic="Semantic disagreement",
            positions={"Process": "pos A", "Entity": "pos B"}
        )
        digest = build_dialogue_digest(state)
        assert "UNRESOLVED" in digest


class TestHasUnresolvedConflicts:
    """Verify has_unresolved_conflicts respects resolutions."""

    def test_all_resolved_returns_false(self):
        """After resolving all conflicts, has_unresolved should be False."""
        state = SharedState()
        state.add_conflict("E", "P", "X: structural disagreement (e vs p)", {})
        state.resolve_conflict(0, "Reframed")
        assert state.has_unresolved_conflicts() is False

    def test_mixed_returns_true(self):
        """With some unresolved, has_unresolved should be True."""
        state = SharedState()
        state.add_conflict("E", "P", "X: structural disagreement (e vs p)", {})
        state.add_conflict("P", "E", "Semantic issue", {})
        state.resolve_conflict(0, "Reframed")
        assert state.has_unresolved_conflicts() is True
