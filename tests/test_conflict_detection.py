"""
Phase 10.2: Active Conflict Detection Tests

Tests for:
- _extract_conflicts() parsing CONFLICT: lines
- Structural conflict detection (_detect_structural_conflicts)
- Conflict recording in SharedState
- AMBIGUOUS termination path activation
"""

import pytest
from unittest.mock import Mock
from core.protocol import SharedState, Message, MessageType, DialogueProtocol, TerminationState
from core.engine import MotherlabsEngine
from core.llm import BaseLLMClient
from agents.base import LLMAgent


class MockLLM:
    """Minimal mock for LLMAgent construction."""
    pass


def make_agent(name="Entity"):
    return LLMAgent(
        name=name,
        perspective="Testing",
        system_prompt="Test",
        llm_client=MockLLM()
    )


class TestExtractConflicts:
    """Test _extract_conflicts() parsing from agent responses."""

    def test_extract_single_conflict(self):
        """Should extract CONFLICT: line and record in state."""
        agent = make_agent()
        state = SharedState()
        response = "Analysis complete.\nCONFLICT: Session is an entity, not a process"
        agent._extract_conflicts(state, response)
        assert len(state.conflicts) == 1
        assert "Session is an entity, not a process" in state.conflicts[0]["topic"]

    def test_extract_multiple_conflicts(self):
        """Should extract all CONFLICT: lines."""
        agent = make_agent()
        state = SharedState()
        response = (
            "Structural analysis:\n"
            "CONFLICT: Payment should be separate entity\n"
            "More analysis\n"
            "CONFLICT: Order is not a simple record"
        )
        agent._extract_conflicts(state, response)
        assert len(state.conflicts) == 2

    def test_extract_conflict_case_insensitive(self):
        """CONFLICT: marker should work in any case."""
        agent = make_agent()
        state = SharedState()
        response = "conflict: booking as process vs entity"
        agent._extract_conflicts(state, response)
        assert len(state.conflicts) == 1

    def test_extract_no_conflicts(self):
        """Should not add anything if no CONFLICT: lines."""
        agent = make_agent()
        state = SharedState()
        response = "This is a normal response without conflicts."
        agent._extract_conflicts(state, response)
        assert len(state.conflicts) == 0

    def test_extract_empty_conflict_ignored(self):
        """CONFLICT: with no text should be ignored."""
        agent = make_agent()
        state = SharedState()
        response = "CONFLICT: "
        agent._extract_conflicts(state, response)
        assert len(state.conflicts) == 0

    def test_conflict_records_agent_name(self):
        """Conflict should record the extracting agent's name."""
        agent = make_agent("Entity")
        state = SharedState()
        response = "CONFLICT: Session handling disagrees"
        agent._extract_conflicts(state, response)
        assert state.conflicts[0]["agents"][0] == "Entity"


class TestStructuralConflictDetection:
    """Test _detect_structural_conflicts() in engine."""

    def _make_engine(self):
        """Create engine with mock client."""
        client = Mock(spec=BaseLLMClient)
        client.deterministic = True
        client.model = "mock"
        client.complete_with_system = Mock(return_value="{}")
        return MotherlabsEngine(llm_client=client, auto_store=False, cache_policy="none")

    def test_detect_entity_vs_process_conflict(self):
        """Should detect when Entity calls something an entity and Process calls it a flow."""
        engine = self._make_engine()
        state = SharedState()

        # Entity says Session is a data entity with attributes (near "Session")
        entity_msg = Message(
            sender="Entity",
            content="Session is a data entity with authentication attributes and token property",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(entity_msg)

        # Process says Session is a process flow (near "Session")
        process_msg = Message(
            sender="Process",
            content="Session is a process flow with state transition from active to expired",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(process_msg)

        engine._detect_structural_conflicts(state, process_msg)
        assert len(state.conflicts) >= 1
        assert "Session" in state.conflicts[0]["topic"]

    def test_no_conflict_same_perspective(self):
        """Should not flag conflict when both describe component similarly."""
        engine = self._make_engine()
        state = SharedState()

        entity_msg = Message(
            sender="Entity",
            content="User is a data entity with email and password attributes",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(entity_msg)

        process_msg = Message(
            sender="Process",
            content="User data entity needs validation before creation",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(process_msg)

        # Both call User an "entity" near the name, no conflict
        engine._detect_structural_conflicts(state, process_msg)
        # User is described as entity in both messages, no entity-vs-process split
        assert len(state.conflicts) == 0

    def test_no_conflict_no_shared_components(self):
        """Should not flag conflict when agents discuss different components."""
        engine = self._make_engine()
        state = SharedState()

        entity_msg = Message(
            sender="Entity",
            content="User entity has email and password attributes",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(entity_msg)

        process_msg = Message(
            sender="Process",
            content="Booking process flow handles scheduling and confirmation",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(process_msg)

        engine._detect_structural_conflicts(state, process_msg)
        # No shared capitalized component names -> no conflict
        assert len(state.conflicts) == 0

    def test_no_conflict_generic_words_filtered(self):
        """Should not flag conflicts on generic words like 'Analyzing' or 'INSIGHT'."""
        engine = self._make_engine()
        state = SharedState()

        entity_msg = Message(
            sender="Entity",
            content="Analyzing structure. INSIGHT: User entity contains fields.",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(entity_msg)

        process_msg = Message(
            sender="Process",
            content="Analyzing behavior. INSIGHT: Login flow triggers validation.",
            message_type=MessageType.PROPOSITION
        )
        state.add_message(process_msg)

        engine._detect_structural_conflicts(state, process_msg)
        # "Analyzing" and "INSIGHT" are noise words, filtered out
        assert len(state.conflicts) == 0


class TestAmbiguousTermination:
    """Test that conflicts now enable the AMBIGUOUS termination path."""

    def test_explicit_conflicts_enable_ambiguous(self):
        """Explicit CONFLICT: markers should trigger AMBIGUOUS termination."""
        agent = make_agent()
        state = SharedState()

        # Agent extracts two conflicts
        response = "CONFLICT: Session nature disagreement\nCONFLICT: Auth flow disputed"
        agent._extract_conflicts(state, response)

        # Verify AMBIGUOUS is possible
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        protocol.turn_count = 6
        for _ in range(6):
            state.add_insight("Insight")

        result = protocol.should_terminate(state)
        assert result == TerminationState.AMBIGUOUS

    def test_early_conflicts_do_not_terminate(self):
        """Conflicts before depth requirements should NOT trigger AMBIGUOUS."""
        agent = make_agent()
        state = SharedState()

        # Agent extracts two conflicts
        response = "CONFLICT: Session nature disagreement\nCONFLICT: Auth flow disputed"
        agent._extract_conflicts(state, response)

        # Depth NOT met: turn_count=2 < min_turns=4
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        protocol.turn_count = 2

        result = protocol.should_terminate(state)
        assert result is None  # Should continue, not AMBIGUOUS
