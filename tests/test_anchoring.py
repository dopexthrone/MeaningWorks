"""
Phase 10.1: Semantic Anchoring Tests

Tests for:
- Three-level context (L1 Core, L2 Evolving, L3 Immediate)
- _extract_discovered_components() from insights
- _compute_uncovered_ground() from intent vs discovered
- Context within token budget
- Phase hints (EXPLORE, CHALLENGE, CONVERGE)
- Graceful fallback with empty state
"""

import pytest
from core.protocol import SharedState, Message, MessageType, ConfidenceVector
from agents.base import LLMAgent, BaseAgent


class MockLLM:
    pass


def make_agent(name="Entity"):
    return LLMAgent(
        name=name,
        perspective="Structure: what exists",
        system_prompt="Test",
        llm_client=MockLLM()
    )


def make_state_with_intent():
    """Create a state with intent already extracted."""
    state = SharedState()
    state.known["input"] = "Build a booking system for tattoo artists"
    state.known["intent"] = {
        "core_need": "Scheduling and booking for tattoo artists",
        "domain": "booking/scheduling",
        "actors": ["Artist", "Client"],
        "constraints": ["Must handle time slots", "Support multiple artists"],
        "explicit_components": ["Artist", "Client", "Booking", "Session", "Payment"],
    }
    return state


class TestL1CoreAnchoring:
    """Test L1 (Core) level — immutable context."""

    def test_l1_contains_core_need(self):
        """L1 should contain the core_need from intent."""
        agent = make_agent()
        state = make_state_with_intent()
        context = agent._build_context(state)
        assert "Scheduling and booking for tattoo artists" in context

    def test_l1_contains_domain(self):
        """L1 should contain the domain."""
        agent = make_agent()
        state = make_state_with_intent()
        context = agent._build_context(state)
        assert "booking/scheduling" in context

    def test_l1_contains_constraints(self):
        """L1 should contain key constraints."""
        agent = make_agent()
        state = make_state_with_intent()
        context = agent._build_context(state)
        assert "Must handle time slots" in context

    def test_l1_fallback_no_intent(self):
        """L1 should fall back to raw input when no intent."""
        agent = make_agent()
        state = SharedState()
        state.known["input"] = "Build something cool"
        context = agent._build_context(state)
        assert "Build something cool" in context


class TestL2EvolvingState:
    """Test L2 (Evolving) level — recomputed per turn."""

    def test_l2_shows_discovered_components(self):
        """L2 should list components found in insights."""
        agent = make_agent()
        state = make_state_with_intent()
        state.insights.append("Artist entity has availability slots")
        state.insights.append("Booking connects Client to Artist")
        context = agent._build_context(state)
        assert "Artist" in context
        assert "Booking" in context

    def test_l2_shows_confidence_dimensions(self):
        """L2 should show structural, behavioral, coverage confidence."""
        agent = make_agent()
        state = make_state_with_intent()
        state.confidence.structural = 0.6
        state.confidence.behavioral = 0.3
        state.confidence.coverage = 0.5
        context = agent._build_context(state)
        assert "S=0.6" in context
        assert "B=0.3" in context
        assert "C=0.5" in context

    def test_l2_shows_conflict_count(self):
        """L2 should show number of unresolved conflicts."""
        agent = make_agent()
        state = make_state_with_intent()
        state.add_conflict("Entity", "Process", "Booking nature", {})
        context = agent._build_context(state)
        assert "conflicts=1" in context

    def test_l2_zero_conflicts_when_none(self):
        """L2 should show conflicts=0 when no conflicts."""
        agent = make_agent()
        state = make_state_with_intent()
        context = agent._build_context(state)
        assert "conflicts=0" in context


class TestL3ImmediateContext:
    """Test L3 (Immediate) level — adaptive per turn."""

    def test_l3_shows_uncovered_components(self):
        """L3 should list components not yet discussed."""
        agent = make_agent()
        state = make_state_with_intent()
        # Only Artist discovered, others uncovered
        state.insights.append("Artist entity has scheduling slots")
        context = agent._build_context(state)
        # At least some of the uncovered components should appear
        assert "uncovered=" in context

    def test_l3_shows_unknowns(self):
        """L3 should list active unknowns."""
        agent = make_agent()
        state = make_state_with_intent()
        state.add_unknown("Payment processing unclear")
        context = agent._build_context(state)
        assert "Payment processing unclear" in context

    def test_l3_phase_explore(self):
        """L3 should say EXPLORE in early dialogue turns."""
        agent = make_agent()
        state = make_state_with_intent()
        # 0-3 dialogue turns = EXPLORE
        context = agent._build_context(state)
        assert "phase=EXPLORE" in context

    def test_l3_phase_challenge(self):
        """L3 should say CHALLENGE in middle dialogue turns."""
        agent = make_agent()
        state = make_state_with_intent()
        for i in range(6):
            state.add_message(Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content=f"Turn {i}",
                message_type=MessageType.PROPOSITION
            ))
        context = agent._build_context(state)
        assert "phase=CHALLENGE" in context

    def test_l3_phase_converge(self):
        """L3 should say CONVERGE in late dialogue turns."""
        agent = make_agent()
        state = make_state_with_intent()
        for i in range(10):
            state.add_message(Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content=f"Turn {i}",
                message_type=MessageType.PROPOSITION
            ))
        context = agent._build_context(state)
        assert "phase=CONVERGE" in context


class TestDiscoveredComponents:
    """Test _extract_discovered_components() helper."""

    def test_extracts_capitalized_words(self):
        """Should extract capitalized words from insights."""
        agent = make_agent()
        state = SharedState()
        state.insights.append("User entity with email")
        state.insights.append("Session manages tokens")
        result = agent._extract_discovered_components(state)
        assert "User" in result
        assert "Session" in result

    def test_filters_noise_words(self):
        """Should filter common noise words."""
        agent = make_agent()
        state = SharedState()
        state.insights.append("This INSIGHT about The system")
        result = agent._extract_discovered_components(state)
        assert "INSIGHT" not in result
        assert "The" not in result
        assert "This" not in result

    def test_empty_insights(self):
        """Should return empty list with no insights."""
        agent = make_agent()
        state = SharedState()
        result = agent._extract_discovered_components(state)
        assert result == []


class TestUncoveredGround:
    """Test _compute_uncovered_ground() helper."""

    def test_all_uncovered_initially(self):
        """With no insights, all explicit components are uncovered."""
        agent = make_agent()
        state = make_state_with_intent()
        result = agent._compute_uncovered_ground(state)
        assert len(result) == 5  # Artist, Client, Booking, Session, Payment

    def test_partial_coverage(self):
        """Components mentioned in insights should be covered."""
        agent = make_agent()
        state = make_state_with_intent()
        state.insights.append("Artist entity has availability")
        state.insights.append("Client books through system")
        result = agent._compute_uncovered_ground(state)
        assert "Artist" not in result
        assert "Client" not in result
        assert "Payment" in result

    def test_no_explicit_components(self):
        """Should return empty when no explicit_components in intent."""
        agent = make_agent()
        state = SharedState()
        state.known["intent"] = {"core_need": "something", "explicit_components": []}
        result = agent._compute_uncovered_ground(state)
        assert result == []


class TestContextBudget:
    """Test that context stays within token budget."""

    def test_context_under_3200_chars(self):
        """Full context should be under 3200 chars (~800 tokens)."""
        agent = make_agent()
        state = make_state_with_intent()
        state.personas = [
            {"name": "Artist", "perspective": "Focus on creative time management"},
            {"name": "Client", "perspective": "Easy booking experience"},
        ]
        for i in range(8):
            state.add_message(Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content=f"Analysis turn {i} with some detail about the system.",
                message_type=MessageType.PROPOSITION
            ))
            state.insights.append(f"Insight about Component{i}")
        context = agent._build_context(state)
        assert len(context) < 3200, f"Context too long: {len(context)} chars"

    def test_recent_messages_limited_to_3(self):
        """Should only include 3 most recent messages."""
        agent = make_agent()
        state = make_state_with_intent()
        for i in range(10):
            state.add_message(Message(
                sender="Entity",
                content=f"UniqueMarker{i} analysis",
                message_type=MessageType.PROPOSITION
            ))
        context = agent._build_context(state)
        # Should contain messages 7, 8, 9 but not 0-6
        assert "UniqueMarker9" in context
        assert "UniqueMarker8" in context
        assert "UniqueMarker7" in context
        assert "UniqueMarker0" not in context
