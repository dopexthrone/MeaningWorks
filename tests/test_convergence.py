"""
Convergence Validation Tests

Derived from: AXIOMS.md Constraints C002, C003, C007

Tests ensure:
1. Challenge protocol enforced (C002: challenge before agreement)
2. Substantive challenges required (C003: must reference + propose alternative)
3. Convergence requires depth (C007: min turns, min insights, 2+ agreements)
4. No premature termination
"""

import pytest
from core.protocol import (
    SharedState, Message, MessageType,
    DialogueProtocol, TerminationState, ConfidenceVector
)


class TestDialogueProtocol:
    """Test DialogueProtocol convergence behavior."""

    def test_depth_requirements_default(self):
        """Protocol should have depth requirements by default."""
        protocol = DialogueProtocol()

        assert protocol.min_turns == 6
        assert protocol.min_insights == 8
        assert protocol.max_turns == 64  # Phase 12.1a: adaptive ceiling default

    def test_no_early_termination_insufficient_turns(self):
        """Should not terminate before min_turns even with agreements."""
        protocol = DialogueProtocol(min_turns=6, min_insights=4)
        state = SharedState()

        # Add 4 turns with agreements
        for i in range(4):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree. SUFFICIENT.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            protocol.turn_count = i + 1

        # Add enough insights
        for _ in range(5):
            state.add_insight("Test insight")

        # Should NOT terminate - not enough turns
        result = protocol.should_terminate(state)
        assert result is None, "Should not terminate before min_turns"

    def test_no_early_termination_insufficient_insights(self):
        """Should not terminate before min_insights even with enough turns."""
        protocol = DialogueProtocol(min_turns=4, min_insights=8)
        state = SharedState()

        # Add 6 turns with agreements
        for i in range(6):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree. SUFFICIENT.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
        protocol.turn_count = 6

        # Only add 3 insights (below min)
        for _ in range(3):
            state.add_insight("Test insight")

        # Should NOT terminate - not enough insights
        result = protocol.should_terminate(state)
        assert result is None, "Should not terminate before min_insights"

    def test_termination_requires_agreements(self):
        """Should not terminate without 2+ agreements in last 4 turns."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()

        # Add enough turns and insights
        for i in range(6):
            # Only propositions, no agreements
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I propose this. SUFFICIENT.",
                message_type=MessageType.PROPOSITION
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")
        protocol.turn_count = 6

        # Should NOT terminate - no agreements
        result = protocol.should_terminate(state)
        assert result is None, "Should not terminate without agreements"

    def test_termination_requires_sufficient_signal(self):
        """Should not terminate without 'sufficient' in recent messages."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()

        # Add enough turns, insights, and agreements but NO "sufficient"
        for i in range(6):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this approach.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")
        protocol.turn_count = 6

        # Should NOT terminate - no "sufficient" signal
        result = protocol.should_terminate(state)
        assert result is None, "Should not terminate without SUFFICIENT signal"

    def test_termination_blocked_by_unknowns(self):
        """Should not terminate SUCCESS if unknowns remain."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()
        state.unknown.append("Unresolved ambiguity")

        # Add proper convergence signals
        for i in range(6):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree. SUFFICIENT.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")
        protocol.turn_count = 6

        # Should NOT terminate SUCCESS - unknowns exist
        result = protocol.should_terminate(state)
        assert result is None, "Should not terminate SUCCESS with unknowns"

    def test_successful_convergence(self):
        """Should terminate SUCCESS when all conditions met."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()

        # Meet all conditions: turns, insights, agreements, sufficient, no unknowns
        for i in range(6):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this analysis. SUFFICIENT.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")
        protocol.turn_count = 6

        result = protocol.should_terminate(state)
        assert result == TerminationState.SUCCESS, "Should terminate SUCCESS"

    def test_exhaustion_at_max_turns(self):
        """Should terminate EXHAUSTION at max_turns regardless of depth."""
        protocol = DialogueProtocol(max_turns=12, min_turns=6, min_insights=8)
        state = SharedState()

        # Only 2 insights - normally wouldn't converge
        state.add_insight("Insight 1")
        state.add_insight("Insight 2")

        protocol.turn_count = 12

        result = protocol.should_terminate(state)
        assert result == TerminationState.EXHAUSTION, "Should EXHAUST at max_turns"

    def test_user_flag_terminates(self):
        """Should terminate USER_FLAG when insight flagged."""
        protocol = DialogueProtocol(min_turns=6, min_insights=8)
        state = SharedState()

        state.add_insight("Test insight")
        state.flag_current()  # Flag it

        protocol.turn_count = 2

        result = protocol.should_terminate(state)
        assert result == TerminationState.USER_FLAG, "Should terminate on flag"


class TestChallengeProtocol:
    """Test challenge-before-agreement constraint (C002, C003)."""

    def test_challenge_markers_detected(self):
        """Challenge markers should be detected in content."""
        from agents.base import LLMAgent

        # Create minimal agent for testing detection
        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        challenge_texts = [
            "But what about edge cases?",
            "You missed the authentication flow.",
            "What happens when the user cancels?",
            "CHALLENGE: This assumption is wrong.",
            "However, this doesn't account for failures.",
            "What about error handling?"
        ]

        for text in challenge_texts:
            msg_type = agent._detect_message_type(text.lower(), None)
            assert msg_type == MessageType.CHALLENGE, f"Should detect '{text}' as challenge"

    def test_accommodation_after_challenge(self):
        """Accommodation should be detected when responding to challenge."""
        from agents.base import LLMAgent

        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        # Create a challenge message
        challenge = Message(
            sender="Process",
            content="What about failure handling?",
            message_type=MessageType.CHALLENGE
        )

        accommodation_texts = [
            "You're right, I missed that.",
            "Good point about the failures.",
            "I missed the error states.",
            "Adding error handling now.",
            "Revising to include that case."
        ]

        for text in accommodation_texts:
            msg_type = agent._detect_message_type(text.lower(), challenge)
            assert msg_type == MessageType.ACCOMMODATION, f"Should detect '{text}' as accommodation"

    def test_agreement_requires_sufficient(self):
        """Agreement should require 'sufficient' marker."""
        from agents.base import LLMAgent

        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        # "i agree" now triggers agreement (expanded detection for cross-model)
        msg_type = agent._detect_message_type("i agree with this", None)
        assert msg_type == MessageType.AGREEMENT, "i agree should trigger agreement"

        # "sufficient" also triggers agreement
        msg_type = agent._detect_message_type("this is sufficient.", None)
        assert msg_type == MessageType.AGREEMENT, "SUFFICIENT should trigger agreement"

        # Neither agree nor sufficient - should be proposition
        msg_type = agent._detect_message_type("this is a structural analysis", None)
        assert msg_type == MessageType.PROPOSITION, "Plain analysis should be proposition"


class TestInsightExtraction:
    """Test insight extraction preserves full data (Issue 1 fix)."""

    def test_insight_full_not_truncated(self):
        """Full insight should never be truncated."""
        from agents.base import LLMAgent

        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        # Long insight (>60 chars)
        long_insight = "INSIGHT: This is a very long insight that describes something complex about the system architecture and its components"

        full, display = agent._extract_insight(long_insight)

        # Full should be complete
        assert len(full) > 60, "Full insight should be preserved"
        assert "architecture" in full, "Full insight should contain all words"

        # Display should be truncated
        assert len(display) <= 60, "Display should be truncated"
        assert display.endswith("..."), "Display should end with ellipsis"

    def test_insight_display_under_60(self):
        """Short insights should not be truncated."""
        from agents.base import LLMAgent

        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        short_insight = "INSIGHT: User needs booking flow"

        full, display = agent._extract_insight(short_insight)

        assert full == display, "Short insights should be identical"
        assert not display.endswith("..."), "Should not truncate short insights"

    def test_symbolic_patterns_extracted(self):
        """Symbolic patterns should be extracted as insights."""
        from agents.base import LLMAgent

        class MockLLM:
            pass

        agent = LLMAgent(
            name="Test",
            perspective="Testing",
            system_prompt="Test",
            llm_client=MockLLM()
        )

        patterns = [
            "- Booking = User + Session + Payment",  # Decomposition
            "- Request → Validation → Processing",   # Implication (unicode)
            "- Request -> Validation -> Processing", # Implication (ascii)
            "- Entity ≠ Process",                    # Contrast (unicode)
            "- Entity != Process",                   # Contrast (ascii)
            "conflict: UI wants X, backend wants Y", # Resolution
            "hidden: implicit state machine",        # Discovery
        ]

        for pattern in patterns:
            full, display = agent._extract_insight(pattern)
            assert full is not None, f"Should extract insight from '{pattern}'"


class TestSharedStateInsights:
    """Test SharedState insight handling."""

    def test_message_insight_added_to_state(self):
        """Message with insight should add to state.insights."""
        state = SharedState()

        msg = Message(
            sender="Entity",
            content="Analysis content",
            message_type=MessageType.PROPOSITION,
            insight="User needs authentication before booking"
        )

        state.add_message(msg)

        assert len(state.insights) == 1
        assert state.insights[0] == "User needs authentication before booking"

    def test_message_without_insight_no_addition(self):
        """Message without insight should not add to state.insights."""
        state = SharedState()

        msg = Message(
            sender="Entity",
            content="Analysis content",
            message_type=MessageType.PROPOSITION,
            insight=None
        )

        state.add_message(msg)

        assert len(state.insights) == 0

    def test_flag_current_marks_last_insight(self):
        """Flagging should mark the last insight index."""
        state = SharedState()

        state.add_insight("First insight")
        state.add_insight("Second insight")
        state.flag_current()

        assert len(state.flags) == 1
        assert state.flags[0] == 1  # Index of "Second insight"


class TestConfidenceVector:
    """Test ConfidenceVector (ConvergenceSignaling component)."""

    def test_overall_calculation(self):
        """Overall should be average of all dimensions."""
        cv = ConfidenceVector(
            structural=0.8,
            behavioral=0.6,
            coverage=0.7,
            consistency=0.9
        )
        assert abs(cv.overall() - 0.75) < 0.001  # (0.8+0.6+0.7+0.9)/4

    def test_is_sufficient_all_above_threshold(self):
        """Should return True when all dimensions meet threshold."""
        cv = ConfidenceVector(
            structural=0.8,
            behavioral=0.8,
            coverage=0.8,
            consistency=0.8
        )
        assert cv.is_sufficient(threshold=0.7) is True

    def test_is_sufficient_one_below_threshold(self):
        """Should return False if any dimension below threshold."""
        cv = ConfidenceVector(
            structural=0.8,
            behavioral=0.5,  # Below 0.7
            coverage=0.8,
            consistency=0.8
        )
        assert cv.is_sufficient(threshold=0.7) is False

    def test_to_dict_serialization(self):
        """Should serialize with all fields including overall."""
        cv = ConfidenceVector(structural=0.5, behavioral=0.5, coverage=0.5, consistency=0.5)
        d = cv.to_dict()
        assert "structural" in d
        assert "behavioral" in d
        assert "coverage" in d
        assert "consistency" in d
        assert "overall" in d
        assert d["overall"] == 0.5


class TestConflictOracle:
    """Test ConflictOracle functionality in SharedState."""

    def test_add_conflict(self):
        """Should track conflicts between agents."""
        state = SharedState()
        state.add_conflict(
            agent_a="Entity",
            agent_b="Process",
            topic="Authentication flow",
            positions={"Entity": "single sign-on", "Process": "multi-factor"}
        )
        assert len(state.conflicts) == 1
        assert state.conflicts[0]["resolved"] is False

    def test_resolve_conflict(self):
        """Should mark conflict as resolved."""
        state = SharedState()
        state.add_conflict("Entity", "Process", "Topic", {"Entity": "A", "Process": "B"})
        state.resolve_conflict(0, "Chose multi-factor with SSO fallback")
        assert state.conflicts[0]["resolved"] is True
        assert state.conflicts[0]["resolution"] == "Chose multi-factor with SSO fallback"

    def test_has_unresolved_conflicts(self):
        """Should detect unresolved conflicts."""
        state = SharedState()
        assert state.has_unresolved_conflicts() is False

        state.add_conflict("Entity", "Process", "Topic", {})
        assert state.has_unresolved_conflicts() is True

        state.resolve_conflict(0, "Resolved")
        assert state.has_unresolved_conflicts() is False

    def test_ambiguous_termination_on_multiple_conflicts(self):
        """Should terminate AMBIGUOUS when 2+ conflicts unresolved."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()

        # Add two unresolved conflicts
        state.add_conflict("Entity", "Process", "Topic1", {})
        state.add_conflict("Entity", "Process", "Topic2", {})

        protocol.turn_count = 6
        for _ in range(6):
            state.add_insight("Insight")

        result = protocol.should_terminate(state)
        assert result == TerminationState.AMBIGUOUS

    def test_confidence_based_convergence(self):
        """Should terminate SUCCESS when confidence vector sufficient."""
        protocol = DialogueProtocol(min_turns=4, min_insights=4)
        state = SharedState()

        # Set high confidence across all dimensions
        state.confidence = ConfidenceVector(
            structural=0.8,
            behavioral=0.8,
            coverage=0.8,
            consistency=0.8
        )

        # Meet depth requirements
        protocol.turn_count = 6
        for i in range(6):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="Analysis content",
                message_type=MessageType.PROPOSITION
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")

        # Should converge on confidence alone (no "sufficient" signal needed)
        result = protocol.should_terminate(state)
        assert result == TerminationState.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
