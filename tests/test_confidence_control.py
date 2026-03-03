"""
Phase 10.4: Confidence-Driven Dialogue Control Tests

Tests for:
- ConfidenceVector.weakest_dimension()
- ConfidenceVector.dimension_spread()
- ConfidenceVector.is_plateauing()
- SharedState.confidence_history tracking
- GovernorAgent low dimension protection
- GovernorAgent uneven confidence block
- GovernorAgent plateau detection
- Hard stop preserved at max_turns=12
"""

import pytest
from core.protocol import (
    SharedState, Message, MessageType, ConfidenceVector,
    CONFIDENCE_THRESHOLD_WARNING,
)
from agents.swarm import GovernorAgent


class TestWeakestDimension:
    """Test ConfidenceVector.weakest_dimension()."""

    def test_weakest_is_structural(self):
        cv = ConfidenceVector(structural=0.1, behavioral=0.5, coverage=0.6, consistency=0.7)
        assert cv.weakest_dimension() == "structural"

    def test_weakest_is_behavioral(self):
        cv = ConfidenceVector(structural=0.8, behavioral=0.2, coverage=0.6, consistency=0.7)
        assert cv.weakest_dimension() == "behavioral"

    def test_weakest_is_coverage(self):
        cv = ConfidenceVector(structural=0.8, behavioral=0.7, coverage=0.1, consistency=0.6)
        assert cv.weakest_dimension() == "coverage"

    def test_weakest_is_consistency(self):
        cv = ConfidenceVector(structural=0.8, behavioral=0.7, coverage=0.6, consistency=0.1)
        assert cv.weakest_dimension() == "consistency"


class TestDimensionSpread:
    """Test ConfidenceVector.dimension_spread()."""

    def test_even_spread(self):
        cv = ConfidenceVector(structural=0.5, behavioral=0.5, coverage=0.5, consistency=0.5)
        assert cv.dimension_spread() == pytest.approx(0.0)

    def test_large_spread(self):
        cv = ConfidenceVector(structural=0.9, behavioral=0.3, coverage=0.5, consistency=0.7)
        assert cv.dimension_spread() == pytest.approx(0.6)

    def test_moderate_spread(self):
        cv = ConfidenceVector(structural=0.6, behavioral=0.4, coverage=0.5, consistency=0.5)
        assert cv.dimension_spread() == pytest.approx(0.2)


class TestIsPlateauing:
    """Test ConfidenceVector.is_plateauing()."""

    def test_plateauing_no_change(self):
        cv = ConfidenceVector()
        history = [0.5, 0.5, 0.5, 0.51]
        assert cv.is_plateauing(history, window=3) is True

    def test_not_plateauing_with_change(self):
        cv = ConfidenceVector()
        history = [0.3, 0.4, 0.5, 0.6]
        assert cv.is_plateauing(history, window=3) is False

    def test_not_plateauing_insufficient_history(self):
        cv = ConfidenceVector()
        history = [0.5, 0.5]
        assert cv.is_plateauing(history, window=3) is False

    def test_custom_threshold(self):
        cv = ConfidenceVector()
        history = [0.5, 0.52, 0.53]
        # With threshold=0.05, this is plateauing
        assert cv.is_plateauing(history, window=3, threshold=0.05) is True
        # With threshold=0.01, this is NOT plateauing
        assert cv.is_plateauing(history, window=3, threshold=0.01) is False


class TestConfidenceHistory:
    """Test SharedState.confidence_history field."""

    def test_confidence_history_starts_empty(self):
        state = SharedState()
        assert state.confidence_history == []

    def test_confidence_history_records_snapshots(self):
        state = SharedState()
        state.confidence = ConfidenceVector(structural=0.5, behavioral=0.5, coverage=0.5, consistency=0.5)
        state.confidence_history.append(state.confidence.overall())
        assert len(state.confidence_history) == 1
        assert state.confidence_history[0] == pytest.approx(0.5)


class TestGovernorLowDimensionProtection:
    """Test Governor blocks convergence when dimensions are below WARNING."""

    def test_blocks_convergence_with_low_dimension(self):
        """Should NOT end dialogue when a dimension is below 0.4 after min_turns."""
        gov = GovernorAgent()
        state = SharedState()
        # Structural is very low
        state.confidence = ConfidenceVector(
            structural=0.2, behavioral=0.6, coverage=0.6, consistency=0.6
        )

        # Meet depth: 6 turns, 8 insights, 2+ agreements
        for i in range(8):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")

        # At turn_count=6, min_turns=6, the low dim protection requires turn < 6+2=8
        result = gov.should_end_dialogue(state, turn_count=6, min_turns=6, min_insights=8)
        assert result is False, "Should block convergence with low structural confidence"

    def test_allows_convergence_after_extra_turns(self):
        """Should allow convergence after extra turns even with low dimension."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.2, behavioral=0.6, coverage=0.6, consistency=0.6
        )

        for i in range(10):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")

        # At turn_count=8, min_turns=6, 8 >= 6+2=8, so protection expires
        result = gov.should_end_dialogue(state, turn_count=8, min_turns=6, min_insights=8)
        assert result is True


class TestGovernorUnevenSpreadBlock:
    """Test Governor blocks convergence with uneven confidence dimensions."""

    def test_blocks_high_spread(self):
        """Should block when max-min spread > 0.4."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.9, behavioral=0.3, coverage=0.6, consistency=0.5
        )

        for i in range(8):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")

        # Spread = 0.9 - 0.3 = 0.6 > 0.4, and turn_count=7 < 6+3=9
        result = gov.should_end_dialogue(state, turn_count=7, min_turns=6, min_insights=8)
        assert result is False

    def test_allows_after_spread_turns(self):
        """Should allow after enough turns even with spread."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.9, behavioral=0.3, coverage=0.6, consistency=0.5
        )

        for i in range(10):
            msg = Message(
                sender="Entity" if i % 2 == 0 else "Process",
                content="I agree with this.",
                message_type=MessageType.AGREEMENT
            )
            state.add_message(msg)
            state.add_insight(f"Insight {i}")

        # turn_count=9 >= 6+3=9, spread block expires
        result = gov.should_end_dialogue(state, turn_count=9, min_turns=6, min_insights=8)
        assert result is True


class TestGovernorHardStop:
    """Test max_turns hard stop is always preserved (Phase 12.1a: now a parameter)."""

    def test_hard_stop_at_max_turns(self):
        """Should always stop at max_turns regardless of confidence."""
        gov = GovernorAgent()
        state = SharedState()
        # All confidence dimensions are low
        state.confidence = ConfidenceVector(
            structural=0.1, behavioral=0.1, coverage=0.1, consistency=0.1
        )
        # Phase 12.1a: pass explicit max_turns=12 to test hard stop
        result = gov.should_end_dialogue(state, turn_count=12, min_turns=6, min_insights=8, max_turns=12)
        assert result is True

    def test_no_convergence_before_depth(self):
        """Should not converge before min_turns even with high confidence."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.9, behavioral=0.9, coverage=0.9, consistency=0.9
        )
        for i in range(4):
            state.add_message(Message(
                sender="Entity", content="Agree",
                message_type=MessageType.AGREEMENT
            ))
        result = gov.should_end_dialogue(state, turn_count=4, min_turns=6, min_insights=8)
        assert result is False
