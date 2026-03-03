"""
Phase 11.3: Tests for L4 Generative Isomorphism — Pattern Transfer.

Tests _extract_pattern_matches(), digest integration, and synthesis prompt
pattern transfer directives.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.digest import (
    _extract_pattern_matches,
    _format_pattern_transfers,
    build_dialogue_digest,
)
from core.protocol import (
    SharedState,
    Message,
    MessageType,
    ConfidenceVector,
)
from agents.base import AgentCallResult


def _synthesis_result(content: str) -> AgentCallResult:
    """Wrap a JSON content string in an AgentCallResult for synthesis mocks."""
    msg = Message(sender="Synthesis", content=content, message_type=MessageType.PROPOSITION)
    return AgentCallResult(
        agent_name="Synthesis", response_text=content, message=msg,
        conflicts=(), unknowns=(), fractures=(),
        confidence_boost=0.0, agent_dimension="", has_insight=False,
        token_usage={},
    )


# =============================================================================
# PATTERN PARSING TESTS (11.3a)
# =============================================================================


class TestExtractWorksLikePattern:
    """Test "X works like Y" pattern extraction."""

    def test_extract_works_like_pattern(self):
        insights = ["flash catalog works like Pinterest — browse + preview + select"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["source"] == "flash catalog"
        assert result[0]["analog"] == "Pinterest"
        assert result[0]["hints"] == ["browse", "preview", "select"]
        assert result[0]["raw"] == insights[0]

    def test_extract_like_pattern(self):
        insights = ["refund flow is like Stripe refund process — validate, calculate, issue"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["source"] == "refund flow"
        assert result[0]["analog"] == "Stripe refund process"
        assert "validate" in result[0]["hints"]
        assert "calculate" in result[0]["hints"]
        assert "issue" in result[0]["hints"]

    def test_extract_no_pattern(self):
        insights = [
            "booking = commitment + scheduling + matching",
            "hidden: deposit requires refund policy entity",
            "user != customer - user is anyone who touches system",
        ]
        result = _extract_pattern_matches(insights)
        assert len(result) == 0

    def test_extract_multiple_patterns(self):
        insights = [
            "flash catalog works like Pinterest — browse + preview + select",
            "booking = commitment + scheduling",
            "walk-in queue works like Restaurant waitlist — check_in, estimate, notify",
        ]
        result = _extract_pattern_matches(insights)
        assert len(result) == 2
        assert result[0]["analog"] == "Pinterest"
        assert result[1]["analog"] == "Restaurant waitlist"

    def test_extract_hints_from_dash(self):
        insights = ["no-show flow works like Airline forfeit — auto_state + slot_release"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert "auto_state" in result[0]["hints"]
        assert "slot_release" in result[0]["hints"]

    def test_extract_hints_from_needs(self):
        insights = ["deposit ledger works like Stripe balance needs: hold, capture, refund"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["hints"] == ["hold", "capture", "refund"]

    def test_extract_hints_with_plus(self):
        insights = ["catalog works like Pinterest — browse + preview + save + select"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert len(result[0]["hints"]) == 4
        assert "browse" in result[0]["hints"]
        assert "save" in result[0]["hints"]

    def test_extract_capitalized_analog(self):
        """Analog must start with uppercase (proper noun / domain reference)."""
        insights = ["queue works like something simple"]
        result = _extract_pattern_matches(insights)
        # "something" starts lowercase — should not match
        assert len(result) == 0


class TestExtractPatternEdgeCases:
    """Edge cases for pattern extraction."""

    def test_insight_prefix_stripped(self):
        """INSIGHT: prefix should be removed from source."""
        insights = ["INSIGHT: catalog works like Pinterest — browse, select"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["source"] == "catalog"

    def test_no_hints_still_matches(self):
        """Pattern without hints should still be extracted."""
        insights = ["booking system works like Airbnb"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["source"] == "booking system"
        assert result[0]["analog"] == "Airbnb"
        assert result[0]["hints"] == []

    def test_needs_keyword_with_dash(self):
        """— needs: combo should work."""
        insights = ["queue works like Uber pool — needs: match, group, route, notify"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert result[0]["hints"] == ["match", "group", "route", "notify"]

    def test_empty_insights_list(self):
        result = _extract_pattern_matches([])
        assert result == []

    def test_hints_with_and_separator(self):
        insights = ["payment works like Stripe checkout — authorize and capture and settle"]
        result = _extract_pattern_matches(insights)
        assert len(result) == 1
        assert "authorize" in result[0]["hints"]
        assert "capture" in result[0]["hints"]
        assert "settle" in result[0]["hints"]


# =============================================================================
# DIGEST INTEGRATION TESTS (11.3a/b)
# =============================================================================


class TestDigestPatternSection:
    """Test PATTERN TRANSFERS section in digest."""

    def _make_state_with_pattern_insights(self):
        state = SharedState()
        m1 = Message(
            sender="Entity",
            content="The flash catalog works like Pinterest for design browsing",
            message_type=MessageType.PROPOSITION,
            insight="flash catalog works like Pinterest — browse + preview + select",
        )
        state.add_message(m1)
        return state

    def _make_state_without_pattern_insights(self):
        state = SharedState()
        m1 = Message(
            sender="Entity",
            content="User entity has email, password fields",
            message_type=MessageType.PROPOSITION,
            insight="User = email + password_hash + role",
        )
        state.add_message(m1)
        return state

    def test_digest_includes_pattern_section(self):
        state = self._make_state_with_pattern_insights()
        digest = build_dialogue_digest(state)
        assert "PATTERN TRANSFERS:" in digest

    def test_digest_no_pattern_section_without_matches(self):
        state = self._make_state_without_pattern_insights()
        digest = build_dialogue_digest(state)
        assert "PATTERN TRANSFERS:" not in digest

    def test_digest_pattern_format(self):
        state = self._make_state_with_pattern_insights()
        digest = build_dialogue_digest(state)
        # Should contain arrow format: source → analog: hints
        assert "flash catalog" in digest
        assert "Pinterest" in digest
        assert "→" in digest
        assert "browse" in digest

    def test_digest_multiple_patterns(self):
        state = SharedState()
        m1 = Message(
            sender="Entity",
            content="catalog analysis",
            message_type=MessageType.PROPOSITION,
            insight="flash catalog works like Pinterest — browse + preview",
        )
        m2 = Message(
            sender="Process",
            content="queue analysis",
            message_type=MessageType.PROPOSITION,
            insight="walk-in queue works like Restaurant waitlist — check_in, notify",
        )
        state.add_message(m1)
        state.add_message(m2)
        digest = build_dialogue_digest(state)
        assert "Pinterest" in digest
        assert "Restaurant waitlist" in digest

    def test_digest_preserves_existing_sections(self):
        """PATTERN TRANSFERS should not break other sections."""
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.7, behavioral=0.6, coverage=0.5, consistency=0.8
        )
        state.unknown = ["some unknown"]
        m1 = Message(
            sender="Entity",
            content="analysis",
            message_type=MessageType.PROPOSITION,
            insight="catalog works like Pinterest — browse, select",
        )
        state.add_message(m1)
        digest = build_dialogue_digest(state)
        assert "INSIGHTS:" in digest
        assert "CONFIDENCE:" in digest
        assert "UNKNOWNS:" in digest
        assert "PATTERN TRANSFERS:" in digest


# =============================================================================
# SYNTHESIS PROMPT TESTS (11.3b)
# =============================================================================


class TestSynthesisPatternDirectives:
    """Test that synthesis prompt includes PATTERN TRANSFER DIRECTIVES."""

    def _make_engine_with_pattern_state(self):
        """Create engine and state with pattern insights."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        mock = MockClient()
        engine = MotherlabsEngine(llm_client=mock, auto_store=False)

        state = SharedState()
        state.known["input"] = "Build a tattoo booking system"
        state.insights = [
            "flash catalog works like Pinterest — browse + preview + select",
            "booking = commitment + scheduling",
        ]
        return engine, state, mock

    def test_synthesis_includes_pattern_directives(self):
        engine, state, mock = self._make_engine_with_pattern_state()

        # Mock the synthesis agent to capture the prompt
        captured_prompts = []

        def capture_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result('{"components": [{"name": "FlashCatalog", "type": "entity"}], "relationships": [], "constraints": [], "unresolved": []}')

        engine.synthesis_agent.run_llm_only = capture_run

        engine._synthesize(state)

        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "PATTERN TRANSFER DIRECTIVES" in prompt

    def test_synthesis_no_directives_without_patterns(self):
        engine, state, mock = self._make_engine_with_pattern_state()
        state.insights = ["booking = commitment + scheduling"]  # No pattern

        captured_prompts = []

        def capture_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result('{"components": [{"name": "Booking", "type": "entity"}], "relationships": [], "constraints": [], "unresolved": []}')

        engine.synthesis_agent.run_llm_only = capture_run
        engine._synthesize(state)

        prompt = captured_prompts[0]
        assert "PATTERN TRANSFER DIRECTIVES" not in prompt

    def test_synthesis_directive_references_matched_domain(self):
        engine, state, mock = self._make_engine_with_pattern_state()

        captured_prompts = []

        def capture_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result('{"components": [{"name": "FlashCatalog", "type": "entity"}], "relationships": [], "constraints": [], "unresolved": []}')

        engine.synthesis_agent.run_llm_only = capture_run
        engine._synthesize(state)

        prompt = captured_prompts[0]
        assert "Pinterest" in prompt
        assert "flash catalog" in prompt

    def test_synthesis_directive_lists_methods(self):
        engine, state, mock = self._make_engine_with_pattern_state()

        captured_prompts = []

        def capture_run(s, msg, max_tokens=4096):
            captured_prompts.append(msg.content)
            return _synthesis_result('{"components": [{"name": "FlashCatalog", "type": "entity"}], "relationships": [], "constraints": [], "unresolved": []}')

        engine.synthesis_agent.run_llm_only = capture_run
        engine._synthesize(state)

        prompt = captured_prompts[0]
        assert "browse" in prompt
        assert "preview" in prompt
        assert "select" in prompt


# =============================================================================
# FORMAT HELPER TESTS
# =============================================================================


class TestFormatPatternTransfers:
    """Test _format_pattern_transfers() helper."""

    def test_single_pattern(self):
        patterns = [{
            "source": "catalog",
            "analog": "Pinterest",
            "hints": ["browse", "preview"],
            "raw": "catalog works like Pinterest — browse + preview",
        }]
        result = _format_pattern_transfers(patterns)
        assert "PATTERN TRANSFERS:" in result
        assert "catalog" in result
        assert "Pinterest" in result
        assert "browse" in result

    def test_no_hints_shows_placeholder(self):
        patterns = [{
            "source": "queue",
            "analog": "Uber",
            "hints": [],
            "raw": "queue works like Uber",
        }]
        result = _format_pattern_transfers(patterns)
        assert "(no specific hints)" in result
