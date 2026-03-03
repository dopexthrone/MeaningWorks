"""
Phase 10.3: Unknown Tracking Tests (K↑/U↓)

Tests for:
- SharedState.add_unknown() deduplication
- SharedState.resolve_unknown() exact match
- SharedState.resolve_unknown_by_keyword() keyword resolution
- _extract_unknowns() parsing from agent responses
- Coverage penalty for unresolved unknowns
- Unknown resolution in dialogue loop
"""

import pytest
from core.protocol import SharedState, Message, MessageType, ConfidenceVector
from agents.base import LLMAgent


class MockLLM:
    """Minimal mock for LLMAgent construction."""
    pass


def make_agent():
    return LLMAgent(
        name="Test",
        perspective="Testing",
        system_prompt="Test",
        llm_client=MockLLM()
    )


class TestSharedStateAddUnknown:
    """Test SharedState.add_unknown() method."""

    def test_add_unknown_basic(self):
        """Should add unknown to list."""
        state = SharedState()
        state.add_unknown("How does auth handle expiry?")
        assert len(state.unknown) == 1
        assert state.unknown[0] == "How does auth handle expiry?"

    def test_add_unknown_dedup(self):
        """Should not add duplicate unknowns."""
        state = SharedState()
        state.add_unknown("How does auth handle expiry?")
        state.add_unknown("How does auth handle expiry?")
        assert len(state.unknown) == 1

    def test_add_unknown_dedup_case_insensitive(self):
        """Deduplication should be case-insensitive."""
        state = SharedState()
        state.add_unknown("Session management unclear")
        state.add_unknown("session management unclear")
        assert len(state.unknown) == 1

    def test_add_unknown_empty_rejected(self):
        """Empty strings should not be added."""
        state = SharedState()
        state.add_unknown("")
        state.add_unknown("   ")
        assert len(state.unknown) == 0

    def test_add_unknown_strips_whitespace(self):
        """Should strip whitespace from unknowns."""
        state = SharedState()
        state.add_unknown("  auth flow  ")
        assert state.unknown[0] == "auth flow"


class TestSharedStateResolveUnknown:
    """Test SharedState.resolve_unknown() method."""

    def test_resolve_unknown_exact(self):
        """Should remove unknown by exact match."""
        state = SharedState()
        state.add_unknown("Session management unclear")
        state.resolve_unknown("Session management unclear")
        assert len(state.unknown) == 0

    def test_resolve_unknown_case_insensitive(self):
        """Resolution should be case-insensitive."""
        state = SharedState()
        state.add_unknown("Session management unclear")
        state.resolve_unknown("session management unclear")
        assert len(state.unknown) == 0

    def test_resolve_unknown_no_match(self):
        """Should not remove anything if no match."""
        state = SharedState()
        state.add_unknown("Session management unclear")
        state.resolve_unknown("Something else")
        assert len(state.unknown) == 1


class TestResolveUnknownByKeyword:
    """Test SharedState.resolve_unknown_by_keyword() method."""

    def test_keyword_resolution_basic(self):
        """Should resolve unknown when 2+ significant words match."""
        state = SharedState()
        state.add_unknown("Session expiry mechanism unclear")
        resolved = state.resolve_unknown_by_keyword(
            "The session expiry uses a TTL-based mechanism with automatic cleanup."
        )
        assert len(resolved) == 1
        assert "Session expiry mechanism unclear" in resolved
        assert len(state.unknown) == 0

    def test_keyword_resolution_no_match(self):
        """Should not resolve when insufficient keyword matches."""
        state = SharedState()
        state.add_unknown("Session expiry mechanism unclear")
        resolved = state.resolve_unknown_by_keyword(
            "The user interface renders components based on routing."
        )
        assert len(resolved) == 0
        assert len(state.unknown) == 1

    def test_keyword_resolution_multiple(self):
        """Should resolve multiple unknowns in one pass."""
        state = SharedState()
        state.add_unknown("Authentication token validation unclear")
        state.add_unknown("Session management lifecycle unknown")
        resolved = state.resolve_unknown_by_keyword(
            "The authentication system validates tokens using JWT and manages "
            "session lifecycle through creation, renewal, and expiry."
        )
        assert len(resolved) == 2
        assert len(state.unknown) == 0

    def test_keyword_resolution_short_words_ignored(self):
        """Words with 3 or fewer chars should not count as matches."""
        state = SharedState()
        state.add_unknown("How the API key works")
        # "how", "the", "API", "key" are all <=3 chars; only "works" is >3
        resolved = state.resolve_unknown_by_keyword(
            "how the api key works in this system"
        )
        # Only 1 significant word ("works"), needs 2 by default
        # But threshold = min(2, 1) = 1, so it should resolve
        assert len(resolved) == 1

    def test_keyword_resolution_returns_resolved_list(self):
        """Should return the list of resolved unknowns."""
        state = SharedState()
        state.add_unknown("Database connection pooling strategy")
        resolved = state.resolve_unknown_by_keyword(
            "We use database connection pooling with a maximum of 10 connections."
        )
        assert resolved == ["Database connection pooling strategy"]


class TestExtractUnknowns:
    """Test LLMAgent._extract_unknowns() parsing."""

    def test_extract_single_unknown(self):
        """Should extract UNKNOWN: line from response."""
        agent = make_agent()
        state = SharedState()
        response = "Analysis complete.\nUNKNOWN: How does caching invalidate?"
        agent._extract_unknowns(state, response)
        assert len(state.unknown) == 1
        assert state.unknown[0] == "How does caching invalidate?"

    def test_extract_multiple_unknowns(self):
        """Should extract all UNKNOWN: lines."""
        agent = make_agent()
        state = SharedState()
        response = (
            "Structural analysis:\n"
            "UNKNOWN: Session expiry unclear\n"
            "Some other text\n"
            "UNKNOWN: User roles not defined"
        )
        agent._extract_unknowns(state, response)
        assert len(state.unknown) == 2

    def test_extract_unknown_case_insensitive(self):
        """UNKNOWN: marker should work in any case."""
        agent = make_agent()
        state = SharedState()
        response = "unknown: what triggers cancellation?"
        agent._extract_unknowns(state, response)
        assert len(state.unknown) == 1

    def test_extract_no_unknowns(self):
        """Should not add anything if no UNKNOWN: lines."""
        agent = make_agent()
        state = SharedState()
        response = "This is a normal response with no unknowns."
        agent._extract_unknowns(state, response)
        assert len(state.unknown) == 0

    def test_extract_empty_unknown_ignored(self):
        """UNKNOWN: with no text should be ignored."""
        agent = make_agent()
        state = SharedState()
        response = "UNKNOWN: "
        agent._extract_unknowns(state, response)
        assert len(state.unknown) == 0


class TestCoveragePenalty:
    """Test coverage confidence penalty for unresolved unknowns."""

    def test_no_penalty_no_unknowns(self):
        """Coverage should not be penalized with no unknowns."""
        agent = make_agent()
        state = SharedState()
        for i in range(5):
            state.add_insight(f"Insight {i}")
        agent._update_confidence(state, "sufficient", MessageType.PROPOSITION)
        # 5 insights * 0.08 = 0.4
        assert state.confidence.coverage == pytest.approx(0.4, abs=0.01)

    def test_penalty_with_unknowns(self):
        """Coverage should be reduced by unknown penalty."""
        agent = make_agent()
        state = SharedState()
        for i in range(5):
            state.add_insight(f"Insight {i}")
        state.add_unknown("Something unclear")
        state.add_unknown("Another ambiguity")
        agent._update_confidence(state, "sufficient", MessageType.PROPOSITION)
        # 5 * 0.08 = 0.4, penalty = 2 * 0.05 = 0.1, result = 0.3
        assert state.confidence.coverage == pytest.approx(0.3, abs=0.01)

    def test_penalty_capped(self):
        """Unknown penalty should cap at 0.3."""
        agent = make_agent()
        state = SharedState()
        for i in range(10):
            state.add_insight(f"Insight {i}")
        for i in range(10):
            state.add_unknown(f"Unknown {i}")
        agent._update_confidence(state, "sufficient", MessageType.PROPOSITION)
        # 10 * 0.08 = 0.8, penalty = capped at 0.3, result = 0.5
        assert state.confidence.coverage == pytest.approx(0.5, abs=0.01)
