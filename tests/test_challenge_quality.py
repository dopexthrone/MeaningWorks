"""
Phase 10.5: Challenge Quality Validation Tests

Tests for:
- Anti-gaming: "However, I completely agree" detected as AGREEMENT not CHALLENGE
- Substance check: generic pushback demoted to PROPOSITION
- Dominant type wins when both markers present
- Substantive challenges pass validation
- _is_substantive_challenge() reference matching
- Common words excluded from matching
"""

import pytest
from core.protocol import Message, MessageType
from agents.base import LLMAgent


class MockLLM:
    pass


def make_agent():
    return LLMAgent(
        name="Test",
        perspective="Testing",
        system_prompt="Test",
        llm_client=MockLLM()
    )


class TestAntiGaming:
    """Test that mixed agreement+challenge markers resolve correctly."""

    def test_however_with_agreement_is_agreement(self):
        """'However, I completely agree' should be AGREEMENT, not CHALLENGE."""
        agent = make_agent()
        text = "however, i completely agree with your analysis of the authentication flow."
        result = agent._detect_message_type(text, None)
        assert result == MessageType.AGREEMENT

    def test_agree_but_missing_is_challenge(self):
        """'I agree but you missed X' should be CHALLENGE when challenge dominates and references prior."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The Session entity manages lifecycle transitions for authentication tokens.",
            message_type=MessageType.PROPOSITION
        )
        # "i agree" (1 agreement) + "you missed" (1 strong) + "missing" (1 weak)
        # References "session" and "lifecycle" and "authentication" from prior
        text = "i agree on structure but you missed the session lifecycle — missing state transitions for authentication."
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE

    def test_weak_challenge_with_agreement_is_agreement(self):
        """Agreement + weak-only challenge markers → AGREEMENT wins."""
        agent = make_agent()
        # "sufficient" (agreement) + "however," (weak challenge) → AGREEMENT
        text = "however, this is sufficient for the current scope."
        result = agent._detect_message_type(text, None)
        assert result == MessageType.AGREEMENT

    def test_pure_agreement_still_works(self):
        """Pure agreement markers should still detect as AGREEMENT."""
        agent = make_agent()
        text = "this is sufficient. i agree with the analysis."
        result = agent._detect_message_type(text, None)
        assert result == MessageType.AGREEMENT

    def test_pure_challenge_still_works(self):
        """Pure challenge markers with substance should detect as CHALLENGE."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The authentication system needs User and Session entities.",
            message_type=MessageType.PROPOSITION
        )
        text = "but what about the session expiry? you missed the authentication token validation."
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE


class TestSubstanceCheck:
    """Test that challenges require specific content references."""

    def test_generic_pushback_demoted(self):
        """Generic 'however' without specific reference should be PROPOSITION."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The User entity has email and password attributes.",
            message_type=MessageType.PROPOSITION
        )
        # "however," is a challenge marker, but no specific words from prior
        text = "however, let me think about something else entirely unrelated."
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.PROPOSITION

    def test_specific_reference_passes(self):
        """Challenge referencing prior content should pass substance check."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The authentication flow handles login and session creation.",
            message_type=MessageType.PROPOSITION
        )
        # References "authentication" and "session" from prior
        text = "what about the authentication session expiry? you missed that."
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE

    def test_no_prior_message_allows_challenge(self):
        """First-turn challenges (no prior) should always pass substance check."""
        agent = make_agent()
        text = "however, what about error handling?"
        result = agent._detect_message_type(text, None)
        assert result == MessageType.CHALLENGE

    def test_substantive_with_component_name(self):
        """Challenge mentioning specific components should pass."""
        agent = make_agent()
        prior = Message(
            sender="Process",
            content="The booking process creates reservations with time slots.",
            message_type=MessageType.PROPOSITION
        )
        # References "booking" and "reservations" from prior
        text = "but what about cancellation? the booking and reservations need rollback."
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE


class TestIsSubstantiveChallenge:
    """Test _is_substantive_challenge() directly."""

    def test_substantive_with_references(self):
        """Should return True when 2+ significant words match."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The database schema includes users, sessions, and tokens.",
            message_type=MessageType.PROPOSITION
        )
        result = agent._is_substantive_challenge(
            "what about database schema migration?",
            prior
        )
        assert result is True

    def test_not_substantive_no_references(self):
        """Should return False when <2 significant words match."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The database schema includes users, sessions, and tokens.",
            message_type=MessageType.PROPOSITION
        )
        result = agent._is_substantive_challenge(
            "what about the weather today?",
            prior
        )
        assert result is False

    def test_substantive_no_prior_returns_true(self):
        """Should return True when no prior message to check against."""
        agent = make_agent()
        result = agent._is_substantive_challenge("what about error handling?", None)
        assert result is True

    def test_common_words_excluded(self):
        """Common words like 'system', 'about' should not count as references."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The system should handle events properly.",
            message_type=MessageType.PROPOSITION
        )
        # "system", "should", "about" are common words — don't count
        # Only "handle" and "events" are significant, but "handle" is 6 chars and
        # "events" is 6 chars
        result = agent._is_substantive_challenge(
            "what about the system here?",
            prior
        )
        # "system" and "about" are in _COMMON_WORDS, so no significant matches
        assert result is False


class TestExplicitChallengeOverride:
    """Phase 10.7b: Explicit 'Challenge:' line overrides polite agreement language."""

    def test_challenge_to_overrides_agreement(self):
        """'Challenge to you:' on its own line should override agreement markers."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The authentication system has User and Session entities with login flow.",
            message_type=MessageType.PROPOSITION
        )
        text = (
            "your entity mapping is solid. i agree with the structural analysis.\n"
            "challenge to you: what about the session expiry authentication flow?"
        )
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE

    def test_challenge_back_overrides_agreement(self):
        """'Challenge back:' should override agreement markers."""
        agent = make_agent()
        prior = Message(
            sender="Process",
            content="The booking flow handles reservation and cancellation processes.",
            message_type=MessageType.PROPOSITION
        )
        text = (
            "i agree your flow analysis is comprehensive enough.\n"
            "challenge back: the booking reservation needs cancellation rollback logic."
        )
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE

    def test_bold_markdown_challenge_overrides(self):
        """'**Challenge:**' with markdown bold should still override."""
        agent = make_agent()
        prior = Message(
            sender="Entity",
            content="The system architecture handles authentication and session management.",
            message_type=MessageType.PROPOSITION
        )
        text = (
            "i agree with the structural foundation.\n"
            "**challenge:** you haven't addressed the authentication session lifecycle."
        )
        result = agent._detect_message_type(text, prior)
        assert result == MessageType.CHALLENGE

    def test_no_override_without_line_start(self):
        """'challenge:' embedded mid-sentence should not override."""
        agent = make_agent()
        text = "i agree with the overall challenge: this is sufficient."
        result = agent._detect_message_type(text, None)
        # "challenge:" is embedded, not at line start — agreement wins
        assert result == MessageType.AGREEMENT


class TestExistingBehaviorPreserved:
    """Verify existing test expectations from test_convergence.py still hold."""

    def test_challenge_markers_still_detected(self):
        """Challenge markers from existing tests should still work."""
        agent = make_agent()
        challenge_texts_with_priors = [
            ("But what about edge cases in the authentication flow?",
             Message(sender="E", content="The authentication flow handles login.",
                     message_type=MessageType.PROPOSITION)),
            ("You missed the authentication flow entirely.",
             Message(sender="E", content="The authentication flow handles login.",
                     message_type=MessageType.PROPOSITION)),
            ("What happens when the user cancels the authentication?",
             Message(sender="E", content="The authentication flow handles user login.",
                     message_type=MessageType.PROPOSITION)),
            ("CHALLENGE: This assumption is wrong about the architecture.",
             Message(sender="E", content="The architecture assumption handles edge cases.",
                     message_type=MessageType.PROPOSITION)),
        ]
        for text, prior in challenge_texts_with_priors:
            msg_type = agent._detect_message_type(text.lower(), prior)
            assert msg_type == MessageType.CHALLENGE, f"Should detect '{text}' as challenge"

    def test_accommodation_still_works(self):
        """Accommodation detection should be unchanged."""
        agent = make_agent()
        challenge = Message(
            sender="Process",
            content="What about failure handling?",
            message_type=MessageType.CHALLENGE
        )
        accommodation_texts = [
            "you're right, i missed that.",
            "good point about the failures.",
        ]
        for text in accommodation_texts:
            msg_type = agent._detect_message_type(text, challenge)
            assert msg_type == MessageType.ACCOMMODATION, f"Should detect '{text}' as accommodation"
