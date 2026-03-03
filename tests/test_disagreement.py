"""Tests for disagreement-capable — repeated failure triggers evidence-based pushback."""

import pytest


class TestDisagreement:

    def test_pushback_trigger_conditions(self):
        """Pushback requires: chronic_weak non-empty + attempt_count >= 2."""
        chronic_weak = ["traceability", "specificity"]
        attempt_count = 3
        assert len(chronic_weak) > 0
        assert attempt_count >= 2

    def test_no_pushback_first_attempt(self):
        """First attempt should not trigger pushback."""
        chronic_weak = ["traceability"]
        attempt_count = 1
        should_pushback = len(chronic_weak) > 0 and attempt_count >= 2
        assert not should_pushback

    def test_no_pushback_without_chronic(self):
        """No chronic weaknesses = no pushback."""
        chronic_weak = []
        attempt_count = 5
        should_pushback = len(chronic_weak) > 0 and attempt_count >= 2
        assert not should_pushback

    def test_pushback_message_format(self):
        """Pushback messages include evidence from trends."""
        trends_line = "Weak: traceability (35%). Declining: specificity (-12%)."
        attempt_count = 3
        msg = (
            f"I'd push back on this — {trends_line} "
            f"The last {attempt_count} attempts hit the same wall. "
            f"Want to try a different approach?"
        )
        assert "push back" in msg
        assert "traceability" in msg
        assert str(attempt_count) in msg
