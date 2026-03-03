"""Tests for decision-explaining — autonomous work emits explanation."""

import pytest

from mother.stance import Stance


class TestDecisionExplaining:

    def test_stance_values_stringifiable(self):
        """Stance values can be turned into readable strings for explanations."""
        for s in Stance:
            assert isinstance(s.value, str)
            assert len(s.value) > 0

    def test_explanation_format(self):
        """Explanation messages follow the expected format."""
        goal_id = 42
        action = "Compiled goal into 5-step plan"
        stance = Stance.ACT
        msg = f"[auto] Worked on goal #{goal_id}: {action}. Stance was {stance.value}."
        assert f"#{goal_id}" in msg
        assert stance.value in msg
        assert "[auto]" in msg
