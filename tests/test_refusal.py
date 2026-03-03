"""Tests for refusal-capable — REFUSE stance produces visible message."""

import pytest

from mother.stance import Stance, StanceContext, compute_stance


class TestRefusalMechanism:

    def test_refuse_stance_emitted(self):
        """REFUSE stance is returned when frustration high and goals stale."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.2,
            user_idle_seconds=600,
            frustration=0.7,
        )
        result = compute_stance(ctx)
        assert result == Stance.REFUSE

    def test_refuse_is_distinct_from_silent(self):
        """REFUSE is a different signal than SILENT — it requires a message."""
        assert Stance.REFUSE != Stance.SILENT
        assert Stance.REFUSE.value == "refuse"
