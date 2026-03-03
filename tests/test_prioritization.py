"""Tests for prioritization-enforcing — low priority goals skipped when budget thin."""

import pytest

from mother.stance import Stance, StanceContext, compute_stance


class TestPrioritizationEnforcing:

    def test_low_priority_skipped_concept(self):
        """Low-priority goals should be deprioritized when budget is tight.

        The actual skip happens in chat.py's _autonomous_work, but
        stance computation gates entry. Here we verify the dynamic budget
        mechanism reduces available actions.
        """
        # Frustrated + 3 actions → budget is 3 (5-2) → capped
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            frustration=0.5,
            autonomous_actions_this_session=3,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_high_priority_still_runs(self):
        """Non-low-priority goals should still execute within budget."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            frustration=0.0,
            autonomous_actions_this_session=3,
        )
        assert compute_stance(ctx) == Stance.ACT
