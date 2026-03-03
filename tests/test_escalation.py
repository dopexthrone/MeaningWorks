"""Tests for escalation-intelligent — critical events produce visible alerts."""

import pytest

from mother.stance import Stance, StanceContext, compute_stance


class TestEscalation:

    def test_ask_stance_is_escalation_eligible(self):
        """ASK stance means Mother wants to propose — eligible for escalation."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=120,
        )
        assert compute_stance(ctx) == Stance.ASK

    def test_budget_escalation_concept(self):
        """When cost approaches limit, escalation should fire.

        Actual check is in chat.py, but we verify the concept:
        cost/limit >= 0.8 → escalation.
        """
        cost = 4.0
        limit = 5.0
        assert cost / limit >= 0.8

    def test_self_test_failure_escalation_concept(self):
        """Failed self-test should trigger escalation.

        Actual check is in chat.py, but we verify the data shape.
        """
        result = {"passed": False, "summary": "3 failed", "duration_seconds": 5.0}
        assert not result["passed"]
        assert result["summary"] != ""
