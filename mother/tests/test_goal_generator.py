"""Tests for goal_generator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from goal_generator import (
    goals_from_feedback,
    goals_from_grid,
    ImprovementGoal,
)


def test_goals_from_feedback_100_rejection():
    goals = goals_from_feedback([], 1.0, "stable")
    assert len(goals) == 1
    goal = goals[0]
    assert goal.priority == "critical"
    assert "CRITICAL: Rejection rate at 100%" in goal.description
    assert goal.success_metric == "Zero rejections over 10 consecutive compilations."
    assert goal.category == "quality"


def test_goals_from_feedback_high_rejection():
    goals = goals_from_feedback([], 0.6, "stable")
    assert len(goals) == 1
    goal = goals[0]
    assert goal.priority == "critical"
    assert "60%" in goal.description
    assert "Rejection rate below 20%." == goal.success_metric


def test_goals_from_feedback_low_no_goal():
    goals = goals_from_feedback([], 0.2, "stable")
    assert len(goals) == 0


def test_goals_from_grid_low_confidence():
    cell_data = [("INT.ENT.001", 0.2, "F", "noun"), ("SEM.BHV.002", 0.1, "P", "verb")]
    goals = goals_from_grid(cell_data, set(), 2)
    assert len(goals) >= 1
    assert any("critical" == g.priority for g in goals)