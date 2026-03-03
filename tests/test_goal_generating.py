"""Tests for goal-generating — Mother creates goals from chronic weaknesses."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mother.goals import GoalStore


class TestGoalGeneration:

    def test_mother_source_accepted(self):
        """GoalStore accepts source='mother' for mother-generated goals."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        store = GoalStore(db_path)
        goal_id = store.add(
            description="Improve traceability — chronically weak (35%)",
            source="mother",
            priority="low",
        )
        assert goal_id > 0

        goal = store.get(goal_id)
        assert goal.source == "mother"
        assert goal.priority == "low"
        assert "traceability" in goal.description
        store.close()
        db_path.unlink(missing_ok=True)

    def test_mother_goals_are_low_priority(self):
        """Mother-generated goals should be low priority by default."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        store = GoalStore(db_path)
        gid = store.add("Improve specificity", source="mother", priority="low")
        goal = store.get(gid)
        assert goal.priority == "low"
        store.close()
        db_path.unlink(missing_ok=True)

    def test_max_two_mother_goals_concept(self):
        """The cap of 2 mother-generated goals per session is enforced in chat.py.

        We just verify the GoalStore can store multiple mother goals.
        """
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        store = GoalStore(db_path)
        store.add("Goal A", source="mother", priority="low")
        store.add("Goal B", source="mother", priority="low")
        store.add("Goal C", source="mother", priority="low")

        active = store.active()
        mother_goals = [g for g in active if g.source == "mother"]
        assert len(mother_goals) == 3  # store accepts all, chat.py caps at 2
        store.close()
        db_path.unlink(missing_ok=True)
