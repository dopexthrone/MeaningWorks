"""Tests for Build 4 genome wiring — entropy, deadlines, handoff, tool health."""

import time
from pathlib import Path

import pytest

from mother.appendage import AppendageStore
from mother.goals import Goal, GoalStore, compute_goal_health


class TestEntropyFighting:
    """#188: Appendage dissolution candidates detected and dissolved."""

    def test_dissolution_candidates_fresh(self, tmp_path):
        """Fresh active appendages are not dissolution candidates."""
        db = tmp_path / "test.db"
        store = AppendageStore(db)
        aid = store.register("test-app", "test", "gap", "prompt", str(tmp_path))
        store.update_status(aid, "active")
        store.record_use(aid)
        store.record_use(aid)
        store.record_use(aid)
        candidates = store.candidates_for_dissolution()
        assert len(candidates) == 0
        store.close()

    def test_dissolution_candidates_idle(self, tmp_path):
        """Idle, underused appendages are candidates for dissolution."""
        db = tmp_path / "test.db"
        store = AppendageStore(db)
        aid = store.register("idle-app", "idle", "gap", "prompt", str(tmp_path))
        store.update_status(aid, "spawned")
        # Never used, last_used = 0.0 which is < cutoff
        candidates = store.candidates_for_dissolution(idle_hours=0, min_uses=3)
        assert len(candidates) == 1
        assert candidates[0].name == "idle-app"
        store.close()

    def test_solidified_not_dissolved(self, tmp_path):
        """Solidified appendages are never dissolution candidates."""
        db = tmp_path / "test.db"
        store = AppendageStore(db)
        aid = store.register("solid-app", "solid", "gap", "prompt", str(tmp_path))
        store.update_status(aid, "solidified")
        candidates = store.candidates_for_dissolution(idle_hours=0, min_uses=100)
        assert len(candidates) == 0
        store.close()

    def test_dissolution_updates_status(self, tmp_path):
        """Dissolving an appendage changes its status."""
        db = tmp_path / "test.db"
        store = AppendageStore(db)
        aid = store.register("dissolve-me", "test", "gap", "prompt", str(tmp_path))
        store.update_status(aid, "active")
        store.update_status(aid, "dissolved")
        spec = store.get(aid)
        assert spec.status == "dissolved"
        store.close()


class TestDeadlineTracking:
    """#51: Goal deadlines affect health computation and ordering."""

    def test_goal_with_no_deadline(self, tmp_path):
        """Goal with no deadline behaves as before (0.0 = no deadline)."""
        goal = Goal(goal_id=1, timestamp=time.time(), description="no deadline")
        health = compute_goal_health(goal)
        assert health >= 0.5

    def test_overdue_goal_health_boosted(self):
        """Overdue goals get a health boost to prevent premature pruning."""
        now = time.time()
        # Overdue by 1 day
        goal = Goal(
            goal_id=1, timestamp=now - 7200, description="overdue task",
            due_timestamp=now - 86400,
        )
        health_with_deadline = compute_goal_health(goal, now)

        goal_no_deadline = Goal(
            goal_id=2, timestamp=now - 7200, description="no deadline task",
        )
        health_without_deadline = compute_goal_health(goal_no_deadline, now)

        assert health_with_deadline > health_without_deadline

    def test_approaching_deadline_adds_pressure(self):
        """Goals with deadlines within 3 days get urgency boost over distant deadlines."""
        now = time.time()
        # Both goals are 5 days old (so base health is lower, room for boost to matter)
        old_ts = now - 5 * 86400
        # Due in 1 day — should get deadline pressure
        goal_urgent = Goal(
            goal_id=1, timestamp=old_ts, description="due soon",
            due_timestamp=now + 86400,
        )
        # Due in 10 days — no pressure
        goal_relaxed = Goal(
            goal_id=2, timestamp=old_ts, description="due later",
            due_timestamp=now + 10 * 86400,
        )
        health_urgent = compute_goal_health(goal_urgent, now)
        health_relaxed = compute_goal_health(goal_relaxed, now)
        assert health_urgent > health_relaxed

    def test_due_timestamp_persists(self, tmp_path):
        """due_timestamp field persists through store add/get cycle."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        due = time.time() + 86400 * 3
        gid = store.add("Deadline goal", due_timestamp=due)
        goal = store.get(gid)
        assert abs(goal.due_timestamp - due) < 1.0
        store.close()

    def test_due_timestamp_defaults_zero(self, tmp_path):
        """Goals without explicit deadline have due_timestamp=0."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        gid = store.add("No deadline goal")
        goal = store.get(gid)
        assert goal.due_timestamp == 0.0
        store.close()


class TestHandoffGeneration:
    """#66: Handoff document generation from bridge."""

    def test_handoff_with_goals(self, tmp_path):
        """Handoff includes active goals."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {}

        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("Build dashboard", priority="high")
        store.add("Fix login bug", priority="normal")
        store.close()

        handoff = bridge.generate_handoff(db_path=db)
        assert "Active Goals" in handoff
        assert "Build dashboard" in handoff
        assert "Fix login bug" in handoff

    def test_handoff_with_snapshot(self, tmp_path):
        """Handoff includes identity from snapshot."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {}

        snapshot = {
            "identity": {"name": "Mother", "personality": "curious", "provider": "claude"},
            "session": {"messages": 42, "cost_usd": 1.50},
        }
        handoff = bridge.generate_handoff(snapshot=snapshot)
        assert "Identity" in handoff
        assert "curious" in handoff
        assert "$1.50" in handoff

    def test_handoff_empty(self, tmp_path):
        """Handoff with no data still produces header."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {}

        handoff = bridge.generate_handoff()
        assert "Handoff" in handoff


class TestToolChainHealth:
    """#77: Tool health tracking in bridge."""

    def test_health_tracking_initialized(self):
        """Bridge initializes with empty tool health."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {}
        assert bridge.get_tool_health() == {}

    def test_health_after_runs(self):
        """Health computed from run/failure counts."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {
            "my-tool": {"runs": 10, "failures": 2},
        }
        health = bridge.get_tool_health()
        assert health["my-tool"]["health_pct"] == 80.0

    def test_health_all_failures(self):
        """All failures = 0% health."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {
            "bad-tool": {"runs": 5, "failures": 5},
        }
        health = bridge.get_tool_health()
        assert health["bad-tool"]["health_pct"] == 0.0

    def test_health_no_runs(self):
        """No runs = 100% health (pristine)."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._tool_health = {
            "new-tool": {"runs": 0, "failures": 0},
        }
        health = bridge.get_tool_health()
        assert health["new-tool"]["health_pct"] == 100.0
