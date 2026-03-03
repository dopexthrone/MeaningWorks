"""Tests for goal lifecycle wiring — counters, health, pruning."""

import sqlite3
import time
from pathlib import Path

import pytest

from mother.goals import Goal, GoalStore, compute_goal_health


class TestGoalCounters:

    def test_attempt_increments(self, tmp_path):
        """Autonomous work increments attempt_count on the goal."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        gid = store.add("Build a dashboard", source="user")
        assert store.get(gid).attempt_count == 0
        store.increment_attempt(gid)
        assert store.get(gid).attempt_count == 1
        store.increment_attempt(gid)
        assert store.get(gid).attempt_count == 2
        store.close()

    def test_stall_increments_and_resets(self, tmp_path):
        """Stall counter increments on idle ticks, resets on progress."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        gid = store.add("Deploy API", source="user")
        assert store.get(gid).stall_count == 0
        store.increment_stall(gid)
        store.increment_stall(gid)
        assert store.get(gid).stall_count == 2
        store.reset_stall(gid)
        assert store.get(gid).stall_count == 0
        store.close()

    def test_engagement_increments(self, tmp_path):
        """User engagement increments when user interacts with goal."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        gid = store.add("Design schema", source="user")
        store.increment_engagement(gid)
        store.increment_engagement(gid)
        assert store.get(gid).engagement_count == 2
        store.close()

    def test_redirect_increments(self, tmp_path):
        """Redirect counter increments when user changes topic."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        gid = store.add("Build frontend", source="user")
        store.increment_redirect(gid)
        assert store.get(gid).redirect_count == 1
        store.close()


class TestGoalHealth:

    def test_fresh_goal_healthy(self, tmp_path):
        """A fresh goal has high health."""
        goal = Goal(goal_id=1, timestamp=time.time(), description="test")
        health = compute_goal_health(goal)
        assert health >= 0.7

    def test_old_goal_decays(self):
        """A 14-day-old goal with no work and some stalls has very low health."""
        old = time.time() - (14 * 24 * 3600)
        goal = Goal(goal_id=1, timestamp=old, description="test", stall_count=3)
        health = compute_goal_health(goal)
        assert health < 0.1

    def test_stalls_reduce_health(self):
        """Stall count reduces health via stall_decay."""
        now = time.time()
        goal_ok = Goal(goal_id=1, timestamp=now, description="test", stall_count=0)
        goal_stalled = Goal(goal_id=2, timestamp=now, description="test", stall_count=4)
        assert compute_goal_health(goal_ok) > compute_goal_health(goal_stalled)

    def test_engagement_boosts_health(self):
        """Engagement bonus boosts health."""
        now = time.time()
        goal_no = Goal(goal_id=1, timestamp=now, description="test", engagement_count=0)
        goal_engaged = Goal(goal_id=2, timestamp=now, description="test", engagement_count=6)
        assert compute_goal_health(goal_engaged) >= compute_goal_health(goal_no)

    def test_redirect_penalty(self):
        """Redirects reduce health."""
        now = time.time()
        goal_ok = Goal(goal_id=1, timestamp=now, description="test", redirect_count=0)
        goal_redirected = Goal(goal_id=2, timestamp=now, description="test", redirect_count=5)
        assert compute_goal_health(goal_ok) > compute_goal_health(goal_redirected)


class TestGoalPruning:

    def test_score_and_prune_removes_stale(self, tmp_path):
        """score_and_prune auto-dismisses goals with health below threshold."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        # Create a very old goal
        conn = sqlite3.connect(str(db))
        old_ts = time.time() - (14 * 24 * 3600)  # 14 days ago
        conn.execute(
            """INSERT INTO goals
               (timestamp, description, source, priority, status,
                progress_note, last_worked, completion_note,
                engagement_count, redirect_count, stall_count, attempt_count)
               VALUES (?, 'ancient goal', 'user', 'normal', 'active',
                       '', 0.0, '', 0, 3, 8, 0)""",
            (old_ts,),
        )
        conn.commit()
        conn.close()

        # Also add a fresh healthy goal
        store.add("Fresh goal", source="user")

        assert store.count_active() == 2
        pruned = store.score_and_prune(threshold=0.1)
        assert pruned >= 1
        # Fresh goal should survive
        remaining = store.active()
        assert any("Fresh" in g.description for g in remaining)
        store.close()

    def test_score_and_prune_preserves_healthy(self, tmp_path):
        """Healthy goals are not pruned."""
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("Healthy goal", source="user")
        pruned = store.score_and_prune(threshold=0.1)
        assert pruned == 0
        assert store.count_active() == 1
        store.close()
