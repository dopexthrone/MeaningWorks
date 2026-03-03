"""Tests for goal health scoring in mother/goals.py."""

import time

import pytest

from mother.goals import Goal, GoalStore, compute_goal_health


# --- compute_goal_health pure function ---


class TestComputeGoalHealth:
    def test_fresh_goal_high_health(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now)
        health = compute_goal_health(g, now)
        assert health >= 0.95

    def test_week_old_goal_low_health(self):
        now = time.time()
        week_ago = now - (7 * 24 * 3600)
        g = Goal(timestamp=week_ago, last_worked=0.0)
        health = compute_goal_health(g, now)
        # age_decay=0, idle_decay=0, redirect_penalty=1.0 → 0.3*0 + 0.4*0 + 0.3*1.0 = 0.3
        assert health <= 0.3

    def test_stale_48h_no_work(self):
        now = time.time()
        g = Goal(timestamp=now - 3600, last_worked=now - (48 * 3600))
        health = compute_goal_health(g, now)
        # idle_decay=0 (48h stale), age_decay≈1.0, redirect=1.0
        # raw ≈ 0.3*1.0 + 0.4*0.0 + 0.3*1.0 = 0.6
        assert health < 0.7

    def test_recently_worked_goal_healthy(self):
        now = time.time()
        g = Goal(timestamp=now - (3 * 24 * 3600), last_worked=now - 3600)
        health = compute_goal_health(g, now)
        assert health >= 0.5

    def test_redirect_penalty(self):
        now = time.time()
        g_no_redirect = Goal(timestamp=now, last_worked=now, redirect_count=0)
        g_redirected = Goal(timestamp=now, last_worked=now, redirect_count=3)
        h_clean = compute_goal_health(g_no_redirect, now)
        h_redirected = compute_goal_health(g_redirected, now)
        assert h_redirected < h_clean

    def test_engagement_bonus(self):
        now = time.time()
        # Use a 3-day old goal so base health isn't already 1.0
        ts = now - (3 * 24 * 3600)
        g_no_engage = Goal(timestamp=ts, last_worked=ts, engagement_count=0)
        g_engaged = Goal(timestamp=ts, last_worked=ts, engagement_count=6)
        h_passive = compute_goal_health(g_no_engage, now)
        h_engaged = compute_goal_health(g_engaged, now)
        assert h_engaged > h_passive

    def test_engagement_bonus_capped(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, engagement_count=100)
        health = compute_goal_health(g, now)
        assert health <= 1.0

    def test_health_clamped_to_unit(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, engagement_count=20)
        health = compute_goal_health(g, now)
        assert 0.0 <= health <= 1.0

    def test_health_never_negative(self):
        now = time.time()
        ancient = now - (365 * 24 * 3600)
        g = Goal(timestamp=ancient, last_worked=0.0, redirect_count=10)
        health = compute_goal_health(g, now)
        assert health >= 0.0

    def test_no_last_worked_uses_age(self):
        now = time.time()
        g = Goal(timestamp=now - (24 * 3600), last_worked=0.0)
        health = compute_goal_health(g, now)
        # idle_hours falls back to age_hours (24h), so idle_decay = 1 - 24/48 = 0.5
        assert health < 0.9

    def test_default_now_uses_current_time(self):
        g = Goal(timestamp=time.time(), last_worked=time.time())
        health = compute_goal_health(g)  # now defaults to time.time()
        assert health >= 0.9


# --- GoalStore engagement/redirect/prune ---


class TestGoalStoreHealth:
    @pytest.fixture
    def store(self, tmp_path):
        s = GoalStore(tmp_path / "test.db")
        yield s
        s.close()

    def test_increment_engagement(self, store):
        gid = store.add("test goal")
        store.increment_engagement(gid)
        store.increment_engagement(gid)
        g = store.get(gid)
        assert g.engagement_count == 2

    def test_increment_redirect(self, store):
        gid = store.add("test goal")
        store.increment_redirect(gid)
        g = store.get(gid)
        assert g.redirect_count == 1

    def test_score_and_prune_dismisses_stale(self, store):
        # Create a very old goal with redirects to push health below 0.1
        now = time.time()
        store._conn.execute(
            """INSERT INTO goals
               (timestamp, description, source, priority, status,
                progress_note, last_worked, completion_note,
                engagement_count, redirect_count)
               VALUES (?, 'ancient goal', 'user', 'normal', 'active', '', 0.0, '', 0, 5)""",
            (now - 30 * 24 * 3600,),
        )
        store._conn.commit()
        dismissed = store.score_and_prune()
        assert dismissed >= 1
        # Verify it's dismissed
        goals = store.active()
        descriptions = [g.description for g in goals]
        assert "ancient goal" not in descriptions

    def test_score_and_prune_keeps_healthy(self, store):
        store.add("fresh goal")  # Just created, healthy
        dismissed = store.score_and_prune()
        assert dismissed == 0
        assert store.count_active() == 1

    def test_new_goal_has_zero_counts(self, store):
        gid = store.add("new goal")
        g = store.get(gid)
        assert g.engagement_count == 0
        assert g.redirect_count == 0


# --- Migration ---


class TestMigration:
    def test_old_db_gets_new_columns(self, tmp_path):
        """Simulate opening an old DB without engagement/redirect columns."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE goals (
            goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'user',
            priority TEXT NOT NULL DEFAULT 'normal',
            status TEXT NOT NULL DEFAULT 'active',
            progress_note TEXT NOT NULL DEFAULT '',
            last_worked REAL NOT NULL DEFAULT 0.0,
            completion_note TEXT NOT NULL DEFAULT ''
        )""")
        conn.execute(
            """INSERT INTO goals VALUES (1, 1000.0, 'old goal', 'user', 'normal', 'active', '', 0.0, '')"""
        )
        conn.commit()
        conn.close()

        # Opening with GoalStore should migrate
        store = GoalStore(db_path)
        g = store.get(1)
        assert g is not None
        assert g.engagement_count == 0
        assert g.redirect_count == 0
        store.close()


import sqlite3
