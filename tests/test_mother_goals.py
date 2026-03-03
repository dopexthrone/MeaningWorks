"""Tests for mother/goals.py — Goal dataclass and GoalStore."""

import sqlite3
import time
from pathlib import Path

import pytest

from mother.goals import Goal, GoalStore, _PRIORITY_ORDER


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_goals.db"


@pytest.fixture
def store(db_path):
    s = GoalStore(db_path)
    yield s
    s.close()


# --- Goal dataclass ---

class TestGoalDataclass:
    def test_goal_defaults(self):
        g = Goal()
        assert g.goal_id == 0
        assert g.description == ""
        assert g.source == "user"
        assert g.priority == "normal"
        assert g.status == "active"
        assert g.progress_note == ""
        assert g.last_worked == 0.0
        assert g.completion_note == ""

    def test_goal_frozen(self):
        g = Goal(description="test")
        with pytest.raises(AttributeError):
            g.description = "changed"

    def test_goal_with_values(self):
        g = Goal(
            goal_id=1,
            timestamp=1000.0,
            description="build booking system",
            source="mother",
            priority="high",
            status="in_progress",
            progress_note="started",
            last_worked=1001.0,
            completion_note="",
        )
        assert g.goal_id == 1
        assert g.source == "mother"
        assert g.priority == "high"


# --- Priority ordering ---

class TestPriorityOrder:
    def test_priority_values(self):
        assert _PRIORITY_ORDER["urgent"] < _PRIORITY_ORDER["high"]
        assert _PRIORITY_ORDER["high"] < _PRIORITY_ORDER["normal"]
        assert _PRIORITY_ORDER["normal"] < _PRIORITY_ORDER["low"]

    def test_all_priorities_present(self):
        for p in ("urgent", "high", "normal", "low"):
            assert p in _PRIORITY_ORDER


# --- GoalStore CRUD ---

class TestGoalStore:
    def test_add_returns_id(self, store):
        gid = store.add("test goal")
        assert isinstance(gid, int)
        assert gid >= 1

    def test_add_and_get(self, store):
        gid = store.add("build something", source="user", priority="high")
        goal = store.get(gid)
        assert goal is not None
        assert goal.description == "build something"
        assert goal.source == "user"
        assert goal.priority == "high"
        assert goal.status == "active"
        assert goal.timestamp > 0

    def test_get_nonexistent(self, store):
        assert store.get(9999) is None

    def test_active_returns_active_goals(self, store):
        store.add("goal 1")
        store.add("goal 2")
        active = store.active()
        assert len(active) == 2
        assert all(g.status in ("active", "in_progress") for g in active)

    def test_active_excludes_done(self, store):
        gid = store.add("will be done")
        store.update_status(gid, "done", completion_note="finished")
        active = store.active()
        assert len(active) == 0

    def test_active_excludes_dismissed(self, store):
        gid = store.add("will dismiss")
        store.update_status(gid, "dismissed")
        active = store.active()
        assert len(active) == 0

    def test_active_priority_ordering(self, store):
        store.add("low goal", priority="low")
        store.add("urgent goal", priority="urgent")
        store.add("normal goal", priority="normal")
        store.add("high goal", priority="high")
        active = store.active()
        priorities = [g.priority for g in active]
        assert priorities[0] == "urgent"
        assert priorities[1] == "high"
        assert priorities[2] == "normal"
        assert priorities[3] == "low"

    def test_update_status(self, store):
        gid = store.add("in progress goal")
        store.update_status(gid, "in_progress", progress_note="working on it")
        goal = store.get(gid)
        assert goal.status == "in_progress"
        assert goal.progress_note == "working on it"
        assert goal.last_worked > 0

    def test_update_status_done(self, store):
        gid = store.add("completable")
        store.update_status(gid, "done", completion_note="all done")
        goal = store.get(gid)
        assert goal.status == "done"
        assert goal.completion_note == "all done"

    def test_count_active(self, store):
        assert store.count_active() == 0
        store.add("g1")
        assert store.count_active() == 1
        gid2 = store.add("g2")
        assert store.count_active() == 2
        store.update_status(gid2, "done")
        assert store.count_active() == 1

    def test_count_active_includes_in_progress(self, store):
        gid = store.add("g1")
        store.update_status(gid, "in_progress")
        assert store.count_active() == 1

    def test_all_goals(self, store):
        store.add("g1")
        store.add("g2")
        gid3 = store.add("g3")
        store.update_status(gid3, "done")
        all_g = store.all_goals()
        assert len(all_g) == 3  # includes done

    def test_all_goals_limit(self, store):
        for i in range(10):
            store.add(f"goal {i}")
        limited = store.all_goals(limit=3)
        assert len(limited) == 3

    def test_all_goals_newest_first(self, store):
        store.add("first")
        time.sleep(0.01)
        store.add("second")
        all_g = store.all_goals()
        assert all_g[0].description == "second"
        assert all_g[1].description == "first"


# --- next_actionable ---

class TestNextActionable:
    def test_empty_store(self, store):
        assert store.next_actionable() is None

    def test_picks_highest_priority(self, store):
        store.add("low", priority="low")
        store.add("high", priority="high")
        store.add("normal", priority="normal")
        goal = store.next_actionable()
        assert goal.priority == "high"

    def test_picks_urgent_over_high(self, store):
        store.add("high", priority="high")
        store.add("urgent", priority="urgent")
        goal = store.next_actionable()
        assert goal.priority == "urgent"

    def test_prefers_unworked(self, store):
        gid1 = store.add("worked", priority="normal")
        store.add("unworked", priority="normal")
        store.update_status(gid1, "in_progress", progress_note="touched")
        goal = store.next_actionable()
        assert goal.description == "unworked"

    def test_skips_done(self, store):
        gid1 = store.add("done goal", priority="urgent")
        store.update_status(gid1, "done")
        store.add("active goal", priority="low")
        goal = store.next_actionable()
        assert goal.description == "active goal"


# --- Source tagging ---

class TestSourceTagging:
    def test_user_source(self, store):
        gid = store.add("user goal", source="user")
        assert store.get(gid).source == "user"

    def test_mother_source(self, store):
        gid = store.add("mother goal", source="mother")
        assert store.get(gid).source == "mother"

    def test_system_source(self, store):
        gid = store.add("system goal", source="system")
        assert store.get(gid).source == "system"


# --- Persistence ---

class TestPersistence:
    def test_survives_close_reopen(self, db_path):
        s1 = GoalStore(db_path)
        gid = s1.add("persistent goal", priority="high")
        s1.close()

        s2 = GoalStore(db_path)
        goal = s2.get(gid)
        assert goal is not None
        assert goal.description == "persistent goal"
        assert goal.priority == "high"
        s2.close()

    def test_table_created_on_init(self, db_path):
        s = GoalStore(db_path)
        s.close()
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='goals'"
        ).fetchall()
        assert len(tables) == 1
        conn.close()


# --- Weekly build governance: approve/batch/pending ---


class TestGoalApprovalFlow:
    def test_approve_goals_basic(self, store):
        id1 = store.add("Improve convergence", source="system", priority="high")
        id2 = store.add("Fix daemon retry", source="system", priority="normal")
        count = store.approve_goals([id1, id2])
        assert count == 2
        approved = store.approved()
        assert len(approved) == 2
        assert all(g.status == "approved" for g in approved)

    def test_approve_goals_empty_list(self, store):
        store.add("Some goal")
        assert store.approve_goals([]) == 0

    def test_approve_idempotent(self, store):
        gid = store.add("Test goal")
        store.approve_goals([gid])
        # Second approval should not error (already approved, not active)
        count = store.approve_goals([gid])
        assert count == 0  # already approved, WHERE clause excludes it

    def test_approve_only_active_goals(self, store):
        gid = store.add("Test goal")
        store.update_status(gid, "done")
        count = store.approve_goals([gid])
        assert count == 0  # done goals can't be approved

    def test_approved_returns_priority_order(self, store):
        id_low = store.add("Low priority goal", priority="low")
        id_urgent = store.add("Urgent goal", priority="urgent")
        id_normal = store.add("Normal goal", priority="normal")
        store.approve_goals([id_low, id_urgent, id_normal])
        approved = store.approved()
        assert approved[0].priority == "urgent"
        assert approved[-1].priority == "low"

    def test_batch_for_window_respects_limit(self, store):
        for i in range(5):
            gid = store.add(f"Goal {i}", priority="normal")
            store.approve_goals([gid])
        batch = store.batch_for_window(limit=3)
        assert len(batch) == 3

    def test_batch_for_window_empty(self, store):
        store.add("Not approved yet")
        batch = store.batch_for_window()
        assert len(batch) == 0

    def test_pending_briefing_goals_all(self, store):
        store.add("Goal 1")
        store.add("Goal 2")
        pending = store.pending_briefing_goals()
        assert len(pending) == 2

    def test_pending_briefing_goals_since_timestamp(self, store):
        store.add("Old goal")
        cutoff = time.time()
        time.sleep(0.01)  # ensure timestamp difference
        store.add("New goal")
        pending = store.pending_briefing_goals(since_timestamp=cutoff)
        assert len(pending) == 1
        assert "New" in pending[0].description

    def test_pending_excludes_approved(self, store):
        gid = store.add("Already approved")
        store.approve_goals([gid])
        pending = store.pending_briefing_goals()
        assert len(pending) == 0

    def test_status_transition_active_to_approved_to_done(self, store):
        gid = store.add("Full lifecycle goal")
        store.approve_goals([gid])
        g = store.get(gid)
        assert g.status == "approved"
        store.update_status(gid, "in_progress")
        g = store.get(gid)
        assert g.status == "in_progress"
        store.update_status(gid, "done", completion_note="Built successfully")
        g = store.get(gid)
        assert g.status == "done"
