"""Tests for action accountability — closing the say/do gap.

Three gaps covered:
  A: No-op detection (stall counting, stuck escalation)
  B: Fire-and-forget actions (ActionResult, pending tracking)
  C: No failure states (attempt counting, stuck on compilation failure)
"""

import time
import sqlite3

import pytest

from mother.goals import Goal, GoalStore, compute_goal_health
from mother.config import MotherConfig
from mother.executive import PlanStore


# ===================================================================
# Gap A: Goal stall/attempt fields and health penalty
# ===================================================================

class TestGoalStallFields:
    """Goal dataclass gains stall_count and attempt_count."""

    def test_goal_defaults(self):
        g = Goal()
        assert g.stall_count == 0
        assert g.attempt_count == 0

    def test_goal_frozen_with_stall(self):
        g = Goal(stall_count=3, attempt_count=2)
        assert g.stall_count == 3
        assert g.attempt_count == 2


class TestGoalHealthStallDecay:
    """Stall decay multiplier in compute_goal_health."""

    def test_zero_stalls_no_penalty(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, stall_count=0)
        h = compute_goal_health(g, now)
        assert h > 0.5  # Healthy

    def test_stalls_reduce_health(self):
        now = time.time()
        g0 = Goal(timestamp=now, last_worked=now, stall_count=0)
        g2 = Goal(timestamp=now, last_worked=now, stall_count=2)
        h0 = compute_goal_health(g0, now)
        h2 = compute_goal_health(g2, now)
        assert h2 < h0  # More stalls = lower health

    def test_four_stalls_zero_health(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, stall_count=4)
        h = compute_goal_health(g, now)
        assert h == 0.0  # 4 stalls => stall_decay = 0

    def test_five_stalls_still_zero(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, stall_count=5)
        h = compute_goal_health(g, now)
        assert h == 0.0

    def test_one_stall_partial_decay(self):
        now = time.time()
        g = Goal(timestamp=now, last_worked=now, stall_count=1)
        h = compute_goal_health(g, now)
        # stall_decay = 0.75, so health is 75% of normal
        g_no_stall = Goal(timestamp=now, last_worked=now, stall_count=0)
        h_no_stall = compute_goal_health(g_no_stall, now)
        assert abs(h - h_no_stall * 0.75) < 0.01


class TestGoalStoreStallMethods:
    """GoalStore.increment_stall, increment_attempt, reset_stall."""

    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "goals.db"
        s = GoalStore(db)
        yield s
        s.close()

    def test_increment_stall_returns_count(self, store):
        gid = store.add("test goal")
        c1 = store.increment_stall(gid)
        assert c1 == 1
        c2 = store.increment_stall(gid)
        assert c2 == 2

    def test_increment_stall_touches_last_worked(self, store):
        gid = store.add("test goal")
        before = store.get(gid).last_worked
        time.sleep(0.01)
        store.increment_stall(gid)
        after = store.get(gid).last_worked
        assert after > before

    def test_increment_attempt_returns_count(self, store):
        gid = store.add("test goal")
        c1 = store.increment_attempt(gid)
        assert c1 == 1
        c2 = store.increment_attempt(gid)
        assert c2 == 2
        c3 = store.increment_attempt(gid)
        assert c3 == 3

    def test_reset_stall_zeros_count(self, store):
        gid = store.add("test goal")
        store.increment_stall(gid)
        store.increment_stall(gid)
        assert store.get(gid).stall_count == 2
        store.reset_stall(gid)
        assert store.get(gid).stall_count == 0

    def test_reset_stall_does_not_affect_attempt(self, store):
        gid = store.add("test goal")
        store.increment_attempt(gid)
        store.increment_stall(gid)
        store.reset_stall(gid)
        g = store.get(gid)
        assert g.stall_count == 0
        assert g.attempt_count == 1

    def test_stuck_status_excluded_from_active(self, store):
        gid = store.add("test goal")
        store.update_status(gid, "stuck", progress_note="test")
        active = store.active()
        assert all(g.goal_id != gid for g in active)

    def test_stuck_status_excluded_from_next_actionable(self, store):
        gid = store.add("test goal")
        store.update_status(gid, "stuck")
        nxt = store.next_actionable()
        assert nxt is None or nxt.goal_id != gid

    def test_stuck_excluded_from_count_active(self, store):
        gid = store.add("test goal")
        assert store.count_active() == 1
        store.update_status(gid, "stuck")
        assert store.count_active() == 0


class TestGoalStoreRowCompat:
    """_row_to_goal handles 9, 11, and 13 column rows."""

    def test_9_col_row(self):
        row = (1, 100.0, "desc", "user", "normal", "active", "", 0.0, "")
        g = GoalStore._row_to_goal(row)
        assert g.goal_id == 1
        assert g.stall_count == 0
        assert g.attempt_count == 0

    def test_11_col_row(self):
        row = (1, 100.0, "desc", "user", "normal", "active", "", 0.0, "", 5, 2)
        g = GoalStore._row_to_goal(row)
        assert g.engagement_count == 5
        assert g.redirect_count == 2
        assert g.stall_count == 0
        assert g.attempt_count == 0

    def test_13_col_row(self):
        row = (1, 100.0, "desc", "user", "normal", "active", "", 0.0, "", 5, 2, 3, 1)
        g = GoalStore._row_to_goal(row)
        assert g.stall_count == 3
        assert g.attempt_count == 1


class TestGoalStoreMigration:
    """Idempotent column migration on existing databases."""

    def test_migration_adds_columns(self, tmp_path):
        db = tmp_path / "migrate.db"
        # Create old schema without stall/attempt columns
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'user',
                priority TEXT NOT NULL DEFAULT 'normal',
                status TEXT NOT NULL DEFAULT 'active',
                progress_note TEXT NOT NULL DEFAULT '',
                last_worked REAL NOT NULL DEFAULT 0.0,
                completion_note TEXT NOT NULL DEFAULT '',
                engagement_count INTEGER NOT NULL DEFAULT 0,
                redirect_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "INSERT INTO goals (timestamp, description) VALUES (?, ?)",
            (time.time(), "old goal"),
        )
        conn.commit()
        conn.close()

        # Opening GoalStore should migrate
        store = GoalStore(db)
        goal = store.next_actionable()
        assert goal is not None
        assert goal.stall_count == 0
        assert goal.attempt_count == 0
        store.close()

    def test_migration_idempotent(self, tmp_path):
        db = tmp_path / "idempotent.db"
        s1 = GoalStore(db)
        s1.add("goal 1")
        s1.close()
        # Open again — should not error
        s2 = GoalStore(db)
        s2.add("goal 2")
        assert s2.count_active() == 2
        s2.close()


# ===================================================================
# Gap B: ActionResult dataclass
# ===================================================================

class TestActionResult:
    """ActionResult tracks pending async work."""

    def test_default_not_pending(self):
        from mother.screens.chat import ActionResult
        ar = ActionResult(message="done")
        assert ar.pending is False
        assert ar.success is True

    def test_pending_chain_text(self):
        from mother.screens.chat import ActionResult
        ar = ActionResult(message="Compilation started", pending=True)
        assert ar.chain_text == "[PENDING] Compilation started"

    def test_non_pending_chain_text(self):
        from mother.screens.chat import ActionResult
        ar = ActionResult(message="Status displayed")
        assert ar.chain_text == "Status displayed"

    def test_frozen(self):
        from mother.screens.chat import ActionResult
        ar = ActionResult(message="test")
        with pytest.raises(AttributeError):
            ar.message = "changed"


# ===================================================================
# Gap B: _execute_action returns ActionResult
# ===================================================================

class TestExecuteActionReturns:
    """_execute_action returns ActionResult or None."""

    @pytest.fixture
    def screen(self):
        from mother.screens.chat import ChatScreen
        config = MotherConfig()
        s = ChatScreen(config=config)
        # Stub action runners
        s._run_compile = lambda x: None
        s._run_build = lambda x: None
        s._run_search = lambda x: None
        s._run_tools = lambda: None
        s._run_status = lambda: None
        s._run_open = lambda x: None
        s._run_file_action = lambda x: None
        s._run_add_goal = lambda x: None
        s._run_list_goals = lambda: None
        s._run_complete_goal = lambda x: None
        s._run_acquire = lambda x: None
        s._run_launch = lambda: None
        s._run_stop = lambda: None
        return s

    def test_compile_returns_pending(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "compile", "action_arg": "test"})
        assert isinstance(r, ActionResult)
        assert r.pending is True

    def test_full_build_returns_pending(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "full_build", "action_arg": "booking system"})
        assert isinstance(r, ActionResult)
        assert r.pending is True
        assert "Full build" in r.message

    def test_build_returns_pending(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "build", "action_arg": "test"})
        assert isinstance(r, ActionResult)
        assert r.pending is True

    def test_search_returns_pending(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "search", "action_arg": "test"})
        assert isinstance(r, ActionResult)
        assert r.pending is True

    def test_status_returns_sync(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "status", "action_arg": ""})
        assert isinstance(r, ActionResult)
        assert r.pending is False

    def test_goal_returns_sync(self, screen):
        from mother.screens.chat import ActionResult
        r = screen._execute_action({"action": "goal", "action_arg": "test goal"})
        assert isinstance(r, ActionResult)
        assert r.pending is False

    def test_done_returns_none(self, screen):
        r = screen._execute_action({"action": "done", "action_arg": ""})
        assert r is None

    def test_no_action_returns_none(self, screen):
        r = screen._execute_action({"action": "", "action_arg": ""})
        assert r is None

    def test_terminal_returns_none(self, screen):
        r = screen._execute_action({"action": "launch", "action_arg": ""})
        assert r is None


# ===================================================================
# Gap C: PlanStore stale fallback in next_step
# ===================================================================

class TestPlanStoreStallFallback:
    """PlanStore.next_step returns stale in_progress steps."""

    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "plans.db"
        s = PlanStore(db)
        yield s
        s.close()

    def test_next_step_returns_pending_first(self, store):
        plan_id = store.create_plan(
            goal_id=1,
            blueprint_json="{}",
            trust_score=80.0,
            steps=[
                {"name": "s1", "action_type": "compile", "action_arg": "x"},
                {"name": "s2", "action_type": "build", "action_arg": "y"},
            ],
        )
        step = store.next_step(plan_id)
        assert step is not None
        assert step.name == "s1"
        assert step.status == "pending"

    def test_next_step_returns_stale_in_progress(self, store):
        plan_id = store.create_plan(
            goal_id=1,
            blueprint_json="{}",
            trust_score=80.0,
            steps=[{"name": "s1", "action_type": "compile", "action_arg": "x"}],
        )
        step = store.next_step(plan_id)
        # Mark in_progress with old timestamp
        store._conn.execute(
            "UPDATE plan_steps SET status = 'in_progress', started_at = ? WHERE step_id = ?",
            (time.time() - 600, step.step_id),  # 10 min ago
        )
        store._conn.commit()

        # No pending steps, but stale in_progress exists
        result = store.next_step(plan_id)
        assert result is not None
        assert result.status == "in_progress"
        assert result.name == "s1"

    def test_next_step_ignores_recent_in_progress(self, store):
        plan_id = store.create_plan(
            goal_id=1,
            blueprint_json="{}",
            trust_score=80.0,
            steps=[{"name": "s1", "action_type": "compile", "action_arg": "x"}],
        )
        step = store.next_step(plan_id)
        # Mark in_progress with recent timestamp
        store.update_step(step.step_id, "in_progress")

        # No pending, but in_progress is too recent (< 5 min)
        result = store.next_step(plan_id)
        assert result is None

    def test_next_step_none_when_all_done(self, store):
        plan_id = store.create_plan(
            goal_id=1,
            blueprint_json="{}",
            trust_score=80.0,
            steps=[{"name": "s1", "action_type": "compile", "action_arg": "x"}],
        )
        step = store.next_step(plan_id)
        store.update_step(step.step_id, "done", result_note="ok")

        result = store.next_step(plan_id)
        assert result is None


# ===================================================================
# Config fields
# ===================================================================

class TestConfigAccountability:
    """Config gains max_goal_stalls and max_goal_attempts."""

    def test_max_goal_stalls_default(self):
        c = MotherConfig()
        assert c.max_goal_stalls == 10

    def test_max_goal_attempts_default(self):
        c = MotherConfig()
        assert c.max_goal_attempts == 10

    def test_configurable(self):
        c = MotherConfig(max_goal_stalls=5, max_goal_attempts=10)
        assert c.max_goal_stalls == 5
        assert c.max_goal_attempts == 10
