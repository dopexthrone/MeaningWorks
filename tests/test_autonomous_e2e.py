"""E2E integration tests for the autonomous operating loop.

These tests exercise the full chain: goals → stance → tick → work,
with mocks only at the LLM boundary. Everything else is real SQLite,
real stance computation, real plan/step stores.
"""

import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from mother.goals import GoalStore, Goal, compute_goal_health
from mother.stance import Stance, StanceContext, compute_stance
from mother.executive import PlanStore
from mother.config import MotherConfig
from mother.screens.chat import ChatScreen


@pytest.fixture
def db_path(tmp_path):
    """Ephemeral SQLite DB for test isolation."""
    return tmp_path / "test_history.db"


class TestFreshGoalReachesActStance:
    """A fresh goal with health=1.0 and sufficient idle time produces ACT."""

    def test_fresh_goal_health_is_high(self, db_path):
        store = GoalStore(db_path)
        goal_id = store.add("Build a booking system", source="cli", priority="high")
        goal = store.get(goal_id)
        now = time.time()
        health = compute_goal_health(goal, now=now)
        assert health > 0.9
        store.close()

    def test_fresh_goal_triggers_act(self, db_path):
        store = GoalStore(db_path)
        goal_id = store.add("Build a booking system", source="cli", priority="high")
        goal = store.get(goal_id)
        now = time.time()
        health = compute_goal_health(goal, now=now)

        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=health,
            user_idle_seconds=400,
            conversation_active=False,
            autonomous_actions_this_session=0,
        )
        assert compute_stance(ctx) == Stance.ACT
        store.close()


class TestTickDispatchesWithFixedInit:
    """ChatScreen with fixed _last_user_message_time fires run_worker."""

    @pytest.fixture
    def screen(self, db_path):
        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = db_path
        s._store.get_history = MagicMock(return_value=[{"role": "user"}] * 3)
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 0
        s.run_worker = MagicMock()
        # Set mount time baseline (simulating on_mount fix)
        s._last_user_message_time = time.time() - 400
        s._session_start_time = time.time() - 500
        return s

    def test_tick_fires_with_goal(self, screen, db_path):
        store = GoalStore(db_path)
        store.add("Build something", source="cli", priority="high")
        store.close()

        screen._autonomous_tick()
        screen.run_worker.assert_called_once()

    def test_tick_silent_without_goal(self, screen, db_path):
        # No goals → SILENT → no worker
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()


class TestPlanCreationAndStepExecution:
    """PlanStore + GoalStore roundtrip: create, execute steps, verify."""

    def test_plan_steps_execute_to_completion(self, db_path):
        goal_store = GoalStore(db_path)
        goal_id = goal_store.add("Build booking system", source="cli", priority="high")

        plan_store = PlanStore(db_path)
        steps = [
            {"name": "scaffold", "description": "Create project", "action_type": "build"},
            {"name": "models", "description": "Define data models", "action_type": "build"},
            {"name": "verify", "description": "Run tests", "action_type": "reason"},
        ]
        plan_id = plan_store.create_plan(
            goal_id=goal_id,
            blueprint_json='{"type": "booking"}',
            trust_score=0.85,
            steps=steps,
        )

        # Execute all steps
        for _ in range(3):
            step = plan_store.next_step(plan_id)
            assert step is not None
            plan_store.update_step(step.step_id, "done", result_note="ok")

        plan_store.update_plan_progress(plan_id)

        # No more pending steps
        assert plan_store.next_step(plan_id) is None

        # Plan moved to "done" — get_plan_for_goal only returns active/executing,
        # so None here confirms the plan completed and left the active set
        assert plan_store.get_plan_for_goal(goal_id) is None

        # Verify via raw SQL that status is "done" and all 3 steps completed
        row = plan_store._conn.execute(
            "SELECT status, completed_steps FROM goal_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        assert row[0] == "done"
        assert row[1] == 3

        goal_store.close()
        plan_store.close()


class TestCLIGoalPersists:
    """Goal added via CLI (source='cli') is visible to active() query."""

    def test_cli_source_goal_is_active(self, db_path):
        store = GoalStore(db_path)
        goal_id = store.add("Build a CRM", source="cli", priority="normal")

        active = store.active()
        assert len(active) == 1
        assert active[0].source == "cli"
        assert active[0].goal_id == goal_id
        store.close()


class TestStaleGoalBlocked:
    """Stale goal (aged + redirected) decays below health threshold → SILENT."""

    def test_stale_redirected_goal_low_health(self, db_path):
        store = GoalStore(db_path)
        goal_id = store.add("Old project", source="user", priority="normal")
        # Simulate redirects (user kept changing topic)
        store.increment_redirect(goal_id)
        store.increment_redirect(goal_id)
        goal = store.get(goal_id)

        # 7+ days old AND redirected → health drops below 0.3
        future = time.time() + (7 * 86400)
        health = compute_goal_health(goal, now=future)
        assert health < 0.3

        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=health,
            user_idle_seconds=600,
            autonomous_actions_this_session=0,
        )
        assert compute_stance(ctx) == Stance.SILENT
        store.close()


class TestSessionCap:
    """5 autonomous actions → SILENT regardless of health/idle."""

    def test_five_actions_blocks(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.95,
            user_idle_seconds=600,
            autonomous_actions_this_session=5,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_four_actions_allows(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.95,
            user_idle_seconds=600,
            autonomous_actions_this_session=4,
        )
        assert compute_stance(ctx) == Stance.ACT
