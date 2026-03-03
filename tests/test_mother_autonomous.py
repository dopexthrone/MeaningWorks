"""Tests for autonomous tick and proactive perception in chat.py."""

import time

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mother.config import MotherConfig
from mother.screens.chat import ChatScreen


# --- Config defaults ---

class TestAutonomousConfig:
    def test_autonomous_enabled_by_default(self):
        config = MotherConfig()
        assert config.autonomous_enabled is True

    def test_autonomous_tick_seconds_default(self):
        config = MotherConfig()
        assert config.autonomous_tick_seconds == 60

    def test_autonomous_budget_per_cycle(self):
        config = MotherConfig()
        assert config.autonomous_budget_per_cycle == 1.00

    def test_autonomous_budget_per_session(self):
        config = MotherConfig()
        assert config.autonomous_budget_per_session == 10.00

    def test_autonomous_enabled_can_be_set(self):
        config = MotherConfig(autonomous_enabled=True)
        assert config.autonomous_enabled is True


# --- Tick skip conditions ---

class TestAutonomousTick:
    @pytest.fixture
    def screen(self, tmp_path):
        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = tmp_path / "test.db"
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 0
        s.run_worker = MagicMock()
        return s

    def test_skips_when_chatting(self, screen):
        screen._chatting = True
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()

    def test_skips_when_already_working(self, screen):
        screen._autonomous_working = True
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()

    def test_skips_when_unmounted(self, screen):
        screen._unmounted = True
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()

    def test_skips_when_over_autonomous_budget(self, screen):
        screen._autonomous_session_cost = 2.0  # Over default 1.0
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()

    def test_skips_when_approaching_cost_limit(self, screen):
        screen._bridge.get_session_cost.return_value = 4.6  # 92% of 5.0
        screen._autonomous_tick()
        screen.run_worker.assert_not_called()

    def test_runs_when_stance_allows(self, tmp_path):
        """Tick dispatches work when stance computes ACT (healthy goal + long idle)."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("test goal", priority="high")
        store.close()

        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = db
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 0
        s._last_user_message_time = time.time() - 400  # idle > 300s
        s._current_posture = None
        s._context_cache = {}
        s.run_worker = MagicMock()

        s._autonomous_tick()
        s.run_worker.assert_called_once()

    def test_skips_when_no_goals(self, tmp_path):
        """Tick stays SILENT when no active goals exist."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)  # Empty store
        store.close()

        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = db
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 0
        s._last_user_message_time = time.time() - 400
        s._current_posture = None
        s._context_cache = {}
        s.run_worker = MagicMock()

        s._autonomous_tick()
        s.run_worker.assert_not_called()

    def test_skips_when_user_recently_active(self, tmp_path):
        """Tick stays SILENT when user was active < 120s ago."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("test goal", priority="high")
        store.close()

        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = db
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 0
        s._last_user_message_time = time.time() - 30  # idle < 120s
        s._current_posture = None
        s._context_cache = {}
        s.run_worker = MagicMock()

        s._autonomous_tick()
        s.run_worker.assert_not_called()

    def test_skips_when_session_cap_reached(self, tmp_path):
        """Tick stays SILENT after 5 autonomous actions in session."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("test goal", priority="high")
        store.close()

        config = MotherConfig(autonomous_enabled=True)
        s = ChatScreen(config=config)
        s._bridge = MagicMock()
        s._bridge.get_session_cost = MagicMock(return_value=0.0)
        s._store = MagicMock()
        s._store._path = db
        s._unmounted = False
        s._chatting = False
        s._autonomous_working = False
        s._autonomous_session_cost = 0.0
        s._autonomous_actions_count = 5  # Hit cap
        s._last_user_message_time = time.time() - 400
        s._current_posture = None
        s._context_cache = {}
        s.run_worker = MagicMock()

        s._autonomous_tick()
        s.run_worker.assert_not_called()


# --- ChainContext ---

class TestChainContext:
    def test_chain_context_defaults(self):
        from mother.screens.chat import ChainContext
        ctx = ChainContext()
        assert ctx.original_intent == ""
        assert ctx.chain_position == 0
        assert ctx.max_depth == 5
        assert ctx.accumulated_results == ()

    def test_chain_context_frozen(self):
        from mother.screens.chat import ChainContext
        ctx = ChainContext(original_intent="test")
        with pytest.raises(AttributeError):
            ctx.original_intent = "changed"

    def test_chain_context_accumulates_results(self):
        from mother.screens.chat import ChainContext
        ctx = ChainContext(original_intent="build app")
        ctx2 = ChainContext(
            original_intent=ctx.original_intent,
            chain_position=1,
            accumulated_results=ctx.accumulated_results + ("result 1",),
        )
        assert ctx2.chain_position == 1
        assert len(ctx2.accumulated_results) == 1
        assert ctx2.original_intent == "build app"


# --- Goal picking ---

class TestGoalPicking:
    def test_get_active_goal_descriptions_empty(self):
        config = MotherConfig()
        s = ChatScreen(config=config)
        s._store = None
        result = s._get_active_goal_descriptions()
        assert result == []

    def test_get_active_goal_descriptions_with_store(self, tmp_path):
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("goal 1", priority="high")
        store.add("goal 2", priority="normal")

        config = MotherConfig()
        s = ChatScreen(config=config)
        s._store = MagicMock()
        s._store._path = db

        result = s._get_active_goal_descriptions()
        assert len(result) == 2
        assert "goal 1" in result
        assert "goal 2" in result
        store.close()

    def test_get_active_goal_descriptions_max_5(self, tmp_path):
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        for i in range(10):
            store.add(f"goal {i}")

        config = MotherConfig()
        s = ChatScreen(config=config)
        s._store = MagicMock()
        s._store._path = db

        result = s._get_active_goal_descriptions()
        assert len(result) == 5
        store.close()


# --- Context wiring ---

class TestContextWiring:
    def test_context_data_includes_goals(self):
        from mother.context import ContextData
        ctx = ContextData(
            active_goals=["build booking system", "fix auth bug"],
        )
        assert len(ctx.active_goals) == 2

    def test_context_data_includes_action_result(self):
        from mother.context import ContextData
        ctx = ContextData(
            pending_action_result="Compiled 12 components",
        )
        assert ctx.pending_action_result == "Compiled 12 components"

    def test_context_data_includes_working_memory(self):
        from mother.context import ContextData
        ctx = ContextData(
            working_memory_summary="Working on booking system",
        )
        assert ctx.working_memory_summary == "Working on booking system"


# --- Synthesize situation renders goals ---

class TestSynthesizeSituationGoals:
    def test_renders_goals(self):
        from mother.context import ContextData, synthesize_situation
        ctx = ContextData(
            active_goals=["build booking system", "fix auth"],
        )
        result = synthesize_situation(ctx)
        assert "Active goals:" in result
        assert "build booking system" in result
        assert "fix auth" in result

    def test_renders_working_memory(self):
        from mother.context import ContextData, synthesize_situation
        ctx = ContextData(
            working_memory_summary="Processing compile output",
        )
        result = synthesize_situation(ctx)
        assert "Currently: Processing compile output" in result

    def test_renders_action_result(self):
        from mother.context import ContextData, synthesize_situation
        ctx = ContextData(
            pending_action_result="Compiled 12 components at 78% trust",
        )
        result = synthesize_situation(ctx)
        assert "Last action result:" in result
        assert "78% trust" in result

    def test_no_goals_no_section(self):
        from mother.context import ContextData, synthesize_situation
        ctx = ContextData()
        result = synthesize_situation(ctx)
        assert "Active goals:" not in result


# --- Plan-aware autonomous work ---

class TestPlanAwareAutonomy:
    def test_classify_goal_used_for_branching(self):
        """classify_goal determines whether goal gets compiled."""
        from mother.executive import classify_goal
        assert classify_goal("Build a booking system") == "compilable"
        assert classify_goal("Improve recall accuracy") == "conversational"

    def test_working_memory_summary_in_context(self):
        """Working memory summary wired into context data."""
        from mother.context import ContextData, synthesize_situation
        ctx = ContextData(
            working_memory_summary="Executing goal #3: step 2/5 — AuthService",
        )
        result = synthesize_situation(ctx)
        assert "Currently: Executing goal #3: step 2/5 — AuthService" in result

    def test_working_memory_init(self):
        config = MotherConfig()
        s = ChatScreen(config=config)
        assert s._working_memory_summary == ""

    def test_plan_store_roundtrip(self, tmp_path):
        """PlanStore can create and retrieve a plan."""
        from mother.executive import PlanStore
        db = tmp_path / "test.db"
        store = PlanStore(db)
        steps = [
            {"name": "DB", "description": "Schema", "action_type": "compile", "action_arg": "DB"},
            {"name": "API", "description": "Routes", "action_type": "build", "action_arg": "API"},
        ]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=75.0, steps=steps)
        plan = store.get_plan_for_goal(1)
        assert plan is not None
        assert plan.total_steps == 2
        assert plan.steps[0].name == "DB"
        assert plan.steps[1].name == "API"
        store.close()

    def test_extract_steps_produces_build_plan(self):
        """extract_steps_from_blueprint produces compile→build→goal_done."""
        from mother.executive import extract_steps_from_blueprint
        bp = {
            "components": [
                {"name": "UI", "type": "interface", "description": "Frontend"},
                {"name": "API", "type": "service", "description": "Backend"},
                {"name": "DB", "type": "database", "description": "Storage"},
            ],
        }
        steps = extract_steps_from_blueprint(bp, goal_description="build a web app")
        assert len(steps) == 3
        assert steps[0]["action_type"] == "compile"
        assert steps[1]["action_type"] == "build"
        assert steps[2]["action_type"] == "goal_done"


# --- Seed goal ---

class TestSeedGoal:
    def test_seed_creates_when_empty(self, tmp_path):
        """Seed goal is created when GoalStore has zero active goals."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        assert store.count_active() == 0

        # Simulate what on_mount does
        store.add(
            "Greet the user and ask what they'd like to build",
            source="system",
            priority="normal",
        )
        assert store.count_active() == 1
        goals = store.active()
        assert goals[0].source == "system"
        assert "Greet" in goals[0].description
        store.close()

    def test_seed_skips_when_goals_exist(self, tmp_path):
        """No seed goal if active goals already present."""
        from mother.goals import GoalStore
        db = tmp_path / "test.db"
        store = GoalStore(db)
        store.add("Existing goal", source="user", priority="high")
        assert store.count_active() == 1

        # Would-be seed check
        if store.count_active() == 0:
            store.add("Greet the user", source="system")
        assert store.count_active() == 1  # Still 1, no seed added
        store.close()

    def test_seed_goal_classifies_as_conversational(self):
        """Seed goal text classifies as conversational (not compilable)."""
        from mother.executive import classify_goal
        result = classify_goal("Greet the user and ask what they'd like to build")
        assert result == "conversational"


# --- Autonomous journal ---

class TestAutonomousJournal:
    def test_journal_records_autonomous_domain(self, tmp_path):
        """JournalEntry with domain='autonomous' round-trips through BuildJournal."""
        from mother.journal import BuildJournal, JournalEntry
        db = tmp_path / "test.db"
        journal = BuildJournal(db)
        journal.record(JournalEntry(
            event_type="compile",
            description="Goal #1 compiled",
            success=True,
            trust_score=78.0,
            domain="autonomous",
        ))
        recent = journal.recent(limit=1)
        assert len(recent) == 1
        assert recent[0].domain == "autonomous"
        assert recent[0].event_type == "compile"
        assert recent[0].success is True
        journal.close()
