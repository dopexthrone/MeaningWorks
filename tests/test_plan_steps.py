"""Tests for executable plan step extraction and direct execution.

Covers:
- _build_plan: standard compile → build → goal_done
- _multi_phase_plan: per-subsystem compile+build + verify
- extract_steps_from_blueprint: strategy selection
- Direct execution path in _execute_plan_step (no LLM)
"""

import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from mother.executive import (
    extract_steps_from_blueprint,
    _build_plan,
    _multi_phase_plan,
    PlanStore,
)


# --- _build_plan ---

class TestBuildPlan:
    def test_produces_three_steps(self):
        bp = {"components": [{"name": "X", "type": "service"}]}
        steps = _build_plan(bp, "build a thing")
        assert len(steps) == 3

    def test_step_types_are_compile_build_goal_done(self):
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "build a CRM")
        assert steps[0]["action_type"] == "compile"
        assert steps[1]["action_type"] == "build"
        assert steps[2]["action_type"] == "goal_done"

    def test_action_arg_is_goal_description(self):
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "build a booking system")
        assert steps[0]["action_arg"] == "build a booking system"
        assert steps[1]["action_arg"] == "build a booking system"
        assert steps[2]["action_arg"] == ""

    def test_fallback_to_blueprint_description(self):
        bp = {"description": "A web app", "components": [{"name": "X"}]}
        steps = _build_plan(bp, "")
        assert steps[0]["action_arg"] == "A web app"

    def test_fallback_to_project_when_no_description(self):
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "")
        assert steps[0]["action_arg"] == "project"

    def test_step_names(self):
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "build it")
        assert steps[0]["name"] == "compile"
        assert steps[1]["name"] == "build"
        assert steps[2]["name"] == "verify"

    def test_step_descriptions_contain_goal(self):
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "build a booking system")
        assert "booking system" in steps[0]["description"]
        assert "booking system" in steps[1]["description"]


# --- _multi_phase_plan ---

class TestMultiPhasePlan:
    def test_two_subsystems_produce_five_steps(self):
        subs = [
            {"name": "Auth", "description": "Auth module", "sub_blueprint": {}},
            {"name": "Billing", "description": "Billing module", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "build platform")
        assert len(steps) == 5

    def test_subsystem_compile_build_pairs(self):
        subs = [
            {"name": "Auth", "description": "Auth module", "sub_blueprint": {}},
            {"name": "Billing", "description": "Billing module", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "")
        assert steps[0]["action_type"] == "compile"
        assert steps[0]["name"] == "compile_Auth"
        assert steps[1]["action_type"] == "build"
        assert steps[1]["name"] == "build_Auth"
        assert steps[2]["action_type"] == "compile"
        assert steps[2]["name"] == "compile_Billing"
        assert steps[3]["action_type"] == "build"
        assert steps[3]["name"] == "build_Billing"

    def test_final_step_is_goal_done(self):
        subs = [
            {"name": "A", "description": "A", "sub_blueprint": {}},
            {"name": "B", "description": "B", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "")
        assert steps[-1]["action_type"] == "goal_done"

    def test_action_arg_is_subsystem_description(self):
        subs = [
            {"name": "Auth", "description": "Authentication service", "sub_blueprint": {}},
            {"name": "Pay", "description": "Payment gateway", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "")
        assert steps[0]["action_arg"] == "Authentication service"
        assert steps[2]["action_arg"] == "Payment gateway"

    def test_three_subsystems(self):
        subs = [
            {"name": f"S{i}", "description": f"Sub {i}", "sub_blueprint": {}}
            for i in range(3)
        ]
        steps = _multi_phase_plan(subs, "")
        # 3 × 2 + 1 = 7 steps
        assert len(steps) == 7
        assert steps[-1]["action_type"] == "goal_done"

    def test_fallback_name_when_missing(self):
        subs = [
            {"description": "First", "sub_blueprint": {}},
            {"description": "Second", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "")
        assert steps[0]["name"] == "compile_subsystem_0"
        assert steps[2]["name"] == "compile_subsystem_1"


# --- extract_steps_from_blueprint strategy selection ---

class TestStrategySelection:
    def test_no_subsystems_uses_build_plan(self):
        bp = {
            "components": [
                {"name": "A", "type": "service"},
                {"name": "B", "type": "entity"},
            ]
        }
        steps = extract_steps_from_blueprint(bp, "build it")
        assert len(steps) == 3
        assert steps[0]["action_type"] == "compile"

    def test_multiple_subsystems_uses_multi_phase(self):
        bp = {
            "components": [
                {"name": "Auth", "description": "Auth", "sub_blueprint": {"c": []}},
                {"name": "Pay", "description": "Pay", "sub_blueprint": {"c": []}},
            ]
        }
        steps = extract_steps_from_blueprint(bp, "build platform")
        assert len(steps) == 5  # 2×2 + 1

    def test_single_subsystem_uses_build_plan(self):
        bp = {
            "components": [
                {"name": "Auth", "description": "Auth", "sub_blueprint": {"c": []}},
                {"name": "Config", "type": "module"},
            ]
        }
        steps = extract_steps_from_blueprint(bp, "build auth")
        assert len(steps) == 3  # build plan

    def test_empty_components_returns_empty(self):
        assert extract_steps_from_blueprint({"components": []}) == []

    def test_no_components_key_returns_empty(self):
        assert extract_steps_from_blueprint({}) == []


# --- PlanStore integration with new step format ---

class TestPlanStoreWithNewSteps:
    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "test.db"
        s = PlanStore(db)
        yield s
        s.close()

    def test_build_plan_roundtrip(self, store):
        """Steps from _build_plan persist and retrieve correctly."""
        bp = {"components": [{"name": "X"}]}
        steps = _build_plan(bp, "build a booking system")
        plan_id = store.create_plan(
            goal_id=1, blueprint_json="{}", trust_score=80.0, steps=steps,
        )

        plan = store.get_plan_for_goal(1)
        assert plan.total_steps == 3
        assert plan.steps[0].action_type == "compile"
        assert plan.steps[0].action_arg == "build a booking system"
        assert plan.steps[1].action_type == "build"
        assert plan.steps[2].action_type == "goal_done"

    def test_multi_phase_roundtrip(self, store):
        """Steps from _multi_phase_plan persist correctly."""
        subs = [
            {"name": "A", "description": "Subsystem A", "sub_blueprint": {}},
            {"name": "B", "description": "Subsystem B", "sub_blueprint": {}},
        ]
        steps = _multi_phase_plan(subs, "build it")
        plan_id = store.create_plan(
            goal_id=2, blueprint_json="{}", trust_score=75.0, steps=steps,
        )

        plan = store.get_plan_for_goal(2)
        assert plan.total_steps == 5
        assert plan.steps[0].action_type == "compile"
        assert plan.steps[1].action_type == "build"
        assert plan.steps[4].action_type == "goal_done"

    def test_execute_build_plan_to_completion(self, store):
        """All 3 steps of a build plan execute to completion."""
        steps = _build_plan({"components": [{"name": "X"}]}, "test project")
        plan_id = store.create_plan(
            goal_id=3, blueprint_json="{}", trust_score=90.0, steps=steps,
        )

        for _ in range(3):
            step = store.next_step(plan_id)
            assert step is not None
            store.update_step(step.step_id, "done", result_note="ok")

        store.update_plan_progress(plan_id)
        assert store.next_step(plan_id) is None

        # Plan should be done
        row = store._conn.execute(
            "SELECT status, completed_steps FROM goal_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        assert row[0] == "done"
        assert row[1] == 3


# --- Direct execution path ---

class TestDirectExecution:
    """Verify that well-formed steps skip LLM and go to _execute_action."""

    def test_compile_step_skips_llm(self):
        """A compile step should call _execute_action directly, not stream_chat."""
        from mother.screens.chat import ChatScreen

        step_dict = {
            "step_id": 1, "position": 0, "name": "compile",
            "description": "Compile the project",
            "action_type": "compile", "action_arg": "build a booking system",
        }

        screen = ChatScreen.__new__(ChatScreen)
        screen._bridge = MagicMock()
        screen._bridge.get_next_plan_step = AsyncMock(return_value=step_dict)
        screen._bridge.update_plan_step = AsyncMock()
        screen._bridge.complete_goal = AsyncMock()
        screen._autonomous_actions_count = 0
        screen._working_memory_summary = ""

        # Mock _execute_action to return a pending ActionResult
        mock_ar = MagicMock()
        mock_ar.chain_text = "Compilation started"
        mock_ar.pending = True
        screen._execute_action = MagicMock(return_value=mock_ar)
        screen._route_output = MagicMock()

        plan = {"plan_id": 1, "total_steps": 3, "status": "executing"}
        goal = {"goal_id": 1, "description": "build a booking system"}

        import asyncio
        asyncio.run(screen._execute_plan_step(
            "/tmp/test.db", goal, plan, MagicMock(), 0.10,
        ))

        # _execute_action called with correct parsed dict
        screen._execute_action.assert_called_once_with({
            "action": "compile",
            "action_arg": "build a booking system",
        })

        # stream_chat NOT called (no LLM round-trip)
        screen._bridge.stream_chat.assert_not_called()

    def test_goal_done_step_completes_goal(self):
        """A goal_done step should complete the goal without LLM."""
        from mother.screens.chat import ChatScreen

        step_dict = {
            "step_id": 3, "position": 2, "name": "verify",
            "description": "Verify and complete",
            "action_type": "goal_done", "action_arg": "",
        }

        screen = ChatScreen.__new__(ChatScreen)
        screen._bridge = MagicMock()
        screen._bridge.get_next_plan_step = AsyncMock(return_value=step_dict)
        screen._bridge.update_plan_step = AsyncMock()
        screen._bridge.complete_goal = AsyncMock()
        screen._autonomous_actions_count = 0
        screen._working_memory_summary = "working"
        screen._route_output = MagicMock()

        plan = {"plan_id": 1, "total_steps": 3, "status": "executing"}
        goal = {"goal_id": 7, "description": "build something"}

        import asyncio
        asyncio.run(screen._execute_plan_step(
            "/tmp/test.db", goal, plan, MagicMock(), 0.10,
        ))

        # Goal completed
        screen._bridge.complete_goal.assert_called_once_with(
            "/tmp/test.db", 7, note="Plan completed",
        )
        # Step marked done
        screen._bridge.update_plan_step.assert_any_call(
            "/tmp/test.db", 3, "done", result_note="goal completed",
        )
        # Working memory cleared
        assert screen._working_memory_summary == ""
        # No LLM call
        screen._bridge.stream_chat.assert_not_called()

    def test_reason_step_uses_llm(self):
        """A 'reason' step should fall back to LLM streaming."""
        from mother.screens.chat import ChatScreen

        step_dict = {
            "step_id": 5, "position": 0, "name": "think",
            "description": "Reason about approach",
            "action_type": "reason", "action_arg": "",
        }

        screen = ChatScreen.__new__(ChatScreen)
        screen._bridge = MagicMock()
        screen._bridge.get_next_plan_step = AsyncMock(return_value=step_dict)
        screen._bridge.update_plan_step = AsyncMock()
        screen._bridge.begin_chat_stream = MagicMock()
        screen._bridge.stream_chat = AsyncMock()
        screen._bridge.get_stream_result = MagicMock(return_value="Some reasoning")
        screen._autonomous_actions_count = 0
        screen._working_memory_summary = ""
        screen._execute_action = MagicMock(return_value=None)
        screen._route_output = MagicMock()
        screen._update_senses = MagicMock()
        screen._current_senses = None
        screen._current_posture = None
        screen._config = MagicMock()
        screen._build_context_data = MagicMock(return_value=MagicMock())

        plan = {"plan_id": 2, "total_steps": 1, "status": "executing"}
        goal = {"goal_id": 3, "description": "figure out caching"}

        import asyncio
        with patch("mother.screens.chat.parse_response", return_value={
            "display": "I think we should...", "action": "done",
            "action_arg": "", "voice": "",
        }), patch("mother.screens.chat.synthesize_context", return_value="ctx"), \
             patch("mother.screens.chat.build_system_prompt", return_value="sys"):
            asyncio.run(screen._execute_plan_step(
                "/tmp/test.db", goal, plan, MagicMock(), 0.10,
            ))

        # LLM WAS called for reason steps
        screen._bridge.stream_chat.assert_called_once()
