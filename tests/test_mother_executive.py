"""Tests for mother/executive.py — goal execution plans."""

import json
import time

import pytest

from mother.executive import (
    PlanStep,
    GoalPlan,
    PlanStore,
    classify_goal,
    extract_steps_from_blueprint,
    _topological_order,
)


# --- classify_goal ---

class TestClassifyGoal:
    def test_compilable_build_app(self):
        assert classify_goal("Build a booking system") == "compilable"

    def test_compilable_create_api(self):
        assert classify_goal("Create an API for user management") == "compilable"

    def test_compilable_implement_service(self):
        assert classify_goal("Implement the authentication service") == "compilable"

    def test_compilable_deploy_dashboard(self):
        assert classify_goal("Deploy the analytics dashboard") == "compilable"

    def test_compilable_setup_pipeline(self):
        assert classify_goal("Set up a CI/CD pipeline") == "compilable"

    def test_compilable_add_feature(self):
        assert classify_goal("Add a notification feature") == "compilable"

    def test_conversational_improve_recall(self):
        assert classify_goal("Improve my recall accuracy") == "conversational"

    def test_conversational_think_about(self):
        assert classify_goal("Think about the meaning of life") == "conversational"

    def test_conversational_no_nouns(self):
        assert classify_goal("Build something amazing") == "conversational"

    def test_conversational_no_verbs(self):
        assert classify_goal("The api is broken") == "conversational"

    def test_conversational_empty(self):
        assert classify_goal("") == "conversational"

    def test_compilable_case_insensitive(self):
        assert classify_goal("BUILD A WEB APP") == "compilable"

    def test_compilable_write_script(self):
        assert classify_goal("Write a migration script") == "compilable"


# --- extract_steps_from_blueprint ---

class TestExtractSteps:
    def test_empty_blueprint(self):
        assert extract_steps_from_blueprint({}) == []

    def test_empty_components(self):
        assert extract_steps_from_blueprint({"components": []}) == []

    def test_build_plan_from_single_component(self):
        """Any blueprint without subsystems produces a 3-step build plan."""
        bp = {
            "components": [
                {"name": "AuthService", "type": "service", "description": "Handles auth"}
            ]
        }
        steps = extract_steps_from_blueprint(bp, goal_description="build auth service")
        assert len(steps) == 3
        assert steps[0]["action_type"] == "compile"
        assert steps[0]["action_arg"] == "build auth service"
        assert steps[1]["action_type"] == "build"
        assert steps[1]["action_arg"] == "build auth service"
        assert steps[2]["action_type"] == "goal_done"

    def test_build_plan_from_multi_component(self):
        """Multiple components without sub_blueprint still produce a 3-step plan."""
        bp = {
            "components": [
                {"name": "UserModel", "type": "entity", "description": "User data"},
                {"name": "ApiService", "type": "service", "description": "API layer"},
                {"name": "Dashboard", "type": "interface", "description": "UI"},
            ]
        }
        steps = extract_steps_from_blueprint(bp, goal_description="build a CRM")
        assert len(steps) == 3
        assert steps[0]["action_type"] == "compile"
        assert steps[1]["action_type"] == "build"
        assert steps[2]["action_type"] == "goal_done"

    def test_goal_description_used_as_action_arg(self):
        bp = {"components": [{"name": "Foo", "type": "service", "description": ""}]}
        steps = extract_steps_from_blueprint(bp, goal_description="build a booking system")
        assert steps[0]["action_arg"] == "build a booking system"
        assert steps[1]["action_arg"] == "build a booking system"

    def test_fallback_to_blueprint_description_when_no_goal(self):
        bp = {
            "description": "A booking system",
            "components": [{"name": "X", "type": "service", "description": ""}],
        }
        steps = extract_steps_from_blueprint(bp)
        assert steps[0]["action_arg"] == "A booking system"

    def test_multi_phase_plan_for_subsystems(self):
        """Components with sub_blueprint produce per-subsystem compile+build."""
        bp = {
            "components": [
                {
                    "name": "Auth",
                    "description": "Authentication subsystem",
                    "sub_blueprint": {"components": [{"name": "Login"}]},
                },
                {
                    "name": "Billing",
                    "description": "Billing subsystem",
                    "sub_blueprint": {"components": [{"name": "Invoice"}]},
                },
            ]
        }
        steps = extract_steps_from_blueprint(bp, goal_description="build platform")
        # 2 subsystems × 2 steps + 1 verify = 5 steps
        assert len(steps) == 5
        assert steps[0]["action_type"] == "compile"
        assert "Authentication subsystem" in steps[0]["action_arg"]
        assert steps[1]["action_type"] == "build"
        assert steps[2]["action_type"] == "compile"
        assert "Billing subsystem" in steps[2]["action_arg"]
        assert steps[3]["action_type"] == "build"
        assert steps[4]["action_type"] == "goal_done"

    def test_single_subsystem_uses_build_plan(self):
        """Only one sub_blueprint component → standard build plan, not multi-phase."""
        bp = {
            "components": [
                {
                    "name": "Auth",
                    "description": "Auth subsystem",
                    "sub_blueprint": {"components": [{"name": "Login"}]},
                },
                {"name": "Config", "type": "module", "description": "Config"},
            ]
        }
        steps = extract_steps_from_blueprint(bp, goal_description="build auth")
        assert len(steps) == 3  # build plan, not multi-phase


# --- _topological_order ---

class TestTopologicalOrder:
    def test_empty_relationships(self):
        comps = [{"name": "A"}, {"name": "B"}]
        result = _topological_order(comps, [])
        assert len(result) == 2

    def test_empty_components(self):
        result = _topological_order([], [{"source": "A", "target": "B"}])
        assert result == []

    def test_simple_chain(self):
        comps = [{"name": "C"}, {"name": "B"}, {"name": "A"}]
        rels = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ]
        result = _topological_order(comps, rels)
        names = [c["name"] for c in result]
        assert names.index("A") < names.index("B")
        assert names.index("B") < names.index("C")

    def test_cycle_falls_back(self):
        comps = [{"name": "A"}, {"name": "B"}]
        rels = [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "A"},
        ]
        result = _topological_order(comps, rels)
        # Falls back to original order
        assert len(result) == 2
        assert result[0]["name"] == "A"
        assert result[1]["name"] == "B"

    def test_diamond_dependency(self):
        comps = [{"name": "D"}, {"name": "B"}, {"name": "C"}, {"name": "A"}]
        rels = [
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "B", "target": "D"},
            {"source": "C", "target": "D"},
        ]
        result = _topological_order(comps, rels)
        names = [c["name"] for c in result]
        assert names[0] == "A"
        assert names[-1] == "D"

    def test_from_to_keys(self):
        """Supports 'from'/'to' as alternative keys."""
        comps = [{"name": "X"}, {"name": "Y"}]
        rels = [{"from": "X", "to": "Y"}]
        result = _topological_order(comps, rels)
        names = [c["name"] for c in result]
        assert names.index("X") < names.index("Y")


# --- PlanStore CRUD ---

class TestPlanStore:
    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "test_plans.db"
        s = PlanStore(db)
        yield s
        s.close()

    def test_create_plan(self, store):
        steps = [
            {"name": "Step1", "description": "Do thing", "action_type": "build", "action_arg": "Step1"},
            {"name": "Step2", "description": "Do other", "action_type": "compile", "action_arg": "Step2"},
        ]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{"test": true}', trust_score=75.0, steps=steps)
        assert plan_id > 0

    def test_get_plan_for_goal(self, store):
        steps = [{"name": "S1", "description": "", "action_type": "build", "action_arg": "S1"}]
        store.create_plan(goal_id=42, blueprint_json='{}', trust_score=80.0, steps=steps)

        plan = store.get_plan_for_goal(42)
        assert plan is not None
        assert plan.goal_id == 42
        assert plan.trust_score == 80.0
        assert plan.status == "active"
        assert plan.total_steps == 1
        assert len(plan.steps) == 1

    def test_get_plan_for_goal_none(self, store):
        assert store.get_plan_for_goal(999) is None

    def test_next_step(self, store):
        steps = [
            {"name": "A", "description": "", "action_type": "build", "action_arg": "A"},
            {"name": "B", "description": "", "action_type": "compile", "action_arg": "B"},
        ]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=70.0, steps=steps)

        step = store.next_step(plan_id)
        assert step is not None
        assert step.name == "A"
        assert step.position == 0
        assert step.status == "pending"

    def test_update_step_in_progress(self, store):
        steps = [{"name": "X", "description": "", "action_type": "reason", "action_arg": "X"}]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=50.0, steps=steps)

        step = store.next_step(plan_id)
        store.update_step(step.step_id, "in_progress")

        # Next step should return None (no pending)
        assert store.next_step(plan_id) is None

    def test_update_step_done(self, store):
        steps = [
            {"name": "A", "description": "", "action_type": "build", "action_arg": "A"},
            {"name": "B", "description": "", "action_type": "build", "action_arg": "B"},
        ]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=60.0, steps=steps)

        step = store.next_step(plan_id)
        store.update_step(step.step_id, "done", result_note="Built successfully")

        next_step = store.next_step(plan_id)
        assert next_step is not None
        assert next_step.name == "B"

    def test_plan_progress_tracking(self, store):
        steps = [
            {"name": "A", "description": "", "action_type": "build", "action_arg": "A"},
            {"name": "B", "description": "", "action_type": "build", "action_arg": "B"},
        ]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=60.0, steps=steps)

        step = store.next_step(plan_id)
        store.update_step(step.step_id, "done")
        store.update_plan_progress(plan_id)

        plan = store.get_plan_for_goal(1)
        assert plan.completed_steps == 1
        assert plan.status == "executing"

    def test_plan_done_when_all_steps_done(self, store):
        steps = [{"name": "Only", "description": "", "action_type": "build", "action_arg": "Only"}]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=90.0, steps=steps)

        step = store.next_step(plan_id)
        store.update_step(step.step_id, "done")
        store.update_plan_progress(plan_id)

        plan = store.get_plan_for_goal(1)
        # Plan is done — get_plan_for_goal only returns active/executing
        assert plan is None

    def test_abandon_plan(self, store):
        steps = [{"name": "X", "description": "", "action_type": "build", "action_arg": "X"}]
        plan_id = store.create_plan(goal_id=1, blueprint_json='{}', trust_score=50.0, steps=steps)

        store.abandon_plan(plan_id)

        # Should not find abandoned plan
        assert store.get_plan_for_goal(1) is None

    def test_blueprint_json_preserved(self, store):
        bp = json.dumps({"name": "Test", "components": [{"name": "A"}]})
        plan_id = store.create_plan(goal_id=1, blueprint_json=bp, trust_score=75.0, steps=[])

        # Direct query to verify
        row = store._conn.execute(
            "SELECT blueprint_json FROM goal_plans WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        assert json.loads(row[0]) == {"name": "Test", "components": [{"name": "A"}]}

    def test_multiple_plans_returns_latest(self, store):
        steps = [{"name": "V1", "description": "", "action_type": "build", "action_arg": "V1"}]
        store.create_plan(goal_id=5, blueprint_json='{"v": 1}', trust_score=60.0, steps=steps)

        # Abandon first plan
        plan1 = store.get_plan_for_goal(5)
        store.abandon_plan(plan1.plan_id)

        # Create new plan
        steps2 = [{"name": "V2", "description": "", "action_type": "build", "action_arg": "V2"}]
        store.create_plan(goal_id=5, blueprint_json='{"v": 2}', trust_score=80.0, steps=steps2)

        plan = store.get_plan_for_goal(5)
        assert plan is not None
        assert plan.trust_score == 80.0
        assert plan.steps[0].name == "V2"


# --- Dataclass frozen ---

class TestDataclasses:
    def test_plan_step_frozen(self):
        step = PlanStep(name="test")
        with pytest.raises(AttributeError):
            step.name = "changed"

    def test_goal_plan_frozen(self):
        plan = GoalPlan(goal_id=1)
        with pytest.raises(AttributeError):
            plan.goal_id = 2

    def test_plan_step_defaults(self):
        step = PlanStep()
        assert step.status == "pending"
        assert step.action_type == ""
        assert step.started_at == 0.0

    def test_goal_plan_defaults(self):
        plan = GoalPlan()
        assert plan.status == "active"
        assert plan.trust_score == 0.0
        assert plan.steps == []
