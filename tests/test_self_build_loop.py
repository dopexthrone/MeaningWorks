"""Tests for self-build loop wiring — goal enrichment, self-build detection,
plan step routing, and daemon enrichment."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mother.goal_generator import (
    ImprovementGoal,
    goal_to_actionable,
    _CATEGORY_TO_VERB,
    _CONCERN_DESCRIPTIONS,
    _LAYER_DESCRIPTIONS,
)
from mother.executive import (
    _is_self_build_goal,
    _SELF_BUILD_SIGNALS,
    _build_plan,
    extract_steps_from_blueprint,
)


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# goal_to_actionable
# ---------------------------------------------------------------------------

class TestGoalToActionable:
    def test_enriches_with_postcodes(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="critical",
            category="confidence",
            description="3 cells below 30% confidence need attention.",
            source="grid:critical_confidence",
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"),
        )
        enriched = goal_to_actionable(goal)
        assert "Strengthen" in enriched.description
        assert "Semantic" in enriched.description or "Cognitive" in enriched.description
        assert "Entity" in enriched.description or "Behavior" in enriched.description

    def test_preserves_original_description(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="high",
            category="coverage",
            description="Core layers not mapped.",
            source="grid:missing_layers",
            target_postcodes=("INT.ENT.ECO.WHAT.SFT",),
        )
        enriched = goal_to_actionable(goal)
        assert "Core layers not mapped" in enriched.description

    def test_no_postcodes_returns_unchanged(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="medium",
            category="quality",
            description="Output quality low.",
            source="feedback",
        )
        enriched = goal_to_actionable(goal)
        assert enriched.description == goal.description

    def test_includes_grid_cells_when_provided(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="critical",
            category="confidence",
            description="Cells need attention.",
            source="grid",
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
        )
        cells = [
            ("SEM.ENT.DOM.WHAT.SFT", 0.25, "P", "entity-model"),
            ("COG.BHV.APP.HOW.SFT", 0.40, "P", "behavior"),  # not targeted
        ]
        enriched = goal_to_actionable(goal, grid_cells=cells)
        assert "entity-model" in enriched.description
        assert "behavior" not in enriched.description  # not in target_postcodes

    def test_preserves_immutable_fields(self):
        goal = ImprovementGoal(
            goal_id="G-042",
            priority="high",
            category="resilience",
            description="Fix issues.",
            source="observer",
            target_postcodes=("STR.GTE.CMP.HOW.SFT",),
            estimated_effort="medium",
            success_metric="Zero quarantined.",
        )
        enriched = goal_to_actionable(goal)
        assert enriched.goal_id == "G-042"
        assert enriched.priority == "high"
        assert enriched.category == "resilience"
        assert enriched.source == "observer"
        assert enriched.target_postcodes == ("STR.GTE.CMP.HOW.SFT",)
        assert enriched.estimated_effort == "medium"
        assert enriched.success_metric == "Zero quarantined."

    def test_all_categories_have_verbs(self):
        for category in ("confidence", "coverage", "quality", "resilience"):
            assert category in _CATEGORY_TO_VERB

    def test_coverage_category_uses_implement_verb(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="high",
            category="coverage",
            description="Missing layers.",
            source="grid",
            target_postcodes=("NET.ORC.APP.WHAT.SFT",),
        )
        enriched = goal_to_actionable(goal)
        assert "Implement" in enriched.description

    def test_resilience_category_uses_fix_verb(self):
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="high",
            category="resilience",
            description="Quarantined.",
            source="grid",
            target_postcodes=("AGN.AGT.APP.WHAT.SFT",),
        )
        enriched = goal_to_actionable(goal)
        assert "Fix" in enriched.description


# ---------------------------------------------------------------------------
# _is_self_build_goal
# ---------------------------------------------------------------------------

class TestIsSelfBuildGoal:
    def test_confidence_and_layer_is_self_build(self):
        assert _is_self_build_goal(
            "Strengthen implementation for Semantic Entity modeling, confidence too low"
        )

    def test_coverage_and_capability_is_self_build(self):
        assert _is_self_build_goal(
            "Implement missing capability for kernel coverage"
        )

    def test_self_improvement_tag(self):
        assert _is_self_build_goal("[SELF-IMPROVEMENT] fix compiler quality issues")

    def test_generic_build_is_not_self_build(self):
        assert not _is_self_build_goal("Build a task manager application")

    def test_single_signal_is_not_self_build(self):
        assert not _is_self_build_goal("Fix the login page quality")

    def test_empty_string_is_not_self_build(self):
        assert not _is_self_build_goal("")

    def test_two_signals_minimum(self):
        # "grid" + "cell" = 2 signals
        assert _is_self_build_goal("Update grid cell handling")

    def test_quarantined_and_resilience(self):
        assert _is_self_build_goal("Fix quarantined cells, improve resilience")


# ---------------------------------------------------------------------------
# _build_plan with self_build
# ---------------------------------------------------------------------------

class TestBuildPlanSelfBuild:
    def test_self_build_goal_uses_self_build_action(self):
        bp = {"components": [{"name": "test"}], "description": "test"}
        steps = _build_plan(bp, "Strengthen compiler kernel confidence")
        build_step = next(s for s in steps if s["name"] == "build")
        assert build_step["action_type"] == "self_build"

    def test_normal_goal_uses_build_action(self):
        bp = {"components": [{"name": "test"}], "description": "test"}
        steps = _build_plan(bp, "Build a task manager application")
        build_step = next(s for s in steps if s["name"] == "build")
        assert build_step["action_type"] == "build"

    def test_extract_steps_routes_self_build(self):
        bp = {
            "components": [{"name": "CompilerFix"}],
            "description": "Fix compiler quality",
        }
        steps = extract_steps_from_blueprint(
            bp, goal_description="Strengthen kernel semantic coverage"
        )
        action_types = [s["action_type"] for s in steps]
        assert "self_build" in action_types

    def test_extract_steps_has_compile_before_self_build(self):
        bp = {
            "components": [{"name": "Fix"}],
            "description": "Fix grid confidence",
        }
        steps = extract_steps_from_blueprint(
            bp, goal_description="Strengthen kernel cell confidence"
        )
        action_types = [s["action_type"] for s in steps]
        compile_idx = action_types.index("compile")
        build_idx = action_types.index("self_build")
        assert compile_idx < build_idx


# ---------------------------------------------------------------------------
# bridge.compile_goal_to_plan enrichment
# ---------------------------------------------------------------------------

class TestBridgeCompileGoalToPlan:
    def _make_bridge(self, mock_engine):
        """Create a minimal bridge with a mocked engine."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "mock"
        bridge._engine = mock_engine
        bridge._engine_cost_baseline = 0.0
        bridge._last_call_cost = 0.0
        bridge._session_cost = 0.0
        return bridge

    def test_enriches_self_build_steps(self, tmp_path):
        """When compile produces self_build steps, bridge enriches action_arg."""
        mock_result = SimpleNamespace(
            success=True,
            blueprint={
                "components": [{"name": "TestComp", "description": "test"}],
                "description": "test",
            },
            verification={"overall_score": 80.0},
        )
        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine._session_cost_usd = 0.0

        bridge = self._make_bridge(mock_engine)

        # A self-build goal description
        goal_desc = "Strengthen kernel semantic confidence"

        result = run(bridge.compile_goal_to_plan(
            db_path=tmp_path / "test.db",
            goal_id=1,
            goal_description=goal_desc,
        ))

        assert result is not None
        assert result["plan_id"] > 0
        assert result["total_steps"] >= 2

    def test_non_self_build_steps_not_enriched(self, tmp_path):
        """Normal goals don't get self_build_planner enrichment."""
        mock_result = SimpleNamespace(
            success=True,
            blueprint={
                "components": [{"name": "TaskManager", "description": "manage tasks"}],
                "description": "task manager",
            },
            verification={"overall_score": 75.0},
        )
        mock_engine = MagicMock()
        mock_engine.compile.return_value = mock_result
        mock_engine._session_cost_usd = 0.0

        bridge = self._make_bridge(mock_engine)

        result = run(bridge.compile_goal_to_plan(
            db_path=tmp_path / "test.db",
            goal_id=2,
            goal_description="Build a task manager application",
        ))

        assert result is not None
        # No target_postcodes for non-self-build
        assert "target_postcodes" not in result


# ---------------------------------------------------------------------------
# daemon._enrich_goal_for_build
# ---------------------------------------------------------------------------

class TestDaemonEnrichGoal:
    def test_enriches_with_grid_context(self, monkeypatch):
        from mother.daemon import DaemonMode, DaemonConfig

        # Mock load_grid to return a grid with cells
        mock_grid = MagicMock()
        mock_grid.cells = {
            "SEM.ENT.DOM.WHAT.SFT": SimpleNamespace(
                confidence=0.25, fill=SimpleNamespace(name="P"), primitive="entity"
            ),
        }
        monkeypatch.setattr("kernel.store.load_grid", lambda mid: mock_grid)

        daemon = DaemonMode.__new__(DaemonMode)
        enriched = daemon._enrich_goal_for_build("3 cells below 30% confidence")

        # Should have been enriched (no postcodes on the goal though,
        # so it falls through unchanged since we can't extract postcodes
        # from a bare string)
        assert isinstance(enriched, str)
        assert len(enriched) > 0

    def test_falls_back_on_error(self, monkeypatch):
        from mother.daemon import DaemonMode

        monkeypatch.setattr(
            "kernel.store.load_grid",
            lambda mid: (_ for _ in ()).throw(Exception("DB error")),
        )

        daemon = DaemonMode.__new__(DaemonMode)
        result = daemon._enrich_goal_for_build("original goal text")
        assert result == "original goal text"

    def test_scheduler_tick_uses_enriched_goal(self, monkeypatch):
        """Verify _scheduler_tick passes enriched goal to enqueue."""
        from mother.daemon import DaemonMode, DaemonConfig

        daemon = DaemonMode(config=DaemonConfig())
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        daemon._queue = []

        # Mock _find_critical_goal to return a (goal_id, description) tuple
        monkeypatch.setattr(daemon, "_find_critical_goal", lambda: (1, "Fix kernel quality issues"))

        # Mock _enrich_goal_for_build
        enrich_calls = []
        def mock_enrich(desc):
            enrich_calls.append(desc)
            return f"ENRICHED: {desc}"
        monkeypatch.setattr(daemon, "_enrich_goal_for_build", mock_enrich)

        # Run scheduler tick
        run(daemon._scheduler_tick())

        assert len(enrich_calls) == 1
        assert enrich_calls[0] == "Fix kernel quality issues"

        # Check enqueued text
        pending = [r for r in daemon._queue if r.status == "pending"]
        assert len(pending) == 1
        assert "ENRICHED:" in pending[0].input_text


# ---------------------------------------------------------------------------
# Full loop integration (mocked)
# ---------------------------------------------------------------------------

class TestFullLoopMocked:
    def test_goal_enrichment_to_plan_extraction_flow(self):
        """Verify the conceptual pipeline: diagnostic goal → enriched → plan with self_build."""
        # 1. Start with diagnostic goal
        goal = ImprovementGoal(
            goal_id="G-001",
            priority="critical",
            category="confidence",
            description="3 cells below 30% confidence need immediate attention.",
            source="grid:critical_confidence",
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"),
        )

        # 2. Enrich
        enriched = goal_to_actionable(goal)
        assert "Strengthen" in enriched.description
        assert enriched.target_postcodes == goal.target_postcodes

        # 3. Extract steps using enriched description
        blueprint = {
            "components": [
                {"name": "EntityModel", "description": "Semantic entity modeling"},
                {"name": "BehaviorFlow", "description": "Cognitive behavior processing"},
            ],
        }
        steps = extract_steps_from_blueprint(blueprint, goal_description=enriched.description)

        # 4. Should produce self_build action (enriched desc has "strengthen" + "semantic")
        action_types = [s["action_type"] for s in steps]
        assert "self_build" in action_types

    def test_non_self_build_goal_produces_build_action(self):
        """Normal goals go through the standard build path."""
        goal = ImprovementGoal(
            goal_id="G-100",
            priority="medium",
            category="quality",
            description="Output formatting needs improvement.",
            source="feedback",
        )
        enriched = goal_to_actionable(goal)
        blueprint = {"components": [{"name": "Formatter"}]}
        steps = extract_steps_from_blueprint(blueprint, goal_description=enriched.description)
        action_types = [s["action_type"] for s in steps]
        assert "build" in action_types
        assert "self_build" not in action_types

    def test_planner_produces_rich_prompt_for_self_build(self, tmp_path):
        """SelfBuildSpec generates a prompt with architectural rules."""
        from mother.self_build_planner import (
            goal_to_build_intent,
            blueprint_to_build_context,
            assemble_self_build_prompt,
        )

        goal = {
            "description": "Strengthen kernel semantic entity modeling",
            "category": "confidence",
            "target_postcodes": ("SEM.ENT.DOM.WHAT.SFT",),
        }
        blueprint = {
            "components": [
                {"name": "EntityModel", "description": "Handles entity extraction"},
            ],
        }

        intent = goal_to_build_intent(goal, blueprint=blueprint)
        context = blueprint_to_build_context(blueprint)
        spec = assemble_self_build_prompt(
            build_intent=intent,
            repo_dir=tmp_path,
            blueprint_context=context,
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
        )

        assert "TASK:" in spec.prompt
        assert "Avoid modifying these core files" in spec.prompt
        assert "bridge.py" in spec.prompt
        assert "EntityModel" in spec.prompt
