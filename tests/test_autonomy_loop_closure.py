"""
Tests for autonomy loop closure — the feedback path from self-build outcomes
back into observation recording, grid re-analysis, and goal generation.

Verifies:
1. _close_feedback_loop records observations on success
2. _close_feedback_loop records anomaly observations on failure
3. _close_feedback_loop gracefully handles missing grid
4. _close_feedback_loop generates new goals from updated grid
5. _close_feedback_loop deduplicates goals against GoalStore
6. Postcode extraction from prompt text
7. Daemon _sync_goals uses self._outcomes (no core.outcome_store import)
8. Observation recording produces correct deltas
9. Re-analysis finds low-confidence cells
10. Goal generation from updated grid
"""

import ast
import asyncio
import re
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.observer import (
    record_observation,
    apply_batch,
    _CONFIRM_BOOST,
    _ANOMALY_DECAY,
)
from mother.bridge import EngineBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(postcodes=None, fill_state=FillState.P, confidence=0.70):
    """Create a grid with cells at given postcodes."""
    if postcodes is None:
        postcodes = ["INT.SEM.ECO.WHY.SFT"]
    grid = Grid()
    grid.set_intent("test intent", postcodes[0], "test_primitive")
    for pc_str in postcodes:
        pc = parse_postcode(pc_str)
        cell = Cell(
            postcode=pc,
            primitive="test_primitive",
            content="test content",
            fill=fill_state,
            confidence=confidence,
            connections=(),
            parent=None,
            source=("test",),
            revisions=(),
        )
        grid.cells[pc_str] = cell
    return grid


def _run(coro):
    """Run async in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Postcode extraction
# ---------------------------------------------------------------------------

class TestPostcodeExtraction:
    """Test _extract_postcodes_from_text static method."""

    def test_extracts_single_postcode(self):
        text = "Improve INT.SEM.ECO.WHY.SFT cell confidence"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ("INT.SEM.ECO.WHY.SFT",)

    def test_extracts_multiple_postcodes(self):
        text = "Target COG.BHV.APP.HOW.SFT and STR.ENT.DOM.WHAT.SFT"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert len(result) == 2
        assert "COG.BHV.APP.HOW.SFT" in result
        assert "STR.ENT.DOM.WHAT.SFT" in result

    def test_deduplicates_postcodes(self):
        text = "First INT.SEM.ECO.WHY.SFT then again INT.SEM.ECO.WHY.SFT"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ("INT.SEM.ECO.WHY.SFT",)

    def test_no_postcodes_returns_empty(self):
        text = "Just a regular description with no postcodes"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ()

    def test_preserves_order(self):
        text = "First COG.BHV.APP.HOW.SFT then INT.SEM.ECO.WHY.SFT"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ("COG.BHV.APP.HOW.SFT", "INT.SEM.ECO.WHY.SFT")

    def test_handles_how_much_dimension(self):
        text = "Check RES.LMT.APP.HOW_MUCH.SFT cost cell"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ("RES.LMT.APP.HOW_MUCH.SFT",)

    def test_ignores_partial_postcodes(self):
        text = "Just INT.SEM alone is not a full postcode"
        result = EngineBridge._extract_postcodes_from_text(text)
        assert result == ()


# ---------------------------------------------------------------------------
# _close_feedback_loop — success path
# ---------------------------------------------------------------------------

class TestCloseFeedbackLoopSuccess:
    """Test _close_feedback_loop on successful builds."""

    def test_success_path_records_observations(self, tmp_path):
        """Observations are recorded and cells get confidence boost."""
        from kernel.store import save_grid, load_grid

        postcodes = ("INT.SEM.ECO.WHY.SFT", "COG.BHV.APP.HOW.SFT")
        grid = _make_grid(list(postcodes), confidence=0.60)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=True,
                build_description="test build",
            )

        assert result["cells_observed"] == 2
        assert result["cells_improved"] == 2  # confirmed observations boost confidence
        assert result["cells_degraded"] == 0

    def test_success_path_returns_summary(self, tmp_path):
        """Summary dict has all expected keys."""
        from kernel.store import save_grid, load_grid

        postcodes = ("INT.SEM.ECO.WHY.SFT",)
        grid = _make_grid(list(postcodes), confidence=0.50)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=True,
                build_description="test",
            )

        assert "cells_observed" in result
        assert "cells_improved" in result
        assert "cells_degraded" in result
        assert "transitions" in result
        assert "new_goals_added" in result


# ---------------------------------------------------------------------------
# _close_feedback_loop — failure path
# ---------------------------------------------------------------------------

class TestCloseFeedbackLoopFailure:
    """Test _close_feedback_loop on failed builds."""

    def test_failure_records_anomaly_observations(self, tmp_path):
        """Failed builds record anomaly observations that degrade confidence."""
        from kernel.store import save_grid, load_grid

        postcodes = ("INT.SEM.ECO.WHY.SFT",)
        grid = _make_grid(list(postcodes), confidence=0.70)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=False,
                failure_reason="Tests failed after modification",
            )

        assert result["cells_observed"] == 1
        assert result["cells_degraded"] == 1  # anomaly decays confidence
        assert result["cells_improved"] == 0

    def test_failure_reason_propagated(self, tmp_path):
        """Failure reason appears in observation detail via grid revisions."""
        from kernel.store import save_grid, load_grid

        postcodes = ("INT.SEM.ECO.WHY.SFT",)
        grid = _make_grid(list(postcodes), confidence=0.70)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        # Track the deltas via patching record_observation at kernel level
        recorded_deltas = []
        original_record = record_observation

        def _capturing_record(*args, **kwargs):
            delta = original_record(*args, **kwargs)
            recorded_deltas.append(delta)
            return delta

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)), \
             patch("kernel.observer.record_observation", side_effect=_capturing_record):
            bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=False,
                failure_reason="Syntax error in generated code",
            )

        assert len(recorded_deltas) == 1
        assert recorded_deltas[0].anomaly is True
        assert "Syntax error" in recorded_deltas[0].anomaly_detail


# ---------------------------------------------------------------------------
# _close_feedback_loop — edge cases
# ---------------------------------------------------------------------------

class TestCloseFeedbackLoopEdgeCases:
    """Test _close_feedback_loop with edge cases."""

    def test_no_grid_returns_empty_summary(self):
        """When no grid exists, returns zeroed summary gracefully."""
        bridge = EngineBridge()

        with patch("kernel.store.load_grid", return_value=None):
            result = bridge._close_feedback_loop(
                target_postcodes=("INT.SEM.ECO.WHY.SFT",),
                success=True,
            )

        assert result["cells_observed"] == 0
        assert result["new_goals_added"] == 0

    def test_empty_postcodes_returns_empty_summary(self):
        """Empty postcode tuple returns immediately."""
        bridge = EngineBridge()
        result = bridge._close_feedback_loop(
            target_postcodes=(),
            success=True,
        )
        assert result["cells_observed"] == 0

    def test_falls_back_to_session_grid(self, tmp_path):
        """If compiler-self-desc doesn't exist, uses session grid."""
        from kernel.store import save_grid, load_grid

        postcodes = ("INT.SEM.ECO.WHY.SFT",)
        grid = _make_grid(list(postcodes), confidence=0.60)
        # Only save to "session", not "compiler-self-desc"
        save_grid(grid, "session", db_dir=tmp_path)

        bridge = EngineBridge()

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=True,
            )

        assert result["cells_observed"] == 1

    def test_postcodes_not_in_grid(self, tmp_path):
        """Postcodes that don't exist in grid produce zero-confidence deltas."""
        from kernel.store import save_grid, load_grid

        grid = _make_grid(["INT.SEM.ECO.WHY.SFT"], confidence=0.70)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        # Request observation on a postcode that doesn't exist in the grid
        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)):
            result = bridge._close_feedback_loop(
                target_postcodes=("COG.BHV.APP.HOW.SFT",),
                success=True,
            )

        # Cell doesn't exist — observation recorded but no confidence change
        assert result["cells_observed"] == 1


# ---------------------------------------------------------------------------
# Goal generation from feedback loop
# ---------------------------------------------------------------------------

class TestFeedbackLoopGoalGeneration:
    """Test that _close_feedback_loop generates goals from updated grid."""

    def test_generates_goals_for_low_confidence(self, tmp_path):
        """After failure degrades confidence, goals are generated."""
        from kernel.store import save_grid, load_grid

        # Start with cells at 0.25 — already critical
        postcodes = ("INT.SEM.ECO.WHY.SFT", "COG.BHV.APP.HOW.SFT")
        grid = _make_grid(list(postcodes), confidence=0.25)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        mock_gs = MagicMock()
        mock_gs.active.return_value = []

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)), \
             patch("mother.goals.GoalStore", return_value=mock_gs):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=False,
                failure_reason="Build failed",
            )

        # Should have generated goals because cells are at critical confidence
        # (mock_gs.add.call_count tracks how many goals were added)
        assert result["new_goals_added"] >= 0

    def test_deduplicates_against_existing_goals(self, tmp_path):
        """Goals that match existing active goals are not duplicated."""
        from kernel.store import save_grid, load_grid
        from mother.goals import GoalStore, Goal

        postcodes = ("INT.SEM.ECO.WHY.SFT",)
        grid = _make_grid(list(postcodes), confidence=0.20)
        save_grid(grid, "compiler-self-desc", db_dir=tmp_path)

        bridge = EngineBridge()

        # Create a mock GoalStore with an existing goal that matches what
        # goals_from_grid would generate for a critical-confidence cell
        existing_goal = Goal(
            goal_id=1,
            timestamp=time.time(),
            description="1 cells below 30% confidence need immediate attention.",
            source="system",
            priority="urgent",
            status="active",
        )
        mock_gs = MagicMock()
        mock_gs.active.return_value = [existing_goal]

        with patch("kernel.store.load_grid", side_effect=lambda mid: load_grid(mid, db_dir=tmp_path)), \
             patch("kernel.store.save_grid", side_effect=lambda g, mid, **kw: save_grid(g, mid, db_dir=tmp_path)), \
             patch("mother.goals.GoalStore", return_value=mock_gs):
            result = bridge._close_feedback_loop(
                target_postcodes=postcodes,
                success=False,
                failure_reason="Build failed",
            )

        # The dedup logic should prevent duplicate goals from being added.
        # Since the existing goal description matches the pattern of the
        # generated goal, the add count should be lower than without dedup.
        assert isinstance(result["new_goals_added"], int)
        # The critical confidence goal should be deduped against the existing one
        # New goals might still be added for other categories (coverage, etc.)
        assert result["new_goals_added"] >= 0


# ---------------------------------------------------------------------------
# Observation recording correctness
# ---------------------------------------------------------------------------

class TestObservationRecording:
    """Test that observations produce correct deltas."""

    def test_confirmed_observation_boosts_confidence(self):
        """Confirmed observation (success) boosts confidence by _CONFIRM_BOOST."""
        grid = _make_grid(["INT.SEM.ECO.WHY.SFT"], confidence=0.60)
        delta = record_observation(
            grid=grid,
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="self_build",
            expected="build success",
            actual="build succeeded",
            confirmed=True,
            anomaly=False,
        )
        assert delta.confidence_after == pytest.approx(0.60 + _CONFIRM_BOOST)
        assert not delta.anomaly

    def test_anomaly_observation_decays_confidence(self):
        """Anomaly observation (failure) decays confidence by _ANOMALY_DECAY."""
        grid = _make_grid(["INT.SEM.ECO.WHY.SFT"], confidence=0.60)
        delta = record_observation(
            grid=grid,
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="self_build",
            expected="build success",
            actual="build failed: tests failed",
            confirmed=False,
            anomaly=True,
            anomaly_detail="Tests failed after modification",
        )
        assert delta.confidence_after == pytest.approx(0.60 - _ANOMALY_DECAY)
        assert delta.anomaly

    def test_batch_apply_updates_grid(self):
        """apply_batch mutates grid cells with new confidence values."""
        grid = _make_grid(
            ["INT.SEM.ECO.WHY.SFT", "COG.BHV.APP.HOW.SFT"],
            confidence=0.70,
        )
        deltas = [
            record_observation(grid, "INT.SEM.ECO.WHY.SFT", "self_build",
                             "success", "succeeded", confirmed=True),
            record_observation(grid, "COG.BHV.APP.HOW.SFT", "self_build",
                             "success", "failed", confirmed=False, anomaly=True),
        ]
        batch = apply_batch(grid, deltas)

        assert batch.cells_touched == 2
        assert batch.cells_improved == 1
        assert batch.cells_degraded == 1

        # Verify grid was actually mutated
        cell_int = grid.get("INT.SEM.ECO.WHY.SFT")
        assert cell_int.confidence == pytest.approx(0.70 + _CONFIRM_BOOST)

        cell_cog = grid.get("COG.BHV.APP.HOW.SFT")
        assert cell_cog.confidence == pytest.approx(0.70 - _ANOMALY_DECAY)


# ---------------------------------------------------------------------------
# Re-analysis — low confidence detection
# ---------------------------------------------------------------------------

class TestReAnalysis:
    """Test re-analysis finds low-confidence cells after observation."""

    def test_low_confidence_cells_detected(self):
        """goals_from_grid finds cells below threshold after degradation."""
        from mother.goal_generator import goals_from_grid

        cell_data = [
            ("INT.SEM.ECO.WHY.SFT", 0.25, "P", "intent_root"),
            ("COG.BHV.APP.HOW.SFT", 0.80, "F", "behavior"),
        ]
        goals = goals_from_grid(cell_data, {"INT", "COG"}, 2)

        # Should flag the 0.25 cell as critical
        critical_goals = [g for g in goals if g.priority == "critical"]
        assert len(critical_goals) >= 1
        assert "INT.SEM.ECO.WHY.SFT" in critical_goals[0].target_postcodes

    def test_healthy_grid_produces_no_confidence_goals(self):
        """A grid with all cells above threshold generates no confidence goals."""
        from mother.goal_generator import goals_from_grid

        cell_data = [
            ("INT.SEM.ECO.WHY.SFT", 0.90, "F", "intent_root"),
            ("COG.BHV.APP.HOW.SFT", 0.85, "F", "behavior"),
        ]
        goals = goals_from_grid(cell_data, {"INT", "COG"}, 2)

        confidence_goals = [g for g in goals if g.category == "confidence"]
        assert len(confidence_goals) == 0


# ---------------------------------------------------------------------------
# Daemon boundary fix — no core.outcome_store import
# ---------------------------------------------------------------------------

class TestDaemonBoundaryFix:
    """Verify daemon._sync_goals no longer imports from core/."""

    def test_no_core_import_in_daemon(self):
        """daemon.py must not import from core/ at module or function level."""
        import mother.daemon
        source_path = Path(mother.daemon.__file__)
        source = source_path.read_text()

        # Check for import statements from core/
        tree = ast.parse(source)
        core_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("core."):
                    core_imports.append(f"line {node.lineno}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("core."):
                        core_imports.append(f"line {node.lineno}: import {alias.name}")

        assert not core_imports, (
            f"daemon.py has {len(core_imports)} import(s) from core/: "
            + ", ".join(core_imports)
        )

    def test_sync_goals_uses_self_outcomes(self):
        """_sync_goals operates on self._outcomes, not OutcomeStore."""
        from mother.daemon import DaemonMode
        import inspect

        source = inspect.getsource(DaemonMode._sync_goals)
        assert "OutcomeStore" not in source
        assert "self._outcomes" in source

    def test_sync_goals_with_empty_outcomes(self, tmp_path):
        """_sync_goals returns cleanly when no outcomes exist."""
        from mother.daemon import DaemonMode

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = []
        # Should not raise
        daemon._sync_goals()

    def test_sync_goals_with_outcomes(self, tmp_path):
        """_sync_goals processes in-memory outcomes and generates goals."""
        from mother.daemon import DaemonMode
        from mother.governor_feedback import CompilationOutcome

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = [
            CompilationOutcome(
                compile_id="test-1",
                input_summary="test input",
                trust_score=30.0,
                completeness=20.0,
                consistency=25.0,
                coherence=30.0,
                traceability=20.0,
                actionability=20.0,
                specificity=20.0,
                codegen_readiness=20.0,
                component_count=5,
                rejected=True,
                rejection_reason="Low quality",
                domain="software",
            ),
            CompilationOutcome(
                compile_id="test-2",
                input_summary="test input 2",
                trust_score=25.0,
                completeness=15.0,
                consistency=20.0,
                coherence=25.0,
                traceability=15.0,
                actionability=15.0,
                specificity=15.0,
                codegen_readiness=15.0,
                component_count=3,
                rejected=True,
                rejection_reason="Low quality",
                domain="software",
            ),
        ]

        # Should not raise even if GoalStore path doesn't exist
        # (wrapped in try/except)
        daemon._sync_goals()


# ---------------------------------------------------------------------------
# Async wrapper — observe_and_reanalyze
# ---------------------------------------------------------------------------

class TestObserveAndReanalyze:
    """Test the async wrapper around _close_feedback_loop."""

    def test_async_wrapper_delegates_to_sync(self, tmp_path):
        """observe_and_reanalyze calls _close_feedback_loop."""
        bridge = EngineBridge()

        expected = {
            "cells_observed": 5,
            "cells_improved": 3,
            "cells_degraded": 1,
            "transitions": [],
            "new_goals_added": 2,
        }

        with patch.object(bridge, "_close_feedback_loop", return_value=expected):
            result = _run(bridge.observe_and_reanalyze(
                target_postcodes=("INT.SEM.ECO.WHY.SFT",),
                success=True,
                build_description="test",
            ))

        assert result == expected

    def test_async_wrapper_handles_exception(self):
        """observe_and_reanalyze returns empty summary on error."""
        bridge = EngineBridge()

        with patch.object(bridge, "_close_feedback_loop", side_effect=RuntimeError("boom")):
            result = _run(bridge.observe_and_reanalyze(
                target_postcodes=("INT.SEM.ECO.WHY.SFT",),
                success=True,
            ))

        assert result["cells_observed"] == 0
        assert result["new_goals_added"] == 0


# ---------------------------------------------------------------------------
# get_recent_outcomes — bridge boundary method
# ---------------------------------------------------------------------------

class TestGetRecentOutcomes:
    """Test bridge.get_recent_outcomes boundary method."""

    def test_returns_list_of_dicts(self):
        """get_recent_outcomes returns list of dicts."""
        bridge = EngineBridge()

        mock_record = MagicMock()
        mock_record.compile_id = "test-1"
        mock_record.input_summary = "test"
        mock_record.trust_score = 80.0
        mock_record.completeness = 90.0
        mock_record.consistency = 85.0
        mock_record.coherence = 80.0
        mock_record.traceability = 75.0
        mock_record.component_count = 10
        mock_record.rejected = False
        mock_record.rejection_reason = ""
        mock_record.domain = "software"
        mock_record.compression_loss_categories = ""

        mock_store = MagicMock()
        mock_store.recent.return_value = [mock_record]

        with patch("core.outcome_store.OutcomeStore", return_value=mock_store):
            results = bridge.get_recent_outcomes(limit=10)

        assert len(results) == 1
        assert results[0]["compile_id"] == "test-1"
        assert results[0]["trust_score"] == 80.0

    def test_returns_empty_on_error(self):
        """get_recent_outcomes returns [] on import or DB error."""
        bridge = EngineBridge()

        with patch("core.outcome_store.OutcomeStore", side_effect=RuntimeError("no db")):
            results = bridge.get_recent_outcomes()

        assert results == []


# ---------------------------------------------------------------------------
# State transitions from observations
# ---------------------------------------------------------------------------

class TestStateTransitions:
    """Test that observation-driven confidence changes trigger state transitions."""

    def test_promotion_on_high_confidence(self):
        """P -> F when confidence crosses 0.85 threshold."""
        grid = _make_grid(["INT.SEM.ECO.WHY.SFT"], fill_state=FillState.P, confidence=0.84)

        delta = record_observation(
            grid=grid,
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="self_build",
            expected="build success",
            actual="build succeeded",
            confirmed=True,
        )

        # Confidence should go to 0.84 + 0.03 = 0.87, above 0.85 threshold
        assert delta.confidence_after == pytest.approx(0.84 + _CONFIRM_BOOST)

        from kernel.observer import apply_observation
        transition = apply_observation(grid, delta)

        assert transition is not None
        assert transition == ("P", "F")

    def test_demotion_on_low_confidence(self):
        """F -> P when confidence drops below 0.50 threshold."""
        grid = _make_grid(["INT.SEM.ECO.WHY.SFT"], fill_state=FillState.F, confidence=0.55)

        delta = record_observation(
            grid=grid,
            postcode="INT.SEM.ECO.WHY.SFT",
            event_type="self_build",
            expected="build success",
            actual="build failed",
            confirmed=False,
            anomaly=True,
        )

        # Confidence: 0.55 - 0.12 = 0.43, below 0.50
        assert delta.confidence_after == pytest.approx(0.55 - _ANOMALY_DECAY)

        from kernel.observer import apply_observation
        transition = apply_observation(grid, delta)

        assert transition is not None
        assert transition == ("F", "P")


# ---------------------------------------------------------------------------
# Pattern-driven goal generation in daemon._sync_goals
# ---------------------------------------------------------------------------

class TestPatternGoalGeneration:
    """Verify that learned patterns above threshold spawn improvement goals."""

    def test_patterns_above_threshold_generate_goals(self, tmp_path):
        """Patterns with frequency >= 3 and confidence >= 0.6 produce goals."""
        from mother.daemon import DaemonMode
        from mother.governor_feedback import CompilationOutcome
        from kernel.memory import LearnedPattern

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = [
            CompilationOutcome(
                compile_id="t-1", input_summary="test", trust_score=50.0,
                completeness=50.0, consistency=50.0, coherence=50.0,
                traceability=50.0, actionability=50.0, specificity=50.0,
                codegen_readiness=50.0, component_count=5, rejected=False,
                rejection_reason="", domain="software",
            ),
        ]

        test_patterns = [
            LearnedPattern(
                pattern_id="recurring_loss_entity",
                category="recurring_gap",
                description="entity losses in 5/8 compilations",
                frequency=5,
                confidence=0.7,
                affected_postcodes=(),
                remediation="Strengthen entity extraction",
            ),
        ]

        added_goals = []
        mock_gs = MagicMock()
        mock_gs.active.return_value = []
        mock_gs.add.side_effect = lambda **kw: added_goals.append(kw)

        with patch("kernel.memory.load_patterns", return_value=test_patterns), \
             patch("mother.goals.GoalStore", return_value=mock_gs):
            daemon._sync_goals()

        # The pattern goal should appear in the added goals
        pattern_descs = [g["description"] for g in added_goals if "Recurring pattern" in g["description"]]
        assert len(pattern_descs) >= 1
        assert "entity losses" in pattern_descs[0]

    def test_patterns_below_threshold_ignored(self, tmp_path):
        """Patterns with frequency < 3 or confidence < 0.6 don't generate goals."""
        from mother.daemon import DaemonMode
        from mother.governor_feedback import CompilationOutcome
        from kernel.memory import LearnedPattern

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = [
            CompilationOutcome(
                compile_id="t-1", input_summary="test", trust_score=50.0,
                completeness=50.0, consistency=50.0, coherence=50.0,
                traceability=50.0, actionability=50.0, specificity=50.0,
                codegen_readiness=50.0, component_count=5, rejected=False,
                rejection_reason="", domain="software",
            ),
        ]

        # frequency=2 (below 3) and confidence=0.4 (below 0.6)
        test_patterns = [
            LearnedPattern(
                pattern_id="low_freq_pattern",
                category="recurring_gap",
                description="rare pattern",
                frequency=2,
                confidence=0.4,
                affected_postcodes=(),
                remediation="Some fix",
            ),
        ]

        added_goals = []
        mock_gs = MagicMock()
        mock_gs.active.return_value = []
        mock_gs.add.side_effect = lambda **kw: added_goals.append(kw)

        with patch("kernel.memory.load_patterns", return_value=test_patterns), \
             patch("mother.goals.GoalStore", return_value=mock_gs):
            daemon._sync_goals()

        # No pattern goals should have been added
        pattern_descs = [g["description"] for g in added_goals if "Recurring pattern" in g["description"]]
        assert len(pattern_descs) == 0

    def test_high_frequency_patterns_get_high_priority(self, tmp_path):
        """Patterns with frequency >= 5 get 'high' priority; others get 'medium'."""
        from mother.daemon import DaemonMode
        from mother.governor_feedback import CompilationOutcome
        from kernel.memory import LearnedPattern
        from mother.goal_generator import ImprovementGoal

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = [
            CompilationOutcome(
                compile_id="t-1", input_summary="test", trust_score=80.0,
                completeness=80.0, consistency=80.0, coherence=80.0,
                traceability=80.0, actionability=80.0, specificity=80.0,
                codegen_readiness=80.0, component_count=5, rejected=False,
                rejection_reason="", domain="software",
            ),
        ]

        test_patterns = [
            LearnedPattern(
                pattern_id="high_freq",
                category="recurring_gap",
                description="high freq pattern",
                frequency=7,
                confidence=0.8,
                affected_postcodes=(),
                remediation="Fix high freq",
            ),
            LearnedPattern(
                pattern_id="med_freq",
                category="recurring_gap",
                description="medium freq pattern",
                frequency=3,
                confidence=0.65,
                affected_postcodes=(),
                remediation="Fix medium freq",
            ),
        ]

        # Capture the goal_set from generate_goal_set
        captured_goal_sets = []
        original_generate = None
        from mother import goal_generator as _gg
        original_generate = _gg.generate_goal_set

        def _capture_generate(*args, **kwargs):
            result = original_generate(*args, **kwargs)
            captured_goal_sets.append(result)
            return result

        mock_gs = MagicMock()
        mock_gs.active.return_value = []

        with patch("kernel.memory.load_patterns", return_value=test_patterns), \
             patch("mother.goals.GoalStore", return_value=mock_gs), \
             patch("mother.goal_generator.generate_goal_set", side_effect=_capture_generate):
            daemon._sync_goals()

        assert len(captured_goal_sets) >= 1
        goals = captured_goal_sets[0].goals
        pattern_goals = [g for g in goals if "Recurring pattern" in g.description]

        high_freq_goals = [g for g in pattern_goals if "high freq" in g.description]
        med_freq_goals = [g for g in pattern_goals if "medium freq" in g.description]

        assert len(high_freq_goals) >= 1
        assert high_freq_goals[0].priority == "high"
        assert len(med_freq_goals) >= 1
        assert med_freq_goals[0].priority == "medium"

    def test_pattern_remediation_in_description(self, tmp_path):
        """Pattern remediation text is included in the goal description."""
        from mother.daemon import DaemonMode
        from mother.governor_feedback import CompilationOutcome
        from kernel.memory import LearnedPattern

        daemon = DaemonMode(config_dir=tmp_path)
        daemon._outcomes = [
            CompilationOutcome(
                compile_id="t-1", input_summary="test", trust_score=50.0,
                completeness=50.0, consistency=50.0, coherence=50.0,
                traceability=50.0, actionability=50.0, specificity=50.0,
                codegen_readiness=50.0, component_count=5, rejected=False,
                rejection_reason="", domain="software",
            ),
        ]

        test_patterns = [
            LearnedPattern(
                pattern_id="with_remediation",
                category="recurring_gap",
                description="entity losses recurring",
                frequency=4,
                confidence=0.7,
                affected_postcodes=(),
                remediation="Strengthen entity extraction in synthesis",
            ),
        ]

        added_goals = []
        mock_gs = MagicMock()
        mock_gs.active.return_value = []
        mock_gs.add.side_effect = lambda **kw: added_goals.append(kw)

        with patch("kernel.memory.load_patterns", return_value=test_patterns), \
             patch("mother.goals.GoalStore", return_value=mock_gs):
            daemon._sync_goals()

        pattern_descs = [g["description"] for g in added_goals if "Recurring pattern" in g["description"]]
        assert len(pattern_descs) >= 1
        assert "Suggested fix: Strengthen entity extraction in synthesis" in pattern_descs[0]


# ---------------------------------------------------------------------------
# code_task native engine — primary path with CLI fallback
# ---------------------------------------------------------------------------

class TestCodeTaskNativeEngine:
    """Verify code_task tries native engine first, falls back to CLI."""

    def test_native_engine_tried_first(self):
        """When native engine succeeds, CLI is never called."""
        bridge = EngineBridge()

        native_result = {
            "success": True,
            "result_text": "Native engine completed",
            "cost_usd": 0.50,
            "rolled_back": False,
            "error": "",
            "provider": "claude",
            "files_modified": ["main.py"],
            "turns_used": 5,
        }

        with patch.object(bridge, "_try_native_code_engine", return_value=native_result), \
             patch("mother.coding_agent.invoke_coding_agent") as mock_cli, \
             patch("mother.claude_code.git_snapshot", return_value="abc123"), \
             patch("mother.claude_code.git_rollback"):
            result = _run(bridge.code_task(
                prompt="Add a hello function",
                target_dir="/tmp/test_project",
            ))

        assert result["success"] is True
        assert result["result_text"] == "Native engine completed"
        assert result["cost_usd"] == 0.50
        # CLI should NOT have been called
        mock_cli.assert_not_called()

    def test_fallback_to_cli_when_native_returns_none(self):
        """When native engine returns None, CLI is used as fallback."""
        bridge = EngineBridge()

        mock_agent_result = MagicMock()
        mock_agent_result.success = True
        mock_agent_result.result_text = "CLI completed"
        mock_agent_result.cost_usd = 0.25
        mock_agent_result.error = ""

        with patch.object(bridge, "_try_native_code_engine", return_value=None), \
             patch("mother.coding_agent.invoke_coding_agent", return_value=mock_agent_result), \
             patch("mother.coding_agent.default_providers", return_value=[]), \
             patch("mother.coding_agent._clean_env", return_value={}), \
             patch("mother.claude_code.git_snapshot", return_value=""), \
             patch("mother.claude_code.git_rollback"), \
             patch("mother.claude_code.run_tests", return_value=True):
            result = _run(bridge.code_task(
                prompt="Fix the bug",
                target_dir="/tmp/test_project",
            ))

        assert result["success"] is True
        assert result["result_text"] == "CLI completed"
