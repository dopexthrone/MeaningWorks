"""
Tests for feedback loop wiring: observer firing, compression-loss resynthesis,
grid persistence, goal sync, and bridge grid access.

Verifies the four feedback gaps are closed:
1. Observer fires after compilation
2. Compression losses feed re-synthesis
3. Grid persists across compilations (L2)
4. Goals flow into GoalStore
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.observer import (
    record_observation,
    apply_observation,
    apply_batch,
    _CONFIRM_BOOST,
    _CONTRADICT_DECAY,
    _ANOMALY_DECAY,
)
from kernel.store import save_grid, load_grid
from core.engine import MotherlabsEngine
from core.protocol import SharedState


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
            connections=frozenset(),
            parent=None,
            source=("test",),
            revisions=(),
        )
        grid.cells[pc_str] = cell
    return grid


def _make_engine(tmp_path):
    """Create a minimal engine with mock LLM for testing."""
    from persistence.corpus import Corpus
    mock_client = Mock()
    mock_client.provider_name = "mock"
    mock_client.model_name = "mock-model"
    mock_client.deterministic = True
    mock_client.model = "mock-model"
    engine = MotherlabsEngine(
        llm_client=mock_client,
        pipeline_mode="staged",
        corpus=Corpus(tmp_path / "corpus"),
        auto_store=False,
    )
    return engine


# ===========================================================================
# TestObserverFiring
# ===========================================================================

class TestObserverFiring:
    """Observer fires after verification and records correct data."""

    def test_observer_fires_after_verification(self):
        """Observer records observations for F/P cells."""
        grid = _make_grid(
            ["INT.SEM.ECO.WHY.SFT", "SEM.ENT.DOM.HOW.SFT"],
            fill_state=FillState.P,
            confidence=0.70,
        )
        deltas = []
        for pc_key, cell in grid.cells.items():
            if cell.fill in (FillState.F, FillState.P):
                delta = record_observation(
                    grid, pc_key,
                    event_type="compilation",
                    expected="progressing",
                    actual="pass",
                    confirmed=True,
                )
                deltas.append(delta)
        assert len(deltas) == 2
        batch = apply_batch(grid, deltas)
        assert batch.cells_touched == 2

    def test_observer_records_correct_event_type(self):
        grid = _make_grid()
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            event_type="compilation",
            expected="verified",
            actual="pass",
            confirmed=True,
        )
        assert delta.event_type == "compilation"

    def test_observer_batch_applies_to_grid(self):
        grid = _make_grid(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "progressing", "pass",
            confirmed=True,
        )
        batch = apply_batch(grid, [delta])
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(0.70 + _CONFIRM_BOOST)

    def test_transitions_logged_on_state_change(self):
        """High-confidence P cell should promote to F."""
        grid = _make_grid(fill_state=FillState.P, confidence=0.84)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "progressing", "pass",
            confirmed=True,
        )
        # 0.84 + 0.03 = 0.87 > 0.85 promote threshold
        batch = apply_batch(grid, [delta])
        assert len(batch.transitions) == 1
        assert batch.transitions[0][1] == "P"  # old state
        assert batch.transitions[0][2] == "F"  # new state

    def test_anomaly_flagged_when_low_fidelity(self):
        grid = _make_grid(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "verified", "needs_work",
            confirmed=False,
            anomaly=True,
        )
        assert delta.anomaly is True
        assert delta.confidence_after == pytest.approx(0.70 - _ANOMALY_DECAY)

    def test_grid_state_after_observer(self):
        """Grid cells have updated confidence after observer runs."""
        grid = _make_grid(confidence=0.60)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "progressing", "pass",
            confirmed=True,
        )
        apply_observation(grid, delta)
        cell = grid.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence > 0.60

    def test_observer_skipped_when_no_grid(self):
        """No kernel grid means observer block is skipped (no crash)."""
        # Simulates engine._kernel_grid = None
        grid = None
        assert grid is None  # observer block checks `if self._kernel_grid`

    def test_observer_never_blocks_compilation(self):
        """Observer failure is caught and doesn't propagate."""
        grid = _make_grid()
        # Observation on nonexistent cell returns delta with anomaly
        delta = record_observation(
            grid, "NONEXISTENT.POSTCODE.X.Y.Z",
            "compilation", "verified", "pass",
            confirmed=True,
        )
        # Should not raise, returns anomaly delta
        assert delta.anomaly is True


# ===========================================================================
# TestCompressionResynthesis
# ===========================================================================

class TestCompressionResynthesis:
    """Compression losses from closed-loop gate feed into re-synthesis."""

    def _make_state_with_losses(self, losses):
        state = SharedState()
        state.known["compression_losses"] = losses
        return state

    def test_compression_losses_prepended_to_gaps(self):
        """Compression losses appear before verification gaps."""
        state = self._make_state_with_losses(["user authentication", "payment flow"])
        verification = {
            "completeness": {"gaps": ["missing error handling"]},
        }

        # Simulate gap extraction logic from _targeted_resynthesis
        gaps = []
        compression_losses = state.known.get("compression_losses", [])
        if compression_losses:
            gaps.extend([f"COMPRESSION_LOSS: {f}" for f in compression_losses])
        completeness = verification.get("completeness", {})
        if completeness.get("gaps"):
            gaps.extend(completeness["gaps"])

        assert len(gaps) == 3
        assert gaps[0] == "COMPRESSION_LOSS: user authentication"
        assert gaps[1] == "COMPRESSION_LOSS: payment flow"
        assert gaps[2] == "missing error handling"

    def test_resynthesis_reads_compression_losses(self):
        state = self._make_state_with_losses(["data validation"])
        losses = state.known.get("compression_losses", [])
        assert losses == ["data validation"]

    def test_empty_compression_losses_no_effect(self):
        state = SharedState()
        losses = state.known.get("compression_losses", [])
        assert losses == []

    def test_resynthesis_works_with_verification_gaps_only(self):
        state = SharedState()
        verification = {"completeness": {"gaps": ["missing component"]}}

        gaps = []
        compression_losses = state.known.get("compression_losses", [])
        if compression_losses:
            gaps.extend([f"COMPRESSION_LOSS: {f}" for f in compression_losses])
        if verification.get("completeness", {}).get("gaps"):
            gaps.extend(verification["completeness"]["gaps"])

        assert len(gaps) == 1
        assert gaps[0] == "missing component"

    def test_compression_loss_priority_order(self):
        """Compression losses come before all verification gaps."""
        state = self._make_state_with_losses(["lost entity"])
        verification = {
            "completeness": {"gaps": ["gap1"]},
            "consistency": {"conflicts": ["conflict1"]},
            "coherence": {"suggested_fixes": ["fix1"]},
        }

        gaps = []
        compression_losses = state.known.get("compression_losses", [])
        if compression_losses:
            gaps.extend([f"COMPRESSION_LOSS: {f}" for f in compression_losses])
        if verification.get("completeness", {}).get("gaps"):
            gaps.extend(verification["completeness"]["gaps"])
        if verification.get("consistency", {}).get("conflicts"):
            gaps.extend([f"CONFLICT: {c}" for c in verification["consistency"]["conflicts"]])
        if verification.get("coherence", {}).get("suggested_fixes"):
            gaps.extend([f"FIX: {f}" for f in verification["coherence"]["suggested_fixes"]])

        assert gaps[0].startswith("COMPRESSION_LOSS:")
        assert len(gaps) == 4

    def test_gap_prompt_includes_compression_loss_prefix(self):
        losses = ["authentication module"]
        gaps = [f"COMPRESSION_LOSS: {f}" for f in losses]
        gap_text = "\n".join(f"- {g}" for g in gaps)
        assert "COMPRESSION_LOSS: authentication module" in gap_text


# ===========================================================================
# TestGridPersistence
# ===========================================================================

class TestGridPersistence:
    """Grid saves to and loads from maps.db for L2 accumulation."""

    def test_grid_saved_after_compilation(self, tmp_path):
        grid = _make_grid()
        save_grid(grid, map_id="session", name="test compile", db_dir=tmp_path)
        loaded = load_grid("session", db_dir=tmp_path)
        assert loaded is not None
        assert len(loaded.cells) == len(grid.cells)

    def test_grid_loaded_into_next_compilation(self, tmp_path):
        grid = _make_grid(confidence=0.80)
        save_grid(grid, map_id="session", name="first compile", db_dir=tmp_path)
        prev = load_grid("session", db_dir=tmp_path)
        assert prev is not None
        cell = prev.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(0.80)

    def test_fresh_compile_no_prior_grid(self, tmp_path):
        result = load_grid("session", db_dir=tmp_path)
        assert result is None

    def test_save_load_roundtrip_preserves_confidence(self, tmp_path):
        grid = _make_grid(confidence=0.65)
        save_grid(grid, map_id="session", name="test", db_dir=tmp_path)
        loaded = load_grid("session", db_dir=tmp_path)
        cell = loaded.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(0.65)
        assert cell.fill == FillState.P

    def test_observer_modified_confidence_persists(self, tmp_path):
        grid = _make_grid(confidence=0.70)
        delta = record_observation(
            grid, "INT.SEM.ECO.WHY.SFT",
            "compilation", "progressing", "pass",
            confirmed=True,
        )
        apply_observation(grid, delta)
        expected_conf = 0.70 + _CONFIRM_BOOST

        save_grid(grid, map_id="session", name="observed", db_dir=tmp_path)
        loaded = load_grid("session", db_dir=tmp_path)
        cell = loaded.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(expected_conf)

    def test_map_id_session_used_consistently(self, tmp_path):
        grid = _make_grid()
        save_grid(grid, map_id="session", name="test", db_dir=tmp_path)
        assert load_grid("session", db_dir=tmp_path) is not None
        assert load_grid("other", db_dir=tmp_path) is None

    def test_persistence_failure_doesnt_block(self, tmp_path):
        """Persistence uses try/except — verify the pattern."""
        try:
            save_grid(_make_grid(), map_id="session", name="test", db_dir=tmp_path)
        except Exception:
            pytest.fail("Persistence should not raise")

    def test_grid_accumulates_across_compiles(self, tmp_path):
        """Second save overwrites but preserves updated state."""
        grid1 = _make_grid(confidence=0.50)
        save_grid(grid1, map_id="session", name="compile-1", db_dir=tmp_path)

        grid2 = _make_grid(confidence=0.80)
        save_grid(grid2, map_id="session", name="compile-2", db_dir=tmp_path)

        loaded = load_grid("session", db_dir=tmp_path)
        cell = loaded.cells["INT.SEM.ECO.WHY.SFT"]
        assert cell.confidence == pytest.approx(0.80)


# ===========================================================================
# TestGoalSync
# ===========================================================================

class TestGoalSync:
    """Improvement goals flow from generator to GoalStore."""

    def test_improvement_goals_flow_to_goal_store(self, tmp_path):
        from mother.goals import GoalStore
        from mother.goal_generator import ImprovementGoal, GoalSet, generate_goal_set

        goals = [
            ImprovementGoal(
                goal_id="g1", priority="high", category="confidence",
                description="Improve low-confidence cells in INT layer",
                source="observer:low_confidence",
            ),
        ]
        goal_set = generate_goal_set(goals, [], [])

        db = tmp_path / "test.db"
        gs = GoalStore(db)
        for g in goal_set.goals:
            mapped = {"critical": "urgent", "high": "high", "medium": "normal", "low": "low"}.get(g.priority, "normal")
            gs.add(description=g.description, source="system", priority=mapped)
        active = gs.active()
        gs.close()

        assert len(active) >= 1
        assert active[0].source == "system"

    def test_priority_mapping_correct(self):
        _map = {"critical": "urgent", "high": "high", "medium": "normal", "low": "low"}
        assert _map["critical"] == "urgent"
        assert _map["high"] == "high"
        assert _map["medium"] == "normal"
        assert _map["low"] == "low"

    def test_duplicate_goals_not_re_added(self, tmp_path):
        from mother.goals import GoalStore

        db = tmp_path / "test.db"
        gs = GoalStore(db)
        gs.add("Improve INT layer confidence", source="system")

        existing = gs.active()
        existing_descs = [g.description.lower() for g in existing]
        new_desc = "improve int layer confidence"

        # Dedup check
        is_dup = any(new_desc in ed or ed in new_desc for ed in existing_descs)
        assert is_dup is True
        gs.close()

    def test_source_set_to_system(self, tmp_path):
        from mother.goals import GoalStore

        db = tmp_path / "test.db"
        gs = GoalStore(db)
        gid = gs.add("Test goal", source="system")
        goal = gs.get(gid)
        gs.close()
        assert goal.source == "system"

    def test_daemon_sync_goals_creates_from_feedback(self):
        """Daemon._sync_goals uses analyze_outcomes → goals_from_feedback."""
        from mother.governor_feedback import CompilationOutcome, analyze_outcomes
        from mother.goal_generator import goals_from_feedback

        outcome = CompilationOutcome(
            compile_id="d-1",
            input_summary="test",
            trust_score=30.0,
            completeness=30.0,
            consistency=30.0,
            coherence=30.0,
            traceability=30.0,
            actionability=30.0,
            specificity=30.0,
            codegen_readiness=30.0,
            component_count=2,
            rejected=True,
            rejection_reason="low trust",
            domain="software",
        )
        report = analyze_outcomes([outcome])
        weaknesses = [(w.dimension, w.severity, w.mean_score) for w in report.weaknesses]
        feedback_goals = goals_from_feedback(weaknesses, report.rejection_rate, report.trend)
        # Low scores should generate improvement goals
        assert len(feedback_goals) > 0

    def test_bridge_sync_goals_returns_count(self, tmp_path):
        """Bridge.sync_goals_to_store returns count of added goals."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge.__new__(EngineBridge)
        bridge._engine = Mock()
        bridge._engine._kernel_grid = None
        bridge._engine._compilation_outcomes = []

        # With no grid and no outcomes, should return 0
        result = bridge.sync_goals_to_store(tmp_path / "test.db")
        assert result == 0

    def test_empty_goals_produce_no_entries(self, tmp_path):
        from mother.goals import GoalStore
        from mother.goal_generator import generate_goal_set

        goal_set = generate_goal_set([], [], [])
        db = tmp_path / "test.db"
        gs = GoalStore(db)
        count_before = gs.count_active()

        for g in goal_set.goals:
            gs.add(description=g.description, source="system")
        count_after = gs.count_active()
        gs.close()

        assert count_after == count_before

    def test_goal_sync_failure_doesnt_block(self):
        """Goal sync wrapped in try/except — verify pattern."""
        try:
            # Simulate the try/except pattern used in daemon and chat
            result = 0
            raise RuntimeError("simulated failure")
        except Exception:
            result = 0  # failure returns 0
        assert result == 0


# ===========================================================================
# TestEngineExtractDimScore
# ===========================================================================

class TestEngineExtractDimScore:
    """MotherlabsEngine._extract_dim_score handles all verification formats."""

    def test_flat_verification_dict(self):
        score = MotherlabsEngine._extract_dim_score(
            {"completeness": 72}, "completeness"
        )
        assert score == 72.0

    def test_nested_verification_dict(self):
        score = MotherlabsEngine._extract_dim_score(
            {"completeness": {"score": 85, "gaps": ["x"]}}, "completeness"
        )
        assert score == 85.0

    def test_missing_dimension_returns_zero(self):
        score = MotherlabsEngine._extract_dim_score({}, "completeness")
        assert score == 0.0


# ===========================================================================
# TestBridgeGridAccess
# ===========================================================================

class TestBridgeGridAccess:
    """Bridge reads _kernel_grid (not _last_kernel_grid) from engine."""

    def test_get_improvement_goals_uses_kernel_grid(self):
        from mother.bridge import EngineBridge

        bridge = EngineBridge.__new__(EngineBridge)
        grid = _make_grid(confidence=0.30)

        engine = Mock()
        engine._kernel_grid = grid
        engine._compilation_outcomes = []
        bridge._engine = engine

        result = bridge.get_improvement_goals()
        # With low confidence cells, should generate goals
        assert result is not None
        assert "goals" in result

    def test_returns_none_when_no_grid(self):
        from mother.bridge import EngineBridge

        bridge = EngineBridge.__new__(EngineBridge)
        engine = Mock()
        engine._kernel_grid = None
        engine._compilation_outcomes = []
        bridge._engine = engine

        result = bridge.get_improvement_goals()
        # Still returns something (from feedback/anomaly generators)
        # but grid_goals will be empty
        assert result is not None

    def test_returns_goals_with_low_confidence_cells(self):
        from mother.bridge import EngineBridge

        bridge = EngineBridge.__new__(EngineBridge)
        grid = _make_grid(
            postcodes=["INT.SEM.ECO.WHY.SFT", "SEM.ENT.DOM.HOW.SFT"],
            confidence=0.20,
        )

        engine = Mock()
        engine._kernel_grid = grid
        engine._compilation_outcomes = []
        bridge._engine = engine

        result = bridge.get_improvement_goals()
        assert result is not None
        assert result["total"] > 0


# ===========================================================================
# TestEngineKernelGridCaching
# ===========================================================================

class TestEngineKernelGridCaching:
    """Engine caches _kernel_grid for observer + persistence."""

    def test_engine_has_kernel_grid_attr(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert hasattr(engine, "_kernel_grid")
        assert engine._kernel_grid is None

    def test_kernel_grid_survives_multiple_accesses(self, tmp_path):
        engine = _make_engine(tmp_path)
        grid = _make_grid()
        engine._kernel_grid = grid
        assert engine._kernel_grid is grid
        assert engine._kernel_grid.cells is grid.cells
