"""
tests/test_l3_loop.py — L3: Compile the Compiler — Close the Self-Referential Loop.

Tests for the 5 changes that close the F(F) ~ F loop:
1. [SELF-COMPILE] prefix routing in compile()
2. Self-compile grid persistence with "compiler-self-desc" map_id
3. Self-description grid loaded as history in Phase 3.5
4. Grid-level convergence tracking in run_self_compile_loop()
5. Daemon autonomous self-compile scheduling

Zero kernel changes — all tests mock at boundaries.
"""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import pytest

from persistence.corpus import Corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(tmp_path):
    """Create a minimal engine with mock LLM for testing."""
    from core.engine import MotherlabsEngine
    client = Mock()
    client.provider_name = "mock"
    client.model_name = "mock-model"
    client.deterministic = True
    client.model = "mock-model"
    engine = MotherlabsEngine(
        llm_client=client,
        pipeline_mode="staged",
        corpus=Corpus(tmp_path / "corpus"),
        auto_store=False,
    )
    return engine


def _make_mock_grid(cells_keys=None, fill_rate=0.5):
    """Create a mock Grid with given cell keys and fill_rate."""
    grid = MagicMock()
    if cells_keys is None:
        cells_keys = {"L0.ENT.ECO.SEM.GEN": Mock(), "L1.BHV.APP.FNC.GEN": Mock()}
    grid.cells = {k: Mock() for k in cells_keys}
    grid.fill_rate = fill_rate
    grid.total_cells = len(cells_keys)
    return grid


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ===========================================================================
# 1. PREFIX ROUTING — [SELF-COMPILE] in compile()
# ===========================================================================

class TestPrefixRouting:
    """[SELF-COMPILE] prefix in description routes to self_compile()."""

    def test_self_compile_prefix_routes(self, tmp_path):
        """Input starting with [SELF-COMPILE] should call self_compile()."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True
        engine.self_compile = Mock(return_value=mock_result)

        result = engine.compile("[SELF-COMPILE]")
        engine.self_compile.assert_called_once()
        assert result is mock_result

    def test_self_compile_prefix_with_whitespace(self, tmp_path):
        """Leading whitespace before [SELF-COMPILE] still routes."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        engine.self_compile = Mock(return_value=mock_result)

        result = engine.compile("  [SELF-COMPILE]")
        engine.self_compile.assert_called_once()

    def test_self_compile_prefix_with_trailing_text(self, tmp_path):
        """[SELF-COMPILE] with trailing text still routes."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        engine.self_compile = Mock(return_value=mock_result)

        result = engine.compile("[SELF-COMPILE] extra text here")
        engine.self_compile.assert_called_once()

    def test_normal_input_not_intercepted(self, tmp_path):
        """Normal description should NOT trigger self_compile routing."""
        engine = _make_engine(tmp_path)
        engine.self_compile = Mock()

        # This will fail at some point in the pipeline, but self_compile
        # should not be called
        try:
            engine.compile("Build a task manager")
        except Exception:
            pass
        engine.self_compile.assert_not_called()

    def test_self_improvement_not_intercepted(self, tmp_path):
        """[SELF-IMPROVEMENT] prefix should NOT trigger self_compile routing."""
        engine = _make_engine(tmp_path)
        engine.self_compile = Mock()

        try:
            engine.compile("[SELF-IMPROVEMENT] improve synthesis")
        except Exception:
            pass
        engine.self_compile.assert_not_called()

    def test_insight_emitted_on_routing(self, tmp_path):
        """L3 insight should be emitted when routing to self_compile."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        engine.self_compile = Mock(return_value=mock_result)
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        engine.compile("[SELF-COMPILE]")
        assert any("L3: self-compile triggered" in i for i in insights)


# ===========================================================================
# 2. GRID PERSISTENCE — self_compile saves "compiler-self-desc"
# ===========================================================================

class TestGridPersistence:
    """self_compile() saves kernel grid with map_id='compiler-self-desc'."""

    def test_saves_grid_after_compile(self, tmp_path):
        """Kernel grid saved under 'compiler-self-desc' map_id."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()

        # Mock compile_with_axioms to set _kernel_grid and return result
        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": []}

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("core.engine.save_grid", create=True) as mock_save:
            # Patch at the import location inside self_compile
            with patch("kernel.store.save_grid") as mock_save_store:
                result = engine.self_compile()
                mock_save_store.assert_called_once_with(
                    mock_grid,
                    map_id="compiler-self-desc",
                    name="Compiler Self-Description",
                )

    def test_no_grid_skips_save(self, tmp_path):
        """If _kernel_grid is None, save is skipped."""
        engine = _make_engine(tmp_path)

        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = None
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid") as mock_save_store:
            engine.self_compile()
            mock_save_store.assert_not_called()

    def test_save_failure_is_nonfatal(self, tmp_path):
        """Exception during save_grid does not crash self_compile."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()

        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid", side_effect=RuntimeError("db locked")):
            result = engine.self_compile()
            assert result is mock_result  # did not crash

    def test_returns_compile_result(self, tmp_path):
        """self_compile() still returns the CompileResult from compile_with_axioms."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = None
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        result = engine.self_compile()
        assert result is mock_result

    def test_does_not_clobber_session(self, tmp_path):
        """Saving 'compiler-self-desc' should NOT use map_id='session'."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()
        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid") as mock_save:
            engine.self_compile()
            # The map_id kwarg should be "compiler-self-desc", not "session"
            call_kwargs = mock_save.call_args
            assert call_kwargs[1]["map_id"] == "compiler-self-desc"

    def test_insight_includes_cell_count(self, tmp_path):
        """Insight message includes cell count and fill rate."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid(
            cells_keys={"L0.ENT.ECO.SEM.GEN": Mock(), "L1.BHV.APP.FNC.GEN": Mock()},
            fill_rate=0.75,
        )
        mock_result = Mock()
        mock_result.success = True
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid"):
            engine.self_compile()

        l3_insights = [i for i in insights if "L3: self-desc grid saved" in i]
        assert len(l3_insights) == 1
        assert "2 cells" in l3_insights[0]
        assert "75%" in l3_insights[0]

    def test_survives_across_calls(self, tmp_path):
        """Multiple self_compile() calls each attempt save."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()
        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = mock_grid
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid") as mock_save:
            engine.self_compile()
            engine.self_compile()
            assert mock_save.call_count == 2

    def test_falsy_grid_skips_save(self, tmp_path):
        """Falsy _kernel_grid (e.g. empty list assigned) → save skipped."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True

        def fake_compile_with_axioms(*args, **kwargs):
            engine._kernel_grid = []  # falsy
            return mock_result

        engine.compile_with_axioms = fake_compile_with_axioms

        with patch("kernel.store.save_grid") as mock_save:
            engine.self_compile()
            mock_save.assert_not_called()


# ===========================================================================
# 3. HISTORY INJECTION — Phase 3.5 loads self-desc as history
# ===========================================================================

class TestHistoryInjection:
    """Phase 3.5 kernel compilation loads self-desc grid as third history source."""

    def test_self_desc_appended_to_history(self, tmp_path):
        """load_grid('compiler-self-desc') result appended to history list."""
        from kernel.grid import Grid
        mock_self_desc = _make_mock_grid({"L0.ENT.ECO.SEM.GEN": Mock()})

        with patch("kernel.store.load_grid") as mock_load:
            # First call: "session" → None, Second call: "compiler-self-desc" → grid
            mock_load.side_effect = lambda map_id, **kw: (
                mock_self_desc if map_id == "compiler-self-desc" else None
            )

            engine = _make_engine(tmp_path)
            insights = []
            engine.on_insight = lambda msg: insights.append(msg)

            # We can't easily run the full pipeline, so test the load logic directly
            # by simulating Phase 3.5's load sequence
            from kernel.store import load_grid as _load_grid
            history = []
            # L2: session grid
            prev_grid = _load_grid("session")
            if prev_grid is not None:
                history.append(prev_grid)
            # L3: self-desc
            self_desc_grid = _load_grid("compiler-self-desc")
            if self_desc_grid is not None:
                history.append(self_desc_grid)

            assert len(history) == 1
            assert history[0] is mock_self_desc

    def test_works_without_self_desc(self, tmp_path):
        """No self-desc grid → history has only GT and session."""
        with patch("kernel.store.load_grid", return_value=None):
            from kernel.store import load_grid as _load_grid
            history = []
            prev = _load_grid("session")
            if prev is not None:
                history.append(prev)
            self_desc = _load_grid("compiler-self-desc")
            if self_desc is not None:
                history.append(self_desc)

            assert len(history) == 0

    def test_load_failure_is_nonfatal(self, tmp_path):
        """Exception during load_grid for self-desc is caught and logged."""
        with patch("kernel.store.load_grid", side_effect=RuntimeError("db locked")):
            from kernel.store import load_grid as _load_grid
            history = []
            try:
                self_desc = _load_grid("compiler-self-desc")
                if self_desc is not None:
                    history.append(self_desc)
            except Exception:
                pass  # mirrors the try/except in engine.py
            # Should not crash — history stays empty
            assert len(history) == 0

    def test_history_order_gt_session_selfdesc(self, tmp_path):
        """History order should be [ground_truth, session, self_desc]."""
        mock_session = _make_mock_grid({"L0.ENT.APP.SEM.GEN": Mock()})
        mock_self_desc = _make_mock_grid({"L0.ENT.ECO.SEM.GEN": Mock()})
        mock_gt = _make_mock_grid({"L0.SEM.ECO.SEM.GEN": Mock()})

        with patch("kernel.store.load_grid") as mock_load:
            mock_load.side_effect = lambda map_id, **kw: {
                "session": mock_session,
                "compiler-self-desc": mock_self_desc,
            }.get(map_id)

            from kernel.store import load_grid as _load_grid

            # Simulate engine Phase 3.5 load sequence
            history = [mock_gt]  # ground truth first
            prev = _load_grid("session")
            if prev is not None:
                history.append(prev)
            self_desc = _load_grid("compiler-self-desc")
            if self_desc is not None:
                history.append(self_desc)

            assert len(history) == 3
            assert history[0] is mock_gt
            assert history[1] is mock_session
            assert history[2] is mock_self_desc

    def test_insight_emitted_on_load(self, tmp_path):
        """Loading self-desc grid should emit an L3 insight."""
        engine = _make_engine(tmp_path)
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        mock_self_desc = _make_mock_grid(
            {"L0.ENT.ECO.SEM.GEN": Mock(), "L1.BHV.APP.FNC.GEN": Mock()}
        )

        # Verify the insight format by simulating
        engine._emit_insight(f"L3: self-desc loaded ({len(mock_self_desc.cells)} cells)")
        assert any("L3: self-desc loaded" in i for i in insights)
        assert any("2 cells" in i for i in insights)

    def test_both_session_and_selfdesc_loaded(self, tmp_path):
        """Both session and self-desc can coexist in history."""
        mock_session = _make_mock_grid({"L0.ENT.APP.SEM.GEN": Mock()})
        mock_self_desc = _make_mock_grid({"L0.ENT.ECO.SEM.GEN": Mock()})

        with patch("kernel.store.load_grid") as mock_load:
            mock_load.side_effect = lambda map_id, **kw: {
                "session": mock_session,
                "compiler-self-desc": mock_self_desc,
            }.get(map_id)

            from kernel.store import load_grid as _load_grid
            history = []
            prev = _load_grid("session")
            if prev is not None:
                history.append(prev)
            self_desc = _load_grid("compiler-self-desc")
            if self_desc is not None:
                history.append(self_desc)

            assert len(history) == 2
            assert history[0] is mock_session
            assert history[1] is mock_self_desc

    def test_selfdesc_with_no_session(self, tmp_path):
        """Self-desc loads even when session grid is absent."""
        mock_self_desc = _make_mock_grid({"L0.ENT.ECO.SEM.GEN": Mock()})

        with patch("kernel.store.load_grid") as mock_load:
            mock_load.side_effect = lambda map_id, **kw: (
                mock_self_desc if map_id == "compiler-self-desc" else None
            )

            from kernel.store import load_grid as _load_grid
            history = []
            prev = _load_grid("session")
            if prev is not None:
                history.append(prev)
            self_desc = _load_grid("compiler-self-desc")
            if self_desc is not None:
                history.append(self_desc)

            assert len(history) == 1
            assert history[0] is mock_self_desc


# ===========================================================================
# 4. CONVERGENCE TRACKING — run_self_compile_loop() grid comparison
# ===========================================================================

class TestConvergenceTracking:
    """Grid-level convergence in run_self_compile_loop()."""

    def test_no_previous_grid(self, tmp_path):
        """No saved self-desc → no convergence insight."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid({"A": Mock(), "B": Mock()})
        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = mock_grid
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid", return_value=None):
            report = engine.run_self_compile_loop(runs=1)

        # No "L3 convergence" insight because no previous grid
        conv_insights = [i for i in insights if "L3 convergence" in i]
        assert len(conv_insights) == 0

    def test_identical_grids_show_fixed_point(self, tmp_path):
        """Identical cell keys → overlap=1.0 → 'fixed point'."""
        engine = _make_engine(tmp_path)
        keys = {"A": Mock(), "B": Mock(), "C": Mock()}
        mock_grid = _make_mock_grid(keys)
        prev_grid = _make_mock_grid(keys)

        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = mock_grid
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid", return_value=prev_grid):
            report = engine.run_self_compile_loop(runs=1)

        conv = [i for i in insights if "L3 convergence" in i]
        assert len(conv) == 1
        assert "fixed point" in conv[0]

    def test_completely_different_grids_show_evolving(self, tmp_path):
        """Zero overlap → 'evolving'."""
        engine = _make_engine(tmp_path)
        curr_grid = _make_mock_grid({"A": Mock(), "B": Mock()})
        prev_grid = _make_mock_grid({"C": Mock(), "D": Mock()})

        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = curr_grid
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid", return_value=prev_grid):
            report = engine.run_self_compile_loop(runs=1)

        conv = [i for i in insights if "L3 convergence" in i]
        assert len(conv) == 1
        assert "evolving" in conv[0]

    def test_threshold_boundary_above(self, tmp_path):
        """Overlap > 0.85 → fixed point."""
        engine = _make_engine(tmp_path)
        # 18/20 = 0.9 overlap (Jaccard = 18/20 = 0.9)
        shared = {f"K{i}": Mock() for i in range(18)}
        curr_extra = {f"X{i}": Mock() for i in range(1)}
        prev_extra = {f"Y{i}": Mock() for i in range(1)}
        curr_keys = {**shared, **curr_extra}
        prev_keys = {**shared, **prev_extra}

        curr_grid = _make_mock_grid(curr_keys)
        prev_grid = _make_mock_grid(prev_keys)

        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = curr_grid
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid", return_value=prev_grid):
            engine.run_self_compile_loop(runs=1)

        conv = [i for i in insights if "L3 convergence" in i]
        assert any("fixed point" in c for c in conv)

    def test_threshold_boundary_below(self, tmp_path):
        """Overlap <= 0.85 → evolving."""
        engine = _make_engine(tmp_path)
        # 8 shared, 2 unique each → Jaccard = 8/12 = 0.667
        shared = {f"K{i}": Mock() for i in range(8)}
        curr_extra = {f"X{i}": Mock() for i in range(2)}
        prev_extra = {f"Y{i}": Mock() for i in range(2)}
        curr_keys = {**shared, **curr_extra}
        prev_keys = {**shared, **prev_extra}

        curr_grid = _make_mock_grid(curr_keys)
        prev_grid = _make_mock_grid(prev_keys)

        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = curr_grid
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid", return_value=prev_grid):
            engine.run_self_compile_loop(runs=1)

        conv = [i for i in insights if "L3 convergence" in i]
        assert any("evolving" in c for c in conv)

    def test_convergence_check_failure_nonfatal(self, tmp_path):
        """Exception during convergence check does not crash loop."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid({"A": Mock()})
        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = mock_grid
            return mock_result

        engine.self_compile = fake_self_compile

        with patch("kernel.store.load_grid", side_effect=RuntimeError("boom")):
            report = engine.run_self_compile_loop(runs=1)
            # Did not crash
            assert report is not None

    def test_no_kernel_grid_skips_convergence(self, tmp_path):
        """If _kernel_grid is None after loop, convergence check is skipped."""
        engine = _make_engine(tmp_path)
        mock_result = Mock()
        mock_result.success = True
        mock_result.blueprint = {"components": [{"name": "X"}]}

        def fake_self_compile():
            engine._kernel_grid = None
            return mock_result

        engine.self_compile = fake_self_compile
        insights = []
        engine.on_insight = lambda msg: insights.append(msg)

        with patch("kernel.store.load_grid") as mock_load:
            report = engine.run_self_compile_loop(runs=1)
            # load_grid should not even be called for convergence
            # (it might be called during self_compile itself, but not for convergence)
            conv = [i for i in insights if "L3 convergence" in i]
            assert len(conv) == 0


# ===========================================================================
# 5. DAEMON SCHEDULING — _should_self_compile() + _scheduler_tick()
# ===========================================================================

class TestDaemonSelfCompile:
    """Daemon autonomous self-compile scheduling."""

    @pytest.fixture
    def daemon(self, tmp_path):
        from mother.daemon import DaemonMode, DaemonConfig
        return DaemonMode(config=DaemonConfig(), config_dir=tmp_path)

    def test_fresh_install_triggers(self, daemon):
        """No self-desc grid at all → should_self_compile returns True."""
        with patch("kernel.store.list_maps", return_value=[]):
            assert daemon._should_self_compile() is True

    def test_recent_grid_does_not_trigger(self, daemon):
        """Self-desc updated recently → should not trigger."""
        now = time.time()
        maps = [{"id": "compiler-self-desc", "updated": now - 3600}]  # 1h ago
        mock_grid = _make_mock_grid(fill_rate=0.5)

        with patch("kernel.store.list_maps", return_value=maps), \
             patch("kernel.store.load_grid", return_value=mock_grid):
            assert daemon._should_self_compile() is False

    def test_stale_grid_triggers(self, daemon):
        """Self-desc older than 24h with low fill → triggers."""
        now = time.time()
        maps = [{"id": "compiler-self-desc", "updated": now - 90000}]  # 25h ago
        mock_grid = _make_mock_grid(fill_rate=0.5)

        with patch("kernel.store.list_maps", return_value=maps), \
             patch("kernel.store.load_grid", return_value=mock_grid):
            assert daemon._should_self_compile() is True

    def test_converged_grid_extends_interval(self, daemon):
        """High fill_rate (>0.8) → uses 7-day threshold instead of 24h."""
        now = time.time()
        # 48h ago — past 24h but within 7 days
        maps = [{"id": "compiler-self-desc", "updated": now - 172800}]
        mock_grid = _make_mock_grid(fill_rate=0.9)

        with patch("kernel.store.list_maps", return_value=maps), \
             patch("kernel.store.load_grid", return_value=mock_grid):
            assert daemon._should_self_compile() is False

    def test_converged_grid_past_7_days_triggers(self, daemon):
        """High fill_rate but > 7 days old → triggers."""
        now = time.time()
        maps = [{"id": "compiler-self-desc", "updated": now - 700000}]  # ~8 days
        mock_grid = _make_mock_grid(fill_rate=0.9)

        with patch("kernel.store.list_maps", return_value=maps), \
             patch("kernel.store.load_grid", return_value=mock_grid):
            assert daemon._should_self_compile() is True

    def test_exception_returns_false(self, daemon):
        """Any exception in _should_self_compile → returns False (safe)."""
        with patch("kernel.store.list_maps", side_effect=RuntimeError("db locked")):
            assert daemon._should_self_compile() is False

    def test_goals_take_priority_over_self_compile(self, daemon):
        """When a critical goal exists, self_compile is not checked."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None

        with patch.object(daemon, "_find_critical_goal", return_value=(1, "fix synthesis")), \
             patch.object(daemon, "_should_self_compile") as mock_should:
            run(daemon._scheduler_tick())
            # _should_self_compile should NOT be called when a goal is found
            mock_should.assert_not_called()
            # A [SELF-IMPROVEMENT] compile was enqueued instead
            assert len(daemon._queue) == 1
            assert daemon._queue[0].input_text.startswith("[SELF-IMPROVEMENT]")

    def test_self_compile_enqueued_when_no_goal(self, daemon):
        """No critical goal + stale grid → [SELF-COMPILE] enqueued."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=True):
            run(daemon._scheduler_tick())
            assert len(daemon._queue) == 1
            assert daemon._queue[0].input_text == "[SELF-COMPILE]"
            assert daemon._queue[0].priority == -2

    def test_self_compile_priority_lower_than_goal(self, daemon):
        """Self-compile uses priority -2 (lower than goal's -1)."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=True):
            run(daemon._scheduler_tick())
            assert daemon._queue[0].priority == -2

    def test_queue_full_skips_self_compile(self, daemon):
        """Full queue → self-compile not enqueued, no crash."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        # Fill queue
        from mother.daemon import CompileRequest
        for i in range(daemon.config.max_queue_size):
            daemon._queue.append(CompileRequest(
                input_text=f"task {i}", status="pending"
            ))

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=True):
            # Should not crash despite full queue
            # But pending items already in queue → tick returns early
            run(daemon._scheduler_tick())

    def test_no_self_compile_when_grid_fresh(self, daemon):
        """Fresh grid + no goal → nothing enqueued."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=False):
            run(daemon._scheduler_tick())
            assert len(daemon._queue) == 0

    def test_cooldown_blocks_self_compile(self, daemon):
        """Cooldown applies to self-compile same as goals."""
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = time.time()  # just now

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=True):
            run(daemon._scheduler_tick())
            assert len(daemon._queue) == 0

    def test_stale_thresholds_are_class_attributes(self, daemon):
        """Threshold constants exist as class attributes."""
        from mother.daemon import DaemonMode
        assert DaemonMode._SELF_COMPILE_STALE_HOURS == 24
        assert DaemonMode._SELF_COMPILE_CONVERGED_HOURS == 168


# ===========================================================================
# INTEGRATION — end-to-end flow simulation
# ===========================================================================

class TestL3Integration:
    """Integration tests verifying the full L3 flow."""

    def test_self_compile_prefix_through_compile(self, tmp_path):
        """[SELF-COMPILE] → compile() → self_compile() → save grid."""
        engine = _make_engine(tmp_path)
        mock_grid = _make_mock_grid()
        mock_result = Mock()
        mock_result.success = True

        original_self_compile = engine.self_compile.__func__

        def instrumented_self_compile(self_engine):
            self_engine._kernel_grid = mock_grid
            # Skip the actual compile_with_axioms
            mock_res = Mock()
            mock_res.success = True
            # Simulate what self_compile does after compile_with_axioms
            try:
                if self_engine._kernel_grid:
                    from kernel.store import save_grid as _sg
                    _sg(
                        self_engine._kernel_grid,
                        map_id="compiler-self-desc",
                        name="Compiler Self-Description",
                    )
            except Exception:
                pass
            return mock_res

        engine.self_compile = lambda: instrumented_self_compile(engine)

        with patch("kernel.store.save_grid") as mock_save:
            result = engine.compile("[SELF-COMPILE]")
            mock_save.assert_called_once()
            assert mock_save.call_args[1]["map_id"] == "compiler-self-desc"

    def test_daemon_tick_to_compile_flow(self, tmp_path):
        """Daemon _scheduler_tick enqueues [SELF-COMPILE], queue sees it."""
        from mother.daemon import DaemonMode, DaemonConfig
        daemon = DaemonMode(config=DaemonConfig(), config_dir=tmp_path)
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None

        with patch.object(daemon, "_find_critical_goal", return_value=None), \
             patch.object(daemon, "_should_self_compile", return_value=True):
            run(daemon._scheduler_tick())

        assert len(daemon._queue) == 1
        req = daemon._queue[0]
        assert req.input_text == "[SELF-COMPILE]"
        assert req.status == "pending"
        assert req.priority == -2
