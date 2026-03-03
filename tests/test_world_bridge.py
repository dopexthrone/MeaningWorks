"""Tests for world grid bridge methods in mother/bridge.py (Phase 6-8)."""

import time
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile


class TestBridgeWorldGridMethods:
    """Test EngineBridge world grid methods."""

    def _make_bridge(self):
        from mother.bridge import EngineBridge
        return EngineBridge(provider="local")

    def test_bootstrap_world_grid(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                assert grid is not None
                assert grid.total_cells > 0

    def test_get_world_grid_when_none_exists(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                result = bridge.get_world_grid()
                assert result is None

    def test_bootstrap_then_get(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                loaded = bridge.get_world_grid()
                assert loaded is not None
                assert loaded.total_cells > 0

    def test_save_world_grid(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                bridge.save_world_grid(grid)
                loaded = bridge.get_world_grid()
                assert loaded is not None

    def test_fill_world_cell(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                result = bridge.fill_world_cell(
                    grid,
                    postcode_key="OBS.ENV.APP.WHAT.USR",
                    primitive="VS Code",
                    content="Editor open",
                    confidence=0.8,
                    source=(f"observation:screen:{time.time()}",),
                )
                assert result is not None
                cell = grid.get("OBS.ENV.APP.WHAT.USR")
                assert cell.is_filled
                assert cell.primitive == "VS Code"

    def test_score_world_candidates(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                from kernel.grid import INTENT_CONTRACT
                from kernel.ops import fill
                fill(grid, "OBS.ENV.APP.WHAT.USR", "screen",
                     "content", 0.8, source=(INTENT_CONTRACT,))

                candidates = bridge.score_world_candidates(grid)
                assert isinstance(candidates, list)

    def test_dispatch_from_cell(self):
        bridge = self._make_bridge()
        action = bridge.dispatch_from_cell("OBS.ENV.APP.WHAT.USR", 5.0, "screen")
        assert action is not None
        assert action.action_type == "perceive"

    def test_dispatch_from_cell_invalid(self):
        bridge = self._make_bridge()
        action = bridge.dispatch_from_cell("INVALID", 1.0)
        assert action is None  # should return None on error

    def test_apply_staleness_decay(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                # Fill an old observation
                from kernel.ops import fill
                old_time = time.time() - 600
                fill(grid, "OBS.ENV.APP.WHAT.USR", "old_screen",
                     "content", 0.8,
                     source=(f"observation:screen:{old_time}",))

                decayed = bridge.apply_staleness_decay(grid)
                assert decayed >= 1

    def test_world_grid_health(self):
        bridge = self._make_bridge()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()
                health = bridge.world_grid_health(grid)
                assert "total_cells" in health
                assert "seed_coverage" in health
                assert health["seed_coverage"] == 1.0


class TestPerceptionBridgeIntegration:
    """Test perception → world grid fill pipeline (Phase 7)."""

    def test_perception_to_fill_to_world_grid(self):
        """Full pipeline: perception event → bridge fill → grid cell."""
        from mother.bridge import EngineBridge
        from kernel.perception_bridge import perception_to_fill

        bridge = EngineBridge(provider="local")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()

                # Simulate perception event
                now = time.time()
                fill_params = perception_to_fill("screen", "VS Code - main.py", 0.8, now)
                result = bridge.fill_world_cell(grid, **fill_params)

                assert result is not None
                cell = grid.get("OBS.ENV.APP.WHAT.USR")
                assert cell.is_filled
                assert cell.primitive == "VS Code - main.py"

    def test_fusion_signal_to_world_grid(self):
        """Full pipeline: fusion signal → bridge fill → grid cell."""
        from mother.bridge import EngineBridge
        from kernel.perception_bridge import fusion_signal_to_fill

        bridge = EngineBridge(provider="local")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()

                fill_params = fusion_signal_to_fill(
                    "focused", 0.6, ("screen",), time.time()
                )
                result = bridge.fill_world_cell(grid, **fill_params)

                assert result is not None
                cell = grid.get("OBS.ENV.APP.HOW.USR")
                assert cell.is_filled
                assert "focused" in cell.primitive

    def test_environment_snapshot_batch_fill(self):
        """Batch fill from environment snapshot entries."""
        from mother.bridge import EngineBridge
        from kernel.perception_bridge import environment_snapshot_to_fills

        bridge = EngineBridge(provider="local")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kernel.store._DEFAULT_DB_DIR", Path(tmpdir)):
                grid = bridge.bootstrap_world_grid()

                entries = [
                    ("screen", "Terminal", 0.8, time.time()),
                    ("speech", "discussing", 0.7, time.time()),
                ]
                fills = environment_snapshot_to_fills(entries)

                for fp in fills:
                    bridge.fill_world_cell(grid, **fp)

                assert grid.get("OBS.ENV.APP.WHAT.USR").is_filled
                assert grid.get("OBS.USR.APP.WHAT.USR").is_filled


class TestDaemonMetaCells:
    """Test daemon META cell fills (Phase 8)."""

    def test_learned_pattern_to_meta_cell(self):
        """Simulate daemon filling MET.MEM cells from learned patterns."""
        from kernel.world_grid import bootstrap_world_grid
        from kernel.ops import fill
        import tempfile

        grid = bootstrap_world_grid()

        # Simulate daemon pattern recording
        now = time.time()
        result = fill(
            grid, "MET.MEM.DOM.WHAT.MTH",
            primitive="recurring_synthesis_failure",
            content="Components extracted but relationships missed in 3/5 compilations",
            confidence=0.7,
            source=(f"observation:compiler:{now}",),
        )
        assert result.status.value in ("ok", "revised")
        cell = grid.get("MET.MEM.DOM.WHAT.MTH")
        assert cell.is_filled
        assert "synthesis_failure" in cell.primitive

    def test_goal_to_meta_cell(self):
        """Simulate daemon filling MET.GOL cells from generated goals."""
        from kernel.world_grid import bootstrap_world_grid
        from kernel.ops import fill

        grid = bootstrap_world_grid()

        now = time.time()
        result = fill(
            grid, "MET.GOL.DOM.WHY.MTH",
            primitive="improve_relationship_extraction",
            content="Synthesis misses relationships in complex intents",
            confidence=0.5,
            source=(f"observation:feedback:{now}",),
        )
        assert result.status.value in ("ok", "revised")
        cell = grid.get("MET.GOL.DOM.WHY.MTH")
        assert cell.is_filled

    def test_meta_cells_use_observation_provenance(self):
        """Verify META cells pass AX1 provenance with observation: prefix."""
        from kernel.world_grid import bootstrap_world_grid
        from kernel.ops import fill

        grid = bootstrap_world_grid()

        # This should NOT be quarantined — observation: is valid provenance
        result = fill(
            grid, "MET.MEM.DOM.WHAT.MTH",
            primitive="test_pattern",
            content="test",
            confidence=0.6,
            source=(f"observation:compiler:{time.time()}",),
        )
        assert result.status.value != "quarantined"

    def test_revision_on_meta_refill(self):
        """Repeated fills to META cells preserve revision history."""
        from kernel.world_grid import bootstrap_world_grid
        from kernel.ops import fill

        grid = bootstrap_world_grid()

        now = time.time()
        fill(grid, "MET.MEM.DOM.WHAT.MTH",
             "pattern_v1", "first observation", 0.5,
             source=(f"observation:compiler:{now}",))

        fill(grid, "MET.MEM.DOM.WHAT.MTH",
             "pattern_v2", "updated observation", 0.7,
             source=(f"observation:compiler:{now + 1}",))

        cell = grid.get("MET.MEM.DOM.WHAT.MTH")
        assert cell.primitive == "pattern_v2"
        assert len(cell.revisions) >= 1


class TestWorldGridEndToEnd:
    """Full end-to-end: bootstrap → perceive → score → dispatch → act."""

    def test_perceive_score_dispatch_cycle(self):
        from kernel.world_grid import bootstrap_world_grid, apply_staleness_decay
        from kernel.navigator import score_world_candidates
        from kernel.dispatch import dispatch_from_cell
        from kernel.ops import fill

        # 1. Bootstrap
        grid = bootstrap_world_grid()

        # 2. Fill perception cells (simulate sensor data)
        now = time.time()
        fill(grid, "OBS.ENV.APP.WHAT.USR", "VS Code with main.py open",
             "User editing Python code", 0.85,
             source=(f"observation:screen:{now}",))

        # 3. Fill a project cell
        from kernel.grid import INTENT_CONTRACT
        fill(grid, "INT.PRJ.APP.WHAT.USR", "task-manager-app",
             "Building a task manager", 0.9,
             source=(INTENT_CONTRACT,))

        # 4. Score candidates
        candidates = score_world_candidates(grid)
        assert len(candidates) > 0

        # 5. Dispatch top candidate
        top = candidates[0]
        action = dispatch_from_cell(top.postcode_key, top.score)
        assert action.action_type in (
            "perceive", "observe_user", "compile", "execute_task",
            "reflect", "self_improve", "check_external", "check_schedule",
            "manage_agent", "observe",
        )

        # 6. Simulate time passing and decay
        future = now + 120
        decayed = apply_staleness_decay(grid, now=future)
        # Screen (60s half-life) should have decayed after 120s
        assert decayed >= 1

    def test_navigator_chooses_stale_observation(self):
        """When observation is stale, navigator should prioritize refresh."""
        from kernel.world_grid import bootstrap_world_grid
        from kernel.navigator import score_world_candidates
        from kernel.ops import fill

        grid = bootstrap_world_grid()

        # Fill observation cells - one fresh, one stale
        now = time.time()
        fill(grid, "OBS.ENV.APP.WHAT.USR", "VS Code",
             "Editor", 0.8,
             source=(f"observation:screen:{now}",))

        # Score with staleness info
        staleness_map = {"OBS.ENV.APP.WHAT.USR": 300.0}  # 5 min stale
        candidates = score_world_candidates(grid, staleness_map=staleness_map)

        # Stale screen should be near top
        screen = next(
            (c for c in candidates if c.postcode_key == "OBS.ENV.APP.WHAT.USR"),
            None
        )
        assert screen is not None
        assert "stale" in screen.reason

    def test_world_grid_vocabulary_valid(self):
        """All world grid postcodes parse successfully."""
        from kernel.cell import parse_postcode
        from kernel.world_grid import WORLD_SEED_CELLS
        from kernel.perception_bridge import MODALITY_POSTCODES

        for pk, _ in WORLD_SEED_CELLS:
            pc = parse_postcode(pk)
            assert pc is not None

        for modality, pk in MODALITY_POSTCODES.items():
            pc = parse_postcode(pk)
            assert pc is not None
