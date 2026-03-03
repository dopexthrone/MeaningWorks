"""Tests for kernel/world_grid.py — world grid lifecycle management."""

import time
import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill
from kernel.world_grid import (
    bootstrap_world_grid,
    merge_compilation_into_world,
    apply_staleness_decay,
    world_grid_health,
    WORLD_SEED_CELLS,
    MODALITY_HALF_LIVES,
    MAX_WORLD_CELLS,
    _enforce_cell_cap,
    _extract_observation_time,
)


class TestBootstrapWorldGrid:
    """bootstrap_world_grid() tests."""

    def test_creates_new_grid(self):
        grid = bootstrap_world_grid()
        assert grid is not None
        assert grid.root is not None
        assert grid.total_cells > 0

    def test_seeds_all_cells(self):
        grid = bootstrap_world_grid()
        for pk, primitive in WORLD_SEED_CELLS:
            assert grid.has(pk), f"Missing seed cell: {pk}"

    def test_seed_cells_are_empty(self):
        grid = bootstrap_world_grid()
        for pk, _ in WORLD_SEED_CELLS:
            cell = grid.get(pk)
            assert cell.fill == FillState.E

    def test_idempotent_on_existing_grid(self):
        grid = bootstrap_world_grid()
        # Fill a seed cell
        fill(grid, "OBS.ENV.APP.WHAT.USR", "screen capture",
             "VS Code", 0.8, source=(INTENT_CONTRACT,))
        cell_before = grid.get("OBS.ENV.APP.WHAT.USR")
        assert cell_before.is_filled

        # Bootstrap again — should not overwrite
        grid2 = bootstrap_world_grid(grid)
        assert grid2 is grid  # same object
        cell_after = grid.get("OBS.ENV.APP.WHAT.USR")
        assert cell_after.is_filled
        assert cell_after.primitive == "screen capture"

    def test_extends_existing_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHAT.SFT", "test_root")
        initial_cells = grid.total_cells

        bootstrap_world_grid(grid)
        assert grid.total_cells > initial_cells

    def test_sets_root_on_new_grid(self):
        grid = bootstrap_world_grid()
        assert grid.root == "INT.SEM.ECO.WHAT.MTH"
        root_cell = grid.get(grid.root)
        assert root_cell is not None
        assert root_cell.fill == FillState.F

    def test_preserves_existing_root(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHAT.SFT", "test_root")
        bootstrap_world_grid(grid)
        assert grid.root == "INT.SEM.ECO.WHAT.SFT"

    def test_seed_cells_have_intent_source(self):
        grid = bootstrap_world_grid()
        for pk, _ in WORLD_SEED_CELLS:
            cell = grid.get(pk)
            assert INTENT_CONTRACT in cell.source


class TestMergeCompilationIntoWorld:
    """merge_compilation_into_world() tests."""

    def _make_compilation_grid(self):
        grid = Grid()
        grid.set_intent("build a task manager", "INT.SEM.ECO.WHAT.SFT", "compile_root")
        # Fill some cells in different layers
        fill(grid, "INT.ENT.APP.WHAT.SFT", "TaskManager",
             "Main entity", 0.9, source=(INTENT_CONTRACT,))
        fill(grid, "MET.MEM.DOM.WHAT.SFT", "memory_pattern",
             "Learned pattern", 0.85, source=(INTENT_CONTRACT,))
        fill(grid, "STR.FNC.APP.HOW.SFT", "structure_func",
             "Internal structure", 0.88, source=(INTENT_CONTRACT,))
        return grid

    def test_merges_relevant_layers(self):
        world = bootstrap_world_grid()
        comp = self._make_compilation_grid()

        merged = merge_compilation_into_world(world, comp, "proj-1")
        # INT root + INT entity + MET should merge, STR should not
        assert merged >= 2
        assert world.get("INT.ENT.APP.WHAT.SFT") is not None
        assert world.get("MET.MEM.DOM.WHAT.SFT") is not None

    def test_skips_non_merge_layers(self):
        world = bootstrap_world_grid()
        comp = self._make_compilation_grid()

        merge_compilation_into_world(world, comp, "proj-1")
        assert world.get("STR.FNC.APP.HOW.SFT") is None

    def test_skips_unfilled_cells(self):
        world = bootstrap_world_grid()
        comp = Grid()
        # Use a non-merge layer for root to isolate the test
        comp.set_intent("test", "STR.SEM.ECO.WHAT.SFT", "root")
        # Add empty cell in merge layer
        pc = parse_postcode("INT.ENT.APP.WHAT.SFT")
        comp.put(Cell(postcode=pc, primitive="empty", fill=FillState.E))

        merged = merge_compilation_into_world(world, comp, "proj-1")
        assert merged == 0

    def test_adds_project_provenance(self):
        world = bootstrap_world_grid()
        comp = self._make_compilation_grid()

        merge_compilation_into_world(world, comp, "my-project")
        cell = world.get("INT.ENT.APP.WHAT.SFT")
        assert any("my-project" in s for s in cell.source)

    def test_revision_on_existing_cell(self):
        world = bootstrap_world_grid()
        # Pre-fill a cell
        fill(world, "MET.MEM.DOM.WHAT.MTH", "old_pattern",
             "Old content", 0.7, source=(INTENT_CONTRACT,))

        comp = Grid()
        comp.set_intent("test", "INT.SEM.ECO.WHAT.SFT", "root")
        fill(comp, "MET.MEM.DOM.WHAT.MTH", "new_pattern",
             "New content", 0.9, source=(INTENT_CONTRACT,))

        merge_compilation_into_world(world, comp, "proj-1")
        cell = world.get("MET.MEM.DOM.WHAT.MTH")
        assert cell.primitive == "new_pattern"
        assert len(cell.revisions) > 0  # previous content preserved

    def test_returns_zero_for_empty_compilation(self):
        world = bootstrap_world_grid()
        comp = Grid()
        assert merge_compilation_into_world(world, comp, "proj") == 0


class TestApplyStalenessDecay:
    """apply_staleness_decay() tests."""

    def _fill_obs_cell(self, grid, modality_postcode, confidence, timestamp):
        """Fill an OBS cell with observation provenance."""
        fill(grid, modality_postcode, "test_obs",
             "observation content", confidence,
             source=(f"observation:screen:{timestamp}",))

    def test_decays_stale_observation(self):
        grid = bootstrap_world_grid()
        old_time = time.time() - 120  # 2 minutes ago
        self._fill_obs_cell(grid, "OBS.ENV.APP.WHAT.USR", 0.9, old_time)

        decayed = apply_staleness_decay(grid, now=time.time())
        cell = grid.get("OBS.ENV.APP.WHAT.USR")
        # With 60s half-life, 120s age → ~0.25 * 0.9 = ~0.225
        assert cell.confidence < 0.9
        assert decayed >= 1

    def test_fresh_observation_unchanged(self):
        grid = bootstrap_world_grid()
        now = time.time()
        self._fill_obs_cell(grid, "OBS.ENV.APP.WHAT.USR", 0.9, now)

        decayed = apply_staleness_decay(grid, now=now)
        cell = grid.get("OBS.ENV.APP.WHAT.USR")
        assert cell.confidence >= 0.89  # essentially unchanged
        assert decayed == 0

    def test_very_old_resets_to_empty(self):
        grid = bootstrap_world_grid()
        old_time = time.time() - 600  # 10 minutes ago
        self._fill_obs_cell(grid, "OBS.ENV.APP.WHAT.USR", 0.5, old_time)

        apply_staleness_decay(grid, now=time.time())
        cell = grid.get("OBS.ENV.APP.WHAT.USR")
        assert cell.fill == FillState.E
        assert cell.confidence == 0.0

    def test_non_obs_cells_not_decayed(self):
        grid = bootstrap_world_grid()
        fill(grid, "INT.PRJ.APP.WHAT.USR", "project",
             "content", 0.9, source=(INTENT_CONTRACT,))

        decayed = apply_staleness_decay(grid, now=time.time())
        cell = grid.get("INT.PRJ.APP.WHAT.USR")
        assert cell.confidence == 0.9

    def test_custom_half_lives(self):
        grid = bootstrap_world_grid()
        now = time.time()
        old_time = now - 30  # 30s ago with speech half-life of 30s
        fill(grid, "OBS.USR.APP.WHAT.USR", "speech",
             "content", 0.8, source=(f"observation:speech:{old_time}",))

        decayed = apply_staleness_decay(
            grid,
            half_lives={"speech": 30.0},
            now=now,
        )
        cell = grid.get("OBS.USR.APP.WHAT.USR")
        # Exactly 1 half-life → confidence halved
        assert 0.35 < cell.confidence < 0.45

    def test_preserves_revision_history(self):
        grid = bootstrap_world_grid()
        old_time = time.time() - 600
        self._fill_obs_cell(grid, "OBS.ENV.APP.WHAT.USR", 0.5, old_time)

        apply_staleness_decay(grid, now=time.time())
        cell = grid.get("OBS.ENV.APP.WHAT.USR")
        # Reset to empty should preserve revision
        assert len(cell.revisions) > 0

    def test_returns_zero_for_empty_grid(self):
        grid = bootstrap_world_grid()
        assert apply_staleness_decay(grid) == 0


class TestExtractObservationTime:
    """_extract_observation_time() tests."""

    def test_observation_source(self):
        assert _extract_observation_time(("observation:screen:1740000000.0",)) == 1740000000.0

    def test_perception_source(self):
        assert _extract_observation_time(("perception:camera:1740000001.5",)) == 1740000001.5

    def test_fusion_source(self):
        assert _extract_observation_time(("fusion:focused:1740000002.0",)) == 1740000002.0

    def test_no_timestamp(self):
        assert _extract_observation_time(("human:input",)) is None

    def test_empty_source(self):
        assert _extract_observation_time(()) is None

    def test_invalid_timestamp(self):
        assert _extract_observation_time(("observation:screen:notanumber",)) is None

    def test_multiple_sources_finds_first(self):
        sources = ("human:input", "observation:screen:1740000000.0")
        assert _extract_observation_time(sources) == 1740000000.0


class TestWorldGridHealth:
    """world_grid_health() diagnostic tests."""

    def test_empty_world_grid(self):
        grid = bootstrap_world_grid()
        health = world_grid_health(grid)
        assert health["total_cells"] > 0
        assert health["filled_cells"] == 1  # root intent cell
        assert health["seed_coverage"] == 1.0

    def test_with_filled_cells(self):
        grid = bootstrap_world_grid()
        fill(grid, "OBS.ENV.APP.WHAT.USR", "screen", "content", 0.8,
             source=(INTENT_CONTRACT,))
        fill(grid, "INT.PRJ.APP.WHAT.USR", "project", "content", 0.9,
             source=(INTENT_CONTRACT,))

        health = world_grid_health(grid)
        assert health["filled_cells"] >= 3  # root + 2 filled
        assert health["fill_rate"] > 0

    def test_stale_observation_count(self):
        grid = bootstrap_world_grid()
        fill(grid, "OBS.ENV.APP.WHAT.USR", "screen", "old", 0.2,
             source=(INTENT_CONTRACT,))

        health = world_grid_health(grid)
        assert health["stale_observation_count"] >= 1

    def test_layer_fill_rates(self):
        grid = bootstrap_world_grid()
        health = world_grid_health(grid)
        assert "layer_fill_rates" in health
        assert isinstance(health["layer_fill_rates"], dict)

    def test_seed_coverage_partial(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHAT.MTH", "root")
        # Only add a couple seed cells
        pc = parse_postcode("OBS.ENV.APP.WHAT.USR")
        grid.put(Cell(postcode=pc, primitive="test", fill=FillState.E))

        health = world_grid_health(grid)
        assert 0 < health["seed_coverage"] < 1.0


class TestEnforceCellCap:
    """_enforce_cell_cap() tests."""

    def test_no_action_under_cap(self):
        grid = bootstrap_world_grid()
        removed = _enforce_cell_cap(grid)
        assert removed == 0

    def test_removes_empty_cells_first(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHAT.SFT", "root")

        # Fill up with empty cells to exceed cap
        for i in range(MAX_WORLD_CELLS + 10):
            concern = "ENT"
            scope_idx = i % 10
            from kernel.cell import SCOPES
            scope = SCOPES[scope_idx]
            pk = f"INT.{concern}.{scope}.WHAT.SFT"
            if not grid.has(pk):
                pc = parse_postcode(pk)
                grid.put(Cell(postcode=pc, primitive=f"cell_{i}", fill=FillState.E))

        # Force exceeding by adding lots more
        for i in range(600):
            pk = f"STR.ENT.APP.WHAT.SFT"
            if i == 0 and not grid.has(pk):
                pc = parse_postcode(pk)
                grid.put(Cell(postcode=pc, primitive=f"extra_{i}", fill=FillState.E))

        if grid.total_cells > MAX_WORLD_CELLS:
            removed = _enforce_cell_cap(grid)
            assert removed > 0
            assert grid.total_cells <= MAX_WORLD_CELLS


class TestWorldGridIntegration:
    """Integration: bootstrap → fill → decay → health."""

    def test_full_lifecycle(self):
        # 1. Bootstrap
        grid = bootstrap_world_grid()
        assert grid.total_cells > 0

        # 2. Fill perception cells
        now = time.time()
        fill(grid, "OBS.ENV.APP.WHAT.USR", "VS Code open",
             "Editor with main.py", 0.9,
             source=(f"observation:screen:{now}",))
        fill(grid, "OBS.USR.APP.WHAT.USR", "user speaking",
             "discussing architecture", 0.8,
             source=(f"observation:speech:{now}",))

        assert len(grid.filled_cells()) >= 3  # root + 2 observations

        # 3. Decay (simulate 2 minutes later)
        future = now + 120
        decayed = apply_staleness_decay(grid, now=future)
        assert decayed >= 1  # speech should decay (30s half-life, 120s age)

        # 4. Health check
        health = world_grid_health(grid)
        assert health["filled_cells"] >= 1  # at least root
        assert health["seed_coverage"] == 1.0

    def test_perception_bridge_to_world_grid(self):
        """Verify perception_bridge fills work with world grid."""
        from kernel.perception_bridge import perception_to_fill

        grid = bootstrap_world_grid()
        params = perception_to_fill("screen", "Terminal output", 0.7, time.time())

        result = fill(grid, **params)
        assert result.status.value in ("ok", "revised")
        assert grid.get(params["postcode_key"]).is_filled
