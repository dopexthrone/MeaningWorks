"""
Kernel integration tests — vocabulary expansion, store, ground truth, engine wiring.

Tests the kernel's connection to:
  1. Ground truth vocabulary (postcodes from MTH-USM-001 parse cleanly)
  2. Persistent storage (save/load/list/delete roundtrip)
  3. Ground truth loader (96 nodes → Grid)
  4. Engine integration (CompileResult.semantic_grid populated)
  5. Bridge integration (get_semantic_grid, get_semantic_nav)
"""

import json
import tempfile
from pathlib import Path

import pytest

from kernel.cell import (
    LAYERS, CONCERNS, SCOPES, SCOPE_SET, DIMENSIONS, DOMAINS,
    FillState, parse_postcode,
)


# ===========================================================================
# 1. Vocabulary expansion — ground truth postcodes parse
# ===========================================================================

class TestVocabularyExpansion:
    """All ground truth postcodes must parse without ValueError."""

    # Representative postcodes from every layer of the ground truth
    GROUND_TRUTH_POSTCODES = [
        # INT
        "INT.SEM.ECO.WHY.SFT",
        "INT.SEM.ECO.WHO.SFT",
        "INT.SEM.ECO.WHAT.SFT",
        "INT.SEM.ECO.HOW.SFT",
        "INT.SEM.ECO.IF.SFT",
        # SEM
        "SEM.SEM.ECO.WHAT.SFT",
        "SEM.SEM.APP.HOW.SFT",
        "SEM.SEM.ECO.WHY.ORG",
        "SEM.SEM.ECO.WHY.COG",
        # STR — uses ENM, SCH (new concerns)
        "STR.ENT.ECO.WHAT.SFT",
        "STR.REL.ECO.WHAT.SFT",
        "STR.ENM.ECO.WHAT.SFT",
        "STR.SCH.ECO.WHAT.SFT",
        # IDN — uses PRM (new concern), ACT
        "IDN.ACT.ECO.WHO.SFT",
        "IDN.PRM.ECO.WHO.SFT",
        # AGN — uses GOL (new concern)
        "AGN.ORC.ECO.WHO.SFT",
        "AGN.AGT.CMP.WHO.SFT",
        "AGN.MEM.CMP.WHO.SFT",
        "AGN.GOL.CMP.WHY.SFT",
        # STA
        "STA.BHV.CMP.WHEN.SFT",
        "STA.TRN.STP.WHEN.SFT",
        # TME — uses TMO, VRS (new concerns)
        "TME.TMO.CMP.WHEN.SFT",
        "TME.SCH.CMP.WHEN.SFT",
        "TME.VRS.ECO.WHEN.SFT",
        # EXC — uses GTE, LCK, RTY (new concerns)
        "EXC.FNC.ECO.HOW.SFT",
        "EXC.FNC.CMP.HOW.SFT",
        "EXC.FNC.STP.HOW.SFT",
        "EXC.GTE.ECO.HOW.SFT",
        "EXC.GTE.CMP.HOW.SFT",
        "EXC.LCK.CMP.HOW.SFT",
        "EXC.RTY.CMP.HOW.SFT",
        # DAT — new layer, uses TRF, COL (new concerns)
        "DAT.FLW.ECO.HOW.SFT",
        "DAT.TRF.CMP.HOW.SFT",
        "DAT.COL.ECO.WHAT.SFT",
        # SFX — new layer, uses WRT, EMT, RED (new concerns)
        "SFX.WRT.ECO.HOW.SFT",
        "SFX.EMT.ECO.HOW.SFT",
        "SFX.RED.ECO.HOW.SFT",
        # NET
        "NET.FLW.ECO.HOW.SFT",
        "NET.FLW.CMP.HOW.SFT",
        "NET.FLW.ECO.HOW.NET",
        # RES
        "RES.MET.ECO.HOW_MUCH.SFT",
        "RES.LMT.ECO.HOW_MUCH.SFT",
        # OBS — uses TRC, ALT (new concerns)
        "OBS.LOG.ECO.HOW.SFT",
        "OBS.MET.ECO.HOW.SFT",
        "OBS.TRC.CMP.HOW.SFT",
        "OBS.ALT.ECO.HOW.SFT",
        # CTR — uses CFG, PLY (new concern)
        "CTR.CFG.ECO.HOW.SFT",
        "CTR.PLY.ECO.HOW.SFT",
        # EMG — uses CND
        "EMG.CND.ECO.WHAT.SFT",
        # MET — uses INT, VRS, CNS, PRV
        "MET.INT.ECO.WHY.SFT",
        "MET.VRS.ECO.WHEN.SFT",
        "MET.CNS.ECO.IF.SFT",
        "MET.PRV.ECO.HOW.SFT",
    ]

    @pytest.mark.parametrize("postcode", GROUND_TRUTH_POSTCODES)
    def test_ground_truth_postcode_parses(self, postcode):
        """Every ground truth postcode must parse without error."""
        pc = parse_postcode(postcode)
        assert pc.key == postcode

    def test_new_layers_in_vocabulary(self):
        """DAT and SFX layers must be in the kernel vocabulary."""
        assert "DAT" in LAYERS
        assert "SFX" in LAYERS

    def test_new_concerns_in_vocabulary(self):
        """All 14 new concerns from ground truth must be present."""
        new_concerns = {"ENM", "PRM", "GOL", "TMO", "LCK", "RTY",
                        "TRF", "COL", "WRT", "EMT", "RED", "ALT",
                        "CFG", "TRC"}
        for c in new_concerns:
            assert c in CONCERNS, f"Missing concern: {c}"

    def test_original_vocabulary_intact(self):
        """Original kernel vocabulary still present."""
        originals = {"SEM", "ENT", "BHV", "FNC", "REL", "PLN", "MEM",
                     "ORC", "AGT", "ACT", "INT", "PRV", "CNS"}
        for c in originals:
            assert c in CONCERNS, f"Lost concern: {c}"

    def test_total_layer_count(self):
        """Should have 18 layers (16 original + DAT + SFX)."""
        assert len(LAYERS) == 18

    def test_total_concern_count(self):
        """Should have 45 concerns (26 original + 14 ground truth + 5 world-model)."""
        assert len(CONCERNS) == 45


# ===========================================================================
# 2. Persistent storage — save/load/list/delete
# ===========================================================================

class TestStore:
    """kernel/store.py roundtrip tests."""

    def _make_grid(self):
        from kernel.grid import Grid
        from kernel.cell import Cell
        grid = Grid()
        grid.set_intent("test intent", "INT.SEM.ECO.WHY.SFT", "intent")
        pc = parse_postcode("SEM.ENT.APP.WHAT.SFT")
        cell = Cell(
            postcode=pc,
            primitive="test_entity",
            content="test content",
            fill=FillState.F,
            confidence=0.92,
            connections=("INT.SEM.ECO.WHY.SFT",),
            source=("test",),
        )
        grid.put(cell)
        return grid

    def test_save_and_load(self):
        from kernel.store import save_grid, load_grid
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "test-1", "Test Map", Path(td))
            loaded = load_grid("test-1", Path(td))
            assert loaded is not None
            assert len(loaded.cells) == len(grid.cells)
            assert loaded.intent_text == grid.intent_text
            assert loaded.root == grid.root

    def test_load_nonexistent(self):
        from kernel.store import load_grid
        with tempfile.TemporaryDirectory() as td:
            assert load_grid("nonexistent", Path(td)) is None

    def test_list_maps(self):
        from kernel.store import save_grid, list_maps
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "map-a", "Alpha", Path(td))
            save_grid(grid, "map-b", "Beta", Path(td))
            maps = list_maps(Path(td))
            assert len(maps) == 2
            ids = {m["id"] for m in maps}
            assert ids == {"map-a", "map-b"}

    def test_delete_map(self):
        from kernel.store import save_grid, delete_map, list_maps
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "to-delete", "Doomed", Path(td))
            assert delete_map("to-delete", Path(td)) is True
            assert list_maps(Path(td)) == []

    def test_delete_nonexistent(self):
        from kernel.store import delete_map
        with tempfile.TemporaryDirectory() as td:
            assert delete_map("nope", Path(td)) is False

    def test_upsert_overwrites(self):
        from kernel.store import save_grid, load_grid, map_cell_count
        grid1 = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid1, "upsert", "V1", Path(td))
            assert map_cell_count("upsert", Path(td)) == 2  # intent + entity

            # Modify grid and re-save
            pc = parse_postcode("COG.BHV.DOM.HOW.COG")
            from kernel.cell import Cell
            grid1.put(Cell(postcode=pc, primitive="new", content="added", fill=FillState.P, confidence=0.5, source=("human:test",)))
            save_grid(grid1, "upsert", "V2", Path(td))
            assert map_cell_count("upsert", Path(td)) == 3

    def test_cell_count(self):
        from kernel.store import save_grid, map_cell_count
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "count-test", "Count", Path(td))
            assert map_cell_count("count-test", Path(td)) == 2

    def test_fill_state_preserved(self):
        from kernel.store import save_grid, load_grid
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "fill-test", "Fill", Path(td))
            loaded = load_grid("fill-test", Path(td))
            for pk, cell in grid.cells.items():
                assert loaded.cells[pk].fill == cell.fill

    def test_connections_preserved(self):
        from kernel.store import save_grid, load_grid
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "conn-test", "Conn", Path(td))
            loaded = load_grid("conn-test", Path(td))
            for pk, cell in grid.cells.items():
                assert set(loaded.cells[pk].connections) == set(cell.connections)

    def test_confidence_preserved(self):
        from kernel.store import save_grid, load_grid
        grid = self._make_grid()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "conf-test", "Conf", Path(td))
            loaded = load_grid("conf-test", Path(td))
            for pk, cell in grid.cells.items():
                assert loaded.cells[pk].confidence == cell.confidence


# ===========================================================================
# 3. Ground truth loader
# ===========================================================================

class TestGroundTruth:
    """kernel/ground_truth.py — MTH-USM-001 loads correctly."""

    def test_load_returns_grid(self):
        from kernel.ground_truth import load_ground_truth
        from kernel.grid import Grid
        grid = load_ground_truth()
        assert isinstance(grid, Grid)

    def test_node_count(self):
        """96 raw nodes collapse to ~54 unique postcodes."""
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        assert len(grid.cells) >= 50
        assert len(grid.cells) <= 60

    def test_layer_count(self):
        """All 16 ground truth layers activated."""
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        assert len(grid.activated_layers) == 16

    def test_all_16_layers_present(self):
        from kernel.ground_truth import load_ground_truth
        expected = {"INT", "SEM", "STR", "IDN", "AGN", "STA", "TME", "EXC",
                    "DAT", "SFX", "NET", "RES", "OBS", "CTR", "EMG", "MET"}
        grid = load_ground_truth()
        assert grid.activated_layers == expected

    def test_fill_state_distribution(self):
        """Majority filled, some partial, a few candidates."""
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        filled = sum(1 for c in grid.cells.values() if c.fill == FillState.F)
        partial = sum(1 for c in grid.cells.values() if c.fill == FillState.P)
        candidate = sum(1 for c in grid.cells.values() if c.fill == FillState.C)
        assert filled >= 45
        assert partial >= 3
        assert candidate >= 1

    def test_intent_set(self):
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        assert "intent" in grid.intent_text.lower() or "Mother" in grid.intent_text

    def test_root_set(self):
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        assert grid.root == "INT.SEM.ECO.WHY.SFT"

    def test_root_cell_exists(self):
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        root = grid.get(grid.root)
        assert root is not None
        assert root.fill == FillState.F
        assert root.confidence >= 0.95

    def test_connections_present(self):
        """Some cells should have connections."""
        from kernel.ground_truth import load_ground_truth
        grid = load_ground_truth()
        with_conns = sum(1 for c in grid.cells.values() if c.connections)
        assert with_conns >= 10

    def test_stats_function(self):
        from kernel.ground_truth import ground_truth_stats
        stats = ground_truth_stats()
        assert stats["map_id"] == "MTH-USM-001"
        assert stats["total_node_definitions"] >= 96
        assert stats["layer_count"] == 16
        assert stats["fill_rate"] >= 0.85

    def test_store_roundtrip(self):
        """Ground truth survives save/load cycle."""
        from kernel.ground_truth import load_ground_truth
        from kernel.store import save_grid, load_grid
        grid = load_ground_truth()
        with tempfile.TemporaryDirectory() as td:
            save_grid(grid, "MTH-USM-001", "Ground Truth", Path(td))
            loaded = load_grid("MTH-USM-001", Path(td))
            assert len(loaded.cells) == len(grid.cells)
            for pk in grid.cells:
                assert pk in loaded.cells

    def test_nav_serialization(self):
        """Ground truth grid serializes to nav format."""
        from kernel.ground_truth import load_ground_truth
        from kernel.nav import grid_to_nav
        grid = load_ground_truth()
        nav = grid_to_nav(grid)
        assert "INT.SEM.ECO.WHY.SFT" in nav
        assert "F 0." in nav or "F 1." in nav
        assert len(nav) > 500  # substantial output

    def test_emission(self):
        """Ground truth grid emits a manifest."""
        from kernel.ground_truth import load_ground_truth
        from kernel.emission import emit
        grid = load_ground_truth()
        manifest = emit(grid, force=True)
        assert manifest is not None
        assert len(manifest.nodes) > 0
        assert manifest.fill_rate > 0.5


# ===========================================================================
# 4. Engine integration — CompileResult has semantic_grid
# ===========================================================================

class TestEngineIntegration:
    """Engine CompileResult includes semantic_grid field."""

    def test_compile_result_has_semantic_grid_field(self):
        """CompileResult dataclass has the new field."""
        from core.engine import CompileResult
        r = CompileResult(success=True)
        assert hasattr(r, "semantic_grid")
        assert r.semantic_grid is None  # default

    def test_compile_result_accepts_semantic_grid(self):
        from core.engine import CompileResult
        data = {"nav": "test", "cells": 10, "layers": 3}
        r = CompileResult(success=True, semantic_grid=data)
        assert r.semantic_grid == data
        assert r.semantic_grid["cells"] == 10


# ===========================================================================
# 5. Bridge integration — convenience methods
# ===========================================================================

class TestBridgeIntegration:
    """Bridge semantic grid extraction methods."""

    def _make_result_like(self, grid_data=None):
        """Create a mock CompileResult-like object."""
        class MockResult:
            def __init__(self, sg):
                self.semantic_grid = sg
        return MockResult(grid_data)

    def test_get_semantic_grid_present(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        result = self._make_result_like({"nav": "test nav", "cells": 5})
        grid = bridge.get_semantic_grid(result)
        assert grid is not None
        assert grid["cells"] == 5

    def test_get_semantic_grid_none(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        result = self._make_result_like(None)
        assert bridge.get_semantic_grid(result) is None

    def test_get_semantic_nav_present(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        result = self._make_result_like({"nav": "INT.SEM.ECO | F 0.99 | intent"})
        nav = bridge.get_semantic_nav(result)
        assert "INT.SEM.ECO" in nav

    def test_get_semantic_nav_none(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        result = self._make_result_like(None)
        assert bridge.get_semantic_nav(result) == ""

    def test_get_semantic_nav_no_attr(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)

        class Bare:
            pass

        assert bridge.get_semantic_grid(Bare()) is None
        assert bridge.get_semantic_nav(Bare()) == ""
