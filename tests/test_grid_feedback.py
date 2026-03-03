"""Tests for grid feedback loop — boost_cell_confidence, record_build_outcome,
and bridge.update_grid_after_build."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import pytest

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid
from kernel.store import (
    boost_cell_confidence,
    cell_history,
    load_grid,
    record_build_outcome,
    save_grid,
)


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_dir(tmp_path):
    """Temporary db directory."""
    return tmp_path


@pytest.fixture
def grid_with_cells():
    """Grid with a few cells at varying confidence levels."""
    g = Grid()
    g.intent_text = "test grid for feedback"
    g.activated_layers.add("SEM")
    g.activated_layers.add("COG")

    pc1 = parse_postcode("SEM.ENT.DOM.WHAT.SFT")
    pc2 = parse_postcode("COG.BHV.APP.HOW.SFT")
    pc3 = parse_postcode("SEM.FNC.DOM.HOW.SFT")

    g.cells[pc1.key] = Cell(
        postcode=pc1, primitive="entity-model",
        content="test", fill=FillState.P, confidence=0.25,
    )
    g.cells[pc2.key] = Cell(
        postcode=pc2, primitive="behavior-flow",
        content="test", fill=FillState.P, confidence=0.40,
    )
    g.cells[pc3.key] = Cell(
        postcode=pc3, primitive="function-handler",
        content="test", fill=FillState.F, confidence=0.90,
    )
    return g


# ---------------------------------------------------------------------------
# boost_cell_confidence
# ---------------------------------------------------------------------------

class TestBoostCellConfidence:
    def test_boosts_matching_cells(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        updated = boost_cell_confidence(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"),
            delta=0.15,
            db_dir=db_dir,
        )
        assert updated == 2

        # Verify new confidence values
        grid = load_grid("test-map", db_dir=db_dir)
        assert grid is not None
        assert grid.cells["SEM.ENT.DOM.WHAT.SFT"].confidence == pytest.approx(0.40)
        assert grid.cells["COG.BHV.APP.HOW.SFT"].confidence == pytest.approx(0.55)

    def test_caps_at_max_confidence(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        updated = boost_cell_confidence(
            "test-map",
            postcodes=("SEM.FNC.DOM.HOW.SFT",),
            delta=0.15,
            max_confidence=0.95,
            db_dir=db_dir,
        )
        assert updated == 1

        grid = load_grid("test-map", db_dir=db_dir)
        assert grid.cells["SEM.FNC.DOM.HOW.SFT"].confidence == pytest.approx(0.95)

    def test_negative_delta_floors(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        updated = boost_cell_confidence(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            delta=-0.30,
            db_dir=db_dir,
        )
        assert updated == 1

        grid = load_grid("test-map", db_dir=db_dir)
        assert grid.cells["SEM.ENT.DOM.WHAT.SFT"].confidence == pytest.approx(0.05)

    def test_ignores_non_matching_postcodes(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        updated = boost_cell_confidence(
            "test-map",
            postcodes=("NET.ENT.DOM.WHAT.SFT",),
            delta=0.15,
            db_dir=db_dir,
        )
        assert updated == 0

    def test_empty_postcodes_returns_zero(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)
        updated = boost_cell_confidence("test-map", postcodes=(), db_dir=db_dir)
        assert updated == 0

    def test_nonexistent_map_returns_zero(self, db_dir):
        updated = boost_cell_confidence(
            "no-such-map", postcodes=("SEM.ENT.DOM.WHAT.SFT",), db_dir=db_dir,
        )
        assert updated == 0

    def test_records_fill_history(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        boost_cell_confidence(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            delta=0.15,
            db_dir=db_dir,
        )

        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        # Should have at least 2 entries: initial save + boost
        assert len(history) >= 2
        latest = history[-1]
        assert latest["agent"] == "self_build"
        assert latest["confidence"] == pytest.approx(0.40)

    def test_no_change_when_already_at_max(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        updated = boost_cell_confidence(
            "test-map",
            postcodes=("SEM.FNC.DOM.HOW.SFT",),
            delta=0.15,
            max_confidence=0.90,  # Cell is already at 0.90
            db_dir=db_dir,
        )
        # 0.90 + 0.15 = 1.05, capped to 0.90 — same as current, so no update
        assert updated == 0


# ---------------------------------------------------------------------------
# record_build_outcome
# ---------------------------------------------------------------------------

class TestRecordBuildOutcome:
    def test_records_success(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        record_build_outcome(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=True,
            build_description="Strengthened entity model",
            db_dir=db_dir,
        )

        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        latest = history[-1]
        assert latest["agent"] == "self_build"
        assert latest["fill_state"] == "build_success"
        assert "Strengthened" in latest["content"]

    def test_records_failure(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        record_build_outcome(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=False,
            build_description="Tests failed",
            db_dir=db_dir,
        )

        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        latest = history[-1]
        assert latest["fill_state"] == "build_failure"

    def test_empty_postcodes_noop(self, db_dir):
        # Should not raise
        record_build_outcome("test-map", postcodes=(), success=True, db_dir=db_dir)

    def test_truncates_long_description(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        long_desc = "x" * 1000
        record_build_outcome(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=True,
            build_description=long_desc,
            db_dir=db_dir,
        )

        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        latest = history[-1]
        assert len(latest["content"]) <= 500

    def test_records_multiple_postcodes(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        record_build_outcome(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT"),
            success=True,
            db_dir=db_dir,
        )

        h1 = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        h2 = cell_history("test-map", "COG.BHV.APP.HOW.SFT", db_dir=db_dir)
        assert len(h1) >= 2
        assert len(h2) >= 2

    def test_source_json_contains_build_tag(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        record_build_outcome(
            "test-map",
            postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=True,
            db_dir=db_dir,
        )

        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        latest = history[-1]
        assert "build:build_success" in latest["source"]


# ---------------------------------------------------------------------------
# Integration: boost + record together
# ---------------------------------------------------------------------------

class TestBoostAndRecordIntegration:
    def test_combined_success_flow(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        postcodes = ("SEM.ENT.DOM.WHAT.SFT", "COG.BHV.APP.HOW.SFT")

        # Boost confidence
        updated = boost_cell_confidence(
            "test-map", postcodes=postcodes, delta=0.15, db_dir=db_dir,
        )
        assert updated == 2

        # Record outcome
        record_build_outcome(
            "test-map", postcodes=postcodes, success=True,
            build_description="Improved entity+behavior", db_dir=db_dir,
        )

        # Verify grid state
        grid = load_grid("test-map", db_dir=db_dir)
        assert grid.cells["SEM.ENT.DOM.WHAT.SFT"].confidence == pytest.approx(0.40)

        # Verify history has both entries
        history = cell_history("test-map", "SEM.ENT.DOM.WHAT.SFT", db_dir=db_dir)
        agents = [h["agent"] for h in history]
        assert agents.count("self_build") >= 2  # boost + record

    def test_combined_failure_flow(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        postcodes = ("SEM.ENT.DOM.WHAT.SFT",)

        # Penalize confidence
        boost_cell_confidence(
            "test-map", postcodes=postcodes, delta=-0.10, db_dir=db_dir,
        )

        # Record failure
        record_build_outcome(
            "test-map", postcodes=postcodes, success=False,
            build_description="Tests failed", db_dir=db_dir,
        )

        grid = load_grid("test-map", db_dir=db_dir)
        assert grid.cells["SEM.ENT.DOM.WHAT.SFT"].confidence == pytest.approx(0.15)

    def test_multiple_boosts_accumulate(self, db_dir, grid_with_cells):
        save_grid(grid_with_cells, "test-map", db_dir=db_dir)

        postcodes = ("SEM.ENT.DOM.WHAT.SFT",)

        boost_cell_confidence("test-map", postcodes=postcodes, delta=0.10, db_dir=db_dir)
        boost_cell_confidence("test-map", postcodes=postcodes, delta=0.10, db_dir=db_dir)

        grid = load_grid("test-map", db_dir=db_dir)
        # 0.25 + 0.10 + 0.10 = 0.45
        assert grid.cells["SEM.ENT.DOM.WHAT.SFT"].confidence == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# bridge.update_grid_after_build (unit test, mocked store)
# ---------------------------------------------------------------------------

class TestBridgeUpdateGridAfterBuild:
    def test_calls_boost_and_record(self, monkeypatch):
        """Test that bridge method calls the right store functions."""
        from mother.bridge import EngineBridge

        calls = []

        def mock_boost(map_id, postcodes, delta, **kw):
            calls.append(("boost", map_id, postcodes, delta))
            return len(postcodes)

        def mock_record(map_id, postcodes, success, **kw):
            calls.append(("record", map_id, postcodes, success))

        monkeypatch.setattr("kernel.store.boost_cell_confidence", mock_boost)
        monkeypatch.setattr("kernel.store.record_build_outcome", mock_record)

        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "mock"

        result = run(bridge.update_grid_after_build(
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=True,
            build_description="test build",
        ))

        # Should have called boost + record for both map_ids
        boost_calls = [c for c in calls if c[0] == "boost"]
        record_calls = [c for c in calls if c[0] == "record"]
        assert len(boost_calls) == 2  # compiler-self-desc + session
        assert len(record_calls) == 2
        assert all(c[3] == 0.15 for c in boost_calls)  # positive delta
        assert result == 2  # 1 postcode × 2 maps

    def test_failure_uses_negative_delta(self, monkeypatch):
        from mother.bridge import EngineBridge

        calls = []

        def mock_boost(map_id, postcodes, delta, **kw):
            calls.append(("boost", delta))
            return 0

        def mock_record(map_id, postcodes, success, **kw):
            pass

        monkeypatch.setattr("kernel.store.boost_cell_confidence", mock_boost)
        monkeypatch.setattr("kernel.store.record_build_outcome", mock_record)

        bridge = EngineBridge.__new__(EngineBridge)
        bridge._provider = "mock"

        run(bridge.update_grid_after_build(
            target_postcodes=("SEM.ENT.DOM.WHAT.SFT",),
            success=False,
        ))

        assert all(c[1] == -0.10 for c in calls)
