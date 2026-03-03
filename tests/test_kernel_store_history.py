"""Tests for kernel/store.py — append-only fill history."""

import json
import time
from pathlib import Path

import pytest

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid
from kernel.store import (
    save_grid,
    load_grid,
    cell_history,
    cell_version_count,
    history_stats,
    _db_path,
    _ensure_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(cells=None, intent="test intent"):
    g = Grid()
    g.intent_text = intent
    if cells:
        for pc_str, content, fill, confidence in cells:
            pc = parse_postcode(pc_str)
            cell = Cell(
                postcode=pc,
                primitive=content.split()[0] if content else "unknown",
                content=content,
                fill=fill,
                confidence=confidence,
                source=("test",),
            )
            g.activated_layers.add(pc.layer)
            g.cells[pc.key] = cell
    return g


PC1 = "INT.ENT.ECO.WHAT.SFT"
PC2 = "SEM.BHV.APP.HOW.SFT"
PC3 = "ORG.FNC.DOM.WHY.SFT"


# ---------------------------------------------------------------------------
# Basic history recording
# ---------------------------------------------------------------------------

class TestFillHistory:
    def test_save_creates_history(self, tmp_path):
        grid = _make_grid([
            (PC1, "User component", FillState.F, 0.9),
        ])
        save_grid(grid, "map-1", db_dir=tmp_path)

        history = cell_history("map-1", PC1, db_dir=tmp_path)
        assert len(history) == 1
        assert history[0]["version"] == 1
        assert history[0]["content"] == "User component"
        assert history[0]["confidence"] == 0.9
        assert history[0]["fill_state"] == "F"

    def test_multiple_saves_append_versions(self, tmp_path):
        grid1 = _make_grid([(PC1, "Version 1", FillState.P, 0.5)])
        save_grid(grid1, "map-1", db_dir=tmp_path)

        grid2 = _make_grid([(PC1, "Version 2", FillState.F, 0.9)])
        save_grid(grid2, "map-1", db_dir=tmp_path)

        history = cell_history("map-1", PC1, db_dir=tmp_path)
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[0]["content"] == "Version 1"
        assert history[1]["version"] == 2
        assert history[1]["content"] == "Version 2"

    def test_history_never_deleted_on_resave(self, tmp_path):
        grid = _make_grid([(PC1, "v1", FillState.P, 0.3)])
        save_grid(grid, "map-1", db_dir=tmp_path)

        # Save again with different content (cells table is replaced, history is not)
        grid2 = _make_grid([(PC1, "v2", FillState.F, 0.9)])
        save_grid(grid2, "map-1", db_dir=tmp_path)

        # Live snapshot only has v2
        loaded = load_grid("map-1", db_dir=tmp_path)
        assert loaded.cells[PC1].content == "v2"

        # History has both
        history = cell_history("map-1", PC1, db_dir=tmp_path)
        assert len(history) == 2

    def test_multiple_cells_tracked_independently(self, tmp_path):
        grid = _make_grid([
            (PC1, "A", FillState.F, 0.8),
            (PC2, "B", FillState.P, 0.4),
        ])
        save_grid(grid, "map-1", db_dir=tmp_path)

        h1 = cell_history("map-1", PC1, db_dir=tmp_path)
        h2 = cell_history("map-1", PC2, db_dir=tmp_path)
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["content"] == "A"
        assert h2[0]["content"] == "B"

    def test_history_has_timestamp(self, tmp_path):
        before = time.time()
        grid = _make_grid([(PC1, "content", FillState.F, 0.9)])
        save_grid(grid, "map-1", db_dir=tmp_path)
        after = time.time()

        history = cell_history("map-1", PC1, db_dir=tmp_path)
        assert before <= history[0]["timestamp"] <= after

    def test_history_has_source(self, tmp_path):
        grid = _make_grid([(PC1, "content", FillState.F, 0.9)])
        save_grid(grid, "map-1", db_dir=tmp_path)

        history = cell_history("map-1", PC1, db_dir=tmp_path)
        assert history[0]["source"] == ["test"]

    def test_different_maps_isolated(self, tmp_path):
        grid1 = _make_grid([(PC1, "map1", FillState.F, 0.9)])
        grid2 = _make_grid([(PC1, "map2", FillState.F, 0.8)])
        save_grid(grid1, "map-a", db_dir=tmp_path)
        save_grid(grid2, "map-b", db_dir=tmp_path)

        ha = cell_history("map-a", PC1, db_dir=tmp_path)
        hb = cell_history("map-b", PC1, db_dir=tmp_path)
        assert len(ha) == 1
        assert len(hb) == 1
        assert ha[0]["content"] == "map1"
        assert hb[0]["content"] == "map2"


# ---------------------------------------------------------------------------
# cell_version_count
# ---------------------------------------------------------------------------

class TestCellVersionCount:
    def test_zero_for_nonexistent(self, tmp_path):
        assert cell_version_count("nope", "nope", db_dir=tmp_path) == 0

    def test_counts_correctly(self, tmp_path):
        grid = _make_grid([(PC1, "v1", FillState.P, 0.3)])
        save_grid(grid, "map-1", db_dir=tmp_path)
        assert cell_version_count("map-1", PC1, db_dir=tmp_path) == 1

        grid2 = _make_grid([(PC1, "v2", FillState.F, 0.9)])
        save_grid(grid2, "map-1", db_dir=tmp_path)
        assert cell_version_count("map-1", PC1, db_dir=tmp_path) == 2

    def test_nonexistent_path(self, tmp_path):
        assert cell_version_count("x", "y", db_dir=tmp_path / "nonexistent") == 0


# ---------------------------------------------------------------------------
# history_stats
# ---------------------------------------------------------------------------

class TestHistoryStats:
    def test_empty(self, tmp_path):
        stats = history_stats("nonexistent", db_dir=tmp_path)
        assert stats["total_versions"] == 0

    def test_aggregates(self, tmp_path):
        grid = _make_grid([
            (PC1, "A", FillState.F, 0.8),
            (PC2, "B", FillState.P, 0.4),
        ])
        save_grid(grid, "map-1", db_dir=tmp_path)

        stats = history_stats("map-1", db_dir=tmp_path)
        assert stats["total_versions"] == 2
        assert stats["unique_cells"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.6, abs=0.01)
        assert stats["max_version"] == 1

    def test_after_multiple_saves(self, tmp_path):
        for i in range(3):
            grid = _make_grid([(PC1, f"v{i}", FillState.F, 0.5 + i * 0.1)])
            save_grid(grid, "map-1", db_dir=tmp_path)

        stats = history_stats("map-1", db_dir=tmp_path)
        assert stats["total_versions"] == 3
        assert stats["unique_cells"] == 1
        assert stats["max_version"] == 3

    def test_nonexistent_path(self, tmp_path):
        stats = history_stats("x", db_dir=tmp_path / "nonexistent")
        assert stats["total_versions"] == 0


# ---------------------------------------------------------------------------
# Schema idempotency
# ---------------------------------------------------------------------------

class TestSchemaIdempotency:
    def test_fills_history_table_exists(self, tmp_path):
        import sqlite3
        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        _ensure_schema(conn)
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "fills_history" in tables
        conn.close()

    def test_double_ensure_no_error(self, tmp_path):
        import sqlite3
        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        _ensure_schema(conn)
        _ensure_schema(conn)  # second call should not error
        conn.close()
