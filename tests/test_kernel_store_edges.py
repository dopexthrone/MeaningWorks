"""Tests for kernel/store.py — typed topology edges."""

import time
from pathlib import Path

import pytest

from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.store import (
    save_grid,
    load_grid,
    load_connection_metadata,
    hub_postcodes,
    save_typed_connection,
    _db_path,
    _ensure_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PC1 = "INT.ENT.ECO.WHAT.SFT"
PC2 = "SEM.BHV.APP.HOW.SFT"
PC3 = "ORG.FNC.DOM.WHY.SFT"
PC4 = "STR.REL.CMP.WHAT.SFT"
PC5 = "COG.PLN.FET.HOW.SFT"


def _make_grid_with_connections(connections_map, intent="test"):
    """connections_map: {pc_str: [connected_pc_str, ...]}"""
    g = Grid()
    g.intent_text = intent
    all_pcs = set(connections_map.keys())
    for targets in connections_map.values():
        all_pcs.update(targets)

    for pc_str in all_pcs:
        pc = parse_postcode(pc_str)
        conns = tuple(connections_map.get(pc_str, []))
        cell = Cell(
            postcode=pc,
            primitive="node",
            content=f"content for {pc_str}",
            fill=FillState.F,
            confidence=0.9,
            connections=conns,
            source=("test",),
        )
        g.activated_layers.add(pc.layer)
        g.cells[pc.key] = cell
    return g


# ---------------------------------------------------------------------------
# Typed connection storage
# ---------------------------------------------------------------------------

class TestTypedConnections:
    def test_connections_have_default_type(self, tmp_path):
        grid = _make_grid_with_connections({PC1: [PC2]})
        save_grid(grid, "map-1", db_dir=tmp_path)

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert len(meta) == 1
        assert meta[0]["from"] == PC1
        assert meta[0]["to"] == PC2
        assert meta[0]["type"] == "association"
        assert meta[0]["strength"] == 0.5

    def test_connections_have_timestamp(self, tmp_path):
        before = time.time()
        grid = _make_grid_with_connections({PC1: [PC2]})
        save_grid(grid, "map-1", db_dir=tmp_path)
        after = time.time()

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert before <= meta[0]["created_at"] <= after

    def test_multiple_connections(self, tmp_path):
        grid = _make_grid_with_connections({
            PC1: [PC2, PC3],
            PC2: [PC3],
        })
        save_grid(grid, "map-1", db_dir=tmp_path)

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert len(meta) == 3
        froms = [m["from"] for m in meta]
        assert froms.count(PC1) == 2
        assert froms.count(PC2) == 1

    def test_load_grid_still_works(self, tmp_path):
        """Typed metadata doesn't break load_grid (Cell.connections is still postcode-only)."""
        grid = _make_grid_with_connections({PC1: [PC2, PC3]})
        save_grid(grid, "map-1", db_dir=tmp_path)

        loaded = load_grid("map-1", db_dir=tmp_path)
        cell = loaded.cells[PC1]
        assert set(cell.connections) == {PC2, PC3}

    def test_empty_connections(self, tmp_path):
        grid = _make_grid_with_connections({PC1: []})
        save_grid(grid, "map-1", db_dir=tmp_path)

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert meta == []

    def test_nonexistent_map(self, tmp_path):
        meta = load_connection_metadata("nope", db_dir=tmp_path)
        assert meta == []

    def test_nonexistent_path(self, tmp_path):
        meta = load_connection_metadata("x", db_dir=tmp_path / "nonexistent")
        assert meta == []


# ---------------------------------------------------------------------------
# save_typed_connection (direct typed insert)
# ---------------------------------------------------------------------------

class TestSaveTypedConnection:
    def test_save_derivation(self, tmp_path):
        # First save a grid to create the schema
        grid = _make_grid_with_connections({PC1: [], PC2: []})
        save_grid(grid, "map-1", db_dir=tmp_path)

        save_typed_connection(
            "map-1", PC1, PC2,
            connection_type="derivation",
            strength=0.9,
            db_dir=tmp_path,
        )

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert len(meta) == 1
        assert meta[0]["type"] == "derivation"
        assert meta[0]["strength"] == 0.9

    def test_upsert_updates_type(self, tmp_path):
        grid = _make_grid_with_connections({PC1: [PC2]})
        save_grid(grid, "map-1", db_dir=tmp_path)

        # Update from default association to containment
        save_typed_connection(
            "map-1", PC1, PC2,
            connection_type="containment",
            strength=1.0,
            db_dir=tmp_path,
        )

        meta = load_connection_metadata("map-1", db_dir=tmp_path)
        assert len(meta) == 1
        assert meta[0]["type"] == "containment"
        assert meta[0]["strength"] == 1.0


# ---------------------------------------------------------------------------
# hub_postcodes
# ---------------------------------------------------------------------------

class TestHubPostcodes:
    def test_identifies_hubs(self, tmp_path):
        # PC1 connects to 5 others (hub)
        grid = _make_grid_with_connections({
            PC1: [PC2, PC3, PC4, PC5, "INT.ENT.APP.WHAT.SFT"],
            PC2: [PC3],
        })
        save_grid(grid, "map-1", db_dir=tmp_path)

        hubs = hub_postcodes("map-1", min_connections=5, db_dir=tmp_path)
        assert PC1 in hubs
        assert PC2 not in hubs

    def test_no_hubs(self, tmp_path):
        grid = _make_grid_with_connections({PC1: [PC2]})
        save_grid(grid, "map-1", db_dir=tmp_path)

        hubs = hub_postcodes("map-1", min_connections=5, db_dir=tmp_path)
        assert hubs == []

    def test_lower_threshold(self, tmp_path):
        grid = _make_grid_with_connections({PC1: [PC2, PC3]})
        save_grid(grid, "map-1", db_dir=tmp_path)

        hubs = hub_postcodes("map-1", min_connections=2, db_dir=tmp_path)
        assert PC1 in hubs

    def test_nonexistent_map(self, tmp_path):
        hubs = hub_postcodes("nope", db_dir=tmp_path)
        assert hubs == []

    def test_nonexistent_path(self, tmp_path):
        hubs = hub_postcodes("x", db_dir=tmp_path / "nonexistent")
        assert hubs == []


# ---------------------------------------------------------------------------
# Migration — existing DB without new columns
# ---------------------------------------------------------------------------

class TestMigration:
    def test_migrate_adds_columns(self, tmp_path):
        """Simulates an old DB without typed columns, then ensures migration works."""
        import sqlite3

        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        # Create old-style schema without new columns
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS maps (
                id TEXT PRIMARY KEY, name TEXT, intent TEXT DEFAULT '',
                root TEXT DEFAULT '', created REAL, updated REAL
            );
            CREATE TABLE IF NOT EXISTS cells (
                map_id TEXT, postcode TEXT, primitive TEXT,
                content TEXT DEFAULT '', fill_state TEXT DEFAULT 'E',
                confidence REAL DEFAULT 0.0, parent TEXT,
                source_json TEXT DEFAULT '[]', revisions_json TEXT DEFAULT '[]',
                PRIMARY KEY (map_id, postcode)
            );
            CREATE TABLE IF NOT EXISTS connections (
                map_id TEXT, from_postcode TEXT, to_postcode TEXT,
                PRIMARY KEY (map_id, from_postcode, to_postcode)
            );
        """)
        # Insert a connection without typed columns
        conn.execute(
            "INSERT INTO connections (map_id, from_postcode, to_postcode) VALUES (?, ?, ?)",
            ("old-map", PC1, PC2),
        )
        conn.commit()
        conn.close()

        # Now call _ensure_schema which should migrate
        conn2 = sqlite3.connect(str(path))
        _ensure_schema(conn2)

        # Verify columns exist
        cols = {r[1] for r in conn2.execute("PRAGMA table_info(connections)").fetchall()}
        assert "connection_type" in cols
        assert "strength" in cols
        assert "created_at" in cols

        # Verify old data has defaults
        conn2.row_factory = sqlite3.Row
        row = conn2.execute("SELECT * FROM connections WHERE map_id='old-map'").fetchone()
        assert row["connection_type"] == "association"
        assert row["strength"] == 0.5
        conn2.close()
