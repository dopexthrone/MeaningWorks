"""
kernel/store.py — Persistent grid storage with SQLite.

LEAF module. Stores and retrieves semantic maps at ~/.motherlabs/maps.db.
Each map is a named grid with cells, connections, and metadata.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid


# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".motherlabs"
_DEFAULT_DB_NAME = "maps.db"


def _db_path(db_dir: Optional[Path] = None) -> Path:
    """Writable db path — creates directory if needed."""
    d = db_dir or _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_DB_NAME


def _read_db_path(db_dir: Optional[Path] = None) -> Path:
    """Read-only db path — does NOT create directories."""
    d = db_dir or _DEFAULT_DB_DIR
    return d / _DEFAULT_DB_NAME


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS maps (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    intent      TEXT NOT NULL DEFAULT '',
    root        TEXT NOT NULL DEFAULT '',
    created     REAL NOT NULL,
    updated     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cells (
    map_id      TEXT NOT NULL,
    postcode    TEXT NOT NULL,
    primitive   TEXT NOT NULL,
    content     TEXT NOT NULL DEFAULT '',
    fill_state  TEXT NOT NULL DEFAULT 'E',
    confidence  REAL NOT NULL DEFAULT 0.0,
    parent      TEXT,
    source_json TEXT NOT NULL DEFAULT '[]',
    revisions_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (map_id, postcode),
    FOREIGN KEY (map_id) REFERENCES maps(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS connections (
    map_id          TEXT NOT NULL,
    from_postcode   TEXT NOT NULL,
    to_postcode     TEXT NOT NULL,
    PRIMARY KEY (map_id, from_postcode, to_postcode),
    FOREIGN KEY (map_id) REFERENCES maps(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS fills_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    map_id TEXT NOT NULL,
    postcode TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    fill_state TEXT NOT NULL DEFAULT 'E',
    agent TEXT NOT NULL DEFAULT '',
    source_json TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL,
    UNIQUE(map_id, postcode, version)
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    _migrate_connections(conn)


def _migrate_connections(conn: sqlite3.Connection) -> None:
    """Add connection_type, strength, created_at columns if missing."""
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(connections)").fetchall()
    }
    if "connection_type" not in cols:
        conn.execute(
            "ALTER TABLE connections ADD COLUMN connection_type TEXT NOT NULL DEFAULT 'association'"
        )
    if "strength" not in cols:
        conn.execute(
            "ALTER TABLE connections ADD COLUMN strength REAL NOT NULL DEFAULT 0.5"
        )
    if "created_at" not in cols:
        conn.execute(
            "ALTER TABLE connections ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_grid(
    grid: Grid,
    map_id: str,
    name: str = "",
    db_dir: Optional[Path] = None,
) -> str:
    """Save a grid to persistent storage.

    If map_id already exists, it is fully replaced (upsert).
    Returns the map_id.
    """
    path = _db_path(db_dir)
    now = time.time()

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)

    try:
        with conn:
            # Upsert map record
            conn.execute(
                """INSERT INTO maps (id, name, intent, root, created, updated)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       name=excluded.name,
                       intent=excluded.intent,
                       root=excluded.root,
                       updated=excluded.updated""",
                (
                    map_id,
                    name or map_id,
                    grid.intent_text,
                    grid.root or "",
                    now,
                    now,
                ),
            )

            # Clear existing cells and connections for this map
            conn.execute("DELETE FROM cells WHERE map_id = ?", (map_id,))
            conn.execute("DELETE FROM connections WHERE map_id = ?", (map_id,))

            # Insert cells + append to fill history
            for pc_key, cell in grid.cells.items():
                conn.execute(
                    """INSERT INTO cells
                       (map_id, postcode, primitive, content, fill_state,
                        confidence, parent, source_json, revisions_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        map_id,
                        pc_key,
                        cell.primitive,
                        cell.content,
                        cell.fill.name,
                        cell.confidence,
                        cell.parent,
                        json.dumps(list(cell.source)),
                        json.dumps(list(cell.revisions)),
                    ),
                )

                # Append-only fill history (never deleted)
                cur_version = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM fills_history WHERE map_id = ? AND postcode = ?",
                    (map_id, pc_key),
                ).fetchone()[0]
                conn.execute(
                    """INSERT OR IGNORE INTO fills_history
                       (map_id, postcode, version, content, confidence,
                        fill_state, agent, source_json, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        map_id,
                        pc_key,
                        cur_version + 1,
                        cell.content,
                        cell.confidence,
                        cell.fill.name,
                        grid._agent_map.get(pc_key, ""),  # agent from fill() calls
                        json.dumps(list(cell.source)),
                        now,
                    ),
                )

                # Insert typed connections
                for conn_pc in cell.connections:
                    conn.execute(
                        """INSERT OR IGNORE INTO connections
                           (map_id, from_postcode, to_postcode,
                            connection_type, strength, created_at)
                           VALUES (?, ?, ?, 'association', 0.5, ?)""",
                        (map_id, pc_key, conn_pc, now),
                    )
    finally:
        conn.close()

    return map_id


def load_grid(
    map_id: str,
    db_dir: Optional[Path] = None,
) -> Optional[Grid]:
    """Load a grid from persistent storage.

    Returns None if map_id not found.
    """
    path = _db_path(db_dir)
    if not path.exists():
        return None

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    try:
        # Get map record
        row = conn.execute(
            "SELECT * FROM maps WHERE id = ?", (map_id,)
        ).fetchone()
        if row is None:
            return None

        intent = row["intent"]
        root_str = row["root"]

        # Get all cells
        cell_rows = conn.execute(
            "SELECT * FROM cells WHERE map_id = ? ORDER BY postcode",
            (map_id,),
        ).fetchall()

        # Get all connections
        conn_rows = conn.execute(
            "SELECT from_postcode, to_postcode FROM connections WHERE map_id = ?",
            (map_id,),
        ).fetchall()

        # Build connection map
        conn_map: dict[str, list[str]] = {}
        for cr in conn_rows:
            fp = cr["from_postcode"]
            tp = cr["to_postcode"]
            conn_map.setdefault(fp, []).append(tp)

        # Build grid
        grid = Grid()
        grid.intent_text = intent
        grid.root = root_str if root_str else None

        for cr in cell_rows:
            pc = parse_postcode(cr["postcode"])
            fill = FillState[cr["fill_state"]]
            source = tuple(json.loads(cr["source_json"]))
            revisions = tuple(
                tuple(r) for r in json.loads(cr["revisions_json"])
            )
            connections = tuple(conn_map.get(cr["postcode"], []))

            cell = Cell(
                postcode=pc,
                primitive=cr["primitive"],
                content=cr["content"],
                fill=fill,
                confidence=cr["confidence"],
                connections=connections,
                parent=cr["parent"],
                source=source,
                revisions=revisions,
            )

            # Ensure layer is activated (direct add, no root cell creation)
            grid.activated_layers.add(pc.layer)

            grid.cells[pc.key] = cell

        return grid
    finally:
        conn.close()


def list_maps(
    db_dir: Optional[Path] = None,
) -> list[dict[str, object]]:
    """List all stored maps.

    Returns list of dicts with: id, name, intent, root, created, updated.
    """
    path = _db_path(db_dir)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    try:
        rows = conn.execute(
            "SELECT id, name, intent, root, created, updated FROM maps ORDER BY updated DESC"
        ).fetchall()

        return [
            {
                "id": r["id"],
                "name": r["name"],
                "intent": r["intent"],
                "root": r["root"],
                "created": r["created"],
                "updated": r["updated"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def delete_map(
    map_id: str,
    db_dir: Optional[Path] = None,
) -> bool:
    """Delete a map and all its cells/connections.

    Returns True if map existed and was deleted, False if not found.
    """
    path = _db_path(db_dir)
    if not path.exists():
        return False

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)

    try:
        with conn:
            cursor = conn.execute("DELETE FROM maps WHERE id = ?", (map_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def map_cell_count(
    map_id: str,
    db_dir: Optional[Path] = None,
) -> int:
    """Get the cell count for a stored map. Returns 0 if not found."""
    path = _db_path(db_dir)
    if not path.exists():
        return 0

    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)

    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM cells WHERE map_id = ?", (map_id,)
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fill history queries
# ---------------------------------------------------------------------------

def cell_history(
    map_id: str,
    postcode: str,
    db_dir: Optional[Path] = None,
) -> list[dict]:
    """Return all historical versions of a cell, oldest first."""
    path = _read_db_path(db_dir)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    try:
        rows = conn.execute(
            """SELECT version, content, confidence, fill_state, agent,
                      source_json, timestamp
               FROM fills_history
               WHERE map_id = ? AND postcode = ?
               ORDER BY version ASC""",
            (map_id, postcode),
        ).fetchall()
        return [
            {
                "version": r["version"],
                "content": r["content"],
                "confidence": r["confidence"],
                "fill_state": r["fill_state"],
                "agent": r["agent"],
                "source": json.loads(r["source_json"]),
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def cell_version_count(
    map_id: str,
    postcode: str,
    db_dir: Optional[Path] = None,
) -> int:
    """Return the number of historical versions for a cell."""
    path = _read_db_path(db_dir)
    if not path.exists():
        return 0

    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)

    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM fills_history WHERE map_id = ? AND postcode = ?",
            (map_id, postcode),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def history_stats(
    map_id: str,
    db_dir: Optional[Path] = None,
) -> dict:
    """Aggregate statistics over fill history for a map."""
    path = _read_db_path(db_dir)
    if not path.exists():
        return {"total_versions": 0}

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    try:
        row = conn.execute(
            """SELECT COUNT(*) as total,
                      COUNT(DISTINCT postcode) as unique_cells,
                      AVG(confidence) as avg_confidence,
                      MAX(version) as max_version
               FROM fills_history WHERE map_id = ?""",
            (map_id,),
        ).fetchone()

        if not row or row["total"] == 0:
            return {"total_versions": 0}

        return {
            "total_versions": row["total"],
            "unique_cells": row["unique_cells"],
            "avg_confidence": round(row["avg_confidence"] or 0, 3),
            "max_version": row["max_version"],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Typed connection queries
# ---------------------------------------------------------------------------

def load_connection_metadata(
    map_id: str,
    db_dir: Optional[Path] = None,
) -> list[dict]:
    """Load all connections with typed metadata for a map."""
    path = _read_db_path(db_dir)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)

    try:
        rows = conn.execute(
            """SELECT from_postcode, to_postcode, connection_type,
                      strength, created_at
               FROM connections WHERE map_id = ?
               ORDER BY from_postcode, to_postcode""",
            (map_id,),
        ).fetchall()
        return [
            {
                "from": r["from_postcode"],
                "to": r["to_postcode"],
                "type": r["connection_type"],
                "strength": r["strength"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def hub_postcodes(
    map_id: str,
    min_connections: int = 5,
    db_dir: Optional[Path] = None,
) -> list[str]:
    """Return postcodes that are hubs (many connections)."""
    path = _read_db_path(db_dir)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)

    try:
        rows = conn.execute(
            """SELECT from_postcode, COUNT(*) as cnt
               FROM connections WHERE map_id = ?
               GROUP BY from_postcode
               HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (map_id, min_connections),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def save_typed_connection(
    map_id: str,
    from_postcode: str,
    to_postcode: str,
    connection_type: str = "association",
    strength: float = 0.5,
    db_dir: Optional[Path] = None,
) -> None:
    """Save or update a typed connection between two cells."""
    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)

    try:
        with conn:
            now = time.time()
            conn.execute(
                """INSERT INTO connections
                   (map_id, from_postcode, to_postcode, connection_type, strength, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(map_id, from_postcode, to_postcode) DO UPDATE SET
                       connection_type=excluded.connection_type,
                       strength=excluded.strength""",
                (map_id, from_postcode, to_postcode, connection_type, strength, now),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Self-build feedback
# ---------------------------------------------------------------------------

def boost_cell_confidence(
    map_id: str,
    postcodes: tuple[str, ...],
    delta: float = 0.15,
    max_confidence: float = 0.95,
    db_dir: Optional[Path] = None,
) -> int:
    """Boost confidence for cells after successful build. Returns count updated.

    Loads grid, applies delta to matching cells (capped at max_confidence).
    On negative delta (failure), floors at 0.05.
    Saves the grid back and records the change in fills_history.
    """
    if not postcodes:
        return 0

    path = _db_path(db_dir)
    if not path.exists():
        return 0

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    _ensure_schema(conn)

    updated = 0
    now = time.time()
    pc_set = set(postcodes)

    try:
        with conn:
            # Read current cells
            rows = conn.execute(
                "SELECT postcode, confidence, fill_state FROM cells WHERE map_id = ?",
                (map_id,),
            ).fetchall()

            for postcode_val, current_conf, fill_state in rows:
                if postcode_val not in pc_set:
                    continue

                new_conf = current_conf + delta
                if delta >= 0:
                    new_conf = min(new_conf, max_confidence)
                else:
                    new_conf = max(new_conf, 0.05)

                if new_conf == current_conf:
                    continue

                # Update cell confidence
                conn.execute(
                    "UPDATE cells SET confidence = ? WHERE map_id = ? AND postcode = ?",
                    (new_conf, map_id, postcode_val),
                )

                # Append to fill history
                cur_version = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM fills_history "
                    "WHERE map_id = ? AND postcode = ?",
                    (map_id, postcode_val),
                ).fetchone()[0]
                conn.execute(
                    """INSERT OR IGNORE INTO fills_history
                       (map_id, postcode, version, content, confidence,
                        fill_state, agent, source_json, timestamp)
                       VALUES (?, ?, ?, '', ?, ?, 'self_build', '[]', ?)""",
                    (map_id, postcode_val, cur_version + 1,
                     new_conf, fill_state, now),
                )

                updated += 1

            # Update map timestamp
            if updated > 0:
                conn.execute(
                    "UPDATE maps SET updated = ? WHERE id = ?",
                    (now, map_id),
                )

    finally:
        conn.close()

    return updated


def record_build_outcome(
    map_id: str,
    postcodes: tuple[str, ...],
    success: bool,
    build_description: str = "",
    db_dir: Optional[Path] = None,
) -> None:
    """Record a build outcome in fills_history with agent='self_build'.

    This is a lightweight audit trail — does not modify cell confidence
    (use boost_cell_confidence for that).
    """
    if not postcodes:
        return

    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    _ensure_schema(conn)

    now = time.time()
    status_tag = "build_success" if success else "build_failure"
    desc_truncated = build_description[:500] if build_description else ""

    try:
        with conn:
            for pc in postcodes:
                cur_version = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM fills_history "
                    "WHERE map_id = ? AND postcode = ?",
                    (map_id, pc),
                ).fetchone()[0]
                conn.execute(
                    """INSERT OR IGNORE INTO fills_history
                       (map_id, postcode, version, content, confidence,
                        fill_state, agent, source_json, timestamp)
                       VALUES (?, ?, ?, ?, 0.0, ?, 'self_build', ?, ?)""",
                    (
                        map_id, pc, cur_version + 1,
                        desc_truncated, status_tag,
                        json.dumps([f"build:{status_tag}"]),
                        now,
                    ),
                )
    finally:
        conn.close()
