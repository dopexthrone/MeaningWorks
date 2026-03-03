"""
Appendage registry — data model and SQLite persistence for spawned capabilities.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

An appendage is a child-process agent that Mother builds and spawns to
acquire capabilities she doesn't natively have. Appendages that prove
useful solidify into permanent parts of her form. Those that don't dissolve.

Lifecycle: spec → building → built → spawned → active → solidified | dissolved | failed
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AppendageSpec:
    """Specification and runtime state for a spawned capability."""

    appendage_id: int = 0
    name: str = ""                      # "screen-recorder", "browser-tracker"
    description: str = ""               # What it does
    capability_gap: str = ""            # The gap this fills
    entry_point: str = "main.py"
    project_dir: str = ""               # ~/motherlabs/appendages/<name>/
    status: str = "spec"                # spec|building|built|spawned|active|failed|solidified|dissolved
    pid: int = 0
    port: int = 0
    created_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    total_cost_usd: float = 0.0
    error: str = ""
    build_prompt: str = ""
    capabilities_json: str = "[]"       # JSON array of capability keywords


def propagate_constraints(
    parent_build_prompt: str,
    parent_capabilities: str,
    child_build_prompt: str,
) -> str:
    """Propagate parent constraints into child build prompt. Pure function.
    Prepends parent context so child inherits domain constraints.
    """
    if not parent_build_prompt and not parent_capabilities:
        return child_build_prompt
    inherited = []
    if parent_capabilities and parent_capabilities != "[]":
        inherited.append(f"[Inherited capabilities: {parent_capabilities}]")
    if parent_build_prompt:
        # Extract constraint lines from parent (lines containing "must", "require", "constraint", "limit")
        constraint_lines = []
        for line in parent_build_prompt.split("\n"):
            lower = line.lower().strip()
            if any(kw in lower for kw in ("must", "require", "constraint", "limit", "boundary", "never")):
                constraint_lines.append(line.strip())
        if constraint_lines:
            inherited.append("[Inherited constraints from parent:]")
            inherited.extend(constraint_lines[:5])  # cap at 5
    if not inherited:
        return child_build_prompt
    prefix = "\n".join(inherited) + "\n\n"
    return prefix + child_build_prompt


_VALID_STATUSES = frozenset({
    "spec", "building", "built", "spawned", "active",
    "failed", "solidified", "dissolved",
})

_CREATE_APPENDAGES = """
CREATE TABLE IF NOT EXISTS appendages (
    appendage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    capability_gap TEXT NOT NULL DEFAULT '',
    entry_point TEXT NOT NULL DEFAULT 'main.py',
    project_dir TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'spec',
    pid INTEGER NOT NULL DEFAULT 0,
    port INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_used REAL NOT NULL DEFAULT 0.0,
    use_count INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    error TEXT NOT NULL DEFAULT '',
    build_prompt TEXT NOT NULL DEFAULT '',
    capabilities_json TEXT NOT NULL DEFAULT '[]'
)
"""


class AppendageStore:
    """SQLite-backed appendage registry. Coexists with history.db tables."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_APPENDAGES)
        self._conn.commit()

    def register(
        self,
        name: str,
        description: str,
        capability_gap: str,
        build_prompt: str,
        project_dir: str,
        capabilities: Optional[List[str]] = None,
    ) -> int:
        """Register a new appendage. Returns appendage_id."""
        now = time.time()
        caps_json = json.dumps(capabilities or [])
        cursor = self._conn.execute(
            """INSERT INTO appendages
               (name, description, capability_gap, build_prompt,
                project_dir, capabilities_json, created_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'spec')""",
            (name, description, capability_gap, build_prompt,
             project_dir, caps_json, now),
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_status(
        self,
        appendage_id: int,
        status: str,
        pid: int = 0,
        port: int = 0,
        error: str = "",
    ) -> None:
        """Update appendage status and optional runtime fields."""
        if status not in _VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        self._conn.execute(
            """UPDATE appendages
               SET status = ?, pid = ?, port = ?, error = ?
               WHERE appendage_id = ?""",
            (status, pid, port, error, appendage_id),
        )
        self._conn.commit()

    def update_cost(self, appendage_id: int, cost_usd: float) -> None:
        """Add to the total cost for an appendage."""
        self._conn.execute(
            """UPDATE appendages
               SET total_cost_usd = total_cost_usd + ?
               WHERE appendage_id = ?""",
            (cost_usd, appendage_id),
        )
        self._conn.commit()

    def get(self, appendage_id: int) -> Optional[AppendageSpec]:
        """Get appendage by ID."""
        row = self._conn.execute(
            "SELECT * FROM appendages WHERE appendage_id = ?",
            (appendage_id,),
        ).fetchone()
        return self._row_to_spec(row) if row else None

    def get_by_name(self, name: str) -> Optional[AppendageSpec]:
        """Get appendage by name."""
        row = self._conn.execute(
            "SELECT * FROM appendages WHERE name = ? ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        return self._row_to_spec(row) if row else None

    def active(self) -> List[AppendageSpec]:
        """Get all appendages with status in (spawned, active, solidified)."""
        rows = self._conn.execute(
            """SELECT * FROM appendages
               WHERE status IN ('spawned', 'active', 'solidified')
               ORDER BY name ASC""",
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def solidified(self) -> List[AppendageSpec]:
        """Get all solidified appendages."""
        rows = self._conn.execute(
            "SELECT * FROM appendages WHERE status = 'solidified' ORDER BY name ASC",
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def find_for_capability(self, keyword: str) -> Optional[AppendageSpec]:
        """Search for an active/solidified appendage matching a capability keyword.

        Searches the capabilities_json field and name/description.
        Returns the best match or None.
        """
        keyword_lower = keyword.lower()

        # First: check active/solidified appendages
        rows = self._conn.execute(
            """SELECT * FROM appendages
               WHERE status IN ('spawned', 'active', 'solidified', 'built')
               ORDER BY use_count DESC""",
        ).fetchall()

        for row in rows:
            spec = self._row_to_spec(row)
            # Check capabilities_json
            try:
                caps = json.loads(spec.capabilities_json)
                if any(keyword_lower in c.lower() for c in caps):
                    return spec
            except (json.JSONDecodeError, TypeError):
                pass
            # Check name and description
            if keyword_lower in spec.name.lower() or keyword_lower in spec.description.lower():
                return spec

        return None

    def record_use(self, appendage_id: int) -> None:
        """Bump use_count and update last_used timestamp."""
        now = time.time()
        self._conn.execute(
            """UPDATE appendages
               SET use_count = use_count + 1, last_used = ?
               WHERE appendage_id = ?""",
            (now, appendage_id),
        )
        self._conn.commit()

    def candidates_for_dissolution(
        self,
        idle_hours: int = 48,
        min_uses: int = 3,
    ) -> List[AppendageSpec]:
        """Find appendages that are idle and underused — candidates for dissolution.

        Returns appendages where:
        - status is 'active' or 'spawned' (not solidified)
        - last_used is older than idle_hours ago (or never used)
        - use_count < min_uses
        """
        cutoff = time.time() - (idle_hours * 3600)
        rows = self._conn.execute(
            """SELECT * FROM appendages
               WHERE status IN ('active', 'spawned')
               AND (last_used < ? OR last_used = 0.0)
               AND use_count < ?
               ORDER BY last_used ASC""",
            (cutoff, min_uses),
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def all(self) -> List[AppendageSpec]:
        """Get all appendages regardless of status."""
        rows = self._conn.execute(
            "SELECT * FROM appendages ORDER BY created_at DESC",
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_spec(row) -> AppendageSpec:
        return AppendageSpec(
            appendage_id=row[0],
            name=row[1],
            description=row[2],
            capability_gap=row[3],
            entry_point=row[4],
            project_dir=row[5],
            status=row[6],
            pid=row[7],
            port=row[8],
            created_at=row[9],
            last_used=row[10],
            use_count=row[11],
            total_cost_usd=row[12],
            error=row[13],
            build_prompt=row[14],
            capabilities_json=row[15],
        )
