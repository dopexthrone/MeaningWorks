"""
Motherlabs Tool Registry — SQLite-backed tool storage and discovery.

LEAF MODULE — stdlib + sqlite3 only. Follows auth.py KeyStore pattern.

Provides:
- ToolRegistry: SQLite-backed CRUD + search + usage tracking
- Module-level singleton: get_tool_registry()

Database: ~/motherlabs/tools.db (separate from auth.db and corpus)
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.tool_package import (
    ToolDigest,
    ToolPackage,
    deserialize_tool_package,
    extract_digest,
    serialize_tool_package,
)


# =============================================================================
# SCHEMA
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tools (
    package_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    fingerprint TEXT NOT NULL,
    trust_score REAL NOT NULL,
    verification_badge TEXT NOT NULL,
    component_count INTEGER NOT NULL,
    relationship_count INTEGER NOT NULL,
    source_instance_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    imported_at TEXT,
    package_json TEXT NOT NULL,
    is_local INTEGER NOT NULL DEFAULT 1,
    usage_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS tool_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id TEXT NOT NULL,
    action TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    target_instance_id TEXT DEFAULT '',
    FOREIGN KEY (package_id) REFERENCES tools(package_id)
);

CREATE INDEX IF NOT EXISTS idx_tools_domain ON tools(domain);
CREATE INDEX IF NOT EXISTS idx_tools_fingerprint ON tools(fingerprint);
CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);
CREATE INDEX IF NOT EXISTS idx_usage_package ON tool_usage_log(package_id);
"""


# =============================================================================
# HELPERS
# =============================================================================

def _default_db_path() -> str:
    """Default tools database path: ~/motherlabs/tools.db"""
    db_dir = Path.home() / "motherlabs"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "tools.db")


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """SQLite-backed tool registry. Thread-safe.

    Stores full ToolPackage JSON alongside indexed metadata columns
    for efficient search and filtering.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or _default_db_path()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection (for use within a lock)."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def register_tool(self, package: ToolPackage, is_local: bool = True) -> None:
        """Register a tool package in the registry.

        Args:
            package: ToolPackage to register
            is_local: True if compiled locally, False if imported

        Raises:
            sqlite3.IntegrityError: If package_id already exists
        """
        digest = extract_digest(package)
        package_json = json.dumps(serialize_tool_package(package))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO tools (
                        package_id, name, domain, version, fingerprint,
                        trust_score, verification_badge, component_count,
                        relationship_count, source_instance_id, created_at,
                        imported_at, package_json, is_local, usage_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        package.package_id,
                        package.name,
                        package.domain,
                        package.version,
                        package.fingerprint,
                        package.trust_score,
                        package.verification_badge,
                        digest.component_count,
                        digest.relationship_count,
                        package.source_instance_id,
                        package.created_at,
                        None if is_local else now,
                        package_json,
                        1 if is_local else 0,
                        package.usage_count,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_tool(self, package_id: str) -> Optional[ToolPackage]:
        """Get full ToolPackage by ID.

        Args:
            package_id: Package ID to look up

        Returns:
            ToolPackage or None if not found
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT package_json FROM tools WHERE package_id = ?",
                    (package_id,),
                ).fetchone()
            finally:
                conn.close()

        if not row:
            return None

        return deserialize_tool_package(json.loads(row["package_json"]))

    def list_tools(
        self,
        domain: Optional[str] = None,
        local_only: bool = False,
    ) -> List[ToolDigest]:
        """List tools as lightweight digests.

        Args:
            domain: Filter by domain (None = all)
            local_only: Only return locally compiled tools

        Returns:
            List of ToolDigest
        """
        query = "SELECT * FROM tools WHERE 1=1"
        params: list = []

        if domain:
            query += " AND domain = ?"
            params.append(domain)

        if local_only:
            query += " AND is_local = 1"

        query += " ORDER BY created_at DESC"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.close()

        return [
            ToolDigest(
                package_id=r["package_id"],
                name=r["name"],
                domain=r["domain"],
                fingerprint=r["fingerprint"],
                trust_score=r["trust_score"],
                verification_badge=r["verification_badge"],
                component_count=r["component_count"],
                relationship_count=r["relationship_count"],
                source_instance_id=r["source_instance_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def search_tools(self, query: str) -> List[ToolDigest]:
        """Search tools by name or domain (case-insensitive substring match).

        Args:
            query: Search string

        Returns:
            List of matching ToolDigest
        """
        search_pattern = f"%{query}%"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM tools
                       WHERE name LIKE ? OR domain LIKE ?
                       ORDER BY trust_score DESC, created_at DESC""",
                    (search_pattern, search_pattern),
                ).fetchall()
            finally:
                conn.close()

        return [
            ToolDigest(
                package_id=r["package_id"],
                name=r["name"],
                domain=r["domain"],
                fingerprint=r["fingerprint"],
                trust_score=r["trust_score"],
                verification_badge=r["verification_badge"],
                component_count=r["component_count"],
                relationship_count=r["relationship_count"],
                source_instance_id=r["source_instance_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def find_by_fingerprint(self, fingerprint: str) -> Optional[ToolDigest]:
        """Find a tool by structural fingerprint hash.

        Args:
            fingerprint: StructuralFingerprint.hash_digest

        Returns:
            ToolDigest or None
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM tools WHERE fingerprint = ? LIMIT 1",
                    (fingerprint,),
                ).fetchone()
            finally:
                conn.close()

        if not row:
            return None

        return ToolDigest(
            package_id=row["package_id"],
            name=row["name"],
            domain=row["domain"],
            fingerprint=row["fingerprint"],
            trust_score=row["trust_score"],
            verification_badge=row["verification_badge"],
            component_count=row["component_count"],
            relationship_count=row["relationship_count"],
            source_instance_id=row["source_instance_id"],
            created_at=row["created_at"],
        )

    def record_usage(
        self,
        package_id: str,
        action: str,
        target_instance_id: str = "",
    ) -> None:
        """Record a usage event for a tool.

        Args:
            package_id: Tool package ID
            action: 'export' | 'import' | 'query' | 'use'
            target_instance_id: Remote instance (for export/import)
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO tool_usage_log
                       (package_id, action, timestamp, target_instance_id)
                       VALUES (?, ?, ?, ?)""",
                    (package_id, action, now, target_instance_id),
                )
                conn.commit()
            finally:
                conn.close()

    def increment_usage_count(self, package_id: str) -> None:
        """Atomically increment usage count for a tool.

        Args:
            package_id: Tool package ID
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE tools SET usage_count = usage_count + 1 WHERE package_id = ?",
                    (package_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def get_usage_stats(self, package_id: str) -> Dict[str, Any]:
        """Get usage statistics for a tool.

        Args:
            package_id: Tool package ID

        Returns:
            Dict with action counts and total usage
        """
        with self._lock:
            conn = self._connect()
            try:
                # Action counts
                rows = conn.execute(
                    """SELECT action, COUNT(*) as cnt
                       FROM tool_usage_log
                       WHERE package_id = ?
                       GROUP BY action""",
                    (package_id,),
                ).fetchall()

                # Total usage_count from tools table
                tool_row = conn.execute(
                    "SELECT usage_count FROM tools WHERE package_id = ?",
                    (package_id,),
                ).fetchone()
            finally:
                conn.close()

        action_counts = {r["action"]: r["cnt"] for r in rows}
        total = tool_row["usage_count"] if tool_row else 0

        return {
            "package_id": package_id,
            "usage_count": total,
            "action_counts": action_counts,
            "total_events": sum(action_counts.values()),
        }

    def remove_tool(self, package_id: str) -> bool:
        """Remove a tool from the registry.

        Args:
            package_id: Tool package ID

        Returns:
            True if tool was found and removed, False if not found
        """
        with self._lock:
            conn = self._connect()
            try:
                # Delete usage logs first (FK constraint)
                conn.execute(
                    "DELETE FROM tool_usage_log WHERE package_id = ?",
                    (package_id,),
                )
                cursor = conn.execute(
                    "DELETE FROM tools WHERE package_id = ?",
                    (package_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()


# =============================================================================
# MODULE SINGLETON
# =============================================================================

_tool_registry: Optional[ToolRegistry] = None
_singleton_lock = threading.Lock()


def get_tool_registry(db_path: Optional[str] = None) -> ToolRegistry:
    """Get or create the singleton ToolRegistry.

    Args:
        db_path: Optional database path override

    Returns:
        ToolRegistry instance
    """
    global _tool_registry
    with _singleton_lock:
        if _tool_registry is None:
            _tool_registry = ToolRegistry(db_path=db_path)
        return _tool_registry
