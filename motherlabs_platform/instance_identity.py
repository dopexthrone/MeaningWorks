"""
Motherlabs Instance Identity — self-identification and peer discovery.

Near-leaf module — stdlib + sqlite3 only.

Provides:
- InstanceIdentityStore: SQLite-backed identity persistence
- InstanceRecord: Frozen dataclass for instance info
- TrustGraphDigest: Summary of an instance's tool portfolio
- build_trust_graph_digest(): Compute digest from tool registry

Database: ~/motherlabs/instance.db
"""

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class InstanceRecord:
    """Identity record for this or a peer instance."""
    instance_id: str           # SHA256[:16] of generated secret
    name: str
    created_at: str
    api_endpoint: str          # URL where V2 API is reachable
    is_self: bool              # True for this instance


@dataclass(frozen=True)
class TrustGraphDigest:
    """Summary of an instance's tool portfolio."""
    instance_id: str
    instance_name: str
    tool_count: int
    verified_tool_count: int
    domain_counts: Dict[str, int]
    total_compilations: int
    avg_trust_score: float
    last_updated: str


# =============================================================================
# SCHEMA
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS instances (
    instance_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    api_endpoint TEXT NOT NULL DEFAULT '',
    is_self INTEGER NOT NULL DEFAULT 0
);
"""


# =============================================================================
# HELPERS
# =============================================================================

def _default_db_path() -> str:
    """Default instance database path: ~/motherlabs/instance.db"""
    db_dir = Path.home() / "motherlabs"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "instance.db")


def _generate_instance_id() -> str:
    """Generate a new instance ID from a random secret."""
    secret = secrets.token_hex(32)
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# INSTANCE IDENTITY STORE
# =============================================================================

class InstanceIdentityStore:
    """SQLite-backed instance identity store. Thread-safe.

    Auto-generates this instance's ID on first use via get_or_create_self().
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

    def get_or_create_self(self, name: str = "default") -> InstanceRecord:
        """Get this instance's identity, creating it if needed.

        Args:
            name: Human-readable instance name

        Returns:
            Frozen InstanceRecord with is_self=True
        """
        with self._lock:
            conn = self._connect()
            try:
                # Check if self already exists
                row = conn.execute(
                    "SELECT * FROM instances WHERE is_self = 1"
                ).fetchone()

                if row:
                    return InstanceRecord(
                        instance_id=row["instance_id"],
                        name=row["name"],
                        created_at=row["created_at"],
                        api_endpoint=row["api_endpoint"],
                        is_self=True,
                    )

                # Create new identity
                instance_id = _generate_instance_id()
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                conn.execute(
                    """INSERT INTO instances
                       (instance_id, name, created_at, api_endpoint, is_self)
                       VALUES (?, ?, ?, '', 1)""",
                    (instance_id, name, now),
                )
                conn.commit()

                return InstanceRecord(
                    instance_id=instance_id,
                    name=name,
                    created_at=now,
                    api_endpoint="",
                    is_self=True,
                )
            finally:
                conn.close()

    def register_peer(
        self,
        instance_id: str,
        name: str,
        api_endpoint: str,
    ) -> None:
        """Register a known peer instance.

        Args:
            instance_id: Peer's instance ID
            name: Human-readable name
            api_endpoint: URL where peer's V2 API is reachable

        Raises:
            sqlite3.IntegrityError: If instance_id already registered
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO instances
                       (instance_id, name, created_at, api_endpoint, is_self)
                       VALUES (?, ?, ?, ?, 0)""",
                    (instance_id, name, now, api_endpoint),
                )
                conn.commit()
            finally:
                conn.close()

    def list_peers(self) -> List[InstanceRecord]:
        """List all known peer instances (excludes self).

        Returns:
            List of InstanceRecord
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM instances WHERE is_self = 0 ORDER BY name"
                ).fetchall()
            finally:
                conn.close()

        return [
            InstanceRecord(
                instance_id=r["instance_id"],
                name=r["name"],
                created_at=r["created_at"],
                api_endpoint=r["api_endpoint"],
                is_self=False,
            )
            for r in rows
        ]

    def get_peer(self, instance_id: str) -> Optional[InstanceRecord]:
        """Get a specific peer by ID.

        Args:
            instance_id: Peer instance ID

        Returns:
            InstanceRecord or None
        """
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM instances WHERE instance_id = ? AND is_self = 0",
                    (instance_id,),
                ).fetchone()
            finally:
                conn.close()

        if not row:
            return None

        return InstanceRecord(
            instance_id=row["instance_id"],
            name=row["name"],
            created_at=row["created_at"],
            api_endpoint=row["api_endpoint"],
            is_self=False,
        )

    def remove_peer(self, instance_id: str) -> bool:
        """Remove a peer from the identity store.

        Args:
            instance_id: Peer instance ID

        Returns:
            True if found and removed, False if not found
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM instances WHERE instance_id = ? AND is_self = 0",
                    (instance_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()


# =============================================================================
# TRUST GRAPH DIGEST
# =============================================================================

def build_trust_graph_digest(
    instance_id: str,
    instance_name: str,
    tool_registry,
) -> TrustGraphDigest:
    """Compute a trust graph digest from the tool registry.

    Summarizes this instance's tool portfolio for sharing with peers.

    Args:
        instance_id: This instance's ID
        instance_name: This instance's name
        tool_registry: ToolRegistry instance

    Returns:
        Frozen TrustGraphDigest
    """
    tools = tool_registry.list_tools()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not tools:
        return TrustGraphDigest(
            instance_id=instance_id,
            instance_name=instance_name,
            tool_count=0,
            verified_tool_count=0,
            domain_counts={},
            total_compilations=0,
            avg_trust_score=0.0,
            last_updated=now,
        )

    verified_count = sum(1 for t in tools if t.verification_badge == "verified")
    domain_counts: Dict[str, int] = {}
    total_trust = 0.0

    for t in tools:
        domain_counts[t.domain] = domain_counts.get(t.domain, 0) + 1
        total_trust += t.trust_score

    return TrustGraphDigest(
        instance_id=instance_id,
        instance_name=instance_name,
        tool_count=len(tools),
        verified_tool_count=verified_count,
        domain_counts=domain_counts,
        total_compilations=len(tools),
        avg_trust_score=round(total_trust / len(tools), 1),
        last_updated=now,
    )


def serialize_trust_graph_digest(digest: TrustGraphDigest) -> Dict[str, Any]:
    """Serialize a TrustGraphDigest to JSON-safe dict.

    Args:
        digest: TrustGraphDigest

    Returns:
        JSON-safe dict
    """
    return {
        "instance_id": digest.instance_id,
        "instance_name": digest.instance_name,
        "tool_count": digest.tool_count,
        "verified_tool_count": digest.verified_tool_count,
        "domain_counts": dict(digest.domain_counts),
        "total_compilations": digest.total_compilations,
        "avg_trust_score": digest.avg_trust_score,
        "last_updated": digest.last_updated,
    }
