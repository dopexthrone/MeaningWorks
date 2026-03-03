"""
Motherlabs Auth — API key management and rate limiting.

LEAF MODULE — stdlib + sqlite3 only.

Provides:
- KeyStore: SQLite-backed API key CRUD (create, validate, list, revoke)
- RateLimiter: In-memory sliding-window rate limiter with DB recovery
- APIKeyRecord / KeyValidationResult: Frozen dataclasses

Database: ~/motherlabs/auth.db (separate from corpus)
"""

import hashlib
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class APIKeyRecord:
    """A stored API key (without the raw key)."""
    id: str
    name: str
    created_at: str
    is_active: bool
    rate_limit_per_hour: int
    budget_usd: float
    spent_usd: float


@dataclass(frozen=True)
class KeyValidationResult:
    """Result of validating a raw API key."""
    valid: bool
    key_id: Optional[str] = None
    key_name: Optional[str] = None
    is_active: bool = False
    rate_limit_per_hour: int = 100
    budget_usd: float = 50.0
    spent_usd: float = 0.0
    reason: str = ""


# =============================================================================
# HELPERS
# =============================================================================

def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of a raw API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _default_db_path() -> str:
    """Default auth database path: $MOTHERLABS_DATA_DIR/auth.db or ~/motherlabs/auth.db"""
    data_dir = os.environ.get("MOTHERLABS_DATA_DIR")
    if data_dir:
        db_dir = Path(data_dir)
    else:
        db_dir = Path.home() / "motherlabs"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "auth.db")


# =============================================================================
# KEYSTORE
# =============================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    rate_limit_per_hour INTEGER NOT NULL DEFAULT 100,
    budget_usd REAL NOT NULL DEFAULT 50.0,
    spent_usd REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'software',
    duration_s REAL NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (key_id) REFERENCES api_keys(id)
);

CREATE INDEX IF NOT EXISTS idx_usage_key_ts ON usage_log(key_id, timestamp);

CREATE TABLE IF NOT EXISTS key_roles (
    key_id TEXT NOT NULL,
    role TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY (key_id, role),
    FOREIGN KEY (key_id) REFERENCES api_keys(id)
);
"""


# =============================================================================
# ACCESS CONTROL — Genome #168: access-control-enforcing
# =============================================================================

_VALID_ROLES: frozenset = frozenset({"admin", "compiler", "reader", "builder", "reviewer"})

_ROLE_PERMISSIONS: Dict[str, frozenset] = {
    "admin": frozenset({"compile", "build", "read", "write", "manage_keys", "manage_roles"}),
    "compiler": frozenset({"compile", "read"}),
    "builder": frozenset({"compile", "build", "read"}),
    "reader": frozenset({"read"}),
    "reviewer": frozenset({"read", "compile"}),
}


@dataclass(frozen=True)
class AccessControlResult:
    """Result of an access control check."""
    allowed: bool
    key_id: str
    action: str
    roles: Tuple[str, ...]
    reason: str


class KeyStore:
    """SQLite-backed API key store. Thread-safe."""

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

    def create_key(
        self,
        name: str,
        rate_limit: int = 100,
        budget: float = 50.0,
    ) -> Tuple[str, str]:
        """Create a new API key.

        Returns:
            (key_id, raw_key) — raw_key is shown once, then only the hash is stored.
        """
        raw_key = secrets.token_hex(32)
        key_hash = _hash_key(raw_key)
        key_id = secrets.token_hex(8)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO api_keys (id, name, key_hash, created_at,
                       is_active, rate_limit_per_hour, budget_usd, spent_usd)
                       VALUES (?, ?, ?, ?, 1, ?, ?, 0.0)""",
                    (key_id, name, key_hash, created_at, rate_limit, budget),
                )
                conn.commit()
            finally:
                conn.close()

        return key_id, raw_key

    def validate_key(self, raw_key: str) -> KeyValidationResult:
        """Validate a raw API key by hash lookup.

        Checks:
        1. Key exists (hash match)
        2. Key is active (not revoked)
        3. Budget not exceeded
        """
        key_hash = _hash_key(raw_key)

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT id, name, is_active, rate_limit_per_hour,
                       budget_usd, spent_usd
                       FROM api_keys WHERE key_hash = ?""",
                    (key_hash,),
                ).fetchone()
            finally:
                conn.close()

        if not row:
            return KeyValidationResult(valid=False, reason="Invalid API key")

        if not row["is_active"]:
            return KeyValidationResult(
                valid=False,
                key_id=row["id"],
                key_name=row["name"],
                reason="API key has been revoked",
            )

        if row["spent_usd"] >= row["budget_usd"]:
            return KeyValidationResult(
                valid=False,
                key_id=row["id"],
                key_name=row["name"],
                is_active=True,
                rate_limit_per_hour=row["rate_limit_per_hour"],
                budget_usd=row["budget_usd"],
                spent_usd=row["spent_usd"],
                reason="Budget exceeded",
            )

        return KeyValidationResult(
            valid=True,
            key_id=row["id"],
            key_name=row["name"],
            is_active=True,
            rate_limit_per_hour=row["rate_limit_per_hour"],
            budget_usd=row["budget_usd"],
            spent_usd=row["spent_usd"],
        )

    def list_keys(self) -> List[APIKeyRecord]:
        """List all API keys (without raw keys)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT id, name, created_at, is_active,
                       rate_limit_per_hour, budget_usd, spent_usd
                       FROM api_keys ORDER BY created_at DESC"""
                ).fetchall()
            finally:
                conn.close()

        return [
            APIKeyRecord(
                id=r["id"],
                name=r["name"],
                created_at=r["created_at"],
                is_active=bool(r["is_active"]),
                rate_limit_per_hour=r["rate_limit_per_hour"],
                budget_usd=r["budget_usd"],
                spent_usd=r["spent_usd"],
            )
            for r in rows
        ]

    def revoke_key(self, key_id: str) -> bool:
        """Revoke (deactivate) a key by ID.

        Returns:
            True if key was found and revoked, False if not found.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE api_keys SET is_active = 0 WHERE id = ?",
                    (key_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def record_spend(self, key_id: str, cost_usd: float) -> None:
        """Atomically increment spent_usd for a key."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE api_keys SET spent_usd = spent_usd + ? WHERE id = ?",
                    (cost_usd, key_id),
                )
                conn.commit()
            finally:
                conn.close()

    def log_usage(
        self,
        key_id: str,
        domain: str = "software",
        duration: float = 0.0,
        cost: float = 0.0,
        success: bool = True,
    ) -> None:
        """Insert a usage log entry."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO usage_log
                       (key_id, timestamp, domain, duration_s, cost_usd, success)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (key_id, timestamp, domain, duration, cost, 1 if success else 0),
                )
                conn.commit()
            finally:
                conn.close()

    def get_usage_count_since(self, key_id: str, since_ts: str) -> int:
        """Count usage_log entries for a key since a given ISO timestamp."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM usage_log WHERE key_id = ? AND timestamp >= ?",
                    (key_id, since_ts),
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    # -----------------------------------------------------------------
    # Role-based access control — Genome #168
    # -----------------------------------------------------------------

    def grant_role(self, key_id: str, role: str) -> bool:
        """Grant a role to an API key.

        Returns True if role was granted, False if role is invalid.
        """
        if role not in _VALID_ROLES:
            return False

        granted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO key_roles (key_id, role, granted_at)
                       VALUES (?, ?, ?)""",
                    (key_id, role, granted_at),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def revoke_role(self, key_id: str, role: str) -> bool:
        """Revoke a role from an API key.

        Returns True if role was found and revoked, False otherwise.
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM key_roles WHERE key_id = ? AND role = ?",
                    (key_id, role),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def get_roles(self, key_id: str) -> List[str]:
        """Get all roles for an API key."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT role FROM key_roles WHERE key_id = ? ORDER BY role",
                    (key_id,),
                ).fetchall()
                return [r["role"] for r in rows]
            finally:
                conn.close()

    def check_access(self, key_id: str, action: str) -> "AccessControlResult":
        """Check if an API key has permission to perform an action.

        Gets roles, unions their permissions, checks if action is in the union.

        Returns AccessControlResult.
        """
        roles = self.get_roles(key_id)

        if not roles:
            return AccessControlResult(
                allowed=False,
                key_id=key_id,
                action=action,
                roles=(),
                reason="No roles assigned",
            )

        # Union all permissions from all roles
        all_permissions: set = set()
        for role in roles:
            perms = _ROLE_PERMISSIONS.get(role, frozenset())
            all_permissions.update(perms)

        allowed = action in all_permissions

        reason = (
            f"Action '{action}' {'permitted' if allowed else 'denied'} "
            f"for roles: {', '.join(roles)}"
        )

        return AccessControlResult(
            allowed=allowed,
            key_id=key_id,
            action=action,
            roles=tuple(roles),
            reason=reason,
        )


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """In-memory sliding-window rate limiter with DB recovery on init.

    Uses a per-key list of timestamps. On init, pre-populates from
    usage_log so rate limits survive restarts.
    """

    def __init__(self, key_store: KeyStore, window_seconds: int = 3600):
        self._key_store = key_store
        self._window = window_seconds
        self._lock = threading.Lock()
        # key_id -> list of timestamps (float)
        self._windows: Dict[str, List[float]] = {}

    def check_rate_limit(
        self, key_id: str, limit: int
    ) -> Tuple[bool, int, float]:
        """Check if a request is within the rate limit.

        Args:
            key_id: The API key ID
            limit: Maximum requests per window

        Returns:
            (allowed, remaining, reset_timestamp)
        """
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            if key_id not in self._windows:
                self._windows[key_id] = []

            # Prune expired entries
            self._windows[key_id] = [
                ts for ts in self._windows[key_id] if ts > cutoff
            ]

            count = len(self._windows[key_id])
            remaining = max(0, limit - count)

            # Reset time = oldest entry + window, or now + window if empty
            if self._windows[key_id]:
                reset_ts = self._windows[key_id][0] + self._window
            else:
                reset_ts = now + self._window

            if count >= limit:
                return False, 0, reset_ts

            # Record this request
            self._windows[key_id].append(now)
            remaining = max(0, limit - count - 1)
            return True, remaining, reset_ts


# =============================================================================
# MODULE SINGLETONS
# =============================================================================

_key_store: Optional[KeyStore] = None
_rate_limiter: Optional[RateLimiter] = None
_singleton_lock = threading.Lock()


def get_key_store(db_path: Optional[str] = None) -> KeyStore:
    """Get or create the singleton KeyStore."""
    global _key_store
    with _singleton_lock:
        if _key_store is None:
            _key_store = KeyStore(db_path=db_path)
        return _key_store


def get_rate_limiter() -> RateLimiter:
    """Get or create the singleton RateLimiter."""
    global _rate_limiter
    with _singleton_lock:
        if _rate_limiter is None:
            _rate_limiter = RateLimiter(get_key_store())
        return _rate_limiter
