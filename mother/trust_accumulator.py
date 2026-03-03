"""
mother/trust_accumulator.py — Cross-session trust accumulation.

LEAF module. No imports from core/ or mother/. Stdlib only.

Tracks compilation success rates, fidelity scores, and trust metrics
across sessions. Provides a persistent trust snapshot for context injection.

Persistence via SQLite (trust_snapshots table in maps.db).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "TrustSnapshot",
    "update_trust",
    "save_trust_snapshot",
    "load_trust_snapshot",
    "format_trust_context",
]


# ---------------------------------------------------------------------------
# Database location (shared with kernel/store.py, kernel/memory.py)
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".motherlabs"
_DEFAULT_DB_NAME = "maps.db"


def _db_path(db_dir: Optional[Path] = None) -> Path:
    d = db_dir or _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_DB_NAME


_TRUST_SCHEMA = """
CREATE TABLE IF NOT EXISTS trust_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    total_compilations INTEGER NOT NULL DEFAULT 0,
    successful_compilations INTEGER NOT NULL DEFAULT 0,
    session_success_rate REAL NOT NULL DEFAULT 0.0,
    rolling_success_rate REAL NOT NULL DEFAULT 0.0,
    avg_fidelity REAL NOT NULL DEFAULT 0.0,
    avg_trust_score REAL NOT NULL DEFAULT 0.0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_fidelity REAL NOT NULL DEFAULT 0.0,
    last_trust_score REAL NOT NULL DEFAULT 0.0
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_TRUST_SCHEMA)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustSnapshot:
    """Cross-session trust accumulator state."""
    total_compilations: int = 0
    successful_compilations: int = 0
    session_success_rate: float = 0.0     # current session
    rolling_success_rate: float = 0.0     # rolling (exponential decay)
    avg_fidelity: float = 0.0
    avg_trust_score: float = 0.0
    consecutive_failures: int = 0
    last_fidelity: float = 0.0
    last_trust_score: float = 0.0

    @property
    def should_pause_autonomous(self) -> bool:
        """True if autonomous compilation should be paused."""
        return self.consecutive_failures >= 3


# ---------------------------------------------------------------------------
# Pure update function
# ---------------------------------------------------------------------------

_ROLLING_ALPHA = 0.3  # Exponential moving average weight for new observations


def update_trust(
    snapshot: TrustSnapshot,
    success: bool,
    fidelity: float = 0.0,
    trust_score: float = 0.0,
) -> TrustSnapshot:
    """Update trust snapshot with a new compilation outcome.

    Pure function — returns a new snapshot.

    Args:
        snapshot: Current trust state.
        success: Whether the compilation succeeded.
        fidelity: Closed-loop fidelity score (0.0-1.0).
        trust_score: Verification overall trust score (0-100).
    """
    total = snapshot.total_compilations + 1
    successful = snapshot.successful_compilations + (1 if success else 0)

    # Session success rate
    session_rate = successful / total if total > 0 else 0.0

    # Rolling success rate (exponential moving average)
    new_obs = 1.0 if success else 0.0
    if snapshot.total_compilations == 0:
        rolling_rate = new_obs
    else:
        rolling_rate = (_ROLLING_ALPHA * new_obs +
                        (1 - _ROLLING_ALPHA) * snapshot.rolling_success_rate)

    # Rolling averages for fidelity and trust
    fidelity = max(0.0, min(1.0, fidelity))
    trust_score = max(0.0, min(100.0, trust_score))

    if snapshot.total_compilations == 0:
        avg_fid = fidelity
        avg_trust = trust_score
    else:
        avg_fid = (_ROLLING_ALPHA * fidelity +
                   (1 - _ROLLING_ALPHA) * snapshot.avg_fidelity)
        avg_trust = (_ROLLING_ALPHA * trust_score +
                     (1 - _ROLLING_ALPHA) * snapshot.avg_trust_score)

    # Consecutive failures
    if success:
        consec = 0
    else:
        consec = snapshot.consecutive_failures + 1

    return TrustSnapshot(
        total_compilations=total,
        successful_compilations=successful,
        session_success_rate=round(session_rate, 4),
        rolling_success_rate=round(rolling_rate, 4),
        avg_fidelity=round(avg_fid, 4),
        avg_trust_score=round(avg_trust, 2),
        consecutive_failures=consec,
        last_fidelity=round(fidelity, 4),
        last_trust_score=round(trust_score, 2),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_trust_snapshot(
    snapshot: TrustSnapshot,
    db_dir: Optional[Path] = None,
) -> None:
    """Persist trust snapshot to maps.db."""
    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    try:
        _ensure_schema(conn)
        conn.execute(
            "INSERT INTO trust_snapshots "
            "(timestamp, total_compilations, successful_compilations, "
            "session_success_rate, rolling_success_rate, avg_fidelity, "
            "avg_trust_score, consecutive_failures, last_fidelity, last_trust_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                snapshot.total_compilations,
                snapshot.successful_compilations,
                snapshot.session_success_rate,
                snapshot.rolling_success_rate,
                snapshot.avg_fidelity,
                snapshot.avg_trust_score,
                snapshot.consecutive_failures,
                snapshot.last_fidelity,
                snapshot.last_trust_score,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_trust_snapshot(
    db_dir: Optional[Path] = None,
) -> TrustSnapshot:
    """Load most recent trust snapshot from maps.db."""
    path = _db_path(db_dir)
    if not path.exists():
        return TrustSnapshot()

    conn = sqlite3.connect(str(path))
    try:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT total_compilations, successful_compilations, "
            "session_success_rate, rolling_success_rate, avg_fidelity, "
            "avg_trust_score, consecutive_failures, last_fidelity, last_trust_score "
            "FROM trust_snapshots ORDER BY id DESC LIMIT 1",
        ).fetchone()

        if not row:
            return TrustSnapshot()

        return TrustSnapshot(
            total_compilations=row[0],
            successful_compilations=row[1],
            session_success_rate=row[2],
            rolling_success_rate=row[3],
            avg_fidelity=row[4],
            avg_trust_score=row[5],
            consecutive_failures=row[6],
            last_fidelity=row[7],
            last_trust_score=row[8],
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_trust_context(snapshot: TrustSnapshot) -> str:
    """Render trust snapshot for prompt injection.

    Returns empty string if no compilations recorded.
    """
    if snapshot.total_compilations == 0:
        return ""

    lines = ["[Trust Accumulator]"]
    lines.append(
        f"  Compilations: {snapshot.total_compilations} "
        f"({snapshot.successful_compilations} successful)"
    )
    lines.append(
        f"  Success rate: {snapshot.rolling_success_rate:.0%} rolling, "
        f"{snapshot.session_success_rate:.0%} session"
    )
    if snapshot.avg_fidelity > 0:
        lines.append(f"  Avg fidelity: {snapshot.avg_fidelity:.2f}")
    if snapshot.avg_trust_score > 0:
        lines.append(f"  Avg trust: {snapshot.avg_trust_score:.0f}/100")
    if snapshot.consecutive_failures > 0:
        lines.append(f"  Consecutive failures: {snapshot.consecutive_failures}")
    if snapshot.consecutive_failures >= 3:
        lines.append("  ⚠ Autonomous compilation paused (3+ consecutive failures)")

    return "\n".join(lines)
