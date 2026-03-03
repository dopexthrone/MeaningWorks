"""
Governor rejection log — SQLite-backed rejection event store.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Records tool import rejections, extracts patterns, generates
remediation hints that feed back into the compile flow.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RejectionEvent:
    """A single governor rejection event."""

    timestamp: str
    package_id: str
    source_instance: str
    rejection_reason: str
    failed_check: str  # "provenance" | "trust" | "code_safety" | "blueprint" | "package_id" | "fingerprint"
    trust_score: float
    provenance_depth: int


@dataclass(frozen=True)
class RejectionSummary:
    """Aggregate rejection patterns."""

    total_rejections: int
    by_check: Dict[str, int]
    recent_reasons: Tuple[str, ...]
    remediation_hints: Tuple[str, ...]


# Check type → remediation hint
_REMEDIATION_MAP = {
    "trust": "Improve verification scores before export — aim for 70%+ trust.",
    "provenance": "Ensure full provenance chain on export — every hop must trace to a compilation.",
    "code_safety": "Avoid exec/eval/subprocess in tool code — these trigger safety checks.",
    "blueprint": "Ensure blueprint has components, relationships, and constraints before export.",
    "package_id": "Package ID validation failed — re-export the tool.",
    "fingerprint": "Duplicate fingerprint — this tool is already registered.",
}


class RejectionLog:
    """SQLite-backed rejection event store.

    Default DB: ~/.motherlabs/rejections.db
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".motherlabs")
            os.makedirs(config_dir, exist_ok=True)
            db_path = os.path.join(config_dir, "rejections.db")
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the rejections table if it doesn't exist."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rejections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    package_id TEXT NOT NULL,
                    source_instance TEXT NOT NULL,
                    rejection_reason TEXT NOT NULL,
                    failed_check TEXT NOT NULL,
                    trust_score REAL NOT NULL,
                    provenance_depth INTEGER NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def record(self, event: RejectionEvent) -> None:
        """Store a rejection event."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """INSERT INTO rejections
                   (timestamp, package_id, source_instance, rejection_reason,
                    failed_check, trust_score, provenance_depth)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.timestamp,
                    event.package_id,
                    event.source_instance,
                    event.rejection_reason,
                    event.failed_check,
                    event.trust_score,
                    event.provenance_depth,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_summary(self) -> RejectionSummary:
        """Aggregate rejection patterns."""
        conn = sqlite3.connect(self._db_path)
        try:
            # Total
            total = conn.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
            if total == 0:
                return RejectionSummary(
                    total_rejections=0,
                    by_check={},
                    recent_reasons=(),
                    remediation_hints=(),
                )

            # By check type
            rows = conn.execute(
                "SELECT failed_check, COUNT(*) FROM rejections GROUP BY failed_check"
            ).fetchall()
            by_check = {row[0]: row[1] for row in rows}

            # Recent reasons (last 5)
            recent = conn.execute(
                "SELECT rejection_reason FROM rejections ORDER BY id DESC LIMIT 5"
            ).fetchall()
            recent_reasons = tuple(row[0] for row in recent)

            # Generate remediation hints
            hints = self.generate_remediation_hints(by_check)

            return RejectionSummary(
                total_rejections=total,
                by_check=by_check,
                recent_reasons=recent_reasons,
                remediation_hints=hints,
            )
        finally:
            conn.close()

    def get_recent(self, limit: int = 10) -> List[RejectionEvent]:
        """Last N rejections, newest first."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                """SELECT timestamp, package_id, source_instance, rejection_reason,
                          failed_check, trust_score, provenance_depth
                   FROM rejections ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [
                RejectionEvent(
                    timestamp=row[0],
                    package_id=row[1],
                    source_instance=row[2],
                    rejection_reason=row[3],
                    failed_check=row[4],
                    trust_score=row[5],
                    provenance_depth=row[6],
                )
                for row in rows
            ]
        finally:
            conn.close()

    @staticmethod
    def generate_remediation_hints(
        by_check: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, ...]:
        """Pattern-based remediation hints from rejection distribution."""
        if not by_check:
            return ()

        hints = []
        # Sort by count descending — most common failures first
        for check, _count in sorted(by_check.items(), key=lambda x: -x[1]):
            hint = _REMEDIATION_MAP.get(check)
            if hint:
                hints.append(hint)
        return tuple(hints)
