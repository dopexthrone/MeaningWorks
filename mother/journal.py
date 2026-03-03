"""
Mother build journal — operational memory of compilations and builds.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Records every compile and build with outcome, trust, cost, and timing.
Provides summary statistics and streak tracking for operational awareness.
"""

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class JournalEntry:
    """Single compile or build event."""

    entry_id: int = 0
    timestamp: float = 0.0
    event_type: str = ""           # "compile" | "build"
    description: str = ""
    success: bool = False
    trust_score: float = 0.0       # 0-100
    component_count: int = 0
    cost_usd: float = 0.0
    domain: str = ""
    error_summary: str = ""
    duration_seconds: float = 0.0
    project_path: str = ""
    weakest_dimension: str = ""
    dimension_scores: str = ""     # JSON: {"completeness": 67, "consistency": 92, ...}
    experiment_tag: str = ""       # Optional experiment/variant label for A/B tracking


@dataclass(frozen=True)
class JournalSummary:
    """Aggregate stats across all journal entries."""

    total_compiles: int = 0
    total_builds: int = 0
    success_rate: float = 0.0
    avg_trust: float = 0.0
    total_cost: float = 0.0
    domains: Dict[str, int] = field(default_factory=dict)
    last_entry: Optional[JournalEntry] = None
    streak: int = 0                # positive=consecutive successes, negative=consecutive failures


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS build_journal (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 0,
    trust_score REAL NOT NULL DEFAULT 0.0,
    component_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    domain TEXT NOT NULL DEFAULT '',
    error_summary TEXT NOT NULL DEFAULT '',
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    project_path TEXT NOT NULL DEFAULT '',
    weakest_dimension TEXT NOT NULL DEFAULT '',
    dimension_scores TEXT NOT NULL DEFAULT ''
)
"""


class BuildJournal:
    """SQLite-backed build journal. Stores in existing history.db."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        # Migration: add columns to existing tables
        for col, coltype in [
            ("dimension_scores", "TEXT NOT NULL DEFAULT ''"),
            ("experiment_tag", "TEXT NOT NULL DEFAULT ''"),
        ]:
            try:
                self._conn.execute(
                    f"ALTER TABLE build_journal ADD COLUMN {col} {coltype}"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
        self._conn.commit()

    def record(self, entry: JournalEntry) -> int:
        """Record a journal entry. Returns the new entry_id."""
        ts = entry.timestamp if entry.timestamp > 0 else time.time()
        cursor = self._conn.execute(
            """INSERT INTO build_journal
               (timestamp, event_type, description, success, trust_score,
                component_count, cost_usd, domain, error_summary,
                duration_seconds, project_path, weakest_dimension,
                dimension_scores, experiment_tag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts,
                entry.event_type,
                entry.description,
                1 if entry.success else 0,
                entry.trust_score,
                entry.component_count,
                entry.cost_usd,
                entry.domain,
                entry.error_summary,
                entry.duration_seconds,
                entry.project_path,
                entry.weakest_dimension,
                entry.dimension_scores,
                entry.experiment_tag,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_summary(self) -> JournalSummary:
        """Compute aggregate statistics across all entries."""
        rows = self._conn.execute(
            "SELECT * FROM build_journal ORDER BY timestamp DESC"
        ).fetchall()

        if not rows:
            return JournalSummary()

        total_compiles = 0
        total_builds = 0
        success_count = 0
        trust_scores: List[float] = []
        total_cost = 0.0
        domains: Dict[str, int] = {}

        for row in rows:
            entry = self._row_to_entry(row)
            if entry.event_type == "compile":
                total_compiles += 1
            elif entry.event_type == "build":
                total_builds += 1
            if entry.success:
                success_count += 1
            if entry.trust_score > 0:
                trust_scores.append(entry.trust_score)
            total_cost += entry.cost_usd
            if entry.domain:
                domains[entry.domain] = domains.get(entry.domain, 0) + 1

        total = total_compiles + total_builds
        success_rate = success_count / total if total > 0 else 0.0
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

        last_entry = self._row_to_entry(rows[0])

        # Streak: walk recent entries, count consecutive same-success
        streak = 0
        if rows:
            first_success = self._row_to_entry(rows[0]).success
            for row in rows:
                entry = self._row_to_entry(row)
                if entry.success == first_success:
                    streak += 1
                else:
                    break
            if not first_success:
                streak = -streak

        return JournalSummary(
            total_compiles=total_compiles,
            total_builds=total_builds,
            success_rate=round(success_rate, 4),
            avg_trust=round(avg_trust, 2),
            total_cost=round(total_cost, 6),
            domains=domains,
            last_entry=last_entry,
            streak=streak,
        )

    def search(self, query: str, limit: int = 10) -> List[JournalEntry]:
        """Search entries by description or error_summary (LIKE match)."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """SELECT * FROM build_journal
               WHERE description LIKE ? OR error_summary LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def recent(self, limit: int = 5) -> List[JournalEntry]:
        """Return most recent entries, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM build_journal ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def by_experiment(self, tag: str) -> List[JournalEntry]:
        """Return all entries with a given experiment_tag."""
        rows = self._conn.execute(
            "SELECT * FROM build_journal WHERE experiment_tag = ? ORDER BY timestamp DESC",
            (tag,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_entry(row) -> JournalEntry:
        """Convert a database row tuple to JournalEntry.

        Handles 13-col, 14-col, and 15-col rows for backwards compatibility.
        """
        return JournalEntry(
            entry_id=row[0],
            timestamp=row[1],
            event_type=row[2],
            description=row[3],
            success=bool(row[4]),
            trust_score=row[5],
            component_count=row[6],
            cost_usd=row[7],
            domain=row[8],
            error_summary=row[9],
            duration_seconds=row[10],
            project_path=row[11],
            weakest_dimension=row[12],
            dimension_scores=row[13] if len(row) > 13 else "",
            experiment_tag=row[14] if len(row) > 14 else "",
        )
