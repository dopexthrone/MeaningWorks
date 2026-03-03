"""
Mother idea journal — persistent store for self-improvement ideas.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Records ideas Mother has about her own improvement, surfaced from
conversation or observation. Ideas can be pending, in_progress, done,
or dismissed. Follows the mother/journal.py pattern exactly.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Idea:
    """A self-improvement idea."""

    idea_id: int = 0
    timestamp: float = 0.0
    description: str = ""
    source_context: str = ""
    priority: str = "normal"       # "low" | "normal" | "high"
    status: str = "pending"        # "pending" | "in_progress" | "done" | "dismissed"
    outcome: str = ""


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS idea_journal (
    idea_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source_context TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'pending',
    outcome TEXT NOT NULL DEFAULT ''
)
"""

_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


class IdeaJournal:
    """SQLite-backed idea journal. Stores in existing history.db."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def add(
        self,
        description: str,
        source_context: str = "",
        priority: str = "normal",
    ) -> int:
        """Record a new idea. Returns the idea_id."""
        ts = time.time()
        cursor = self._conn.execute(
            """INSERT INTO idea_journal
               (timestamp, description, source_context, priority, status, outcome)
               VALUES (?, ?, ?, ?, 'pending', '')""",
            (ts, description, source_context, priority),
        )
        self._conn.commit()
        return cursor.lastrowid

    def pending(self, limit: int = 10) -> List[Idea]:
        """Return pending ideas, ordered by priority (high first) then recency."""
        rows = self._conn.execute(
            """SELECT * FROM idea_journal
               WHERE status = 'pending'
               ORDER BY
                 CASE priority
                   WHEN 'high' THEN 0
                   WHEN 'normal' THEN 1
                   WHEN 'low' THEN 2
                   ELSE 3
                 END,
                 timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_idea(r) for r in rows]

    def update_status(
        self,
        idea_id: int,
        status: str,
        outcome: str = "",
    ) -> None:
        """Update the status (and optionally outcome) of an idea."""
        self._conn.execute(
            "UPDATE idea_journal SET status = ?, outcome = ? WHERE idea_id = ?",
            (status, outcome, idea_id),
        )
        self._conn.commit()

    def count_pending(self) -> int:
        """Return the number of pending ideas."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM idea_journal WHERE status = 'pending'"
        ).fetchone()
        return row[0] if row else 0

    def get(self, idea_id: int) -> Optional[Idea]:
        """Get a single idea by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM idea_journal WHERE idea_id = ?",
            (idea_id,),
        ).fetchone()
        return self._row_to_idea(row) if row else None

    def all_ideas(self, limit: int = 50) -> List[Idea]:
        """Return all ideas, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM idea_journal ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_idea(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_idea(row) -> Idea:
        """Convert a database row tuple to Idea."""
        return Idea(
            idea_id=row[0],
            timestamp=row[1],
            description=row[2],
            source_context=row[3],
            priority=row[4],
            status=row[5],
            outcome=row[6],
        )
