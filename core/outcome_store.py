"""
core/outcome_store.py — Persistent CompilationOutcome storage.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Append-only store for compilation outcomes. Every compilation writes
a record. Records survive restart. This is what makes L2 memory real —
the feedback analysis operates on accumulated history, not session state.

Schema matches mother/governor_feedback.py CompilationOutcome exactly.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_DEFAULT_DB_DIR = Path.home() / ".motherlabs"
_DEFAULT_DB_NAME = "outcomes.db"


def _db_path(db_dir: Optional[Path] = None) -> Path:
    d = db_dir or _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_DB_NAME


_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    compile_id  TEXT NOT NULL,
    input_summary TEXT NOT NULL DEFAULT '',
    trust_score REAL NOT NULL DEFAULT 0.0,
    completeness REAL NOT NULL DEFAULT 0.0,
    consistency REAL NOT NULL DEFAULT 0.0,
    coherence   REAL NOT NULL DEFAULT 0.0,
    traceability REAL NOT NULL DEFAULT 0.0,
    component_count INTEGER NOT NULL DEFAULT 0,
    rejected    INTEGER NOT NULL DEFAULT 0,
    rejection_reason TEXT NOT NULL DEFAULT '',
    domain      TEXT NOT NULL DEFAULT 'software',
    compression_loss_categories TEXT NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class OutcomeRecord:
    """A persisted compilation outcome — mirrors CompilationOutcome."""

    id: int
    timestamp: float
    compile_id: str
    input_summary: str
    trust_score: float
    completeness: float
    consistency: float
    coherence: float
    traceability: float
    component_count: int
    rejected: bool
    rejection_reason: str
    domain: str
    compression_loss_categories: str = ""


class OutcomeStore:
    """Append-only SQLite store for compilation outcomes."""

    def __init__(self, db_dir: Optional[Path] = None):
        self._path = _db_path(db_dir)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        # Migration: add compression_loss_categories to existing DBs
        try:
            self._conn.execute(
                "ALTER TABLE outcomes ADD COLUMN compression_loss_categories TEXT NOT NULL DEFAULT ''"
            )
        except Exception as e:  # noqa: column already exists
            pass
        self._conn.commit()

    def append(
        self,
        compile_id: str,
        input_summary: str = "",
        trust_score: float = 0.0,
        completeness: float = 0.0,
        consistency: float = 0.0,
        coherence: float = 0.0,
        traceability: float = 0.0,
        component_count: int = 0,
        rejected: bool = False,
        rejection_reason: str = "",
        domain: str = "software",
        compression_loss_categories: str = "",
    ) -> int:
        """Append a compilation outcome. Returns row id."""
        ts = time.time()
        cursor = self._conn.execute(
            """INSERT INTO outcomes
               (timestamp, compile_id, input_summary, trust_score,
                completeness, consistency, coherence, traceability,
                component_count, rejected, rejection_reason, domain,
                compression_loss_categories)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ts, compile_id, input_summary, trust_score,
                completeness, consistency, coherence, traceability,
                component_count, 1 if rejected else 0, rejection_reason, domain,
                compression_loss_categories,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def recent(self, limit: int = 50) -> List[OutcomeRecord]:
        """Load most recent outcomes, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count(self) -> int:
        """Total outcome count."""
        row = self._conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()
        return row[0] if row else 0

    def rejection_rate(self, last_n: int = 50) -> float:
        """Rejection rate over last N outcomes."""
        rows = self._conn.execute(
            """SELECT rejected FROM outcomes
               ORDER BY timestamp DESC LIMIT ?""",
            (last_n,),
        ).fetchall()
        if not rows:
            return 0.0
        return sum(r[0] for r in rows) / len(rows)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_record(row) -> OutcomeRecord:
        return OutcomeRecord(
            id=row[0],
            timestamp=row[1],
            compile_id=row[2],
            input_summary=row[3],
            trust_score=row[4],
            completeness=row[5],
            consistency=row[6],
            coherence=row[7],
            traceability=row[8],
            component_count=row[9],
            rejected=bool(row[10]),
            rejection_reason=row[11],
            domain=row[12],
            compression_loss_categories=row[13] if len(row) > 13 else "",
        )
