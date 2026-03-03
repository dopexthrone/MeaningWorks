"""Per-user state tracking — SQLite-backed concurrency gate + history.

Enforces 1 concurrent compilation per user. Cleans up stale locks on startup.
"""

import os
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("motherlabs.bots.user_state")

STALE_THRESHOLD = 300  # 5 minutes — compilations older than this are considered stale

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_compilations (
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    task_id TEXT,
    started_at REAL NOT NULL,
    finished_at REAL,
    status TEXT NOT NULL DEFAULT 'running',
    PRIMARY KEY (user_id, platform, started_at)
);
CREATE INDEX IF NOT EXISTS idx_user_active
    ON user_compilations(user_id, platform, status);
"""


class UserStateStore:
    """SQLite-backed user state with atomic concurrency gating."""

    def __init__(self, db_path: Optional[str] = None):
        data_dir = os.environ.get("MOTHERLABS_DATA_DIR", "")
        if db_path:
            self._db_path = db_path
        elif data_dir:
            self._db_path = os.path.join(data_dir, "bot_users.db")
        else:
            self._db_path = str(Path.home() / ".motherlabs" / "bot_users.db")

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._cleanup_stale()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _cleanup_stale(self):
        """Mark stale running compilations as failed (bot restart recovery)."""
        cutoff = time.time() - STALE_THRESHOLD
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE user_compilations SET status='failed', finished_at=? "
                "WHERE status='running' AND started_at < ?",
                (time.time(), cutoff),
            )
            conn.commit()
        finally:
            conn.close()

    def try_start_compilation(self, user_id: str, platform: str, task_id: str) -> bool:
        """Atomically try to start a compilation. Returns False if user already has one running."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM user_compilations "
                "WHERE user_id=? AND platform=? AND status='running'",
                (user_id, platform),
            ).fetchone()

            if row:
                return False

            conn.execute(
                "INSERT INTO user_compilations (user_id, platform, task_id, started_at, status) "
                "VALUES (?, ?, ?, ?, 'running')",
                (user_id, platform, task_id, time.time()),
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def finish_compilation(self, user_id: str, platform: str, status: str = "done"):
        """Mark the user's active compilation as finished."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE user_compilations SET status=?, finished_at=? "
                "WHERE user_id=? AND platform=? AND status='running'",
                (status, time.time(), user_id, platform),
            )
            conn.commit()
        finally:
            conn.close()
