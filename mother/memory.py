"""
Conversation store — persistent chat history in SQLite.

Database at ~/.motherlabs/history.db.
"""

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any


DEFAULT_HISTORY_PATH = Path.home() / ".motherlabs" / "history.db"


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str
    timestamp: float = 0.0
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def to_llm_message(self) -> Dict[str, str]:
        """Convert to LLM-compatible message dict."""
        return {"role": self.role, "content": self.content}


class ConversationStore:
    """SQLite-backed conversation history."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or DEFAULT_HISTORY_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._session_id = str(uuid.uuid4())

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages (session_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sense_memory (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS relationship_insights (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL DEFAULT '{}',
                computed_at REAL NOT NULL DEFAULT 0.0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experience_memory (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self._conn.commit()

    @property
    def session_id(self) -> str:
        return self._session_id

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Add a message. Returns the message ID."""
        sid = session_id or self._session_id
        ts = time.time()
        meta = json.dumps(metadata or {})
        cursor = self._conn.execute(
            "INSERT INTO messages (role, content, timestamp, session_id, metadata) VALUES (?, ?, ?, ?, ?)",
            (role, content, ts, sid, meta),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_history(
        self,
        limit: int = 50,
        session_id: Optional[str] = None,
    ) -> List[ChatMessage]:
        """Get recent messages for a session."""
        sid = session_id or self._session_id
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp, session_id, metadata "
            "FROM messages WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
            (sid, limit),
        ).fetchall()
        messages = []
        for row in reversed(rows):
            messages.append(ChatMessage(
                id=row[0],
                role=row[1],
                content=row[2],
                timestamp=row[3],
                session_id=row[4],
                metadata=json.loads(row[5]) if row[5] else {},
            ))
        return messages

    def get_context_window(
        self,
        max_tokens: int = 4000,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Get recent messages that fit within a token budget.

        Rough estimate: 1 token ~ 4 chars.
        """
        messages = self.get_history(limit=100, session_id=session_id)
        result = []
        budget = max_tokens * 3  # chars (conservative: ~3 chars/token)
        used = 0
        for msg in reversed(messages):
            cost = len(msg.content)
            if used + cost > budget:
                break
            result.insert(0, msg.to_llm_message())
            used += cost
        return result

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with their message counts."""
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*) as count, MIN(timestamp) as first, MAX(timestamp) as last "
            "FROM messages GROUP BY session_id ORDER BY last DESC"
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "message_count": row[1],
                "first_message": row[2],
                "last_message": row[3],
            }
            for row in rows
        ]

    def clear_session(self, session_id: Optional[str] = None) -> int:
        """Delete all messages in a session. Returns count deleted."""
        sid = session_id or self._session_id
        cursor = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (sid,)
        )
        self._conn.commit()
        return cursor.rowcount

    def get_message_count_all(self) -> int:
        """Total messages across all sessions."""
        row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def get_session_topics(self, session_id: Optional[str] = None, limit: int = 3) -> List[str]:
        """First N user messages in a session, truncated to 80 chars."""
        sid = session_id or self._session_id
        rows = self._conn.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
            "ORDER BY timestamp ASC LIMIT ?",
            (sid, limit),
        ).fetchall()
        return [row[0][:80] for row in rows]

    def get_cross_session_summary(self, max_sessions: int = 5) -> Dict[str, Any]:
        """Summary across recent sessions for memory injection.

        Returns dict with: total_sessions, total_messages, last_session_time,
        topics (first user msg per session), days_since_last.
        """
        sessions = self.list_sessions()
        total_sessions = len(sessions)
        total_messages = self.get_message_count_all()

        if not sessions:
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "last_session_time": None,
                "topics": [],
                "days_since_last": None,
            }

        last_session_time = sessions[0]["last_message"]
        days_since_last = (time.time() - last_session_time) / 86400

        # Gather first user message from each recent session as topics
        topics = []
        for session in sessions[:max_sessions]:
            sid = session["session_id"]
            session_topics = self.get_session_topics(session_id=sid, limit=1)
            if session_topics:
                topics.append(session_topics[0])

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "last_session_time": last_session_time,
            "topics": topics,
            "days_since_last": days_since_last,
        }

    def load_sense_memory(self) -> Optional[str]:
        """Load serialized sense memory JSON. Returns None if not stored."""
        row = self._conn.execute(
            "SELECT data FROM sense_memory WHERE id = 1"
        ).fetchone()
        return row[0] if row else None

    def save_sense_memory(self, data: str) -> None:
        """Upsert singleton sense memory row."""
        self._conn.execute(
            "INSERT INTO sense_memory (id, data) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
            (data,),
        )
        self._conn.commit()

    def load_experience_memory(self) -> Optional[str]:
        """Load serialized experience memory JSON. Returns None if not stored."""
        try:
            row = self._conn.execute(
                "SELECT data FROM experience_memory WHERE id = 1"
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def save_experience_memory(self, data: str) -> None:
        """Upsert singleton experience memory row."""
        try:
            self._conn.execute(
                "INSERT INTO experience_memory (id, data) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (data,),
            )
            self._conn.commit()
        except Exception:
            # Table may not exist yet in older DBs — create and retry
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS experience_memory (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data TEXT NOT NULL DEFAULT '{}'
                )
            """)
            self._conn.execute(
                "INSERT INTO experience_memory (id, data) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (data,),
            )
            self._conn.commit()

    def get_sessions_since(self, timestamp: float) -> int:
        """Count distinct sessions with messages after timestamp."""
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages WHERE timestamp > ?",
            (timestamp,),
        ).fetchone()
        return row[0] if row else 0

    def get_average_user_message_length(self, session_id: Optional[str] = None) -> float:
        """Average character length of user messages in a session."""
        sid = session_id or self._session_id
        row = self._conn.execute(
            "SELECT AVG(LENGTH(content)) FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_all_messages(self, limit: int = 5000) -> List:
        """All messages across all sessions, oldest first.

        Returns list of (role, content, timestamp, session_id) tuples.
        """
        rows = self._conn.execute(
            "SELECT role, content, timestamp, session_id "
            "FROM messages ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    def save_relationship_insights(self, data: str, computed_at: float) -> None:
        """Upsert singleton relationship insights row."""
        self._conn.execute(
            "INSERT INTO relationship_insights (id, data, computed_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET data = excluded.data, computed_at = excluded.computed_at",
            (data, computed_at),
        )
        self._conn.commit()

    def load_relationship_insights(self) -> Optional[tuple]:
        """Load cached relationship insights. Returns (json_str, computed_at) or None."""
        row = self._conn.execute(
            "SELECT data, computed_at FROM relationship_insights WHERE id = 1"
        ).fetchone()
        return (row[0], row[1]) if row else None

    def get_last_user_message_time(self, session_id: Optional[str] = None) -> Optional[float]:
        """Get timestamp of most recent user message. Returns None if no user messages."""
        sid = session_id or self._session_id
        row = self._conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def close(self) -> None:
        self._conn.close()
