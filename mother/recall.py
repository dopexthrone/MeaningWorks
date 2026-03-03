"""
Mother semantic recall — FTS5-backed conversation retrieval.

LEAF module. Stdlib + sqlite3. No imports from core/ or mother/.

Provides RecallEngine that indexes conversation messages into
an FTS5 virtual table in the existing history.db and enables
keyword-based semantic retrieval for context injection.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Common English stopwords for keyword extraction
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
    "and", "or", "but", "not", "for", "with", "this", "that",
    "was", "are", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "can", "may", "might", "shall", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "his", "her", "our",
    "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "if", "then", "so", "just", "about",
})


@dataclass(frozen=True)
class RecallResult:
    """A single recalled message with relevance metadata."""

    message_id: int
    role: str
    content: str
    timestamp: float
    session_id: str
    relevance_rank: int


class RecallEngine:
    """FTS5-backed conversation retrieval engine.

    Operates on the same SQLite database as ConversationStore.
    Creates an FTS5 virtual table for full-text search over messages.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._fts_available = True
        self._open()

    def _open(self) -> None:
        """Open connection and ensure FTS table exists."""
        try:
            self._conn = sqlite3.connect(str(self._db_path), timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_fts_table()
        except Exception:
            self._fts_available = False

    def _ensure_fts_table(self) -> None:
        """Create FTS5 table and backfill from messages if needed."""
        try:
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(message_id UNINDEXED, content)
            """)
            self._conn.commit()
            self._backfill()
        except sqlite3.OperationalError:
            # FTS5 not available in this SQLite build
            self._fts_available = False

    def _backfill(self) -> None:
        """Backfill FTS table from messages if it's behind."""
        if not self._fts_available or not self._conn:
            return
        try:
            # Count existing FTS entries
            fts_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages_fts"
            ).fetchone()[0]

            # Count indexable messages
            msg_count = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role IN ('user', 'assistant')"
            ).fetchone()[0]

            if fts_count < msg_count:
                # Clear and re-insert all
                self._conn.execute("DELETE FROM messages_fts")
                self._conn.execute("""
                    INSERT INTO messages_fts (message_id, content)
                    SELECT id, content FROM messages
                    WHERE role IN ('user', 'assistant')
                """)
                self._conn.commit()
        except sqlite3.OperationalError:
            self._fts_available = False

    def index_message(self, message_id: int, content: str) -> None:
        """Index a single new message for recall."""
        if not self._fts_available or not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO messages_fts (message_id, content) VALUES (?, ?)",
                (message_id, content),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract search keywords from query string."""
        words = query.lower().split()
        keywords = [
            w for w in words
            if w not in _STOPWORDS and len(w) > 1
        ]
        return keywords[:5]  # Limit to 5 keywords

    def search(self, query: str, limit: int = 10) -> List[RecallResult]:
        """Search messages by keyword. Returns BM25-ranked results."""
        if not self._fts_available or not self._conn:
            return []

        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        # OR-joined FTS5 MATCH
        match_expr = " OR ".join(keywords)

        try:
            rows = self._conn.execute(
                """
                SELECT f.message_id, m.role, m.content, m.timestamp, m.session_id,
                       rank
                FROM messages_fts f
                JOIN messages m ON m.id = f.message_id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, limit),
            ).fetchall()

            results = []
            for i, row in enumerate(rows):
                results.append(RecallResult(
                    message_id=row[0],
                    role=row[1],
                    content=row[2],
                    timestamp=row[3],
                    session_id=row[4],
                    relevance_rank=i + 1,
                ))
            return results
        except sqlite3.OperationalError:
            return []

    def recall_for_context(self, query: str, max_tokens: int = 1000) -> str:
        """Search and format results as a context block.

        Returns a [Recalled] block with truncated snippets that fits
        within the token budget (estimated at ~4 chars/token).
        """
        results = self.search(query, limit=10)
        if not results:
            return ""

        char_budget = max_tokens * 4
        lines = ["[Recalled]"]
        used = len(lines[0])

        for r in results:
            # Truncate content to 200 chars
            snippet = r.content[:200]
            if len(r.content) > 200:
                snippet += "..."
            from mother.temporal import is_stale
            staleness = "[stale] " if is_stale(r.timestamp, max_age_hours=168.0) else ""
            line = f"- {r.role}: {staleness}\"{snippet}\""
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

        if len(lines) == 1:
            return ""  # Only header, no results fit

        return "\n".join(lines)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
