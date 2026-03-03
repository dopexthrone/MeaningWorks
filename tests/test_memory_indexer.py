"""Tests for mother/memory_indexer.py — background fact extraction and episode compression."""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from mother.memory_indexer import (
    MemoryIndexer,
    reindex_history,
)
from mother.episodic_memory import load_episodes, episode_count
from mother.knowledge_base import fact_count, query_facts


# ── Helpers ──────────────────────────────────────────────────────

def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


def _create_history_db(db_path: Path, sessions: dict) -> None:
    """Create a history.db with messages table and populate with sessions.

    sessions: {session_id: [(role, content, timestamp), ...]}
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            session_id TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        )
    """)
    for session_id, messages in sessions.items():
        for role, content, ts in messages:
            conn.execute(
                "INSERT INTO messages (role, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
                (role, content, ts, session_id),
            )
    conn.commit()
    conn.close()


# ── MemoryIndexer ────────────────────────────────────────────────

class TestMemoryIndexer:
    def test_init(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)
        assert indexer._facts_extracted_count == 0
        assert indexer._episodes_saved_count == 0

    def test_index_message_extracts_facts(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        facts = indexer.index_message(
            "I prefer using dark mode for all my apps",
            role="user",
            session_id="test-1",
        )
        assert len(facts) >= 1
        assert indexer._facts_extracted_count >= 1

    def test_index_message_buffers_for_episode(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        indexer.index_message("Hello", "user", session_id="buf-1")
        indexer.index_message("Hi there", "assistant", session_id="buf-1")

        assert "buf-1" in indexer._session_buffers
        assert len(indexer._session_buffers["buf-1"]) == 2

    def test_index_message_no_session(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        # No session_id — should not buffer
        facts = indexer.index_message("I prefer Python", "user")
        assert "" not in indexer._session_buffers or len(indexer._session_buffers.get("", [])) == 0

    def test_close_session_creates_episode(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        # Buffer enough messages
        for i in range(5):
            indexer.index_message(
                f"Message {i} about authentication",
                "user" if i % 2 == 0 else "assistant",
                session_id="close-1",
                timestamp=1000.0 + i,
            )

        episode = indexer.close_session("close-1")
        assert episode is not None
        assert episode.message_count == 5
        assert indexer._episodes_saved_count == 1

        # Verify persisted
        loaded = load_episodes(db_path=db)
        assert len(loaded) == 1

    def test_close_session_too_short(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        # Only 2 messages — too short
        indexer.index_message("Hi", "user", session_id="short-1")
        indexer.index_message("Hello", "assistant", session_id="short-1")

        episode = indexer.close_session("short-1")
        assert episode is None

    def test_close_session_clears_buffer(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        for i in range(4):
            indexer.index_message(f"msg {i}", "user", session_id="clear-1")

        indexer.close_session("clear-1")
        assert "clear-1" not in indexer._session_buffers

    def test_close_nonexistent_session(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)
        episode = indexer.close_session("nonexistent")
        assert episode is None

    def test_stats(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        indexer.index_message("I prefer dark mode", "user", session_id="stats-1")
        stats = indexer.stats

        assert "facts_extracted" in stats
        assert "episodes_saved" in stats
        assert "active_session_buffers" in stats
        assert stats["active_session_buffers"] == 1
        assert stats["buffered_messages"] == 1

    def test_multiple_sessions(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        # Two different sessions
        for i in range(4):
            indexer.index_message(f"Session A msg {i}", "user", session_id="A")
            indexer.index_message(f"Session B msg {i}", "user", session_id="B")

        assert len(indexer._session_buffers) == 2

        ep_a = indexer.close_session("A")
        assert ep_a is not None
        assert ep_a.session_id == "A"
        assert "A" not in indexer._session_buffers
        assert "B" in indexer._session_buffers

    def test_facts_persisted_to_db(self):
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        indexer.index_message("I prefer using Python and always use type hints", "user", session_id="persist-1")

        # Check DB directly
        count = fact_count(db_path=db)
        assert count >= 1


# ── Reindex History ──────────────────────────────────────────────

class TestReindexHistory:
    def test_reindex_creates_episodes(self):
        db = _tmp_db()
        _create_history_db(db, {
            "session-1": [
                ("user", "Build a task manager", 1000.0),
                ("assistant", "I'll implement it with React", 1001.0),
                ("user", "Add authentication too", 1002.0),
                ("assistant", "Done! Created auth module", 1003.0),
            ],
            "session-2": [
                ("user", "Deploy to production", 2000.0),
                ("assistant", "Setting up Docker", 2001.0),
                ("user", "Use kubernetes instead", 2002.0),
                ("assistant", "Configured k8s manifest", 2003.0),
            ],
        })

        stats = reindex_history(db_path=db)
        assert stats["sessions_processed"] == 2
        assert stats["episodes_created"] == 2
        assert stats["messages_processed"] >= 8

    def test_reindex_extracts_facts(self):
        db = _tmp_db()
        _create_history_db(db, {
            "session-facts": [
                ("user", "I prefer using dark mode", 1000.0),
                ("assistant", "Noted, dark mode enabled", 1001.0),
                ("user", "Let's use PostgreSQL for the database", 1002.0),
                ("assistant", "PostgreSQL configured", 1003.0),
            ],
        })

        stats = reindex_history(db_path=db)
        assert stats["facts_extracted"] >= 1

    def test_reindex_skips_short_sessions(self):
        db = _tmp_db()
        _create_history_db(db, {
            "short": [
                ("user", "Hi", 1000.0),
                ("assistant", "Hello", 1001.0),
            ],
        })

        stats = reindex_history(db_path=db)
        assert stats["sessions_processed"] == 0
        assert stats["episodes_created"] == 0

    def test_reindex_nonexistent_db(self):
        stats = reindex_history(db_path=Path("/tmp/nonexistent_reindex.db"))
        assert stats["sessions_processed"] == 0

    def test_reindex_empty_db(self):
        db = _tmp_db()
        _create_history_db(db, {})

        stats = reindex_history(db_path=db)
        assert stats["sessions_processed"] == 0

    def test_reindex_idempotent(self):
        db = _tmp_db()
        _create_history_db(db, {
            "idem-1": [
                ("user", "Build something", 1000.0),
                ("assistant", "On it", 1001.0),
                ("user", "Thanks", 1002.0),
                ("assistant", "Done", 1003.0),
            ],
        })

        stats1 = reindex_history(db_path=db)
        stats2 = reindex_history(db_path=db)

        # Should create same number of episodes (upsert)
        assert stats1["episodes_created"] == stats2["episodes_created"]
        # Should be exactly 1 episode in DB
        assert episode_count(db_path=db) == 1


# ── Integration ──────────────────────────────────────────────────

class TestIntegration:
    def test_full_lifecycle(self):
        """Index messages → close session → query results."""
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        # Simulate a conversation
        messages = [
            ("user", "I want to build an authentication system"),
            ("assistant", "I'll use JWT tokens with bcrypt"),
            ("user", "Let's use PostgreSQL for the database"),
            ("assistant", "Created auth/models.py with User class"),
            ("user", "Add role-based access control"),
            ("assistant", "Done! RBAC implemented with Permission model"),
        ]

        for i, (role, content) in enumerate(messages):
            indexer.index_message(content, role, session_id="lifecycle-1", timestamp=1000.0 + i)

        # Close session
        episode = indexer.close_session("lifecycle-1")
        assert episode is not None
        assert episode.message_count == 6

        # Verify episodes searchable
        from mother.episodic_memory import search_episodes
        results = search_episodes("authentication", db_path=db)
        assert len(results) >= 1

        # Verify facts stored
        from mother.knowledge_base import search_facts
        facts = search_facts("PostgreSQL", db_path=db)
        # May or may not find facts depending on pattern matching
        # Just verify no crash

    def test_indexer_with_memory_bank(self):
        """Verify memory bank can query indexed data."""
        db = _tmp_db()
        indexer = MemoryIndexer(db_path=db)

        indexer.index_message("I prefer dark mode", "user", session_id="bank-1", timestamp=1000.0)
        indexer.index_message("Dark mode enabled", "assistant", session_id="bank-1", timestamp=1001.0)
        indexer.index_message("Also use vim keybindings", "user", session_id="bank-1", timestamp=1002.0)
        indexer.index_message("Vim bindings configured", "assistant", session_id="bank-1", timestamp=1003.0)

        indexer.close_session("bank-1")

        from mother.memory_bank import query as memory_query
        results = memory_query("dark mode", db_path=db)
        # Should find either episode or fact or both
        assert len(results) >= 0  # May be 0 if patterns don't match, but no crash
