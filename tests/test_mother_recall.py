"""
Tests for mother/recall.py — semantic retrieval via FTS5.
"""

import sqlite3
import time

import pytest
from pathlib import Path

from mother.recall import RecallEngine, RecallResult, _STOPWORDS


@pytest.fixture
def recall_db(tmp_path):
    """Create a test DB with messages table pre-populated."""
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
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
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def populated_db(recall_db):
    """DB with several messages for search testing."""
    conn = sqlite3.connect(str(recall_db))
    messages = [
        ("user", "How do I build a tattoo booking system?", 1000.0, "s1"),
        ("assistant", "I can help you build a tattoo booking system with appointments and payments.", 1001.0, "s1"),
        ("user", "Can you add a calendar view?", 1002.0, "s1"),
        ("assistant", "Adding a calendar view for managing appointments.", 1003.0, "s1"),
        ("user", "What about payment processing?", 1004.0, "s2"),
        ("assistant", "For payment processing, we can integrate Stripe or Square.", 1005.0, "s2"),
        ("user", "I want to manage inventory too", 1006.0, "s2"),
        ("assistant", "Inventory management can track ink, needles, and supplies.", 1007.0, "s2"),
    ]
    for role, content, ts, sid in messages:
        conn.execute(
            "INSERT INTO messages (role, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
            (role, content, ts, sid),
        )
    conn.commit()
    conn.close()
    return recall_db


class TestRecallResult:

    def test_frozen(self):
        r = RecallResult(
            message_id=1, role="user", content="test",
            timestamp=1000.0, session_id="s1", relevance_rank=1,
        )
        with pytest.raises(AttributeError):
            r.message_id = 2

    def test_fields(self):
        r = RecallResult(
            message_id=1, role="user", content="hello",
            timestamp=1000.0, session_id="s1", relevance_rank=1,
        )
        assert r.message_id == 1
        assert r.role == "user"
        assert r.content == "hello"
        assert r.relevance_rank == 1


class TestFTSTableCreation:

    def test_creates_fts_table(self, recall_db):
        engine = RecallEngine(recall_db)
        conn = sqlite3.connect(str(recall_db))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ).fetchall()
        assert len(tables) == 1
        engine.close()
        conn.close()

    def test_backfill_existing_messages(self, populated_db):
        engine = RecallEngine(populated_db)
        conn = sqlite3.connect(str(populated_db))
        count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        assert count == 8  # All 8 messages backfilled
        engine.close()
        conn.close()


class TestIndexMessage:

    def test_index_and_search_roundtrip(self, recall_db):
        engine = RecallEngine(recall_db)
        # Manually insert a message
        conn = sqlite3.connect(str(recall_db))
        conn.execute(
            "INSERT INTO messages (role, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
            ("user", "Build me a weather dashboard", 1000.0, "s1"),
        )
        conn.commit()
        msg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        engine.index_message(msg_id, "Build me a weather dashboard")
        results = engine.search("weather dashboard")
        assert len(results) >= 1
        assert results[0].content == "Build me a weather dashboard"
        engine.close()


class TestSearch:

    def test_search_finds_relevant(self, populated_db):
        engine = RecallEngine(populated_db)
        results = engine.search("tattoo booking")
        assert len(results) >= 1
        assert any("tattoo" in r.content.lower() for r in results)
        engine.close()

    def test_search_empty_query(self, populated_db):
        engine = RecallEngine(populated_db)
        results = engine.search("")
        assert results == []
        engine.close()

    def test_search_no_matches(self, populated_db):
        engine = RecallEngine(populated_db)
        results = engine.search("xyznonexistent")
        assert results == []
        engine.close()

    def test_search_limit(self, populated_db):
        engine = RecallEngine(populated_db)
        results = engine.search("booking system calendar payment", limit=2)
        assert len(results) <= 2
        engine.close()

    def test_keyword_extraction_filters_stopwords(self, populated_db):
        engine = RecallEngine(populated_db)
        keywords = engine._extract_keywords("what is the best way to build a system")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "best" in keywords or "way" in keywords or "build" in keywords
        engine.close()

    def test_keyword_extraction_limits_to_5(self, populated_db):
        engine = RecallEngine(populated_db)
        keywords = engine._extract_keywords(
            "alpha bravo charlie delta echo foxtrot golf hotel"
        )
        assert len(keywords) <= 5
        engine.close()

    def test_cross_session_search(self, populated_db):
        """Search should find messages from both sessions."""
        engine = RecallEngine(populated_db)
        results = engine.search("payment")
        assert len(results) >= 1
        # Check we can find s2 content
        sessions = {r.session_id for r in results}
        assert "s2" in sessions
        engine.close()

    def test_relevance_rank_assigned(self, populated_db):
        engine = RecallEngine(populated_db)
        results = engine.search("tattoo")
        for i, r in enumerate(results):
            assert r.relevance_rank == i + 1
        engine.close()


class TestRecallForContext:

    def test_format_output(self, populated_db):
        engine = RecallEngine(populated_db)
        block = engine.recall_for_context("tattoo booking")
        assert block.startswith("[Recalled]")
        assert "tattoo" in block.lower()
        engine.close()

    def test_token_budget(self, populated_db):
        engine = RecallEngine(populated_db)
        # Very small budget
        block = engine.recall_for_context("tattoo booking calendar payment", max_tokens=50)
        # Should be limited
        assert len(block) < 250  # 50 tokens * ~4 chars + header
        engine.close()

    def test_empty_result(self, populated_db):
        engine = RecallEngine(populated_db)
        block = engine.recall_for_context("xyznonexistent")
        assert block == ""
        engine.close()

    def test_content_truncation(self, recall_db):
        """Long messages should be truncated in recall output."""
        conn = sqlite3.connect(str(recall_db))
        long_content = "x" * 500
        conn.execute(
            "INSERT INTO messages (role, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
            ("user", long_content, 1000.0, "s1"),
        )
        conn.commit()
        msg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        engine = RecallEngine(recall_db)
        engine.index_message(msg_id, long_content)
        block = engine.recall_for_context("xxx")
        if block:
            # Content should be truncated to 200 chars + "..."
            assert "..." in block
        engine.close()


class TestSpecialCases:

    def test_special_characters_in_search(self, populated_db):
        """Search with special characters should not crash."""
        engine = RecallEngine(populated_db)
        results = engine.search('build "system" (test)')
        # Should not raise
        assert isinstance(results, list)
        engine.close()

    def test_close_and_reopen(self, populated_db):
        engine = RecallEngine(populated_db)
        results1 = engine.search("tattoo")
        engine.close()

        # Reopen
        engine2 = RecallEngine(populated_db)
        results2 = engine2.search("tattoo")
        assert len(results2) == len(results1)
        engine2.close()

    def test_graceful_degradation_no_messages_table(self, tmp_path):
        """If messages table doesn't exist, should degrade gracefully."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()

        engine = RecallEngine(db_path)
        results = engine.search("anything")
        assert results == []
        engine.close()

    def test_empty_db_no_crash(self, recall_db):
        """Empty messages table should work fine."""
        engine = RecallEngine(recall_db)
        results = engine.search("anything")
        assert results == []
        engine.close()
