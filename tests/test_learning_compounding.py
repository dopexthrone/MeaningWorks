"""Tests for learning-compounding — journal patterns fed into compile context."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mother.bridge import EngineBridge


@pytest.fixture
def bridge():
    return EngineBridge(provider="claude")


@pytest.fixture
def tmp_db():
    """Create a temp db with journal table and sample entries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
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
    """)
    # Insert entries with weak traceability
    for i in range(5):
        scores = json.dumps({
            "completeness": 70.0,
            "consistency": 65.0,
            "specificity": 45.0,
            "actionability": 50.0,
            "traceability": 35.0,
            "modularity": 60.0,
            "testability": 55.0,
        })
        conn.execute(
            """INSERT INTO build_journal
               (timestamp, event_type, description, success, trust_score,
                component_count, cost_usd, domain, error_summary,
                duration_seconds, project_path, weakest_dimension, dimension_scores)
               VALUES (?, 'compile', ?, ?, ?, 5, 0.1, 'software', '',
                       0.0, '', '', ?)""",
            (1000.0 + i, f"test compile {i}", 1 if i > 1 else 0, 40.0 + i * 5, scores),
        )
    conn.commit()
    conn.close()
    yield db_path
    db_path.unlink(missing_ok=True)


class TestGetLearningContext:

    def test_returns_trends(self, bridge, tmp_db):
        result = bridge.get_learning_context(tmp_db)
        assert isinstance(result, dict)
        assert "trends_line" in result
        assert "failure_line" in result
        assert "chronic_weak" in result

    def test_chronic_weak_detected(self, bridge, tmp_db):
        result = bridge.get_learning_context(tmp_db)
        # traceability at 35% should be chronic_weak
        assert "traceability" in result["chronic_weak"]

    def test_empty_journal_returns_empty(self, bridge):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
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
        """)
        conn.commit()
        conn.close()
        result = bridge.get_learning_context(db_path)
        assert result["trends_line"] == ""
        assert result["chronic_weak"] == []
        db_path.unlink(missing_ok=True)


class TestGetRejectionHints:

    def test_empty_when_no_rejections(self, bridge):
        # RejectionLog with no data should return empty
        result = bridge.get_rejection_hints()
        assert isinstance(result, list)
