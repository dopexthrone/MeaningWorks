"""Tests for failure-preventing — failure_line injected into compile context."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mother.bridge import EngineBridge


class TestFailurePrevention:

    def test_failure_line_populated(self):
        """failure_line should be non-empty when there are co-occurring weak dimensions."""
        bridge = EngineBridge(provider="claude")
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
        # Multiple failures with co-occurring weak dimensions
        for i in range(6):
            scores = json.dumps({
                "completeness": 70.0,
                "specificity": 30.0,
                "actionability": 35.0,
                "traceability": 60.0,
            })
            conn.execute(
                """INSERT INTO build_journal
                   (timestamp, event_type, description, success, trust_score,
                    component_count, cost_usd, domain, error_summary,
                    duration_seconds, project_path, weakest_dimension, dimension_scores)
                   VALUES (?, 'compile', ?, 0, 30.0, 3, 0.1, 'software', '',
                           0.0, '', '', ?)""",
                (1000.0 + i, f"test {i}", scores),
            )
        conn.commit()
        conn.close()

        result = bridge.get_learning_context(db_path)
        assert result["failure_line"] != ""
        assert "pattern" in result["failure_line"].lower()
        db_path.unlink(missing_ok=True)

    def test_no_failure_line_on_success(self):
        """No failure_line when everything succeeds."""
        bridge = EngineBridge(provider="claude")
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
        for i in range(5):
            scores = json.dumps({
                "completeness": 80.0,
                "specificity": 75.0,
                "actionability": 85.0,
            })
            conn.execute(
                """INSERT INTO build_journal
                   (timestamp, event_type, description, success, trust_score,
                    component_count, cost_usd, domain, error_summary,
                    duration_seconds, project_path, weakest_dimension, dimension_scores)
                   VALUES (?, 'compile', ?, 1, 80.0, 5, 0.1, 'software', '',
                           0.0, '', '', ?)""",
                (1000.0 + i, f"test {i}", scores),
            )
        conn.commit()
        conn.close()

        result = bridge.get_learning_context(db_path)
        assert result["failure_line"] == ""
        db_path.unlink(missing_ok=True)
