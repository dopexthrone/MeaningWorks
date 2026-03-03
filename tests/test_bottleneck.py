"""Tests for bottleneck-identifying — chronic_weak surfaced after failure."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mother.bridge import EngineBridge


class TestBottleneckSurfacing:

    def test_chronic_weak_detected_in_learning_context(self):
        """chronic_weak list is populated when dimension averages are below threshold."""
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
                "completeness": 75.0,
                "traceability": 30.0,
                "specificity": 40.0,
            })
            conn.execute(
                """INSERT INTO build_journal
                   (timestamp, event_type, description, success, trust_score,
                    component_count, cost_usd, domain, error_summary,
                    duration_seconds, project_path, weakest_dimension, dimension_scores)
                   VALUES (?, 'compile', ?, 0, 35.0, 3, 0.1, 'software', '',
                           0.0, '', '', ?)""",
                (1000.0 + i, f"test {i}", scores),
            )
        conn.commit()
        conn.close()

        result = bridge.get_learning_context(db_path)
        assert "traceability" in result["chronic_weak"]
        assert "specificity" in result["chronic_weak"]
        db_path.unlink(missing_ok=True)
