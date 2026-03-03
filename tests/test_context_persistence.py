"""Tests for CONTEXT mode memory persistence."""

import os
import time
import pytest
import sqlite3
import tempfile
from pathlib import Path

from mother.memory_indexer import index_context_map
from mother.knowledge_base import KnowledgeFact


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_db(tmp_path):
    """Temporary database path for testing."""
    return tmp_path / "test_history.db"


SAMPLE_CONTEXT_MAP = {
    "original_intent": "Build a task management system",
    "concepts": [
        {
            "name": "TaskService",
            "description": "Manages task lifecycle",
            "layer": "ORG",
            "concern": "ENT",
            "confidence": 0.9,
            "source_postcode": "ORG.ENT.GLB.STR.SFT",
        },
        {
            "name": "Authentication",
            "description": "User login and session management",
            "layer": "ORG",
            "concern": "BHV",
            "confidence": 0.85,
            "source_postcode": "ORG.BHV.GLB.STR.SFT",
        },
    ],
    "relationships": [
        {
            "source": "TaskService",
            "target": "Authentication",
            "relation_type": "connected",
            "strength": 0.7,
        },
    ],
    "assumptions": [
        {
            "text": "All users must have an account to create tasks",
            "category": "constraint",
            "confidence": 0.6,
        },
    ],
    "unknowns": [
        {
            "question": "What about guest access?",
            "category": "unspecified_constraint",
            "priority": 0.7,
        },
    ],
    "vocabulary": ["TaskService", "Authentication", "Sprint", "Backlog"],
    "memory_connections": [],
    "confidence": 0.875,
}


# ============================================================
# index_context_map function
# ============================================================

class TestIndexContextMap:
    """index_context_map() bulk-indexes context to knowledge facts."""

    def test_returns_stats(self, tmp_db):
        stats = index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        assert isinstance(stats, dict)
        assert "concepts_indexed" in stats
        assert "vocabulary_indexed" in stats
        assert "facts_saved" in stats

    def test_indexes_concepts(self, tmp_db):
        stats = index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        assert stats["concepts_indexed"] == 2

    def test_indexes_vocabulary(self, tmp_db):
        stats = index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        assert stats["vocabulary_indexed"] == 4

    def test_indexes_assumptions(self, tmp_db):
        stats = index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        assert stats["assumptions_indexed"] == 1

    def test_persists_to_database(self, tmp_db):
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        assert tmp_db.exists()
        conn = sqlite3.connect(str(tmp_db))
        try:
            count = conn.execute("SELECT COUNT(*) FROM knowledge_facts").fetchone()[0]
            assert count > 0
        finally:
            conn.close()

    def test_concept_facts_have_correct_category(self, tmp_db):
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            rows = conn.execute(
                "SELECT category, subject FROM knowledge_facts WHERE fact_id LIKE 'ctx:%'"
            ).fetchall()
            assert len(rows) == 2
            for cat, _ in rows:
                assert cat == "decision"
        finally:
            conn.close()

    def test_vocabulary_facts_have_correct_category(self, tmp_db):
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            rows = conn.execute(
                "SELECT category FROM knowledge_facts WHERE fact_id LIKE 'vocab:%'"
            ).fetchall()
            assert len(rows) == 4
            for (cat,) in rows:
                assert cat == "tool"
        finally:
            conn.close()

    def test_assumption_facts_have_correct_category(self, tmp_db):
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        conn = sqlite3.connect(str(tmp_db))
        try:
            rows = conn.execute(
                "SELECT category FROM knowledge_facts WHERE fact_id LIKE 'assumption:%'"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "constraint"
        finally:
            conn.close()


# ============================================================
# Edge cases
# ============================================================

class TestIndexContextMapEdgeCases:
    """Edge case handling for index_context_map."""

    def test_empty_context_map(self, tmp_db):
        empty = {
            "concepts": [],
            "vocabulary": [],
            "assumptions": [],
        }
        stats = index_context_map(empty, "test", db_path=tmp_db)
        assert stats["concepts_indexed"] == 0
        assert stats["vocabulary_indexed"] == 0
        assert stats["facts_saved"] == 0

    def test_missing_concepts_key(self, tmp_db):
        minimal = {"vocabulary": ["foo"]}
        stats = index_context_map(minimal, "test", db_path=tmp_db)
        assert stats["concepts_indexed"] == 0
        assert stats["vocabulary_indexed"] == 1

    def test_concept_without_description(self, tmp_db):
        ctx = {
            "concepts": [{"name": "Foo", "confidence": 0.7}],
            "vocabulary": [],
            "assumptions": [],
        }
        stats = index_context_map(ctx, "test intent", db_path=tmp_db)
        assert stats["concepts_indexed"] == 1
        assert stats["facts_saved"] >= 1

    def test_empty_name_skipped(self, tmp_db):
        ctx = {
            "concepts": [{"name": "", "description": "empty"}],
            "vocabulary": [""],
            "assumptions": [],
        }
        stats = index_context_map(ctx, "test", db_path=tmp_db)
        assert stats["facts_saved"] == 0

    def test_long_intent_truncated(self, tmp_db):
        long_intent = "a" * 500
        stats = index_context_map(SAMPLE_CONTEXT_MAP, long_intent, db_path=tmp_db)
        assert stats["facts_saved"] > 0
        # Source should be truncated
        conn = sqlite3.connect(str(tmp_db))
        try:
            sources = conn.execute("SELECT source FROM knowledge_facts LIMIT 1").fetchone()
            assert len(sources[0]) < 500
        finally:
            conn.close()

    def test_assumptions_capped_at_10(self, tmp_db):
        ctx = {
            "concepts": [],
            "vocabulary": [],
            "assumptions": [
                {"text": f"Assumption {i}", "category": "structural", "confidence": 0.5}
                for i in range(15)
            ],
        }
        stats = index_context_map(ctx, "test", db_path=tmp_db)
        assert stats["assumptions_indexed"] == 10

    def test_idempotent_on_reindex(self, tmp_db):
        """Re-indexing same context shouldn't duplicate facts."""
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        stats1_count = _count_facts(tmp_db)
        index_context_map(SAMPLE_CONTEXT_MAP, "test intent", db_path=tmp_db)
        stats2_count = _count_facts(tmp_db)
        # ON CONFLICT DO UPDATE means count stays same
        assert stats1_count == stats2_count


# ============================================================
# Bridge integration (structural)
# ============================================================

class TestBridgeContextPersistence:
    """Bridge compile() calls index_context_map for CONTEXT mode."""

    def test_bridge_compile_has_context_persistence_code(self):
        import inspect
        from mother.bridge import EngineBridge
        source = inspect.getsource(EngineBridge.compile)
        assert "index_context_map" in source
        assert 'mode == "context"' in source


# ============================================================
# Helpers
# ============================================================

def _count_facts(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM knowledge_facts").fetchone()[0]
    finally:
        conn.close()
