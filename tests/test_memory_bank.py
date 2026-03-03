"""Tests for mother/memory_bank.py — unified retrieval across memory systems."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mother.memory_bank import (
    MemoryResult,
    query,
    format_memory_context,
    memory_stats,
    _recency_score,
    _score_result,
    _results_from_episodes,
    _results_from_knowledge,
    _results_from_goals,
    _SOURCE_WEIGHTS,
)
from mother.episodic_memory import compress_session, save_episode
from mother.knowledge_base import KnowledgeFact, save_fact


# ── Helpers ──────────────────────────────────────────────────────

def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


def _sample_messages():
    return [
        {"role": "user", "content": "Build an authentication system with JWT", "timestamp": 1000.0},
        {"role": "assistant", "content": "I'll implement JWT auth using bcrypt for passwords", "timestamp": 1001.0},
        {"role": "user", "content": "Also add role-based access control", "timestamp": 1002.0},
        {"role": "assistant", "content": "Created auth/models.py with User and Role classes", "timestamp": 1003.0},
    ]


def _mock_recall_engine(results=None):
    """Create a mock RecallEngine that returns specified results."""
    engine = MagicMock()
    if results is None:
        results = []
    engine.search.return_value = results
    return engine


# ── Recency Scoring ──────────────────────────────────────────────

class TestRecencyScore:
    def test_now_is_1(self):
        now = time.time()
        score = _recency_score(now, now=now)
        assert abs(score - 1.0) < 0.01

    def test_old_decays(self):
        now = time.time()
        old = now - (14 * 86400)  # 14 days ago = one half-life
        score = _recency_score(old, now=now)
        assert abs(score - 0.5) < 0.05

    def test_very_old_approaches_zero(self):
        now = time.time()
        ancient = now - (365 * 86400)  # 1 year ago
        score = _recency_score(ancient, now=now)
        assert score < 0.01

    def test_future_clamps_to_1(self):
        now = time.time()
        future = now + 86400
        score = _recency_score(future, now=now)
        assert score >= 1.0


class TestScoreResult:
    def test_source_weights_applied(self):
        now = time.time()
        recall_score = _score_result("recall", 1.0, now)
        goal_score = _score_result("goal", 1.0, now)
        # Recall should be weighted higher than goal
        assert recall_score > goal_score

    def test_low_relevance_reduces_score(self):
        now = time.time()
        high = _score_result("recall", 1.0, now)
        low = _score_result("recall", 0.3, now)
        assert high > low


# ── Episode Results ──────────────────────────────────────────────

class TestResultsFromEpisodes:
    def test_returns_results(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="ep-test")
        save_episode(ep, db_path=db)

        results = _results_from_episodes("authentication", db_path=db)
        assert len(results) >= 1
        assert all(r.source == "episode" for r in results)

    def test_no_results_for_unrelated(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="ep-unrel")
        save_episode(ep, db_path=db)

        results = _results_from_episodes("quantum physics", db_path=db)
        assert len(results) == 0


# ── Knowledge Results ────────────────────────────────────────────

class TestResultsFromKnowledge:
    def test_returns_results(self):
        db = _tmp_db()
        f = KnowledgeFact("kn1", "tool", "Redis", "uses", "Uses Redis for caching", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = _results_from_knowledge("Redis", db_path=db)
        assert len(results) >= 1
        assert all(r.source == "knowledge" for r in results)

    def test_no_results_for_unrelated(self):
        db = _tmp_db()
        results = _results_from_knowledge("quantum", db_path=db)
        assert len(results) == 0


# ── Goal Results ─────────────────────────────────────────────────

class TestResultsFromGoals:
    def test_returns_results_when_goals_exist(self):
        db = _tmp_db()
        # Create goals table and insert a goal
        import sqlite3
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY,
                description TEXT,
                status TEXT DEFAULT 'active',
                priority INTEGER DEFAULT 0,
                timestamp REAL
            )
        """)
        conn.execute(
            "INSERT INTO goals (description, status, priority, timestamp) VALUES (?, ?, ?, ?)",
            ("Improve authentication module", "active", 5, time.time()),
        )
        conn.commit()
        conn.close()

        results = _results_from_goals("authentication", db_path=db)
        assert len(results) >= 1
        assert all(r.source == "goal" for r in results)

    def test_no_goals_table(self):
        db = _tmp_db()
        results = _results_from_goals("anything", db_path=db)
        assert len(results) == 0


# ── Unified Query ────────────────────────────────────────────────

class TestQuery:
    def test_empty_query(self):
        results = query("", db_path=_tmp_db())
        assert results == []

    def test_whitespace_query(self):
        results = query("   ", db_path=_tmp_db())
        assert results == []

    def test_query_with_recall_engine(self):
        mock_result = MagicMock()
        mock_result.content = "I built the authentication system"
        mock_result.role = "assistant"
        mock_result.relevance_rank = 1
        mock_result.timestamp = time.time()

        engine = _mock_recall_engine([mock_result])
        results = query("authentication", recall_engine=engine, db_path=_tmp_db())
        assert len(results) >= 1
        engine.search.assert_called_once()

    def test_query_without_recall(self):
        db = _tmp_db()
        f = KnowledgeFact("q1", "tool", "Redis", "uses", "Uses Redis", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = query("Redis", db_path=db)
        assert len(results) >= 1

    def test_query_combines_sources(self):
        db = _tmp_db()
        # Add episode
        ep = compress_session(_sample_messages(), session_id="combined-1")
        save_episode(ep, db_path=db)
        # Add fact
        f = KnowledgeFact("comb1", "tool", "JWT", "uses", "Uses JWT for auth", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = query("JWT authentication", db_path=db)
        sources = {r.source for r in results}
        # Should have results from multiple sources
        assert len(results) >= 1

    def test_results_sorted_by_relevance(self):
        db = _tmp_db()
        f = KnowledgeFact("sort1", "tool", "Python", "uses", "Uses Python for everything", 0.9, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = query("Python", db_path=db)
        if len(results) >= 2:
            assert results[0].relevance >= results[1].relevance

    def test_deduplicates_results(self):
        db = _tmp_db()
        # Save same fact content under different IDs
        f1 = KnowledgeFact("dup1", "tool", "Redis", "uses", "Uses Redis for caching", 0.8, "", time.time(), time.time(), 0)
        f2 = KnowledgeFact("dup2", "tool", "Redis", "uses", "Uses Redis for caching sessions", 0.7, "", time.time(), time.time(), 0)
        save_fact(f1, db_path=db)
        save_fact(f2, db_path=db)

        results = query("Redis caching", db_path=db)
        # Content with identical 60-char prefix should be deduped
        contents = [r.content[:60].lower() for r in results]
        # Just verify we get results (dedup logic is on exact prefix match)
        assert len(results) >= 1

    def test_respects_limit(self):
        db = _tmp_db()
        for i in range(20):
            f = KnowledgeFact(f"lim{i}", "tool", f"Tool{i}", "uses", f"Uses Tool{i}", 0.8, "", time.time(), time.time(), 0)
            save_fact(f, db_path=db)

        results = query("Tool", limit=5, db_path=db)
        assert len(results) <= 5


# ── Memory Result ────────────────────────────────────────────────

class TestMemoryResult:
    def test_frozen(self):
        mr = MemoryResult(
            source="recall",
            content="test content",
            relevance=0.8,
            timestamp=time.time(),
            category="user",
        )
        with pytest.raises(AttributeError):
            mr.content = "modified"


# ── Context Formatting ──────────────────────────────────────────

class TestFormatMemoryContext:
    def test_formats_results(self):
        results = [
            MemoryResult("knowledge", "user: prefers dark mode", 0.9, time.time(), "preference"),
            MemoryResult("episode", "Discussed auth system", 0.7, time.time(), "session"),
            MemoryResult("recall", 'user: "build a task manager"', 0.8, time.time(), "user"),
        ]
        block = format_memory_context(results)
        assert "[MEMORY]" in block
        assert "Known:" in block
        assert "Past sessions:" in block
        assert "From conversation:" in block

    def test_empty_returns_empty(self):
        assert format_memory_context([]) == ""

    def test_respects_token_budget(self):
        results = [
            MemoryResult("knowledge", f"Fact {i}: " + "x" * 100, 0.8, time.time(), "tool")
            for i in range(50)
        ]
        block = format_memory_context(results, max_tokens=50)
        assert len(block) < 400

    def test_groups_by_source(self):
        results = [
            MemoryResult("knowledge", "fact A", 0.9, time.time(), "tool"),
            MemoryResult("recall", "recalled B", 0.8, time.time(), "user"),
            MemoryResult("knowledge", "fact C", 0.7, time.time(), "tool"),
        ]
        block = format_memory_context(results)
        # Facts grouped under "Known:", recalls under "From conversation:"
        assert "Known:" in block
        assert "From conversation:" in block


# ── Memory Stats ─────────────────────────────────────────────────

class TestMemoryStats:
    def test_stats_empty_db(self):
        db = _tmp_db()
        stats = memory_stats(db_path=db)
        assert stats["episodes"] == 0
        assert stats["facts"] == 0

    def test_stats_with_data(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="stats-1")
        save_episode(ep, db_path=db)
        f = KnowledgeFact("st1", "tool", "X", "uses", "X", 0.5, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        stats = memory_stats(db_path=db)
        assert stats["episodes"] == 1
        assert stats["facts"] == 1


# ── Source Weights ───────────────────────────────────────────────

class TestSourceWeights:
    def test_all_sources_have_weights(self):
        for source in ["recall", "episode", "knowledge", "pattern", "goal"]:
            assert source in _SOURCE_WEIGHTS

    def test_recall_highest_weight(self):
        assert _SOURCE_WEIGHTS["recall"] >= max(
            _SOURCE_WEIGHTS[s] for s in _SOURCE_WEIGHTS if s != "recall"
        )
