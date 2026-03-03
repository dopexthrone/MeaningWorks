"""Tests for mother/episodic_memory.py — session compression into episodes."""

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from mother.episodic_memory import (
    Episode,
    compress_session,
    save_episode,
    load_episodes,
    search_episodes,
    episode_count,
    format_episode_context,
    _extract_topics,
    _extract_decisions,
    _extract_artifacts,
    _extract_questions,
    _build_summary,
)


# ── Helpers ──────────────────────────────────────────────────────

def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


def _sample_messages():
    return [
        {"role": "user", "content": "Let's build a task manager with authentication", "timestamp": 1000.0},
        {"role": "assistant", "content": "I'll implement authentication using JWT tokens and bcrypt for password hashing.", "timestamp": 1001.0},
        {"role": "user", "content": "Should we use PostgreSQL or SQLite?", "timestamp": 1002.0},
        {"role": "assistant", "content": "Let's use PostgreSQL for production. I'll create the schema with users, tasks, and sessions tables.", "timestamp": 1003.0},
        {"role": "user", "content": "Good. Also add role-based access control.", "timestamp": 1004.0},
        {"role": "assistant", "content": "I've created auth/models.py with User, Role, and Permission classes. Also created tasks/api.py with CRUD endpoints.", "timestamp": 1005.0},
    ]


# ── Topic Extraction ────────────────────────────────────────────

class TestExtractTopics:
    def test_extracts_frequent_words(self):
        msgs = [
            {"content": "authentication system with JWT tokens"},
            {"content": "authentication tokens are important"},
            {"content": "implement the token verification"},
        ]
        topics = _extract_topics(msgs)
        assert "authentication" in topics or "tokens" in topics

    def test_filters_stopwords(self):
        msgs = [
            {"content": "the and or but this that"},
            {"content": "the and or but this that"},
        ]
        topics = _extract_topics(msgs)
        assert len(topics) == 0

    def test_caps_at_max(self):
        msgs = [
            {"content": " ".join(f"word{i}" * 3 for i in range(20))},
            {"content": " ".join(f"word{i}" * 3 for i in range(20))},
        ]
        topics = _extract_topics(msgs, max_topics=5)
        assert len(topics) <= 5

    def test_requires_frequency_2(self):
        msgs = [
            {"content": "unique_single_word xyz"},
            {"content": "different content entirely"},
        ]
        topics = _extract_topics(msgs)
        # "unique_single_word" appears only once, should be excluded
        assert "unique_single_word" not in topics


# ── Decision Extraction ─────────────────────────────────────────

class TestExtractDecisions:
    def test_extracts_lets_use(self):
        msgs = [{"content": "Let's use PostgreSQL for the database"}]
        decisions = _extract_decisions(msgs)
        assert len(decisions) >= 1
        assert any("PostgreSQL" in d for d in decisions)

    def test_extracts_decided_to(self):
        msgs = [{"content": "We decided to implement caching with Redis"}]
        decisions = _extract_decisions(msgs)
        assert len(decisions) >= 1

    def test_extracts_going_to(self):
        msgs = [{"content": "Going to build the API endpoints first"}]
        decisions = _extract_decisions(msgs)
        assert len(decisions) >= 1

    def test_caps_at_max(self):
        msgs = [{"content": f"Let's use tool{i} for something" } for i in range(20)]
        decisions = _extract_decisions(msgs, max_decisions=3)
        assert len(decisions) <= 3


# ── Artifact Extraction ─────────────────────────────────────────

class TestExtractArtifacts:
    def test_extracts_python_files(self):
        msgs = [{"content": "I've created auth/models.py and tasks/api.py"}]
        artifacts = _extract_artifacts(msgs)
        assert any("models.py" in a for a in artifacts)

    def test_extracts_inline_code(self):
        msgs = [{"content": "Use `calculate_total()` for pricing"}]
        artifacts = _extract_artifacts(msgs)
        assert len(artifacts) >= 1

    def test_extracts_created_file(self):
        msgs = [{"content": "Created a new module for authentication"}]
        artifacts = _extract_artifacts(msgs)
        assert len(artifacts) >= 1

    def test_caps_at_max(self):
        msgs = [{"content": " ".join(f"file{i}.py" for i in range(20))}]
        artifacts = _extract_artifacts(msgs, max_artifacts=5)
        assert len(artifacts) <= 5


# ── Question Extraction ─────────────────────────────────────────

class TestExtractQuestions:
    def test_extracts_user_questions(self):
        msgs = [
            {"role": "user", "content": "Should we use PostgreSQL or SQLite?"},
        ]
        questions = _extract_questions(msgs)
        assert len(questions) >= 1

    def test_ignores_assistant_questions(self):
        msgs = [
            {"role": "assistant", "content": "What would you like me to build?"},
        ]
        questions = _extract_questions(msgs)
        assert len(questions) == 0

    def test_extracts_how_questions(self):
        msgs = [
            {"role": "user", "content": "How do we handle concurrent requests?"},
        ]
        questions = _extract_questions(msgs)
        assert len(questions) >= 1


# ── Session Compression ─────────────────────────────────────────

class TestCompressSession:
    def test_compress_returns_episode(self):
        ep = compress_session(_sample_messages(), session_id="test-1")
        assert isinstance(ep, Episode)
        assert ep.session_id == "test-1"
        assert ep.message_count == 6
        assert ep.user_turns == 3

    def test_compress_empty(self):
        ep = compress_session([], session_id="empty")
        assert ep.message_count == 0
        assert ep.summary == "Empty session."

    def test_compress_derives_timestamps(self):
        msgs = _sample_messages()
        ep = compress_session(msgs, session_id="ts-test")
        assert ep.start_time == 1000.0
        assert ep.end_time == 1005.0
        assert ep.duration_seconds == 5.0

    def test_compress_explicit_timestamps(self):
        ep = compress_session(
            _sample_messages(),
            session_id="explicit",
            start_time=500.0,
            end_time=2000.0,
        )
        assert ep.start_time == 500.0
        assert ep.end_time == 2000.0

    def test_compress_extracts_topics(self):
        ep = compress_session(_sample_messages(), session_id="topics")
        assert len(ep.topics) > 0

    def test_compress_extracts_decisions(self):
        ep = compress_session(_sample_messages(), session_id="decisions")
        # Messages contain "Let's use PostgreSQL"
        assert len(ep.decisions) > 0

    def test_compress_extracts_artifacts(self):
        ep = compress_session(_sample_messages(), session_id="artifacts")
        # Messages mention .py files
        assert len(ep.artifacts) > 0

    def test_compress_builds_summary(self):
        ep = compress_session(_sample_messages(), session_id="summary")
        assert len(ep.summary) > 10
        assert "task manager" in ep.summary.lower() or "Started with" in ep.summary

    def test_episode_is_frozen(self):
        ep = compress_session(_sample_messages())
        with pytest.raises(AttributeError):
            ep.summary = "modified"


# ── Summary Building ────────────────────────────────────────────

class TestBuildSummary:
    def test_includes_opener(self):
        msgs = [{"role": "user", "content": "Build me a web scraper"}]
        summary = _build_summary(msgs, ["scraper"], [], [])
        assert "scraper" in summary.lower() or "Started with" in summary

    def test_includes_topics(self):
        msgs = [{"role": "user", "content": "hello"}]
        summary = _build_summary(msgs, ["python", "flask", "api"], [], [])
        assert "python" in summary.lower() or "flask" in summary.lower()

    def test_includes_decisions(self):
        msgs = [{"role": "user", "content": "hello"}]
        summary = _build_summary(msgs, [], ["use Redis for caching"], [])
        assert "Redis" in summary

    def test_includes_artifact_count(self):
        msgs = [{"role": "user", "content": "hello"}]
        summary = _build_summary(msgs, [], [], ["a.py", "b.py", "c.py"])
        assert "3" in summary


# ── Persistence ──────────────────────────────────────────────────

class TestEpisodePersistence:
    def test_save_and_load(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="persist-1")
        save_episode(ep, db_path=db)

        loaded = load_episodes(db_path=db)
        assert len(loaded) == 1
        assert loaded[0].session_id == "persist-1"
        assert loaded[0].message_count == 6

    def test_upsert_on_conflict(self):
        db = _tmp_db()
        ep1 = compress_session(_sample_messages(), session_id="upsert-1")
        save_episode(ep1, db_path=db)

        # Save again with same episode_id — should upsert
        msgs2 = _sample_messages() + [
            {"role": "user", "content": "One more thing", "timestamp": 1006.0},
        ]
        ep2 = compress_session(msgs2, session_id="upsert-1")
        save_episode(ep2, db_path=db)

        loaded = load_episodes(db_path=db)
        assert len(loaded) == 1
        assert loaded[0].message_count == 7  # Updated

    def test_load_with_since_filter(self):
        db = _tmp_db()
        ep1 = compress_session(
            _sample_messages(), session_id="old",
            start_time=100.0, end_time=200.0,
        )
        ep2 = compress_session(
            _sample_messages(), session_id="new",
            start_time=1000.0, end_time=2000.0,
        )
        save_episode(ep1, db_path=db)
        save_episode(ep2, db_path=db)

        loaded = load_episodes(since=500.0, db_path=db)
        assert len(loaded) == 1
        assert loaded[0].session_id == "new"

    def test_load_limit(self):
        db = _tmp_db()
        for i in range(5):
            ep = compress_session(
                _sample_messages(), session_id=f"limit-{i}",
                start_time=float(i * 1000), end_time=float(i * 1000 + 100),
            )
            save_episode(ep, db_path=db)

        loaded = load_episodes(limit=3, db_path=db)
        assert len(loaded) == 3

    def test_episode_count(self):
        db = _tmp_db()
        assert episode_count(db_path=db) == 0

        ep = compress_session(_sample_messages(), session_id="count-1")
        save_episode(ep, db_path=db)
        assert episode_count(db_path=db) == 1


# ── Search ───────────────────────────────────────────────────────

class TestEpisodeSearch:
    def test_search_by_topic(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="search-1")
        save_episode(ep, db_path=db)

        results = search_episodes("authentication", db_path=db)
        assert len(results) >= 1

    def test_search_by_decision(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="search-2")
        save_episode(ep, db_path=db)

        results = search_episodes("PostgreSQL", db_path=db)
        assert len(results) >= 1

    def test_search_no_results(self):
        db = _tmp_db()
        ep = compress_session(_sample_messages(), session_id="search-3")
        save_episode(ep, db_path=db)

        results = search_episodes("quantum computing blockchain", db_path=db)
        assert len(results) == 0

    def test_search_missing_db(self):
        results = search_episodes("anything", db_path=Path("/tmp/nonexistent.db"))
        assert results == []


# ── Context Formatting ───────────────────────────────────────────

class TestFormatEpisodeContext:
    def test_formats_episodes(self):
        ep = compress_session(
            _sample_messages(), session_id="fmt-1",
            start_time=time.time() - 3600, end_time=time.time(),
        )
        block = format_episode_context([ep])
        assert "[EPISODIC MEMORY]" in block
        assert len(block) > 20

    def test_empty_returns_empty(self):
        assert format_episode_context([]) == ""

    def test_respects_token_budget(self):
        episodes = []
        for i in range(20):
            ep = compress_session(
                _sample_messages(), session_id=f"budget-{i}",
                start_time=time.time() - i * 3600,
                end_time=time.time() - i * 3600 + 100,
            )
            episodes.append(ep)

        block = format_episode_context(episodes, max_tokens=100)
        # 100 tokens ≈ 400 chars — should truncate
        assert len(block) < 600

    def test_includes_topics(self):
        ep = compress_session(
            _sample_messages(), session_id="topics-fmt",
            start_time=time.time(), end_time=time.time(),
        )
        if ep.topics:
            block = format_episode_context([ep])
            # At least one topic should appear
            assert any(t in block.lower() for t in ep.topics)


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_message_session(self):
        msgs = [{"role": "user", "content": "hello", "timestamp": 1000.0}]
        ep = compress_session(msgs, session_id="single")
        assert ep.message_count == 1
        assert ep.user_turns == 1

    def test_only_assistant_messages(self):
        msgs = [
            {"role": "assistant", "content": "I can help with that.", "timestamp": 1000.0},
            {"role": "assistant", "content": "Here's the solution.", "timestamp": 1001.0},
        ]
        ep = compress_session(msgs, session_id="asst-only")
        assert ep.user_turns == 0

    def test_very_long_message(self):
        msgs = [
            {"role": "user", "content": "x" * 10000, "timestamp": 1000.0},
            {"role": "assistant", "content": "y" * 10000, "timestamp": 1001.0},
        ]
        ep = compress_session(msgs, session_id="long")
        assert ep.message_count == 2
        # Summary should be truncated
        assert len(ep.summary) < 500

    def test_unicode_content(self):
        msgs = [
            {"role": "user", "content": "Créer un système d'authentification", "timestamp": 1000.0},
            {"role": "assistant", "content": "日本語のテスト", "timestamp": 1001.0},
        ]
        ep = compress_session(msgs, session_id="unicode")
        assert ep.message_count == 2

    def test_missing_db_path(self):
        # load_episodes with nonexistent path should return empty
        result = load_episodes(db_path=Path("/tmp/doesnt_exist_ep.db"))
        assert result == []
