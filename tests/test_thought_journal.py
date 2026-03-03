"""Tests for mother/thought_journal.py — SQLite-backed thought persistence."""

import tempfile
import time
from pathlib import Path

import pytest

from mother.thought_journal import ThoughtJournal, ThoughtRecord


@pytest.fixture
def journal(tmp_path):
    """Create a ThoughtJournal with a temp database."""
    db = tmp_path / "test_thoughts.db"
    j = ThoughtJournal(db)
    yield j
    j.close()


@pytest.fixture
def populated_journal(journal):
    """Journal with some pre-recorded thoughts."""
    now = time.time()
    records = [
        ThoughtRecord(
            timestamp=now - 300,
            thought_type="curiosity",
            disposition="surface",
            subject="auth patterns",
            trigger="recall",
            depth=0.4,
            mode="idle",
            session_id="s1",
        ),
        ThoughtRecord(
            timestamp=now - 200,
            thought_type="connection",
            disposition="surface",
            subject="caching relates to auth",
            trigger="topics",
            depth=0.5,
            mode="parallel",
            session_id="s1",
        ),
        ThoughtRecord(
            timestamp=now - 100,
            thought_type="consolidation",
            disposition="journal",
            subject="auth patterns",
            trigger="sleep",
            depth=0.6,
            mode="sleep",
            session_id="s2",
        ),
        ThoughtRecord(
            timestamp=now,
            thought_type="pattern",
            disposition="internal",
            subject="recurring failures",
            trigger="journal",
            depth=0.3,
            mode="idle",
            session_id="s2",
        ),
    ]
    for r in records:
        journal.record(r)
    return journal


class TestThoughtJournal:
    """Core persistence tests."""

    def test_record_returns_id(self, journal):
        rec = ThoughtRecord(thought_type="curiosity", subject="test")
        rid = journal.record(rec)
        assert rid >= 1

    def test_record_auto_timestamp(self, journal):
        rec = ThoughtRecord(thought_type="curiosity", subject="test")
        journal.record(rec)
        rows = journal.recent(1)
        assert len(rows) == 1
        assert rows[0].timestamp > 0

    def test_recent_order(self, populated_journal):
        rows = populated_journal.recent(limit=4)
        assert len(rows) == 4
        # Newest first
        assert rows[0].subject == "recurring failures"
        assert rows[-1].subject == "auth patterns"

    def test_recent_limit(self, populated_journal):
        rows = populated_journal.recent(limit=2)
        assert len(rows) == 2

    def test_surfaceable(self, populated_journal):
        rows = populated_journal.surfaceable()
        assert len(rows) == 2
        for r in rows:
            assert r.disposition == "surface"
            assert r.surfaced is False

    def test_mark_surfaced(self, populated_journal):
        surfaceable = populated_journal.surfaceable()
        assert len(surfaceable) >= 1
        rid = surfaceable[0].record_id
        populated_journal.mark_surfaced(rid)
        # Should be one fewer surfaceable now
        remaining = populated_journal.surfaceable()
        assert len(remaining) == len(surfaceable) - 1

    def test_subjects_for_consolidation(self, populated_journal):
        subjects = populated_journal.subjects_for_consolidation(min_count=2)
        assert "auth patterns" in subjects

    def test_subjects_min_count_filter(self, populated_journal):
        subjects = populated_journal.subjects_for_consolidation(min_count=3)
        assert "auth patterns" not in subjects  # only appears 2x

    def test_empty_journal(self, journal):
        assert journal.recent() == []
        assert journal.surfaceable() == []
        assert journal.subjects_for_consolidation() == []

    def test_record_frozen(self):
        rec = ThoughtRecord()
        with pytest.raises(AttributeError):
            rec.subject = "nope"

    def test_roundtrip_fields(self, journal):
        rec = ThoughtRecord(
            timestamp=1000.0,
            thought_type="question",
            disposition="surface",
            subject="why trust drops",
            trigger="compile_failure",
            depth=0.8,
            mode="deep",
            session_id="sess-abc",
        )
        journal.record(rec)
        rows = journal.recent(1)
        r = rows[0]
        assert r.thought_type == "question"
        assert r.disposition == "surface"
        assert r.subject == "why trust drops"
        assert r.trigger == "compile_failure"
        assert abs(r.depth - 0.8) < 0.001
        assert r.mode == "deep"
        assert r.session_id == "sess-abc"
        assert r.surfaced is False

    def test_multiple_sessions(self, journal):
        journal.record(ThoughtRecord(subject="a", session_id="s1"))
        journal.record(ThoughtRecord(subject="b", session_id="s2"))
        rows = journal.recent()
        sessions = {r.session_id for r in rows}
        assert sessions == {"s1", "s2"}

    def test_reopen_db(self, tmp_path):
        """Data persists across journal instances."""
        db = tmp_path / "persist.db"
        j1 = ThoughtJournal(db)
        j1.record(ThoughtRecord(subject="persistent", thought_type="pattern"))
        j1.close()

        j2 = ThoughtJournal(db)
        rows = j2.recent()
        j2.close()
        assert len(rows) == 1
        assert rows[0].subject == "persistent"
