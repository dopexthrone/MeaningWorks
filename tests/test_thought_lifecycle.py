"""Tests for thought lifecycle — persist, surface, mark, consolidation."""

import time
from pathlib import Path

import pytest

from mother.thought_journal import ThoughtJournal, ThoughtRecord


class TestThoughtPersistence:

    def test_record_and_retrieve(self, tmp_path):
        """Thoughts are persisted and retrievable."""
        db = tmp_path / "test.db"
        journal = ThoughtJournal(db)
        rec = ThoughtRecord(
            thought_type="curiosity",
            disposition="surface",
            subject="Why does the user keep refactoring auth?",
            trigger="idle",
            depth=0.4,
            mode="idle",
            session_id="test-session",
        )
        rec_id = journal.record(rec)
        assert rec_id > 0
        recent = journal.recent(limit=5)
        assert len(recent) == 1
        assert recent[0].subject == "Why does the user keep refactoring auth?"
        journal.close()

    def test_surfaceable_returns_unsurfaced_only(self, tmp_path):
        """surfaceable() returns only thoughts not yet marked surfaced."""
        db = tmp_path / "test.db"
        journal = ThoughtJournal(db)

        # Record two surface thoughts
        id1 = journal.record(ThoughtRecord(
            thought_type="connection",
            disposition="surface",
            subject="Auth and logging share error patterns",
            trigger="idle",
            depth=0.5,
            mode="idle",
        ))
        id2 = journal.record(ThoughtRecord(
            thought_type="question",
            disposition="surface",
            subject="Is the deployment pipeline tested?",
            trigger="deep",
            depth=0.7,
            mode="deep",
        ))

        assert len(journal.surfaceable()) == 2

        # Mark one as surfaced
        journal.mark_surfaced(id1)
        remaining = journal.surfaceable()
        assert len(remaining) == 1
        assert remaining[0].subject == "Is the deployment pipeline tested?"
        journal.close()


class TestMarkSurfaced:

    def test_mark_surfaced_prevents_reappearance(self, tmp_path):
        """Once surfaced, a thought should not appear in surfaceable() again."""
        db = tmp_path / "test.db"
        journal = ThoughtJournal(db)
        rec_id = journal.record(ThoughtRecord(
            thought_type="implication",
            disposition="surface",
            subject="Cost tracking should feed into velocity",
            trigger="parallel",
            depth=0.3,
            mode="parallel",
        ))
        assert len(journal.surfaceable()) == 1
        journal.mark_surfaced(rec_id)
        assert len(journal.surfaceable()) == 0
        journal.close()


class TestConsolidationSubjects:

    def test_recurring_subjects_detected(self, tmp_path):
        """Subjects appearing >= min_count times are candidates for consolidation."""
        db = tmp_path / "test.db"
        journal = ThoughtJournal(db)

        # Record same subject 3 times
        for _ in range(3):
            journal.record(ThoughtRecord(
                thought_type="pattern",
                disposition="journal",
                subject="Auth error handling keeps failing",
                trigger="sleep",
                depth=0.6,
                mode="sleep",
            ))

        # Record a different subject once
        journal.record(ThoughtRecord(
            thought_type="curiosity",
            disposition="internal",
            subject="API rate limits are interesting",
            trigger="idle",
            depth=0.3,
            mode="idle",
        ))

        subjects = journal.subjects_for_consolidation(min_count=2)
        assert "Auth error handling keeps failing" in subjects
        assert "API rate limits are interesting" not in subjects
        journal.close()

    def test_no_consolidation_on_unique_subjects(self, tmp_path):
        """Unique subjects are not candidates for consolidation."""
        db = tmp_path / "test.db"
        journal = ThoughtJournal(db)

        journal.record(ThoughtRecord(
            thought_type="curiosity",
            disposition="internal",
            subject="Topic A",
            trigger="idle",
            depth=0.3,
            mode="idle",
        ))
        journal.record(ThoughtRecord(
            thought_type="curiosity",
            disposition="internal",
            subject="Topic B",
            trigger="idle",
            depth=0.3,
            mode="idle",
        ))

        subjects = journal.subjects_for_consolidation(min_count=2)
        assert len(subjects) == 0
        journal.close()
