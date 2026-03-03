"""Tests for mother/idea_journal.py — persistent idea store."""

import tempfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mother.idea_journal import Idea, IdeaJournal


@pytest.fixture
def journal(tmp_path):
    """Create an IdeaJournal with a temporary database."""
    db = tmp_path / "test_ideas.db"
    j = IdeaJournal(db)
    yield j
    j.close()


# --- Idea dataclass ---

class TestIdea:
    def test_frozen(self):
        idea = Idea(description="test")
        with pytest.raises(FrozenInstanceError):
            idea.status = "done"

    def test_defaults(self):
        idea = Idea()
        assert idea.idea_id == 0
        assert idea.timestamp == 0.0
        assert idea.description == ""
        assert idea.source_context == ""
        assert idea.priority == "normal"
        assert idea.status == "pending"
        assert idea.outcome == ""

    def test_values(self):
        idea = Idea(
            idea_id=1,
            description="Add dark mode",
            source_context="user mentioned wanting dark UI",
            priority="high",
            status="pending",
        )
        assert idea.idea_id == 1
        assert idea.description == "Add dark mode"
        assert idea.priority == "high"


# --- IdeaJournal ---

class TestIdeaJournal:
    def test_add_returns_incrementing_ids(self, journal):
        id1 = journal.add("First idea")
        id2 = journal.add("Second idea")
        id3 = journal.add("Third idea")
        assert id1 < id2 < id3

    def test_add_stores_source_context(self, journal):
        idea_id = journal.add(
            "Better error messages",
            source_context="user was confused by error output",
        )
        idea = journal.get(idea_id)
        assert idea is not None
        assert idea.source_context == "user was confused by error output"

    def test_add_stores_priority(self, journal):
        id_high = journal.add("Urgent fix", priority="high")
        id_low = journal.add("Nice to have", priority="low")
        assert journal.get(id_high).priority == "high"
        assert journal.get(id_low).priority == "low"

    def test_pending_filters_by_status(self, journal):
        id1 = journal.add("Pending idea")
        id2 = journal.add("Done idea")
        journal.update_status(id2, "done", "Completed successfully")

        pending = journal.pending()
        assert len(pending) == 1
        assert pending[0].idea_id == id1

    def test_pending_orders_by_priority(self, journal):
        journal.add("Low priority", priority="low")
        journal.add("High priority", priority="high")
        journal.add("Normal priority", priority="normal")

        pending = journal.pending()
        assert len(pending) == 3
        assert pending[0].priority == "high"
        assert pending[1].priority == "normal"
        assert pending[2].priority == "low"

    def test_pending_orders_by_recency_within_priority(self, journal):
        id1 = journal.add("First normal")
        id2 = journal.add("Second normal")

        pending = journal.pending()
        # Most recent first within same priority
        assert pending[0].idea_id == id2
        assert pending[1].idea_id == id1

    def test_pending_respects_limit(self, journal):
        for i in range(10):
            journal.add(f"Idea {i}")

        pending = journal.pending(limit=3)
        assert len(pending) == 3

    def test_update_status(self, journal):
        idea_id = journal.add("Test idea")
        journal.update_status(idea_id, "in_progress")
        idea = journal.get(idea_id)
        assert idea.status == "in_progress"
        assert idea.outcome == ""

    def test_update_status_with_outcome(self, journal):
        idea_id = journal.add("Test idea")
        journal.update_status(idea_id, "done", "Implemented in v1.2")
        idea = journal.get(idea_id)
        assert idea.status == "done"
        assert idea.outcome == "Implemented in v1.2"

    def test_count_pending(self, journal):
        assert journal.count_pending() == 0
        journal.add("Idea 1")
        journal.add("Idea 2")
        assert journal.count_pending() == 2
        journal.update_status(1, "done")
        assert journal.count_pending() == 1

    def test_get_existing(self, journal):
        idea_id = journal.add("Find me", source_context="test context")
        idea = journal.get(idea_id)
        assert idea is not None
        assert idea.description == "Find me"
        assert idea.source_context == "test context"
        assert idea.status == "pending"
        assert idea.timestamp > 0

    def test_get_nonexistent(self, journal):
        assert journal.get(99999) is None

    def test_all_ideas(self, journal):
        journal.add("A")
        journal.add("B")
        journal.add("C")

        all_ideas = journal.all_ideas()
        assert len(all_ideas) == 3
        # Newest first
        assert all_ideas[0].description == "C"
        assert all_ideas[2].description == "A"

    def test_all_ideas_limit(self, journal):
        for i in range(10):
            journal.add(f"Idea {i}")

        all_ideas = journal.all_ideas(limit=5)
        assert len(all_ideas) == 5

    def test_dismissed_status(self, journal):
        idea_id = journal.add("Bad idea")
        journal.update_status(idea_id, "dismissed", "Not feasible")
        idea = journal.get(idea_id)
        assert idea.status == "dismissed"
        assert idea.outcome == "Not feasible"
        # Should not appear in pending
        assert journal.count_pending() == 0

    def test_persistence(self, tmp_path):
        """Ideas persist across IdeaJournal instances."""
        db = tmp_path / "persist_test.db"

        j1 = IdeaJournal(db)
        j1.add("Persistent idea")
        j1.close()

        j2 = IdeaJournal(db)
        ideas = j2.all_ideas()
        assert len(ideas) == 1
        assert ideas[0].description == "Persistent idea"
        j2.close()
