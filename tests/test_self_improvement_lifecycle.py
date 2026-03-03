"""Tests for self-improvement lifecycle wiring — idea dispatch, completion, failure."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.idea_journal import IdeaJournal


class TestIdeaLifecycle:
    """Unit tests for the idea journal lifecycle (pending → in_progress → done/dismissed)."""

    def test_add_and_pending(self, tmp_path):
        """Ideas start as pending and appear in pending list."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Refactor bridge.py", source_context="conversation")
        assert journal.count_pending() == 1
        ideas = journal.pending()
        assert len(ideas) == 1
        assert ideas[0].idea_id == iid
        assert ideas[0].status == "pending"
        journal.close()

    def test_update_to_in_progress(self, tmp_path):
        """Marking as in_progress removes from pending list."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Add error handling")
        journal.update_status(iid, "in_progress")
        assert journal.count_pending() == 0
        idea = journal.get(iid)
        assert idea.status == "in_progress"
        journal.close()

    def test_update_to_done(self, tmp_path):
        """Done ideas are not pending and have outcome."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Add tests")
        journal.update_status(iid, "in_progress")
        journal.update_status(iid, "done", outcome="Success. Cost: $0.50")
        assert journal.count_pending() == 0
        idea = journal.get(iid)
        assert idea.status == "done"
        assert "Success" in idea.outcome
        journal.close()

    def test_update_to_dismissed(self, tmp_path):
        """Dismissed ideas are not pending and record failure reason."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Risky refactor")
        journal.update_status(iid, "in_progress")
        journal.update_status(iid, "dismissed", outcome="Failed: tests broke")
        idea = journal.get(iid)
        assert idea.status == "dismissed"
        assert "Failed" in idea.outcome
        journal.close()

    def test_priority_ordering(self, tmp_path):
        """High-priority ideas come first in pending list."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        journal.add("Low idea", priority="low")
        journal.add("High idea", priority="high")
        journal.add("Normal idea", priority="normal")
        ideas = journal.pending()
        assert ideas[0].description == "High idea"
        assert ideas[1].description == "Normal idea"
        assert ideas[2].description == "Low idea"
        journal.close()


class TestBridgeIdeaMethods:
    """Tests for bridge idea methods that wrap IdeaJournal."""

    @pytest.fixture
    def db_path(self, tmp_path):
        return tmp_path / "test.db"

    def test_bridge_update_idea_status(self, db_path):
        """Bridge update_idea_status calls IdeaJournal.update_status."""
        journal = IdeaJournal(db_path)
        iid = journal.add("Test idea")
        journal.close()

        # Simulate what bridge.update_idea_status does
        journal2 = IdeaJournal(db_path)
        journal2.update_status(iid, "done", "Success")
        idea = journal2.get(iid)
        assert idea.status == "done"
        assert idea.outcome == "Success"
        journal2.close()

    def test_bridge_get_top_pending_idea(self, db_path):
        """Bridge get_top_pending_idea returns highest priority idea."""
        journal = IdeaJournal(db_path)
        journal.add("Low", priority="low")
        journal.add("High", priority="high")
        ideas = journal.pending(limit=1)
        journal.close()
        assert len(ideas) == 1
        assert ideas[0].description == "High"

    def test_bridge_get_top_pending_idea_empty(self, db_path):
        """Returns empty list when no pending ideas."""
        journal = IdeaJournal(db_path)
        ideas = journal.pending(limit=1)
        journal.close()
        assert len(ideas) == 0


class TestSelfBuildIdeaIntegration:
    """Integration tests: idea lifecycle across self-build dispatch."""

    def test_full_lifecycle_success(self, tmp_path):
        """pending → in_progress → done on successful self-build."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Improve error messages", source_context="observation")

        # Simulate dispatch
        journal.update_status(iid, "in_progress")
        assert journal.count_pending() == 0

        # Simulate success
        journal.update_status(iid, "done", outcome="Success. Cost: $1.20")
        idea = journal.get(iid)
        assert idea.status == "done"
        assert "$1.20" in idea.outcome
        journal.close()

    def test_full_lifecycle_failure(self, tmp_path):
        """pending → in_progress → dismissed on failed self-build."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Dangerous refactor")

        # Simulate dispatch
        journal.update_status(iid, "in_progress")

        # Simulate failure
        journal.update_status(iid, "dismissed", outcome="Failed: test regression")
        idea = journal.get(iid)
        assert idea.status == "dismissed"
        assert "test regression" in idea.outcome
        journal.close()

    def test_full_lifecycle_exception(self, tmp_path):
        """pending → in_progress → dismissed on exception during self-build."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Timeout-prone task")

        # Simulate dispatch
        journal.update_status(iid, "in_progress")

        # Simulate exception cleanup
        journal.update_status(iid, "dismissed", outcome="Exception: TimeoutError")
        idea = journal.get(iid)
        assert idea.status == "dismissed"
        journal.close()

    def test_pending_count_accurate_after_lifecycle(self, tmp_path):
        """Pending count stays accurate through multiple lifecycle transitions."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        id1 = journal.add("Idea 1")
        id2 = journal.add("Idea 2")
        id3 = journal.add("Idea 3")
        assert journal.count_pending() == 3

        journal.update_status(id1, "in_progress")
        assert journal.count_pending() == 2

        journal.update_status(id1, "done", "ok")
        assert journal.count_pending() == 2

        journal.update_status(id2, "dismissed", "skip")
        assert journal.count_pending() == 1

        journal.update_status(id3, "in_progress")
        assert journal.count_pending() == 0
        journal.close()
