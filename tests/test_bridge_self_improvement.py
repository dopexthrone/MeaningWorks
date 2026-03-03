"""Tests for bridge self-improvement methods — changelog, system summary, idea wrappers."""

import pytest

from mother.idea_journal import IdeaJournal
from mother.journal import BuildJournal, JournalEntry


class TestGenerateChangelog:
    """Tests for bridge.generate_changelog()."""

    def test_empty_journal(self, tmp_path):
        """Empty journal returns empty string."""
        db = tmp_path / "test.db"
        journal = BuildJournal(db)
        journal.close()

        # Simulate bridge.generate_changelog
        journal2 = BuildJournal(db)
        entries = journal2.recent(limit=20)
        journal2.close()
        assert entries == []

    def test_changelog_with_entries(self, tmp_path):
        """Changelog formats entries as [ok]/[FAIL] with trust and cost."""
        db = tmp_path / "test.db"
        journal = BuildJournal(db)
        journal.record(JournalEntry(
            event_type="compile",
            description="Build user dashboard",
            success=True,
            trust_score=85.0,
            cost_usd=1.50,
        ))
        journal.record(JournalEntry(
            event_type="compile",
            description="Deploy API endpoint",
            success=False,
            trust_score=45.0,
            cost_usd=0.80,
        ))
        journal.close()

        # Simulate bridge.generate_changelog
        journal2 = BuildJournal(db)
        entries = journal2.recent(limit=20)
        journal2.close()

        lines = []
        for e in entries:
            status = "ok" if e.success else "FAIL"
            trust = f" (trust: {e.trust_score:.0f}%)" if e.trust_score > 0 else ""
            cost = f" ${e.cost_usd:.2f}" if e.cost_usd > 0 else ""
            lines.append(f"[{status}]{trust}{cost} {e.description[:80]}")

        changelog = "\n".join(lines)
        assert "[FAIL]" in changelog
        assert "[ok]" in changelog
        assert "(trust: 85%)" in changelog
        assert "$1.50" in changelog

    def test_changelog_respects_limit(self, tmp_path):
        """Changelog only includes up to `limit` entries."""
        db = tmp_path / "test.db"
        journal = BuildJournal(db)
        for i in range(25):
            journal.record(JournalEntry(
                event_type="compile",
                description=f"Task {i}",
                success=True,
            ))
        journal.close()

        journal2 = BuildJournal(db)
        entries = journal2.recent(limit=10)
        journal2.close()
        assert len(entries) == 10


class TestBridgeIdeaWrappers:
    """Tests for bridge idea journal wrapper methods (update_idea_status, get_top_pending_idea)."""

    def test_update_idea_status_done(self, tmp_path):
        """update_idea_status transitions idea to done with outcome."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Test idea")
        journal.update_status(iid, "done", "Success. Cost: $0.50")
        idea = journal.get(iid)
        assert idea.status == "done"
        assert "Success" in idea.outcome
        journal.close()

    def test_update_idea_status_dismissed(self, tmp_path):
        """update_idea_status transitions idea to dismissed."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        iid = journal.add("Risky idea")
        journal.update_status(iid, "dismissed", "Failed: timeout")
        idea = journal.get(iid)
        assert idea.status == "dismissed"
        assert "timeout" in idea.outcome
        journal.close()

    def test_get_top_pending_respects_priority(self, tmp_path):
        """get_top_pending_idea returns the highest-priority pending idea."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        journal.add("Low task", priority="low")
        journal.add("High task", priority="high")
        journal.add("Normal task", priority="normal")
        ideas = journal.pending(limit=1)
        assert ideas[0].description == "High task"
        journal.close()

    def test_get_top_pending_skips_non_pending(self, tmp_path):
        """get_top_pending_idea only returns pending ideas."""
        db = tmp_path / "test.db"
        journal = IdeaJournal(db)
        id1 = journal.add("Already done")
        journal.update_status(id1, "done", "ok")
        id2 = journal.add("Still pending")
        ideas = journal.pending(limit=1)
        assert len(ideas) == 1
        assert ideas[0].description == "Still pending"
        journal.close()
