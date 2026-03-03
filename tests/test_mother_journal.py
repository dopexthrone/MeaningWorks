"""
Tests for Mother build journal — operational memory of compilations and builds.

Covers: JournalEntry/JournalSummary frozen dataclasses, BuildJournal SQLite
operations, streak computation, search, recent, domain tracking, cost aggregation.
"""

import time

import pytest
from pathlib import Path

from mother.journal import JournalEntry, JournalSummary, BuildJournal


@pytest.fixture
def journal(tmp_path):
    """Create a journal in a temp directory."""
    db = tmp_path / "test_journal.db"
    j = BuildJournal(db)
    yield j
    j.close()


class TestJournalEntryDefaults:

    def test_defaults(self):
        e = JournalEntry()
        assert e.entry_id == 0
        assert e.timestamp == 0.0
        assert e.event_type == ""
        assert e.description == ""
        assert e.success is False
        assert e.trust_score == 0.0
        assert e.component_count == 0
        assert e.cost_usd == 0.0
        assert e.domain == ""
        assert e.error_summary == ""
        assert e.duration_seconds == 0.0
        assert e.project_path == ""
        assert e.weakest_dimension == ""

    def test_frozen(self):
        e = JournalEntry()
        with pytest.raises(AttributeError):
            e.success = True


class TestJournalSummaryDefaults:

    def test_defaults(self):
        s = JournalSummary()
        assert s.total_compiles == 0
        assert s.total_builds == 0
        assert s.success_rate == 0.0
        assert s.avg_trust == 0.0
        assert s.total_cost == 0.0
        assert s.domains == {}
        assert s.last_entry is None
        assert s.streak == 0

    def test_frozen(self):
        s = JournalSummary()
        with pytest.raises(AttributeError):
            s.streak = 5


class TestBuildJournalTable:

    def test_creates_table(self, journal):
        # Should be able to query without error
        result = journal.recent(limit=1)
        assert result == []


class TestBuildJournalRecord:

    def test_record_and_retrieve(self, journal):
        entry = JournalEntry(
            event_type="compile",
            description="A chat agent",
            success=True,
            trust_score=72.5,
            component_count=6,
            cost_usd=0.15,
            domain="software",
        )
        entry_id = journal.record(entry)
        assert entry_id > 0

        recent = journal.recent(limit=1)
        assert len(recent) == 1
        assert recent[0].entry_id == entry_id
        assert recent[0].event_type == "compile"
        assert recent[0].description == "A chat agent"
        assert recent[0].success is True
        assert recent[0].trust_score == 72.5
        assert recent[0].component_count == 6

    def test_compile_success(self, journal):
        entry = JournalEntry(
            event_type="compile",
            success=True,
            trust_score=85.0,
        )
        journal.record(entry)
        s = journal.get_summary()
        assert s.total_compiles == 1
        assert s.success_rate == 1.0

    def test_build_failure(self, journal):
        entry = JournalEntry(
            event_type="build",
            success=False,
            error_summary="Syntax error in main.py",
        )
        journal.record(entry)
        s = journal.get_summary()
        assert s.total_builds == 1
        assert s.success_rate == 0.0
        assert s.last_entry.error_summary == "Syntax error in main.py"


class TestBuildJournalSummary:

    def test_empty_summary(self, journal):
        s = journal.get_summary()
        assert s.total_compiles == 0
        assert s.total_builds == 0
        assert s.success_rate == 0.0
        assert s.streak == 0

    def test_computed_summary(self, journal):
        for i in range(3):
            journal.record(JournalEntry(
                event_type="compile",
                success=True,
                trust_score=70.0 + i * 10,
                cost_usd=0.10,
                domain="software",
            ))
        journal.record(JournalEntry(
            event_type="build",
            success=True,
            trust_score=80.0,
            cost_usd=0.25,
            domain="api",
        ))

        s = journal.get_summary()
        assert s.total_compiles == 3
        assert s.total_builds == 1
        assert s.success_rate == 1.0
        assert s.avg_trust > 0
        assert s.total_cost == pytest.approx(0.55, rel=0.01)
        assert "software" in s.domains
        assert "api" in s.domains

    def test_streak_positive(self, journal):
        for _ in range(4):
            journal.record(JournalEntry(event_type="compile", success=True))
        s = journal.get_summary()
        assert s.streak == 4

    def test_streak_negative(self, journal):
        journal.record(JournalEntry(event_type="compile", success=True))
        journal.record(JournalEntry(event_type="compile", success=False))
        journal.record(JournalEntry(event_type="build", success=False))
        s = journal.get_summary()
        # Last 2 entries are failures (most recent first)
        assert s.streak == -2


class TestBuildJournalSearch:

    def test_search_by_description(self, journal):
        journal.record(JournalEntry(
            event_type="compile",
            description="A REST API for pets",
            success=True,
        ))
        journal.record(JournalEntry(
            event_type="compile",
            description="A chat bot",
            success=True,
        ))
        results = journal.search("REST API")
        assert len(results) == 1
        assert "REST API" in results[0].description

    def test_search_by_error(self, journal):
        journal.record(JournalEntry(
            event_type="build",
            success=False,
            error_summary="ImportError: no module named flask",
        ))
        results = journal.search("flask")
        assert len(results) == 1

    def test_search_no_results(self, journal):
        journal.record(JournalEntry(event_type="compile", description="hello"))
        results = journal.search("nonexistent_term_xyz")
        assert results == []

    def test_search_limit(self, journal):
        for i in range(5):
            journal.record(JournalEntry(
                event_type="compile",
                description=f"api endpoint {i}",
                success=True,
            ))
        results = journal.search("api", limit=3)
        assert len(results) == 3


class TestBuildJournalRecent:

    def test_recent_newest_first(self, journal):
        journal.record(JournalEntry(
            event_type="compile",
            description="first",
            timestamp=1000.0,
        ))
        journal.record(JournalEntry(
            event_type="compile",
            description="second",
            timestamp=2000.0,
        ))
        recent = journal.recent(limit=5)
        assert len(recent) == 2
        assert recent[0].description == "second"
        assert recent[1].description == "first"

    def test_recent_limit(self, journal):
        for i in range(10):
            journal.record(JournalEntry(event_type="compile", description=f"entry {i}"))
        recent = journal.recent(limit=3)
        assert len(recent) == 3


class TestBuildJournalPersistence:

    def test_close_and_reopen(self, tmp_path):
        db = tmp_path / "persist.db"
        j1 = BuildJournal(db)
        j1.record(JournalEntry(event_type="compile", description="test", success=True))
        j1.close()

        j2 = BuildJournal(db)
        recent = j2.recent(limit=1)
        assert len(recent) == 1
        assert recent[0].description == "test"
        j2.close()


class TestBuildJournalDomainTracking:

    def test_domain_counting(self, journal):
        journal.record(JournalEntry(event_type="compile", domain="software"))
        journal.record(JournalEntry(event_type="compile", domain="software"))
        journal.record(JournalEntry(event_type="compile", domain="api"))

        s = journal.get_summary()
        assert s.domains["software"] == 2
        assert s.domains["api"] == 1


class TestBuildJournalCost:

    def test_cost_aggregation(self, journal):
        journal.record(JournalEntry(event_type="compile", cost_usd=0.05))
        journal.record(JournalEntry(event_type="build", cost_usd=0.15))
        journal.record(JournalEntry(event_type="compile", cost_usd=0.10))

        s = journal.get_summary()
        assert s.total_cost == pytest.approx(0.30, rel=0.01)
