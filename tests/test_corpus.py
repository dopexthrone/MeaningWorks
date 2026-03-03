"""
Tests for SQLite Corpus - Persistence Layer.

Comprehensive test suite for SQLiteCorpus, covering:
- CRUD operations (store, get, list_all, list_by_domain)
- Full-text search (FTS5)
- Pagination
- Provider stats
- Domain suggestions
- Migration from JSON to SQLite
- Empty corpus handling
- Concurrent access (basic)
- Blueprint/context graph storage and retrieval
"""

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest

from persistence.corpus import Corpus, CompilationRecord
from persistence.sqlite_corpus import SQLiteCorpus
from persistence.migrations import migrate_json_to_sqlite, MigrationResult


# =============================================================================
# FIXTURES
# =============================================================================


def _make_context_graph(domain: str = "authentication", core_need: str = "auth system") -> Dict[str, Any]:
    """Build a minimal context graph for testing."""
    return {
        "known": {
            "intent": {
                "domain": domain,
                "core_need": core_need,
            }
        },
        "insights": [
            f"Insight 1 for {domain}",
            f"Insight 2 for {domain}",
        ],
    }


def _make_blueprint(components: List[str] = None, relationships: List[Dict] = None) -> Dict[str, Any]:
    """Build a minimal blueprint for testing."""
    if components is None:
        components = ["User", "Session", "AuthService"]

    comps = [
        {
            "name": name,
            "type": "entity",
            "description": f"{name} component",
            "derived_from": f"Derived from {name}",
        }
        for name in components
    ]

    if relationships is None:
        relationships = []
        if len(components) >= 2:
            relationships = [
                {
                    "from": components[0],
                    "to": components[1],
                    "type": "depends_on",
                    "description": f"{components[0]} depends on {components[1]}",
                }
            ]

    return {
        "components": comps,
        "relationships": relationships,
        "constraints": [],
        "unresolved": [],
    }


@pytest.fixture
def corpus_dir(tmp_path):
    """Provide a temporary directory for corpus storage."""
    return tmp_path / "test_corpus"


@pytest.fixture
def corpus(corpus_dir):
    """Provide a fresh SQLiteCorpus instance."""
    return SQLiteCorpus(corpus_path=corpus_dir)


@pytest.fixture
def populated_corpus(corpus):
    """Provide a corpus with several pre-stored compilations."""
    # Auth domain - 4 records (3 successful, 1 failed)
    for i in range(3):
        corpus.store(
            input_text=f"Build an authentication system variant {i}",
            context_graph=_make_context_graph("authentication", f"auth need {i}"),
            blueprint=_make_blueprint(["User", "Session", "AuthService"]),
            insights=[f"Auth insight {i}-a", f"Auth insight {i}-b"],
            success=True,
            provider="grok",
            model="grok-4-1-fast-reasoning",
            stage_timings={"synthesis": 10.0 + i, "verify": 2.0 + i},
            retry_counts={"synthesis": i, "verify": 0},
        )
    corpus.store(
        input_text="Build a broken auth system",
        context_graph=_make_context_graph("authentication", "broken auth"),
        blueprint=_make_blueprint(["User"]),
        insights=["Failed insight"],
        success=False,
        provider="grok",
        model="grok-4-1-fast-reasoning",
        stage_timings={"synthesis": 30.0},
        retry_counts={"synthesis": 3},
    )

    # E-commerce domain - 2 records
    corpus.store(
        input_text="Build an e-commerce product catalog",
        context_graph=_make_context_graph("e-commerce", "product catalog"),
        blueprint=_make_blueprint(["Product", "Category", "Cart"]),
        insights=["Products need categories", "Cart tracks items"],
        success=True,
        provider="claude",
        model="claude-3-opus",
        stage_timings={"synthesis": 15.0},
        retry_counts={"synthesis": 0},
    )
    corpus.store(
        input_text="Build an e-commerce checkout flow",
        context_graph=_make_context_graph("e-commerce", "checkout flow"),
        blueprint=_make_blueprint(["Order", "Payment", "Cart"]),
        insights=["Orders contain payments"],
        success=True,
        provider="openai",
        model="gpt-4",
        stage_timings={"synthesis": 8.0},
        retry_counts={"synthesis": 1},
    )

    return corpus


# =============================================================================
# CRUD OPERATIONS
# =============================================================================


class TestSQLiteCorpusStore:
    """Test store() operations."""

    def test_store_returns_record(self, corpus):
        """store() should return a CompilationRecord with correct fields."""
        record = corpus.store(
            input_text="Build a task manager",
            context_graph=_make_context_graph("productivity", "task management"),
            blueprint=_make_blueprint(["Task", "Project"]),
            insights=["Tasks belong to projects"],
            success=True,
            provider="grok",
            model="grok-4",
        )

        assert isinstance(record, CompilationRecord)
        assert record.domain == "productivity"
        assert record.components_count == 2
        assert record.insights_count == 1
        assert record.success is True
        assert record.provider == "grok"
        assert record.model == "grok-4"
        assert record.id  # Non-empty

    def test_store_creates_json_files(self, corpus):
        """store() should create backward-compatible JSON files."""
        record = corpus.store(
            input_text="Build a chat app",
            context_graph=_make_context_graph("messaging", "chat"),
            blueprint=_make_blueprint(["Message", "Channel"]),
            insights=["Messages flow through channels"],
            success=True,
        )

        compilation_dir = Path(record.file_path)
        assert (compilation_dir / "context-graph.json").exists()
        assert (compilation_dir / "blueprint.json").exists()
        assert (compilation_dir / "trace.md").exists()

    def test_store_truncates_input_text(self, corpus):
        """store() should truncate input_text to 500 chars in the record."""
        long_text = "A" * 1000
        record = corpus.store(
            input_text=long_text,
            context_graph=_make_context_graph(),
            blueprint=_make_blueprint(),
            insights=[],
            success=True,
        )

        assert len(record.input_text) == 500

    def test_store_upserts_on_same_input(self, corpus):
        """store() with same input_text should update, not duplicate."""
        input_text = "Build a user system"

        corpus.store(
            input_text=input_text,
            context_graph=_make_context_graph(),
            blueprint=_make_blueprint(["User"]),
            insights=["First attempt"],
            success=False,
        )
        record2 = corpus.store(
            input_text=input_text,
            context_graph=_make_context_graph(),
            blueprint=_make_blueprint(["User", "Role"]),
            insights=["Second attempt", "With roles"],
            success=True,
        )

        # Should only have one record
        all_records = corpus.list_all()
        assert len(all_records) == 1
        assert all_records[0].success is True
        assert all_records[0].components_count == 2

    def test_store_with_stage_timings_and_retries(self, corpus):
        """store() should persist stage_timings and retry_counts."""
        timings = {"synthesis": 12.5, "verify": 3.2}
        retries = {"synthesis": 2, "verify": 0}

        record = corpus.store(
            input_text="Build a system with metrics",
            context_graph=_make_context_graph(),
            blueprint=_make_blueprint(),
            insights=[],
            success=True,
            stage_timings=timings,
            retry_counts=retries,
        )

        fetched = corpus.get(record.id)
        assert fetched.stage_timings == timings
        assert fetched.retry_counts == retries


class TestSQLiteCorpusGet:
    """Test get() and list operations."""

    def test_get_existing_record(self, populated_corpus):
        """get() should return record by ID."""
        all_records = populated_corpus.list_all()
        target = all_records[0]

        fetched = populated_corpus.get(target.id)
        assert fetched is not None
        assert fetched.id == target.id
        assert fetched.domain == target.domain

    def test_get_nonexistent_returns_none(self, corpus):
        """get() should return None for missing ID."""
        assert corpus.get("nonexistent_id") is None

    def test_list_all_returns_sorted(self, populated_corpus):
        """list_all() should return records sorted by timestamp descending."""
        records = populated_corpus.list_all()
        assert len(records) == 6

        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_by_domain(self, populated_corpus):
        """list_by_domain() should filter by domain case-insensitively."""
        auth_records = populated_corpus.list_by_domain("authentication")
        assert len(auth_records) == 4

        ecomm_records = populated_corpus.list_by_domain("e-commerce")
        assert len(ecomm_records) == 2

        # Case insensitive
        upper_records = populated_corpus.list_by_domain("AUTHENTICATION")
        assert len(upper_records) == 4

    def test_list_by_domain_empty(self, populated_corpus):
        """list_by_domain() should return empty list for unknown domain."""
        records = populated_corpus.list_by_domain("nonexistent_domain")
        assert records == []


# =============================================================================
# FULL-TEXT SEARCH
# =============================================================================


class TestSQLiteCorpusSearch:
    """Test FTS5 full-text search."""

    def test_search_by_input_text(self, populated_corpus):
        """search() should find records matching input text."""
        results = populated_corpus.search("authentication")
        assert len(results) >= 3  # At least the auth records

    def test_search_by_domain_keyword(self, populated_corpus):
        """search() should match domain text."""
        results = populated_corpus.search("e-commerce")
        assert len(results) >= 2

    def test_search_partial_match(self, populated_corpus):
        """search() should handle partial word matches."""
        results = populated_corpus.search("catalog")
        assert len(results) >= 1

    def test_search_no_results(self, populated_corpus):
        """search() should return empty list when nothing matches."""
        results = populated_corpus.search("xyzzy_no_match_ever_12345")
        assert results == []

    def test_search_empty_corpus(self, corpus):
        """search() on empty corpus should return empty list."""
        results = corpus.search("anything")
        assert results == []


# =============================================================================
# PAGINATION
# =============================================================================


class TestSQLiteCorpusPagination:
    """Test pagination on list methods."""

    def test_list_all_pagination(self, populated_corpus):
        """list_all() with page/per_page should return correct slice."""
        page1 = populated_corpus.list_all(page=1, per_page=2)
        page2 = populated_corpus.list_all(page=2, per_page=2)
        page3 = populated_corpus.list_all(page=3, per_page=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 2

        # All IDs should be unique across pages
        all_ids = [r.id for r in page1 + page2 + page3]
        assert len(set(all_ids)) == 6

    def test_list_all_no_pagination_returns_all(self, populated_corpus):
        """list_all() without page/per_page returns all records."""
        all_records = populated_corpus.list_all()
        assert len(all_records) == 6

    def test_list_by_domain_pagination(self, populated_corpus):
        """list_by_domain() with pagination should slice correctly."""
        page1 = populated_corpus.list_by_domain("authentication", page=1, per_page=2)
        page2 = populated_corpus.list_by_domain("authentication", page=2, per_page=2)

        assert len(page1) == 2
        assert len(page2) == 2

        ids = [r.id for r in page1 + page2]
        assert len(set(ids)) == 4

    def test_pagination_beyond_results(self, populated_corpus):
        """Requesting a page beyond available results should return empty."""
        page_far = populated_corpus.list_all(page=100, per_page=10)
        assert page_far == []


# =============================================================================
# BLUEPRINT / CONTEXT GRAPH RETRIEVAL
# =============================================================================


class TestSQLiteCorpusBlobRetrieval:
    """Test blueprint and context graph loading."""

    def test_load_blueprint(self, populated_corpus):
        """load_blueprint() should return the stored blueprint dict."""
        records = populated_corpus.list_all()
        record = records[0]

        blueprint = populated_corpus.load_blueprint(record.id)
        assert blueprint is not None
        assert "components" in blueprint
        assert isinstance(blueprint["components"], list)

    def test_load_blueprint_nonexistent(self, corpus):
        """load_blueprint() should return None for missing ID."""
        assert corpus.load_blueprint("nonexistent") is None

    def test_load_context_graph(self, populated_corpus):
        """load_context_graph() should return the stored context graph."""
        records = populated_corpus.list_all()
        record = records[0]

        cg = populated_corpus.load_context_graph(record.id)
        assert cg is not None
        assert "known" in cg
        assert "intent" in cg["known"]

    def test_load_context_graph_nonexistent(self, corpus):
        """load_context_graph() should return None for missing ID."""
        assert corpus.load_context_graph("nonexistent") is None


# =============================================================================
# STATISTICS
# =============================================================================


class TestSQLiteCorpusStats:
    """Test get_stats() and get_provider_stats()."""

    def test_stats_empty_corpus(self, corpus):
        """get_stats() on empty corpus should return zeroed stats."""
        stats = corpus.get_stats()
        assert stats["total_compilations"] == 0
        assert stats["domains"] == {}
        assert stats["success_rate"] == 0.0
        assert stats["total_components"] == 0
        assert stats["total_insights"] == 0
        assert stats["domains_with_suggestions"] == []

    def test_stats_populated(self, populated_corpus):
        """get_stats() should aggregate correctly across all records."""
        stats = populated_corpus.get_stats()
        assert stats["total_compilations"] == 6
        assert stats["domains"]["authentication"] == 4
        assert stats["domains"]["e-commerce"] == 2
        assert stats["total_components"] > 0
        assert stats["total_insights"] > 0

        # 5 out of 6 successful
        assert abs(stats["success_rate"] - 5 / 6) < 0.01

    def test_stats_domains_with_suggestions(self, populated_corpus):
        """get_stats() should identify domains with 3+ successful compilations."""
        stats = populated_corpus.get_stats()
        # authentication has 3 successful, e-commerce has 2
        assert "authentication" in stats["domains_with_suggestions"]
        assert "e-commerce" not in stats["domains_with_suggestions"]

    def test_provider_stats_all(self, populated_corpus):
        """get_provider_stats() should return stats for all providers."""
        stats = populated_corpus.get_provider_stats()

        assert "grok" in stats
        assert stats["grok"]["total_compilations"] == 4
        assert "grok-4-1-fast-reasoning" in stats["grok"]["models_used"]

        assert "claude" in stats
        assert stats["claude"]["total_compilations"] == 1

        assert "openai" in stats
        assert stats["openai"]["total_compilations"] == 1

    def test_provider_stats_filtered(self, populated_corpus):
        """get_provider_stats(provider) should filter to one provider."""
        stats = populated_corpus.get_provider_stats(provider="grok")
        assert "grok" in stats
        assert len(stats) == 1

    def test_provider_stats_avg_synthesis(self, populated_corpus):
        """Provider stats should calculate average synthesis time."""
        stats = populated_corpus.get_provider_stats(provider="grok")
        # Grok records had synthesis times: 10.0, 11.0, 12.0, 30.0
        avg = (10.0 + 11.0 + 12.0 + 30.0) / 4
        assert abs(stats["grok"]["avg_synthesis_time"] - avg) < 0.1

    def test_provider_stats_empty(self, corpus):
        """get_provider_stats() on empty corpus should return empty dict."""
        stats = corpus.get_provider_stats()
        assert stats == {}


# =============================================================================
# DOMAIN SUGGESTIONS
# =============================================================================


class TestSQLiteCorpusDomainSuggestions:
    """Test get_domain_suggestions()."""

    def test_suggestions_insufficient_samples(self, corpus):
        """Should report insufficient samples when below threshold."""
        corpus.store(
            input_text="Single auth record",
            context_graph=_make_context_graph("authentication"),
            blueprint=_make_blueprint(["User"]),
            insights=["One insight"],
            success=True,
        )

        suggestions = corpus.get_domain_suggestions("authentication", min_samples=3)
        assert suggestions["has_suggestions"] is False
        assert "Insufficient samples" in suggestions.get("reason", "")

    def test_suggestions_with_enough_samples(self, populated_corpus):
        """Should return component suggestions when enough samples exist."""
        suggestions = populated_corpus.get_domain_suggestions(
            "authentication", min_frequency=0.6, min_samples=3
        )
        assert suggestions["has_suggestions"] is True
        assert suggestions["sample_size"] == 3
        assert len(suggestions["suggested_components"]) > 0
        # User and Session appear in all 3 successful auth blueprints
        assert "User" in suggestions["suggested_components"]
        assert "Session" in suggestions["suggested_components"]

    def test_suggestions_unknown_domain(self, populated_corpus):
        """Should return no suggestions for unknown domain."""
        suggestions = populated_corpus.get_domain_suggestions("nonexistent")
        assert suggestions["has_suggestions"] is False
        assert suggestions["sample_size"] == 0


# =============================================================================
# EXPORT FOR RECOMPILE
# =============================================================================


class TestSQLiteCorpusExport:
    """Test export_for_recompile()."""

    def test_export_existing_record(self, populated_corpus):
        """export_for_recompile() should produce formatted recompile string."""
        records = populated_corpus.list_all()
        record = records[0]

        export = populated_corpus.export_for_recompile(record.id)
        assert export is not None
        assert record.id in export
        assert "Prior compilation" in export
        assert "Key insights" in export

    def test_export_nonexistent_returns_none(self, corpus):
        """export_for_recompile() should return None for missing ID."""
        assert corpus.export_for_recompile("nonexistent") is None


# =============================================================================
# MIGRATION
# =============================================================================


class TestMigration:
    """Test JSON-to-SQLite migration."""

    def test_migrate_populated_json_corpus(self, tmp_path):
        """Migration should transfer all records from JSON to SQLite."""
        json_path = tmp_path / "json_corpus"
        sqlite_path = tmp_path / "sqlite_corpus"

        # Create and populate a JSON corpus
        json_corpus = Corpus(corpus_path=json_path)
        for i in range(3):
            json_corpus.store(
                input_text=f"Migration test input {i}",
                context_graph=_make_context_graph("test", f"test need {i}"),
                blueprint=_make_blueprint(["CompA", "CompB"]),
                insights=[f"Insight {i}"],
                success=True,
                provider="grok",
                model="grok-4",
            )

        # Run migration
        result = migrate_json_to_sqlite(json_path, sqlite_path)

        assert result.records_found == 3
        assert result.records_migrated == 3
        assert result.records_failed == 0
        assert result.success is True

        # Verify SQLite corpus has the data
        sqlite_corpus = SQLiteCorpus(corpus_path=sqlite_path)
        assert sqlite_corpus.count() == 3

        all_records = sqlite_corpus.list_all()
        assert len(all_records) == 3

    def test_migrate_empty_corpus(self, tmp_path):
        """Migration of empty JSON corpus should succeed with zero records."""
        json_path = tmp_path / "empty_corpus"
        sqlite_path = tmp_path / "sqlite_corpus"

        # Create empty JSON corpus (just the directory and empty index)
        json_corpus = Corpus(corpus_path=json_path)

        result = migrate_json_to_sqlite(json_path, sqlite_path)

        assert result.records_found == 0
        assert result.records_migrated == 0
        assert result.success is True

    def test_migrate_no_json_corpus(self, tmp_path):
        """Migration with no existing JSON corpus should do nothing."""
        nonexistent_path = tmp_path / "does_not_exist"

        result = migrate_json_to_sqlite(nonexistent_path)

        assert result.records_found == 0
        assert result.records_migrated == 0
        assert result.success is True

    def test_migrate_same_path(self, tmp_path):
        """Migration into same directory should work (SQLite alongside JSON)."""
        corpus_path = tmp_path / "shared_corpus"

        # Create JSON corpus
        json_corpus = Corpus(corpus_path=corpus_path)
        json_corpus.store(
            input_text="Same-path migration test",
            context_graph=_make_context_graph("test"),
            blueprint=_make_blueprint(["CompA"]),
            insights=["An insight"],
            success=True,
        )

        # Migrate in-place
        result = migrate_json_to_sqlite(corpus_path)

        assert result.records_migrated == 1
        assert result.success is True

        # Both index.json and corpus.db should exist
        assert (corpus_path / "index.json").exists()
        assert (corpus_path / "corpus.db").exists()


# =============================================================================
# EMPTY CORPUS
# =============================================================================


class TestSQLiteCorpusEmpty:
    """Test behavior of empty corpus."""

    def test_empty_list_all(self, corpus):
        """list_all() on empty corpus should return empty list."""
        assert corpus.list_all() == []

    def test_empty_count(self, corpus):
        """count() on empty corpus should return 0."""
        assert corpus.count() == 0

    def test_empty_search(self, corpus):
        """search() on empty corpus should return empty list."""
        assert corpus.search("anything") == []

    def test_empty_domain_suggestions(self, corpus):
        """get_domain_suggestions() on empty corpus returns no suggestions."""
        s = corpus.get_domain_suggestions("anything")
        assert s["has_suggestions"] is False


# =============================================================================
# CONCURRENT ACCESS
# =============================================================================


class TestSQLiteCorpusConcurrency:
    """Test basic concurrent access to the corpus."""

    def test_concurrent_writes(self, corpus_dir):
        """Multiple threads writing to the same corpus should not corrupt data."""
        # Pre-initialize schema so threads don't race on CREATE TABLE
        SQLiteCorpus(corpus_path=corpus_dir)
        errors = []

        def write_record(thread_id: int):
            try:
                # Each thread gets its own connection via a fresh corpus instance
                c = SQLiteCorpus(corpus_path=corpus_dir)
                c.store(
                    input_text=f"Concurrent write from thread {thread_id}",
                    context_graph=_make_context_graph("concurrency", f"thread {thread_id}"),
                    blueprint=_make_blueprint([f"Comp_{thread_id}"]),
                    insights=[f"Thread {thread_id} insight"],
                    success=True,
                    provider="test",
                    model="test-model",
                )
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Concurrent write errors: {errors}"

        # Verify all records were stored
        corpus = SQLiteCorpus(corpus_path=corpus_dir)
        all_records = corpus.list_all()
        assert len(all_records) == 10

    def test_concurrent_reads_and_writes(self, corpus_dir):
        """Reads and writes happening concurrently should not error."""
        # Pre-populate
        corpus = SQLiteCorpus(corpus_path=corpus_dir)
        corpus.store(
            input_text="Pre-existing record for concurrent test",
            context_graph=_make_context_graph("concurrency"),
            blueprint=_make_blueprint(["Base"]),
            insights=["Base insight"],
            success=True,
        )

        read_errors = []
        write_errors = []

        def reader(iteration: int):
            try:
                c = SQLiteCorpus(corpus_path=corpus_dir)
                records = c.list_all()
                # Should always get at least the pre-existing record
                assert len(records) >= 1
            except Exception as e:
                read_errors.append((iteration, str(e)))

        def writer(iteration: int):
            try:
                c = SQLiteCorpus(corpus_path=corpus_dir)
                c.store(
                    input_text=f"Concurrent mixed write {iteration}",
                    context_graph=_make_context_graph("concurrency"),
                    blueprint=_make_blueprint([f"Mixed_{iteration}"]),
                    insights=[f"Mixed insight {iteration}"],
                    success=True,
                )
            except Exception as e:
                write_errors.append((iteration, str(e)))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader, args=(i,)))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert read_errors == [], f"Concurrent read errors: {read_errors}"
        assert write_errors == [], f"Concurrent write errors: {write_errors}"


# =============================================================================
# DATABASE INTEGRITY
# =============================================================================


class TestSQLiteCorpusIntegrity:
    """Test database integrity and schema correctness."""

    def test_db_file_created(self, corpus):
        """Initializing corpus should create the database file."""
        assert corpus.db_path.exists()

    def test_tables_exist(self, corpus):
        """All expected tables should exist in the database."""
        conn = sqlite3.connect(str(corpus.db_path))
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {row[0] for row in tables}

            assert "compilations" in table_names
            assert "blueprints" in table_names
            assert "context_graphs" in table_names
            assert "compilations_fts" in table_names
        finally:
            conn.close()

    def test_wal_mode_enabled(self, corpus):
        """Database should use WAL journal mode."""
        conn = sqlite3.connect(str(corpus.db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
        finally:
            conn.close()
