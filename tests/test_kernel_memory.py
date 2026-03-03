"""Tests for kernel/memory.py — episodic memory consolidation."""

import json
import sqlite3
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel.memory import (
    MemoryRecord,
    consolidate,
    save_memory,
    load_recent_memories,
    memory_stats,
    _ensure_memory_schema,
    _extract_trust,
    _extract_gate_results,
    _extract_learnings,
    _extract_gaps,
    _db_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary db directory."""
    return tmp_path


def _make_state(known=None):
    """Minimal SharedState-like object."""
    s = SimpleNamespace()
    s.known = known or {}
    return s


def _make_result(
    success=True,
    verification=None,
    insights=None,
    semantic_grid=None,
    blueprint=None,
    cache_stats=None,
    stage_timings=None,
):
    return SimpleNamespace(
        success=success,
        verification=verification or {},
        insights=insights or [],
        semantic_grid=semantic_grid,
        blueprint=blueprint or {},
        cache_stats=cache_stats or {},
        stage_timings=stage_timings or {},
    )


def _make_record(run_id="test-001", **kwargs):
    defaults = dict(
        run_id=run_id,
        timestamp=time.time(),
        intent_summary="Build a task manager",
        domain="software",
        trust_score=75.0,
        fidelity_score=0.8,
        compression_losses={"entity": 2},
        gate_results={"compilation": "pass"},
        learnings=["pattern reuse detected"],
        gaps=["Low completeness: 55/100"],
        cell_count=20,
        fill_rate=0.7,
        cost_usd=0.15,
        duration_seconds=120.0,
    )
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


# ---------------------------------------------------------------------------
# MemoryRecord dataclass
# ---------------------------------------------------------------------------

class TestMemoryRecord:
    def test_frozen(self):
        r = _make_record()
        with pytest.raises(AttributeError):
            r.run_id = "changed"

    def test_fields(self):
        r = _make_record()
        assert r.run_id == "test-001"
        assert r.domain == "software"
        assert r.trust_score == 75.0
        assert isinstance(r.compression_losses, dict)
        assert isinstance(r.learnings, list)


# ---------------------------------------------------------------------------
# consolidate()
# ---------------------------------------------------------------------------

class TestConsolidate:
    def test_basic_consolidation(self):
        state = _make_state({"input": "Build a task manager for teams", "domain": "software"})
        result = _make_result(
            success=True,
            verification={"completeness": {"score": 80}, "consistency": {"score": 90}},
            insights=["pattern reuse detected", "missing auth component"],
        )
        mem = consolidate(result, state, "run-001", 120.0)

        assert mem.run_id == "run-001"
        assert mem.intent_summary == "Build a task manager for teams"
        assert mem.domain == "software"
        assert mem.trust_score > 0
        assert mem.duration_seconds == 120.0
        assert "pattern" in " ".join(mem.learnings).lower()

    def test_intent_truncation(self):
        long_input = "x" * 500
        state = _make_state({"input": long_input})
        result = _make_result()
        mem = consolidate(result, state, "run-002", 10.0)
        assert len(mem.intent_summary) <= 200

    def test_empty_state(self):
        state = _make_state({})
        result = _make_result()
        mem = consolidate(result, state, "run-003", 0.0)
        assert mem.intent_summary == ""
        assert mem.domain == "software"  # default

    def test_fidelity_from_closed_loop(self):
        state = _make_state({
            "input": "test",
            "closed_loop_result": {"fidelity": 0.85, "passed": True},
        })
        result = _make_result()
        mem = consolidate(result, state, "run-004", 5.0)
        assert mem.fidelity_score == 0.85

    def test_compression_losses_extracted(self):
        state = _make_state({
            "input": "test",
            "compression_loss_categories": {"entity": 3, "behavior": 1},
        })
        result = _make_result()
        mem = consolidate(result, state, "run-005", 5.0)
        assert mem.compression_losses == {"entity": 3, "behavior": 1}

    def test_grid_stats_from_semantic_grid(self):
        result = _make_result(semantic_grid={"cells": 25, "fill_rate": 0.8})
        state = _make_state({"input": "test"})
        mem = consolidate(result, state, "run-006", 5.0)
        assert mem.cell_count == 25
        assert mem.fill_rate == 0.8

    def test_gate_results_success(self):
        state = _make_state({
            "input": "test",
            "closed_loop_result": {"passed": True, "fidelity": 0.9},
        })
        result = _make_result(success=True, verification={"completeness": 80})
        mem = consolidate(result, state, "run-007", 5.0)
        assert mem.gate_results["compilation"] == "pass"
        assert mem.gate_results["closed_loop"] == "pass"

    def test_gate_results_failure(self):
        state = _make_state({
            "input": "test",
            "closed_loop_result": {"passed": False, "fidelity": 0.3},
        })
        result = _make_result(success=False, verification={})
        mem = consolidate(result, state, "run-008", 5.0)
        assert mem.gate_results["compilation"] == "fail"
        assert mem.gate_results["closed_loop"] == "fail"


# ---------------------------------------------------------------------------
# _extract_trust
# ---------------------------------------------------------------------------

class TestExtractTrust:
    def test_dict_scores(self):
        v = {"completeness": {"score": 80}, "consistency": {"score": 60}}
        assert _extract_trust(v) == pytest.approx(35.0)  # (80+60+0+0)/4

    def test_numeric_scores(self):
        v = {"completeness": 100, "consistency": 80, "coherence": 90, "traceability": 70}
        assert _extract_trust(v) == pytest.approx(85.0)

    def test_empty(self):
        assert _extract_trust({}) == 0.0
        assert _extract_trust(None) == 0.0


# ---------------------------------------------------------------------------
# _extract_learnings / _extract_gaps
# ---------------------------------------------------------------------------

class TestExtractLearnings:
    def test_keyword_match(self):
        insights = ["pattern detected in auth", "random noise", "improved throughput"]
        result = _extract_learnings(insights)
        assert len(result) == 2

    def test_cap_at_10(self):
        insights = [f"pattern {i}" for i in range(20)]
        assert len(_extract_learnings(insights)) == 10

    def test_non_string_ignored(self):
        assert _extract_learnings([None, 42, "pattern match"]) == ["pattern match"]


class TestExtractGaps:
    def test_insight_gaps(self):
        insights = ["missing component", "all good", "incomplete spec"]
        result = _extract_gaps(insights, _make_result())
        assert len(result) == 2

    def test_low_verification_scores(self):
        result = _make_result(verification={
            "completeness": {"score": 40},
            "consistency": {"score": 90},
        })
        gaps = _extract_gaps([], result)
        assert any("completeness" in g.lower() for g in gaps)

    def test_cap_at_10(self):
        insights = [f"missing item {i}" for i in range(20)]
        assert len(_extract_gaps(insights, _make_result())) == 10


# ---------------------------------------------------------------------------
# save_memory / load_recent_memories
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_db):
        rec = _make_record("run-100")
        row_id = save_memory(rec, db_dir=tmp_db)
        assert row_id > 0

        loaded = load_recent_memories(limit=10, db_dir=tmp_db)
        assert len(loaded) == 1
        assert loaded[0].run_id == "run-100"
        assert loaded[0].trust_score == 75.0

    def test_upsert_on_duplicate_run_id(self, tmp_db):
        rec1 = _make_record("run-dup", trust_score=50.0)
        save_memory(rec1, db_dir=tmp_db)
        rec2 = _make_record("run-dup", trust_score=90.0)
        save_memory(rec2, db_dir=tmp_db)

        loaded = load_recent_memories(limit=10, db_dir=tmp_db)
        assert len(loaded) == 1
        assert loaded[0].trust_score == 90.0

    def test_ordering_most_recent_first(self, tmp_db):
        for i in range(5):
            rec = _make_record(f"run-{i}", timestamp=1000.0 + i)
            save_memory(rec, db_dir=tmp_db)

        loaded = load_recent_memories(limit=10, db_dir=tmp_db)
        assert loaded[0].run_id == "run-4"
        assert loaded[-1].run_id == "run-0"

    def test_limit_works(self, tmp_db):
        for i in range(10):
            save_memory(_make_record(f"run-{i}"), db_dir=tmp_db)

        loaded = load_recent_memories(limit=3, db_dir=tmp_db)
        assert len(loaded) == 3

    def test_domain_filter(self, tmp_db):
        save_memory(_make_record("run-sw", domain="software"), db_dir=tmp_db)
        save_memory(_make_record("run-api", domain="api"), db_dir=tmp_db)

        sw = load_recent_memories(domain="software", db_dir=tmp_db)
        assert len(sw) == 1
        assert sw[0].domain == "software"

    def test_empty_db(self, tmp_db):
        loaded = load_recent_memories(db_dir=tmp_db)
        assert loaded == []

    def test_nonexistent_dir(self):
        loaded = load_recent_memories(db_dir=Path("/nonexistent/path/abc123"))
        assert loaded == []

    def test_json_roundtrip(self, tmp_db):
        rec = _make_record(
            "run-json",
            compression_losses={"entity": 3, "behavior": 1},
            gate_results={"compilation": "pass", "closed_loop": "fail"},
            learnings=["learned A", "learned B"],
            gaps=["gap X"],
        )
        save_memory(rec, db_dir=tmp_db)
        loaded = load_recent_memories(db_dir=tmp_db)[0]
        assert loaded.compression_losses == {"entity": 3, "behavior": 1}
        assert loaded.gate_results == {"compilation": "pass", "closed_loop": "fail"}
        assert loaded.learnings == ["learned A", "learned B"]
        assert loaded.gaps == ["gap X"]


# ---------------------------------------------------------------------------
# memory_stats
# ---------------------------------------------------------------------------

class TestMemoryStats:
    def test_empty(self, tmp_db):
        stats = memory_stats(db_dir=tmp_db)
        assert stats["total"] == 0

    def test_aggregates(self, tmp_db):
        save_memory(_make_record("r1", trust_score=80.0, cost_usd=0.10, duration_seconds=100), db_dir=tmp_db)
        save_memory(_make_record("r2", trust_score=60.0, cost_usd=0.20, duration_seconds=200), db_dir=tmp_db)

        stats = memory_stats(db_dir=tmp_db)
        assert stats["total"] == 2
        assert stats["avg_trust"] == pytest.approx(70.0)
        assert stats["total_cost"] == pytest.approx(0.30, abs=0.01)
        assert stats["avg_duration"] == pytest.approx(150.0)

    def test_domain_breakdown(self, tmp_db):
        save_memory(_make_record("r1", domain="software"), db_dir=tmp_db)
        save_memory(_make_record("r2", domain="software"), db_dir=tmp_db)
        save_memory(_make_record("r3", domain="api"), db_dir=tmp_db)

        stats = memory_stats(db_dir=tmp_db)
        assert stats["domains"]["software"] == 2
        assert stats["domains"]["api"] == 1

    def test_nonexistent_dir(self):
        stats = memory_stats(db_dir=Path("/nonexistent/path/abc123"))
        assert stats["total"] == 0


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchema:
    def test_schema_idempotent(self, tmp_db):
        path = _db_path(tmp_db)
        conn = sqlite3.connect(str(path))
        _ensure_memory_schema(conn)
        _ensure_memory_schema(conn)  # second call should not error
        conn.close()

    def test_table_exists(self, tmp_db):
        path = _db_path(tmp_db)
        conn = sqlite3.connect(str(path))
        _ensure_memory_schema(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "memory_records" in table_names
        conn.close()
