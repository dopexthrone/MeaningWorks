"""Tests for kernel/memory.py — L2 pattern detection extensions."""

import json
import sqlite3
import time
from pathlib import Path

import pytest

from kernel.memory import (
    MemoryRecord,
    LearnedPattern,
    detect_patterns,
    save_patterns,
    load_patterns,
    format_pattern_context,
    save_memory,
    _ensure_patterns_schema,
    _db_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(run_id, **kwargs):
    defaults = dict(
        run_id=run_id,
        timestamp=time.time(),
        intent_summary="Build something",
        domain="software",
        trust_score=75.0,
        fidelity_score=0.8,
        compression_losses={},
        gate_results={"compilation": "pass"},
        learnings=[],
        gaps=[],
        cell_count=20,
        fill_rate=0.7,
        cost_usd=0.15,
        duration_seconds=120.0,
    )
    defaults.update(kwargs)
    return MemoryRecord(**defaults)


# ---------------------------------------------------------------------------
# LearnedPattern dataclass
# ---------------------------------------------------------------------------

class TestLearnedPattern:
    def test_frozen(self):
        p = LearnedPattern(
            pattern_id="test", category="recurring_gap",
            description="desc", frequency=3, confidence=0.7,
            affected_postcodes=(), remediation="fix",
        )
        with pytest.raises(AttributeError):
            p.pattern_id = "changed"

    def test_fields(self):
        p = LearnedPattern(
            pattern_id="p1", category="domain_confidence",
            description="d", frequency=5, confidence=0.9,
            affected_postcodes=("INT.ENT.ECO.WHAT.SFT",),
            remediation="review",
        )
        assert p.category == "domain_confidence"
        assert len(p.affected_postcodes) == 1


# ---------------------------------------------------------------------------
# detect_patterns
# ---------------------------------------------------------------------------

class TestDetectPatterns:
    def test_too_few_memories(self):
        memories = [_make_record(f"r{i}") for i in range(2)]
        assert detect_patterns(memories, min_occurrences=3) == []

    def test_recurring_compression_loss(self):
        memories = [
            _make_record(f"r{i}", compression_losses={"entity": 2})
            for i in range(5)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert "recurring_loss_entity" in ids

    def test_multiple_loss_categories(self):
        memories = []
        for i in range(4):
            losses = {"entity": 1}
            if i < 3:
                losses["behavior"] = 1
            memories.append(_make_record(f"r{i}", compression_losses=losses))

        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert "recurring_loss_entity" in ids
        assert "recurring_loss_behavior" in ids

    def test_low_trust_domain(self):
        memories = [
            _make_record(f"r{i}", domain="api", trust_score=40.0)
            for i in range(5)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert "low_trust_api" in ids

    def test_high_trust_domain_not_flagged(self):
        memories = [
            _make_record(f"r{i}", domain="software", trust_score=85.0)
            for i in range(5)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert not any("low_trust" in pid for pid in ids)

    def test_recurring_gaps(self):
        memories = [
            _make_record(f"r{i}", gaps=["Low completeness: 40/100"])
            for i in range(4)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        assert any("recurring_gap" in p.pattern_id for p in patterns)

    def test_gate_failure_pattern(self):
        memories = [
            _make_record(f"r{i}", gate_results={"compilation": "fail", "closed_loop": "fail"})
            for i in range(5)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert "gate_failure_compilation" in ids
        assert "gate_failure_closed_loop" in ids

    def test_confidence_capped_at_1(self):
        memories = [
            _make_record(f"r{i}", compression_losses={"entity": 1})
            for i in range(100)
        ]
        patterns = detect_patterns(memories, min_occurrences=3)
        for p in patterns:
            assert p.confidence <= 1.0

    def test_min_occurrences_respected(self):
        memories = [
            _make_record(f"r{i}", compression_losses={"rare": 1})
            for i in range(2)
        ]
        # Add more to meet minimum count but not for the "rare" category
        memories.extend([_make_record(f"x{i}") for i in range(3)])
        patterns = detect_patterns(memories, min_occurrences=3)
        ids = [p.pattern_id for p in patterns]
        assert "recurring_loss_rare" not in ids


# ---------------------------------------------------------------------------
# save_patterns / load_patterns
# ---------------------------------------------------------------------------

class TestPatternPersistence:
    def test_save_and_load(self, tmp_path):
        patterns = [
            LearnedPattern(
                pattern_id="p1", category="recurring_gap",
                description="Entity loss recurring", frequency=5,
                confidence=0.8, affected_postcodes=(), remediation="fix entity",
            ),
        ]
        count = save_patterns(patterns, db_dir=tmp_path)
        assert count == 1

        loaded = load_patterns(min_confidence=0.5, db_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded[0].pattern_id == "p1"
        assert loaded[0].confidence == 0.8

    def test_upsert(self, tmp_path):
        p1 = LearnedPattern("p1", "gap", "v1", 3, 0.5, (), "fix")
        p2 = LearnedPattern("p1", "gap", "v2", 7, 0.9, (), "fix better")
        save_patterns([p1], db_dir=tmp_path)
        save_patterns([p2], db_dir=tmp_path)

        loaded = load_patterns(min_confidence=0.0, db_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded[0].frequency == 7
        assert loaded[0].description == "v2"

    def test_confidence_filter(self, tmp_path):
        patterns = [
            LearnedPattern("p1", "gap", "high", 5, 0.9, (), "fix"),
            LearnedPattern("p2", "gap", "low", 2, 0.2, (), "fix"),
        ]
        save_patterns(patterns, db_dir=tmp_path)

        high_only = load_patterns(min_confidence=0.5, db_dir=tmp_path)
        assert len(high_only) == 1
        assert high_only[0].pattern_id == "p1"

    def test_empty_save(self, tmp_path):
        assert save_patterns([], db_dir=tmp_path) == 0

    def test_nonexistent_dir(self, tmp_path):
        loaded = load_patterns(db_dir=tmp_path / "nonexistent")
        assert loaded == []


# ---------------------------------------------------------------------------
# format_pattern_context
# ---------------------------------------------------------------------------

class TestFormatPatternContext:
    def test_empty(self):
        assert format_pattern_context([]) == ""

    def test_formats_patterns(self):
        patterns = [
            LearnedPattern("p1", "recurring_gap", "Entity loss", 5, 0.8, (), "Fix entity"),
        ]
        result = format_pattern_context(patterns)
        assert "[LEARNED PATTERNS" in result
        assert "Entity loss" in result
        assert "Fix entity" in result

    def test_capped_at_10(self):
        patterns = [
            LearnedPattern(f"p{i}", "gap", f"desc {i}", i, 0.5, (), "fix")
            for i in range(20)
        ]
        result = format_pattern_context(patterns)
        # Count bullet points
        assert result.count("- [") <= 10


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestPatternsSchema:
    def test_idempotent(self, tmp_path):
        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        _ensure_patterns_schema(conn)
        _ensure_patterns_schema(conn)
        conn.close()

    def test_table_exists(self, tmp_path):
        path = _db_path(tmp_path)
        conn = sqlite3.connect(str(path))
        _ensure_patterns_schema(conn)
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "learned_patterns" in tables
        conn.close()
