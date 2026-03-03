"""Tests for mother/knowledge_base.py — structured fact extraction and storage."""

import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from mother.knowledge_base import (
    KnowledgeFact,
    CATEGORIES,
    extract_facts,
    save_fact,
    save_facts,
    query_facts,
    search_facts,
    decay_stale_facts,
    fact_count,
    format_knowledge_context,
    _normalize_fact_id,
)


# ── Helpers ──────────────────────────────────────────────────────

def _tmp_db() -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return Path(f.name)


# ── Fact ID Normalization ────────────────────────────────────────

class TestNormalizeFactId:
    def test_deterministic(self):
        id1 = _normalize_fact_id("preference", "user", "dark mode")
        id2 = _normalize_fact_id("preference", "user", "dark mode")
        assert id1 == id2

    def test_case_insensitive(self):
        id1 = _normalize_fact_id("preference", "User", "Dark Mode")
        id2 = _normalize_fact_id("preference", "user", "dark mode")
        assert id1 == id2

    def test_caps_length(self):
        long_value = "x" * 200
        fid = _normalize_fact_id("preference", "user", long_value)
        assert len(fid) <= 120


# ── Fact Extraction ──────────────────────────────────────────────

class TestExtractFacts:
    def test_extracts_preference(self):
        facts = extract_facts("I prefer using dark mode", "user", session_id="s1")
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) >= 1
        assert prefs[0].subject == "user"
        assert prefs[0].predicate == "prefers"

    def test_extracts_always_preference(self):
        facts = extract_facts("I always use TypeScript for backends", "user", session_id="s1")
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) >= 1

    def test_extracts_never_constraint(self):
        facts = extract_facts("Never use var in JavaScript", "user", session_id="s1")
        constraints = [f for f in facts if f.category == "constraint"]
        assert len(constraints) >= 1

    def test_extracts_must_constraint(self):
        facts = extract_facts("You must not modify the config file", "user")
        constraints = [f for f in facts if f.category == "constraint"]
        assert len(constraints) >= 1

    def test_extracts_decision(self):
        facts = extract_facts("Let's use PostgreSQL for the database", "user")
        decisions = [f for f in facts if f.category == "decision"]
        assert len(decisions) >= 1

    def test_extracts_tool(self):
        facts = extract_facts("We switched to Redis for caching", "assistant")
        tools = [f for f in facts if f.category == "tool"]
        assert len(tools) >= 1
        assert any("Redis" in f.subject for f in tools)

    def test_extracts_tool_from_install(self):
        facts = extract_facts("pip install flask-cors", "assistant")
        tools = [f for f in facts if f.category == "tool"]
        assert len(tools) >= 1

    def test_preference_only_from_user(self):
        facts = extract_facts("I prefer functional programming", "assistant")
        prefs = [f for f in facts if f.category == "preference"]
        assert len(prefs) == 0  # Assistant preferences don't count

    def test_decision_from_both_roles(self):
        facts_user = extract_facts("Let's use FastAPI", "user")
        facts_asst = extract_facts("Let's use FastAPI", "assistant")
        user_decisions = [f for f in facts_user if f.category == "decision"]
        asst_decisions = [f for f in facts_asst if f.category == "decision"]
        assert len(user_decisions) >= 1
        assert len(asst_decisions) >= 1

    def test_empty_message(self):
        facts = extract_facts("", "user")
        assert facts == []

    def test_no_false_positives_stopwords(self):
        facts = extract_facts("the and or but this that", "user")
        # Should not extract tools from stopwords
        tools = [f for f in facts if f.category == "tool"]
        assert len(tools) == 0

    def test_deduplicates_within_extraction(self):
        facts = extract_facts(
            "I prefer dark mode. I also prefer dark mode for all apps.", "user"
        )
        prefs = [f for f in facts if f.category == "preference"]
        ids = [f.fact_id for f in prefs]
        assert len(ids) == len(set(ids))  # No duplicate IDs

    def test_fact_is_frozen(self):
        facts = extract_facts("I prefer Python", "user")
        if facts:
            with pytest.raises(AttributeError):
                facts[0].value = "modified"

    def test_confidence_levels(self):
        pref_facts = extract_facts("I prefer dark mode", "user")
        constraint_facts = extract_facts("Never use eval", "user")
        tool_facts = extract_facts("We're using Django", "assistant")

        if pref_facts:
            assert pref_facts[0].confidence == 0.8
        if constraint_facts:
            constraints = [f for f in constraint_facts if f.category == "constraint"]
            if constraints:
                assert constraints[0].confidence == 0.9
        if tool_facts:
            tools = [f for f in tool_facts if f.category == "tool"]
            if tools:
                assert tools[0].confidence == 0.6

    def test_person_extraction(self):
        facts = extract_facts("My colleague John helped with the deployment", "user")
        persons = [f for f in facts if f.category == "person"]
        assert len(persons) >= 1


# ── Persistence ──────────────────────────────────────────────────

class TestFactPersistence:
    def test_save_and_query(self):
        db = _tmp_db()
        fact = KnowledgeFact(
            fact_id="test-1",
            category="preference",
            subject="user",
            predicate="prefers",
            value="dark mode",
            confidence=0.8,
            source="conversation:s1",
            first_seen=time.time(),
            last_confirmed=time.time(),
            access_count=0,
        )
        save_fact(fact, db_path=db)

        results = query_facts(category="preference", db_path=db)
        assert len(results) == 1
        assert results[0].value == "dark mode"

    def test_upsert_bumps_confidence(self):
        db = _tmp_db()
        fact = KnowledgeFact(
            fact_id="bump-1",
            category="tool",
            subject="Redis",
            predicate="in_use",
            value="Uses Redis",
            confidence=0.6,
            source="conversation:s1",
            first_seen=time.time(),
            last_confirmed=time.time(),
            access_count=0,
        )
        save_fact(fact, db_path=db)
        save_fact(fact, db_path=db)  # Second save = confirmation

        results = query_facts(subject="Redis", db_path=db)
        assert len(results) == 1
        assert results[0].confidence == 0.7  # 0.6 + 0.1

    def test_save_facts_batch(self):
        db = _tmp_db()
        facts = extract_facts("I prefer Python and let's use FastAPI", "user")
        if facts:
            count = save_facts(facts, db_path=db)
            assert count == len(facts)
            assert fact_count(db_path=db) == len(facts)

    def test_save_facts_empty(self):
        db = _tmp_db()
        count = save_facts([], db_path=db)
        assert count == 0


# ── Query ────────────────────────────────────────────────────────

class TestFactQuery:
    def test_query_by_subject(self):
        db = _tmp_db()
        f1 = KnowledgeFact("s1", "tool", "Redis", "uses", "Uses Redis", 0.8, "", time.time(), time.time(), 0)
        f2 = KnowledgeFact("s2", "tool", "PostgreSQL", "uses", "Uses PostgreSQL", 0.7, "", time.time(), time.time(), 0)
        save_fact(f1, db_path=db)
        save_fact(f2, db_path=db)

        results = query_facts(subject="Redis", db_path=db)
        assert len(results) == 1
        assert results[0].subject == "Redis"

    def test_query_by_category(self):
        db = _tmp_db()
        f1 = KnowledgeFact("c1", "preference", "user", "prefers", "dark mode", 0.8, "", time.time(), time.time(), 0)
        f2 = KnowledgeFact("c2", "tool", "Redis", "uses", "Uses Redis", 0.7, "", time.time(), time.time(), 0)
        save_fact(f1, db_path=db)
        save_fact(f2, db_path=db)

        results = query_facts(category="preference", db_path=db)
        assert len(results) == 1
        assert results[0].category == "preference"

    def test_query_min_confidence(self):
        db = _tmp_db()
        f1 = KnowledgeFact("mc1", "tool", "A", "uses", "Low conf", 0.2, "", time.time(), time.time(), 0)
        f2 = KnowledgeFact("mc2", "tool", "B", "uses", "High conf", 0.9, "", time.time(), time.time(), 0)
        save_fact(f1, db_path=db)
        save_fact(f2, db_path=db)

        results = query_facts(min_confidence=0.5, db_path=db)
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_query_missing_db(self):
        results = query_facts(db_path=Path("/tmp/nonexistent_kb.db"))
        assert results == []


# ── Search ───────────────────────────────────────────────────────

class TestFactSearch:
    def test_search_by_value(self):
        db = _tmp_db()
        f = KnowledgeFact("sv1", "preference", "user", "prefers", "dark mode theme", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = search_facts("dark mode", db_path=db)
        assert len(results) >= 1

    def test_search_by_subject(self):
        db = _tmp_db()
        f = KnowledgeFact("ss1", "tool", "PostgreSQL", "uses", "Uses PostgreSQL", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        results = search_facts("PostgreSQL", db_path=db)
        assert len(results) >= 1

    def test_search_no_results(self):
        db = _tmp_db()
        results = search_facts("quantum blockchain", db_path=db)
        assert len(results) == 0

    def test_search_missing_db(self):
        results = search_facts("anything", db_path=Path("/tmp/nonexistent_search.db"))
        assert results == []


# ── Decay ────────────────────────────────────────────────────────

class TestDecayFacts:
    def test_decay_stale_facts(self):
        db = _tmp_db()
        old_ts = time.time() - (60 * 86400)  # 60 days ago
        f = KnowledgeFact("decay1", "tool", "OldTool", "uses", "Old tool", 0.3, "", old_ts, old_ts, 0)
        save_fact(f, db_path=db)

        # Decay with 30-day threshold
        deleted = decay_stale_facts(max_age_days=30.0, decay_rate=0.3, db_path=db)
        # 0.3 - 0.3 = 0.0 → deleted
        assert deleted >= 1

    def test_no_decay_recent_facts(self):
        db = _tmp_db()
        f = KnowledgeFact("nodecy1", "tool", "NewTool", "uses", "New tool", 0.8, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)

        deleted = decay_stale_facts(max_age_days=30.0, db_path=db)
        assert deleted == 0
        assert fact_count(db_path=db) == 1


# ── Fact Count ───────────────────────────────────────────────────

class TestFactCount:
    def test_count_empty(self):
        db = _tmp_db()
        assert fact_count(db_path=db) == 0

    def test_count_after_insert(self):
        db = _tmp_db()
        f = KnowledgeFact("cnt1", "tool", "X", "uses", "X", 0.5, "", time.time(), time.time(), 0)
        save_fact(f, db_path=db)
        assert fact_count(db_path=db) == 1

    def test_count_missing_db(self):
        assert fact_count(db_path=Path("/tmp/nonexistent_count.db")) == 0


# ── Context Formatting ──────────────────────────────────────────

class TestFormatKnowledgeContext:
    def test_formats_facts(self):
        facts = [
            KnowledgeFact("f1", "preference", "user", "prefers", "dark mode", 0.8, "", time.time(), time.time(), 0),
            KnowledgeFact("f2", "tool", "Redis", "uses", "Uses Redis", 0.7, "", time.time(), time.time(), 0),
        ]
        block = format_knowledge_context(facts)
        assert "[KNOWN FACTS]" in block
        assert "dark mode" in block
        assert "Redis" in block

    def test_empty_returns_empty(self):
        assert format_knowledge_context([]) == ""

    def test_groups_by_category(self):
        facts = [
            KnowledgeFact("g1", "preference", "user", "prefers", "A", 0.8, "", time.time(), time.time(), 0),
            KnowledgeFact("g2", "preference", "user", "prefers", "B", 0.7, "", time.time(), time.time(), 0),
            KnowledgeFact("g3", "tool", "Redis", "uses", "C", 0.9, "", time.time(), time.time(), 0),
        ]
        block = format_knowledge_context(facts)
        assert "Preference:" in block
        assert "Tool:" in block

    def test_respects_token_budget(self):
        facts = [
            KnowledgeFact(f"b{i}", "tool", f"Tool{i}", "uses", f"Uses Tool{i} " * 10, 0.8, "", time.time(), time.time(), 0)
            for i in range(50)
        ]
        block = format_knowledge_context(facts, max_tokens=50)
        assert len(block) < 400

    def test_priority_order(self):
        facts = [
            KnowledgeFact("o1", "tool", "X", "uses", "tool fact", 0.8, "", time.time(), time.time(), 0),
            KnowledgeFact("o2", "preference", "user", "prefers", "pref fact", 0.8, "", time.time(), time.time(), 0),
        ]
        block = format_knowledge_context(facts)
        # Preference should appear before Tool in output
        pref_pos = block.find("Preference:")
        tool_pos = block.find("Tool:")
        if pref_pos >= 0 and tool_pos >= 0:
            assert pref_pos < tool_pos


# ── Categories ───────────────────────────────────────────────────

class TestCategories:
    def test_all_categories_are_strings(self):
        for cat in CATEGORIES:
            assert isinstance(cat, str)
            assert len(cat) > 2

    def test_expected_categories_exist(self):
        assert "preference" in CATEGORIES
        assert "decision" in CATEGORIES
        assert "tool" in CATEGORIES
        assert "constraint" in CATEGORIES
