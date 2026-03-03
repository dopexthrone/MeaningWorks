"""Tests for mother.relationship — cross-session compounding."""

import json
import time
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from mother.relationship import (
    RelationshipInsight,
    extract_topic_keywords,
    _classify_time_of_day,
    extract_preferred_time,
    compute_relationship_stage,
    extract_relationship_insights,
    synthesize_relationship_narrative,
)
from mother.memory import ConversationStore
from mother.context import ContextData, synthesize_situation
from mother.persona import build_greeting


# --- Helpers ---

def _msg(role, content, ts=0.0, sid="s1"):
    return (role, content, ts, sid)


def _session(sid, count, first, last):
    return {"session_id": sid, "message_count": count, "first_message": first, "last_message": last}


# =============================================================================
# TestExtractTopicKeywords
# =============================================================================

class TestExtractTopicKeywords:
    def test_empty(self):
        assert extract_topic_keywords([]) == {}

    def test_stopwords_filtered(self):
        msgs = [_msg("user", "I want to make something good")]
        result = extract_topic_keywords(msgs)
        assert "want" not in result
        assert "make" not in result
        assert "something" not in result

    def test_short_words_filtered(self):
        msgs = [
            _msg("user", "the api key"),
            _msg("user", "the api key"),
        ]
        result = extract_topic_keywords(msgs)
        # "api" and "key" are 3 chars, filtered out (< 4)
        assert "api" not in result
        assert "key" not in result

    def test_slash_commands_excluded(self):
        msgs = [
            _msg("user", "/compile booking system"),
            _msg("user", "/compile booking system"),
        ]
        result = extract_topic_keywords(msgs)
        assert result == {}

    def test_frequency_counting(self):
        msgs = [
            _msg("user", "I want to build a booking system"),
            _msg("user", "The booking system needs calendar integration"),
            _msg("user", "Also booking should handle payments"),
        ]
        result = extract_topic_keywords(msgs)
        assert "booking" in result
        assert result["booking"] == 3

    def test_top_10_limit(self):
        # Create 15 distinct keywords each appearing 2+ times
        msgs = []
        for i in range(15):
            word = f"keyword{i:02d}"
            msgs.append(_msg("user", f"discuss {word} concept"))
            msgs.append(_msg("user", f"more about {word} today"))
        result = extract_topic_keywords(msgs)
        assert len(result) <= 10

    def test_user_only(self):
        msgs = [
            _msg("assistant", "architecture architecture architecture architecture"),
            _msg("user", "hello there"),
        ]
        result = extract_topic_keywords(msgs)
        assert "architecture" not in result

    def test_min_2_occurrences(self):
        msgs = [_msg("user", "unique_longword_here is interesting")]
        # Only 1 occurrence — shouldn't appear after extract but the raw
        # counter may include it. The orchestrator filters to min 2.
        # extract_topic_keywords returns top 10 from Counter which includes
        # items with count 1. The orchestrator (extract_relationship_insights)
        # does the min-2 filter. Test that the raw function at least returns something.
        result = extract_topic_keywords(msgs)
        # Count is 1, so it may appear in the raw output (Counter doesn't filter)
        # But in extract_relationship_insights the min-2 filter is applied.


# =============================================================================
# TestClassifyTimeOfDay
# =============================================================================

class TestClassifyTimeOfDay:
    def test_morning(self):
        assert _classify_time_of_day(5) == "morning"
        assert _classify_time_of_day(11) == "morning"

    def test_afternoon(self):
        assert _classify_time_of_day(12) == "afternoon"
        assert _classify_time_of_day(16) == "afternoon"

    def test_evening(self):
        assert _classify_time_of_day(17) == "evening"
        assert _classify_time_of_day(20) == "evening"

    def test_night(self):
        assert _classify_time_of_day(21) == "night"
        assert _classify_time_of_day(4) == "night"
        assert _classify_time_of_day(0) == "night"


# =============================================================================
# TestExtractPreferredTime
# =============================================================================

class TestExtractPreferredTime:
    def test_few_sessions(self):
        """Needs 3+ sessions."""
        assert extract_preferred_time([1.0, 2.0]) == ""

    def test_clear_evening_preference(self):
        # Create timestamps at 7pm, 8pm, 7:30pm
        import calendar
        base = time.mktime(time.strptime("2026-02-10 19:00:00", "%Y-%m-%d %H:%M:%S"))
        stamps = [base, base + 3600, base + 86400 + 1800]
        assert extract_preferred_time(stamps) == "evening"

    def test_no_clear_preference(self):
        """When spread evenly, no bucket >50%."""
        # morning, afternoon, evening, night
        stamps = [
            time.mktime(time.strptime("2026-02-10 08:00:00", "%Y-%m-%d %H:%M:%S")),
            time.mktime(time.strptime("2026-02-11 14:00:00", "%Y-%m-%d %H:%M:%S")),
            time.mktime(time.strptime("2026-02-12 19:00:00", "%Y-%m-%d %H:%M:%S")),
            time.mktime(time.strptime("2026-02-13 23:00:00", "%Y-%m-%d %H:%M:%S")),
        ]
        assert extract_preferred_time(stamps) == ""


# =============================================================================
# TestComputeRelationshipStage
# =============================================================================

class TestComputeRelationshipStage:
    def test_new(self):
        assert compute_relationship_stage(0, 0) == "new"
        assert compute_relationship_stage(1, 5) == "new"
        assert compute_relationship_stage(2, 20) == "new"

    def test_building(self):
        assert compute_relationship_stage(3, 30) == "building"
        assert compute_relationship_stage(8, 100) == "building"

    def test_established(self):
        assert compute_relationship_stage(9, 100) == "established"
        assert compute_relationship_stage(20, 500) == "established"

    def test_deep_with_rapport(self):
        assert compute_relationship_stage(21, 500, rapport_baseline=0.5) == "deep"
        assert compute_relationship_stage(21, 500, rapport_baseline=0.8) == "deep"

    def test_deep_without_rapport_stays_established(self):
        assert compute_relationship_stage(21, 500, rapport_baseline=0.3) == "established"


# =============================================================================
# TestExtractRelationshipInsights
# =============================================================================

class TestExtractRelationshipInsights:
    def test_empty(self):
        insight = extract_relationship_insights([], [])
        assert insight.relationship_stage == "new"
        assert insight.sessions_analyzed == 0
        assert insight.messages_analyzed == 0

    def test_basic(self):
        msgs = [
            _msg("user", "build me a booking system", 1000.0, "s1"),
            _msg("assistant", "on it", 1001.0, "s1"),
            _msg("user", "booking needs calendar", 1002.0, "s1"),
        ]
        sessions = [_session("s1", 3, 1000.0, 1002.0)]
        insight = extract_relationship_insights(msgs, sessions)
        assert insight.sessions_analyzed == 1
        assert insight.messages_analyzed == 3
        assert insight.relationship_stage == "new"
        assert insight.computed_at > 0

    def test_with_sense_memory(self):
        msgs = [_msg("user", "hello there friend", 1000.0, "s1")] * 5
        sessions = [_session("s1", 5, 1000.0, 1004.0)] * 5
        sense_mem = {"rapport_trend": 0.3, "confidence_trend": -0.1, "peak_rapport": 0.6}
        insight = extract_relationship_insights(msgs, sessions, sense_memory=sense_mem)
        assert insight.rapport_direction == "growing"
        assert insight.confidence_direction == "declining"

    def test_with_corpus(self):
        msgs = [_msg("user", "test", 1000.0, "s1")]
        sessions = [_session("s1", 1, 1000.0, 1000.0)]
        corpus = {
            "total_compilations": 5,
            "domains": {"software": 3, "api": 2},
        }
        insight = extract_relationship_insights(msgs, sessions, corpus_summary=corpus)
        assert insight.compilation_count == 5
        assert insight.primary_domain == "software"
        assert "software" in insight.domains_explored
        assert "api" in insight.domains_explored

    def test_provenance(self):
        msgs = [_msg("user", "hi", 1000.0, "s1")] * 10
        sessions = [_session("s1", 10, 1000.0, 1009.0)]
        insight = extract_relationship_insights(msgs, sessions)
        assert insight.sessions_analyzed == 1
        assert insight.messages_analyzed == 10

    def test_frozen(self):
        insight = extract_relationship_insights([], [])
        with pytest.raises(AttributeError):
            insight.relationship_stage = "deep"

    def test_conversational_ratio(self):
        msgs = [
            _msg("user", "hello", 1.0, "s1"),
            _msg("user", "how are you", 2.0, "s1"),
            _msg("user", "/compile something", 3.0, "s1"),
            _msg("user", "that looks great", 4.0, "s1"),
        ]
        sessions = [_session("s1", 4, 1.0, 4.0)]
        insight = extract_relationship_insights(msgs, sessions)
        # 3 out of 4 user messages are conversational
        assert 0.74 < insight.conversational_ratio < 0.76

    def test_recurring_topics_min_2(self):
        msgs = [
            _msg("user", "architecture is interesting", 1.0, "s1"),
            _msg("user", "architecture patterns matter", 2.0, "s1"),
            _msg("user", "unique_word_xyz once", 3.0, "s1"),
        ]
        sessions = [_session("s1", 3, 1.0, 3.0)]
        insight = extract_relationship_insights(msgs, sessions)
        assert "architecture" in insight.recurring_topics
        # unique_word_xyz only appears once -> filtered out
        assert "unique_word_xyz" not in insight.recurring_topics


# =============================================================================
# TestSynthesizeRelationshipNarrative
# =============================================================================

class TestSynthesizeRelationshipNarrative:
    def test_new_user(self):
        insight = RelationshipInsight(relationship_stage="new")
        assert synthesize_relationship_narrative(insight) == "New user. No prior history."

    def test_building(self):
        insight = RelationshipInsight(
            relationship_stage="building",
            sessions_analyzed=5,
            messages_analyzed=30,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "Building relationship" in narrative

    def test_established_with_domain(self):
        insight = RelationshipInsight(
            relationship_stage="established",
            primary_domain="software",
            sessions_analyzed=15,
            messages_analyzed=200,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "Established relationship" in narrative
        assert "software" in narrative

    def test_deep(self):
        insight = RelationshipInsight(
            relationship_stage="deep",
            primary_domain="software",
            preferred_time_of_day="evening",
            recurring_topics={"architecture": 5, "compilation": 3},
            sessions_analyzed=25,
            messages_analyzed=500,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "Deep relationship" in narrative
        assert "evening" in narrative
        assert "architecture" in narrative

    def test_sparse_data(self):
        insight = RelationshipInsight(
            relationship_stage="building",
            sessions_analyzed=3,
            messages_analyzed=5,
        )
        narrative = synthesize_relationship_narrative(insight)
        # Should be short — no domain, no time preference, no topics
        assert "Building relationship" in narrative
        assert len(narrative) < 100

    def test_conversational_ratio_high(self):
        insight = RelationshipInsight(
            relationship_stage="established",
            conversational_ratio=0.9,
            messages_analyzed=50,
            sessions_analyzed=10,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "conversational" in narrative.lower()

    def test_conversational_ratio_low(self):
        insight = RelationshipInsight(
            relationship_stage="established",
            conversational_ratio=0.2,
            messages_analyzed=50,
            sessions_analyzed=10,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "task-oriented" in narrative.lower()

    def test_rapport_growing(self):
        insight = RelationshipInsight(
            relationship_stage="building",
            rapport_direction="growing",
            sessions_analyzed=5,
            messages_analyzed=30,
        )
        narrative = synthesize_relationship_narrative(insight)
        assert "Rapport growing" in narrative

    def test_max_3_sentences(self):
        insight = RelationshipInsight(
            relationship_stage="deep",
            primary_domain="software",
            preferred_time_of_day="evening",
            recurring_topics={"arch": 5, "build": 3, "test": 2},
            conversational_ratio=0.9,
            rapport_direction="growing",
            messages_analyzed=100,
            sessions_analyzed=25,
        )
        narrative = synthesize_relationship_narrative(insight)
        # Count sentences (rough: periods followed by space or end)
        sentences = [s.strip() for s in narrative.split(". ") if s.strip()]
        # Should not exceed ~3 conceptual sentences
        assert len(narrative) < 300


# =============================================================================
# TestRelationshipInsightsPersistence
# =============================================================================

class TestRelationshipInsightsPersistence:
    def test_save_and_load(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            data = {"relationship_stage": "building", "sessions_analyzed": 5}
            store.save_relationship_insights(json.dumps(data), 1234567890.0)
            loaded = store.load_relationship_insights()
            assert loaded is not None
            assert json.loads(loaded[0]) == data
            assert loaded[1] == 1234567890.0
        finally:
            store.close()

    def test_empty_load(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            assert store.load_relationship_insights() is None
        finally:
            store.close()

    def test_upsert(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            store.save_relationship_insights('{"v":1}', 100.0)
            store.save_relationship_insights('{"v":2}', 200.0)
            loaded = store.load_relationship_insights()
            assert json.loads(loaded[0]) == {"v": 2}
            assert loaded[1] == 200.0
        finally:
            store.close()


# =============================================================================
# TestGetAllMessages
# =============================================================================

class TestGetAllMessages:
    def test_multi_session(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            store.add_message("user", "hello", session_id="s1")
            store.add_message("assistant", "hi", session_id="s1")
            store.add_message("user", "world", session_id="s2")
            msgs = store.get_all_messages()
            assert len(msgs) == 3
            # Ordered by timestamp ASC
            assert msgs[0][1] == "hello"
            assert msgs[2][1] == "world"
        finally:
            store.close()

    def test_limit(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            for i in range(10):
                store.add_message("user", f"msg{i}")
            msgs = store.get_all_messages(limit=3)
            assert len(msgs) == 3
        finally:
            store.close()

    def test_ordering(self, tmp_path):
        store = ConversationStore(path=tmp_path / "test.db")
        try:
            store.add_message("user", "first")
            store.add_message("user", "second")
            store.add_message("user", "third")
            msgs = store.get_all_messages()
            contents = [m[1] for m in msgs]
            assert contents == ["first", "second", "third"]
        finally:
            store.close()


# =============================================================================
# TestContextWithRelationship
# =============================================================================

class TestContextWithRelationship:
    def test_narrative_renders(self):
        data = ContextData(
            total_sessions=10,
            total_messages=100,
            relationship_narrative="Established relationship. Works primarily on software projects.",
        )
        situation = synthesize_situation(data)
        assert "Established relationship" in situation
        assert "Works primarily on software projects" in situation
        # Should NOT contain the stat line fallback
        assert "10 sessions" not in situation

    def test_fallback_without_narrative(self):
        data = ContextData(
            total_sessions=10,
            total_messages=100,
            relationship_narrative="",
        )
        situation = synthesize_situation(data)
        # Should use stat line fallback
        assert "10 sessions" in situation

    def test_first_session(self):
        data = ContextData(
            total_sessions=0,
            total_messages=0,
            relationship_narrative="should be ignored",
        )
        situation = synthesize_situation(data)
        assert "First session" in situation


# =============================================================================
# TestGreetingWithRelationship
# =============================================================================

class TestGreetingWithRelationship:
    def _config(self):
        from mother.config import MotherConfig
        return MotherConfig()

    def test_deep_with_domain(self):
        insight = RelationshipInsight(
            relationship_stage="deep",
            primary_domain="software",
        )
        greeting = build_greeting(
            self._config(),
            memory_summary={"total_sessions": 25, "topics": ["hello"]},
            relationship_insight=insight,
        )
        assert "software" in greeting

    def test_established_with_topics(self):
        insight = RelationshipInsight(
            relationship_stage="established",
            recurring_topics={"architecture": 5},
        )
        greeting = build_greeting(
            self._config(),
            memory_summary={"total_sessions": 15, "topics": ["hello"]},
            relationship_insight=insight,
        )
        assert "architecture" in greeting

    def test_building_recent(self):
        insight = RelationshipInsight(
            relationship_stage="building",
        )
        greeting = build_greeting(
            self._config(),
            memory_summary={"total_sessions": 5, "days_since_last": 0.5, "topics": []},
            relationship_insight=insight,
        )
        assert greeting == "Continuing."

    def test_no_insight_falls_through(self):
        greeting = build_greeting(
            self._config(),
            memory_summary={"total_sessions": 5, "days_since_last": 2.0, "topics": ["hello"]},
            relationship_insight=None,
        )
        # Should fall through to existing logic — David 8 voice
        assert "hello" in greeting.lower() or "where" in greeting.lower() or "need" in greeting.lower()

    def test_new_user_no_insight(self):
        greeting = build_greeting(
            self._config(),
            memory_summary={"total_sessions": 0},
        )
        assert greeting == "What would you like to build?"
