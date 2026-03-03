"""
Tests for core/classification.py — deterministic component classification.

Phase 15: Component Classification Algorithm
~20 tests — mention frequency, grammatical role, semantic centrality,
type inference, component detection, full classification pipeline.
"""

import pytest

from core.classification import (
    ClassificationScore,
    compute_mention_frequency,
    detect_grammatical_role,
    compute_semantic_centrality,
    infer_component_type,
    is_likely_component,
    classify_components,
    filter_by_confidence,
    needs_llm_fallback,
)


# =============================================================================
# Mention Frequency Tests
# =============================================================================

class TestMentionFrequency:
    def test_zero_mentions(self):
        freq = compute_mention_frequency("XyzFoo", "build a login system", [])
        assert freq == 0.0

    def test_single_input_mention(self):
        freq = compute_mention_frequency("login", "build a login system", [])
        assert freq > 0.0

    def test_multiple_mentions_higher(self):
        f1 = compute_mention_frequency("login", "build a login system", [])
        f2 = compute_mention_frequency(
            "login",
            "build a login system with login page and login flow",
            ["the login handles authentication", "login is the entry point"],
        )
        assert f2 > f1

    def test_input_weighted_higher(self):
        # Same word count but all in input vs all in dialogue
        f_input = compute_mention_frequency("auth", "auth auth auth auth auth", [])
        f_dialogue = compute_mention_frequency("auth", "", ["auth auth auth auth auth"])
        assert f_input > f_dialogue

    def test_multi_word_name(self):
        freq = compute_mention_frequency(
            "Intent Agent",
            "the intent agent extracts intent from user input",
            [],
        )
        assert freq > 0.0


# =============================================================================
# Grammatical Role Tests
# =============================================================================

class TestGrammaticalRole:
    def test_subject_detected_by_verb(self):
        role = detect_grammatical_role(
            "Governor", "Governor handles agent orchestration", [],
        )
        assert role == "subject"

    def test_object_detected_by_contains(self):
        role = detect_grammatical_role(
            "SharedState", "the system stores SharedState between turns", [],
        )
        assert role == "object"

    def test_agent_keyword_forces_subject(self):
        role = detect_grammatical_role("Entity Agent", "", [])
        assert role == "subject"

    def test_state_keyword_forces_object(self):
        role = detect_grammatical_role("SharedState", "", [])
        assert role == "object"

    def test_modifier_detected(self):
        role = detect_grammatical_role(
            "confidence", "confidence value tracks progress", [],
        )
        # "confidence" contains keyword "score"-like patterns
        assert role in ("modifier", "object")

    def test_unknown_for_no_context(self):
        role = detect_grammatical_role("Xyzzy", "", [])
        assert role == "unknown"


# =============================================================================
# Semantic Centrality Tests
# =============================================================================

class TestSemanticCentrality:
    def test_zero_relationships(self):
        cent = compute_semantic_centrality("A", [], 5)
        assert cent == 0.0

    def test_high_centrality(self):
        rels = [
            {"from": "A", "to": "B"},
            {"from": "C", "to": "A"},
            {"from": "A", "to": "D"},
            {"from": "E", "to": "A"},
        ]
        cent = compute_semantic_centrality("A", rels, 5)
        assert cent > 0.5

    def test_low_centrality(self):
        rels = [
            {"from": "B", "to": "C"},
            {"from": "D", "to": "E"},
        ]
        cent = compute_semantic_centrality("A", rels, 5)
        assert cent == 0.0

    def test_single_component(self):
        cent = compute_semantic_centrality("A", [], 1)
        assert cent == 0.5  # Default for single component


# =============================================================================
# Type Inference Tests
# =============================================================================

class TestInferComponentType:
    def test_agent_keyword(self):
        t, c = infer_component_type("Entity Agent", "unknown")
        assert t == "agent"
        assert c > 0.5

    def test_state_keyword(self):
        t, c = infer_component_type("SharedState", "unknown")
        assert t == "entity"

    def test_protocol_keyword(self):
        t, c = infer_component_type("DialogueProtocol", "unknown")
        assert t == "process"

    def test_no_keywords_defaults_entity(self):
        t, c = infer_component_type("Reservation", "unknown")
        assert t == "entity"
        assert c < 0.5

    def test_llm_type_as_tiebreaker(self):
        t, c = infer_component_type("Reservation", "subject", "process")
        # With subject role + LLM saying process, should lean process
        assert t in ("process", "agent")

    def test_event_keyword(self):
        t, c = infer_component_type("BookingEvent", "unknown")
        assert t == "event"


# =============================================================================
# Component Detection Tests
# =============================================================================

class TestIsLikelyComponent:
    def test_generic_term_rejected(self):
        is_comp, _ = is_likely_component("data", 0.5, "object", 0.3)
        assert not is_comp

    def test_short_name_rejected(self):
        is_comp, _ = is_likely_component("DB", 0.5, "object", 0.3)
        assert not is_comp

    def test_enum_value_rejected(self):
        is_comp, _ = is_likely_component("AWAITING_INPUT", 0.5, "object", 0.3)
        assert not is_comp

    def test_real_component_accepted(self):
        is_comp, _ = is_likely_component("SharedState", 0.5, "object", 0.5)
        assert is_comp

    def test_modifier_low_freq_low_centrality_rejected(self):
        is_comp, _ = is_likely_component("flags", 0.1, "modifier", 0.0)
        assert not is_comp

    def test_zero_everything_rejected(self):
        is_comp, _ = is_likely_component("Phantom", 0.0, "unknown", 0.0)
        assert not is_comp


# =============================================================================
# Full Classification Pipeline Tests
# =============================================================================

class TestClassifyComponents:
    def test_empty_candidates(self):
        result = classify_components([], "test input", [], [])
        assert result == []

    def test_single_candidate(self):
        candidates = [{"name": "SharedState", "type": "entity", "derived_from": "test"}]
        result = classify_components(candidates, "SharedState stores data", [], [])
        assert len(result) == 1
        assert result[0].name == "SharedState"
        assert result[0].is_component

    def test_sorts_by_confidence(self):
        candidates = [
            {"name": "flags", "type": "entity", "derived_from": ""},
            {"name": "SharedState", "type": "entity", "derived_from": "test"},
            {"name": "Governor Agent", "type": "agent", "derived_from": "test"},
        ]
        result = classify_components(
            candidates,
            "Governor Agent manages SharedState. flags are internal",
            [],
            [{"from": "Governor Agent", "to": "SharedState"}],
        )
        # Governor Agent should rank highest (subject + agent keyword + relationship)
        assert result[0].name == "Governor Agent"

    def test_filter_by_confidence(self):
        candidates = [
            {"name": "SharedState", "type": "entity", "derived_from": "test"},
            {"name": "data", "type": "entity", "derived_from": "test"},
        ]
        scores = classify_components(candidates, "SharedState stores data", [], [])
        accepted, rejected = filter_by_confidence(scores, threshold=0.3)
        accepted_names = {s.name for s in accepted}
        rejected_names = {s.name for s in rejected}
        assert "SharedState" in accepted_names
        assert "data" in rejected_names


class TestNeedsLlmFallback:
    def test_high_confidence_no_fallback(self):
        score = ClassificationScore(
            name="X", mention_frequency=0.8, grammatical_role="subject",
            semantic_centrality=0.5, inferred_type="agent", type_confidence=0.9,
            is_component=True, overall_confidence=0.8, reasoning="",
        )
        assert not needs_llm_fallback(score)

    def test_low_confidence_needs_fallback(self):
        score = ClassificationScore(
            name="X", mention_frequency=0.3, grammatical_role="unknown",
            semantic_centrality=0.1, inferred_type="entity", type_confidence=0.4,
            is_component=True, overall_confidence=0.4, reasoning="",
        )
        assert needs_llm_fallback(score)

    def test_rejected_no_fallback(self):
        score = ClassificationScore(
            name="X", mention_frequency=0.0, grammatical_role="modifier",
            semantic_centrality=0.0, inferred_type="entity", type_confidence=0.3,
            is_component=False, overall_confidence=0.1, reasoning="",
        )
        assert not needs_llm_fallback(score)


# =============================================================================
# has_relationships gate tests (DECOMPOSE stage fix)
# =============================================================================

class TestHasRelationshipsGate:
    """When relationships=[] (DECOMPOSE stage), centrality is structurally 0.0.
    The classifier should not use absent data as negative evidence."""

    def test_zero_freq_zero_centrality_accepted_without_relationships(self):
        """Component with no mentions and no centrality passes when relationships don't exist."""
        is_comp, _ = is_likely_component("TaskManager", 0.0, "unknown", 0.0, has_relationships=False)
        assert is_comp

    def test_zero_freq_zero_centrality_rejected_with_relationships(self):
        """Same component rejected when relationships exist (existing behavior)."""
        is_comp, _ = is_likely_component("TaskManager", 0.0, "unknown", 0.0, has_relationships=True)
        assert not is_comp

    def test_modifier_low_freq_accepted_without_relationships(self):
        """Modifier with low freq but no relationship data should not be rejected on centrality."""
        is_comp, _ = is_likely_component("priority", 0.1, "modifier", 0.0, has_relationships=False)
        assert is_comp

    def test_modifier_low_freq_rejected_with_relationships(self):
        """Modifier with low freq and low centrality rejected when relationships exist."""
        is_comp, _ = is_likely_component("priority", 0.1, "modifier", 0.0, has_relationships=True)
        assert not is_comp

    def test_classify_components_empty_relationships_keeps_components(self):
        """classify_components with relationships=[] should not filter everything."""
        candidates = [
            {"name": "TaskManager", "type": "process", "derived_from": "user"},
            {"name": "TaskStore", "type": "entity", "derived_from": "user"},
            {"name": "NotificationService", "type": "agent", "derived_from": "user"},
        ]
        scores = classify_components(candidates, "build a task manager", [], [])
        accepted, _ = filter_by_confidence(scores, threshold=0.15)
        assert len(accepted) >= 1, f"Expected at least 1 accepted, got {len(accepted)}: {[(s.name, s.overall_confidence, s.is_component) for s in scores]}"

    def test_classify_components_with_relationships_still_filters(self):
        """classify_components with real relationships still filters low-quality components."""
        candidates = [
            {"name": "Governor Agent", "type": "agent", "derived_from": "test"},
            {"name": "data", "type": "entity", "derived_from": "test"},
        ]
        rels = [{"from": "Governor Agent", "to": "SharedState"}]
        scores = classify_components(
            candidates, "Governor Agent manages state", [], rels,
        )
        accepted, rejected = filter_by_confidence(scores, threshold=0.15)
        accepted_names = {s.name for s in accepted}
        rejected_names = {s.name for s in rejected}
        assert "Governor Agent" in accepted_names
        assert "data" in rejected_names
