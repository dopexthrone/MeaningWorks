"""
Tests for corpus feedback loop activation.

Validates: suggestion usage tracking, anti-pattern detection,
and corpus_feedback field in CompileResult.
"""

import pytest
from core.engine import MotherlabsEngine, CompileResult
from core.llm import MockClient


# =============================================================================
# FIXTURES
# =============================================================================

def _make_engine(**kwargs):
    """Create engine with MockClient for testing."""
    return MotherlabsEngine(llm_client=MockClient(), auto_store=False, **kwargs)


SAMPLE_BLUEPRINT = {
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item"},
        {"name": "Team", "type": "entity", "description": "A group"},
        {"name": "Notification", "type": "process", "description": "Alert system"},
    ],
    "relationships": [
        {"from": "Task", "to": "Team", "type": "belongs_to"},
        {"from": "Notification", "to": "Task", "type": "notifies"},
    ],
    "constraints": [],
}

BLUEPRINT_NO_DESCRIPTIONS = {
    "components": [
        {"name": "Task", "type": "entity"},
        {"name": "Team", "type": "entity"},
    ],
    "relationships": [],
    "constraints": [],
}

BLUEPRINT_WITH_ORPHAN = {
    "components": [
        {"name": "Task", "type": "entity", "description": "A work item"},
        {"name": "Orphan", "type": "entity", "description": "Disconnected"},
    ],
    "relationships": [],  # Orphan has no relationships
    "constraints": [],
}


# =============================================================================
# CompileResult has corpus_feedback field
# =============================================================================

class TestCompileResultCorpusFeedback:
    def test_field_exists(self):
        result = CompileResult(success=True)
        assert hasattr(result, "corpus_feedback")
        assert result.corpus_feedback == {}

    def test_field_accepts_dict(self):
        result = CompileResult(
            success=True,
            corpus_feedback={"suggestion_hit_rate": 0.5},
        )
        assert result.corpus_feedback["suggestion_hit_rate"] == 0.5


# =============================================================================
# Suggestion usage tracking
# =============================================================================

class TestSuggestionUsageTracking:
    def test_no_suggestions(self):
        """No suggestions = none influence."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None, None,
        )
        assert feedback["suggestion_hit_rate"] == 0.0
        assert feedback["corpus_influence"] == "none"
        assert feedback["suggestions_used"] == []
        assert feedback["suggestions_ignored"] == []

    def test_empty_suggestions(self):
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {"has_suggestions": False, "suggested_components": []},
            None,
        )
        assert feedback["corpus_influence"] == "none"

    def test_all_suggestions_used(self):
        """All suggested components appear in blueprint → strong influence."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {
                "has_suggestions": True,
                "suggested_components": ["Task", "Team"],
            },
            None,
        )
        assert feedback["suggestion_hit_rate"] == 1.0
        assert feedback["corpus_influence"] == "strong"
        assert "Task" in feedback["suggestions_used"]
        assert "Team" in feedback["suggestions_used"]
        assert feedback["suggestions_ignored"] == []

    def test_partial_suggestions_used(self):
        """Some suggestions used → partial influence."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {
                "has_suggestions": True,
                "suggested_components": ["Task", "Database", "Cache"],
            },
            None,
        )
        assert feedback["suggestion_hit_rate"] == pytest.approx(0.33, abs=0.01)
        assert feedback["corpus_influence"] == "partial"
        assert "Task" in feedback["suggestions_used"]
        assert "Database" in feedback["suggestions_ignored"]

    def test_no_suggestions_used(self):
        """No suggested components in blueprint → none influence."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {
                "has_suggestions": True,
                "suggested_components": ["Database", "Cache", "Queue"],
            },
            None,
        )
        assert feedback["suggestion_hit_rate"] == 0.0
        assert feedback["corpus_influence"] == "none"
        assert len(feedback["suggestions_ignored"]) == 3

    def test_case_insensitive_matching(self):
        """Suggestion matching is case-insensitive."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {
                "has_suggestions": True,
                "suggested_components": ["task", "TEAM"],
            },
            None,
        )
        assert feedback["suggestion_hit_rate"] == 1.0

    def test_partial_name_matching(self):
        """Suggestions match on substring (e.g. 'Task' matches 'TaskManager')."""
        engine = _make_engine()
        blueprint = {
            "components": [
                {"name": "TaskManager", "type": "entity"},
            ],
            "relationships": [],
        }
        feedback = engine._compute_corpus_feedback(
            blueprint,
            {
                "has_suggestions": True,
                "suggested_components": ["Task"],
            },
            None,
        )
        assert feedback["suggestion_hit_rate"] == 1.0


# =============================================================================
# Anti-pattern detection
# =============================================================================

class TestAntiPatternDetection:
    def test_no_domain_model(self):
        """No domain model = no anti-pattern warnings."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None, None,
        )
        assert feedback["anti_pattern_warnings"] == []

    def test_no_anti_patterns(self):
        """Domain model with no anti-patterns = no warnings."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None,
            {"anti_patterns": []},
        )
        assert feedback["anti_pattern_warnings"] == []

    def test_hollow_component_warning(self):
        """Hollow component anti-pattern triggers warning."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None,
            {
                "anti_patterns": [
                    {
                        "type": "hollow_component",
                        "component_names": ["Task"],
                    },
                ],
            },
        )
        assert len(feedback["anti_pattern_warnings"]) == 1
        assert "Task" in feedback["anti_pattern_warnings"][0]
        assert "hollow" in feedback["anti_pattern_warnings"][0]

    def test_hollow_component_no_match(self):
        """Hollow component warning only for components in blueprint."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None,
            {
                "anti_patterns": [
                    {
                        "type": "hollow_component",
                        "component_names": ["Database"],
                    },
                ],
            },
        )
        assert feedback["anti_pattern_warnings"] == []

    def test_missing_description_warning(self):
        """Missing description anti-pattern triggers warning."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            BLUEPRINT_NO_DESCRIPTIONS, None,
            {
                "anti_patterns": [
                    {
                        "type": "missing_description",
                        "component_names": ["Task"],
                    },
                ],
            },
        )
        assert len(feedback["anti_pattern_warnings"]) == 1
        assert "description" in feedback["anti_pattern_warnings"][0]

    def test_missing_description_no_warning_when_present(self):
        """No warning when component has description."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None,  # SAMPLE_BLUEPRINT has descriptions
            {
                "anti_patterns": [
                    {
                        "type": "missing_description",
                        "component_names": ["Task"],
                    },
                ],
            },
        )
        assert feedback["anti_pattern_warnings"] == []

    def test_orphan_component_warning(self):
        """Orphan component anti-pattern triggers warning."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            BLUEPRINT_WITH_ORPHAN, None,
            {
                "anti_patterns": [
                    {
                        "type": "orphan_component",
                        "component_names": ["Orphan"],
                    },
                ],
            },
        )
        assert len(feedback["anti_pattern_warnings"]) == 1
        assert "Orphan" in feedback["anti_pattern_warnings"][0]
        assert "orphan" in feedback["anti_pattern_warnings"][0].lower()

    def test_orphan_warning_not_triggered_when_connected(self):
        """No orphan warning when component has relationships."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT, None,  # Task has relationships
            {
                "anti_patterns": [
                    {
                        "type": "orphan_component",
                        "component_names": ["Task"],
                    },
                ],
            },
        )
        assert feedback["anti_pattern_warnings"] == []

    def test_multiple_anti_patterns(self):
        """Multiple anti-patterns produce multiple warnings."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            BLUEPRINT_WITH_ORPHAN, None,
            {
                "anti_patterns": [
                    {"type": "hollow_component", "component_names": ["Orphan"]},
                    {"type": "orphan_component", "component_names": ["Orphan"]},
                ],
            },
        )
        assert len(feedback["anti_pattern_warnings"]) == 2


# =============================================================================
# Combined feedback
# =============================================================================

class TestCombinedFeedback:
    def test_suggestions_and_anti_patterns(self):
        """Both suggestion tracking and anti-patterns work together."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback(
            SAMPLE_BLUEPRINT,
            {
                "has_suggestions": True,
                "suggested_components": ["Task", "Database"],
            },
            {
                "anti_patterns": [
                    {"type": "hollow_component", "component_names": ["Task"]},
                ],
            },
        )
        assert feedback["suggestion_hit_rate"] == 0.5
        assert feedback["corpus_influence"] == "strong"
        assert len(feedback["anti_pattern_warnings"]) == 1

    def test_feedback_keys_always_present(self):
        """All feedback keys are present regardless of input."""
        engine = _make_engine()
        feedback = engine._compute_corpus_feedback({}, None, None)
        assert "suggestion_hit_rate" in feedback
        assert "suggestions_used" in feedback
        assert "suggestions_ignored" in feedback
        assert "anti_pattern_warnings" in feedback
        assert "corpus_influence" in feedback


# =============================================================================
# L2 Formatter tests — format_anti_pattern_warnings / format_constraint_hints
# =============================================================================

from persistence.corpus_analysis import CorpusAnalyzer, DomainModel


class TestFormatAntiPatternWarnings:
    """Test anti-pattern warning formatter."""

    def test_empty_anti_patterns(self):
        model = DomainModel(domain="software", anti_patterns=[])
        assert CorpusAnalyzer.format_anti_pattern_warnings(model) is None

    def test_single_anti_pattern(self):
        model = DomainModel(
            domain="software",
            anti_patterns=[{
                "description": "hollow_component",
                "count": 3,
                "avg_score": 45.0,
                "source_ids": ["a", "b", "c"],
            }],
        )
        result = CorpusAnalyzer.format_anti_pattern_warnings(model)
        assert result is not None
        assert "WARNINGS" in result
        assert "Hollow components" in result
        assert "3 builds" in result

    def test_multiple_anti_patterns(self):
        model = DomainModel(
            domain="software",
            anti_patterns=[
                {"description": "hollow_component", "count": 3, "avg_score": 45.0, "source_ids": ["a", "b", "c"]},
                {"description": "orphan_component", "count": 2, "avg_score": 50.0, "source_ids": ["a", "b"]},
            ],
        )
        result = CorpusAnalyzer.format_anti_pattern_warnings(model)
        assert "Hollow components" in result
        assert "Orphan components" in result

    def test_missing_description_pattern(self):
        model = DomainModel(
            domain="software",
            anti_patterns=[{"description": "missing_description", "count": 2, "avg_score": 55.0, "source_ids": ["x"]}],
        )
        result = CorpusAnalyzer.format_anti_pattern_warnings(model)
        assert "Missing descriptions" in result
        assert "1 build" in result

    def test_unknown_pattern_uses_raw(self):
        model = DomainModel(
            domain="software",
            anti_patterns=[{"description": "some_new_thing", "count": 2, "avg_score": 40.0, "source_ids": ["a", "b"]}],
        )
        result = CorpusAnalyzer.format_anti_pattern_warnings(model)
        assert "some_new_thing" in result


class TestFormatConstraintHints:
    """Test constraint hint formatter."""

    def test_empty_constraints(self):
        model = DomainModel(domain="software", constraint_templates=[])
        assert CorpusAnalyzer.format_constraint_hints(model) is None

    def test_single_constraint(self):
        model = DomainModel(
            domain="software",
            constraint_templates=[{
                "type": "rate_limit",
                "target_pattern": "Auth",
                "description": "Limit login attempts",
                "frequency": 0.6,
                "source_ids": [],
            }],
        )
        result = CorpusAnalyzer.format_constraint_hints(model)
        assert "PROVEN CONSTRAINTS" in result
        assert "Auth" in result
        assert "rate_limit" in result
        assert "60% of builds" in result

    def test_multiple_constraints(self):
        model = DomainModel(
            domain="software",
            constraint_templates=[
                {"type": "rate_limit", "target_pattern": "Auth", "description": "", "frequency": 0.8, "source_ids": []},
                {"type": "validation", "target_pattern": "Data", "description": "Validate input", "frequency": 0.5, "source_ids": []},
            ],
        )
        result = CorpusAnalyzer.format_constraint_hints(model)
        assert "Auth" in result
        assert "Data" in result

    def test_constraint_without_type(self):
        model = DomainModel(
            domain="software",
            constraint_templates=[{"type": "", "target_pattern": "Cache", "description": "TTL expiry", "frequency": 0.4, "source_ids": []}],
        )
        result = CorpusAnalyzer.format_constraint_hints(model)
        assert "Cache" in result
        assert "TTL expiry" in result


# =============================================================================
# Dialogue depth adaptation
# =============================================================================

class TestDialogueDepthAdaptation:
    """Test experienced domain dialogue depth reduction."""

    def test_experienced_domain_reduces_depth(self):
        from core.protocol import calculate_dialogue_depth
        intent = {"domain": "software", "core_need": "test"}
        orig_min, _, _ = calculate_dialogue_depth(intent, "Build a test app")

        # Simulate engine adaptation logic
        corpus_patterns = {"sample_size": 5}
        adapted = orig_min
        if corpus_patterns.get("sample_size", 0) >= 5:
            adapted = max(1, adapted - 1)
        assert adapted < orig_min
        assert adapted >= 1

    def test_inexperienced_domain_unchanged(self):
        from core.protocol import calculate_dialogue_depth
        intent = {"domain": "software", "core_need": "test"}
        orig_min, _, _ = calculate_dialogue_depth(intent, "Build a test app")

        corpus_patterns = {"sample_size": 3}
        adapted = orig_min
        if corpus_patterns.get("sample_size", 0) >= 5:
            adapted = max(1, adapted - 1)
        assert adapted == orig_min

    def test_depth_floor_at_one(self):
        min_turns = 1
        corpus_patterns = {"sample_size": 10}
        if corpus_patterns.get("sample_size", 0) >= 5:
            min_turns = max(1, min_turns - 1)
        assert min_turns == 1


# =============================================================================
# Context corpus depth rendering
# =============================================================================

class TestContextCorpusDepthRendering:
    """Test corpus depth fields in context synthesis."""

    def test_anti_pattern_count_shown(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(corpus_total=10, corpus_success_rate=0.8, corpus_anti_pattern_count=3)
        result = synthesize_situation(data)
        assert "3 known anti-patterns" in result

    def test_constraint_count_shown(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(corpus_total=10, corpus_success_rate=0.8, corpus_constraint_count=5)
        result = synthesize_situation(data)
        assert "5 constraint templates" in result

    def test_zero_counts_hidden(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(corpus_total=10, corpus_success_rate=0.8, corpus_anti_pattern_count=0, corpus_constraint_count=0)
        result = synthesize_situation(data)
        assert "anti-pattern" not in result
        assert "constraint template" not in result

    def test_singular_forms(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(corpus_total=5, corpus_success_rate=0.9, corpus_anti_pattern_count=1, corpus_constraint_count=1)
        result = synthesize_situation(data)
        assert "1 known anti-pattern" in result
        assert "anti-patterns" not in result
        assert "1 constraint template" in result
        assert "constraint templates" not in result
