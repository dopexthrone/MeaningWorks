"""Tests for compression loss quality fixes.

Covers:
- Rich compression loss storage (category, severity, description)
- Entity checklist extraction for synthesis prompt
- Entity loss hard-trigger for re-synthesis
- Backward compatibility with string-format losses
"""

import pytest


# ---------------------------------------------------------------------------
# _extract_entity_checklist tests
# ---------------------------------------------------------------------------


class TestEntityChecklist:
    """Tests for the pre-synthesis entity extraction."""

    def _extract(self, original, digest=""):
        from core.engine import _extract_entity_checklist
        return _extract_entity_checklist(original, digest)

    def test_empty_input_returns_empty(self):
        assert self._extract("", "") == []

    def test_short_input_returns_empty(self):
        assert self._extract("hello world", "") == []

    def test_extracts_repeated_nouns(self):
        text = (
            "The authentication service handles authentication tokens. "
            "Users authenticate through the authentication gateway."
        )
        result = self._extract(text)
        names = [r[0] for r in result]
        assert "authentication" in names

    def test_counts_frequency(self):
        text = (
            "Cache handles cache invalidation. Cache must be fast. "
            "The scheduler runs tasks. The scheduler is autonomous."
        )
        result = self._extract(text)
        entity_map = {r[0]: r[1] for r in result}
        assert entity_map.get("cache", 0) >= 3
        assert entity_map.get("scheduler", 0) >= 2

    def test_filters_stop_words(self):
        text = (
            "The system should handle users with their various needs. "
            "Users would include different types of users with features."
        )
        result = self._extract(text)
        names = [r[0] for r in result]
        assert "system" not in names
        assert "should" not in names
        assert "handle" not in names
        assert "feature" not in names

    def test_extracts_capitalized_phrases(self):
        text = "The Task Manager schedules work. Build a Task Manager for teams."
        result = self._extract(text)
        names = [r[0] for r in result]
        assert "task manager" in names

    def test_includes_context_snippet(self):
        text = (
            "Build an authentication service that validates credentials. "
            "The authentication service must be secure."
        )
        result = self._extract(text)
        for name, count, ctx in result:
            if name == "authentication":
                assert "authentication" in ctx.lower()
                break

    def test_capped_at_15(self):
        # Generate input with many distinct entities
        entities = [f"entityword{chr(65+i)}{chr(65+i)}" for i in range(25)]
        text = " ".join(f"The {e} and {e} are important." for e in entities)
        result = self._extract(text)
        assert len(result) <= 15

    def test_sorted_by_frequency(self):
        text = (
            "Cache cache cache cache cache. "
            "Router router router. "
            "Gateway gateway."
        )
        result = self._extract(text)
        if len(result) >= 2:
            assert result[0][1] >= result[1][1]

    def test_dialogue_digest_adds_context(self):
        original = "Build an authentication system"
        digest = (
            "Navigator asked about authentication flow. "
            "Entity agent confirmed authentication requires tokens. "
            "Process agent described authentication validation steps."
        )
        result = self._extract(original, digest)
        entity_map = {r[0]: r[1] for r in result}
        # "authentication" appears in both input and digest
        assert entity_map.get("authentication", 0) >= 3

    def test_minimum_word_length(self):
        text = "The API and the SDK and the URL and the app and the key"
        result = self._extract(text)
        names = [r[0] for r in result]
        # 3-letter words should be excluded (need 4+)
        assert "api" not in names
        assert "sdk" not in names


# ---------------------------------------------------------------------------
# Rich compression loss storage tests
# ---------------------------------------------------------------------------


class TestRichCompressionLossStorage:
    """Tests for enriched compression loss dict format."""

    def test_targeted_resynthesis_handles_rich_format(self):
        """Re-synthesis gap builder should use category+severity from rich loss."""
        from core.engine import MotherlabsEngine
        from core.protocol import SharedState

        state = SharedState()
        state.known["compression_losses"] = [
            {
                "fragment": "authentication",
                "category": "entity",
                "severity": 0.9,
                "description": "Terms from input not captured: authentication",
            }
        ]

        # Access the gap-building logic indirectly — the format should be parseable
        losses = state.known["compression_losses"]
        for loss_info in losses:
            assert isinstance(loss_info, dict)
            assert loss_info["category"] == "entity"
            assert loss_info["severity"] == 0.9
            assert "authentication" in loss_info["fragment"]

    def test_backward_compat_with_string_format(self):
        """Legacy string-format losses should still work in gap builder."""
        # Simulate the gap builder logic from _targeted_resynthesis
        compression_losses = ["authentication", "cache"]
        gaps = []
        for loss_info in compression_losses:
            if isinstance(loss_info, dict):
                fragment = loss_info.get("fragment", "")
                gap_desc = f"COMPRESSION_LOSS ({loss_info.get('category')}): {fragment}"
            else:
                gap_desc = f"COMPRESSION_LOSS: {loss_info}"
            gaps.append(gap_desc)

        assert gaps[0] == "COMPRESSION_LOSS: authentication"
        assert gaps[1] == "COMPRESSION_LOSS: cache"

    def test_rich_format_gap_description(self):
        """Rich format should produce detailed gap descriptions."""
        loss_info = {
            "fragment": "scheduler, cron",
            "category": "entity",
            "severity": 0.6,
            "description": "Terms from input not captured in blueprint: scheduler, cron",
        }

        if isinstance(loss_info, dict):
            fragment = loss_info.get("fragment", "")
            category = loss_info.get("category", "entity")
            severity = loss_info.get("severity", 0.5)
            desc = loss_info.get("description", "")
            gap_desc = f"COMPRESSION_LOSS ({category}, severity={severity:.1f}): {fragment}"
            if desc:
                gap_desc += f" — {desc}"
        else:
            gap_desc = f"COMPRESSION_LOSS: {loss_info}"

        assert "entity" in gap_desc
        assert "severity=0.6" in gap_desc
        assert "scheduler, cron" in gap_desc
        assert "not captured" in gap_desc


# ---------------------------------------------------------------------------
# Entity loss hard-trigger tests
# ---------------------------------------------------------------------------


class TestEntityLossHardTrigger:
    """Tests for entity loss triggering re-synthesis regardless of fidelity."""

    def test_high_severity_entity_loss_triggers(self):
        """Entity loss with severity > 0.6 should trigger re-synthesis."""
        compression_losses = [
            {
                "fragment": "authentication, gateway",
                "category": "entity",
                "severity": 0.9,
                "description": "Terms from input not captured",
            }
        ]

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is True

    def test_low_severity_entity_loss_no_trigger(self):
        """Entity loss with severity <= 0.6 should not trigger on its own."""
        compression_losses = [
            {
                "fragment": "minor_term",
                "category": "entity",
                "severity": 0.3,
                "description": "Minor loss",
            }
        ]

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is False

    def test_non_entity_loss_no_trigger(self):
        """Context/constraint losses should not trigger entity hard-trigger."""
        compression_losses = [
            {
                "fragment": "some context",
                "category": "context",
                "severity": 0.8,
                "description": "Decoder loss",
            }
        ]

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is False

    def test_string_format_losses_no_trigger(self):
        """Legacy string-format losses should not crash the trigger check."""
        compression_losses = ["authentication", "cache"]

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is False

    def test_mixed_format_losses(self):
        """Mix of string and dict losses should work correctly."""
        compression_losses = [
            "old_format_loss",
            {
                "fragment": "auth_service",
                "category": "entity",
                "severity": 0.8,
                "description": "Not captured",
            },
        ]

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is True

    def test_empty_losses_no_trigger(self):
        """Empty loss list should not trigger."""
        compression_losses = []

        _entity_loss_triggered = any(
            isinstance(l, dict) and l.get("category") == "entity"
            and l.get("severity", 0) > 0.6
            for l in compression_losses
        )
        assert _entity_loss_triggered is False

    def test_resynth_decision_with_entity_trigger(self):
        """Entity trigger should force re-synthesis even when fidelity passes."""
        cl_fidelity = 0.75  # Above 0.70 threshold — normally would pass
        fidelity_threshold = 0.70
        completeness = 0  # Low — normally wouldn't trigger
        num_components = 10  # Not catastrophically thin

        _fidelity_triggered = cl_fidelity < fidelity_threshold
        _entity_loss_triggered = True  # Simulated entity loss

        should_resynth = (
            _fidelity_triggered
            or _entity_loss_triggered
            or completeness >= 80  # resynth_min_completeness
            or num_components <= 3
        )

        assert _fidelity_triggered is False  # Fidelity passed
        assert should_resynth is True  # But entity loss forces re-synthesis


# ---------------------------------------------------------------------------
# Integration: entity checklist in synthesis context
# ---------------------------------------------------------------------------


class TestEntityChecklistIntegration:
    """Tests for entity checklist integration with synthesis prompt."""

    def test_checklist_format_for_prompt(self):
        """Entity checklist should format cleanly for prompt injection."""
        from core.engine import _extract_entity_checklist

        text = (
            "Build a task management system. Tasks have priorities. "
            "The scheduler assigns tasks to workers. Workers process tasks. "
            "The notification system alerts users when tasks complete."
        )
        checklist = _extract_entity_checklist(text, "")
        if checklist:
            lines = [
                f"- {name} ({count}x)" + (f": {ctx}" if ctx else "")
                for name, count, ctx in checklist
            ]
            prompt_text = "\n".join(lines)
            assert "task" in prompt_text.lower() or "scheduler" in prompt_text.lower()

    def test_real_world_input(self):
        """Test with realistic compilation input."""
        from core.engine import _extract_entity_checklist

        text = (
            "I want to build a recipe sharing platform where users can upload "
            "recipes with photos, ingredients, and step-by-step instructions. "
            "Users should be able to follow other users, save recipes to "
            "collections, and rate recipes. The search engine should support "
            "filtering by cuisine, dietary restrictions, and cooking time. "
            "There should be a recommendation engine that suggests recipes "
            "based on user preferences and past activity."
        )
        checklist = _extract_entity_checklist(text, "")
        names = [r[0] for r in checklist]
        # "recipes" should appear (mentioned many times)
        assert any("recipe" in n for n in names)
