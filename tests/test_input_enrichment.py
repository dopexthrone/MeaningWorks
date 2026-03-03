"""
Tests for core/input_enrichment.py — LEAF MODULE.

Phase 2 of Agent Ship: Input Enrichment tests.
"""

import pytest

from core.input_enrichment import (
    EnrichmentResult,
    ENRICHMENT_SYSTEM_PROMPT,
    build_enrichment_prompt,
    parse_enrichment_response,
    _strip_preamble,
)


# =============================================================================
# FROZEN DATACLASS
# =============================================================================

class TestEnrichmentResult:
    def test_frozen(self):
        result = EnrichmentResult(
            original_input="todo app",
            enriched_input="A task management application...",
            expansion_ratio=5.0,
        )
        with pytest.raises(AttributeError):
            result.enriched_input = "changed"

    def test_fields(self):
        result = EnrichmentResult(
            original_input="todo app",
            enriched_input="expanded",
            expansion_ratio=2.5,
        )
        assert result.original_input == "todo app"
        assert result.enriched_input == "expanded"
        assert result.expansion_ratio == 2.5


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

class TestSystemPrompt:
    def test_contains_critical_instructions(self):
        assert "do not add features" in ENRICHMENT_SYSTEM_PROMPT.lower()
        assert "depth" in ENRICHMENT_SYSTEM_PROMPT.lower()
        assert "breadth" in ENRICHMENT_SYSTEM_PROMPT.lower()

    def test_contains_expansion_guidance(self):
        assert "actors" in ENRICHMENT_SYSTEM_PROMPT.lower()
        assert "actions" in ENRICHMENT_SYSTEM_PROMPT.lower()
        assert "constraints" in ENRICHMENT_SYSTEM_PROMPT.lower()


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

class TestBuildEnrichmentPrompt:
    def test_contains_original_input(self):
        prompt = build_enrichment_prompt("todo app with teams")
        assert "todo app with teams" in prompt

    def test_contains_instructions(self):
        prompt = build_enrichment_prompt("build a store")
        assert "actors" in prompt.lower()
        assert "actions" in prompt.lower()
        assert "enriched" in prompt.lower()

    def test_strips_whitespace(self):
        prompt = build_enrichment_prompt("  todo app  ")
        assert '"todo app"' in prompt

    def test_returns_string(self):
        result = build_enrichment_prompt("test")
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# RESPONSE PARSING
# =============================================================================

class TestParseEnrichmentResponse:
    def test_normal_response(self):
        original = "todo app"
        enriched_text = (
            "A task management application designed for small teams. "
            "Users can create tasks, assign them to team members, set deadlines, "
            "and track progress. The system supports notifications when deadlines "
            "approach. Key actors include team leads, team members, and administrators. "
            "Similar to Trello or Asana in scope but focused on simplicity."
        )
        result = parse_enrichment_response(enriched_text, original)
        assert result.original_input == original
        assert result.enriched_input == enriched_text
        assert result.expansion_ratio > 1.0

    def test_empty_response_fallback(self):
        result = parse_enrichment_response("", "todo app")
        assert result.enriched_input == "todo app"
        assert result.expansion_ratio == 1.0

    def test_none_response_fallback(self):
        result = parse_enrichment_response(None, "todo app")
        assert result.enriched_input == "todo app"
        assert result.expansion_ratio == 1.0

    def test_too_short_response_fallback(self):
        result = parse_enrichment_response("ok", "todo app with teams and deadlines")
        assert result.enriched_input == "todo app with teams and deadlines"
        assert result.expansion_ratio == 1.0

    def test_strips_preamble(self):
        response = "Here is the enriched description:\nA detailed task management system for teams with deadlines and assignments and notifications and more."
        result = parse_enrichment_response(response, "todo")
        assert not result.enriched_input.startswith("Here is")

    def test_expansion_ratio_calculated(self):
        original = "app"
        enriched = "A comprehensive application for managing various tasks and workflows in a team environment."
        result = parse_enrichment_response(enriched, original)
        expected = round(len(enriched) / len(original), 2)
        assert result.expansion_ratio == expected


# =============================================================================
# PREAMBLE STRIPPING
# =============================================================================

class TestStripPreamble:
    def test_here_is_pattern(self):
        text = "Here is the enriched description:\nActual content here."
        assert _strip_preamble(text) == "Actual content here."

    def test_heres_pattern(self):
        text = "Here's the expanded specification:\nActual content."
        assert _strip_preamble(text) == "Actual content."

    def test_enriched_description_pattern(self):
        text = "Enriched Description:\nActual content."
        assert _strip_preamble(text) == "Actual content."

    def test_no_preamble(self):
        text = "A task management application for teams."
        assert _strip_preamble(text) == text

    def test_sure_here_is(self):
        text = "Sure! Here is the enriched description:\nContent."
        assert _strip_preamble(text) == "Content."


# =============================================================================
# LEAF MODULE CONSTRAINT
# =============================================================================

class TestLeafModuleConstraint:
    def test_no_engine_imports(self):
        import core.input_enrichment as mod
        source_file = mod.__file__
        with open(source_file) as f:
            source = f.read()
        for forbidden in ["from core.engine", "from core.protocol",
                          "from core.pipeline", "from core.llm"]:
            assert forbidden not in source, f"LEAF MODULE violated: found '{forbidden}'"

    def test_stdlib_only_imports(self):
        import core.input_enrichment as mod
        source_file = mod.__file__
        with open(source_file) as f:
            source = f.read()
        import re as _re
        imports = _re.findall(r'^(?:from|import)\s+(\S+)', source, _re.MULTILINE)
        allowed_prefixes = {"re", "dataclasses", "typing"}
        for imp in imports:
            top = imp.split('.')[0]
            assert top in allowed_prefixes, f"Non-stdlib import: {imp}"
