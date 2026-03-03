"""Tests for mother/client_docs.py.

Covers ClientBrief, generate_client_brief(), format_client_markdown(),
and internal helpers.
"""

import pytest
from mother.client_docs import (
    ClientBrief,
    generate_client_brief,
    format_client_markdown,
    _extract_components,
    _extract_relationships,
    _extract_constraints,
    _extract_acceptance_criteria,
    _infer_title,
    _build_executive_summary,
    _extract_dimension_highlights,
)


# === Sample data ===

def _sample_blueprint():
    return {
        "components": [
            {"name": "AuthService", "type": "service", "description": "Handles user auth"},
            {"name": "UserDB", "type": "entity", "description": "User data store"},
            {"name": "RateLimiter", "type": "constraint", "description": "Max 100 req/s",
             "validation_rules": ["Must enforce rate limits", "Must log violations"]},
        ],
        "relationships": [
            {"from": "AuthService", "to": "UserDB", "type": "depends_on",
             "description": "Reads user credentials"},
            {"from": "RateLimiter", "to": "AuthService", "type": "constrains"},
        ],
        "constraints": [
            {"description": "Response time under 200ms"},
            {"description": "99.9% uptime SLA"},
        ],
    }


def _sample_verification():
    return {
        "overall_score": 85.0,
        "completeness": 90.0,
        "consistency": 82.0,
        "coherence": {"score": 88.0},
        "traceability": 45.0,
        "actionability": 75.0,
    }


# === ClientBrief dataclass ===

class TestClientBrief:
    def test_frozen(self):
        brief = ClientBrief(
            title="Test",
            executive_summary="Summary",
            components=[],
            relationships=[],
            acceptance_criteria=[],
            constraints=[],
            trust_score=80.0,
            dimension_highlights=[],
        )
        with pytest.raises(AttributeError):
            brief.title = "Changed"

    def test_default_generated_at(self):
        brief = ClientBrief(
            title="T", executive_summary="S",
            components=[], relationships=[],
            acceptance_criteria=[], constraints=[],
            trust_score=0.0, dimension_highlights=[],
        )
        assert brief.generated_at == ""


# === generate_client_brief ===

class TestGenerateClientBrief:
    def test_basic_generation(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        assert isinstance(brief, ClientBrief)
        assert len(brief.components) == 3
        assert len(brief.relationships) == 2
        assert len(brief.constraints) == 2

    def test_with_project_name(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp, project_name="My Project")
        assert brief.title == "My Project"

    def test_without_project_name_infers_title(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        assert brief.title  # non-empty

    def test_with_verification(self):
        bp = _sample_blueprint()
        ver = _sample_verification()
        brief = generate_client_brief(bp, verification=ver)
        assert brief.trust_score == 85.0
        assert len(brief.dimension_highlights) > 0

    def test_without_verification(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        assert brief.trust_score == 0.0
        assert brief.dimension_highlights == []

    def test_empty_blueprint(self):
        brief = generate_client_brief({})
        assert brief.components == []
        assert brief.relationships == []

    def test_acceptance_criteria_from_validation_rules(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        assert len(brief.acceptance_criteria) >= 2
        assert "Must enforce rate limits" in brief.acceptance_criteria

    def test_generated_at_populated(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        assert brief.generated_at  # non-empty


# === format_client_markdown ===

class TestFormatClientMarkdown:
    def test_contains_title(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp, project_name="Auth System")
        md = format_client_markdown(brief)
        assert "# Auth System" in md

    def test_contains_executive_summary(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Executive Summary" in md
        assert "3 component" in md

    def test_contains_components_table(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "System Components" in md
        assert "AuthService" in md
        assert "UserDB" in md

    def test_contains_relationships(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Component Interactions" in md
        assert "AuthService" in md

    def test_contains_constraints(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Design Constraints" in md
        assert "200ms" in md

    def test_contains_acceptance_criteria(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Acceptance Criteria" in md
        assert "- [ ]" in md

    def test_quality_assessment_with_verification(self):
        bp = _sample_blueprint()
        ver = _sample_verification()
        brief = generate_client_brief(bp, verification=ver)
        md = format_client_markdown(brief)
        assert "Quality Assessment" in md
        assert "85%" in md

    def test_no_quality_without_verification(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Quality Assessment" not in md

    def test_generated_at_shown(self):
        bp = _sample_blueprint()
        brief = generate_client_brief(bp)
        md = format_client_markdown(brief)
        assert "Generated:" in md


# === Internal helpers ===

class TestExtractComponents:
    def test_extracts_name_type_description(self):
        bp = _sample_blueprint()
        comps = _extract_components(bp)
        assert len(comps) == 3
        assert comps[0]["name"] == "AuthService"
        assert comps[0]["type"] == "service"
        assert comps[0]["description"] == "Handles user auth"

    def test_empty_blueprint(self):
        assert _extract_components({}) == []

    def test_fallback_to_derived_from(self):
        bp = {"components": [{"name": "X", "type": "t", "derived_from": "user said so"}]}
        comps = _extract_components(bp)
        assert comps[0]["description"] == "user said so"


class TestExtractRelationships:
    def test_human_readable_format(self):
        bp = _sample_blueprint()
        rels = _extract_relationships(bp)
        assert len(rels) == 2
        assert "AuthService" in rels[0]
        assert "UserDB" in rels[0]
        assert "depends_on" in rels[0]

    def test_includes_description(self):
        bp = _sample_blueprint()
        rels = _extract_relationships(bp)
        assert "Reads user credentials" in rels[0]

    def test_empty_blueprint(self):
        assert _extract_relationships({}) == []


class TestExtractConstraints:
    def test_extracts_descriptions(self):
        bp = _sample_blueprint()
        constraints = _extract_constraints(bp)
        assert "Response time under 200ms" in constraints
        assert "99.9% uptime SLA" in constraints

    def test_skips_empty_descriptions(self):
        bp = {"constraints": [{"description": ""}, {"description": "Valid"}]}
        constraints = _extract_constraints(bp)
        assert constraints == ["Valid"]


class TestInferTitle:
    def test_subsystem_preferred(self):
        bp = {"components": [
            {"name": "Main", "type": "component"},
            {"name": "CoreEngine", "type": "subsystem"},
        ]}
        title = _infer_title(bp)
        assert "CoreEngine" in title

    def test_falls_back_to_first(self):
        bp = {"components": [{"name": "Widget", "type": "component"}]}
        title = _infer_title(bp)
        assert "Widget" in title

    def test_empty_components(self):
        title = _infer_title({})
        assert title == "System Specification"


class TestBuildExecutiveSummary:
    def test_includes_counts(self):
        summary = _build_executive_summary("Test", 5, 3, 0.0, [])
        assert "5 component" in summary
        assert "3 interaction" in summary

    def test_high_trust_message(self):
        summary = _build_executive_summary("Test", 1, 1, 85.0, [])
        assert "high confidence" in summary

    def test_moderate_trust_message(self):
        summary = _build_executive_summary("Test", 1, 1, 65.0, [])
        assert "moderate confidence" in summary

    def test_low_trust_message(self):
        summary = _build_executive_summary("Test", 1, 1, 30.0, [])
        assert "refinement" in summary

    def test_constraints_mentioned(self):
        summary = _build_executive_summary("Test", 1, 1, 0.0, ["c1", "c2"])
        assert "2 constraint" in summary


class TestExtractDimensionHighlights:
    def test_strong_dimension(self):
        ver = {"completeness": 92.0}
        highlights = _extract_dimension_highlights(ver)
        assert any("Strong" in h for h in highlights)

    def test_weak_dimension(self):
        ver = {"traceability": 45.0}
        highlights = _extract_dimension_highlights(ver)
        assert any("Needs work" in h for h in highlights)

    def test_dict_score_format(self):
        ver = {"coherence": {"score": 95.0}}
        highlights = _extract_dimension_highlights(ver)
        assert any("Strong" in h and "coherence" in h.lower() for h in highlights)

    def test_unknown_dimension_ignored(self):
        ver = {"unknown_field": 99.0}
        highlights = _extract_dimension_highlights(ver)
        assert len(highlights) == 0
