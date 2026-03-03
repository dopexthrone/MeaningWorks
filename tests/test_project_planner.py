"""Tests for mother/project_planner.py — blueprint→code engine prompt translation.

Pure function tests. No I/O, no mocking needed.
"""

import os
import pytest

from mother.project_planner import (
    ProjectBuildSpec,
    infer_project_name,
    extract_language_and_framework,
    blueprint_to_project_context,
    build_project_system_prompt,
    assemble_project_build_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_blueprint(
    domain="task-management",
    core_need="A task manager with auth and notifications",
    components=None,
    relationships=None,
    **kwargs,
):
    bp = {"domain": domain, "core_need": core_need}
    if components is not None:
        bp["components"] = components
    if relationships is not None:
        bp["relationships"] = relationships
    bp.update(kwargs)
    return bp


def _make_verification(overall_score=85.0, **kwargs):
    v = {"overall_score": overall_score}
    v.update(kwargs)
    return v


# ---------------------------------------------------------------------------
# infer_project_name
# ---------------------------------------------------------------------------

class TestInferProjectName:
    def test_from_core_need(self):
        bp = _make_blueprint(core_need="Task manager with auth")
        name = infer_project_name(bp)
        assert name == "task-manager-with-auth"

    def test_from_domain(self):
        bp = _make_blueprint(core_need="", domain="E-Commerce Platform")
        name = infer_project_name(bp)
        assert name == "ecommerce-platform"

    def test_empty_blueprint(self):
        assert infer_project_name({}) == "project"

    def test_special_chars_stripped(self):
        bp = _make_blueprint(core_need="Build a TODO app (v2)!")
        name = infer_project_name(bp)
        assert "(" not in name
        assert ")" not in name
        assert "!" not in name

    def test_caps_at_four_words(self):
        bp = _make_blueprint(
            core_need="A very long project name that exceeds the word limit"
        )
        name = infer_project_name(bp)
        words = name.split("-")
        assert len(words) <= 4

    def test_none_values(self):
        bp = {"core_need": None, "domain": None}
        assert infer_project_name(bp) == "project"

    def test_whitespace_only(self):
        bp = {"core_need": "   ", "domain": "  "}
        assert infer_project_name(bp) == "project"

    def test_description_fallback(self):
        bp = {"description": "Simple Chat Application"}
        name = infer_project_name(bp)
        assert "chat" in name


# ---------------------------------------------------------------------------
# extract_language_and_framework
# ---------------------------------------------------------------------------

class TestExtractLanguageAndFramework:
    def test_default_python(self):
        lang, fw = extract_language_and_framework({})
        assert lang == "python"
        assert fw == ()

    def test_react_detection(self):
        bp = _make_blueprint(
            components=[
                {"name": "UserDashboard", "description": "React component for user dashboard"}
            ]
        )
        lang, fw = extract_language_and_framework(bp)
        assert lang == "typescript"
        assert "react" in fw

    def test_express_detection(self):
        bp = _make_blueprint(
            description="Express.js REST API with MongoDB"
        )
        lang, fw = extract_language_and_framework(bp)
        assert "express" in fw

    def test_flask_detection(self):
        bp = _make_blueprint(
            core_need="Flask web application with SQLAlchemy"
        )
        lang, fw = extract_language_and_framework(bp)
        assert lang == "python"
        assert "flask" in fw

    def test_fastapi_detection(self):
        bp = _make_blueprint(
            description="FastAPI backend with async endpoints"
        )
        lang, fw = extract_language_and_framework(bp)
        assert lang == "python"
        assert "fastapi" in fw

    def test_rust_detection(self):
        bp = _make_blueprint(
            description="High-performance Rust web server using Axum"
        )
        lang, fw = extract_language_and_framework(bp)
        assert lang == "rust"
        assert "axum" in fw

    def test_constraint_scanning(self):
        bp = _make_blueprint(
            components=[
                {"name": "API", "constraints": ["Must use Django REST Framework"]}
            ]
        )
        lang, fw = extract_language_and_framework(bp)
        assert lang == "python"
        assert "django" in fw

    def test_tech_stack_field(self):
        bp = _make_blueprint(tech_stack=["react", "typescript", "node"])
        lang, fw = extract_language_and_framework(bp)
        assert lang == "typescript"

    def test_empty_components(self):
        bp = _make_blueprint(components=[])
        lang, fw = extract_language_and_framework(bp)
        assert lang == "python"

    def test_malformed_components(self):
        bp = _make_blueprint(components=["not a dict", 42, None])
        lang, fw = extract_language_and_framework(bp)
        assert isinstance(lang, str)
        assert isinstance(fw, tuple)


# ---------------------------------------------------------------------------
# blueprint_to_project_context
# ---------------------------------------------------------------------------

class TestBlueprintToProjectContext:
    def test_empty_blueprint(self):
        result = blueprint_to_project_context({})
        assert result == ""

    def test_components_rendered(self):
        bp = _make_blueprint(
            components=[
                {
                    "name": "AuthService",
                    "description": "Handles user authentication",
                    "methods": [{"name": "login"}, {"name": "logout"}],
                }
            ]
        )
        ctx = blueprint_to_project_context(bp)
        assert "AuthService" in ctx
        assert "authentication" in ctx
        assert "login" in ctx
        assert "logout" in ctx

    def test_relationships_rendered(self):
        bp = _make_blueprint(
            relationships=[
                {"source": "AuthService", "target": "Database", "type": "uses"}
            ]
        )
        ctx = blueprint_to_project_context(bp)
        assert "AuthService" in ctx
        assert "Database" in ctx
        assert "uses" in ctx

    def test_max_cap(self):
        comps = [{"name": f"Component{i}"} for i in range(30)]
        bp = _make_blueprint(components=comps)
        ctx = blueprint_to_project_context(bp, max_components=5)
        assert "Component0" in ctx
        assert "Component4" in ctx
        assert "Component5" not in ctx
        assert "25 more" in ctx

    def test_insights_rendered(self):
        bp = _make_blueprint(
            insights=[{"description": "Consider caching for performance"}]
        )
        ctx = blueprint_to_project_context(bp)
        assert "caching" in ctx

    def test_malformed_keys(self):
        bp = {"components": [{"name": 123, "methods": "not a list"}]}
        ctx = blueprint_to_project_context(bp)
        assert isinstance(ctx, str)

    def test_component_constraints_rendered(self):
        bp = _make_blueprint(
            components=[
                {
                    "name": "API",
                    "constraints": [
                        {"description": "Rate limit to 100 req/min"},
                        "Must support CORS",
                    ],
                }
            ]
        )
        ctx = blueprint_to_project_context(bp)
        assert "Rate limit" in ctx
        assert "CORS" in ctx

    def test_component_type_rendered(self):
        bp = _make_blueprint(
            components=[{"name": "Gateway", "type": "service"}]
        )
        ctx = blueprint_to_project_context(bp)
        assert "service" in ctx

    def test_component_relationships_rendered(self):
        bp = _make_blueprint(
            components=[
                {
                    "name": "TaskService",
                    "relationships": [
                        {"target": "Database", "type": "reads-from"}
                    ],
                }
            ]
        )
        ctx = blueprint_to_project_context(bp)
        assert "Database" in ctx
        assert "reads-from" in ctx

    def test_domain_and_core_need_included(self):
        bp = _make_blueprint(
            domain="healthcare", core_need="Patient records management"
        )
        ctx = blueprint_to_project_context(bp)
        assert "healthcare" in ctx
        assert "Patient records" in ctx


# ---------------------------------------------------------------------------
# build_project_system_prompt
# ---------------------------------------------------------------------------

class TestBuildProjectSystemPrompt:
    def test_contains_language(self):
        prompt = build_project_system_prompt("python", (), "/tmp/proj")
        assert "python" in prompt.lower()

    def test_contains_framework_hints(self):
        prompt = build_project_system_prompt(
            "typescript", ("react", "next"), "/tmp/proj"
        )
        assert "react" in prompt
        assert "next" in prompt

    def test_contains_project_dir(self):
        prompt = build_project_system_prompt("python", (), "/home/user/proj")
        assert "/home/user/proj" in prompt

    def test_no_self_modification_rules(self):
        prompt = build_project_system_prompt("python", (), "/tmp/proj")
        # Should not contain motherlabs-internal rules
        assert "bridge.py" not in prompt
        assert "persona.py" not in prompt
        assert "DomainAdapter" not in prompt

    def test_requires_entry_point(self):
        prompt = build_project_system_prompt("python", (), "/tmp/proj")
        assert "entry point" in prompt.lower()

    def test_requires_tests(self):
        prompt = build_project_system_prompt("python", (), "/tmp/proj")
        assert "test" in prompt.lower()

    def test_no_stubs(self):
        prompt = build_project_system_prompt("python", (), "/tmp/proj")
        assert "no stubs" in prompt.lower() or "no placeholder" in prompt.lower()


# ---------------------------------------------------------------------------
# assemble_project_build_spec
# ---------------------------------------------------------------------------

class TestAssembleProjectBuildSpec:
    def test_full_assembly(self):
        bp = _make_blueprint(
            components=[
                {"name": "AuthService", "description": "Handles auth"},
                {"name": "TaskAPI", "description": "Task CRUD"},
            ]
        )
        v = _make_verification(overall_score=88.0)
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")

        assert isinstance(spec, ProjectBuildSpec)
        assert spec.project_name
        assert spec.project_dir.startswith("/tmp/projects/")
        assert spec.component_count == 2
        assert spec.trust_score == 88.0
        assert spec.language == "python"
        assert "AuthService" in spec.blueprint_summary
        assert "TaskAPI" in spec.blueprint_summary
        assert len(spec.prompt) > 0
        assert len(spec.system_prompt) > 0

    def test_custom_name_override(self):
        bp = _make_blueprint()
        v = _make_verification()
        spec = assemble_project_build_spec(
            bp, v, "/tmp/projects", project_name="my-app"
        )
        assert spec.project_name == "my-app"
        assert spec.project_dir == "/tmp/projects/my-app"

    def test_custom_language_override(self):
        bp = _make_blueprint()
        v = _make_verification()
        spec = assemble_project_build_spec(
            bp, v, "/tmp/projects", language="rust"
        )
        assert spec.language == "rust"
        assert "rust" in spec.system_prompt.lower()

    def test_trust_passthrough(self):
        bp = _make_blueprint()
        v = _make_verification(overall_score=72.5)
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        assert spec.trust_score == 72.5

    def test_empty_blueprint(self):
        spec = assemble_project_build_spec({}, {}, "/tmp/projects")
        assert spec.project_name == "project"
        assert spec.component_count == 0
        assert spec.trust_score == 0.0
        assert spec.language == "python"

    def test_none_verification_values(self):
        bp = _make_blueprint()
        spec = assemble_project_build_spec(bp, {}, "/tmp/projects")
        assert spec.trust_score == 0.0

    def test_fifty_plus_components(self):
        comps = [{"name": f"Comp{i}"} for i in range(55)]
        bp = _make_blueprint(components=comps)
        v = _make_verification()
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        assert spec.component_count == 55
        # Context should be capped
        assert "more components" in spec.blueprint_summary

    def test_frozen_dataclass(self):
        bp = _make_blueprint()
        v = _make_verification()
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        with pytest.raises(AttributeError):
            spec.project_name = "new-name"

    def test_prompt_contains_core_need(self):
        bp = _make_blueprint(core_need="Real-time chat application")
        v = _make_verification()
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        assert "Real-time chat application" in spec.prompt

    def test_domain_extraction(self):
        bp = _make_blueprint(domain="fintech")
        v = _make_verification()
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        assert spec.domain == "fintech"

    def test_framework_detection_flows_through(self):
        bp = _make_blueprint(
            description="React dashboard with Express backend"
        )
        v = _make_verification()
        spec = assemble_project_build_spec(bp, v, "/tmp/projects")
        assert "react" in spec.framework_hints or "express" in spec.framework_hints
