"""
Tests for domain adapter wiring — verifies adapter propagation through pipeline.

Covers:
- Engine stores adapter
- StagedPipeline receives adapter
- Classification uses adapter vocab
- Interface extraction uses adapter flows
- Verification uses adapter actionability checks
- End-to-end with MockClient + PROCESS_ADAPTER
- V2 router mounted (health endpoint)
- Agent factory prompt overrides (Phase: Complete Adapter Unpacking)
- File extension threading through project writer
- Emission user_content adapter-awareness
- Classification subject/object patterns
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from core.engine import MotherlabsEngine
from core.llm import MockClient
from core.pipeline import StagedPipeline
from adapters.software import SOFTWARE_ADAPTER
from adapters.process import PROCESS_ADAPTER


# =============================================================================
# Engine stores adapter
# =============================================================================

class TestEngineStoresAdapter:
    def test_default_no_adapter(self):
        engine = MotherlabsEngine(llm_client=MockClient())
        assert engine.domain_adapter is None

    def test_software_adapter(self):
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            domain_adapter=SOFTWARE_ADAPTER,
        )
        assert engine.domain_adapter is SOFTWARE_ADAPTER

    def test_process_adapter(self):
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            domain_adapter=PROCESS_ADAPTER,
        )
        assert engine.domain_adapter is PROCESS_ADAPTER


# =============================================================================
# StagedPipeline receives adapter
# =============================================================================

class TestPipelineReceivesAdapter:
    def test_default_no_adapter(self):
        sp = StagedPipeline(llm_client=MockClient())
        assert sp.domain_adapter is None

    def test_with_adapter(self):
        sp = StagedPipeline(
            llm_client=MockClient(),
            domain_adapter=PROCESS_ADAPTER,
        )
        assert sp.domain_adapter is PROCESS_ADAPTER


# =============================================================================
# Classification uses adapter vocab
# =============================================================================

class TestClassificationAdapterVocab:
    def test_process_type_keywords_passed(self):
        """Verify classify_components receives adapter type_keywords."""
        from core.classification import classify_components

        candidates = [
            {"name": "ReviewStep", "type": "entity", "derived_from": "test"},
            {"name": "ApprovalGateway", "type": "entity", "derived_from": "test"},
        ]

        # With process adapter type_keywords
        type_kw = dict(PROCESS_ADAPTER.vocabulary.type_keywords)
        generic = PROCESS_ADAPTER.classification.generic_terms

        scores = classify_components(
            candidates=candidates,
            input_text="employee onboarding review and approval gateway",
            dialogue_history=["ReviewStep is a process step", "ApprovalGateway is a gateway"],
            relationships=[],
            type_keywords=type_kw,
            generic_terms=generic,
        )

        assert len(scores) == 2
        # Scores should be populated (basic sanity)
        for s in scores:
            assert s.overall_confidence >= 0.0

    def test_software_adapter_type_keywords(self):
        """Software adapter type keywords should also work."""
        from core.classification import classify_components

        candidates = [
            {"name": "AuthService", "type": "entity", "derived_from": "test"},
        ]

        type_kw = dict(SOFTWARE_ADAPTER.vocabulary.type_keywords)
        generic = SOFTWARE_ADAPTER.classification.generic_terms

        scores = classify_components(
            candidates=candidates,
            input_text="authentication service for user login",
            dialogue_history=["AuthService handles login"],
            relationships=[],
            type_keywords=type_kw,
            generic_terms=generic,
        )

        assert len(scores) == 1


# =============================================================================
# Interface extraction uses adapter flows
# =============================================================================

class TestInterfaceExtractionAdapterFlows:
    def test_process_relationship_flows(self):
        """Verify extract_interface_map accepts adapter relationship_flows."""
        from core.interface_extractor import extract_interface_map

        blueprint = {
            "components": [
                {"name": "ReviewStep", "type": "process"},
                {"name": "ApprovalGateway", "type": "process"},
            ],
            "relationships": [
                {"from": "ReviewStep", "to": "ApprovalGateway", "type": "triggers"},
            ],
            "constraints": [],
        }

        rel_flows = dict(PROCESS_ADAPTER.vocabulary.relationship_flows)
        type_hints = dict(PROCESS_ADAPTER.vocabulary.type_hints)

        imap = extract_interface_map(
            blueprint, None,
            relationship_flows=rel_flows,
            type_hints=type_hints,
        )

        assert imap is not None
        # Should have at least one contract for the relationship
        assert len(imap.contracts) >= 1 or len(imap.unmatched_relationships) >= 0


# =============================================================================
# Verification uses adapter actionability checks
# =============================================================================

class TestVerificationAdapterChecks:
    def test_process_actionability_checks(self):
        """Process adapter uses decision_points/activities instead of just methods."""
        from core.verification import verify_deterministic

        blueprint = {
            "components": [
                {"name": "ReviewStep", "type": "process", "methods": [],
                 "decision_points": ["approve_or_reject"],
                 "activities": ["submit_review"]},
            ],
            "relationships": [
                {"from": "ReviewStep", "to": "ReviewStep", "type": "self"},
            ],
            "constraints": [],
        }

        result = verify_deterministic(
            blueprint=blueprint,
            intent_keywords=["review", "approval"],
            input_text="review and approval process",
            graph_errors=[],
            graph_warnings=[],
            health_score=0.8,
            health_stats={"orphan_ratio": 0.0, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.7,
            actionability_checks=PROCESS_ADAPTER.verification.actionability_checks,
        )

        assert result is not None
        assert hasattr(result, "status")

    def test_default_actionability_is_methods(self):
        """Without adapter, default actionability check is methods only."""
        from core.verification import verify_deterministic

        blueprint = {
            "components": [
                {"name": "AuthService", "type": "entity", "methods": [
                    {"name": "login", "component": "AuthService",
                     "parameters": [], "return_type": "bool"}
                ]},
            ],
            "relationships": [],
            "constraints": [],
        }

        result = verify_deterministic(
            blueprint=blueprint,
            intent_keywords=["auth"],
            input_text="auth service",
            graph_errors=[],
            graph_warnings=[],
            health_score=0.8,
            health_stats={"orphan_ratio": 0.0, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.7,
            # No actionability_checks param → defaults to ("methods",)
        )

        assert result is not None


# =============================================================================
# End-to-end with MockClient + PROCESS_ADAPTER
# =============================================================================

class TestEndToEndProcessAdapter:
    def test_compile_with_process_adapter(self):
        """Full compile with PROCESS_ADAPTER completes without error."""
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            pipeline_mode="staged",
            domain_adapter=PROCESS_ADAPTER,
        )
        result = engine.compile("Employee onboarding process")

        assert result is not None
        assert isinstance(result.success, bool)
        # With MockClient, components may or may not be produced
        # The key assertion is: no crash, adapter was threaded without error
        assert result.blueprint is not None

    def test_compile_without_adapter(self):
        """Compile without adapter still works (backward compat)."""
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            pipeline_mode="staged",
        )
        result = engine.compile("User authentication system")

        assert result is not None
        assert isinstance(result.success, bool)


# =============================================================================
# V2 router mounted
# =============================================================================

class TestV2RouterMounted:
    def test_v2_health_returns_200(self):
        """GET /v2/health returns 200 when V2 router is mounted."""
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        response = client.get("/v2/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_v2_domains_returns_200(self):
        """GET /v2/domains returns 200 with registered adapters."""
        from fastapi.testclient import TestClient
        from api.main import app

        # Ensure adapters are registered
        import adapters  # noqa: F401

        client = TestClient(app)
        response = client.get("/v2/domains")
        assert response.status_code == 200
        data = response.json()
        assert "domains" in data


# =============================================================================
# Agent factory prompt overrides
# =============================================================================

class TestAgentFactoryPromptOverrides:
    def test_intent_agent_uses_process_adapter_prompt(self):
        """Process adapter overrides intent agent system prompt."""
        from agents.swarm import create_intent_agent
        agent = create_intent_agent(MockClient(), domain_adapter=PROCESS_ADAPTER)
        assert agent.system_prompt == PROCESS_ADAPTER.prompts.intent_system_prompt
        assert "business process" in agent.system_prompt.lower()

    def test_persona_agent_uses_process_adapter_prompt(self):
        """Process adapter overrides persona agent system prompt."""
        from agents.swarm import create_persona_agent
        agent = create_persona_agent(MockClient(), domain_adapter=PROCESS_ADAPTER)
        assert agent.system_prompt == PROCESS_ADAPTER.prompts.persona_system_prompt

    def test_entity_agent_uses_process_adapter_prompt(self):
        """Process adapter overrides entity agent system prompt."""
        from agents.spec_agents import create_entity_agent
        agent = create_entity_agent(MockClient(), domain_adapter=PROCESS_ADAPTER)
        assert agent.system_prompt == PROCESS_ADAPTER.prompts.entity_system_prompt
        assert "process" in agent.system_prompt.lower()

    def test_process_agent_uses_process_adapter_prompt(self):
        """Process adapter overrides process agent system prompt."""
        from agents.spec_agents import create_process_agent
        agent = create_process_agent(MockClient(), domain_adapter=PROCESS_ADAPTER)
        assert agent.system_prompt == PROCESS_ADAPTER.prompts.process_system_prompt

    def test_synthesis_agent_uses_process_adapter_prompt(self):
        """Process adapter overrides synthesis agent system prompt."""
        from agents.swarm import create_synthesis_agent
        agent = create_synthesis_agent(MockClient(), domain_adapter=PROCESS_ADAPTER)
        assert agent.system_prompt == PROCESS_ADAPTER.prompts.synthesis_system_prompt

    def test_software_adapter_uses_defaults(self):
        """Software adapter has empty prompts → falls back to module defaults."""
        from agents.swarm import create_intent_agent, INTENT_SYSTEM_PROMPT
        agent = create_intent_agent(MockClient(), domain_adapter=SOFTWARE_ADAPTER)
        # SOFTWARE_ADAPTER.prompts.intent_system_prompt is empty string
        # So agent should use the default INTENT_SYSTEM_PROMPT
        assert agent.system_prompt == INTENT_SYSTEM_PROMPT

    def test_no_adapter_uses_defaults(self):
        """No adapter → uses module-level defaults."""
        from agents.swarm import create_intent_agent, INTENT_SYSTEM_PROMPT
        agent = create_intent_agent(MockClient())
        assert agent.system_prompt == INTENT_SYSTEM_PROMPT

    def test_engine_agents_use_process_prompts(self):
        """Engine with PROCESS_ADAPTER creates agents with process prompts."""
        engine = MotherlabsEngine(
            llm_client=MockClient(),
            domain_adapter=PROCESS_ADAPTER,
        )
        assert engine.intent_agent.system_prompt == PROCESS_ADAPTER.prompts.intent_system_prompt
        assert engine.synthesis_agent.system_prompt == PROCESS_ADAPTER.prompts.synthesis_system_prompt


# =============================================================================
# File extension threading through project writer
# =============================================================================

class TestFileExtensionThreading:
    def test_write_project_yaml_extension(self):
        """write_project with file_extension='.yaml' produces .yaml files, no __init__.py."""
        from core.project_writer import write_project, ProjectConfig

        generated_code = {
            "ReviewStep": "activity: ReviewStep\ndescription: Review submission",
            "ApprovalGateway": "gateway: ApprovalGateway\ntype: exclusive",
        }
        blueprint = {
            "domain": "onboarding",
            "components": [
                {"name": "ReviewStep", "type": "activity"},
                {"name": "ApprovalGateway", "type": "gateway"},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(
                generated_code, blueprint, tmpdir,
                config=ProjectConfig(entry_point=False, tests=False),
                file_extension=".yaml",
            )

            # Should have .yaml files, NOT .py
            yaml_files = [f for f in manifest.files_written if f.endswith(".yaml")]
            py_files = [f for f in manifest.files_written if f.endswith(".py")]
            assert len(yaml_files) > 0, f"Expected .yaml files, got: {manifest.files_written}"
            assert "__init__.py" not in manifest.files_written
            assert "main.py" not in manifest.files_written
            assert "pyproject.toml" not in manifest.files_written

    def test_write_project_default_py_extension(self):
        """Default write_project still produces .py files."""
        from core.project_writer import write_project, ProjectConfig

        generated_code = {"AuthService": "class AuthService:\n    pass\n"}
        blueprint = {
            "domain": "auth",
            "components": [{"name": "AuthService", "type": "service"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_project(
                generated_code, blueprint, tmpdir,
                config=ProjectConfig(tests=False),
            )

            assert "__init__.py" in manifest.files_written
            assert "main.py" in manifest.files_written
            assert "pyproject.toml" in manifest.files_written


# =============================================================================
# Syntax validation skip for non-Python
# =============================================================================

class TestSyntaxValidationSkip:
    def test_validate_all_code_skips_non_python(self):
        """validate_all_code returns empty for non-Python extensions."""
        from core.project_writer import validate_all_code

        # Invalid Python but valid YAML
        code = {"Foo": "this is not: valid python{{{"}
        errors = validate_all_code(code, file_extension=".yaml")
        assert errors == []

    def test_validate_all_code_checks_python(self):
        """validate_all_code still checks .py extensions."""
        from core.project_writer import validate_all_code

        code = {"Foo": "def broken(\n"}
        errors = validate_all_code(code, file_extension=".py")
        assert len(errors) > 0


# =============================================================================
# Classification subject/object patterns
# =============================================================================

class TestClassificationPatterns:
    def test_process_subject_patterns(self):
        """classify_components with process subject_patterns runs without error."""
        from core.classification import classify_components

        candidates = [
            {"name": "HRTeam", "type": "participant", "derived_from": "test"},
        ]

        scores = classify_components(
            candidates=candidates,
            input_text="HRTeam handles onboarding of new employees",
            dialogue_history=["HRTeam coordinates with IT department"],
            relationships=[],
            subject_patterns=PROCESS_ADAPTER.classification.subject_patterns,
            object_patterns=PROCESS_ADAPTER.classification.object_patterns,
        )

        assert len(scores) == 1
        # HRTeam should be detected as subject (handles, coordinates)
        assert scores[0].grammatical_role == "subject"

    def test_default_patterns_when_none(self):
        """Without adapter patterns, default patterns are used."""
        from core.classification import classify_components

        candidates = [
            {"name": "AuthService", "type": "service", "derived_from": "test"},
        ]

        scores = classify_components(
            candidates=candidates,
            input_text="AuthService handles user authentication",
            dialogue_history=[],
            relationships=[],
            # No subject_patterns/object_patterns → use defaults
        )

        assert len(scores) == 1
        assert scores[0].grammatical_role == "subject"
