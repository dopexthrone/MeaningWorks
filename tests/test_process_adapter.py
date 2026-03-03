"""
Tests for adapters/process.py — business process domain adapter.

Phase B: Second Domain Adapter — proves generalization.
"""

import pytest
from adapters.process import (
    PROCESS_ADAPTER,
    PROCESS_VOCABULARY,
    PROCESS_CLASSIFICATION,
    PROCESS_PROMPTS,
    PROCESS_VERIFICATION,
    PROCESS_MATERIALIZATION,
)
from core.domain_adapter import DomainAdapter
from core.adapter_registry import register_adapter, get_adapter, clear_registry
from core.classification import (
    classify_components,
    infer_component_type,
    is_likely_component,
)
from core.interface_extractor import extract_data_flows, extract_interface_map
from core.verification import score_actionability, verify_deterministic


@pytest.fixture(autouse=True)
def ensure_registered():
    """Ensure process adapter is registered."""
    import adapters  # noqa: F401 — auto-registers
    yield


# =============================================================================
# Adapter structure
# =============================================================================

class TestProcessAdapterStructure:
    def test_is_domain_adapter(self):
        assert isinstance(PROCESS_ADAPTER, DomainAdapter)

    def test_name(self):
        assert PROCESS_ADAPTER.name == "process"

    def test_version(self):
        assert PROCESS_ADAPTER.version == "1.0"

    def test_registered(self):
        adapter = get_adapter("process")
        assert adapter.name == "process"

    def test_yaml_output(self):
        assert PROCESS_MATERIALIZATION.output_format == "yaml"
        assert PROCESS_MATERIALIZATION.file_extension == ".yaml"


# =============================================================================
# Process vocabulary
# =============================================================================

class TestProcessVocabulary:
    def test_has_activity_keywords(self):
        assert "activity" in PROCESS_VOCABULARY.type_keywords
        assert "task" in PROCESS_VOCABULARY.type_keywords["activity"]
        assert "approval" in PROCESS_VOCABULARY.type_keywords["activity"]

    def test_has_gateway_keywords(self):
        assert "gateway" in PROCESS_VOCABULARY.type_keywords
        assert "decision" in PROCESS_VOCABULARY.type_keywords["gateway"]

    def test_has_event_keywords(self):
        assert "event" in PROCESS_VOCABULARY.type_keywords
        assert "trigger" in PROCESS_VOCABULARY.type_keywords["event"]
        assert "deadline" in PROCESS_VOCABULARY.type_keywords["event"]

    def test_has_participant_keywords(self):
        assert "participant" in PROCESS_VOCABULARY.type_keywords
        assert "role" in PROCESS_VOCABULARY.type_keywords["participant"]

    def test_has_relationship_flows(self):
        assert "follows" in PROCESS_VOCABULARY.relationship_flows
        assert "decides" in PROCESS_VOCABULARY.relationship_flows
        assert "escalates_to" in PROCESS_VOCABULARY.relationship_flows

    def test_entity_types_for_artifacts(self):
        assert "artifact" in PROCESS_VOCABULARY.entity_types
        assert "document" in PROCESS_VOCABULARY.entity_types
        assert "form" in PROCESS_VOCABULARY.entity_types

    def test_process_types_for_activities(self):
        assert "activity" in PROCESS_VOCABULARY.process_types
        assert "gateway" in PROCESS_VOCABULARY.process_types
        assert "event" in PROCESS_VOCABULARY.process_types


# =============================================================================
# Classification with process vocabulary
# =============================================================================

class TestProcessClassification:
    def test_activity_type_inference(self):
        """Activity components should be classified using process vocabulary."""
        comp_type, confidence = infer_component_type(
            "Document Review Task",
            "subject",
            "",
            PROCESS_VOCABULARY.type_keywords,
        )
        assert comp_type == "activity"
        assert confidence > 0.3

    def test_gateway_type_inference(self):
        comp_type, confidence = infer_component_type(
            "Approval Decision Gateway",
            "subject",
            "",
            PROCESS_VOCABULARY.type_keywords,
        )
        assert comp_type == "gateway"
        assert confidence > 0.3

    def test_event_type_inference(self):
        comp_type, confidence = infer_component_type(
            "Deadline Timer Event",
            "subject",
            "",
            PROCESS_VOCABULARY.type_keywords,
        )
        assert comp_type == "event"
        assert confidence > 0.3

    def test_generic_terms_filter(self):
        """Process generic terms should reject non-components."""
        is_comp, reason = is_likely_component(
            "status", 0.1, "modifier", 0.0,
            PROCESS_CLASSIFICATION.generic_terms,
        )
        assert not is_comp

    def test_classify_process_components(self):
        """Full classification with process vocabulary."""
        candidates = [
            {"name": "Document Review", "type": "activity"},
            {"name": "Manager Approval", "type": "activity"},
            {"name": "Escalation Gateway", "type": "gateway"},
            {"name": "status", "type": ""},
        ]
        results = classify_components(
            candidates,
            "Employee onboarding with document review, manager approval, and escalation",
            ["The review task happens before approval"],
            [],
            PROCESS_VOCABULARY.type_keywords,
            PROCESS_CLASSIFICATION.generic_terms,
        )
        names = [r.name for r in results if r.is_component]
        assert "Document Review" in names
        assert "Manager Approval" in names
        assert "status" not in names


# =============================================================================
# Interface extraction with process vocabulary
# =============================================================================

class TestProcessInterfaceExtraction:
    def test_follows_relationship(self):
        """Process 'follows' relationship produces sequence flow."""
        rel = {"type": "follows", "description": "review follows submission"}
        flows = extract_data_flows(
            rel, "Submission", "Review",
            PROCESS_VOCABULARY.relationship_flows,
            PROCESS_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "sequence_flow"
        assert flows[0].direction == "A_to_B"

    def test_decides_relationship(self):
        rel = {"type": "decides", "description": "gateway decides approval path"}
        flows = extract_data_flows(
            rel, "Gateway", "Approval",
            PROCESS_VOCABULARY.relationship_flows,
        )
        assert len(flows) == 1
        assert flows[0].name == "decision_flow"

    def test_escalates_relationship(self):
        rel = {"type": "escalates_to", "description": "overdue escalates to manager"}
        flows = extract_data_flows(
            rel, "Timer", "Manager Review",
            PROCESS_VOCABULARY.relationship_flows,
        )
        assert len(flows) == 1
        assert flows[0].name == "escalation"

    def test_unknown_relationship_is_bidirectional(self):
        rel = {"type": "unknown_type", "description": "some process relation"}
        flows = extract_data_flows(
            rel, "A", "B",
            PROCESS_VOCABULARY.relationship_flows,
        )
        assert len(flows) == 1
        assert flows[0].direction == "bidirectional"


# =============================================================================
# Verification with process checks
# =============================================================================

class TestProcessVerification:
    def test_actionability_with_decision_points(self):
        """Process components with decision_points should score well."""
        components = [
            {"name": "Review", "type": "activity", "description": "Review the application thoroughly",
             "decision_points": ["approve", "reject"]},
            {"name": "Gateway", "type": "gateway", "description": "Route to approval or rejection path",
             "activities": ["check_eligibility"]},
        ]
        score = score_actionability(
            0.5, components,
            PROCESS_VERIFICATION.actionability_checks,
        )
        assert score.score > 0
        # Both components have actionability indicators
        assert score.name == "actionability"

    def test_actionability_without_methods_but_with_activities(self):
        """Process domain doesn't require 'methods' — it checks 'activities'."""
        components = [
            {"name": "Step1", "type": "activity", "description": "First step in the process",
             "activities": ["submit_form", "validate"]},
        ]
        score = score_actionability(
            0.5, components,
            PROCESS_VERIFICATION.actionability_checks,
        )
        # Should score higher than 0 because "activities" is in actionability_checks
        assert score.score > 0

    def test_verify_deterministic_with_process_checks(self):
        """Full verification using process actionability checks."""
        blueprint = {
            "components": [
                {"name": "Submission", "type": "activity", "description": "Submit the application form",
                 "derived_from": "Employee submits application", "activities": ["fill_form"]},
                {"name": "Review", "type": "activity", "description": "HR reviews the submitted application",
                 "derived_from": "HR reviews application", "decision_points": ["approve", "reject"]},
            ],
            "relationships": [
                {"from": "Submission", "to": "Review", "type": "follows",
                 "description": "Review follows submission"},
            ],
            "constraints": [],
        }
        result = verify_deterministic(
            blueprint=blueprint,
            intent_keywords=["submission", "review", "application"],
            input_text="Employee submits application, HR reviews",
            graph_errors=[],
            graph_warnings=[],
            health_score=0.8,
            health_stats={"orphan_ratio": 0.0, "dangling_ref_count": 0},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.7,
            actionability_checks=PROCESS_VERIFICATION.actionability_checks,
        )
        assert result.overall_score > 0
        assert result.actionability.score > 0


# =============================================================================
# Prompts
# =============================================================================

class TestProcessPrompts:
    def test_intent_prompt_mentions_process(self):
        assert "business process" in PROCESS_PROMPTS.intent_system_prompt.lower()

    def test_entity_prompt_mentions_activities(self):
        assert "activities" in PROCESS_PROMPTS.entity_system_prompt.lower()

    def test_process_prompt_mentions_flows(self):
        assert "flow" in PROCESS_PROMPTS.process_system_prompt.lower()

    def test_emission_preamble_mentions_yaml(self):
        assert "yaml" in PROCESS_PROMPTS.emission_preamble.lower()

    def test_synthesis_prompt_mentions_process_types(self):
        assert "activity" in PROCESS_PROMPTS.synthesis_system_prompt.lower()
        assert "gateway" in PROCESS_PROMPTS.synthesis_system_prompt.lower()


# =============================================================================
# Cross-domain isolation
# =============================================================================

class TestCrossDomainIsolation:
    def test_software_and_process_both_registered(self):
        from core.adapter_registry import list_adapters
        adapters = list_adapters()
        assert "software" in adapters
        assert "process" in adapters

    def test_different_type_keywords(self):
        from adapters.software import SOFTWARE_VOCABULARY
        sw_types = set(SOFTWARE_VOCABULARY.type_keywords.keys())
        proc_types = set(PROCESS_VOCABULARY.type_keywords.keys())
        # They should have different type taxonomies
        assert sw_types != proc_types
        assert "agent" in sw_types
        assert "activity" in proc_types

    def test_different_output_formats(self):
        from adapters.software import SOFTWARE_MATERIALIZATION
        assert SOFTWARE_MATERIALIZATION.output_format == "python"
        assert PROCESS_MATERIALIZATION.output_format == "yaml"
