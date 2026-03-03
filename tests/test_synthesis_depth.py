"""
Phase 12.5: Synthesis Depth Fix — Post-Synthesis Method & Property Enrichment.

Tests for:
- _enrich_blueprint_methods() deterministic backfill (8 tests)
- Output template shape in swarm.py (2 tests)
- Method completeness with natural language input (2 tests)
"""

import pytest
from unittest.mock import Mock

from core.engine import MotherlabsEngine
from core.protocol import SharedState
from core.llm import BaseLLMClient


# =============================================================================
# HELPERS
# =============================================================================

def _make_engine():
    """Create a minimal engine instance for testing."""
    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock"
    return MotherlabsEngine(llm_client=client)


def _make_state(**known_overrides):
    """Create a SharedState with optional known overrides."""
    state = SharedState()
    for k, v in known_overrides.items():
        state.known[k] = v
    return state


def _make_blueprint(*component_dicts):
    """Create a blueprint with the given components."""
    return {
        "components": list(component_dicts),
        "relationships": [],
    }


def _make_method(component, name, params=None, return_type="None", derived_from=""):
    """Create an extracted method dict matching the shape from extract_dialogue_methods."""
    return {
        "component": component,
        "name": name,
        "parameters": params or [],
        "return_type": return_type,
        "derived_from": derived_from or f"METHOD: {component}.{name}()",
        "source": "dialogue",
    }


def _make_component(name, comp_type="entity", methods=None):
    """Create a blueprint component dict."""
    comp = {
        "name": name,
        "type": comp_type,
        "description": f"Test {name}",
        "derived_from": f"test input for {name}",
    }
    if methods is not None:
        comp["methods"] = methods
    return comp


# =============================================================================
# TestEnrichBlueprintMethods
# =============================================================================

class TestEnrichBlueprintMethods:
    """Tests for _enrich_blueprint_methods() deterministic backfill."""

    def test_no_extracted_methods_runs_inference(self):
        """Empty state.known triggers deterministic method inference fallback."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),  # type="entity" → gets CRUD
        )
        state = _make_state()

        result = engine._enrich_blueprint_methods(blueprint, state)

        assert result is blueprint
        # Entity type gets CRUD methods from _infer_component_methods
        methods = result["components"][0]["methods"]
        assert len(methods) == 4
        names = [m["name"] for m in methods]
        assert "create_taskmanager" in names

    def test_methods_mapped_to_matching_component(self):
        """Extracted methods get mapped to the correct blueprint component."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),
            _make_component("UserService"),
        )
        methods = [
            _make_method("TaskManager", "create_task"),
            _make_method("TaskManager", "delete_task"),
            _make_method("UserService", "authenticate"),
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        tm = result["components"][0]
        us = result["components"][1]
        assert len(tm["methods"]) == 2
        assert {m["name"] for m in tm["methods"]} == {"create_task", "delete_task"}
        assert len(us["methods"]) == 1
        assert us["methods"][0]["name"] == "authenticate"

    def test_methods_normalized_component_name(self):
        """Component name normalization maps variant names to blueprint names."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("Governor Agent", comp_type="agent"),
        )
        # "Governor" should normalize to "Governor Agent" via _normalize_method_component
        methods = [
            _make_method("Governor", "validate_trust"),
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        gov = result["components"][0]
        assert "methods" in gov
        assert len(gov["methods"]) == 1
        assert gov["methods"][0]["name"] == "validate_trust"

    def test_methods_fuzzy_prefix_match(self):
        """Partial name prefix matching works when exact fails."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManagerService"),
        )
        # "taskmanager" is a prefix of "taskmanagerservice"
        methods = [
            _make_method("TaskManager", "create_task"),
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        comp = result["components"][0]
        assert "methods" in comp
        assert len(comp["methods"]) == 1

    def test_no_duplicate_methods(self):
        """Same method name is not added twice to a component."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager", methods=[
                {"name": "create_task", "parameters": [], "return_type": "Task", "derived_from": "spec"}
            ]),
        )
        methods = [
            _make_method("TaskManager", "create_task"),  # duplicate
            _make_method("TaskManager", "delete_task"),  # new
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        tm = result["components"][0]
        assert len(tm["methods"]) == 2  # 1 existing + 1 new
        names = [m["name"] for m in tm["methods"]]
        assert names.count("create_task") == 1

    def test_state_machine_enrichment(self):
        """State machines from dialogue get mapped to matching components."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),
        )
        state_machines = [{
            "component": "TaskManager",
            "states": ["OPEN", "IN_PROGRESS", "DONE"],
            "transitions": [
                {"from": "OPEN", "to": "IN_PROGRESS", "trigger": "start"},
                {"from": "IN_PROGRESS", "to": "DONE", "trigger": "complete"},
            ],
            "derived_from": "STATES: TaskManager",
        }]
        state = _make_state(extracted_state_machines=state_machines)

        result = engine._enrich_blueprint_methods(blueprint, state)

        tm = result["components"][0]
        assert "state_machine" in tm
        assert tm["state_machine"]["states"] == ["OPEN", "IN_PROGRESS", "DONE"]

    def test_unmatched_methods_skipped_but_inferred(self):
        """Methods for nonexistent components are skipped; gap-fill infers methods."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),
        )
        methods = [
            _make_method("NonexistentService", "do_thing"),
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        tm = result["components"][0]
        # Dialogue method "do_thing" is NOT added (wrong component)
        method_names = [m["name"] for m in tm.get("methods", [])]
        assert "do_thing" not in method_names
        # But gap-fill adds inferred CRUD methods (type=entity)
        assert len(tm["methods"]) == 4

    def test_empty_methods_array_initialized(self):
        """Component with no methods key gets one when methods are added."""
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),  # no "methods" key
        )
        assert "methods" not in blueprint["components"][0]

        methods = [
            _make_method("TaskManager", "create_task"),
        ]
        state = _make_state(extracted_methods=methods)

        result = engine._enrich_blueprint_methods(blueprint, state)

        tm = result["components"][0]
        assert "methods" in tm
        assert isinstance(tm["methods"], list)
        assert len(tm["methods"]) == 1


# =============================================================================
# TestOutputTemplateShape
# =============================================================================

class TestOutputTemplateShape:
    """Verify swarm.py synthesis template includes methods and attributes fields."""

    def test_template_includes_methods_field(self):
        """The SYNTHESIS_SYSTEM_PROMPT must include 'methods' in component template."""
        from agents.swarm import SYNTHESIS_SYSTEM_PROMPT
        assert '"methods"' in SYNTHESIS_SYSTEM_PROMPT

    def test_template_includes_attributes_field(self):
        """The SYNTHESIS_SYSTEM_PROMPT must include 'attributes' in component template."""
        from agents.swarm import SYNTHESIS_SYSTEM_PROMPT
        assert '"attributes"' in SYNTHESIS_SYSTEM_PROMPT


# =============================================================================
# TestMethodCompletenessNaturalLanguage
# =============================================================================

class TestMethodCompletenessNaturalLanguage:
    """Test that natural language inputs don't vacuously pass method validation."""

    def test_natural_language_input_no_false_pass(self):
        """
        'Build a task manager' has no method signatures, so _validate_method_completeness
        returns empty missing list — this is the vacuous pass we're documenting.
        The fix is _enrich_blueprint_methods, not changing validation.
        """
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),
        )

        missing = engine._validate_method_completeness(
            "Build a task manager with CRUD operations",
            blueprint,
        )
        # No method signatures in natural language → empty missing list (vacuous pass)
        assert missing == []

    def test_extracted_methods_validated_via_enrichment(self):
        """
        When extracted_methods exist from dialogue, they should end up in the
        blueprint via _enrich_blueprint_methods, making them available for
        downstream validation.
        """
        engine = _make_engine()
        blueprint = _make_blueprint(
            _make_component("TaskManager"),
        )
        methods = [
            _make_method("TaskManager", "create_task", return_type="Task"),
            _make_method("TaskManager", "delete_task", return_type="bool"),
        ]
        state = _make_state(extracted_methods=methods)

        # After enrichment, methods should be in blueprint
        enriched = engine._enrich_blueprint_methods(blueprint, state)

        # Now _validate_method_completeness should find these methods
        tm = enriched["components"][0]
        method_names = {m["name"] for m in tm.get("methods", [])}
        assert "create_task" in method_names
        assert "delete_task" in method_names
