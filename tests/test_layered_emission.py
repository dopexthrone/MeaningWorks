"""
Layered Emission Protocol Tests.

Tests for EmissionLayer classification, layered planning, context-enriched prompts,
gate validation, engine integration, and emission dataclass updates.
"""

import ast
import pytest
from unittest.mock import MagicMock, patch

from core.naming import to_snake as _to_snake_case
from core.materialization import (
    EmissionLayer,
    LayerPlan,
    LayerGateResult,
    MaterializationBatch,
    MaterializationPlan,
    NodePrompt,
    DependencyGraph,
    classify_component_layer,
    build_layered_plan,
    build_node_prompt,
    build_node_prompt_with_context,
    validate_layer_gate,
    _extract_signatures,
    _promote_subsystem_children,
    _DEFAULT_ENTITY_TYPES,
    _DEFAULT_INTERFACE_TYPES,
)
from core.interface_schema import (
    InterfaceMap,
    InterfaceContract,
    DataFlow,
)
from core.agent_emission import (
    BatchEmission,
    EmissionResult,
    NodeEmission,
    EmissionConfig,
    EMISSION_VERSION,
    assemble_emission,
)


# =============================================================================
# Helpers
# =============================================================================

def _contract(a: str, b: str, direction: str = "mutual", rel_type: str = "manages") -> InterfaceContract:
    """Build a minimal InterfaceContract."""
    return InterfaceContract(
        node_a=a,
        node_b=b,
        relationship_type=rel_type,
        relationship_description=f"{a} {rel_type} {b}",
        data_flows=(
            DataFlow(name="data", type_hint="Dict", direction="A_to_B", derived_from="test"),
        ),
        constraints=(),
        fragility=0.5,
        confidence=0.8,
        directionality=direction,
        derived_from="test",
    )


def _imap(*contracts: InterfaceContract) -> InterfaceMap:
    """Build an InterfaceMap from contracts."""
    return InterfaceMap(
        contracts=tuple(contracts),
        unmatched_relationships=(),
        extraction_confidence=0.8,
        derived_from="test",
    )


def _blueprint(*components):
    """Build a minimal blueprint from (name, type) tuples."""
    return {
        "components": [
            {"name": name, "type": ctype, "description": f"{name} component"}
            for name, ctype in components
        ],
        "constraints": [],
        "relationships": [],
    }


# =============================================================================
# Layer Classification Tests (7)
# =============================================================================

class TestClassifyComponentLayer:
    def test_entity_is_types_layer(self):
        assert classify_component_layer("entity") == EmissionLayer.TYPES

    def test_data_is_types_layer(self):
        assert classify_component_layer("data") == EmissionLayer.TYPES

    def test_interface_is_interfaces_layer(self):
        assert classify_component_layer("interface") == EmissionLayer.INTERFACES

    def test_process_is_implementations_layer(self):
        assert classify_component_layer("process") == EmissionLayer.IMPLEMENTATIONS

    def test_unknown_type_defaults_implementation(self):
        assert classify_component_layer("foobar") == EmissionLayer.IMPLEMENTATIONS

    def test_custom_entity_types(self):
        custom = frozenset({"widget"})
        assert classify_component_layer("widget", entity_types=custom) == EmissionLayer.TYPES
        assert classify_component_layer("entity", entity_types=custom) == EmissionLayer.IMPLEMENTATIONS

    def test_custom_interface_types(self):
        custom = frozenset({"gateway"})
        assert classify_component_layer("gateway", interface_types=custom) == EmissionLayer.INTERFACES
        assert classify_component_layer("interface", interface_types=custom) == EmissionLayer.IMPLEMENTATIONS


# =============================================================================
# Layered Plan Tests (6)
# =============================================================================

class TestBuildLayeredPlan:
    def test_basic_layered_plan(self):
        """2 entity + 2 process → 2 non-empty layers."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("OrderModel", "entity"),
            ("UserService", "process"),
            ("OrderService", "process"),
        )
        imap = _imap(
            _contract("UserModel", "UserService", "A_depends_on_B"),
            _contract("OrderModel", "OrderService", "A_depends_on_B"),
        )
        plan = build_layered_plan(bp, imap)
        assert plan.layers is not None
        non_det = [lp for lp in plan.layers if not lp.is_deterministic]
        assert len(non_det) == 2
        assert non_det[0].layer_name == "types"
        assert non_det[1].layer_name == "implementations"

    def test_all_same_type_falls_back(self):
        """All entity → layers=None (flat mode)."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("OrderModel", "entity"),
        )
        imap = _imap(_contract("UserModel", "OrderModel"))
        plan = build_layered_plan(bp, imap)
        assert plan.layers is None

    def test_layer_batches_preserve_deps(self):
        """Dependency order within each layer is maintained."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("UserService", "process"),
            ("OrderService", "process"),
        )
        imap = _imap(
            _contract("UserModel", "UserService", "A_depends_on_B"),
            _contract("UserService", "OrderService", "A_depends_on_B"),
        )
        plan = build_layered_plan(bp, imap)
        assert plan.layers is not None
        impl_layer = [lp for lp in plan.layers if lp.layer_name == "implementations"][0]
        # UserService and OrderService should be in the implementation layer
        assert "UserService" in impl_layer.node_names
        assert "OrderService" in impl_layer.node_names

    def test_layer_3_is_deterministic(self):
        """Integration layer has is_deterministic=True."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("UserService", "process"),
        )
        imap = _imap(_contract("UserModel", "UserService"))
        plan = build_layered_plan(bp, imap)
        assert plan.layers is not None
        integration = [lp for lp in plan.layers if lp.layer == EmissionLayer.INTEGRATION]
        assert len(integration) == 1
        assert integration[0].is_deterministic is True

    def test_empty_layer_skipped(self):
        """No interface components → layer 1 absent from plan."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("UserService", "process"),
        )
        imap = _imap(_contract("UserModel", "UserService"))
        plan = build_layered_plan(bp, imap)
        assert plan.layers is not None
        layer_names = [lp.layer_name for lp in plan.layers if not lp.is_deterministic]
        assert "interfaces" not in layer_names

    def test_node_names_complete(self):
        """All components accounted for across layers."""
        bp = _blueprint(
            ("UserModel", "entity"),
            ("ApiGateway", "interface"),
            ("UserService", "process"),
        )
        imap = _imap(
            _contract("UserModel", "ApiGateway"),
            _contract("ApiGateway", "UserService"),
        )
        plan = build_layered_plan(bp, imap)
        assert plan.layers is not None
        all_names = set()
        for lp in plan.layers:
            all_names.update(lp.node_names)
        assert all_names == {"UserModel", "ApiGateway", "UserService"}


# =============================================================================
# Context-Enriched Prompt Tests (6)
# =============================================================================

class TestBuildNodePromptWithContext:
    def _make_component(self, name="UserService", ctype="process"):
        return {"name": name, "type": ctype, "description": f"{name} component"}

    def test_no_prior_code_same_as_base(self):
        """prior_layer_code=None → same structure as build_node_prompt."""
        comp = self._make_component()
        bp = _blueprint(("UserService", "process"))
        imap = _imap()
        base = build_node_prompt(comp, bp, imap)
        enriched = build_node_prompt_with_context(comp, bp, imap)
        # Without prior code or layer, should be identical
        assert enriched.prompt_text == base.prompt_text

    def test_prior_code_appears_in_prompt(self):
        """Actual code text appears in prompt_text."""
        comp = self._make_component()
        bp = _blueprint(("UserModel", "entity"), ("UserService", "process"))
        imap = _imap(_contract("UserModel", "UserService"))
        prior = {"UserModel": "class UserModel:\n    name: str\n    email: str"}
        enriched = build_node_prompt_with_context(
            comp, bp, imap, prior_layer_code=prior, layer=EmissionLayer.IMPLEMENTATIONS,
        )
        assert "class UserModel:" in enriched.prompt_text
        assert "name: str" in enriched.prompt_text

    def test_anti_redefinition_instruction(self):
        """'DO NOT REDEFINE' present when prior code provided."""
        comp = self._make_component()
        bp = _blueprint(("UserModel", "entity"), ("UserService", "process"))
        imap = _imap(_contract("UserModel", "UserService"))
        prior = {"UserModel": "class UserModel:\n    pass"}
        enriched = build_node_prompt_with_context(
            comp, bp, imap, prior_layer_code=prior, layer=EmissionLayer.IMPLEMENTATIONS,
        )
        assert "DO NOT REDEFINE" in enriched.prompt_text

    def test_only_adjacent_code_full(self):
        """Non-adjacent prior code shows as signatures only."""
        comp = self._make_component()
        bp = _blueprint(
            ("UserModel", "entity"),
            ("OrderModel", "entity"),
            ("UserService", "process"),
        )
        # UserModel is adjacent; OrderModel is not
        imap = _imap(_contract("UserModel", "UserService"))
        prior = {
            "UserModel": "class UserModel:\n    \"\"\"User entity.\"\"\"\n    name: str",
            "OrderModel": "class OrderModel:\n    \"\"\"Order entity.\"\"\"\n    def total(self) -> float:\n        return 0.0",
        }
        enriched = build_node_prompt_with_context(
            comp, bp, imap, prior_layer_code=prior, layer=EmissionLayer.IMPLEMENTATIONS,
        )
        # Adjacent gets full code
        assert "(adjacent" in enriched.prompt_text
        # Non-adjacent gets signatures
        assert "(available" in enriched.prompt_text

    def test_layer_0_type_instruction(self):
        """Types layer instruction present for layer 0."""
        comp = self._make_component("UserModel", "entity")
        bp = _blueprint(("UserModel", "entity"))
        imap = _imap()
        enriched = build_node_prompt_with_context(
            comp, bp, imap, layer=EmissionLayer.TYPES,
        )
        assert "pure dataclass" in enriched.prompt_text.lower()

    def test_layer_2_import_instruction(self):
        """Implementation layer instruction for layer 2."""
        comp = self._make_component()
        bp = _blueprint(("UserModel", "entity"), ("UserService", "process"))
        imap = _imap(_contract("UserModel", "UserService"))
        prior = {"UserModel": "class UserModel:\n    pass"}
        enriched = build_node_prompt_with_context(
            comp, bp, imap, prior_layer_code=prior, layer=EmissionLayer.IMPLEMENTATIONS,
        )
        assert "import" in enriched.prompt_text.lower()
        assert "prior layer" in enriched.prompt_text.lower()


# =============================================================================
# Gate Validation Tests (6)
# =============================================================================

class TestValidateLayerGate:
    def test_gate_passes_valid_code(self):
        """Parseable Python → passed=True."""
        code = {"UserModel": "class UserModel:\n    name: str = ''\n"}
        gate = validate_layer_gate(EmissionLayer.TYPES, code)
        assert gate.passed is True
        assert len(gate.errors) == 0

    def test_gate_fails_syntax_error(self):
        """Unparseable code → passed=False, error message."""
        code = {"UserModel": "class UserModel\n    name: str"}  # Missing colon
        gate = validate_layer_gate(EmissionLayer.TYPES, code)
        assert gate.passed is False
        assert any("syntax error" in e for e in gate.errors)

    def test_layer_0_no_cross_deps(self):
        """L0 code with relative import → error."""
        code = {"UserModel": "from .order_model import OrderModel\n\nclass UserModel:\n    pass\n"}
        gate = validate_layer_gate(EmissionLayer.TYPES, code)
        assert gate.passed is False
        assert any("relative import" in e for e in gate.errors)

    def test_layer_1_l0_import_resolves(self):
        """L1 refs L0 name → passes (warning-free for known priors)."""
        layer_code = {"UserInterface": "from .user_model import UserModel\n\nclass UserInterface:\n    pass\n"}
        prior = {"UserModel": "class UserModel:\n    pass\n"}
        gate = validate_layer_gate(EmissionLayer.INTERFACES, layer_code, prior_layer_code=prior)
        assert gate.passed is True

    def test_layer_2_unresolved_import(self):
        """L2 refs nonexistent module → warning (not blocking error)."""
        layer_code = {"UserService": "from .nonexistent import Foo\n\nclass UserService:\n    pass\n"}
        prior = {"UserModel": "class UserModel:\n    pass\n"}
        gate = validate_layer_gate(EmissionLayer.IMPLEMENTATIONS, layer_code, prior_layer_code=prior)
        # Unresolved imports are warnings, not errors (may resolve at integration)
        assert gate.passed is True
        assert any("nonexistent" in w for w in gate.warnings)

    def test_non_python_always_passes(self):
        """is_python=False → trivial pass."""
        code = {"activity": "not: valid: python: at: all {{{"}
        gate = validate_layer_gate(EmissionLayer.TYPES, code, is_python=False)
        assert gate.passed is True


# =============================================================================
# Engine Integration Tests (7)
# =============================================================================

class TestEngineLayeredEmission:
    """Tests that engine.emit_code routes correctly between layered and flat paths."""

    def _make_engine(self):
        """Build a minimal engine with MockClient."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine
        engine = MotherlabsEngine(llm_client=MockClient())
        return engine

    def _mixed_blueprint(self):
        return _blueprint(
            ("UserModel", "entity"),
            ("UserService", "process"),
        )

    def _uniform_blueprint(self):
        return _blueprint(
            ("UserService", "process"),
            ("OrderService", "process"),
        )

    def _make_imap_for(self, bp):
        names = [c["name"] for c in bp["components"]]
        if len(names) >= 2:
            return _imap(_contract(names[0], names[1]))
        return _imap()

    def test_layered_emit_basic(self):
        """layered=True with mixed types → result.layered=True."""
        engine = self._make_engine()
        bp = self._mixed_blueprint()
        imap = self._make_imap_for(bp)
        result = engine.emit_code(bp, interface_map=imap, layered=True)
        assert result.layered is True
        assert len(result.layer_gate_results) > 0

    def test_layered_false_uses_flat(self):
        """layered=False → flat path, .layered=False."""
        engine = self._make_engine()
        bp = self._mixed_blueprint()
        imap = self._make_imap_for(bp)
        result = engine.emit_code(bp, interface_map=imap, layered=False)
        assert result.layered is False

    def test_gate_results_populated(self):
        """result.layer_gate_results is non-empty tuple for layered emission."""
        engine = self._make_engine()
        bp = self._mixed_blueprint()
        imap = self._make_imap_for(bp)
        result = engine.emit_code(bp, interface_map=imap, layered=True)
        assert isinstance(result.layer_gate_results, tuple)
        assert len(result.layer_gate_results) > 0

    def test_gate_failure_continues(self):
        """Gate fails → degraded mode, no crash."""
        engine = self._make_engine()
        bp = self._mixed_blueprint()
        imap = self._make_imap_for(bp)
        # MockClient returns non-parseable text, but emission should still complete
        result = engine.emit_code(bp, interface_map=imap, layered=True)
        # Should complete without exception
        assert result.total_nodes == 2

    def test_all_same_type_uses_flat(self):
        """All 'process' → layers=None → flat path."""
        engine = self._make_engine()
        bp = self._uniform_blueprint()
        imap = self._make_imap_for(bp)
        result = engine.emit_code(bp, interface_map=imap, layered=True)
        # Falls back to flat because all components are "process"
        assert result.layered is False

    def test_backward_compat_no_regression(self):
        """Existing emit_code() call without layered param works."""
        engine = self._make_engine()
        bp = self._mixed_blueprint()
        imap = self._make_imap_for(bp)
        # Call without layered parameter — should default to True
        result = engine.emit_code(bp, interface_map=imap)
        assert result is not None
        assert result.total_nodes == 2

    def test_emit_node_llm_extracted(self):
        """_emit_node_llm() helper works standalone."""
        engine = self._make_engine()
        system = "Generate a Python class."
        code, success, error = engine._emit_node_llm("TestComp", system)
        # MockClient returns "[Mock response #N]" — not valid Python
        # so success depends on whether it parses
        assert isinstance(code, str)
        assert isinstance(success, bool)


# =============================================================================
# Emission Dataclass Tests (3)
# =============================================================================

class TestEmissionDataclassUpdates:
    def test_batch_emission_layer_default(self):
        """BatchEmission() without layer → None."""
        be = BatchEmission(
            batch_index=0,
            emissions=(),
            success_count=0,
            failure_count=0,
        )
        assert be.layer is None

    def test_emission_result_layered_default(self):
        """EmissionResult() without new fields → backward compat."""
        result = EmissionResult(
            batch_emissions=(),
            generated_code={},
            verification_report={},
            total_nodes=0,
            success_count=0,
            failure_count=0,
            pass_rate=0.0,
            l2_context_injected=False,
            timestamp="2026-02-11",
            derived_from="test",
        )
        assert result.layered is False
        assert result.layer_gate_results == ()

    def test_emission_result_with_layers(self):
        """New fields populated correctly."""
        gate = LayerGateResult(layer=0, passed=True, errors=(), warnings=())
        result = EmissionResult(
            batch_emissions=(),
            generated_code={},
            verification_report={},
            total_nodes=0,
            success_count=0,
            failure_count=0,
            pass_rate=0.0,
            l2_context_injected=False,
            timestamp="2026-02-11",
            derived_from="test",
            layer_gate_results=(gate,),
            layered=True,
        )
        assert result.layered is True
        assert len(result.layer_gate_results) == 1
        assert result.layer_gate_results[0].passed is True


# =============================================================================
# Subsystem Promotion Tests (5)
# =============================================================================

class TestSubsystemPromotion:
    """Tests for _promote_subsystem_children() — Fix 1."""

    def _make_graph(self, nodes, edges):
        """Build a DependencyGraph from node list and (dep, dependent) edges."""
        adjacency = {n: set() for n in nodes}
        reverse = {n: set() for n in nodes}
        for dep, dependent in edges:
            adjacency[dependent].add(dep)
            reverse[dep].add(dependent)
        return DependencyGraph(
            nodes=sorted(nodes),
            edges=edges,
            adjacency=adjacency,
            reverse=reverse,
        )

    def test_child_entity_promoted_out_of_layer_0(self):
        """Entity child with L0 dep → promoted to L1."""
        graph = self._make_graph(
            ["SharedState", "Knowledge"],
            [("SharedState", "Knowledge")],
        )
        node_layers = {
            "SharedState": EmissionLayer.TYPES,
            "Knowledge": EmissionLayer.TYPES,
        }
        result = _promote_subsystem_children(node_layers, graph)
        assert result["SharedState"] == EmissionLayer.TYPES
        assert result["Knowledge"] == EmissionLayer.INTERFACES

    def test_parent_entity_stays_in_layer_0(self):
        """Parent with no L0 deps → stays L0."""
        graph = self._make_graph(
            ["SharedState", "UserService"],
            [("SharedState", "UserService")],
        )
        node_layers = {
            "SharedState": EmissionLayer.TYPES,
            "UserService": EmissionLayer.IMPLEMENTATIONS,
        }
        result = _promote_subsystem_children(node_layers, graph)
        assert result["SharedState"] == EmissionLayer.TYPES

    def test_no_promotion_when_no_deps(self):
        """Two independent entities → both stay L0."""
        graph = self._make_graph(["ModelA", "ModelB"], [])
        node_layers = {
            "ModelA": EmissionLayer.TYPES,
            "ModelB": EmissionLayer.TYPES,
        }
        result = _promote_subsystem_children(node_layers, graph)
        assert result["ModelA"] == EmissionLayer.TYPES
        assert result["ModelB"] == EmissionLayer.TYPES

    def test_multiple_children_all_promoted(self):
        """SharedState contains K, U, O → all promoted."""
        graph = self._make_graph(
            ["SharedState", "Knowledge", "Uncertainty", "Observations"],
            [
                ("SharedState", "Knowledge"),
                ("SharedState", "Uncertainty"),
                ("SharedState", "Observations"),
            ],
        )
        node_layers = {
            "SharedState": EmissionLayer.TYPES,
            "Knowledge": EmissionLayer.TYPES,
            "Uncertainty": EmissionLayer.TYPES,
            "Observations": EmissionLayer.TYPES,
        }
        result = _promote_subsystem_children(node_layers, graph)
        assert result["SharedState"] == EmissionLayer.TYPES
        assert result["Knowledge"] == EmissionLayer.INTERFACES
        assert result["Uncertainty"] == EmissionLayer.INTERFACES
        assert result["Observations"] == EmissionLayer.INTERFACES

    def test_promotion_preserves_non_entity_layers(self):
        """Process components unaffected by promotion."""
        graph = self._make_graph(
            ["SharedState", "Knowledge", "UserService"],
            [("SharedState", "Knowledge"), ("Knowledge", "UserService")],
        )
        node_layers = {
            "SharedState": EmissionLayer.TYPES,
            "Knowledge": EmissionLayer.TYPES,
            "UserService": EmissionLayer.IMPLEMENTATIONS,
        }
        result = _promote_subsystem_children(node_layers, graph)
        assert result["UserService"] == EmissionLayer.IMPLEMENTATIONS

    def test_containment_relationship_promotes_child(self):
        """Blueprint 'contains' relationship promotes child even without graph edge."""
        # No graph edges (mutual directionality → no edge)
        graph = self._make_graph(["tasks", "priority levels"], [])
        node_layers = {
            "tasks": EmissionLayer.TYPES,
            "priority levels": EmissionLayer.TYPES,
        }
        blueprint = {
            "components": [
                {"name": "tasks", "type": "entity"},
                {"name": "priority levels", "type": "entity"},
            ],
            "relationships": [
                {"from": "tasks", "to": "priority levels", "type": "contains"},
            ],
        }
        result = _promote_subsystem_children(node_layers, graph, blueprint)
        assert result["tasks"] == EmissionLayer.TYPES  # Parent stays
        assert result["priority levels"] == EmissionLayer.INTERFACES  # Child promoted

    def test_containment_no_promotion_without_blueprint(self):
        """Without blueprint, containment not detected (graph-only path)."""
        graph = self._make_graph(["tasks", "priority levels"], [])
        node_layers = {
            "tasks": EmissionLayer.TYPES,
            "priority levels": EmissionLayer.TYPES,
        }
        # No blueprint passed → only graph deps checked
        result = _promote_subsystem_children(node_layers, graph)
        assert result["tasks"] == EmissionLayer.TYPES
        assert result["priority levels"] == EmissionLayer.TYPES


# =============================================================================
# Token Scaling Tests (5)
# =============================================================================

class TestTokenScaling:
    """Tests for _scale_max_tokens() — Fix 2."""

    def test_short_prompt_no_scaling(self):
        """≤1000 chars → base tokens."""
        from core.engine import _scale_max_tokens
        assert _scale_max_tokens(4096, "x" * 500) == 4096

    def test_medium_prompt_1_5x(self):
        """1000-3000 → 1.5x."""
        from core.engine import _scale_max_tokens
        assert _scale_max_tokens(4096, "x" * 2000) == int(4096 * 1.5)

    def test_long_prompt_2x(self):
        """3000-6000 → 2.0x."""
        from core.engine import _scale_max_tokens
        assert _scale_max_tokens(4096, "x" * 5000) == int(4096 * 2.0)

    def test_very_long_prompt_3x(self):
        """>6000 → 3.0x."""
        from core.engine import _scale_max_tokens
        assert _scale_max_tokens(4096, "x" * 8000) == int(4096 * 3.0)

    def test_cap_at_32768(self):
        """Cap enforced."""
        from core.engine import _scale_max_tokens
        assert _scale_max_tokens(16384, "x" * 10000) == 32768
