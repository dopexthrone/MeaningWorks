"""
Phase D: Agent Emission Tests.

Tests for core/agent_emission.py (LEAF MODULE) and engine.emit_code().
"""

import json
import pytest
from datetime import datetime

from core.agent_emission import (
    EmissionConfig,
    NodeEmission,
    BatchEmission,
    EmissionResult,
    EMISSION_PREAMBLE,
    EMISSION_VERSION,
    build_emission_system_prompt,
    compute_prompt_hash,
    extract_code_from_response,
    assemble_emission,
    serialize_emission_result,
    deserialize_emission_result,
)
from core.materialization import (
    NodePrompt,
    MaterializationPlan,
    MaterializationBatch,
    DependencyGraph,
    build_materialization_plan,
    verify_interfaces,
)
from core.interface_schema import (
    InterfaceMap,
    InterfaceContract,
    DataFlow,
    InterfaceConstraint,
)
from core.compilation_tree import (
    L2Synthesis,
    CrossCuttingComponent,
    InterfaceGap,
    format_l2_patterns_section,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_node_prompt():
    """A minimal NodePrompt for testing."""
    return NodePrompt(
        component_name="UserService",
        component_type="service",
        description="Handles user management",
        dimensional_position={"complexity": 0.5, "stability": 0.8},
        interfaces=[{
            "adjacent_node": "AuthService",
            "relationship": "depends_on",
            "direction": "A_depends_on_B",
            "data_flows": [{"name": "auth_token", "type": "str", "direction": "B_to_A"}],
            "fragility": 0.3,
        }],
        constraints=["Must validate email format"],
        methods=[{"name": "create_user", "parameters": [{"name": "email", "type_hint": "str"}], "return_type": "User"}],
        prompt_text="# Materialize: UserService\nType: service\nDescription: Handles user management\n\nGenerate a complete Python implementation for this component.",
    )


@pytest.fixture
def empty_interface_map():
    """An InterfaceMap with no contracts."""
    return InterfaceMap(
        contracts=(),
        unmatched_relationships=(),
        extraction_confidence=1.0,
        derived_from="test",
    )


@pytest.fixture
def simple_interface_map():
    """An InterfaceMap with one contract."""
    contract = InterfaceContract(
        node_a="UserService",
        node_b="AuthService",
        relationship_type="depends_on",
        relationship_description="UserService depends on AuthService",
        data_flows=(
            DataFlow(
                name="auth_token",
                type_hint="str",
                direction="B_to_A",
                derived_from="test",
            ),
        ),
        constraints=(),
        fragility=0.3,
        confidence=0.9,
        directionality="A_depends_on_B",
        derived_from="test",
    )
    return InterfaceMap(
        contracts=(contract,),
        unmatched_relationships=(),
        extraction_confidence=0.9,
        derived_from="test",
    )


@pytest.fixture
def sample_blueprint():
    """A minimal blueprint for testing."""
    return {
        "components": [
            {"name": "UserService", "type": "service", "description": "User management"},
            {"name": "AuthService", "type": "service", "description": "Authentication"},
        ],
        "relationships": [
            {"from": "UserService", "to": "AuthService", "type": "depends_on"},
        ],
        "constraints": [],
    }


@pytest.fixture
def sample_l2_synthesis():
    """A minimal L2Synthesis for testing."""
    return L2Synthesis(
        shared_vocabulary=(("user", 3), ("auth", 2)),
        cross_cutting_components=(
            CrossCuttingComponent(
                normalized_name="userservice",
                variants=("UserService",),
                frequency=1.0,
                child_sources=("auth_sub", "user_sub"),
                component_type="service",
            ),
        ),
        relationship_patterns=(("userservice", "authservice", "depends_on", 2),),
        interface_gaps=(),
        integration_constraints=("Shared component 'userservice' (service) must have consistent interface across: auth_sub, user_sub",),
        pattern_count=5,
        synthesis_confidence=0.5,
    )


# =============================================================================
# D.1 TESTS: Dataclass Construction + Immutability
# =============================================================================


class TestEmissionConfig:
    """Tests for EmissionConfig frozen dataclass."""

    def test_defaults(self):
        config = EmissionConfig()
        assert config.max_tokens == 16384
        assert config.temperature == 0.0
        assert config.system_preamble == ""
        assert config.retry_failed is True
        assert config.max_retries == 2

    def test_custom_values(self):
        config = EmissionConfig(max_tokens=32768, temperature=0.2, system_preamble="Custom")
        assert config.max_tokens == 32768
        assert config.temperature == 0.2
        assert config.system_preamble == "Custom"

    def test_frozen(self):
        config = EmissionConfig()
        with pytest.raises(AttributeError):
            config.max_tokens = 1234


class TestNodeEmission:
    """Tests for NodeEmission frozen dataclass."""

    def test_construction(self):
        ne = NodeEmission(
            component_name="Foo",
            component_type="service",
            code="class Foo: pass",
            success=True,
            error=None,
            prompt_hash="abc123",
            derived_from=EMISSION_VERSION,
        )
        assert ne.component_name == "Foo"
        assert ne.success is True
        assert ne.error is None

    def test_with_error(self):
        ne = NodeEmission(
            component_name="Foo",
            component_type="service",
            code="",
            success=False,
            error="LLM timeout",
            prompt_hash="abc123",
            derived_from=EMISSION_VERSION,
        )
        assert ne.success is False
        assert ne.error == "LLM timeout"

    def test_frozen(self):
        ne = NodeEmission("A", "service", "", True, None, "h", "v")
        with pytest.raises(AttributeError):
            ne.code = "new code"


class TestBatchEmission:
    """Tests for BatchEmission frozen dataclass."""

    def test_construction(self):
        ne = NodeEmission("A", "service", "code", True, None, "h", "v")
        be = BatchEmission(batch_index=0, emissions=(ne,), success_count=1, failure_count=0)
        assert be.batch_index == 0
        assert len(be.emissions) == 1
        assert be.success_count == 1

    def test_empty(self):
        be = BatchEmission(batch_index=0, emissions=(), success_count=0, failure_count=0)
        assert len(be.emissions) == 0

    def test_frozen(self):
        be = BatchEmission(0, (), 0, 0)
        with pytest.raises(AttributeError):
            be.batch_index = 1


class TestEmissionResult:
    """Tests for EmissionResult frozen dataclass."""

    def test_construction(self):
        er = EmissionResult(
            batch_emissions=(),
            generated_code={},
            verification_report={"pass_rate": 1.0},
            total_nodes=0,
            success_count=0,
            failure_count=0,
            pass_rate=1.0,
            l2_context_injected=False,
            timestamp="2026-01-01T00:00:00",
            derived_from=EMISSION_VERSION,
        )
        assert er.total_nodes == 0
        assert er.derived_from == EMISSION_VERSION

    def test_frozen(self):
        er = EmissionResult((), {}, {}, 0, 0, 0, 0.0, False, "", "v")
        with pytest.raises(AttributeError):
            er.total_nodes = 5


# =============================================================================
# D.1 TESTS: build_emission_system_prompt
# =============================================================================


class TestBuildEmissionSystemPrompt:
    """Tests for build_emission_system_prompt()."""

    def test_basic(self, simple_node_prompt):
        prompt = build_emission_system_prompt(simple_node_prompt)
        assert EMISSION_PREAMBLE.strip() in prompt
        assert "Materialize: UserService" in prompt

    def test_with_l2(self, simple_node_prompt, sample_l2_synthesis):
        l2_section = format_l2_patterns_section(sample_l2_synthesis)
        prompt = build_emission_system_prompt(simple_node_prompt, l2_section=l2_section)
        assert "Cross-Subsystem Patterns" in prompt
        assert "Materialize: UserService" in prompt

    def test_with_config_preamble(self, simple_node_prompt):
        config = EmissionConfig(system_preamble="Use FastAPI for all services.")
        prompt = build_emission_system_prompt(simple_node_prompt, config=config)
        assert "Use FastAPI for all services." in prompt
        assert EMISSION_PREAMBLE.strip() in prompt

    def test_without_l2(self, simple_node_prompt):
        prompt = build_emission_system_prompt(simple_node_prompt, l2_section=None)
        assert "Cross-Subsystem Patterns" not in prompt

    def test_all_combined(self, simple_node_prompt, sample_l2_synthesis):
        l2_section = format_l2_patterns_section(sample_l2_synthesis)
        config = EmissionConfig(system_preamble="Project: Motherlabs")
        prompt = build_emission_system_prompt(simple_node_prompt, l2_section, config)
        assert "Project: Motherlabs" in prompt
        assert "Materialize: UserService" in prompt
        assert "Cross-Subsystem Patterns" in prompt


# =============================================================================
# D.1 TESTS: extract_code_from_response
# =============================================================================


class TestExtractCodeFromResponse:
    """Tests for extract_code_from_response()."""

    def test_python_block(self):
        response = "Here's the code:\n```python\nclass Foo:\n    pass\n```\nDone."
        code = extract_code_from_response(response)
        assert "class Foo:" in code
        assert "pass" in code

    def test_bare_code(self):
        response = "class Foo:\n    def bar(self):\n        return 42"
        code = extract_code_from_response(response)
        assert "class Foo:" in code
        assert "return 42" in code

    def test_explanation_wrapping(self):
        response = "I'll create a Foo class.\n```python\nclass Foo:\n    pass\n```\nThis implements the component."
        code = extract_code_from_response(response)
        assert "class Foo:" in code
        assert "I'll create" not in code

    def test_multiple_blocks(self):
        response = "```python\nclass A:\n    pass\n```\n\nAlso:\n```python\nclass B:\n    pass\n```"
        code = extract_code_from_response(response)
        assert "class A:" in code
        assert "class B:" in code

    def test_empty_response(self):
        assert extract_code_from_response("") == ""
        assert extract_code_from_response("   ") == ""
        assert extract_code_from_response(None) == ""

    def test_malformed_no_code(self):
        response = "I don't know how to implement this component."
        code = extract_code_from_response(response)
        assert code == ""

    def test_generic_code_block(self):
        response = "```\ndef hello():\n    print('hi')\n```"
        code = extract_code_from_response(response)
        assert "def hello():" in code

    def test_generic_block_non_python(self):
        response = "```\nSome random text with no code markers\n```"
        code = extract_code_from_response(response)
        # Should still extract since it's inside a code block
        assert "Some random text" in code

    def test_preserves_indentation(self):
        response = "```python\nclass Foo:\n    def bar(self):\n        if True:\n            return 1\n```"
        code = extract_code_from_response(response)
        assert "        if True:" in code
        assert "            return 1" in code


# =============================================================================
# D.1 TESTS: compute_prompt_hash
# =============================================================================


class TestComputePromptHash:
    """Tests for compute_prompt_hash()."""

    def test_deterministic(self):
        h1 = compute_prompt_hash("hello world")
        h2 = compute_prompt_hash("hello world")
        assert h1 == h2

    def test_different_inputs_differ(self):
        h1 = compute_prompt_hash("hello")
        h2 = compute_prompt_hash("world")
        assert h1 != h2

    def test_length(self):
        h = compute_prompt_hash("test")
        assert len(h) == 16

    def test_hex_chars(self):
        h = compute_prompt_hash("test prompt")
        assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# D.1 TESTS: assemble_emission
# =============================================================================


class TestAssembleEmission:
    """Tests for assemble_emission()."""

    def test_empty(self, empty_interface_map):
        result = assemble_emission([], empty_interface_map, False)
        assert result.total_nodes == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.generated_code == {}
        assert result.l2_context_injected is False

    def test_single_batch(self, empty_interface_map):
        ne = NodeEmission("A", "service", "class A: pass", True, None, "h", "v")
        be = BatchEmission(0, (ne,), 1, 0)
        result = assemble_emission([be], empty_interface_map, False)
        assert result.total_nodes == 1
        assert result.success_count == 1
        assert "A" in result.generated_code

    def test_mixed_success_failure(self, empty_interface_map):
        ne1 = NodeEmission("A", "service", "class A: pass", True, None, "h", "v")
        ne2 = NodeEmission("B", "service", "", False, "error", "h2", "v")
        be = BatchEmission(0, (ne1, ne2), 1, 1)
        result = assemble_emission([be], empty_interface_map, False)
        assert result.total_nodes == 2
        assert result.success_count == 1
        assert result.failure_count == 1
        assert "A" in result.generated_code
        assert "B" not in result.generated_code

    def test_verification_called(self, simple_interface_map):
        ne1 = NodeEmission("UserService", "service", "class UserService:\n    def auth_token(self): return self.auth_service", True, None, "h", "v")
        ne2 = NodeEmission("AuthService", "service", "class AuthService:\n    def auth_token(self): return 'token'", True, None, "h2", "v")
        be = BatchEmission(0, (ne1, ne2), 2, 0)
        result = assemble_emission([be], simple_interface_map, False)
        assert "total_contracts" in result.verification_report
        assert result.verification_report["total_contracts"] == 1

    def test_l2_flag(self, empty_interface_map):
        result = assemble_emission([], empty_interface_map, True)
        assert result.l2_context_injected is True

    def test_pass_rate(self, empty_interface_map):
        result = assemble_emission([], empty_interface_map, False)
        # Empty = pass_rate 1.0 (no contracts to fail)
        assert result.pass_rate == 1.0

    def test_derived_from(self, empty_interface_map):
        result = assemble_emission([], empty_interface_map, False)
        assert result.derived_from == EMISSION_VERSION

    def test_multiple_batches(self, empty_interface_map):
        ne1 = NodeEmission("A", "service", "class A: pass", True, None, "h", "v")
        ne2 = NodeEmission("B", "service", "class B: pass", True, None, "h2", "v")
        be1 = BatchEmission(0, (ne1,), 1, 0)
        be2 = BatchEmission(1, (ne2,), 1, 0)
        result = assemble_emission([be1, be2], empty_interface_map, False)
        assert result.total_nodes == 2
        assert len(result.batch_emissions) == 2
        assert "A" in result.generated_code
        assert "B" in result.generated_code


# =============================================================================
# D.1 TESTS: Serialization Round-Trip
# =============================================================================


class TestSerialization:
    """Tests for serialize/deserialize_emission_result."""

    def test_full_round_trip(self, empty_interface_map):
        ne = NodeEmission("A", "service", "class A: pass", True, None, "abc123", EMISSION_VERSION)
        be = BatchEmission(0, (ne,), 1, 0)
        original = assemble_emission([be], empty_interface_map, True)

        data = serialize_emission_result(original)
        restored = deserialize_emission_result(data)

        assert restored.total_nodes == original.total_nodes
        assert restored.success_count == original.success_count
        assert restored.failure_count == original.failure_count
        assert restored.pass_rate == original.pass_rate
        assert restored.l2_context_injected == original.l2_context_injected
        assert restored.derived_from == original.derived_from
        assert restored.generated_code == original.generated_code
        assert len(restored.batch_emissions) == len(original.batch_emissions)

    def test_empty_round_trip(self, empty_interface_map):
        original = assemble_emission([], empty_interface_map, False)
        data = serialize_emission_result(original)
        restored = deserialize_emission_result(data)
        assert restored.total_nodes == 0
        assert restored.generated_code == {}

    def test_failed_emissions_round_trip(self, empty_interface_map):
        ne = NodeEmission("A", "service", "", False, "LLM timeout", "h", EMISSION_VERSION)
        be = BatchEmission(0, (ne,), 0, 1)
        original = assemble_emission([be], empty_interface_map, False)

        data = serialize_emission_result(original)
        restored = deserialize_emission_result(data)

        assert restored.failure_count == 1
        assert restored.batch_emissions[0].emissions[0].error == "LLM timeout"
        assert restored.batch_emissions[0].emissions[0].success is False

    def test_json_serializable(self, empty_interface_map):
        ne = NodeEmission("A", "service", "class A: pass", True, None, "h", EMISSION_VERSION)
        be = BatchEmission(0, (ne,), 1, 0)
        result = assemble_emission([be], empty_interface_map, False)
        data = serialize_emission_result(result)
        # Must be JSON-serializable
        json_str = json.dumps(data)
        assert json_str


# =============================================================================
# D.2 TESTS: Engine emit_code()
# =============================================================================


class TestEngineEmitCode:
    """Tests for MotherlabsEngine.emit_code()."""

    def _make_engine(self, mock_client=None):
        from core.engine import MotherlabsEngine
        from core.llm import MockClient
        client = mock_client or MockClient()
        return MotherlabsEngine(
            llm_client=client,
            auto_store=False,
        )

    def _make_blueprint(self):
        return {
            "components": [
                {"name": "UserService", "type": "service", "description": "User management"},
                {"name": "AuthService", "type": "service", "description": "Authentication"},
            ],
            "relationships": [
                {"from": "UserService", "to": "AuthService", "type": "depends_on"},
            ],
            "constraints": [],
        }

    def _make_interface_map(self):
        contract = InterfaceContract(
            node_a="UserService",
            node_b="AuthService",
            relationship_type="depends_on",
            relationship_description="UserService depends on AuthService",
            data_flows=(
                DataFlow(name="auth_token", type_hint="str", direction="B_to_A", derived_from="test"),
            ),
            constraints=(),
            fragility=0.3,
            confidence=0.9,
            directionality="A_depends_on_B",
            derived_from="test",
        )
        return InterfaceMap(
            contracts=(contract,),
            unmatched_relationships=(),
            extraction_confidence=0.9,
            derived_from="test",
        )

    def test_basic_emit(self):
        engine = self._make_engine()
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        result = engine.emit_code(blueprint, interface_map=imap)
        assert isinstance(result, EmissionResult)
        assert result.total_nodes == 2
        assert result.derived_from == EMISSION_VERSION

    def test_llm_call_count(self):
        from core.llm import MockClient
        client = MockClient()
        engine = self._make_engine(client)
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        engine.emit_code(blueprint, interface_map=imap)
        # Should call LLM once per component
        assert client.call_count == 2

    def test_batch_ordering_preserved(self):
        engine = self._make_engine()
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        result = engine.emit_code(blueprint, interface_map=imap)
        # Batches should be in order
        for i, be in enumerate(result.batch_emissions):
            assert be.batch_index == i

    def test_l2_injection(self):
        engine = self._make_engine()
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        l2 = L2Synthesis(
            shared_vocabulary=(("user", 3),),
            cross_cutting_components=(),
            relationship_patterns=(),
            interface_gaps=(),
            integration_constraints=(),
            pattern_count=1,
            synthesis_confidence=0.5,
        )
        result = engine.emit_code(blueprint, interface_map=imap, l2_synthesis=l2)
        assert result.l2_context_injected is True

    def test_error_handling(self):
        """Failed LLM calls should not crash — graceful degradation."""
        from core.llm import BaseLLMClient

        class FailingClient(BaseLLMClient):
            def __init__(self):
                super().__init__(deterministic=True)
                self.call_count = 0

            def complete(self, messages, **kwargs):
                self.call_count += 1
                raise RuntimeError("LLM down")

            def complete_with_system(self, system_prompt, user_content, **kwargs):
                self.call_count += 1
                raise RuntimeError("LLM down")

        engine = self._make_engine(FailingClient())
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        result = engine.emit_code(blueprint, interface_map=imap)
        assert result.failure_count == 2
        assert result.success_count == 0

    def test_result_shape(self):
        engine = self._make_engine()
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        result = engine.emit_code(blueprint, interface_map=imap)
        assert hasattr(result, 'batch_emissions')
        assert hasattr(result, 'generated_code')
        assert hasattr(result, 'verification_report')
        assert hasattr(result, 'total_nodes')
        assert hasattr(result, 'pass_rate')
        assert hasattr(result, 'timestamp')

    def test_empty_blueprint(self):
        engine = self._make_engine()
        result = engine.emit_code({}, interface_map=InterfaceMap((), (), 1.0, "test"))
        assert result.total_nodes == 0

    def test_retry_on_failure(self):
        """With retry_failed=True, failed nodes get retried."""
        from core.llm import BaseLLMClient

        class FailOnceClient(BaseLLMClient):
            def __init__(self):
                super().__init__(deterministic=True)
                self.call_count = 0
                self.calls_per_node = {}

            def complete(self, messages, **kwargs):
                self.call_count += 1
                return "[Mock]"

            def complete_with_system(self, system_prompt, user_content, **kwargs):
                self.call_count += 1
                # Fail first call, succeed on retry
                key = user_content[:30]
                self.calls_per_node[key] = self.calls_per_node.get(key, 0) + 1
                if self.calls_per_node[key] == 1:
                    raise RuntimeError("transient failure")
                return "```python\nclass Recovered: pass\n```"

        engine = self._make_engine(FailOnceClient())
        blueprint = self._make_blueprint()
        imap = self._make_interface_map()
        config = EmissionConfig(retry_failed=True, max_retries=1)
        result = engine.emit_code(blueprint, interface_map=imap, config=config)
        # After retry, should recover
        assert result.success_count >= 1


# =============================================================================
# LEAF MODULE CONSTRAINT
# =============================================================================


class TestLeafModuleConstraint:
    """Verify agent_emission.py is a LEAF MODULE."""

    def test_no_engine_imports(self):
        import ast
        from pathlib import Path
        source = Path("core/agent_emission.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "core.engine" not in node.module, "LEAF MODULE must not import engine"
                    assert "core.protocol" not in node.module, "LEAF MODULE must not import protocol"
                    assert "core.pipeline" not in node.module, "LEAF MODULE must not import pipeline"

    def test_import_succeeds(self):
        from core.agent_emission import EmissionResult
        assert EmissionResult is not None


# =============================================================================
# D.3 TESTS: Strategy 0 Code Extraction (truncated fences)
# =============================================================================


class TestExtractCodeTruncatedFence:
    """Tests for Strategy 0 — truncated code fence handling."""

    def test_truncated_python_fence(self):
        """Opening ```python fence, no closing ``` → clean extraction."""
        response = "```python\nclass Foo:\n    pass\n"
        code = extract_code_from_response(response)
        assert "class Foo:" in code
        assert "```" not in code

    def test_complete_fence_still_works(self):
        """Both markers → still works via Strategy 0."""
        response = "```python\nclass Bar:\n    x: int = 0\n```"
        code = extract_code_from_response(response)
        assert "class Bar:" in code
        assert "```" not in code

    def test_truncated_generic_fence(self):
        """Generic ``` fence, no closing → extracted if Python-like."""
        response = "```\nimport os\ndef hello():\n    print('hi')\n"
        code = extract_code_from_response(response)
        assert "def hello():" in code
        assert "```" not in code


# =============================================================================
# D.4 TESTS: dedup_emitted_classes
# =============================================================================


class TestDedupEmittedClasses:
    """Tests for dedup_emitted_classes() — Fix 4."""

    def test_no_duplicates_unchanged(self):
        from core.agent_emission import dedup_emitted_classes
        code = {
            "UserService": "class UserService:\n    pass\n",
            "AuthService": "class AuthService:\n    pass\n",
        }
        result, log = dedup_emitted_classes(code)
        assert result == code
        assert log == []

    def test_duplicate_class_deduped(self):
        from core.agent_emission import dedup_emitted_classes
        code = {
            "ModuleA": "class Signal:\n    name: str = ''\n\nclass ModuleA:\n    pass\n",
            "ModuleB": "class Signal:\n    name: str = ''\n    value: int = 0\n\nclass ModuleB:\n    pass\n",
        }
        result, log = dedup_emitted_classes(code)
        assert len(log) == 1
        # The shorter definition should be replaced with an import
        assert "from .module_" in result["ModuleA"] or "from .module_" in result["ModuleB"]

    def test_canonical_is_richest(self):
        """Most body nodes → canonical."""
        from core.agent_emission import dedup_emitted_classes
        # ModuleB has richer Signal (3 fields vs 1)
        code = {
            "ModuleA": "class Signal:\n    name: str = ''\n\nclass ModuleA:\n    pass\n",
            "ModuleB": "class Signal:\n    name: str = ''\n    value: int = 0\n    kind: str = 'default'\n\nclass ModuleB:\n    pass\n",
        }
        result, log = dedup_emitted_classes(code)
        # ModuleB should keep Signal, ModuleA gets import
        assert "class Signal:" in result["ModuleB"]
        assert "from .module_b import Signal" in result["ModuleA"]

    def test_unparseable_file_skipped(self):
        """Syntax error files → unchanged."""
        from core.agent_emission import dedup_emitted_classes
        code = {
            "Good": "class Signal:\n    pass\n",
            "Bad": "class Signal\n    pass\n",  # Missing colon
        }
        result, log = dedup_emitted_classes(code)
        # Bad file can't be parsed, so no dedup happens
        assert result == code
        assert log == []

    def test_multiple_duplicates(self):
        """Two classes each duped → both deduped."""
        from core.agent_emission import dedup_emitted_classes
        code = {
            "FileA": "class Signal:\n    a: int = 1\n\nclass Event:\n    b: str = ''\n",
            "FileB": "class Signal:\n    a: int = 1\n    extra: bool = True\n\nclass Event:\n    b: str = ''\n    extra: bool = True\n",
        }
        result, log = dedup_emitted_classes(code)
        assert len(log) == 2  # Both Signal and Event deduped

    def test_import_uses_snake_case(self):
        """Import uses snake_case module name."""
        from core.agent_emission import dedup_emitted_classes
        code = {
            "ConflictOracle": "class Signal:\n    pass\n\nclass ConflictOracle:\n    pass\n",
            "GovernorAgent": "class Signal:\n    name: str = ''\n    value: int = 0\n\nclass GovernorAgent:\n    pass\n",
        }
        result, log = dedup_emitted_classes(code)
        # ConflictOracle should have import from governor_agent (richer)
        assert "from .governor_agent import Signal" in result["ConflictOracle"]
