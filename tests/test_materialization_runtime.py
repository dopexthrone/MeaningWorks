"""Tests for materialization runtime contract extensions."""

import pytest

from core.domain_adapter import RuntimeCapabilities
from core.materialization import (
    build_node_prompt_with_context,
    _build_runtime_contract,
    NodePrompt,
)
from core.interface_schema import InterfaceMap


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_COMPONENT = {
    "name": "ChatAgent",
    "type": "agent",
    "description": "Handles user messages",
    "methods": [{"name": "handle", "parameters": [], "return_type": "dict"}],
}

SAMPLE_BLUEPRINT = {
    "domain": "agent_system",
    "components": [SAMPLE_COMPONENT],
    "constraints": [],
    "relationships": [],
}

EMPTY_INTERFACE_MAP = InterfaceMap(
    contracts=(), unmatched_relationships=(), extraction_confidence=1.0, derived_from="test",
)

FULL_RUNTIME = RuntimeCapabilities(
    has_event_loop=True,
    has_llm_client=True,
    has_persistent_state=True,
    has_self_recompile=True,
    has_tool_execution=True,
    event_loop_type="asyncio",
    state_backend="sqlite",
    tool_allowlist=("web_search",),
)


# =============================================================================
# RUNTIME CONTRACT BUILDER
# =============================================================================

class TestBuildRuntimeContract:
    """Test _build_runtime_contract helper."""

    def test_full_contract(self):
        text = _build_runtime_contract(FULL_RUNTIME)
        assert "## Runtime Contract" in text
        assert "self.state" in text
        assert "self.llm" in text
        assert "self.tools" in text
        assert "self.emit" in text
        assert "handle" in text

    def test_empty_contract(self):
        text = _build_runtime_contract(RuntimeCapabilities())
        assert text == ""

    def test_only_state(self):
        cap = RuntimeCapabilities(has_persistent_state=True)
        text = _build_runtime_contract(cap)
        assert "self.state" in text
        assert "self.llm" not in text

    def test_only_llm(self):
        cap = RuntimeCapabilities(has_llm_client=True)
        text = _build_runtime_contract(cap)
        assert "self.llm" in text
        assert "self.state" not in text

    def test_only_tools(self):
        cap = RuntimeCapabilities(has_tool_execution=True)
        text = _build_runtime_contract(cap)
        assert "self.tools" in text

    def test_async_instruction(self):
        cap = RuntimeCapabilities(has_event_loop=True, event_loop_type="asyncio")
        text = _build_runtime_contract(cap)
        assert "async" in text.lower()


# =============================================================================
# NODE PROMPT WITH RUNTIME
# =============================================================================

class TestNodePromptWithRuntime:
    """Test build_node_prompt_with_context with runtime_capabilities."""

    def test_adds_runtime_section(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "Runtime Contract" in prompt.prompt_text

    def test_no_runtime_no_section(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
        )
        assert "Runtime Contract" not in prompt.prompt_text

    def test_runtime_none_no_section(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=None,
        )
        assert "Runtime Contract" not in prompt.prompt_text

    def test_disabled_runtime_no_section(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=RuntimeCapabilities(),
        )
        assert "Runtime Contract" not in prompt.prompt_text

    def test_runtime_section_contains_state_store(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "StateStore" in prompt.prompt_text

    def test_runtime_with_layer_instruction(self):
        """Runtime contract should coexist with layer instructions."""
        from core.materialization import EmissionLayer
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            layer=EmissionLayer.IMPLEMENTATIONS,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "Runtime Contract" in prompt.prompt_text
        assert "Layer Instruction" in prompt.prompt_text

    def test_runtime_with_prior_code(self):
        """Runtime contract should coexist with prior layer code."""
        prior = {"BaseModel": "class BaseModel:\n    pass\n"}
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            prior_layer_code=prior,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert "Runtime Contract" in prompt.prompt_text
        assert "Imported Code" in prompt.prompt_text

    def test_prompt_text_is_string(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert isinstance(prompt.prompt_text, str)
        assert isinstance(prompt, NodePrompt)

    def test_original_fields_preserved(self):
        prompt = build_node_prompt_with_context(
            SAMPLE_COMPONENT,
            SAMPLE_BLUEPRINT,
            EMPTY_INTERFACE_MAP,
            runtime_capabilities=FULL_RUNTIME,
        )
        assert prompt.component_name == "ChatAgent"
        assert prompt.component_type == "agent"
        assert prompt.description == "Handles user messages"
