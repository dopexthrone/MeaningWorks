"""Tests for the Agent System domain adapter."""

import pytest

from core.domain_adapter import (
    DomainAdapter,
    RuntimeCapabilities,
    VocabularyMap,
    PromptTemplates,
    ClassificationConfig,
    VerificationOverrides,
    MaterializationConfig,
)
from core.adapter_registry import get_adapter, list_adapters, clear_registry
from adapters.agent_system import (
    AGENT_SYSTEM_ADAPTER,
    AGENT_SYSTEM_VOCABULARY,
    AGENT_SYSTEM_CLASSIFICATION,
    AGENT_SYSTEM_PROMPTS,
    AGENT_SYSTEM_VERIFICATION,
    AGENT_SYSTEM_MATERIALIZATION,
    AGENT_SYSTEM_RUNTIME,
)


@pytest.fixture(autouse=True)
def _register_adapters():
    """Ensure all adapters are registered for each test."""
    import adapters  # noqa: F401
    yield
    # Re-register after clear (if any test clears)
    from core.adapter_registry import register_adapter
    from adapters.software import SOFTWARE_ADAPTER
    from adapters.process import PROCESS_ADAPTER
    from adapters.api import API_ADAPTER
    register_adapter(SOFTWARE_ADAPTER)
    register_adapter(PROCESS_ADAPTER)
    register_adapter(API_ADAPTER)
    register_adapter(AGENT_SYSTEM_ADAPTER)


# =============================================================================
# ADAPTER STRUCTURE
# =============================================================================

class TestAgentSystemAdapterStructure:
    """Test adapter is correctly constructed and frozen."""

    def test_adapter_is_domain_adapter(self):
        assert isinstance(AGENT_SYSTEM_ADAPTER, DomainAdapter)

    def test_adapter_name(self):
        assert AGENT_SYSTEM_ADAPTER.name == "agent_system"

    def test_adapter_version(self):
        assert AGENT_SYSTEM_ADAPTER.version == "1.0"

    def test_adapter_is_frozen(self):
        with pytest.raises(AttributeError):
            AGENT_SYSTEM_ADAPTER.name = "modified"

    def test_adapter_has_runtime(self):
        assert AGENT_SYSTEM_ADAPTER.runtime is not None
        assert isinstance(AGENT_SYSTEM_ADAPTER.runtime, RuntimeCapabilities)

    def test_adapter_materialization_is_python(self):
        assert AGENT_SYSTEM_ADAPTER.materialization.output_format == "python"
        assert AGENT_SYSTEM_ADAPTER.materialization.file_extension == ".py"


# =============================================================================
# REGISTRATION
# =============================================================================

class TestAgentSystemRegistration:
    """Test adapter registers and retrieves correctly."""

    def test_adapter_in_list(self):
        assert "agent_system" in list_adapters()

    def test_adapter_retrieval(self):
        adapter = get_adapter("agent_system")
        assert adapter.name == "agent_system"
        assert adapter is AGENT_SYSTEM_ADAPTER

    def test_coexists_with_other_adapters(self):
        names = list_adapters()
        assert "software" in names
        assert "process" in names
        assert "api" in names
        assert "agent_system" in names
        assert len(names) >= 4


# =============================================================================
# VOCABULARY
# =============================================================================

class TestAgentSystemVocabulary:
    """Test vocabulary covers agent/tool/state/handler types."""

    def test_entity_types_non_empty(self):
        assert len(AGENT_SYSTEM_VOCABULARY.entity_types) > 0

    def test_process_types_non_empty(self):
        assert len(AGENT_SYSTEM_VOCABULARY.process_types) > 0

    def test_has_agent_type_keywords(self):
        assert "agent" in AGENT_SYSTEM_VOCABULARY.type_keywords
        assert "agent" in AGENT_SYSTEM_VOCABULARY.type_keywords["agent"]

    def test_has_tool_type_keywords(self):
        assert "tool" in AGENT_SYSTEM_VOCABULARY.type_keywords
        assert "tool" in AGENT_SYSTEM_VOCABULARY.type_keywords["tool"]

    def test_has_state_store_type_keywords(self):
        assert "state_store" in AGENT_SYSTEM_VOCABULARY.type_keywords

    def test_has_message_handler_type_keywords(self):
        assert "message_handler" in AGENT_SYSTEM_VOCABULARY.type_keywords

    def test_has_skill_type_keywords(self):
        assert "skill" in AGENT_SYSTEM_VOCABULARY.type_keywords

    def test_relationship_flows_cover_agent_domain(self):
        flows = AGENT_SYSTEM_VOCABULARY.relationship_flows
        assert "delegates_to" in flows
        assert "uses_tool" in flows
        assert "reads_state" in flows
        assert "writes_state" in flows
        assert "handles" in flows

    def test_interface_types_set(self):
        assert len(AGENT_SYSTEM_VOCABULARY.interface_types) > 0

    def test_type_hints_for_agent_concepts(self):
        hints = AGENT_SYSTEM_VOCABULARY.type_hints
        assert "message" in hints
        assert "event" in hints


# =============================================================================
# CLASSIFICATION
# =============================================================================

class TestAgentSystemClassification:
    """Test classification patterns score agent-domain components."""

    def test_subject_patterns_not_empty(self):
        assert len(AGENT_SYSTEM_CLASSIFICATION.subject_patterns) > 0

    def test_object_patterns_not_empty(self):
        assert len(AGENT_SYSTEM_CLASSIFICATION.object_patterns) > 0

    def test_generic_terms_populated(self):
        assert "data" in AGENT_SYSTEM_CLASSIFICATION.generic_terms
        assert "input" in AGENT_SYSTEM_CLASSIFICATION.generic_terms

    def test_min_name_length(self):
        assert AGENT_SYSTEM_CLASSIFICATION.min_name_length == 3

    def test_subject_patterns_match_agent_verbs(self):
        """Subject patterns should match agent-domain verbs."""
        import re
        pattern = AGENT_SYSTEM_CLASSIFICATION.subject_patterns[0]
        text = "ChatAgent handles user messages"
        regex = pattern.format("ChatAgent")
        assert re.search(regex, text, re.IGNORECASE) is not None


# =============================================================================
# RUNTIME CAPABILITIES
# =============================================================================

class TestAgentSystemRuntime:
    """Test runtime capabilities configuration."""

    def test_has_event_loop(self):
        assert AGENT_SYSTEM_RUNTIME.has_event_loop is True

    def test_has_llm_client(self):
        assert AGENT_SYSTEM_RUNTIME.has_llm_client is True

    def test_has_persistent_state(self):
        assert AGENT_SYSTEM_RUNTIME.has_persistent_state is True

    def test_has_self_recompile(self):
        assert AGENT_SYSTEM_RUNTIME.has_self_recompile is True

    def test_has_tool_execution(self):
        assert AGENT_SYSTEM_RUNTIME.has_tool_execution is True

    def test_event_loop_type_asyncio(self):
        assert AGENT_SYSTEM_RUNTIME.event_loop_type == "asyncio"

    def test_state_backend_sqlite(self):
        assert AGENT_SYSTEM_RUNTIME.state_backend == "sqlite"

    def test_default_port(self):
        assert AGENT_SYSTEM_RUNTIME.default_port == 8080

    def test_tool_allowlist(self):
        assert "web_search" in AGENT_SYSTEM_RUNTIME.tool_allowlist
        assert "file_read" in AGENT_SYSTEM_RUNTIME.tool_allowlist
        assert len(AGENT_SYSTEM_RUNTIME.tool_allowlist) == 4

    def test_runtime_is_frozen(self):
        with pytest.raises(AttributeError):
            AGENT_SYSTEM_RUNTIME.has_event_loop = False

    def test_default_runtime_capabilities_all_false(self):
        """Default RuntimeCapabilities has everything disabled."""
        default = RuntimeCapabilities()
        assert default.has_event_loop is False
        assert default.has_llm_client is False
        assert default.has_persistent_state is False
        assert default.has_self_recompile is False
        assert default.has_tool_execution is False
        assert default.event_loop_type == "none"
        assert default.state_backend == "none"

    def test_can_compile_enabled(self):
        assert AGENT_SYSTEM_RUNTIME.can_compile is True

    def test_can_share_tools_enabled(self):
        assert AGENT_SYSTEM_RUNTIME.can_share_tools is True

    def test_corpus_path_set(self):
        assert AGENT_SYSTEM_RUNTIME.corpus_path == "~/motherlabs/corpus.db"


# =============================================================================
# PROMPTS
# =============================================================================

class TestAgentSystemPrompts:
    """Test prompt templates are populated."""

    def test_intent_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.intent_system_prompt) > 50

    def test_persona_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.persona_system_prompt) > 50

    def test_entity_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.entity_system_prompt) > 50

    def test_process_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.process_system_prompt) > 50

    def test_synthesis_prompt_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.synthesis_system_prompt) > 50

    def test_emission_preamble_non_empty(self):
        assert len(AGENT_SYSTEM_PROMPTS.emission_preamble) > 50

    def test_emission_preamble_mentions_async(self):
        assert "async" in AGENT_SYSTEM_PROMPTS.emission_preamble

    def test_intent_prompt_mentions_agent(self):
        assert "agent" in AGENT_SYSTEM_PROMPTS.intent_system_prompt.lower()


# =============================================================================
# SOFTWARE ADAPTER UNCHANGED
# =============================================================================

class TestSoftwareAdapterUnchanged:
    """Verify existing adapters have no runtime by default."""

    def test_software_adapter_no_runtime(self):
        adapter = get_adapter("software")
        assert adapter.runtime is None

    def test_process_adapter_no_runtime(self):
        adapter = get_adapter("process")
        assert adapter.runtime is None

    def test_api_adapter_no_runtime(self):
        adapter = get_adapter("api")
        assert adapter.runtime is None
