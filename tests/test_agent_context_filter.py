"""Tests for agent context filtering (Build 4) — structural blindness enforcement."""

import pytest
from unittest.mock import MagicMock, patch

# Import core first to avoid circular import
from core.protocol import Message, MessageType, SharedState, ConfidenceVector
from agents.base import (
    AgentRole, BaseAgent, LLMAgent, AgentCallResult,
    _BEHAVIORAL_VOCABULARY, _STRUCTURAL_VOCABULARY,
)


def _make_state(**kwargs):
    """Create a minimal SharedState for testing."""
    state = SharedState()
    state.known["input"] = kwargs.get("input", "Build a task manager")
    state.known["intent"] = kwargs.get("intent", {"core_need": "task manager", "domain": "software"})
    for msg in kwargs.get("history", []):
        state.history.append(msg)
    return state


def _make_message(sender, content, msg_type=MessageType.PROPOSITION):
    return Message(sender=sender, content=content, message_type=msg_type)


class TestAgentRoleEnum:
    """AgentRole enum exists with correct values."""

    def test_structural_role(self):
        assert AgentRole.STRUCTURAL.value == "structural"

    def test_behavioral_role(self):
        assert AgentRole.BEHAVIORAL.value == "behavioral"

    def test_integrative_role(self):
        assert AgentRole.INTEGRATIVE.value == "integrative"

    def test_evaluative_role(self):
        assert AgentRole.EVALUATIVE.value == "evaluative"

    def test_orchestrative_role(self):
        assert AgentRole.ORCHESTRATIVE.value == "orchestrative"


class TestFilteredContext:
    """Structural agents filter behavioral messages and vice versa."""

    def test_structural_excludes_process_messages(self):
        """Entity agent should not see Process messages."""
        state = _make_state(history=[
            _make_message("Entity", "Found User entity with attributes"),
            _make_message("Process", "Authentication workflow has 3 steps"),
            _make_message("Entity", "Database schema has 5 tables"),
        ])
        agent = LLMAgent(
            name="Entity", perspective="structure",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.STRUCTURAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "Authentication workflow" not in ctx
        assert "Found User entity" in ctx
        assert "Database schema" in ctx

    def test_behavioral_excludes_entity_messages(self):
        """Process agent should not see Entity messages."""
        state = _make_state(history=[
            _make_message("Entity", "User entity with name attribute"),
            _make_message("Process", "Login flow triggers session creation"),
        ])
        agent = LLMAgent(
            name="Process", perspective="behavior",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.BEHAVIORAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "User entity with name" not in ctx
        assert "Login flow" in ctx

    def test_structural_strips_behavioral_vocabulary(self):
        """Structural agent context should strip behavioral vocabulary."""
        state = _make_state(history=[
            _make_message("Intent", "The system workflow handles pipeline orchestration"),
        ])
        agent = LLMAgent(
            name="Entity", perspective="structure",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.STRUCTURAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "workflow" not in ctx
        assert "pipeline" not in ctx
        assert "[...]" in ctx

    def test_behavioral_strips_structural_vocabulary(self):
        """Behavioral agent context should strip structural vocabulary."""
        state = _make_state(history=[
            _make_message("Intent", "The entity schema defines the data model interface"),
        ])
        agent = LLMAgent(
            name="Process", perspective="behavior",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.BEHAVIORAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "schema" not in ctx
        assert "data model" not in ctx

    def test_structural_hides_behavioral_confidence(self):
        """Structural agent should not see confidence.behavioral."""
        state = _make_state()
        state.confidence.structural = 0.5
        state.confidence.behavioral = 0.8
        agent = LLMAgent(
            name="Entity", perspective="structure",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.STRUCTURAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "S=0.5" in ctx
        assert "B=0.8" not in ctx

    def test_behavioral_hides_structural_confidence(self):
        """Behavioral agent should not see confidence.structural."""
        state = _make_state()
        state.confidence.structural = 0.5
        state.confidence.behavioral = 0.8
        agent = LLMAgent(
            name="Process", perspective="behavior",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.BEHAVIORAL,
        )
        ctx = agent._build_filtered_context(state)
        assert "B=0.8" in ctx
        assert "S=0.5" not in ctx


class TestRoleBasedConfidence:
    """Confidence boost dispatches on role, not name."""

    def test_structural_role_sets_structural_dimension(self):
        """Agent with STRUCTURAL role reports structural dimension."""
        agent = LLMAgent(
            name="CustomAgent", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.STRUCTURAL,
        )
        state = _make_state()
        boost, dim = agent._compute_confidence_boost(
            state, "INSIGHT: Found a key component", MessageType.PROPOSITION
        )
        assert dim == "structural"

    def test_behavioral_role_sets_behavioral_dimension(self):
        """Agent with BEHAVIORAL role reports behavioral dimension."""
        agent = LLMAgent(
            name="CustomAgent", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.BEHAVIORAL,
        )
        state = _make_state()
        boost, dim = agent._compute_confidence_boost(
            state, "INSIGHT: Found a workflow", MessageType.PROPOSITION
        )
        assert dim == "behavioral"

    def test_name_fallback_entity(self):
        """Without role, falls back to name-based dispatch."""
        agent = LLMAgent(
            name="Entity", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
        )
        state = _make_state()
        boost, dim = agent._compute_confidence_boost(
            state, "test response", MessageType.PROPOSITION
        )
        assert dim == "structural"

    def test_name_fallback_process(self):
        """Without role, falls back to name-based dispatch."""
        agent = LLMAgent(
            name="Process", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
        )
        state = _make_state()
        boost, dim = agent._compute_confidence_boost(
            state, "test response", MessageType.PROPOSITION
        )
        assert dim == "behavioral"


class TestNonFilteredRoles:
    """Non-structural/behavioral roles get unfiltered context."""

    def test_integrative_sees_all(self):
        """Integrative role sees all messages."""
        state = _make_state(history=[
            _make_message("Entity", "User entity found"),
            _make_message("Process", "Login flow defined"),
        ])
        agent = LLMAgent(
            name="Synthesis", perspective="integration",
            system_prompt="test", llm_client=MagicMock(),
            role=AgentRole.INTEGRATIVE,
        )
        ctx = agent._build_filtered_context(state)
        assert "User entity" in ctx
        assert "Login flow" in ctx

    def test_no_role_sees_all(self):
        """Agent with no role sees all messages."""
        state = _make_state(history=[
            _make_message("Entity", "User entity found"),
            _make_message("Process", "Login flow defined"),
        ])
        agent = LLMAgent(
            name="Test", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
        )
        ctx = agent._build_filtered_context(state)
        assert "User entity" in ctx
        assert "Login flow" in ctx


class TestBackwardCompat:
    """Existing tests should not break."""

    def test_role_defaults_to_none(self):
        """Default role is None."""
        agent = LLMAgent(
            name="Test", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
        )
        assert agent.role is None

    def test_base_agent_role_defaults_to_none(self):
        """BaseAgent default role is None."""
        class TestAgent(BaseAgent):
            def run(self, state, input_msg=None):
                pass
        agent = TestAgent("Test", "test")
        assert agent.role is None
