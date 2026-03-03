"""Tests for agent output validation (Build 5)."""

import pytest
from unittest.mock import MagicMock

# Import core first to avoid circular import
from core.protocol import Message, MessageType, SharedState
from agents.base import (
    AgentRole, LLMAgent, OutputValidation,
    _STRUCTURAL_OUTPUT_VOCAB, _BEHAVIORAL_OUTPUT_VOCAB,
)


def _make_agent(role):
    return LLMAgent(
        name="TestAgent", perspective="test",
        system_prompt="test", llm_client=MagicMock(),
        role=role,
    )


class TestStructuralOutputValidation:
    """Structural agent output should contain entity/schema vocabulary."""

    def test_structural_output_high_score(self):
        """Rich structural output scores high."""
        agent = _make_agent(AgentRole.STRUCTURAL)
        result = agent.validate_agent_output(
            "The User entity has a name attribute and email field. "
            "The schema defines a record type with an identifier property. "
            "The component hierarchy uses composition."
        )
        assert result.role_match_score > 0.3
        assert len(result.vocabulary_found) > 3

    def test_behavioral_output_in_structural_agent(self):
        """Behavioral output in structural agent scores lower."""
        agent = _make_agent(AgentRole.STRUCTURAL)
        structural_result = agent.validate_agent_output(
            "The User entity has a name attribute and email field. "
            "The schema defines record types."
        )
        behavioral_result = agent.validate_agent_output(
            "The workflow processes events in sequence. "
            "Each step triggers the next handler."
        )
        assert structural_result.role_match_score > behavioral_result.role_match_score

    def test_empty_output(self):
        """Empty output scores zero."""
        agent = _make_agent(AgentRole.STRUCTURAL)
        result = agent.validate_agent_output("")
        assert result.role_match_score == 0.0


class TestBehavioralOutputValidation:
    """Behavioral agent output should contain process/flow vocabulary."""

    def test_behavioral_output_high_score(self):
        """Rich behavioral output scores high."""
        agent = _make_agent(AgentRole.BEHAVIORAL)
        result = agent.validate_agent_output(
            "The authentication process follows a step-by-step sequence. "
            "Each transition triggers the next event handler. "
            "The workflow dispatches actions through the pipeline."
        )
        assert result.role_match_score > 0.3
        assert len(result.vocabulary_found) > 3

    def test_structural_output_in_behavioral_agent(self):
        """Structural output in behavioral agent scores lower."""
        agent = _make_agent(AgentRole.BEHAVIORAL)
        behavioral_result = agent.validate_agent_output(
            "The process flow triggers transitions and dispatches events."
        )
        structural_result = agent.validate_agent_output(
            "The entity schema defines attributes and field types."
        )
        assert behavioral_result.role_match_score > structural_result.role_match_score


class TestNonFilteredRoles:
    """Non-structural/behavioral roles always pass."""

    def test_integrative_always_passes(self):
        agent = _make_agent(AgentRole.INTEGRATIVE)
        result = agent.validate_agent_output("anything at all")
        assert result.role_match_score == 1.0

    def test_evaluative_always_passes(self):
        agent = _make_agent(AgentRole.EVALUATIVE)
        result = agent.validate_agent_output("anything at all")
        assert result.role_match_score == 1.0

    def test_no_role_always_passes(self):
        agent = LLMAgent(
            name="Test", perspective="test",
            system_prompt="test", llm_client=MagicMock(),
        )
        result = agent.validate_agent_output("anything at all")
        assert result.role_match_score == 1.0


class TestOutputValidationDataclass:
    """OutputValidation is frozen and has correct structure."""

    def test_frozen(self):
        v = OutputValidation(
            role_match_score=0.5,
            vocabulary_found=("entity", "schema"),
        )
        with pytest.raises(AttributeError):
            v.role_match_score = 0.9

    def test_vocabulary_missing_populated(self):
        agent = _make_agent(AgentRole.STRUCTURAL)
        result = agent.validate_agent_output("no relevant vocabulary here at all")
        assert len(result.vocabulary_missing) > 0
        assert result.role_match_score < 0.3
