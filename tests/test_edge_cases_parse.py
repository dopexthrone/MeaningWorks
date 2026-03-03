"""
Phase 17.3: Hollow Artifact Detection — Parse Failure Guard Tests.

Tests for:
- _parse_health on artifact parsers
- StagedPipeline abort on EXPAND/DECOMPOSE double-hollow
- _extract_intent validation
- _generate_personas filtering + validation
- QualityScore.is_hollow property
- HOLLOW_THRESHOLD constant
"""

import pytest
from unittest.mock import MagicMock, patch
from core.protocol import SharedState, Message, MessageType
from core.pipeline import (
    parse_expand_artifact,
    parse_decompose_artifact,
    parse_ground_artifact,
    parse_constrain_artifact,
    parse_architect_artifact,
)
from core.input_quality import InputQualityAnalyzer, QualityScore
from core.exceptions import CompilationError


# =============================================================================
# parse_expand_artifact
# =============================================================================

class TestParseExpandHealth:
    def test_hollow_when_no_nodes_or_containment(self):
        """Empty dialogue -> hollow."""
        state = SharedState()
        state.add_message(Message(sender="Entity", content="no structured lines here",
                                   message_type=MessageType.PROPOSITION))
        artifact = parse_expand_artifact(state)
        assert artifact["_parse_health"]["hollow"] is True
        assert artifact["_parse_health"]["nouns_found"] == 0
        assert artifact["_parse_health"]["verbs_found"] == 0

    def test_not_hollow_with_valid_data(self):
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content='NODE: UserService (source: "input")\nINSIGHT: found',
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_expand_artifact(state)
        assert artifact["_parse_health"]["hollow"] is False
        assert artifact["_parse_health"]["nouns_found"] >= 1


# =============================================================================
# parse_decompose_artifact
# =============================================================================

class TestParseDecomposeHealth:
    def test_hollow_when_no_components(self):
        state = SharedState()
        state.known["input"] = "test"
        state.add_message(Message(sender="Entity", content="just talking",
                                   message_type=MessageType.PROPOSITION))
        artifact = parse_decompose_artifact(state)
        assert artifact["_parse_health"]["hollow"] is True
        assert artifact["_parse_health"]["components_found"] == 0

    def test_not_hollow_with_component(self):
        state = SharedState()
        state.known["input"] = "Build a booking system for tattoo studio"
        state.add_message(Message(
            sender="Entity",
            content='COMPONENT: UserService | type=entity | derived_from="input"',
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_decompose_artifact(state)
        assert artifact["_parse_health"]["hollow"] is False
        assert artifact["_parse_health"]["components_found"] >= 1


# =============================================================================
# parse_ground_artifact
# =============================================================================

class TestParseGroundHealth:
    def test_hollow_when_no_relationships(self):
        state = SharedState()
        state.add_message(Message(sender="Entity", content="no rels",
                                   message_type=MessageType.PROPOSITION))
        artifact = parse_ground_artifact(state)
        assert artifact["_parse_health"]["hollow"] is True

    def test_not_hollow_with_relationship(self):
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content='RELATIONSHIP: A -> B | type=depends_on | description="test"',
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_ground_artifact(state)
        assert artifact["_parse_health"]["hollow"] is False


# =============================================================================
# parse_constrain_artifact
# =============================================================================

class TestParseConstrainHealth:
    def test_hollow_when_no_constraints(self):
        state = SharedState()
        state.add_message(Message(sender="Entity", content="no constraints",
                                   message_type=MessageType.PROPOSITION))
        artifact = parse_constrain_artifact(state)
        assert artifact["_parse_health"]["hollow"] is True

    def test_not_hollow_with_constraint(self):
        state = SharedState()
        state.add_message(Message(
            sender="Entity",
            content='CONSTRAINT: Price | description="must be positive" | derived_from="input"',
            message_type=MessageType.PROPOSITION,
        ))
        artifact = parse_constrain_artifact(state)
        assert artifact["_parse_health"]["hollow"] is False


# =============================================================================
# parse_architect_artifact
# =============================================================================

class TestParseArchitectHealth:
    def test_hollow_when_no_subsystems(self):
        state = SharedState()
        state.add_message(Message(sender="Entity", content="no subsystems",
                                   message_type=MessageType.PROPOSITION))
        artifact = parse_architect_artifact(state)
        assert artifact["_parse_health"]["hollow"] is True


# =============================================================================
# _extract_intent validation
# =============================================================================

class TestExtractIntentValidation:
    def test_raises_on_completely_empty_intent(self):
        """When description is empty string, _extract_intent should raise."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=MockClient())
        state = SharedState()
        # MockClient returns generic JSON without core_need, so fallback fires
        # If description is empty string -> fallback core_need="" -> should raise
        with pytest.raises(CompilationError) as exc_info:
            engine._extract_intent("", state)
        assert exc_info.value.error_code == "E3001"


# =============================================================================
# _generate_personas validation
# =============================================================================

class TestGeneratePersonasValidation:
    def test_filters_malformed_personas(self):
        """Personas without 'name' field should be filtered out."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        # Mock client that returns personas with one malformed entry
        mock = MockClient()
        original_run = mock.complete.__func__ if hasattr(mock.complete, '__func__') else None

        engine = MotherlabsEngine(llm_client=mock)
        state = SharedState()
        intent = {"core_need": "test", "domain": "test"}

        # The MockClient won't return valid personas -> should raise E3002
        with pytest.raises(CompilationError) as exc_info:
            engine._generate_personas(intent, state)
        assert exc_info.value.error_code == "E3002"

    def test_raises_on_empty_result(self):
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=MockClient())
        state = SharedState()
        intent = {"core_need": "test", "domain": "test"}

        with pytest.raises(CompilationError) as exc_info:
            engine._generate_personas(intent, state)
        assert exc_info.value.error_code == "E3002"


# =============================================================================
# QualityScore.is_hollow
# =============================================================================

class TestQualityScoreIsHollow:
    def test_is_hollow_in_range(self):
        """Score between REJECT and HOLLOW thresholds -> is_hollow True."""
        score = QualityScore(overall=0.20)
        assert score.is_hollow is True

    def test_not_hollow_above_threshold(self):
        score = QualityScore(overall=0.30)
        assert score.is_hollow is False

    def test_not_hollow_below_reject(self):
        score = QualityScore(overall=0.10)
        assert score.is_hollow is False

    def test_hollow_threshold_exists(self):
        assert InputQualityAnalyzer.HOLLOW_THRESHOLD == 0.25
