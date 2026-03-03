"""
Phase 10.6/10.9: Confidence Signal Tests

Tests for _update_confidence() with:
- Insight-grounded blending (structural/behavioral stabilization)
- Message type boosts (proposition, challenge, agreement, accommodation)
- Content markers (positive, negative, territory-claiming)
- Consistency formula (productive = agreement + accommodation + proposition + challenge-with-insight)
"""

import pytest
from core.protocol import SharedState, Message, MessageType
from agents.base import LLMAgent


class MockLLM:
    pass


def make_entity_agent():
    return LLMAgent(
        name="Entity",
        perspective="Structure",
        system_prompt="Test",
        llm_client=MockLLM()
    )


def make_process_agent():
    return LLMAgent(
        name="Process",
        perspective="Behavior",
        system_prompt="Test",
        llm_client=MockLLM()
    )


def make_generic_agent():
    return LLMAgent(
        name="Test",
        perspective="Testing",
        system_prompt="Test",
        llm_client=MockLLM()
    )


def _add_entity_insight(state, text="Analysis", insight="X = Y"):
    """Helper: add an Entity message with insight to state history."""
    state.add_message(Message(
        sender="Entity", content=text,
        message_type=MessageType.PROPOSITION, insight=insight
    ))


def _add_process_insight(state, text="Flow analysis", insight="A -> B"):
    """Helper: add a Process message with insight to state history."""
    state.add_message(Message(
        sender="Process", content=text,
        message_type=MessageType.PROPOSITION, insight=insight
    ))


class TestPropositionWithInsight:
    """PROPOSITION + INSIGHT: line should boost confidence."""

    def test_entity_structural_boost(self):
        """Entity proposition with insight should boost structural."""
        agent = make_entity_agent()
        state = SharedState()
        response = "Analyzing the system.\nINSIGHT: User = identity + credentials"
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # No prior insights in history, grounded=0, accumulated=0.06
        # blended = 0*0.5 + 0.06*0.5 = 0.03
        assert state.confidence.structural == pytest.approx(0.03, abs=0.01)

    def test_process_behavioral_boost(self):
        """Process proposition with insight should boost behavioral."""
        agent = make_process_agent()
        state = SharedState()
        response = "The flow works like this.\nINSIGHT: Login -> Validate -> Session"
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        assert state.confidence.behavioral == pytest.approx(0.03, abs=0.01)

    def test_cumulative_with_history(self):
        """Multiple turns with insight history should accumulate via grounding."""
        agent = make_entity_agent()
        state = SharedState()
        # Add 4 prior Entity messages with insights to history
        for i in range(4):
            _add_entity_insight(state, f"Turn {i}", f"Component{i} = structure")
        # Now do the 5th turn
        response = "Turn 4 analysis.\nINSIGHT: Component4 = structure"
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # grounded = 4 * 0.08 = 0.32 (4 prior insights in history)
        # accumulated grows from each prior call, but we only called once here
        # so accumulated = 0 + 0.06 = 0.06
        # blended = 0.32*0.5 + 0.06*0.5 = 0.19
        assert state.confidence.structural > 0.15


class TestPropositionWithoutInsight:
    """PROPOSITION without insight should give small boost."""

    def test_entity_small_boost(self):
        """Entity proposition without insight → small boost."""
        agent = make_entity_agent()
        state = SharedState()
        response = "Let me analyze the structure of this system."
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # grounded=0, accumulated=0.02, blended=0.01
        assert state.confidence.structural == pytest.approx(0.01, abs=0.01)

    def test_process_small_boost(self):
        """Process proposition without insight → small boost."""
        agent = make_process_agent()
        state = SharedState()
        response = "The behavioral flow shows several steps."
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        assert state.confidence.behavioral == pytest.approx(0.01, abs=0.01)


class TestChallengeWithInsight:
    """CHALLENGE with insight should be net positive."""

    def test_challenge_with_insight_positive(self):
        """Substantive challenge with discovery → positive boost."""
        agent = make_entity_agent()
        state = SharedState()
        response = "What about errors?\nINSIGHT: ErrorHandler = fallback + recovery"
        agent._update_confidence(state, response, MessageType.CHALLENGE)
        # grounded=0, accumulated=0.03, blended=0.015
        assert state.confidence.structural > 0.0

    def test_challenge_without_insight_slight_negative(self):
        """Pure challenge without insight → slightly negative."""
        agent = make_entity_agent()
        state = SharedState()
        state.confidence.structural = 0.5
        response = "But what about the edge cases here?"
        agent._update_confidence(state, response, MessageType.CHALLENGE)
        # grounded=0, accumulated=0.5-0.02=0.48, blended=0*0.5+0.48*0.5=0.24
        assert state.confidence.structural == pytest.approx(0.24, abs=0.01)


class TestAgreementUnchanged:
    """AGREEMENT should still give significant boost."""

    def test_agreement_boost(self):
        agent = make_entity_agent()
        state = SharedState()
        response = "I agree with the structural analysis."
        agent._update_confidence(state, response, MessageType.AGREEMENT)
        # grounded=0, accumulated=0.15, blended=0.075
        assert state.confidence.structural == pytest.approx(0.075, abs=0.01)


class TestAccommodationUnchanged:
    """ACCOMMODATION should give moderate boost."""

    def test_accommodation_boost(self):
        agent = make_entity_agent()
        state = SharedState()
        response = "Good point, revising my analysis."
        agent._update_confidence(state, response, MessageType.ACCOMMODATION)
        # +0.08 (accommodation) + 0.03 (positive marker: "good point") = 0.11
        # grounded=0, accumulated=0.11, blended=0.055
        assert state.confidence.structural == pytest.approx(0.055, abs=0.01)


class TestContentMarkers:
    """Content-based positive/negative markers still work."""

    def test_positive_marker_adds(self):
        """'comprehensive' in response should add boost."""
        agent = make_entity_agent()
        state = SharedState()
        response = "This is a comprehensive analysis.\nINSIGHT: X = Y"
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # 0.06 (proposition+insight) + 0.03 (positive marker) = 0.09
        # grounded=0, accumulated=0.09, blended=0.045
        assert state.confidence.structural == pytest.approx(0.045, abs=0.01)

    def test_territory_claiming_negative_boosts(self):
        """'missing' as territory-claiming → discovery boost."""
        agent = make_entity_agent()
        state = SharedState()
        response = "There are missing components in the authentication flow."
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # 0.02 (proposition) + 0.02 (territory-claiming discovery) = 0.04
        # grounded=0, accumulated=0.04, blended=0.02
        assert state.confidence.structural == pytest.approx(0.02, abs=0.01)

    def test_self_directed_negative_penalizes(self):
        """'I missed' → self-directed, penalize."""
        agent = make_entity_agent()
        state = SharedState()
        response = "I missed the session lifecycle entirely."
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # 0.02 (proposition) - 0.05 (self-directed negative) = -0.03 → clamped to 0.0
        # grounded=0, accumulated=0.0, blended=0.0
        assert state.confidence.structural == pytest.approx(0.0, abs=0.01)

    def test_challenge_negative_no_double_penalty(self):
        """Negative marker in CHALLENGE → already scored by type, no extra penalty."""
        agent = make_entity_agent()
        state = SharedState()
        state.confidence.structural = 0.5
        response = "You overlooked the error handling gap."
        agent._update_confidence(state, response, MessageType.CHALLENGE)
        # -0.02 (challenge without insight), "gap" detected but skipped for CHALLENGE
        # grounded=0, accumulated=0.48, blended=0.24
        assert state.confidence.structural == pytest.approx(0.24, abs=0.01)


class TestConsistencyCalculation:
    """Consistency uses productive/total ratio over 6 recent messages."""

    def test_all_propositions_high_consistency(self):
        """All constructive turns → high consistency."""
        agent = make_generic_agent()
        state = SharedState()
        for i in range(6):
            state.add_message(Message(
                sender="Entity", content=f"Analysis {i}",
                message_type=MessageType.PROPOSITION
            ))
        agent._update_confidence(state, "test", MessageType.PROPOSITION)
        assert state.confidence.consistency == pytest.approx(6/7, abs=0.01)

    def test_mixed_constructive_challenge(self):
        """Mix of constructive and challenges → moderate consistency."""
        agent = make_generic_agent()
        state = SharedState()
        for i in range(4):
            state.add_message(Message(
                sender="Entity", content=f"Analysis {i}",
                message_type=MessageType.PROPOSITION
            ))
        for i in range(2):
            state.add_message(Message(
                sender="Process", content=f"Challenge {i}",
                message_type=MessageType.CHALLENGE
            ))
        agent._update_confidence(state, "test", MessageType.PROPOSITION)
        assert state.confidence.consistency == pytest.approx(4/7, abs=0.01)

    def test_all_challenges_no_insight_low_consistency(self):
        """All challenges without insights → low consistency."""
        agent = make_generic_agent()
        state = SharedState()
        for i in range(4):
            state.add_message(Message(
                sender="Entity", content=f"But what about {i}?",
                message_type=MessageType.CHALLENGE
            ))
        agent._update_confidence(state, "test", MessageType.PROPOSITION)
        assert state.confidence.consistency == pytest.approx(0.0, abs=0.01)

    def test_challenges_with_insights_are_productive(self):
        """Challenges that yield insights count as productive turns."""
        agent = make_generic_agent()
        state = SharedState()
        for i in range(3):
            state.add_message(Message(
                sender="Entity", content=f"Challenge {i}",
                message_type=MessageType.CHALLENGE,
                insight=f"insight {i}"
            ))
        state.add_message(Message(
            sender="Process", content="Empty challenge",
            message_type=MessageType.CHALLENGE
        ))
        agent._update_confidence(state, "test", MessageType.PROPOSITION)
        assert state.confidence.consistency == pytest.approx(3/5, abs=0.01)


class TestGroundingStabilization:
    """Test that insight-grounded blending stabilizes structural/behavioral."""

    def test_grounding_anchors_with_prior_insights(self):
        """With prior insights in history, grounding provides stable floor."""
        agent = make_entity_agent()
        state = SharedState()
        # Add 6 prior Entity messages with insights
        for i in range(6):
            _add_entity_insight(state, f"Turn {i}", f"Component{i}")
        # Even a negative turn can't tank structural due to grounding
        response = "I missed something."
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # grounded = 6 * 0.08 = 0.48
        # accumulated = 0 + 0.02 - 0.05 = -0.03 → 0.0
        # blended = 0.48*0.5 + 0.0*0.5 = 0.24
        assert state.confidence.structural >= 0.20

    def test_no_prior_insights_means_no_grounding(self):
        """Without prior insights, only accumulated noise matters."""
        agent = make_entity_agent()
        state = SharedState()
        # No prior messages in history
        response = "Analysis.\nINSIGHT: X = Y"
        agent._update_confidence(state, response, MessageType.PROPOSITION)
        # grounded=0, accumulated=0.06, blended=0.03
        assert state.confidence.structural < 0.05

    def test_entity_and_process_grounded_independently(self):
        """Entity insights don't ground behavioral, and vice versa."""
        agent_e = make_entity_agent()
        agent_p = make_process_agent()
        state = SharedState()
        # Add 5 Entity insights but no Process insights
        for i in range(5):
            _add_entity_insight(state, f"E turn {i}", f"Entity{i}")
        agent_e._update_confidence(state, "Test.\nINSIGHT: X", MessageType.PROPOSITION)
        agent_p._update_confidence(state, "Test.\nINSIGHT: Y", MessageType.PROPOSITION)
        # Entity has 5 prior insights → grounded=0.40
        assert state.confidence.structural > 0.15
        # Process has 0 prior insights → grounded=0
        assert state.confidence.behavioral < 0.05


class TestEndToEndConfidenceGrowth:
    """Simulate a realistic dialogue and verify confidence grows."""

    def test_10_turn_dialogue_produces_nonzero_confidence(self):
        """After 10 turns of normal dialogue, all dimensions should be nonzero."""
        entity = make_entity_agent()
        process = make_process_agent()
        state = SharedState()

        # Simulate alternating turns with insights
        for i in range(5):
            # Entity turn
            resp_e = f"Analyzing structure.\nINSIGHT: Component{i} = attributes + relationships"
            entity._update_confidence(state, resp_e, MessageType.PROPOSITION)
            state.add_message(Message(
                sender="Entity", content=resp_e,
                message_type=MessageType.PROPOSITION,
                insight=f"Component{i} = attributes + relationships"
            ))

            # Process turn
            resp_p = f"Analyzing behavior.\nINSIGHT: Flow{i} -> Step{i} -> Result{i}"
            process._update_confidence(state, resp_p, MessageType.PROPOSITION)
            state.add_message(Message(
                sender="Process", content=resp_p,
                message_type=MessageType.PROPOSITION,
                insight=f"Flow{i} -> Step{i} -> Result{i}"
            ))

        # All dimensions should have grown
        assert state.confidence.structural > 0.05, f"structural={state.confidence.structural}"
        assert state.confidence.behavioral > 0.05, f"behavioral={state.confidence.behavioral}"
        assert state.confidence.coverage > 0.3, f"coverage={state.confidence.coverage}"
        assert state.confidence.consistency > 0.5, f"consistency={state.confidence.consistency}"
        assert state.confidence.overall() > 0.1, f"overall={state.confidence.overall()}"

    def test_agreement_after_propositions_reaches_convergence(self):
        """Dialogue with insights then agreements should reach convergence-worthy levels."""
        entity = make_entity_agent()
        process = make_process_agent()
        state = SharedState()

        # 3 insight turns each
        for i in range(3):
            resp = f"Structure found.\nINSIGHT: Entity{i} = core"
            entity._update_confidence(state, resp, MessageType.PROPOSITION)
            state.add_message(Message(
                sender="Entity", content=resp,
                message_type=MessageType.PROPOSITION,
                insight=f"Entity{i} = core"
            ))

            resp = f"Flow mapped.\nINSIGHT: Process{i} -> done"
            process._update_confidence(state, resp, MessageType.PROPOSITION)
            state.add_message(Message(
                sender="Process", content=resp,
                message_type=MessageType.PROPOSITION,
                insight=f"Process{i} -> done"
            ))

        # 2 agreement turns
        entity._update_confidence(state, "I agree, sufficient.", MessageType.AGREEMENT)
        state.add_message(Message(
            sender="Entity", content="I agree.", message_type=MessageType.AGREEMENT
        ))
        process._update_confidence(state, "Agreed, comprehensive.", MessageType.AGREEMENT)
        state.add_message(Message(
            sender="Process", content="Agreed.", message_type=MessageType.AGREEMENT
        ))

        # Should have meaningful confidence now
        assert state.confidence.structural > 0.1
        assert state.confidence.behavioral > 0.1
        assert state.confidence.overall() > 0.1
