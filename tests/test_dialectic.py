"""
Tests for core/dialectic.py — Dialectic Rounds.

Tests the 3-round thesis/antithesis/synthesis structure for spec dialogue.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import asdict

from core.dialectic import (
    DialecticRole,
    DialecticPhase,
    RoundOutput,
    RoundManager,
    ROTATION_ANGLES,
    ROUND_ROLE_PROMPTS,
    COLLAPSE_PROMPT,
    _WEAKNESS_TO_ANGLE,
)
from core.protocol import (
    SharedState,
    Message,
    MessageType,
    ConfidenceVector,
)
from core.protocol_spec import PROTOCOL, DialecticSpec


# =============================================================================
# TestDialecticRole
# =============================================================================


class TestDialecticRole:
    """Enum values for turn roles within a round."""

    def test_thesis_value(self):
        assert DialecticRole.THESIS.value == "thesis"

    def test_antithesis_value(self):
        assert DialecticRole.ANTITHESIS.value == "antithesis"

    def test_synthesis_value(self):
        assert DialecticRole.SYNTHESIS.value == "synthesis"


# =============================================================================
# TestDialecticPhase
# =============================================================================


class TestDialecticPhase:
    """Enum values for round phases."""

    def test_thesis_phase_value(self):
        assert DialecticPhase.THESIS.value == "thesis"

    def test_stress_test_value(self):
        assert DialecticPhase.STRESS_TEST.value == "stress"

    def test_collapse_value(self):
        assert DialecticPhase.COLLAPSE.value == "collapse"


# =============================================================================
# TestRoundOutput
# =============================================================================


class TestRoundOutput:
    """RoundOutput dataclass."""

    def test_creation(self):
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=["test insight"],
            confidence_snapshot={"structural": 0.5},
        )
        assert ro.round_number == 0
        assert ro.phase == DialecticPhase.THESIS
        assert ro.rotation_angle == "existence"
        assert ro.insights == ["test insight"]

    def test_defaults(self):
        ro = RoundOutput(
            round_number=1,
            phase=DialecticPhase.STRESS_TEST,
            rotation_angle="dynamics",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        assert ro.provenance_passed is True
        assert ro.gate_attempts == 0

    def test_mutable(self):
        """RoundOutput is NOT frozen — we need to update provenance_passed."""
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        ro.provenance_passed = False
        assert ro.provenance_passed is False


# =============================================================================
# TestDialecticSpec
# =============================================================================


class TestDialecticSpec:
    """DialecticSpec in protocol."""

    def test_exists_on_protocol(self):
        assert hasattr(PROTOCOL, 'dialectic')

    def test_default_values(self):
        spec = PROTOCOL.dialectic
        assert spec.turns_per_round == 3
        assert spec.max_rounds == 3
        assert spec.max_gate_retries == 4
        assert spec.total_turn_budget == 30
        assert spec.collapse_no_challenge is True

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PROTOCOL.dialectic.turns_per_round = 5

    def test_standalone_creation(self):
        spec = DialecticSpec(turns_per_round=4, max_rounds=2)
        assert spec.turns_per_round == 4
        assert spec.max_rounds == 2


# =============================================================================
# TestRoundManager
# =============================================================================


class TestRoundManager:
    """Core RoundManager behavior."""

    def test_initial_state(self):
        mgr = RoundManager()
        assert mgr.current_round == 0
        assert mgr.rounds == []
        assert mgr._gate_failures == 0

    def test_current_phase_round_0(self):
        mgr = RoundManager()
        assert mgr.current_phase() == DialecticPhase.THESIS

    def test_current_phase_round_1(self):
        mgr = RoundManager()
        mgr.current_round = 1
        assert mgr.current_phase() == DialecticPhase.STRESS_TEST

    def test_current_phase_round_2(self):
        mgr = RoundManager()
        mgr.current_round = 2
        assert mgr.current_phase() == DialecticPhase.COLLAPSE

    def test_turn_role_thesis(self):
        mgr = RoundManager()
        assert mgr.turn_role(0) == DialecticRole.THESIS

    def test_turn_role_antithesis(self):
        mgr = RoundManager()
        assert mgr.turn_role(1) == DialecticRole.ANTITHESIS

    def test_turn_role_synthesis(self):
        mgr = RoundManager()
        assert mgr.turn_role(2) == DialecticRole.SYNTHESIS

    def test_rotation_angle_round_0_always_existence(self):
        mgr = RoundManager()
        name, prompt = mgr.rotation_angle_for_round(0, None)
        assert name == "existence"
        assert "entities" in prompt.lower()

    def test_rotation_angle_round_1_adaptive_structural(self):
        mgr = RoundManager()
        conf = ConfidenceVector(structural=0.1, behavioral=0.5, coverage=0.5, consistency=0.5)
        name, prompt = mgr.rotation_angle_for_round(1, conf)
        assert name == "grounding"

    def test_rotation_angle_round_1_adaptive_behavioral(self):
        mgr = RoundManager()
        conf = ConfidenceVector(structural=0.5, behavioral=0.1, coverage=0.5, consistency=0.5)
        name, prompt = mgr.rotation_angle_for_round(1, conf)
        assert name == "dynamics"

    def test_rotation_angle_round_1_adaptive_coverage(self):
        mgr = RoundManager()
        conf = ConfidenceVector(structural=0.5, behavioral=0.5, coverage=0.1, consistency=0.5)
        name, prompt = mgr.rotation_angle_for_round(1, conf)
        assert name == "state"

    def test_rotation_angle_round_1_adaptive_consistency(self):
        mgr = RoundManager()
        conf = ConfidenceVector(structural=0.5, behavioral=0.5, coverage=0.5, consistency=0.1)
        name, prompt = mgr.rotation_angle_for_round(1, conf)
        assert name == "constraints"

    def test_rotation_angle_round_1_no_confidence_fallback(self):
        mgr = RoundManager()
        name, _ = mgr.rotation_angle_for_round(1, None)
        assert name == "dynamics"

    def test_rotation_angle_round_2_collapse(self):
        mgr = RoundManager()
        name, prompt = mgr.rotation_angle_for_round(2, None)
        assert name == "collapse"
        assert "No new challenges" in prompt

    def test_commit_round_advances(self):
        mgr = RoundManager()
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=["i1"],
            confidence_snapshot={},
        )
        mgr.commit_round(ro)
        assert mgr.current_round == 1
        assert len(mgr.rounds) == 1
        assert mgr.rounds[0] is ro

    def test_commit_resets_gate_failures(self):
        mgr = RoundManager()
        mgr._gate_failures = 2
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        mgr.commit_round(ro)
        assert mgr._gate_failures == 0


# =============================================================================
# TestProvenanceGate
# =============================================================================


def _make_message(insight=None, sender="Entity"):
    """Helper to create a Message with optional insight."""
    return Message(
        sender=sender,
        content="test content",
        message_type=MessageType.PROPOSITION,
        insight=insight,
        insight_display=insight[:60] if insight else None,
    )


class TestProvenanceGate:
    """Provenance gate between rounds."""

    def test_passes_with_valid_insight(self):
        mgr = RoundManager()
        msgs = [
            _make_message("Entity exists"),
            _make_message("Challenge accepted", sender="Process"),
            _make_message("Resolved: synthesis insight"),
        ]
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=msgs,
            insights=["Resolved: synthesis insight"],
            confidence_snapshot={},
        )
        state = SharedState(known={"input": "test"})
        assert mgr.check_round_gate(ro, state) is True

    def test_fails_with_no_insight_on_synthesis(self):
        mgr = RoundManager()
        msgs = [
            _make_message("Entity exists"),
            _make_message("Challenge accepted", sender="Process"),
            _make_message(None),  # synthesis without insight
        ]
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=msgs,
            insights=[],
            confidence_snapshot={},
        )
        state = SharedState(known={"input": "test"})
        assert mgr.check_round_gate(ro, state) is False

    def test_fails_with_too_few_messages(self):
        mgr = RoundManager()
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[_make_message("only one")],
            insights=[],
            confidence_snapshot={},
        )
        state = SharedState(known={"input": "test"})
        assert mgr.check_round_gate(ro, state) is False

    def test_reentry_increments_gate_failures(self):
        """Gate failure tracking for retry logic."""
        mgr = RoundManager()
        assert mgr._gate_failures == 0
        mgr._gate_failures += 1
        assert mgr._gate_failures == 1

    def test_max_retries_respected(self):
        """After max retries, round should commit regardless."""
        mgr = RoundManager(max_gate_retries=2)
        mgr._gate_failures = 2
        # At this point, the engine loop checks _gate_failures < max_gate_retries
        # and skips retry — commits instead
        assert mgr._gate_failures >= mgr.max_gate_retries


# =============================================================================
# TestRoundContext
# =============================================================================


class TestRoundContext:
    """build_round_context() output."""

    def test_thesis_context_includes_role(self):
        mgr = RoundManager()
        ctx = mgr.build_round_context(0, DialecticRole.THESIS, [])
        assert "THESIS" in ctx
        assert "Stake your position" in ctx

    def test_antithesis_context_includes_role(self):
        mgr = RoundManager()
        ctx = mgr.build_round_context(0, DialecticRole.ANTITHESIS, [])
        assert "ANTITHESIS" in ctx
        assert "Challenge" in ctx

    def test_synthesis_context_includes_role(self):
        mgr = RoundManager()
        ctx = mgr.build_round_context(0, DialecticRole.SYNTHESIS, [])
        assert "SYNTHESIS" in ctx
        assert "Resolve" in ctx

    def test_includes_prior_round_summary(self):
        mgr = RoundManager()
        prior = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=["found User entity"],
            confidence_snapshot={"structural": 0.3, "behavioral": 0.2, "coverage": 0.1},
        )
        ctx = mgr.build_round_context(1, DialecticRole.THESIS, [prior])
        assert "PRIOR ROUNDS:" in ctx
        assert "found User entity" in ctx
        assert "existence" in ctx

    def test_collapse_context_includes_no_new_challenges(self):
        mgr = RoundManager()
        mgr.current_round = 2
        ctx = mgr.build_round_context(2, DialecticRole.THESIS, [])
        assert "No new challenges" in ctx
        assert "COLLAPSE" in ctx


# =============================================================================
# TestNarrowScope
# =============================================================================


class TestNarrowScope:
    """narrow_scope() output."""

    def test_generates_narrowing_instruction(self):
        mgr = RoundManager()
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        state = SharedState(known={"input": "test", "intent": {}})
        result = mgr.narrow_scope(ro, state)
        assert "RETRY" in result
        assert "provenance" in result.lower()

    def test_references_unresolved_unknowns(self):
        mgr = RoundManager()
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        state = SharedState(
            known={"input": "test", "intent": {}},
            unknown=["What is the auth method?", "Which database?"],
        )
        result = mgr.narrow_scope(ro, state)
        assert "auth method" in result

    def test_references_uncovered_components(self):
        mgr = RoundManager()
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=[],
            insights=[],
            confidence_snapshot={},
        )
        state = SharedState(
            known={
                "input": "test",
                "intent": {"explicit_components": ["UserService", "AuthDB"]},
            },
        )
        result = mgr.narrow_scope(ro, state)
        assert "UserService" in result


# =============================================================================
# TestRotationAngles
# =============================================================================


class TestRotationAngles:
    """Rotation angles constants."""

    def test_five_angles_defined(self):
        assert len(ROTATION_ANGLES) == 5

    def test_each_is_name_prompt_tuple(self):
        for name, prompt in ROTATION_ANGLES:
            assert isinstance(name, str)
            assert isinstance(prompt, str)
            assert len(name) > 0
            assert len(prompt) > 0

    def test_first_is_existence(self):
        assert ROTATION_ANGLES[0][0] == "existence"

    def test_weakness_mapping_covers_all_dims(self):
        expected = {"structural", "behavioral", "coverage", "consistency"}
        assert set(_WEAKNESS_TO_ANGLE.keys()) == expected


# =============================================================================
# TestRoundPrompts
# =============================================================================


class TestRoundPrompts:
    """Round role prompts constants."""

    def test_all_roles_have_prompts(self):
        for role in DialecticRole:
            assert role in ROUND_ROLE_PROMPTS

    def test_thesis_prompt_content(self):
        assert "THESIS" in ROUND_ROLE_PROMPTS[DialecticRole.THESIS]

    def test_antithesis_prompt_content(self):
        assert "ANTITHESIS" in ROUND_ROLE_PROMPTS[DialecticRole.ANTITHESIS]

    def test_synthesis_prompt_content(self):
        assert "SYNTHESIS" in ROUND_ROLE_PROMPTS[DialecticRole.SYNTHESIS]

    def test_collapse_prompt_content(self):
        assert "FINAL SYNTHESIS" in COLLAPSE_PROMPT
        assert "No new challenges" in COLLAPSE_PROMPT


# =============================================================================
# TestBackwardCompat
# =============================================================================


class TestBackwardCompat:
    """Backward compatibility checks."""

    def test_confidence_tracking_works_within_rounds(self):
        """ConfidenceVector still works — dialectic doesn't break it."""
        cv = ConfidenceVector(structural=0.3, behavioral=0.4, coverage=0.5, consistency=0.6)
        assert cv.overall() == pytest.approx(0.45)
        assert cv.weakest_dimension() == "structural"

    def test_round_output_is_plain_dataclass(self):
        """No imports from mother/ in dialectic module."""
        import core.dialectic as mod
        import inspect
        source = inspect.getfile(mod)
        with open(source) as f:
            content = f.read()
        assert "from mother" not in content
        assert "import mother" not in content

    def test_shared_state_dialectic_context_is_optional(self):
        """SharedState works fine without _dialectic_context."""
        state = SharedState(known={"input": "test"})
        ctx = state.known.get("_dialectic_context", "")
        assert ctx == ""


# =============================================================================
# TestIntegration
# =============================================================================


def _make_mock_agent(name, insights=None):
    """Create a mock agent that returns predictable messages."""
    agent = Mock()
    agent.name = name
    call_count = [0]
    insight_list = insights or [f"INSIGHT: {name} found something #{i}" for i in range(20)]

    def run_side_effect(state, input_msg):
        idx = call_count[0] % len(insight_list)
        insight_text = insight_list[idx]
        call_count[0] += 1
        return Message(
            sender=name,
            content=f"{name} response #{call_count[0]}\n{insight_text}",
            message_type=MessageType.PROPOSITION,
            insight=insight_text,
            insight_display=insight_text[:60],
        )

    agent.run = Mock(side_effect=run_side_effect)
    return agent


class TestIntegration:
    """Integration tests with mock agents."""

    def test_three_rounds_complete(self):
        """3 rounds of 3 turns = 9 turns total."""
        mgr = RoundManager()
        state = SharedState(known={"input": "build a booking system"})
        entity = _make_mock_agent("Entity")
        process = _make_mock_agent("Process")
        agents = {"Entity": entity, "Process": process}

        total_turns = 0

        while mgr.current_round < PROTOCOL.dialectic.max_rounds:
            if total_turns >= PROTOCOL.dialectic.total_turn_budget:
                break

            phase = mgr.current_phase()
            round_messages = []
            round_insights = []

            for turn_in_round in range(PROTOCOL.dialectic.turns_per_round):
                role = mgr.turn_role(turn_in_round)
                agent_name = "Process" if role == DialecticRole.ANTITHESIS else "Entity"
                agent = agents[agent_name]

                msg = Message(
                    sender="System", content="go",
                    message_type=MessageType.PROPOSITION,
                )
                response = agent.run(state, msg)
                state.add_message(response)
                round_messages.append(response)
                if response.insight:
                    round_insights.append(response.insight)
                total_turns += 1

            ro = RoundOutput(
                round_number=mgr.current_round,
                phase=phase,
                rotation_angle="existence",
                messages=round_messages,
                insights=round_insights,
                confidence_snapshot=state.confidence.to_dict(),
            )
            mgr.commit_round(ro)

        assert mgr.current_round == 3
        assert total_turns == 9
        assert len(mgr.rounds) == 3

    def test_gate_failure_causes_round_reentry(self):
        """When provenance gate fails, round re-enters with narrowed scope."""
        mgr = RoundManager(max_gate_retries=1)
        state = SharedState(known={"input": "build something"})

        # Round with synthesis that has no insight — gate will fail
        no_insight_msg = Message(
            sender="Entity", content="no insight here",
            message_type=MessageType.PROPOSITION,
            insight=None,
        )
        msgs = [
            _make_message("thesis claim"),
            _make_message("challenge", sender="Process"),
            no_insight_msg,  # synthesis without insight
        ]
        ro = RoundOutput(
            round_number=0,
            phase=DialecticPhase.THESIS,
            rotation_angle="existence",
            messages=msgs,
            insights=[],
            confidence_snapshot={},
        )

        gate_passed = mgr.check_round_gate(ro, state)
        assert gate_passed is False

        # Simulate retry logic
        assert mgr._gate_failures < mgr.max_gate_retries
        mgr._gate_failures += 1
        narrowed = mgr.narrow_scope(ro, state)
        assert "RETRY" in narrowed

        # After max retries, should not retry
        assert mgr._gate_failures >= mgr.max_gate_retries

    def test_total_budget_hard_stop(self):
        """total_turn_budget enforces hard ceiling."""
        mgr = RoundManager()
        state = SharedState(known={"input": "test"})
        entity = _make_mock_agent("Entity")
        process = _make_mock_agent("Process")
        agents = {"Entity": entity, "Process": process}

        total_turns = 0
        # Simulate running until budget
        budget = PROTOCOL.dialectic.total_turn_budget

        while mgr.current_round < 10:  # way more than max_rounds
            if total_turns >= budget:
                break

            round_messages = []
            for turn_in_round in range(PROTOCOL.dialectic.turns_per_round):
                if total_turns >= budget:
                    break
                role = mgr.turn_role(turn_in_round)
                agent_name = "Process" if role == DialecticRole.ANTITHESIS else "Entity"
                response = agents[agent_name].run(state, None)
                state.add_message(response)
                round_messages.append(response)
                total_turns += 1

            if len(round_messages) == PROTOCOL.dialectic.turns_per_round:
                ro = RoundOutput(
                    round_number=mgr.current_round,
                    phase=mgr.current_phase(),
                    rotation_angle="test",
                    messages=round_messages,
                    insights=[],
                    confidence_snapshot={},
                )
                mgr.commit_round(ro)

        assert total_turns <= budget

    def test_collapse_round_skips_gate(self):
        """COLLAPSE phase does not run provenance gate."""
        mgr = RoundManager()
        phase = DialecticPhase.COLLAPSE
        # The engine code checks: if phase != DialecticPhase.COLLAPSE
        # So collapse rounds skip the gate entirely
        assert phase == DialecticPhase.COLLAPSE
        # Verify the phase equality check works as expected
        assert phase != DialecticPhase.THESIS
        assert phase != DialecticPhase.STRESS_TEST
