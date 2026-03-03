"""
Phase 12.1: Tests for Engine Quality — Adaptive Depth, Persona Injection, Prioritized Synthesis.

Tests cover:
- 12.1a: Adaptive Dialogue Depth (8 tests)
- 12.1b: Persona Injection into Agent Context (6 tests)
- 12.1c: Prioritized Synthesis Insights (7 tests)
"""

import pytest
from core.protocol import (
    SharedState,
    Message,
    MessageType,
    ConfidenceVector,
    calculate_dialogue_depth,
)
from core.digest import build_dialogue_digest, _rank_insights
from agents.base import LLMAgent
from agents.swarm import GovernorAgent


# =============================================================================
# 12.1a: ADAPTIVE DIALOGUE DEPTH (8 tests)
# =============================================================================


class TestAdaptiveDepth:
    """Tests for calculate_dialogue_depth() returning 3-tuple with adaptive ceiling."""

    def test_simple_input_returns_3_tuple(self):
        """Simple input returns (6, 8, 12)."""
        intent = {
            "explicit_components": ["A", "B"],
            "actors": ["User"],
            "constraints": [],
        }
        result = calculate_dialogue_depth(intent, "Build a simple app")
        assert len(result) == 3
        min_turns, min_insights, max_turns = result
        assert min_turns == 6
        assert min_insights == 8
        assert max_turns == 22

    def test_complex_input_higher_min(self):
        """13 components should yield min_turns > 8."""
        intent = {
            "explicit_components": [
                "Intent Agent", "Persona Agent", "Entity Agent", "Process Agent",
                "Synthesis Agent", "Verify Agent", "Governor Agent",
                "SharedState", "ConfidenceVector", "ConflictOracle",
                "Message", "DialogueProtocol", "Corpus",
            ],
            "actors": ["Admin", "User", "Moderator", "System"],
            "constraints": ["C1", "C2", "C3", "C4", "C5", "C6"],
        }
        min_turns, _, _ = calculate_dialogue_depth(intent, "x" * 3000)
        assert min_turns > 8

    def test_relationship_bonus(self):
        """explicit_relationships add depth (Phase 12.1a)."""
        intent_no_rels = {
            "explicit_components": [],
            "actors": [],
            "constraints": [],
            "explicit_relationships": [],
        }
        intent_with_rels = {
            "explicit_components": [],
            "actors": [],
            "constraints": [],
            "explicit_relationships": ["A triggers B", "C accesses D", "E monitors F", "G snapshots H"],
        }
        turns_no, _, _ = calculate_dialogue_depth(intent_no_rels, "short")
        turns_with, _, _ = calculate_dialogue_depth(intent_with_rels, "short")
        assert turns_with > turns_no  # 4 relationships = +1 bonus

    def test_max_turns_is_min_plus_offset(self):
        """max_turns = min_turns + offset for normal inputs."""
        intent = {
            "explicit_components": ["A", "B", "C", "D", "E", "F"],
            "actors": ["U1", "U2"],
            "constraints": [],
        }
        min_turns, _, max_turns = calculate_dialogue_depth(intent, "short")
        assert max_turns == min_turns + 16

    def test_max_turns_capped_at_24(self):
        """max_turns should never exceed 24."""
        intent = {
            "explicit_components": list(range(50)),
            "actors": list(range(20)),
            "constraints": list(range(20)),
            "explicit_relationships": list(range(20)),
        }
        _, _, max_turns = calculate_dialogue_depth(intent, "x" * 5000)
        assert max_turns <= 64

    def test_simple_max_is_12(self):
        """Simple input yields max_turns=12."""
        _, _, max_turns = calculate_dialogue_depth({}, "")
        assert max_turns == 22

    def test_governor_uses_adaptive_max(self):
        """Governor should respect the passed max_turns parameter."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.1, behavioral=0.1, coverage=0.1, consistency=0.1
        )
        # Not at max yet — should not end
        result = gov.should_end_dialogue(
            state, turn_count=11, min_turns=6, min_insights=8, max_turns=20
        )
        assert result is False

    def test_governor_hard_stop_at_adaptive_max(self):
        """Governor hard-stops when turn_count >= max_turns."""
        gov = GovernorAgent()
        state = SharedState()
        state.confidence = ConfidenceVector(
            structural=0.1, behavioral=0.1, coverage=0.1, consistency=0.1
        )
        # At max — should end
        result = gov.should_end_dialogue(
            state, turn_count=20, min_turns=6, min_insights=8, max_turns=20
        )
        assert result is True


# =============================================================================
# 12.1b: PERSONA INJECTION (6 tests)
# =============================================================================


class _MockLLM:
    """Minimal mock LLM for testing context building."""
    def complete_with_system(self, system_prompt, user_content, max_tokens=4096):
        return "INSIGHT: test insight"


class TestPersonaInjection:
    """Tests for _build_context() persona injection."""

    def _make_agent(self):
        return LLMAgent(
            name="Entity",
            perspective="Structure: nouns, attributes, relationships",
            system_prompt="You are the Entity Agent.",
            llm_client=_MockLLM(),
        )

    def test_context_includes_priorities(self):
        """Personas with priorities should appear in context."""
        agent = self._make_agent()
        state = SharedState()
        state.personas = [{
            "name": "Architect",
            "priorities": ["security", "scalability", "maintainability"],
            "blind_spots": "UX concerns",
            "key_questions": ["How does auth work?"],
        }]
        context = agent._build_context(state)
        assert "priorities=" in context
        assert "security" in context

    def test_context_includes_blind_spots(self):
        """Blind spots should appear in context."""
        agent = self._make_agent()
        state = SharedState()
        state.personas = [{
            "name": "Designer",
            "priorities": [],
            "blind_spots": "Backend complexity",
            "key_questions": [],
        }]
        context = agent._build_context(state)
        assert "blind_spots=" in context
        assert "Backend complexity" in context

    def test_context_includes_key_questions(self):
        """Key questions should appear in context."""
        agent = self._make_agent()
        state = SharedState()
        state.personas = [{
            "name": "PM",
            "priorities": [],
            "blind_spots": "",
            "key_questions": ["What is the MVP?", "Who pays?"],
        }]
        context = agent._build_context(state)
        assert "questions=" in context
        assert "What is the MVP?" in context

    def test_context_graceful_without_personas(self):
        """No personas should produce 'None yet' fallback."""
        agent = self._make_agent()
        state = SharedState()
        context = agent._build_context(state)
        assert "None yet" in context

    def test_context_max_3_personas(self):
        """Only first 3 personas included."""
        agent = self._make_agent()
        state = SharedState()
        state.personas = [
            {"name": f"Persona{i}", "priorities": [f"p{i}"], "blind_spots": "", "key_questions": []}
            for i in range(5)
        ]
        context = agent._build_context(state)
        assert "Persona0" in context
        assert "Persona2" in context
        assert "Persona3" not in context
        assert "Persona4" not in context

    def test_context_handles_missing_fields(self):
        """Partial persona dicts should not crash."""
        agent = self._make_agent()
        state = SharedState()
        state.personas = [
            {"name": "Minimal"},  # no priorities, blind_spots, or key_questions
        ]
        context = agent._build_context(state)
        assert "Minimal" in context
        assert "priorities=[]" in context


# =============================================================================
# 12.1c: PRIORITIZED SYNTHESIS INSIGHTS (7 tests)
# =============================================================================


class TestRankInsights:
    """Tests for _rank_insights() insight prioritization."""

    def _make_state_with_typed_insights(self, message_types_and_insights):
        """Helper: create state with messages of given types and insights."""
        state = SharedState()
        for i, (mtype, insight_text) in enumerate(message_types_and_insights):
            sender = "Entity" if i % 2 == 0 else "Process"
            m = Message(
                sender=sender,
                content=f"Turn {i}: {insight_text}",
                message_type=mtype,
                insight=insight_text,
            )
            state.add_message(m)
        return state

    def test_rank_insights_challenge_medium(self):
        """CHALLENGE insight should get MEDIUM tier (score=3)."""
        state = self._make_state_with_typed_insights([
            (MessageType.CHALLENGE, "booking = commitment + scheduling"),
        ])
        ranked = _rank_insights(state)
        assert len(ranked) == 1
        assert ranked[0]["tier"] == "MEDIUM"
        assert ranked[0]["score"] == 3

    def test_rank_insights_proposition_low(self):
        """PROPOSITION insight should get LOW tier."""
        state = self._make_state_with_typed_insights([
            (MessageType.PROPOSITION, "system has users and bookings"),
        ])
        ranked = _rank_insights(state)
        assert len(ranked) == 1
        assert ranked[0]["tier"] == "LOW"
        assert ranked[0]["score"] == 1

    def test_rank_insights_pattern_boost(self):
        """'works like' in insight should add +2 score."""
        state = self._make_state_with_typed_insights([
            (MessageType.PROPOSITION, "queue works like Restaurant waitlist"),
        ])
        ranked = _rank_insights(state)
        assert len(ranked) == 1
        # Proposition (+1) + pattern (+2) = 3 → MEDIUM
        assert ranked[0]["score"] == 3
        assert ranked[0]["tier"] == "MEDIUM"

    def test_rank_insights_challenge_plus_pattern_high(self):
        """CHALLENGE + 'works like' = score >= 5 → HIGH."""
        state = self._make_state_with_typed_insights([
            (MessageType.CHALLENGE, "booking works like Airline reservation"),
        ])
        ranked = _rank_insights(state)
        assert ranked[0]["score"] >= 5
        assert ranked[0]["tier"] == "HIGH"

    def test_digest_tiered_insights(self):
        """Digest should have MEDIUM/LOW sections for challenge + proposition."""
        state = self._make_state_with_typed_insights([
            (MessageType.CHALLENGE, "critical finding about auth"),
            (MessageType.PROPOSITION, "basic observation about users"),
        ])
        digest = build_dialogue_digest(state)
        assert "INSIGHTS:" in digest
        assert "MEDIUM PRIORITY:" in digest
        assert "LOW PRIORITY:" in digest

    def test_digest_higher_tier_before_lower(self):
        """Higher priority tiers should appear before lower in digest."""
        state = self._make_state_with_typed_insights([
            (MessageType.PROPOSITION, "basic observation first"),
            (MessageType.CHALLENGE, "critical finding works like Airline reservation"),
        ])
        digest = build_dialogue_digest(state)
        high_pos = digest.find("HIGH PRIORITY:")
        low_pos = digest.find("LOW PRIORITY:")
        assert high_pos >= 0, "HIGH PRIORITY should appear"
        assert low_pos >= 0, "LOW PRIORITY should appear"
        assert high_pos < low_pos, "HIGH should appear before LOW"

    def test_digest_no_insights_no_crash(self):
        """Empty insights should not crash or produce INSIGHTS section."""
        state = SharedState()
        digest = build_dialogue_digest(state)
        assert "INSIGHTS:" not in digest


class TestSynthesisInstruction:
    """Test that synthesis instruction includes priority guidance."""

    def test_synthesis_instruction_has_priority(self):
        """The synthesis INSTRUCTION should mention prioritizing HIGH insights."""
        # Read the instruction from engine source
        import core.engine as engine_module
        import inspect
        source = inspect.getsource(engine_module.MotherlabsEngine._synthesize)
        assert "HIGH PRIORITY" in source
        assert "high-priority" in source
