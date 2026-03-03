"""Tests for Build 6 genome wiring — readiness-staging, abstraction-leveling."""

import pytest

from mother.executive import (
    PlanStep,
    _needs_preparation,
    _build_plan,
    extract_steps_from_blueprint,
)
from mother.impulse import (
    classify_abstraction_level,
    compute_impulse,
    Impulse,
    ImpulseContext,
)
from mother.metabolism import (
    classify_thought_type,
    MetabolicContext,
    MetabolicMode,
    ThoughtType,
)


class TestReadinessStaging:
    """#101: Goals that need preparation get a prepare step."""

    def test_needs_preparation_research(self):
        """Research goals trigger preparation."""
        assert _needs_preparation({}, "Research best database options")

    def test_needs_preparation_gather(self):
        """Data gathering goals trigger preparation."""
        assert _needs_preparation({}, "Gather user feedback from surveys")

    def test_needs_preparation_integrate(self):
        """Integration goals trigger preparation."""
        assert _needs_preparation({"description": "integrate with Stripe"}, "connect payment")

    def test_no_preparation_simple_build(self):
        """Simple build goals don't need preparation."""
        assert not _needs_preparation({}, "Build a login page")

    def test_no_preparation_empty(self):
        """Empty descriptions don't need preparation."""
        assert not _needs_preparation({}, "")

    def test_build_plan_with_preparation(self):
        """Goals needing preparation get 4-step plan."""
        bp = {"description": "research-based project"}
        steps = _build_plan(bp, "Research and build API integration")
        assert len(steps) == 4
        assert steps[0]["action_type"] == "prepare"
        assert steps[0]["name"] == "prepare"
        assert steps[1]["action_type"] == "compile"
        assert steps[2]["action_type"] == "build"
        assert steps[3]["action_type"] == "goal_done"

    def test_build_plan_without_preparation(self):
        """Simple goals get 3-step plan (no prepare)."""
        bp = {"description": "simple app"}
        steps = _build_plan(bp, "Build a simple app")
        assert len(steps) == 3
        assert steps[0]["action_type"] == "compile"

    def test_extract_steps_passes_preparation(self):
        """extract_steps_from_blueprint delegates to _build_plan with preparation."""
        bp = {
            "description": "investigate and migrate database",
            "components": [{"name": "db"}],
        }
        steps = extract_steps_from_blueprint(bp, "Investigate and migrate database")
        assert any(s["action_type"] == "prepare" for s in steps)

    def test_prepare_action_type_in_plan_step(self):
        """PlanStep accepts 'prepare' action_type."""
        step = PlanStep(action_type="prepare", name="prep", description="stage materials")
        assert step.action_type == "prepare"


class TestAbstractionLeveling:
    """#108: Abstraction level computed from topics, adapts impulse/metabolism."""

    def test_strategic_topics_high_level(self):
        """Strategic keywords yield high abstraction level."""
        topics = ["vision", "strategy", "roadmap", "architecture"]
        level = classify_abstraction_level(topics, 5)
        assert level >= 0.7

    def test_concrete_topics_low_level(self):
        """Concrete keywords yield low abstraction level."""
        topics = ["bug", "fix", "test", "refactor"]
        level = classify_abstraction_level(topics, 5)
        assert level <= 0.3

    def test_mixed_topics_mid_level(self):
        """Mixed topics yield middle abstraction level."""
        topics = ["vision", "bug", "strategy", "test"]
        level = classify_abstraction_level(topics, 5)
        assert 0.3 <= level <= 0.7

    def test_empty_topics_default(self):
        """Empty topics return default 0.5."""
        assert classify_abstraction_level([], 0) == 0.5

    def test_few_messages_default(self):
        """Too few messages return default 0.5."""
        assert classify_abstraction_level(["vision"], 1) == 0.5

    def test_no_keyword_matches_default(self):
        """Topics with no keyword matches return default 0.5."""
        topics = ["lunch", "weather", "hello"]
        assert classify_abstraction_level(topics, 5) == 0.5

    def test_high_abstraction_impulse_prefers_reflect(self):
        """High abstraction level + recall → REFLECT over SPEAK."""
        ctx = ImpulseContext(
            user_idle_seconds=100,
            curiosity=0.6,
            rapport=0.3,
            recall_hit_count=3,
            messages_this_session=5,
            abstraction_level=0.8,
            impulse_budget_remaining=0.5,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.REFLECT

    def test_low_abstraction_impulse_prefers_speak(self):
        """Low abstraction level → SPEAK (concrete questions)."""
        ctx = ImpulseContext(
            user_idle_seconds=100,
            curiosity=0.6,
            rapport=0.3,
            recall_hit_count=1,
            messages_this_session=5,
            abstraction_level=0.2,
            impulse_budget_remaining=0.5,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.SPEAK

    def test_high_abstraction_metabolism_idle_connection(self):
        """High abstraction + IDLE → CONNECTION/IMPLICATION thought type."""
        ctx = MetabolicContext(
            user_idle_seconds=300,
            recall_hit_count=3,
            messages_this_session=5,
            abstraction_level=0.8,
        )
        tt = classify_thought_type(MetabolicMode.IDLE, ctx)
        assert tt == ThoughtType.CONNECTION

    def test_high_abstraction_metabolism_idle_implication(self):
        """High abstraction + IDLE + no recall → IMPLICATION."""
        ctx = MetabolicContext(
            user_idle_seconds=300,
            recall_hit_count=0,
            messages_this_session=5,
            abstraction_level=0.8,
        )
        tt = classify_thought_type(MetabolicMode.IDLE, ctx)
        assert tt == ThoughtType.IMPLICATION

    def test_low_abstraction_metabolism_idle_curiosity(self):
        """Low abstraction + IDLE + curiosity → CURIOSITY."""
        ctx = MetabolicContext(
            user_idle_seconds=300,
            curiosity=0.6,
            messages_this_session=5,
            abstraction_level=0.2,
        )
        tt = classify_thought_type(MetabolicMode.IDLE, ctx)
        assert tt == ThoughtType.CURIOSITY

    def test_impulse_context_has_abstraction_field(self):
        """ImpulseContext has abstraction_level field."""
        ctx = ImpulseContext(abstraction_level=0.8)
        assert ctx.abstraction_level == 0.8

    def test_metabolic_context_has_abstraction_field(self):
        """MetabolicContext has abstraction_level field."""
        ctx = MetabolicContext(abstraction_level=0.3)
        assert ctx.abstraction_level == 0.3
