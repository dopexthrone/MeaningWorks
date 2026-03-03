"""
Tests for mother/impulse.py — dialogue initiative system.

Tests the impulse computation and prompt generation that gives
Mother the ability to speak unprompted: curiosity, observations,
reflections, re-engagement.
"""

import pytest

from mother.impulse import (
    Impulse,
    ImpulseContext,
    compute_impulse,
    impulse_prompt,
)


# =============================================================================
# Hard gates — must always produce QUIET
# =============================================================================

class TestImpulseHardGates:
    """Hard gates that suppress all impulses regardless of other signals."""

    def test_conversation_active_blocks_all(self):
        ctx = ImpulseContext(
            curiosity=0.9, rapport=0.8, user_idle_seconds=300,
            conversation_active=True,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_autonomous_working_blocks_all(self):
        ctx = ImpulseContext(
            curiosity=0.9, rapport=0.8, user_idle_seconds=300,
            autonomous_working=True,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_budget_exhausted_blocks_all(self):
        ctx = ImpulseContext(
            curiosity=0.9, rapport=0.8, user_idle_seconds=300,
            impulse_budget_remaining=0.0,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_low_vitality_blocks_all(self):
        ctx = ImpulseContext(
            curiosity=0.9, rapport=0.8, user_idle_seconds=300,
            vitality=0.1,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_rate_limit_blocks_all(self):
        ctx = ImpulseContext(
            curiosity=0.9, rapport=0.8, user_idle_seconds=300,
            impulse_actions_this_session=5,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_default_context_is_quiet(self):
        """Default ImpulseContext should produce QUIET."""
        ctx = ImpulseContext()
        assert compute_impulse(ctx) == Impulse.QUIET


# =============================================================================
# GREET — re-engagement after absence
# =============================================================================

class TestImpulseGreet:
    def test_new_session_long_absence(self):
        ctx = ImpulseContext(
            is_new_session=True,
            hours_since_last_session=5.0,
            user_idle_seconds=0.0,
        )
        assert compute_impulse(ctx) == Impulse.GREET

    def test_new_session_short_absence_no_greet(self):
        """Less than 2h absence doesn't trigger GREET."""
        ctx = ImpulseContext(
            is_new_session=True,
            hours_since_last_session=1.0,
            user_idle_seconds=0.0,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_long_idle_with_rapport(self):
        """30+ min idle + rapport triggers GREET."""
        ctx = ImpulseContext(
            user_idle_seconds=1800,
            rapport=0.3,
        )
        assert compute_impulse(ctx) == Impulse.GREET

    def test_long_idle_without_rapport_no_greet(self):
        """Long idle but no rapport = QUIET (don't presume familiarity)."""
        ctx = ImpulseContext(
            user_idle_seconds=1800,
            rapport=0.1,
        )
        assert compute_impulse(ctx) != Impulse.GREET


# =============================================================================
# OBSERVE — perception-driven observations
# =============================================================================

class TestImpulseObserve:
    def test_pending_screen_with_idle(self):
        ctx = ImpulseContext(
            has_pending_screen=True,
            user_idle_seconds=90,
            attentiveness=0.5,
        )
        assert compute_impulse(ctx) == Impulse.OBSERVE

    def test_pending_camera_with_idle(self):
        ctx = ImpulseContext(
            has_pending_camera=True,
            user_idle_seconds=90,
            attentiveness=0.5,
        )
        assert compute_impulse(ctx) == Impulse.OBSERVE

    def test_pending_screen_too_soon(self):
        """User only idle 20s — don't interrupt."""
        ctx = ImpulseContext(
            has_pending_screen=True,
            user_idle_seconds=20,
            attentiveness=0.5,
        )
        assert compute_impulse(ctx) == Impulse.QUIET

    def test_pending_screen_low_attentiveness(self):
        """Low attentiveness = not watching closely enough to observe."""
        ctx = ImpulseContext(
            has_pending_screen=True,
            user_idle_seconds=90,
            attentiveness=0.2,
        )
        assert compute_impulse(ctx) != Impulse.OBSERVE

    def test_no_pending_perception_no_observe(self):
        ctx = ImpulseContext(
            has_pending_screen=False,
            has_pending_camera=False,
            user_idle_seconds=90,
            attentiveness=0.8,
        )
        assert compute_impulse(ctx) != Impulse.OBSERVE


# =============================================================================
# REFLECT — memory/journal-driven insights
# =============================================================================

class TestImpulseReflect:
    def test_failure_streak_triggers_reflect(self):
        ctx = ImpulseContext(
            user_idle_seconds=150,
            journal_failure_streak=-3,
            confidence=0.3,
        )
        assert compute_impulse(ctx) == Impulse.REFLECT

    def test_failure_streak_not_enough_idle(self):
        ctx = ImpulseContext(
            user_idle_seconds=60,
            journal_failure_streak=-3,
            confidence=0.3,
        )
        assert compute_impulse(ctx) != Impulse.REFLECT

    def test_recall_patterns_with_rapport(self):
        ctx = ImpulseContext(
            user_idle_seconds=150,
            recall_hit_count=5,
            rapport=0.4,
        )
        assert compute_impulse(ctx) == Impulse.REFLECT

    def test_build_history_curiosity_rapport(self):
        ctx = ImpulseContext(
            user_idle_seconds=150,
            journal_total_builds=8,
            curiosity=0.6,
            rapport=0.4,
        )
        assert compute_impulse(ctx) == Impulse.REFLECT

    def test_build_history_low_curiosity(self):
        """History exists but curiosity too low — no reflection."""
        ctx = ImpulseContext(
            user_idle_seconds=150,
            journal_total_builds=8,
            curiosity=0.3,
            rapport=0.4,
        )
        assert compute_impulse(ctx) != Impulse.REFLECT


# =============================================================================
# SPEAK — curiosity-driven dialogue
# =============================================================================

class TestImpulseSpeak:
    def test_high_curiosity_decent_rapport(self):
        ctx = ImpulseContext(
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=90,
        )
        assert compute_impulse(ctx) == Impulse.SPEAK

    def test_very_high_curiosity_low_rapport(self):
        """Very high curiosity can override low rapport if enough messages."""
        ctx = ImpulseContext(
            curiosity=0.8,
            rapport=0.1,
            user_idle_seconds=90,
            messages_this_session=5,
        )
        assert compute_impulse(ctx) == Impulse.SPEAK

    def test_high_attentiveness_diverse_topics(self):
        ctx = ImpulseContext(
            attentiveness=0.7,
            unique_topic_count=4,
            user_idle_seconds=90,
        )
        assert compute_impulse(ctx) == Impulse.SPEAK

    def test_low_curiosity_no_speak(self):
        ctx = ImpulseContext(
            curiosity=0.2,
            rapport=0.5,
            user_idle_seconds=90,
        )
        assert compute_impulse(ctx) != Impulse.SPEAK

    def test_too_soon_no_speak(self):
        """User idle only 20s — don't initiate."""
        ctx = ImpulseContext(
            curiosity=0.8,
            rapport=0.5,
            user_idle_seconds=20,
        )
        assert compute_impulse(ctx) == Impulse.QUIET


# =============================================================================
# No goals required — the key test
# =============================================================================

class TestImpulseNoGoalsRequired:
    """Impulse system must work WITHOUT active goals. This is the whole point."""

    def test_speak_without_goals(self):
        """Mother can speak from curiosity alone, no goals needed."""
        ctx = ImpulseContext(
            curiosity=0.7,
            rapport=0.4,
            user_idle_seconds=120,
            # No goal-related fields at all
        )
        result = compute_impulse(ctx)
        assert result == Impulse.SPEAK

    def test_observe_without_goals(self):
        ctx = ImpulseContext(
            has_pending_screen=True,
            user_idle_seconds=90,
            attentiveness=0.5,
        )
        assert compute_impulse(ctx) == Impulse.OBSERVE

    def test_greet_without_goals(self):
        ctx = ImpulseContext(
            is_new_session=True,
            hours_since_last_session=24.0,
        )
        assert compute_impulse(ctx) == Impulse.GREET


# =============================================================================
# Priority ordering
# =============================================================================

class TestImpulsePriority:
    """When multiple impulses could fire, verify priority."""

    def test_greet_beats_observe(self):
        """GREET has priority over OBSERVE."""
        ctx = ImpulseContext(
            is_new_session=True,
            hours_since_last_session=5.0,
            has_pending_screen=True,
            attentiveness=0.8,
            user_idle_seconds=120,
        )
        assert compute_impulse(ctx) == Impulse.GREET

    def test_observe_beats_speak(self):
        """OBSERVE has priority over SPEAK when both could fire."""
        ctx = ImpulseContext(
            has_pending_screen=True,
            user_idle_seconds=90,
            attentiveness=0.6,
            curiosity=0.8,
            rapport=0.5,
        )
        assert compute_impulse(ctx) == Impulse.OBSERVE


# =============================================================================
# Prompt generation
# =============================================================================

class TestImpulsePrompt:
    def test_quiet_returns_none(self):
        ctx = ImpulseContext()
        assert impulse_prompt(Impulse.QUIET, ctx) is None

    def test_greet_new_session(self):
        ctx = ImpulseContext(
            is_new_session=True,
            hours_since_last_session=48.0,
        )
        prompt = impulse_prompt(Impulse.GREET, ctx)
        assert prompt is not None
        assert "re-engagement" in prompt
        assert "2 days" in prompt

    def test_greet_idle(self):
        ctx = ImpulseContext(
            user_idle_seconds=2000,
            rapport=0.3,
        )
        prompt = impulse_prompt(Impulse.GREET, ctx)
        assert prompt is not None
        assert "idle" in prompt.lower() or "quiet" in prompt.lower()

    def test_observe_screen(self):
        ctx = ImpulseContext(has_pending_screen=True)
        prompt = impulse_prompt(Impulse.OBSERVE, ctx)
        assert prompt is not None
        assert "screen" in prompt.lower()

    def test_observe_camera(self):
        ctx = ImpulseContext(has_pending_camera=True)
        prompt = impulse_prompt(Impulse.OBSERVE, ctx)
        assert prompt is not None
        assert "camera" in prompt.lower()

    def test_reflect_failure_streak(self):
        ctx = ImpulseContext(journal_failure_streak=-4)
        prompt = impulse_prompt(Impulse.REFLECT, ctx)
        assert prompt is not None
        assert "4 consecutive" in prompt

    def test_speak_with_topics(self):
        ctx = ImpulseContext(unique_topic_count=5)
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert prompt is not None
        assert "5 different topics" in prompt

    def test_speak_without_topics(self):
        ctx = ImpulseContext(unique_topic_count=1)
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert prompt is not None
        assert "curious" in prompt.lower()


# =============================================================================
# Integration with stance (independence)
# =============================================================================

class TestImpulseStanceIndependence:
    """Impulse and Stance are independent axes."""

    def test_stance_silent_impulse_can_fire(self):
        """Even when Stance would say SILENT (no goals), impulse can fire."""
        from mother.stance import compute_stance, StanceContext, Stance

        # Stance: no goals → SILENT
        stance_ctx = StanceContext(has_active_goals=False)
        assert compute_stance(stance_ctx) == Stance.SILENT

        # Impulse: curious → SPEAK
        impulse_ctx = ImpulseContext(
            curiosity=0.7,
            rapport=0.4,
            user_idle_seconds=120,
        )
        assert compute_impulse(impulse_ctx) == Impulse.SPEAK

    def test_both_can_coexist(self):
        """Stance ACT and Impulse SPEAK can both be true simultaneously."""
        from mother.stance import compute_stance, StanceContext, Stance

        stance_ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.8,
            user_idle_seconds=600,
        )
        assert compute_stance(stance_ctx) == Stance.ACT

        impulse_ctx = ImpulseContext(
            curiosity=0.7,
            rapport=0.4,
            user_idle_seconds=600,
        )
        assert compute_impulse(impulse_ctx) == Impulse.SPEAK


# =============================================================================
# Config defaults
# =============================================================================

class TestConfigDefaults:
    """Config changes reflect the new defaults."""

    def test_autonomous_enabled_by_default(self):
        from mother.config import MotherConfig
        cfg = MotherConfig()
        assert cfg.autonomous_enabled is True

    def test_dialogue_initiative_disabled_by_default(self):
        from mother.config import MotherConfig
        cfg = MotherConfig()
        assert cfg.dialogue_initiative_enabled is False

    def test_impulse_tick_seconds_default(self):
        from mother.config import MotherConfig
        cfg = MotherConfig()
        assert cfg.impulse_tick_seconds == 90

    def test_impulse_budget_default(self):
        from mother.config import MotherConfig
        cfg = MotherConfig()
        assert cfg.impulse_budget_per_session == 0.50


# =============================================================================
# Frame rule update
# =============================================================================

class TestFrameRuleUpdate:
    """Verify the frame rule now permits unsolicited dialogue."""

    def test_frame_rules_contain_talk_freely(self):
        from mother.context import _FRAME_RULES
        assert "Talk freely" in _FRAME_RULES
        assert "observe, wonder, ask" in _FRAME_RULES

    def test_frame_rules_no_longer_restrict_talking(self):
        from mother.context import _FRAME_RULES
        assert "Talk when talking" not in _FRAME_RULES
