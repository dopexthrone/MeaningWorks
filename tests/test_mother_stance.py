"""Tests for mother/stance.py — Stance computation."""

import pytest

from mother.stance import Stance, StanceContext, compute_stance


class TestStanceDefaults:
    def test_default_is_silent(self):
        ctx = StanceContext()
        assert compute_stance(ctx) == Stance.SILENT

    def test_stance_enum_values(self):
        assert Stance.ACT.value == "act"
        assert Stance.WAIT.value == "wait"
        assert Stance.ASK.value == "ask"
        assert Stance.SILENT.value == "silent"

    def test_stance_context_frozen(self):
        ctx = StanceContext()
        with pytest.raises(AttributeError):
            ctx.has_active_goals = True


class TestConversationActive:
    def test_conversation_active_always_silent(self):
        ctx = StanceContext(
            conversation_active=True,
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) == Stance.SILENT


class TestNoGoals:
    def test_no_goals_silent(self):
        ctx = StanceContext(
            has_active_goals=False,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) == Stance.SILENT


class TestLowHealth:
    def test_health_below_threshold_silent(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.1,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_health_at_threshold_boundary(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.2,
            user_idle_seconds=600,
        )
        # 0.2 is NOT < 0.2, so passes health gate. But 0.2 < 0.3 → WAIT
        assert compute_stance(ctx) != Stance.SILENT


class TestSessionCap:
    def test_five_actions_caps_to_silent(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            autonomous_actions_this_session=5,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_under_cap_not_silent(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            autonomous_actions_this_session=4,
        )
        assert compute_stance(ctx) != Stance.SILENT


class TestUserIdle:
    def test_recently_active_silent(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=30,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_at_idle_threshold_not_silent(self):
        # 60 seconds is NOT < 60, so passes idle gate
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=60,
        )
        assert compute_stance(ctx) != Stance.SILENT


class TestGraduatedResponse:
    def test_high_health_long_idle_act(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.8,
            user_idle_seconds=300,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_health_exactly_0_5_idle_120_act(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.5,
            user_idle_seconds=120,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_medium_health_medium_idle_act(self):
        # h=0.5 >= 0.5, i=180 >= 120 → ACT (was ASK with old thresholds)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.5,
            user_idle_seconds=180,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_health_0_6_idle_200_act(self):
        # h=0.6 >= 0.5, i=200 >= 120 → ACT (was ASK with old thresholds)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=200,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_medium_health_short_idle_act(self):
        # h=0.5 >= 0.5, i=130 >= 120 → ACT (was WAIT with old thresholds)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.5,
            user_idle_seconds=130,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_low_medium_health_long_idle_ask(self):
        # h=0.4 >= 0.3, i=600 >= 60 → ASK (was WAIT with old thresholds)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) == Stance.ASK

    def test_high_health_moderate_idle_act(self):
        # h=0.8 >= 0.5, i=200 >= 120 → ACT (was ASK with old thresholds)
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.8,
            user_idle_seconds=200,
        )
        assert compute_stance(ctx) == Stance.ACT


class TestNewThresholds:
    """Boundary tests for the updated stance thresholds."""

    def test_health_exactly_0_2_passes_floor(self):
        # 0.2 is NOT < 0.2 → passes health floor
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.2,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_health_0_19_blocked_by_floor(self):
        # 0.19 < 0.2 → SILENT
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.19,
            user_idle_seconds=600,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_session_cap_at_5(self):
        # 5 actions → SILENT
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            autonomous_actions_this_session=5,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_session_cap_at_4_allows(self):
        # 4 actions → not capped
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            autonomous_actions_this_session=4,
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_idle_59_blocked(self):
        # 59s < 60s → SILENT
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=59,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_idle_60_passes(self):
        # 60s is NOT < 60s → passes
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=60,
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_act_threshold_health_0_5_idle_120(self):
        # h=0.5 >= 0.5, i=120 >= 120 → ACT
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.5,
            user_idle_seconds=120,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_ask_threshold_health_0_3_idle_60(self):
        # h=0.3 >= 0.3, i=60 >= 60 → ASK
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.3,
            user_idle_seconds=60,
        )
        assert compute_stance(ctx) == Stance.ASK


class TestDeepFlowProtection:
    """Step 4: Momentum-protecting — deep flow forces SILENT."""

    def test_deep_flow_forces_silent(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            flow_state="deep",
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_shallow_flow_allows_action(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            flow_state="shallow",
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_idle_flow_allows_action(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            flow_state="idle",
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_empty_flow_allows_action(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            flow_state="",
        )
        assert compute_stance(ctx) != Stance.SILENT


class TestRefuseStance:
    """Step 7: Decision-reducing — REFUSE on high frustration + stale goals."""

    def test_refuse_on_high_frustration(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.25,
            user_idle_seconds=600,
            frustration=0.7,
        )
        assert compute_stance(ctx) == Stance.REFUSE

    def test_no_refuse_when_goals_healthy(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.5,
            user_idle_seconds=600,
            frustration=0.7,
        )
        assert compute_stance(ctx) != Stance.REFUSE

    def test_no_refuse_low_frustration(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.25,
            user_idle_seconds=600,
            frustration=0.3,
        )
        # Low frustration + low health → WAIT or ASK, not REFUSE
        assert compute_stance(ctx) != Stance.REFUSE

    def test_refuse_enum_value(self):
        assert Stance.REFUSE.value == "refuse"


class TestDynamicBudget:
    """Step 16: Threshold-aware — dynamic budget based on frustration/posture."""

    def test_dynamic_budget_frustration(self):
        # Frustrated (>=0.4) → budget drops from 5 to 3
        # 3 actions should cap it
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            frustration=0.5,
            autonomous_actions_this_session=3,
        )
        assert compute_stance(ctx) == Stance.SILENT

    def test_dynamic_budget_energized(self):
        # Energized → budget goes from 5 to 7
        # 5 actions should NOT cap it
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            posture_state="energized",
            autonomous_actions_this_session=5,
        )
        assert compute_stance(ctx) != Stance.SILENT

    def test_dynamic_budget_normal(self):
        # Normal: budget=5, 5 actions → SILENT
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.9,
            user_idle_seconds=600,
            autonomous_actions_this_session=5,
        )
        assert compute_stance(ctx) == Stance.SILENT


class TestMotherSourcedGoals:
    """Step 17: Sovereign — mother-sourced goals auto-execute."""

    def test_mother_sourced_goals_auto_execute(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=120,
            goal_source="mother",
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_user_sourced_same_conditions_asks(self):
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.4,
            user_idle_seconds=120,
            goal_source="user",
        )
        # h=0.4 >= 0.3, i=120 >= 60 → ASK (not ACT, needs h>=0.5 or i>=120)
        assert compute_stance(ctx) == Stance.ASK
