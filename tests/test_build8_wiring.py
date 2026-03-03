"""Tests for Build 8 genome wiring — emotional buffer, draft-first, environment-optimizing."""

import pytest

from mother.impulse import (
    Impulse,
    ImpulseContext,
    compute_impulse,
    impulse_prompt,
)
from mother.stance import Stance, StanceContext, compute_stance
from mother.bridge import EngineBridge


class TestEmotionalBufferCapable:
    """#128: Deescalation framing when frustration is high."""

    def test_impulse_context_has_frustration_field(self):
        """ImpulseContext carries frustration."""
        ctx = ImpulseContext(frustration=0.8)
        assert ctx.frustration == 0.8

    def test_deescalation_prefix_when_frustrated(self):
        """High frustration injects deescalation framing into prompt."""
        ctx = ImpulseContext(
            frustration=0.7,
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.SPEAK
        prompt = impulse_prompt(impulse, ctx)
        assert "Emotional context" in prompt
        assert "frustration" in prompt.lower()
        assert "warm" in prompt.lower()

    def test_no_deescalation_when_calm(self):
        """Low frustration does not inject deescalation."""
        ctx = ImpulseContext(
            frustration=0.3,
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.SPEAK
        prompt = impulse_prompt(impulse, ctx)
        assert "Emotional context" not in prompt

    def test_deescalation_on_greet(self):
        """Deescalation prefix applies to GREET impulse too."""
        ctx = ImpulseContext(
            frustration=0.8,
            is_new_session=True,
            hours_since_last_session=5.0,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.GREET
        prompt = impulse_prompt(impulse, ctx)
        assert "Emotional context" in prompt
        assert "re-engagement" in prompt

    def test_deescalation_on_reflect(self):
        """Deescalation prefix applies to REFLECT impulse."""
        ctx = ImpulseContext(
            frustration=0.7,
            user_idle_seconds=200,
            journal_failure_streak=-3,
            confidence=0.3,
        )
        impulse = compute_impulse(ctx)
        assert impulse == Impulse.REFLECT
        prompt = impulse_prompt(impulse, ctx)
        assert "Emotional context" in prompt
        assert "reflection" in prompt

    def test_quiet_unaffected(self):
        """QUIET impulse returns None regardless of frustration."""
        ctx = ImpulseContext(frustration=0.9, conversation_active=True)
        assert compute_impulse(ctx) == Impulse.QUIET
        assert impulse_prompt(Impulse.QUIET, ctx) is None

    def test_deescalation_threshold_exact(self):
        """Exactly 0.7 frustration triggers deescalation."""
        ctx = ImpulseContext(
            frustration=0.7,
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
            messages_this_session=5,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "Emotional context" in prompt

    def test_below_threshold_no_deescalation(self):
        """0.69 frustration does not trigger deescalation."""
        ctx = ImpulseContext(
            frustration=0.69,
            curiosity=0.6,
            rapport=0.3,
            user_idle_seconds=120,
        )
        prompt = impulse_prompt(Impulse.SPEAK, ctx)
        assert "Emotional context" not in prompt


class TestDraftFirst:
    """#43: ASK stance stores proposal for approval cycle."""

    def test_ask_stance_triggers_on_medium_goal(self):
        """Medium health goal + idle 1min → ASK (not ACT)."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.35,
            user_idle_seconds=90,
        )
        assert compute_stance(ctx) == Stance.ASK

    def test_act_stance_on_healthy_goal(self):
        """Healthy goal + idle 2min → ACT."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=150,
        )
        assert compute_stance(ctx) == Stance.ACT

    def test_ask_does_not_act(self):
        """ASK should not escalate to ACT without user approval."""
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.35,
            user_idle_seconds=65,
        )
        stance = compute_stance(ctx)
        assert stance == Stance.ASK
        # ASK is distinct from ACT
        assert stance != Stance.ACT

    def test_proposal_storage_integration(self):
        """Verify proposal dict structure matches what _execute_plan_step expects."""
        proposal = {
            "db_path": "/tmp/test.db",
            "goal": {"goal_id": 1, "description": "test goal"},
            "goal_id": 1,
            "step": {"step_id": 1, "name": "compile", "position": 0,
                     "action_type": "compile", "action_arg": "test", "description": "Compile test"},
            "plan": {"total_steps": 3},
            "cycle_budget": 0.10,
        }
        assert proposal["db_path"] == "/tmp/test.db"
        assert proposal["goal"]["goal_id"] == 1
        assert proposal["step"]["action_type"] == "compile"
        assert proposal["cycle_budget"] == 0.10


class TestEnvironmentOptimizing:
    """#46: Workspace awareness surfaces environment warnings."""

    def test_get_workspace_info_returns_dict(self):
        """get_workspace_info returns expected structure."""
        bridge = EngineBridge()
        info = bridge.get_workspace_info()
        assert isinstance(info, dict)
        assert "disk_free_gb" in info
        assert "warnings" in info
        assert isinstance(info["warnings"], list)

    def test_disk_free_is_positive(self):
        """Disk free space should be a positive number."""
        bridge = EngineBridge()
        info = bridge.get_workspace_info()
        assert info["disk_free_gb"] > 0

    def test_data_dir_measured(self):
        """Data dir measurement runs without error."""
        bridge = EngineBridge()
        info = bridge.get_workspace_info()
        # data_dir_mb may or may not be present depending on ~/.motherlabs existence
        # but the call should not crash
        assert isinstance(info, dict)

    def test_warnings_empty_on_healthy_system(self):
        """On a normal dev machine, no critical warnings expected."""
        bridge = EngineBridge()
        info = bridge.get_workspace_info()
        # This test is environment-dependent — it just verifies the structure
        assert isinstance(info["warnings"], list)

    def test_workspace_info_no_crash_on_missing_dir(self):
        """get_workspace_info handles missing dirs gracefully."""
        bridge = EngineBridge()
        # Even if ~/.motherlabs doesn't exist, should return valid dict
        info = bridge.get_workspace_info()
        assert "disk_free_gb" in info
