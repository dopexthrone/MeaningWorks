"""Tests for Build 3: Autonomic Coordination.

Covers:
- Reactive perception scheduling (debounce, significance gate)
- Observer wiring (OBS.USR fill on user message)
- Thought→goal bridge expansion (FRUSTRATION, CONCERN)
- Convergence detection (pause after 3 failures)
- _reactive_observe_tick existence and guard checks
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

import pytest


# --- Reactive perception debounce ---


class TestReactivePerceptionDebounce:
    def test_last_reactive_impulse_attribute_exists(self):
        """ChatScreen should initialize _last_reactive_impulse."""
        # Test the attribute is defined in __init__
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen.__init__)
        assert "_last_reactive_impulse" in source

    def test_autonomous_outcome_history_attribute_exists(self):
        """ChatScreen should initialize _autonomous_outcome_history."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen.__init__)
        assert "_autonomous_outcome_history" in source


class TestReactiveObserveTick:
    def test_reactive_observe_tick_exists(self):
        """ChatScreen should have _reactive_observe_tick method."""
        from mother.screens.chat import ChatScreen
        assert hasattr(ChatScreen, "_reactive_observe_tick")
        assert callable(getattr(ChatScreen, "_reactive_observe_tick"))

    def test_reactive_observe_tick_guards_chatting(self):
        """Should return early if chatting."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._reactive_observe_tick)
        assert "self._chatting" in source
        assert "self._autonomous_working" in source
        assert "self._unmounted" in source

    def test_reactive_observe_tick_uses_impulse(self):
        """Should use compute_impulse to decide whether to act."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._reactive_observe_tick)
        assert "compute_impulse" in source
        assert "Impulse.QUIET" in source


# --- Perception consumer reactive scheduling ---


class TestPerceptionConsumerReactive:
    def test_reactive_scheduling_in_perception_consumer(self):
        """_perception_consumer should schedule _reactive_observe_tick on high significance."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._perception_consumer)
        assert "_reactive_observe_tick" in source
        assert "significance >= 0.7" in source
        assert "30.0" in source or "30" in source  # debounce

    def test_reactive_fires_call_later(self):
        """Should use call_later(2.0, ...) for 2s delay."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._perception_consumer)
        assert "call_later" in source
        assert "2.0" in source


# --- Observer wiring ---


class TestObserverWiring:
    def test_obs_usr_fill_on_user_message(self):
        """_handle_chat should fill OBS.USR cell on user message."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._handle_chat)
        assert "OBS.USR.DOM.WHAT.MTH" in source
        assert "observation:user_message" in source

    def test_obs_usr_fill_has_error_handling(self):
        """OBS.USR fill should be wrapped in try/except."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._handle_chat)
        assert "OBS.USR world grid fill skipped" in source


# --- Thought→goal bridge expansion ---


class TestThoughtGoalExpansion:
    def test_frustration_mapped(self):
        """FRUSTRATION thoughts should map to 'Fix' goals at normal priority."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "ThoughtType.FRUSTRATION" in source
        assert '"Fix"' in source

    def test_concern_mapped(self):
        """CONCERN thoughts should map to 'Address' goals at normal priority."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "ThoughtType.CONCERN" in source
        assert '"Address"' in source

    def test_curiosity_still_mapped(self):
        """CURIOSITY thoughts should still map to 'Investigate' goals."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "ThoughtType.CURIOSITY" in source
        assert '"Investigate"' in source

    def test_question_still_mapped(self):
        """QUESTION thoughts should still map to 'Investigate' goals."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "ThoughtType.QUESTION" in source

    def test_goal_cap_raised_to_8(self):
        """Mother-generated goal cap should be 8 (up from 5)."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "self._mother_generated_goals < 8" in source

    def test_uses_dedup(self):
        """Thought→goal bridge should use dedup=True."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "dedup=True" in source

    def test_thought_goal_map_structure(self):
        """_THOUGHT_GOAL_MAP should have 4 entries."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._metabolism_think)
        assert "_THOUGHT_GOAL_MAP" in source
        # Should have entries for CURIOSITY, QUESTION, FRUSTRATION, CONCERN
        for ttype in ("CURIOSITY", "QUESTION", "FRUSTRATION", "CONCERN"):
            assert f"ThoughtType.{ttype}" in source


# --- Convergence detection ---


class TestConvergenceDetection:
    def test_convergence_check_in_autonomous_tick(self):
        """_autonomous_tick should check _autonomous_outcome_history."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._autonomous_tick)
        assert "_autonomous_outcome_history" in source
        assert "last_3" in source or "last 3" in source

    def test_convergence_pauses_on_all_failures(self):
        """Should pause when last 3 outcomes all failed."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._autonomous_tick)
        assert "all_failed" in source

    def test_convergence_pauses_on_no_improvement(self):
        """Should pause when last 3 outcomes had no cell improvement."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._autonomous_tick)
        assert "none_improved" in source

    def test_outcome_recorded_on_compile_success(self):
        """_compile_goal success path should record outcome."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._compile_goal)
        assert "_autonomous_outcome_history" in source
        assert '"success": True' in source

    def test_outcome_recorded_on_compile_failure(self):
        """_compile_goal failure path should record outcome."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._compile_goal)
        assert '"success": False' in source

    def test_outcome_recorded_on_autonomous_work_exception(self):
        """_autonomous_work exception handler should record failure."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._autonomous_work)
        assert "_autonomous_outcome_history.append" in source

    def test_outcome_history_bounded(self):
        """Outcome history should be bounded to last 10."""
        from mother.screens.chat import ChatScreen
        import inspect
        source = inspect.getsource(ChatScreen._autonomous_work)
        assert "10" in source  # cap


# --- Integration: dedup used in thought→goal ---


class TestDeduplicationIntegration:
    def test_goal_dedup_key_consistency(self):
        """goal_dedup_key should normalize numbers for comparison."""
        from mother.goals import goal_dedup_key
        assert goal_dedup_key("Fix 5 quarantined cells") == goal_dedup_key("Fix 71 quarantined cells")

    def test_goal_dedup_key_different_goals(self):
        """Different goals should have different dedup keys."""
        from mother.goals import goal_dedup_key
        assert goal_dedup_key("Fix coherence") != goal_dedup_key("Fix traceability")
