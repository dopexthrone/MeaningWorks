"""Tests for kernel/dispatch.py — cell → action type dispatch."""

import pytest
from kernel.dispatch import (
    ActionDispatch,
    ACTION_MAP,
    dispatch_from_cell,
    action_requires_llm,
)


class TestDispatchFromCell:
    """dispatch_from_cell() mapping tests."""

    def test_obs_env_maps_to_perceive(self):
        action = dispatch_from_cell("OBS.ENV.APP.WHAT.USR", 5.0, "screen capture")
        assert action.action_type == "perceive"
        assert action.postcode == "OBS.ENV.APP.WHAT.USR"
        assert action.priority == 5.0
        assert action.context == "screen capture"

    def test_obs_usr_maps_to_observe_user(self):
        action = dispatch_from_cell("OBS.USR.APP.WHAT.USR", 3.0)
        assert action.action_type == "observe_user"

    def test_int_prj_maps_to_compile(self):
        action = dispatch_from_cell("INT.PRJ.APP.WHAT.USR", 9.0, "my project")
        assert action.action_type == "compile"
        assert action.requires_llm is True

    def test_int_tsk_maps_to_execute_task(self):
        action = dispatch_from_cell("INT.TSK.APP.WHAT.USR", 4.0)
        assert action.action_type == "execute_task"
        assert action.requires_llm is True

    def test_int_sem_maps_to_compile(self):
        action = dispatch_from_cell("INT.SEM.ECO.WHAT.SFT", 7.0)
        assert action.action_type == "compile"

    def test_met_mem_maps_to_reflect(self):
        action = dispatch_from_cell("MET.MEM.DOM.WHAT.MTH", 2.0)
        assert action.action_type == "reflect"
        assert action.requires_llm is True

    def test_met_gol_maps_to_self_improve(self):
        action = dispatch_from_cell("MET.GOL.DOM.WHY.MTH", 6.0)
        assert action.action_type == "self_improve"
        assert action.requires_llm is True

    def test_net_sta_maps_to_check_external(self):
        action = dispatch_from_cell("NET.STA.APP.WHAT.USR", 1.0)
        assert action.action_type == "check_external"

    def test_tme_sch_maps_to_check_schedule(self):
        action = dispatch_from_cell("TME.SCH.APP.WHEN.USR", 1.5)
        assert action.action_type == "check_schedule"

    def test_agn_tsk_maps_to_manage_agent(self):
        action = dispatch_from_cell("AGN.TSK.APP.WHAT.USR", 2.0)
        assert action.action_type == "manage_agent"

    def test_unknown_mapping_defaults_to_observe(self):
        action = dispatch_from_cell("STR.FNC.APP.HOW.SFT", 1.0)
        assert action.action_type == "observe"

    def test_invalid_postcode_raises(self):
        with pytest.raises(ValueError):
            dispatch_from_cell("INVALID", 1.0)

    def test_frozen_dataclass(self):
        action = dispatch_from_cell("OBS.ENV.APP.WHAT.USR", 5.0)
        with pytest.raises(AttributeError):
            action.action_type = "something_else"

    def test_empty_primitive(self):
        action = dispatch_from_cell("OBS.ENV.APP.WHAT.USR", 5.0)
        assert action.context == ""

    def test_priority_preserved(self):
        action = dispatch_from_cell("OBS.ENV.APP.WHAT.USR", 42.5)
        assert action.priority == 42.5


class TestActionRequiresLlm:
    """action_requires_llm() cost gate tests."""

    def test_compile_requires_llm(self):
        assert action_requires_llm("compile") is True

    def test_self_improve_requires_llm(self):
        assert action_requires_llm("self_improve") is True

    def test_reflect_requires_llm(self):
        assert action_requires_llm("reflect") is True

    def test_execute_task_requires_llm(self):
        assert action_requires_llm("execute_task") is True

    def test_perceive_does_not_require_llm(self):
        assert action_requires_llm("perceive") is False

    def test_observe_user_does_not_require_llm(self):
        assert action_requires_llm("observe_user") is False

    def test_check_external_does_not_require_llm(self):
        assert action_requires_llm("check_external") is False

    def test_check_schedule_does_not_require_llm(self):
        assert action_requires_llm("check_schedule") is False

    def test_manage_agent_does_not_require_llm(self):
        assert action_requires_llm("manage_agent") is False

    def test_unknown_action_does_not_require_llm(self):
        assert action_requires_llm("unknown_action") is False


class TestActionMapConsistency:
    """Verify ACTION_MAP entries use valid postcodes."""

    def test_all_actions_have_valid_layer_concern(self):
        from kernel.cell import LAYERS, CONCERNS
        for (layer, concern), action in ACTION_MAP.items():
            assert layer in LAYERS, f"Invalid layer {layer} in ACTION_MAP"
            assert concern in CONCERNS, f"Invalid concern {concern} in ACTION_MAP"

    def test_no_duplicate_actions_for_same_mapping(self):
        # Each (layer, concern) pair should map to exactly one action
        seen = set()
        for key in ACTION_MAP:
            assert key not in seen, f"Duplicate ACTION_MAP key: {key}"
            seen.add(key)

    def test_perceive_action_is_not_llm(self):
        """Perception refresh should be cheap (no LLM)."""
        for (layer, concern), action in ACTION_MAP.items():
            if action == "perceive":
                assert not action_requires_llm(action)
