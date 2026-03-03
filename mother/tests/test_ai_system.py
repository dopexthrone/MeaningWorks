import pytest
from core.ai_system import (
    AISystem,
    PromptCompilation,
    AgentOrchestration,
    VerificationLogic,
    DetailedFailureModesInAgentOrchestrationHandoffs,
    CompileResult
)

def test_prompt_compilation():
    pc = PromptCompilation("test_instance_id")
    pc.initialize()
    result = pc.process("test task")
    assert "CRITICAL" in result
    assert "test_instance_id" in result
    assert len(pc.compiler_prompts) == 1
    pc.configure({})
    assert pc.configured

def test_agent_orchestration():
    ao = AgentOrchestration("test_instance_id")
    ao.initialize()
    result = ao.process("test prompt")
    assert "BLUEPRINT" in result
    assert "test_instance_id" in result
    ao.configure({})
    assert ao.configured
    assert "ready" in ao.agent_orchestration

def test_verification_logic():
    vl = VerificationLogic("test_instance_id")
    vl.initialize()
    blueprint = "test blueprint " + "instance_id=test_instance_id " * 5
    result = vl.process(blueprint)
    assert isinstance(result, CompileResult)
    assert not result.rejected
    assert all(s >= 92.0 for s in result.scores.values())
    assert result.reason == ""
    vl.configure({})
    assert vl.configured

def test_failure_modes():
    fm = DetailedFailureModesInAgentOrchestrationHandoffs()
    mode_id = fm.create_detailed_failure_modes_in_agent_orchestration_handoffs("fm1", "test desc")
    assert mode_id == "fm1"
    got = fm.get_detailed_failure_modes_in_agent_orchestration_handoffs("fm1")
    assert got["description"] == "test desc"
    assert fm.update_detailed_failure_modes_in_agent_orchestration_handoffs("fm1", severity="high")
    assert fm.get_detailed_failure_modes_in_agent_orchestration_handoffs("fm1")["severity"] == "high"
    assert fm.delete_detailed_failure_modes_in_agent_orchestration_handoffs("fm1")
    assert fm.get_detailed_failure_modes_in_agent_orchestration_handoffs("fm1") is None

def test_ai_system_integrate_pipeline():
    ai = AISystem("test_instance_id")
    ai.integrate_pipeline()
    # Checks initialization
    assert ai.prompt_compilation.configured is False  # configure not called
    # failure mode created
    fm = ai.detailed_failure_modes_in_agent_orchestration_handoffs
    assert fm.get_detailed_failure_modes_in_agent_orchestration_handoffs("missing_provenance") is not None

def test_pipeline_end_to_end():
    ai = AISystem("test_instance_id")
    ai.integrate_pipeline()
    prompt = ai.prompt_compilation.process("test")
    orch = ai.agent_orchestration.process(prompt)
    verified = ai.verification_logic.process(orch)
    assert not verified.rejected
