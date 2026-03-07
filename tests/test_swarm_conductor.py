"""Tests for SwarmConductor — plan routing, execution, error handling."""

from unittest.mock import MagicMock, patch
from swarm.conductor import SwarmConductor, SwarmPlan, SwarmStep
from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState


class TestSwarmPlan:
    """Plan and step dataclasses."""

    def test_step_to_dict(self):
        step = SwarmStep(agent="compile", action="compile", depends_on=(0,), config={"k": "v"})
        d = step.to_dict()
        assert d["agent"] == "compile"
        assert d["depends_on"] == [0]
        assert d["config"] == {"k": "v"}

    def test_plan_to_dict(self):
        plan = SwarmPlan(
            steps=(SwarmStep(agent="compile", action="compile"),),
            estimated_cost_usd=0.15,
        )
        d = plan.to_dict()
        assert len(d["steps"]) == 1
        assert d["estimated_cost_usd"] == 0.15


class TestConductorPlanning:
    """SwarmConductor.plan() routing."""

    def test_plan_compile_only(self):
        state = SwarmState(intent="test", request_type="compile_only")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == "compile"

    def test_plan_full_build(self):
        state = SwarmState(intent="test", request_type="full_build")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        assert len(plan.steps) == 4
        agents = [s.agent for s in plan.steps]
        assert agents == ["retrieval", "memory", "compile", "coding"]
        # compile depends on retrieval + memory
        assert plan.steps[2].depends_on == (0, 1)
        # coding depends on compile
        assert plan.steps[3].depends_on == (2,)

    def test_plan_research(self):
        state = SwarmState(intent="test", request_type="research")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == "retrieval"

    def test_plan_evolve(self):
        state = SwarmState(intent="test", request_type="evolve")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        assert len(plan.steps) == 4
        agents = [s.agent for s in plan.steps]
        assert agents == ["memory", "retrieval", "compile", "coding"]
        assert plan.steps[2].depends_on == (0, 1)
        assert plan.steps[2].config.get("enrich") is True
        assert plan.steps[3].depends_on == (2,)

    def test_plan_full_build_cost_estimate(self):
        """Full build cost includes all 4 agents."""
        state = SwarmState(intent="test", request_type="full_build")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        # 0.15 (compile) + 0.00 (retrieval) + 0.00 (memory) + 0.50 (coding)
        assert plan.estimated_cost_usd == 0.65

    def test_plan_unknown_defaults_compile(self):
        state = SwarmState(intent="test", request_type="unknown_type")
        conductor = SwarmConductor()
        plan = conductor.plan(state)
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == "compile"


class TestConductorExecution:
    """SwarmConductor.execute() sequential execution."""

    def test_execute_empty_plan(self):
        """Empty plan → success with warning."""
        # Override plan() to return empty plan
        conductor = SwarmConductor()
        original_plan = conductor.plan
        conductor.plan = lambda state: SwarmPlan(steps=(), estimated_cost_usd=0.0)

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)
        assert result.success is True
        assert len(result.warnings) > 0

        conductor.plan = original_plan

    def test_execute_with_mock_agent(self):
        """Successful execution with a mock agent."""
        class MockAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state.with_updates(
                    blueprint={"components": [{"name": "Test"}]},
                )

        conductor = SwarmConductor()
        conductor.agents["compile"] = MockAgent()

        state = SwarmState(intent="Build test", request_type="compile_only")
        result = conductor.execute(state)

        assert result.success is True
        assert result.state.blueprint == {"components": [{"name": "Test"}]}
        assert "compile" in result.agent_timings
        assert result.total_duration_s >= 0

    def test_execute_critical_agent_failure(self):
        """Critical agent failure aborts the swarm."""
        class FailAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                raise RuntimeError("Compilation failed")

        conductor = SwarmConductor()
        conductor.agents["compile"] = FailAgent()

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0]["error_type"] == "RuntimeError"

    def test_execute_non_critical_agent_failure(self):
        """Non-critical agent failure continues execution."""
        class OptionalAgent(SwarmAgent):
            name = "compile"
            criticality = "low"
            def execute(self, state, config):
                raise RuntimeError("Non-critical failure")

        conductor = SwarmConductor()
        conductor.agents["compile"] = OptionalAgent()

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)

        # Should continue (success depends on no critical failures)
        assert len(result.warnings) > 0
        assert "Non-critical failure" in result.warnings[0]

    def test_execute_cost_cap(self):
        """Execution stops when cost cap is reached."""
        state = SwarmState(
            intent="test",
            request_type="compile_only",
            cost_accumulated_usd=10.0,
            cost_cap_usd=5.0,
        )
        conductor = SwarmConductor()
        result = conductor.execute(state)

        assert result.success is False
        assert any("Cost cap" in w for w in result.warnings)

    def test_execute_missing_agent(self):
        """Missing agent in plan → failure."""
        conductor = SwarmConductor()
        # Remove all agents
        conductor.agents.clear()

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)

        assert result.success is False
        assert len(result.errors) == 1
        assert "not registered" in result.errors[0]["message"]

    def test_execute_sets_plan_on_state(self):
        """Plan is written to state before execution."""
        class MockAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state

        conductor = SwarmConductor()
        conductor.agents["compile"] = MockAgent()

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)

        assert result.state.plan is not None
        assert len(result.state.plan["steps"]) == 1

    def test_execute_progress_callback(self):
        """Progress callback is called during execution."""
        events = []

        class MockAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state

        def on_progress(agent, step_index, stage, message):
            events.append((agent, step_index, stage))

        conductor = SwarmConductor()
        conductor.agents["compile"] = MockAgent()

        state = SwarmState(intent="test", request_type="compile_only")
        conductor.execute(state, progress_callback=on_progress)

        assert len(events) >= 2  # starting + completed
        assert events[0] == ("compile", 0, "starting")
        assert events[1] == ("compile", 0, "completed")

    def test_execute_stops_after_blocked_compile(self):
        """An unsuccessful compile step should halt before downstream agents run."""
        class BlockedCompileAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state.with_updates(
                    compile_result={
                        "success": False,
                        "error": "Clarification required before compilation can continue",
                        "fracture": {
                            "stage": "interrogation",
                            "competing_configs": ["A", "B"],
                            "collapsing_constraint": "Which direction should I take?",
                        },
                    },
                    blueprint={},
                )

        class ExplodingCodingAgent(SwarmAgent):
            name = "coding"
            criticality = "high"
            def execute(self, state, config):
                raise AssertionError("coding should not execute after a blocked compile")

        conductor = SwarmConductor()
        conductor.agents["compile"] = BlockedCompileAgent()
        conductor.agents["coding"] = ExplodingCodingAgent()

        state = SwarmState(intent="test", request_type="full_build")
        result = conductor.execute(state)

        assert result.success is False
        assert any(err["error_type"] == "awaiting_decision" for err in result.errors)

    def test_execute_stops_after_blocking_semantic_gate(self):
        """A successful compile can still halt before coding if a blocked node remains."""
        class BlockingCompileAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state.with_updates(
                    compile_result={
                        "success": True,
                        "blocking_escalations": [
                            {
                                "postcode": "CTR.LMT.APP.IF.SFT",
                                "question": "Numeric bounds target is still unresolved",
                                "options": [],
                            }
                        ],
                    },
                    blueprint={"components": [{"name": "Calculator"}]},
                )

        class ExplodingCodingAgent(SwarmAgent):
            name = "coding"
            criticality = "high"
            def execute(self, state, config):
                raise AssertionError("coding should not execute after a blocked semantic gate")

        conductor = SwarmConductor()
        conductor.agents["compile"] = BlockingCompileAgent()
        conductor.agents["coding"] = ExplodingCodingAgent()

        state = SwarmState(intent="test", request_type="full_build")
        result = conductor.execute(state)

        assert result.success is False
        assert any(err["error_type"] == "awaiting_decision" for err in result.errors)


class TestConductorAgentRegistry:
    """Agent registration and listing."""

    def test_default_agents(self):
        conductor = SwarmConductor()
        assert "compile" in conductor.agents
        assert "retrieval" in conductor.agents
        assert "memory" in conductor.agents
        assert "coding" in conductor.agents

    def test_register_custom_agent(self):
        class CustomAgent(SwarmAgent):
            name = "custom"
            criticality = "low"
            def execute(self, state, config):
                return state

        conductor = SwarmConductor()
        conductor.register_agent(CustomAgent())
        assert "custom" in conductor.agents

    def test_list_agents(self):
        conductor = SwarmConductor()
        agents = conductor.list_agents()
        assert len(agents) >= 4
        names = {a["name"] for a in agents}
        assert {"compile", "retrieval", "memory", "coding"} <= names
        assert "input_keys" in agents[0]
        assert "output_keys" in agents[0]


class TestConductorParallel:
    """Parallel execution via DAGExecutor."""

    def test_constructor_defaults(self):
        conductor = SwarmConductor()
        assert conductor.use_parallel is False
        assert conductor.max_workers == 3

    def test_constructor_parallel(self):
        conductor = SwarmConductor(use_parallel=True, max_workers=5)
        assert conductor.use_parallel is True
        assert conductor.max_workers == 5

    def test_sequential_when_no_deps(self):
        """compile_only has no deps → sequential even with use_parallel=True."""
        class MockAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"
            def execute(self, state, config):
                return state.with_updates(blueprint={"ok": True})

        conductor = SwarmConductor(use_parallel=True)
        conductor.agents["compile"] = MockAgent()

        state = SwarmState(intent="test", request_type="compile_only")
        result = conductor.execute(state)

        assert result.success is True
        assert result.state.blueprint == {"ok": True}

    def test_parallel_execution_full_build(self):
        """Full build with use_parallel delegates to DAGExecutor."""
        class StubAgent(SwarmAgent):
            def __init__(self, name_val, crit, out_keys, out_vals):
                self._name = name_val
                self._crit = crit
                self._out_keys = out_keys
                self._out_vals = out_vals
            @property
            def name(self): return self._name
            @property
            def criticality(self): return self._crit
            @property
            def output_keys(self): return self._out_keys
            def execute(self, state, config):
                return state.with_updates(**self._out_vals)

        conductor = SwarmConductor(use_parallel=True)
        conductor.agents["retrieval"] = StubAgent(
            "retrieval", "medium", ["retrieval_context"],
            {"retrieval_context": {"relevant_documents": "docs"}},
        )
        conductor.agents["memory"] = StubAgent(
            "memory", "low", ["memory_context"],
            {"memory_context": {"relevant_patterns": "pats"}},
        )
        conductor.agents["compile"] = StubAgent(
            "compile", "critical",
            ["blueprint", "verification", "context_graph", "compile_result", "trust"],
            {"blueprint": {"components": []}, "compile_result": {"success": True}},
        )
        conductor.agents["coding"] = StubAgent(
            "coding", "high", ["generated_code", "project_manifest"],
            {"generated_code": {"App": "code"}},
        )

        state = SwarmState(intent="test", request_type="full_build")
        result = conductor.execute(state)

        assert result.success is True
        assert result.state.retrieval_context is not None
        assert result.state.memory_context is not None
        assert result.state.blueprint is not None
        assert result.state.generated_code is not None
