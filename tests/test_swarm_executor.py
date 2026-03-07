"""Tests for DAGExecutor — topology, parallel merge, failure handling."""

import time
from unittest.mock import MagicMock
import pytest

from swarm.executor import compute_execution_groups, DAGExecutor
from swarm.conductor import SwarmPlan, SwarmStep
from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState


# --- Helper agents for testing ---

class WriterAgent(SwarmAgent):
    """Agent that writes a specific field."""

    def __init__(self, name_val, field, value, criticality_val="medium", delay=0):
        self._name = name_val
        self._field = field
        self._value = value
        self._criticality = criticality_val
        self._delay = delay

    @property
    def name(self):
        return self._name

    @property
    def criticality(self):
        return self._criticality

    @property
    def output_keys(self):
        return [self._field]

    def execute(self, state, config):
        if self._delay:
            time.sleep(self._delay)
        return state.with_updates(**{self._field: self._value})


class FailAgent(SwarmAgent):
    """Agent that always raises."""

    def __init__(self, name_val, criticality_val="medium"):
        self._name = name_val
        self._criticality = criticality_val

    @property
    def name(self):
        return self._name

    @property
    def criticality(self):
        return self._criticality

    @property
    def output_keys(self):
        return []

    def execute(self, state, config):
        raise RuntimeError(f"{self._name} failed")


# --- compute_execution_groups tests ---

class TestComputeExecutionGroups:
    """Kahn's topological sort with level grouping."""

    def test_empty_plan(self):
        plan = SwarmPlan(steps=())
        assert compute_execution_groups(plan) == []

    def test_single_step(self):
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
        ))
        groups = compute_execution_groups(plan)
        assert groups == [(0,)]

    def test_all_independent(self):
        """Three independent steps → single parallel group."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
            SwarmStep(agent="b", action="run"),
            SwarmStep(agent="c", action="run"),
        ))
        groups = compute_execution_groups(plan)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_linear_chain(self):
        """0 → 1 → 2 → all sequential."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
            SwarmStep(agent="b", action="run", depends_on=(0,)),
            SwarmStep(agent="c", action="run", depends_on=(1,)),
        ))
        groups = compute_execution_groups(plan)
        assert groups == [(0,), (1,), (2,)]

    def test_diamond_topology(self):
        """
        0 → 1
        0 → 2
        1,2 → 3
        Groups: (0,), (1,2), (3,)
        """
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
            SwarmStep(agent="b", action="run", depends_on=(0,)),
            SwarmStep(agent="c", action="run", depends_on=(0,)),
            SwarmStep(agent="d", action="run", depends_on=(1, 2)),
        ))
        groups = compute_execution_groups(plan)
        assert len(groups) == 3
        assert groups[0] == (0,)
        assert set(groups[1]) == {1, 2}
        assert groups[2] == (3,)

    def test_full_build_topology(self):
        """
        retrieval(0), memory(1) independent
        compile(2) depends on (0, 1)
        coding(3) depends on (2)
        Groups: (0, 1), (2,), (3,)
        """
        plan = SwarmPlan(steps=(
            SwarmStep(agent="retrieval", action="run"),
            SwarmStep(agent="memory", action="run"),
            SwarmStep(agent="compile", action="compile", depends_on=(0, 1)),
            SwarmStep(agent="coding", action="emit", depends_on=(2,)),
        ))
        groups = compute_execution_groups(plan)
        assert len(groups) == 3
        assert set(groups[0]) == {0, 1}
        assert groups[1] == (2,)
        assert groups[2] == (3,)

    def test_cycle_raises(self):
        """Circular dependency → ValueError."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run", depends_on=(1,)),
            SwarmStep(agent="b", action="run", depends_on=(0,)),
        ))
        with pytest.raises(ValueError, match="cycle"):
            compute_execution_groups(plan)


# --- DAGExecutor tests ---

class TestDAGExecutor:
    """Parallel group execution."""

    def test_single_step_execution(self):
        plan = SwarmPlan(steps=(
            SwarmStep(agent="retrieval", action="run"),
        ))
        agents = {
            "retrieval": WriterAgent("retrieval", "retrieval_context", {"relevant_documents": "test"}),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.retrieval_context == {"relevant_documents": "test"}
        assert "retrieval" in timings
        assert len(errors) == 0

    def test_parallel_merge(self):
        """Two agents in parallel writing different fields."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="retrieval", action="run"),
            SwarmStep(agent="memory", action="run"),
        ))
        agents = {
            "retrieval": WriterAgent("retrieval", "retrieval_context", {"relevant_documents": "docs"}),
            "memory": WriterAgent("memory", "memory_context", {"relevant_patterns": "patterns"}),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.retrieval_context == {"relevant_documents": "docs"}
        assert final.memory_context == {"relevant_patterns": "patterns"}
        assert len(errors) == 0

    def test_sequential_groups(self):
        """Groups execute in order: parallel group → sequential step."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="retrieval", action="run"),
            SwarmStep(agent="memory", action="run"),
            SwarmStep(agent="compile", action="compile", depends_on=(0, 1)),
        ))
        agents = {
            "retrieval": WriterAgent("retrieval", "retrieval_context", {"relevant_documents": "docs"}),
            "memory": WriterAgent("memory", "memory_context", {"relevant_patterns": "pats"}),
            "compile": WriterAgent("compile", "blueprint", {"components": []}, "critical"),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.retrieval_context is not None
        assert final.memory_context is not None
        assert final.blueprint == {"components": []}
        assert len(errors) == 0

    def test_critical_failure_aborts(self):
        """Critical agent failure stops the swarm."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="compile", action="compile"),
            SwarmStep(agent="coding", action="emit", depends_on=(0,)),
        ))
        agents = {
            "compile": FailAgent("compile", "critical"),
            "coding": WriterAgent("coding", "generated_code", {}, "high"),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert len(errors) == 1
        assert errors[0]["agent"] == "compile"
        # coding should not have run
        assert final.generated_code is None

    def test_non_critical_failure_continues(self):
        """Non-critical failure in parallel group doesn't abort."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="retrieval", action="run"),
            SwarmStep(agent="memory", action="run"),
        ))
        agents = {
            "retrieval": WriterAgent("retrieval", "retrieval_context", {"relevant_documents": "docs"}),
            "memory": FailAgent("memory", "low"),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        # retrieval should still have succeeded
        assert final.retrieval_context == {"relevant_documents": "docs"}
        assert len(errors) == 1
        assert errors[0]["agent"] == "memory"

    def test_cost_cap_stops_execution(self):
        """Cost cap prevents further groups from running."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
            SwarmStep(agent="b", action="run", depends_on=(0,)),
        ))
        agents = {
            "a": WriterAgent("a", "retrieval_context", {"relevant_documents": "x"}),
            "b": WriterAgent("b", "memory_context", {"relevant_patterns": "y"}),
        }
        state = SwarmState(intent="test", cost_accumulated_usd=10.0, cost_cap_usd=5.0)

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert any("Cost cap" in w for w in warnings)
        # Neither agent should have run
        assert final.retrieval_context is None

    def test_output_keys_only_merge(self):
        """Only declared output_keys are merged from agent results."""

        class SelectiveAgent(SwarmAgent):
            name = "selective"
            criticality = "medium"

            @property
            def output_keys(self):
                return ["retrieval_context"]

            def execute(self, state, config):
                # Writes to both retrieval_context AND memory_context
                return state.with_updates(
                    retrieval_context={"relevant_documents": "from_selective"},
                    memory_context={"relevant_patterns": "should_not_merge"},
                )

        plan = SwarmPlan(steps=(
            SwarmStep(agent="selective", action="run"),
        ))
        agents = {"selective": SelectiveAgent()}
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=1)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        # retrieval_context IS in output_keys → merged
        assert final.retrieval_context == {"relevant_documents": "from_selective"}
        # memory_context is NOT in output_keys → not merged
        assert final.memory_context is None

    def test_missing_agent(self):
        """Missing agent produces error."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="nonexistent", action="run"),
        ))
        agents = {}
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=1)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert len(errors) == 1
        assert "not registered" in errors[0]["message"]

    def test_progress_callback(self):
        """Progress callback fires for each step."""
        events = []

        def on_progress(agent, step_index, stage, message):
            events.append((agent, step_index, stage))

        plan = SwarmPlan(steps=(
            SwarmStep(agent="a", action="run"),
        ))
        agents = {"a": WriterAgent("a", "retrieval_context", {"relevant_documents": "x"})}
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=1)
        executor.execute(plan, state, agents, progress_callback=on_progress)

        assert ("a", 0, "starting") in events
        assert ("a", 0, "completed") in events

    def test_blocked_compile_emits_error(self):
        """A blocked compile should surface as an awaiting_decision error."""
        class BlockedCompileAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"

            @property
            def output_keys(self):
                return ["compile_result", "blueprint"]

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

        plan = SwarmPlan(steps=(
            SwarmStep(agent="compile", action="run"),
        ))
        agents = {"compile": BlockedCompileAgent()}
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=1)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.compile_result is not None
        assert final.compile_result["success"] is False
        assert any(err["error_type"] == "awaiting_decision" for err in errors)

    def test_blocking_semantic_gate_emits_error(self):
        """A successful compile with blocking escalations should still halt."""
        class BlockingCompileAgent(SwarmAgent):
            name = "compile"
            criticality = "critical"

            @property
            def output_keys(self):
                return ["compile_result", "blueprint"]

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

        plan = SwarmPlan(steps=(
            SwarmStep(agent="compile", action="run"),
        ))
        agents = {"compile": BlockingCompileAgent()}
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=1)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.compile_result is not None
        assert final.compile_result["success"] is True
        assert any(err["error_type"] == "awaiting_decision" for err in errors)

    def test_parallel_missing_agent_in_group(self):
        """Missing agent in parallel group doesn't crash other agents."""
        plan = SwarmPlan(steps=(
            SwarmStep(agent="good", action="run"),
            SwarmStep(agent="missing", action="run"),
        ))
        agents = {
            "good": WriterAgent("good", "retrieval_context", {"relevant_documents": "ok"}),
        }
        state = SwarmState(intent="test")

        executor = DAGExecutor(max_workers=2)
        final, timings, errors, warnings = executor.execute(plan, state, agents)

        assert final.retrieval_context == {"relevant_documents": "ok"}
        assert len(errors) == 1
        assert errors[0]["agent"] == "missing"
