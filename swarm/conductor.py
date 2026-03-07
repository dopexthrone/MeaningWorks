"""SwarmConductor — plans and executes multi-agent workflows.

Phase 1: Deterministic routing, sequential execution.
Phase 2: Intelligence agents (retrieval, memory, coding) + parallel DAG executor.
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

from swarm.agents.base import SwarmAgent
from swarm.agents.compile import CompileAgent, get_compile_halt_info
from swarm.agents.retrieval import RetrievalAgent
from swarm.agents.memory import MemoryAgent
from swarm.agents.coding import CodingAgent
from swarm.state import SwarmState, SwarmProgress, SwarmError, SwarmResult

logger = logging.getLogger("motherlabs.swarm.conductor")


@dataclass(frozen=True)
class SwarmStep:
    """A single step in a swarm execution plan."""

    agent: str
    action: str
    depends_on: Tuple[int, ...] = ()
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "depends_on": list(self.depends_on),
            "config": self.config,
        }


@dataclass(frozen=True)
class SwarmPlan:
    """Execution plan — ordered steps with dependency graph."""

    steps: Tuple[SwarmStep, ...]
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "estimated_cost_usd": self.estimated_cost_usd,
        }


# Cost estimates per agent (USD per invocation)
_AGENT_COST_ESTIMATES = {
    "compile": 0.15,    # ~6 LLM calls at ~$0.025 each
    "retrieval": 0.00,  # no LLM
    "memory": 0.00,     # no LLM
    "coding": 0.50,     # code emission LLM calls
}


class SwarmConductor:
    """Plans and executes swarm workflows.

    Phase 1: Deterministic plan routing, sequential execution.
    Phase 2: Intelligence agents + optional parallel DAG executor.
    """

    def __init__(self, use_parallel: bool = False, max_workers: int = 3):
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        self.agents: Dict[str, SwarmAgent] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in agents."""
        self.register_agent(CompileAgent())
        self.register_agent(RetrievalAgent())
        self.register_agent(MemoryAgent())
        self.register_agent(CodingAgent())

    def register_agent(self, agent: SwarmAgent):
        """Register an agent for use in plans."""
        self.agents[agent.name] = agent

    def list_agents(self) -> List[Dict[str, Any]]:
        """List available agents with metadata."""
        return [
            {
                "name": agent.name,
                "criticality": agent.criticality,
                "input_keys": agent.input_keys,
                "output_keys": agent.output_keys,
            }
            for agent in self.agents.values()
        ]

    def plan(self, state: SwarmState) -> SwarmPlan:
        """Route request_type to an execution plan.

        Deterministic routing, no LLM involved.
        Phase 2 plans include dependencies for parallel execution.
        """
        request_type = state.request_type

        if request_type == "compile_only":
            steps = (
                SwarmStep(agent="compile", action="compile", config={}),
            )

        elif request_type == "full_build":
            steps = (
                SwarmStep(agent="retrieval", action="search", config={}),
                SwarmStep(agent="memory", action="recall", config={}),
                SwarmStep(agent="compile", action="compile", depends_on=(0, 1), config={}),
                SwarmStep(agent="coding", action="emit", depends_on=(2,), config={}),
            )

        elif request_type == "research":
            steps = (
                SwarmStep(agent="retrieval", action="search", config={}),
            )

        elif request_type == "evolve":
            steps = (
                SwarmStep(agent="memory", action="recall", config={}),
                SwarmStep(agent="retrieval", action="search", config={}),
                SwarmStep(agent="compile", action="compile", depends_on=(0, 1), config={"enrich": True}),
                SwarmStep(agent="coding", action="emit", depends_on=(2,), config={}),
            )

        else:
            # Default to compile_only for unknown request types
            logger.warning("Unknown request_type '%s', defaulting to compile_only", request_type)
            steps = (
                SwarmStep(agent="compile", action="compile", config={}),
            )

        est_cost = sum(_AGENT_COST_ESTIMATES.get(s.agent, 0.0) for s in steps)
        return SwarmPlan(steps=steps, estimated_cost_usd=est_cost)

    def execute(self, state: SwarmState, progress_callback=None) -> SwarmResult:
        """Execute a swarm plan.

        When use_parallel=True and plan has dependencies, delegates to DAGExecutor.
        Otherwise falls back to sequential execution (Phase 1 compatibility).

        Args:
            state: Initial swarm state
            progress_callback: Optional fn(agent, step_index, stage, message)

        Returns:
            SwarmResult with final state, timings, and errors
        """
        start = time.time()
        plan = self.plan(state)
        state = state.with_updates(plan=plan.to_dict())

        # Empty plan = nothing to do
        if not plan.steps:
            return SwarmResult(
                success=True,
                state=state,
                plan_executed=plan.to_dict(),
                agent_timings={},
                total_cost_usd=0.0,
                total_duration_s=round(time.time() - start, 3),
                warnings=["No agents available for request_type '%s'" % state.request_type],
            )

        # Delegate to DAGExecutor if parallel mode and plan has dependencies
        has_deps = any(step.depends_on for step in plan.steps)
        if self.use_parallel and has_deps:
            return self._execute_parallel(plan, state, start, progress_callback)

        return self._execute_sequential(plan, state, start, progress_callback)

    def _execute_parallel(self, plan, state, start, progress_callback) -> SwarmResult:
        """Execute via DAGExecutor for plans with dependencies."""
        from swarm.executor import DAGExecutor

        executor = DAGExecutor(max_workers=self.max_workers)
        final_state, agent_timings, errors, warnings = executor.execute(
            plan, state, self.agents, progress_callback
        )

        total_duration = time.time() - start
        has_critical = any(
            not e.get("recoverable", True) for e in errors
        )

        return SwarmResult(
            success=len(errors) == 0 or not has_critical,
            state=final_state,
            plan_executed=plan.to_dict(),
            agent_timings=agent_timings,
            total_cost_usd=final_state.cost_accumulated_usd,
            total_duration_s=round(total_duration, 3),
            errors=errors,
            warnings=warnings,
        )

    def _execute_sequential(self, plan, state, start, progress_callback) -> SwarmResult:
        """Sequential execution (Phase 1 compatible)."""
        agent_timings: Dict[str, float] = {}
        errors: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for i, step in enumerate(plan.steps):
            agent_name = step.agent

            if agent_name not in self.agents:
                err = SwarmError(
                    agent=agent_name,
                    step_index=i,
                    error_type="agent_not_found",
                    message=f"Agent '{agent_name}' not registered",
                    recoverable=False,
                )
                errors.append(err.to_dict())
                return SwarmResult(
                    success=False,
                    state=state,
                    plan_executed=plan.to_dict(),
                    agent_timings=agent_timings,
                    total_cost_usd=state.cost_accumulated_usd,
                    total_duration_s=round(time.time() - start, 3),
                    errors=errors,
                )

            agent = self.agents[agent_name]

            # Emit progress
            if progress_callback:
                try:
                    progress_callback(agent_name, i, "starting", f"Starting {agent_name}")
                except Exception:
                    pass

            # Check cost cap
            if state.cost_accumulated_usd >= state.cost_cap_usd:
                warnings.append(
                    f"Cost cap ${state.cost_cap_usd:.2f} reached at step {i} "
                    f"(accumulated ${state.cost_accumulated_usd:.2f})"
                )
                return SwarmResult(
                    success=False,
                    state=state,
                    plan_executed=plan.to_dict(),
                    agent_timings=agent_timings,
                    total_cost_usd=state.cost_accumulated_usd,
                    total_duration_s=round(time.time() - start, 3),
                    errors=errors,
                    warnings=warnings,
                )

            # Execute agent — inject progress_callback into config for compile agent
            step_config = dict(step.config) if step.config else {}
            if progress_callback and agent_name == "compile":
                step_config["_progress_callback"] = progress_callback
            step_start = time.time()
            try:
                state = agent.execute(state, step_config)
                step_duration = time.time() - step_start
                agent_timings[agent_name] = round(step_duration, 3)

                logger.info(
                    "Step %d/%d (%s) completed in %.1fs",
                    i + 1, len(plan.steps), agent_name, step_duration,
                )

                if progress_callback:
                    try:
                        progress_callback(agent_name, i, "completed", f"{agent_name} completed")
                    except Exception:
                        pass

                if agent_name == "compile":
                    halt = get_compile_halt_info(state.compile_result)
                    if halt:
                        errors.append(SwarmError(
                            agent=agent_name,
                            step_index=i,
                            error_type=halt["error_type"],
                            message=halt["message"],
                            recoverable=bool(halt.get("recoverable", False)),
                        ).to_dict())
                        return SwarmResult(
                            success=False,
                            state=state,
                            plan_executed=plan.to_dict(),
                            agent_timings=agent_timings,
                            total_cost_usd=state.cost_accumulated_usd,
                            total_duration_s=round(time.time() - start, 3),
                            errors=errors,
                            warnings=warnings,
                        )

            except Exception as e:
                step_duration = time.time() - step_start
                agent_timings[agent_name] = round(step_duration, 3)

                err = SwarmError(
                    agent=agent_name,
                    step_index=i,
                    error_type=type(e).__name__,
                    message=str(e),
                    recoverable=agent.criticality != "critical",
                )
                errors.append(err.to_dict())

                logger.error(
                    "Step %d/%d (%s) failed after %.1fs: %s",
                    i + 1, len(plan.steps), agent_name, step_duration, e,
                )

                if agent.criticality == "critical":
                    # Critical agent failure → abort entire swarm
                    return SwarmResult(
                        success=False,
                        state=state,
                        plan_executed=plan.to_dict(),
                        agent_timings=agent_timings,
                        total_cost_usd=state.cost_accumulated_usd,
                        total_duration_s=round(time.time() - start, 3),
                        errors=errors,
                        warnings=warnings,
                    )
                else:
                    # Non-critical → record warning, continue
                    warnings.append(f"{agent_name} failed (non-critical): {e}")

        total_duration = time.time() - start
        return SwarmResult(
            success=len(errors) == 0 or all(
                not e.get("recoverable", True) is False
                for e in errors
            ),
            state=state,
            plan_executed=plan.to_dict(),
            agent_timings=agent_timings,
            total_cost_usd=state.cost_accumulated_usd,
            total_duration_s=round(total_duration, 3),
            errors=errors,
            warnings=warnings,
        )
