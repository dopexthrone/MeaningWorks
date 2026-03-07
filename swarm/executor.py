"""DAGExecutor — parallel group execution via ThreadPoolExecutor.

Computes execution groups from a SwarmPlan's dependency graph (Kahn's algorithm),
then runs each group in parallel while groups themselves run sequentially.

Single-writer guarantee: each SwarmState field is written by exactly one agent,
so parallel agents within a group never conflict. Merge uses output_keys only.
"""

import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from swarm.agents.compile import get_compile_halt_info
from swarm.state import SwarmState, SwarmError

logger = logging.getLogger("motherlabs.swarm.executor")


def compute_execution_groups(plan) -> List[Tuple[int, ...]]:
    """Compute parallel execution groups via Kahn's topological sort.

    Steps with no unresolved dependencies form a group that can run concurrently.
    Groups are returned in execution order.

    Args:
        plan: SwarmPlan with .steps tuple of SwarmStep

    Returns:
        List of tuples, each tuple contains step indices that can run in parallel.
        Example: [(0, 1), (2,), (3,)] means steps 0+1 parallel, then 2, then 3.

    Raises:
        ValueError: If dependency graph has a cycle.
    """
    steps = plan.steps
    n = len(steps)

    if n == 0:
        return []

    # Build in-degree map and adjacency list
    in_degree = [0] * n
    dependents = defaultdict(list)  # step_idx -> list of steps that depend on it

    for i, step in enumerate(steps):
        for dep in step.depends_on:
            if 0 <= dep < n:
                in_degree[i] += 1
                dependents[dep].append(i)

    # Kahn's algorithm with level grouping
    groups = []
    queue = deque(i for i in range(n) if in_degree[i] == 0)

    processed = 0
    while queue:
        # All nodes in the current queue form one parallel group
        group = tuple(queue)
        groups.append(group)
        next_queue = deque()

        for idx in group:
            processed += 1
            for dep_idx in dependents[idx]:
                in_degree[dep_idx] -= 1
                if in_degree[dep_idx] == 0:
                    next_queue.append(dep_idx)

        queue = next_queue

    if processed != n:
        raise ValueError(
            f"Dependency cycle detected: processed {processed}/{n} steps"
        )

    return groups


class DAGExecutor:
    """Execute swarm plans with parallel group execution.

    Groups run sequentially; steps within a group run via ThreadPoolExecutor.
    Merge is safe because each agent writes to distinct output_keys.
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers

    def execute(
        self,
        plan,
        state: SwarmState,
        agents: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[SwarmState, Dict[str, float], List[Dict[str, Any]], List[str]]:
        """Execute a plan with parallel groups.

        Args:
            plan: SwarmPlan with steps and dependencies
            state: Initial SwarmState
            agents: Dict of agent_name -> SwarmAgent
            progress_callback: Optional fn(agent, step_index, stage, message)

        Returns:
            (final_state, agent_timings, errors, warnings)
        """
        groups = compute_execution_groups(plan)

        agent_timings: Dict[str, float] = {}
        errors: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for group in groups:
            # Check cost cap before each group
            if state.cost_accumulated_usd >= state.cost_cap_usd:
                warnings.append(
                    f"Cost cap ${state.cost_cap_usd:.2f} reached "
                    f"(accumulated ${state.cost_accumulated_usd:.2f})"
                )
                break

            if len(group) == 1:
                # Single step — no thread overhead
                idx = group[0]
                state, step_errors = self._execute_step(
                    idx, plan.steps[idx], state, agents, agent_timings, progress_callback
                )
                errors.extend(step_errors)

                # Check for critical failure
                if step_errors and self._is_critical_failure(step_errors, plan.steps[idx], agents):
                    break
            else:
                # Parallel execution
                state, group_errors = self._execute_group(
                    group, plan, state, agents, agent_timings, progress_callback
                )
                errors.extend(group_errors)

                # Check for critical failures in group
                for err in group_errors:
                    step_idx = err.get("step_index", -1)
                    if 0 <= step_idx < len(plan.steps):
                        if self._is_critical_failure([err], plan.steps[step_idx], agents):
                            warnings.append(f"Critical failure in parallel group, aborting")
                            return state, agent_timings, errors, warnings

        return state, agent_timings, errors, warnings

    def _execute_step(
        self,
        idx: int,
        step,
        state: SwarmState,
        agents: Dict[str, Any],
        agent_timings: Dict[str, float],
        progress_callback: Optional[Callable],
    ) -> Tuple[SwarmState, List[Dict[str, Any]]]:
        """Execute a single step, return (updated_state, errors)."""
        agent_name = step.agent
        step_errors = []

        if agent_name not in agents:
            err = SwarmError(
                agent=agent_name,
                step_index=idx,
                error_type="agent_not_found",
                message=f"Agent '{agent_name}' not registered",
                recoverable=False,
            )
            step_errors.append(err.to_dict())
            return state, step_errors

        agent = agents[agent_name]

        if progress_callback:
            try:
                progress_callback(agent_name, idx, "starting", f"Starting {agent_name}")
            except Exception:
                pass

        step_start = time.time()
        try:
            new_state = agent.execute(state, step.config)
            step_duration = time.time() - step_start
            agent_timings[agent_name] = round(step_duration, 3)

            # Merge only output_keys from new_state
            state = self._merge_output_keys(state, new_state, agent)

            if agent_name == "compile":
                halt = get_compile_halt_info(getattr(state, "compile_result", None))
                if halt:
                    step_errors.append(SwarmError(
                        agent=agent_name,
                        step_index=idx,
                        error_type=halt["error_type"],
                        message=halt["message"],
                        recoverable=bool(halt.get("recoverable", False)),
                    ).to_dict())
                    return state, step_errors

            if progress_callback:
                try:
                    progress_callback(agent_name, idx, "completed", f"{agent_name} completed")
                except Exception:
                    pass

        except Exception as e:
            step_duration = time.time() - step_start
            agent_timings[agent_name] = round(step_duration, 3)

            err = SwarmError(
                agent=agent_name,
                step_index=idx,
                error_type=type(e).__name__,
                message=str(e),
                recoverable=agent.criticality != "critical",
            )
            step_errors.append(err.to_dict())

            logger.error("Step %d (%s) failed: %s", idx, agent_name, e)

        return state, step_errors

    def _execute_group(
        self,
        group: Tuple[int, ...],
        plan,
        state: SwarmState,
        agents: Dict[str, Any],
        agent_timings: Dict[str, float],
        progress_callback: Optional[Callable],
    ) -> Tuple[SwarmState, List[Dict[str, Any]]]:
        """Execute a parallel group, merge results, return (updated_state, errors)."""
        group_errors = []
        results: Dict[int, SwarmState] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for idx in group:
                step = plan.steps[idx]
                agent_name = step.agent

                if agent_name not in agents:
                    err = SwarmError(
                        agent=agent_name,
                        step_index=idx,
                        error_type="agent_not_found",
                        message=f"Agent '{agent_name}' not registered",
                        recoverable=False,
                    )
                    group_errors.append(err.to_dict())
                    continue

                if progress_callback:
                    try:
                        progress_callback(agent_name, idx, "starting", f"Starting {agent_name}")
                    except Exception:
                        pass

                future = pool.submit(self._run_agent, agents[agent_name], state, step.config)
                futures[future] = (idx, step)

            for future in as_completed(futures):
                idx, step = futures[future]
                agent_name = step.agent
                agent = agents.get(agent_name)

                try:
                    new_state, duration = future.result()
                    agent_timings[agent_name] = round(duration, 3)
                    results[idx] = new_state

                    if agent_name == "compile":
                        halt = get_compile_halt_info(getattr(new_state, "compile_result", None))
                        if halt:
                            group_errors.append(SwarmError(
                                agent=agent_name,
                                step_index=idx,
                                error_type=halt["error_type"],
                                message=halt["message"],
                                recoverable=bool(halt.get("recoverable", False)),
                            ).to_dict())

                    if progress_callback:
                        try:
                            progress_callback(agent_name, idx, "completed", f"{agent_name} completed")
                        except Exception:
                            pass

                except Exception as e:
                    agent_timings[agent_name] = 0.0
                    err = SwarmError(
                        agent=agent_name,
                        step_index=idx,
                        error_type=type(e).__name__,
                        message=str(e),
                        recoverable=agent.criticality != "critical" if agent else False,
                    )
                    group_errors.append(err.to_dict())
                    logger.error("Parallel step %d (%s) failed: %s", idx, agent_name, e)

        # Merge all successful results using output_keys
        for idx in sorted(results.keys()):
            step = plan.steps[idx]
            agent = agents.get(step.agent)
            if agent:
                state = self._merge_output_keys(state, results[idx], agent)

        return state, group_errors

    @staticmethod
    def _run_agent(agent, state: SwarmState, config: Dict[str, Any]) -> Tuple[SwarmState, float]:
        """Run an agent and return (new_state, duration). Thread target."""
        start = time.time()
        new_state = agent.execute(state, config)
        return new_state, time.time() - start

    @staticmethod
    def _merge_output_keys(base_state: SwarmState, new_state: SwarmState, agent) -> SwarmState:
        """Merge only the agent's declared output_keys from new_state into base_state."""
        updates = {}
        for key in agent.output_keys:
            if hasattr(new_state, key):
                val = getattr(new_state, key)
                if val is not None:
                    updates[key] = val
        if updates:
            return base_state.with_updates(**updates)
        return base_state

    @staticmethod
    def _is_critical_failure(step_errors: List[Dict[str, Any]], step, agents: Dict[str, Any]) -> bool:
        """Check if any error in step_errors is from a critical agent."""
        agent = agents.get(step.agent)
        if agent and agent.criticality == "critical":
            return True
        return False
