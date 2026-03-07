"""Motherlabs Swarm — composable multi-agent workflows.

Phase 1: Foundation — SwarmState, CompileAgent, sequential Conductor.
Phase 2: Intelligence agents (Retrieval, Memory, Coding) + DAGExecutor.
"""

from swarm.state import SwarmState, SwarmProgress, SwarmError, SwarmResult
from swarm.agents.base import SwarmAgent
from swarm.agents.compile import CompileAgent
from swarm.agents.retrieval import RetrievalAgent
from swarm.agents.memory import MemoryAgent
from swarm.agents.coding import CodingAgent
from swarm.conductor import SwarmConductor, SwarmPlan, SwarmStep
from swarm.executor import DAGExecutor

__all__ = [
    "SwarmState",
    "SwarmProgress",
    "SwarmError",
    "SwarmResult",
    "SwarmAgent",
    "CompileAgent",
    "RetrievalAgent",
    "MemoryAgent",
    "CodingAgent",
    "SwarmConductor",
    "SwarmPlan",
    "SwarmStep",
    "DAGExecutor",
]
