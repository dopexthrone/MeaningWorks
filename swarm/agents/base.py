"""SwarmAgent ABC — base class for all swarm agents.

Each agent reads specific SwarmState fields (input_keys),
executes its logic, and writes to specific fields (output_keys).
Agents must never mutate the input state — always use state.with_updates().
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from swarm.state import SwarmState


class SwarmAgent(ABC):
    """Base class for all swarm agents."""

    name: str = "base"
    criticality: str = "medium"  # critical|high|medium|low

    @abstractmethod
    def execute(self, state: SwarmState, config: Dict[str, Any]) -> SwarmState:
        """Execute agent logic, return updated state.

        Must never mutate input state. Use state.with_updates() to
        produce a new state with this agent's outputs.

        Args:
            state: Current swarm state (read-only)
            config: Step-specific configuration from the plan

        Returns:
            New SwarmState with this agent's outputs written
        """

    @property
    def input_keys(self) -> List[str]:
        """SwarmState fields this agent reads."""
        return []

    @property
    def output_keys(self) -> List[str]:
        """SwarmState fields this agent writes."""
        return []
