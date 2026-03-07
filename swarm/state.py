"""SwarmState — immutable-handoff state for multi-agent workflows.

Each agent reads from state and returns a new state via with_updates().
Fields are typed slots written by exactly one agent.
"""

import copy
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SwarmState:
    """Central state object passed between swarm agents.

    Immutable handoff: agents call state.with_updates() to produce
    a new state rather than mutating in place.
    """

    swarm_id: str = ""
    user_id: str = "anonymous"
    session_id: str = ""
    intent: str = ""                              # Original user input (L4 locked)
    domain: str = "software"                      # Target domain (L4 locked)
    request_type: str = "compile_only"            # compile_only|full_build|research|evolve

    # LLM config
    llm_api_key: Optional[str] = None
    provider: str = "auto"

    # Agent outputs (each written by exactly one agent)
    research_context: Optional[Dict[str, Any]] = None   # Written by: Research
    retrieval_context: Optional[Dict[str, Any]] = None  # Written by: Retrieval
    memory_context: Optional[Dict[str, Any]] = None     # Written by: Memory
    blueprint: Optional[Dict[str, Any]] = None          # Written by: Compile
    verification: Optional[Dict[str, Any]] = None       # Written by: Compile
    context_graph: Optional[Dict[str, Any]] = None      # Written by: Compile
    compile_result: Optional[Dict[str, Any]] = None     # Written by: Compile (serialized)
    generated_code: Optional[Dict[str, str]] = None     # Written by: Coding
    project_manifest: Optional[Dict[str, Any]] = None   # Written by: Coding
    security_audit: Optional[Dict[str, Any]] = None     # Written by: Security
    test_suite: Optional[Dict[str, Any]] = None         # Written by: Testing
    trust: Optional[Dict[str, Any]] = None              # Written by: Compile
    stub_report: Optional[Dict[str, Any]] = None        # Written by: Coding (post-emission)

    # Evolve: carry forward from previous compilation
    previous_blueprint: Optional[Dict[str, Any]] = None   # Blueprint from prior task
    previous_task_id: Optional[str] = None                 # Source task ID
    previous_generated_code: Optional[Dict[str, str]] = None  # Code from prior task
    evolve_applied: bool = False                           # True if previous blueprint was actually used

    # Coordination
    plan: Optional[Dict[str, Any]] = None               # Written by: Conductor
    progress: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    cost_accumulated_usd: float = 0.0
    cost_cap_usd: float = 5.0

    def __post_init__(self):
        if not self.swarm_id:
            self.swarm_id = str(uuid.uuid4())
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

    def with_updates(self, **kwargs: Any) -> "SwarmState":
        """Immutable handoff — return new state with updates applied."""
        new = copy.copy(self)
        for k, v in kwargs.items():
            if not hasattr(new, k):
                raise AttributeError(f"SwarmState has no field '{k}'")
            setattr(new, k, v)
        return new

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Huey task result storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SwarmState":
        """Deserialize from Huey task result."""
        # Filter to only known fields
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        state = cls(**filtered)
        return state


@dataclass(frozen=True)
class SwarmProgress:
    """Progress event emitted by an agent during execution."""

    agent: str
    step_index: int
    stage: str
    message: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SwarmError:
    """Error event from an agent."""

    agent: str
    step_index: int
    error_type: str
    message: str
    recoverable: bool
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SwarmResult:
    """Final result from a swarm execution."""

    success: bool
    state: SwarmState
    plan_executed: Dict[str, Any] = field(default_factory=dict)
    agent_timings: Dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "success": self.success,
            "state": self.state.to_dict(),
            "plan_executed": self.plan_executed,
            "agent_timings": self.agent_timings,
            "total_cost_usd": self.total_cost_usd,
            "total_duration_s": self.total_duration_s,
            "errors": self.errors,
            "warnings": self.warnings,
        }
