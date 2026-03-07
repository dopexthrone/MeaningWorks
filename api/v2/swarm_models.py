"""Pydantic models for swarm API endpoints.

Follows patterns from api/v2/models.py.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SwarmExecuteRequest(BaseModel):
    """Request body for POST /v2/swarm/execute."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Natural language description of intent",
    )
    domain: str = Field(
        "software",
        description="Target domain (software, process, api, agent_system)",
    )
    request_type: str = Field(
        "compile_only",
        description="Workflow type: compile_only|full_build|research|evolve",
    )
    cost_cap_usd: float = Field(
        5.0,
        ge=0.1,
        le=100.0,
        description="Maximum cost in USD before aborting",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override (claude, openai, grok, gemini)",
    )
    previous_task_id: Optional[str] = Field(
        None,
        description="Task ID of a prior compilation to evolve from (carries blueprint forward)",
    )


class SwarmExecuteResponse(BaseModel):
    """Response for POST /v2/swarm/execute (async)."""

    task_id: str
    estimated_cost_usd: float = 0.0


class SwarmStatusResponse(BaseModel):
    """Response for GET /v2/swarm/status/{task_id}."""

    task_id: str
    status: str = "queued"  # queued|running|completed|failed
    progress_pct: float = 0.0
    current_agent: Optional[str] = None
    agent_statuses: Dict[str, str] = Field(default_factory=dict)


class SwarmResultResponse(BaseModel):
    """Response for GET /v2/swarm/result/{task_id} and POST /v2/swarm/execute/sync."""

    success: bool
    swarm_id: str = ""
    evolve_applied: bool = False
    blueprint: Optional[Dict[str, Any]] = None
    materialized_output: Optional[Dict[str, str]] = None
    trust: Optional[Dict[str, Any]] = None
    verification: Optional[Dict[str, Any]] = None
    generated_code: Optional[Dict[str, str]] = None
    project_files: Optional[Dict[str, str]] = None
    project_name: Optional[str] = None
    agent_timings: Dict[str, float] = Field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SwarmAgentInfo(BaseModel):
    """Agent metadata for GET /v2/swarm/agents."""

    name: str
    criticality: str
    input_keys: List[str] = Field(default_factory=list)
    output_keys: List[str] = Field(default_factory=list)


class SwarmAgentsResponse(BaseModel):
    """Response for GET /v2/swarm/agents."""

    agents: List[SwarmAgentInfo] = Field(default_factory=list)
