"""
Motherlabs V2 API Models — request/response with trust indicators and domain routing.

Phase D: V2 API + Platform Layer
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# V2 REQUEST MODELS
# =============================================================================

class V2CompileRequest(BaseModel):
    """Request body for POST /v2/compile — universal domain-parameterized compilation."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of intent",
    )
    domain: str = Field(
        "software",
        description="Domain adapter to use (software, process, etc.)",
    )
    trust_level: str = Field(
        "standard",
        description="Trust computation depth: 'fast' | 'standard' | 'thorough'",
    )
    materialize: bool = Field(
        True,
        description="Whether to produce output files",
    )
    output_format: Optional[str] = Field(
        None,
        description="Override adapter default output format",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override (claude, openai, grok, gemini)",
    )
    enrich: bool = Field(
        False,
        description="Auto-enrich sparse input before compilation",
    )
    canonical_components: Optional[List[str]] = Field(
        None,
        description="Components that MUST appear in output",
    )
    canonical_relationships: Optional[List[List[str]]] = Field(
        None,
        description="Relationships that MUST appear: [[from, to, type], ...]",
    )


class V2CompileTreeRequest(BaseModel):
    """Request body for POST /v2/compile-tree — multi-subsystem compilation."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of intent",
    )
    domain: str = Field("software")
    max_children: int = Field(8, ge=1, le=20)
    enrich: bool = Field(False)


class V2MaterializeRequest(BaseModel):
    """Request body for POST /v2/materialize — materialize blueprint per domain."""

    blueprint: Dict[str, Any] = Field(
        ...,
        description="Blueprint JSON to materialize",
    )
    domain: str = Field("software")
    interface_map: Optional[Dict[str, Any]] = None
    dim_meta: Optional[Dict[str, Any]] = None
    max_tokens: int = Field(4096, ge=256, le=16384)


class V2RecompileRequest(BaseModel):
    """Request body for POST /v2/recompile — evolve a running system."""

    current_blueprint: Dict[str, Any] = Field(
        ...,
        description="The current system blueprint (embedded at generation time)",
    )
    enhancement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="What capability the system is missing",
    )
    domain: str = Field(
        "agent_system",
        description="Domain adapter to use for recompilation",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override",
    )


class V2ValidateRequest(BaseModel):
    """Request body for POST /v2/validate — trust-only mode (no materialization)."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )
    domain: str = Field("software")


# =============================================================================
# V2 TRUST RESPONSE MODEL
# =============================================================================

class TrustResponse(BaseModel):
    """Trust indicators — THE PRODUCT."""

    overall_score: float = 0.0
    provenance_depth: int = 0
    fidelity_scores: Dict[str, int] = {}
    gap_report: List[str] = []
    verification_badge: str = "unverified"
    silence_zones: List[str] = []
    confidence_trajectory: List[float] = []
    derivation_chain_length: float = 0.0
    dimensional_coverage: Dict[str, float] = {}


class UsageResponse(BaseModel):
    """Usage/metering info."""

    tokens: int = 0
    cost_usd: float = 0.0
    domain: str = "software"
    adapter_version: str = "1.0"


class V2RecompileResponse(BaseModel):
    """Response body for POST /v2/recompile."""

    success: bool
    blueprint: Dict[str, Any] = {}
    materialized_output: Dict[str, str] = {}
    trust: TrustResponse = Field(default_factory=TrustResponse)
    domain: str = "agent_system"
    enhancement_applied: str = ""
    error: Optional[str] = None


# =============================================================================
# V2 RESPONSE MODELS
# =============================================================================

class V2CompileResponse(BaseModel):
    """Response body for POST /v2/compile."""

    success: bool
    blueprint: Dict[str, Any] = {}
    materialized_output: Dict[str, str] = {}

    # THE PRODUCT
    trust: TrustResponse = Field(default_factory=TrustResponse)

    # Semantic data
    dimensional_metadata: Dict[str, Any] = {}
    interface_map: Dict[str, Any] = {}
    verification: Dict[str, Any] = {}
    context_graph: Dict[str, Any] = {}

    # Metadata
    domain: str = "software"
    adapter_version: str = "1.0"
    usage: UsageResponse = Field(default_factory=UsageResponse)
    error: Optional[str] = None


class V2CompileTreeResponse(BaseModel):
    """Response body for POST /v2/compile-tree."""

    success: bool
    root_blueprint: Dict[str, Any] = {}
    child_blueprints: List[Dict[str, Any]] = []
    l2_synthesis: Dict[str, Any] = {}
    trust: TrustResponse = Field(default_factory=TrustResponse)
    domain: str = "software"
    error: Optional[str] = None


class V2MaterializeResponse(BaseModel):
    """Response body for POST /v2/materialize."""

    success: bool
    files: Dict[str, str] = {}
    verification: Dict[str, Any] = {}
    trust: TrustResponse = Field(default_factory=TrustResponse)
    domain: str = "software"
    error: Optional[str] = None


class V2ValidateResponse(BaseModel):
    """Response body for POST /v2/validate — trust-only, no materialization."""

    success: bool
    trust: TrustResponse = Field(default_factory=TrustResponse)
    blueprint: Dict[str, Any] = {}
    domain: str = "software"
    error: Optional[str] = None


class DomainInfoResponse(BaseModel):
    """Response body for GET /v2/domains/{name}."""

    name: str
    version: str
    output_format: str = "python"
    file_extension: str = ".py"
    vocabulary_types: List[str] = []
    relationship_types: List[str] = []
    actionability_checks: List[str] = []


class DomainListResponse(BaseModel):
    """Response body for GET /v2/domains."""

    domains: List[DomainInfoResponse] = []


class V2HealthResponse(BaseModel):
    """Response body for GET /v2/health."""

    status: str = "ok"
    version: str = "2.0.0"
    domains_available: List[str] = []
    corpus_size: int = 0
    uptime_seconds: float = 0.0
    worker_queue_depth: int = 0
    disk_free_mb: int = 0
    disk_total_mb: int = 0


class V2MetricsResponse(BaseModel):
    """Response body for GET /v2/metrics."""

    per_domain: Dict[str, Dict[str, Any]] = {}
    total_compilations: int = 0
    total_cost_usd: float = 0.0


# =============================================================================
# TOOL SHARING MODELS
# =============================================================================

class ToolDigestResponse(BaseModel):
    """Lightweight tool summary (no code)."""

    package_id: str
    name: str
    domain: str
    fingerprint: str
    trust_score: float
    verification_badge: str
    component_count: int
    relationship_count: int
    source_instance_id: str
    created_at: str


class ToolSearchResponse(BaseModel):
    """Response for tool list/search endpoints."""

    tools: List[ToolDigestResponse] = []
    total: int = 0


class ToolExportRequest(BaseModel):
    """Request body for POST /v2/tools/export."""

    compilation_id: str = Field(
        ...,
        min_length=1,
        description="Corpus compilation ID to export",
    )
    name: Optional[str] = Field(
        None,
        description="Human-readable tool name (defaults to blueprint core_need)",
    )
    version: str = Field(
        "1.0.0",
        description="Semantic version",
    )


class ToolExportResponse(BaseModel):
    """Response body for POST /v2/tools/export."""

    success: bool
    package_id: str = ""
    name: str = ""
    trust_score: float = 0.0
    verification_badge: str = "unverified"
    error: Optional[str] = None


class ToolImportRequest(BaseModel):
    """Request body for POST /v2/tools/import."""

    package: Dict[str, Any] = Field(
        ...,
        description="Serialized ToolPackage JSON",
    )
    min_trust_score: float = Field(
        60.0,
        description="Minimum trust score for import",
    )
    require_verified: bool = Field(
        False,
        description="Require 'verified' badge",
    )


class ToolImportResponse(BaseModel):
    """Response body for POST /v2/tools/import."""

    success: bool
    allowed: bool = False
    rejection_reason: str = ""
    provenance_valid: bool = False
    trust_sufficient: bool = False
    code_safe: bool = False
    warnings: List[str] = []
    checks_performed: List[str] = []
    error: Optional[str] = None


class ToolPackageResponse(BaseModel):
    """Full tool package response."""

    package_id: str
    name: str
    version: str
    domain: str
    trust_score: float
    verification_badge: str
    fingerprint: str
    blueprint: Dict[str, Any] = {}
    generated_code: Dict[str, str] = {}
    fidelity_scores: Dict[str, int] = {}
    provenance_chain: List[Dict[str, Any]] = []
    source_instance_id: str = ""
    created_at: str = ""


class TrustGraphDigestResponse(BaseModel):
    """Response for GET /v2/instance/digest."""

    instance_id: str
    instance_name: str
    tool_count: int = 0
    verified_tool_count: int = 0
    domain_counts: Dict[str, int] = {}
    total_compilations: int = 0
    avg_trust_score: float = 0.0
    last_updated: str = ""


class PeerRegisterRequest(BaseModel):
    """Request body for POST /v2/instance/peers."""

    instance_id: str = Field(..., min_length=4)
    name: str = Field(..., min_length=1)
    api_endpoint: str = Field(..., min_length=1)


class PeerResponse(BaseModel):
    """Peer instance info."""

    instance_id: str
    name: str
    created_at: str = ""
    api_endpoint: str = ""


class PeerStatusResponse(BaseModel):
    """Peer liveness status."""

    instance_id: str
    name: str
    api_endpoint: str = ""
    reachable: bool = False
    latency_ms: float = 0.0
    last_checked: str = ""
    error: str = ""


class PeerEventModel(BaseModel):
    """WebSocket peer event."""

    event_type: str
    instance_id: str
    payload: Dict[str, Any] = {}
    timestamp: str = ""


# =============================================================================
# ASYNC COMPILATION MODELS
# =============================================================================

class AsyncCompileResponse(BaseModel):
    """Response for POST /v2/compile/async — returns task ID for polling."""

    task_id: str
    status: str = "queued"
    poll_url: str = ""


class TaskStatusResponse(BaseModel):
    """Response for GET /v2/tasks/{task_id} — poll compilation status."""

    task_id: str
    status: str = "pending"  # pending | running | complete | error | cancelled
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
