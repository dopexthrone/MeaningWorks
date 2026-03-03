"""
Motherlabs API - Request/Response models.

Phase 7.3: REST API
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS
# =============================================================================


class CompileRequest(BaseModel):
    """Request body for POST /v1/compile."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of what to build",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override (claude, openai, grok, gemini)",
    )
    canonical_components: Optional[List[str]] = Field(
        None,
        description="Components that MUST appear in the output blueprint",
    )
    canonical_relationships: Optional[List[List[str]]] = Field(
        None,
        description="Relationships that MUST appear: [[from, to, type], ...]",
    )
    cache_policy: Optional[str] = Field(
        None,
        description="Cache policy override (none, intent, full)",
    )
    min_quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum input quality score (0.0-1.0). Default uses system threshold.",
    )
    pipeline_mode: Optional[str] = Field(
        None,
        description="Pipeline mode: 'legacy' (single dialogue) or 'staged' (5-stage pipeline)",
    )
    enrich: bool = Field(
        False,
        description="Enrich sparse input before compilation (auto-detects hollow input)",
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class StageResultResponse(BaseModel):
    """A single stage result in the compilation."""

    stage: str
    success: bool
    errors: List[str] = []
    warnings: List[str] = []


class CompileResponse(BaseModel):
    """Response body for POST /v1/compile."""

    success: bool
    blueprint: Dict[str, Any] = {}
    insights: List[str] = []
    verification: Dict[str, Any] = {}
    stage_results: List[StageResultResponse] = []
    schema_validation: Dict[str, Any] = {}
    graph_validation: Dict[str, Any] = {}
    cache_stats: Dict[str, Any] = {}
    input_quality: Dict[str, Any] = {}
    dimensional_metadata: Dict[str, Any] = {}
    interface_map: Dict[str, Any] = {}
    corpus_suggestions: Dict[str, Any] = {}
    context_graph: Dict[str, Any] = {}
    error: Optional[str] = None


class CompilationRecordResponse(BaseModel):
    """A corpus compilation record."""

    id: str
    input_text: str
    domain: str
    timestamp: str
    components_count: int
    insights_count: int
    success: bool
    provider: str = "unknown"
    model: str = "unknown"


class CorpusListResponse(BaseModel):
    """Response body for GET /v1/corpus."""

    records: List[CompilationRecordResponse]
    total: int
    page: int = 1
    per_page: int = 20


class CorpusStatsResponse(BaseModel):
    """Response body for GET /v1/corpus/stats."""

    total_compilations: int
    domains: Dict[str, int] = {}
    success_rate: float = 0.0
    total_components: int = 0
    total_insights: int = 0


class HealthResponse(BaseModel):
    """Response body for GET /v1/health (enhanced Phase 19)."""

    status: str = "ok"
    version: str = "0.2.0"
    corpus_size: int = 0
    # Phase 19 additive fields
    uptime_seconds: float = 0.0
    compilations_recent: int = 0
    recent_success_rate: float = 0.0
    recent_avg_duration: float = 0.0
    cache_hit_rate: float = 0.0
    issues: List[str] = []
    # Phase 21: Cost tracking
    recent_total_cost_usd: float = 0.0


class MetricsResponse(BaseModel):
    """Response body for GET /v1/metrics (Phase 19)."""

    aggregate: Dict[str, Any] = {}
    corpus: Dict[str, Any] = {}
    providers: Dict[str, Any] = {}
    cache: Dict[str, Any] = {}
    # Phase 21: Cost section
    cost: Dict[str, Any] = {}


class CodegenRequest(BaseModel):
    """Request body for POST /v1/codegen."""

    blueprint: Dict[str, Any] = Field(
        ...,
        description="Blueprint JSON to generate code from",
    )
    include_tests: bool = Field(
        True,
        description="Include generated test file",
    )


class CodegenResponse(BaseModel):
    """Response body for POST /v1/codegen."""

    success: bool
    files: Dict[str, str] = {}
    component_count: int = 0
    error: Optional[str] = None


class CompileAndGenerateRequest(BaseModel):
    """Request body for POST /v1/compile-and-generate."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of what to build",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider override",
    )
    canonical_components: Optional[List[str]] = Field(
        None,
        description="Components that MUST appear in output",
    )
    canonical_relationships: Optional[List[List[str]]] = Field(
        None,
        description="Relationships that MUST appear: [[from, to, type], ...]",
    )
    include_tests: bool = Field(
        True,
        description="Include generated test file",
    )
    min_quality_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum input quality score (0.0-1.0)",
    )
    codegen_mode: str = Field(
        "template",
        description="Code generation mode: 'template' (fast, deterministic) or 'llm' (richer, agent-based)",
    )


class CompileAndGenerateResponse(BaseModel):
    """Response body for POST /v1/compile-and-generate."""

    compilation: CompileResponse
    codegen: CodegenResponse


class EditOperation(BaseModel):
    """A single edit operation for Phase 16."""

    operation: str = Field(
        ...,
        description="Edit type: rename, remove, merge, add_constraint, flag_hollow, add_component",
    )
    # rename
    old_name: Optional[str] = Field(None, description="Component to rename (for rename)")
    new_name: Optional[str] = Field(None, description="New name (for rename)")
    # remove / flag_hollow / add_component
    name: Optional[str] = Field(None, description="Component name (for remove/flag_hollow/add_component)")
    # merge
    names: Optional[List[str]] = Field(None, description="Components to merge")
    merged_name: Optional[str] = Field(None, description="Name for merged component")
    merged_type: Optional[str] = Field(None, description="Type for merged component")
    # add_constraint
    description: Optional[str] = Field(None, description="Constraint description (for add_constraint)")
    applies_to: Optional[List[str]] = Field(None, description="Components constraint applies to")
    # add_component
    type: Optional[str] = Field(None, description="Component type (for add_component)")


class EditBlueprintRequest(BaseModel):
    """Request body for POST /v1/blueprint/edit."""

    compilation_id: str = Field(
        ...,
        description="ID of the compilation to edit",
    )
    edits: List[EditOperation] = Field(
        ...,
        min_length=1,
        description="List of edit operations to apply",
    )


class EditBlueprintResponse(BaseModel):
    """Response body for POST /v1/blueprint/edit."""

    success: bool
    blueprint: Dict[str, Any] = {}
    operations_applied: int = 0
    warnings: List[str] = []
    schema_validation: Dict[str, Any] = {}
    graph_validation: Dict[str, Any] = {}
    error: Optional[str] = None


class RecompileStageRequest(BaseModel):
    """Request body for POST /v1/blueprint/{id}/recompile-stage."""

    stage: str = Field(
        ...,
        description="Pipeline stage to re-run: EXPAND, DECOMPOSE, GROUND, CONSTRAIN, or ARCHITECT",
    )
    edits: Optional[List[EditOperation]] = Field(
        None,
        description="Optional edits to apply before re-running the stage",
    )


class RecompileStageResponse(BaseModel):
    """Response body for POST /v1/blueprint/{id}/recompile-stage."""

    success: bool
    blueprint: Dict[str, Any] = {}
    schema_validation: Dict[str, Any] = {}
    error: Optional[str] = None


class MaterializeRequest(BaseModel):
    """Request body for POST /v1/materialize."""

    blueprint: Dict[str, Any] = Field(
        ...,
        description="Blueprint JSON to materialize into code via LLM agents",
    )
    interface_map: Optional[Dict[str, Any]] = Field(
        None,
        description="Interface map JSON (extracted from blueprint if omitted)",
    )
    dim_meta: Optional[Dict[str, Any]] = Field(
        None,
        description="Dimensional metadata JSON (optional)",
    )
    l2_synthesis: Optional[Dict[str, Any]] = Field(
        None,
        description="L2 synthesis JSON from compile_tree (optional)",
    )
    max_tokens: int = Field(
        4096,
        ge=256,
        le=16384,
        description="Max tokens per LLM call",
    )


class MaterializeResponse(BaseModel):
    """Response body for POST /v1/materialize."""

    success: bool
    files: Dict[str, str] = {}
    verification: Dict[str, Any] = {}
    total_nodes: int = 0
    success_count: int = 0
    failure_count: int = 0
    pass_rate: float = 0.0
    error: Optional[str] = None


class EstimateCostRequest(BaseModel):
    """Request body for POST /v1/estimate-cost."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description to estimate cost for",
    )
    provider: Optional[str] = Field(
        None,
        description="LLM provider (claude, openai, grok, gemini)",
    )
    model: Optional[str] = Field(
        None,
        description="Model name override",
    )


class EstimateCostResponse(BaseModel):
    """Response body for POST /v1/estimate-cost."""

    estimated_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    breakdown: Dict[str, Any] = {}


class AgentRequest(BaseModel):
    """Request body for POST /v1/agent."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of what to build",
    )
    codegen_mode: str = Field(
        "llm",
        description="Code generation mode: 'llm' (agent-based) or 'template' (deterministic)",
    )
    enrich: bool = Field(
        True,
        description="Enrich sparse input before compilation",
    )
    write_project: bool = Field(
        False,
        description="Write project files to disk (default: return code only)",
    )
    output_dir: str = Field(
        "./output",
        description="Output directory for project files (if write_project=True)",
    )
    build: bool = Field(
        False,
        description="Run build loop after writing: install deps, test, fix errors (Phase 27)",
    )


class AgentResponse(BaseModel):
    """Response body for POST /v1/agent."""

    success: bool
    project_files: Dict[str, str] = {}      # filename → content (from project manifest)
    blueprint: Dict[str, Any] = {}
    generated_code: Dict[str, str] = {}
    enrichment: Optional[Dict[str, Any]] = None
    quality_score: float = 0.0
    timing: Dict[str, float] = {}
    # Semantic data from compilation
    verification: Dict[str, Any] = {}
    dimensional_metadata: Dict[str, Any] = {}
    interface_map: Dict[str, Any] = {}
    stage_results: List[Dict[str, Any]] = []
    # Phase 27: Build loop result
    build_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CompileTreeRequest(BaseModel):
    """Request body for POST /v1/compile-tree."""

    description: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Natural language description of what to build",
    )
    canonical_components: Optional[List[str]] = Field(
        None,
        description="Components that MUST appear in output",
    )
    canonical_relationships: Optional[List[List[str]]] = Field(
        None,
        description="Relationships that MUST appear: [[from, to, type], ...]",
    )
    max_children: int = Field(
        8,
        ge=1,
        le=20,
        description="Maximum subsystem decompositions",
    )
    enrich: bool = Field(
        False,
        description="Enrich sparse input before compilation",
    )


class CompileTreeResponse(BaseModel):
    """Response body for POST /v1/compile-tree."""

    success: bool
    root_blueprint: Dict[str, Any] = {}
    child_blueprints: List[Dict[str, Any]] = []
    l2_synthesis: Dict[str, Any] = {}
    integration_report: Dict[str, Any] = {}
    tree_health: float = 0.0
    total_components: int = 0
    timestamp: str = ""
    error: Optional[str] = None


class SelfCompileLoopRequest(BaseModel):
    """Request body for POST /v1/self-compile-loop."""

    runs: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of self-compile iterations",
    )


class SelfCompileLoopResponse(BaseModel):
    """Response body for POST /v1/self-compile-loop."""

    converged: bool = False
    stability_score: float = 0.0
    run_count: int = 0
    variance_summary: Dict[str, Any] = {}
    code_diffs: List[Dict[str, Any]] = []
    patterns: List[Dict[str, Any]] = []
    overall_health: float = 0.0
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    suggestion: str = ""
    error_type: str = "Error"
    error_code: Optional[str] = None
    root_cause: Optional[str] = None
    fix_examples: List[str] = []
