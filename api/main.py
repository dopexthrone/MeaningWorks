"""
Motherlabs REST API - FastAPI application.

Phase 7.3: REST API
Phase 9.3: Code Generation API
Phase 16: Blueprint Editing API

Endpoints:
    POST /v1/compile              - Compile natural language to blueprint
    POST /v1/self-compile         - Motherlabs compiles itself
    POST /v1/codegen              - Generate Python code from blueprint
    POST /v1/compile-and-generate - Full pipeline + codegen in one call
    POST /v1/blueprint/edit       - Edit a stored blueprint (Phase 16)
    POST /v1/blueprint/{id}/recompile-stage - Re-run single stage (Phase 16)
    POST /v1/self-compile-loop    - Self-compile loop with convergence tracking
    POST /v1/agent                - Full agent pipeline: enrich → compile → emit → write
    GET  /v1/corpus               - List compilations (paginated)
    GET  /v1/corpus/{id}          - Get specific compilation
    GET  /v1/health               - Health check (enhanced Phase 19)
    GET  /v1/metrics              - Aggregate metrics (Phase 19)
"""

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import (
    CompileRequest,
    CompileResponse,
    StageResultResponse,
    CompilationRecordResponse,
    CorpusListResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    CodegenRequest,
    CodegenResponse,
    CompileAndGenerateRequest,
    CompileAndGenerateResponse,
    EditBlueprintRequest,
    EditBlueprintResponse,
    RecompileStageRequest,
    RecompileStageResponse,
    MaterializeRequest,
    MaterializeResponse,
    EstimateCostRequest,
    EstimateCostResponse,
    AgentRequest,
    AgentResponse,
    SelfCompileLoopRequest,
    SelfCompileLoopResponse,
    CompileTreeRequest,
    CompileTreeResponse,
)
from core.engine import MotherlabsEngine, CompileResult
from core.exceptions import MotherlabsError
from codegen.generator import BlueprintCodeGenerator
from persistence.corpus import Corpus

logger = logging.getLogger("motherlabs.api")

# Fail fast if no LLM API key is configured
if os.environ.get("MOTHERLABS_ENV") == "production":
    _has_llm_key = any(os.environ.get(k) for k in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY", "GOOGLE_API_KEY",
    ))
    if not _has_llm_key:
        raise SystemExit(
            "FATAL: No LLM API key configured. "
            "Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GOOGLE_API_KEY"
        )

# Initialize Sentry before anything else
from motherlabs_platform.sentry import init_sentry  # noqa: E402
init_sentry()


# =============================================================================
# ENGINE FACTORY
# =============================================================================

# Shared engine instance (lazy-initialized)
_engine: Optional[MotherlabsEngine] = None


def get_engine() -> MotherlabsEngine:
    """Get or create the shared engine instance."""
    global _engine
    if _engine is None:
        _engine = MotherlabsEngine()
    return _engine


def set_engine(engine: MotherlabsEngine) -> None:
    """Set custom engine (for testing)."""
    global _engine
    _engine = engine


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Motherlabs API",
    description="Semantic compilation for the rest of us",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — configurable via MOTHERLABS_CORS_ORIGINS (comma-separated)
_cors_origins = os.environ.get("MOTHERLABS_CORS_ORIGINS", "").strip()
_allowed_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# V2 API routes (multi-domain compilation)
from api.v2.routes import router as v2_router  # noqa: E402
app.include_router(v2_router)

# Register built-in domain adapters
import adapters  # noqa: E402, F401 — registers built-in domain adapters

# Auth middleware — opt-in via MOTHERLABS_REQUIRE_AUTH=1
if os.environ.get("MOTHERLABS_REQUIRE_AUTH"):
    from motherlabs_platform.auth import get_key_store, get_rate_limiter
    from motherlabs_platform.middleware import APIKeyMiddleware

    app.add_middleware(
        APIKeyMiddleware,
        require_key=True,
        key_store=get_key_store(),
        rate_limiter=get_rate_limiter(),
    )


# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.exception_handler(MotherlabsError)
async def motherlabs_error_handler(request, exc: MotherlabsError):
    """Convert MotherlabsError to user-friendly JSON response."""
    return JSONResponse(
        status_code=422,
        content=exc.to_user_dict(),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _compile_result_to_response(result: CompileResult) -> CompileResponse:
    """Convert engine CompileResult to API response."""
    stage_results = [
        StageResultResponse(
            stage=sr.stage,
            success=sr.success,
            errors=sr.errors,
            warnings=sr.warnings,
        )
        for sr in result.stage_results
    ]

    return CompileResponse(
        success=result.success,
        blueprint=result.blueprint,
        insights=result.insights,
        verification=result.verification,
        stage_results=stage_results,
        schema_validation=result.schema_validation,
        graph_validation=result.graph_validation,
        cache_stats=result.cache_stats,
        input_quality=result.input_quality,
        dimensional_metadata=result.dimensional_metadata or {},
        interface_map=result.interface_map or {},
        corpus_suggestions=result.corpus_suggestions or {},
        context_graph=result.context_graph or {},
        error=result.error,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post(
    "/v1/compile",
    response_model=CompileResponse,
    responses={422: {"model": ErrorResponse}},
)
async def compile_description(request: CompileRequest):
    """
    Compile a natural language description into a blueprint.

    The engine runs the full 5-stage pipeline:
    1. Intent extraction
    2. Persona generation
    3. Spec dialogue (Entity <-> Process)
    4. Synthesis
    5. Verification

    With enrich=True, sparse input is automatically expanded before compilation.
    """
    engine = get_engine()

    # Enrich sparse input if requested
    description = request.description
    if request.enrich:
        from core.input_quality import InputQualityAnalyzer
        from core.input_enrichment import (
            ENRICHMENT_SYSTEM_PROMPT,
            build_enrichment_prompt,
            parse_enrichment_response,
        )
        analyzer = InputQualityAnalyzer()
        quality = analyzer.analyze(description)
        if quality.is_hollow:
            try:
                prompt = build_enrichment_prompt(description)
                response = engine.llm.complete_with_system(
                    system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                enrichment = parse_enrichment_response(response, description)
                description = enrichment.enriched_input
            except Exception:
                pass  # Graceful fallback: use original

    # Convert relationships from list-of-lists to list-of-tuples
    canonical_rels = None
    if request.canonical_relationships:
        canonical_rels = [
            tuple(r) for r in request.canonical_relationships
            if len(r) >= 3
        ]

    result = engine.compile(
        description=description,
        canonical_components=request.canonical_components,
        canonical_relationships=canonical_rels,
        min_quality_score=request.min_quality_score,
        pipeline_mode=request.pipeline_mode,
    )

    return _compile_result_to_response(result)


@app.post(
    "/v1/self-compile",
    response_model=CompileResponse,
    responses={422: {"model": ErrorResponse}},
)
async def self_compile():
    """
    Motherlabs compiles itself (dogfood test).

    Uses canonical components and axioms to verify the system
    can accurately specify its own architecture.
    """
    engine = get_engine()
    result = engine.self_compile()
    return _compile_result_to_response(result)


@app.post(
    "/v1/self-compile-loop",
    response_model=SelfCompileLoopResponse,
)
async def self_compile_loop(request: SelfCompileLoopRequest):
    """
    Run self-compile N times with convergence tracking.

    Compiles the system's own architecture repeatedly, tracks blueprint
    stability across runs, diffs against actual source code, and extracts
    self-observation patterns (Stratum 3 provenance).
    """
    from dataclasses import asdict

    engine = get_engine()
    report = engine.run_self_compile_loop(runs=request.runs)

    return SelfCompileLoopResponse(
        converged=report.convergence.is_converged,
        stability_score=1.0 - report.convergence.variance.variance_score,
        run_count=report.convergence.variance.run_count,
        variance_summary=asdict(report.convergence.variance),
        code_diffs=[asdict(d) for d in report.code_diffs],
        patterns=[asdict(p) for p in report.patterns],
        overall_health=report.overall_health,
        timestamp=report.timestamp,
    )


@app.post(
    "/v1/compile-tree",
    response_model=CompileTreeResponse,
)
async def compile_tree(request: CompileTreeRequest):
    """
    Compile a large task as a tree of subsystems.

    Decomposes the description into independent subsystems, compiles each
    child separately, runs L2 synthesis for cross-cutting patterns, and
    verifies integration. For complex multi-subsystem projects.
    """
    from dataclasses import asdict

    engine = get_engine()

    # Enrich if requested and input is hollow
    description = request.description
    if request.enrich:
        from core.input_quality import InputQualityAnalyzer
        from core.input_enrichment import (
            ENRICHMENT_SYSTEM_PROMPT,
            build_enrichment_prompt,
            parse_enrichment_response,
        )
        analyzer = InputQualityAnalyzer()
        quality = analyzer.analyze(description)
        if quality.is_hollow:
            try:
                prompt = build_enrichment_prompt(description)
                response = engine.llm.complete_with_system(
                    system_prompt=ENRICHMENT_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                enrichment = parse_enrichment_response(response, description)
                description = enrichment.enriched_input
            except Exception:
                pass

    # Convert relationships
    canonical_rels = None
    if request.canonical_relationships:
        canonical_rels = [
            tuple(r) for r in request.canonical_relationships
            if len(r) >= 3
        ]

    try:
        tree_result = engine.compile_tree(
            description=description,
            canonical_components=request.canonical_components,
            canonical_relationships=canonical_rels,
            max_children=request.max_children,
        )

        child_blueprints = []
        for child in tree_result.child_results:
            child_blueprints.append({
                "subsystem": child.subsystem_name,
                "blueprint": child.blueprint,
                "success": child.success,
                "component_count": child.component_count,
                "relationship_count": child.relationship_count,
            })

        return CompileTreeResponse(
            success=True,
            root_blueprint=tree_result.root_blueprint,
            child_blueprints=child_blueprints,
            l2_synthesis=asdict(tree_result.l2_synthesis),
            integration_report=asdict(tree_result.integration_report),
            tree_health=tree_result.tree_health,
            total_components=tree_result.total_components,
            timestamp=tree_result.timestamp,
        )
    except Exception as e:
        return CompileTreeResponse(
            success=False,
            error=str(e),
        )


@app.get(
    "/v1/corpus",
    response_model=CorpusListResponse,
)
async def list_corpus(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Results per page"),
):
    """
    List compilations in the corpus.

    Supports pagination and optional domain filtering.
    """
    engine = get_engine()

    if domain:
        all_records = engine.list_compilations(domain=domain)
    else:
        all_records = engine.list_compilations()

    total = len(all_records)
    start = (page - 1) * per_page
    end = start + per_page
    page_records = all_records[start:end]

    records = [
        CompilationRecordResponse(
            id=r.id,
            input_text=r.input_text,
            domain=r.domain,
            timestamp=r.timestamp,
            components_count=r.components_count,
            insights_count=r.insights_count,
            success=r.success,
            provider=r.provider,
            model=r.model,
        )
        for r in page_records
    ]

    return CorpusListResponse(
        records=records,
        total=total,
        page=page,
        per_page=per_page,
    )


@app.get(
    "/v1/corpus/{compilation_id}",
    responses={404: {"model": ErrorResponse}},
)
async def get_compilation(compilation_id: str):
    """
    Get a specific compilation by ID.

    Returns the full compilation record including blueprint and context graph.
    """
    engine = get_engine()
    record = engine.corpus.get(compilation_id)

    if not record:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Compilation {compilation_id} not found", "suggestion": "Check the ID and try again."},
        )

    blueprint = engine.corpus.load_blueprint(compilation_id)
    context_graph = engine.corpus.load_context_graph(compilation_id)

    return {
        "record": CompilationRecordResponse(
            id=record.id,
            input_text=record.input_text,
            domain=record.domain,
            timestamp=record.timestamp,
            components_count=record.components_count,
            insights_count=record.insights_count,
            success=record.success,
            provider=record.provider,
            model=record.model,
        ).model_dump(),
        "blueprint": blueprint,
        "context_graph": context_graph,
    }


# =============================================================================
# CODEGEN ENDPOINTS - Phase 9.3
# =============================================================================


def _generate_code_from_blueprint(blueprint: dict, include_tests: bool) -> CodegenResponse:
    """Generate Python code from a blueprint dict."""
    try:
        gen = BlueprintCodeGenerator(blueprint)
        files = {"blueprint_code.py": gen.generate()}

        if include_tests:
            files["test_blueprint.py"] = gen.generate_tests()

        component_count = len(blueprint.get("components", []))

        return CodegenResponse(
            success=True,
            files=files,
            component_count=component_count,
        )
    except Exception as e:
        logger.error(f"Codegen failed: {e}")
        return CodegenResponse(
            success=False,
            error=str(e),
        )


@app.post(
    "/v1/codegen",
    response_model=CodegenResponse,
    responses={422: {"model": ErrorResponse}},
)
async def generate_code(request: CodegenRequest):
    """
    Generate Python code from a blueprint.

    Phase 9.3: Takes blueprint JSON and returns generated Python files
    with constraint-aware code, validation stubs, and example tests.
    """
    return _generate_code_from_blueprint(request.blueprint, request.include_tests)


@app.post(
    "/v1/compile-and-generate",
    response_model=CompileAndGenerateResponse,
    responses={422: {"model": ErrorResponse}},
)
async def compile_and_generate(request: CompileAndGenerateRequest):
    """
    Full pipeline: compile description to blueprint, then generate code.

    Phase 9.3: Combines /v1/compile and /v1/codegen in one call.
    """
    engine = get_engine()

    # Convert relationships
    canonical_rels = None
    if request.canonical_relationships:
        canonical_rels = [
            tuple(r) for r in request.canonical_relationships
            if len(r) >= 3
        ]

    result = engine.compile(
        description=request.description,
        canonical_components=request.canonical_components,
        canonical_relationships=canonical_rels,
        min_quality_score=request.min_quality_score,
    )

    compilation = _compile_result_to_response(result)

    if request.codegen_mode == "llm":
        # Phase D: LLM-based code emission
        from core.agent_emission import EmissionConfig
        config = EmissionConfig()
        emission_result = engine.emit_code(result.blueprint, config=config)
        codegen = CodegenResponse(
            success=emission_result.success_count > 0,
            files=dict(emission_result.generated_code),
            component_count=emission_result.total_nodes,
            error=None if emission_result.success_count > 0 else "All emissions failed",
        )
    else:
        # Default: template-based code generation
        codegen = _generate_code_from_blueprint(result.blueprint, request.include_tests)

    return CompileAndGenerateResponse(
        compilation=compilation,
        codegen=codegen,
    )


# =============================================================================
# MATERIALIZATION (Phase D)
# =============================================================================


@app.post(
    "/v1/materialize",
    response_model=MaterializeResponse,
    responses={422: {"model": ErrorResponse}},
)
async def materialize_blueprint(request: MaterializeRequest):
    """
    Materialize a blueprint into executable code via LLM agents.

    Phase D: Agent Emission. Takes a blueprint and dispatches LLM calls
    per component in dependency order. Returns generated code files
    with interface verification.
    """
    engine = get_engine()

    try:
        from core.agent_emission import EmissionConfig

        # Deserialize interface_map if provided
        interface_map = None
        if request.interface_map:
            from core.interface_schema import deserialize_interface_map
            interface_map = deserialize_interface_map(request.interface_map)

        # Deserialize dim_meta if provided
        dim_meta = None
        if request.dim_meta:
            from core.dimensional import deserialize_dimensional_metadata
            dim_meta = deserialize_dimensional_metadata(request.dim_meta)

        # Deserialize l2_synthesis if provided
        l2_synthesis = None
        if request.l2_synthesis:
            from core.compilation_tree import deserialize_tree_result
            # L2 synthesis is a sub-object; reconstruct directly
            l2_data = request.l2_synthesis
            from core.compilation_tree import L2Synthesis, CrossCuttingComponent, InterfaceGap
            cross_cutting = tuple(
                CrossCuttingComponent(
                    normalized_name=cc["normalized_name"],
                    variants=tuple(cc["variants"]),
                    frequency=cc["frequency"],
                    child_sources=tuple(cc["child_sources"]),
                    component_type=cc["component_type"],
                )
                for cc in l2_data.get("cross_cutting_components", [])
            )
            interface_gaps = tuple(
                InterfaceGap(
                    component_a=g["component_a"],
                    component_b=g["component_b"],
                    child_a=g["child_a"],
                    child_b=g["child_b"],
                    gap_type=g["gap_type"],
                    description=g["description"],
                )
                for g in l2_data.get("interface_gaps", [])
            )
            l2_synthesis = L2Synthesis(
                shared_vocabulary=tuple(tuple(sv) for sv in l2_data.get("shared_vocabulary", [])),
                cross_cutting_components=cross_cutting,
                relationship_patterns=tuple(tuple(rp) for rp in l2_data.get("relationship_patterns", [])),
                interface_gaps=interface_gaps,
                integration_constraints=tuple(l2_data.get("integration_constraints", [])),
                pattern_count=l2_data.get("pattern_count", 0),
                synthesis_confidence=l2_data.get("synthesis_confidence", 0.0),
            )

        config = EmissionConfig(max_tokens=request.max_tokens)
        result = engine.emit_code(
            blueprint=request.blueprint,
            interface_map=interface_map,
            dim_meta=dim_meta,
            l2_synthesis=l2_synthesis,
            config=config,
        )

        return MaterializeResponse(
            success=result.success_count > 0,
            files=dict(result.generated_code),
            verification=result.verification_report,
            total_nodes=result.total_nodes,
            success_count=result.success_count,
            failure_count=result.failure_count,
            pass_rate=result.pass_rate,
        )
    except Exception as e:
        logger.error(f"Materialization failed: {e}")
        return MaterializeResponse(
            success=False,
            error=str(e),
        )


# =============================================================================
# BLUEPRINT EDITING (Phase 16)
# =============================================================================


@app.post(
    "/v1/blueprint/edit",
    response_model=EditBlueprintResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def edit_blueprint(
    request: EditBlueprintRequest,
    engine: MotherlabsEngine = Depends(get_engine),
):
    """
    Apply edits to a stored blueprint.

    Phase 16: Human-in-the-Loop Iteration. Rename, remove, merge components,
    add constraints, flag hollow components. Saves as a new compilation with
    lineage tracking.
    """
    try:
        edits = [op.model_dump(exclude_none=True) for op in request.edits]
        result = engine.edit_blueprint(request.compilation_id, edits)
        return EditBlueprintResponse(
            success=result.success,
            blueprint=result.blueprint,
            operations_applied=len(request.edits),
            warnings=[],
            schema_validation=result.schema_validation,
            graph_validation=result.graph_validation,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post(
    "/v1/blueprint/{compilation_id}/recompile-stage",
    response_model=RecompileStageResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def recompile_stage(
    compilation_id: str,
    request: RecompileStageRequest,
    engine: MotherlabsEngine = Depends(get_engine),
):
    """
    Re-run a single pipeline stage on a stored compilation.

    Phase 16: Selective stage re-execution. Optionally apply edits before
    re-running the specified stage.
    """
    try:
        edits = None
        if request.edits:
            edits = [op.model_dump(exclude_none=True) for op in request.edits]
        result = engine.recompile_stage(compilation_id, request.stage, edits)
        return RecompileStageResponse(
            success=result.success,
            blueprint=result.blueprint,
            schema_validation=result.schema_validation,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# HEALTH
# =============================================================================


@app.get(
    "/v1/health",
    response_model=HealthResponse,
)
async def health_check():
    """Health check endpoint (enhanced Phase 19)."""
    engine = get_engine()
    snapshot = engine.get_health_snapshot()

    return HealthResponse(
        status=snapshot.get("status", "ok"),
        version="0.2.0",
        corpus_size=snapshot.get("corpus_size", 0),
        uptime_seconds=snapshot.get("uptime_seconds", 0.0),
        compilations_recent=snapshot.get("compilations_recent", 0),
        recent_success_rate=snapshot.get("recent_success_rate", 0.0),
        recent_avg_duration=snapshot.get("recent_avg_duration", 0.0),
        cache_hit_rate=snapshot.get("cache_hit_rate", 0.0),
        issues=snapshot.get("issues", []),
        recent_total_cost_usd=snapshot.get("recent_total_cost_usd", 0.0),
    )


@app.get(
    "/v1/metrics",
    response_model=MetricsResponse,
)
async def get_metrics(
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """Aggregate metrics endpoint (Phase 19)."""
    engine = get_engine()
    aggregate = engine.get_metrics()
    corpus_stats = engine.get_corpus_stats()
    provider_stats = (
        engine.corpus.get_provider_stats(provider)
        if provider
        else engine.corpus.get_provider_stats()
    )
    cache_stats = engine._cache.stats()

    cost_section = {
        "total_input_tokens": aggregate.get("total_input_tokens", 0),
        "total_output_tokens": aggregate.get("total_output_tokens", 0),
        "total_cost_usd": aggregate.get("total_cost_usd", 0.0),
        "avg_cost_usd": aggregate.get("avg_cost_usd", 0.0),
        "session_cost_usd": engine._session_cost_usd,
    }

    return MetricsResponse(
        aggregate=aggregate,
        corpus=corpus_stats,
        providers=provider_stats,
        cache=cache_stats,
        cost=cost_section,
    )


# =============================================================================
# COST ESTIMATION (Phase 21)
# =============================================================================


@app.post(
    "/v1/estimate-cost",
    response_model=EstimateCostResponse,
)
async def estimate_cost(request: EstimateCostRequest):
    """
    Estimate the cost of compiling a description.

    Phase 21: Heuristic estimation based on description length and model pricing.
    Assumes ~4 chars per input token and ~6 LLM calls with ~2000 output tokens each.
    """
    from core.telemetry import TokenUsage, estimate_cost as _estimate_cost, PRICING_TABLE

    # Determine model
    model = request.model
    if not model:
        provider = (request.provider or "claude").lower()
        defaults = {
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-5.1",
            "grok": "grok-4-1-fast-reasoning",
            "gemini": "gemini-2.0-flash",
        }
        model = defaults.get(provider, "claude-sonnet-4-20250514")

    # Heuristic: ~4 chars per token for input, ~6 LLM calls, ~2000 tokens output each
    input_chars = len(request.description)
    est_input_tokens_per_call = max(input_chars // 4, 100)  # minimum 100
    num_calls = 6
    est_output_tokens_per_call = 2000

    total_input = est_input_tokens_per_call * num_calls
    total_output = est_output_tokens_per_call * num_calls
    total_tokens = total_input + total_output

    usage = TokenUsage(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_tokens,
        provider=request.provider or "claude",
        model=model,
    )
    cost = _estimate_cost(usage)

    return EstimateCostResponse(
        estimated_tokens=total_tokens,
        estimated_cost_usd=cost.total_cost,
        model=model,
        breakdown={
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "input_cost_usd": cost.input_cost,
            "output_cost_usd": cost.output_cost,
            "num_llm_calls": num_calls,
        },
    )


# =============================================================================
# AGENT (Phase: Ship the Compiler as a Product)
# =============================================================================


@app.post(
    "/v1/agent",
    response_model=AgentResponse,
)
async def run_agent(request: AgentRequest):
    """
    Run the full agent pipeline: enrich → compile → emit → (optionally) write.

    This is the primary product endpoint. Takes a natural language description
    and returns a complete blueprint + generated code, optionally written to disk.
    """
    from core.agent_orchestrator import (
        AgentOrchestrator,
        AgentConfig,
        serialize_agent_result,
    )

    engine = get_engine()
    agent_config = AgentConfig(
        codegen_mode=request.codegen_mode,
        enrich_input=request.enrich,
        write_project=request.write_project,
        output_dir=request.output_dir,
        build=request.build and request.write_project,
    )
    orch = AgentOrchestrator(engine, agent_config)
    result = orch.run(request.description)

    serialized = serialize_agent_result(result)

    # project_files: filename → content (from manifest.file_contents)
    project_files = {}
    if result.project_manifest and result.project_manifest.file_contents:
        project_files = dict(result.project_manifest.file_contents)

    # Extract semantic data from compile_result
    verification = {}
    dimensional_metadata = {}
    interface_map = {}
    stage_results = []
    if result.compile_result:
        cr = result.compile_result
        verification = getattr(cr, 'verification', {}) or {}
        dimensional_metadata = getattr(cr, 'dimensional_metadata', {}) or {}
        interface_map = getattr(cr, 'interface_map', {}) or {}
        stage_results = [
            {"stage": sr.stage, "success": sr.success,
             "errors": list(sr.errors), "warnings": list(sr.warnings)}
            for sr in (getattr(cr, 'stage_results', []) or [])
        ]

    return AgentResponse(
        success=result.success,
        project_files=project_files,
        blueprint=result.blueprint,
        generated_code=dict(result.generated_code),
        enrichment=serialized.get("enrichment"),
        quality_score=result.quality_score,
        timing=dict(result.timing),
        verification=verification,
        dimensional_metadata=dimensional_metadata,
        interface_map=interface_map,
        stage_results=stage_results,
        build_result=serialized.get("build_result"),
        error=result.error,
    )
