"""
Motherlabs V2 API Routes — multi-domain compilation endpoints.

Phase D: V2 API + Platform Layer

V2 Endpoints:
- POST /v2/compile              — Universal domain-parameterized compilation
- POST /v2/compile-tree         — Multi-subsystem compilation
- POST /v2/materialize          — Materialize blueprint per domain
- POST /v2/validate             — Trust-only mode (no materialization)
- GET  /v2/trust/{compile_id}   — Trust indicators for a compilation
- GET  /v2/domains              — List available domain adapters
- GET  /v2/domains/{name}       — Domain adapter metadata
- GET  /v2/health               — Platform health
- GET  /v2/metrics              — Per-domain metrics
"""

import time
import logging
from typing import Dict, Any

_start_time = time.time()

from fastapi import APIRouter, HTTPException, Request

from api.v2.models import (
    V2CompileRequest,
    V2CompileResponse,
    V2CompileTreeRequest,
    V2CompileTreeResponse,
    V2MaterializeRequest,
    V2MaterializeResponse,
    V2RecompileRequest,
    V2RecompileResponse,
    V2ValidateRequest,
    V2ValidateResponse,
    TrustResponse,
    UsageResponse,
    StageResultResponse,
    DomainInfoResponse,
    DomainListResponse,
    V2HealthResponse,
    V2MetricsResponse,
    ToolDigestResponse,
    ToolSearchResponse,
    ToolExportRequest,
    ToolExportResponse,
    ToolImportRequest,
    ToolImportResponse,
    ToolPackageResponse,
    TrustGraphDigestResponse,
    PeerRegisterRequest,
    PeerResponse,
    PeerStatusResponse,
    PeerEventModel,
    AsyncCompileResponse,
    TaskStatusResponse,
    TaskDecisionRequest,
    TaskDecisionResponse,
    CorpusBenchmarkResponse,
)
from core.adapter_registry import get_adapter, list_adapters
from core.blueprint_protocol import (
    BlueprintNode,
    CostReport,
    DepthReport,
    GovernanceReport,
    build_blueprint_semantic_gates,
    build_blueprint_semantic_nodes,
    build_semantic_gate_escalations,
    make_node_ref,
)
from core.exceptions import ConfigurationError
from core.trust import compute_trust_indicators, serialize_trust_indicators
from motherlabs_platform.metering import MeteringTracker

logger = logging.getLogger("motherlabs.api.v2")

router = APIRouter(prefix="/v2", tags=["v2"])

# Module-level metering tracker
_metering = MeteringTracker()

_MAX_CONTINUATION_DEPTH = 6
_MAX_REPEATED_PAUSE_SIGNATURES = 3


# =============================================================================
# HELPER: Build trust response from compile result
# =============================================================================

def _build_trust_response(
    blueprint: Dict[str, Any],
    verification: Dict[str, Any],
    context_graph: Dict[str, Any],
    dimensional_metadata: Dict[str, Any],
    intent_keywords: list,
) -> TrustResponse:
    """Compute trust indicators and return as TrustResponse model."""
    trust = compute_trust_indicators(
        blueprint=blueprint,
        verification=verification,
        context_graph=context_graph,
        dimensional_metadata=dimensional_metadata,
        intent_keywords=intent_keywords,
    )
    data = serialize_trust_indicators(trust)
    return TrustResponse(**data)


def _build_yaml_output(
    blueprint: Dict[str, Any],
    verification: Dict[str, Any],
    context_graph: Dict[str, Any],
) -> Dict[str, str]:
    """Best-effort flat artifact tree for blueprint inspection/export."""
    try:
        from core.compilation_output import build_flat_output
        return build_flat_output(blueprint or {}, verification or {}, context_graph or {})
    except Exception as yaml_err:
        logger.warning("YAML output generation failed: %s", yaml_err)
        return {}


def _normalize_swarm_compile_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Huey swarm results to the frontend compile result shape."""
    if not (isinstance(raw_result, dict) and "state" in raw_result and "blueprint" not in raw_result):
        return raw_result

    state = raw_result.get("state") or {}
    compile_result = state.get("compile_result") or {}

    blueprint = state.get("blueprint") or compile_result.get("blueprint") or {}
    verification = state.get("verification") or compile_result.get("verification") or {}
    context_graph = state.get("context_graph") or compile_result.get("context_graph") or {}
    dim_meta = compile_result.get("dimensional_metadata") or {}
    interface_map = compile_result.get("interface_map") or {}
    structured_insights = compile_result.get("structured_insights") or []
    difficulty = compile_result.get("difficulty") or {}
    stage_results = compile_result.get("stage_results") or []
    stage_timings = compile_result.get("stage_timings") or {}
    retry_counts = compile_result.get("retry_counts") or {}
    termination_condition = compile_result.get("termination_condition") or state.get("termination_condition") or {}
    blueprint["semantic_gates"] = build_blueprint_semantic_gates(
        blueprint,
        trust=state.get("trust") or {},
        verification=verification,
        context_graph=context_graph,
    )
    semantic_nodes = _serialize_semantic_nodes(
        compile_result.get("semantic_nodes")
        or state.get("semantic_nodes")
        or build_blueprint_semantic_nodes(
            blueprint,
            seed_text=state.get("intent") or blueprint.get("core_need") or "",
            trust=state.get("trust") or {},
            verification=verification,
            run_id=str(raw_result.get("swarm_id") or raw_result.get("task_id") or "swarm"),
        )
    )
    blocking_escalations = compile_result.get("blocking_escalations") or (
        build_semantic_gate_escalations(
            semantic_nodes,
            blueprint=blueprint,
            trust=state.get("trust") or {},
            context_graph=context_graph,
        )
        if not raw_result.get("success", True)
        else []
    )
    if not termination_condition:
        if blocking_escalations:
            termination_condition = {
                "status": "awaiting_human",
                "reason": "human_decision_required",
                "message": blocking_escalations[0].get(
                    "question",
                    "A human decision is required before compilation can continue.",
                ),
                "next_action": "Answer the blocking question to resume compilation.",
            }
        elif compile_result.get("error"):
            termination_condition = {
                "status": "halted",
                "reason": "compile_error",
                "message": str(compile_result.get("error")),
                "next_action": "Resolve the reported compiler error before retrying.",
            }
        elif raw_result.get("success", False):
            termination_condition = {
                "status": "complete",
                "reason": "quality_floor_reached",
                "message": "Compilation completed.",
                "next_action": "Inspect the blueprint, export it, or compile deeper.",
            }
    error = compile_result.get("error")
    fracture = compile_result.get("fracture") or None
    interrogation = compile_result.get("interrogation") or {}
    usage_payload = {
        "cost_usd": raw_result.get("total_cost_usd"),
    }
    governance_report = _build_governance_report(
        semantic_nodes=semantic_nodes,
        trust_payload=state.get("trust") or {},
        verification=verification,
        context_graph=context_graph,
        blueprint=blueprint,
        fracture=fracture,
        stage_results=stage_results,
        stage_timings=stage_timings,
        usage_payload=usage_payload,
        success=bool(raw_result.get("success", False)),
    )
    project_manifest = state.get("project_manifest") or {}

    return {
        "success": raw_result.get("success", False),
        "blueprint": blueprint,
        "semantic_nodes": [node.model_dump() for node in semantic_nodes],
        "termination_condition": termination_condition,
        "governance_report": governance_report.model_dump(),
        "materialized_output": state.get("generated_code") or {},
        "trust": state.get("trust"),
        "verification": verification,
        "context_graph": context_graph,
        "domain": state.get("domain", "software"),
        "project_files": project_manifest.get("file_contents"),
        "project_name": project_manifest.get("project_name"),
        "input_text": state.get("intent"),
        "duration_seconds": raw_result.get("total_duration_s"),
        "evolve_applied": state.get("evolve_applied", False),
        "benchmark": raw_result.get("benchmark"),
        "stub_report": state.get("stub_report"),
        "structured_insights": structured_insights,
        "difficulty": difficulty,
        "stage_results": stage_results,
        "stage_timings": stage_timings,
        "retry_counts": retry_counts,
        "dimensional_metadata": dim_meta,
        "interface_map": interface_map,
        "yaml_output": _build_yaml_output(blueprint, verification, context_graph),
        "error": error,
        "fracture": fracture,
        "interrogation": interrogation,
        "blocking_escalations": blocking_escalations,
    }


def _fracture_to_escalation(fracture: Dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(fracture, dict) or not fracture:
        return None

    question = str(fracture.get("collapsing_constraint") or "").strip()
    if not question:
        return None

    return {
        "postcode": "INT.SEM.APP.IF.SFT",
        "question": question,
        "options": [str(option) for option in fracture.get("competing_configs", []) or [] if str(option).strip()],
        "node_ref": None,
        "kind": "fracture",
        "stage": str(fracture.get("stage") or "interrogation"),
    }


def _append_human_decision_to_intent(
    base_intent: str,
    fracture: Dict[str, Any] | None,
    decision: Dict[str, Any],
    pause_options: list[str] | None = None,
) -> str:
    lines = [str(base_intent or "").strip()]
    lines.extend([
        "",
        "[Human Decision]",
        f"Question: {decision.get('question', '')}",
        f"Answer: {decision.get('answer', '')}",
    ])

    options = [str(option) for option in (fracture or {}).get("competing_configs", []) or [] if str(option).strip()]
    if pause_options:
        options.extend(
            str(option) for option in pause_options
            if str(option).strip()
        )
        options = list(dict.fromkeys(options))
    if options:
        lines.append(f"Available options at pause: {', '.join(options)}")

    lines.append("Continue compilation with this decision treated as locked context.")
    return "\n".join(line for line in lines if line is not None)


def _pause_signature(postcode: str, question: str, options: list[str] | None = None) -> str:
    normalized_options = [str(option).strip() for option in options or [] if str(option).strip()]
    return "|".join([
        str(postcode or "").strip(),
        str(question or "").strip(),
        " / ".join(normalized_options),
    ])


def _get_previous_task_id(raw_result: Dict[str, Any] | None) -> str | None:
    if not isinstance(raw_result, dict):
        return None
    state = raw_result.get("state") or {}
    previous_task_id = state.get("previous_task_id")
    return str(previous_task_id) if previous_task_id else None


def _build_continuation_guard(
    huey: Any,
    task_id: str,
    postcode: str,
    question: str,
    options: list[str] | None = None,
) -> Dict[str, Any] | None:
    signature = _pause_signature(postcode, question, options)
    seen_task_ids = set()
    repeated_signatures = 0
    lineage_depth = 0
    current_task_id: str | None = task_id

    while current_task_id and current_task_id not in seen_task_ids:
        seen_task_ids.add(current_task_id)
        lineage_depth += 1

        raw = huey.result(current_task_id, preserve=True)
        if not isinstance(raw, dict):
            break

        normalized = _normalize_swarm_compile_result(raw)
        escalations = list(normalized.get("blocking_escalations") or []) if isinstance(normalized, dict) else []
        fracture = _fracture_to_escalation(normalized.get("fracture")) if isinstance(normalized, dict) else None
        for escalation in escalations or ([fracture] if fracture else []):
            escalation_signature = _pause_signature(
                str(escalation.get("postcode") or ""),
                str(escalation.get("question") or ""),
                list(escalation.get("options") or []),
            )
            if escalation_signature == signature:
                repeated_signatures += 1
                break

        current_task_id = _get_previous_task_id(raw)

    if current_task_id in seen_task_ids:
        return {
            "status": "halted",
            "reason": "continuation_cycle_detected",
            "message": "This continuation chain looped back to a previous task. Motherlabs stopped to prevent an unbounded cycle.",
            "next_action": "Start a fresh compile or branch the problem into a narrower semantic scope.",
            "semantic_progress": {
                "fingerprint_changed": False,
                "verification_score_delta": 0.0,
                "components_delta": 0,
                "gates_changed": False,
            },
        }

    if lineage_depth >= _MAX_CONTINUATION_DEPTH:
        return {
            "status": "stalled",
            "reason": "continuation_depth_exceeded",
            "message": "This compile chain has resumed too many times. Motherlabs stopped to keep the problem bounded.",
            "next_action": "Narrow the seed, split the branch, or start a fresh compile from the last stable blueprint.",
            "semantic_progress": {
                "fingerprint_changed": False,
                "verification_score_delta": 0.0,
                "components_delta": 0,
                "gates_changed": False,
            },
        }

    if repeated_signatures >= _MAX_REPEATED_PAUSE_SIGNATURES:
        return {
            "status": "stalled",
            "reason": "continuation_cycle_detected",
            "message": "The same semantic pause has reappeared across continuation tasks. Motherlabs stopped instead of looping on the same unresolved issue.",
            "next_action": "Change the scope, revise the blueprint path, or answer the problem in a more concrete way before recompiling.",
            "semantic_progress": {
                "fingerprint_changed": False,
                "verification_score_delta": 0.0,
                "components_delta": 0,
                "gates_changed": False,
            },
        }

    return None


def _serialize_stage_results(stage_results: Any) -> list[StageResultResponse]:
    """Normalize engine stage results into API-safe summaries."""
    serialized = []
    for sr in stage_results or []:
        if isinstance(sr, dict):
            serialized.append(StageResultResponse(
                stage=sr.get("stage", ""),
                success=bool(sr.get("success", False)),
                errors=list(sr.get("errors", []) or []),
                warnings=list(sr.get("warnings", []) or []),
                retries=int(sr.get("retries", 0) or 0),
            ))
            continue

        serialized.append(StageResultResponse(
            stage=getattr(sr, "stage", ""),
            success=bool(getattr(sr, "success", False)),
            errors=list(getattr(sr, "errors", []) or []),
            warnings=list(getattr(sr, "warnings", []) or []),
            retries=int(getattr(sr, "retries", 0) or 0),
        ))
    return serialized


def _serialize_semantic_nodes(nodes: Any) -> list[BlueprintNode]:
    """Normalize semantic node payloads to canonical protocol models."""
    serialized = []
    for node in nodes or []:
        if isinstance(node, BlueprintNode):
            serialized.append(node)
            continue
        serialized.append(BlueprintNode(**node))
    return serialized


def _match_semantic_node(semantic_nodes: list[BlueprintNode], text: str) -> BlueprintNode | None:
    lower = text.lower()
    for node in semantic_nodes:
        node_ref = make_node_ref(node.postcode, node.primitive)
        if (
            node.primitive.lower() in lower
            or node.postcode.lower() in lower
            or node_ref.lower() in lower
        ):
            return node
    return None


def _build_depth_report(
    semantic_nodes: list[BlueprintNode],
    trust_payload: Dict[str, Any],
    gap_count: int,
) -> DepthReport:
    total = max(len(semantic_nodes), 1)
    filled = sum(1 for node in semantic_nodes if node.fill_state == "F")
    partial = sum(1 for node in semantic_nodes if node.fill_state == "P")
    empty = sum(1 for node in semantic_nodes if node.fill_state == "E")
    blocked = sum(1 for node in semantic_nodes if node.fill_state == "B")
    candidate = sum(1 for node in semantic_nodes if node.fill_state == "C")
    weighted_fill = (
        filled
        + partial * 0.7
        + blocked * 0.4
        + candidate * 0.35
    ) / total

    trust_score = float(trust_payload.get("overall_score") or 0.0) / 100.0
    score = max(0.1, min(0.97, ((weighted_fill + trust_score) / 2.0) - min(0.22, gap_count * 0.03)))

    if score < 0.35:
        label = "sketch"
    elif score < 0.55:
        label = "demo"
    elif score < 0.78:
        label = "standard"
    else:
        label = "production"

    active_layers = {node.layer for node in semantic_nodes}
    average_scope_depth = (
        sum(node.depth for node in semantic_nodes) / len(semantic_nodes)
        if semantic_nodes else None
    )

    return DepthReport(
        label=label,
        average_scope_depth=average_scope_depth,
        filled_ratio=filled / total,
        partial_ratio=partial / total,
        empty_ratio=empty / total,
        activated_layer_ratio=(len(active_layers) / 19.0) if active_layers else 0.0,
        gaps_remaining=gap_count,
    )


def _build_cost_report(
    usage_payload: Dict[str, Any],
    stage_timings: Dict[str, float],
    success: bool,
) -> CostReport:
    actual_usd = usage_payload.get("cost_usd")
    by_agent: Dict[str, float] = {}
    if isinstance(actual_usd, (float, int)) and actual_usd > 0 and stage_timings:
        total_timing = sum(stage_timings.values()) or 0.0
        if total_timing > 0:
            by_agent = {
                stage: round(float(actual_usd) * (timing / total_timing), 4)
                for stage, timing in stage_timings.items()
            }

    return CostReport(
        actual_usd=float(actual_usd) if isinstance(actual_usd, (float, int)) else None,
        by_agent=by_agent,
        halted=not success,
    )


def _extract_human_decisions(
    semantic_nodes: list[BlueprintNode],
    context_graph: Dict[str, Any],
) -> list[dict[str, Any]]:
    decisions = []
    for entry in context_graph.get("decision_trace", []) or []:
        sender = str(entry.get("sender") or "").lower()
        entry_type = str(entry.get("type") or "").lower()
        if sender not in {"human", "user"} and entry_type not in {"decision", "approval", "human_decision"}:
            continue

        content = str(entry.get("content") or "").strip() or "Human decision recorded"
        node = _match_semantic_node(semantic_nodes, content)
        decisions.append({
            "postcode": node.postcode if node else "INT.SEM.APP.WHY.SFT",
            "question": content,
            "answer": str(entry.get("insight") or entry.get("answer") or "confirmed"),
            "timestamp": str(entry.get("timestamp") or ""),
        })
    return decisions[:10]


def _build_governance_report(
    semantic_nodes: list[BlueprintNode],
    trust_payload: Dict[str, Any],
    verification: Dict[str, Any],
    context_graph: Dict[str, Any],
    blueprint: Dict[str, Any],
    fracture: Dict[str, Any] | None,
    stage_results: Any,
    stage_timings: Dict[str, float],
    usage_payload: Dict[str, Any],
    success: bool,
) -> GovernanceReport:
    gap_texts = [
        *(str(gap) for gap in trust_payload.get("gap_report", []) or []),
        *(f"Unresolved: {gap}" for gap in blueprint.get("unresolved", []) or []),
    ]
    promoted = sum(1 for node in semantic_nodes if node.fill_state == "F")
    partial = sum(1 for node in semantic_nodes if node.fill_state == "P")
    total_nodes = len(semantic_nodes)
    structural_coverage = ((promoted + partial * 0.6) / max(total_nodes, 1)) * 100.0
    trust_coverage = float(trust_payload.get("overall_score") or structural_coverage)
    coverage = max(0.0, min(100.0, round((structural_coverage + trust_coverage) / 2.0, 1)))

    escalations: list[dict[str, Any]] = []
    seen_escalations = set()
    fracture_escalation = _fracture_to_escalation(fracture)
    if fracture_escalation:
        item = (fracture_escalation["postcode"], fracture_escalation["question"])
        seen_escalations.add(item)
        escalations.append(fracture_escalation)

    for escalation in build_semantic_gate_escalations(
        semantic_nodes,
        blueprint=blueprint,
        trust=trust_payload,
        context_graph=context_graph,
    ):
        item = (escalation["postcode"], escalation["question"])
        if item in seen_escalations:
            continue
        seen_escalations.add(item)
        escalations.append(escalation)

    for gap in gap_texts[:8]:
        node = _match_semantic_node(semantic_nodes, gap)
        item = (
            node.postcode if node else "INT.SEM.APP.WHY.SFT",
            gap,
        )
        if item in seen_escalations:
            continue
        seen_escalations.add(item)
        escalations.append({
            "postcode": item[0],
            "question": item[1],
            "node_ref": make_node_ref(node.postcode, node.primitive) if node else None,
            "kind": "gap",
            "stage": "governance",
        })

    human_decisions = _extract_human_decisions(semantic_nodes, context_graph)
    stage_failures = [
        sr for sr in _serialize_stage_results(stage_results)
        if not sr.success
    ]
    anti_goals_checked = len(
        [
            value for value in verification.values()
            if isinstance(value, dict) and value.get("status") == "pass"
        ]
    )

    return GovernanceReport(
        total_nodes=total_nodes,
        promoted=promoted,
        quarantined=[
            {
                "postcode": node.postcode,
                "reason": "fill_state=Q",
            }
            for node in semantic_nodes
            if node.fill_state == "Q"
        ],
        escalated=escalations,
        axiom_violations=[],
        human_decisions=human_decisions,
        coverage=coverage,
        anti_goals_checked=anti_goals_checked,
        compilation_depth=_build_depth_report(semantic_nodes, trust_payload, len(gap_texts) + len(stage_failures)),
        cost_report=_build_cost_report(usage_payload, stage_timings, success),
    )


# =============================================================================
# DOMAIN ENDPOINTS
# =============================================================================

@router.get("/domains", response_model=DomainListResponse)
async def list_domains():
    """List all available domain adapters."""
    domains = []
    for name in list_adapters():
        adapter = get_adapter(name)
        domains.append(DomainInfoResponse(
            name=adapter.name,
            version=adapter.version,
            output_format=adapter.materialization.output_format,
            file_extension=adapter.materialization.file_extension,
            vocabulary_types=sorted(adapter.vocabulary.type_keywords.keys()),
            relationship_types=sorted(adapter.vocabulary.relationship_flows.keys()),
            actionability_checks=list(adapter.verification.actionability_checks),
        ))
    return DomainListResponse(domains=domains)


@router.get("/domains/{name}", response_model=DomainInfoResponse)
async def get_domain_info(name: str):
    """Get metadata for a specific domain adapter."""
    try:
        adapter = get_adapter(name)
    except ConfigurationError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return DomainInfoResponse(
        name=adapter.name,
        version=adapter.version,
        output_format=adapter.materialization.output_format,
        file_extension=adapter.materialization.file_extension,
        vocabulary_types=sorted(adapter.vocabulary.type_keywords.keys()),
        relationship_types=sorted(adapter.vocabulary.relationship_flows.keys()),
        actionability_checks=list(adapter.verification.actionability_checks),
    )


# =============================================================================
# COMPILE ENDPOINT
# =============================================================================

@router.post("/compile", response_model=V2CompileResponse)
async def compile_v2(request: V2CompileRequest, raw_request: Request):
    """Universal domain-parameterized compilation.

    The core compilation endpoint. Routes through the appropriate domain adapter
    and returns trust indicators alongside the compiled output.

    Runs synchronous engine.compile() in a thread pool to avoid blocking
    the asyncio event loop (which would freeze health checks and polling).
    """
    import asyncio

    # Validate domain adapter exists
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # BYO LLM key: pass directly to engine (no env var mutation)
    llm_api_key = raw_request.headers.get("x-llm-api-key")
    llm_provider = raw_request.headers.get("x-llm-provider")

    # Detect provider from key prefix if not explicit
    effective_provider = request.provider or "auto"
    if llm_api_key and effective_provider == "auto":
        if llm_api_key.startswith("sk-ant-"):
            effective_provider = "claude"
        elif llm_api_key.startswith("sk-"):
            effective_provider = "openai"
        elif llm_api_key.startswith("xai-"):
            effective_provider = "grok"

    start_time = time.time()

    def _run_compile():
        """Run compilation in thread pool — blocking call."""
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            api_key=llm_api_key,
            provider=effective_provider,
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        return engine.compile(
            description=request.description,
            canonical_components=request.canonical_components,
            canonical_relationships=request.canonical_relationships,
            enrich=request.enrich,
        )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_compile)

        # Extract intent keywords for trust computation
        intent_keywords = []
        if hasattr(result, 'context_graph'):
            intent_keywords = result.context_graph.get("keywords", [])
        elif isinstance(result, dict):
            intent_keywords = result.get("context_graph", {}).get("keywords", [])

        # Build result dicts
        blueprint = result.blueprint if hasattr(result, 'blueprint') else result.get("blueprint", {})
        verification = result.verification if hasattr(result, 'verification') else result.get("verification", {})
        context_graph = result.context_graph if hasattr(result, 'context_graph') else result.get("context_graph", {})
        dim_meta = result.dimensional_metadata if hasattr(result, 'dimensional_metadata') else result.get("dimensional_metadata", {})
        interface_map = result.interface_map if hasattr(result, 'interface_map') else result.get("interface_map", {})
        stage_results = result.stage_results if hasattr(result, 'stage_results') else result.get("stage_results", [])
        stage_timings = result.stage_timings if hasattr(result, 'stage_timings') else result.get("stage_timings", {})
        retry_counts = result.retry_counts if hasattr(result, 'retry_counts') else result.get("retry_counts", {})
        fracture = result.fracture if hasattr(result, 'fracture') else result.get("fracture")
        error = result.error if hasattr(result, 'error') else result.get("error")
        termination_condition = (
            result.termination_condition
            if hasattr(result, "termination_condition")
            else result.get("termination_condition", {})
        )

        # Compute trust indicators
        trust = _build_trust_response(
            blueprint, verification, context_graph, dim_meta, intent_keywords,
        )
        semantic_nodes = _serialize_semantic_nodes(
            getattr(result, "semantic_nodes", None)
            or (result.get("semantic_nodes", []) if isinstance(result, dict) else [])
            or build_blueprint_semantic_nodes(
                blueprint,
                seed_text=request.description,
                trust=trust,
                verification=verification,
                run_id=f"sync:{request.domain}",
            )
        )
        usage_payload = {
            "tokens": getattr(getattr(result, "usage", None), "tokens", None)
            if not isinstance(result, dict) else (result.get("usage", {}) or {}).get("tokens"),
            "cost_usd": getattr(getattr(result, "usage", None), "cost_usd", None)
            if not isinstance(result, dict) else (result.get("usage", {}) or {}).get("cost_usd"),
        }

        duration = time.time() - start_time
        key_id = getattr(raw_request.state, "api_key_id", None)
        _metering.record_compilation(request.domain, duration, 0.0, key_id=key_id)

        success = result.success if hasattr(result, 'success') else result.get("success", False)
        governance_report = _build_governance_report(
            semantic_nodes=semantic_nodes,
            trust_payload=trust.model_dump(),
            verification=verification,
            context_graph=context_graph,
            blueprint=blueprint,
            fracture=fracture,
            stage_results=stage_results,
            stage_timings=stage_timings,
            usage_payload=usage_payload,
            success=success,
        )

        return V2CompileResponse(
            success=success,
            blueprint=blueprint,
            semantic_nodes=semantic_nodes,
            termination_condition=termination_condition,
            governance_report=governance_report,
            materialized_output={},
            trust=trust,
            yaml_output=_build_yaml_output(blueprint, verification, context_graph),
            dimensional_metadata=dim_meta,
            interface_map=interface_map,
            verification=verification,
            context_graph=context_graph,
            structured_insights=(
                result.structured_insights
                if hasattr(result, "structured_insights")
                else result.get("structured_insights", [])
            ),
            difficulty=(
                result.difficulty
                if hasattr(result, "difficulty")
                else result.get("difficulty", {})
            ),
            stage_results=_serialize_stage_results(stage_results),
            stage_timings=stage_timings,
            retry_counts=retry_counts,
            domain=request.domain,
            adapter_version=adapter.version,
            usage=UsageResponse(
                tokens=int(usage_payload.get("tokens") or 0),
                cost_usd=float(usage_payload.get("cost_usd") or 0.0),
                domain=request.domain,
                adapter_version=adapter.version,
            ),
            error=error,
        )

    except Exception as e:
        logger.exception("V2 compilation failed")
        return V2CompileResponse(
            success=False,
            domain=request.domain,
            adapter_version=adapter.version,
            error=str(e),
        )


# =============================================================================
# RECOMPILE ENDPOINT (evolve a running system)
# =============================================================================

@router.post("/recompile", response_model=V2RecompileResponse)
async def recompile_v2(request: V2RecompileRequest, raw_request: Request):
    """Recompile a running system with an enhancement description.

    Called by generated systems' SelfRecompiler to evolve themselves.
    Takes the current blueprint + a gap description and produces a new
    compilation that addresses the gap.
    """
    import asyncio

    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm_api_key = raw_request.headers.get("x-llm-api-key")
    llm_provider = raw_request.headers.get("x-llm-provider")

    effective_provider = request.provider or "auto"
    if llm_api_key and effective_provider == "auto":
        if llm_api_key.startswith("sk-ant-"):
            effective_provider = "claude"
        elif llm_api_key.startswith("sk-"):
            effective_provider = "openai"

    start_time = time.time()

    def _run_recompile():
        from core.engine import MotherlabsEngine

        current_desc = request.current_blueprint.get("core_need", "")
        components = request.current_blueprint.get("components", [])
        comp_names = [c.get("name", "") for c in components if c.get("name")]

        enhanced_description = (
            f"{current_desc}\n\n"
            f"Existing components: {', '.join(comp_names)}\n\n"
            f"Enhancement needed: {request.enhancement}"
        )

        engine = MotherlabsEngine(
            api_key=llm_api_key,
            provider=effective_provider,
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        result = engine.compile(
            description=enhanced_description,
            canonical_components=comp_names or None,
        )
        return engine, result, comp_names

    try:
        loop = asyncio.get_event_loop()
        engine, result, comp_names = await loop.run_in_executor(None, _run_recompile)

        blueprint = result.blueprint if hasattr(result, 'blueprint') else {}
        verification = result.verification if hasattr(result, 'verification') else {}
        context_graph = result.context_graph if hasattr(result, 'context_graph') else {}
        dim_meta = result.dimensional_metadata if hasattr(result, 'dimensional_metadata') else {}
        intent_keywords = context_graph.get("keywords", []) if isinstance(context_graph, dict) else []

        trust = _build_trust_response(
            blueprint, verification, context_graph, dim_meta, intent_keywords,
        )

        # Emit code from the new blueprint
        materialized: Dict[str, str] = {}
        success = result.success if hasattr(result, 'success') else False
        if success and blueprint.get("components"):
            try:
                emission = engine.emit_code(blueprint)
                if hasattr(emission, 'generated_code') and emission.generated_code:
                    materialized = dict(emission.generated_code)
                    if adapter.runtime is not None:
                        from core.runtime_scaffold import (
                            generate_runtime_py, generate_state_py,
                            generate_tools_py, generate_llm_client_py,
                            generate_config_py, generate_recompile_py,
                        )
                        comp_names_new = list(emission.generated_code.keys())
                        scaffolds = {
                            "runtime.py": generate_runtime_py(adapter.runtime, blueprint, comp_names_new),
                            "state.py": generate_state_py(adapter.runtime, blueprint, comp_names_new),
                            "tools.py": generate_tools_py(adapter.runtime, blueprint, comp_names_new),
                            "llm_client.py": generate_llm_client_py(adapter.runtime, blueprint, comp_names_new),
                            "config.py": generate_config_py(adapter.runtime, blueprint, comp_names_new),
                            "recompile.py": generate_recompile_py(adapter.runtime, blueprint, comp_names_new),
                        }
                        for fname, code in scaffolds.items():
                            if code:
                                materialized[fname] = code
            except Exception as emit_err:
                logger.warning("Code emission after recompile failed: %s", emit_err)

        duration = time.time() - start_time
        key_id = getattr(raw_request.state, "api_key_id", None)
        _metering.record_compilation(request.domain, duration, 0.0, key_id=key_id)

        return V2RecompileResponse(
            success=success,
            blueprint=blueprint,
            materialized_output=materialized,
            trust=trust,
            domain=request.domain,
            enhancement_applied=request.enhancement,
        )

    except Exception as e:
        logger.exception("V2 recompilation failed")
        return V2RecompileResponse(
            success=False,
            domain=request.domain,
            error=str(e),
        )


# =============================================================================
# VALIDATE ENDPOINT (trust-only, no materialization)
# =============================================================================

@router.post("/validate", response_model=V2ValidateResponse)
async def validate_v2(request: V2ValidateRequest, raw_request: Request):
    """Trust-only mode — compile and return trust indicators without materialization."""
    import asyncio

    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm_api_key = raw_request.headers.get("x-llm-api-key")

    start_time = time.time()

    def _run_validate():
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            api_key=llm_api_key,
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        return engine.compile(description=request.description)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_validate)

        blueprint = result.blueprint if hasattr(result, 'blueprint') else {}
        verification = result.verification if hasattr(result, 'verification') else {}
        context_graph = result.context_graph if hasattr(result, 'context_graph') else {}
        dim_meta = result.dimensional_metadata if hasattr(result, 'dimensional_metadata') else {}
        intent_keywords = context_graph.get("keywords", []) if isinstance(context_graph, dict) else []

        trust = _build_trust_response(
            blueprint, verification, context_graph, dim_meta, intent_keywords,
        )

        success = result.success if hasattr(result, 'success') else False

        duration = time.time() - start_time
        key_id = getattr(raw_request.state, "api_key_id", None)
        _metering.record_compilation(request.domain, duration, 0.0, key_id=key_id)

        return V2ValidateResponse(
            success=success,
            trust=trust,
            blueprint=blueprint,
            domain=request.domain,
        )

    except Exception as e:
        logger.exception("V2 validation failed")
        return V2ValidateResponse(
            success=False,
            domain=request.domain,
            error=str(e),
        )


# =============================================================================
# HEALTH & METRICS
# =============================================================================

@router.get("/health", response_model=V2HealthResponse)
async def health_v2():
    """Platform health check with worker queue depth and disk info."""
    import os
    import shutil
    import adapters  # noqa: F401 — ensure adapters registered

    data_dir = os.environ.get("MOTHERLABS_DATA_DIR", "/data")
    disk = shutil.disk_usage(data_dir) if os.path.exists(data_dir) else None

    # Worker queue depth
    queue_depth = 0
    try:
        from worker.config import huey as _huey
        queue_depth = _huey.pending_count()
    except Exception:
        pass

    # Corpus size
    corpus_size = 0
    try:
        from persistence.corpus import Corpus
        _corpus = Corpus(data_dir=data_dir)
        corpus_size = len(_corpus.list())
    except Exception:
        pass

    # Uptime
    uptime = 0.0
    try:
        uptime = time.time() - _start_time
    except Exception:
        pass

    return V2HealthResponse(
        status="ok",
        version="2.0.0",
        domains_available=list_adapters(),
        corpus_size=corpus_size,
        uptime_seconds=round(uptime, 1),
        worker_queue_depth=queue_depth,
        disk_free_mb=round(disk.free / (1024 * 1024)) if disk else 0,
        disk_total_mb=round(disk.total / (1024 * 1024)) if disk else 0,
    )


@router.get("/metrics", response_model=V2MetricsResponse)
async def metrics_v2():
    """Per-domain metrics."""
    metrics = _metering.get_metrics()
    return V2MetricsResponse(
        per_domain=metrics.get("per_domain", {}),
        total_compilations=metrics.get("total_compilations", 0),
        total_cost_usd=metrics.get("total_cost_usd", 0.0),
    )


# =============================================================================
# TOOL SHARING ENDPOINTS
# =============================================================================

@router.get("/tools", response_model=ToolSearchResponse)
async def list_tools_v2(domain: str = None, local_only: bool = False):
    """List tools in local registry."""
    from motherlabs_platform.tool_registry import get_tool_registry
    from core.tool_package import serialize_digest

    registry = get_tool_registry()
    tools = registry.list_tools(domain=domain, local_only=local_only)

    return ToolSearchResponse(
        tools=[
            ToolDigestResponse(
                package_id=t.package_id,
                name=t.name,
                domain=t.domain,
                fingerprint=t.fingerprint,
                trust_score=t.trust_score,
                verification_badge=t.verification_badge,
                component_count=t.component_count,
                relationship_count=t.relationship_count,
                source_instance_id=t.source_instance_id,
                created_at=t.created_at,
            )
            for t in tools
        ],
        total=len(tools),
    )


@router.get("/tools/search", response_model=ToolSearchResponse)
async def search_tools_v2(q: str = "", domain: str = None):
    """Search tools by name or domain."""
    from motherlabs_platform.tool_registry import get_tool_registry

    registry = get_tool_registry()

    if q:
        tools = registry.search_tools(q)
    else:
        tools = registry.list_tools(domain=domain)

    return ToolSearchResponse(
        tools=[
            ToolDigestResponse(
                package_id=t.package_id,
                name=t.name,
                domain=t.domain,
                fingerprint=t.fingerprint,
                trust_score=t.trust_score,
                verification_badge=t.verification_badge,
                component_count=t.component_count,
                relationship_count=t.relationship_count,
                source_instance_id=t.source_instance_id,
                created_at=t.created_at,
            )
            for t in tools
        ],
        total=len(tools),
    )


@router.post("/tools/export", response_model=ToolExportResponse)
async def export_tool_v2(request: ToolExportRequest):
    """Package a compilation for export."""
    from core.tool_export import export_tool
    from motherlabs_platform.instance_identity import InstanceIdentityStore
    from motherlabs_platform.tool_registry import get_tool_registry
    from persistence.corpus import Corpus

    try:
        identity_store = InstanceIdentityStore()
        identity = identity_store.get_or_create_self()
        corpus = Corpus()
        registry = get_tool_registry()

        pkg = export_tool(
            compilation_id=request.compilation_id,
            corpus=corpus,
            instance_id=identity.instance_id,
            name=request.name,
            version=request.version,
        )

        # Register locally
        try:
            registry.register_tool(pkg, is_local=True)
        except Exception:
            pass  # Already registered (idempotent)

        return ToolExportResponse(
            success=True,
            package_id=pkg.package_id,
            name=pkg.name,
            trust_score=pkg.trust_score,
            verification_badge=pkg.verification_badge,
        )
    except ValueError as e:
        return ToolExportResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Tool export failed")
        return ToolExportResponse(success=False, error=str(e))


@router.post("/tools/import", response_model=ToolImportResponse)
async def import_tool_v2(request: ToolImportRequest):
    """Import a tool package with governor validation."""
    from core.tool_package import deserialize_tool_package
    from core.tool_export import import_tool
    from motherlabs_platform.instance_identity import InstanceIdentityStore
    from motherlabs_platform.tool_registry import get_tool_registry

    try:
        pkg = deserialize_tool_package(request.package)
    except (KeyError, ValueError) as e:
        return ToolImportResponse(
            success=False, allowed=False,
            rejection_reason=f"Deserialization failed: {e}",
            error=str(e),
        )

    try:
        identity_store = InstanceIdentityStore()
        identity = identity_store.get_or_create_self()
        registry = get_tool_registry()

        result = import_tool(
            package=pkg,
            registry=registry,
            instance_id=identity.instance_id,
            min_trust_score=request.min_trust_score,
            require_verified=request.require_verified,
        )

        return ToolImportResponse(
            success=result.allowed,
            allowed=result.allowed,
            rejection_reason=result.rejection_reason,
            provenance_valid=result.provenance_valid,
            trust_sufficient=result.trust_sufficient,
            code_safe=result.code_safe,
            warnings=list(result.warnings),
            checks_performed=list(result.checks_performed),
        )
    except Exception as e:
        logger.exception("Tool import failed")
        return ToolImportResponse(success=False, error=str(e))


@router.get("/tools/{package_id}", response_model=ToolPackageResponse)
async def get_tool_v2(package_id: str):
    """Get full tool package by ID."""
    from motherlabs_platform.tool_registry import get_tool_registry
    from core.tool_package import serialize_tool_package

    registry = get_tool_registry()
    pkg = registry.get_tool(package_id)

    if not pkg:
        raise HTTPException(status_code=404, detail=f"Tool not found: {package_id}")

    data = serialize_tool_package(pkg)
    return ToolPackageResponse(
        package_id=pkg.package_id,
        name=pkg.name,
        version=pkg.version,
        domain=pkg.domain,
        trust_score=pkg.trust_score,
        verification_badge=pkg.verification_badge,
        fingerprint=pkg.fingerprint,
        blueprint=pkg.blueprint,
        generated_code=pkg.generated_code,
        fidelity_scores=pkg.fidelity_scores,
        provenance_chain=data.get("provenance_chain", []),
        source_instance_id=pkg.source_instance_id,
        created_at=pkg.created_at,
    )


# =============================================================================
# INSTANCE DISCOVERY ENDPOINTS
# =============================================================================

@router.get("/instance/digest", response_model=TrustGraphDigestResponse)
async def instance_digest_v2():
    """This instance's trust graph digest."""
    from motherlabs_platform.instance_identity import (
        InstanceIdentityStore,
        build_trust_graph_digest,
        serialize_trust_graph_digest,
    )
    from motherlabs_platform.tool_registry import get_tool_registry

    store = InstanceIdentityStore()
    identity = store.get_or_create_self()
    registry = get_tool_registry()

    digest = build_trust_graph_digest(
        identity.instance_id,
        identity.name,
        registry,
    )
    data = serialize_trust_graph_digest(digest)
    return TrustGraphDigestResponse(**data)


@router.post("/instance/peers")
async def register_peer_v2(request: PeerRegisterRequest):
    """Register a known peer instance."""
    from motherlabs_platform.instance_identity import InstanceIdentityStore

    store = InstanceIdentityStore()
    store.register_peer(request.instance_id, request.name, request.api_endpoint)
    return {"success": True, "instance_id": request.instance_id}


@router.get("/instance/peers")
async def list_peers_v2():
    """List known peer instances."""
    from motherlabs_platform.instance_identity import InstanceIdentityStore

    store = InstanceIdentityStore()
    peers = store.list_peers()

    return {
        "peers": [
            {
                "instance_id": p.instance_id,
                "name": p.name,
                "created_at": p.created_at,
                "api_endpoint": p.api_endpoint,
            }
            for p in peers
        ],
        "total": len(peers),
    }


@router.get("/instance/peers/status")
async def peers_status_v2():
    """Check liveness of all known peers.

    Returns health status for each registered peer. Does not require
    PeerManager — performs one-shot health checks via PeerClient.
    """
    from motherlabs_platform.instance_identity import InstanceIdentityStore
    from motherlabs_platform.peer_client import PeerClient

    store = InstanceIdentityStore()
    peers = store.list_peers()

    statuses = []
    for p in peers:
        client = PeerClient(p.api_endpoint, timeout=5.0)
        try:
            status = await client.health_check(p.instance_id)
            statuses.append(PeerStatusResponse(
                instance_id=p.instance_id,
                name=p.name,
                api_endpoint=p.api_endpoint,
                reachable=status.reachable,
                latency_ms=status.latency_ms,
                last_checked=status.last_checked,
                error=status.error,
            ))
        except Exception as e:
            statuses.append(PeerStatusResponse(
                instance_id=p.instance_id,
                name=p.name,
                api_endpoint=p.api_endpoint,
                reachable=False,
                error=str(e),
            ))
        finally:
            await client.close()

    return {"peers": [s.model_dump() for s in statuses], "total": len(statuses)}


# =============================================================================
# ASYNC COMPILATION ENDPOINTS
# =============================================================================

@router.post("/compile/async", response_model=AsyncCompileResponse)
async def compile_async(request: V2CompileRequest, raw_request: Request):
    """Enqueue a compilation for async execution. Returns a task ID for polling.

    Routes through the swarm full_build pipeline so pass 1 gets the
    complete agent chain (memory + retrieval + compile + coding) — no
    separate "2nd pass" needed for production-quality output.
    """
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # BYO LLM key: encrypt before passing to Huey (never store plaintext on disk)
    from worker.keystore import encrypt_if_present
    llm_api_key_enc = encrypt_if_present(raw_request.headers.get("x-llm-api-key"))
    llm_provider = raw_request.headers.get("x-llm-provider")

    from worker.swarm_tasks import swarm_execute_task

    task = swarm_execute_task(
        intent=request.description,
        domain=request.domain,
        request_type="full_build",
        provider=request.provider,
        llm_api_key_enc=llm_api_key_enc,
        cost_cap_usd=5.0,
    )

    return AsyncCompileResponse(
        task_id=task.id,
        status="queued",
        poll_url=f"/v2/tasks/{task.id}",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Poll the status of an async compilation task."""
    from worker.config import huey
    from worker.progress import read_progress, write_task_ledger

    raw_result = huey.result(task_id, preserve=True)

    progress = None
    try:
        progress = read_progress(task_id)
    except Exception:
        progress = None

    if raw_result is None:
        status = "running" if progress and progress.get("stage_index", 0) > 0 else "pending"
        return TaskStatusResponse(task_id=task_id, status=status, progress=progress)

    result = _normalize_swarm_compile_result(raw_result)
    progress_termination = (progress or {}).get("termination_condition") or {}
    if isinstance(result, dict) and progress_termination:
        result["termination_condition"] = progress_termination

    if progress_termination:
        return TaskStatusResponse(
            task_id=task_id,
            status="complete",
            result=result,
            progress=progress,
        )

    blocking_escalations = list(result.get("blocking_escalations") or []) if isinstance(result, dict) else []
    fracture_escalation = _fracture_to_escalation(result.get("fracture")) if isinstance(result, dict) else None
    pause_escalations = blocking_escalations or ([fracture_escalation] if fracture_escalation else [])

    if pause_escalations:
        try:
            write_task_ledger(
                task_id,
                escalations=pause_escalations,
                stage="awaiting_decision",
                index=progress.get("stage_index", 0) if progress else 0,
            )
            progress = read_progress(task_id)
        except Exception:
            progress = progress or None
        return TaskStatusResponse(
            task_id=task_id,
            status="awaiting_decision",
            result=result,
            progress=progress,
        )

    if isinstance(result, dict) and result.get("error") and not result.get("success", False):
        return TaskStatusResponse(
            task_id=task_id,
            status="error",
            result=result,
            error=result["error"],
            progress=progress,
        )
    try:
        governance = result.get("governance_report") or {}
        write_task_ledger(
            task_id,
            escalations=list(governance.get("escalated", []) or []),
            human_decisions=list(governance.get("human_decisions", []) or []),
        )
        progress = read_progress(task_id)
    except Exception:
        progress = progress or None

    return TaskStatusResponse(
        task_id=task_id,
        status="complete",
        result=result,
        progress=progress,
    )


@router.post("/tasks/{task_id}/decisions", response_model=TaskDecisionResponse)
async def record_task_decision(task_id: str, request: TaskDecisionRequest, raw_request: Request):
    """Append a human decision and continue a paused compile when possible."""
    from datetime import datetime, timezone
    from worker.config import huey
    from worker.progress import append_human_decision, read_progress, write_task_termination

    timestamp = request.timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    decision = {
        "postcode": request.postcode,
        "question": request.question,
        "answer": request.answer,
        "timestamp": timestamp,
    }
    append_human_decision(
        task_id,
        decision,
    )

    next_task_id = None
    raw_result = huey.result(task_id, preserve=True)
    normalized = _normalize_swarm_compile_result(raw_result) if isinstance(raw_result, dict) else {}
    fracture = normalized.get("fracture") if isinstance(normalized, dict) else None
    blocking_escalations = list(normalized.get("blocking_escalations") or []) if isinstance(normalized, dict) else []

    if fracture or blocking_escalations:
        from worker.keystore import encrypt_if_present
        from worker.swarm_tasks import swarm_execute_task

        state = raw_result.get("state", {}) if isinstance(raw_result, dict) else {}
        matching_pause = next(
            (
                escalation for escalation in blocking_escalations
                if str(escalation.get("postcode") or "") == request.postcode
                and str(escalation.get("question") or "") == request.question
            ),
            None,
        )
        continuation_guard = _build_continuation_guard(
            huey,
            task_id,
            request.postcode,
            request.question,
            list((matching_pause or {}).get("options") or []),
        )
        if continuation_guard:
            write_task_termination(task_id, continuation_guard)
            return TaskDecisionResponse(
                task_id=task_id,
                saved=True,
                progress=read_progress(task_id),
                next_task_id=None,
                termination_condition=continuation_guard,
            )
        resumed = swarm_execute_task(
            intent=_append_human_decision_to_intent(
                normalized.get("input_text") or state.get("intent") or "",
                fracture,
                decision,
                pause_options=list((matching_pause or {}).get("options") or []),
            ),
            domain=str(normalized.get("domain") or state.get("domain") or "software"),
            request_type=str(state.get("request_type") or "full_build"),
            provider=state.get("provider"),
            llm_api_key_enc=encrypt_if_present(raw_request.headers.get("x-llm-api-key")),
            llm_provider=raw_request.headers.get("x-llm-provider"),
            cost_cap_usd=float(state.get("cost_cap_usd") or 5.0),
            previous_task_id=task_id,
        )
        next_task_id = resumed.id

    return TaskDecisionResponse(
        task_id=task_id,
        saved=True,
        progress=read_progress(task_id),
        next_task_id=next_task_id,
        termination_condition=None,
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending async compilation task."""
    from worker.config import huey

    revoked = huey.revoke_by_id(task_id)
    return {"task_id": task_id, "cancelled": True}


# =============================================================================
# CORPUS BENCHMARKING (Feature 5)
# =============================================================================

@router.get("/corpus/benchmarks", response_model=CorpusBenchmarkResponse)
async def get_corpus_benchmarks(domain: str = "software"):
    """Get corpus benchmarks for a domain — averages for comparison."""
    try:
        from persistence.sqlite_corpus import SQLiteCorpus
        corpus = SQLiteCorpus()
        stats = corpus.get_stats()

        total = stats.get("total_compilations", 0)
        domain_count = stats.get("domains", {}).get(domain, 0)

        if domain_count == 0:
            return CorpusBenchmarkResponse(
                domain=domain,
                total_compilations=0,
                avg_component_count=0.0,
                avg_trust_score=0.0,
                avg_gap_count=0.0,
            )

        # Query domain-specific averages
        conn = corpus._get_connection()
        try:
            row = conn.execute(
                """
                SELECT
                    AVG(components_count) as avg_components,
                    AVG(insights_count) as avg_insights
                FROM compilations
                WHERE LOWER(domain) = LOWER(?) AND success = 1
                """,
                (domain,),
            ).fetchone()

            avg_components = row["avg_components"] or 0.0
            avg_insights = row["avg_insights"] or 0.0
        finally:
            conn.close()

        return CorpusBenchmarkResponse(
            domain=domain,
            total_compilations=domain_count,
            avg_component_count=round(avg_components, 1),
            avg_trust_score=round(stats.get("success_rate", 0.0) * 100, 1),
            avg_gap_count=round(avg_insights, 1),
        )
    except Exception as e:
        logger.warning("Corpus benchmarks failed: %s", e)
        return CorpusBenchmarkResponse(domain=domain)
