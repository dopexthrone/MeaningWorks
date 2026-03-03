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
)
from core.adapter_registry import get_adapter, list_adapters
from core.exceptions import ConfigurationError
from core.trust import compute_trust_indicators, serialize_trust_indicators
from motherlabs_platform.metering import MeteringTracker

logger = logging.getLogger("motherlabs.api.v2")

router = APIRouter(prefix="/v2", tags=["v2"])

# Module-level metering tracker
_metering = MeteringTracker()


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
    """
    # Validate domain adapter exists
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.time()

    try:
        # Import engine lazily to avoid circular imports
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            provider=request.provider or "auto",
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        result = engine.compile(
            description=request.description,
            canonical_components=request.canonical_components,
            canonical_relationships=request.canonical_relationships,
            enrich=request.enrich,
        )

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

        # Compute trust indicators
        trust = _build_trust_response(
            blueprint, verification, context_graph, dim_meta, intent_keywords,
        )

        duration = time.time() - start_time
        key_id = getattr(raw_request.state, "api_key_id", None)
        _metering.record_compilation(request.domain, duration, 0.0, key_id=key_id)

        success = result.success if hasattr(result, 'success') else result.get("success", False)

        return V2CompileResponse(
            success=success,
            blueprint=blueprint,
            materialized_output={},
            trust=trust,
            dimensional_metadata=dim_meta,
            interface_map=interface_map,
            verification=verification,
            context_graph=context_graph,
            domain=request.domain,
            adapter_version=adapter.version,
            usage=UsageResponse(
                domain=request.domain,
                adapter_version=adapter.version,
            ),
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
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.time()

    try:
        from core.engine import MotherlabsEngine

        # Build an enhanced description from blueprint + enhancement
        current_desc = request.current_blueprint.get("core_need", "")
        components = request.current_blueprint.get("components", [])
        comp_names = [c.get("name", "") for c in components if c.get("name")]

        enhanced_description = (
            f"{current_desc}\n\n"
            f"Existing components: {', '.join(comp_names)}\n\n"
            f"Enhancement needed: {request.enhancement}"
        )

        engine = MotherlabsEngine(
            provider=request.provider or "auto",
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        result = engine.compile(
            description=enhanced_description,
            canonical_components=comp_names or None,
        )

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
                    # Also generate scaffold files for runtime-capable systems
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
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.time()

    try:
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        result = engine.compile(description=request.description)

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

    return V2HealthResponse(
        status="ok",
        version="2.0.0",
        domains_available=list_adapters(),
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
    """Enqueue a compilation for async execution. Returns a task ID for polling."""
    try:
        adapter = get_adapter(request.domain)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from worker.tasks import compile_task

    task = compile_task(
        description=request.description,
        domain=request.domain,
        provider=request.provider,
        enrich=request.enrich,
        canonical_components=request.canonical_components,
        canonical_relationships=request.canonical_relationships,
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

    result = huey.result(task_id, preserve=True)

    if result is None:
        # Task is still pending or running
        return TaskStatusResponse(task_id=task_id, status="pending")

    if isinstance(result, dict) and result.get("error"):
        return TaskStatusResponse(
            task_id=task_id,
            status="error",
            result=result,
            error=result["error"],
        )

    return TaskStatusResponse(
        task_id=task_id,
        status="complete",
        result=result,
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending async compilation task."""
    from worker.config import huey

    revoked = huey.revoke_by_id(task_id)
    return {"task_id": task_id, "cancelled": True}
