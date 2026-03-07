"""FastAPI routes for swarm endpoints.

POST /v2/swarm/execute                   → async, returns task_id
POST /v2/swarm/execute/sync              → blocks until done
GET  /v2/swarm/status/{task_id}          → poll status
GET  /v2/swarm/result/{task_id}          → get result
GET  /v2/swarm/result/{task_id}/download → download project as ZIP
GET  /v2/swarm/agents                    → list available agents
"""

import asyncio
import io
import logging
import zipfile
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.v2.swarm_models import (
    SwarmExecuteRequest,
    SwarmExecuteResponse,
    SwarmStatusResponse,
    SwarmResultResponse,
    SwarmAgentInfo,
    SwarmAgentsResponse,
)

logger = logging.getLogger("motherlabs.api.v2.swarm")

router = APIRouter(prefix="/v2/swarm", tags=["swarm"])


@router.post("/execute", response_model=SwarmExecuteResponse)
async def swarm_execute_async(request: SwarmExecuteRequest, raw_request: Request):
    """Enqueue a swarm execution for async processing. Returns task_id."""
    from worker.keystore import encrypt_if_present
    from worker.swarm_tasks import swarm_execute_task
    from swarm.conductor import SwarmConductor
    from swarm.state import SwarmState

    # Encrypt BYO key before passing to Huey
    llm_api_key_enc = encrypt_if_present(raw_request.headers.get("x-llm-api-key"))

    # Estimate cost from plan
    temp_state = SwarmState(
        intent=request.description,
        domain=request.domain,
        request_type=request.request_type,
    )
    conductor = SwarmConductor()
    plan = conductor.plan(temp_state)

    # Evolve: fetch previous blueprint if previous_task_id provided
    previous_blueprint = None
    evolve_applied = False
    if request.previous_task_id:
        from worker.config import huey as _huey
        prev_result = _huey.result(request.previous_task_id, preserve=True)
        if prev_result and isinstance(prev_result, dict):
            prev_state = prev_result.get("state", {})
            previous_blueprint = prev_state.get("blueprint") or prev_result.get("blueprint")
            previous_generated_code = prev_state.get("generated_code") or prev_result.get("materialized_output")
            if previous_blueprint and previous_blueprint.get("components"):
                evolve_applied = True
        if not evolve_applied:
            logger.warning(
                "Evolve requested but previous task %s has no usable blueprint — running cold",
                request.previous_task_id,
            )

    task = swarm_execute_task(
        intent=request.description,
        domain=request.domain,
        request_type=request.request_type,
        provider=request.provider,
        llm_api_key_enc=llm_api_key_enc,
        cost_cap_usd=request.cost_cap_usd,
        previous_blueprint=previous_blueprint,
        previous_generated_code=previous_generated_code if evolve_applied else None,
        previous_task_id=request.previous_task_id,
        evolve_applied=evolve_applied,
    )

    return SwarmExecuteResponse(
        task_id=task.id,
        estimated_cost_usd=plan.estimated_cost_usd,
    )


@router.post("/execute/sync", response_model=SwarmResultResponse)
async def swarm_execute_sync(request: SwarmExecuteRequest, raw_request: Request):
    """Execute swarm synchronously — blocks until done."""
    from swarm.state import SwarmState
    from swarm.conductor import SwarmConductor

    llm_api_key = raw_request.headers.get("x-llm-api-key")

    # Detect provider
    effective_provider = request.provider or "auto"
    if llm_api_key and effective_provider == "auto":
        if llm_api_key.startswith("sk-ant-"):
            effective_provider = "claude"
        elif llm_api_key.startswith("sk-"):
            effective_provider = "openai"
        elif llm_api_key.startswith("xai-"):
            effective_provider = "grok"

    # Evolve: fetch previous blueprint + code if previous_task_id provided
    previous_blueprint = None
    previous_generated_code = None
    evolve_applied = False
    if request.previous_task_id:
        from worker.config import huey as _huey
        prev_result = _huey.result(request.previous_task_id, preserve=True)
        if prev_result and isinstance(prev_result, dict):
            prev_state = prev_result.get("state", {})
            previous_blueprint = prev_state.get("blueprint") or prev_result.get("blueprint")
            previous_generated_code = prev_state.get("generated_code") or prev_result.get("materialized_output")
            if previous_blueprint and previous_blueprint.get("components"):
                evolve_applied = True
        if not evolve_applied:
            logger.warning(
                "Evolve requested but previous task %s has no usable blueprint — running cold",
                request.previous_task_id,
            )

    state = SwarmState(
        intent=request.description,
        domain=request.domain,
        request_type=request.request_type,
        llm_api_key=llm_api_key,
        provider=effective_provider,
        cost_cap_usd=request.cost_cap_usd,
        previous_blueprint=previous_blueprint,
        previous_generated_code=previous_generated_code if evolve_applied else None,
        previous_task_id=request.previous_task_id,
        evolve_applied=evolve_applied,
    )

    conductor = SwarmConductor()

    # Run in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, conductor.execute, state)

    final_state = result.state
    return SwarmResultResponse(
        success=result.success,
        swarm_id=final_state.swarm_id,
        evolve_applied=final_state.evolve_applied,
        blueprint=final_state.blueprint,
        trust=final_state.trust,
        verification=final_state.verification,
        generated_code=final_state.generated_code,
        project_files=(final_state.project_manifest or {}).get("file_contents"),
        project_name=(final_state.project_manifest or {}).get("project_name"),
        agent_timings=result.agent_timings,
        total_cost_usd=result.total_cost_usd,
        total_duration_s=result.total_duration_s,
        errors=result.errors,
        warnings=result.warnings,
    )


@router.get("/status/{task_id}", response_model=SwarmStatusResponse)
async def swarm_status(task_id: str):
    """Poll status of an async swarm execution."""
    from worker.config import huey
    from worker.progress import read_progress

    result = huey.result(task_id, preserve=True)

    # Read progress
    progress = None
    try:
        progress = read_progress(task_id)
    except Exception:
        pass

    if result is not None:
        # Task completed
        success = result.get("success", False) if isinstance(result, dict) else False
        status = "completed" if success else "failed"
        return SwarmStatusResponse(
            task_id=task_id,
            status=status,
            progress_pct=100.0,
        )

    # Still running or queued
    if progress and progress.get("stage_index", 0) > 0:
        current_stage = progress.get("current_stage", "")
        # Parse agent name from "swarm:agent:stage" format
        current_agent = None
        if current_stage.startswith("swarm:"):
            parts = current_stage.split(":")
            if len(parts) >= 2:
                current_agent = parts[1]

        return SwarmStatusResponse(
            task_id=task_id,
            status="running",
            progress_pct=min(progress.get("stage_index", 0) / max(progress.get("total_stages", 1), 1) * 100, 99.0),
            current_agent=current_agent,
        )

    return SwarmStatusResponse(
        task_id=task_id,
        status="queued",
    )


@router.get("/result/{task_id}", response_model=SwarmResultResponse)
async def swarm_result(task_id: str):
    """Get the result of a completed swarm execution."""
    from worker.config import huey

    result = huey.result(task_id, preserve=True)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No result for task {task_id} — task may still be running",
        )

    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Unexpected result format")

    state = result.get("state", {})
    return SwarmResultResponse(
        success=result.get("success", False),
        swarm_id=state.get("swarm_id", ""),
        evolve_applied=state.get("evolve_applied", False),
        blueprint=state.get("blueprint"),
        trust=state.get("trust"),
        verification=state.get("verification"),
        generated_code=state.get("generated_code"),
        project_files=(state.get("project_manifest") or {}).get("file_contents"),
        project_name=(state.get("project_manifest") or {}).get("project_name"),
        agent_timings=result.get("agent_timings", {}),
        total_cost_usd=result.get("total_cost_usd", 0.0),
        total_duration_s=result.get("total_duration_s", 0.0),
        errors=result.get("errors", []),
        warnings=result.get("warnings", []),
    )


@router.get("/result/{task_id}/download")
async def swarm_download(task_id: str):
    """Download project files as a ZIP archive."""
    from worker.config import huey

    result = huey.result(task_id, preserve=True)

    if result is None or not isinstance(result, dict):
        raise HTTPException(
            status_code=404,
            detail=f"No result for task {task_id}",
        )

    state = result.get("state", {})
    manifest = state.get("project_manifest") or {}
    file_contents = manifest.get("file_contents")

    if not file_contents:
        raise HTTPException(
            status_code=404,
            detail="No project files available for download",
        )

    project_name = manifest.get("project_name", "project")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in file_contents.items():
            zf.writestr(f"{project_name}/{filename}", content)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{project_name}.zip"',
        },
    )


@router.get("/agents", response_model=SwarmAgentsResponse)
async def swarm_agents():
    """List available swarm agents."""
    from swarm.conductor import SwarmConductor

    conductor = SwarmConductor()
    agents = conductor.list_agents()

    return SwarmAgentsResponse(
        agents=[SwarmAgentInfo(**a) for a in agents],
    )
