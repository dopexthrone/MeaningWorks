"""Huey tasks for swarm execution.

Follows the same pattern as worker/tasks.py: encrypt keys in API route,
decrypt in worker, set env vars under lock, run, restore.
"""

import logging
import signal
import time
from typing import Any, Dict, Optional

from worker.config import huey

logger = logging.getLogger("motherlabs.worker.swarm")

SWARM_TIMEOUT = 600  # 10 minutes — swarm can run multiple agents


class SwarmTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SwarmTimeout(f"Swarm execution exceeded {SWARM_TIMEOUT}s timeout")


@huey.task(retries=0, context=True)
def swarm_execute_task(
    intent: str,
    domain: str = "software",
    request_type: str = "compile_only",
    provider: Optional[str] = None,
    llm_api_key_enc: Optional[str] = None,
    cost_cap_usd: float = 5.0,
    user_id: str = "anonymous",
    session_id: Optional[str] = None,
    previous_blueprint: Optional[Dict] = None,
    previous_generated_code: Optional[Dict] = None,
    previous_task_id: Optional[str] = None,
    evolve_applied: bool = False,
    task=None,
) -> Dict[str, Any]:
    """Async swarm execution via Huey worker.

    Pattern matches compile_task() exactly:
    1. Decrypt BYO key
    2. Apply env vars under lock
    3. Build SwarmState
    4. SwarmConductor.execute(state)
    5. Restore env vars
    6. Return serialized SwarmResult
    """
    from worker.keystore import decrypt_if_present, apply_llm_key, restore_llm_key, env_lock
    from worker.progress import write_progress
    from swarm.state import SwarmState
    from swarm.conductor import SwarmConductor

    start = time.time()
    task_id = task.id if task else None

    # Decrypt BYO key
    llm_api_key = decrypt_if_present(llm_api_key_enc)

    # Detect provider from key prefix
    effective_provider = provider or "auto"
    if llm_api_key and effective_provider == "auto":
        if llm_api_key.startswith("sk-ant-"):
            effective_provider = "claude"
        elif llm_api_key.startswith("sk-"):
            effective_provider = "openai"
        elif llm_api_key.startswith("xai-"):
            effective_provider = "grok"

    # Set timeout
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SWARM_TIMEOUT)
    except (AttributeError, ValueError):
        pass

    # Apply BYO key under lock
    original_env = {}
    try:
        with env_lock:
            original_env = apply_llm_key(llm_api_key, effective_provider)

        # Build state
        state = SwarmState(
            intent=intent,
            domain=domain,
            request_type=request_type,
            llm_api_key=llm_api_key,
            provider=effective_provider,
            user_id=user_id,
            session_id=session_id or "",
            cost_cap_usd=cost_cap_usd,
            previous_blueprint=previous_blueprint,
            previous_generated_code=previous_generated_code,
            previous_task_id=previous_task_id,
            evolve_applied=evolve_applied,
        )

        # Progress callback that writes to progress store
        def _on_progress(agent: str, step_index: int, stage: str, message: str, **kwargs):
            if task_id:
                try:
                    write_progress(
                        task_id, f"swarm:{agent}:{stage}", step_index, message,
                        structured_insight=kwargs.get("structured_insight"),
                        difficulty=kwargs.get("difficulty"),
                        escalations=kwargs.get("escalations"),
                        human_decisions=kwargs.get("human_decisions"),
                    )
                except Exception:
                    pass

        # Execute
        conductor = SwarmConductor()
        result = conductor.execute(state, progress_callback=_on_progress)

        duration = time.time() - start
        logger.info(
            "Swarm completed: request_type=%s domain=%s duration=%.1fs success=%s",
            request_type, domain, duration, result.success,
        )

        # Benchmark quality scoring
        final_state = result.state
        benchmark_data = None
        if final_state.project_manifest and final_state.project_manifest.get("file_contents"):
            bench_files = final_state.project_manifest["file_contents"]
        elif final_state.generated_code:
            bench_files = final_state.generated_code
        else:
            bench_files = None

        if bench_files:
            try:
                from core.benchmark import benchmark_project, serialize_benchmark
                bench = benchmark_project(
                    bench_files,
                    final_state.blueprint,
                    final_state.trust,
                    gen_label="swarm",
                )
                benchmark_data = serialize_benchmark(bench)
                logger.info(
                    "Swarm benchmark: composite=%.1f%%",
                    bench.composite_score * 100,
                )
            except Exception as bench_err:
                logger.warning("Benchmark scoring failed: %s", bench_err)

        # Corpus auto-ingest: feed successful compilations back for L2 learning
        if result.success and final_state.blueprint and final_state.blueprint.get("components"):
            try:
                from persistence.corpus import Corpus
                corpus = Corpus()
                corpus.store(
                    input_text=intent,
                    context_graph=final_state.context_graph or {},
                    blueprint=final_state.blueprint,
                    insights=[],
                    success=True,
                    provider=effective_provider,
                    verification_score=(final_state.trust or {}).get("overall_score", 0) / 100.0,
                )
            except Exception as corpus_err:
                logger.warning("Corpus auto-ingest failed: %s", corpus_err)

        result_dict = result.to_dict()
        if benchmark_data:
            result_dict["benchmark"] = benchmark_data
        return result_dict

    except SwarmTimeout:
        duration = time.time() - start
        logger.error(
            "Swarm timed out after %.0fs: request_type=%s domain=%s",
            duration, request_type, domain,
        )
        return {
            "success": False,
            "state": {"intent": intent, "domain": domain},
            "plan_executed": {},
            "agent_timings": {},
            "total_cost_usd": 0.0,
            "total_duration_s": round(duration, 2),
            "errors": [{"error_type": "timeout", "message": f"Swarm timed out after {SWARM_TIMEOUT}s"}],
            "warnings": [],
        }

    except Exception as e:
        duration = time.time() - start
        logger.exception(
            "Swarm failed after %.1fs: request_type=%s domain=%s error=%s",
            duration, request_type, domain, str(e),
        )
        return {
            "success": False,
            "state": {"intent": intent, "domain": domain},
            "plan_executed": {},
            "agent_timings": {},
            "total_cost_usd": 0.0,
            "total_duration_s": round(duration, 2),
            "errors": [{"error_type": type(e).__name__, "message": str(e)}],
            "warnings": [],
        }

    finally:
        # Restore env vars
        with env_lock:
            restore_llm_key(original_env)
        # Clear timeout
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass
