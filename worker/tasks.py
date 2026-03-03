"""Huey tasks — wraps MotherlabsEngine.compile() for async execution.

Each task creates its own MotherlabsEngine instance (stateless).
retries=0 because the engine has its own retry logic via FailoverClient.
"""

import logging
import signal
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from worker.config import huey

logger = logging.getLogger("motherlabs.worker")

TASK_TIMEOUT = 300  # 5 minutes — kills hung compilations


class TaskTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TaskTimeout(f"Compilation exceeded {TASK_TIMEOUT}s timeout")


@huey.task(retries=0)
def compile_task(
    description: str,
    domain: str = "software",
    provider: Optional[str] = None,
    enrich: bool = False,
    canonical_components: Optional[list] = None,
    canonical_relationships: Optional[list] = None,
) -> Dict[str, Any]:
    """Run a compilation asynchronously.

    Returns a serializable dict (CompileResult fields + metadata).
    """
    from core.engine import MotherlabsEngine
    from core.adapter_registry import get_adapter
    from core.trust import compute_trust_indicators, serialize_trust_indicators

    start = time.time()

    # Set timeout (Unix only — signal.SIGALRM not available on Windows)
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TASK_TIMEOUT)
    except (AttributeError, ValueError):
        old_handler = None  # Windows or non-main thread — skip timeout

    try:
        adapter = get_adapter(domain)
        engine = MotherlabsEngine(
            provider=provider or "auto",
            pipeline_mode="staged",
            domain_adapter=adapter,
        )

        result = engine.compile(
            description=description,
            canonical_components=canonical_components,
            canonical_relationships=canonical_relationships,
            enrich=enrich,
        )

        blueprint = result.blueprint if hasattr(result, "blueprint") else {}
        verification = result.verification if hasattr(result, "verification") else {}
        context_graph = result.context_graph if hasattr(result, "context_graph") else {}
        dim_meta = result.dimensional_metadata if hasattr(result, "dimensional_metadata") else {}
        interface_map = result.interface_map if hasattr(result, "interface_map") else {}
        intent_keywords = context_graph.get("keywords", []) if isinstance(context_graph, dict) else []

        trust_indicators = compute_trust_indicators(
            blueprint=blueprint,
            verification=verification,
            context_graph=context_graph,
            dimensional_metadata=dim_meta,
            intent_keywords=intent_keywords,
        )
        trust_data = serialize_trust_indicators(trust_indicators)

        duration = time.time() - start
        success = result.success if hasattr(result, "success") else False

        logger.info("Compilation completed: domain=%s duration=%.1fs success=%s", domain, duration, success)
        return {
            "success": success,
            "blueprint": blueprint,
            "trust": trust_data,
            "verification": verification,
            "context_graph": context_graph,
            "dimensional_metadata": dim_meta,
            "interface_map": interface_map,
            "domain": domain,
            "adapter_version": adapter.version,
            "duration_seconds": round(duration, 2),
            "error": None,
        }

    except TaskTimeout:
        duration = time.time() - start
        logger.error(
            "DEAD LETTER: Compilation timed out after %.0fs: domain=%s description=%.100s",
            duration, domain, description,
        )
        return {
            "success": False,
            "blueprint": {},
            "trust": {},
            "verification": {},
            "context_graph": {},
            "dimensional_metadata": {},
            "interface_map": {},
            "domain": domain,
            "adapter_version": "unknown",
            "duration_seconds": round(duration, 2),
            "error": f"Compilation timed out after {TASK_TIMEOUT}s. Try a shorter description or 'fast' trust level.",
        }

    except Exception as e:
        duration = time.time() - start
        logger.exception(
            "DEAD LETTER: Compilation failed after %.1fs: domain=%s error=%s",
            duration, domain, str(e),
        )
        return {
            "success": False,
            "blueprint": {},
            "trust": {},
            "verification": {},
            "context_graph": {},
            "dimensional_metadata": {},
            "interface_map": {},
            "domain": domain,
            "adapter_version": "unknown",
            "duration_seconds": round(duration, 2),
            "error": str(e),
        }

    finally:
        # Clear timeout
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (AttributeError, ValueError):
            pass
