"""CodingAgent — blueprint-to-code via engine.emit_code().

Reads:  state.blueprint, state.domain, state.compile_result, state.llm_api_key, state.provider
Writes: state.generated_code, state.project_manifest

Uses LLM. Wraps core/engine.py emit_code() — NOT AgentOrchestrator.run().
"""

import copy
import logging
import re
from typing import Any, Dict, List, Optional

from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState

logger = logging.getLogger("motherlabs.swarm.coding")


def _infer_name(intent: str) -> str:
    """Slugify intent into a project name."""
    slug = re.sub(r"[^a-z0-9]+", "_", intent.lower().strip())[:60].strip("_")
    return slug or "project"


def _match_previous_code(
    blueprint: Dict[str, Any],
    previous_generated_code: Dict[str, str],
) -> Dict[str, Any]:
    """Enrich blueprint components with previous code for evolve pass.

    Matches component names to file keys in previous_generated_code
    (keys are component names from the prior emission) and injects
    a `previous_implementation` field into each matching component.

    Returns a deep copy of the blueprint with enrichments applied.
    """
    from core.naming import to_snake

    enriched = copy.deepcopy(blueprint)
    components = enriched.get("components", [])

    # Build lookup: snake_name -> code, pascal_name -> code, raw_key -> code
    code_by_key: Dict[str, str] = {}
    for key, code in previous_generated_code.items():
        code_by_key[key] = code
        code_by_key[key.lower()] = code
        # Also index by snake form
        snake = to_snake(key)
        code_by_key[snake] = code

    matched = 0
    for comp in components:
        name = comp.get("name", "")
        snake = to_snake(name)
        # Try: exact name, lowercase, snake
        code = code_by_key.get(name) or code_by_key.get(name.lower()) or code_by_key.get(snake)
        if code:
            # Cap at 4000 chars to avoid blowing prompt budget
            comp["previous_implementation"] = code[:4000]
            matched += 1

    if matched:
        logger.info(
            "Evolve: injected previous code for %d/%d components",
            matched, len(components),
        )

    return enriched


class CodingAgent(SwarmAgent):
    """Generate code from a compiled blueprint."""

    name = "coding"
    criticality = "high"

    @property
    def input_keys(self) -> List[str]:
        return ["blueprint", "domain", "compile_result", "llm_api_key", "provider"]

    @property
    def output_keys(self) -> List[str]:
        return ["generated_code", "project_manifest"]

    def execute(self, state: SwarmState, config: Dict[str, Any]) -> SwarmState:
        """Emit code from blueprint via MotherlabsEngine.emit_code().

        Steps:
        1. Validate blueprint exists
        2. Create engine with user's LLM config
        3. Extract interface_map and dim_meta from compile_result
        4. Call engine.emit_code()
        5. Optionally write_project() if config.write_project=True
        """
        from core.engine import MotherlabsEngine
        from core.adapter_registry import get_adapter

        # 1. Validate
        if not state.blueprint:
            raise ValueError("CodingAgent requires a compiled blueprint (state.blueprint is empty)")

        # 1b. Evolve: enrich blueprint with previous code so LLM can improve
        blueprint = state.blueprint
        if state.previous_generated_code:
            blueprint = _match_previous_code(blueprint, state.previous_generated_code)

        # 2. Create engine
        adapter = get_adapter(state.domain)
        engine = MotherlabsEngine(
            api_key=state.llm_api_key,
            provider=state.provider,
            domain_adapter=adapter,
        )

        # 3. Extract from compile_result
        compile_result = state.compile_result or {}
        interface_map = compile_result.get("interface_map")
        dim_meta = compile_result.get("dimensional_metadata")

        # 4. Emit code (uses enriched blueprint with previous_implementation if evolving)
        emission_result = engine.emit_code(
            blueprint=blueprint,
            interface_map=interface_map,
            dim_meta=dim_meta,
            layered=True,
        )

        generated_code = emission_result.generated_code
        project_manifest = None

        # 5. Always build in-memory project when code was generated
        if generated_code:
            from core.project_writer import build_project_in_memory

            manifest = build_project_in_memory(
                generated_code=generated_code,
                blueprint=blueprint,
            )
            project_manifest = {
                "file_contents": manifest.file_contents,
                "files_written": list(manifest.files_written),
                "entry_point": manifest.entry_point,
                "total_lines": manifest.total_lines,
                "project_name": _infer_name(state.intent),
                "cross_module_warnings": list(manifest.cross_module_warnings),
            }

        # 6. Optional disk write (in addition to in-memory)
        if config.get("write_project") and generated_code:
            from core.project_writer import write_project

            output_dir = config.get("output_dir", "/tmp/motherlabs_output")
            write_project(
                generated_code=generated_code,
                blueprint=blueprint,
                output_dir=output_dir,
            )
            if project_manifest:
                project_manifest["output_dir"] = output_dir

        # 7. Post-emission stub detection
        stub_report = None
        if generated_code:
            try:
                from core.agent_emission import detect_stub_methods
                stubs, stub_count, total_methods = detect_stub_methods(generated_code)
                stub_ratio = stub_count / total_methods if total_methods else 0.0
                stub_report = {
                    "stub_count": stub_count,
                    "total_methods": total_methods,
                    "stub_ratio": round(stub_ratio, 3),
                    "stubs": [
                        {"file": s.file, "class": s.class_name, "method": s.method_name}
                        for s in stubs[:50]  # Cap detail list
                    ],
                }
                if stub_count > 0:
                    logger.warning(
                        "CodingAgent: %d/%d methods are stubs (%.0f%%)",
                        stub_count, total_methods, stub_ratio * 100,
                    )
            except Exception as e:
                logger.warning("Stub detection failed: %s", e)

        logger.info(
            "CodingAgent completed: components=%d pass_rate=%.1f%%",
            emission_result.total_nodes,
            emission_result.pass_rate * 100,
        )

        return state.with_updates(
            generated_code=generated_code,
            project_manifest=project_manifest,
            stub_report=stub_report,
        )
