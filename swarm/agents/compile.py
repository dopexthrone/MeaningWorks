"""CompileAgent — wraps MotherlabsEngine.compile() as a swarm agent.

Reads: intent, domain, research_context, retrieval_context, memory_context
Writes: blueprint, verification, context_graph, compile_result, trust
"""

import logging
from typing import Any, Dict, List

from swarm.agents.base import SwarmAgent
from swarm.state import SwarmState

logger = logging.getLogger("motherlabs.swarm.compile")

DEFAULT_ESCALATION_POSTCODE = "INT.SEM.APP.IF.SFT"


def get_compile_halt_info(compile_result: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Describe whether the compile step must stop the swarm."""
    if not isinstance(compile_result, dict):
        return None

    fracture = compile_result.get("fracture") or {}
    if fracture:
        question = fracture.get("collapsing_constraint") or "Clarification required before compilation can continue."
        return {
            "error_type": "awaiting_decision",
            "message": compile_result.get("error") or question,
            "recoverable": True,
            "escalation": {
                "postcode": DEFAULT_ESCALATION_POSTCODE,
                "question": question,
                "options": list(fracture.get("competing_configs") or []),
            },
        }

    blocking_escalations = list(compile_result.get("blocking_escalations") or [])
    if blocking_escalations:
        escalation = dict(blocking_escalations[0])
        return {
            "error_type": "awaiting_decision",
            "message": escalation.get("question") or "A semantic gate requires human input before coding can continue.",
            "recoverable": True,
            "escalation": {
                "postcode": escalation.get("postcode") or DEFAULT_ESCALATION_POSTCODE,
                "question": escalation.get("question") or "A semantic gate requires human input before coding can continue.",
                "options": list(escalation.get("options") or []),
            },
        }

    if compile_result.get("success", True):
        return None

    return {
        "error_type": "compile_failed",
        "message": compile_result.get("error") or "Compile step failed.",
        "recoverable": False,
    }


class CompileAgent(SwarmAgent):
    """Run the semantic compiler pipeline as a swarm step."""

    name = "compile"
    criticality = "critical"

    @property
    def input_keys(self) -> List[str]:
        return ["intent", "domain", "research_context", "retrieval_context", "memory_context"]

    @property
    def output_keys(self) -> List[str]:
        return ["blueprint", "verification", "context_graph", "compile_result", "trust"]

    def execute(self, state: SwarmState, config: Dict[str, Any]) -> SwarmState:
        """Run MotherlabsEngine.compile() with enriched description.

        Prepends any pre-compilation context (research, retrieval, memory)
        to the user's intent before sending to the compiler.
        """
        from core.engine import MotherlabsEngine
        from core.adapter_registry import get_adapter
        from core.blueprint_protocol import (
            build_semantic_gate_escalations,
            project_legacy_blueprint_nodes,
        )
        from core.exceptions import FractureError
        from core.protocol_spec import FractureSignal
        from core.trust import compute_trust_indicators, serialize_trust_indicators

        # Build enriched description from pre-compilation agents
        description = self._enrich_description(state)

        # Bridge swarm progress callback to core engine format
        swarm_callback = config.pop("_progress_callback", None)
        config.pop("_task_id", None)  # Clean up — task_id handled by swarm_callback closure
        # Create engine with user's LLM config
        adapter = get_adapter(state.domain)
        on_interrogate = None
        if swarm_callback:
            def on_interrogate(request):
                if not request.questions:
                    return None

                question = request.questions[0]
                escalation = {
                    "postcode": DEFAULT_ESCALATION_POSTCODE,
                    "question": question.question,
                    "options": list(question.options or []),
                }
                try:
                    swarm_callback(
                        "compile",
                        0,
                        "awaiting_decision",
                        question.question,
                        escalations=[escalation],
                    )
                except Exception:
                    pass

                raise FractureError(
                    "Clarification required before compilation can continue",
                    stage="interrogation",
                    signal=FractureSignal(
                        stage="interrogation",
                        competing_configs=list(question.options or []),
                        collapsing_constraint=question.question,
                        agent="Interrogation",
                        context=request.context,
                    ),
                )

        engine = MotherlabsEngine(
            api_key=state.llm_api_key,
            provider=state.provider,
            pipeline_mode="staged",
            domain_adapter=adapter,
            on_interrogate=on_interrogate,
        )

        engine_progress = None
        if swarm_callback:
            def engine_progress(stage: str, index: int, insight: str = "", **kwargs):
                """Adapter: core engine callback → swarm progress store."""
                try:
                    swarm_callback(
                        "compile", index, stage, insight or f"compile:{stage}",
                        **kwargs,
                    )
                except Exception:
                    pass

        # Run compilation
        result = engine.compile(
            description=description,
            enrich=config.get("enrich", False),
            progress_callback=engine_progress,
        )

        # Extract outputs
        blueprint = result.blueprint if hasattr(result, "blueprint") else {}
        verification = result.verification if hasattr(result, "verification") else {}
        context_graph = result.context_graph if hasattr(result, "context_graph") else {}
        dim_meta = result.dimensional_metadata if hasattr(result, "dimensional_metadata") else {}
        intent_keywords = context_graph.get("keywords", []) if isinstance(context_graph, dict) else []

        # Compute trust
        trust_indicators = compute_trust_indicators(
            blueprint=blueprint,
            verification=verification,
            context_graph=context_graph,
            dimensional_metadata=dim_meta,
            intent_keywords=intent_keywords,
        )
        trust_data = serialize_trust_indicators(trust_indicators)

        success = result.success if hasattr(result, "success") else False
        raw_semantic_nodes = getattr(result, "semantic_nodes", None) if hasattr(result, "semantic_nodes") else None
        semantic_nodes = (
            list(raw_semantic_nodes)
            if isinstance(raw_semantic_nodes, list) and raw_semantic_nodes
            else [
                node.model_dump()
                for node in project_legacy_blueprint_nodes(
                    blueprint,
                    seed_text=state.intent,
                    trust=trust_data,
                    verification=verification,
                    run_id=state.swarm_id or "swarm",
                )
            ]
        )
        raw_blocking_escalations = getattr(result, "blocking_escalations", None) if hasattr(result, "blocking_escalations") else None
        blocking_escalations = (
            list(raw_blocking_escalations)
            if isinstance(raw_blocking_escalations, list) and raw_blocking_escalations
            else build_semantic_gate_escalations(
                semantic_nodes,
                blueprint=blueprint,
                trust=trust_data,
                context_graph=context_graph,
            )
        )

        # Serialize compile result for downstream agents
        compile_result_dict = {
            "success": success,
            "blueprint": blueprint,
            "verification": verification,
            "context_graph": context_graph,
            "dimensional_metadata": dim_meta,
            "interface_map": result.interface_map if hasattr(result, "interface_map") else {},
            "structured_insights": (
                result.structured_insights if hasattr(result, "structured_insights") else []
            ),
            "difficulty": (
                result.difficulty if hasattr(result, "difficulty") else {}
            ),
            "stage_results": [
                {
                    "stage": sr.stage,
                    "success": sr.success,
                    "errors": list(sr.errors),
                    "warnings": list(sr.warnings),
                    "retries": getattr(sr, "retries", 0),
                }
                for sr in (result.stage_results if hasattr(result, "stage_results") else [])
            ],
            "stage_timings": (
                result.stage_timings if hasattr(result, "stage_timings") else {}
            ),
            "retry_counts": (
                result.retry_counts if hasattr(result, "retry_counts") else {}
            ),
            "semantic_nodes": semantic_nodes,
            "blocking_escalations": blocking_escalations,
            "error": (
                result.error if hasattr(result, "error") else None
            ),
            "fracture": (
                result.fracture if hasattr(result, "fracture") else None
            ),
            "interrogation": (
                result.interrogation if hasattr(result, "interrogation") else {}
            ),
            "termination_condition": (
                result.termination_condition if hasattr(result, "termination_condition") else {}
            ),
        }

        logger.info(
            "CompileAgent completed: domain=%s success=%s components=%d",
            state.domain,
            success,
            len(blueprint.get("components", [])),
        )

        return state.with_updates(
            blueprint=blueprint,
            verification=verification,
            context_graph=context_graph,
            compile_result=compile_result_dict,
            trust=trust_data,
        )

    def _enrich_description(self, state: SwarmState) -> str:
        """Prepend pre-compilation context to the user's intent.

        This is how Research/Retrieval/Memory agents improve compilation
        quality — they provide domain context that gets woven into the prompt.
        """
        parts = []

        if state.research_context:
            findings = state.research_context.get("findings", "")
            if findings:
                parts.append(f"[Research Context]\n{findings}\n")

        if state.retrieval_context:
            docs = state.retrieval_context.get("relevant_documents", "")
            if docs:
                parts.append(f"[Retrieved Context]\n{docs}\n")

        if state.memory_context:
            patterns = state.memory_context.get("relevant_patterns", "")
            if patterns:
                parts.append(f"[Memory Context]\n{patterns}\n")

        if state.previous_blueprint:
            bp_summary = _summarize_blueprint(state.previous_blueprint)
            if bp_summary:
                parts.append(
                    f"[Previous Compilation Blueprint]\n{bp_summary}\n"
                    f"Improve this: fill method bodies, add cross-module wiring, "
                    f"resolve integration gaps.\n"
                )

        if parts:
            parts.append(f"[User Intent]\n{state.intent}")
            return "\n".join(parts)

        return state.intent


def _summarize_blueprint(blueprint: Dict[str, Any]) -> str:
    """Summarize a blueprint into compact text for LLM context.

    Extracts component names, types, relationships, and constraints
    into a text representation suitable for enrichment context.
    """
    lines: list = []
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])
    constraints = blueprint.get("constraints", [])

    if components:
        lines.append(f"Components ({len(components)}):")
        for comp in components:
            name = comp.get("name", "?")
            ctype = comp.get("type", "?")
            methods = comp.get("methods", [])
            method_names = [m.get("name", "?") for m in methods] if methods else []
            method_str = f" methods=[{', '.join(method_names)}]" if method_names else ""
            lines.append(f"  - {name} ({ctype}){method_str}")

    if relationships:
        lines.append(f"Relationships ({len(relationships)}):")
        for rel in relationships[:20]:  # Cap at 20 to avoid context overflow
            src = rel.get("from_component", "?")
            tgt = rel.get("to_component", "?")
            rtype = rel.get("type", "?")
            lines.append(f"  - {src} --{rtype}--> {tgt}")

    if constraints:
        lines.append(f"Constraints ({len(constraints)}):")
        for con in constraints[:10]:
            lines.append(f"  - {con.get('description', '?')}")

    return "\n".join(lines)
