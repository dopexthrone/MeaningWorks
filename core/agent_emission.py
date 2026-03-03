"""
Motherlabs Agent Emission — LLM-driven code generation from blueprints.

Phase D: Agent Emission
Derived from: DIMENSIONAL_BLUEPRINT.md — "AI agents = print heads.
Materialize layers of the model into executable reality."

Uses MaterializationPlan to dispatch LLM calls per-node, extract code,
verify interfaces. The engine orchestrates; this module provides
frozen dataclasses + pure functions only.

This is a LEAF MODULE — imports only materialization + interface_schema +
compilation_tree + stdlib. No engine/protocol/pipeline imports.
"""

import ast
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from core.materialization import (
    MaterializationPlan,
    NodePrompt,
    verify_interfaces,
)
from core.interface_schema import InterfaceMap
from core.compilation_tree import format_l2_patterns_section, L2Synthesis
from core.naming import to_snake


# =============================================================================
# CONSTANTS
# =============================================================================

EMISSION_PREAMBLE = """You are a code generation agent for the Motherlabs semantic compiler.
Generate a FULLY IMPLEMENTED, production-ready Python implementation for one component.
Rules:
1. Output ONLY Python code inside a ```python block.
2. Honor ALL declared interfaces exactly.
3. Do NOT add undeclared dependencies.
4. Include docstrings and type hints.
5. The code must be syntactically valid Python.
6. Do NOT redefine types that are available via import. Use the imported version exactly as shown.
7. EVERY method MUST have a real implementation. No stubs, no `pass`, no `...`, no placeholder comments, no TODO markers.
8. If a method's behavior is described in the component spec, implement that exact behavior.
9. If a method's behavior is not fully specified, implement the most reasonable behavior based on the component's purpose and its relationships to other components.
10. The output must be RUNNABLE — a user should be able to import and use this component immediately with no modifications.
"""

EMISSION_VERSION = "agent_emission:v1.0"


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class EmissionConfig:
    """Configuration for agent emission.

    Derived from: Phase D — emission configuration
    """
    max_tokens: int = 16384
    temperature: float = 0.0          # C008 determinism
    system_preamble: str = ""         # Extra context (e.g. project conventions)
    retry_failed: bool = True
    max_retries: int = 2              # Allows syntax repair + one API retry


@dataclass(frozen=True)
class NodeEmission:
    """Result of emitting code for a single blueprint node.

    Derived from: Phase D — per-node emission result
    """
    component_name: str
    component_type: str
    code: str                         # Generated Python code
    success: bool
    error: Optional[str]
    prompt_hash: str                  # SHA256[:16] of prompt
    derived_from: str                 # "agent_emission:v1.0"


@dataclass(frozen=True)
class BatchEmission:
    """Result of emitting code for one materialization batch.

    Derived from: Phase D — per-batch emission result
    """
    batch_index: int
    emissions: Tuple[NodeEmission, ...]
    success_count: int
    failure_count: int
    layer: Optional[int] = None  # EmissionLayer value when layered


@dataclass(frozen=True)
class EmissionResult:
    """Complete emission result across all batches.

    Derived from: Phase D — aggregate emission output
    """
    batch_emissions: Tuple[BatchEmission, ...]
    generated_code: Dict[str, str]    # component_name -> code
    verification_report: Dict[str, Any]
    total_nodes: int
    success_count: int
    failure_count: int
    pass_rate: float                  # verification pass rate
    l2_context_injected: bool
    timestamp: str
    derived_from: str
    layer_gate_results: Tuple = ()   # Tuple[LayerGateResult, ...] when layered
    layered: bool = False            # Whether layered emission was used


# =============================================================================
# PURE FUNCTIONS — Prompt Building
# =============================================================================

def build_emission_system_prompt(
    node_prompt: NodePrompt,
    l2_section: Optional[str] = None,
    config: Optional[EmissionConfig] = None,
    emission_preamble: Optional[str] = None,
) -> str:
    """Assemble system prompt for a single node emission.

    Combines: preamble + config preamble + node prompt + L2 context.

    Args:
        node_prompt: NodePrompt from materialization plan
        l2_section: Optional formatted L2 patterns section
        config: Optional EmissionConfig with system_preamble
        emission_preamble: Optional domain-specific preamble (default: software)

    Returns:
        Complete system prompt string

    Derived from: Phase D — prompt assembly
    """
    preamble = emission_preamble if emission_preamble else EMISSION_PREAMBLE
    parts = [preamble.strip()]

    if config and config.system_preamble:
        parts.append("")
        parts.append(config.system_preamble.strip())

    parts.append("")
    parts.append(node_prompt.prompt_text)

    if l2_section:
        parts.append("")
        parts.append(l2_section)

    return "\n".join(parts)


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256[:16] hash of prompt text for traceability.

    Args:
        prompt_text: The full prompt string

    Returns:
        First 16 hex characters of SHA256 digest

    Derived from: Phase D — prompt traceability
    """
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# PURE FUNCTIONS — Code Extraction
# =============================================================================

def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles:
    1. ```python ... ``` blocks (preferred)
    2. ``` ... ``` blocks without language tag
    3. Bare code (no markdown wrapping)
    4. Multiple code blocks (concatenated)
    5. Empty or malformed responses

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code (may be empty string if nothing found)

    Derived from: Phase D — code extraction from LLM output
    """
    if not response or not response.strip():
        return ""

    # Strategy 0: Response starts with code fence — strip markers
    # Handles truncated LLM output where closing ``` may be absent
    stripped = response.strip()
    if stripped.startswith("```python"):
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3].rstrip()
        return body.strip()
    if stripped.startswith("```") and not stripped.startswith("```\n"):
        # ```lang tag other than python — skip, let strategies below handle
        pass
    elif stripped.startswith("```"):
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3].rstrip()
        if _looks_like_python(body):
            return body.strip()

    # Strategy 1: Extract ```python ... ``` blocks
    python_blocks = re.findall(
        r'```python\s*\n(.*?)```',
        response,
        re.DOTALL,
    )
    if python_blocks:
        return "\n\n".join(block.strip() for block in python_blocks)

    # Strategy 2: Extract ``` ... ``` blocks (no language tag)
    generic_blocks = re.findall(
        r'```\s*\n(.*?)```',
        response,
        re.DOTALL,
    )
    if generic_blocks:
        # Filter to blocks that look like Python
        py_candidates = [
            block.strip() for block in generic_blocks
            if _looks_like_python(block)
        ]
        if py_candidates:
            return "\n\n".join(py_candidates)
        # If no Python-looking blocks, take all
        return "\n\n".join(block.strip() for block in generic_blocks)

    # Strategy 3: Bare code — response itself looks like Python
    stripped = response.strip()
    if _looks_like_python(stripped):
        return stripped

    return ""


def _looks_like_python(text: str) -> bool:
    """Heuristic check whether text looks like Python code."""
    markers = (
        "def ", "class ", "import ", "from ", "if ", "for ",
        "while ", "return ", "self.", "print(", "raise ",
    )
    return any(marker in text for marker in markers)


# =============================================================================
# PURE FUNCTIONS — Assembly
# =============================================================================

def assemble_emission(
    batch_emissions: List[BatchEmission],
    interface_map: InterfaceMap,
    l2_injected: bool,
) -> EmissionResult:
    """Aggregate batch emissions into final EmissionResult.

    Collects generated code, runs interface verification, computes metrics.

    Args:
        batch_emissions: List of BatchEmission from all batches
        interface_map: InterfaceMap for verification
        l2_injected: Whether L2 context was injected into prompts

    Returns:
        Complete EmissionResult with verification report

    Derived from: Phase D — emission assembly
    """
    generated_code: Dict[str, str] = {}
    total_success = 0
    total_failure = 0

    for batch in batch_emissions:
        total_success += batch.success_count
        total_failure += batch.failure_count
        for emission in batch.emissions:
            if emission.success and emission.code:
                generated_code[emission.component_name] = emission.code

    total_nodes = total_success + total_failure

    # Run interface verification
    verification_report = verify_interfaces(generated_code, interface_map)

    return EmissionResult(
        batch_emissions=tuple(batch_emissions),
        generated_code=generated_code,
        verification_report=verification_report,
        total_nodes=total_nodes,
        success_count=total_success,
        failure_count=total_failure,
        pass_rate=verification_report.get("pass_rate", 0.0),
        l2_context_injected=l2_injected,
        timestamp=datetime.now().isoformat(),
        derived_from=EMISSION_VERSION,
    )


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_emission_result(result: EmissionResult) -> dict:
    """Serialize EmissionResult to JSON-safe dict.

    Derived from: Phase D — emission serialization
    """
    return {
        "batch_emissions": [
            {
                "batch_index": be.batch_index,
                "emissions": [
                    {
                        "component_name": ne.component_name,
                        "component_type": ne.component_type,
                        "code": ne.code,
                        "success": ne.success,
                        "error": ne.error,
                        "prompt_hash": ne.prompt_hash,
                        "derived_from": ne.derived_from,
                    }
                    for ne in be.emissions
                ],
                "success_count": be.success_count,
                "failure_count": be.failure_count,
            }
            for be in result.batch_emissions
        ],
        "generated_code": dict(result.generated_code),
        "verification_report": result.verification_report,
        "total_nodes": result.total_nodes,
        "success_count": result.success_count,
        "failure_count": result.failure_count,
        "pass_rate": result.pass_rate,
        "l2_context_injected": result.l2_context_injected,
        "timestamp": result.timestamp,
        "derived_from": result.derived_from,
    }


def deserialize_emission_result(data: dict) -> EmissionResult:
    """Deserialize EmissionResult from JSON-safe dict.

    Derived from: Phase D — emission deserialization
    """
    batch_emissions = tuple(
        BatchEmission(
            batch_index=be["batch_index"],
            emissions=tuple(
                NodeEmission(
                    component_name=ne["component_name"],
                    component_type=ne["component_type"],
                    code=ne["code"],
                    success=ne["success"],
                    error=ne.get("error"),
                    prompt_hash=ne["prompt_hash"],
                    derived_from=ne["derived_from"],
                )
                for ne in be["emissions"]
            ),
            success_count=be["success_count"],
            failure_count=be["failure_count"],
        )
        for be in data["batch_emissions"]
    )

    return EmissionResult(
        batch_emissions=batch_emissions,
        generated_code=data["generated_code"],
        verification_report=data["verification_report"],
        total_nodes=data["total_nodes"],
        success_count=data["success_count"],
        failure_count=data["failure_count"],
        pass_rate=data["pass_rate"],
        l2_context_injected=data["l2_context_injected"],
        timestamp=data["timestamp"],
        derived_from=data["derived_from"],
    )


# =============================================================================
# PURE FUNCTIONS — Deduplication
# =============================================================================

def dedup_emitted_classes(
    generated_code: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    """Deduplicate classes defined in multiple emitted files.

    When multiple agents independently define the same helper class (e.g. Signal),
    keep the richest definition (most AST body nodes) and replace others with imports.

    Args:
        generated_code: component_name → code string

    Returns:
        Tuple of (modified_code, dedup_log) where dedup_log lists actions taken.

    Derived from: Layered Emission Validation — Fix 4
    """
    # Step 1: Parse each file, collect top-level ClassDef nodes
    file_classes: Dict[str, Dict[str, ast.ClassDef]] = {}  # file -> {class_name -> node}
    parseable: Dict[str, ast.Module] = {}

    for comp_name, code in generated_code.items():
        try:
            tree = ast.parse(code, filename=f"{comp_name}.py")
            parseable[comp_name] = tree
            classes = {}
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    classes[node.name] = node
            file_classes[comp_name] = classes
        except SyntaxError:
            file_classes[comp_name] = {}

    # Step 2: Find classes defined in 2+ files
    class_locations: Dict[str, List[str]] = {}  # class_name -> [file_names]
    for comp_name, classes in file_classes.items():
        for cls_name in classes:
            class_locations.setdefault(cls_name, []).append(comp_name)

    duplicated = {name: files for name, files in class_locations.items() if len(files) >= 2}

    if not duplicated:
        return dict(generated_code), []

    # Step 3: For each duplicated class, keep richest, replace others with import
    result = dict(generated_code)
    dedup_log: List[str] = []

    for cls_name, file_list in duplicated.items():
        # Find richest definition (most body nodes)
        best_file = max(
            file_list,
            key=lambda f: len(file_classes[f][cls_name].body) if cls_name in file_classes[f] else 0,
        )

        for comp_name in file_list:
            if comp_name == best_file:
                continue

            cls_node = file_classes[comp_name].get(cls_name)
            if not cls_node:
                continue

            # Replace class definition with import statement
            code = result[comp_name]
            lines = code.split("\n")

            # Find the line range of the class definition
            start_line = cls_node.lineno - 1  # 0-indexed
            end_line = cls_node.end_lineno if cls_node.end_lineno else start_line + 1

            # Build import line
            module_name = to_snake(best_file)
            import_line = f"from .{module_name} import {cls_name}"

            # Replace class lines with import
            new_lines = lines[:start_line] + [import_line] + lines[end_line:]
            result[comp_name] = "\n".join(new_lines)

            dedup_log.append(
                f"Replaced {cls_name} in {comp_name} with import from {best_file}"
            )

    return result, dedup_log
