"""
Motherlabs Build Loop — iterative fix loop for generated projects.

Phase 27: Runtime Build Loop (Step 4)

For each failing component: build fix prompt → LLM call → extract fixed code →
re-write file → re-validate. Repeat up to max_iterations.

This is a LEAF MODULE — imports only runtime_validator + agent_emission +
project_writer + stdlib.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from core.runtime_validator import (
    RuntimeConfig,
    ValidationResult,
    ComponentError,
    validate_project,
)
from core.agent_emission import extract_code_from_response
from core.project_writer import ProjectManifest, write_project, ProjectConfig


# =============================================================================
# CONSTANTS
# =============================================================================

BUILD_FIX_PREAMBLE = """You are a code repair agent for the Motherlabs semantic compiler.
Fix the error in the generated Python component below.
Rules:
1. Output ONLY the fixed Python code inside a ```python block.
2. Honor ALL declared interfaces exactly.
3. Do NOT change the class/function API — only fix the internal implementation.
4. The code must be syntactically valid Python.
5. Do NOT add new dependencies unless absolutely necessary.
"""


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class BuildConfig:
    """Configuration for the build loop."""
    max_iterations: int = 10
    max_fixes_per_component: int = 5
    runtime_config: RuntimeConfig = RuntimeConfig()


@dataclass(frozen=True)
class FixAttempt:
    """Record of a single fix attempt for a component."""
    component_name: str
    iteration: int
    error: str
    prompt: str
    original_code: str
    fixed_code: str
    succeeded: bool


@dataclass(frozen=True)
class BuildIteration:
    """Record of one build-validate-fix cycle."""
    iteration: int
    validation: ValidationResult
    fixes_attempted: Tuple[FixAttempt, ...] = ()
    components_fixed: Tuple[str, ...] = ()
    components_still_broken: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    """Complete result of the build loop."""
    success: bool
    iterations: Tuple[BuildIteration, ...] = ()
    final_code: Dict[str, str] = field(default_factory=dict)
    final_manifest: Optional[ProjectManifest] = None
    components_fixed: Tuple[str, ...] = ()
    components_unfixed: Tuple[str, ...] = ()
    total_fix_attempts: int = 0


# =============================================================================
# FIX PROMPT CONSTRUCTION
# =============================================================================

def build_fix_prompt(
    component_name: str,
    error: ComponentError,
    code: str,
    blueprint: Dict[str, Any],
    interfaces: Optional[Dict[str, Any]] = None,
    generated_code: Optional[Dict[str, str]] = None,
) -> str:
    """Build a prompt for the LLM to fix a specific component error.

    Args:
        component_name: Name of the failing component
        error: The ComponentError describing the failure
        code: Current code for the component
        blueprint: Blueprint dict (for context)
        interfaces: Optional interface contracts
        generated_code: Optional dict of all generated code (for adjacent module context)

    Returns:
        Fix prompt string
    """
    # Find component in blueprint for context
    comp_info = ""
    for comp in blueprint.get("components", []):
        if comp.get("name") == component_name:
            comp_info = (
                f"Component type: {comp.get('type', 'unknown')}\n"
                f"Description: {comp.get('description', 'N/A')}\n"
            )
            break

    # Build interface section
    interface_section = ""
    if interfaces:
        # Look for this component's contracts
        contracts = interfaces.get("contracts", {})
        comp_contract = contracts.get(component_name, {})
        if comp_contract:
            interface_section = f"\n## Interface Contract\n{_format_contract(comp_contract)}\n"

    # Build constraint section
    constraint_section = ""
    constraints = blueprint.get("constraints", [])
    relevant = [c for c in constraints
                if isinstance(c, dict) and component_name in str(c.get("applies_to", []))]
    if relevant:
        constraint_section = "\n## Relevant Constraints\n"
        for c in relevant:
            constraint_section += f"- {c.get('description', str(c))}\n"

    # Build referenced module section for import/name errors
    referenced_section = ""
    if generated_code and error.error_type in ("import", "name"):
        # Try to find the referenced module name from the error message
        msg = error.error_message
        for ref_name, ref_code in generated_code.items():
            if ref_name == component_name:
                continue
            if ref_name in msg or ref_name.lower() in msg.lower():
                # Show first 30 lines (class def + method signatures)
                snippet_lines = ref_code.split("\n")[:30]
                snippet = "\n".join(snippet_lines)
                referenced_section = f"\n## Referenced Module: {ref_name}\n```python\n{snippet}\n```\n"
                break

    prompt = f"""{BUILD_FIX_PREAMBLE}

## Component: {component_name}
{comp_info}
## Error
Type: {error.error_type}
Message: {error.error_message}
File: {error.file_path}
Line: {error.line_number}
{interface_section}{constraint_section}{referenced_section}
## Current Code
```python
{code}
```

Fix the error above and return the corrected code.
"""
    return prompt


def _format_contract(contract: Dict[str, Any]) -> str:
    """Format an interface contract for inclusion in a prompt."""
    lines = []
    if "provides" in contract:
        lines.append("Provides:")
        for item in contract["provides"]:
            lines.append(f"  - {item}")
    if "requires" in contract:
        lines.append("Requires:")
        for item in contract["requires"]:
            lines.append(f"  - {item}")
    if "endpoints" in contract:
        lines.append("Endpoints:")
        for ep in contract["endpoints"]:
            lines.append(f"  - {ep}")
    return "\n".join(lines)


# =============================================================================
# FIX SELECTION
# =============================================================================

def identify_components_to_fix(
    validation: ValidationResult,
    fix_history: Dict[str, int],
    max_fixes: int = 2,
) -> List[ComponentError]:
    """Select which components to attempt fixing this iteration.

    Prioritizes components with fewer previous fix attempts.
    Skips components that have already exceeded max_fixes.

    Args:
        validation: Current validation result
        fix_history: Dict[component_name, fix_count] of past attempts
        max_fixes: Maximum fixes per component

    Returns:
        List of ComponentErrors to fix (deduplicated by component)
    """
    seen = set()
    to_fix = []

    for error in validation.component_errors:
        name = error.component_name
        if name in seen:
            continue
        if fix_history.get(name, 0) >= max_fixes:
            continue
        seen.add(name)
        to_fix.append(error)

    # Sort by fewest previous attempts first
    to_fix.sort(key=lambda e: fix_history.get(e.component_name, 0))

    return to_fix


# =============================================================================
# APPLY FIX
# =============================================================================

def apply_fix_to_project(
    project_dir: str,
    component_name: str,
    new_code: str,
    generated_code: Dict[str, str],
    blueprint: Dict[str, Any],
    entity_types: Optional[frozenset] = None,
    file_extension: str = ".py",
) -> Tuple[Dict[str, str], Optional[ProjectManifest]]:
    """Apply a fixed component's code to the project.

    Updates generated_code dict and re-writes the project.

    Args:
        project_dir: Project directory
        component_name: Component to replace
        new_code: Fixed code
        generated_code: Mutable dict of all component code
        blueprint: Blueprint dict
        entity_types: Optional domain-specific entity types for project writer
        file_extension: Output file extension (default: ".py")

    Returns:
        (updated generated_code, new manifest)
    """
    # Update the code dict
    updated = dict(generated_code)
    updated[component_name] = new_code

    # Re-write the project
    parent_dir = os.path.dirname(project_dir)
    project_name = os.path.basename(project_dir)
    config = ProjectConfig(project_name=project_name)

    manifest = write_project(
        updated, blueprint, parent_dir, config,
        entity_types=entity_types, file_extension=file_extension,
    )

    return updated, manifest


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_build_result(result: BuildResult) -> dict:
    """Serialize BuildResult to JSON-safe dict."""
    iterations = []
    for it in result.iterations:
        iter_dict = {
            "iteration": it.iteration,
            "validation_success": it.validation.success,
            "component_errors": [
                {
                    "component": e.component_name,
                    "type": e.error_type,
                    "message": e.error_message,
                    "file": e.file_path,
                    "line": e.line_number,
                }
                for e in it.validation.component_errors
            ],
            "unmapped_errors": list(it.validation.unmapped_errors),
            "fixes_attempted": [
                {
                    "component": f.component_name,
                    "error": f.error,
                    "succeeded": f.succeeded,
                }
                for f in it.fixes_attempted
            ],
            "components_fixed": list(it.components_fixed),
            "components_still_broken": list(it.components_still_broken),
        }
        iterations.append(iter_dict)

    return {
        "success": result.success,
        "iterations": iterations,
        "components_fixed": list(result.components_fixed),
        "components_unfixed": list(result.components_unfixed),
        "total_fix_attempts": result.total_fix_attempts,
    }
