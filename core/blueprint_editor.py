"""
Motherlabs Blueprint Editor — human-in-the-loop edit operations.

Phase 16: Human-in-the-Loop Iteration
Derived from: NEXT-STEPS.md — "Users cannot correct or refine output
without re-running entire compilation."

Provides atomic edit operations on compiled blueprints. Each operation
takes a blueprint dict, applies a change, and returns a new dict with
edit history for lineage tracking.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# EDIT OPERATIONS
# =============================================================================

@dataclass(frozen=True)
class EditOperation:
    """
    Record of a single edit applied to a blueprint.

    Immutable — forms an append-only lineage chain.
    """
    operation: str        # "rename" | "remove" | "merge" | "add_constraint" | "flag_hollow" | "add_component"
    target: str           # Component name or constraint description
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EditResult:
    """
    Result of applying edits to a blueprint.

    Contains the modified blueprint plus lineage metadata.
    """
    blueprint: Dict[str, Any]
    operations_applied: List[EditOperation]
    warnings: List[str]
    components_before: int
    components_after: int


def rename_component(
    blueprint: Dict[str, Any],
    old_name: str,
    new_name: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Rename a component throughout the blueprint.

    Updates: component name, relationship endpoints, constraint applies_to,
    and unresolved mentions.

    Args:
        blueprint: Blueprint dict to modify (will be deep-copied)
        old_name: Current component name
        new_name: New component name

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []
    found = False

    # Rename in components
    for comp in bp.get("components", []):
        if comp.get("name", "").lower() == old_name.lower():
            comp["name"] = new_name
            found = True

        # Rename in subsystem children
        sub = comp.get("sub_blueprint")
        if sub:
            for sub_comp in sub.get("components", []):
                if sub_comp.get("name", "").lower() == old_name.lower():
                    sub_comp["name"] = new_name
                    found = True

    if not found:
        warnings.append(f"Component '{old_name}' not found in blueprint")
        return bp, warnings

    # Rename in relationships
    for rel in bp.get("relationships", []):
        if rel.get("from", "").lower() == old_name.lower():
            rel["from"] = new_name
        if rel.get("to", "").lower() == old_name.lower():
            rel["to"] = new_name

    # Rename in constraints
    for constraint in bp.get("constraints", []):
        applies_to = constraint.get("applies_to", [])
        constraint["applies_to"] = [
            new_name if a.lower() == old_name.lower() else a
            for a in applies_to
        ]
        # Also update description if it mentions the old name
        desc = constraint.get("description", "")
        if old_name.lower() in desc.lower():
            constraint["description"] = desc.replace(old_name, new_name)

    # Update unresolved mentions
    bp["unresolved"] = [
        u.replace(old_name, new_name) if old_name.lower() in u.lower() else u
        for u in bp.get("unresolved", [])
    ]

    return bp, warnings


def remove_component(
    blueprint: Dict[str, Any],
    name: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Remove a component and its relationships from the blueprint.

    Args:
        blueprint: Blueprint dict to modify (will be deep-copied)
        name: Component name to remove

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []
    name_lower = name.lower()

    # Remove from components
    original_count = len(bp.get("components", []))
    bp["components"] = [
        c for c in bp.get("components", [])
        if c.get("name", "").lower() != name_lower
    ]
    if len(bp.get("components", [])) == original_count:
        warnings.append(f"Component '{name}' not found in blueprint")
        return bp, warnings

    # Remove relationships involving this component
    bp["relationships"] = [
        r for r in bp.get("relationships", [])
        if r.get("from", "").lower() != name_lower and r.get("to", "").lower() != name_lower
    ]

    # Remove from constraint applies_to
    for constraint in bp.get("constraints", []):
        constraint["applies_to"] = [
            a for a in constraint.get("applies_to", [])
            if a.lower() != name_lower
        ]

    return bp, warnings


def merge_components(
    blueprint: Dict[str, Any],
    names: List[str],
    merged_name: str,
    merged_type: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Merge multiple components into one.

    Combines methods, relationships, and constraints. Preserves the richest
    description among the merged components.

    Args:
        blueprint: Blueprint dict to modify
        names: Component names to merge
        merged_name: Name for the merged component
        merged_type: Type for merged component (auto-detect if empty)

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []
    names_lower = {n.lower() for n in names}

    # Collect components to merge
    to_merge = []
    remaining = []
    for comp in bp.get("components", []):
        if comp.get("name", "").lower() in names_lower:
            to_merge.append(comp)
        else:
            remaining.append(comp)

    if len(to_merge) < 2:
        warnings.append(f"Need at least 2 components to merge, found {len(to_merge)}")
        return bp, warnings

    # Build merged component
    # Use richest description (longest)
    descriptions = [c.get("description", "") for c in to_merge]
    best_desc = max(descriptions, key=len) if descriptions else ""

    # Combine methods
    all_methods = []
    seen_methods = set()
    for comp in to_merge:
        for m in comp.get("methods", []):
            m_name = m.get("name", "")
            if m_name not in seen_methods:
                all_methods.append(m)
                seen_methods.add(m_name)

    # Auto-detect type from majority
    if not merged_type:
        types = [c.get("type", "entity") for c in to_merge]
        merged_type = max(set(types), key=types.count)

    merged = {
        "name": merged_name,
        "type": merged_type,
        "description": best_desc,
        "derived_from": f"merged from: {', '.join(names)}",
        "methods": all_methods,
    }

    remaining.append(merged)
    bp["components"] = remaining

    # Update relationships to point to merged name
    for rel in bp.get("relationships", []):
        if rel.get("from", "").lower() in names_lower:
            rel["from"] = merged_name
        if rel.get("to", "").lower() in names_lower:
            rel["to"] = merged_name

    # Remove self-referential relationships created by merge
    bp["relationships"] = [
        r for r in bp.get("relationships", [])
        if not (r.get("from", "").lower() == merged_name.lower()
                and r.get("to", "").lower() == merged_name.lower())
    ]

    # Update constraints
    for constraint in bp.get("constraints", []):
        applies = constraint.get("applies_to", [])
        if any(a.lower() in names_lower for a in applies):
            constraint["applies_to"] = [
                merged_name if a.lower() in names_lower else a
                for a in applies
            ]
            # Deduplicate
            constraint["applies_to"] = list(dict.fromkeys(constraint["applies_to"]))

    return bp, warnings


def add_constraint(
    blueprint: Dict[str, Any],
    description: str,
    applies_to: List[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Add a new constraint to the blueprint.

    Args:
        blueprint: Blueprint dict to modify
        description: Constraint description
        applies_to: Component names this constraint applies to

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []

    # Verify components exist
    existing = {c.get("name", "").lower() for c in bp.get("components", [])}
    for name in applies_to:
        if name.lower() not in existing:
            warnings.append(f"Component '{name}' not found — constraint may be orphaned")

    bp.setdefault("constraints", []).append({
        "description": description,
        "applies_to": applies_to,
        "derived_from": "user edit (Phase 16)",
    })

    return bp, warnings


def flag_hollow(
    blueprint: Dict[str, Any],
    name: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Flag a component as hollow (placeholder without real substance).

    Adds a warning to unresolved and marks the component for review.

    Args:
        blueprint: Blueprint dict to modify
        name: Component name to flag

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []
    name_lower = name.lower()
    found = False

    for comp in bp.get("components", []):
        if comp.get("name", "").lower() == name_lower:
            comp["hollow"] = True
            found = True
            break

    if not found:
        warnings.append(f"Component '{name}' not found in blueprint")
        return bp, warnings

    bp.setdefault("unresolved", []).append(
        f"HOLLOW: '{name}' flagged as potentially hollow — needs review or removal"
    )

    return bp, warnings


def add_component(
    blueprint: Dict[str, Any],
    name: str,
    comp_type: str = "entity",
    description: str = "",
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Add a new component to the blueprint.

    Args:
        blueprint: Blueprint dict to modify
        name: Component name
        comp_type: Component type
        description: Component description

    Returns:
        (modified_blueprint, warnings)
    """
    bp = copy.deepcopy(blueprint)
    warnings = []

    # Check for duplicates
    existing = {c.get("name", "").lower() for c in bp.get("components", [])}
    if name.lower() in existing:
        warnings.append(f"Component '{name}' already exists")
        return bp, warnings

    bp.setdefault("components", []).append({
        "name": name,
        "type": comp_type,
        "description": description or f"User-added component: {name}",
        "derived_from": "user edit (Phase 16)",
        "methods": [],
    })

    return bp, warnings


# =============================================================================
# EDIT ORCHESTRATOR
# =============================================================================

def apply_edits(
    blueprint: Dict[str, Any],
    edits: List[Dict[str, Any]],
) -> EditResult:
    """
    Apply a sequence of edit operations to a blueprint.

    Each edit is a dict with:
    - "operation": str — one of rename/remove/merge/add_constraint/flag_hollow/add_component
    - Plus operation-specific fields (see individual functions)

    Args:
        blueprint: Original blueprint dict
        edits: List of edit operation dicts

    Returns:
        EditResult with modified blueprint, applied operations, and warnings
    """
    bp = copy.deepcopy(blueprint)
    all_warnings = []
    operations = []
    components_before = len(bp.get("components", []))

    for edit in edits:
        op = edit.get("operation", "")
        warnings = []

        if op == "rename":
            bp, warnings = rename_component(bp, edit.get("old_name", ""), edit.get("new_name", ""))
            operations.append(EditOperation(
                operation="rename",
                target=edit.get("old_name", ""),
                details={"new_name": edit.get("new_name", "")},
            ))

        elif op == "remove":
            bp, warnings = remove_component(bp, edit.get("name", ""))
            operations.append(EditOperation(
                operation="remove",
                target=edit.get("name", ""),
            ))

        elif op == "merge":
            bp, warnings = merge_components(
                bp, edit.get("names", []), edit.get("merged_name", ""),
                edit.get("merged_type", ""),
            )
            operations.append(EditOperation(
                operation="merge",
                target=edit.get("merged_name", ""),
                details={"merged_from": edit.get("names", [])},
            ))

        elif op == "add_constraint":
            bp, warnings = add_constraint(
                bp, edit.get("description", ""), edit.get("applies_to", []),
            )
            operations.append(EditOperation(
                operation="add_constraint",
                target=edit.get("description", ""),
                details={"applies_to": edit.get("applies_to", [])},
            ))

        elif op == "flag_hollow":
            bp, warnings = flag_hollow(bp, edit.get("name", ""))
            operations.append(EditOperation(
                operation="flag_hollow",
                target=edit.get("name", ""),
            ))

        elif op == "add_component":
            bp, warnings = add_component(
                bp, edit.get("name", ""), edit.get("type", "entity"),
                edit.get("description", ""),
            )
            operations.append(EditOperation(
                operation="add_component",
                target=edit.get("name", ""),
            ))

        else:
            all_warnings.append(f"Unknown operation: '{op}'")
            continue

        all_warnings.extend(warnings)

    return EditResult(
        blueprint=bp,
        operations_applied=operations,
        warnings=all_warnings,
        components_before=components_before,
        components_after=len(bp.get("components", [])),
    )
