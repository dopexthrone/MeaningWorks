"""
Motherlabs Blueprint Health — pre-materialization validation.

Phase 17.2: Edge Case Handling

LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.

Checks:
- component_count >= 1 (error if 0)
- no unnamed components (error)
- no case-insensitive name collisions (error)
- orphan ratio < 0.5 (warning)
- component_count <= 100 (warning >50, error >100)
- dangling relationship references (warning)
- input size guard (truncation for >10k words)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class HealthReport:
    """Result of a blueprint health check."""
    healthy: bool
    score: float               # 0.0-1.0
    errors: Tuple[str, ...]    # blocking issues
    warnings: Tuple[str, ...]  # non-blocking issues
    stats: Dict[str, Any]


def check_blueprint_health(blueprint: dict) -> HealthReport:
    """
    Check blueprint structural health before materialization.

    Args:
        blueprint: Compiled blueprint dict

    Returns:
        HealthReport with errors, warnings, score
    """
    errors = []
    warnings = []
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    component_count = len(components)
    stats = {"component_count": component_count, "relationship_count": len(relationships)}

    # Check 1: At least 1 component
    if component_count == 0:
        errors.append("Blueprint has 0 components")

    # Check 2: No unnamed components
    unnamed_count = 0
    for comp in components:
        name = comp.get("name", "")
        if not name or not name.strip():
            unnamed_count += 1
    if unnamed_count > 0:
        errors.append(f"{unnamed_count} component(s) have no name")
    stats["unnamed_count"] = unnamed_count

    # Check 3: No case-insensitive name collisions
    name_lower_map: Dict[str, str] = {}  # lowercase -> original
    collisions = []
    for comp in components:
        name = comp.get("name", "")
        if not name:
            continue
        key = name.lower().replace(" ", "").replace("_", "")
        if key in name_lower_map:
            collisions.append((name_lower_map[key], name))
        else:
            name_lower_map[key] = name
    if collisions:
        for a, b in collisions:
            errors.append(f"Name collision (case-insensitive): '{a}' vs '{b}'")
    stats["collision_count"] = len(collisions)

    # Check 4: Orphan ratio
    component_names = {comp.get("name", "") for comp in components if comp.get("name")}
    connected = set()
    dangling_refs = []
    for rel in relationships:
        from_c = rel.get("from", "")
        to_c = rel.get("to", "")
        if from_c:
            connected.add(from_c)
        if to_c:
            connected.add(to_c)
        # Check 6: Dangling relationship references
        if from_c and from_c not in component_names:
            dangling_refs.append(from_c)
        if to_c and to_c not in component_names:
            dangling_refs.append(to_c)

    orphan_count = len(component_names - connected) if component_names else 0
    orphan_ratio = orphan_count / component_count if component_count > 0 else 0.0
    stats["orphan_count"] = orphan_count
    stats["orphan_ratio"] = orphan_ratio

    if orphan_ratio >= 0.5 and component_count > 1:
        warnings.append(
            f"High orphan ratio: {orphan_count}/{component_count} components "
            f"have no relationships ({orphan_ratio:.0%})"
        )

    # Check 5: Component count bounds
    if component_count > 100:
        errors.append(f"Too many components: {component_count} (max 100)")
    elif component_count > 50:
        warnings.append(f"Large blueprint: {component_count} components (consider splitting)")
    stats["over_limit"] = component_count > 100

    # Check 6: Dangling references
    if dangling_refs:
        unique_dangling = sorted(set(dangling_refs))
        warnings.append(
            f"Dangling relationship references: {', '.join(unique_dangling[:5])}"
            + (f" (+{len(unique_dangling) - 5} more)" if len(unique_dangling) > 5 else "")
        )
    stats["dangling_ref_count"] = len(set(dangling_refs))

    # Compute score
    score = 1.0
    if errors:
        score -= 0.3 * len(errors)
    if warnings:
        score -= 0.1 * len(warnings)
    score = max(0.0, min(1.0, score))

    healthy = len(errors) == 0

    return HealthReport(
        healthy=healthy,
        score=round(score, 2),
        errors=tuple(errors),
        warnings=tuple(warnings),
        stats=stats,
    )


def check_input_size(text: str, max_words: int = 10000) -> Tuple[bool, str]:
    """
    Guard against very large inputs that cause token explosion.

    Args:
        text: Input text
        max_words: Maximum word count (default 10000)

    Returns:
        (ok, text_or_truncated): ok=True if within limit, text may be truncated
    """
    if not text:
        return True, text

    words = text.split()
    if len(words) <= max_words:
        return True, text

    # Truncate to max_words
    truncated = " ".join(words[:max_words])
    return False, truncated
