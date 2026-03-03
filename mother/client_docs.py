"""
Client-facing document generation — transform blueprints into external docs.

LEAF module (stdlib only). Generates polished client-facing documentation
from compilation results. Strips internal metadata, formats for
non-technical stakeholders.

Genome #127: client-facing-capable — generates documents for external parties.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ClientBrief:
    """A client-facing document generated from a blueprint."""

    title: str
    executive_summary: str
    components: List[Dict[str, str]]     # [{name, type, description}]
    relationships: List[str]              # human-readable relationship lines
    acceptance_criteria: List[str]
    constraints: List[str]
    trust_score: float
    dimension_highlights: List[str]       # strengths and weaknesses
    generated_at: str = ""


def generate_client_brief(
    blueprint: Dict[str, Any],
    verification: Optional[Dict[str, Any]] = None,
    project_name: str = "",
) -> ClientBrief:
    """Generate a client-facing brief from a blueprint.

    Extracts components, relationships, constraints, and verification
    results. Formats everything for a non-technical audience.
    """
    components = _extract_components(blueprint)
    relationships = _extract_relationships(blueprint)
    constraints = _extract_constraints(blueprint)
    criteria = _extract_acceptance_criteria(blueprint)

    # Verification highlights
    trust_score = 0.0
    highlights: List[str] = []
    if verification:
        trust_score = verification.get("overall_score", 0.0)
        highlights = _extract_dimension_highlights(verification)

    # Executive summary
    comp_count = len(components)
    rel_count = len(relationships)
    title = project_name or _infer_title(blueprint)
    summary = _build_executive_summary(
        title, comp_count, rel_count, trust_score, constraints,
    )

    import time
    return ClientBrief(
        title=title,
        executive_summary=summary,
        components=components,
        relationships=relationships,
        acceptance_criteria=criteria,
        constraints=constraints,
        trust_score=trust_score,
        dimension_highlights=highlights,
        generated_at=time.strftime("%Y-%m-%d %H:%M"),
    )


def format_client_markdown(brief: ClientBrief) -> str:
    """Format a ClientBrief as a polished Markdown document."""
    lines = [f"# {brief.title}\n"]

    if brief.generated_at:
        lines.append(f"*Generated: {brief.generated_at}*\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(brief.executive_summary + "\n")

    # Quality score
    if brief.trust_score > 0:
        lines.append("## Quality Assessment\n")
        lines.append(f"**Overall confidence:** {brief.trust_score:.0f}%\n")
        if brief.dimension_highlights:
            for h in brief.dimension_highlights:
                lines.append(f"- {h}")
            lines.append("")

    # Components
    if brief.components:
        lines.append("## System Components\n")
        lines.append("| Component | Type | Description |")
        lines.append("|-----------|------|-------------|")
        for comp in brief.components:
            name = comp.get("name", "")
            ctype = comp.get("type", "")
            desc = comp.get("description", "").replace("\n", " ")
            # Truncate long descriptions for table
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"| {name} | {ctype} | {desc} |")
        lines.append("")

    # Relationships
    if brief.relationships:
        lines.append("## Component Interactions\n")
        for rel in brief.relationships:
            lines.append(f"- {rel}")
        lines.append("")

    # Constraints
    if brief.constraints:
        lines.append("## Design Constraints\n")
        for c in brief.constraints:
            lines.append(f"- {c}")
        lines.append("")

    # Acceptance criteria
    if brief.acceptance_criteria:
        lines.append("## Acceptance Criteria\n")
        for criterion in brief.acceptance_criteria:
            lines.append(f"- [ ] {criterion}")
        lines.append("")

    return "\n".join(lines)


# --- Internal helpers ---

def _extract_components(blueprint: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract component info, stripping internal fields."""
    components: List[Dict[str, str]] = []
    for comp in blueprint.get("components", []):
        name = comp.get("name", "Unknown")
        ctype = comp.get("type", "component")
        desc = comp.get("description", "")
        if not desc:
            # Fallback: use derived_from as description
            desc = comp.get("derived_from", "")
        components.append({
            "name": name,
            "type": str(ctype),
            "description": desc,
        })
    return components


def _extract_relationships(blueprint: Dict[str, Any]) -> List[str]:
    """Convert relationships to human-readable strings."""
    lines: List[str] = []
    for rel in blueprint.get("relationships", []):
        from_comp = rel.get("from", "?")
        to_comp = rel.get("to", "?")
        rel_type = rel.get("type", "relates to")
        desc = rel.get("description", "")
        line = f"{from_comp} {rel_type} {to_comp}"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return lines


def _extract_constraints(blueprint: Dict[str, Any]) -> List[str]:
    """Extract constraint descriptions."""
    constraints: List[str] = []
    for c in blueprint.get("constraints", []):
        desc = c.get("description", "")
        if desc:
            constraints.append(desc)
    return constraints


def _extract_acceptance_criteria(blueprint: Dict[str, Any]) -> List[str]:
    """Extract validation rules as acceptance criteria."""
    criteria: List[str] = []
    for comp in blueprint.get("components", []):
        rules = comp.get("validation_rules", [])
        if isinstance(rules, list):
            for rule in rules:
                if isinstance(rule, str) and rule:
                    criteria.append(rule)
    return criteria


def _infer_title(blueprint: Dict[str, Any]) -> str:
    """Infer a project title from blueprint content."""
    # Use the first subsystem or largest component name
    components = blueprint.get("components", [])
    if not components:
        return "System Specification"

    # Prefer subsystem components
    for comp in components:
        if comp.get("type") in ("subsystem", "SUBSYSTEM"):
            return comp.get("name", "System") + " — Specification"

    # Fall back to first component
    return components[0].get("name", "System") + " — Specification"


def _build_executive_summary(
    title: str,
    comp_count: int,
    rel_count: int,
    trust_score: float,
    constraints: List[str],
) -> str:
    """Build a 2-3 sentence executive summary."""
    parts: List[str] = []

    parts.append(
        f"This specification defines a system with "
        f"{comp_count} component{'s' if comp_count != 1 else ''} "
        f"and {rel_count} interaction{'s' if rel_count != 1 else ''}."
    )

    if constraints:
        parts.append(
            f"The design is bounded by {len(constraints)} "
            f"constraint{'s' if len(constraints) != 1 else ''}."
        )

    if trust_score >= 80:
        parts.append("The specification has been validated with high confidence.")
    elif trust_score >= 60:
        parts.append("The specification has been validated with moderate confidence.")
    elif trust_score > 0:
        parts.append("The specification requires further refinement before implementation.")

    return " ".join(parts)


_DIMENSION_LABELS = {
    "completeness": "Coverage",
    "consistency": "Internal consistency",
    "coherence": "Structural coherence",
    "traceability": "Requirement traceability",
    "actionability": "Implementation readiness",
    "specificity": "Detail level",
    "codegen_readiness": "Code generation readiness",
}


def _extract_dimension_highlights(
    verification: Dict[str, Any],
    strong_threshold: float = 80.0,
    weak_threshold: float = 60.0,
) -> List[str]:
    """Extract strengths and weaknesses from verification dimensions."""
    highlights: List[str] = []

    for dim, label in _DIMENSION_LABELS.items():
        v = verification.get(dim)
        score = None
        if isinstance(v, (int, float)):
            score = float(v)
        elif isinstance(v, dict) and "score" in v:
            score = float(v["score"])

        if score is None:
            continue

        if score >= strong_threshold:
            highlights.append(f"**Strong:** {label} ({score:.0f}%)")
        elif score < weak_threshold:
            highlights.append(f"**Needs work:** {label} ({score:.0f}%)")

    return highlights
