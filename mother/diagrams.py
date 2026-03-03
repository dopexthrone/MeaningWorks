"""
Diagram generation — Mermaid flowcharts from blueprints.

LEAF module (stdlib only). Converts blueprint component/relationship
structures into Mermaid diagram syntax (plain text, no renderer needed).

Genome #143: visual-first — generates diagrams from blueprint data.
"""

import re
from typing import Any, Dict, List, Optional


# Max components before falling back to simplified view
_MAX_COMPONENTS = 25
_MAX_RELATIONSHIPS = 40


def blueprint_to_mermaid(
    blueprint: Dict[str, Any],
    direction: str = "TD",
    title: str = "",
) -> str:
    """Convert a blueprint to a Mermaid flowchart diagram.

    Args:
        blueprint: Blueprint dict with components and relationships.
        direction: Flow direction — "TD" (top-down), "LR" (left-right).
        title: Optional diagram title.

    Returns:
        Mermaid diagram syntax as a string.
    """
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    if not components:
        return ""

    # Complexity check
    if len(components) > _MAX_COMPONENTS:
        return _simplified_diagram(components, relationships, direction, title)

    lines: List[str] = []

    # Header
    if title:
        lines.append(f"---")
        lines.append(f"title: {title}")
        lines.append(f"---")
    lines.append(f"flowchart {direction}")

    # Group components by subsystem
    top_level: List[Dict] = []
    subsystems: List[Dict] = []

    for comp in components:
        if comp.get("type") in ("subsystem", "SUBSYSTEM") and comp.get("sub_blueprint"):
            subsystems.append(comp)
        else:
            top_level.append(comp)

    # Render subsystems as subgraphs
    for sub in subsystems:
        sub_name = sub.get("name", "Subsystem")
        sub_id = _safe_id(sub_name)
        lines.append(f"    subgraph {sub_id}[\"{sub_name}\"]")

        sub_bp = sub.get("sub_blueprint", {})
        for child in sub_bp.get("components", []):
            child_id = _safe_id(child.get("name", ""))
            child_label = _node_label(child)
            lines.append(f"        {child_id}[\"{child_label}\"]")

        lines.append(f"    end")

    # Render top-level components
    for comp in top_level:
        comp_id = _safe_id(comp.get("name", ""))
        label = _node_label(comp)
        shape = _node_shape(comp)
        lines.append(f"    {comp_id}{shape[0]}\"{label}\"{shape[1]}")

    # Render relationships
    rendered_rels = set()
    for rel in relationships[:_MAX_RELATIONSHIPS]:
        from_comp = rel.get("from", "")
        to_comp = rel.get("to", "")
        rel_type = rel.get("type", "")

        if not from_comp or not to_comp:
            continue

        from_id = _safe_id(from_comp)
        to_id = _safe_id(to_comp)
        rel_key = (from_id, to_id, rel_type)
        if rel_key in rendered_rels:
            continue
        rendered_rels.add(rel_key)

        arrow = _arrow_style(rel_type)
        if rel_type:
            lines.append(f"    {from_id} {arrow}|{rel_type}| {to_id}")
        else:
            lines.append(f"    {from_id} {arrow} {to_id}")

    # Styling by component type
    style_groups = _group_by_type(components)
    for ctype, ids in style_groups.items():
        style = _type_style(ctype)
        if style and ids:
            id_list = ",".join(ids)
            lines.append(f"    style {id_list} {style}")

    return "\n".join(lines)


def blueprint_to_mermaid_block(
    blueprint: Dict[str, Any],
    direction: str = "TD",
    title: str = "",
) -> str:
    """Wrap Mermaid output in a Markdown code block."""
    diagram = blueprint_to_mermaid(blueprint, direction, title)
    if not diagram:
        return ""
    return f"```mermaid\n{diagram}\n```"


def component_tree(blueprint: Dict[str, Any], indent: int = 0) -> str:
    """Generate a text-based component tree as fallback for complex blueprints.

    Returns an indented tree view showing containment hierarchy.
    """
    lines: List[str] = []
    prefix = "  " * indent

    for comp in blueprint.get("components", []):
        name = comp.get("name", "?")
        ctype = comp.get("type", "")
        type_tag = f" ({ctype})" if ctype else ""
        lines.append(f"{prefix}- {name}{type_tag}")

        sub_bp = comp.get("sub_blueprint")
        if sub_bp and isinstance(sub_bp, dict):
            lines.append(component_tree(sub_bp, indent + 1))

    return "\n".join(lines)


# --- Internal helpers ---

def _safe_id(name: str) -> str:
    """Convert a component name to a safe Mermaid node ID."""
    # Replace non-alphanumeric with underscore
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name.strip())
    safe = re.sub(r'_+', '_', safe).strip('_')
    if not safe:
        safe = "node"
    # Ensure doesn't start with digit
    if safe[0].isdigit():
        safe = "n_" + safe
    return safe


def _node_label(comp: Dict[str, Any]) -> str:
    """Generate a node label from component data."""
    name = comp.get("name", "?")
    ctype = comp.get("type", "")
    if ctype and ctype not in ("component", ""):
        return f"{name}<br/><small>{ctype}</small>"
    return name


def _node_shape(comp: Dict[str, Any]) -> tuple:
    """Return Mermaid shape delimiters based on component type."""
    ctype = str(comp.get("type", "")).lower()
    shapes = {
        "entity": ("[", "]"),         # rectangle
        "process": ("([", "])"),      # stadium
        "service": ("([", "])"),      # stadium
        "event": (">", "]"),          # asymmetric
        "constraint": ("{{", "}}"),   # hexagon
        "interface": ("[[", "]]"),    # subroutine
    }
    return shapes.get(ctype, ("[", "]"))


def _arrow_style(rel_type: str) -> str:
    """Return Mermaid arrow style based on relationship type."""
    rel_lower = rel_type.lower() if rel_type else ""
    if rel_lower in ("depends_on", "requires"):
        return "-.->"
    if rel_lower in ("triggers", "emits", "sends"):
        return "==>"
    if rel_lower in ("contains", "owns", "has"):
        return "-->"
    return "-->"


def _type_style(ctype: str) -> str:
    """Return Mermaid style for a component type."""
    styles = {
        "entity": "fill:#e1f5fe",
        "process": "fill:#f3e5f5",
        "service": "fill:#e8f5e9",
        "event": "fill:#fff3e0",
        "constraint": "fill:#fce4ec",
    }
    return styles.get(ctype.lower(), "")


def _group_by_type(components: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Group component IDs by their type."""
    groups: Dict[str, List[str]] = {}
    for comp in components:
        ctype = str(comp.get("type", "component")).lower()
        comp_id = _safe_id(comp.get("name", ""))
        if ctype not in groups:
            groups[ctype] = []
        groups[ctype].append(comp_id)
    return groups


# =============================================================================
# Translation-native — Genome #148: multi-format blueprint translation
# =============================================================================

def blueprint_to_sequence_diagram(
    blueprint: Dict[str, Any],
    title: str = "",
) -> str:
    """Convert a blueprint to a Mermaid sequence diagram.

    Extracts relationships and maps from/to as participants,
    relationship types as message labels.

    Returns Mermaid sequenceDiagram syntax string.
    """
    relationships = blueprint.get("relationships", [])
    if not relationships:
        return ""

    lines: List[str] = []
    if title:
        lines.append(f"---")
        lines.append(f"title: {title}")
        lines.append(f"---")
    lines.append("sequenceDiagram")

    # Collect unique participants in order of appearance
    participants: List[str] = []
    seen: set = set()
    for rel in relationships[:_MAX_RELATIONSHIPS]:
        for entity in (rel.get("from", ""), rel.get("to", "")):
            if entity and entity not in seen:
                participants.append(entity)
                seen.add(entity)

    # Declare participants
    for p in participants:
        pid = _safe_id(p)
        lines.append(f"    participant {pid} as {p}")

    # Render messages
    for rel in relationships[:_MAX_RELATIONSHIPS]:
        from_comp = rel.get("from", "")
        to_comp = rel.get("to", "")
        if not from_comp or not to_comp:
            continue

        from_id = _safe_id(from_comp)
        to_id = _safe_id(to_comp)
        rel_type = rel.get("type", "")
        label = rel_type if rel_type else "sends"

        lines.append(f"    {from_id}->>+{to_id}: {label}")

    return "\n".join(lines)


def blueprint_to_wireframe(
    blueprint: Dict[str, Any],
    title: str = "",
) -> str:
    """Convert a blueprint to an ASCII wireframe.

    Renders components as ASCII boxes stacked vertically with connection lines.

    Returns ASCII string.
    """
    components = blueprint.get("components", [])
    if not components:
        return ""

    lines: List[str] = []
    if title:
        lines.append(f"=== {title} ===")
        lines.append("")

    for i, comp in enumerate(components):
        name = comp.get("name", "?")
        ctype = comp.get("type", "")
        label = f"{name} ({ctype})" if ctype else name

        box_width = max(len(label) + 4, 20)
        border = "+" + "-" * (box_width - 2) + "+"
        padding = box_width - 2 - len(label)
        left_pad = padding // 2
        right_pad = padding - left_pad
        content = "|" + " " * left_pad + label + " " * right_pad + "|"

        lines.append(border)
        lines.append(content)
        lines.append(border)

        # Connection line between components
        if i < len(components) - 1:
            center = box_width // 2
            lines.append(" " * center + "|")

    return "\n".join(lines)


def translate_blueprint(
    blueprint: Dict[str, Any],
    target_format: str = "flowchart",
    title: str = "",
) -> str:
    """Translate a blueprint to the specified output format.

    Dispatches to format-specific renderers:
    - "flowchart" → blueprint_to_mermaid
    - "sequence" → blueprint_to_sequence_diagram
    - "wireframe" → blueprint_to_wireframe
    - "tree" → component_tree
    - unknown → component_tree (fallback)

    Returns formatted string.
    """
    dispatchers = {
        "flowchart": lambda: blueprint_to_mermaid(blueprint, title=title),
        "sequence": lambda: blueprint_to_sequence_diagram(blueprint, title=title),
        "wireframe": lambda: blueprint_to_wireframe(blueprint, title=title),
        "tree": lambda: component_tree(blueprint),
    }

    renderer = dispatchers.get(target_format)
    if renderer:
        return renderer()

    # Unknown format — fallback to tree
    return component_tree(blueprint)


def _simplified_diagram(
    components: List[Dict],
    relationships: List[Dict],
    direction: str,
    title: str,
) -> str:
    """Simplified diagram for complex blueprints (>25 components).

    Groups by type, shows counts, only renders subsystem-level nodes.
    """
    lines: List[str] = []
    if title:
        lines.append(f"---")
        lines.append(f"title: {title} (simplified)")
        lines.append(f"---")
    lines.append(f"flowchart {direction}")

    # Count by type
    type_counts: Dict[str, int] = {}
    subsystems: List[str] = []
    for comp in components:
        ctype = str(comp.get("type", "component")).lower()
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        if ctype == "subsystem":
            subsystems.append(comp.get("name", "?"))

    # Render subsystems as nodes
    for name in subsystems:
        sid = _safe_id(name)
        lines.append(f"    {sid}[\"{name}\"]")

    # Summary node
    non_sub = {k: v for k, v in type_counts.items() if k != "subsystem"}
    if non_sub:
        summary_parts = [f"{count} {ctype}{'s' if count > 1 else ''}"
                        for ctype, count in sorted(non_sub.items())]
        summary = ", ".join(summary_parts)
        lines.append(f'    _summary["{summary}"]')

    # Render subsystem-level relationships only
    sub_names = set(s.lower() for s in subsystems)
    for rel in relationships[:_MAX_RELATIONSHIPS]:
        from_c = rel.get("from", "")
        to_c = rel.get("to", "")
        if from_c.lower() in sub_names or to_c.lower() in sub_names:
            from_id = _safe_id(from_c)
            to_id = _safe_id(to_c)
            rel_type = rel.get("type", "")
            if rel_type:
                lines.append(f"    {from_id} -->|{rel_type}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

    return "\n".join(lines)
