"""
Motherlabs CLI Viewport — pure rendering functions for terminal visualization.

Phase C: CLI Viewport
Derived from: DIMENSIONAL_BLUEPRINT.md convergence — dimensional blueprints
and agent emissions need human-readable visualization, not just JSON.

All functions take data dicts (serialized forms), return formatted strings.
Zero I/O, zero side effects. Uses ANSI styles from cli/main.py.

This is a LEAF MODULE — imports only cli/main.py Style + stdlib.
"""

import math
from typing import Dict, List, Optional, Any, Tuple


# Import Style from cli.main — the only project dependency
from cli.main import Style, styled


# =============================================================================
# HELPERS (private)
# =============================================================================

def _bar(value: float, width: int = 10) -> str:
    """Render a progress bar: [####------]

    Args:
        value: 0.0 to 1.0 (clamped)
        width: total bar width (default 10)

    Returns:
        Formatted bar string like [####------]
    """
    value = max(0.0, min(1.0, value))
    filled = round(value * width)
    empty = width - filled
    return f"[{'#' * filled}{'-' * empty}]"


def _sparkline(values: list) -> str:
    """Render a sparkline from a list of 0-1 floats.

    Uses Unicode block characters for compact visualization.

    Args:
        values: list of floats (0.0 to 1.0)

    Returns:
        Sparkline string like "..#%@"
    """
    if not values:
        return ""
    blocks = " .:-=+*#%@"
    chars = []
    for v in values:
        v = max(0.0, min(1.0, v))
        idx = min(int(v * (len(blocks) - 1)), len(blocks) - 1)
        chars.append(blocks[idx])
    return "".join(chars)


def _scatter_2d(
    points: List[Tuple[float, float, str]],
    x_label: str = "x",
    y_label: str = "y",
    w: int = 40,
    h: int = 20,
) -> str:
    """Render a 2D ASCII scatter plot.

    Args:
        points: list of (x, y, label) where x,y are 0-1 floats
        x_label: axis label for x
        y_label: axis label for y
        w: plot width in characters
        h: plot height in characters

    Returns:
        Multi-line ASCII scatter plot
    """
    if not points:
        return "  (no data points)"

    # Build grid
    grid = [[' ' for _ in range(w)] for _ in range(h)]

    # Place points (numbered markers)
    legend_entries = []
    for idx, (x, y, label) in enumerate(points):
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        col = min(int(x * (w - 1)), w - 1)
        row = min(int((1.0 - y) * (h - 1)), h - 1)  # Invert y for display
        marker = str(idx + 1) if idx < 9 else chr(ord('A') + idx - 9)
        if idx >= 35:
            marker = '*'
        grid[row][col] = marker
        legend_entries.append((marker, label))

    # Render
    lines = []
    lines.append(f"  {y_label}")
    lines.append(f"  {'1.0':>3} {'.' * w}")
    for row_idx, row in enumerate(grid):
        if row_idx == 0 or row_idx == h - 1 or row_idx == h // 2:
            y_val = 1.0 - (row_idx / (h - 1)) if h > 1 else 0.5
            tick = f"{y_val:.1f}"
        else:
            tick = "   "
        lines.append(f"  {tick:>3} |{''.join(row)}|")
    lines.append(f"  {'0.0':>3} {'.' * w}")
    lines.append(f"      {'0.0':<{w // 2}}{'1.0':>{w // 2}}  {x_label}")

    # Legend
    lines.append("")
    lines.append("  Legend:")
    for marker, label in legend_entries:
        lines.append(f"    {marker} = {label}")

    return "\n".join(lines)


def _color_risk(risk: str) -> str:
    """Color a risk label based on severity.

    Args:
        risk: "high", "medium", or "low"

    Returns:
        ANSI-colored risk string
    """
    risk_lower = risk.lower() if risk else "unknown"
    if risk_lower == "high":
        return styled(risk, Style.RED, Style.BOLD)
    elif risk_lower == "medium":
        return styled(risk, Style.YELLOW)
    elif risk_lower == "low":
        return styled(risk, Style.GREEN)
    return styled(risk, Style.GRAY)


def _truncate(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis if too long.

    Args:
        text: input text
        max_len: maximum length (default 60)

    Returns:
        Truncated text with ... if needed
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# =============================================================================
# RENDER FUNCTIONS — Public API
# =============================================================================

def render_dimensional_summary(dim_meta: dict) -> str:
    """Render a summary of dimensional metadata.

    Shows axes with exploration depth bars, silence zones,
    and confidence trajectory sparkline.

    Args:
        dim_meta: Serialized DimensionalMetadata dict (from serialize_dimensional_metadata)

    Returns:
        Formatted multi-line string
    """
    if not dim_meta:
        return styled("  Dimensional metadata: not available", Style.DIM)

    lines = []
    lines.append(styled("  Dimensional Space", Style.CYAN, Style.BOLD))
    lines.append("")

    # Axes table
    axes = dim_meta.get("axes", [])
    if axes:
        # Header
        lines.append(styled("  Axis                    Range                              Depth", Style.WHITE))
        lines.append(styled("  " + "-" * 72, Style.DIM))

        for ax in axes:
            name = _truncate(ax.get("name", "?"), 22)
            range_low = _truncate(ax.get("range_low", "?"), 12)
            range_high = _truncate(ax.get("range_high", "?"), 12)
            depth = ax.get("exploration_depth", 0.0)
            bar = _bar(depth)

            range_str = f"{range_low} -> {range_high}"
            lines.append(
                f"  {name:<24}{range_str:<35}{bar} {depth:.0%}"
            )

            # Silence zones for this axis
            sz = ax.get("silence_zones", [])
            if sz:
                for zone in sz:
                    lines.append(styled(f"    silence: {_truncate(zone, 50)}", Style.YELLOW))
    else:
        lines.append(styled("  No dimensions extracted", Style.DIM))

    # Global silence zones
    global_sz = dim_meta.get("silence_zones", [])
    if global_sz:
        lines.append("")
        lines.append(styled("  Silence Zones (global)", Style.YELLOW))
        for zone in global_sz:
            lines.append(styled(f"    - {_truncate(zone, 60)}", Style.DIM))

    # Confidence trajectory
    trajectory = dim_meta.get("confidence_trajectory", [])
    if trajectory:
        lines.append("")
        spark = _sparkline(trajectory)
        final = trajectory[-1] if trajectory else 0.0
        lines.append(
            f"  Confidence: {styled(spark, Style.CYAN)} "
            f"{styled(f'{final:.0%}', Style.WHITE, Style.BOLD)}"
        )

    # Dialogue depth
    depth = dim_meta.get("dialogue_depth", 0)
    if depth:
        lines.append(f"  Dialogue depth: {styled(str(depth), Style.WHITE)} turns")

    return "\n".join(lines)


def render_node_map(
    dim_meta: dict,
    x_axis: Optional[str] = None,
    y_axis: Optional[str] = None,
) -> str:
    """Render a 2D ASCII scatter plot of node positions.

    Projects N-dimensional positions onto 2 chosen axes.
    Defaults to first two dimensions if not specified.

    Args:
        dim_meta: Serialized DimensionalMetadata dict
        x_axis: Name of axis for X (default: first dimension)
        y_axis: Name of axis for Y (default: second dimension)

    Returns:
        Formatted ASCII scatter plot
    """
    if not dim_meta:
        return styled("  Node map: not available", Style.DIM)

    axes = dim_meta.get("axes", [])
    positions = dim_meta.get("node_positions", {})

    if not axes or not positions:
        return styled("  Node map: insufficient data (need axes + positions)", Style.DIM)

    # Default axes
    axis_names = [a["name"] for a in axes]
    if x_axis is None:
        x_axis = axis_names[0] if axis_names else "x"
    if y_axis is None:
        y_axis = axis_names[1] if len(axis_names) > 1 else axis_names[0] if axis_names else "y"

    # Build points
    points = []
    for name, pos_data in positions.items():
        dim_values = pos_data.get("dimension_values", {})
        x = dim_values.get(x_axis, 0.0)
        y = dim_values.get(y_axis, 0.0)
        points.append((x, y, name))

    lines = []
    lines.append(styled("  Node Map", Style.CYAN, Style.BOLD))
    lines.append(styled(f"  Projection: {x_axis} x {y_axis}", Style.DIM))
    lines.append("")
    lines.append(_scatter_2d(points, x_label=x_axis, y_label=y_axis))

    return "\n".join(lines)


def render_interface_contracts(imap: dict) -> str:
    """Render interface contracts in a readable format.

    Shows per-contract: node_a <-> node_b, type, direction, fragility.
    Lists data flows under each contract.

    Args:
        imap: Serialized InterfaceMap dict (from serialize_interface_map)

    Returns:
        Formatted multi-line string
    """
    if not imap:
        return styled("  Interface contracts: not available", Style.DIM)

    lines = []
    lines.append(styled("  Interface Contracts", Style.CYAN, Style.BOLD))
    lines.append("")

    contracts = imap.get("contracts", [])
    if not contracts:
        lines.append(styled("  No contracts extracted", Style.DIM))
        return "\n".join(lines)

    for i, c in enumerate(contracts):
        node_a = c.get("node_a", "?")
        node_b = c.get("node_b", "?")
        rel_type = c.get("relationship_type", "?")
        direction = c.get("directionality", "mutual")
        fragility = c.get("fragility", 0.0)
        confidence = c.get("confidence", 0.0)

        # Direction arrow
        if direction == "A_depends_on_B":
            arrow = "<-"
        elif direction == "B_depends_on_A":
            arrow = "->"
        else:
            arrow = "<>"

        frag_bar = _bar(fragility)
        frag_color = Style.RED if fragility > 0.7 else Style.YELLOW if fragility > 0.3 else Style.GREEN

        lines.append(
            f"  {styled(node_a, Style.WHITE)} {arrow} {styled(node_b, Style.WHITE)}"
            f"  {styled(rel_type, Style.DIM)}"
            f"  fragility: {styled(frag_bar, frag_color)}"
        )

        # Data flows
        flows = c.get("data_flows", [])
        for flow in flows:
            fname = flow.get("name", "?")
            ftype = flow.get("type_hint", "?")
            fdir = flow.get("direction", "?")
            lines.append(styled(f"    {fname}: {ftype} ({fdir})", Style.DIM))

        if i < len(contracts) - 1:
            lines.append("")

    # Unmatched relationships
    unmatched = imap.get("unmatched_relationships", [])
    if unmatched:
        lines.append("")
        lines.append(styled("  Unmatched Relationships", Style.YELLOW))
        for u in unmatched:
            lines.append(styled(f"    - {u}", Style.DIM))

    # Overall confidence
    ext_conf = imap.get("extraction_confidence", 0.0)
    lines.append("")
    lines.append(
        f"  Extraction confidence: {styled(f'{ext_conf:.0%}', Style.WHITE, Style.BOLD)}"
    )

    return "\n".join(lines)


def render_fragile_edges(dim_meta: dict) -> str:
    """Render fragile edges with risk indicators.

    Shows affected nodes, drift risk (colored), and reasoning.

    Args:
        dim_meta: Serialized DimensionalMetadata dict

    Returns:
        Formatted multi-line string
    """
    if not dim_meta:
        return styled("  Fragile edges: not available", Style.DIM)

    edges = dim_meta.get("fragile_edges", [])
    if not edges:
        return styled("  No fragile edges detected", Style.GREEN)

    lines = []
    lines.append(styled("  Fragile Edges", Style.CYAN, Style.BOLD))
    lines.append("")

    for edge in edges:
        desc = edge.get("description", "?")
        nodes = edge.get("affected_nodes", [])
        risk = edge.get("drift_risk", "unknown")
        reasoning = edge.get("reasoning", "")

        nodes_str = ", ".join(nodes) if nodes else "?"
        lines.append(f"  {styled(desc, Style.WHITE)}")
        lines.append(f"    nodes: {styled(nodes_str, Style.DIM)}  risk: {_color_risk(risk)}")
        if reasoning:
            lines.append(f"    reason: {styled(_truncate(reasoning, 60), Style.DIM)}")
        lines.append("")

    return "\n".join(lines)


def render_emission_result(emission: dict) -> str:
    """Render agent emission results.

    Shows batch timeline with per-node status, verification summary.

    Args:
        emission: Serialized EmissionResult dict (from serialize_emission_result)

    Returns:
        Formatted multi-line string
    """
    if not emission:
        return styled("  Emission results: not available", Style.DIM)

    lines = []
    lines.append(styled("  Agent Emission", Style.CYAN, Style.BOLD))
    lines.append("")

    # Batch timeline
    batches = emission.get("batch_emissions", [])
    for batch in batches:
        batch_idx = batch.get("batch_index", 0)
        emissions = batch.get("emissions", [])
        success_count = batch.get("success_count", 0)
        failure_count = batch.get("failure_count", 0)

        total = success_count + failure_count
        lines.append(
            styled(f"  Batch {batch_idx}", Style.WHITE, Style.BOLD)
            + styled(f"  ({success_count}/{total} succeeded)", Style.DIM)
        )

        for ne in emissions:
            name = ne.get("component_name", "?")
            comp_type = ne.get("component_type", "?")
            success = ne.get("success", False)
            code = ne.get("code", "")
            prompt_hash = ne.get("prompt_hash", "?")[:8]
            error = ne.get("error")

            status = styled("ok", Style.GREEN) if success else styled("FAIL", Style.RED)
            code_size = f"{len(code)} chars" if code else "no code"

            lines.append(
                f"    {status}  {styled(name, Style.WHITE)} ({comp_type})"
                f"  {styled(code_size, Style.DIM)}  #{prompt_hash}"
            )
            if error:
                lines.append(styled(f"         error: {_truncate(error, 50)}", Style.RED))

        lines.append("")

    # Verification summary
    report = emission.get("verification_report", {})
    total_contracts = report.get("total_contracts", 0)
    passed = report.get("passed", 0)
    pass_rate = emission.get("pass_rate", 0.0)

    rate_bar = _bar(pass_rate)
    rate_color = Style.GREEN if pass_rate >= 0.8 else Style.YELLOW if pass_rate >= 0.5 else Style.RED

    lines.append(styled("  Verification", Style.WHITE, Style.BOLD))
    lines.append(
        f"    Contracts: {passed}/{total_contracts} passed"
        f"  {styled(rate_bar, rate_color)} {pass_rate:.0%}"
    )

    # Total
    total_nodes = emission.get("total_nodes", 0)
    success_total = emission.get("success_count", 0)
    lines.append("")
    lines.append(
        f"  Total: {styled(str(success_total), Style.GREEN)}/{total_nodes} succeeded"
        f"  pass rate: {styled(f'{pass_rate:.0%}', Style.WHITE, Style.BOLD)}"
    )

    return "\n".join(lines)


def render_blueprint_tree(blueprint: dict) -> str:
    """Render a blueprint as a component tree.

    Shows components with type and description, relationships, and constraints.

    Args:
        blueprint: Blueprint dict with components, relationships, constraints

    Returns:
        Formatted multi-line string
    """
    if not blueprint:
        return styled("  Blueprint: not available", Style.DIM)

    lines = []
    lines.append(styled("  Blueprint", Style.CYAN, Style.BOLD))
    lines.append("")

    # Components
    components = blueprint.get("components", [])
    if components:
        lines.append(styled("  Components", Style.WHITE))
        for i, comp in enumerate(components):
            name = comp.get("name", "?")
            ctype = comp.get("type", "?")
            desc = _truncate(comp.get("description", ""), 50)
            last = i == len(components) - 1
            prefix = "  +--" if last else "  |--"
            lines.append(
                f"{prefix} {styled(name, Style.WHITE)} ({styled(ctype, Style.DIM)})"
            )
            if desc:
                connector = "     " if last else "  |  "
                lines.append(f"{connector} {styled(desc, Style.DIM)}")

    # Relationships
    relationships = blueprint.get("relationships", [])
    if relationships:
        lines.append("")
        lines.append(styled("  Relationships", Style.WHITE))
        for rel in relationships:
            src = rel.get("from", "?")
            dst = rel.get("to", "?")
            rtype = rel.get("type", "related")
            lines.append(
                f"    {styled(src, Style.WHITE)} -> {styled(dst, Style.WHITE)}"
                f"  ({styled(rtype, Style.DIM)})"
            )

    # Constraints
    constraints = blueprint.get("constraints", [])
    if constraints:
        lines.append("")
        lines.append(styled("  Constraints", Style.WHITE))
        for con in constraints:
            desc = con.get("description", str(con)) if isinstance(con, dict) else str(con)
            lines.append(f"    - {styled(_truncate(desc, 60), Style.DIM)}")

    return "\n".join(lines)


def render_compilation_overview(
    blueprint: dict,
    dim_meta: Optional[dict] = None,
    imap: Optional[dict] = None,
    emission: Optional[dict] = None,
) -> str:
    """Render a single-screen compilation overview.

    Compact summary with component count, relationship count, dimension count,
    health bars for exploration depth, fragility, and verification.

    Args:
        blueprint: Blueprint dict
        dim_meta: Optional serialized DimensionalMetadata
        imap: Optional serialized InterfaceMap
        emission: Optional serialized EmissionResult

    Returns:
        Formatted multi-line string
    """
    if not blueprint:
        return styled("  Compilation overview: not available", Style.DIM)

    lines = []
    lines.append(styled("  Compilation Overview", Style.CYAN, Style.BOLD))
    lines.append("")

    # Blueprint stats
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])
    constraints = blueprint.get("constraints", [])

    lines.append(
        f"  Components: {styled(str(len(components)), Style.WHITE, Style.BOLD)}"
        f"    Relationships: {styled(str(len(relationships)), Style.WHITE, Style.BOLD)}"
        f"    Constraints: {styled(str(len(constraints)), Style.WHITE, Style.BOLD)}"
    )

    # Dimensional stats
    if dim_meta:
        axes = dim_meta.get("axes", [])
        positions = dim_meta.get("node_positions", {})
        fragile = dim_meta.get("fragile_edges", [])

        # Average exploration depth
        depths = [a.get("exploration_depth", 0.0) for a in axes]
        avg_depth = sum(depths) / len(depths) if depths else 0.0

        # Fragility: proportion of edges that are fragile
        n_fragile = len(fragile)

        lines.append(
            f"  Dimensions: {styled(str(len(axes)), Style.WHITE, Style.BOLD)}"
            f"    Positions: {styled(str(len(positions)), Style.WHITE, Style.BOLD)}"
            f"    Fragile edges: {styled(str(n_fragile), Style.YELLOW if n_fragile > 0 else Style.GREEN)}"
        )
        lines.append(
            f"  Exploration: {_bar(avg_depth)} {avg_depth:.0%}"
        )
    else:
        lines.append(styled("  Dimensions: not available", Style.DIM))

    # Interface stats
    if imap:
        contracts = imap.get("contracts", [])
        ext_conf = imap.get("extraction_confidence", 0.0)
        unmatched = imap.get("unmatched_relationships", [])

        # Average fragility
        frags = [c.get("fragility", 0.0) for c in contracts]
        avg_frag = sum(frags) / len(frags) if frags else 0.0
        frag_color = Style.RED if avg_frag > 0.7 else Style.YELLOW if avg_frag > 0.3 else Style.GREEN

        lines.append(
            f"  Contracts: {styled(str(len(contracts)), Style.WHITE, Style.BOLD)}"
            f"    Unmatched: {styled(str(len(unmatched)), Style.YELLOW if unmatched else Style.GREEN)}"
            f"    Confidence: {ext_conf:.0%}"
        )
        lines.append(
            f"  Fragility: {styled(_bar(avg_frag), frag_color)} {avg_frag:.0%}"
        )
    else:
        lines.append(styled("  Interfaces: not available", Style.DIM))

    # Emission stats
    if emission:
        total = emission.get("total_nodes", 0)
        success = emission.get("success_count", 0)
        pass_rate = emission.get("pass_rate", 0.0)
        rate_color = Style.GREEN if pass_rate >= 0.8 else Style.YELLOW if pass_rate >= 0.5 else Style.RED

        lines.append(
            f"  Emission: {styled(str(success), Style.GREEN)}/{total} nodes"
            f"    Verification: {styled(_bar(pass_rate), rate_color)} {pass_rate:.0%}"
        )
    else:
        lines.append(styled("  Emission: not run", Style.DIM))

    return "\n".join(lines)
