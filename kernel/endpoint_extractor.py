"""
kernel/endpoint_extractor.py — Extract unexplored territory from a grid as intent chains.

LEAF module. No imports from core/ or mother/.

When a compilation finishes with the grid only partially explored (~50% fill),
valuable territory remains undiscovered. This module extracts that territory
as structured intents for chaining — feeding output endpoints back as input
for deeper exploration passes.

Three chain types:
  frontier    — unfilled connections from filled cells (the "next tokens")
  low_conf    — filled cells with confidence < threshold (need deepening)
  isolated    — active layers with no cross-layer connections (island territory)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from kernel.cell import Cell, FillState, parse_postcode, LAYERS
from kernel.grid import Grid


# ---------------------------------------------------------------------------
# Chain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EndpointChain:
    """An unexplored endpoint that can be chained as a new intent."""
    chain_type: str         # "frontier", "low_conf", "isolated"
    intent_text: str        # Actionable intent string for next compile pass
    source_postcodes: tuple[str, ...]  # Where this chain was extracted from
    priority: float         # Higher = more important (0.0 - 1.0)
    layer: str              # Primary layer this chain targets
    concern: str            # Primary concern axis


# Layer descriptions for readable intent generation
_LAYER_NAMES = {
    "INT": "Intent", "SEM": "Semantic", "ORG": "Organization",
    "COG": "Cognitive", "AGN": "Agency", "STR": "Structure",
    "STA": "State", "IDN": "Identity", "TME": "Time",
    "EXC": "Execution", "CTR": "Control", "RES": "Resource",
    "OBS": "Observability", "NET": "Network", "EMG": "Emergence",
    "MET": "Meta", "DAT": "Data", "SFX": "Side Effects",
}

_CONCERN_NAMES = {
    "SEM": "Semantic", "ENT": "Entity", "BHV": "Behavior",
    "FNC": "Function", "REL": "Relation", "PLN": "Plan",
    "MEM": "Memory", "ORC": "Orchestration", "AGT": "Agent",
    "ACT": "Actor", "SCO": "Scope", "STA": "State",
    "TRN": "Transition", "SNP": "Snapshot", "VRS": "Version",
    "SCH": "Schedule", "GTE": "Gate", "PLY": "Policy",
    "MET": "Metric", "LOG": "Log", "LMT": "Limit",
    "FLW": "Flow", "CND": "Candidate", "INT": "Integrity",
    "PRV": "Provenance", "CNS": "Constraint",
    "ENM": "Enumeration", "PRM": "Permission", "GOL": "Goal",
    "TMO": "Timeout", "LCK": "Lock", "RTY": "Retry",
    "TRF": "Transform", "COL": "Collection", "WRT": "Write",
    "EMT": "Emit", "RED": "Read", "ALT": "Alert",
    "CFG": "Config", "TRC": "Trace",
}


# Confidence threshold below which a filled cell is considered "shallow"
LOW_CONFIDENCE_THRESHOLD = 0.60

# Minimum chains to extract (don't bother with tiny grids)
MIN_CELLS_FOR_CHAINING = 3

# Maximum chains to return (cap exploration breadth)
MAX_CHAINS = 8


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_exploration_endpoints(
    grid: Grid,
    original_intent: str = "",
) -> list[EndpointChain]:
    """Extract unexplored territory from a grid as chainable intents.

    Scans three dimensions:
      1. Frontier — unfilled connections from filled cells
      2. Low confidence — filled cells that need deepening
      3. Isolated layers — active layers with no cross-layer connections

    Returns sorted by priority (highest first), capped at MAX_CHAINS.
    Only produces chains if the grid has >= MIN_CELLS_FOR_CHAINING cells.
    """
    if grid.total_cells < MIN_CELLS_FOR_CHAINING:
        return []

    chains: list[EndpointChain] = []

    chains.extend(_extract_frontier_chains(grid, original_intent))
    chains.extend(_extract_low_confidence_chains(grid, original_intent))
    chains.extend(_extract_isolated_layer_chains(grid, original_intent))

    # Deduplicate by intent_text (keep highest priority)
    seen: dict[str, EndpointChain] = {}
    for chain in chains:
        key = chain.intent_text
        if key not in seen or chain.priority > seen[key].priority:
            seen[key] = chain

    # Sort by priority descending, cap at MAX_CHAINS
    result = sorted(seen.values(), key=lambda c: c.priority, reverse=True)
    return result[:MAX_CHAINS]


def chain_summary(chains: list[EndpointChain]) -> str:
    """Format chains as a concise summary string."""
    if not chains:
        return "No exploration endpoints found."
    lines = [f"Found {len(chains)} exploration endpoint(s):"]
    for i, chain in enumerate(chains, 1):
        lines.append(
            f"  {i}. [{chain.chain_type}] {chain.intent_text} "
            f"(priority={chain.priority:.2f}, layer={chain.layer})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frontier chains — unfilled connections from filled cells
# ---------------------------------------------------------------------------

def _extract_frontier_chains(
    grid: Grid,
    original_intent: str,
) -> list[EndpointChain]:
    """Extract chains from unfilled connections (the 'next tokens')."""
    chains = []

    # Group unfilled connections by layer+concern
    frontier_groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for cell in grid.filled_cells():
        for conn_key in cell.connections:
            target = grid.get(conn_key)
            if target is None or target.is_empty or target.is_candidate:
                try:
                    pc = parse_postcode(conn_key)
                    frontier_groups[(pc.layer, pc.concern)].append(conn_key)
                except ValueError:
                    pass

    for (layer, concern), postcodes in frontier_groups.items():
        layer_name = _LAYER_NAMES.get(layer, layer)
        concern_name = _CONCERN_NAMES.get(concern, concern)

        # Priority scales with number of unfilled connections
        priority = min(1.0, len(postcodes) * 0.2)

        intent_prefix = f"Explore {layer_name} {concern_name}"
        if original_intent:
            intent = (
                f"{intent_prefix} aspects of: {original_intent}. "
                f"Focus on: {', '.join(postcodes[:3])} "
                f"({len(postcodes)} unexplored connection(s) in this territory)."
            )
        else:
            intent = (
                f"{intent_prefix}: "
                f"{len(postcodes)} unexplored connection(s) at "
                f"{', '.join(postcodes[:3])}."
            )

        chains.append(EndpointChain(
            chain_type="frontier",
            intent_text=intent,
            source_postcodes=tuple(postcodes),
            priority=priority,
            layer=layer,
            concern=concern,
        ))

    return chains


# ---------------------------------------------------------------------------
# Low-confidence chains — filled cells that need deepening
# ---------------------------------------------------------------------------

def _extract_low_confidence_chains(
    grid: Grid,
    original_intent: str,
) -> list[EndpointChain]:
    """Extract chains from filled cells with low confidence."""
    chains = []

    # Group low-confidence cells by layer+concern
    low_conf_groups: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)

    for cell in grid.filled_cells():
        if cell.confidence < LOW_CONFIDENCE_THRESHOLD:
            low_conf_groups[(cell.postcode.layer, cell.postcode.concern)].append(
                (cell.postcode.key, cell.confidence)
            )

    for (layer, concern), cells in low_conf_groups.items():
        layer_name = _LAYER_NAMES.get(layer, layer)
        concern_name = _CONCERN_NAMES.get(concern, concern)

        # Priority: lower confidence = higher priority
        avg_conf = sum(c for _, c in cells) / len(cells)
        priority = 1.0 - avg_conf

        postcodes = [pc for pc, _ in cells]
        conf_desc = ", ".join(f"{pc}@{c:.0%}" for pc, c in cells[:3])

        if original_intent:
            intent = (
                f"Deepen {layer_name} {concern_name} understanding of: {original_intent}. "
                f"Current confidence is low ({conf_desc}). "
                f"Provide more specific details about {concern_name.lower()} "
                f"in the {layer_name.lower()} layer."
            )
        else:
            intent = (
                f"Deepen {layer_name} {concern_name} analysis. "
                f"Current confidence is low: {conf_desc}."
            )

        chains.append(EndpointChain(
            chain_type="low_conf",
            intent_text=intent,
            source_postcodes=tuple(postcodes),
            priority=priority,
            layer=layer,
            concern=concern,
        ))

    return chains


# ---------------------------------------------------------------------------
# Isolated layer chains — active layers with no cross-layer connections
# ---------------------------------------------------------------------------

def _extract_isolated_layer_chains(
    grid: Grid,
    original_intent: str,
) -> list[EndpointChain]:
    """Extract chains from active layers that have no cross-layer connections."""
    chains = []

    # Find layers with filled cells
    layer_cells: dict[str, list[Cell]] = defaultdict(list)
    for cell in grid.filled_cells():
        layer_cells[cell.postcode.layer].append(cell)

    if len(layer_cells) < 2:
        # Need at least 2 layers to detect isolation
        return []

    # Check each layer for cross-layer connections
    for layer, cells in layer_cells.items():
        has_cross_layer = False
        for cell in cells:
            for conn_key in cell.connections:
                try:
                    target_pc = parse_postcode(conn_key)
                    if target_pc.layer != layer:
                        has_cross_layer = True
                        break
                except ValueError:
                    pass
            if has_cross_layer:
                break

        if not has_cross_layer and len(cells) >= 1:
            layer_name = _LAYER_NAMES.get(layer, layer)

            # Get concerns present in this layer
            concerns = {c.postcode.concern for c in cells}
            concern = sorted(concerns)[0] if concerns else "SEM"
            concern_name = _CONCERN_NAMES.get(concern, concern)

            postcodes = [c.postcode.key for c in cells]

            # Priority: isolated layers are medium priority
            priority = 0.5

            if original_intent:
                intent = (
                    f"Connect {layer_name} layer to other system layers for: {original_intent}. "
                    f"The {layer_name.lower()} territory ({len(cells)} cell(s)) is isolated — "
                    f"explore how it relates to other aspects of the system."
                )
            else:
                intent = (
                    f"Connect {layer_name} layer ({len(cells)} cell(s)) to other layers. "
                    f"Currently isolated with no cross-layer relationships."
                )

            chains.append(EndpointChain(
                chain_type="isolated",
                intent_text=intent,
                source_postcodes=tuple(postcodes),
                priority=priority,
                layer=layer,
                concern=concern,
            ))

    return chains
