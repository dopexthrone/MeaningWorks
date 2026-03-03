"""
kernel/emission.py — Manifest export to Claude Code.

The emission module transforms a verified grid into an actionable
manifest that Claude Code can execute. Only F/P cells are emitted,
dependency-ordered (parents before children), with escalations
extracted from B/P cells.

Emission is simulation-gated: the governor must pass before export.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from kernel.cell import Cell, FillState, Postcode, parse_postcode, SCOPE_DEPTH
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.agents import governor, SimulationResult


# ---------------------------------------------------------------------------
# Escalation extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Escalation:
    """A structured question surfaced to the human."""
    id: str
    urgency: str        # "blocking" or "non-blocking"
    postcode: str
    primitive: str
    question: str
    context: str
    options: tuple[str, ...] = ()


def extract_escalations(grid: Grid) -> list[Escalation]:
    """Extract escalations from blocked and partial cells.

    Blocked (B) → blocking escalation (must resolve before emission)
    Partial (P) with confidence < 0.80 → non-blocking clarification
    """
    escalations: list[Escalation] = []
    esc_id = 0

    for cell in grid.cells.values():
        if cell.is_blocked:
            esc_id += 1
            escalations.append(Escalation(
                id=f"ESC-{esc_id}",
                urgency="blocking",
                postcode=cell.postcode.key,
                primitive=cell.primitive,
                question=f"What should {cell.primitive or cell.postcode.key} contain?",
                context=cell.content or "No content yet — this cell is blocked.",
            ))

        elif cell.fill == FillState.P and cell.confidence < 0.80:
            esc_id += 1
            escalations.append(Escalation(
                id=f"ESC-{esc_id}",
                urgency="non-blocking",
                postcode=cell.postcode.key,
                primitive=cell.primitive,
                question=f"Can you clarify {cell.primitive or cell.postcode.key}?",
                context=cell.content or "Partial content with low confidence.",
            ))

    return escalations


# ---------------------------------------------------------------------------
# Manifest node
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestNode:
    """A single node in the emitted manifest."""
    postcode: str
    primitive: str
    content: str
    fill_state: str      # "F" or "P"
    confidence: float
    layer: str
    concern: str
    scope: str
    dimension: str
    domain: str
    depth: int
    connections: tuple[str, ...]
    parent: Optional[str]
    source: tuple[str, ...]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    """The emitted manifest — what Claude Code receives.

    Contains:
      - Ordered nodes (parents before children, high confidence first within layer)
      - Escalations (questions that need human input)
      - Metadata (grid stats, simulation result)
    """
    nodes: list[ManifestNode] = field(default_factory=list)
    escalations: list[Escalation] = field(default_factory=list)
    intent: str = ""
    root_postcode: str = ""
    total_grid_cells: int = 0
    emitted_cells: int = 0
    fill_rate: float = 0.0
    simulation_passed: bool = False
    layers_active: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "version": "1.0",
            "type": "motherlabs_manifest",
            "intent": self.intent,
            "root": self.root_postcode,
            "stats": {
                "total_grid_cells": self.total_grid_cells,
                "emitted_cells": self.emitted_cells,
                "fill_rate": round(self.fill_rate, 3),
                "simulation_passed": self.simulation_passed,
                "layers_active": list(self.layers_active),
            },
            "nodes": [
                {
                    "postcode": n.postcode,
                    "primitive": n.primitive,
                    "content": n.content,
                    "fill_state": n.fill_state,
                    "confidence": round(n.confidence, 3),
                    "layer": n.layer,
                    "scope": n.scope,
                    "depth": n.depth,
                    "connections": list(n.connections),
                    "parent": n.parent,
                    "source": list(n.source),
                }
                for n in self.nodes
            ],
            "escalations": [
                {
                    "id": e.id,
                    "urgency": e.urgency,
                    "postcode": e.postcode,
                    "primitive": e.primitive,
                    "question": e.question,
                    "context": e.context,
                    "options": list(e.options),
                }
                for e in self.escalations
            ],
        }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def emit(grid: Grid, force: bool = False) -> Optional[Manifest]:
    """Emit a manifest from the grid.

    Simulation-gated: runs governor first. If simulation fails and
    force=False, returns None. If force=True, emits anyway (with
    simulation_passed=False).

    Args:
        grid: The grid to emit from.
        force: If True, emit even if simulation fails.

    Returns:
        Manifest if simulation passes (or force=True), else None.
    """
    # Run simulation gate
    sim = governor(grid)

    if not sim.passed and not force:
        return None

    # Extract emittable cells (F and P only)
    emittable = [
        c for c in grid.cells.values()
        if c.fill in (FillState.F, FillState.P)
    ]

    # Dependency-order: parents before children, then by depth, then confidence
    ordered = _dependency_order(emittable, grid)

    # Build manifest nodes
    nodes = [
        ManifestNode(
            postcode=c.postcode.key,
            primitive=c.primitive,
            content=c.content,
            fill_state=c.fill.name,
            confidence=c.confidence,
            layer=c.postcode.layer,
            concern=c.postcode.concern,
            scope=c.postcode.scope,
            dimension=c.postcode.dimension,
            domain=c.postcode.domain,
            depth=c.postcode.depth,
            connections=c.connections,
            parent=c.parent,
            source=c.source,
        )
        for c in ordered
    ]

    # Extract escalations
    escalations = extract_escalations(grid)

    return Manifest(
        nodes=nodes,
        escalations=escalations,
        intent=grid.intent_text,
        root_postcode=grid.root or "",
        total_grid_cells=grid.total_cells,
        emitted_cells=len(nodes),
        fill_rate=grid.fill_rate,
        simulation_passed=sim.passed,
        layers_active=tuple(sorted(grid.activated_layers)),
    )


def _dependency_order(cells: list[Cell], grid: Grid) -> list[Cell]:
    """Order cells so parents come before children.

    Strategy:
      1. Cells with no parent (or parent=intent contract) come first
      2. Then cells whose parent is already in the output
      3. Within same depth, sort by confidence descending
      4. Fall back to postcode alphabetical for determinism
    """
    ordered: list[Cell] = []
    emitted_keys: set[str] = set()
    remaining = list(cells)

    # Multiple passes to resolve dependencies
    max_passes = 20
    for _ in range(max_passes):
        if not remaining:
            break

        next_remaining: list[Cell] = []
        made_progress = False

        for cell in remaining:
            can_emit = False
            if cell.parent is None or cell.parent == INTENT_CONTRACT:
                can_emit = True
            elif cell.parent in emitted_keys:
                can_emit = True
            # If parent isn't in our emit set at all, emit anyway
            elif cell.parent not in {c.postcode.key for c in cells}:
                can_emit = True

            if can_emit:
                ordered.append(cell)
                emitted_keys.add(cell.postcode.key)
                made_progress = True
            else:
                next_remaining.append(cell)

        remaining = next_remaining
        if not made_progress:
            # Circular or unresolvable — emit remaining as-is
            ordered.extend(remaining)
            break

    # Secondary sort within dependency tiers: depth ascending, confidence descending
    # We preserve the dependency order but sort within same-depth groups
    return ordered
