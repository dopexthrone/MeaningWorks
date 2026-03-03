"""
kernel/world_grid.py — World grid lifecycle management.

LEAF module. Imports only from kernel/ (cell, grid, ops).

The world grid is Mother's persistent world model — a single always-live
grid (map_id="world") that holds environment observations, user state,
self-knowledge, project context, and goals. The navigator scores cells
across all these regions to decide what Mother should do next.

Persistence uses existing save_grid("world") / load_grid("world").
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Optional

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.perception_bridge import MODALITY_POSTCODES, modality_for_postcode

__all__ = [
    "bootstrap_world_grid",
    "merge_compilation_into_world",
    "apply_staleness_decay",
    "world_grid_health",
    "WORLD_SEED_CELLS",
    "MODALITY_HALF_LIVES",
    "MAX_WORLD_CELLS",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Modality half-lives for staleness decay (seconds)
MODALITY_HALF_LIVES: dict[str, float] = {
    "screen": 60.0,
    "speech": 30.0,
    "camera": 120.0,
    "fusion": 45.0,
}

# Hard cap on world grid size to prevent unbounded growth
MAX_WORLD_CELLS = 500

# Confidence threshold below which decayed cells reset to Empty
_DECAY_PRUNE_THRESHOLD = 0.05

# Seed cells: these are always present in a world grid.
# Each tuple: (postcode_key, primitive)
WORLD_SEED_CELLS: tuple[tuple[str, str], ...] = (
    # Observation layer — environment + user
    ("OBS.ENV.APP.WHAT.USR", "environment_screen"),
    ("OBS.USR.APP.WHAT.USR", "user_speech"),
    ("OBS.USR.APP.WHERE.USR", "user_presence"),
    ("OBS.ENV.APP.HOW.USR", "activity_fusion"),
    # Intent layer — project + tasks
    ("INT.PRJ.APP.WHAT.USR", "active_project"),
    ("INT.TSK.APP.WHAT.USR", "pending_tasks"),
    # Meta layer — self-knowledge
    ("MET.MEM.DOM.WHAT.MTH", "learned_patterns"),
    ("MET.GOL.DOM.WHY.MTH", "improvement_goals"),
    # Network layer — external connections
    ("NET.STA.APP.WHAT.USR", "external_services"),
    # Time layer — temporal state
    ("TME.SCH.APP.WHEN.USR", "schedule_state"),
    # Agency layer — active agents
    ("AGN.TSK.APP.WHAT.USR", "active_agents"),
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_world_grid(existing: Grid | None = None) -> Grid:
    """Create or extend a world grid with required structure.

    Seeds empty cells for all WORLD_SEED_CELLS postcodes.
    Idempotent — only adds cells that don't exist.

    Args:
        existing: An existing grid to extend. If None, creates a new grid.

    Returns:
        The grid (mutated in-place if existing was provided).
    """
    grid = existing if existing is not None else Grid()

    # Set intent contract if not already set
    if grid.root is None:
        grid.set_intent(
            "Mother world model",
            "INT.SEM.ECO.WHAT.MTH",
            "world_model_root",
        )

    # Seed cells — only add if not already present
    for postcode_key, primitive in WORLD_SEED_CELLS:
        if not grid.has(postcode_key):
            pc = parse_postcode(postcode_key)
            cell = Cell(
                postcode=pc,
                primitive=primitive,
                content="",
                fill=FillState.E,
                confidence=0.0,
                source=(INTENT_CONTRACT,),
            )
            grid.put(cell)

    return grid


# ---------------------------------------------------------------------------
# Compilation merge
# ---------------------------------------------------------------------------

def merge_compilation_into_world(
    world: Grid,
    compilation: Grid,
    project_id: str,
) -> int:
    """Copy relevant cells from a compilation grid into the world grid.

    Merges filled cells from the compilation that belong to layers relevant
    to the world model (INT, MET, NET). Cells are tagged with project_id
    in their source.

    Args:
        world: The world grid to merge into (mutated in-place).
        compilation: The compilation grid to merge from.
        project_id: Identifier for the project (used in source provenance).

    Returns:
        Count of cells merged.
    """
    merged = 0

    # Only merge certain layers from compilation
    merge_layers = {"INT", "MET", "NET"}

    for pk, cell in compilation.cells.items():
        if not cell.is_filled:
            continue
        if cell.postcode.layer not in merge_layers:
            continue

        # Tag with project provenance
        new_source = cell.source + (f"observation:compile:{project_id}",)

        existing = world.get(pk)
        if existing and existing.is_filled:
            # AX3 revision: keep revision history
            revisions = existing.revisions + ((existing.content, existing.confidence),)
        else:
            revisions = ()

        merged_cell = Cell(
            postcode=cell.postcode,
            primitive=cell.primitive,
            content=cell.content,
            fill=cell.fill,
            confidence=cell.confidence,
            connections=cell.connections,
            parent=cell.parent,
            source=new_source,
            revisions=revisions,
        )
        world.put(merged_cell)
        merged += 1

    # Enforce cell cap after merge
    _enforce_cell_cap(world)

    return merged


# ---------------------------------------------------------------------------
# Staleness decay
# ---------------------------------------------------------------------------

def apply_staleness_decay(
    grid: Grid,
    half_lives: dict[str, float] | None = None,
    now: float | None = None,
) -> int:
    """Decay confidence of observation cells based on modality half-life.

    Only decays cells in the OBS layer whose postcodes map to known modalities.
    Cells below _DECAY_PRUNE_THRESHOLD are reset to Empty state.

    Args:
        grid: The world grid (mutated in-place).
        half_lives: Modality → half-life seconds. Defaults to MODALITY_HALF_LIVES.
        now: Current time. Defaults to time.time().

    Returns:
        Count of cells decayed (confidence reduced or reset to empty).
    """
    ts = now if now is not None else time.time()
    hl = half_lives if half_lives is not None else MODALITY_HALF_LIVES
    decayed = 0

    obs_cells = grid.cells_in_layer("OBS")
    for cell in obs_cells:
        if not cell.is_filled:
            continue

        # Find the modality for this cell
        modality = modality_for_postcode(cell.postcode.key)
        if modality is None:
            continue

        half_life = hl.get(modality)
        if half_life is None or half_life <= 0:
            continue

        # Find the observation timestamp from source
        obs_time = _extract_observation_time(cell.source)
        if obs_time is None:
            continue

        age = ts - obs_time
        if age <= 0:
            continue

        decay_factor = math.pow(0.5, age / half_life)
        new_confidence = cell.confidence * decay_factor

        if new_confidence < _DECAY_PRUNE_THRESHOLD:
            # Reset to empty
            reset_cell = Cell(
                postcode=cell.postcode,
                primitive=cell.primitive,
                content="",
                fill=FillState.E,
                confidence=0.0,
                source=cell.source,
                revisions=cell.revisions + ((cell.content, cell.confidence),),
            )
            grid.put(reset_cell)
            decayed += 1
        elif abs(new_confidence - cell.confidence) > 0.001:
            # Reduce confidence
            new_fill = FillState.P if new_confidence < 0.85 else cell.fill
            decayed_cell = replace(
                cell,
                confidence=new_confidence,
                fill=new_fill,
            )
            grid.put(decayed_cell)
            decayed += 1

    return decayed


def _extract_observation_time(source: tuple[str, ...]) -> float | None:
    """Extract timestamp from observation/perception/fusion source strings.

    Looks for patterns like "observation:screen:1740000000.0"
    """
    for s in source:
        parts = s.split(":")
        if len(parts) >= 3 and parts[0] in ("observation", "perception", "fusion"):
            try:
                return float(parts[-1])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Health diagnostics
# ---------------------------------------------------------------------------

def world_grid_health(grid: Grid) -> dict:
    """Diagnostic: staleness, coverage, layer fill rates, cell count.

    Returns dict with:
        total_cells: int
        filled_cells: int
        empty_cells: int
        fill_rate: float
        layer_fill_rates: dict[str, float]
        stale_observation_count: int  (OBS cells with confidence < 0.3)
        seed_coverage: float  (fraction of WORLD_SEED_CELLS that exist)
    """
    total = grid.total_cells
    filled = len(grid.filled_cells())
    empty = len(grid.empty_cells())

    # Layer fill rates
    layer_rates: dict[str, float] = {}
    for layer in sorted(grid.activated_layers):
        cells = grid.cells_in_layer(layer)
        if cells:
            layer_filled = sum(1 for c in cells if c.is_filled)
            layer_rates[layer] = layer_filled / len(cells)
        else:
            layer_rates[layer] = 0.0

    # Stale observations
    obs_cells = grid.cells_in_layer("OBS")
    stale = sum(
        1 for c in obs_cells
        if c.is_filled and c.confidence < 0.3
    )

    # Seed coverage
    seed_present = sum(1 for pk, _ in WORLD_SEED_CELLS if grid.has(pk))
    seed_coverage = seed_present / len(WORLD_SEED_CELLS) if WORLD_SEED_CELLS else 1.0

    return {
        "total_cells": total,
        "filled_cells": filled,
        "empty_cells": empty,
        "fill_rate": grid.fill_rate,
        "layer_fill_rates": layer_rates,
        "stale_observation_count": stale,
        "seed_coverage": seed_coverage,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enforce_cell_cap(grid: Grid) -> int:
    """Remove lowest-confidence empty/partial cells if grid exceeds MAX_WORLD_CELLS.

    Never removes Filled (F) cells with confidence >= 0.85.
    Returns count of cells removed.
    """
    if grid.total_cells <= MAX_WORLD_CELLS:
        return 0

    excess = grid.total_cells - MAX_WORLD_CELLS

    # Candidates for removal: Empty cells first, then low-confidence Partial
    candidates: list[tuple[str, float]] = []
    for pk, cell in grid.cells.items():
        if cell.fill == FillState.E:
            candidates.append((pk, -1.0))  # Empty cells removed first
        elif cell.fill == FillState.P and cell.confidence < 0.3:
            candidates.append((pk, cell.confidence))

    # Sort: lowest confidence first (empty at -1.0 = first)
    candidates.sort(key=lambda x: x[1])

    removed = 0
    for pk, _ in candidates[:excess]:
        if pk in grid.cells:
            del grid.cells[pk]
            removed += 1

    return removed
