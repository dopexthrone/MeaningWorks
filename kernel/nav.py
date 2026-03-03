"""
kernel/nav.py — Navigation layer: lightweight text format + context budget.

The nav format is what the AI holds in context during compilation.
~3 lines per node, ~30 tokens per node. The full grid nav fits in
a few thousand tokens for most compilations.

Two representations:
  - nav string: lightweight, fits in LLM context, used for navigation
  - Grid: full structure, used for operations

Roundtrip: grid_to_nav(grid) → text, nav_to_grid(text) → grid.
Budget: budget_nav(grid, max_tokens) → truncated text.
"""

from __future__ import annotations

import re
from typing import Optional

from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
)
from kernel.grid import Grid, INTENT_CONTRACT


# ---------------------------------------------------------------------------
# Serialization: Grid → nav text
# ---------------------------------------------------------------------------

_FILL_NAMES = {s: s.name for s in FillState}
_FILL_FROM_NAME = {s.name: s for s in FillState}

# Approximate tokens per node in nav format
TOKENS_PER_NODE = 30


def grid_to_nav(grid: Grid, include_empty: bool = False) -> str:
    """Serialize a grid to the lightweight navigation format.

    Format per cell:
      POSTCODE | FILL CONF | primitive [parent:POSTCODE]
        → conn1, conn2, ...

    Cells grouped by layer, sorted by postcode within layer.
    Empty cells excluded by default (set include_empty=True to include).

    Args:
        grid: The grid to serialize.
        include_empty: Whether to include E (empty) cells.

    Returns:
        Navigation text string.
    """
    lines: list[str] = []

    # Group by layer
    by_layer: dict[str, list[Cell]] = {}
    for cell in grid.cells.values():
        if not include_empty and cell.is_empty:
            continue
        layer = cell.postcode.layer
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(cell)

    # Metadata header
    filled = len(grid.filled_cells())
    total = grid.total_cells
    lines.append(f"# MAP {filled}F/{total}T root:{grid.root or 'none'}")
    if grid.intent_text:
        # Truncate intent for nav
        intent_short = grid.intent_text[:120].replace("\n", " ")
        lines.append(f"# INTENT {intent_short}")
    lines.append("")

    for layer in sorted(by_layer.keys()):
        lines.append(f"## {layer}")
        cells = sorted(by_layer[layer], key=lambda c: c.postcode.key)
        for cell in cells:
            lines.append(_cell_to_nav(cell))
        lines.append("")

    return "\n".join(lines)


def _cell_to_nav(cell: Cell) -> str:
    """Serialize a single cell to nav format."""
    key = cell.postcode.key
    fill = cell.fill.name
    conf = f"{cell.confidence:.2f}"
    prim = cell.primitive or "(unnamed)"

    # Parent annotation
    parent_str = ""
    if cell.parent:
        parent_str = f" ^{cell.parent}"

    line = f"{key} | {fill} {conf} | {prim}{parent_str}"

    # Connections on next line
    if cell.connections:
        conns = ", ".join(cell.connections)
        line += f"\n  -> {conns}"

    return line


# ---------------------------------------------------------------------------
# Deserialization: nav text → Grid
# ---------------------------------------------------------------------------

# Regex patterns for parsing
_HEADER_RE = re.compile(r"^# MAP (\d+)F/(\d+)T root:(.+)$")
_INTENT_RE = re.compile(r"^# INTENT (.+)$")
_LAYER_RE = re.compile(r"^## ([A-Z]{3})$")
_CELL_RE = re.compile(
    r"^([A-Z_]+\.[A-Z_]+\.[A-Z_]+\.[A-Z_]+\.[A-Z_]+)"  # postcode
    r"\s*\|\s*([FPEBQC])\s+([\d.]+)"                      # fill + confidence
    r"\s*\|\s*(.+)$"                                       # primitive + optional parent
)
_CONN_RE = re.compile(r"^\s+->\s+(.+)$")
_PARENT_RE = re.compile(r"^(.+?)\s+\^(.+)$")


def nav_to_grid(text: str) -> Grid:
    """Deserialize nav text back to a Grid.

    Reconstructs cells with postcode, fill state, confidence,
    primitive, connections, and parent. Content is NOT preserved
    (nav format doesn't carry full content — that's the storage layer).

    Args:
        text: Navigation format text.

    Returns:
        Reconstructed Grid.
    """
    grid = Grid()
    root_key: Optional[str] = None
    intent_text = ""

    lines = text.split("\n")
    i = 0
    current_cell_key: Optional[str] = None

    while i < len(lines):
        line = lines[i].rstrip()

        # Header
        m = _HEADER_RE.match(line)
        if m:
            root_key = m.group(3) if m.group(3) != "none" else None
            i += 1
            continue

        # Intent
        m = _INTENT_RE.match(line)
        if m:
            intent_text = m.group(1)
            i += 1
            continue

        # Layer header (skip, informational)
        m = _LAYER_RE.match(line)
        if m:
            i += 1
            continue

        # Connection line (belongs to previous cell)
        m = _CONN_RE.match(line)
        if m and current_cell_key:
            conn_str = m.group(1).strip()
            connections = tuple(
                c.strip() for c in conn_str.split(",") if c.strip()
            )
            # Update the cell with connections
            existing = grid.get(current_cell_key)
            if existing:
                from dataclasses import replace
                updated = replace(existing, connections=connections)
                grid.put(updated)
            i += 1
            continue

        # Cell line
        m = _CELL_RE.match(line)
        if m:
            postcode_str = m.group(1)
            fill_name = m.group(2)
            confidence = float(m.group(3))
            prim_and_parent = m.group(4).strip()

            # Parse parent
            parent = None
            primitive = prim_and_parent
            pm = _PARENT_RE.match(prim_and_parent)
            if pm:
                primitive = pm.group(1).strip()
                parent = pm.group(2).strip()

            try:
                postcode = parse_postcode(postcode_str)
            except ValueError:
                i += 1
                continue

            fill_state = _FILL_FROM_NAME.get(fill_name, FillState.E)

            # Determine source
            source: tuple[str, ...] = ()
            if parent:
                source = (parent,)
            elif postcode_str == root_key:
                source = (INTENT_CONTRACT,)
            else:
                source = (INTENT_CONTRACT,)

            cell = Cell(
                postcode=postcode,
                primitive=primitive,
                content="",  # nav format doesn't carry content
                fill=fill_state,
                confidence=confidence,
                parent=parent,
                source=source,
            )
            grid.put(cell)
            current_cell_key = postcode_str
            i += 1
            continue

        # Empty or unrecognized line
        current_cell_key = None
        i += 1

    # Set root
    if root_key:
        grid.root = root_key
    if intent_text:
        grid.intent_text = intent_text

    return grid


# ---------------------------------------------------------------------------
# Context budget
# ---------------------------------------------------------------------------

def estimate_tokens(grid: Grid, include_empty: bool = False) -> int:
    """Estimate the token count of the nav format.

    ~30 tokens per non-empty cell. Empty cells ~15 tokens.
    Header ~20 tokens.
    """
    count = 20  # header
    for cell in grid.cells.values():
        if cell.is_empty and not include_empty:
            continue
        if cell.is_empty:
            count += 15
        else:
            count += TOKENS_PER_NODE
            # Extra tokens for connections
            count += len(cell.connections) * 5
    return count


def budget_nav(
    grid: Grid,
    max_tokens: int,
    include_empty: bool = False,
) -> str:
    """Produce a nav string that fits within a token budget.

    Truncation strategy (priority order):
      1. Always include the root cell
      2. Include filled (F) cells sorted by confidence descending
      3. Include partial (P) cells sorted by confidence descending
      4. Include blocked (B) cells (important — they're escalations)
      5. Include candidate (C) cells
      6. Include quarantined (Q) cells
      7. Include empty (E) cells if include_empty and budget allows

    Args:
        grid: The grid to serialize.
        max_tokens: Maximum token budget.
        include_empty: Whether to consider E cells at all.

    Returns:
        Truncated navigation text string.
    """
    # Priority-sorted cells
    priority_cells: list[Cell] = []

    # Root always first
    if grid.root:
        root_cell = grid.get(grid.root)
        if root_cell:
            priority_cells.append(root_cell)

    # F cells by confidence (descending)
    f_cells = sorted(
        [c for c in grid.cells.values()
         if c.fill == FillState.F and c.postcode.key != grid.root],
        key=lambda c: c.confidence,
        reverse=True,
    )
    priority_cells.extend(f_cells)

    # P cells by confidence (descending)
    p_cells = sorted(
        [c for c in grid.cells.values() if c.fill == FillState.P],
        key=lambda c: c.confidence,
        reverse=True,
    )
    priority_cells.extend(p_cells)

    # B cells
    priority_cells.extend(grid.blocked_cells())

    # C cells
    priority_cells.extend(grid.candidate_cells())

    # Q cells
    priority_cells.extend(grid.quarantined_cells())

    # E cells
    if include_empty:
        priority_cells.extend(grid.empty_cells())

    # Build nav with budget
    lines: list[str] = []
    token_count = 20  # header

    # Header
    filled = len(grid.filled_cells())
    total = grid.total_cells
    lines.append(f"# MAP {filled}F/{total}T root:{grid.root or 'none'}")
    if grid.intent_text:
        intent_short = grid.intent_text[:120].replace("\n", " ")
        lines.append(f"# INTENT {intent_short}")
    lines.append("")

    included_keys: set[str] = set()
    current_layer: Optional[str] = None

    for cell in priority_cells:
        key = cell.postcode.key
        if key in included_keys:
            continue

        # Estimate tokens for this cell
        cell_tokens = TOKENS_PER_NODE + len(cell.connections) * 5
        if token_count + cell_tokens > max_tokens:
            break

        # Layer header if new layer
        if cell.postcode.layer != current_layer:
            if current_layer is not None:
                lines.append("")
            current_layer = cell.postcode.layer
            lines.append(f"## {current_layer}")
            token_count += 5

        lines.append(_cell_to_nav(cell))
        token_count += cell_tokens
        included_keys.add(key)

    # Footer with truncation notice
    excluded = grid.total_cells - len(included_keys)
    if excluded > 0:
        lines.append("")
        lines.append(f"# TRUNCATED {excluded} cells omitted (budget: {max_tokens} tokens)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured grid export (lossless bridge for synthesis provenance)
# ---------------------------------------------------------------------------

def grid_to_structured(grid: Grid) -> list[dict]:
    """Export grid cells as structured dicts with full data.

    Unlike grid_to_nav() which produces text, this returns machine-readable
    dicts preserving all cell data including content, source tuples, and
    revision counts. Used by synthesis to maintain structural provenance.

    Returns list of dicts for non-empty cells, sorted by postcode.
    """
    result = []
    for key in sorted(grid.cells.keys()):
        cell = grid.cells[key]
        if cell.is_empty:
            continue
        result.append({
            "postcode": cell.postcode.key,
            "primitive": cell.primitive,
            "content": cell.content,
            "fill_state": cell.fill.name,
            "confidence": round(cell.confidence, 4),
            "source": list(cell.source),
            "connections": list(cell.connections),
            "parent": cell.parent,
            "revision_count": len(cell.revisions),
            "layer": cell.postcode.layer,
            "concern": cell.postcode.concern,
            "scope": cell.postcode.scope,
            "dimension": cell.postcode.dimension,
            "domain": cell.postcode.domain,
        })
    return result
