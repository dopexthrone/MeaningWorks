"""
kernel/ops.py — Fill and Connect operations.

The only two mutations in the kernel. Everything the compiler does
reduces to: fill a cell, or connect two cells.

Axiom enforcement happens here, not in the cell or grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
    SCOPE_DEPTH,
)
from kernel.grid import Grid, INTENT_CONTRACT


class FillStatus(Enum):
    """Outcome of a fill operation."""
    OK = "ok"                          # Cell filled successfully
    REVISED = "revised"                # Cell re-filled (AX3 FEEDBACK)
    QUARANTINED = "quarantined"        # Axiom violation, cell quarantined
    BLOCKED = "blocked"                # Cell is blocked, fill rejected
    PROMOTED = "promoted"              # Candidate promoted to filled


@dataclass(frozen=True)
class FillResult:
    """Result of a fill operation."""
    status: FillStatus
    cell: Cell                         # The cell after the operation
    violation: Optional[str] = None    # Which axiom was violated (if quarantined)
    activated_layers: tuple[str, ...] = ()  # Layers activated by connections
    content_warnings: tuple[str, ...] = ()  # Content-postcode alignment warnings


class ConnectStatus(Enum):
    """Outcome of a connect operation."""
    OK = "ok"                          # Connection created
    DUPLICATE = "duplicate"            # Already connected (idempotent)
    MISSING_SOURCE = "missing_source"  # Source cell doesn't exist
    MISSING_TARGET = "missing_target"  # Target cell doesn't exist (created as E)
    LAYER_ACTIVATED = "layer_activated"  # Connection triggered layer activation


@dataclass(frozen=True)
class ConnectResult:
    """Result of a connect operation."""
    status: ConnectStatus
    from_cell: Optional[Cell] = None
    to_cell: Optional[Cell] = None
    activated_layer: Optional[str] = None


# --- Fill -------------------------------------------------------------------

def fill(
    grid: Grid,
    postcode_key: str,
    primitive: str,
    content: str,
    confidence: float,
    connections: tuple[str, ...] = (),
    source: tuple[str, ...] = (),
    parent: Optional[str] = None,
    content_validator=None,
    agent: str = "",
) -> FillResult:
    """Fill a cell in the grid.

    Enforces all axioms:
      AX1 PROVENANCE — source must reference a filled cell or intent contract
      AX2 DESCENT    — parent must be filled before child
      AX3 FEEDBACK   — re-fill preserves history
      AX5 CONSTRAINT — blocked cells reject fill

    Returns FillResult with status and the resulting cell.
    The grid is mutated in place.
    """
    postcode = parse_postcode(postcode_key)
    confidence = max(0.0, min(1.0, confidence))

    existing = grid.get(postcode_key)

    # AX5 CONSTRAINT — blocked cells reject fill
    if existing and existing.is_blocked:
        return FillResult(
            status=FillStatus.BLOCKED,
            cell=existing,
            violation="AX5_CONSTRAINT",
        )

    # AX4 EMERGENCE — agents propose, never self-approve
    # When promoting a candidate (C→F/P), the approving agent must differ from proposer
    if (
        agent
        and existing
        and existing.proposer
        and existing.is_candidate
        and agent == existing.proposer
    ):
        q_cell = Cell(
            postcode=postcode,
            primitive=primitive,
            content=content,
            fill=FillState.Q,
            confidence=0.0,
            connections=connections,
            parent=parent,
            source=source,
            proposer=existing.proposer,
        )
        grid.put(q_cell)
        return FillResult(
            status=FillStatus.QUARANTINED,
            cell=q_cell,
            violation="AX4_SELF_APPROVAL",
        )

    # AX1 PROVENANCE — source must reference filled cell or intent contract
    if not _check_provenance(grid, source, parent):
        # Quarantine — provenance violation
        q_cell = Cell(
            postcode=postcode,
            primitive=primitive,
            content=content,
            fill=FillState.Q,
            confidence=0.0,
            connections=connections,
            parent=parent,
            source=source,
        )
        grid.put(q_cell)
        return FillResult(
            status=FillStatus.QUARANTINED,
            cell=q_cell,
            violation="AX1_PROVENANCE",
        )

    # AX2 DESCENT — parent must be filled (if specified and not root)
    if parent and parent != INTENT_CONTRACT:
        parent_cell = grid.get(parent)
        if parent_cell is None or not parent_cell.is_filled:
            q_cell = Cell(
                postcode=postcode,
                primitive=primitive,
                content=content,
                fill=FillState.Q,
                confidence=0.0,
                connections=connections,
                parent=parent,
                source=source,
            )
            grid.put(q_cell)
            return FillResult(
                status=FillStatus.QUARANTINED,
                cell=q_cell,
                violation="AX2_DESCENT",
            )

    # Determine fill state from confidence
    if confidence >= 0.85:
        fill_state = FillState.F
    elif confidence > 0.0:
        fill_state = FillState.P
    else:
        fill_state = FillState.E

    # AX3 FEEDBACK — re-fill is a revision
    revisions: tuple[tuple[str, float], ...] = ()
    status = FillStatus.OK

    if existing and existing.is_filled:
        # Preserve previous content as revision
        revisions = existing.revisions + ((existing.content, existing.confidence),)
        status = FillStatus.REVISED
    elif existing and existing.is_candidate:
        status = FillStatus.PROMOTED

    new_cell = Cell(
        postcode=postcode,
        primitive=primitive,
        content=content,
        fill=fill_state,
        confidence=confidence,
        connections=connections,
        parent=parent,
        source=source,
        proposer=agent or (existing.proposer if existing else ""),
        revisions=revisions,
    )
    grid.put(new_cell)

    # Record agent in grid's agent map (storage-layer concern)
    if agent:
        grid._agent_map[postcode_key] = agent

    # Sign provenance (non-blocking — failures don't affect fill)
    try:
        from kernel.provenance_signing import ProvenanceSigner, content_hash
        signer = getattr(grid, "_signer", None)
        if signer is None:
            signer = ProvenanceSigner()
            object.__setattr__(grid, "_signer", signer)
        sig = signer.sign_provenance(
            source=source,
            postcode=postcode_key,
            fill_state=fill_state.name,
            content_hash=content_hash(content) if content else "",
        )
        grid._signatures[postcode_key] = sig
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Provenance signing skipped: {e}")

    # Check connections for layer activation
    activated = _activate_from_connections(grid, connections, postcode)

    # Content validation (measurement, not gate)
    content_warnings: tuple[str, ...] = ()
    if content_validator is not None and content:
        from kernel.content_validator import validate_content
        fit = validate_content(content, postcode.dimension, postcode.concern)
        content_warnings = fit.warnings

    return FillResult(
        status=status,
        cell=new_cell,
        activated_layers=tuple(activated),
        content_warnings=content_warnings,
    )


def _check_provenance(
    grid: Grid,
    source: tuple[str, ...],
    parent: Optional[str],
) -> bool:
    """Check AX1: source must reference a filled cell or intent contract.

    Valid provenance if ANY of:
    - source contains INTENT_CONTRACT
    - source contains a postcode of a filled cell
    - parent is the intent contract
    - parent is a filled cell
    - source contains a string starting with "human:" (human input ref)
    - source contains a string starting with "contract:" (external contract ref)
    """
    if not source and not parent:
        return False

    # Check parent
    if parent == INTENT_CONTRACT:
        return True
    if parent:
        parent_cell = grid.get(parent)
        if parent_cell and parent_cell.is_filled:
            return True

    # Check sources
    for s in source:
        if s == INTENT_CONTRACT:
            return True
        if s.startswith(("human:", "contract:", "memory:", "observation:", "perception:", "fusion:")):
            return True
        cell = grid.get(s)
        if cell and cell.is_filled:
            return True

    return False


def _activate_from_connections(
    grid: Grid,
    connections: tuple[str, ...],
    source_postcode: Postcode,
) -> list[str]:
    """Activate layers referenced by connections that aren't active yet."""
    activated = []
    for conn_key in connections:
        try:
            conn_pc = parse_postcode(conn_key)
        except ValueError:
            continue
        if not grid.is_layer_active(conn_pc.layer):
            grid.activate_layer(
                conn_pc.layer,
                conn_pc.concern,
                conn_pc.dimension,
                conn_pc.domain,
            )
            activated.append(conn_pc.layer)
        # Create target cell as E if it doesn't exist — never overwrite
        if not grid.has(conn_key):
            target_cell = Cell(
                postcode=conn_pc,
                primitive="",
                content="",
                fill=FillState.E,
                confidence=0.0,
                source=(source_postcode.key,),
            )
            grid.put(target_cell)
        # If target already exists (E, C, F, etc.), leave it alone
    return activated


# --- Block ------------------------------------------------------------------

def block(
    grid: Grid,
    postcode_key: str,
    reason: str = "",
) -> FillResult:
    """Transition a cell to blocked state (FillState.B).

    Preserves content in revisions. Used when:
    - Verifier blocks after repeated quarantines
    - Governor blocks cycle participants

    Returns FillResult with status BLOCKED.
    """
    existing = grid.get(postcode_key)
    if existing is None:
        return FillResult(
            status=FillStatus.BLOCKED,
            cell=Cell(
                postcode=parse_postcode(postcode_key),
                primitive="",
                content="",
                fill=FillState.B,
                confidence=0.0,
            ),
            violation=reason or "blocked",
        )

    if existing.is_blocked:
        return FillResult(
            status=FillStatus.BLOCKED,
            cell=existing,
            violation="already_blocked",
        )

    # Preserve current content in revisions before blocking
    revisions = existing.revisions
    if existing.content:
        revisions = revisions + ((existing.content, existing.confidence),)

    blocked_cell = Cell(
        postcode=existing.postcode,
        primitive=existing.primitive,
        content=existing.content,
        fill=FillState.B,
        confidence=0.0,
        connections=existing.connections,
        parent=existing.parent,
        source=existing.source,
        revisions=revisions,
    )
    grid.put(blocked_cell)

    return FillResult(
        status=FillStatus.BLOCKED,
        cell=blocked_cell,
        violation=reason or "blocked",
    )


# --- Connect ----------------------------------------------------------------

def connect(
    grid: Grid,
    from_key: str,
    to_key: str,
) -> ConnectResult:
    """Connect two cells in the grid.

    Creates a directed wire from_key -> to_key.
    Both cells must exist (target created as E if missing).
    Idempotent on duplicate connections.
    Connection to inactive layer triggers activation.

    The grid is mutated in place.
    """
    from_cell = grid.get(from_key)
    if from_cell is None:
        return ConnectResult(
            status=ConnectStatus.MISSING_SOURCE,
        )

    # Check for duplicate
    if to_key in from_cell.connections:
        to_cell = grid.get(to_key)
        return ConnectResult(
            status=ConnectStatus.DUPLICATE,
            from_cell=from_cell,
            to_cell=to_cell,
        )

    # Parse target postcode
    try:
        to_pc = parse_postcode(to_key)
    except ValueError:
        return ConnectResult(
            status=ConnectStatus.MISSING_TARGET,
        )

    # Layer activation check
    activated_layer = None
    if not grid.is_layer_active(to_pc.layer):
        grid.activate_layer(to_pc.layer, to_pc.concern, to_pc.dimension, to_pc.domain)
        activated_layer = to_pc.layer

    # Create target cell as E if missing
    status = ConnectStatus.OK
    to_cell = grid.get(to_key)
    if to_cell is None:
        to_cell = Cell(
            postcode=to_pc,
            primitive="",
            content="",
            fill=FillState.E,
            confidence=0.0,
            source=(from_key,),
        )
        grid.put(to_cell)
        status = ConnectStatus.MISSING_TARGET

    # Update from_cell with new connection
    new_connections = from_cell.connections + (to_key,)
    updated_from = replace(from_cell, connections=new_connections)
    grid.put(updated_from)

    if activated_layer:
        status = ConnectStatus.LAYER_ACTIVATED

    return ConnectResult(
        status=status,
        from_cell=updated_from,
        to_cell=to_cell,
        activated_layer=activated_layer,
    )
