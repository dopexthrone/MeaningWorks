"""
kernel/observer.py — Observation loop that writes deltas back to the grid.

LEAF module. After every compilation or interaction, records what happened
versus what was expected. Observations modify cell confidence scores and
can trigger state transitions.

This is the AX3 FEEDBACK axiom in action: every run writes a delta.
The grid doesn't just hold structure — it evolves from real usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid


# ---------------------------------------------------------------------------
# Observation types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationDelta:
    """A single observation: expected vs actual for a cell."""
    postcode: str
    event_type: str  # "compilation", "interaction", "execution", "verification"
    expected: str  # what was expected to happen
    actual: str  # what actually happened
    confidence_before: float
    confidence_after: float
    anomaly: bool = False  # did something unexpected happen?
    anomaly_detail: str = ""


@dataclass
class ObservationBatch:
    """Collection of observations from one run."""
    run_id: str
    deltas: list[ObservationDelta] = field(default_factory=list)
    cells_touched: int = 0
    cells_improved: int = 0
    cells_degraded: int = 0
    transitions: list[tuple[str, str, str]] = field(default_factory=list)  # (postcode, old_state, new_state)


# ---------------------------------------------------------------------------
# Confidence adjustment
# ---------------------------------------------------------------------------

# How much a single observation moves confidence
_CONFIRM_BOOST = 0.03  # observation confirms expectation
_CONTRADICT_DECAY = 0.08  # observation contradicts expectation
_ANOMALY_DECAY = 0.12  # unexpected anomaly
_MIN_CONFIDENCE = 0.0
_MAX_CONFIDENCE = 1.0

# Thresholds for state transitions
_PROMOTE_THRESHOLD = 0.85  # P→F when confidence crosses this
_DEMOTE_THRESHOLD = 0.50  # F→P when confidence drops below this
_QUARANTINE_THRESHOLD = 0.20  # P→Q when confidence drops below this


def _clamp(v: float) -> float:
    return max(_MIN_CONFIDENCE, min(_MAX_CONFIDENCE, v))


def _adjust_confidence(
    current: float,
    confirmed: bool,
    anomaly: bool = False,
) -> float:
    """Adjust confidence based on observation outcome."""
    if anomaly:
        return _clamp(current - _ANOMALY_DECAY)
    if confirmed:
        return _clamp(current + _CONFIRM_BOOST)
    return _clamp(current - _CONTRADICT_DECAY)


def _should_transition(cell: Cell, new_confidence: float) -> Optional[FillState]:
    """Determine if a cell should change fill state based on new confidence."""
    if cell.fill == FillState.P and new_confidence >= _PROMOTE_THRESHOLD:
        return FillState.F
    if cell.fill == FillState.F and new_confidence < _DEMOTE_THRESHOLD:
        return FillState.P
    if cell.fill == FillState.P and new_confidence < _QUARANTINE_THRESHOLD:
        return FillState.Q
    return None


# ---------------------------------------------------------------------------
# Core observation functions
# ---------------------------------------------------------------------------

def record_observation(
    grid: Grid,
    postcode: str,
    event_type: str,
    expected: str,
    actual: str,
    confirmed: bool = True,
    anomaly: bool = False,
    anomaly_detail: str = "",
) -> ObservationDelta:
    """Record a single observation against a cell in the grid.

    Adjusts cell confidence and may trigger state transitions.
    Returns the ObservationDelta describing what changed.

    Does NOT modify the grid directly — call apply_observation() to mutate.
    """
    cell = grid.get(postcode)
    if cell is None:
        # Observation for a cell that doesn't exist — create it as candidate
        return ObservationDelta(
            postcode=postcode,
            event_type=event_type,
            expected=expected,
            actual=actual,
            confidence_before=0.0,
            confidence_after=0.0,
            anomaly=True,
            anomaly_detail=f"Cell {postcode} does not exist in grid. {anomaly_detail}",
        )

    old_conf = cell.confidence
    new_conf = _adjust_confidence(old_conf, confirmed, anomaly)

    return ObservationDelta(
        postcode=postcode,
        event_type=event_type,
        expected=expected,
        actual=actual,
        confidence_before=old_conf,
        confidence_after=new_conf,
        anomaly=anomaly,
        anomaly_detail=anomaly_detail,
    )


def apply_observation(grid: Grid, delta: ObservationDelta) -> Optional[tuple[str, str]]:
    """Apply a single observation delta to the grid.

    Mutates the grid: updates cell confidence, may transition fill state.
    Returns (old_state, new_state) tuple if a transition happened, else None.
    """
    cell = grid.get(delta.postcode)
    if cell is None:
        return None

    new_conf = delta.confidence_after
    new_fill = _should_transition(cell, new_conf) or cell.fill

    # Record the observation in revisions
    new_revisions = cell.revisions + ((delta.event_type, delta.confidence_before),)

    updated = Cell(
        postcode=cell.postcode,
        primitive=cell.primitive,
        content=cell.content,
        fill=new_fill,
        confidence=new_conf,
        connections=cell.connections,
        parent=cell.parent,
        source=cell.source + (f"obs:{delta.event_type}",),
        revisions=new_revisions,
    )
    grid.cells[delta.postcode] = updated

    if new_fill != cell.fill:
        return (cell.fill.name, new_fill.name)
    return None


def apply_batch(grid: Grid, deltas: list[ObservationDelta]) -> ObservationBatch:
    """Apply a batch of observation deltas to the grid.

    Returns an ObservationBatch summarizing all changes.
    """
    batch = ObservationBatch(
        run_id=f"obs-{len(deltas)}",
    )

    for delta in deltas:
        transition = apply_observation(grid, delta)
        batch.cells_touched += 1

        if delta.confidence_after > delta.confidence_before:
            batch.cells_improved += 1
        elif delta.confidence_after < delta.confidence_before:
            batch.cells_degraded += 1

        if transition:
            batch.transitions.append(
                (delta.postcode, transition[0], transition[1])
            )

        batch.deltas.append(delta)

    return batch


# ---------------------------------------------------------------------------
# Confidence drift analysis
# ---------------------------------------------------------------------------

def compute_confidence_drift(
    grid: Grid,
    previous_grid: Optional[Grid] = None,
) -> dict[str, object]:
    """Compute confidence drift between two grid states.

    If no previous_grid, returns current confidence distribution.
    """
    current_cells = {k: c for k, c in grid.cells.items() if c.fill != FillState.E}

    if previous_grid is None:
        confidences = [c.confidence for c in current_cells.values()]
        if not confidences:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0, "drift": 0.0}
        return {
            "mean": round(sum(confidences) / len(confidences), 4),
            "min": round(min(confidences), 4),
            "max": round(max(confidences), 4),
            "count": len(confidences),
            "drift": 0.0,
        }

    prev_cells = {k: c for k, c in previous_grid.cells.items() if c.fill != FillState.E}

    drifts: list[float] = []
    for pk, cell in current_cells.items():
        prev = prev_cells.get(pk)
        if prev is not None:
            drifts.append(cell.confidence - prev.confidence)

    if not drifts:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0, "drift": 0.0}

    return {
        "mean": round(sum(c.confidence for c in current_cells.values()) / len(current_cells), 4),
        "min": round(min(c.confidence for c in current_cells.values()), 4),
        "max": round(max(c.confidence for c in current_cells.values()), 4),
        "count": len(current_cells),
        "drift": round(sum(drifts) / len(drifts), 4),
        "improved": sum(1 for d in drifts if d > 0),
        "degraded": sum(1 for d in drifts if d < 0),
        "stable": sum(1 for d in drifts if d == 0),
    }


def find_low_confidence_cells(
    grid: Grid,
    threshold: float = 0.60,
) -> list[tuple[str, float, str]]:
    """Find cells below confidence threshold.

    Returns list of (postcode, confidence, primitive) sorted by confidence ascending.
    These are candidates for human escalation or additional observation.
    """
    results: list[tuple[str, float, str]] = []
    for pk, cell in grid.cells.items():
        if cell.fill in (FillState.F, FillState.P) and cell.confidence < threshold:
            results.append((pk, cell.confidence, cell.primitive))

    results.sort(key=lambda x: x[1])
    return results


def find_anomalous_cells(
    observations: list[ObservationDelta],
) -> list[ObservationDelta]:
    """Extract observations that detected anomalies."""
    return [o for o in observations if o.anomaly]
