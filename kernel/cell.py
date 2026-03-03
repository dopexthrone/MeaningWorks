"""
kernel/cell.py — The atomic unit of the semantic map.

A Cell is a position in coordinate space that can hold content.
8 fields. Nothing else. All metadata (agent logs, run history, locks)
belongs to the storage layer, not the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FillState(Enum):
    """How populated a cell is."""
    F = "filled"        # Complete, verified, confidence >= 0.85
    P = "partial"       # Content exists but incomplete or low confidence
    E = "empty"         # Cell exists in grid, nothing filled yet
    B = "blocked"       # Dependency unresolved — cannot fill until blocker clears
    Q = "quarantined"   # Governor rejected but trace preserved for review
    C = "candidate"     # Emerged pattern awaiting promotion


# Valid axis values -----------------------------------------------------------

LAYERS = frozenset({
    "INT",  # Intent
    "SEM",  # Semantic
    "ORG",  # Organization
    "COG",  # Cognitive
    "AGN",  # Agency
    "STR",  # Structure
    "STA",  # State
    "IDN",  # Identity
    "TME",  # Time
    "EXC",  # Execution
    "CTR",  # Control
    "RES",  # Resource
    "OBS",  # Observability
    "NET",  # Network
    "EMG",  # Emergence
    "MET",  # Meta
    "DAT",  # Data
    "SFX",  # Side Effects
})

CONCERNS = frozenset({
    "SEM",  # Semantic
    "ENT",  # Entity
    "BHV",  # Behavior
    "FNC",  # Function
    "REL",  # Relation
    "PLN",  # Plan
    "MEM",  # Memory
    "ORC",  # Orchestration
    "AGT",  # Agent
    "ACT",  # Actor
    "SCO",  # Scope
    "STA",  # State
    "TRN",  # Transition
    "SNP",  # Snapshot
    "VRS",  # Version
    "SCH",  # Schedule
    "GTE",  # Gate
    "PLY",  # Policy
    "MET",  # Metric
    "LOG",  # Log
    "LMT",  # Limit
    "FLW",  # Flow
    "CND",  # Candidate
    "INT",  # Integrity
    "PRV",  # Provenance
    "CNS",  # Constraint
    # World-model concerns
    "USR",  # User state
    "ENV",  # Environment
    "PRJ",  # Project
    "HAB",  # Habit/pattern
    "TSK",  # Task
    # Ground truth concerns (from MTH-ORG-001)
    "ENM",  # Enumeration
    "PRM",  # Permission
    "GOL",  # Goal
    "TMO",  # Timeout
    "LCK",  # Lock
    "RTY",  # Retry
    "TRF",  # Transform
    "COL",  # Collection
    "WRT",  # Write
    "EMT",  # Emit
    "RED",  # Read
    "ALT",  # Alert
    "CFG",  # Config
    "TRC",  # Trace
})

SCOPES = (
    "ECO",  # Ecosystem — index 0
    "APP",  # Application — index 1
    "DOM",  # Domain — index 2
    "FET",  # Feature — index 3
    "CMP",  # Component — index 4
    "FNC",  # Function — index 5
    "STP",  # Step — index 6
    "OPR",  # Operation — index 7
    "EXP",  # Expression — index 8
    "VAL",  # Value — index 9
)

SCOPE_SET = frozenset(SCOPES)

SCOPE_DEPTH = {s: i for i, s in enumerate(SCOPES)}

DIMENSIONS = frozenset({
    "WHAT",      # Identity, definition
    "HOW",       # Method, process
    "WHY",       # Purpose, rationale
    "WHO",       # Actor, responsibility
    "WHEN",      # Timing, sequence
    "WHERE",     # Location, scope
    "IF",        # Condition, constraint
    "HOW_MUCH",  # Quantity, cost
})

# Domain is extensible — these are the initial set
DOMAINS = frozenset({
    "SFT",  # Software
    "ORG",  # Organization
    "COG",  # Cognitive
    "NET",  # Network
    "ECN",  # Economics
    "PHY",  # Physical
    "SOC",  # Social
    "EDU",  # Education
    "MED",  # Medical
    "LGL",  # Legal
})


@dataclass(frozen=True)
class Postcode:
    """Parsed 5-axis coordinate."""
    layer: str
    concern: str
    scope: str
    dimension: str
    domain: str

    @property
    def depth(self) -> int:
        """Scope depth: ECO=0, VAL=9."""
        return SCOPE_DEPTH.get(self.scope, -1)

    @property
    def key(self) -> str:
        """Reconstruct the postcode string."""
        return f"{self.layer}.{self.concern}.{self.scope}.{self.dimension}.{self.domain}"

    def parent_scope(self) -> Optional[str]:
        """Return the scope one level up, or None if at ECO."""
        idx = SCOPE_DEPTH.get(self.scope, 0)
        if idx <= 0:
            return None
        return SCOPES[idx - 1]

    def child_scope(self) -> Optional[str]:
        """Return the scope one level down, or None if at VAL."""
        idx = SCOPE_DEPTH.get(self.scope, 9)
        if idx >= 9:
            return None
        return SCOPES[idx + 1]

    def __str__(self) -> str:
        return self.key


def parse_postcode(raw: str) -> Postcode:
    """Parse a 5-axis postcode string into a Postcode object.

    Format: LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN
    Example: INT.SEM.ECO.WHY.ORG

    Raises ValueError on invalid format or unknown axis values.
    """
    parts = raw.strip().split(".")
    if len(parts) != 5:
        raise ValueError(
            f"Postcode must have exactly 5 axes (got {len(parts)}): {raw!r}"
        )

    layer, concern, scope, dimension, domain = parts

    if layer not in LAYERS:
        raise ValueError(f"Unknown layer {layer!r} in postcode {raw!r}")
    if concern not in CONCERNS:
        raise ValueError(f"Unknown concern {concern!r} in postcode {raw!r}")
    if scope not in SCOPE_SET:
        raise ValueError(f"Unknown scope {scope!r} in postcode {raw!r}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unknown dimension {dimension!r} in postcode {raw!r}")
    # Domain is extensible — accept any 2-4 char uppercase string
    if not (2 <= len(domain) <= 4 and domain.isalpha() and domain.isupper()):
        raise ValueError(
            f"Domain must be 2-4 uppercase letters, got {domain!r} in postcode {raw!r}"
        )

    return Postcode(
        layer=layer,
        concern=concern,
        scope=scope,
        dimension=dimension,
        domain=domain,
    )


@dataclass(frozen=True)
class Cell:
    """The atomic unit of the semantic map.

    A Cell is a position in coordinate space that can hold content.
    Everything else (agent logs, run history, locks) belongs to the
    storage layer, not the kernel.
    """
    postcode: Postcode                              # 5-axis address
    primitive: str                                   # what lives here (name)
    content: str = ""                                # description
    fill: FillState = FillState.E                    # fill state
    confidence: float = 0.0                          # 0.0 - 1.0
    connections: tuple[str, ...] = ()                # postcodes of connected cells
    parent: Optional[str] = None                     # postcode of parent cell
    source: tuple[str, ...] = ()                     # provenance refs
    proposer: str = ""                                # agent that proposed this content (AX4)
    # Revision history — only populated on re-fill (AX3 FEEDBACK)
    revisions: tuple[tuple[str, float], ...] = ()    # (previous_content, previous_confidence)

    def __post_init__(self):
        # Clamp confidence
        if not (0.0 <= self.confidence <= 1.0):
            object.__setattr__(
                self, "confidence",
                max(0.0, min(1.0, self.confidence))
            )

    @property
    def is_filled(self) -> bool:
        return self.fill in (FillState.F, FillState.P)

    @property
    def is_empty(self) -> bool:
        return self.fill == FillState.E

    @property
    def is_blocked(self) -> bool:
        return self.fill == FillState.B

    @property
    def is_quarantined(self) -> bool:
        return self.fill == FillState.Q

    @property
    def is_candidate(self) -> bool:
        return self.fill == FillState.C

    @property
    def depth(self) -> int:
        return self.postcode.depth

    def nav_line(self) -> str:
        """Lightweight navigation representation (3-line format)."""
        state = self.fill.name
        conns = ", ".join(self.connections) if self.connections else ""
        line = f"{self.postcode.key} | {state} {self.confidence:.2f} | {self.primitive}"
        if conns:
            line += f"\n  -> {conns}"
        return line
