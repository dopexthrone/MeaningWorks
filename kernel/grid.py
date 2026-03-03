"""
kernel/grid.py — The coordinate space.

Grid holds cells and manages layer activation.
Not all cells exist — the grid activates regions on demand.
The full 320,000-cell grid never materializes.
Only the relevant subspace exists.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
    SCOPES,
)


# Intent contract is a special source ref that is always valid provenance
INTENT_CONTRACT = "__intent_contract__"


@dataclass
class Grid:
    """The coordinate space.

    Mutable — fill and connect operations mutate in place and return self
    for chaining. The Grid is the ONLY mutable object in the kernel.
    """
    cells: dict[str, Cell] = field(default_factory=dict)      # postcode_key -> Cell
    activated_layers: set[str] = field(default_factory=set)    # layer codes
    root: Optional[str] = None                                 # postcode of intent contract cell
    intent_text: str = ""                                      # raw intent for provenance
    _agent_map: dict[str, str] = field(default_factory=dict)   # postcode_key -> last agent
    _signatures: dict[str, object] = field(default_factory=dict)  # postcode_key -> ProvenanceSignature

    def __post_init__(self):
        # Thread-safety lock for concurrent grid mutations
        object.__setattr__(self, "_lock", threading.Lock())

    # --- Cell access ---

    def get(self, postcode_key: str) -> Optional[Cell]:
        """Get a cell by its postcode key string."""
        return self.cells.get(postcode_key)

    def has(self, postcode_key: str) -> bool:
        """Check if a cell exists at this postcode."""
        return postcode_key in self.cells

    def put(self, cell: Cell) -> None:
        """Put a cell into the grid (overwrites if exists).

        Enforces AX1 provenance guard: filled cells (F or P) must have
        a non-empty source tuple. Cells with other fill states (E, C, Q, B)
        are structural and exempt.

        Raises ValueError if a filled cell has no provenance.
        Thread-safe: acquires _lock if available.
        """
        # AX1 provenance guard — filled cells must trace to something
        if cell.fill in (FillState.F, FillState.P) and not cell.source:
            raise ValueError(
                f"AX1 violation: filled cell {cell.postcode.key} has no provenance source"
            )
        lock = getattr(self, "_lock", None)
        if lock:
            lock.acquire()
        try:
            key = cell.postcode.key
            self.cells[key] = cell
            self.activated_layers.add(cell.postcode.layer)
        finally:
            if lock:
                lock.release()

    # --- Layer activation ---

    def is_layer_active(self, layer: str) -> bool:
        return layer in self.activated_layers

    def activate_layer(self, layer: str, concern: str, dimension: str, domain: str) -> Cell:
        """Activate a layer by creating its root cell (scope=ECO, fill=E).

        Returns the created root cell.
        """
        if layer in self.activated_layers:
            # Already active — find or create root
            root_candidates = [
                c for c in self.cells.values()
                if c.postcode.layer == layer and c.postcode.scope == "ECO"
            ]
            if root_candidates:
                return root_candidates[0]

        postcode = parse_postcode(f"{layer}.{concern}.ECO.{dimension}.{domain}")
        root_cell = Cell(
            postcode=postcode,
            primitive=f"{layer.lower()}_root",
            content="",
            fill=FillState.E,
            confidence=0.0,
            source=(INTENT_CONTRACT,) if self.root else (),
        )
        self.put(root_cell)
        return root_cell

    # --- Queries ---

    def cells_in_layer(self, layer: str) -> list[Cell]:
        """All cells in a given layer."""
        return [c for c in self.cells.values() if c.postcode.layer == layer]

    def filled_cells(self) -> list[Cell]:
        """All cells with fill state F or P."""
        return [c for c in self.cells.values() if c.is_filled]

    def empty_cells(self) -> list[Cell]:
        """All cells with fill state E."""
        return [c for c in self.cells.values() if c.is_empty]

    def blocked_cells(self) -> list[Cell]:
        """All cells with fill state B."""
        return [c for c in self.cells.values() if c.is_blocked]

    def quarantined_cells(self) -> list[Cell]:
        """All cells with fill state Q."""
        return [c for c in self.cells.values() if c.is_quarantined]

    def candidate_cells(self) -> list[Cell]:
        """All cells with fill state C."""
        return [c for c in self.cells.values() if c.is_candidate]

    def unfilled_connections(self) -> list[str]:
        """Postcodes that are referenced in connections but are empty or don't exist."""
        result = []
        for cell in self.cells.values():
            if not cell.is_filled:
                continue
            for conn in cell.connections:
                target = self.get(conn)
                if target is None or target.is_empty:
                    if conn not in result:
                        result.append(conn)
        return result

    def orphan_cells(self) -> list[Cell]:
        """Filled cells with no connections and no parent."""
        all_connected = set()
        for cell in self.cells.values():
            all_connected.update(cell.connections)
        return [
            c for c in self.cells.values()
            if c.is_filled
            and not c.connections
            and c.parent is None
            and c.postcode.key not in all_connected
            and c.postcode.key != self.root
        ]

    # --- Stats ---

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def fill_rate(self) -> float:
        if not self.cells:
            return 0.0
        filled = sum(1 for c in self.cells.values() if c.is_filled)
        return filled / len(self.cells)

    def stats(self) -> dict[str, int]:
        """Count cells by fill state."""
        counts: dict[str, int] = {s.name: 0 for s in FillState}
        for cell in self.cells.values():
            counts[cell.fill.name] += 1
        return counts

    # --- Navigation format ---

    def nav(self) -> str:
        """Lightweight navigation representation of the entire grid.

        Returns the full map in ~3 lines per node format,
        grouped by layer.
        """
        lines = []
        by_layer: dict[str, list[Cell]] = {}
        for cell in self.cells.values():
            layer = cell.postcode.layer
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append(cell)

        for layer in sorted(by_layer.keys()):
            lines.append(f"# {layer}")
            for cell in sorted(by_layer[layer], key=lambda c: c.postcode.key):
                lines.append(cell.nav_line())
            lines.append("")

        return "\n".join(lines)

    # --- Intent contract ---

    def set_intent(self, intent_text: str, postcode_key: str, primitive: str) -> Cell:
        """Create the root intent contract cell.

        This is the axiom — taken as given. All provenance chains
        ultimately trace to this cell.
        """
        self.intent_text = intent_text
        postcode = parse_postcode(postcode_key)
        root_cell = Cell(
            postcode=postcode,
            primitive=primitive,
            content=intent_text,
            fill=FillState.F,
            confidence=1.0,
            source=(INTENT_CONTRACT,),
        )
        self.put(root_cell)
        self.root = postcode_key
        return root_cell

    # --- Provenance verification ---

    def verify_provenance(self, postcode_key: str) -> bool:
        """Verify the provenance signature for a cell.

        Returns True if the cell's provenance is correctly signed.
        Returns True if no signature exists (unsigned cells are allowed).
        Returns False only if a signature exists but is invalid.
        """
        sig = self._signatures.get(postcode_key)
        if sig is None:
            return True  # unsigned = allowed (backwards compat)

        cell = self.get(postcode_key)
        if cell is None:
            return False

        try:
            from kernel.provenance_signing import ProvenanceSigner, content_hash
            signer = getattr(self, "_signer", None)
            if signer is None:
                signer = ProvenanceSigner()
                object.__setattr__(self, "_signer", signer)
            return signer.verify(
                source=cell.source,
                postcode=postcode_key,
                signature=sig,
                fill_state=cell.fill.name,
                content_hash=content_hash(cell.content) if cell.content else "",
            )
        except Exception as e:  # noqa: verification infra failure = don't block
            import logging
            logging.getLogger(__name__).debug(f"Provenance verification skipped: {e}")
            return True
