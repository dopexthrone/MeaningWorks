"""
kernel/navigator.py — Next-cell prediction, descent, emergence detection.

The navigator decides WHERE to fill next. This is the token-prediction
analogy: each fill conditions the next. The connection graph is the
attention mask.

Three functions:
  next_cell(grid) → postcode | None
  descend(grid, postcode) → list of child postcodes created
  detect_emergence(grid) → list of candidate cells created
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
    SCOPES,
    SCOPE_DEPTH,
)
from kernel.grid import Grid
from kernel._text_utils import semantic_jaccard, normalize_tokens, expand_synonyms


# ---------------------------------------------------------------------------
# Scoring weights for next-cell prediction
# ---------------------------------------------------------------------------

# Priority 1: Unfilled connections from filled cells
# These are the "next tokens" — most probable continuations.
_WEIGHT_UNFILLED_CONN = 10.0

# Priority 2: Cross-layer gaps (active layer, empty root)
_WEIGHT_CROSS_LAYER_GAP = 8.0

# Priority 3: Low-confidence neighbor — uncertainty propagates, resolve first
_WEIGHT_LOW_CONF_NEIGHBOR = 5.0

# Priority 4: Depth pressure — push downward when current scope is saturated
_WEIGHT_DEPTH_PRESSURE = 3.0

# Bonus: cell has a parent that is filled (descent-ready)
_WEIGHT_PARENT_FILLED = 2.0

# Penalty: deeper scope = harder to fill accurately
_PENALTY_DEPTH = 0.5


# ---------------------------------------------------------------------------
# Convergence thresholds
# ---------------------------------------------------------------------------

# A cell doesn't need children if confidence >= this
CONFIDENCE_THRESHOLD = 0.95

# Minimum fill rate before navigator considers the grid "converged"
MIN_CONVERGENCE_FILL_RATE = 0.80

# Minimum filled cells before emergence detection kicks in
MIN_FILLS_FOR_EMERGENCE = 5

# Minimum times a primitive must appear to be a candidate pattern
MIN_PATTERN_OCCURRENCE = 2


# ---------------------------------------------------------------------------
# Next-cell prediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoredCell:
    """A candidate cell with its navigation score."""
    postcode_key: str
    score: float
    reason: str


def next_cell(grid: Grid) -> Optional[str]:
    """Predict the next cell to fill.

    Scores all unfilled cells and returns the postcode of the
    highest-scored candidate. Returns None if the grid is converged
    (no good candidates remain).

    Scoring:
      1. Unfilled connections from filled cells (highest weight)
      2. Cross-layer gaps (active layer with empty root)
      3. Low-confidence neighbors (uncertainty propagates)
      4. Depth pressure (push downward when scope is saturated)
    """
    candidates = score_candidates(grid)
    if not candidates:
        return None
    return candidates[0].postcode_key


def score_candidates(grid: Grid) -> list[ScoredCell]:
    """Score all unfillable cells and return sorted by score (descending)."""
    scores: dict[str, float] = defaultdict(float)
    reasons: dict[str, list[str]] = defaultdict(list)

    # --- Priority 1: Unfilled connections from filled cells ---
    for cell in grid.filled_cells():
        for conn_key in cell.connections:
            target = grid.get(conn_key)
            if target is None or target.is_empty or target.is_candidate:
                scores[conn_key] += _WEIGHT_UNFILLED_CONN
                reasons[conn_key].append(f"unfilled_conn_from:{cell.postcode.key}")

    # --- Priority 2: Cross-layer gaps ---
    for layer in grid.activated_layers:
        layer_cells = grid.cells_in_layer(layer)
        eco_cells = [c for c in layer_cells if c.postcode.scope == "ECO"]
        if eco_cells and all(c.is_empty for c in eco_cells):
            # Active layer with empty root = critical gap
            for c in eco_cells:
                scores[c.postcode.key] += _WEIGHT_CROSS_LAYER_GAP
                reasons[c.postcode.key].append("cross_layer_gap")

    # --- Priority 3: Low-confidence neighbors ---
    for cell in grid.cells.values():
        if cell.is_filled and cell.confidence < 0.85:
            # This cell's connections may need attention
            for conn_key in cell.connections:
                target = grid.get(conn_key)
                if target and (target.is_empty or target.is_candidate):
                    bonus = _WEIGHT_LOW_CONF_NEIGHBOR * (1.0 - cell.confidence)
                    scores[conn_key] += bonus
                    reasons[conn_key].append(f"low_conf_neighbor:{cell.confidence:.2f}")

    # --- Priority 4: Depth pressure ---
    _apply_depth_pressure(grid, scores, reasons)

    # --- Bonus: parent filled ---
    for key, _ in scores.items():
        cell = grid.get(key)
        if cell and cell.parent:
            parent = grid.get(cell.parent)
            if parent and parent.is_filled:
                scores[key] += _WEIGHT_PARENT_FILLED
                reasons[key].append("parent_filled")

    # --- Penalty: depth ---
    for key in list(scores.keys()):
        try:
            pc = parse_postcode(key)
            scores[key] -= pc.depth * _PENALTY_DEPTH
        except ValueError:
            pass

    # --- Penalty: heavily-revised cells (instability signal) ---
    for key in list(scores.keys()):
        cell = grid.get(key)
        if cell and len(cell.revisions) > 3:
            penalty = (len(cell.revisions) - 3) * 2.0
            scores[key] -= penalty
            reasons[key].append(f"revision_penalty:{len(cell.revisions)}_revisions")

    # Filter: only cells that are empty, candidate, or don't exist yet
    fillable = {}
    for key, score in scores.items():
        cell = grid.get(key)
        if cell is None or cell.is_empty or cell.is_candidate:
            fillable[key] = score

    # Sort descending by score
    sorted_candidates = sorted(fillable.items(), key=lambda x: x[1], reverse=True)

    return [
        ScoredCell(
            postcode_key=key,
            score=score,
            reason="; ".join(reasons.get(key, [])),
        )
        for key, score in sorted_candidates
        if score > 0
    ]


def _apply_depth_pressure(
    grid: Grid,
    scores: dict[str, float],
    reasons: dict[str, list[str]],
) -> None:
    """When a scope level is saturated, push toward deeper cells."""
    # Group filled cells by layer+scope
    layer_scope_counts: dict[tuple[str, str], int] = defaultdict(int)
    layer_scope_total: dict[tuple[str, str], int] = defaultdict(int)

    for cell in grid.cells.values():
        key = (cell.postcode.layer, cell.postcode.scope)
        layer_scope_total[key] += 1
        if cell.is_filled:
            layer_scope_counts[key] += 1

    # For each layer where ECO is filled, check if children exist
    for cell in grid.filled_cells():
        if cell.postcode.scope == "ECO" and cell.confidence < CONFIDENCE_THRESHOLD:
            child_scope = cell.postcode.child_scope()
            if child_scope is None:
                continue
            # Check if any children exist at the next scope
            child_key = (cell.postcode.layer, child_scope)
            if layer_scope_total.get(child_key, 0) == 0:
                # No children yet — depth pressure to create them
                # We score a hypothetical child postcode
                hyp_key = f"{cell.postcode.layer}.{cell.postcode.concern}.{child_scope}.{cell.postcode.dimension}.{cell.postcode.domain}"
                scores[hyp_key] += _WEIGHT_DEPTH_PRESSURE
                reasons[hyp_key].append(f"depth_pressure_from:{cell.postcode.key}")


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def is_converged(grid: Grid) -> bool:
    """Check if the grid has converged (no more useful fills possible).

    Converged when:
      1. No unfilled connections from filled cells, AND
      2. No cross-layer gaps, AND
      3. All filled cells have confidence >= threshold OR
         fill rate exceeds minimum convergence rate
    """
    # Check 1: unfilled connections (includes candidates — they need promotion)
    for cell in grid.filled_cells():
        for conn_key in cell.connections:
            target = grid.get(conn_key)
            if target is None or target.is_empty or target.is_candidate:
                return False

    # Check 2: cross-layer gaps
    for layer in grid.activated_layers:
        eco_cells = [
            c for c in grid.cells_in_layer(layer)
            if c.postcode.scope == "ECO"
        ]
        if eco_cells and all(c.is_empty for c in eco_cells):
            return False

    # Check 2.5: unfilled children from descent
    # If descend() created child cells, they must be filled before convergence.
    for cell in grid.cells.values():
        if cell.parent is not None and (cell.is_empty or cell.is_candidate):
            return False

    # Check 3: sufficient confidence or fill rate
    filled = grid.filled_cells()
    if not filled:
        return False

    avg_confidence = sum(c.confidence for c in filled) / len(filled)
    if avg_confidence >= CONFIDENCE_THRESHOLD:
        return True

    if grid.fill_rate >= MIN_CONVERGENCE_FILL_RATE:
        return True

    return False


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------

def should_descend(cell: Cell) -> bool:
    """Determine if a cell needs children at the next scope level.

    A cell needs children if:
      - It is filled (F or P)
      - Confidence < CONFIDENCE_THRESHOLD
      - It is not at VAL scope (deepest level)
    """
    if not cell.is_filled:
        return False
    if cell.confidence >= CONFIDENCE_THRESHOLD:
        return False
    if cell.postcode.child_scope() is None:
        return False
    return True


def _relevant_dimensions(grid: Grid, layer: str, concern: str, domain: str, own_dim: str) -> list[str]:
    """Find dimensions worth descending into.

    Returns the parent's own dimension plus any dimensions that already have
    filled cells in this layer+concern at any scope. This prevents the 8x
    dimensional fan-out where every concept spawns children for all 8 dimensions.
    """
    dims = {own_dim}
    for cell in grid.cells.values():
        if (cell.is_filled
                and cell.postcode.layer == layer
                and cell.postcode.concern == concern
                and cell.postcode.domain == domain):
            dims.add(cell.postcode.dimension)
    return sorted(dims)


def descend(grid: Grid, postcode_key: str) -> list[str]:
    """Create child cells at the next scope level for the given cell.

    Returns list of postcode keys for the created children.
    Children are created as Empty cells with parent set to the given cell.

    Only creates children for relevant dimensions — the parent's own dimension
    plus dimensions that already have filled cells in the same layer+concern.
    This prevents the 8x dimensional fan-out that produces noise.
    """
    cell = grid.get(postcode_key)
    if cell is None:
        return []

    if not should_descend(cell):
        return []

    child_scope = cell.postcode.child_scope()
    if child_scope is None:
        return []

    pc = cell.postcode
    created = []

    # Only create children for dimensions that are relevant
    dims = _relevant_dimensions(grid, pc.layer, pc.concern, pc.domain, pc.dimension)
    for dim in dims:
        child_key = f"{pc.layer}.{pc.concern}.{child_scope}.{dim}.{pc.domain}"
        if not grid.has(child_key):
            child_pc = parse_postcode(child_key)
            child_cell = Cell(
                postcode=child_pc,
                primitive="",
                content="",
                fill=FillState.E,
                confidence=0.0,
                parent=postcode_key,
                source=(postcode_key,),
            )
            grid.put(child_cell)
            created.append(child_key)

    return created


def descend_selective(
    grid: Grid,
    postcode_key: str,
    dimensions: tuple[str, ...],
) -> list[str]:
    """Create child cells only for specific dimensions.

    Like descend() but only for the given dimensions.
    Use when the navigator knows which dimensions are relevant.
    """
    cell = grid.get(postcode_key)
    if cell is None:
        return []

    if not should_descend(cell):
        return []

    child_scope = cell.postcode.child_scope()
    if child_scope is None:
        return []

    pc = cell.postcode
    created = []

    for dim in dimensions:
        child_key = f"{pc.layer}.{pc.concern}.{child_scope}.{dim}.{pc.domain}"
        if not grid.has(child_key):
            child_pc = parse_postcode(child_key)
            child_cell = Cell(
                postcode=child_pc,
                primitive="",
                content="",
                fill=FillState.E,
                confidence=0.0,
                parent=postcode_key,
                source=(postcode_key,),
            )
            grid.put(child_cell)
            created.append(child_key)

    return created


# ---------------------------------------------------------------------------
# Emergence detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmergenceSignal:
    """A detected pattern that may warrant a new node."""
    signal_type: str       # "repeated_primitive", "shared_connection", "orphan_cluster"
    evidence: tuple[str, ...]  # postcode keys involved
    primitive: str         # suggested primitive name
    description: str       # what was detected


def detect_emergence(grid: Grid) -> list[EmergenceSignal]:
    """Scan the grid for emergent patterns.

    Detects:
      1. Repeated primitives — same primitive at multiple postcodes
      2. Shared connections — two cells in different layers connect to same target
      3. Orphan clusters — filled cells with no connections

    Only runs after MIN_FILLS_FOR_EMERGENCE fills.
    """
    filled = grid.filled_cells()
    if len(filled) < MIN_FILLS_FOR_EMERGENCE:
        return []

    signals: list[EmergenceSignal] = []

    # --- Pattern 1: Semantically similar primitives ---
    # Group primitives by semantic similarity (Jaccard > 0.5 on stemmed tokens)
    primitive_groups = _semantic_group_primitives(filled)
    for group_label, locations in primitive_groups.items():
        if len(locations) >= MIN_PATTERN_OCCURRENCE:
            layers = {parse_postcode(k).layer for k in locations}
            if len(layers) > 1:
                signals.append(EmergenceSignal(
                    signal_type="repeated_primitive",
                    evidence=tuple(locations),
                    primitive=f"emerged_{group_label}",
                    description=f"Semantic group '{group_label}' at {len(locations)} postcodes across {len(layers)} layers",
                ))

    # --- Pattern 2: Shared connections ---
    # Two cells in different layers that both connect to the same target
    target_sources: dict[str, list[str]] = defaultdict(list)
    for cell in filled:
        for conn in cell.connections:
            target_sources[conn].append(cell.postcode.key)

    for target, sources in target_sources.items():
        if len(sources) >= 2:
            source_layers = {parse_postcode(s).layer for s in sources}
            if len(source_layers) > 1:
                signals.append(EmergenceSignal(
                    signal_type="shared_connection",
                    evidence=tuple(sources) + (target,),
                    primitive=f"hub_{target.split('.')[0].lower()}",
                    description=f"Target '{target}' connected from {len(sources)} cells across {len(source_layers)} layers",
                ))

    # --- Pattern 3: Orphan clusters ---
    orphans = grid.orphan_cells()
    if len(orphans) >= 2:
        # Multiple orphans = potential cluster
        orphan_layers = {c.postcode.layer for c in orphans}
        signals.append(EmergenceSignal(
            signal_type="orphan_cluster",
            evidence=tuple(c.postcode.key for c in orphans),
            primitive="orphan_cluster",
            description=f"{len(orphans)} disconnected cells across {len(orphan_layers)} layers",
        ))

    return signals


# ---------------------------------------------------------------------------
# World-model-aware scoring (Phase 4)
# ---------------------------------------------------------------------------

_WEIGHT_STALENESS = 6.0           # stale observation needs refresh
_WEIGHT_USER_FACING = 7.0         # USR/ENV concerns get priority
_WEIGHT_ACTIVE_PROJECT = 9.0      # cells connected to active project
_PENALTY_RECENTLY_CHECKED = -4.0  # suppress re-checking fresh cells

_WORLD_USER_CONCERNS = frozenset({"USR", "ENV"})


def score_world_candidates(
    grid: Grid,
    staleness_map: dict[str, float] | None = None,
    active_project: str | None = None,
    recently_filled: frozenset[str] = frozenset(),
) -> list[ScoredCell]:
    """Score candidates with world-model-aware weights.

    Calls existing score_candidates() internally, then adds:
    - Staleness pressure (observation cells past their half-life)
    - User-facing priority (USR/ENV concerns scored higher)
    - Active project boost
    - Recently-checked penalty (cooldown)

    Args:
        grid: The world grid to score.
        staleness_map: postcode_key → seconds since last observation.
            If None, staleness scoring is skipped.
        active_project: Project identifier. Cells connected to this
            project get a boost.
        recently_filled: Set of postcode keys filled recently (cooldown).

    Returns:
        List of ScoredCell sorted by score descending.
    """
    # Start with base scores from existing navigator
    base = score_candidates(grid)
    scores: dict[str, float] = {sc.postcode_key: sc.score for sc in base}
    reasons: dict[str, list[str]] = {
        sc.postcode_key: [sc.reason] for sc in base
    }

    # Also consider filled cells that might need refreshing
    # (existing score_candidates only looks at unfilled cells)
    for cell in grid.cells.values():
        if cell.postcode.key not in scores:
            scores[cell.postcode.key] = 0.0
            reasons[cell.postcode.key] = []

    # --- Staleness pressure ---
    if staleness_map:
        for pk, age_seconds in staleness_map.items():
            cell = grid.get(pk)
            if cell is None:
                continue
            # Stale filled OBS cells need refreshing
            if cell.postcode.layer == "OBS" and cell.is_filled:
                # Normalize: 1.0 at 1 half-life, 2.0 at 2 half-lives
                staleness_score = _WEIGHT_STALENESS * min(age_seconds / 60.0, 3.0)
                scores[pk] = scores.get(pk, 0.0) + staleness_score
                reasons.setdefault(pk, []).append(f"stale:{age_seconds:.0f}s")

    # --- User-facing priority ---
    for pk in list(scores.keys()):
        try:
            pc = parse_postcode(pk)
        except ValueError:
            continue
        if pc.concern in _WORLD_USER_CONCERNS:
            scores[pk] += _WEIGHT_USER_FACING
            reasons.setdefault(pk, []).append("user_facing")

    # --- Active project boost ---
    if active_project:
        project_cells: set[str] = set()
        for cell in grid.cells.values():
            for s in cell.source:
                if active_project in s:
                    project_cells.add(cell.postcode.key)
                    # Also boost connected cells
                    for conn in cell.connections:
                        project_cells.add(conn)

        for pk in project_cells:
            if pk in scores:
                scores[pk] += _WEIGHT_ACTIVE_PROJECT
                reasons.setdefault(pk, []).append(f"active_project:{active_project}")

    # --- Recently-checked penalty ---
    for pk in recently_filled:
        if pk in scores:
            scores[pk] += _PENALTY_RECENTLY_CHECKED
            reasons.setdefault(pk, []).append("recently_checked")

    # Filter out cells that are already well-filled AND not stale
    # (world model cares about filled cells that need refresh too)
    result: list[ScoredCell] = []
    for pk, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score <= 0:
            continue
        result.append(ScoredCell(
            postcode_key=pk,
            score=score,
            reason="; ".join(reasons.get(pk, [])),
        ))

    return result


def _semantic_group_primitives(cells: list[Cell]) -> dict[str, list[str]]:
    """Group cells by semantic similarity of their primitives.

    Uses synonym-expanded stemmed-token Jaccard > 0.4. Returns dict mapping
    group label (first primitive in group) to list of postcode keys.
    """
    prims_with_keys: list[tuple[str, str]] = []
    for cell in cells:
        if cell.primitive:
            prims_with_keys.append((cell.primitive, cell.postcode.key))

    if not prims_with_keys:
        return {}

    # Pre-compute expanded token sets for each primitive
    token_cache: dict[str, set[str]] = {}
    for prim, _ in prims_with_keys:
        if prim not in token_cache:
            token_cache[prim] = expand_synonyms(normalize_tokens(prim))

    def _similarity(a: str, b: str) -> float:
        ta = token_cache.get(a, set())
        tb = token_cache.get(b, set())
        if not ta or not tb:
            return 0.0
        intersection = ta & tb
        union = ta | tb
        return len(intersection) / len(union) if union else 0.0

    # Greedy clustering
    assigned: dict[str, str] = {}
    groups: dict[str, list[str]] = {}

    for i, (prim_a, key_a) in enumerate(prims_with_keys):
        if key_a in assigned:
            continue
        label = prim_a
        group = [key_a]
        assigned[key_a] = label

        for j in range(i + 1, len(prims_with_keys)):
            prim_b, key_b = prims_with_keys[j]
            if key_b in assigned:
                continue
            if prim_a == prim_b or _similarity(prim_a, prim_b) > 0.4:
                group.append(key_b)
                assigned[key_b] = label

        groups[label] = group

    return groups


def promote_emergence(
    grid: Grid,
    signal: EmergenceSignal,
    target_postcode_key: str,
) -> Cell:
    """Create a candidate cell from an emergence signal.

    The candidate cell is placed at target_postcode_key with fill=C.
    It must be validated and promoted to F/P by the EMERGENCE agent.
    """
    pc = parse_postcode(target_postcode_key)
    candidate = Cell(
        postcode=pc,
        primitive=signal.primitive,
        content=signal.description,
        fill=FillState.C,
        confidence=0.0,
        connections=signal.evidence,
        source=signal.evidence,
    )
    grid.put(candidate)
    return candidate
