"""
kernel/agents.py — Six grid operators that form the compilation pipeline.

MEMORY    → Bootstrap from history (pre-fill at reduced confidence)
AUTHOR    → Fill cells from input (the ONLY agent that touches an LLM)
VERIFIER  → Enforce AX1 provenance (promote or quarantine)
OBSERVER  → Detect cross-cell patterns (wraps detect_emergence)
EMERGENCE → Promote candidates to filled nodes
GOVERNOR  → Simulation gate (conflict, gap, cycle, dead-end)

Each agent is a function: grid → grid (mutates in place, returns same grid).
The pipeline loops AUTHOR→VERIFIER→OBSERVER→EMERGENCE→GOVERNOR until
GOVERNOR passes or max iterations reached.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
    LAYERS,
    SCOPES,
)
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, block, FillStatus
from kernel.navigator import (
    next_cell,
    score_candidates,
    descend,
    detect_emergence,
    promote_emergence,
    is_converged,
    EmergenceSignal,
)


# ---------------------------------------------------------------------------
# LLM interface — injectable function type
# ---------------------------------------------------------------------------

class LLMFunction(Protocol):
    """Protocol for the LLM call used by AUTHOR.

    Takes a prompt string, returns structured extraction:
    list of dicts with keys: postcode, primitive, content, confidence, connections.
    """
    def __call__(self, prompt: str) -> list[dict]: ...


# ---------------------------------------------------------------------------
# MEMORY agent
# ---------------------------------------------------------------------------

def memory(grid: Grid, previous_grids: list[Grid], confidence_decay: float = 0.7) -> Grid:
    """Bootstrap from history.

    For each filled cell in previous grids that shares a domain with
    the current grid's intent, pre-fill at reduced confidence.

    confidence_decay: multiplier applied to historical confidence (default 0.7).
    A cell from history at 0.95 becomes 0.665 in the new grid.

    Only imports cells that don't already exist in the current grid.
    """
    if not previous_grids:
        return grid

    for prev_grid in previous_grids:
        # Carry forward ALL activated layers from history
        for layer in prev_grid.activated_layers:
            if layer not in grid.activated_layers:
                grid.activated_layers.add(layer)

        for cell in prev_grid.filled_cells():
            key = cell.postcode.key
            # Don't overwrite existing cells
            if grid.has(key):
                continue

            # Import at reduced confidence with stability factor
            # Heavily-revised cells imported at lower confidence
            stability = 1.0 / (1 + len(cell.revisions))
            decayed_conf = cell.confidence * confidence_decay * stability
            fill(
                grid,
                key,
                cell.primitive,
                cell.content,
                decayed_conf,
                connections=cell.connections,
                source=("memory:" + (prev_grid.root or "unknown"),),
                parent=cell.parent,
                agent="memory",
            )

    return grid


# ---------------------------------------------------------------------------
# AUTHOR agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthorExtraction:
    """A concept extracted by the AUTHOR from input text."""
    postcode: str
    primitive: str
    content: str
    confidence: float
    connections: tuple[str, ...] = ()


def _depth_coverage_block(grid: Grid) -> str:
    """Return per-scope cell counts for prompt injection."""
    scope_counts: dict[str, int] = {}
    for c in grid.cells.values():
        s = c.postcode.scope
        scope_counts[s] = scope_counts.get(s, 0) + 1
    return chr(10).join(f"  {s}: {scope_counts.get(s, 0)} cells" for s in SCOPES)


def _extraction_range(target_scopes: tuple[str, ...]) -> str:
    """Return extraction count range based on scope depth."""
    if target_scopes:
        deep_scopes = {"CMP", "FNC", "STP", "OPR", "EXP", "VAL"}
        if any(s in deep_scopes for s in target_scopes):
            return "10-25"
    return "5-15"


def _build_author_prompt(grid: Grid, input_text: str, target_scopes: tuple[str, ...] = ()) -> str:
    """Build the prompt for the LLM to extract concepts.

    Includes: the input text, current grid nav (what's already filled),
    and instructions for structured extraction.

    target_scopes: if non-empty, instructs the LLM to focus on these scope
    levels (e.g., ("CMP", "FNC", "STP") for technical primitives).
    """
    nav = grid.nav()
    filled_count = len(grid.filled_cells())
    empty_targets = grid.unfilled_connections()
    # Also include empty descendant cells as fill targets
    empty_descendants = [
        c.postcode.key for c in grid.empty_cells()
        if c.postcode.key not in empty_targets
    ]
    all_targets = empty_targets + empty_descendants[:30]  # Cap at 30 to fit context

    # Filter targets by scope when targeting is active
    if target_scopes:
        scope_set = set(target_scopes)
        filtered_targets = []
        for t in all_targets:
            parts = t.split(".")
            if len(parts) >= 3 and parts[2] in scope_set:
                filtered_targets.append(t)
        all_targets = filtered_targets if filtered_targets else all_targets[:10]

    # Scope-aware extraction guidance
    scope_guidance = ""
    if target_scopes:
        scope_names = " ".join(target_scopes)
        # Map scope depth to extraction style
        deep_scopes = {"CMP", "FNC", "STP", "OPR", "EXP", "VAL"}
        mid_scopes = {"DOM", "FET"}
        if any(s in deep_scopes for s in target_scopes):
            scope_guidance = f"""
SCOPE FOCUS: Use ONLY these scopes: {scope_names}
You are extracting TECHNICAL PRIMITIVES — the actual data structures, functions,
methods, parameters, and code-level constructs. Each extraction should name a
specific implementation artifact (e.g., "Cell-dataclass", "fill-function",
"parse-postcode", "Grid-cells-dict", "confidence-float").

CRITICAL: Each extraction must be a DIFFERENT technical artifact. Do NOT take
one concept and expand it across dimensions (WHAT/HOW/WHO/WHEN/WHERE/IF/HOW_MUCH).
That is WRONG and will be rejected. Instead, extract distinct code constructs:
  WRONG: auth-WHAT, auth-HOW, auth-WHO, auth-WHEN (same thing, 4 angles)
  RIGHT: AuthToken-class, validate-jwt-function, hash-password-method, refresh-token-endpoint"""
        elif any(s in mid_scopes for s in target_scopes):
            scope_guidance = f"""
SCOPE FOCUS: Use ONLY these scopes: {scope_names}
You are extracting ARCHITECTURAL COMPONENTS — modules, subsystems, protocols,
and interfaces. Each extraction should name a distinct structural element
(e.g., "agent-pipeline", "compilation-loop", "emission-gate", "trust-mesh-protocol").

CRITICAL: Each extraction must be a DIFFERENT component. Do NOT take one concept
and expand it across dimensions. Extract distinct subsystems and modules instead."""
        else:
            scope_guidance = f"""
SCOPE FOCUS: Prefer these scopes: {scope_names}
You are extracting high-level concepts and ecosystem relationships."""

    prompt = f"""You are a semantic compiler. Extract structured concepts from the input text and encode each as a 5-axis postcode.

INPUT TEXT:
{input_text}

CURRENT MAP STATE ({filled_count} filled cells):
{nav if nav.strip() else "(empty — this is the first pass)"}

UNFILLED TARGETS (priority — fill these with content from the input text):
{chr(10).join(all_targets[:30]) if all_targets else "(none — all cells filled)"}
{scope_guidance}
COORDINATE SCHEMA:
Postcode = LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN (exactly 5 dot-separated parts)

AXIS 1 — LAYER (what level of abstraction):
  INT SEM ORG COG AGN STR STA IDN TME EXC CTR RES OBS NET EMG MET

AXIS 2 — CONCERN (what aspect):
  SEM(Semantic) ENT(Entity) BHV(Behavior) FNC(Function) REL(Relation) PLN(Plan) MEM(Memory) ORC(Orchestration) AGT(Agent) ACT(Actor) SCO(Scope) STA(State) TRN(Transition) SNP(Snapshot) VRS(Version) SCH(Schedule) GTE(Gate) PLY(Policy) MET(Metric) LOG(Log) LMT(Limit) FLW(Flow) CND(Candidate) INT(Integrity) PRV(Provenance) CNS(Constraint)

AXIS 3 — SCOPE (how broad):
  ECO(Ecosystem) APP(Application) DOM(Domain) FET(Feature) CMP(Component) FNC(Function) STP(Step) OPR(Operation) EXP(Expression) VAL(Value)

AXIS 4 — DIMENSION (what question it answers):
  WHAT HOW WHY WHO WHEN WHERE IF HOW_MUCH

AXIS 5 — DOMAIN (knowledge domain):
  SFT(Software) ORG(Organization) COG(Cognitive) NET(Network) ECN(Economics) PHY(Physical) SOC(Social) EDU(Education) MED(Medical) LGL(Legal)

RULES:
- Each axis value MUST come from the lists above. Do NOT invent new values.
- Example valid postcodes: SEM.ENT.ECO.WHAT.SFT, COG.BHV.APP.HOW.COG, ORG.FNC.FET.WHO.SFT
- Example INVALID: SEM.APP.WHO.USR.SFT (APP is a scope, not a concern; USR is not a dimension)
- Every postcode must have exactly 5 parts from the 5 axes in order.
- Connections should reference other postcodes in the map (existing or new from this extraction).

LAYER COVERAGE:
Try to spread extractions across DIFFERENT layers. The following layers are underused
and should be prioritized if the input text contains relevant concepts:
  AGN(Agency) STR(Structure) CTR(Control) IDN(Identity) TME(Time) EXC(Exception) OBS(Observation) RES(Resource)
Not every extraction needs a different layer, but avoid putting everything in SEM or STA.

DEPTH COVERAGE (cells per scope level):
{_depth_coverage_block(grid)}
Prioritize scopes with 0 cells — they represent structural blind spots.

INSTRUCTIONS:
1. Extract {_extraction_range(target_scopes)} concepts from the input text — each must be a DISTINCT concept
2. Assign each a valid 5-axis postcode using ONLY values from the lists above
3. Rate confidence 0.0-1.0 (how certain is this extraction)
4. List connections to other postcodes
5. Every extraction MUST trace to the input text — do not hallucinate
6. Vary the LAYER and CONCERN axes — avoid repeating the same layer+concern pair

Return ONLY a JSON array:
[{{"postcode": "LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN", "primitive": "short-name", "content": "description", "confidence": 0.0-1.0, "connections": ["OTHER.POST.CODE.DIM.DOM"]}}]"""
    return prompt


def author(grid: Grid, input_text: str, llm_fn: LLMFunction, target_scopes: tuple[str, ...] = ()) -> Grid:
    """Fill cells from input text using an LLM.

    The ONLY agent that touches an LLM. All others are pure functions.

    1. Builds prompt with current grid state + input text
    2. Calls llm_fn to extract structured concepts
    3. Fills cells using fill() with AX1-AX5 enforcement
    4. Descends on filled cells where appropriate
    """
    prompt = _build_author_prompt(grid, input_text, target_scopes=target_scopes)
    extractions = llm_fn(prompt)

    if not extractions:
        return grid

    for ext in extractions:
        postcode_key = ext.get("postcode", "")
        primitive = ext.get("primitive", "")
        content = ext.get("content", "")
        confidence = float(ext.get("confidence", 0.0))
        connections = tuple(ext.get("connections", []))

        if not postcode_key or not primitive:
            continue

        # Validate postcode
        try:
            parse_postcode(postcode_key)
        except ValueError:
            continue

        # Determine source — trace to intent contract or existing cells
        source = _determine_source(grid, postcode_key, connections)

        # Determine parent — first connection that's at a higher scope
        parent = _determine_parent(grid, postcode_key, connections)

        fill(
            grid,
            postcode_key,
            primitive,
            content,
            confidence,
            connections=connections,
            source=source,
            parent=parent,
            agent="author",
        )

    # Descend on cells that need deeper exploration
    for cell in list(grid.filled_cells()):
        if cell.confidence < 0.95 and cell.postcode.child_scope() is not None:
            descend(grid, cell.postcode.key)

    return grid


def _determine_source(
    grid: Grid,
    postcode_key: str,
    connections: tuple[str, ...],
) -> tuple[str, ...]:
    """Determine provenance source for a new cell.

    Priority:
    1. Connected filled cells
    2. Intent contract
    """
    sources = []
    for conn in connections:
        target = grid.get(conn)
        if target and target.is_filled:
            sources.append(conn)

    if not sources:
        sources.append(INTENT_CONTRACT)

    return tuple(sources)


def _determine_parent(
    grid: Grid,
    postcode_key: str,
    connections: tuple[str, ...],
) -> Optional[str]:
    """Find the best parent for a new cell.

    Parent = a connected cell at a higher (shallower) scope in the same layer.
    """
    try:
        pc = parse_postcode(postcode_key)
    except ValueError:
        return None

    for conn in connections:
        target = grid.get(conn)
        if target and target.is_filled:
            if (target.postcode.layer == pc.layer and
                target.postcode.depth < pc.depth):
                return conn

    return None


# ---------------------------------------------------------------------------
# DEDUPLICATION — collapse dimensional fan-out
# ---------------------------------------------------------------------------

@dataclass
class DedupeResult:
    """Result of dimensional deduplication pass."""
    groups_found: int = 0
    cells_quarantined: int = 0


def deduplicate_dimensional(grid: Grid, min_group_size: int = 3) -> DedupeResult:
    """Collapse dimensional fan-out.

    When the LLM generates N cells that share the same layer.concern.scope.domain
    but vary only by dimension (e.g., provenance-WHAT, provenance-HOW, provenance-WHO...),
    keep the highest-confidence one and quarantine the rest.

    min_group_size: only collapse groups of this size or larger (default 3).
    A pair of cells differing by dimension is fine — that's legitimate.
    But 5+ cells that are "the same concept from different angles" is noise.
    """
    result = DedupeResult()

    # Group filled cells by (layer, concern, scope, domain) — the 4 axes
    # that should be unique. Dimension is the axis that varies in fan-out.
    groups: dict[tuple[str, str, str, str], list[Cell]] = defaultdict(list)
    for cell in grid.filled_cells():
        if cell.postcode.key == grid.root:
            continue  # never touch the root
        key = (cell.postcode.layer, cell.postcode.concern,
               cell.postcode.scope, cell.postcode.domain)
        groups[key].append(cell)

    for group_key, cells in groups.items():
        if len(cells) < min_group_size:
            continue

        # Check if this is actually dimensional fan-out:
        # all cells should have similar content (same concept, different angle)
        # Heuristic: if they share the same first word of primitive, it's fan-out
        primitives = [c.primitive for c in cells]
        base_words = [p.split("-")[0] if "-" in p else p for p in primitives]
        most_common = Counter(base_words).most_common(1)[0]
        if most_common[1] < min_group_size:
            continue  # Not fan-out — different concepts at same coordinates

        result.groups_found += 1

        # Keep the highest-confidence cell, quarantine the rest
        cells_sorted = sorted(cells, key=lambda c: c.confidence, reverse=True)
        keeper = cells_sorted[0]

        for cell in cells_sorted[1:]:
            # Replace with quarantined version
            quarantined = Cell(
                postcode=cell.postcode,
                primitive=cell.primitive,
                content=cell.content,
                fill=FillState.Q,
                confidence=0.0,
                connections=cell.connections,
                parent=cell.parent,
                source=cell.source,
                revisions=cell.revisions,
            )
            grid.put(quarantined)
            result.cells_quarantined += 1

    return result


# ---------------------------------------------------------------------------
# VERIFIER agent
# ---------------------------------------------------------------------------

@dataclass
class VerifyResult:
    """Result of verification pass."""
    checked: int = 0
    promoted: int = 0
    quarantined: int = 0
    blocked: int = 0
    issues: list[str] = field(default_factory=list)


def verifier(grid: Grid, confidence_boost: float = 0.02) -> VerifyResult:
    """Enforce AX1 provenance on all filled cells.

    For each filled cell:
      - If source traces to intent contract or another filled cell → boost confidence
      - If source is broken (referenced cells no longer filled) → quarantine
      - If cell has 2+ prior quarantines in revision history → block

    Returns VerifyResult with counts.
    """
    result = VerifyResult()

    for cell in list(grid.filled_cells()):
        result.checked += 1
        provenance_valid = _check_cell_provenance(grid, cell)

        if provenance_valid:
            # Check for repeat quarantine pattern: if revisions contain
            # 2+ zero-confidence entries (from prior quarantines), block
            quarantine_count = sum(
                1 for _, conf in cell.revisions if conf == 0.0
            )
            if quarantine_count >= 2:
                block(grid, cell.postcode.key, reason="repeat_quarantine")
                result.blocked += 1
                result.issues.append(
                    f"Blocked {cell.postcode.key}: {quarantine_count} prior quarantines"
                )
                continue

            # Boost confidence slightly (verified = more trustworthy)
            new_conf = min(1.0, cell.confidence + confidence_boost)
            if new_conf != cell.confidence:
                fill(
                    grid,
                    cell.postcode.key,
                    cell.primitive,
                    cell.content,
                    new_conf,
                    connections=cell.connections,
                    source=cell.source,
                    parent=cell.parent,
                    agent="verifier",
                )
                result.promoted += 1
        else:
            # Quarantine — provenance broken
            q_cell = Cell(
                postcode=cell.postcode,
                primitive=cell.primitive,
                content=cell.content,
                fill=FillState.Q,
                confidence=0.0,
                connections=cell.connections,
                parent=cell.parent,
                source=cell.source,
            )
            grid.put(q_cell)
            result.quarantined += 1
            result.issues.append(
                f"AX1 violation at {cell.postcode.key}: source chain broken"
            )

    return result


def _check_cell_provenance(grid: Grid, cell: Cell) -> bool:
    """Check if a cell's provenance chain is intact.

    Valid if ANY source is:
      - INTENT_CONTRACT
      - A filled cell in the grid
      - A human: or contract: reference
      - A memory: reference
    """
    for s in cell.source:
        if s == INTENT_CONTRACT:
            return True
        if s.startswith(("human:", "contract:", "memory:")):
            return True
        source_cell = grid.get(s)
        if source_cell and source_cell.is_filled:
            return True

    # Check parent
    if cell.parent:
        if cell.parent == INTENT_CONTRACT:
            return True
        parent_cell = grid.get(cell.parent)
        if parent_cell and parent_cell.is_filled:
            return True

    return False


# ---------------------------------------------------------------------------
# OBSERVER agent
# ---------------------------------------------------------------------------

@dataclass
class ObserveResult:
    """Result of observation pass."""
    signals_detected: int = 0
    candidates_created: int = 0
    signals: list[EmergenceSignal] = field(default_factory=list)


def observer(grid: Grid) -> ObserveResult:
    """Detect cross-cell patterns and create candidate nodes.

    Wraps detect_emergence() and places candidates in the EMG layer.
    """
    result = ObserveResult()

    signals = detect_emergence(grid)
    result.signals = signals
    result.signals_detected = len(signals)

    for i, signal in enumerate(signals):
        # Place candidate in EMG layer
        target_key = _emergence_target(signal, index=i)
        if not grid.has(target_key):
            promote_emergence(grid, signal, target_key)
            result.candidates_created += 1

    return result


def _emergence_target(signal: EmergenceSignal, index: int = 0) -> str:
    """Determine the postcode for an emergence candidate.

    Uses signal type + evidence hash to generate unique postcodes,
    preventing collisions when multiple signals of the same type exist.

    Args:
        signal: The emergence signal.
        index: Signal index within current observer call (for uniqueness).
    """
    # Derive dimension from evidence to create unique coordinates
    evidence_hash = hash(signal.evidence) % 8
    dimensions = ["WHAT", "HOW", "WHY", "WHO", "WHEN", "WHERE", "IF", "HOW_MUCH"]
    dim = dimensions[evidence_hash]

    # Derive scope from signal type
    if signal.signal_type == "repeated_primitive":
        scope = "ECO"
        domain = "COG"
    elif signal.signal_type == "shared_connection":
        scope = "APP"
        domain = "SFT"
    elif signal.signal_type == "orphan_cluster":
        scope = "DOM"
        domain = "ORG"
    else:
        scope = "ECO"
        domain = "COG"

    # Use concern axis to add uniqueness based on index
    concerns = ["CND", "INT", "PRV", "MET"]
    concern = concerns[index % len(concerns)]

    return f"EMG.{concern}.{scope}.{dim}.{domain}"


# ---------------------------------------------------------------------------
# EMERGENCE agent
# ---------------------------------------------------------------------------

@dataclass
class EmergeResult:
    """Result of emergence promotion pass."""
    candidates_checked: int = 0
    promoted: int = 0
    rejected: int = 0


def emergence(grid: Grid, min_evidence: int = 2) -> EmergeResult:
    """Validate and promote candidate cells.

    For each C (candidate) cell:
      - Check evidence references are still valid (filled cells)
      - If >= min_evidence valid references → promote to P at 0.75 confidence
      - If insufficient evidence → leave as C (don't delete — may gather more evidence)
    """
    result = EmergeResult()

    for cell in list(grid.candidate_cells()):
        result.candidates_checked += 1

        # Count valid evidence
        valid_evidence = 0
        for ref in cell.connections:
            ref_cell = grid.get(ref)
            if ref_cell and ref_cell.is_filled:
                valid_evidence += 1

        if valid_evidence >= min_evidence:
            # Promote: fill as P with moderate confidence
            fill(
                grid,
                cell.postcode.key,
                cell.primitive,
                cell.content,
                0.75,
                connections=cell.connections,
                source=tuple(
                    ref for ref in cell.connections
                    if grid.get(ref) and grid.get(ref).is_filled
                ),
                agent="emergence",
            )
            result.promoted += 1
        else:
            result.rejected += 1

    return result


# ---------------------------------------------------------------------------
# GOVERNOR agent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationIssue:
    """A specific issue found during simulation."""
    check: str          # "conflict", "gap", "cycle", "dead_end"
    severity: str       # "hard" (blocks emission) or "soft" (warning)
    postcodes: tuple[str, ...]
    description: str


@dataclass
class SimulationResult:
    """Result of the governor's simulation gate."""
    passed: bool = False
    issues: list[SimulationIssue] = field(default_factory=list)
    hard_failures: int = 0
    soft_warnings: int = 0
    fill_rate: float = 0.0
    total_cells: int = 0
    filled_cells: int = 0
    quarantined_cells: int = 0

    @property
    def can_emit(self) -> bool:
        """Emission allowed only if no hard failures."""
        return self.hard_failures == 0


def governor(grid: Grid) -> SimulationResult:
    """Simulation gate — final check before emission.

    Runs 4 checks:
      1. CONFLICT — contradictory cells (same postcode, different content at high confidence)
      2. GAP — critical paths with empty intermediate nodes
      3. CYCLE — circular parent dependencies
      4. DEAD_END — filled cells with zero outgoing connections (orphans)

    Hard failures block emission. Soft warnings are informational.
    """
    result = SimulationResult(
        total_cells=grid.total_cells,
        filled_cells=len(grid.filled_cells()),
        quarantined_cells=len(grid.quarantined_cells()),
        fill_rate=grid.fill_rate,
    )

    _check_conflicts(grid, result)
    _check_gaps(grid, result)
    _check_cycles(grid, result)
    _check_dead_ends(grid, result)

    result.passed = result.hard_failures == 0

    return result


def _check_conflicts(grid: Grid, result: SimulationResult) -> None:
    """Check for contradictory fills.

    Two cells connected to each other with fill=F but content that
    suggests contradiction (one says X, the other says not-X).

    Heuristic: if two connected F cells both have confidence > 0.90
    but their content shares no words (beyond stop words) → potential conflict.
    """
    _STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "no", "nor", "so", "yet", "if",
        "then", "that", "this", "these", "those", "it", "its",
    })

    filled = grid.filled_cells()
    for cell in filled:
        if cell.confidence < 0.90:
            continue
        cell_words = {
            w.lower().strip(".,!?;:'\"()[]{}") for w in cell.content.split()
        } - _STOP_WORDS

        for conn_key in cell.connections:
            target = grid.get(conn_key)
            if not target or not target.is_filled or target.confidence < 0.90:
                continue
            target_words = {
                w.lower().strip(".,!?;:'\"()[]{}") for w in target.content.split()
            } - _STOP_WORDS

            if cell_words and target_words and not cell_words & target_words:
                # Zero word overlap between two high-confidence connected cells
                result.issues.append(SimulationIssue(
                    check="conflict",
                    severity="soft",
                    postcodes=(cell.postcode.key, target.postcode.key),
                    description=f"Zero content overlap between high-confidence connected cells",
                ))
                result.soft_warnings += 1


def _check_gaps(grid: Grid, result: SimulationResult) -> None:
    """Check for critical gaps.

    A gap is a filled cell connecting to an empty cell that connects
    to another filled cell. The empty intermediate breaks the chain.
    """
    for cell in grid.filled_cells():
        for conn_key in cell.connections:
            target = grid.get(conn_key)
            if target is None or not target.is_empty:
                continue
            # Target is empty — check if it connects to something filled
            for onward_key in target.connections:
                onward = grid.get(onward_key)
                if onward and onward.is_filled:
                    result.issues.append(SimulationIssue(
                        check="gap",
                        severity="hard",
                        postcodes=(cell.postcode.key, conn_key, onward_key),
                        description=f"Empty cell {conn_key} sits between two filled cells",
                    ))
                    result.hard_failures += 1
                    return  # One hard gap is enough


def _check_cycles(grid: Grid, result: SimulationResult) -> None:
    """Check for circular parent dependencies.

    If A.parent = B and B.parent = A, that's a cycle.
    Uses iterative ancestor walk with visited set.
    Blocks cycle participants via block().
    """
    for cell in grid.cells.values():
        if not cell.parent:
            continue
        if cell.is_blocked:
            continue
        visited = {cell.postcode.key}
        current_key = cell.parent

        depth = 0
        while current_key and depth < 100:
            if current_key == INTENT_CONTRACT:
                break
            if current_key in visited:
                # Block all cycle participants
                for cycle_key in visited:
                    cycle_cell = grid.get(cycle_key)
                    if cycle_cell and not cycle_cell.is_blocked:
                        block(grid, cycle_key, reason="parent_cycle")
                result.issues.append(SimulationIssue(
                    check="cycle",
                    severity="hard",
                    postcodes=(cell.postcode.key, current_key),
                    description=f"Parent cycle detected and blocked: {cell.postcode.key} → ... → {current_key}",
                ))
                result.hard_failures += 1
                return  # One cycle is enough
            visited.add(current_key)
            parent_cell = grid.get(current_key)
            current_key = parent_cell.parent if parent_cell else None
            depth += 1


def _check_dead_ends(grid: Grid, result: SimulationResult) -> None:
    """Check for filled cells with no connections and no children.

    A dead end is a filled cell that:
      - Has no outgoing connections
      - Is not connected TO by any other cell
      - Has no children (no cells with this as parent)
      - Is not the root

    Dead ends are soft warnings (they're valid but potentially incomplete).
    """
    # Build reverse connection map
    connected_to: set[str] = set()
    has_children: set[str] = set()

    for cell in grid.cells.values():
        for conn in cell.connections:
            connected_to.add(conn)
        if cell.parent:
            has_children.add(cell.parent)

    for cell in grid.filled_cells():
        key = cell.postcode.key
        if key == grid.root:
            continue
        if not cell.connections and key not in connected_to and key not in has_children:
            result.issues.append(SimulationIssue(
                check="dead_end",
                severity="soft",
                postcodes=(key,),
                description=f"Isolated filled cell with no connections",
            ))
            result.soft_warnings += 1


# ---------------------------------------------------------------------------
# Orchestrator — the compilation pipeline
# ---------------------------------------------------------------------------

@dataclass
class CompileConfig:
    """Configuration for the compilation pipeline."""
    max_iterations: int = 5
    confidence_boost: float = 0.02
    confidence_decay: float = 0.7
    min_evidence: int = 2
    min_fill_rate: float = 0.0  # Minimum fill rate before early exit (0.0 = no minimum)
    target_scopes: tuple[str, ...] = ()  # Scope hints for AUTHOR (empty = no restriction)
    scope_schedule: tuple[tuple[str, ...], ...] = ()  # Per-iteration scope overrides


@dataclass
class CompileResult:
    """Result of a full compilation run."""
    grid: Grid
    iterations: int = 0
    converged: bool = False
    simulation: Optional[SimulationResult] = None
    author_calls: int = 0
    verify_results: list[VerifyResult] = field(default_factory=list)
    observe_results: list[ObserveResult] = field(default_factory=list)
    emerge_results: list[EmergeResult] = field(default_factory=list)


def compile(
    input_text: str,
    llm_fn: LLMFunction,
    intent_postcode: str = "INT.SEM.ECO.WHY.SFT",
    intent_primitive: str = "intent",
    history: Optional[list[Grid]] = None,
    config: Optional[CompileConfig] = None,
) -> CompileResult:
    """Run the full compilation pipeline.

    MEMORY → [AUTHOR → VERIFIER → OBSERVER → EMERGENCE → GOVERNOR]
    Loop until GOVERNOR passes or max_iterations reached.

    Args:
        input_text: The raw intent to compile
        llm_fn: Injectable LLM function for AUTHOR
        intent_postcode: Postcode for the root intent cell
        intent_primitive: Primitive name for the root intent cell
        history: Previous grids for MEMORY bootstrap
        config: Pipeline configuration

    Returns:
        CompileResult with the final grid and metadata
    """
    if config is None:
        config = CompileConfig()

    result = CompileResult(grid=Grid())
    grid = result.grid

    # Phase 0: Set intent contract
    grid.set_intent(input_text, intent_postcode, intent_primitive)

    # Phase 1: MEMORY — bootstrap from history
    if history:
        memory(grid, history, confidence_decay=config.confidence_decay)

    # Phase 2-6: Loop
    for i in range(config.max_iterations):
        result.iterations = i + 1

        # AUTHOR — fill from input (scope_schedule overrides target_scopes per iteration)
        iteration_scopes = config.target_scopes
        if config.scope_schedule:
            iteration_scopes = config.scope_schedule[min(i, len(config.scope_schedule) - 1)]
        author(grid, input_text, llm_fn, target_scopes=iteration_scopes)
        result.author_calls += 1

        # DEDUP — collapse dimensional fan-out before verification
        deduplicate_dimensional(grid)

        # VERIFIER — check provenance
        v_result = verifier(grid, confidence_boost=config.confidence_boost)
        result.verify_results.append(v_result)

        # OBSERVER — detect patterns
        o_result = observer(grid)
        result.observe_results.append(o_result)

        # EMERGENCE — promote candidates
        e_result = emergence(grid, min_evidence=config.min_evidence)
        result.emerge_results.append(e_result)

        # GOVERNOR — simulation gate
        sim = governor(grid)
        result.simulation = sim

        if sim.passed:
            # Don't exit early if fill rate is below minimum threshold
            if config.min_fill_rate > 0.0 and grid.fill_rate < config.min_fill_rate:
                continue  # Keep iterating — grid is still sparse
            result.converged = True
            break

        # If not passed but converged (no more useful fills), stop anyway
        if is_converged(grid):
            result.converged = True
            break

    return result
