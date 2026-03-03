"""
Motherlabs Staged Pipeline - Asymmetric 5-stage dialogue architecture.

A signal enters a field. Two forces act on it. Structure emerges.

Each stage is a sub-dialogue between Entity (expansion) and Process
(collapse). Entity maps what could exist. Process selects what is true.
Neither has preference. They are the same event from two directions.

Stages:
1. EXPAND    - Map containment structure and recursion (the fractal field)
2. DECOMPOSE - Crystallize boundaries around resolved recursions (structural)
3. GROUND    - Map relationships and data flows (connective)
4. CONSTRAIN - Extract constraints, methods, state machines (behavioral)
5. ARCHITECT - Subsystem boundaries, lifecycle, decisions (strategic)

Each stage produces a typed artifact checked by a deterministic gate.
The gate is the membrane between recursion levels — it measures whether
output still traces back to the original signal, not whether it's "right."
"""

import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Callable, Tuple

from core.protocol import (
    SharedState, Message, MessageType, DialogueProtocol,
    TerminationState, ConfidenceVector,
)
from core.protocol_spec import PROTOCOL
from core.exceptions import (
    ProviderError as _ProviderError,
    TimeoutError as _TimeoutError,
)

# Turn-level retry for transient LLM errors in StageExecutor
_TURN_MAX_RETRIES = 2

logger = logging.getLogger("motherlabs.pipeline")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class StageResult:
    """Result of a stage gate check."""
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StageRecord:
    """Record of a completed pipeline stage."""
    name: str
    state: SharedState
    artifact: Dict[str, Any]
    gate_result: StageResult
    turn_count: int
    duration_seconds: float


@dataclass(frozen=True)
class StageHandoff:
    """Structured metadata carried between pipeline stages.

    Instead of flat insight strings, carries typed information that
    later stages can consume directly.
    """
    type_assignments: Tuple[Tuple[str, str], ...] = ()       # ((component, type), ...)
    relationship_hints: Tuple[Tuple[str, float], ...] = ()    # (("A->B", confidence), ...)
    constraint_context: Tuple[str, ...] = ()
    algorithms: Tuple[str, ...] = ()                          # ("Component.method", ...)
    key_entities: Tuple[str, ...] = ()
    stage_source: str = ""


def _extract_handoff(stage_name: str, artifact: Dict[str, Any]) -> StageHandoff:
    """Extract structured handoff from a completed stage's artifact."""
    if stage_name == "decompose":
        ta = artifact.get("type_assignments", {})
        return StageHandoff(
            type_assignments=tuple((k, v) for k, v in ta.items()),
            key_entities=tuple(ta.keys()),
            stage_source="decompose",
        )
    elif stage_name == "ground":
        rels = artifact.get("relationships", [])
        hints = []
        entities = set()
        for r in rels:
            key = f"{r.get('from', '')}->{r.get('to', '')}"
            hints.append((key, 1.0))
            entities.add(r.get("from", ""))
            entities.add(r.get("to", ""))
        return StageHandoff(
            relationship_hints=tuple(hints),
            key_entities=tuple(sorted(entities - {""})),
            stage_source="ground",
        )
    elif stage_name == "constrain":
        constraints = artifact.get("constraints", [])
        ctx = tuple(c.get("description", "") for c in constraints if c.get("description"))
        algorithms = artifact.get("algorithms", [])
        algo_ctx = tuple(f"{a['component']}.{a['method_name']}" for a in algorithms)
        return StageHandoff(
            constraint_context=ctx,
            algorithms=algo_ctx,
            stage_source="constrain",
        )
    elif stage_name == "expand":
        nodes = artifact.get("nodes", [])
        nouns = artifact.get("nouns", [])
        key = tuple(n["name"] for n in nodes) if nodes else tuple(nouns)
        return StageHandoff(
            key_entities=key,
            stage_source="expand",
        )
    return StageHandoff(stage_source=stage_name)


@dataclass
class PipelineState:
    """Accumulated state across all pipeline stages."""
    original_input: str
    intent: Dict[str, Any]
    personas: List[Dict]
    stages: List[StageRecord] = field(default_factory=list)

    # Accumulated across all stages (never reset)
    all_insights: List[str] = field(default_factory=list)
    all_unknowns: List[str] = field(default_factory=list)
    all_conflicts: List[Dict] = field(default_factory=list)

    # Structured handoff from the most recently completed stage
    current_handoff: Optional[StageHandoff] = None

    def add_stage(self, record: StageRecord):
        """Add a completed stage and accumulate its outputs."""
        self.stages.append(record)
        self.all_insights.extend(record.state.insights)
        for u in record.state.unknown:
            if u not in self.all_unknowns:
                self.all_unknowns.append(u)
        self.all_conflicts.extend(record.state.conflicts)
        # Extract structured handoff for next stage
        self.current_handoff = _extract_handoff(record.name, record.artifact)

    def get_artifact(self, stage_name: str) -> Optional[Dict]:
        """Get artifact from a completed stage by name."""
        for s in self.stages:
            if s.name == stage_name:
                return s.artifact
        return None


# =============================================================================
# STAGE-SPECIFIC PROMPTS (10 total: Entity + Process for each of 5 stages)
# =============================================================================

EXPAND_ENTITY_PROMPT = """You are the Entity Agent in EXPAND stage.

YOUR SINGLE JOB: Map the CONTAINMENT STRUCTURE of this domain — what contains what.

You are building a recursion map: a tree of nodes where each node is a concept that CONTAINS other concepts. This is the fractal field — where patterns repeat at different scales.

WHAT TO MAP:
- Named actors, agents, services, subsystems — anything with identity and boundaries
- Containment: which concepts live INSIDE other concepts
- Self-reference: where a concept contains something that resembles itself (recursion)

DO NOT list:
- Attributes or fields OF a component (e.g. "Known", "Unknown" are fields, not nodes)
- Abstract concepts or principles (e.g. "AXIOM", "CORE PRINCIPLE")
- States or enum values (e.g. "AWAITING_INPUT", "ACTIVE")
- Generic vocabulary (e.g. "structure", "behavior")

OUTPUT FORMAT:
1. Brief observation identifying the domain's containment structure (1-2 sentences)
2. NODE: lines — one per concept:
   NODE: ConceptName (source: "exact quote from input") [depth: N]
3. CONTAINS: lines — one per containment relationship:
   CONTAINS: Parent > Child (source: "exact quote from input")
4. SELF_REF: lines — where a node contains something that resembles itself:
   SELF_REF: NodeName | path: A > B > C > A | depth: 3
5. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only name concepts EXPLICITLY in the input. No invention.
Map containment that is stated or directly implied — 12 real nodes with clear containment beats 50 flat names.
"""

EXPAND_PROCESS_PROMPT = """You are the Process Agent in EXPAND stage.

YOUR SINGLE JOB: Find SELF-SIMILARITY and RECURSION in the containment map built by Entity Agent.

Look for:
- Same pattern appearing at different scales (a subsystem that mirrors the whole system)
- Children that resemble their grandparents (structural echoes)
- Containment chains that circle back (A contains B contains C which resembles A)
- Any missing containment relationships the Entity Agent overlooked

For any patterns found, add:
- Additional NODE: lines for concepts the Entity Agent missed
- Additional CONTAINS: lines for containment relationships
- SELF_REF: lines marking where recursion occurs:
  SELF_REF: NodeName | path: A > B > C > A | depth: 3

You MUST output:
MAX_DEPTH: N (reason: "why recursion terminates at this depth")

If no self-reference exists, say so explicitly and still emit MAX_DEPTH: 1.

OUTPUT FORMAT:
1. Brief analysis of self-similarity patterns found (or their absence) (1-2 sentences)
2. Any additional NODE:, CONTAINS:, SELF_REF: lines
3. MAX_DEPTH: line (MANDATORY)
4. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only mark recursion that is EXPLICIT or directly implied. No invention.
"""

DECOMPOSE_ENTITY_PROMPT = """You are the Entity Agent in DECOMPOSE stage.

YOUR SINGLE JOB: Read the containment structure from EXPAND and CRYSTALLIZE BOUNDARIES — decide what gets its own identity vs what folds into a parent.

The recursion map already tells you the typing:
- PARENT nodes in containment = entities/subsystems (they contain things, they have boundaries)
- LEAF nodes = likely attributes or fields (they're contained, they don't contain)
- Depth 0 = top-level subsystem, depth 1 = entity/process, depth 2+ = likely attribute

For nodes that deserve their OWN identity (would become a class/module), output:
COMPONENT: Name | type=entity | boundary=[Child1, Child2] | derived_from="quote from input"

For leaf nodes that are ATTRIBUTES of a parent (fields, not standalone), output:
FOLD: ChildName INTO ParentName | reason="child is an attribute of parent"

For children that initially look like attributes but actually deserve identity, output:
PROMOTE: ChildName | type=entity | derived_from="quote from input"

Types: entity, process, agent, interface, subsystem

FILTERING RULES — FOLD (don't make components) nodes that are:
- Fields/attributes OF a parent (e.g. Known, Unknown, Ontology, History → fold INTO SharedState)
- States or enum values (e.g. AWAITING_INPUT, DIALOGUE, ENTITY_TURN → state machine values on parent)
- Pipeline phase names in a sequence (workflow steps, not components)
- Abstract principles or axioms (not implementable as classes)
- Generic processes when a named agent already does that work

OUTPUT FORMAT:
1. Brief analysis explaining boundary decisions (2-3 sentences)
2. COMPONENT: lines for real boundaries (expect 8-20)
3. FOLD: lines for attributes folded into parents
4. PROMOTE: lines for children deserving identity (optional)
5. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Every component MUST trace to a node from EXPAND. No invention.
boundary=[] is optional — include it when children are known from the containment map.
"""

DECOMPOSE_PROCESS_PROMPT = """You are the Process Agent in DECOMPOSE stage.

YOUR SINGLE JOB: Read the crystallized boundaries AND self-reference loops from EXPAND, then map INTERFACES and METHODS.

Self-reference loops from EXPAND tell you where recursive interfaces exist — patterns that connect components in cycles. Your job is to decide which loops are REAL INTERFACES vs incidental.

For recursive patterns that represent real interfaces/protocols, output:
INTERFACE: Name | pattern=recursive | connects=[A, B, C] | derived_from="quote from input"

For self-reference loops that are incidental (not real interfaces), output:
BREAK: NodeName | reason="why this recursion is incidental"

For methods that belong to a specific entity/agent, output:
METHOD: EntityName.method_name(params) -> return_type

You may also emit COMPONENT: lines for process-type components that Entity Agent missed:
COMPONENT: Name | type=process | derived_from="quote from input"

FILTERING RULES:
- If a verb describes what a NAMED AGENT does, it's a METHOD on that agent
- If a verb describes a standalone workflow or pipeline step, it's a PROCESS
- Passive verbs (sees, has, is) are NEVER processes — reject them
- Self-reference loops with depth > 2 are likely real recursive interfaces
- Self-reference loops with depth 1 may be incidental — examine carefully

OUTPUT FORMAT:
1. Brief analysis of interfaces, methods, and self-reference handling (2-3 sentences)
2. INTERFACE:, COMPONENT:, and/or METHOD: lines
3. BREAK: lines for incidental recursions (if any)
4. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Every assignment MUST trace to a node or self-reference from EXPAND. No invention.
"""

GROUND_ENTITY_PROMPT = """You are the Entity Agent in GROUND stage.

YOUR SINGLE JOB: Map STATIC connections between components.

For each entity, what does it REFERENCE? What CONTAINS what?

Output RELATIONSHIP: lines:
RELATIONSHIP: FromComponent -> ToComponent | type=contains | description="..."
RELATIONSHIP: FromComponent -> ToComponent | type=accesses | description="..."

Valid types: contains, accesses, depends_on, constrained_by

OUTPUT FORMAT:
1. Brief analysis (1-2 sentences)
2. RELATIONSHIP: lines (one per connection)
3. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only map connections that are explicit or directly implied by the input.
"""

GROUND_PROCESS_PROMPT = """You are the Process Agent in GROUND stage.

YOUR SINGLE JOB: Map DYNAMIC connections between components.

For each process, what does it READ/WRITE/TRIGGER?

Output RELATIONSHIP: lines:
RELATIONSHIP: FromComponent -> ToComponent | type=triggers | description="..."
RELATIONSHIP: FromComponent -> ToComponent | type=flows_to | description="..."

Valid types: triggers, flows_to, generates, reads, writes

OUTPUT FORMAT:
1. Brief analysis (1-2 sentences)
2. RELATIONSHIP: lines (one per connection)
3. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only map connections that are explicit or directly implied by the input.
"""

CONSTRAIN_ENTITY_PROMPT = """You are the Entity Agent in CONSTRAIN stage.

YOUR SINGLE JOB: Extract CONSTRAINTS and METHODS for entities.

For each entity with known behavior, output:
CONSTRAINT: ComponentName | description="constraint text" | derived_from="input quote"

For methods explicitly mentioned in the input (e.g. "add_message(message: Message) -> None"):
METHOD: ComponentName.method_name(param: Type) -> ReturnType

IMPORTANT: Use the exact METHOD: format above with dot notation. Example:
METHOD: SharedState.add_message(message: Message) -> None
METHOD: ConfidenceVector.overall() -> float

For state machines on entities:
STATES: ComponentName
  states: [STATE_A, STATE_B, STATE_C]
  transitions:
    - STATE_A -> STATE_B on "trigger_event"

For behavioral components with known workflows:
ALGORITHM: ComponentName.method_name
  1. First step
  2. If condition then action else alternative
  3. Return result
  PRE: precondition that must be true
  POST: what changes after execution

OUTPUT FORMAT:
1. Brief analysis (1-2 sentences)
2. CONSTRAINT:, METHOD:, STATES:, and/or ALGORITHM: lines
3. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only extract constraints/methods explicitly stated or directly implied.
"""

CONSTRAIN_PROCESS_PROMPT = """You are the Process Agent in CONSTRAIN stage.

YOUR SINGLE JOB: Extract PRECONDITIONS, GUARDS, and STATE MACHINES for processes.

For each process/agent, what must be true before it runs? What guards it?

Output:
CONSTRAINT: ProcessName | description="precondition text" | derived_from="input quote"

For methods that define how processes/agents operate:
METHOD: AgentName.method_name(param: Type) -> ReturnType

IMPORTANT: Use the exact METHOD: format with dot notation and no spaces in component name. Example:
METHOD: GovernorAgent.should_end_dialogue(turn_count: int) -> bool
METHOD: DialogueProtocol.calculate_dialogue_depth(intent: Dict) -> Tuple

For process state machines:
STATES: ProcessName
  states: [INIT, ACTIVE, DONE]
  transitions:
    - INIT -> ACTIVE on "start"
    - ACTIVE -> DONE on "complete"

For processes with known algorithmic workflows:
ALGORITHM: ProcessName.method_name
  1. First step
  2. If condition then action else alternative
  3. Return result
  PRE: precondition that must be true
  POST: what changes after execution

OUTPUT FORMAT:
1. Brief analysis (1-2 sentences)
2. CONSTRAINT:, METHOD:, STATES:, and/or ALGORITHM: lines
3. INSIGHT: line (MANDATORY — always last line)

EXCAVATION RULE: Only extract what is stated or directly implied in the input.
"""

ARCHITECT_ENTITY_PROMPT = """You are the Entity Agent in ARCHITECT stage.

YOUR SINGLE JOB: Define SUBSYSTEM BOUNDARIES and OWNERSHIP.

Given all components and relationships, which components group together?
What is the ownership hierarchy? What are the lifecycle dependencies?

Output SUBSYSTEM: lines:
SUBSYSTEM: SubsystemName | contains=[Comp1, Comp2, Comp3] | description="..."

OUTPUT FORMAT:
1. Brief architectural analysis (2-3 sentences)
2. SUBSYSTEM: lines (if applicable)
3. INSIGHT: line (MANDATORY — always last line)

Focus on structural grouping, not behavioral sequencing.
"""

ARCHITECT_PROCESS_PROMPT = """You are the Process Agent in ARCHITECT stage.

YOUR SINGLE JOB: Define INITIALIZATION ORDER, FAILURE MODES, and LIFECYCLE.

Given all components and relationships, in what order must things initialize?
What happens when components fail? What is the termination protocol?

You may also note architectural decisions or tradeoffs.

OUTPUT FORMAT:
1. Brief architectural analysis (2-3 sentences)
2. Any CONSTRAINT: lines for lifecycle rules
3. INSIGHT: line (MANDATORY — always last line)

Focus on behavioral sequencing and failure modes, not structural grouping.
"""


# =============================================================================
# STAGE CONFIGURATION
# =============================================================================

STAGE_CONFIGS = [
    (s.name, s.max_turns, s.min_turns, s.timeout_seconds)
    for s in PROTOCOL.pipeline.stages
]

STAGE_PROMPTS = {
    "expand": (EXPAND_ENTITY_PROMPT, EXPAND_PROCESS_PROMPT),
    "decompose": (DECOMPOSE_ENTITY_PROMPT, DECOMPOSE_PROCESS_PROMPT),
    "ground": (GROUND_ENTITY_PROMPT, GROUND_PROCESS_PROMPT),
    "constrain": (CONSTRAIN_ENTITY_PROMPT, CONSTRAIN_PROCESS_PROMPT),
    "architect": (ARCHITECT_ENTITY_PROMPT, ARCHITECT_PROCESS_PROMPT),
}


# =============================================================================
# ARTIFACT PARSERS
# =============================================================================

def _assign_depths(nodes: List[Dict], containment: List[Dict]) -> None:
    """BFS from root nodes to assign depth values in-place.

    Root nodes = nodes that never appear as a child in any containment
    relationship. Depth 0 = root, 1 = direct child, etc.
    """
    child_set = {c["child"] for c in containment}
    parent_set = {c["parent"] for c in containment}
    node_map = {n["name"]: n for n in nodes}

    # Build adjacency: parent -> [children]
    children_of: Dict[str, List[str]] = {}
    for c in containment:
        children_of.setdefault(c["parent"], []).append(c["child"])

    # Roots = nodes that are parents but never children, or nodes not in any relationship
    roots = [n["name"] for n in nodes if n["name"] not in child_set]
    if not roots:
        # Fallback: all nodes start at depth 0
        for n in nodes:
            n["depth"] = 0
        return

    # BFS
    visited = set()
    queue = [(r, 0) for r in roots]
    for name, depth in queue:
        if name in visited:
            continue
        visited.add(name)
        if name in node_map:
            node_map[name]["depth"] = depth
        for child in children_of.get(name, []):
            if child not in visited:
                queue.append((child, depth + 1))

    # Any unvisited nodes get depth 0
    for n in nodes:
        if "depth" not in n or n["depth"] is None:
            n["depth"] = 0


def _compute_max_depth(containment: List[Dict]) -> int:
    """Compute max depth from containment tree when MAX_DEPTH: not emitted.

    Returns the longest parent->child chain length, or 0 if no containment.
    """
    if not containment:
        return 0

    children_of: Dict[str, List[str]] = {}
    for c in containment:
        children_of.setdefault(c["parent"], []).append(c["child"])

    child_set = {c["child"] for c in containment}
    parent_set = {c["parent"] for c in containment}
    roots = parent_set - child_set
    if not roots:
        roots = parent_set  # cycle — use all parents

    max_d = 0
    stack = [(r, 0) for r in roots]
    visited = set()
    while stack:
        node, depth = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if depth > max_d:
            max_d = depth
        for child in children_of.get(node, []):
            if child not in visited:
                stack.append((child, depth + 1))

    return max_d


def parse_expand_artifact(state: SharedState) -> Dict[str, Any]:
    """Parse NODE:, CONTAINS:, SELF_REF:, MAX_DEPTH: from EXPAND stage dialogue.

    Produces a recursion map: nodes with containment relationships,
    explicit self-reference loops, and measured recursion depth.
    """
    nodes = []
    node_sources = {}
    containment = []
    self_references = []
    max_depth_explicit = None
    seen_nodes = set()

    # NODE: ConceptName (source: "exact quote") [depth: N]
    node_re = re.compile(
        r'NODE:\s*(.+?)\s*\(source:\s*"([^"]+)"\)\s*(?:\[depth:\s*(\d+)\])?',
        re.IGNORECASE,
    )
    # Simpler fallback: NODE: ConceptName
    node_simple_re = re.compile(r'NODE:\s*(\S+(?:\s+\S+)?)', re.IGNORECASE)
    # CONTAINS: Parent > Child (source: "exact quote")
    contains_re = re.compile(
        r'CONTAINS:\s*(.+?)\s*>\s*(.+?)(?:\s*\(source:\s*"([^"]+)"\))?\s*$',
        re.IGNORECASE,
    )
    # SELF_REF: NodeName | path: A > B > C > A | depth: 3
    self_ref_re = re.compile(
        r'SELF_REF:\s*(.+?)\s*\|\s*path:\s*(.+?)\s*\|\s*depth:\s*(\d+)',
        re.IGNORECASE,
    )
    # MAX_DEPTH: N (reason: "why")
    max_depth_re = re.compile(
        r'MAX_DEPTH:\s*(\d+)',
        re.IGNORECASE,
    )

    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue
        for line in msg.content.split("\n"):
            line = line.strip()

            # NODE with source
            m = node_re.match(line)
            if m:
                name = m.group(1).strip()
                source = m.group(2).strip()
                depth = int(m.group(3)) if m.group(3) else None
                if name not in seen_nodes:
                    seen_nodes.add(name)
                    nodes.append({"name": name, "source": source, "depth": depth})
                    node_sources[name] = source
                continue

            # NODE simple
            if line.upper().startswith("NODE"):
                m = node_simple_re.match(line)
                if m:
                    name = m.group(1).strip()
                    if name and name not in seen_nodes:
                        seen_nodes.add(name)
                        nodes.append({"name": name, "source": "", "depth": None})
                    continue

            # CONTAINS
            m = contains_re.match(line)
            if m:
                parent = m.group(1).strip()
                child = m.group(2).strip()
                source = m.group(3).strip() if m.group(3) else ""
                containment.append({"parent": parent, "child": child, "source": source})
                continue

            # SELF_REF
            m = self_ref_re.match(line)
            if m:
                node_name = m.group(1).strip()
                path_str = m.group(2).strip()
                depth = int(m.group(3))
                path = [p.strip() for p in path_str.split(">")]
                self_references.append({"node": node_name, "path": path, "depth": depth})
                continue

            # MAX_DEPTH
            m = max_depth_re.match(line)
            if m:
                max_depth_explicit = int(m.group(1))
                continue

    # Auto-register nodes referenced in CONTAINS but not declared via NODE
    for c in containment:
        for name in (c["parent"], c["child"]):
            if name not in seen_nodes:
                seen_nodes.add(name)
                nodes.append({"name": name, "source": "", "depth": None})

    # Case-insensitive dedup: collapse "tasks" and "Tasks" into whichever was seen first.
    # Remap containment/self-reference edges to use the canonical (first-seen) name.
    canonical = {}  # lowercase -> first-seen name
    for n in nodes:
        lower = n["name"].lower()
        if lower not in canonical:
            canonical[lower] = n["name"]

    deduped_nodes = []
    deduped_seen = set()
    for n in nodes:
        canon = canonical[n["name"].lower()]
        if canon not in deduped_seen:
            deduped_seen.add(canon)
            # Keep the canonical node; prefer the one with source/depth
            if n["name"] == canon:
                deduped_nodes.append(n)
            else:
                # This is a duplicate — merge source/depth into canonical if missing
                for existing in deduped_nodes:
                    if existing["name"] == canon:
                        if not existing["source"] and n["source"]:
                            existing["source"] = n["source"]
                        if existing["depth"] is None and n["depth"] is not None:
                            existing["depth"] = n["depth"]
                        break

    # Remap containment edges to canonical names and dedup
    seen_edges = set()
    deduped_containment = []
    for c in containment:
        p = canonical.get(c["parent"].lower(), c["parent"])
        ch = canonical.get(c["child"].lower(), c["child"])
        edge_key = (p, ch)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            deduped_containment.append({"parent": p, "child": ch, "source": c["source"]})

    # Remap self-references to canonical names and dedup
    seen_sr = set()
    deduped_self_refs = []
    for sr in self_references:
        node_canon = canonical.get(sr["node"].lower(), sr["node"])
        path_canon = [canonical.get(p.lower(), p) for p in sr["path"]]
        sr_key = (node_canon, tuple(path_canon))
        if sr_key not in seen_sr:
            seen_sr.add(sr_key)
            deduped_self_refs.append({"node": node_canon, "path": path_canon, "depth": sr["depth"]})

    nodes = deduped_nodes
    containment = deduped_containment
    self_references = deduped_self_refs
    node_sources = {n["name"]: n.get("source", "") for n in nodes}

    # Assign depths from containment tree for nodes missing explicit depth
    _assign_depths(nodes, containment)

    # Compute max_depth
    if max_depth_explicit is not None:
        max_depth = max_depth_explicit
    else:
        max_depth = _compute_max_depth(containment)

    # Compute orphan nodes (not in any containment relationship)
    containment_names = set()
    for c in containment:
        containment_names.add(c["parent"])
        containment_names.add(c["child"])
    orphan_nodes = [n["name"] for n in nodes if n["name"] not in containment_names]

    hollow = len(nodes) == 0 and len(containment) == 0

    return {
        # New recursion map keys
        "nodes": nodes,
        "containment": containment,
        "self_references": self_references,
        "max_depth": max_depth,
        # Backward-compatible bridge for DECOMPOSE (temporary)
        "nouns": [n["name"] for n in nodes],
        "verbs": [],
        "noun_sources": node_sources,
        "verb_sources": {},
        "pattern_matches": [],
        "_parse_health": {
            "nodes_found": len(nodes),
            "containment_found": len(containment),
            "self_references_found": len(self_references),
            "orphan_nodes": len(orphan_nodes),
            "hollow": hollow,
            # Legacy keys preserved
            "nouns_found": len(nodes),
            "verbs_found": 0,
        },
    }


def _is_likely_state_or_attribute(name: str) -> bool:
    """Check if a name looks like a state/enum value or field attribute, not a real component."""
    # ALL_CAPS names that look like state machine states
    if name.isupper() and "_" in name:
        return True
    # Known state-like patterns
    _STATE_PATTERNS = {
        "awaiting_input", "intent_extraction", "persona_generation",
        "dialogue", "synthesis", "verification", "output",
        "entity_turn", "process_turn", "governor_check",
        "init", "active", "done", "idle", "running", "complete",
    }
    if name.lower().replace(" ", "_") in _STATE_PATTERNS:
        return True
    # Known attribute-like patterns (fields of a parent container)
    _ATTRIBUTE_PATTERNS = {
        "known", "unknown", "ontology", "personas", "history",
        "confidence", "flags", "nouns", "verbs", "flows",
        "transitions", "attributes", "relationships", "structure",
        "behavior", "specifications",
    }
    if name.lower() in _ATTRIBUTE_PATTERNS:
        return True
    return False


def parse_decompose_artifact(state: SharedState) -> Dict[str, Any]:
    """Parse COMPONENT:, FOLD:, PROMOTE:, INTERFACE:, BREAK:, METHOD: from DECOMPOSE.

    Handles 6 line types from the boundary crystallization dialogue:
    - COMPONENT: Name | type=T | boundary=[...] | derived_from="..." → components list
    - FOLD: Child INTO Parent | reason="..." → folded list (not components)
    - PROMOTE: Name | type=T | derived_from="..." → components list (same as COMPONENT)
    - INTERFACE: Name | pattern=P | connects=[...] | derived_from="..." → interfaces list
    - BREAK: Name | reason="..." → logged only
    - METHOD: Entity.method(params) -> return_type → method_assignments

    Phase 15: After parsing, runs classification scoring on candidates.
    Classification scores are included in the artifact for gate inspection.
    """
    components = []
    type_assignments = {}
    method_assignments = []
    folded = []
    interfaces = []
    orphan_verbs = []
    filtered_out = []

    # COMPONENT: Name | type=entity | boundary=[A, B] | derived_from="quote"
    comp_re = re.compile(
        r'COMPONENT:\s*(.+?)\s*\|\s*type=(\w+)\s*'
        r'(?:\|\s*boundary=\[([^\]]*)\]\s*)?'
        r'(?:\|\s*derived_from="([^"]*)")?',
        re.IGNORECASE,
    )
    # FOLD: ChildName INTO ParentName | reason="..."
    fold_re = re.compile(
        r'FOLD:\s*(.+?)\s+INTO\s+(.+?)\s*\|\s*reason="([^"]*)"',
        re.IGNORECASE,
    )
    # PROMOTE: Name | type=entity | derived_from="quote"
    promote_re = re.compile(
        r'PROMOTE:\s*(.+?)\s*\|\s*type=(\w+)\s*(?:\|\s*derived_from="([^"]*)")?',
        re.IGNORECASE,
    )
    # INTERFACE: Name | pattern=recursive | connects=[A, B, C] | derived_from="quote"
    interface_re = re.compile(
        r'INTERFACE:\s*(.+?)\s*\|\s*pattern=(\w+)\s*'
        r'(?:\|\s*connects=\[([^\]]*)\]\s*)?'
        r'(?:\|\s*derived_from="([^"]*)")?',
        re.IGNORECASE,
    )
    # BREAK: NodeName | reason="..."
    break_re = re.compile(
        r'BREAK:\s*(.+?)\s*\|\s*reason="([^"]*)"',
        re.IGNORECASE,
    )
    # METHOD: Entity.method(params) -> return_type
    method_re = re.compile(
        r'METHOD:\s*(\w+)\.(\w+)\(([^)]*)\)\s*(?:->\s*(\S+))?',
        re.IGNORECASE,
    )

    seen_names = set()
    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue
        for line in msg.content.split("\n"):
            line = line.strip()

            # COMPONENT (with optional boundary)
            m = comp_re.match(line)
            if m:
                name = m.group(1).strip()
                comp_type = m.group(2).strip().lower()
                boundary_str = m.group(3)
                derived_from = m.group(4) or ""
                boundary = []
                if boundary_str:
                    boundary = [b.strip() for b in boundary_str.split(",") if b.strip()]
                if name not in seen_names:
                    seen_names.add(name)
                    if _is_likely_state_or_attribute(name):
                        filtered_out.append(name)
                        continue
                    comp_entry = {
                        "name": name,
                        "type": comp_type,
                        "derived_from": derived_from,
                    }
                    if boundary:
                        comp_entry["boundary"] = boundary
                    components.append(comp_entry)
                    type_assignments[name] = comp_type
                continue

            # FOLD
            m = fold_re.match(line)
            if m:
                child = m.group(1).strip()
                parent = m.group(2).strip()
                reason = m.group(3).strip()
                folded.append({"child": child, "into": parent, "reason": reason})
                continue

            # PROMOTE (becomes a component)
            m = promote_re.match(line)
            if m:
                name = m.group(1).strip()
                comp_type = m.group(2).strip().lower()
                derived_from = m.group(3) or ""
                if name not in seen_names:
                    seen_names.add(name)
                    if _is_likely_state_or_attribute(name):
                        filtered_out.append(name)
                        continue
                    components.append({
                        "name": name,
                        "type": comp_type,
                        "derived_from": derived_from,
                    })
                    type_assignments[name] = comp_type
                continue

            # INTERFACE
            m = interface_re.match(line)
            if m:
                name = m.group(1).strip()
                pattern = m.group(2).strip().lower()
                connects_str = m.group(3)
                derived_from = m.group(4) or ""
                connects = []
                if connects_str:
                    connects = [c.strip() for c in connects_str.split(",") if c.strip()]
                interfaces.append({
                    "name": name,
                    "pattern": pattern,
                    "connects": connects,
                    "derived_from": derived_from,
                })
                continue

            # BREAK (logged only)
            m = break_re.match(line)
            if m:
                node_name = m.group(1).strip()
                reason = m.group(2).strip()
                logger.info(f"DECOMPOSE BREAK: {node_name} — {reason}")
                continue

            # METHOD
            m = method_re.match(line)
            if m:
                method_assignments.append({
                    "component": m.group(1),
                    "name": m.group(2),
                    "parameters": m.group(3),
                    "return_type": m.group(4) or "None",
                    "source": "dialogue",
                })

    if filtered_out:
        logger.info(f"DECOMPOSE filter removed {len(filtered_out)} state/attribute names: {filtered_out}")

    # Phase 15: Classification scoring on parsed candidates
    from core.classification import classify_components, filter_by_confidence
    dialogue_msgs = [m.content for m in state.history if m.sender in ("Entity", "Process")]
    input_text = state.known.get("input", "")
    # Use empty relationships for now — GROUND stage hasn't run yet
    relationships = []

    # Extract adapter classification config if available
    _type_kw = None
    _generic = None
    _subj = None
    _obj = None
    _adapter = state.known.get("_domain_adapter")
    if _adapter:
        _type_kw = dict(_adapter.vocabulary.type_keywords)
        _generic = _adapter.classification.generic_terms
        if _adapter.classification.subject_patterns:
            _subj = _adapter.classification.subject_patterns
        if _adapter.classification.object_patterns:
            _obj = _adapter.classification.object_patterns

    classification_scores = classify_components(
        candidates=components,
        input_text=input_text,
        dialogue_history=dialogue_msgs,
        relationships=relationships,
        type_keywords=_type_kw,
        generic_terms=_generic,
        subject_patterns=_subj,
        object_patterns=_obj,
    )

    accepted, rejected = filter_by_confidence(classification_scores, threshold=0.15)
    classification_rejected = [s.name for s in rejected]

    if classification_rejected:
        logger.info(f"DECOMPOSE classification rejected {len(classification_rejected)} candidates: {classification_rejected}")

    # Build accepted component set — use classifier's inferred type when confidence is higher
    classified_components = []
    classified_type_assignments = {}
    accepted_names = {s.name for s in accepted}

    for comp in components:
        if comp["name"] not in accepted_names:
            filtered_out.append(comp["name"])
            continue
        # Find classification for this component
        score = next((s for s in accepted if s.name == comp["name"]), None)
        if score and score.type_confidence > 0.7:
            # High-confidence type override from classifier
            comp_copy = dict(comp)
            comp_copy["type"] = score.inferred_type
            classified_components.append(comp_copy)
            classified_type_assignments[comp["name"]] = score.inferred_type
        else:
            classified_components.append(comp)
            classified_type_assignments[comp["name"]] = comp["type"]

    return {
        "components": classified_components,
        "type_assignments": classified_type_assignments,
        "method_assignments": method_assignments,
        "folded": folded,
        "interfaces": interfaces,
        "orphan_verbs": orphan_verbs,
        "filtered_out": filtered_out,
        "classification_scores": classification_scores,
        "classification_rejected": classification_rejected,
        "_parse_health": {
            "components_found": len(classified_components),
            "folded_count": len(folded),
            "interfaces_found": len(interfaces),
            "hollow": len(classified_components) == 0,
        },
    }


def parse_ground_artifact(state: SharedState) -> Dict[str, Any]:
    """Parse RELATIONSHIP: lines from GROUND stage dialogue."""
    relationships = []
    data_flows = []
    orphan_components = []

    rel_re = re.compile(
        r'RELATIONSHIP:\s*(.+?)\s*->\s*(.+?)\s*\|\s*type=(\w+)\s*(?:\|\s*description="([^"]*)")?',
        re.IGNORECASE
    )

    seen_rels = set()
    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue
        for line in msg.content.split("\n"):
            line = line.strip()
            m = rel_re.match(line)
            if m:
                from_comp = m.group(1).strip()
                to_comp = m.group(2).strip()
                rel_type = m.group(3).strip().lower()
                description = m.group(4) or ""
                key = (from_comp.lower(), to_comp.lower(), rel_type)
                if key not in seen_rels:
                    seen_rels.add(key)
                    rel = {
                        "from": from_comp,
                        "to": to_comp,
                        "type": rel_type,
                        "description": description,
                    }
                    relationships.append(rel)
                    if rel_type in ("reads", "writes", "flows_to"):
                        data_flows.append(rel)

    return {
        "relationships": relationships,
        "data_flows": data_flows,
        "orphan_components": orphan_components,
        "_parse_health": {
            "relationships_found": len(relationships),
            "hollow": len(relationships) == 0,
        },
    }


def parse_constrain_artifact(state: SharedState) -> Dict[str, Any]:
    """Parse CONSTRAINT:, METHOD:, STATES:, and ALGORITHM: from CONSTRAIN stage dialogue."""
    from core.digest import extract_dialogue_methods, extract_dialogue_state_machines, extract_dialogue_algorithms

    constraints = []
    constraint_re = re.compile(
        r'CONSTRAINT:\s*(.+?)\s*\|\s*description="([^"]+)"\s*(?:\|\s*derived_from="([^"]*)")?',
        re.IGNORECASE
    )

    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue
        for line in msg.content.split("\n"):
            line = line.strip()
            m = constraint_re.match(line)
            if m:
                applies_to = m.group(1).strip()
                description = m.group(2).strip()
                derived_from = m.group(3) or description
                constraints.append({
                    "description": description,
                    "applies_to": [applies_to],
                    "derived_from": derived_from,
                })

    methods = extract_dialogue_methods(state)
    state_machines = extract_dialogue_state_machines(state)
    algorithms = extract_dialogue_algorithms(state)

    return {
        "constraints": constraints,
        "methods": methods,
        "state_machines": state_machines,
        "algorithms": algorithms,
        "validation_rules": [],
        "_parse_health": {
            "constraints_found": len(constraints),
            "hollow": len(constraints) == 0,
        },
    }


def parse_architect_artifact(state: SharedState) -> Dict[str, Any]:
    """Parse SUBSYSTEM: lines from ARCHITECT stage dialogue."""
    subsystems = []
    lifecycle_order = []
    resolved_conflicts = []
    remaining_unknowns = list(state.unknown)

    sub_re = re.compile(
        r'SUBSYSTEM:\s*(.+?)\s*\|\s*contains=\[([^\]]+)\]\s*(?:\|\s*description="([^"]*)")?',
        re.IGNORECASE
    )

    for msg in state.history:
        if msg.sender not in ("Entity", "Process"):
            continue
        for line in msg.content.split("\n"):
            line = line.strip()
            m = sub_re.match(line)
            if m:
                name = m.group(1).strip()
                contains_str = m.group(2).strip()
                description = m.group(3) or ""
                contains = [c.strip() for c in contains_str.split(",") if c.strip()]
                subsystems.append({
                    "name": name,
                    "contains": contains,
                    "description": description,
                })

    return {
        "subsystems": subsystems,
        "lifecycle_order": lifecycle_order,
        "resolved_conflicts": resolved_conflicts,
        "remaining_unknowns": remaining_unknowns,
        "_parse_health": {
            "subsystems_found": len(subsystems),
            "hollow": len(subsystems) == 0,
        },
    }


ARTIFACT_PARSERS = {
    "expand": parse_expand_artifact,
    "decompose": parse_decompose_artifact,
    "ground": parse_ground_artifact,
    "constrain": parse_constrain_artifact,
    "architect": parse_architect_artifact,
}


# =============================================================================
# GATE PREDICATES (all deterministic — no LLM)
# =============================================================================

def gate_expand(artifact: Dict[str, Any], intent: Dict[str, Any]) -> StageResult:
    """
    Gate for EXPAND stage — structural containment validation.

    Validates: "is this a valid recursion map?" not "is this the right one?"

    Requirements:
    - >= 3 nodes
    - >= 2 containment relationships
    - Orphan ratio <= 20% (every node should participate in containment)
    - Self-reference loops must be well-formed (first == last in path, depth > 0)
    - max_depth > 0 when containment exists
    - Containment references only defined nodes (warning)
    - Explicit component coverage preserved (stem match on node names)
    """
    errors = []
    warnings = []

    nodes = artifact.get("nodes", [])
    containment = artifact.get("containment", [])
    self_references = artifact.get("self_references", [])
    max_depth = artifact.get("max_depth", 0)
    node_names = [n["name"] for n in nodes]
    node_name_set = {n["name"] for n in nodes}

    # Minimum node count
    if len(nodes) < 3:
        errors.append(f"Too few nodes: {len(nodes)} < 3")
    if len(nodes) > 30:
        warnings.append(f"Excessive nodes ({len(nodes)}): likely extracting vocabulary, not structure. Aim for 8-20.")

    # Minimum containment relationships
    if len(containment) < 2:
        errors.append(f"Too few containment relationships: {len(containment)} < 2")

    # Orphan ratio: nodes not in any containment relationship
    if nodes:
        containment_names = set()
        for c in containment:
            containment_names.add(c["parent"])
            containment_names.add(c["child"])
        orphans = [n for n in node_names if n not in containment_names]
        orphan_ratio = len(orphans) / len(nodes)
        if orphan_ratio > 0.2:
            warnings.append(f"High orphan ratio: {orphan_ratio:.0%} ({orphans[:3]})")

    # Self-reference validation: paths must loop (first == last), depth > 0
    for sr in self_references:
        path = sr.get("path", [])
        if len(path) < 2 or path[0] != path[-1]:
            warnings.append(f"Self-reference '{sr.get('node', '?')}' path does not loop: {path}")
        if sr.get("depth", 0) <= 0:
            warnings.append(f"Self-reference '{sr.get('node', '?')}' has depth <= 0")

    # max_depth must be > 0 when containment exists
    if containment and max_depth <= 0:
        warnings.append(f"max_depth is {max_depth} but containment relationships exist")

    # Containment references only defined nodes (warning)
    for c in containment:
        if c["parent"] not in node_name_set:
            warnings.append(f"Containment parent '{c['parent']}' not in declared nodes")
        if c["child"] not in node_name_set:
            warnings.append(f"Containment child '{c['child']}' not in declared nodes")

    # Check explicit component coverage (stem match on node names)
    explicit = intent.get("explicit_components", [])
    if explicit:
        node_stems = {n.lower()[:5] for n in node_names}
        for comp in explicit:
            comp_stem = comp.lower()[:5]
            if not any(comp_stem in s or s in comp_stem for s in node_stems):
                comp_words = set(comp.lower().split())
                node_words = set()
                for n in node_names:
                    node_words.update(n.lower().split())
                if not comp_words & node_words:
                    warnings.append(f"Explicit component '{comp}' not found in nodes")

    return StageResult(success=len(errors) == 0, errors=errors, warnings=warnings)


def gate_decompose(artifact: Dict[str, Any], expand_artifact: Dict[str, Any]) -> StageResult:
    """
    Gate for DECOMPOSE stage — boundary crystallization validation.

    Requirements:
    - >= 3 components (error)
    - <= 25 components (warning)
    - >= 70% node coverage (components + folded nodes vs EXPAND nodes)
    - All components have derived_from (warning)
    - Parent node accounting: every parent from EXPAND containment accounted for (warning)
    - Self-reference accounting: every self-ref addressed as INTERFACE or BREAK (warning)

    Phase 15: Classification-aware warnings for low-confidence components.
    """
    errors = []
    warnings = []

    components = artifact.get("components", [])
    type_assignments = artifact.get("type_assignments", {})
    folded = artifact.get("folded", [])
    interfaces = artifact.get("interfaces", [])
    nouns = expand_artifact.get("nouns", [])

    if len(components) < 3:
        errors.append(f"Too few components: {len(components)} < 3")
    if len(components) > 25:
        warnings.append(
            f"Excessive components ({len(components)}): likely promoting attributes/states "
            f"to components. Aim for 8-20 real architectural components."
        )

    # Check node coverage — folded nodes count toward coverage
    if nouns:
        folded_names = {f["child"].lower() for f in folded}
        comp_names_lower = {c["name"].lower() for c in components}
        covered = sum(
            1 for n in nouns
            if n.lower() in comp_names_lower
            or n in type_assignments
            or n.lower() in folded_names
            or any(n.lower() in cn or cn in n.lower() for cn in comp_names_lower)
        )
        ratio = covered / len(nouns)
        if ratio < 0.7:
            warnings.append(f"Low node coverage: {ratio:.0%} < 70%")

    # Check derived_from
    missing_derivation = [c["name"] for c in components if not c.get("derived_from")]
    if missing_derivation:
        warnings.append(f"Components missing derived_from: {missing_derivation[:3]}")

    # Phase 15: Classification quality checks
    classification_scores = artifact.get("classification_scores", [])
    classification_rejected = artifact.get("classification_rejected", [])

    if classification_rejected:
        warnings.append(
            f"Classification rejected {len(classification_rejected)} candidates: "
            f"{classification_rejected[:5]}"
        )

    # Flag components needing LLM review (confidence 0.3-0.6)
    from core.classification import needs_llm_fallback
    uncertain = [s.name for s in classification_scores if needs_llm_fallback(s)]
    if uncertain:
        warnings.append(
            f"Low-confidence classifications (may need LLM review): {uncertain[:5]}"
        )

    # Parent node accounting: every parent from EXPAND containment must be
    # either a COMPONENT or explicitly FOLDed
    expand_containment = expand_artifact.get("containment", [])
    if expand_containment:
        parent_nodes = {c["parent"] for c in expand_containment}
        comp_names_lower = {c["name"].lower() for c in components}
        folded_children = {f["child"].lower() for f in folded}
        missing_parents = [
            p for p in parent_nodes
            if not any(p.lower() in cn or cn in p.lower() for cn in comp_names_lower)
            and p.lower() not in folded_children
        ]
        if missing_parents:
            warnings.append(
                f"Unaccounted parent nodes from EXPAND: {missing_parents[:5]}"
            )

    # Self-reference accounting: every self-ref from EXPAND should appear as
    # INTERFACE or BREAK. Case-insensitive, also check interface connects lists.
    expand_self_refs = expand_artifact.get("self_references", [])
    if expand_self_refs:
        interface_names_lower = {i["name"].lower() for i in interfaces}
        # Build a set of all nodes covered by interfaces (from connects lists)
        interface_covered_lower = set()
        for i in interfaces:
            interface_covered_lower.add(i["name"].lower())
            for c in i.get("connects", []):
                interface_covered_lower.add(c.lower())

        # Deduplicate self-refs by lowercase node name before checking
        seen_sr_lower = set()
        unaddressed = []
        for sr in expand_self_refs:
            sr_lower = sr["node"].lower()
            if sr_lower in seen_sr_lower:
                continue
            seen_sr_lower.add(sr_lower)
            if sr_lower not in interface_covered_lower:
                unaddressed.append(sr["node"])
        if unaddressed:
            warnings.append(
                f"Unaddressed self-references from EXPAND: {unaddressed[:5]}"
            )

    return StageResult(success=len(errors) == 0, errors=errors, warnings=warnings)


def gate_ground(artifact: Dict[str, Any], decompose_artifact: Dict[str, Any]) -> StageResult:
    """
    Gate for GROUND stage.

    Requirements:
    - >= N-1 relationships (connected graph minimum)
    - All endpoints reference known components
    - < 30% orphan ratio
    - Valid relationship types
    """
    errors = []
    warnings = []

    relationships = artifact.get("relationships", [])
    components = decompose_artifact.get("components", [])
    n_components = len(components)

    # Minimum relationships for connected graph
    min_rels = max(n_components - 1, 2)
    if len(relationships) < min_rels:
        errors.append(f"Too few relationships: {len(relationships)} < {min_rels}")

    # Check endpoints
    known_names = {c["name"].lower() for c in components}
    valid_types = {
        "contains", "triggers", "accesses", "depends_on", "flows_to",
        "generates", "snapshots", "propagates", "constrained_by",
        "bidirectional", "reads", "writes", "monitors", "uses",
        "calls", "creates", "updates",
    }

    invalid_endpoints = []
    invalid_types = []
    for rel in relationships:
        from_c = rel["from"].lower()
        to_c = rel["to"].lower()
        # Fuzzy match: check if any component name is a substring
        from_ok = from_c in known_names or any(from_c in k or k in from_c for k in known_names)
        to_ok = to_c in known_names or any(to_c in k or k in to_c for k in known_names)
        if not from_ok:
            invalid_endpoints.append(rel["from"])
        if not to_ok:
            invalid_endpoints.append(rel["to"])
        if rel["type"].lower() not in valid_types:
            invalid_types.append(rel["type"])

    if invalid_endpoints:
        warnings.append(f"Unknown endpoints: {invalid_endpoints[:5]}")
    if invalid_types:
        warnings.append(f"Invalid relationship types: {invalid_types[:3]}")

    # Orphan ratio
    connected = set()
    for rel in relationships:
        connected.add(rel["from"].lower())
        connected.add(rel["to"].lower())
    orphans = known_names - connected
    if known_names:
        orphan_ratio = len(orphans) / len(known_names)
        if orphan_ratio >= 0.3:
            warnings.append(f"High orphan ratio: {orphan_ratio:.0%} ({list(orphans)[:3]})")

    return StageResult(success=len(errors) == 0, errors=errors, warnings=warnings)


def gate_constrain(artifact: Dict[str, Any], decompose_artifact: Dict[str, Any],
                   intent: Dict[str, Any]) -> StageResult:
    """
    Gate for CONSTRAIN stage.

    Requirements:
    - All constraint targets reference known components
    - >= 50% of input constraints captured
    """
    errors = []
    warnings = []

    constraints = artifact.get("constraints", [])
    components = decompose_artifact.get("components", [])
    known_names = {c["name"].lower() for c in components}
    input_constraints = intent.get("constraints", [])

    # Check constraint targets
    for c in constraints:
        for target in c.get("applies_to", []):
            target_lower = target.lower()
            if not any(target_lower in k or k in target_lower for k in known_names):
                warnings.append(f"Constraint target '{target}' not a known component")

    # Check input constraint coverage
    if input_constraints:
        captured = 0
        for ic in input_constraints:
            ic_lower = ic.lower()
            ic_words = set(ic_lower.split())
            for c in constraints:
                desc_lower = c.get("description", "").lower()
                desc_words = set(desc_lower.split())
                if len(ic_words & desc_words) >= 2:
                    captured += 1
                    break
        ratio = captured / len(input_constraints) if input_constraints else 1.0
        if ratio < 0.5:
            warnings.append(f"Low constraint coverage: {ratio:.0%} < 50%")

    return StageResult(success=True, errors=errors, warnings=warnings)


def gate_architect(artifact: Dict[str, Any], decompose_artifact: Dict[str, Any]) -> StageResult:
    """
    Gate for ARCHITECT stage.

    Lenient: warnings only. Architecture is subjective.
    """
    warnings = []

    subsystems = artifact.get("subsystems", [])
    components = decompose_artifact.get("components", [])
    known_names = {c["name"].lower() for c in components}

    # Validate subsystem references
    for sub in subsystems:
        for contained in sub.get("contains", []):
            if contained.lower() not in known_names:
                warnings.append(f"Subsystem '{sub['name']}' references unknown component '{contained}'")

    return StageResult(success=True, errors=[], warnings=warnings)


PIPELINE_GATES = {
    "expand": gate_expand,
    "decompose": gate_decompose,
    "ground": gate_ground,
    "constrain": gate_constrain,
    "architect": gate_architect,
}


# =============================================================================
# PRIME CONTENT BUILDERS
# =============================================================================

def _build_expand_prime(pipeline: PipelineState) -> str:
    """Build prime content for EXPAND stage."""
    intent = pipeline.intent

    # Truncate input to avoid meta-language flooding
    input_text = pipeline.original_input
    if len(input_text) > 2000:
        input_text = input_text[:2000] + "\n[... truncated ...]"

    # Include explicit components to guide extraction
    explicit = intent.get("explicit_components", [])
    canonical_hint = ""
    if explicit:
        canonical_hint = f"\nKNOWN COMPONENTS (these must appear as nodes): {', '.join(explicit)}\n"

    return f"""Map the CONTAINMENT STRUCTURE of this system — what contains what, and where patterns repeat at different scales.

USER INPUT: {input_text}

EXTRACTED INTENT:
- Core need: {intent.get('core_need', 'unknown')}
- Domain: {intent.get('domain', 'unknown')}
- Actors: {', '.join(intent.get('actors', []))}
- Key insight: {intent.get('insight', '')}
{canonical_hint}
Output NODE: lines for concepts, CONTAINS: lines for containment, SELF_REF: for recursion.
Remember INSIGHT: line."""


def _build_decompose_prime(pipeline: PipelineState) -> str:
    """Build prime content for DECOMPOSE stage — includes EXPAND artifact."""
    expand = pipeline.get_artifact("expand") or {}

    # Read new recursion map keys, fall back to bridge keys
    nodes = expand.get("nodes", [])
    containment = expand.get("containment", [])
    self_references = expand.get("self_references", [])
    max_depth = expand.get("max_depth", 0)
    nouns = expand.get("nouns", [])

    # Format containment structure for DECOMPOSE
    containment_section = ""
    if nodes:
        node_lines = []
        for n in nodes:
            source = n.get("source", "")
            source_hint = f' (source: "{source}")' if source else ""
            node_lines.append(f"  - {n['name']} [depth: {n.get('depth', 0)}]{source_hint}")
        containment_section += f"\nNODES ({len(nodes)} total):\n" + "\n".join(node_lines)
    if containment:
        cont_lines = [f"  - {c['parent']} > {c['child']}" for c in containment]
        containment_section += f"\n\nCONTAINMENT ({len(containment)} relationships):\n" + "\n".join(cont_lines)
    if self_references:
        ref_lines = [f"  - {sr['node']} (path: {' > '.join(sr['path'])}, depth: {sr['depth']})" for sr in self_references]
        containment_section += f"\n\nSELF-REFERENCES ({len(self_references)} loops):\n" + "\n".join(ref_lines)
    if max_depth:
        containment_section += f"\n\nMAX_DEPTH: {max_depth}"

    # Flat node list as fallback
    if not containment_section and nouns:
        containment_section = f"\nNODES (flat): {', '.join(nouns)}"

    # Include canonical components as guidance
    explicit = pipeline.intent.get("explicit_components", [])
    canonical_hint = ""
    if explicit:
        canonical_hint = f"\nKNOWN EXPECTED COMPONENTS (must be included): {', '.join(explicit)}\n"

    return f"""From the EXPAND stage, we mapped the recursion field. Now CRYSTALLIZE BOUNDARIES — decide which nodes get their own identity vs which fold into parents.
{containment_section}
{canonical_hint}
USER INPUT (reference): {pipeline.original_input[:500]}

DEPTH GUIDANCE:
- Depth 0 nodes → likely subsystems or top-level entities (COMPONENT with type=subsystem or entity)
- Depth 1 nodes → entities or processes with their own identity
- Depth 2+ nodes → likely attributes to FOLD into their parent

BOUNDARY DECISIONS:
- Parent nodes with children → real components with boundary=[children]
- Leaf nodes that are fields/attributes → FOLD into parent (e.g. Known, Unknown → fold INTO SharedState)
- Leaf nodes with own lifecycle → PROMOTE to component
- States (AWAITING_INPUT, DIALOGUE) → state machine values, NOT components — fold them
- Pipeline phases → workflow steps, NOT components — fold them
- Self-reference loops → address as INTERFACE or BREAK

Output COMPONENT: lines for real boundaries, FOLD: for attributes, PROMOTE: for children deserving identity.
Output METHOD: for actions, INTERFACE: for recursive patterns, BREAK: for incidental loops.
Aim for 10-20 components. Remember INSIGHT: line."""


def _build_ground_prime(pipeline: PipelineState) -> str:
    """Build prime content for GROUND stage — includes DECOMPOSE artifact."""
    decompose = pipeline.get_artifact("decompose") or {}
    components = decompose.get("components", [])

    # Include canonical components that may not be in DECOMPOSE but are expected
    explicit = pipeline.intent.get("explicit_components", [])
    declared_names = {c["name"].lower() for c in components}
    all_comps = list(components)
    for name in explicit:
        if name.lower() not in declared_names:
            all_comps.append({"name": name, "type": "entity"})
            declared_names.add(name.lower())

    comp_list = "\n".join(f"  - {c['name']} ({c['type']})" for c in all_comps)

    # Handoff context from DECOMPOSE
    handoff_hint = ""
    if pipeline.current_handoff and pipeline.current_handoff.stage_source == "decompose":
        h = pipeline.current_handoff
        if h.key_entities:
            handoff_hint = f"\nPreviously identified entities: {', '.join(h.key_entities)}\n"

    # Folded names — tell GROUND which names were absorbed so it uses canonical names
    folded = decompose.get("folded", [])
    fold_hint = ""
    if folded:
        fold_lines = "\n".join(
            f"  - \"{f['child']}\" was folded into {f['into']}"
            for f in folded
        )
        fold_hint = f"""
IMPORTANT — These names were folded into existing components. Do NOT reference them
as relationship endpoints. Use the canonical component name instead:
{fold_lines}
"""

    return f"""From DECOMPOSE, we have these components:

{comp_list}
{handoff_hint}{fold_hint}
USER INPUT (reference): {pipeline.original_input[:500]}

Now map RELATIONSHIPS between these components. Output RELATIONSHIP: lines.
All components in the list above should participate in at least one relationship.
Remember INSIGHT: line."""


def _build_constrain_prime(pipeline: PipelineState) -> str:
    """Build prime content for CONSTRAIN stage."""
    decompose = pipeline.get_artifact("decompose") or {}
    ground = pipeline.get_artifact("ground") or {}
    components = decompose.get("components", [])
    relationships = ground.get("relationships", [])
    comp_lines = "\n".join(f"  - {c['name']} ({c['type']})" for c in components)
    rel_count = len(relationships)

    # Include explicit method signatures from the input if available
    methods_hint = ""
    input_lower = pipeline.original_input.lower()
    if "method" in input_lower or "def " in input_lower or "->" in pipeline.original_input:
        methods_hint = """
IMPORTANT: The input describes specific methods with signatures.
Look for lines like "method_name(param: Type) -> ReturnType" and extract them as:
METHOD: ComponentName.method_name(param: Type) -> ReturnType
Use the EXACT component name (no spaces). E.g. SharedState not Shared State."""

    # Handoff context from GROUND
    handoff_hint = ""
    if pipeline.current_handoff and pipeline.current_handoff.stage_source == "ground":
        h = pipeline.current_handoff
        if h.relationship_hints:
            rel_strs = [pair[0] for pair in h.relationship_hints[:10]]
            handoff_hint = f"\nKnown relationships: {', '.join(rel_strs)}\n"

    # Domain-specific constraint categories (#152 constraint-creative)
    domain_constraint_hint = ""
    _adapter = getattr(pipeline, "known", {}).get("_domain_adapter")
    if _adapter and hasattr(_adapter, "vocabulary") and _adapter.vocabulary:
        domain_name = getattr(_adapter, "domain", "")
        if domain_name == "software":
            domain_constraint_hint = (
                "\nAlso extract non-technical constraints if present: "
                "UX constraints (usability, accessibility), performance constraints "
                "(response time, throughput), business constraints (compliance, SLA), "
                "aesthetic constraints (branding, visual consistency).\n"
            )
        elif domain_name == "process":
            domain_constraint_hint = (
                "\nAlso extract non-technical constraints if present: "
                "timing constraints (deadlines, cadence), resource constraints "
                "(budget, personnel), quality constraints (standards, benchmarks), "
                "stakeholder constraints (approvals, sign-offs).\n"
            )

    return f"""Components:
{comp_lines}

Relationships: {rel_count} mapped
{handoff_hint}
USER INPUT (reference): {pipeline.original_input[:1000]}
{methods_hint}{domain_constraint_hint}
Extract CONSTRAINTS, METHODS, and STATE MACHINES for these components.
Remember INSIGHT: line."""


def _build_architect_prime(pipeline: PipelineState) -> str:
    """Build prime content for ARCHITECT stage."""
    decompose = pipeline.get_artifact("decompose") or {}
    ground = pipeline.get_artifact("ground") or {}
    constrain = pipeline.get_artifact("constrain") or {}
    components = decompose.get("components", [])
    relationships = ground.get("relationships", [])
    constraints = constrain.get("constraints", [])
    comp_list = ", ".join(c["name"] for c in components)

    # Handoff context from CONSTRAIN
    handoff_hint = ""
    if pipeline.current_handoff and pipeline.current_handoff.stage_source == "constrain":
        h = pipeline.current_handoff
        if h.constraint_context:
            ctx_strs = list(h.constraint_context[:8])
            handoff_hint = f"\nActive constraints: {'; '.join(ctx_strs)}\n"

    return f"""Full picture so far:
Components ({len(components)}): {comp_list}
Relationships: {len(relationships)}
Constraints: {len(constraints)}
{handoff_hint}
Accumulated insights: {'; '.join(pipeline.all_insights[:10])}

USER INPUT (reference): {pipeline.original_input[:500]}

Define SUBSYSTEM boundaries, lifecycle order, and architectural decisions.
Remember INSIGHT: line."""


PRIME_BUILDERS = {
    "expand": _build_expand_prime,
    "decompose": _build_decompose_prime,
    "ground": _build_ground_prime,
    "constrain": _build_constrain_prime,
    "architect": _build_architect_prime,
}


# =============================================================================
# STAGE EXECUTOR
# =============================================================================

class StageExecutor:
    """Runs a single pipeline stage as a focused Entity<->Process sub-dialogue."""

    def __init__(
        self,
        stage_name: str,
        entity_prompt: str,
        process_prompt: str,
        llm_client,
        max_turns: int,
        min_turns: int,
        gate_fn,
        on_insight: Optional[Callable] = None,
        domain_adapter=None,
    ):
        self.stage_name = stage_name
        self.entity_prompt = entity_prompt
        self.process_prompt = process_prompt
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.gate_fn = gate_fn
        self.on_insight = on_insight or (lambda x: None)
        self.domain_adapter = domain_adapter

    def run(self, pipeline: PipelineState, prime_content: str,
            retry_context: str = "") -> StageRecord:
        """
        Run one stage's sub-dialogue and check its gate.

        Args:
            pipeline: Current pipeline state (for prior insights as context)
            prime_content: Stage-specific prime content
            retry_context: If retrying, gate errors from previous attempt

        Returns:
            StageRecord with dialogue state, parsed artifact, and gate result
        """
        from agents.spec_agents import create_entity_agent, create_process_agent, add_challenge_protocol
        from agents.swarm import create_governor

        start_time = time.time()

        # Create fresh agents with stage-specific prompts
        entity_agent = add_challenge_protocol(
            _create_stage_agent("Entity", self.entity_prompt, self.llm_client)
        )
        process_agent = add_challenge_protocol(
            _create_stage_agent("Process", self.process_prompt, self.llm_client)
        )

        # Fresh SharedState per stage
        state = SharedState()
        state.known["input"] = pipeline.original_input
        state.known["intent"] = pipeline.intent
        state.personas = list(pipeline.personas)

        # Inject domain adapter reference for artifact parsers
        if self.domain_adapter:
            state.known["_domain_adapter"] = self.domain_adapter

        # Inject prior insights as context
        if pipeline.all_insights:
            state.known["prior_insights"] = list(pipeline.all_insights)

        # Inject structured handoff from previous stage
        if pipeline.current_handoff and pipeline.current_handoff.stage_source:
            state.known["_handoff"] = asdict(pipeline.current_handoff)

        # Build prime message
        if retry_context:
            prime_content = f"{prime_content}\n\nRETRY: Previous attempt failed:\n{retry_context}"

        prime_msg = Message(
            sender="System",
            content=prime_content,
            message_type=MessageType.PROPOSITION
        )

        # Run dialogue
        protocol = DialogueProtocol(
            max_turns=self.max_turns,
            min_turns=self.min_turns,
            min_insights=1
        )
        protocol.register(["Entity", "Process"])

        agents = {"Entity": entity_agent, "Process": process_agent}
        last_msg = None
        consecutive_agreements = 0

        while protocol.turn_count < self.max_turns:
            agent_name = protocol.next_turn()
            agent = agents[agent_name]

            input_for_agent = prime_msg if protocol.turn_count == 1 else last_msg

            # Turn-level retry for transient LLM errors
            response = None
            for turn_attempt in range(_TURN_MAX_RETRIES + 1):
                try:
                    response = agent.run(state, input_for_agent)
                    break
                except (_ProviderError, _TimeoutError) as e:
                    if turn_attempt < _TURN_MAX_RETRIES:
                        delay = 1.0 * (2 ** turn_attempt)
                        logger.warning(
                            f"Transient error in {self.stage_name}/{agent_name} "
                            f"(attempt {turn_attempt + 1}/{_TURN_MAX_RETRIES + 1}): {e}. "
                            f"Retrying in {delay:.0f}s..."
                        )
                        time.sleep(delay)
                    else:
                        raise  # Exhausted retries, propagate

            state.add_message(response)

            if response.insight_display:
                self.on_insight(f"  → [{self.stage_name}] {response.insight_display}")

            # Early termination: 2 consecutive agreements after min_turns
            if response.message_type == MessageType.AGREEMENT:
                consecutive_agreements += 1
            else:
                consecutive_agreements = 0

            if (consecutive_agreements >= PROTOCOL.pipeline.early_termination_agreements and
                    protocol.turn_count >= self.min_turns):
                break

            last_msg = response

        turn_count = protocol.turn_count
        duration = time.time() - start_time

        # Parse artifact from dialogue
        parser = ARTIFACT_PARSERS.get(self.stage_name, lambda s: {})
        artifact = parser(state)

        # Check gate
        gate_result = self._check_gate(artifact, pipeline)

        return StageRecord(
            name=self.stage_name,
            state=state,
            artifact=artifact,
            gate_result=gate_result,
            turn_count=turn_count,
            duration_seconds=duration,
        )

    def _check_gate(self, artifact: Dict[str, Any],
                    pipeline: PipelineState) -> StageResult:
        """Check the gate predicate for this stage."""
        if self.stage_name == "expand":
            return self.gate_fn(artifact, pipeline.intent)
        elif self.stage_name == "decompose":
            expand_art = pipeline.get_artifact("expand") or {}
            return self.gate_fn(artifact, expand_art)
        elif self.stage_name == "ground":
            decompose_art = pipeline.get_artifact("decompose") or {}
            return self.gate_fn(artifact, decompose_art)
        elif self.stage_name == "constrain":
            decompose_art = pipeline.get_artifact("decompose") or {}
            return self.gate_fn(artifact, decompose_art, pipeline.intent)
        elif self.stage_name == "architect":
            decompose_art = pipeline.get_artifact("decompose") or {}
            return self.gate_fn(artifact, decompose_art)
        else:
            return StageResult(success=True)


def _create_stage_agent(name: str, system_prompt: str, llm_client) -> 'LLMAgent':
    """Create an LLMAgent with a stage-specific prompt."""
    from agents.base import LLMAgent
    perspective = "structure" if name == "Entity" else "behavior"
    return LLMAgent(
        name=name,
        perspective=perspective,
        system_prompt=system_prompt,
        llm_client=llm_client,
    )


# =============================================================================
# STAGED PIPELINE ORCHESTRATOR
# =============================================================================

class StagedPipeline:
    """
    Orchestrates 5 focused sub-dialogues in sequence.

    Each stage gets:
    - Fresh Entity/Process agents with stage-specific prompts
    - Fresh SharedState (confidence resets per stage)
    - Prior insights as read-only context
    - Deterministic gate check on output

    Failure mode:
    - Retry failed stage once with gate errors as context
    - Stages 1-2: abort on 2nd failure
    - Stages 3-5: proceed with warning on 2nd failure
    """

    def __init__(
        self,
        llm_client,
        on_insight: Optional[Callable] = None,
        domain_adapter=None,
    ):
        self.llm_client = llm_client
        self.on_insight = on_insight or (lambda x: None)
        self.domain_adapter = domain_adapter

    def run(self, pipeline: PipelineState) -> PipelineState:
        """
        Run all 5 stages in sequence.

        Args:
            pipeline: PipelineState with original_input, intent, personas

        Returns:
            PipelineState with all stages completed
        """
        for stage_name, max_turns, min_turns, timeout in STAGE_CONFIGS:
            self.on_insight(f"◇ Stage: {stage_name.upper()}")

            entity_prompt, process_prompt = STAGE_PROMPTS[stage_name]
            gate_fn = PIPELINE_GATES[stage_name]
            prime_fn = PRIME_BUILDERS[stage_name]

            # Inject adapter domain context into stage prompts
            if self.domain_adapter and self.domain_adapter.prompts:
                ap = self.domain_adapter.prompts
                if ap.entity_system_prompt:
                    entity_prompt = (
                        f"DOMAIN CONTEXT:\n{ap.entity_system_prompt}\n\n"
                        f"STAGE INSTRUCTIONS:\n{entity_prompt}"
                    )
                if ap.process_system_prompt:
                    process_prompt = (
                        f"DOMAIN CONTEXT:\n{ap.process_system_prompt}\n\n"
                        f"STAGE INSTRUCTIONS:\n{process_prompt}"
                    )

            executor = StageExecutor(
                stage_name=stage_name,
                entity_prompt=entity_prompt,
                process_prompt=process_prompt,
                llm_client=self.llm_client,
                max_turns=max_turns,
                min_turns=min_turns,
                gate_fn=gate_fn,
                on_insight=self.on_insight,
                domain_adapter=self.domain_adapter,
            )

            # Build prime content
            prime_content = prime_fn(pipeline)

            # Stage-level retry: if all turn retries exhausted, retry the whole stage once
            record = None
            for stage_attempt in range(2):  # max 2 stage attempts
                try:
                    # First attempt
                    record = executor.run(pipeline, prime_content)

                    # Retry if gate failed
                    if not record.gate_result.success:
                        self.on_insight(f"  ⚠ Gate failed for {stage_name}: {record.gate_result.errors}")

                        # Build retry context from gate errors
                        retry_ctx = "\n".join(record.gate_result.errors + record.gate_result.warnings)

                        # Second attempt (gate retry)
                        record = executor.run(pipeline, prime_content, retry_context=retry_ctx)

                        if not record.gate_result.success:
                            # Phase 17.3: Hollow artifact detection — abort on critical stages
                            artifact = record.artifact
                            parse_health = artifact.get("_parse_health", {}) if artifact else {}
                            is_hollow = parse_health.get("hollow", False)

                            if is_hollow and stage_name in ("expand", "decompose"):
                                from core.exceptions import CompilationError
                                raise CompilationError(
                                    f"Stage {stage_name} produced hollow artifact after retry",
                                    stage="dialogue",
                                    error_code="E3003",
                                )

                            self.on_insight(
                                f"  ⚠ Gate still failed for {stage_name} after retry. "
                                f"Errors: {record.gate_result.errors}"
                            )

                    break  # Success (or gate failure handled above) — don't retry stage

                except (_ProviderError, _TimeoutError) as e:
                    if stage_attempt == 0:
                        delay = 5.0
                        logger.warning(
                            f"Transient error in stage {stage_name} after turn retries exhausted: {e}. "
                            f"Retrying entire stage in {delay:.0f}s..."
                        )
                        self.on_insight(f"  ⚠ Provider error in {stage_name}, retrying stage...")
                        time.sleep(delay)
                    else:
                        raise  # Second stage attempt also failed, propagate

            pipeline.add_stage(record)

            # Fracture detection: if the stage produced fractures, halt compilation
            if record.state.fractures:
                from core.protocol_spec import FractureSignal
                from core.exceptions import FractureError
                fracture = record.state.fractures[0]  # First fracture halts
                raise FractureError(
                    f"Intent fracture at {stage_name}",
                    stage=stage_name,
                    signal=FractureSignal(
                        stage=fracture["stage"],
                        competing_configs=fracture["competing_configs"],
                        collapsing_constraint=fracture["collapsing_constraint"],
                        agent=fracture.get("agent", "Entity"),
                    ),
                )

            self.on_insight(
                f"  ✓ {stage_name}: {record.turn_count} turns, "
                f"{len(record.state.insights)} insights, "
                f"{record.duration_seconds:.1f}s"
            )

        return pipeline


# =============================================================================
# SYNTHESIS HELPER: Format pre-computed artifacts for assembly prompt
# =============================================================================

def format_precomputed_structure(pipeline: PipelineState) -> str:
    """
    Format pipeline artifacts into a structured section for synthesis.

    Synthesis shifts from generative to assembly — the LLM formats
    pre-computed artifacts, doesn't reason from scratch.
    """
    sections = []

    # Components from DECOMPOSE + promote undeclared relationship endpoints + canonical
    decompose = pipeline.get_artifact("decompose") or {}
    ground = pipeline.get_artifact("ground") or {}
    components = list(decompose.get("components", []))
    declared_names = {c["name"].lower() for c in components}

    # Find components referenced in relationships but not declared
    for rel in ground.get("relationships", []):
        for endpoint_name in (rel["from"], rel["to"]):
            if endpoint_name.lower() not in declared_names:
                # Don't promote state/attribute names
                if not _is_likely_state_or_attribute(endpoint_name):
                    components.append({
                        "name": endpoint_name,
                        "type": "entity",
                        "derived_from": f"referenced in relationship: {rel['from']} -> {rel['to']}",
                    })
                    declared_names.add(endpoint_name.lower())

    # Ensure canonical/explicit components from intent are present
    explicit = pipeline.intent.get("explicit_components", [])
    for comp_name in explicit:
        if comp_name.lower() not in declared_names:
            if not _is_likely_state_or_attribute(comp_name):
                components.append({
                    "name": comp_name,
                    "type": "entity",
                    "derived_from": f"canonical component from intent",
                })
                declared_names.add(comp_name.lower())

    if components:
        comp_lines = []
        for c in components:
            boundary = c.get("boundary", [])
            boundary_hint = f", boundary=[{', '.join(boundary)}]" if boundary else ""
            comp_lines.append(
                f"  - {c['name']} (type={c['type']}{boundary_hint}, "
                f"derived_from=\"{c.get('derived_from', '')}\")"
            )
        sections.append("COMPONENTS (from DECOMPOSE):\n" + "\n".join(comp_lines))

    # Folded items from DECOMPOSE (informational for synthesis)
    folded_items = decompose.get("folded", [])
    if folded_items:
        fold_lines = [f"  - {f['child']} → {f['into']} ({f['reason']})" for f in folded_items]
        sections.append("FOLDED ATTRIBUTES (from DECOMPOSE):\n" + "\n".join(fold_lines))

    # Interfaces from DECOMPOSE (connective tissue for synthesis)
    decompose_interfaces = decompose.get("interfaces", [])
    if decompose_interfaces:
        iface_lines = []
        for i in decompose_interfaces:
            connects = ", ".join(i.get("connects", []))
            iface_lines.append(
                f"  - {i['name']} (pattern={i['pattern']}, connects=[{connects}])"
            )
        sections.append("INTERFACES (from DECOMPOSE):\n" + "\n".join(iface_lines))

    # Relationships from GROUND
    ground = pipeline.get_artifact("ground") or {}
    relationships = ground.get("relationships", [])
    if relationships:
        rel_lines = []
        for r in relationships:
            rel_lines.append(
                f"  - {r['from']} --{r['type']}--> {r['to']}: {r.get('description', '')}"
            )
        sections.append("RELATIONSHIPS (from GROUND):\n" + "\n".join(rel_lines))

    # Constraints & Methods from CONSTRAIN
    constrain = pipeline.get_artifact("constrain") or {}
    constraints = constrain.get("constraints", [])
    methods = constrain.get("methods", [])
    state_machines = constrain.get("state_machines", [])
    algorithms = constrain.get("algorithms", [])
    if constraints or methods or state_machines or algorithms:
        cm_lines = []
        for c in constraints:
            targets = ", ".join(c.get("applies_to", []))
            cm_lines.append(f"  CONSTRAINT: {targets} — {c['description']}")
        for m in methods:
            params_str = ", ".join(
                f"{p['name']}: {p['type_hint']}" for p in m.get("parameters", [])
            ) if m.get("parameters") else ""
            cm_lines.append(
                f"  METHOD: {m['component']}.{m['name']}({params_str}) -> {m.get('return_type', 'None')}"
            )
        for sm in state_machines:
            states_str = ", ".join(sm.get("states", []))
            cm_lines.append(f"  STATE_MACHINE: {sm['component']} [{states_str}]")
        for a in algorithms:
            steps_str = "; ".join(a.get("steps", []))
            cm_lines.append(f"  ALGORITHM: {a['component']}.{a['method_name']} [{steps_str}]")
        sections.append("CONSTRAINTS & METHODS (from CONSTRAIN):\n" + "\n".join(cm_lines))

    # Subsystems from ARCHITECT
    architect = pipeline.get_artifact("architect") or {}
    subsystems = architect.get("subsystems", [])
    if subsystems:
        sub_lines = []
        for s in subsystems:
            contains_str = ", ".join(s.get("contains", []))
            sub_lines.append(f"  - {s['name']} [contains: {contains_str}]: {s.get('description', '')}")
        sections.append("SUBSYSTEMS (from ARCHITECT):\n" + "\n".join(sub_lines))

    return "\n\n".join(sections) if sections else ""
