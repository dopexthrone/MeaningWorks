"""
Motherlabs Interface Extractor — algorithmic extraction of interface contracts.

Phase B.1.2: Extract pairwise interface contracts from dimensional blueprints.
Prerequisite for parallel materialization (B.2+).

Key design decision: interfaces extracted ALGORITHMICALLY, not by LLM.
Pure computation, no generation. Consistent with SRE identity.

Imports: core.interface_schema (B1.1) + core.dimensional (A.1)
"""

import math
from typing import Dict, Any, FrozenSet, List, Optional, Set, Tuple

from core.interface_schema import (
    DataFlow,
    InterfaceConstraint,
    InterfaceContract,
    InterfaceMap,
)
from core.dimensional import DimensionalMetadata


# =============================================================================
# Relationship-to-data-flow mapping
# Reuses pattern from codegen/generator.py:RELATIONSHIP_TO_METHOD
# =============================================================================

_RELATIONSHIP_TO_FLOW = {
    "triggers": ("trigger_signal", "Signal", "A_to_B"),
    "accesses": ("data_access", "Any", "B_to_A"),
    "monitors": ("monitoring_data", "Any", "B_to_A"),
    "contains": ("contained_ref", "Any", "A_to_B"),
    "snapshots": ("snapshot_data", "Any", "A_to_B"),
    "depends_on": ("dependency", "Any", "B_to_A"),
    "flows_to": ("flow_data", "Any", "A_to_B"),
    "generates": ("generated_output", "Any", "A_to_B"),
    "propagates": ("propagated_data", "Any", "A_to_B"),
    "constrained_by": ("constraint_ref", "Any", "B_to_A"),
    "bidirectional": ("shared_data", "Any", "bidirectional"),
}

# Component type heuristics for inferring data types
_TYPE_HINTS_BY_COMPONENT = {
    "sharedstate": "SharedState",
    "shared state": "SharedState",
    "confidencevector": "ConfidenceVector",
    "confidence vector": "ConfidenceVector",
    "message": "Message",
    "dialogueprotocol": "DialogueProtocol",
    "dialogue protocol": "DialogueProtocol",
    "conflictoracle": "ConflictOracle",
    "conflict oracle": "ConflictOracle",
    "corpus": "CompilationRecord",
}


def _infer_type_hint(
    component_name: str,
    relationship_type: str,
    type_hints: Optional[Dict[str, str]] = None,
) -> str:
    """
    Infer the type hint for data flowing to/from a component.

    Uses component name heuristics and relationship type.
    Falls back to "Any" when uncertain.

    Args:
        component_name: Component name to match
        relationship_type: Relationship type string
        type_hints: Optional domain-specific type hints (default: software)
    """
    hints = type_hints if type_hints is not None else _TYPE_HINTS_BY_COMPONENT
    name_lower = component_name.lower().replace("_", "").replace(" ", "")
    for key, hint in hints.items():
        if key.replace(" ", "") in name_lower:
            return hint

    # Relationship-specific defaults
    if relationship_type == "triggers":
        return "Signal"
    if relationship_type == "snapshots":
        return "Dict[str, Any]"

    return "Any"


def extract_data_flows(
    relationship: Dict[str, Any],
    comp_a_name: str,
    comp_b_name: str,
    relationship_flows: Optional[Dict[str, Tuple[str, str, str]]] = None,
    type_hints: Optional[Dict[str, str]] = None,
) -> Tuple[DataFlow, ...]:
    """
    Infer data flows from a relationship between two components.

    Uses relationship type to determine flow direction and data types.
    Reuses pattern from codegen/generator.py:RELATIONSHIP_TO_METHOD.

    Args:
        relationship: The relationship dict from the blueprint
        comp_a_name: Source component name (from)
        comp_b_name: Target component name (to)
        relationship_flows: Optional domain-specific flow mapping (default: software)
        type_hints: Optional domain-specific type hints (default: software)

    Returns:
        Tuple of DataFlow instances

    Derived from: Phase B.1.2 — data flow extraction
    """
    flows = relationship_flows if relationship_flows is not None else _RELATIONSHIP_TO_FLOW
    rel_type = relationship.get("type", "").lower()
    rel_desc = relationship.get("description", "")

    flow_info = flows.get(rel_type)
    if not flow_info:
        # Unknown relationship type — create generic bidirectional flow
        return (DataFlow(
            name=f"{rel_type}_data",
            type_hint="Any",
            direction="bidirectional",
            derived_from=f"relationship: {comp_a_name}->{comp_b_name} ({rel_type})",
        ),)

    flow_name, default_type, direction = flow_info

    # Try to infer better type hints from component names
    if direction == "A_to_B":
        type_hint = _infer_type_hint(comp_b_name, rel_type, type_hints)
    elif direction == "B_to_A":
        type_hint = _infer_type_hint(comp_b_name, rel_type, type_hints)
    else:
        type_hint = _infer_type_hint(comp_a_name, rel_type, type_hints)

    if type_hint == "Any":
        type_hint = default_type

    return (DataFlow(
        name=flow_name,
        type_hint=type_hint,
        direction=direction,
        derived_from=f"relationship: {comp_a_name}->{comp_b_name} ({rel_type}): {rel_desc[:80]}",
    ),)


def extract_interface_constraints(
    node_a: str,
    node_b: str,
    constraints: List[Dict[str, Any]],
) -> Tuple[InterfaceConstraint, ...]:
    """
    Extract constraints that apply to the interface between two nodes.

    Matches constraints that mention both nodes (by name) or apply to
    either node. Uses fuzzy matching (case-insensitive, partial).

    Args:
        node_a: First component name
        node_b: Second component name
        constraints: Blueprint constraints list

    Returns:
        Tuple of InterfaceConstraint instances

    Derived from: Phase B.1.2 — constraint matching
    """
    result = []
    a_lower = node_a.lower()
    b_lower = node_b.lower()

    for constraint in constraints:
        desc = constraint.get("description", "")
        applies_to = constraint.get("applies_to", [])
        desc_lower = desc.lower()

        # Check if constraint mentions both nodes
        mentions_a = (
            a_lower in desc_lower
            or any(a_lower in at.lower() for at in applies_to)
        )
        mentions_b = (
            b_lower in desc_lower
            or any(b_lower in at.lower() for at in applies_to)
        )

        if mentions_a and mentions_b:
            # Constraint spans both nodes — definitely an interface constraint
            ctype = _classify_constraint(desc)
            result.append(InterfaceConstraint(
                description=desc,
                constraint_type=ctype,
                derived_from=constraint.get("derived_from", desc),
            ))
        elif mentions_a or mentions_b:
            # Constraint applies to one node — may affect the interface
            # Only include if it's about data validation (range, not_null)
            ctype = _classify_constraint(desc)
            if ctype in ("range", "not_null"):
                result.append(InterfaceConstraint(
                    description=desc,
                    constraint_type=ctype,
                    derived_from=constraint.get("derived_from", desc),
                ))

    return tuple(result)


def _classify_constraint(description: str) -> str:
    """Classify a constraint description into a type."""
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in ("range", "between", "<=", ">=")):
        return "range"
    if any(kw in desc_lower for kw in ("not null", "required", "must not be none", "cannot be none")):
        return "not_null"
    return "custom"


def compute_edge_fragility(
    node_a: str,
    node_b: str,
    dim_meta: Optional[DimensionalMetadata],
) -> float:
    """
    Compute fragility score for an edge between two nodes.

    Sources:
    - FragileEdge mentions of either node
    - Position distance in dimensional space
    - Node confidence values

    Args:
        node_a: First component name
        node_b: Second component name
        dim_meta: Dimensional metadata (may be None)

    Returns:
        Fragility score in [0, 1] where 1 = maximally fragile

    Derived from: Phase B.1.2 — edge fragility computation
    """
    if dim_meta is None:
        return 0.5  # Unknown fragility

    fragility_signals = []

    # 1. Check fragile edges mentioning either node
    for edge in dim_meta.fragile_edges:
        affected = set(n.lower() for n in edge.affected_nodes)
        if node_a.lower() in affected or node_b.lower() in affected:
            risk_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            fragility_signals.append(risk_map.get(edge.drift_risk, 0.5))

    # 2. Position distance (larger distance = more fragile)
    pos_a = dim_meta.get_position(node_a)
    pos_b = dim_meta.get_position(node_b)

    if pos_a.dimension_values and pos_b.dimension_values:
        # Compute Euclidean distance in dimensional space
        dims_a = dict(pos_a.dimension_values)
        dims_b = dict(pos_b.dimension_values)
        all_dims = set(dims_a.keys()) | set(dims_b.keys())
        if all_dims:
            sq_sum = sum(
                (dims_a.get(d, 0.0) - dims_b.get(d, 0.0)) ** 2
                for d in all_dims
            )
            distance = math.sqrt(sq_sum / len(all_dims))  # Normalized
            fragility_signals.append(min(distance, 1.0))

    # 3. Node confidence (lower confidence = more fragile)
    avg_confidence = (pos_a.confidence + pos_b.confidence) / 2.0
    if avg_confidence > 0:
        fragility_signals.append(1.0 - avg_confidence)

    if not fragility_signals:
        return 0.5

    return sum(fragility_signals) / len(fragility_signals)


def determine_directionality(relationship: Dict[str, Any]) -> str:
    """
    Determine the directionality of a relationship.

    Mapping:
    - triggers/flows_to/generates/propagates/snapshots → A_depends_on_B (A drives B)
    - accesses/depends_on/constrained_by/monitors → B_depends_on_A (B serves A)
    - bidirectional/contains → mutual

    Args:
        relationship: The relationship dict

    Returns:
        "A_depends_on_B" | "B_depends_on_A" | "mutual"

    Derived from: Phase B.1.2 — directionality inference
    """
    rel_type = relationship.get("type", "").lower()

    a_drives_b = {"triggers", "flows_to", "generates", "propagates", "snapshots"}
    b_serves_a = {"accesses", "depends_on", "constrained_by", "monitors"}
    mutual_types = {"bidirectional", "contains"}

    if rel_type in a_drives_b:
        return "A_depends_on_B"
    elif rel_type in b_serves_a:
        return "B_depends_on_A"
    elif rel_type in mutual_types:
        return "mutual"
    else:
        return "mutual"  # Default for unknown types


def _fuzzy_find_component(name: str, component_names: Set[str]) -> Optional[str]:
    """
    Find a component by fuzzy matching against known names.

    Reuses logic from core.schema._fuzzy_match_component but returns
    the matched name instead of bool.
    """
    if name in component_names:
        return name

    name_lower = name.lower().strip()
    for cn in component_names:
        if cn.lower().strip() == name_lower:
            return cn

    # Plural/singular
    if name_lower.endswith('s'):
        for cn in component_names:
            if cn.lower().strip() == name_lower[:-1]:
                return cn
    for cn in component_names:
        if cn.lower().strip() == name_lower + 's':
            return cn

    return None


def extract_interface_map(
    blueprint: Dict[str, Any],
    dim_meta: Optional[DimensionalMetadata] = None,
    relationship_flows: Optional[Dict[str, Tuple[str, str, str]]] = None,
    type_hints: Optional[Dict[str, str]] = None,
) -> InterfaceMap:
    """
    Orchestrator: extract complete InterfaceMap from blueprint and dimensional metadata.

    Creates one InterfaceContract per relationship. Uses fuzzy component matching
    for endpoint resolution.

    Args:
        blueprint: The blueprint dict
        dim_meta: Optional dimensional metadata for fragility computation
        relationship_flows: Optional domain-specific flow mapping (default: software)
        type_hints: Optional domain-specific type hints (default: software)

    Returns:
        Frozen InterfaceMap with all contracts

    Derived from: Phase B.1.2 — full interface extraction
    """
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])
    constraints = blueprint.get("constraints", [])

    component_names = {c.get("name", "") for c in components if c.get("name")}

    contracts = []
    unmatched = []
    confidence_sum = 0.0

    for rel in relationships:
        from_name = rel.get("from", "")
        to_name = rel.get("to", "")
        rel_type = rel.get("type", "")
        rel_desc = rel.get("description", "")

        # Fuzzy match endpoints
        matched_from = _fuzzy_find_component(from_name, component_names)
        matched_to = _fuzzy_find_component(to_name, component_names)

        if not matched_from or not matched_to:
            unmatched.append(f"{from_name}->{to_name} ({rel_type})")
            continue

        # Extract data flows
        data_flows = extract_data_flows(rel, matched_from, matched_to, relationship_flows, type_hints)

        # Extract interface constraints
        iface_constraints = extract_interface_constraints(
            matched_from, matched_to, constraints
        )

        # Compute fragility
        fragility = compute_edge_fragility(matched_from, matched_to, dim_meta)

        # Determine directionality
        directionality = determine_directionality(rel)

        # Confidence: higher if we have dimensional metadata and constraints
        confidence = 0.5
        if dim_meta is not None:
            confidence += 0.2
        if iface_constraints:
            confidence += 0.1
        if data_flows and data_flows[0].type_hint != "Any":
            confidence += 0.2
        confidence = min(confidence, 1.0)

        confidence_sum += confidence

        contracts.append(InterfaceContract(
            node_a=matched_from,
            node_b=matched_to,
            relationship_type=rel_type,
            relationship_description=rel_desc,
            data_flows=data_flows,
            constraints=iface_constraints,
            fragility=fragility,
            confidence=confidence,
            directionality=directionality,
            derived_from=f"relationship: {matched_from}->{matched_to} ({rel_type})",
        ))

    # Overall extraction confidence
    total = len(contracts) + len(unmatched)
    extraction_confidence = confidence_sum / len(contracts) if contracts else 0.0
    if unmatched:
        extraction_confidence *= (len(contracts) / total)

    return InterfaceMap(
        contracts=tuple(contracts),
        unmatched_relationships=tuple(unmatched),
        extraction_confidence=extraction_confidence,
        derived_from="Phase B.1: algorithmic interface extraction",
    )
