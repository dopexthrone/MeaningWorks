"""
Motherlabs Interface Schema — frozen dataclass hierarchy for interface contracts.

Phase B.1: Interface Extraction
Derived from: DIMENSIONAL_BLUEPRINT.md — parallel materialization requires
explicit interface contracts between adjacent blueprint nodes.

This is a LEAF MODULE — zero project imports, only stdlib.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataFlow:
    """
    A single data flow between two adjacent nodes.

    Describes what crosses the boundary: name, inferred type,
    and direction of flow.

    Derived from: DIMENSIONAL_BLUEPRINT.md — "blueprint declares interfaces"
    """
    name: str              # What flows (e.g. "compilation_state")
    type_hint: str         # Inferred type (e.g. "SharedState")
    direction: str         # "A_to_B" | "B_to_A" | "bidirectional"
    derived_from: str      # Traceability back to source


@dataclass(frozen=True)
class InterfaceConstraint:
    """
    A constraint that applies to the interface between two nodes.

    Extracted from blueprint constraints that mention both nodes,
    or global constraints that apply to the edge.

    Derived from: Phase B.1 — constraint matching across node pairs
    """
    description: str
    constraint_type: str   # "range" | "not_null" | "custom"
    derived_from: str


@dataclass(frozen=True)
class InterfaceContract:
    """
    Complete interface contract between two adjacent blueprint nodes.

    One contract per relationship. Contains all data flows, constraints,
    fragility assessment, and directionality.

    Derived from: DIMENSIONAL_BLUEPRINT.md — "parallel print heads:
    blueprint declares interfaces -> agents materialize nodes simultaneously
    -> no merge conflicts by construction"
    """
    node_a: str
    node_b: str
    relationship_type: str
    relationship_description: str
    data_flows: Tuple[DataFlow, ...]
    constraints: Tuple[InterfaceConstraint, ...]
    fragility: float       # 0-1 from dimensional metadata
    confidence: float      # 0-1 extraction confidence
    directionality: str    # "A_depends_on_B" | "B_depends_on_A" | "mutual"
    derived_from: str


@dataclass(frozen=True)
class InterfaceMap:
    """
    Complete set of interface contracts for a blueprint.

    The InterfaceMap is the prerequisite for parallel materialization:
    each agent can materialize its node knowing exactly what interfaces
    it must honor with adjacent nodes.

    Derived from: Phase B.1 — full interface extraction from blueprint
    """
    contracts: Tuple[InterfaceContract, ...]
    unmatched_relationships: Tuple[str, ...]
    extraction_confidence: float   # 0-1 overall confidence
    derived_from: str


def serialize_interface_map(imap: InterfaceMap) -> dict:
    """
    Convert frozen InterfaceMap to JSON-serializable dict.

    Derived from: Phase B.1 — serialization for blueprint output.
    """
    return {
        "contracts": [
            {
                "node_a": c.node_a,
                "node_b": c.node_b,
                "relationship_type": c.relationship_type,
                "relationship_description": c.relationship_description,
                "data_flows": [
                    {
                        "name": df.name,
                        "type_hint": df.type_hint,
                        "direction": df.direction,
                        "derived_from": df.derived_from,
                    }
                    for df in c.data_flows
                ],
                "constraints": [
                    {
                        "description": ic.description,
                        "constraint_type": ic.constraint_type,
                        "derived_from": ic.derived_from,
                    }
                    for ic in c.constraints
                ],
                "fragility": c.fragility,
                "confidence": c.confidence,
                "directionality": c.directionality,
                "derived_from": c.derived_from,
            }
            for c in imap.contracts
        ],
        "unmatched_relationships": list(imap.unmatched_relationships),
        "extraction_confidence": imap.extraction_confidence,
        "derived_from": imap.derived_from,
    }


def deserialize_interface_map(data: dict) -> InterfaceMap:
    """
    Reconstruct frozen InterfaceMap from JSON-serializable dict.

    Derived from: Phase B.1 — round-trip serialization.
    """
    contracts = tuple(
        InterfaceContract(
            node_a=c["node_a"],
            node_b=c["node_b"],
            relationship_type=c["relationship_type"],
            relationship_description=c.get("relationship_description", ""),
            data_flows=tuple(
                DataFlow(
                    name=df["name"],
                    type_hint=df["type_hint"],
                    direction=df["direction"],
                    derived_from=df.get("derived_from", ""),
                )
                for df in c.get("data_flows", [])
            ),
            constraints=tuple(
                InterfaceConstraint(
                    description=ic["description"],
                    constraint_type=ic["constraint_type"],
                    derived_from=ic.get("derived_from", ""),
                )
                for ic in c.get("constraints", [])
            ),
            fragility=c.get("fragility", 0.0),
            confidence=c.get("confidence", 0.0),
            directionality=c.get("directionality", "mutual"),
            derived_from=c.get("derived_from", ""),
        )
        for c in data.get("contracts", [])
    )

    return InterfaceMap(
        contracts=contracts,
        unmatched_relationships=tuple(data.get("unmatched_relationships", [])),
        extraction_confidence=data.get("extraction_confidence", 0.0),
        derived_from=data.get("derived_from", ""),
    )
