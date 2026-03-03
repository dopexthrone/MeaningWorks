"""
Motherlabs Dimensional Blueprint Schema — frozen dataclass hierarchy.

Phase A: Dimensional Blueprint Output
Derived from: DIMENSIONAL_BLUEPRINT.md convergence (Feb 2026)

Blueprint = dimensional semantic field. Nodes positioned in derived N-dimensional
space with fractal depth, silence zones, fragile edges, and typed edges.

This is a LEAF MODULE — zero project imports, only stdlib.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


@dataclass(frozen=True)
class DimensionAxis:
    """
    A single dimension in the blueprint's semantic space.

    Each axis is derived from compilation material, not predetermined.
    The compiler discovers which dimensions matter for this particular input.

    Derived from: DIMENSIONAL_BLUEPRINT.md — "Nodes positioned in derived N-dimensional space"
    """
    name: str                                     # Derived from material, not predetermined
    range_low: str                                # Low end description
    range_high: str                               # High end description
    exploration_depth: float                      # 0-1: how thoroughly explored
    derived_from: str                             # What produced this axis
    silence_zones: Tuple[str, ...] = ()           # Regions of this axis unexplored


@dataclass(frozen=True)
class NodePosition:
    """
    A component's position in the dimensional space.

    Derived from: DIMENSIONAL_BLUEPRINT.md — fractal schema {node, position, adjacent, constraints}
    """
    dimension_values: Tuple[Tuple[str, float], ...]   # (axis_name, 0-1 position) pairs
    confidence: float                                  # 0-1: positioning confidence

    def get_value(self, axis_name: str) -> float:
        """Get position on a specific axis. Returns 0.0 if axis not found."""
        for name, value in self.dimension_values:
            if name == axis_name:
                return value
        return 0.0


@dataclass(frozen=True)
class FragileEdge:
    """
    An edge in the blueprint that may break under change.

    Fragile edges represent connections that are under-validated,
    late-discovered, or span low-confidence dimensions.

    Derived from: DIMENSIONAL_BLUEPRINT.md — "fragile edges"
    """
    description: str
    affected_nodes: Tuple[str, ...]
    drift_risk: str                               # "high" / "medium" / "low"
    reasoning: str
    derived_from: str


@dataclass(frozen=True)
class DimensionalMetadata:
    """
    Complete dimensional metadata for a blueprint.

    This is the dimensional layer that sits alongside the flat blueprint.
    The flat blueprint (components, relationships, constraints, unresolved)
    remains the canonical output — dimensional metadata adds positioning
    and confidence information derived from the compilation process.

    Derived from: DIMENSIONAL_BLUEPRINT.md convergence
    """
    dimensions: Tuple[DimensionAxis, ...]
    node_positions: Tuple[Tuple[str, NodePosition], ...]   # (component_name, position) pairs
    fragile_edges: Tuple[FragileEdge, ...]
    silence_zones: Tuple[str, ...]
    confidence_trajectory: Tuple[float, ...]
    dimension_confidence: Tuple[Tuple[str, float], ...]    # (dim_name, final_confidence) pairs
    dialogue_depth: int
    stage_discovery: Tuple[Tuple[str, str], ...]           # (component_name, discovering_stage) pairs

    def get_position(self, component_name: str) -> NodePosition:
        """Get a component's position. Returns zero-position if not found."""
        for name, pos in self.node_positions:
            if name == component_name:
                return pos
        return NodePosition(dimension_values=(), confidence=0.0)

    def get_dimension(self, axis_name: str) -> DimensionAxis:
        """Get a dimension by name. Returns None if not found."""
        for dim in self.dimensions:
            if dim.name == axis_name:
                return dim
        return None


def serialize_dimensional_metadata(meta: DimensionalMetadata) -> dict:
    """
    Convert frozen DimensionalMetadata to JSON-serializable dict.

    Derived from: Phase A.1 — serialization for blueprint output.
    """
    return {
        "axes": [
            {
                "name": dim.name,
                "range_low": dim.range_low,
                "range_high": dim.range_high,
                "exploration_depth": dim.exploration_depth,
                "derived_from": dim.derived_from,
                "silence_zones": list(dim.silence_zones),
            }
            for dim in meta.dimensions
        ],
        "node_positions": {
            name: {
                "dimension_values": {
                    axis_name: value
                    for axis_name, value in pos.dimension_values
                },
                "confidence": pos.confidence,
            }
            for name, pos in meta.node_positions
        },
        "fragile_edges": [
            {
                "description": edge.description,
                "affected_nodes": list(edge.affected_nodes),
                "drift_risk": edge.drift_risk,
                "reasoning": edge.reasoning,
                "derived_from": edge.derived_from,
            }
            for edge in meta.fragile_edges
        ],
        "silence_zones": list(meta.silence_zones),
        "confidence_trajectory": list(meta.confidence_trajectory),
        "dimension_confidence": {
            name: conf
            for name, conf in meta.dimension_confidence
        },
        "dialogue_depth": meta.dialogue_depth,
        "stage_discovery": {
            name: stage
            for name, stage in meta.stage_discovery
        },
    }


def deserialize_dimensional_metadata(data: dict) -> DimensionalMetadata:
    """
    Reconstruct frozen DimensionalMetadata from JSON-serializable dict.

    Derived from: Phase A.1 — round-trip serialization.
    """
    dimensions = tuple(
        DimensionAxis(
            name=ax["name"],
            range_low=ax["range_low"],
            range_high=ax["range_high"],
            exploration_depth=ax["exploration_depth"],
            derived_from=ax["derived_from"],
            silence_zones=tuple(ax.get("silence_zones", ())),
        )
        for ax in data.get("axes", [])
    )

    node_positions = tuple(
        (
            name,
            NodePosition(
                dimension_values=tuple(
                    (axis, val)
                    for axis, val in pos_data["dimension_values"].items()
                ),
                confidence=pos_data["confidence"],
            ),
        )
        for name, pos_data in data.get("node_positions", {}).items()
    )

    fragile_edges = tuple(
        FragileEdge(
            description=edge["description"],
            affected_nodes=tuple(edge["affected_nodes"]),
            drift_risk=edge["drift_risk"],
            reasoning=edge["reasoning"],
            derived_from=edge["derived_from"],
        )
        for edge in data.get("fragile_edges", [])
    )

    dimension_confidence = tuple(
        (name, conf)
        for name, conf in data.get("dimension_confidence", {}).items()
    )

    stage_discovery = tuple(
        (name, stage)
        for name, stage in data.get("stage_discovery", {}).items()
    )

    return DimensionalMetadata(
        dimensions=dimensions,
        node_positions=node_positions,
        fragile_edges=fragile_edges,
        silence_zones=tuple(data.get("silence_zones", [])),
        confidence_trajectory=tuple(data.get("confidence_trajectory", [])),
        dimension_confidence=dimension_confidence,
        dialogue_depth=data.get("dialogue_depth", 0),
        stage_discovery=stage_discovery,
    )
