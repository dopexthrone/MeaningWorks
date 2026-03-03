"""
Motherlabs Dimension Extractor — algorithmic extraction of dimensional metadata.

Phase A.2: Extract dimensions from SharedState and blueprint.

Key design decision: dimensions extracted ALGORITHMICALLY, not by LLM.
The LLM did its work during dialogue. Dimension extraction is deterministic
reduction — pure computation, no generation. Consistent with SRE identity.

Imports: core.dimensional (A.1) + core.protocol (SharedState, ConfidenceVector)
"""

from core.dimensional import (
    DimensionAxis,
    NodePosition,
    FragileEdge,
    DimensionalMetadata,
)
from core.protocol import SharedState, ConfidenceVector


# =============================================================================
# The four base dimensions derived from ConfidenceVector
# =============================================================================

_BASE_DIMENSIONS = {
    "structural": ("low structure", "high structure", "ConfidenceVector.structural"),
    "behavioral": ("low behavior", "high behavior", "ConfidenceVector.behavioral"),
    "coverage": ("low coverage", "high coverage", "ConfidenceVector.coverage"),
    "consistency": ("low consistency", "high consistency", "ConfidenceVector.consistency"),
}


def extract_dimensions(state: SharedState) -> tuple:
    """
    Extract dimension axes from SharedState.

    Sources:
    - ConfidenceVector dimensions (structural/behavioral/coverage/consistency)
    - Conflict categories (if conflicts span distinct topics, each topic is an axis)

    Each axis derived from actual dialogue data.
    Exploration depth = confidence value for that dimension.

    Returns:
        Tuple[DimensionAxis, ...]
    """
    axes = []

    # Base 4 axes from ConfidenceVector
    cv = state.confidence
    dim_values = {
        "structural": cv.structural,
        "behavioral": cv.behavioral,
        "coverage": cv.coverage,
        "consistency": cv.consistency,
    }

    for name, (low, high, source) in _BASE_DIMENSIONS.items():
        depth = dim_values[name]

        # Find silence zones: regions of this dimension below threshold
        silence = []
        if depth < 0.3:
            silence.append(f"{name} largely unexplored")

        axes.append(DimensionAxis(
            name=name,
            range_low=low,
            range_high=high,
            exploration_depth=depth,
            derived_from=source,
            silence_zones=tuple(silence),
        ))

    # Additional axes from conflict categories
    conflict_categories = set()
    for conflict in state.conflicts:
        category = conflict.get("category", "")
        if category and category not in conflict_categories:
            conflict_categories.add(category)

    for category in sorted(conflict_categories):
        # Conflicts reveal dimensions of disagreement
        axes.append(DimensionAxis(
            name=f"conflict:{category}",
            range_low=f"no {category} conflict",
            range_high=f"active {category} conflict",
            exploration_depth=0.5,  # Conflicts indicate partial exploration
            derived_from=f"ConflictOracle category: {category}",
        ))

    return tuple(axes)


def extract_node_positions(
    state: SharedState,
    blueprint: dict,
    dimensions: tuple,
) -> tuple:
    """
    Extract node positions in the dimensional space from blueprint components.

    Positioning logic:
    - Component type drives base position:
      entity=high structural, process=high behavioral
    - Relationship density affects coverage dimension
    - Confidence = overall confidence at time of extraction

    Returns:
        Tuple of (component_name, NodePosition) pairs
    """
    positions = []

    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    # Count relationships per component
    rel_counts = {}
    for rel in relationships:
        from_c = rel.get("from", "")
        to_c = rel.get("to", "")
        rel_counts[from_c] = rel_counts.get(from_c, 0) + 1
        rel_counts[to_c] = rel_counts.get(to_c, 0) + 1

    max_rels = max(rel_counts.values()) if rel_counts else 1

    # Build dimension name set for lookup
    dim_names = {d.name for d in dimensions}

    for comp in components:
        name = comp.get("name", "")
        comp_type = comp.get("type", "entity").lower()

        # Base positions from component type
        structural_pos = 0.5
        behavioral_pos = 0.5

        if comp_type in ("entity", "constraint"):
            structural_pos = 0.8
            behavioral_pos = 0.2
        elif comp_type in ("process", "event"):
            structural_pos = 0.2
            behavioral_pos = 0.8
        elif comp_type == "interface":
            structural_pos = 0.6
            behavioral_pos = 0.6
        elif comp_type == "subsystem":
            structural_pos = 0.7
            behavioral_pos = 0.5

        # Coverage position from relationship density
        density = rel_counts.get(name, 0) / max_rels if max_rels > 0 else 0.0
        coverage_pos = min(density, 1.0)

        # Consistency position from overall confidence
        consistency_pos = state.confidence.overall()

        # Build dimension values (only for dimensions that exist)
        dim_values = []
        if "structural" in dim_names:
            dim_values.append(("structural", structural_pos))
        if "behavioral" in dim_names:
            dim_values.append(("behavioral", behavioral_pos))
        if "coverage" in dim_names:
            dim_values.append(("coverage", coverage_pos))
        if "consistency" in dim_names:
            dim_values.append(("consistency", consistency_pos))

        # Confidence = overall confidence of the compilation
        confidence = state.confidence.overall()

        positions.append((
            name,
            NodePosition(
                dimension_values=tuple(dim_values),
                confidence=confidence,
            ),
        ))

    return tuple(positions)


def extract_fragile_edges(state: SharedState, blueprint: dict) -> tuple:
    """
    Extract fragile edges from state and blueprint.

    Sources:
    - Unresolved conflicts → fragile edges (direct mapping)
    - Components with few relationships → potentially orphaned
    - Low overall confidence → general fragility

    Returns:
        Tuple[FragileEdge, ...]
    """
    edges = []

    # 1. Unresolved conflicts → fragile edges
    for conflict in state.conflicts:
        if not conflict.get("resolved", False):
            topic = conflict.get("topic", "unknown")
            agents = conflict.get("agents", [])
            turn = conflict.get("turn", 0)

            edges.append(FragileEdge(
                description=f"Unresolved conflict: {topic}",
                affected_nodes=tuple(agents),
                drift_risk="high",
                reasoning=f"Conflict at turn {turn} never resolved",
                derived_from=f"ConflictOracle: {topic}",
            ))

    # 2. Orphan components (0 relationships) → fragile
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    connected = set()
    for rel in relationships:
        connected.add(rel.get("from", ""))
        connected.add(rel.get("to", ""))

    for comp in components:
        name = comp.get("name", "")
        if name and name not in connected:
            edges.append(FragileEdge(
                description=f"Orphan component: {name}",
                affected_nodes=(name,),
                drift_risk="medium",
                reasoning="Component has no relationships — may be disconnected",
                derived_from="Graph analysis: no incident edges",
            ))

    # 3. Low confidence dimensions → fragile region
    cv = state.confidence
    weak_dims = cv.needs_attention()
    if weak_dims:
        edges.append(FragileEdge(
            description=f"Weak dimensions: {', '.join(weak_dims)}",
            affected_nodes=tuple(weak_dims),
            drift_risk="medium",
            reasoning=f"Confidence below warning threshold in {len(weak_dims)} dimension(s)",
            derived_from="ConfidenceVector.needs_attention()",
        ))

    return tuple(edges)


def extract_silence_zones(state: SharedState, dimensions: tuple) -> tuple:
    """
    Extract silence zones — areas of the semantic space that were not explored.

    Sources:
    - Dimensions with exploration_depth < 0.3
    - Unresolved unknowns
    - Persona blind spots never challenged

    Returns:
        Tuple[str, ...]
    """
    zones = []

    # 1. Low-exploration dimensions
    for dim in dimensions:
        if dim.exploration_depth < 0.3:
            zones.append(f"Dimension '{dim.name}' underexplored (depth={dim.exploration_depth:.2f})")

    # 2. Unresolved unknowns
    for unknown in state.unknown:
        zones.append(f"Unresolved: {unknown}")

    # 3. Persona blind spots (personas with low perspective coverage)
    for persona in state.personas:
        if isinstance(persona, dict):
            blind_spots = persona.get("blind_spots", [])
            if isinstance(blind_spots, str):
                blind_spots = [blind_spots]
            if isinstance(blind_spots, list):
                for spot in blind_spots:
                    if isinstance(spot, str) and len(spot) > 1:
                        zones.append(f"Blind spot ({persona.get('name', 'unknown')}): {spot}")

    return tuple(zones)


def _extract_stage_discovery(state: SharedState, blueprint: dict) -> tuple:
    """
    Determine which pipeline stage discovered each component.

    Uses heuristic: component names mentioned in stage artifacts.
    Falls back to "synthesis" for components not found in stage data.

    Returns:
        Tuple of (component_name, stage_name) pairs
    """
    discoveries = []
    components = blueprint.get("components", [])

    # Try to find pipeline state in known (it's popped before corpus store,
    # but may still be available during extraction)
    pipeline_state = state.known.get("pipeline_state")

    if pipeline_state and hasattr(pipeline_state, 'stages'):
        # Map component names to the earliest stage that mentions them
        comp_to_stage = {}
        for record in pipeline_state.stages:
            artifact = record.artifact or {}
            artifact_str = str(artifact).lower()
            for comp in components:
                name = comp.get("name", "")
                if name and name.lower() in artifact_str and name not in comp_to_stage:
                    comp_to_stage[name] = record.name

        for comp in components:
            name = comp.get("name", "")
            if name:
                stage = comp_to_stage.get(name, "synthesis")
                discoveries.append((name, stage))
    else:
        # No pipeline state available — attribute all to synthesis
        for comp in components:
            name = comp.get("name", "")
            if name:
                discoveries.append((name, "synthesis"))

    return tuple(discoveries)


def build_dimensional_metadata(
    state: SharedState,
    blueprint: dict,
) -> DimensionalMetadata:
    """
    Orchestrator: build complete DimensionalMetadata from state and blueprint.

    Calls all individual extractors, assembles the frozen result.
    Pure computation — no LLM calls, no side effects.

    Args:
        state: SharedState from compilation
        blueprint: The flat blueprint dict

    Returns:
        Frozen DimensionalMetadata instance
    """
    dimensions = extract_dimensions(state)
    node_positions = extract_node_positions(state, blueprint, dimensions)
    fragile_edges = extract_fragile_edges(state, blueprint)
    silence_zones = extract_silence_zones(state, dimensions)
    stage_discovery = _extract_stage_discovery(state, blueprint)

    # Dimension confidence = final value of each base dimension
    cv = state.confidence
    dim_confidence = (
        ("structural", cv.structural),
        ("behavioral", cv.behavioral),
        ("coverage", cv.coverage),
        ("consistency", cv.consistency),
    )

    # Dialogue depth = total history length
    dialogue_depth = len(state.history)

    return DimensionalMetadata(
        dimensions=dimensions,
        node_positions=node_positions,
        fragile_edges=fragile_edges,
        silence_zones=silence_zones,
        confidence_trajectory=tuple(state.confidence_history),
        dimension_confidence=dim_confidence,
        dialogue_depth=dialogue_depth,
        stage_discovery=stage_discovery,
    )
