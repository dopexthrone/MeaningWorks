"""
Motherlabs Trust Computation — the product is the trust indicators.

Phase C: Trust Computation Module

LEAF MODULE — stdlib only. No engine/protocol/pipeline imports.
All frozen dataclasses. All pure functions.

Like HTTPS: you don't think about TLS, you see the lock icon.
This module computes the lock icon equivalent for compiled output.

Trust computation sources:
- Verification scores (7 dimensions)
- Provenance depth (stratum count from context_graph)
- Confidence trajectory (per-turn confidence history)
- Dimensional metadata (for gap detection)
- Silence zones (semantic regions not covered)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class TrustIndicators:
    """Complete trust assessment for a compilation result.

    This is THE PRODUCT — the trust indicators that make AI output trustworthy.
    Every field maps to a visual component in the Semantic IDE.
    """
    overall_score: float                    # 0-100, weighted average of fidelity dimensions
    provenance_depth: int                   # 1-3, how many strata of provenance exist
    fidelity_scores: Dict[str, int]         # 7 verification dimensions → scores (0-100)
    gap_report: Tuple[str, ...]             # Explicit gaps: input not covered, output not sourced
    dimensional_coverage: Dict[str, float]  # Dimension name → coverage ratio
    verification_badge: str                 # "verified" | "partial" | "unverified"
    confidence_trajectory: Tuple[float, ...]  # Per-turn confidence for line chart
    silence_zones: Tuple[str, ...]          # Unexplored semantic regions
    derivation_chain_length: float          # Average provenance chain depth
    component_count: int                    # Number of blueprint components
    relationship_count: int                 # Number of blueprint relationships
    constraint_count: int                   # Number of blueprint constraints
    method_coverage: float                  # 0.0-1.0, fraction of components with methods


# =============================================================================
# BADGE LOGIC
# =============================================================================

# Core dimensions that drive badge computation
_CORE_DIMENSIONS = ("completeness", "consistency", "coherence", "traceability")

# Badge thresholds
_VERIFIED_THRESHOLD = 70
_PARTIAL_THRESHOLD = 40


def compute_badge(fidelity_scores: Dict[str, int]) -> str:
    """Compute verification badge from fidelity scores.

    Logic:
    - "verified" — All 4 core dimensions >= 70
    - "partial" — No core dimension < 40, but some < 70
    - "unverified" — Any core dimension < 40

    Args:
        fidelity_scores: Dimension name → score (0-100)

    Returns:
        "verified" | "partial" | "unverified"
    """
    core_scores = []
    for dim in _CORE_DIMENSIONS:
        score = fidelity_scores.get(dim, 0)
        core_scores.append(score)

    if not core_scores:
        return "unverified"

    if any(s < _PARTIAL_THRESHOLD for s in core_scores):
        return "unverified"

    if all(s >= _VERIFIED_THRESHOLD for s in core_scores):
        return "verified"

    return "partial"


# =============================================================================
# PROVENANCE DEPTH
# =============================================================================

def compute_provenance_depth(context_graph: Dict[str, Any]) -> int:
    """Compute provenance depth from context graph.

    Checks which strata are populated:
    - Stratum 1: Input provenance (always exists if context_graph populated)
    - Stratum 2: Dialogue provenance (insights, conflicts)
    - Stratum 3: Compilation provenance (stable patterns)

    Args:
        context_graph: The context_graph from compilation result

    Returns:
        Depth 1-3 (or 0 if empty)
    """
    if not context_graph:
        return 0

    depth = 0

    # Stratum 1: Input tracking
    if context_graph.get("input_hash") or context_graph.get("keywords"):
        depth = 1

    # Stratum 2: Dialogue provenance
    insights = context_graph.get("insights", [])
    conflicts = context_graph.get("conflicts", [])
    if insights or conflicts:
        depth = max(depth, 2)

    # Stratum 3: Compilation patterns
    patterns = context_graph.get("self_compile_patterns", [])
    stable = context_graph.get("stable_components", [])
    if patterns or stable:
        depth = max(depth, 3)

    return depth


# =============================================================================
# GAP DETECTION
# =============================================================================

def detect_gaps(
    blueprint: Dict[str, Any],
    intent_keywords: List[str],
    verification: Dict[str, Any],
) -> Tuple[str, ...]:
    """Detect gaps between input intent and compiled output.

    Sources:
    1. Verification gaps (from dimension scores)
    2. Uncovered intent keywords
    3. Components without derived_from

    Args:
        blueprint: Compiled blueprint dict
        intent_keywords: Keywords from intent extraction
        verification: Verification result dict

    Returns:
        Tuple of gap description strings
    """
    gaps = []

    # 1. Verification gaps
    for dim_name in ("completeness", "consistency", "coherence", "traceability",
                      "actionability", "specificity"):
        dim = verification.get(dim_name, {})
        dim_gaps = dim.get("gaps", [])
        for g in dim_gaps[:3]:  # Cap per dimension
            gaps.append(f"[{dim_name}] {g}")

    # 2. Uncovered keywords
    components = blueprint.get("components", [])
    bp_text = " ".join(
        f"{c.get('name', '')} {c.get('description', '')} {c.get('derived_from', '')}"
        for c in components
    ).lower()
    for kw in intent_keywords:
        if kw.lower() not in bp_text and len(kw) >= 3:
            gaps.append(f"Input keyword not covered: '{kw}'")
            if len(gaps) > 20:
                break

    # 3. Components without provenance
    for comp in components:
        derived = comp.get("derived_from", "")
        if not derived or len(derived.strip()) < 5:
            gaps.append(f"No provenance: component '{comp.get('name', '?')}'")

    return tuple(gaps[:25])  # Cap total gaps


# =============================================================================
# SILENCE ZONES
# =============================================================================

def detect_silence_zones(
    dimensional_metadata: Dict[str, Any],
    confidence_trajectory: List[float],
) -> Tuple[str, ...]:
    """Detect semantic regions that were under-explored.

    Sources:
    1. Low-confidence dimensional positions
    2. Dimensions with no node positions
    3. Confidence trajectory drops

    Args:
        dimensional_metadata: Serialized DimensionalMetadata
        confidence_trajectory: Per-turn confidence values

    Returns:
        Tuple of silence zone descriptions
    """
    zones = []

    # 1. Low-confidence dimensions
    dimensions = dimensional_metadata.get("dimensions", [])
    for dim in dimensions:
        dim_name = dim.get("name", "unknown") if isinstance(dim, dict) else str(dim)
        # Check if any node has low confidence in this dimension
        zones_for_dim = []
        node_positions = dimensional_metadata.get("node_positions", [])
        for pos in node_positions:
            if isinstance(pos, dict):
                conf = pos.get("confidence", 0.5)
                if conf < 0.4:
                    zones_for_dim.append(pos.get("component_name", "?"))

    # 2. Trajectory drops (confidence decreased significantly)
    if len(confidence_trajectory) >= 3:
        for i in range(1, len(confidence_trajectory)):
            if confidence_trajectory[i] < confidence_trajectory[i - 1] - 0.15:
                zones.append(f"Confidence dropped at turn {i + 1} ({confidence_trajectory[i - 1]:.2f} → {confidence_trajectory[i]:.2f})")

    # 3. Fragile edges from dimensional metadata
    fragile = dimensional_metadata.get("fragile_edges", [])
    for edge in fragile[:5]:
        if isinstance(edge, dict):
            risk = edge.get("drift_risk", "unknown")
            nodes = edge.get("affected_nodes", [])
            if risk in ("high", "medium"):
                zones.append(f"Fragile connection: {', '.join(nodes[:3])} (risk: {risk})")

    return tuple(zones[:15])


# =============================================================================
# DERIVATION CHAIN
# =============================================================================

def compute_derivation_chain_length(components: List[Dict[str, Any]]) -> float:
    """Compute average derivation chain depth.

    A chain of length N means the output traces through N provenance hops.
    Longer chains = deeper provenance = more trustworthy.

    Simple heuristic: count non-trivial derived_from fields.
    "user input" = depth 1, "dialogue turn 3: ..." = depth 2,
    "synthesis of insights 1,2,3" = depth 3.

    Args:
        components: Blueprint component dicts

    Returns:
        Average chain depth (0.0 if no components)
    """
    if not components:
        return 0.0

    total_depth = 0.0
    for comp in components:
        derived = comp.get("derived_from", "")
        if not derived:
            total_depth += 0.0
        elif len(derived) < 15:
            total_depth += 1.0  # Shallow reference
        elif any(kw in derived.lower() for kw in ("synthesis", "multiple", "combined", "merged")):
            total_depth += 3.0  # Deep synthesis
        elif any(kw in derived.lower() for kw in ("dialogue", "turn", "insight", "exchange")):
            total_depth += 2.0  # Dialogue provenance
        else:
            total_depth += 1.5  # Standard reference

    return total_depth / len(components)


# =============================================================================
# CONFIDENCE TRAJECTORY
# =============================================================================

def extract_confidence_trajectory(
    context_graph: Dict[str, Any],
) -> Tuple[float, ...]:
    """Extract per-turn confidence values from context graph.

    Args:
        context_graph: Context graph with confidence data

    Returns:
        Tuple of per-turn confidence floats
    """
    # Check for explicit trajectory
    trajectory = context_graph.get("confidence_trajectory", [])
    if trajectory:
        return tuple(float(v) for v in trajectory)

    # Fallback: derive from insights count (rough proxy)
    insights = context_graph.get("insights", [])
    if not insights:
        return ()

    # Build approximate trajectory from insight accumulation
    total = len(insights)
    return tuple(min((i + 1) / total, 1.0) for i in range(total))


# =============================================================================
# DIMENSIONAL COVERAGE
# =============================================================================

def compute_dimensional_coverage(
    dimensional_metadata: Dict[str, Any],
) -> Dict[str, float]:
    """Compute coverage ratio per dimension.

    For each dimension, what fraction of components have a positioned value?

    Args:
        dimensional_metadata: Serialized DimensionalMetadata

    Returns:
        Dict of dimension name → coverage ratio (0.0-1.0)
    """
    dimensions = dimensional_metadata.get("dimensions", [])
    node_positions = dimensional_metadata.get("node_positions", [])

    if not dimensions or not node_positions:
        return {}

    total_nodes = len(node_positions)
    if total_nodes == 0:
        return {}

    coverage = {}
    for dim in dimensions:
        dim_name = dim.get("name", str(dim)) if isinstance(dim, dict) else str(dim)
        # Count nodes that have values for this dimension
        positioned = 0
        for pos in node_positions:
            if isinstance(pos, dict):
                dim_values = pos.get("dimension_values", [])
                for dv in dim_values:
                    if isinstance(dv, (list, tuple)) and len(dv) >= 2:
                        if dv[0] == dim_name:
                            positioned += 1
                            break
        coverage[dim_name] = positioned / total_nodes

    return coverage


def _compute_method_coverage(components: List[Dict[str, Any]]) -> float:
    """Compute fraction of components that have non-empty methods arrays."""
    if not components:
        return 1.0  # No components = vacuously covered
    with_methods = sum(1 for c in components if c.get("methods"))
    return with_methods / len(components)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def compute_trust_indicators(
    blueprint: Dict[str, Any],
    verification: Dict[str, Any],
    context_graph: Dict[str, Any],
    dimensional_metadata: Dict[str, Any],
    intent_keywords: List[str],
) -> TrustIndicators:
    """Compute complete trust indicators for a compilation result.

    This is the main entry point. Takes all compilation outputs and produces
    the TrustIndicators that are THE PRODUCT.

    Args:
        blueprint: Compiled blueprint dict
        verification: Verification result dict (from verify_deterministic or LLM)
        context_graph: Context graph with provenance data
        dimensional_metadata: Serialized DimensionalMetadata
        intent_keywords: Keywords from intent extraction

    Returns:
        Frozen TrustIndicators
    """
    # Extract fidelity scores
    fidelity_scores = {}
    for dim in ("completeness", "consistency", "coherence", "traceability",
                "actionability", "specificity", "codegen_readiness"):
        dim_data = verification.get(dim, {})
        if isinstance(dim_data, dict):
            fidelity_scores[dim] = dim_data.get("score", 0)
        elif isinstance(dim_data, (int, float)):
            fidelity_scores[dim] = int(dim_data)

    # Compute overall score (weighted average of fidelity dimensions)
    weights = {
        "completeness": 0.20, "consistency": 0.20, "coherence": 0.15,
        "traceability": 0.15, "actionability": 0.10, "specificity": 0.10,
        "codegen_readiness": 0.10,
    }
    weighted_sum = sum(
        fidelity_scores.get(dim, 0) * w
        for dim, w in weights.items()
    )
    overall_score = min(max(weighted_sum, 0.0), 100.0)

    # Compute all trust components
    badge = compute_badge(fidelity_scores)
    provenance_depth = compute_provenance_depth(context_graph)
    gaps = detect_gaps(blueprint, intent_keywords, verification)
    confidence_trajectory = extract_confidence_trajectory(context_graph)
    silence_zones = detect_silence_zones(dimensional_metadata, list(confidence_trajectory))
    chain_length = compute_derivation_chain_length(blueprint.get("components", []))
    dim_coverage = compute_dimensional_coverage(dimensional_metadata)

    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])
    constraints = blueprint.get("constraints", [])

    # Method coverage: fraction of components that have non-empty methods
    method_cov = _compute_method_coverage(components)

    # Apply method coverage penalty to overall score.
    # If <50% of components have methods, reduce trust proportionally.
    # Full penalty at 0% coverage = -15 points. No penalty at >=70%.
    if method_cov < 0.70:
        penalty = (0.70 - method_cov) / 0.70 * 15.0
        overall_score = max(0.0, overall_score - penalty)

    return TrustIndicators(
        overall_score=round(overall_score, 1),
        provenance_depth=provenance_depth,
        fidelity_scores=fidelity_scores,
        gap_report=gaps,
        dimensional_coverage=dim_coverage,
        verification_badge=badge,
        confidence_trajectory=confidence_trajectory,
        silence_zones=silence_zones,
        derivation_chain_length=round(chain_length, 2),
        component_count=len(components),
        relationship_count=len(relationships),
        constraint_count=len(constraints),
        method_coverage=round(method_cov, 3),
    )


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_trust_indicators(trust: TrustIndicators) -> Dict[str, Any]:
    """Serialize TrustIndicators to JSON-safe dict for API responses."""
    return {
        "overall_score": trust.overall_score,
        "provenance_depth": trust.provenance_depth,
        "fidelity_scores": dict(trust.fidelity_scores),
        "gap_report": list(trust.gap_report),
        "dimensional_coverage": dict(trust.dimensional_coverage),
        "verification_badge": trust.verification_badge,
        "confidence_trajectory": list(trust.confidence_trajectory),
        "silence_zones": list(trust.silence_zones),
        "derivation_chain_length": trust.derivation_chain_length,
        "component_count": trust.component_count,
        "relationship_count": trust.relationship_count,
        "constraint_count": trust.constraint_count,
        "method_coverage": trust.method_coverage,
    }
