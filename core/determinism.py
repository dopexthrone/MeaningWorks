"""
Motherlabs Determinism — structural fingerprinting and variance measurement.

Phase 13: Determinism & Reproducibility
Derived from: C008 (Determinism) — temperature=0, structural fingerprinting,
variance measurement, pipeline determinism verification.

This is a LEAF MODULE — zero project imports, only stdlib (hashlib, json).
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any


@dataclass(frozen=True)
class StructuralFingerprint:
    """
    Topology-only fingerprint of a blueprint.

    Ignores descriptions, derived_from, ordering — only topology matters.
    Two blueprints with the same components, types, and relationships
    produce the same fingerprint regardless of field order or descriptions.

    Derived from: Phase 13.1 — structural fingerprinting
    """
    component_set: Tuple[str, ...]                      # Sorted names
    component_types: Tuple[Tuple[str, str], ...]        # Sorted (name, type)
    relationship_set: Tuple[Tuple[str, str, str], ...]  # Sorted (from, to, type)
    constraint_count: int
    unresolved_count: int
    hash_digest: str                                     # SHA256[:16]


@dataclass(frozen=True)
class StructuralDistance:
    """
    Distance between two structural fingerprints.

    Derived from: Phase 13.1 — variance measurement
    """
    jaccard_components: float    # 0-1 Jaccard similarity on component sets
    jaccard_relationships: float # 0-1 Jaccard similarity on relationship sets
    type_mismatches: Tuple[Tuple[str, str, str], ...]  # (name, type_in_a, type_in_b)
    added_components: Tuple[str, ...]     # In fp2 but not fp1
    removed_components: Tuple[str, ...]   # In fp1 but not fp2
    overall_distance: float               # 0=identical, 1=completely different


@dataclass(frozen=True)
class VarianceReport:
    """
    Variance analysis across multiple compilation runs.

    Derived from: Phase 13.1 — build_variance_report
    """
    run_count: int
    unique_structures: int
    dominant_hash: str
    dominant_frequency: int
    variance_score: float       # 0=perfect determinism, 1=total chaos
    fingerprints: Tuple[StructuralFingerprint, ...]


def compute_structural_fingerprint(blueprint: Dict[str, Any]) -> StructuralFingerprint:
    """
    Compute topology-only fingerprint of a blueprint.

    Normalizes ordering, ignores descriptions and derived_from fields.
    Only the structural skeleton matters: which components exist,
    their types, and how they connect.

    Args:
        blueprint: The blueprint dict to fingerprint

    Returns:
        Frozen StructuralFingerprint with SHA256[:16] hash

    Derived from: Phase 13.1 — replaces tests/consistency_test.py:structural_hash()
    """
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])
    constraints = blueprint.get("constraints", [])
    unresolved = blueprint.get("unresolved", [])

    # Sorted component names
    component_names = tuple(sorted(
        c.get("name", "") for c in components
    ))

    # Sorted (name, type) pairs
    component_types = tuple(sorted(
        (c.get("name", ""), c.get("type", "entity"))
        for c in components
    ))

    # Sorted (from, to, type) triples
    relationship_set = tuple(sorted(
        (r.get("from", ""), r.get("to", ""), r.get("type", ""))
        for r in relationships
    ))

    # Counts only (not content) for constraints and unresolved
    constraint_count = len(constraints)
    unresolved_count = len(unresolved)

    # Build canonical structure for hashing
    canonical = {
        "components": list(component_names),
        "types": [(n, t) for n, t in component_types],
        "relationships": [(f, t, r) for f, t, r in relationship_set],
        "constraint_count": constraint_count,
        "unresolved_count": unresolved_count,
    }

    hash_input = json.dumps(canonical, sort_keys=True).encode("utf-8")
    hash_digest = hashlib.sha256(hash_input).hexdigest()[:16]

    return StructuralFingerprint(
        component_set=component_names,
        component_types=component_types,
        relationship_set=relationship_set,
        constraint_count=constraint_count,
        unresolved_count=unresolved_count,
        hash_digest=hash_digest,
    )


def compute_structural_distance(
    fp1: StructuralFingerprint,
    fp2: StructuralFingerprint,
) -> StructuralDistance:
    """
    Compute distance between two structural fingerprints.

    Uses Jaccard similarity on component and relationship sets.
    Identifies type mismatches and added/removed components.

    Args:
        fp1: First fingerprint
        fp2: Second fingerprint

    Returns:
        StructuralDistance with overall_distance in [0, 1]

    Derived from: Phase 13.1 — structural distance metrics
    """
    # Jaccard on component sets
    set1_c = set(fp1.component_set)
    set2_c = set(fp2.component_set)
    union_c = set1_c | set2_c
    intersection_c = set1_c & set2_c
    jaccard_c = len(intersection_c) / len(union_c) if union_c else 1.0

    # Jaccard on relationship sets
    set1_r = set(fp1.relationship_set)
    set2_r = set(fp2.relationship_set)
    union_r = set1_r | set2_r
    intersection_r = set1_r & set2_r
    jaccard_r = len(intersection_r) / len(union_r) if union_r else 1.0

    # Type mismatches for common components
    types1 = dict(fp1.component_types)
    types2 = dict(fp2.component_types)
    common_names = set(types1.keys()) & set(types2.keys())
    type_mismatches = tuple(sorted(
        (name, types1[name], types2[name])
        for name in common_names
        if types1[name] != types2[name]
    ))

    # Added/removed
    added = tuple(sorted(set2_c - set1_c))
    removed = tuple(sorted(set1_c - set2_c))

    # Overall distance: weighted combination
    # Component structure is more important than relationships
    type_mismatch_penalty = len(type_mismatches) / len(common_names) if common_names else 0.0
    overall = 1.0 - (0.4 * jaccard_c + 0.4 * jaccard_r + 0.2 * (1.0 - type_mismatch_penalty))

    return StructuralDistance(
        jaccard_components=jaccard_c,
        jaccard_relationships=jaccard_r,
        type_mismatches=type_mismatches,
        added_components=added,
        removed_components=removed,
        overall_distance=max(0.0, min(1.0, overall)),
    )


def build_variance_report(
    fingerprints: List[StructuralFingerprint],
) -> VarianceReport:
    """
    Analyze variance across multiple compilation fingerprints.

    Args:
        fingerprints: List of fingerprints from repeated compilations

    Returns:
        VarianceReport with variance_score:
            0.0 = perfect determinism (all identical)
            1.0 = total chaos (all different)

    Derived from: Phase 13.1 — variance analysis
    """
    if not fingerprints:
        return VarianceReport(
            run_count=0,
            unique_structures=0,
            dominant_hash="",
            dominant_frequency=0,
            variance_score=0.0,
            fingerprints=(),
        )

    # Count hash frequencies
    hash_counts: Dict[str, int] = {}
    for fp in fingerprints:
        hash_counts[fp.hash_digest] = hash_counts.get(fp.hash_digest, 0) + 1

    unique_count = len(hash_counts)
    dominant_hash = max(hash_counts, key=hash_counts.get)
    dominant_freq = hash_counts[dominant_hash]

    # Variance score: 0 if all same, approaches 1 as more unique structures
    # Formula: 1 - (dominant_frequency / total_runs)
    variance_score = 1.0 - (dominant_freq / len(fingerprints))

    return VarianceReport(
        run_count=len(fingerprints),
        unique_structures=unique_count,
        dominant_hash=dominant_hash,
        dominant_frequency=dominant_freq,
        variance_score=variance_score,
        fingerprints=tuple(fingerprints),
    )
