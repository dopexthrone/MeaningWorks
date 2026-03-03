"""
Motherlabs Compilation Trees — tree decomposition, L2 synthesis, integration verification.

Phase 25: Compilation Trees
Derived from: VISION.md L2 (compile compilations → patterns), DIMENSIONAL_BLUEPRINT.md

This is a LEAF MODULE — imports only core/determinism and stdlib.
No engine/protocol/pipeline imports.

Problem: compile() produces one blueprint. Large systems need N connected blueprints.
Solution: decompose_root() → children compile independently → synthesize_l2_patterns() → verify_integration()
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

from core.determinism import compute_structural_fingerprint


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class SubsystemSpec:
    """One subsystem identified during root decomposition.

    Derived from: Phase 25.1 — tree decomposition
    """
    name: str                                # Subsystem name from ARCHITECT
    description: str                         # Human-readable description
    canonical_components: Tuple[str, ...]    # Components in this subsystem
    parent_components: Tuple[str, ...]       # Root components related to subsystem
    derived_from: str                        # "architect_artifact" | "subsystem_hint" | "blueprint_structure"


@dataclass(frozen=True)
class TreeDecomposition:
    """Result of decomposing a root blueprint into subsystems.

    Derived from: Phase 25.1 — tree decomposition
    """
    subsystem_specs: Tuple[SubsystemSpec, ...]
    root_component_count: int
    decomposition_source: str                # "architect_artifact" | "subsystem_hint" | "blueprint_subsystem_type" | "none"
    decomposition_confidence: float          # 0-1
    unassigned_components: Tuple[str, ...]   # Root components not in any subsystem


@dataclass(frozen=True)
class CrossCuttingComponent:
    """A component that appears across multiple sibling blueprints.

    Derived from: Phase 25.1 — L2 synthesis
    """
    normalized_name: str
    variants: Tuple[str, ...]                # Name variants across siblings
    frequency: float                         # 0-1 across children
    child_sources: Tuple[str, ...]           # Which children contain it
    component_type: str                      # Most common type


@dataclass(frozen=True)
class InterfaceGap:
    """A gap in interface contracts between sibling blueprints.

    Derived from: Phase 25.1 — integration verification
    """
    component_a: str
    component_b: str
    child_a: str                             # Subsystem name
    child_b: str                             # Subsystem name
    gap_type: str                            # "missing_contract" | "type_mismatch" | "dangling_reference"
    description: str


@dataclass(frozen=True)
class L2Synthesis:
    """Cross-sibling pattern extraction (L2 = compile compilations → patterns).

    Derived from: Phase 25.1 + 25.3 — L2 synthesis
    """
    shared_vocabulary: Tuple[Tuple[str, int], ...]           # (term, freq)
    cross_cutting_components: Tuple[CrossCuttingComponent, ...]
    relationship_patterns: Tuple[Tuple[str, str, str, int], ...]  # (from_norm, to_norm, type, freq)
    interface_gaps: Tuple[InterfaceGap, ...]
    integration_constraints: Tuple[str, ...]
    pattern_count: int
    synthesis_confidence: float


@dataclass(frozen=True)
class IntegrationReport:
    """Cross-child consistency check.

    Derived from: Phase 25.1 — integration verification
    """
    total_interfaces_checked: int
    consistent_count: int
    gap_count: int
    gaps: Tuple[InterfaceGap, ...]
    overall_score: float                     # 0-1


@dataclass(frozen=True)
class ChildResult:
    """Result of compiling one subsystem child.

    Derived from: Phase 25.1 — tree result assembly
    """
    subsystem_name: str
    success: bool
    blueprint: Dict[str, Any]
    fingerprint_hash: str                    # SHA256[:16] from StructuralFingerprint
    component_count: int
    relationship_count: int
    verification_score: float


@dataclass(frozen=True)
class TreeResult:
    """Complete compilation tree output.

    Derived from: Phase 25.1 — tree result assembly
    """
    root_blueprint: Dict[str, Any]
    decomposition: TreeDecomposition
    child_results: Tuple[ChildResult, ...]
    l2_synthesis: L2Synthesis
    integration_report: IntegrationReport
    tree_health: float
    total_components: int
    timestamp: str


# =============================================================================
# PURE FUNCTIONS — Name Normalization
# =============================================================================

def normalize_component_name(name: str) -> str:
    """Normalize component name: lowercase, strip spaces/underscores/hyphens.

    Mirrors CorpusAnalyzer.normalize_name for cross-module consistency.

    Args:
        name: Raw component name

    Returns:
        Normalized lowercase string with separators removed

    Derived from: Phase 25.1 — mirrors corpus_analysis.py:normalize_name
    """
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


# =============================================================================
# PURE FUNCTIONS — Decomposition
# =============================================================================

def decompose_root(
    blueprint: Dict[str, Any],
    architect_artifact: Optional[Dict[str, Any]] = None,
    subsystem_hints: Optional[Dict[str, List[str]]] = None,
) -> TreeDecomposition:
    """Extract SubsystemSpecs from root blueprint context.

    Priority order:
    1. architect_artifact — parsed SUBSYSTEM: lines from ARCHITECT stage
    2. subsystem_hints — from canonical_components markers
    3. blueprint subsystem-type components — fallback from schema

    Args:
        blueprint: Root compilation blueprint dict
        architect_artifact: Output from parse_architect_artifact() (optional)
        subsystem_hints: Dict of subsystem_name → [component_names] (optional)

    Returns:
        TreeDecomposition with extracted subsystem specs

    Derived from: Phase 25.1 — tree decomposition
    """
    components = blueprint.get("components", [])
    all_component_names = [c.get("name", "") for c in components if c.get("name")]
    root_count = len(all_component_names)

    # Strategy 1: architect_artifact
    if architect_artifact and architect_artifact.get("subsystems"):
        specs = []
        assigned = set()
        for sub in architect_artifact["subsystems"]:
            name = sub.get("name", "")
            contains = tuple(sub.get("contains", []))
            desc = sub.get("description", "")
            if name and contains:
                assigned.update(contains)
                # Find parent components that relate to this subsystem
                parent_comps = _find_parent_components(
                    contains, blueprint.get("relationships", []), all_component_names
                )
                specs.append(SubsystemSpec(
                    name=name,
                    description=desc,
                    canonical_components=contains,
                    parent_components=parent_comps,
                    derived_from="architect_artifact",
                ))

        unassigned = tuple(sorted(
            n for n in all_component_names if n not in assigned
        ))

        confidence = min(1.0, len(specs) * 0.3 + 0.1) if specs else 0.0
        return TreeDecomposition(
            subsystem_specs=tuple(specs),
            root_component_count=root_count,
            decomposition_source="architect_artifact",
            decomposition_confidence=confidence,
            unassigned_components=unassigned,
        )

    # Strategy 2: subsystem_hints
    if subsystem_hints:
        specs = []
        assigned = set()
        for sub_name, sub_components in subsystem_hints.items():
            if sub_name and sub_components:
                contains = tuple(sub_components)
                assigned.update(sub_components)
                parent_comps = _find_parent_components(
                    contains, blueprint.get("relationships", []), all_component_names
                )
                specs.append(SubsystemSpec(
                    name=sub_name,
                    description=f"Subsystem: {sub_name}",
                    canonical_components=contains,
                    parent_components=parent_comps,
                    derived_from="subsystem_hint",
                ))

        unassigned = tuple(sorted(
            n for n in all_component_names if n not in assigned
        ))

        confidence = min(1.0, len(specs) * 0.25 + 0.1) if specs else 0.0
        return TreeDecomposition(
            subsystem_specs=tuple(specs),
            root_component_count=root_count,
            decomposition_source="subsystem_hint",
            decomposition_confidence=confidence,
            unassigned_components=unassigned,
        )

    # Strategy 3: blueprint subsystem-type components
    specs = []
    assigned = set()
    for comp in components:
        if comp.get("type") == "subsystem" and comp.get("name"):
            name = comp["name"]
            # Find components contained by this subsystem via relationships
            contained = _find_contained_components(
                name, blueprint.get("relationships", [])
            )
            if contained:
                assigned.update(contained)
                specs.append(SubsystemSpec(
                    name=name,
                    description=comp.get("description", f"Subsystem: {name}"),
                    canonical_components=tuple(contained),
                    parent_components=(name,),
                    derived_from="blueprint_structure",
                ))

    if specs:
        unassigned = tuple(sorted(
            n for n in all_component_names
            if n not in assigned and n not in [s.name for s in specs]
        ))
        confidence = min(1.0, len(specs) * 0.2 + 0.1)
        return TreeDecomposition(
            subsystem_specs=tuple(specs),
            root_component_count=root_count,
            decomposition_source="blueprint_subsystem_type",
            decomposition_confidence=confidence,
            unassigned_components=unassigned,
        )

    # No subsystems found
    return TreeDecomposition(
        subsystem_specs=(),
        root_component_count=root_count,
        decomposition_source="none",
        decomposition_confidence=0.0,
        unassigned_components=tuple(sorted(all_component_names)),
    )


def _find_parent_components(
    subsystem_components: Tuple[str, ...],
    relationships: List[Dict[str, Any]],
    all_component_names: List[str],
) -> Tuple[str, ...]:
    """Find root components that connect to subsystem components but aren't in the subsystem."""
    sub_set = set(subsystem_components)
    parents = set()
    for rel in relationships:
        from_c = rel.get("from", "")
        to_c = rel.get("to", "")
        if from_c in sub_set and to_c not in sub_set and to_c in all_component_names:
            parents.add(to_c)
        if to_c in sub_set and from_c not in sub_set and from_c in all_component_names:
            parents.add(from_c)
    return tuple(sorted(parents))


def _find_contained_components(
    subsystem_name: str,
    relationships: List[Dict[str, Any]],
) -> List[str]:
    """Find components contained by a subsystem via 'contains' relationships."""
    contained = []
    for rel in relationships:
        if rel.get("from") == subsystem_name and rel.get("type") == "contains":
            to_c = rel.get("to", "")
            if to_c:
                contained.append(to_c)
    return contained


# =============================================================================
# PURE FUNCTIONS — Subsystem Description Building
# =============================================================================

def build_subsystem_description(
    root_description: str,
    spec: SubsystemSpec,
    root_blueprint: Dict[str, Any],
) -> str:
    """Construct child compilation input from root context + subsystem spec.

    Args:
        root_description: Original user description
        spec: SubsystemSpec for this child
        root_blueprint: Root compilation blueprint for context

    Returns:
        Description string for child compilation

    Derived from: Phase 25.1 — subsystem description building
    """
    parts = [
        f"Subsystem: {spec.name}",
        "",
        f"Part of larger system: {root_description[:500]}",
        "",
    ]

    if spec.description:
        parts.append(f"Subsystem purpose: {spec.description}")
        parts.append("")

    if spec.canonical_components:
        parts.append(f"Core components: {', '.join(spec.canonical_components)}")

    # Add relevant relationships from root blueprint
    relevant_rels = _extract_relevant_relationships(
        spec.canonical_components, root_blueprint.get("relationships", [])
    )
    if relevant_rels:
        parts.append("")
        parts.append("Key relationships:")
        for rel in relevant_rels[:10]:  # Cap at 10
            parts.append(f"  - {rel['from']} {rel['type']} {rel['to']}")

    # Add relevant constraints from root blueprint
    relevant_constraints = _extract_relevant_constraints(
        spec.canonical_components, root_blueprint.get("constraints", [])
    )
    if relevant_constraints:
        parts.append("")
        parts.append("Constraints:")
        for c in relevant_constraints[:5]:  # Cap at 5
            parts.append(f"  - {c['description']}")

    return "\n".join(parts)


def _extract_relevant_relationships(
    component_names: Tuple[str, ...],
    relationships: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract relationships involving the given components."""
    name_set = set(component_names)
    return [
        r for r in relationships
        if r.get("from") in name_set or r.get("to") in name_set
    ]


def _extract_relevant_constraints(
    component_names: Tuple[str, ...],
    constraints: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract constraints that apply to the given components."""
    name_set = set(component_names)
    result = []
    for c in constraints:
        applies_to = c.get("applies_to", [])
        if any(a in name_set for a in applies_to):
            result.append(c)
    return result


# =============================================================================
# PURE FUNCTIONS — L2 Pattern Synthesis
# =============================================================================

def extract_shared_vocabulary(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> Tuple[Tuple[str, int], ...]:
    """Extract terms appearing in 2+ sibling blueprints.

    Analyzes component names, descriptions, and relationship descriptions
    to find shared domain vocabulary across children.

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        Sorted tuple of (term, frequency) for terms in 2+ siblings

    Derived from: Phase 25.1 — L2 vocabulary extraction
    """
    if len(child_blueprints) < 2:
        return ()

    # Collect normalized terms per child
    child_terms: List[set] = []
    for bp in child_blueprints:
        terms = set()
        for comp in bp.get("components", []):
            name = comp.get("name", "")
            if name:
                terms.add(normalize_component_name(name))
            # Extract words from description
            desc = comp.get("description", "")
            for word in _extract_content_words(desc):
                terms.add(word)
        for rel in bp.get("relationships", []):
            desc = rel.get("description", "")
            for word in _extract_content_words(desc):
                terms.add(word)
        child_terms.append(terms)

    # Count how many children contain each term
    term_freq: Dict[str, int] = {}
    for terms in child_terms:
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1

    # Filter: term must appear in 2+ children
    shared = [
        (term, freq)
        for term, freq in term_freq.items()
        if freq >= 2
    ]

    return tuple(sorted(shared, key=lambda x: (-x[1], x[0])))


def _extract_content_words(text: str) -> List[str]:
    """Extract meaningful words from text (>=4 chars, lowercase)."""
    import re
    words = re.findall(r'[a-zA-Z]{4,}', text)
    # Filter common stop words
    stop_words = {
        "that", "this", "with", "from", "have", "will", "been", "were",
        "they", "them", "then", "than", "each", "when", "what", "which",
        "their", "there", "about", "would", "could", "should", "into",
        "also", "some", "more", "very", "just", "like", "only", "over",
    }
    return [w.lower() for w in words if w.lower() not in stop_words]


def find_cross_cutting_components(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> Tuple[CrossCuttingComponent, ...]:
    """Find components that appear in 2+ sibling blueprints.

    Uses normalized names to detect variants (e.g. "UserAuth" and "User Auth").

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        Tuple of CrossCuttingComponent, sorted by frequency descending

    Derived from: Phase 25.1 — L2 cross-cutting detection
    """
    if len(child_blueprints) < 2:
        return ()

    # Map normalized_name → {child_name: [original_names, types]}
    norm_map: Dict[str, Dict[str, List[tuple]]] = {}
    for bp, child_name in zip(child_blueprints, child_names):
        for comp in bp.get("components", []):
            name = comp.get("name", "")
            comp_type = comp.get("type", "entity")
            if name:
                norm = normalize_component_name(name)
                if norm not in norm_map:
                    norm_map[norm] = {}
                if child_name not in norm_map[norm]:
                    norm_map[norm][child_name] = []
                norm_map[norm][child_name].append((name, comp_type))

    # Filter: must appear in 2+ children
    n_children = len(child_blueprints)
    results = []
    for norm, child_data in norm_map.items():
        if len(child_data) >= 2:
            all_variants = []
            all_types = []
            child_sources = []
            for child_name, entries in child_data.items():
                child_sources.append(child_name)
                for orig_name, comp_type in entries:
                    all_variants.append(orig_name)
                    all_types.append(comp_type)

            # Most common type
            type_freq: Dict[str, int] = {}
            for t in all_types:
                type_freq[t] = type_freq.get(t, 0) + 1
            most_common_type = max(type_freq, key=type_freq.get)

            results.append(CrossCuttingComponent(
                normalized_name=norm,
                variants=tuple(sorted(set(all_variants))),
                frequency=len(child_data) / n_children,
                child_sources=tuple(sorted(child_sources)),
                component_type=most_common_type,
            ))

    return tuple(sorted(results, key=lambda x: (-x.frequency, x.normalized_name)))


def detect_interface_gaps(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> Tuple[InterfaceGap, ...]:
    """Detect interface gaps between sibling blueprints.

    Gap types:
    - missing_contract: cross-child reference with no matching interface
    - type_mismatch: same normalized component has different types across children
    - dangling_reference: relationship references component not in any child

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        Tuple of InterfaceGap

    Derived from: Phase 25.1 — integration verification
    """
    if len(child_blueprints) < 2:
        return ()

    gaps = []

    # Build component registry: norm_name → {child_name: (orig_name, type)}
    comp_registry: Dict[str, Dict[str, tuple]] = {}
    for bp, child_name in zip(child_blueprints, child_names):
        for comp in bp.get("components", []):
            name = comp.get("name", "")
            comp_type = comp.get("type", "entity")
            if name:
                norm = normalize_component_name(name)
                if norm not in comp_registry:
                    comp_registry[norm] = {}
                comp_registry[norm][child_name] = (name, comp_type)

    # Type mismatches: same normalized name, different types
    for norm, children in comp_registry.items():
        if len(children) >= 2:
            types_seen = {}
            for child_name, (orig_name, comp_type) in children.items():
                if comp_type not in types_seen:
                    types_seen[comp_type] = []
                types_seen[comp_type].append((child_name, orig_name))

            if len(types_seen) >= 2:
                # There's a type mismatch
                type_items = list(types_seen.items())
                for i in range(len(type_items)):
                    for j in range(i + 1, len(type_items)):
                        type_a, entries_a = type_items[i]
                        type_b, entries_b = type_items[j]
                        child_a, comp_a = entries_a[0]
                        child_b, comp_b = entries_b[0]
                        gaps.append(InterfaceGap(
                            component_a=comp_a,
                            component_b=comp_b,
                            child_a=child_a,
                            child_b=child_b,
                            gap_type="type_mismatch",
                            description=f"'{norm}' is '{type_a}' in {child_a} but '{type_b}' in {child_b}",
                        ))

    # Dangling references: relationship targets not in same child's components
    for bp, child_name in zip(child_blueprints, child_names):
        local_norms = set()
        for comp in bp.get("components", []):
            name = comp.get("name", "")
            if name:
                local_norms.add(normalize_component_name(name))

        for rel in bp.get("relationships", []):
            for endpoint in ("from", "to"):
                ref_name = rel.get(endpoint, "")
                if ref_name and normalize_component_name(ref_name) not in local_norms:
                    # Reference to component not in this child
                    norm_ref = normalize_component_name(ref_name)
                    # Check if it exists in another child
                    found_in = None
                    for other_child, child_data in comp_registry.get(norm_ref, {}).items():
                        if other_child != child_name:
                            found_in = other_child
                            break

                    if found_in:
                        # Cross-child reference — needs an interface contract
                        gaps.append(InterfaceGap(
                            component_a=ref_name,
                            component_b=rel.get("from" if endpoint == "to" else "to", ""),
                            child_a=found_in,
                            child_b=child_name,
                            gap_type="missing_contract",
                            description=f"'{ref_name}' referenced in {child_name} but defined in {found_in}",
                        ))
                    else:
                        # Truly dangling — not in any child
                        gaps.append(InterfaceGap(
                            component_a=ref_name,
                            component_b=rel.get("from" if endpoint == "to" else "to", ""),
                            child_a=child_name,
                            child_b="",
                            gap_type="dangling_reference",
                            description=f"'{ref_name}' referenced in {child_name} but not found in any child",
                        ))

    return tuple(gaps)


def extract_relationship_patterns(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> Tuple[Tuple[str, str, str, int], ...]:
    """Extract relationship triples that appear in 2+ sibling blueprints.

    Uses normalized names for matching.

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        Tuple of (from_norm, to_norm, type, frequency) sorted by frequency desc

    Derived from: Phase 25.3 — L2 relationship pattern extraction
    """
    if len(child_blueprints) < 2:
        return ()

    # Count normalized relationship triples per child
    triple_children: Dict[Tuple[str, str, str], set] = {}
    for bp, child_name in zip(child_blueprints, child_names):
        for rel in bp.get("relationships", []):
            from_n = normalize_component_name(rel.get("from", ""))
            to_n = normalize_component_name(rel.get("to", ""))
            rel_type = rel.get("type", "")
            if from_n and to_n and rel_type:
                triple = (from_n, to_n, rel_type)
                if triple not in triple_children:
                    triple_children[triple] = set()
                triple_children[triple].add(child_name)

    # Filter: must appear in 2+ children
    results = [
        (from_n, to_n, rel_type, len(children))
        for (from_n, to_n, rel_type), children in triple_children.items()
        if len(children) >= 2
    ]

    return tuple(sorted(results, key=lambda x: (-x[3], x[0], x[1])))


def synthesize_l2_patterns(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> L2Synthesis:
    """Orchestrate all L2 pattern extraction across sibling blueprints.

    Combines shared vocabulary, cross-cutting components, relationship patterns,
    and interface gap detection into a single L2Synthesis.

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        L2Synthesis with all extracted patterns

    Derived from: Phase 25.1 + 25.3 — L2 synthesis orchestration
    """
    if not child_blueprints:
        return L2Synthesis(
            shared_vocabulary=(),
            cross_cutting_components=(),
            relationship_patterns=(),
            interface_gaps=(),
            integration_constraints=(),
            pattern_count=0,
            synthesis_confidence=0.0,
        )

    vocab = extract_shared_vocabulary(child_blueprints, child_names)
    cross_cutting = find_cross_cutting_components(child_blueprints, child_names)
    rel_patterns = extract_relationship_patterns(child_blueprints, child_names)
    gaps = detect_interface_gaps(child_blueprints, child_names)

    # Build integration constraints from cross-cutting components
    constraints = []
    for cc in cross_cutting:
        if cc.frequency >= 0.5:
            constraints.append(
                f"Shared component '{cc.normalized_name}' ({cc.component_type}) "
                f"must have consistent interface across: {', '.join(cc.child_sources)}"
            )

    pattern_count = len(vocab) + len(cross_cutting) + len(rel_patterns)

    # Confidence: based on signal strength and gap count
    if not child_blueprints or len(child_blueprints) < 2:
        confidence = 0.0
    else:
        signal = min(1.0, pattern_count / 10.0)
        gap_penalty = min(0.5, len(gaps) * 0.1)
        confidence = max(0.0, signal - gap_penalty)

    return L2Synthesis(
        shared_vocabulary=vocab,
        cross_cutting_components=cross_cutting,
        relationship_patterns=rel_patterns,
        interface_gaps=gaps,
        integration_constraints=tuple(constraints),
        pattern_count=pattern_count,
        synthesis_confidence=confidence,
    )


# =============================================================================
# PURE FUNCTIONS — Integration Verification
# =============================================================================

def verify_integration(
    child_blueprints: List[Dict[str, Any]],
    child_names: List[str],
) -> IntegrationReport:
    """Verify cross-child consistency.

    Checks:
    - Cross-cutting components have consistent types
    - No dangling references
    - Interface contracts exist for cross-child dependencies

    Args:
        child_blueprints: List of blueprint dicts from children
        child_names: Corresponding subsystem names

    Returns:
        IntegrationReport with overall consistency score

    Derived from: Phase 25.1 — integration verification
    """
    if not child_blueprints or len(child_blueprints) < 2:
        return IntegrationReport(
            total_interfaces_checked=0,
            consistent_count=0,
            gap_count=0,
            gaps=(),
            overall_score=1.0,
        )

    gaps = detect_interface_gaps(child_blueprints, child_names)
    cross_cutting = find_cross_cutting_components(child_blueprints, child_names)

    # Total interfaces = cross-cutting components (potential interface points)
    total_checked = len(cross_cutting)
    gap_count = len(gaps)

    # Consistent = interfaces without gaps
    # A cross-cutting component is consistent if none of the gaps reference it
    gap_norms = set()
    for g in gaps:
        gap_norms.add(normalize_component_name(g.component_a))
        gap_norms.add(normalize_component_name(g.component_b))

    consistent = sum(
        1 for cc in cross_cutting
        if cc.normalized_name not in gap_norms
    )

    if total_checked > 0:
        score = consistent / total_checked
    else:
        # No cross-cutting components = no interface issues possible
        score = 1.0 if gap_count == 0 else 0.0

    return IntegrationReport(
        total_interfaces_checked=total_checked,
        consistent_count=consistent,
        gap_count=gap_count,
        gaps=gaps,
        overall_score=max(0.0, min(1.0, score)),
    )


# =============================================================================
# PURE FUNCTIONS — Tree Health
# =============================================================================

def compute_tree_health(
    child_results: Tuple[ChildResult, ...],
    l2_synthesis: L2Synthesis,
    integration_report: IntegrationReport,
) -> float:
    """Compute overall tree compilation health score.

    Formula: 0.4*success_rate + 0.2*avg_verification + 0.25*integration + 0.15*synthesis_signal

    Args:
        child_results: Tuple of ChildResult from child compilations
        l2_synthesis: L2Synthesis from cross-sibling analysis
        integration_report: IntegrationReport from consistency check

    Returns:
        Health score in [0.0, 1.0]

    Derived from: Phase 25.1 — tree health computation
    """
    if not child_results:
        return 0.0

    # Success rate (0.4 weight)
    success_count = sum(1 for cr in child_results if cr.success)
    success_rate = success_count / len(child_results)

    # Average verification score (0.2 weight)
    avg_verification = sum(cr.verification_score for cr in child_results) / len(child_results)

    # Integration score (0.25 weight)
    integration_score = integration_report.overall_score

    # Synthesis signal (0.15 weight) — based on pattern count and confidence
    synthesis_signal = l2_synthesis.synthesis_confidence

    health = (
        0.4 * success_rate
        + 0.2 * avg_verification
        + 0.25 * integration_score
        + 0.15 * synthesis_signal
    )

    return max(0.0, min(1.0, health))


# =============================================================================
# PURE FUNCTIONS — L2 Formatting
# =============================================================================

def format_l2_patterns_section(l2_synthesis: L2Synthesis) -> Optional[str]:
    """Format L2 patterns as text for re-compilation prompt injection.

    Returns None if no meaningful patterns to inject.

    Args:
        l2_synthesis: L2Synthesis from synthesize_l2_patterns()

    Returns:
        Formatted text section or None

    Derived from: Phase 25.3 — L2 formatting
    """
    if l2_synthesis.pattern_count == 0:
        return None

    parts = ["## Cross-Subsystem Patterns (L2 Synthesis)", ""]

    if l2_synthesis.cross_cutting_components:
        parts.append("### Shared Components")
        for cc in l2_synthesis.cross_cutting_components:
            parts.append(
                f"- **{cc.normalized_name}** ({cc.component_type}): "
                f"appears in {', '.join(cc.child_sources)} "
                f"[{cc.frequency:.0%} of subsystems]"
            )
        parts.append("")

    if l2_synthesis.relationship_patterns:
        parts.append("### Recurring Relationships")
        for from_n, to_n, rel_type, freq in l2_synthesis.relationship_patterns:
            parts.append(f"- {from_n} → {to_n} ({rel_type}): {freq} subsystems")
        parts.append("")

    if l2_synthesis.shared_vocabulary:
        # Top 10 terms
        top_terms = l2_synthesis.shared_vocabulary[:10]
        parts.append("### Shared Vocabulary")
        terms_str = ", ".join(f"{t} ({f})" for t, f in top_terms)
        parts.append(f"- {terms_str}")
        parts.append("")

    if l2_synthesis.integration_constraints:
        parts.append("### Integration Constraints")
        for c in l2_synthesis.integration_constraints:
            parts.append(f"- {c}")
        parts.append("")

    if l2_synthesis.interface_gaps:
        parts.append(f"### Interface Gaps ({len(l2_synthesis.interface_gaps)} detected)")
        for gap in l2_synthesis.interface_gaps[:5]:  # Cap display
            parts.append(f"- [{gap.gap_type}] {gap.description}")
        parts.append("")

    result = "\n".join(parts).strip()
    return result if len(result) > 50 else None


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_tree_result(result: TreeResult) -> dict:
    """Serialize TreeResult to JSON-safe dict.

    Derived from: Phase 25.1 — tree result serialization
    """
    return {
        "root_blueprint": result.root_blueprint,
        "decomposition": {
            "subsystem_specs": [
                {
                    "name": s.name,
                    "description": s.description,
                    "canonical_components": list(s.canonical_components),
                    "parent_components": list(s.parent_components),
                    "derived_from": s.derived_from,
                }
                for s in result.decomposition.subsystem_specs
            ],
            "root_component_count": result.decomposition.root_component_count,
            "decomposition_source": result.decomposition.decomposition_source,
            "decomposition_confidence": result.decomposition.decomposition_confidence,
            "unassigned_components": list(result.decomposition.unassigned_components),
        },
        "child_results": [
            {
                "subsystem_name": cr.subsystem_name,
                "success": cr.success,
                "blueprint": cr.blueprint,
                "fingerprint_hash": cr.fingerprint_hash,
                "component_count": cr.component_count,
                "relationship_count": cr.relationship_count,
                "verification_score": cr.verification_score,
            }
            for cr in result.child_results
        ],
        "l2_synthesis": {
            "shared_vocabulary": [list(sv) for sv in result.l2_synthesis.shared_vocabulary],
            "cross_cutting_components": [
                {
                    "normalized_name": cc.normalized_name,
                    "variants": list(cc.variants),
                    "frequency": cc.frequency,
                    "child_sources": list(cc.child_sources),
                    "component_type": cc.component_type,
                }
                for cc in result.l2_synthesis.cross_cutting_components
            ],
            "relationship_patterns": [list(rp) for rp in result.l2_synthesis.relationship_patterns],
            "interface_gaps": [
                {
                    "component_a": g.component_a,
                    "component_b": g.component_b,
                    "child_a": g.child_a,
                    "child_b": g.child_b,
                    "gap_type": g.gap_type,
                    "description": g.description,
                }
                for g in result.l2_synthesis.interface_gaps
            ],
            "integration_constraints": list(result.l2_synthesis.integration_constraints),
            "pattern_count": result.l2_synthesis.pattern_count,
            "synthesis_confidence": result.l2_synthesis.synthesis_confidence,
        },
        "integration_report": {
            "total_interfaces_checked": result.integration_report.total_interfaces_checked,
            "consistent_count": result.integration_report.consistent_count,
            "gap_count": result.integration_report.gap_count,
            "gaps": [
                {
                    "component_a": g.component_a,
                    "component_b": g.component_b,
                    "child_a": g.child_a,
                    "child_b": g.child_b,
                    "gap_type": g.gap_type,
                    "description": g.description,
                }
                for g in result.integration_report.gaps
            ],
            "overall_score": result.integration_report.overall_score,
        },
        "tree_health": result.tree_health,
        "total_components": result.total_components,
        "timestamp": result.timestamp,
    }


def deserialize_tree_result(data: dict) -> TreeResult:
    """Deserialize TreeResult from JSON-safe dict.

    Derived from: Phase 25.1 — tree result deserialization
    """
    decomp_data = data["decomposition"]
    specs = tuple(
        SubsystemSpec(
            name=s["name"],
            description=s["description"],
            canonical_components=tuple(s["canonical_components"]),
            parent_components=tuple(s["parent_components"]),
            derived_from=s["derived_from"],
        )
        for s in decomp_data["subsystem_specs"]
    )
    decomposition = TreeDecomposition(
        subsystem_specs=specs,
        root_component_count=decomp_data["root_component_count"],
        decomposition_source=decomp_data["decomposition_source"],
        decomposition_confidence=decomp_data["decomposition_confidence"],
        unassigned_components=tuple(decomp_data["unassigned_components"]),
    )

    child_results = tuple(
        ChildResult(
            subsystem_name=cr["subsystem_name"],
            success=cr["success"],
            blueprint=cr["blueprint"],
            fingerprint_hash=cr["fingerprint_hash"],
            component_count=cr["component_count"],
            relationship_count=cr["relationship_count"],
            verification_score=cr["verification_score"],
        )
        for cr in data["child_results"]
    )

    l2_data = data["l2_synthesis"]
    cross_cutting = tuple(
        CrossCuttingComponent(
            normalized_name=cc["normalized_name"],
            variants=tuple(cc["variants"]),
            frequency=cc["frequency"],
            child_sources=tuple(cc["child_sources"]),
            component_type=cc["component_type"],
        )
        for cc in l2_data["cross_cutting_components"]
    )
    interface_gaps_l2 = tuple(
        InterfaceGap(
            component_a=g["component_a"],
            component_b=g["component_b"],
            child_a=g["child_a"],
            child_b=g["child_b"],
            gap_type=g["gap_type"],
            description=g["description"],
        )
        for g in l2_data["interface_gaps"]
    )
    l2_synthesis = L2Synthesis(
        shared_vocabulary=tuple(tuple(sv) for sv in l2_data["shared_vocabulary"]),
        cross_cutting_components=cross_cutting,
        relationship_patterns=tuple(tuple(rp) for rp in l2_data["relationship_patterns"]),
        interface_gaps=interface_gaps_l2,
        integration_constraints=tuple(l2_data["integration_constraints"]),
        pattern_count=l2_data["pattern_count"],
        synthesis_confidence=l2_data["synthesis_confidence"],
    )

    ir_data = data["integration_report"]
    integration_gaps = tuple(
        InterfaceGap(
            component_a=g["component_a"],
            component_b=g["component_b"],
            child_a=g["child_a"],
            child_b=g["child_b"],
            gap_type=g["gap_type"],
            description=g["description"],
        )
        for g in ir_data["gaps"]
    )
    integration_report = IntegrationReport(
        total_interfaces_checked=ir_data["total_interfaces_checked"],
        consistent_count=ir_data["consistent_count"],
        gap_count=ir_data["gap_count"],
        gaps=integration_gaps,
        overall_score=ir_data["overall_score"],
    )

    return TreeResult(
        root_blueprint=data["root_blueprint"],
        decomposition=decomposition,
        child_results=child_results,
        l2_synthesis=l2_synthesis,
        integration_report=integration_report,
        tree_health=data["tree_health"],
        total_components=data["total_components"],
        timestamp=data["timestamp"],
    )
