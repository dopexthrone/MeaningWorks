"""
Motherlabs Deterministic Verification — three-layer verification stack (Layer 1).

Phase 18: Verification Overhaul

LEAF MODULE — stdlib only. No engine/protocol/pipeline/schema imports.
Receives pre-computed data as arguments. All frozen dataclasses. All pure functions.

Scoring dimensions:
- completeness: intent keyword coverage in blueprint
- consistency: contradiction + graph error penalty
- coherence: orphan ratio, relationship density, health score, dangling refs
- traceability: derived_from coverage + specificity
- actionability: parseable constraints, methods, typed attributes (NEW)
- specificity: description length, type confidence, derived_from quality (NEW)
- codegen_readiness: weighted composite of above dimensions (NEW)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single verification dimension."""
    name: str
    score: int              # 0-100
    confidence: float       # 0.0-1.0
    gaps: Tuple[str, ...]
    details: str


@dataclass(frozen=True)
class DeterministicVerification:
    """Result of deterministic verification (Layer 1)."""
    completeness: DimensionScore
    consistency: DimensionScore
    coherence: DimensionScore
    traceability: DimensionScore
    actionability: DimensionScore
    specificity: DimensionScore
    codegen_readiness: DimensionScore
    overall_score: int          # 0-100 weighted average
    status: str                 # "pass" | "needs_work" | "ambiguous"
    needs_llm: bool
    ambiguous_dimensions: Tuple[str, ...]


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    """Clamp integer to [lo, hi]."""
    return max(lo, min(hi, value))


def score_completeness(
    blueprint: Dict[str, Any],
    intent_keywords: List[str],
    input_text: str,
) -> DimensionScore:
    """
    Score intent keyword coverage in blueprint components.

    Checks component names, descriptions, and derived_from fields for
    presence of intent keywords (domain, actors, goals, constraints text).

    Args:
        blueprint: Compiled blueprint dict
        intent_keywords: Keywords extracted from intent (domain terms, actor names, goal phrases)
        input_text: Original user input text

    Returns:
        DimensionScore for completeness
    """
    if not intent_keywords:
        return DimensionScore(
            name="completeness",
            score=50,
            confidence=0.3,
            gaps=(),
            details="No intent keywords to check against",
        )

    components = blueprint.get("components", [])

    # Build searchable text from blueprint
    bp_text_parts = []
    for comp in components:
        bp_text_parts.append(comp.get("name", "").lower())
        bp_text_parts.append(comp.get("description", "").lower())
        bp_text_parts.append(comp.get("derived_from", "").lower())
    bp_text = " ".join(bp_text_parts)

    # Also include relationship descriptions
    for rel in blueprint.get("relationships", []):
        bp_text += " " + rel.get("description", "").lower()

    matched = []
    unmatched = []
    for kw in intent_keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower or len(kw_lower) < 3:
            continue
        if kw_lower in bp_text:
            matched.append(kw)
        else:
            unmatched.append(kw)

    total = len(matched) + len(unmatched)
    if total == 0:
        return DimensionScore(
            name="completeness",
            score=50,
            confidence=0.3,
            gaps=(),
            details="No usable intent keywords",
        )

    coverage_ratio = len(matched) / total
    score = _clamp(int(coverage_ratio * 100))

    gaps = tuple(f"Missing: '{kw}'" for kw in unmatched[:10])

    return DimensionScore(
        name="completeness",
        score=score,
        confidence=min(0.5 + coverage_ratio * 0.5, 1.0),
        gaps=gaps,
        details=f"{len(matched)}/{total} intent keywords covered ({coverage_ratio:.0%})",
    )


def _find_cross_subsystem_duplicates(
    components: List[Dict[str, Any]],
) -> List[str]:
    """
    Find components that appear both at top-level AND inside sub_blueprints.

    Returns list of duplicate component names.
    """
    # Collect all names inside sub_blueprints
    sub_names: Dict[str, str] = {}  # normalized -> original name
    for comp in components:
        if comp.get("type") == "subsystem" and comp.get("sub_blueprint"):
            for sub_comp in comp["sub_blueprint"].get("components", []):
                raw = sub_comp.get("name", "")
                norm = _normalize_for_dedup(raw)
                if norm:
                    sub_names[norm] = raw

    if not sub_names:
        return []

    # Check top-level non-subsystem components against sub_blueprint names
    duplicates = []
    for comp in components:
        if comp.get("type") == "subsystem":
            continue
        raw = comp.get("name", "")
        norm = _normalize_for_dedup(raw)
        if norm and norm in sub_names:
            duplicates.append(raw)

    return duplicates


def _normalize_for_dedup(name: str) -> str:
    """Normalize a component name for dedup comparison.

    Strips whitespace, lowercases, removes common trailing type words
    (service, manager, handler, controller, module, component, engine)
    when separated by space/underscore/hyphen, and collapses non-alpha to space.
    """
    import re
    norm = name.lower().strip()
    # Collapse non-alphanumeric to single space first
    norm = re.sub(r'[^a-z0-9]+', ' ', norm).strip()
    # Remove trailing type words (only when they are separate words)
    norm = re.sub(r'\s+(service|manager|handler|controller|module|component|engine)$', '', norm)
    return norm


def score_consistency(
    contradiction_count: int,
    graph_errors: List[str],
    graph_warnings: List[str],
    components: List[Dict[str, Any]] = (),
) -> DimensionScore:
    """
    Score blueprint consistency from contradictions, graph errors, and cross-subsystem duplicates.

    Start at 100, deduct for issues:
    - -20 per contradiction
    - -15 per graph error (cycles)
    - -5 per graph warning
    - -15 per cross-subsystem duplicate (component in both top-level and sub_blueprint)

    Args:
        contradiction_count: Number of detected contradictions
        graph_errors: Error strings from validate_graph()
        graph_warnings: Warning strings from validate_graph()
        components: Blueprint component list (for cross-subsystem duplicate detection)

    Returns:
        DimensionScore for consistency
    """
    score = 100
    gaps = []

    score -= contradiction_count * 20
    if contradiction_count > 0:
        gaps.append(f"{contradiction_count} constraint contradiction(s)")

    cycle_count = sum(1 for e in graph_errors if "cycle" in e.lower())
    other_errors = len(graph_errors) - cycle_count
    score -= cycle_count * 15
    score -= other_errors * 10
    if graph_errors:
        gaps.extend(graph_errors[:5])

    score -= len(graph_warnings) * 5
    if graph_warnings:
        gaps.extend(graph_warnings[:3])

    # Cross-subsystem duplicate detection
    cross_dupes = _find_cross_subsystem_duplicates(list(components))
    score -= len(cross_dupes) * 15
    if cross_dupes:
        gaps.append(f"{len(cross_dupes)} cross-subsystem duplicate(s): {', '.join(cross_dupes[:5])}")

    score = _clamp(score)
    total_issues = contradiction_count + len(graph_errors) + len(graph_warnings) + len(cross_dupes)
    confidence = 0.9 if total_issues == 0 else max(0.5, 0.9 - total_issues * 0.1)

    return DimensionScore(
        name="consistency",
        score=score,
        confidence=confidence,
        gaps=tuple(gaps),
        details=f"Contradictions: {contradiction_count}, graph errors: {len(graph_errors)}, warnings: {len(graph_warnings)}, cross-subsystem dupes: {len(cross_dupes)}",
    )


def score_coherence(
    orphan_ratio: float,
    relationship_density: float,
    health_score: float,
    dangling_count: int,
    component_count: int = 0,
) -> DimensionScore:
    """
    Score blueprint structural coherence.

    Formula: 100 - orphan_penalty + density_bonus + health_bonus - dangling_penalty
    - orphan_penalty: orphan_ratio * orphan_weight (scaled by system size)
    - density_bonus: relationship_density * 45 (capped at 1.5× density)
    - health_bonus: health_score * 30
    - dangling_penalty: min(dangling_count * 5, 20)

    Args:
        orphan_ratio: Fraction of components with no relationships (0.0-1.0)
        relationship_density: relationships / components ratio
        health_score: Blueprint health score (0.0-1.0)
        dangling_count: Number of dangling relationship references
        component_count: Number of components (scales orphan penalty)

    Returns:
        DimensionScore for coherence
    """
    # Scale orphan penalty by system size — small systems penalized less
    if component_count <= 5:
        orphan_weight = 25
    elif component_count <= 10:
        orphan_weight = 32
    else:
        orphan_weight = 40
    orphan_penalty = orphan_ratio * orphan_weight
    density_bonus = min(relationship_density, 1.5) * 30
    health_bonus = health_score * 30
    dangling_penalty = min(dangling_count * 5, 20)

    raw = 100 - orphan_penalty + density_bonus + health_bonus - dangling_penalty
    # Normalize: the base is 100 and bonuses push it higher, so we need to scale
    # Max possible: 100 - 0 + 45 + 30 - 0 = 175 → scale to [0,100]
    # Min possible: 100 - 40 + 0 + 0 - 20 = 40 with worst case
    score = _clamp(int(raw * 100 / 175))

    gaps = []
    if orphan_ratio > 0.3:
        gaps.append(f"High orphan ratio: {orphan_ratio:.0%}")
    if dangling_count > 0:
        gaps.append(f"{dangling_count} dangling reference(s)")
    if health_score < 0.5:
        gaps.append(f"Low health score: {health_score:.2f}")

    return DimensionScore(
        name="coherence",
        score=score,
        confidence=0.8,
        gaps=tuple(gaps),
        details=f"orphan_ratio={orphan_ratio:.2f}, density={relationship_density:.2f}, health={health_score:.2f}, dangling={dangling_count}",
    )


def validate_provenance_refs(
    components: List[Dict[str, Any]],
    grid_postcodes: List[str],
) -> Dict[str, Any]:
    """
    Validate grid: provenance references in component derived_from fields.

    Parses "grid:POSTCODE" references, validates they exist in the grid,
    and classifies each component's provenance quality.

    Args:
        components: Blueprint component dicts
        grid_postcodes: Valid postcode keys from the semantic grid

    Returns:
        Dict with:
        - validated: count of components with valid grid refs
        - text_only: count with derived_from but no grid refs
        - invalid_refs: count with grid refs pointing to non-existent postcodes
        - missing: count with no derived_from at all
        - details: per-component breakdown list
    """
    import re
    grid_ref_re = re.compile(r'grid:([A-Z_]+\.[A-Z_]+\.[A-Z_]+\.[A-Z_]+\.[A-Z_]+)')
    valid_set = set(grid_postcodes)

    validated = 0
    text_only = 0
    invalid_refs = 0
    missing = 0
    details = []

    for comp in components:
        derived = comp.get("derived_from", "")
        name = comp.get("name", "unknown")
        if not derived or not derived.strip():
            missing += 1
            details.append({"name": name, "quality": "missing"})
            continue

        refs = grid_ref_re.findall(derived)
        if not refs:
            text_only += 1
            details.append({"name": name, "quality": "text_only"})
            continue

        all_valid = all(r in valid_set for r in refs)
        if all_valid:
            validated += 1
            details.append({"name": name, "quality": "validated", "refs": refs})
        else:
            invalid_refs += 1
            bad = [r for r in refs if r not in valid_set]
            details.append({"name": name, "quality": "invalid", "bad_refs": bad})

    return {
        "validated": validated,
        "text_only": text_only,
        "invalid_refs": invalid_refs,
        "missing": missing,
        "details": details,
    }


def provenance_integrity_ratio(
    components: List[Dict[str, Any]],
    grid_postcodes: List[str],
) -> float:
    """
    Compute ratio of components with validated grid provenance.

    Returns 0.0 if no grid postcodes available (no grid compiled).
    Returns 1.0 if all components have validated grid refs.
    """
    if not components or not grid_postcodes:
        return 0.0
    result = validate_provenance_refs(components, grid_postcodes)
    total = len(components)
    if total == 0:
        return 0.0
    return result["validated"] / total


def score_traceability(
    components: List[Dict[str, Any]],
    grid_postcodes: List[str] = (),
) -> DimensionScore:
    """
    Score derived_from traceability of components.

    Formula: base = ratio_with_derived * 60 + ratio_with_specific_derived * 40
    Grid bonus: base + grid_validated_ratio * 15 (additive, never penalizes)
    Grid presence can only increase the score, never decrease it.
    "Specific" means derived_from has > 10 characters (not just "user input").

    Args:
        components: Blueprint component dicts
        grid_postcodes: Valid postcode keys from the semantic grid (empty if no grid)

    Returns:
        DimensionScore for traceability
    """
    if not components:
        return DimensionScore(
            name="traceability",
            score=0,
            confidence=0.9,
            gaps=("No components to check",),
            details="Empty blueprint",
        )

    _GENERIC_PHRASES = frozenset({
        "user input", "user requirement", "inferred from input",
        "from dialogue", "dialogue context", "from intent",
        "user request", "inferred", "derived from input",
        # Self-citation patterns (synthesis/enrichment prompts generate these)
        "domain invariant", "enrichment", "resynthesis",
    })

    has_derived = 0
    has_specific = 0
    missing = []

    for comp in components:
        derived = comp.get("derived_from", "")
        name = comp.get("name", "unknown")
        if derived and derived.strip():
            has_derived += 1
            d_stripped = derived.strip()
            # Specificity: >20 chars AND not a known generic phrase or prefix
            is_generic = d_stripped.lower() in _GENERIC_PHRASES
            if not is_generic:
                is_generic = any(
                    d_stripped.lower().startswith(p + ":") or d_stripped.lower().startswith(p + " ")
                    for p in _GENERIC_PHRASES
                )
            if len(d_stripped) > 20 and not is_generic:
                has_specific += 1
        else:
            missing.append(name)

    total = len(components)
    ratio_derived = has_derived / total
    ratio_specific = has_specific / total

    # Base score (backward compatible — same as original formula)
    base_score = ratio_derived * 60 + ratio_specific * 40

    # Grid provenance bonus (purely additive — grid can only help, never hurt)
    grid_ratio = 0.0
    grid_detail = ""
    if grid_postcodes:
        prov = validate_provenance_refs(components, list(grid_postcodes))
        grid_ratio = prov["validated"] / total if total > 0 else 0.0
        grid_detail = f", grid_validated={prov['validated']}/{total}"

    score = _clamp(int(base_score + grid_ratio * 15))

    gaps = tuple(f"No derived_from: '{n}'" for n in missing[:8])

    return DimensionScore(
        name="traceability",
        score=score,
        confidence=0.85,
        gaps=gaps,
        details=f"{has_derived}/{total} have derived_from, {has_specific}/{total} specific{grid_detail}",
    )


def _infer_methods_from_type(comp_type: str) -> List[str]:
    """Infer minimal method set from component type."""
    _TYPE_METHODS = {
        "service": ["handle", "process"],
        "agent": ["run", "handle"],
        "entity": ["create", "read", "update", "delete"],
        "database": ["query", "store"],
        "interface": ["render", "handle_input"],
        "subsystem": [],
        "process": ["execute", "validate"],
        "controller": ["route", "handle"],
        "manager": ["manage", "coordinate"],
        "worker": ["run", "process"],
        "queue": ["enqueue", "dequeue"],
        "cache": ["get", "set", "invalidate"],
        "gateway": ["route", "transform"],
        "adapter": ["adapt", "transform"],
    }
    return _TYPE_METHODS.get(comp_type.lower(), [])


def _infer_methods_from_relationships(
    comp_name: str,
    relationships: List[Dict[str, Any]],
) -> List[str]:
    """Infer methods from how this component participates in relationships."""
    _REL_METHODS = {
        "manages": "manage",
        "triggers": "trigger",
        "consumes": "consume",
        "produces": "produce",
        "authenticates": "authenticate",
        "validates": "validate",
        "transforms": "transform",
        "stores": "store",
        "queries": "query",
        "notifies": "notify",
        "monitors": "monitor",
    }
    methods = []
    name_lower = comp_name.lower()
    for rel in relationships:
        source = rel.get("from", "").lower()
        if source == name_lower or source.replace(" ", "") == name_lower.replace(" ", ""):
            rel_type = rel.get("type", "").lower()
            if rel_type in _REL_METHODS:
                methods.append(_REL_METHODS[rel_type])
    return methods


def score_actionability(
    parseable_constraint_ratio: float,
    components: List[Dict[str, Any]],
    actionability_checks: Tuple[str, ...] = ("methods",),
    relationships: List[Dict[str, Any]] = (),
) -> DimensionScore:
    """
    Score how actionable/implementable the blueprint is.

    Infers methods from component types and relationships when a component
    has no explicit methods — preventing short user input from killing
    actionability scores.

    Formula:
    - parseable_constraint_ratio * 25 (constraints that parse to formal types)
    - methods_ratio * 35 (components that have actionability indicators)
    - typed_ratio * 20 (components with non-default type)
    - substance_ratio * 20 (components with description > 20 chars)

    Args:
        parseable_constraint_ratio: Fraction of constraints that parse to non-CUSTOM types
        components: Blueprint component dicts
        actionability_checks: Tuple of component dict keys to check for actionability
                             (default: ("methods",) for software domain)
        relationships: Blueprint relationship dicts (for method inference)

    Returns:
        DimensionScore for actionability
    """
    if not components:
        return DimensionScore(
            name="actionability",
            score=0,
            confidence=0.8,
            gaps=("No components",),
            details="Empty blueprint",
        )

    total = len(components)
    inferred_count = 0

    has_methods = 0
    for c in components:
        has_explicit = any(c.get(check) for check in actionability_checks)
        if has_explicit:
            has_methods += 1
        else:
            # Infer from type and relationships
            comp_type = c.get("type", "entity")
            comp_name = c.get("name", "")
            type_methods = _infer_methods_from_type(comp_type)
            rel_methods = _infer_methods_from_relationships(comp_name, list(relationships))
            if type_methods or rel_methods:
                has_methods += 1
                inferred_count += 1

    has_typed = sum(1 for c in components if c.get("type", "entity") != "entity")
    has_substance = sum(1 for c in components if len(c.get("description", "")) > 20)

    methods_ratio = has_methods / total
    typed_ratio = has_typed / total
    substance_ratio = has_substance / total

    score = _clamp(int(
        parseable_constraint_ratio * 25
        + methods_ratio * 35
        + typed_ratio * 20
        + substance_ratio * 20
    ))

    gaps = []
    if methods_ratio < 0.3:
        gaps.append(f"Only {has_methods}/{total} components have methods")
    if substance_ratio < 0.5:
        gaps.append(f"Only {has_substance}/{total} components have substantial descriptions")

    return DimensionScore(
        name="actionability",
        score=score,
        confidence=0.75,
        gaps=tuple(gaps),
        details=f"parseable={parseable_constraint_ratio:.0%}, methods={methods_ratio:.0%} ({inferred_count} inferred), typed={typed_ratio:.0%}, substance={substance_ratio:.0%}",
    )


def score_specificity(
    components: List[Dict[str, Any]],
    avg_type_confidence: float,
) -> DimensionScore:
    """
    Score how specific/detailed the blueprint is.

    Formula:
    - description_length_score * 40: avg desc length mapped to 0-100
    - type_confidence * 30
    - derived_from_quality * 30: avg derived_from length mapped to 0-100

    Args:
        components: Blueprint component dicts
        avg_type_confidence: Average type confidence from classification (0.0-1.0)

    Returns:
        DimensionScore for specificity
    """
    if not components:
        return DimensionScore(
            name="specificity",
            score=0,
            confidence=0.7,
            gaps=("No components",),
            details="Empty blueprint",
        )

    # Average description length → score (0 chars=0, 50+=80, 100+=100)
    desc_lengths = [len(c.get("description", "")) for c in components]
    avg_desc = sum(desc_lengths) / len(desc_lengths)
    desc_score = min(avg_desc / 100.0, 1.0)

    # Average derived_from length → quality (short = vague)
    df_lengths = [len(c.get("derived_from", "")) for c in components]
    avg_df = sum(df_lengths) / len(df_lengths)
    df_score = min(avg_df / 50.0, 1.0)

    score = _clamp(int(
        desc_score * 40
        + avg_type_confidence * 30
        + df_score * 30
    ))

    gaps = []
    weak = [c.get("name", "?") for c in components if len(c.get("description", "")) < 15]
    if weak:
        gaps.append(f"Weak descriptions: {', '.join(weak[:5])}")

    return DimensionScore(
        name="specificity",
        score=score,
        confidence=0.7,
        gaps=tuple(gaps),
        details=f"avg_desc={avg_desc:.0f} chars, avg_derived={avg_df:.0f} chars, type_conf={avg_type_confidence:.2f}",
    )


def score_codegen_readiness(
    completeness_score: int,
    consistency_score: int,
    traceability_score: int,
    actionability_score: int,
    specificity_score: int,
) -> DimensionScore:
    """
    Score codegen readiness as weighted average of other dimensions.

    Weights: comp×0.25 + cons×0.20 + trace×0.15 + action×0.25 + spec×0.15

    Args:
        completeness_score: 0-100
        consistency_score: 0-100
        traceability_score: 0-100
        actionability_score: 0-100
        specificity_score: 0-100

    Returns:
        DimensionScore for codegen_readiness
    """
    weighted = (
        completeness_score * 0.25
        + consistency_score * 0.20
        + traceability_score * 0.15
        + actionability_score * 0.25
        + specificity_score * 0.15
    )
    score = _clamp(int(weighted))

    blockers = []
    if consistency_score < 40:
        blockers.append(f"Consistency too low ({consistency_score})")
    if actionability_score < 30:
        blockers.append(f"Actionability too low ({actionability_score})")
    if completeness_score < 30:
        blockers.append(f"Completeness too low ({completeness_score})")

    return DimensionScore(
        name="codegen_readiness",
        score=score,
        confidence=0.8,
        gaps=tuple(blockers),
        details=f"Weighted: comp={completeness_score}×0.25 + cons={consistency_score}×0.20 + trace={traceability_score}×0.15 + act={actionability_score}×0.25 + spec={specificity_score}×0.15",
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def verify_deterministic(
    blueprint: Dict[str, Any],
    intent_keywords: List[str],
    input_text: str,
    graph_errors: List[str],
    graph_warnings: List[str],
    health_score: float,
    health_stats: Dict[str, Any],
    contradiction_count: int,
    parseable_constraint_ratio: float,
    avg_type_confidence: float,
    skip_threshold: int = 70,
    fail_threshold: int = 40,
    actionability_checks: Tuple[str, ...] = ("methods",),
    grid_postcodes: List[str] = (),
) -> DeterministicVerification:
    """
    Run all deterministic scoring and produce verification result.

    Decision logic:
    - All scores >= skip_threshold → status="pass", needs_llm=False
    - Any score < fail_threshold → status="needs_work", needs_llm=False
    - Otherwise → status="ambiguous", needs_llm=True (list ambiguous dimensions)

    Args:
        blueprint: Compiled blueprint dict
        intent_keywords: Keywords from intent extraction
        input_text: Original user input
        graph_errors: From validate_graph()
        graph_warnings: From validate_graph()
        health_score: Blueprint health score (0.0-1.0)
        health_stats: Health report stats dict
        contradiction_count: Number of constraint contradictions
        parseable_constraint_ratio: Fraction of parseable constraints
        avg_type_confidence: Average classification type confidence
        skip_threshold: Score above which LLM is skipped (default 70)
        fail_threshold: Score below which blueprint fails (default 40)

    Returns:
        DeterministicVerification with routing decision
    """
    components = blueprint.get("components", [])
    relationships = blueprint.get("relationships", [])

    orphan_ratio = health_stats.get("orphan_ratio", 0.0)
    dangling_count = health_stats.get("dangling_ref_count", 0)
    component_count = len(components)
    rel_count = len(relationships)
    relationship_density = rel_count / component_count if component_count > 0 else 0.0

    # Score all dimensions
    comp = score_completeness(blueprint, intent_keywords, input_text)
    cons = score_consistency(contradiction_count, graph_errors, graph_warnings, components)
    cohe = score_coherence(orphan_ratio, relationship_density, health_score, dangling_count, component_count)
    trac = score_traceability(components, list(grid_postcodes))
    acti = score_actionability(parseable_constraint_ratio, components, actionability_checks, relationships)
    spec = score_specificity(components, avg_type_confidence)
    codegen = score_codegen_readiness(comp.score, cons.score, trac.score, acti.score, spec.score)

    # Determine routing
    all_scores = [comp, cons, cohe, trac, acti, spec, codegen]
    core_scores = [comp, cons, cohe, trac]  # Only core dimensions drive routing

    ambiguous = []
    has_fail = False

    for ds in core_scores:
        if ds.score < fail_threshold:
            has_fail = True
        elif ds.score < skip_threshold:
            ambiguous.append(ds.name)

    if has_fail:
        status = "needs_work"
        needs_llm = False
    elif not ambiguous:
        status = "pass"
        needs_llm = False
    else:
        status = "ambiguous"
        needs_llm = True

    # Overall score: weighted average of all 7 dimensions
    overall = int(
        comp.score * 0.20
        + cons.score * 0.20
        + cohe.score * 0.15
        + trac.score * 0.15
        + acti.score * 0.10
        + spec.score * 0.10
        + codegen.score * 0.10
    )

    return DeterministicVerification(
        completeness=comp,
        consistency=cons,
        coherence=cohe,
        traceability=trac,
        actionability=acti,
        specificity=spec,
        codegen_readiness=codegen,
        overall_score=_clamp(overall),
        status=status,
        needs_llm=needs_llm,
        ambiguous_dimensions=tuple(ambiguous),
    )


# =============================================================================
# FORMAT CONVERTER
# =============================================================================


def to_verification_dict(det: DeterministicVerification) -> Dict[str, Any]:
    """
    Convert DeterministicVerification to the dict format downstream code expects.

    Produces same shape as the legacy LLM _verify() output plus new additive fields.

    Args:
        det: Deterministic verification result

    Returns:
        Dict with status, completeness, consistency, coherence, traceability,
        actionability, specificity, codegen_readiness, verification_mode
    """
    def _dim_dict(ds: DimensionScore) -> Dict[str, Any]:
        result: Dict[str, Any] = {"score": ds.score, "details": ds.details}
        if ds.gaps:
            result["gaps"] = list(ds.gaps)
        return result

    d: Dict[str, Any] = {
        "status": "pass" if det.status == "pass" else "needs_work",
        "overall_score": det.overall_score,
        "completeness": _dim_dict(det.completeness),
        "consistency": _dim_dict(det.consistency),
        "coherence": _dim_dict(det.coherence),
        "traceability": _dim_dict(det.traceability),
        "actionability": _dim_dict(det.actionability),
        "specificity": _dim_dict(det.specificity),
        "codegen_readiness": _dim_dict(det.codegen_readiness),
        "verification_mode": "deterministic" if not det.needs_llm else "hybrid",
        "semantic_gates": [],
    }

    # Populate legacy fields that _targeted_resynthesis reads
    if det.completeness.gaps:
        d["completeness"]["gaps"] = list(det.completeness.gaps)
    if det.consistency.gaps:
        d["consistency"]["conflicts"] = list(det.consistency.gaps)
    if det.coherence.gaps:
        d["coherence"]["issues"] = list(det.coherence.gaps)
        d["coherence"]["suggested_fixes"] = list(det.coherence.gaps)
    if det.traceability.gaps:
        d["traceability"]["orphans"] = list(det.traceability.gaps)
        d["traceability"]["weak_links"] = list(det.traceability.gaps)

    return d
