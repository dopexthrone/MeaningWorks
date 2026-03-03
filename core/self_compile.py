"""
Motherlabs Self-Compile Loop — convergence tracking and code diffing.

Phase 24: Self-Compile Loop
Derived from: VISION.md L2 (compile compilations → patterns), L3 (compile self → evolution)

This is a LEAF MODULE — imports only core/determinism, codegen/comparison, and stdlib.
No engine/protocol imports.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Dict, Any

from core.determinism import (
    StructuralFingerprint,
    StructuralDistance,
    VarianceReport,
    compute_structural_fingerprint,
    compute_structural_distance,
    build_variance_report,
)
from codegen.comparison import CodeComparisonTool


# =============================================================================
# FROZEN DATACLASSES
# =============================================================================

@dataclass(frozen=True)
class CodeDiffReport:
    """Blueprint-vs-code comparison for one source file.

    Derived from: Phase 24.1 — self-compile code diffing
    """
    file_path: str
    syntax_valid: bool
    classes_found: int
    classes_total: int
    overall_score: float                          # 0-1 from CodeComparisonTool
    class_scores: Tuple[Tuple[str, float], ...]   # (class_name, score)
    missing_classes: Tuple[str, ...]
    derived_from: str                              # "self-compile:v3.0"


@dataclass(frozen=True)
class ConvergencePoint:
    """One self-compile run's summary.

    Derived from: Phase 24.1 — convergence tracking
    """
    fingerprint: StructuralFingerprint
    component_count: int
    relationship_count: int
    constraint_count: int
    canonical_coverage: float    # fraction of canonical components found
    timestamp: str


@dataclass(frozen=True)
class ConvergenceReport:
    """Convergence analysis across multiple self-compile runs.

    Derived from: Phase 24.1 — convergence tracking
    """
    points: Tuple[ConvergencePoint, ...]
    variance: VarianceReport
    is_converged: bool            # variance_score == 0.0
    structural_drift: float       # distance from first to last run
    derived_from: str


@dataclass(frozen=True)
class SelfPattern:
    """A pattern extracted from self-observation.

    Derived from: Phase 24.1 — self-observation patterns (Stratum 3 source)

    pattern_type values:
        "stable_component"      — appears in >=90% of runs
        "stable_relationship"   — relationship triple appears in >=90% of runs
        "drift_point"           — appears in 30-70% of runs (unstable)
        "canonical_gap"         — canonical component NOT found in any run
    """
    pattern_type: str
    name: str
    frequency: float             # 0-1 across runs
    details: str
    derived_from: str            # "self-compile:v3.0"


@dataclass(frozen=True)
class SelfCompileReport:
    """Complete self-compile loop report.

    Derived from: Phase 24.1 — self-compile loop output
    """
    convergence: ConvergenceReport
    code_diffs: Tuple[CodeDiffReport, ...]
    patterns: Tuple[SelfPattern, ...]
    overall_health: float        # 0-1 composite score
    timestamp: str


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def diff_blueprint_vs_code(
    blueprint: Dict[str, Any],
    source_files: List[Tuple[str, str]],
) -> Tuple[CodeDiffReport, ...]:
    """
    Compare blueprint-generated code against actual source files.

    For each source file, generates code from the blueprint using
    BlueprintCodeGenerator, then compares with CodeComparisonTool.

    Args:
        blueprint: Blueprint dict with components/relationships
        source_files: List of (file_path, source_code) tuples

    Returns:
        Tuple of CodeDiffReport, one per source file

    Derived from: Phase 24.1 — code diffing
    """
    from codegen.generator import BlueprintCodeGenerator

    gen = BlueprintCodeGenerator(blueprint)
    generated_code = gen.generate()

    reports = []
    for file_path, actual_code in source_files:
        try:
            tool = CodeComparisonTool(actual_code, generated_code)
            summary = tool.get_summary()

            class_scores = tuple(
                (name, score)
                for name, score in summary.get("class_scores", {}).items()
            )
            # Missing classes: in actual but not found in generated
            all_classes_result = tool.compare_all_classes() if tool.syntax_valid else {}
            missing = tuple(
                name for name, result in all_classes_result.items()
                if not result.get("class_found", False)
            )

            reports.append(CodeDiffReport(
                file_path=file_path,
                syntax_valid=summary.get("syntax_valid", False),
                classes_found=summary.get("classes_found", 0),
                classes_total=summary.get("classes_total", 0),
                overall_score=summary.get("overall_score", 0.0),
                class_scores=class_scores,
                missing_classes=missing,
                derived_from="self-compile:v3.0",
            ))
        except SyntaxError:
            reports.append(CodeDiffReport(
                file_path=file_path,
                syntax_valid=False,
                classes_found=0,
                classes_total=0,
                overall_score=0.0,
                class_scores=(),
                missing_classes=(),
                derived_from="self-compile:v3.0",
            ))

    return tuple(reports)


def track_convergence(
    fingerprints: List[StructuralFingerprint],
    canonical_components: List[str],
) -> ConvergenceReport:
    """
    Analyze convergence across multiple self-compile runs.

    Args:
        fingerprints: List of StructuralFingerprint from repeated runs
        canonical_components: List of required component names

    Returns:
        ConvergenceReport with variance, drift, and convergence status

    Derived from: Phase 24.1 — convergence tracking
    """
    if not fingerprints:
        empty_variance = build_variance_report([])
        return ConvergenceReport(
            points=(),
            variance=empty_variance,
            is_converged=True,
            structural_drift=0.0,
            derived_from="self-compile:v3.0",
        )

    variance = build_variance_report(fingerprints)

    # Build convergence points
    points = []
    canonical_set = set(c.lower() for c in canonical_components)
    for fp in fingerprints:
        found = sum(
            1 for c in fp.component_set
            if c.lower() in canonical_set
        )
        coverage = found / len(canonical_set) if canonical_set else 1.0

        points.append(ConvergencePoint(
            fingerprint=fp,
            component_count=len(fp.component_set),
            relationship_count=len(fp.relationship_set),
            constraint_count=fp.constraint_count,
            canonical_coverage=coverage,
            timestamp=datetime.now().isoformat(),
        ))

    # Structural drift: distance from first to last
    if len(fingerprints) >= 2:
        distance = compute_structural_distance(fingerprints[0], fingerprints[-1])
        structural_drift = distance.overall_distance
    else:
        structural_drift = 0.0

    return ConvergenceReport(
        points=tuple(points),
        variance=variance,
        is_converged=(variance.variance_score == 0.0),
        structural_drift=structural_drift,
        derived_from="self-compile:v3.0",
    )


def extract_self_patterns(
    blueprints: List[Dict[str, Any]],
    canonical_components: List[str],
) -> Tuple[SelfPattern, ...]:
    """
    Extract patterns from repeated self-compile runs.

    Pattern types:
        stable_component:      appears in >=90% of runs
        stable_relationship:   relationship triple in >=90% of runs
        drift_point:           component/relationship in 30-70% of runs
        canonical_gap:         canonical component NOT found in any run

    Args:
        blueprints: List of blueprint dicts from repeated runs
        canonical_components: Required component names

    Returns:
        Tuple of SelfPattern

    Derived from: Phase 24.1 — self-observation patterns
    """
    if not blueprints:
        return ()

    n = len(blueprints)
    patterns = []

    # Count component frequencies
    component_freq: Dict[str, int] = {}
    for bp in blueprints:
        for comp in bp.get("components", []):
            name = comp.get("name", "")
            if name:
                component_freq[name] = component_freq.get(name, 0) + 1

    # Count relationship frequencies
    rel_freq: Dict[Tuple[str, str, str], int] = {}
    for bp in blueprints:
        for rel in bp.get("relationships", []):
            triple = (rel.get("from", ""), rel.get("to", ""), rel.get("type", ""))
            if all(triple):
                rel_freq[triple] = rel_freq.get(triple, 0) + 1

    # Stable components (>=90%)
    for name, count in component_freq.items():
        freq = count / n
        if freq >= 0.9:
            patterns.append(SelfPattern(
                pattern_type="stable_component",
                name=name,
                frequency=freq,
                details=f"Appears in {count}/{n} runs ({freq:.0%})",
                derived_from="self-compile:v3.0",
            ))
        elif 0.3 <= freq <= 0.7:
            patterns.append(SelfPattern(
                pattern_type="drift_point",
                name=name,
                frequency=freq,
                details=f"Unstable: appears in {count}/{n} runs ({freq:.0%})",
                derived_from="self-compile:v3.0",
            ))

    # Stable relationships (>=90%)
    for triple, count in rel_freq.items():
        freq = count / n
        rel_name = f"{triple[0]} -> {triple[1]} ({triple[2]})"
        if freq >= 0.9:
            patterns.append(SelfPattern(
                pattern_type="stable_relationship",
                name=rel_name,
                frequency=freq,
                details=f"Appears in {count}/{n} runs ({freq:.0%})",
                derived_from="self-compile:v3.0",
            ))
        elif 0.3 <= freq <= 0.7:
            patterns.append(SelfPattern(
                pattern_type="drift_point",
                name=rel_name,
                frequency=freq,
                details=f"Unstable relationship: {count}/{n} runs ({freq:.0%})",
                derived_from="self-compile:v3.0",
            ))

    # Canonical gaps — canonical components not found in ANY run
    canonical_set = set(c.lower() for c in canonical_components)
    found_components = set(c.lower() for c in component_freq.keys())
    for canonical in canonical_components:
        if canonical.lower() not in found_components:
            patterns.append(SelfPattern(
                pattern_type="canonical_gap",
                name=canonical,
                frequency=0.0,
                details=f"Canonical component '{canonical}' not found in any run",
                derived_from="self-compile:v3.0",
            ))

    return tuple(patterns)


def compute_overall_health(
    convergence: ConvergenceReport,
    code_diffs: Tuple[CodeDiffReport, ...],
) -> float:
    """
    Compute composite health score.

    Formula: 0.4 * (1 - variance_score) + 0.3 * avg_code_score + 0.3 * canonical_coverage

    Args:
        convergence: ConvergenceReport from track_convergence()
        code_diffs: Tuple of CodeDiffReport from diff_blueprint_vs_code()

    Returns:
        Health score in [0.0, 1.0]

    Derived from: Phase 24.1 — overall health
    """
    # Variance component (0.4 weight)
    variance_component = 1.0 - convergence.variance.variance_score

    # Code diff component (0.3 weight)
    if code_diffs:
        avg_code_score = sum(d.overall_score for d in code_diffs) / len(code_diffs)
    else:
        avg_code_score = 0.0

    # Canonical coverage component (0.3 weight)
    if convergence.points:
        canonical_coverage = convergence.points[-1].canonical_coverage
    else:
        canonical_coverage = 0.0

    health = 0.4 * variance_component + 0.3 * avg_code_score + 0.3 * canonical_coverage
    return max(0.0, min(1.0, health))


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_self_compile_report(report: SelfCompileReport) -> dict:
    """
    Serialize a SelfCompileReport to JSON-safe dict.

    Derived from: Phase 24.1 — report serialization
    """
    return {
        "convergence": {
            "points": [
                {
                    "fingerprint_hash": p.fingerprint.hash_digest,
                    "component_count": p.component_count,
                    "relationship_count": p.relationship_count,
                    "constraint_count": p.constraint_count,
                    "canonical_coverage": p.canonical_coverage,
                    "timestamp": p.timestamp,
                }
                for p in report.convergence.points
            ],
            "variance": {
                "run_count": report.convergence.variance.run_count,
                "unique_structures": report.convergence.variance.unique_structures,
                "dominant_hash": report.convergence.variance.dominant_hash,
                "dominant_frequency": report.convergence.variance.dominant_frequency,
                "variance_score": report.convergence.variance.variance_score,
            },
            "is_converged": report.convergence.is_converged,
            "structural_drift": report.convergence.structural_drift,
            "derived_from": report.convergence.derived_from,
        },
        "code_diffs": [
            {
                "file_path": d.file_path,
                "syntax_valid": d.syntax_valid,
                "classes_found": d.classes_found,
                "classes_total": d.classes_total,
                "overall_score": d.overall_score,
                "class_scores": list(d.class_scores),
                "missing_classes": list(d.missing_classes),
                "derived_from": d.derived_from,
            }
            for d in report.code_diffs
        ],
        "patterns": [
            {
                "pattern_type": p.pattern_type,
                "name": p.name,
                "frequency": p.frequency,
                "details": p.details,
                "derived_from": p.derived_from,
            }
            for p in report.patterns
        ],
        "overall_health": report.overall_health,
        "timestamp": report.timestamp,
    }


def deserialize_self_compile_report(data: dict) -> SelfCompileReport:
    """
    Deserialize a SelfCompileReport from JSON-safe dict.

    Note: StructuralFingerprint within ConvergencePoints are stored as
    hash_digest only — full fingerprints are not round-tripped since they
    require the original blueprint. Points are reconstructed with minimal
    fingerprint stubs.

    Derived from: Phase 24.1 — report deserialization
    """
    conv_data = data["convergence"]
    var_data = conv_data["variance"]

    variance = VarianceReport(
        run_count=var_data["run_count"],
        unique_structures=var_data["unique_structures"],
        dominant_hash=var_data["dominant_hash"],
        dominant_frequency=var_data["dominant_frequency"],
        variance_score=var_data["variance_score"],
        fingerprints=(),
    )

    points = []
    for p_data in conv_data["points"]:
        stub_fp = StructuralFingerprint(
            component_set=(),
            component_types=(),
            relationship_set=(),
            constraint_count=p_data["constraint_count"],
            unresolved_count=0,
            hash_digest=p_data["fingerprint_hash"],
        )
        points.append(ConvergencePoint(
            fingerprint=stub_fp,
            component_count=p_data["component_count"],
            relationship_count=p_data["relationship_count"],
            constraint_count=p_data["constraint_count"],
            canonical_coverage=p_data["canonical_coverage"],
            timestamp=p_data["timestamp"],
        ))

    convergence = ConvergenceReport(
        points=tuple(points),
        variance=variance,
        is_converged=conv_data["is_converged"],
        structural_drift=conv_data["structural_drift"],
        derived_from=conv_data["derived_from"],
    )

    code_diffs = tuple(
        CodeDiffReport(
            file_path=d["file_path"],
            syntax_valid=d["syntax_valid"],
            classes_found=d["classes_found"],
            classes_total=d["classes_total"],
            overall_score=d["overall_score"],
            class_scores=tuple(tuple(cs) for cs in d["class_scores"]),
            missing_classes=tuple(d["missing_classes"]),
            derived_from=d["derived_from"],
        )
        for d in data["code_diffs"]
    )

    patterns = tuple(
        SelfPattern(
            pattern_type=p["pattern_type"],
            name=p["name"],
            frequency=p["frequency"],
            details=p["details"],
            derived_from=p["derived_from"],
        )
        for p in data["patterns"]
    )

    return SelfCompileReport(
        convergence=convergence,
        code_diffs=code_diffs,
        patterns=patterns,
        overall_health=data["overall_health"],
        timestamp=data["timestamp"],
    )
