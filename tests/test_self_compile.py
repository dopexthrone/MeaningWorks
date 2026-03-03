"""
Tests for core/self_compile.py — Phase 24: Self-Compile Loop.

Tests frozen dataclasses, code diffing, convergence tracking,
self-observation pattern extraction, health scoring, and serialization.
"""

import pytest
from datetime import datetime

from core.self_compile import (
    CodeDiffReport,
    ConvergencePoint,
    ConvergenceReport,
    SelfPattern,
    SelfCompileReport,
    diff_blueprint_vs_code,
    track_convergence,
    extract_self_patterns,
    compute_overall_health,
    serialize_self_compile_report,
    deserialize_self_compile_report,
)
from core.determinism import (
    StructuralFingerprint,
    VarianceReport,
    compute_structural_fingerprint,
    build_variance_report,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_fingerprint(components=("A", "B"), relationships=(("A", "B", "uses"),)):
    """Build a StructuralFingerprint from minimal data."""
    blueprint = {
        "components": [{"name": c, "type": "entity"} for c in components],
        "relationships": [{"from": r[0], "to": r[1], "type": r[2]} for r in relationships],
        "constraints": [],
        "unresolved": [],
    }
    return compute_structural_fingerprint(blueprint)


def make_blueprint(components=None, relationships=None, constraints=None):
    """Build a minimal blueprint dict."""
    return {
        "components": [
            {"name": c, "type": "entity", "description": f"The {c} component", "derived_from": "test"}
            for c in (components or ["CompA", "CompB"])
        ],
        "relationships": [
            {"from": r[0], "to": r[1], "type": r[2], "description": "test"}
            for r in (relationships or [("CompA", "CompB", "uses")])
        ],
        "constraints": constraints or [],
        "unresolved": [],
    }


def make_convergence_report(variance_score=0.0, canonical_coverage=1.0, drift=0.0):
    """Build a ConvergenceReport with specified metrics."""
    fp = make_fingerprint()
    variance = VarianceReport(
        run_count=3,
        unique_structures=1,
        dominant_hash=fp.hash_digest,
        dominant_frequency=3,
        variance_score=variance_score,
        fingerprints=(fp,),
    )
    point = ConvergencePoint(
        fingerprint=fp,
        component_count=2,
        relationship_count=1,
        constraint_count=0,
        canonical_coverage=canonical_coverage,
        timestamp=datetime.now().isoformat(),
    )
    return ConvergenceReport(
        points=(point,),
        variance=variance,
        is_converged=(variance_score == 0.0),
        structural_drift=drift,
        derived_from="self-compile:v3.0",
    )


# =============================================================================
# FROZEN DATACLASS TESTS
# =============================================================================

class TestCodeDiffReport:
    def test_construction(self):
        r = CodeDiffReport(
            file_path="core/engine.py",
            syntax_valid=True,
            classes_found=3,
            classes_total=5,
            overall_score=0.75,
            class_scores=(("Foo", 0.8), ("Bar", 0.7)),
            missing_classes=("Baz",),
            derived_from="self-compile:v3.0",
        )
        assert r.file_path == "core/engine.py"
        assert r.syntax_valid is True
        assert r.overall_score == 0.75
        assert len(r.class_scores) == 2
        assert r.missing_classes == ("Baz",)

    def test_immutability(self):
        r = CodeDiffReport(
            file_path="test.py", syntax_valid=True, classes_found=1,
            classes_total=1, overall_score=1.0, class_scores=(),
            missing_classes=(), derived_from="test",
        )
        with pytest.raises(AttributeError):
            r.overall_score = 0.5


class TestConvergencePoint:
    def test_construction(self):
        fp = make_fingerprint()
        p = ConvergencePoint(
            fingerprint=fp, component_count=2, relationship_count=1,
            constraint_count=0, canonical_coverage=0.8,
            timestamp="2026-02-09T00:00:00",
        )
        assert p.component_count == 2
        assert p.canonical_coverage == 0.8
        assert p.fingerprint.hash_digest == fp.hash_digest

    def test_immutability(self):
        fp = make_fingerprint()
        p = ConvergencePoint(
            fingerprint=fp, component_count=2, relationship_count=1,
            constraint_count=0, canonical_coverage=0.8,
            timestamp="2026-02-09T00:00:00",
        )
        with pytest.raises(AttributeError):
            p.canonical_coverage = 1.0


class TestConvergenceReport:
    def test_converged_when_variance_zero(self):
        report = make_convergence_report(variance_score=0.0)
        assert report.is_converged is True

    def test_not_converged_when_variance_nonzero(self):
        report = make_convergence_report(variance_score=0.5)
        assert report.is_converged is False

    def test_immutability(self):
        report = make_convergence_report()
        with pytest.raises(AttributeError):
            report.is_converged = False


class TestSelfPattern:
    def test_stable_component(self):
        p = SelfPattern(
            pattern_type="stable_component", name="SharedState",
            frequency=1.0, details="Appears in 3/3 runs (100%)",
            derived_from="self-compile:v3.0",
        )
        assert p.pattern_type == "stable_component"
        assert p.frequency == 1.0

    def test_drift_point(self):
        p = SelfPattern(
            pattern_type="drift_point", name="AuditTrail",
            frequency=0.5, details="Unstable: appears in 1/2 runs (50%)",
            derived_from="self-compile:v3.0",
        )
        assert p.pattern_type == "drift_point"

    def test_canonical_gap(self):
        p = SelfPattern(
            pattern_type="canonical_gap", name="ConflictOracle",
            frequency=0.0, details="Canonical component 'ConflictOracle' not found",
            derived_from="self-compile:v3.0",
        )
        assert p.pattern_type == "canonical_gap"
        assert p.frequency == 0.0

    def test_immutability(self):
        p = SelfPattern(
            pattern_type="stable_component", name="X",
            frequency=1.0, details="test", derived_from="test",
        )
        with pytest.raises(AttributeError):
            p.frequency = 0.5


class TestSelfCompileReport:
    def test_construction(self):
        convergence = make_convergence_report()
        report = SelfCompileReport(
            convergence=convergence,
            code_diffs=(),
            patterns=(),
            overall_health=0.7,
            timestamp="2026-02-09T00:00:00",
        )
        assert report.overall_health == 0.7
        assert report.convergence.is_converged is True

    def test_immutability(self):
        convergence = make_convergence_report()
        report = SelfCompileReport(
            convergence=convergence, code_diffs=(), patterns=(),
            overall_health=0.7, timestamp="2026-02-09T00:00:00",
        )
        with pytest.raises(AttributeError):
            report.overall_health = 0.0


# =============================================================================
# CORE FUNCTION TESTS
# =============================================================================

class TestDiffBlueprintVsCode:
    def test_diff_with_matching_code(self):
        """Generated code compared against itself should score high."""
        from codegen.generator import BlueprintCodeGenerator

        blueprint = make_blueprint(["User", "Session"])
        gen = BlueprintCodeGenerator(blueprint)
        generated = gen.generate()

        # Compare generated code against itself
        reports = diff_blueprint_vs_code(blueprint, [("test.py", generated)])
        assert len(reports) == 1
        assert reports[0].syntax_valid is True
        assert reports[0].overall_score > 0.0

    def test_diff_with_empty_source(self):
        """Empty source code should still produce a report."""
        blueprint = make_blueprint(["User"])
        reports = diff_blueprint_vs_code(blueprint, [("empty.py", "")])
        assert len(reports) == 1
        # Empty code is valid Python (no syntax error)
        assert reports[0].syntax_valid is True

    def test_diff_with_syntax_error(self):
        """Source with syntax errors produces report with syntax_valid=False."""
        blueprint = make_blueprint(["User"])
        bad_code = "def broken(\nclass what:"
        reports = diff_blueprint_vs_code(blueprint, [("bad.py", bad_code)])
        assert len(reports) == 1
        assert reports[0].syntax_valid is False
        assert reports[0].overall_score == 0.0

    def test_diff_multiple_files(self):
        """Multiple source files produce multiple reports."""
        blueprint = make_blueprint(["User", "Session"])
        code = "class User:\n    pass\n"
        reports = diff_blueprint_vs_code(
            blueprint,
            [("a.py", code), ("b.py", code)],
        )
        assert len(reports) == 2

    def test_diff_empty_source_files(self):
        """Empty source files list produces empty tuple."""
        blueprint = make_blueprint()
        reports = diff_blueprint_vs_code(blueprint, [])
        assert reports == ()

    def test_diff_derived_from(self):
        """All reports have derived_from set."""
        blueprint = make_blueprint(["User"])
        reports = diff_blueprint_vs_code(blueprint, [("test.py", "x = 1\n")])
        assert reports[0].derived_from == "self-compile:v3.0"


class TestTrackConvergence:
    def test_identical_fingerprints_converge(self):
        """Identical fingerprints → variance_score 0.0, is_converged True."""
        fp = make_fingerprint()
        report = track_convergence([fp, fp, fp], ["A", "B"])
        assert report.is_converged is True
        assert report.variance.variance_score == 0.0
        assert report.structural_drift == 0.0
        assert len(report.points) == 3

    def test_different_fingerprints_drift(self):
        """Different fingerprints → non-zero variance and drift."""
        fp1 = make_fingerprint(("A", "B"), (("A", "B", "uses"),))
        fp2 = make_fingerprint(("C", "D"), (("C", "D", "triggers"),))
        report = track_convergence([fp1, fp2], ["A"])
        assert report.is_converged is False
        assert report.variance.variance_score > 0.0
        assert report.structural_drift > 0.0

    def test_single_fingerprint(self):
        """Single fingerprint → converged, no drift."""
        fp = make_fingerprint()
        report = track_convergence([fp], ["A", "B"])
        assert report.is_converged is True
        assert report.structural_drift == 0.0

    def test_empty_fingerprints(self):
        """Empty fingerprint list → converged, empty report."""
        report = track_convergence([], ["A"])
        assert report.is_converged is True
        assert len(report.points) == 0

    def test_canonical_coverage_computed(self):
        """Canonical coverage reflects fraction of canonical components found."""
        fp = make_fingerprint(("A", "B", "C"), ())
        report = track_convergence([fp], ["A", "B", "D", "E"])
        # A and B found (2/4 = 0.5)
        assert report.points[0].canonical_coverage == 0.5

    def test_canonical_coverage_case_insensitive(self):
        """Coverage matching is case-insensitive."""
        fp = make_fingerprint(("sharedstate",), ())
        report = track_convergence([fp], ["SharedState"])
        assert report.points[0].canonical_coverage == 1.0

    def test_derived_from(self):
        """Report has derived_from set."""
        report = track_convergence([], [])
        assert report.derived_from == "self-compile:v3.0"


class TestExtractSelfPatterns:
    def test_stable_component(self):
        """Component in >=90% of runs → stable_component."""
        bps = [make_blueprint(["X", "Y"]) for _ in range(10)]
        patterns = extract_self_patterns(bps, [])
        stable = [p for p in patterns if p.pattern_type == "stable_component"]
        names = {p.name for p in stable}
        assert "X" in names
        assert "Y" in names

    def test_drift_point(self):
        """Component in 30-70% of runs → drift_point."""
        bps = [make_blueprint(["X", "Y"]) for _ in range(5)]
        # Add 5 more without "Y"
        bps.extend([make_blueprint(["X"]) for _ in range(5)])
        patterns = extract_self_patterns(bps, [])
        drift = [p for p in patterns if p.pattern_type == "drift_point" and p.name == "Y"]
        assert len(drift) == 1
        assert drift[0].frequency == 0.5

    def test_canonical_gap(self):
        """Canonical component not in any run → canonical_gap."""
        bps = [make_blueprint(["X"]) for _ in range(3)]
        patterns = extract_self_patterns(bps, ["MissingComp"])
        gaps = [p for p in patterns if p.pattern_type == "canonical_gap"]
        assert len(gaps) == 1
        assert gaps[0].name == "MissingComp"

    def test_stable_relationship(self):
        """Relationship in >=90% of runs → stable_relationship."""
        bps = [make_blueprint(["A", "B"], [("A", "B", "uses")]) for _ in range(10)]
        patterns = extract_self_patterns(bps, [])
        stable_rels = [p for p in patterns if p.pattern_type == "stable_relationship"]
        assert len(stable_rels) >= 1
        assert "A -> B (uses)" in stable_rels[0].name

    def test_empty_blueprints(self):
        """No blueprints → no patterns."""
        patterns = extract_self_patterns([], ["A"])
        assert patterns == ()

    def test_single_blueprint(self):
        """Single blueprint → all components at 100% frequency."""
        bps = [make_blueprint(["Alpha", "Beta"])]
        patterns = extract_self_patterns(bps, [])
        stable = [p for p in patterns if p.pattern_type == "stable_component"]
        assert all(p.frequency == 1.0 for p in stable)

    def test_no_false_canonical_gap_for_present(self):
        """Canonical component that IS found should NOT be a gap."""
        bps = [make_blueprint(["SharedState", "Message"])]
        patterns = extract_self_patterns(bps, ["SharedState"])
        gaps = [p for p in patterns if p.pattern_type == "canonical_gap"]
        assert len(gaps) == 0


class TestComputeOverallHealth:
    def test_perfect_health(self):
        """Perfect variance, code, and coverage → 1.0."""
        convergence = make_convergence_report(
            variance_score=0.0, canonical_coverage=1.0,
        )
        code_diffs = (
            CodeDiffReport(
                file_path="test.py", syntax_valid=True, classes_found=1,
                classes_total=1, overall_score=1.0, class_scores=(),
                missing_classes=(), derived_from="test",
            ),
        )
        health = compute_overall_health(convergence, code_diffs)
        assert health == pytest.approx(1.0)

    def test_zero_health(self):
        """Worst variance, no code, no coverage → 0.0."""
        convergence = make_convergence_report(
            variance_score=1.0, canonical_coverage=0.0,
        )
        health = compute_overall_health(convergence, ())
        assert health == pytest.approx(0.0)

    def test_partial_health(self):
        """Partial metrics → intermediate health."""
        convergence = make_convergence_report(
            variance_score=0.0, canonical_coverage=0.5,
        )
        code_diffs = (
            CodeDiffReport(
                file_path="test.py", syntax_valid=True, classes_found=1,
                classes_total=2, overall_score=0.6, class_scores=(),
                missing_classes=(), derived_from="test",
            ),
        )
        health = compute_overall_health(convergence, code_diffs)
        # 0.4 * 1.0 + 0.3 * 0.6 + 0.3 * 0.5 = 0.4 + 0.18 + 0.15 = 0.73
        assert health == pytest.approx(0.73)

    def test_health_clamped_to_01(self):
        """Health is always in [0, 1]."""
        convergence = make_convergence_report(variance_score=0.0)
        health = compute_overall_health(convergence, ())
        assert 0.0 <= health <= 1.0

    def test_no_code_diffs(self):
        """No code diffs → code component is 0."""
        convergence = make_convergence_report(
            variance_score=0.0, canonical_coverage=1.0,
        )
        health = compute_overall_health(convergence, ())
        # 0.4 * 1.0 + 0.3 * 0.0 + 0.3 * 1.0 = 0.7
        assert health == pytest.approx(0.7)


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    def _make_report(self):
        convergence = make_convergence_report(variance_score=0.0, canonical_coverage=0.8)
        diff = CodeDiffReport(
            file_path="core/engine.py", syntax_valid=True,
            classes_found=3, classes_total=5, overall_score=0.6,
            class_scores=(("Foo", 0.8), ("Bar", 0.4)),
            missing_classes=("Baz",),
            derived_from="self-compile:v3.0",
        )
        pattern = SelfPattern(
            pattern_type="stable_component", name="SharedState",
            frequency=1.0, details="Appears in 3/3 runs (100%)",
            derived_from="self-compile:v3.0",
        )
        return SelfCompileReport(
            convergence=convergence,
            code_diffs=(diff,),
            patterns=(pattern,),
            overall_health=0.85,
            timestamp="2026-02-09T12:00:00",
        )

    def test_round_trip(self):
        """Serialize then deserialize produces equivalent report."""
        original = self._make_report()
        data = serialize_self_compile_report(original)
        restored = deserialize_self_compile_report(data)

        assert restored.overall_health == original.overall_health
        assert restored.timestamp == original.timestamp
        assert len(restored.code_diffs) == len(original.code_diffs)
        assert restored.code_diffs[0].file_path == original.code_diffs[0].file_path
        assert restored.code_diffs[0].overall_score == original.code_diffs[0].overall_score
        assert len(restored.patterns) == len(original.patterns)
        assert restored.patterns[0].name == original.patterns[0].name
        assert restored.convergence.is_converged == original.convergence.is_converged
        assert restored.convergence.variance.variance_score == original.convergence.variance.variance_score

    def test_serialized_is_json_safe(self):
        """Serialized output is JSON-serializable."""
        import json
        report = self._make_report()
        data = serialize_self_compile_report(report)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_class_scores_round_trip(self):
        """Class scores survive serialization as tuples."""
        original = self._make_report()
        data = serialize_self_compile_report(original)
        restored = deserialize_self_compile_report(data)
        assert restored.code_diffs[0].class_scores == original.code_diffs[0].class_scores

    def test_empty_report_round_trip(self):
        """Empty report serializes and deserializes correctly."""
        convergence = make_convergence_report()
        report = SelfCompileReport(
            convergence=convergence, code_diffs=(), patterns=(),
            overall_health=0.4, timestamp="2026-01-01",
        )
        data = serialize_self_compile_report(report)
        restored = deserialize_self_compile_report(data)
        assert restored.overall_health == 0.4
        assert len(restored.code_diffs) == 0
        assert len(restored.patterns) == 0
