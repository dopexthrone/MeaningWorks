"""
Tests for core/determinism.py — structural fingerprinting and variance.

Phase 13: Determinism & Reproducibility
~20 tests — identity, ordering invariance, distance metrics, variance report.
Plus pipeline determinism integration tests.
"""

import pytest

from core.determinism import (
    StructuralFingerprint,
    StructuralDistance,
    VarianceReport,
    compute_structural_fingerprint,
    compute_structural_distance,
    build_variance_report,
)


# =============================================================================
# Fixtures
# =============================================================================

def _bp(components=None, relationships=None, constraints=None, unresolved=None):
    """Build a minimal blueprint dict."""
    return {
        "components": components or [],
        "relationships": relationships or [],
        "constraints": constraints or [],
        "unresolved": unresolved or [],
    }


def _comp(name, ctype="entity"):
    return {"name": name, "type": ctype, "description": f"{name} desc", "derived_from": "test"}


def _rel(from_c, to_c, rtype="triggers"):
    return {"from": from_c, "to": to_c, "type": rtype, "description": "test"}


# =============================================================================
# StructuralFingerprint Tests
# =============================================================================

class TestStructuralFingerprint:
    def test_frozen(self):
        fp = compute_structural_fingerprint(_bp())
        with pytest.raises(AttributeError):
            fp.hash_digest = "xxx"

    def test_empty_blueprint(self):
        fp = compute_structural_fingerprint(_bp())
        assert fp.component_set == ()
        assert fp.relationship_set == ()
        assert fp.constraint_count == 0
        assert fp.unresolved_count == 0
        assert len(fp.hash_digest) == 16

    def test_components_sorted(self):
        bp = _bp(components=[_comp("Z"), _comp("A"), _comp("M")])
        fp = compute_structural_fingerprint(bp)
        assert fp.component_set == ("A", "M", "Z")

    def test_relationships_sorted(self):
        bp = _bp(
            components=[_comp("A"), _comp("B"), _comp("C")],
            relationships=[
                _rel("C", "A", "accesses"),
                _rel("A", "B", "triggers"),
            ],
        )
        fp = compute_structural_fingerprint(bp)
        assert fp.relationship_set[0][0] == "A"  # A->B before C->A

    def test_component_types_tracked(self):
        bp = _bp(components=[
            _comp("Auth", "process"),
            _comp("User", "entity"),
        ])
        fp = compute_structural_fingerprint(bp)
        types_dict = dict(fp.component_types)
        assert types_dict["Auth"] == "process"
        assert types_dict["User"] == "entity"


# =============================================================================
# Identity and Ordering Invariance Tests
# =============================================================================

class TestFingerprintIdentity:
    def test_identical_blueprints_same_hash(self):
        bp = _bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B")],
        )
        fp1 = compute_structural_fingerprint(bp)
        fp2 = compute_structural_fingerprint(bp)
        assert fp1.hash_digest == fp2.hash_digest

    def test_component_order_invariant(self):
        bp1 = _bp(components=[_comp("A"), _comp("B"), _comp("C")])
        bp2 = _bp(components=[_comp("C"), _comp("A"), _comp("B")])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        assert fp1.hash_digest == fp2.hash_digest

    def test_relationship_order_invariant(self):
        bp1 = _bp(
            components=[_comp("A"), _comp("B"), _comp("C")],
            relationships=[_rel("A", "B"), _rel("B", "C")],
        )
        bp2 = _bp(
            components=[_comp("A"), _comp("B"), _comp("C")],
            relationships=[_rel("B", "C"), _rel("A", "B")],
        )
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        assert fp1.hash_digest == fp2.hash_digest

    def test_description_ignored(self):
        comp1 = {"name": "A", "type": "entity", "description": "short", "derived_from": "x"}
        comp2 = {"name": "A", "type": "entity", "description": "very long different description", "derived_from": "y"}
        fp1 = compute_structural_fingerprint(_bp(components=[comp1]))
        fp2 = compute_structural_fingerprint(_bp(components=[comp2]))
        assert fp1.hash_digest == fp2.hash_digest

    def test_different_components_different_hash(self):
        bp1 = _bp(components=[_comp("A")])
        bp2 = _bp(components=[_comp("B")])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        assert fp1.hash_digest != fp2.hash_digest

    def test_different_types_different_hash(self):
        bp1 = _bp(components=[_comp("A", "entity")])
        bp2 = _bp(components=[_comp("A", "process")])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        assert fp1.hash_digest != fp2.hash_digest

    def test_different_relationships_different_hash(self):
        bp1 = _bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B", "triggers")],
        )
        bp2 = _bp(
            components=[_comp("A"), _comp("B")],
            relationships=[_rel("A", "B", "accesses")],
        )
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        assert fp1.hash_digest != fp2.hash_digest


# =============================================================================
# Structural Distance Tests
# =============================================================================

class TestStructuralDistance:
    def test_identical_zero_distance(self):
        bp = _bp(components=[_comp("A"), _comp("B")], relationships=[_rel("A", "B")])
        fp = compute_structural_fingerprint(bp)
        dist = compute_structural_distance(fp, fp)
        assert dist.overall_distance == 0.0
        assert dist.jaccard_components == 1.0
        assert dist.jaccard_relationships == 1.0
        assert len(dist.type_mismatches) == 0

    def test_completely_different_high_distance(self):
        bp1 = _bp(components=[_comp("A")], relationships=[])
        bp2 = _bp(components=[_comp("Z")], relationships=[])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        dist = compute_structural_distance(fp1, fp2)
        assert dist.overall_distance > 0.0
        assert dist.jaccard_components == 0.0
        assert dist.added_components == ("Z",)
        assert dist.removed_components == ("A",)

    def test_partial_overlap(self):
        bp1 = _bp(components=[_comp("A"), _comp("B")])
        bp2 = _bp(components=[_comp("B"), _comp("C")])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        dist = compute_structural_distance(fp1, fp2)
        # Jaccard: |{B}| / |{A,B,C}| = 1/3
        assert abs(dist.jaccard_components - 1.0/3.0) < 0.01

    def test_type_mismatches_detected(self):
        bp1 = _bp(components=[_comp("A", "entity")])
        bp2 = _bp(components=[_comp("A", "process")])
        fp1 = compute_structural_fingerprint(bp1)
        fp2 = compute_structural_fingerprint(bp2)
        dist = compute_structural_distance(fp1, fp2)
        assert len(dist.type_mismatches) == 1
        assert dist.type_mismatches[0] == ("A", "entity", "process")

    def test_empty_blueprints_zero_distance(self):
        fp1 = compute_structural_fingerprint(_bp())
        fp2 = compute_structural_fingerprint(_bp())
        dist = compute_structural_distance(fp1, fp2)
        assert dist.overall_distance == 0.0


# =============================================================================
# Variance Report Tests
# =============================================================================

class TestVarianceReport:
    def test_empty_input(self):
        report = build_variance_report([])
        assert report.run_count == 0
        assert report.variance_score == 0.0

    def test_single_run(self):
        fp = compute_structural_fingerprint(_bp(components=[_comp("A")]))
        report = build_variance_report([fp])
        assert report.run_count == 1
        assert report.unique_structures == 1
        assert report.variance_score == 0.0

    def test_identical_runs_zero_variance(self):
        bp = _bp(components=[_comp("A"), _comp("B")], relationships=[_rel("A", "B")])
        fp = compute_structural_fingerprint(bp)
        report = build_variance_report([fp, fp, fp])
        assert report.run_count == 3
        assert report.unique_structures == 1
        assert report.variance_score == 0.0
        assert report.dominant_frequency == 3

    def test_all_different_max_variance(self):
        fps = [
            compute_structural_fingerprint(_bp(components=[_comp("A")])),
            compute_structural_fingerprint(_bp(components=[_comp("B")])),
            compute_structural_fingerprint(_bp(components=[_comp("C")])),
        ]
        report = build_variance_report(fps)
        assert report.run_count == 3
        assert report.unique_structures == 3
        # variance = 1 - (1/3) = 0.667
        assert abs(report.variance_score - 2.0/3.0) < 0.01

    def test_mostly_deterministic(self):
        bp_main = _bp(components=[_comp("A"), _comp("B")])
        bp_outlier = _bp(components=[_comp("A"), _comp("C")])
        fp_main = compute_structural_fingerprint(bp_main)
        fp_outlier = compute_structural_fingerprint(bp_outlier)
        report = build_variance_report([fp_main, fp_main, fp_main, fp_outlier])
        assert report.unique_structures == 2
        assert report.dominant_frequency == 3
        assert report.variance_score == 0.25  # 1 - 3/4

    def test_report_is_frozen(self):
        fp = compute_structural_fingerprint(_bp())
        report = build_variance_report([fp])
        with pytest.raises(AttributeError):
            report.variance_score = 1.0


# =============================================================================
# Pipeline Determinism Tests (Phase 13.2)
# =============================================================================

class TestPipelineDeterminism:
    """
    Verify that the compilation pipeline is deterministic when using MockClient.

    These tests catch non-deterministic code paths (set iteration, dict ordering,
    random choices) independent of LLM output.
    """

    @pytest.fixture
    def engine(self, tmp_path):
        """Create a test engine with MockClient."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine
        from persistence.corpus import Corpus

        corpus = Corpus(corpus_path=tmp_path / "corpus")
        return MotherlabsEngine(
            llm_client=MockClient(),
            corpus=corpus,
            auto_store=False,
        )

    def test_compile_deterministic_2_runs(self, engine):
        """Same input + MockClient -> identical fingerprint."""
        result1 = engine.compile("Build a login system")
        result2 = engine.compile("Build a login system")

        fp1 = compute_structural_fingerprint(result1.blueprint)
        fp2 = compute_structural_fingerprint(result2.blueprint)

        assert fp1.hash_digest == fp2.hash_digest

    def test_compile_3_runs_perfect_variance(self, engine):
        """Three runs -> variance_score == 0.0."""
        fps = []
        for _ in range(3):
            result = engine.compile("Build a login system")
            fps.append(compute_structural_fingerprint(result.blueprint))

        report = build_variance_report(fps)
        assert report.variance_score == 0.0
        assert report.unique_structures == 1
