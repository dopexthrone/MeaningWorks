"""
Tests for core/trust.py — trust computation module.

Phase C: Trust Computation Module
"""

import pytest
from core.trust import (
    TrustIndicators,
    compute_badge,
    compute_provenance_depth,
    detect_gaps,
    detect_silence_zones,
    compute_derivation_chain_length,
    extract_confidence_trajectory,
    compute_dimensional_coverage,
    compute_trust_indicators,
    serialize_trust_indicators,
)


# =============================================================================
# Badge computation
# =============================================================================

class TestComputeBadge:
    def test_verified_all_high(self):
        scores = {"completeness": 80, "consistency": 90, "coherence": 75, "traceability": 85}
        assert compute_badge(scores) == "verified"

    def test_verified_at_threshold(self):
        scores = {"completeness": 70, "consistency": 70, "coherence": 70, "traceability": 70}
        assert compute_badge(scores) == "verified"

    def test_partial_one_low(self):
        scores = {"completeness": 60, "consistency": 90, "coherence": 75, "traceability": 85}
        assert compute_badge(scores) == "partial"

    def test_unverified_below_40(self):
        scores = {"completeness": 30, "consistency": 90, "coherence": 75, "traceability": 85}
        assert compute_badge(scores) == "unverified"

    def test_unverified_all_low(self):
        scores = {"completeness": 10, "consistency": 20, "coherence": 15, "traceability": 5}
        assert compute_badge(scores) == "unverified"

    def test_empty_scores(self):
        assert compute_badge({}) == "unverified"

    def test_missing_dimensions_treated_as_zero(self):
        scores = {"completeness": 80}
        assert compute_badge(scores) == "unverified"  # Others default to 0


# =============================================================================
# Provenance depth
# =============================================================================

class TestProvenanceDepth:
    def test_empty(self):
        assert compute_provenance_depth({}) == 0

    def test_stratum_1(self):
        cg = {"input_hash": "abc123"}
        assert compute_provenance_depth(cg) == 1

    def test_stratum_2(self):
        cg = {"input_hash": "abc", "insights": ["insight1", "insight2"]}
        assert compute_provenance_depth(cg) == 2

    def test_stratum_3(self):
        cg = {"input_hash": "abc", "insights": ["i1"], "self_compile_patterns": ["p1"]}
        assert compute_provenance_depth(cg) == 3

    def test_stratum_3_stable_components(self):
        cg = {"keywords": ["k1"], "stable_components": ["c1"]}
        assert compute_provenance_depth(cg) == 3

    def test_stratum_2_conflicts(self):
        cg = {"input_hash": "x", "conflicts": [{"a": "b"}]}
        assert compute_provenance_depth(cg) == 2


# =============================================================================
# Gap detection
# =============================================================================

class TestDetectGaps:
    def test_no_gaps(self):
        blueprint = {"components": [
            {"name": "Auth", "description": "auth system", "derived_from": "user wants authentication"},
        ]}
        verification = {"completeness": {"score": 90, "gaps": []}}
        gaps = detect_gaps(blueprint, ["auth"], verification)
        assert len(gaps) == 0

    def test_verification_gaps(self):
        verification = {"completeness": {"score": 50, "gaps": ["Missing: error handling"]}}
        gaps = detect_gaps({"components": []}, [], verification)
        assert any("error handling" in g for g in gaps)

    def test_uncovered_keywords(self):
        blueprint = {"components": [{"name": "Auth", "description": "auth", "derived_from": "input"}]}
        gaps = detect_gaps(blueprint, ["auth", "payment", "notification"], {})
        assert any("payment" in g for g in gaps)
        assert any("notification" in g for g in gaps)

    def test_no_provenance(self):
        blueprint = {"components": [{"name": "Foo", "description": "bar"}]}
        gaps = detect_gaps(blueprint, [], {})
        assert any("No provenance" in g for g in gaps)

    def test_gap_cap(self):
        # Generate many gaps
        blueprint = {"components": [{"name": f"C{i}"} for i in range(30)]}
        gaps = detect_gaps(blueprint, [], {})
        assert len(gaps) <= 25


# =============================================================================
# Silence zones
# =============================================================================

class TestSilenceZones:
    def test_empty_metadata(self):
        zones = detect_silence_zones({}, [])
        assert zones == ()

    def test_confidence_drop(self):
        trajectory = [0.5, 0.6, 0.7, 0.4, 0.5]  # Drop at index 3
        zones = detect_silence_zones({}, trajectory)
        assert any("dropped" in z.lower() for z in zones)

    def test_fragile_edges(self):
        meta = {
            "dimensions": [],
            "node_positions": [],
            "fragile_edges": [
                {"drift_risk": "high", "affected_nodes": ["A", "B"]},
            ],
        }
        zones = detect_silence_zones(meta, [])
        assert any("fragile" in z.lower() for z in zones)

    def test_no_confidence_drop_if_steady(self):
        trajectory = [0.3, 0.4, 0.5, 0.6, 0.7]
        zones = detect_silence_zones({}, trajectory)
        assert not any("dropped" in z.lower() for z in zones)


# =============================================================================
# Derivation chain
# =============================================================================

class TestDerivationChainLength:
    def test_empty(self):
        assert compute_derivation_chain_length([]) == 0.0

    def test_no_derived_from(self):
        components = [{"name": "A"}, {"name": "B"}]
        assert compute_derivation_chain_length(components) == 0.0

    def test_shallow(self):
        components = [{"name": "A", "derived_from": "user input"}]
        result = compute_derivation_chain_length(components)
        assert result == 1.0

    def test_dialogue_depth(self):
        components = [{"name": "A", "derived_from": "dialogue turn 3: entity agent identified this"}]
        result = compute_derivation_chain_length(components)
        assert result == 2.0

    def test_synthesis_depth(self):
        components = [{"name": "A", "derived_from": "synthesis of multiple dialogue insights"}]
        result = compute_derivation_chain_length(components)
        assert result == 3.0

    def test_mixed(self):
        components = [
            {"name": "A", "derived_from": "user input"},
            {"name": "B", "derived_from": "synthesis of insights 1,2,3"},
        ]
        result = compute_derivation_chain_length(components)
        assert result == 2.0  # (1.0 + 3.0) / 2


# =============================================================================
# Confidence trajectory
# =============================================================================

class TestConfidenceTrajectory:
    def test_explicit_trajectory(self):
        cg = {"confidence_trajectory": [0.1, 0.3, 0.5, 0.8]}
        result = extract_confidence_trajectory(cg)
        assert result == (0.1, 0.3, 0.5, 0.8)

    def test_fallback_from_insights(self):
        cg = {"insights": ["i1", "i2", "i3", "i4"]}
        result = extract_confidence_trajectory(cg)
        assert len(result) == 4
        assert result[-1] == 1.0

    def test_empty(self):
        result = extract_confidence_trajectory({})
        assert result == ()


# =============================================================================
# Dimensional coverage
# =============================================================================

class TestDimensionalCoverage:
    def test_empty(self):
        assert compute_dimensional_coverage({}) == {}

    def test_full_coverage(self):
        meta = {
            "dimensions": [{"name": "complexity"}],
            "node_positions": [
                {"component_name": "A", "dimension_values": [("complexity", 0.5)]},
                {"component_name": "B", "dimension_values": [("complexity", 0.8)]},
            ],
        }
        cov = compute_dimensional_coverage(meta)
        assert cov.get("complexity", 0) == 1.0

    def test_partial_coverage(self):
        meta = {
            "dimensions": [{"name": "complexity"}],
            "node_positions": [
                {"component_name": "A", "dimension_values": [("complexity", 0.5)]},
                {"component_name": "B", "dimension_values": []},
            ],
        }
        cov = compute_dimensional_coverage(meta)
        assert cov.get("complexity", 0) == 0.5


# =============================================================================
# Full trust computation
# =============================================================================

class TestComputeTrustIndicators:
    def test_basic_computation(self):
        blueprint = {
            "components": [
                {"name": "Auth", "type": "entity", "description": "Authentication system",
                 "derived_from": "user wants authentication system"},
            ],
            "relationships": [{"from": "Auth", "to": "DB", "type": "accesses"}],
            "constraints": [{"description": "Must be secure"}],
        }
        verification = {
            "completeness": {"score": 80},
            "consistency": {"score": 90},
            "coherence": {"score": 75},
            "traceability": {"score": 85},
            "actionability": {"score": 60},
            "specificity": {"score": 55},
            "codegen_readiness": {"score": 70},
        }
        context_graph = {"input_hash": "abc", "insights": ["i1", "i2"]}

        trust = compute_trust_indicators(
            blueprint, verification, context_graph, {}, ["auth"],
        )

        assert isinstance(trust, TrustIndicators)
        assert trust.overall_score > 0
        assert trust.verification_badge == "verified"
        assert trust.provenance_depth == 2
        assert trust.component_count == 1
        assert trust.relationship_count == 1
        assert trust.constraint_count == 1

    def test_unverified_result(self):
        verification = {
            "completeness": {"score": 20},
            "consistency": {"score": 30},
            "coherence": {"score": 15},
            "traceability": {"score": 25},
        }
        trust = compute_trust_indicators(
            {"components": [], "relationships": [], "constraints": []},
            verification, {}, {}, [],
        )
        assert trust.verification_badge == "unverified"
        assert trust.overall_score < 30

    def test_partial_result(self):
        verification = {
            "completeness": {"score": 55},
            "consistency": {"score": 80},
            "coherence": {"score": 60},
            "traceability": {"score": 50},
        }
        trust = compute_trust_indicators(
            {"components": [], "relationships": [], "constraints": []},
            verification, {}, {}, [],
        )
        assert trust.verification_badge == "partial"

    def test_frozen(self):
        trust = compute_trust_indicators(
            {"components": [], "relationships": [], "constraints": []},
            {}, {}, {}, [],
        )
        with pytest.raises(AttributeError):
            trust.overall_score = 99.0


# =============================================================================
# Serialization
# =============================================================================

class TestSerializeTrustIndicators:
    def test_round_trip(self):
        trust = compute_trust_indicators(
            {"components": [{"name": "X", "derived_from": "input"}],
             "relationships": [], "constraints": []},
            {"completeness": {"score": 80}, "consistency": {"score": 90},
             "coherence": {"score": 75}, "traceability": {"score": 85}},
            {"input_hash": "abc"}, {}, ["x"],
        )
        data = serialize_trust_indicators(trust)
        assert isinstance(data, dict)
        assert data["overall_score"] == trust.overall_score
        assert data["verification_badge"] == trust.verification_badge
        assert isinstance(data["gap_report"], list)
        assert isinstance(data["confidence_trajectory"], list)

    def test_json_safe(self):
        import json
        trust = compute_trust_indicators(
            {"components": [], "relationships": [], "constraints": []},
            {}, {}, {}, [],
        )
        data = serialize_trust_indicators(trust)
        # Must be JSON serializable
        json_str = json.dumps(data)
        assert json_str
