"""
Tests for core/interface_extractor.py — algorithmic interface extraction.

Phase B.1.2: Interface Extractor
~25 tests — direction per relationship type, constraint matching,
fragility from dimensional metadata, empty/edge cases.
"""

import pytest

from core.interface_schema import (
    DataFlow,
    InterfaceConstraint,
    InterfaceContract,
    InterfaceMap,
)
from core.interface_extractor import (
    extract_data_flows,
    extract_interface_constraints,
    compute_edge_fragility,
    determine_directionality,
    extract_interface_map,
)
from core.dimensional import (
    DimensionalMetadata,
    DimensionAxis,
    NodePosition,
    FragileEdge,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_dim_meta(
    node_positions=None,
    fragile_edges=None,
) -> DimensionalMetadata:
    """Build a minimal DimensionalMetadata for testing."""
    return DimensionalMetadata(
        dimensions=(
            DimensionAxis(
                name="structural", range_low="low", range_high="high",
                exploration_depth=0.8, derived_from="test",
            ),
        ),
        node_positions=node_positions or (),
        fragile_edges=fragile_edges or (),
        silence_zones=(),
        confidence_trajectory=(0.5, 0.7),
        dimension_confidence=(("structural", 0.8),),
        dialogue_depth=3,
        stage_discovery=(),
    )


def _make_blueprint(
    components=None,
    relationships=None,
    constraints=None,
):
    """Build a minimal blueprint dict for testing."""
    return {
        "components": components or [],
        "relationships": relationships or [],
        "constraints": constraints or [],
        "unresolved": [],
    }


# =============================================================================
# Directionality Tests
# =============================================================================

class TestDetermineDirectionality:
    def test_triggers_a_depends_on_b(self):
        assert determine_directionality({"type": "triggers"}) == "A_depends_on_B"

    def test_flows_to_a_depends_on_b(self):
        assert determine_directionality({"type": "flows_to"}) == "A_depends_on_B"

    def test_generates_a_depends_on_b(self):
        assert determine_directionality({"type": "generates"}) == "A_depends_on_B"

    def test_propagates_a_depends_on_b(self):
        assert determine_directionality({"type": "propagates"}) == "A_depends_on_B"

    def test_snapshots_a_depends_on_b(self):
        assert determine_directionality({"type": "snapshots"}) == "A_depends_on_B"

    def test_accesses_b_depends_on_a(self):
        assert determine_directionality({"type": "accesses"}) == "B_depends_on_A"

    def test_depends_on_b_depends_on_a(self):
        assert determine_directionality({"type": "depends_on"}) == "B_depends_on_A"

    def test_constrained_by_b_depends_on_a(self):
        assert determine_directionality({"type": "constrained_by"}) == "B_depends_on_A"

    def test_monitors_b_depends_on_a(self):
        assert determine_directionality({"type": "monitors"}) == "B_depends_on_A"

    def test_bidirectional_mutual(self):
        assert determine_directionality({"type": "bidirectional"}) == "mutual"

    def test_contains_mutual(self):
        assert determine_directionality({"type": "contains"}) == "mutual"

    def test_unknown_type_defaults_mutual(self):
        assert determine_directionality({"type": "custom_type"}) == "mutual"


# =============================================================================
# Data Flow Extraction Tests
# =============================================================================

class TestExtractDataFlows:
    def test_triggers_creates_signal_flow(self):
        rel = {"type": "triggers", "description": "Governor triggers Intent"}
        flows = extract_data_flows(rel, "Governor Agent", "Intent Agent")
        assert len(flows) == 1
        assert flows[0].direction == "A_to_B"
        assert flows[0].name == "trigger_signal"

    def test_accesses_creates_data_access_flow(self):
        rel = {"type": "accesses", "description": "Agent accesses SharedState"}
        flows = extract_data_flows(rel, "Entity Agent", "SharedState")
        assert len(flows) == 1
        assert flows[0].direction == "B_to_A"
        assert flows[0].type_hint == "SharedState"

    def test_unknown_type_creates_generic_flow(self):
        rel = {"type": "custom_thing", "description": "custom relationship"}
        flows = extract_data_flows(rel, "A", "B")
        assert len(flows) == 1
        assert flows[0].direction == "bidirectional"
        assert "custom_thing" in flows[0].name

    def test_snapshots_flow(self):
        rel = {"type": "snapshots", "description": "Corpus snapshots state"}
        flows = extract_data_flows(rel, "Corpus", "SharedState")
        assert flows[0].direction == "A_to_B"

    def test_type_hint_inference_for_sharedstate(self):
        rel = {"type": "accesses", "description": "reads state"}
        flows = extract_data_flows(rel, "Entity Agent", "SharedState")
        assert flows[0].type_hint == "SharedState"

    def test_type_hint_inference_for_confidence(self):
        rel = {"type": "monitors", "description": "monitors confidence"}
        flows = extract_data_flows(rel, "ConflictOracle", "ConfidenceVector")
        assert flows[0].type_hint == "ConfidenceVector"


# =============================================================================
# Interface Constraint Extraction Tests
# =============================================================================

class TestExtractInterfaceConstraints:
    def test_constraint_mentioning_both_nodes(self):
        constraints = [{
            "description": "Governor must trigger Intent Agent before others",
            "applies_to": ["Governor Agent", "Intent Agent"],
            "derived_from": "test",
        }]
        result = extract_interface_constraints("Governor Agent", "Intent Agent", constraints)
        assert len(result) == 1
        assert result[0].constraint_type == "custom"

    def test_constraint_mentioning_one_node_range(self):
        constraints = [{
            "description": "confidence in range [0, 1]",
            "applies_to": ["ConfidenceVector"],
            "derived_from": "test",
        }]
        result = extract_interface_constraints("ConflictOracle", "ConfidenceVector", constraints)
        assert len(result) == 1
        assert result[0].constraint_type == "range"

    def test_no_matching_constraints(self):
        constraints = [{
            "description": "unrelated constraint",
            "applies_to": ["SomeOtherComponent"],
            "derived_from": "test",
        }]
        result = extract_interface_constraints("A", "B", constraints)
        assert len(result) == 0

    def test_constraint_in_description_text(self):
        constraints = [{
            "description": "SharedState must not be null when accessed by Entity Agent",
            "applies_to": [],
            "derived_from": "test",
        }]
        result = extract_interface_constraints("Entity Agent", "SharedState", constraints)
        assert len(result) >= 1


# =============================================================================
# Edge Fragility Tests
# =============================================================================

class TestComputeEdgeFragility:
    def test_no_dim_meta_returns_default(self):
        result = compute_edge_fragility("A", "B", None)
        assert result == 0.5

    def test_fragile_edge_increases_fragility(self):
        dim_meta = _make_dim_meta(
            fragile_edges=(
                FragileEdge(
                    description="high risk edge",
                    affected_nodes=("A",),
                    drift_risk="high",
                    reasoning="test",
                    derived_from="test",
                ),
            ),
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.8)),
                ("B", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.8)),
            ),
        )
        result = compute_edge_fragility("A", "B", dim_meta)
        assert result > 0.0

    def test_close_positions_lower_fragility(self):
        dim_meta = _make_dim_meta(
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.9)),
                ("B", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.9)),
            ),
        )
        close = compute_edge_fragility("A", "B", dim_meta)

        dim_meta_far = _make_dim_meta(
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.0),), confidence=0.9)),
                ("B", NodePosition(dimension_values=(("structural", 1.0),), confidence=0.9)),
            ),
        )
        far = compute_edge_fragility("A", "B", dim_meta_far)

        assert close < far

    def test_low_confidence_increases_fragility(self):
        dim_meta = _make_dim_meta(
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.1)),
                ("B", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.1)),
            ),
        )
        low_conf = compute_edge_fragility("A", "B", dim_meta)

        dim_meta_high = _make_dim_meta(
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.9)),
                ("B", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.9)),
            ),
        )
        high_conf = compute_edge_fragility("A", "B", dim_meta_high)

        assert low_conf > high_conf


# =============================================================================
# Full Interface Map Extraction Tests
# =============================================================================

class TestExtractInterfaceMap:
    def test_empty_blueprint(self):
        bp = _make_blueprint()
        result = extract_interface_map(bp)
        assert isinstance(result, InterfaceMap)
        assert len(result.contracts) == 0

    def test_single_relationship(self):
        bp = _make_blueprint(
            components=[
                {"name": "A", "type": "process", "description": "test", "derived_from": "test"},
                {"name": "B", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "A", "to": "B", "type": "triggers", "description": "A triggers B"},
            ],
        )
        result = extract_interface_map(bp)
        assert len(result.contracts) == 1
        assert result.contracts[0].node_a == "A"
        assert result.contracts[0].node_b == "B"
        assert result.contracts[0].directionality == "A_depends_on_B"

    def test_multiple_relationships(self):
        bp = _make_blueprint(
            components=[
                {"name": "Gov", "type": "process", "description": "test", "derived_from": "test"},
                {"name": "Intent", "type": "process", "description": "test", "derived_from": "test"},
                {"name": "State", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "Gov", "to": "Intent", "type": "triggers", "description": "triggers"},
                {"from": "Intent", "to": "State", "type": "accesses", "description": "reads state"},
            ],
        )
        result = extract_interface_map(bp)
        assert len(result.contracts) == 2

    def test_unmatched_relationship(self):
        bp = _make_blueprint(
            components=[
                {"name": "A", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "A", "to": "NonExistent", "type": "triggers", "description": "test"},
            ],
        )
        result = extract_interface_map(bp)
        assert len(result.contracts) == 0
        assert len(result.unmatched_relationships) == 1

    def test_with_dimensional_metadata_increases_confidence(self):
        bp = _make_blueprint(
            components=[
                {"name": "A", "type": "process", "description": "test", "derived_from": "test"},
                {"name": "B", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "A", "to": "B", "type": "triggers", "description": "test"},
            ],
        )
        result_without = extract_interface_map(bp, dim_meta=None)
        dim_meta = _make_dim_meta(
            node_positions=(
                ("A", NodePosition(dimension_values=(("structural", 0.5),), confidence=0.8)),
                ("B", NodePosition(dimension_values=(("structural", 0.6),), confidence=0.8)),
            ),
        )
        result_with = extract_interface_map(bp, dim_meta=dim_meta)

        assert result_with.contracts[0].confidence > result_without.contracts[0].confidence

    def test_extraction_confidence_reduced_by_unmatched(self):
        bp = _make_blueprint(
            components=[
                {"name": "A", "type": "entity", "description": "test", "derived_from": "test"},
                {"name": "B", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "A", "to": "B", "type": "triggers", "description": "test"},
                {"from": "A", "to": "Missing", "type": "triggers", "description": "test"},
            ],
        )
        result = extract_interface_map(bp)
        # With 1 matched and 1 unmatched, confidence is penalized
        assert result.extraction_confidence < 1.0

    def test_constraints_included_in_contracts(self):
        bp = _make_blueprint(
            components=[
                {"name": "ConflictOracle", "type": "entity", "description": "test", "derived_from": "test"},
                {"name": "ConfidenceVector", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "ConflictOracle", "to": "ConfidenceVector", "type": "monitors", "description": "monitors"},
            ],
            constraints=[{
                "description": "ConfidenceVector values in range [0, 1]",
                "applies_to": ["ConfidenceVector"],
                "derived_from": "test",
            }],
        )
        result = extract_interface_map(bp)
        assert len(result.contracts) == 1
        assert len(result.contracts[0].constraints) >= 1

    def test_interface_map_is_frozen(self):
        bp = _make_blueprint(
            components=[
                {"name": "A", "type": "entity", "description": "test", "derived_from": "test"},
                {"name": "B", "type": "entity", "description": "test", "derived_from": "test"},
            ],
            relationships=[
                {"from": "A", "to": "B", "type": "triggers", "description": "test"},
            ],
        )
        result = extract_interface_map(bp)
        with pytest.raises(AttributeError):
            result.extraction_confidence = 0.0
