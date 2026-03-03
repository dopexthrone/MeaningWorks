"""
Tests for core/dimensional.py — Dimensional Blueprint Schema.

Phase A.1: Frozen dataclass hierarchy for dimensional blueprints.
"""

import json
import pytest

from core.dimensional import (
    DimensionAxis,
    NodePosition,
    FragileEdge,
    DimensionalMetadata,
    serialize_dimensional_metadata,
    deserialize_dimensional_metadata,
)


# =============================================================================
# Fixtures
# =============================================================================

def _make_axis(name="structural", depth=0.8):
    return DimensionAxis(
        name=name,
        range_low="low structure",
        range_high="high structure",
        exploration_depth=depth,
        derived_from="ConfidenceVector.structural",
    )


def _make_position(structural=0.9, behavioral=0.3, confidence=0.85):
    return NodePosition(
        dimension_values=(("structural", structural), ("behavioral", behavioral)),
        confidence=confidence,
    )


def _make_edge():
    return FragileEdge(
        description="Weak link between A and B",
        affected_nodes=("ComponentA", "ComponentB"),
        drift_risk="high",
        reasoning="Low confidence at discovery time",
        derived_from="Unresolved conflict at turn 5",
    )


def _make_metadata():
    axes = (_make_axis("structural", 0.8), _make_axis("behavioral", 0.6))
    positions = (
        ("ComponentA", _make_position(0.9, 0.3, 0.85)),
        ("ComponentB", _make_position(0.2, 0.8, 0.7)),
    )
    return DimensionalMetadata(
        dimensions=axes,
        node_positions=positions,
        fragile_edges=(_make_edge(),),
        silence_zones=("security", "performance"),
        confidence_trajectory=(0.1, 0.3, 0.5, 0.7, 0.8),
        dimension_confidence=(("structural", 0.8), ("behavioral", 0.6)),
        dialogue_depth=17,
        stage_discovery=(("ComponentA", "EXPAND"), ("ComponentB", "DECOMPOSE")),
    )


# =============================================================================
# Immutability tests
# =============================================================================

class TestImmutability:
    def test_dimension_axis_frozen(self):
        axis = _make_axis()
        with pytest.raises(AttributeError):
            axis.name = "modified"

    def test_node_position_frozen(self):
        pos = _make_position()
        with pytest.raises(AttributeError):
            pos.confidence = 0.5

    def test_fragile_edge_frozen(self):
        edge = _make_edge()
        with pytest.raises(AttributeError):
            edge.drift_risk = "low"

    def test_dimensional_metadata_frozen(self):
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.dialogue_depth = 99


# =============================================================================
# DimensionAxis tests
# =============================================================================

class TestDimensionAxis:
    def test_construction(self):
        axis = _make_axis()
        assert axis.name == "structural"
        assert axis.exploration_depth == 0.8
        assert axis.derived_from == "ConfidenceVector.structural"

    def test_derived_from_present(self):
        axis = _make_axis()
        assert axis.derived_from != ""

    def test_silence_zones_default(self):
        axis = _make_axis()
        assert axis.silence_zones == ()

    def test_silence_zones_populated(self):
        axis = DimensionAxis(
            name="security",
            range_low="none",
            range_high="full",
            exploration_depth=0.2,
            derived_from="dialogue gap",
            silence_zones=("auth", "encryption"),
        )
        assert len(axis.silence_zones) == 2


# =============================================================================
# NodePosition tests
# =============================================================================

class TestNodePosition:
    def test_construction(self):
        pos = _make_position()
        assert pos.confidence == 0.85

    def test_get_value_existing_axis(self):
        pos = _make_position(structural=0.9)
        assert pos.get_value("structural") == 0.9

    def test_get_value_missing_axis(self):
        pos = _make_position()
        assert pos.get_value("nonexistent") == 0.0

    def test_dimension_values_tuple(self):
        pos = _make_position()
        assert isinstance(pos.dimension_values, tuple)
        assert all(isinstance(pair, tuple) for pair in pos.dimension_values)


# =============================================================================
# FragileEdge tests
# =============================================================================

class TestFragileEdge:
    def test_construction(self):
        edge = _make_edge()
        assert edge.drift_risk == "high"
        assert len(edge.affected_nodes) == 2

    def test_derived_from_present(self):
        edge = _make_edge()
        assert edge.derived_from != ""


# =============================================================================
# DimensionalMetadata tests
# =============================================================================

class TestDimensionalMetadata:
    def test_construction(self):
        meta = _make_metadata()
        assert len(meta.dimensions) == 2
        assert len(meta.node_positions) == 2
        assert meta.dialogue_depth == 17

    def test_get_position_existing(self):
        meta = _make_metadata()
        pos = meta.get_position("ComponentA")
        assert pos.confidence == 0.85

    def test_get_position_missing(self):
        meta = _make_metadata()
        pos = meta.get_position("NonExistent")
        assert pos.confidence == 0.0

    def test_get_dimension_existing(self):
        meta = _make_metadata()
        dim = meta.get_dimension("structural")
        assert dim.exploration_depth == 0.8

    def test_get_dimension_missing(self):
        meta = _make_metadata()
        dim = meta.get_dimension("nonexistent")
        assert dim is None

    def test_confidence_trajectory(self):
        meta = _make_metadata()
        assert meta.confidence_trajectory == (0.1, 0.3, 0.5, 0.7, 0.8)


# =============================================================================
# Serialization tests
# =============================================================================

class TestSerialization:
    def test_serialize_produces_dict(self):
        meta = _make_metadata()
        result = serialize_dimensional_metadata(meta)
        assert isinstance(result, dict)

    def test_serialize_json_compatible(self):
        meta = _make_metadata()
        result = serialize_dimensional_metadata(meta)
        # Must not raise
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_serialize_has_all_keys(self):
        meta = _make_metadata()
        result = serialize_dimensional_metadata(meta)
        expected_keys = {
            "axes", "node_positions", "fragile_edges", "silence_zones",
            "confidence_trajectory", "dimension_confidence", "dialogue_depth",
            "stage_discovery",
        }
        assert set(result.keys()) == expected_keys

    def test_round_trip(self):
        """Serialize then deserialize should produce equivalent data."""
        meta = _make_metadata()
        serialized = serialize_dimensional_metadata(meta)
        restored = deserialize_dimensional_metadata(serialized)

        assert len(restored.dimensions) == len(meta.dimensions)
        assert restored.dialogue_depth == meta.dialogue_depth
        assert restored.silence_zones == meta.silence_zones
        assert restored.confidence_trajectory == meta.confidence_trajectory

        # Check node positions
        for name, pos in meta.node_positions:
            restored_pos = restored.get_position(name)
            assert restored_pos.confidence == pos.confidence

    def test_deserialize_empty(self):
        """Deserializing empty dict should produce empty metadata."""
        meta = deserialize_dimensional_metadata({})
        assert len(meta.dimensions) == 0
        assert len(meta.node_positions) == 0
        assert meta.dialogue_depth == 0

    def test_serialize_axes_structure(self):
        meta = _make_metadata()
        result = serialize_dimensional_metadata(meta)
        assert len(result["axes"]) == 2
        axis = result["axes"][0]
        assert "name" in axis
        assert "exploration_depth" in axis
        assert "derived_from" in axis

    def test_serialize_fragile_edges_structure(self):
        meta = _make_metadata()
        result = serialize_dimensional_metadata(meta)
        assert len(result["fragile_edges"]) == 1
        edge = result["fragile_edges"][0]
        assert edge["drift_risk"] == "high"
        assert isinstance(edge["affected_nodes"], list)
