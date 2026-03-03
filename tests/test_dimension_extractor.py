"""
Tests for core/dimension_extractor.py — Dimension extraction from SharedState.

Phase A.2: Algorithmic extraction of dimensional metadata.
"""

import pytest

from core.protocol import SharedState, ConfidenceVector, Message, MessageType
from core.dimension_extractor import (
    extract_dimensions,
    extract_node_positions,
    extract_fragile_edges,
    extract_silence_zones,
    build_dimensional_metadata,
)
from core.dimensional import DimensionAxis, NodePosition, FragileEdge, DimensionalMetadata


# =============================================================================
# Fixtures
# =============================================================================

def _make_state(structural=0.7, behavioral=0.6, coverage=0.5, consistency=0.8):
    """Create SharedState with specific confidence values."""
    state = SharedState()
    state.confidence = ConfidenceVector(
        structural=structural,
        behavioral=behavioral,
        coverage=coverage,
        consistency=consistency,
    )
    return state


def _make_blueprint(component_names=None, relationships=None):
    """Create a minimal blueprint dict."""
    if component_names is None:
        component_names = [
            ("UserService", "entity"),
            ("AuthProcess", "process"),
            ("LoginAPI", "interface"),
        ]

    components = [
        {
            "name": name,
            "type": ctype,
            "description": f"{name} description",
            "derived_from": f"Input mentioned {name}",
        }
        for name, ctype in component_names
    ]

    if relationships is None:
        relationships = [
            {"from": "UserService", "to": "AuthProcess", "type": "triggers", "description": "triggers auth"},
            {"from": "AuthProcess", "to": "LoginAPI", "type": "accesses", "description": "accesses api"},
        ]

    return {
        "components": components,
        "relationships": relationships,
        "constraints": [],
        "unresolved": [],
    }


# =============================================================================
# extract_dimensions tests
# =============================================================================

class TestExtractDimensions:
    def test_produces_base_four_axes(self):
        state = _make_state()
        dims = extract_dimensions(state)
        names = {d.name for d in dims}
        assert "structural" in names
        assert "behavioral" in names
        assert "coverage" in names
        assert "consistency" in names

    def test_exploration_depth_matches_confidence(self):
        state = _make_state(structural=0.9, behavioral=0.4)
        dims = extract_dimensions(state)
        dim_map = {d.name: d for d in dims}
        assert dim_map["structural"].exploration_depth == 0.9
        assert dim_map["behavioral"].exploration_depth == 0.4

    def test_silence_zones_for_low_confidence(self):
        state = _make_state(structural=0.1)
        dims = extract_dimensions(state)
        dim_map = {d.name: d for d in dims}
        assert len(dim_map["structural"].silence_zones) > 0

    def test_no_silence_zones_for_high_confidence(self):
        state = _make_state(structural=0.9)
        dims = extract_dimensions(state)
        dim_map = {d.name: d for d in dims}
        assert len(dim_map["structural"].silence_zones) == 0

    def test_conflict_categories_add_axes(self):
        state = _make_state()
        state.conflicts = [
            {"topic": "auth method", "category": "TRADEOFF", "agents": ["Entity", "Process"], "resolved": False, "turn": 3},
            {"topic": "storage", "category": "MISSING_INFO", "agents": ["Entity", "Process"], "resolved": False, "turn": 5},
        ]
        dims = extract_dimensions(state)
        names = {d.name for d in dims}
        assert "conflict:TRADEOFF" in names
        assert "conflict:MISSING_INFO" in names
        # Base 4 + 2 conflict = 6 total
        assert len(dims) == 6

    def test_empty_state(self):
        state = SharedState()
        dims = extract_dimensions(state)
        assert len(dims) == 4  # Always at least the base 4

    def test_derived_from_populated(self):
        state = _make_state()
        dims = extract_dimensions(state)
        for dim in dims:
            assert dim.derived_from != ""

    def test_dimensions_are_frozen(self):
        state = _make_state()
        dims = extract_dimensions(state)
        with pytest.raises(AttributeError):
            dims[0].name = "modified"


# =============================================================================
# extract_node_positions tests
# =============================================================================

class TestExtractNodePositions:
    def test_all_components_positioned(self):
        state = _make_state()
        bp = _make_blueprint()
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        names = {name for name, _ in positions}
        assert "UserService" in names
        assert "AuthProcess" in names
        assert "LoginAPI" in names

    def test_entity_high_structural(self):
        state = _make_state()
        bp = _make_blueprint()
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        pos_map = {name: pos for name, pos in positions}
        entity_pos = pos_map["UserService"]
        assert entity_pos.get_value("structural") == 0.8

    def test_process_high_behavioral(self):
        state = _make_state()
        bp = _make_blueprint()
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        pos_map = {name: pos for name, pos in positions}
        process_pos = pos_map["AuthProcess"]
        assert process_pos.get_value("behavioral") == 0.8

    def test_interface_balanced(self):
        state = _make_state()
        bp = _make_blueprint()
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        pos_map = {name: pos for name, pos in positions}
        iface_pos = pos_map["LoginAPI"]
        assert iface_pos.get_value("structural") == 0.6
        assert iface_pos.get_value("behavioral") == 0.6

    def test_empty_blueprint(self):
        state = _make_state()
        bp = {"components": [], "relationships": []}
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        assert len(positions) == 0

    def test_single_component_no_relationships(self):
        state = _make_state()
        bp = _make_blueprint(
            component_names=[("Solo", "entity")],
            relationships=[],
        )
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        assert len(positions) == 1
        _, pos = positions[0]
        # Coverage should be 0 (no relationships)
        assert pos.get_value("coverage") == 0.0

    def test_positions_are_frozen(self):
        state = _make_state()
        bp = _make_blueprint()
        dims = extract_dimensions(state)
        positions = extract_node_positions(state, bp, dims)
        _, pos = positions[0]
        with pytest.raises(AttributeError):
            pos.confidence = 0.0


# =============================================================================
# extract_fragile_edges tests
# =============================================================================

class TestExtractFragileEdges:
    def test_unresolved_conflicts_produce_edges(self):
        state = _make_state()
        state.conflicts = [
            {"topic": "auth", "agents": ["Entity", "Process"], "resolved": False, "turn": 3},
        ]
        bp = _make_blueprint()
        edges = extract_fragile_edges(state, bp)
        conflict_edges = [e for e in edges if "Unresolved conflict" in e.description]
        assert len(conflict_edges) == 1
        assert conflict_edges[0].drift_risk == "high"

    def test_resolved_conflicts_no_edges(self):
        state = _make_state()
        state.conflicts = [
            {"topic": "auth", "agents": ["Entity", "Process"], "resolved": True, "turn": 3},
        ]
        bp = _make_blueprint()
        edges = extract_fragile_edges(state, bp)
        conflict_edges = [e for e in edges if "Unresolved conflict" in e.description]
        assert len(conflict_edges) == 0

    def test_orphan_components_produce_edges(self):
        state = _make_state()
        bp = _make_blueprint(
            component_names=[("UserService", "entity"), ("Orphan", "entity")],
            relationships=[
                {"from": "UserService", "to": "SomeOther", "type": "triggers", "description": "x"},
            ],
        )
        edges = extract_fragile_edges(state, bp)
        orphan_edges = [e for e in edges if "Orphan component" in e.description]
        assert len(orphan_edges) == 1
        assert orphan_edges[0].affected_nodes == ("Orphan",)

    def test_low_confidence_produces_edge(self):
        state = _make_state(structural=0.1, behavioral=0.1)
        bp = _make_blueprint()
        edges = extract_fragile_edges(state, bp)
        weak_edges = [e for e in edges if "Weak dimensions" in e.description]
        assert len(weak_edges) == 1

    def test_no_fragility_when_all_strong(self):
        state = _make_state(structural=0.9, behavioral=0.9, coverage=0.9, consistency=0.9)
        bp = _make_blueprint()  # All components are connected
        edges = extract_fragile_edges(state, bp)
        # Should have no fragile edges (all connected, high confidence, no conflicts)
        assert len(edges) == 0

    def test_edges_are_frozen(self):
        state = _make_state()
        state.conflicts = [
            {"topic": "x", "agents": ["A", "B"], "resolved": False, "turn": 1},
        ]
        bp = _make_blueprint()
        edges = extract_fragile_edges(state, bp)
        with pytest.raises(AttributeError):
            edges[0].drift_risk = "low"


# =============================================================================
# extract_silence_zones tests
# =============================================================================

class TestExtractSilenceZones:
    def test_low_exploration_creates_zone(self):
        state = _make_state(structural=0.1)
        dims = extract_dimensions(state)
        zones = extract_silence_zones(state, dims)
        assert any("structural" in z for z in zones)

    def test_high_exploration_no_zone(self):
        state = _make_state(structural=0.9, behavioral=0.9, coverage=0.9, consistency=0.9)
        dims = extract_dimensions(state)
        zones = extract_silence_zones(state, dims)
        dim_zones = [z for z in zones if "underexplored" in z]
        assert len(dim_zones) == 0

    def test_unresolved_unknowns_create_zones(self):
        state = _make_state()
        state.unknown = ["What auth method?", "Database choice?"]
        dims = extract_dimensions(state)
        zones = extract_silence_zones(state, dims)
        unknown_zones = [z for z in zones if "Unresolved" in z]
        assert len(unknown_zones) == 2

    def test_persona_blind_spots_create_zones(self):
        state = _make_state()
        state.personas = [
            {"name": "Security Expert", "perspective": "auth", "blind_spots": ["UX"]},
        ]
        dims = extract_dimensions(state)
        zones = extract_silence_zones(state, dims)
        blind_zones = [z for z in zones if "Blind spot" in z]
        assert len(blind_zones) == 1

    def test_empty_state_minimal_zones(self):
        state = SharedState()
        dims = extract_dimensions(state)
        zones = extract_silence_zones(state, dims)
        # All 4 base dimensions have 0.0 depth → 4 underexplored zones
        dim_zones = [z for z in zones if "underexplored" in z]
        assert len(dim_zones) == 4


# =============================================================================
# build_dimensional_metadata tests
# =============================================================================

class TestBuildDimensionalMetadata:
    def test_produces_dimensional_metadata(self):
        state = _make_state()
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        assert isinstance(meta, DimensionalMetadata)

    def test_has_dimensions(self):
        state = _make_state()
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        assert len(meta.dimensions) >= 4

    def test_has_positions_for_all_components(self):
        state = _make_state()
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        assert len(meta.node_positions) == len(bp["components"])

    def test_confidence_trajectory_from_history(self):
        state = _make_state()
        state.confidence_history = [0.1, 0.3, 0.5]
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        assert meta.confidence_trajectory == (0.1, 0.3, 0.5)

    def test_dialogue_depth_from_history(self):
        state = _make_state()
        state.add_message(Message(
            sender="Entity", content="test", message_type=MessageType.PROPOSITION
        ))
        state.add_message(Message(
            sender="Process", content="reply", message_type=MessageType.AGREEMENT
        ))
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        assert meta.dialogue_depth == 2

    def test_dimension_confidence_matches_cv(self):
        state = _make_state(structural=0.7, behavioral=0.6)
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        dim_conf = dict(meta.dimension_confidence)
        assert dim_conf["structural"] == 0.7
        assert dim_conf["behavioral"] == 0.6

    def test_stage_discovery_defaults_to_synthesis(self):
        state = _make_state()
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        # Without pipeline_state, all should be "synthesis"
        stage_map = dict(meta.stage_discovery)
        for comp in bp["components"]:
            assert stage_map.get(comp["name"]) == "synthesis"

    def test_empty_state_and_blueprint(self):
        state = SharedState()
        bp = {"components": [], "relationships": [], "constraints": [], "unresolved": []}
        meta = build_dimensional_metadata(state, bp)
        assert isinstance(meta, DimensionalMetadata)
        assert len(meta.dimensions) == 4
        assert len(meta.node_positions) == 0
        assert meta.dialogue_depth == 0

    def test_result_is_frozen(self):
        state = _make_state()
        bp = _make_blueprint()
        meta = build_dimensional_metadata(state, bp)
        with pytest.raises(AttributeError):
            meta.dialogue_depth = 99
