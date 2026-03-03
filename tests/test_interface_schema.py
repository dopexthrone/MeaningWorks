"""
Tests for core/interface_schema.py — frozen dataclass hierarchy.

Phase B.1.1: Interface Schema
~15 tests — immutability, construction, round-trip serialization.
"""

import json
import pytest

from core.interface_schema import (
    DataFlow,
    InterfaceConstraint,
    InterfaceContract,
    InterfaceMap,
    serialize_interface_map,
    deserialize_interface_map,
)


# =============================================================================
# DataFlow Tests
# =============================================================================

class TestDataFlow:
    def test_construction(self):
        df = DataFlow(
            name="trigger_signal",
            type_hint="Signal",
            direction="A_to_B",
            derived_from="test",
        )
        assert df.name == "trigger_signal"
        assert df.type_hint == "Signal"
        assert df.direction == "A_to_B"

    def test_frozen(self):
        df = DataFlow(name="x", type_hint="int", direction="A_to_B", derived_from="test")
        with pytest.raises(AttributeError):
            df.name = "y"

    def test_equality(self):
        df1 = DataFlow(name="x", type_hint="int", direction="A_to_B", derived_from="test")
        df2 = DataFlow(name="x", type_hint="int", direction="A_to_B", derived_from="test")
        assert df1 == df2

    def test_hashable(self):
        df = DataFlow(name="x", type_hint="int", direction="A_to_B", derived_from="test")
        s = {df}
        assert len(s) == 1


# =============================================================================
# InterfaceConstraint Tests
# =============================================================================

class TestInterfaceConstraint:
    def test_construction(self):
        ic = InterfaceConstraint(
            description="confidence in range [0, 1]",
            constraint_type="range",
            derived_from="test",
        )
        assert ic.constraint_type == "range"

    def test_frozen(self):
        ic = InterfaceConstraint(description="x", constraint_type="custom", derived_from="test")
        with pytest.raises(AttributeError):
            ic.description = "y"


# =============================================================================
# InterfaceContract Tests
# =============================================================================

class TestInterfaceContract:
    def test_construction(self):
        contract = InterfaceContract(
            node_a="Governor Agent",
            node_b="Intent Agent",
            relationship_type="triggers",
            relationship_description="Governor triggers Intent",
            data_flows=(
                DataFlow(name="signal", type_hint="Signal", direction="A_to_B", derived_from="test"),
            ),
            constraints=(),
            fragility=0.3,
            confidence=0.8,
            directionality="A_depends_on_B",
            derived_from="test",
        )
        assert contract.node_a == "Governor Agent"
        assert contract.node_b == "Intent Agent"
        assert len(contract.data_flows) == 1
        assert contract.fragility == 0.3

    def test_frozen(self):
        contract = InterfaceContract(
            node_a="A", node_b="B", relationship_type="triggers",
            relationship_description="", data_flows=(), constraints=(),
            fragility=0.0, confidence=0.0, directionality="mutual",
            derived_from="test",
        )
        with pytest.raises(AttributeError):
            contract.fragility = 0.5

    def test_empty_data_flows(self):
        contract = InterfaceContract(
            node_a="A", node_b="B", relationship_type="triggers",
            relationship_description="", data_flows=(), constraints=(),
            fragility=0.0, confidence=0.0, directionality="mutual",
            derived_from="test",
        )
        assert len(contract.data_flows) == 0


# =============================================================================
# InterfaceMap Tests
# =============================================================================

class TestInterfaceMap:
    def test_construction(self):
        imap = InterfaceMap(
            contracts=(),
            unmatched_relationships=(),
            extraction_confidence=0.0,
            derived_from="test",
        )
        assert len(imap.contracts) == 0
        assert imap.extraction_confidence == 0.0

    def test_frozen(self):
        imap = InterfaceMap(
            contracts=(), unmatched_relationships=(),
            extraction_confidence=0.5, derived_from="test",
        )
        with pytest.raises(AttributeError):
            imap.extraction_confidence = 0.9

    def test_with_contracts(self):
        contract = InterfaceContract(
            node_a="A", node_b="B", relationship_type="triggers",
            relationship_description="test", data_flows=(), constraints=(),
            fragility=0.1, confidence=0.9, directionality="A_depends_on_B",
            derived_from="test",
        )
        imap = InterfaceMap(
            contracts=(contract,),
            unmatched_relationships=("X->Y (unknown)",),
            extraction_confidence=0.7,
            derived_from="test",
        )
        assert len(imap.contracts) == 1
        assert len(imap.unmatched_relationships) == 1


# =============================================================================
# Serialization Round-Trip Tests
# =============================================================================

class TestSerialization:
    def _make_full_map(self):
        df = DataFlow(
            name="trigger_signal", type_hint="Signal",
            direction="A_to_B", derived_from="relationship: A->B (triggers)",
        )
        ic = InterfaceConstraint(
            description="confidence in range [0, 1]",
            constraint_type="range",
            derived_from="constraint: confidence range",
        )
        contract = InterfaceContract(
            node_a="Governor Agent", node_b="Intent Agent",
            relationship_type="triggers",
            relationship_description="Governor triggers Intent Agent",
            data_flows=(df,),
            constraints=(ic,),
            fragility=0.3,
            confidence=0.8,
            directionality="A_depends_on_B",
            derived_from="relationship: Governor Agent->Intent Agent (triggers)",
        )
        return InterfaceMap(
            contracts=(contract,),
            unmatched_relationships=("X->Y (unknown)",),
            extraction_confidence=0.75,
            derived_from="Phase B.1: algorithmic interface extraction",
        )

    def test_serialize_produces_dict(self):
        imap = self._make_full_map()
        result = serialize_interface_map(imap)
        assert isinstance(result, dict)
        assert "contracts" in result
        assert len(result["contracts"]) == 1

    def test_serialize_json_serializable(self):
        imap = self._make_full_map()
        result = serialize_interface_map(imap)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_round_trip(self):
        original = self._make_full_map()
        serialized = serialize_interface_map(original)
        restored = deserialize_interface_map(serialized)
        assert restored == original

    def test_round_trip_preserves_data_flows(self):
        original = self._make_full_map()
        serialized = serialize_interface_map(original)
        restored = deserialize_interface_map(serialized)
        assert restored.contracts[0].data_flows[0].name == "trigger_signal"
        assert restored.contracts[0].data_flows[0].type_hint == "Signal"

    def test_round_trip_preserves_constraints(self):
        original = self._make_full_map()
        serialized = serialize_interface_map(original)
        restored = deserialize_interface_map(serialized)
        assert restored.contracts[0].constraints[0].constraint_type == "range"

    def test_deserialize_empty(self):
        restored = deserialize_interface_map({})
        assert len(restored.contracts) == 0
        assert restored.extraction_confidence == 0.0

    def test_deserialize_partial(self):
        data = {
            "contracts": [{
                "node_a": "A", "node_b": "B",
                "relationship_type": "triggers",
                "fragility": 0.5,
            }],
            "extraction_confidence": 0.6,
        }
        restored = deserialize_interface_map(data)
        assert len(restored.contracts) == 1
        assert restored.contracts[0].fragility == 0.5
