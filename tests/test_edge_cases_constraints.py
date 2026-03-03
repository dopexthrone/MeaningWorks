"""
Phase 17.4: Constraint Contradiction Detection Tests.

Tests for:
- detect_contradictions() — range, enum, polarity
- Contradiction frozen dataclass
- Integration with parsed constraints
- E7004 catalog entry
"""

import pytest
from core.schema import (
    detect_contradictions,
    Contradiction,
    FormalConstraint,
    ConstraintType,
    parse_constraint,
)
from core.error_catalog import get_entry


# =============================================================================
# Helpers
# =============================================================================

def _range_constraint(field, min_val, max_val, desc=None):
    return FormalConstraint(
        constraint_type=ConstraintType.RANGE,
        field=field,
        params={"min": min_val, "max": max_val},
        description=desc or f"{field} in range [{min_val}, {max_val}]",
    )


def _enum_constraint(field, values, desc=None):
    return FormalConstraint(
        constraint_type=ConstraintType.ENUM,
        field=field,
        params={"values": values},
        description=desc or f"{field} one of: {values}",
    )


def _positive_constraint(field, desc=None):
    return FormalConstraint(
        constraint_type=ConstraintType.POSITIVE,
        field=field,
        params={},
        description=desc or f"{field} must be positive",
    )


# =============================================================================
# Range conflicts
# =============================================================================

class TestRangeConflict:
    def test_non_overlapping_ranges_detected(self):
        constraints = [
            _range_constraint("price", 0, 10),
            _range_constraint("price", 20, 30),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 1
        assert result[0].contradiction_type == "range_conflict"
        assert result[0].field == "price"

    def test_overlapping_ranges_not_detected(self):
        constraints = [
            _range_constraint("price", 0, 20),
            _range_constraint("price", 10, 30),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 0

    def test_different_fields_not_detected(self):
        constraints = [
            _range_constraint("price", 0, 10),
            _range_constraint("quantity", 20, 30),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 0

    def test_adjacent_ranges_not_conflict(self):
        """[0, 10] and [10, 20] overlap at 10."""
        constraints = [
            _range_constraint("x", 0, 10),
            _range_constraint("x", 10, 20),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 0


# =============================================================================
# Enum disjoint
# =============================================================================

class TestEnumDisjoint:
    def test_disjoint_enums_detected(self):
        constraints = [
            _enum_constraint("status", ["active", "paused"]),
            _enum_constraint("status", ["deleted", "archived"]),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 1
        assert result[0].contradiction_type == "enum_disjoint"

    def test_overlapping_enums_not_detected(self):
        constraints = [
            _enum_constraint("status", ["active", "paused"]),
            _enum_constraint("status", ["active", "archived"]),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 0


# =============================================================================
# Polarity conflict
# =============================================================================

class TestPolarityConflict:
    def test_positive_plus_negative_range_detected(self):
        constraints = [
            _positive_constraint("price"),
            _range_constraint("price", -100, -1),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 1
        assert result[0].contradiction_type == "polarity_conflict"

    def test_positive_plus_positive_range_no_conflict(self):
        constraints = [
            _positive_constraint("price"),
            _range_constraint("price", 0, 100),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 0


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_empty_constraints_no_contradictions(self):
        assert detect_contradictions([]) == []

    def test_single_constraint_no_contradictions(self):
        assert detect_contradictions([_range_constraint("x", 0, 10)]) == []

    def test_works_with_dict_constraints(self):
        """Should also accept raw dict constraints from blueprint."""
        constraints = [
            {"description": "price in range [0, 10]", "applies_to": ["Product"]},
            {"description": "price in range [20, 30]", "applies_to": ["Product"]},
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 1
        assert result[0].contradiction_type == "range_conflict"


# =============================================================================
# Contradiction dataclass
# =============================================================================

class TestContradiction:
    def test_frozen(self):
        c = Contradiction(
            constraint_a="a", constraint_b="b", field="x",
            contradiction_type="range_conflict", description="test",
        )
        with pytest.raises(AttributeError):
            c.field = "y"

    def test_description_human_readable(self):
        constraints = [
            _range_constraint("price", 0, 10),
            _range_constraint("price", 20, 30),
        ]
        result = detect_contradictions(constraints)
        assert len(result) == 1
        desc = result[0].description
        assert "price" in desc
        assert "overlap" in desc.lower() or "not overlap" in desc.lower()


# =============================================================================
# E7004 catalog entry
# =============================================================================

class TestE7004Catalog:
    def test_entry_exists(self):
        entry = get_entry("E7004")
        assert entry is not None
        assert entry.code == "E7004"
        assert "contradict" in entry.title.lower()
