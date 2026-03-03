"""Tests for structured grid-to-synthesis provenance (Build 6)."""

import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill
from kernel.nav import grid_to_structured
from core.verification import (
    validate_provenance_refs,
    provenance_integrity_ratio,
    score_traceability,
)


class TestGridToStructured:
    """grid_to_structured() exports lossless cell data."""

    def _make_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=(INTENT_CONTRACT,),
             connections=("EXC.BHV.APP.HOW.SFT",))
        fill(grid, "EXC.BHV.APP.HOW.SFT", "auth", "auth flow",
             0.85, source=(INTENT_CONTRACT,))
        return grid

    def test_returns_non_empty_list(self):
        grid = self._make_grid()
        result = grid_to_structured(grid)
        assert isinstance(result, list)
        assert len(result) >= 2  # at least 2 filled cells

    def test_cell_has_all_fields(self):
        grid = self._make_grid()
        result = grid_to_structured(grid)
        cell = result[0]
        required_fields = [
            "postcode", "primitive", "content", "fill_state",
            "confidence", "source", "connections", "parent",
            "revision_count", "layer", "concern", "scope",
            "dimension", "domain",
        ]
        for field in required_fields:
            assert field in cell, f"Missing field: {field}"

    def test_excludes_empty_cells(self):
        grid = self._make_grid()
        result = grid_to_structured(grid)
        for cell in result:
            assert cell["fill_state"] != "E"

    def test_preserves_connections(self):
        grid = self._make_grid()
        result = grid_to_structured(grid)
        user_cell = [c for c in result if c["primitive"] == "user"]
        assert len(user_cell) == 1
        assert "EXC.BHV.APP.HOW.SFT" in user_cell[0]["connections"]

    def test_sorted_by_postcode(self):
        grid = self._make_grid()
        result = grid_to_structured(grid)
        postcodes = [c["postcode"] for c in result]
        assert postcodes == sorted(postcodes)


class TestValidateProvenanceRefs:
    """validate_provenance_refs() parses and validates grid: refs."""

    def test_validated_refs(self):
        components = [
            {"name": "Auth", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"},
            {"name": "Flow", "derived_from": "grid:EXC.BHV.APP.HOW.SFT"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT", "EXC.BHV.APP.HOW.SFT"]
        result = validate_provenance_refs(components, postcodes)
        assert result["validated"] == 2
        assert result["text_only"] == 0
        assert result["invalid_refs"] == 0
        assert result["missing"] == 0

    def test_text_only_refs(self):
        components = [
            {"name": "Auth", "derived_from": "user described authentication"},
        ]
        result = validate_provenance_refs(components, ["STR.ENT.CMP.WHAT.SFT"])
        assert result["validated"] == 0
        assert result["text_only"] == 1

    def test_invalid_refs(self):
        components = [
            {"name": "Auth", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"},
        ]
        result = validate_provenance_refs(components, ["EXC.BHV.APP.HOW.SFT"])
        assert result["validated"] == 0
        assert result["invalid_refs"] == 1

    def test_missing_derived_from(self):
        components = [
            {"name": "Auth", "derived_from": ""},
            {"name": "Flow"},
        ]
        result = validate_provenance_refs(components, ["STR.ENT.CMP.WHAT.SFT"])
        assert result["missing"] == 2

    def test_multiple_refs_in_one_field(self):
        components = [
            {"name": "Auth", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT|grid:EXC.BHV.APP.HOW.SFT"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT", "EXC.BHV.APP.HOW.SFT"]
        result = validate_provenance_refs(components, postcodes)
        assert result["validated"] == 1

    def test_mixed_valid_and_invalid(self):
        components = [
            {"name": "Auth", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT|grid:FAKE.BAD.REF.WHY.SFT"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT"]
        result = validate_provenance_refs(components, postcodes)
        assert result["invalid_refs"] == 1  # one ref is bad


class TestProvenanceIntegrityRatio:
    """provenance_integrity_ratio() computes validated fraction."""

    def test_all_validated(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"},
            {"name": "B", "derived_from": "grid:EXC.BHV.APP.HOW.SFT"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT", "EXC.BHV.APP.HOW.SFT"]
        assert provenance_integrity_ratio(components, postcodes) == 1.0

    def test_none_validated(self):
        components = [
            {"name": "A", "derived_from": "user said so"},
        ]
        postcodes = ["STR.ENT.CMP.WHAT.SFT"]
        assert provenance_integrity_ratio(components, postcodes) == 0.0

    def test_no_grid_returns_zero(self):
        components = [{"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT"}]
        assert provenance_integrity_ratio(components, []) == 0.0

    def test_empty_components_returns_zero(self):
        assert provenance_integrity_ratio([], ["STR.ENT.CMP.WHAT.SFT"]) == 0.0


class TestScoreTraceabilityWithGrid:
    """score_traceability() gives bonus for validated grid refs."""

    def test_grid_bonus_increases_score(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT some text"},
        ]
        # Without grid postcodes
        score_no_grid = score_traceability(components).score
        # With grid postcodes
        score_with_grid = score_traceability(
            components, grid_postcodes=["STR.ENT.CMP.WHAT.SFT"]
        ).score
        assert score_with_grid >= score_no_grid

    def test_backward_compat_no_grid(self):
        """Calling without grid_postcodes still works."""
        components = [
            {"name": "A", "derived_from": "user described this component in detail"},
        ]
        result = score_traceability(components)
        assert result.score > 0
        assert "grid_validated" not in result.details
