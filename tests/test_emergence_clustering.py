"""Tests for emergence semantic clustering (Build 2a)."""

import pytest
from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid
from kernel.navigator import detect_emergence, _semantic_group_primitives
from kernel._text_utils import semantic_jaccard


class TestSemanticJaccard:
    """Stemmed-token Jaccard similarity."""

    def test_identical_strings(self):
        assert semantic_jaccard("auth service", "auth service") == 1.0

    def test_synonym_overlap(self):
        """Similar strings should have non-zero similarity."""
        score = semantic_jaccard("auth-service", "authentication-service")
        assert score > 0.3

    def test_empty_strings(self):
        assert semantic_jaccard("", "something") == 0.0
        assert semantic_jaccard("something", "") == 0.0

    def test_unrelated_strings(self):
        score = semantic_jaccard("user login", "database migration")
        assert score < 0.3


class TestSemanticGroupPrimitives:
    """Group primitives by stemmed-token Jaccard > 0.5."""

    def _make_cell(self, postcode_key, primitive):
        return Cell(
            postcode=parse_postcode(postcode_key),
            primitive=primitive,
            content="test",
            fill=FillState.F,
            confidence=0.9,
            source=("human:test",),
        )

    def test_exact_matches_grouped(self):
        """Identical primitives are grouped together."""
        cells = [
            self._make_cell("STR.ENT.CMP.WHAT.SFT", "auth-service"),
            self._make_cell("STA.BHV.CMP.HOW.SFT", "auth-service"),
            self._make_cell("ORG.FNC.CMP.WHAT.SFT", "data-store"),
        ]
        groups = _semantic_group_primitives(cells)
        assert any(len(v) == 2 for v in groups.values())

    def test_similar_primitives_grouped(self):
        """auth-service and authentication-service should group together."""
        cells = [
            self._make_cell("STR.ENT.CMP.WHAT.SFT", "auth-service"),
            self._make_cell("STA.BHV.CMP.HOW.SFT", "authentication-service"),
        ]
        groups = _semantic_group_primitives(cells)
        # Should be in same group
        all_members = [v for v in groups.values()]
        assert any(len(v) == 2 for v in all_members)

    def test_unrelated_not_grouped(self):
        """Unrelated primitives stay separate."""
        cells = [
            self._make_cell("STR.ENT.CMP.WHAT.SFT", "user-login"),
            self._make_cell("STA.BHV.CMP.HOW.SFT", "database-migration"),
        ]
        groups = _semantic_group_primitives(cells)
        assert all(len(v) == 1 for v in groups.values())


class TestEmergenceSemantic:
    """detect_emergence uses semantic grouping."""

    def _make_grid_with_similar_primitives(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        from kernel.grid import INTENT_CONTRACT
        from kernel.ops import fill
        # 5+ filled cells needed for emergence detection
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "auth-service", "handles auth",
             0.9, source=(INTENT_CONTRACT,))
        fill(grid, "STA.BHV.CMP.HOW.SFT", "authentication-service", "auth flow",
             0.9, source=(INTENT_CONTRACT,))
        fill(grid, "ORG.FNC.DOM.WHAT.SFT", "data-store", "persists data",
             0.9, source=(INTENT_CONTRACT,))
        fill(grid, "COG.REL.APP.WHY.SFT", "decision-engine", "makes decisions",
             0.9, source=(INTENT_CONTRACT,))
        fill(grid, "AGN.AGT.FET.WHO.SFT", "agent-runner", "runs agents",
             0.9, source=(INTENT_CONTRACT,))
        return grid

    def test_semantic_grouping_detects_similar(self):
        """Similar primitives across layers trigger emergence."""
        grid = self._make_grid_with_similar_primitives()
        signals = detect_emergence(grid)
        repeated = [s for s in signals if s.signal_type == "repeated_primitive"]
        # auth-service and authentication-service should group
        assert len(repeated) >= 1
        assert any("auth" in s.primitive.lower() for s in repeated)

    def test_unique_emergence_postcodes(self):
        """Multiple emergence signals get unique postcodes."""
        grid = self._make_grid_with_similar_primitives()
        # Add another similar pair
        from kernel.ops import fill
        fill(grid, "NET.FLW.CMP.HOW.SFT", "data-store", "stores data",
             0.9, source=("human:test",))
        signals = detect_emergence(grid)
        from kernel.agents import _emergence_target
        targets = set()
        for i, s in enumerate(signals):
            target = _emergence_target(s, index=i)
            targets.add(target)
        # All targets should be unique when index is passed
        assert len(targets) == len(signals) or len(signals) <= 1
