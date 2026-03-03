"""
tests/test_kernel.py — Comprehensive tests for the semantic map kernel.

Tests all 4 primitives: Cell, Grid, Fill, Connect.
Tests all 5 axioms: PROVENANCE, DESCENT, FEEDBACK, EMERGENCE, CONSTRAINT.
Adversarial cases included.
"""

import pytest
from kernel.cell import (
    Cell,
    FillState,
    Postcode,
    parse_postcode,
    LAYERS,
    CONCERNS,
    SCOPE_SET,
    DIMENSIONS,
    SCOPE_DEPTH,
)
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import (
    fill,
    connect,
    FillResult,
    FillStatus,
    ConnectResult,
    ConnectStatus,
)


# ============================================================
# Cell
# ============================================================

class TestPostcode:
    """Postcode parsing and validation."""

    def test_parse_valid(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        assert pc.layer == "INT"
        assert pc.concern == "SEM"
        assert pc.scope == "ECO"
        assert pc.dimension == "WHY"
        assert pc.domain == "ORG"

    def test_parse_roundtrip(self):
        raw = "AGN.AGT.CMP.WHO.SFT"
        pc = parse_postcode(raw)
        assert pc.key == raw
        assert str(pc) == raw

    def test_parse_wrong_part_count(self):
        with pytest.raises(ValueError, match="exactly 5 axes"):
            parse_postcode("INT.SEM.ECO")

    def test_parse_too_many_parts(self):
        with pytest.raises(ValueError, match="exactly 5 axes"):
            parse_postcode("INT.SEM.ECO.WHY.ORG.EXTRA")

    def test_parse_unknown_layer(self):
        with pytest.raises(ValueError, match="Unknown layer"):
            parse_postcode("ZZZ.SEM.ECO.WHY.ORG")

    def test_parse_unknown_concern(self):
        with pytest.raises(ValueError, match="Unknown concern"):
            parse_postcode("INT.ZZZ.ECO.WHY.ORG")

    def test_parse_unknown_scope(self):
        with pytest.raises(ValueError, match="Unknown scope"):
            parse_postcode("INT.SEM.ZZZ.WHY.ORG")

    def test_parse_unknown_dimension(self):
        with pytest.raises(ValueError, match="Unknown dimension"):
            parse_postcode("INT.SEM.ECO.ZZZ.ORG")

    def test_parse_invalid_domain_lowercase(self):
        with pytest.raises(ValueError, match="Domain must be"):
            parse_postcode("INT.SEM.ECO.WHY.org")

    def test_parse_invalid_domain_too_long(self):
        with pytest.raises(ValueError, match="Domain must be"):
            parse_postcode("INT.SEM.ECO.WHY.TOOLONG")

    def test_parse_invalid_domain_single_char(self):
        with pytest.raises(ValueError, match="Domain must be"):
            parse_postcode("INT.SEM.ECO.WHY.X")

    def test_domain_extensible(self):
        """Domains not in the initial set should still parse."""
        pc = parse_postcode("INT.SEM.ECO.WHY.XYZ")
        assert pc.domain == "XYZ"

    def test_depth_eco_is_zero(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        assert pc.depth == 0

    def test_depth_val_is_nine(self):
        pc = parse_postcode("INT.SEM.VAL.WHY.ORG")
        assert pc.depth == 9

    def test_depth_cmp_is_four(self):
        pc = parse_postcode("INT.SEM.CMP.WHY.ORG")
        assert pc.depth == 4

    def test_parent_scope_of_eco_is_none(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        assert pc.parent_scope() is None

    def test_parent_scope_of_app_is_eco(self):
        pc = parse_postcode("INT.SEM.APP.WHY.ORG")
        assert pc.parent_scope() == "ECO"

    def test_child_scope_of_val_is_none(self):
        pc = parse_postcode("INT.SEM.VAL.WHY.ORG")
        assert pc.child_scope() is None

    def test_child_scope_of_eco_is_app(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        assert pc.child_scope() == "APP"

    def test_frozen(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        with pytest.raises(AttributeError):
            pc.layer = "SEM"


class TestCell:
    """Cell construction and properties."""

    def test_empty_cell(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test")
        assert cell.fill == FillState.E
        assert cell.confidence == 0.0
        assert cell.is_empty
        assert not cell.is_filled

    def test_filled_cell(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(
            postcode=pc,
            primitive="founding_problem",
            content="The gap between expertise and implementation",
            fill=FillState.F,
            confidence=0.99,
            source=(INTENT_CONTRACT,),
        )
        assert cell.is_filled
        assert not cell.is_empty
        assert cell.depth == 0

    def test_blocked_cell(self):
        pc = parse_postcode("RES.MET.ECO.HOW_MUCH.ECN")
        cell = Cell(postcode=pc, primitive="business_model", fill=FillState.B)
        assert cell.is_blocked
        assert not cell.is_filled

    def test_quarantined_cell(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="bad", fill=FillState.Q)
        assert cell.is_quarantined

    def test_candidate_cell(self):
        pc = parse_postcode("EMG.CND.ECO.WHAT.COG")
        cell = Cell(postcode=pc, primitive="discovery", fill=FillState.C)
        assert cell.is_candidate

    def test_confidence_clamp_high(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test", confidence=1.5)
        assert cell.confidence == 1.0

    def test_confidence_clamp_low(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test", confidence=-0.5)
        assert cell.confidence == 0.0

    def test_frozen(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test")
        with pytest.raises(AttributeError):
            cell.content = "mutated"

    def test_nav_line_empty(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test")
        line = cell.nav_line()
        assert "INT.SEM.ECO.WHY.ORG" in line
        assert "E 0.00" in line
        assert "test" in line

    def test_nav_line_with_connections(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(
            postcode=pc,
            primitive="test",
            fill=FillState.F,
            confidence=0.99,
            connections=("SEM.SEM.ECO.WHAT.SFT", "IDN.ACT.ECO.WHO.ORG"),
        )
        line = cell.nav_line()
        assert "->" in line
        assert "SEM.SEM.ECO.WHAT.SFT" in line

    def test_connections_are_tuple(self):
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test", connections=("a", "b"))
        assert isinstance(cell.connections, tuple)


# ============================================================
# Grid
# ============================================================

class TestGrid:
    """Grid construction, activation, and queries."""

    def test_empty_grid(self):
        g = Grid()
        assert g.total_cells == 0
        assert g.fill_rate == 0.0
        assert g.root is None

    def test_set_intent(self):
        g = Grid()
        root = g.set_intent(
            "Build a tattoo booking system",
            "INT.SEM.ECO.WHY.ORG",
            "founding_problem",
        )
        assert g.root == "INT.SEM.ECO.WHY.ORG"
        assert root.fill == FillState.F
        assert root.confidence == 1.0
        assert g.total_cells == 1
        assert g.is_layer_active("INT")

    def test_put_and_get(self):
        g = Grid()
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        cell = Cell(postcode=pc, primitive="test")
        g.put(cell)
        assert g.has("INT.SEM.ECO.WHY.ORG")
        got = g.get("INT.SEM.ECO.WHY.ORG")
        assert got is cell

    def test_get_missing(self):
        g = Grid()
        assert g.get("INT.SEM.ECO.WHY.ORG") is None

    def test_layer_activation(self):
        g = Grid()
        assert not g.is_layer_active("INT")
        g.activate_layer("INT", "SEM", "WHY", "ORG")
        assert g.is_layer_active("INT")
        assert g.total_cells == 1  # root cell created

    def test_layer_activation_idempotent(self):
        g = Grid()
        c1 = g.activate_layer("INT", "SEM", "WHY", "ORG")
        c2 = g.activate_layer("INT", "SEM", "WHY", "ORG")
        assert c1.postcode.key == c2.postcode.key

    def test_cells_in_layer(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc2 = parse_postcode("INT.SEM.ECO.WHAT.ORG")
        g.put(Cell(postcode=pc2, primitive="mission"))
        assert len(g.cells_in_layer("INT")) == 2
        assert len(g.cells_in_layer("SEM")) == 0

    def test_filled_cells(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc2 = parse_postcode("INT.SEM.ECO.WHAT.ORG")
        g.put(Cell(postcode=pc2, primitive="empty"))
        assert len(g.filled_cells()) == 1

    def test_empty_cells(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc2 = parse_postcode("INT.SEM.ECO.WHAT.ORG")
        g.put(Cell(postcode=pc2, primitive="empty"))
        assert len(g.empty_cells()) == 1

    def test_stats(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc2 = parse_postcode("INT.SEM.ECO.WHAT.ORG")
        g.put(Cell(postcode=pc2, primitive="empty"))
        s = g.stats()
        assert s["F"] == 1
        assert s["E"] == 1

    def test_fill_rate(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc2 = parse_postcode("INT.SEM.ECO.WHAT.ORG")
        g.put(Cell(postcode=pc2, primitive="empty"))
        assert g.fill_rate == 0.5

    def test_nav_output(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        nav = g.nav()
        assert "# INT" in nav
        assert "INT.SEM.ECO.WHY.ORG" in nav

    def test_unfilled_connections(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        pc = parse_postcode("INT.SEM.ECO.WHY.ORG")
        filled = Cell(
            postcode=pc,
            primitive="root",
            content="Test",
            fill=FillState.F,
            confidence=1.0,
            connections=("SEM.SEM.ECO.WHAT.SFT",),
            source=(INTENT_CONTRACT,),
        )
        g.put(filled)
        # Create target as empty
        target_pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        g.put(Cell(postcode=target_pc, primitive="target"))
        unfilled = g.unfilled_connections()
        assert "SEM.SEM.ECO.WHAT.SFT" in unfilled

    def test_orphan_cells(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Add orphan — filled but no connections, no parent, not connected to
        orphan_pc = parse_postcode("RES.MET.ECO.HOW_MUCH.ECN")
        orphan = Cell(
            postcode=orphan_pc,
            primitive="orphan",
            content="Orphaned",
            fill=FillState.F,
            confidence=0.9,
            source=(INTENT_CONTRACT,),
        )
        g.put(orphan)
        orphans = g.orphan_cells()
        assert len(orphans) == 1
        assert orphans[0].primitive == "orphan"


# ============================================================
# Fill — Axiom Enforcement
# ============================================================

class TestFillAX1Provenance:
    """AX1: every fill must trace to a source."""

    def test_fill_with_intent_contract(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Make it possible", 0.95,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.OK
        assert result.cell.fill == FillState.F

    def test_fill_with_parent_ref(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Make it possible", 0.95,
            parent="INT.SEM.ECO.WHY.ORG",
            source=("INT.SEM.ECO.WHY.ORG",),
        )
        assert result.status == FillStatus.OK

    def test_fill_with_human_source(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Make it possible", 0.95,
            source=("human:alex_session",),
        )
        assert result.status == FillStatus.OK

    def test_fill_with_contract_source(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Make it possible", 0.95,
            source=("contract:MTH-ORG-001",),
        )
        assert result.status == FillStatus.OK

    def test_fill_no_provenance_quarantines(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Hallucinated content", 0.95,
        )
        assert result.status == FillStatus.QUARANTINED
        assert result.violation == "AX1_PROVENANCE"
        assert result.cell.fill == FillState.Q

    def test_fill_with_unfilled_source_quarantines(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Source references a cell that exists but is empty
        empty_pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        g.put(Cell(postcode=empty_pc, primitive="empty_source"))
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Content", 0.95,
            source=("SEM.SEM.ECO.WHAT.SFT",),
        )
        assert result.status == FillStatus.QUARANTINED
        assert result.violation == "AX1_PROVENANCE"

    def test_fill_with_filled_source_ok(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Fill a source cell first
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Semantic compiler", 0.95,
             source=(INTENT_CONTRACT,))
        # Now fill with that as source
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Content", 0.95,
            source=("SEM.SEM.ECO.WHAT.SFT",),
        )
        assert result.status == FillStatus.OK


class TestFillAX2Descent:
    """AX2: parent must be filled before child."""

    def test_fill_with_filled_parent(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.APP.WHY.ORG", "child",
            "Child content", 0.90,
            parent="INT.SEM.ECO.WHY.ORG",
            source=("INT.SEM.ECO.WHY.ORG",),
        )
        assert result.status == FillStatus.OK

    def test_fill_with_unfilled_parent_quarantines(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Create but don't fill parent
        parent_pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        g.put(Cell(postcode=parent_pc, primitive="unfilled_parent"))
        result = fill(
            g, "SEM.SEM.APP.WHAT.SFT", "child",
            "Child content", 0.90,
            parent="SEM.SEM.ECO.WHAT.SFT",
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.QUARANTINED
        assert result.violation == "AX2_DESCENT"

    def test_fill_with_intent_contract_parent(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Content", 0.95,
            parent=INTENT_CONTRACT,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.OK

    def test_fill_no_parent_ok_if_source_valid(self):
        """Cells without parent are OK if source is valid."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(
            g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
            "Content", 0.95,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.OK


class TestFillAX3Feedback:
    """AX3: re-fill preserves history."""

    def test_refill_creates_revision(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # First fill
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Original content", 0.90,
             source=(INTENT_CONTRACT,))
        # Second fill (revision)
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Revised content", 0.95,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.REVISED
        assert len(result.cell.revisions) == 1
        assert result.cell.revisions[0] == ("Original content", 0.90)
        assert result.cell.content == "Revised content"

    def test_multiple_revisions_accumulate(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "V1", 0.80, source=(INTENT_CONTRACT,))
        fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "V2", 0.85, source=(INTENT_CONTRACT,))
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "V3", 0.90, source=(INTENT_CONTRACT,))
        assert result.status == FillStatus.REVISED
        assert len(result.cell.revisions) == 2
        assert result.cell.revisions[0] == ("V1", 0.80)
        assert result.cell.revisions[1] == ("V2", 0.85)


class TestFillAX5Constraint:
    """AX5: blocked cells reject fill."""

    def test_fill_blocked_cell_rejected(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Create blocked cell
        blocked_pc = parse_postcode("RES.MET.ECO.HOW_MUCH.ECN")
        g.put(Cell(postcode=blocked_pc, primitive="blocked", fill=FillState.B))
        result = fill(
            g, "RES.MET.ECO.HOW_MUCH.ECN", "business_model",
            "SaaS subscription", 0.90,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.BLOCKED
        assert result.violation == "AX5_CONSTRAINT"


class TestFillConfidence:
    """Confidence determines fill state."""

    def test_high_confidence_fills(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", 0.95, source=(INTENT_CONTRACT,))
        assert result.cell.fill == FillState.F

    def test_threshold_confidence_fills(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", 0.85, source=(INTENT_CONTRACT,))
        assert result.cell.fill == FillState.F

    def test_low_confidence_partial(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", 0.50, source=(INTENT_CONTRACT,))
        assert result.cell.fill == FillState.P

    def test_zero_confidence_empty(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", 0.0, source=(INTENT_CONTRACT,))
        assert result.cell.fill == FillState.E

    def test_confidence_clamped_above_one(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", 1.5, source=(INTENT_CONTRACT,))
        assert result.cell.confidence == 1.0

    def test_confidence_clamped_below_zero(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "INT.SEM.ECO.WHAT.ORG", "m", "C", -0.5, source=(INTENT_CONTRACT,))
        assert result.cell.confidence == 0.0


class TestFillLayerActivation:
    """Fill connections trigger layer activation."""

    def test_connection_activates_layer(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        assert not g.is_layer_active("SEM")
        result = fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Content", 0.95,
            connections=("SEM.SEM.ECO.WHAT.SFT",),
            source=(INTENT_CONTRACT,),
        )
        assert g.is_layer_active("SEM")
        assert "SEM" in result.activated_layers

    def test_connection_creates_target_cell(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        fill(
            g, "INT.SEM.ECO.WHAT.ORG", "mission",
            "Content", 0.95,
            connections=("SEM.SEM.ECO.WHAT.SFT",),
            source=(INTENT_CONTRACT,),
        )
        target = g.get("SEM.SEM.ECO.WHAT.SFT")
        assert target is not None
        assert target.is_empty

    def test_already_active_layer_no_duplicate(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        fill(g, "INT.SEM.ECO.WHAT.ORG", "m1", "C1", 0.90,
             connections=("SEM.SEM.ECO.WHAT.SFT",), source=(INTENT_CONTRACT,))
        result = fill(g, "INT.SEM.ECO.HOW.ORG", "m2", "C2", 0.90,
                      connections=("SEM.SEM.ECO.WHAT.SFT",), source=(INTENT_CONTRACT,))
        # SEM already active — should not be in activated_layers
        assert "SEM" not in result.activated_layers


class TestFillCandidatePromotion:
    """Filling a candidate cell promotes it."""

    def test_promote_candidate(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        # Place candidate
        cand_pc = parse_postcode("EMG.CND.ECO.WHAT.COG")
        g.put(Cell(postcode=cand_pc, primitive="discovery", fill=FillState.C, confidence=0.7))
        # Fill it (promote)
        result = fill(
            g, "EMG.CND.ECO.WHAT.COG", "discovery",
            "Validated pattern", 0.90,
            source=(INTENT_CONTRACT,),
        )
        assert result.status == FillStatus.PROMOTED
        assert result.cell.fill == FillState.F


# ============================================================
# Connect
# ============================================================

class TestConnect:
    """Connect operation — wiring cells."""

    def test_connect_two_existing_cells(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler", "C", 0.95, source=(INTENT_CONTRACT,))
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert result.status == ConnectStatus.OK
        assert result.from_cell is not None
        assert "SEM.SEM.ECO.WHAT.SFT" in result.from_cell.connections

    def test_connect_duplicate_idempotent(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler", "C", 0.95, source=(INTENT_CONTRACT,))
        connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert result.status == ConnectStatus.DUPLICATE

    def test_connect_missing_source(self):
        g = Grid()
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert result.status == ConnectStatus.MISSING_SOURCE

    def test_connect_missing_target_creates_empty(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert result.status in (ConnectStatus.MISSING_TARGET, ConnectStatus.LAYER_ACTIVATED)
        assert g.has("SEM.SEM.ECO.WHAT.SFT")
        target = g.get("SEM.SEM.ECO.WHAT.SFT")
        assert target.is_empty

    def test_connect_activates_layer(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        assert not g.is_layer_active("SEM")
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert g.is_layer_active("SEM")
        assert result.status == ConnectStatus.LAYER_ACTIVATED
        assert result.activated_layer == "SEM"

    def test_connect_invalid_target_postcode(self):
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "not.a.valid.postcode")
        assert result.status == ConnectStatus.MISSING_TARGET


# ============================================================
# Integration — Full Pipeline Simulation
# ============================================================

class TestKernelIntegration:
    """End-to-end test: simulate a mini compilation."""

    def test_mini_compilation(self):
        """Simulate: intent → fill 3 layers → connect → check nav."""
        g = Grid()

        # Step 1: Set intent
        root = g.set_intent(
            "Build a tattoo booking system",
            "INT.SEM.ECO.WHY.ORG",
            "founding_problem",
        )
        assert root.fill == FillState.F

        # Step 2: Fill semantic layer
        r1 = fill(
            g, "SEM.SEM.ECO.WHAT.SFT", "booking_system",
            "Online booking system for tattoo studios",
            0.95,
            connections=("STR.ENT.ECO.WHAT.SFT",),
            source=(INTENT_CONTRACT,),
        )
        assert r1.status == FillStatus.OK
        assert g.is_layer_active("STR")  # activated by connection

        # Step 3: Fill structure layer
        r2 = fill(
            g, "STR.ENT.ECO.WHAT.SFT", "product_booking",
            "Booking product with slot management",
            0.90,
            parent="SEM.SEM.ECO.WHAT.SFT",
            source=("SEM.SEM.ECO.WHAT.SFT",),
        )
        assert r2.status == FillStatus.OK

        # Step 4: Connect intent → semantic
        c1 = connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        assert c1.status == ConnectStatus.OK

        # Step 5: Check grid state
        assert g.total_cells >= 3
        assert g.fill_rate > 0
        assert len(g.filled_cells()) == 3

        # Step 6: Check navigation output
        nav = g.nav()
        assert "# INT" in nav
        assert "# SEM" in nav
        assert "# STR" in nav
        assert "booking_system" in nav

    def test_quarantine_does_not_leak(self):
        """Quarantined cells should not appear as filled."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")

        # Try to fill without provenance
        result = fill(g, "SEM.SEM.ECO.WHAT.SFT", "bad", "Hallucinated", 0.99)
        assert result.status == FillStatus.QUARANTINED

        # Filled cells should not include it
        assert len(g.filled_cells()) == 1  # only intent root
        assert len(g.quarantined_cells()) == 1

    def test_descent_ordering(self):
        """ECO fills before APP, APP fills before DOM."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")

        # Fill ECO
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "eco_node",
             "Ecosystem level", 0.95, source=(INTENT_CONTRACT,))

        # Fill APP (parent = ECO) — should work
        r_app = fill(g, "SEM.SEM.APP.WHAT.SFT", "app_node",
                      "App level", 0.90,
                      parent="SEM.SEM.ECO.WHAT.SFT",
                      source=("SEM.SEM.ECO.WHAT.SFT",))
        assert r_app.status == FillStatus.OK

        # Fill DOM (parent = APP) — should work
        r_dom = fill(g, "SEM.SEM.DOM.WHAT.SFT", "dom_node",
                      "Domain level", 0.85,
                      parent="SEM.SEM.APP.WHAT.SFT",
                      source=("SEM.SEM.APP.WHAT.SFT",))
        assert r_dom.status == FillStatus.OK

    def test_blocked_then_unblocked(self):
        """Blocked cell rejects fill, then after unblocking, accepts fill."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")

        # Create blocked cell
        blocked_pc = parse_postcode("RES.MET.ECO.HOW_MUCH.ECN")
        g.put(Cell(postcode=blocked_pc, primitive="biz", fill=FillState.B))

        # Try to fill — rejected
        r1 = fill(g, "RES.MET.ECO.HOW_MUCH.ECN", "biz",
                  "SaaS model", 0.90, source=(INTENT_CONTRACT,))
        assert r1.status == FillStatus.BLOCKED

        # Unblock by replacing with empty
        g.put(Cell(postcode=blocked_pc, primitive="biz", fill=FillState.E))

        # Now fill — should work
        r2 = fill(g, "RES.MET.ECO.HOW_MUCH.ECN", "biz",
                  "SaaS model", 0.90, source=(INTENT_CONTRACT,))
        assert r2.status == FillStatus.OK

    def test_sparse_input_mostly_empty(self):
        """Sparse input should NOT produce hallucinated fills."""
        g = Grid()
        g.set_intent("Build me a thing", "INT.SEM.ECO.WHY.ORG", "root")

        # Only the intent root should be filled — nothing hallucinated
        assert len(g.filled_cells()) == 1
        assert g.total_cells == 1  # no extra cells materialized

    def test_self_referential_grid(self):
        """The grid can describe itself (fixed-point test)."""
        g = Grid()
        g.set_intent(
            "A coordinate space where every concept has a postcode",
            "INT.SEM.ECO.WHY.SFT",
            "semantic_map_intent",
        )

        # Fill meta layer
        fill(g, "MET.INT.ECO.WHY.ORG", "meta_self",
             "The map describes itself",
             0.99,
             connections=("INT.SEM.ECO.WHY.SFT",),
             source=(INTENT_CONTRACT,))

        # Meta connects to intent — circular reference is valid
        connect(g, "MET.INT.ECO.WHY.ORG", "INT.SEM.ECO.WHY.SFT")

        # Both cells exist and are filled
        assert g.get("MET.INT.ECO.WHY.ORG").is_filled
        assert g.get("INT.SEM.ECO.WHY.SFT").is_filled

        # Grid has 2 filled cells + possibly activated empty cells
        assert len(g.filled_cells()) >= 2


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    """Corner cases and adversarial inputs."""

    def test_fill_nonexistent_postcode_creates_cell(self):
        """Filling a postcode that doesn't exist in the grid creates it."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = fill(g, "SEM.SEM.ECO.WHAT.SFT", "new",
                      "Content", 0.95, source=(INTENT_CONTRACT,))
        assert result.status == FillStatus.OK
        assert g.has("SEM.SEM.ECO.WHAT.SFT")

    def test_fill_invalid_postcode_raises(self):
        g = Grid()
        with pytest.raises(ValueError):
            fill(g, "NOT.VALID", "bad", "Content", 0.95)

    def test_empty_grid_nav(self):
        g = Grid()
        nav = g.nav()
        assert nav.strip() == ""

    def test_connect_to_self(self):
        """Self-connection should work (a cell can reference itself)."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        result = connect(g, "INT.SEM.ECO.WHY.ORG", "INT.SEM.ECO.WHY.ORG")
        assert result.status == ConnectStatus.OK

    def test_many_connections(self):
        """A cell can have many connections."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")
        targets = [
            f"SEM.SEM.ECO.{dim}.SFT"
            for dim in ["WHAT", "HOW", "WHY", "WHO", "WHEN"]
        ]
        for t in targets:
            connect(g, "INT.SEM.ECO.WHY.ORG", t)
        root = g.get("INT.SEM.ECO.WHY.ORG")
        assert len(root.connections) == 5

    def test_grid_stats_all_states(self):
        """Grid stats should count all fill states."""
        g = Grid()
        g.set_intent("Test", "INT.SEM.ECO.WHY.ORG", "root")                # F
        g.put(Cell(postcode=parse_postcode("SEM.SEM.ECO.WHAT.SFT"),
                   primitive="a"))                                            # E
        g.put(Cell(postcode=parse_postcode("RES.MET.ECO.HOW_MUCH.ECN"),
                   primitive="b", fill=FillState.B))                          # B
        g.put(Cell(postcode=parse_postcode("EMG.CND.ECO.WHAT.COG"),
                   primitive="c", fill=FillState.C))                          # C
        fill(g, "STR.ENT.ECO.WHAT.SFT", "d", "X", 0.50,
             source=(INTENT_CONTRACT,))                                       # P
        fill(g, "OBS.MET.ECO.HOW.ORG", "e", "No source", 0.95)             # Q

        s = g.stats()
        assert s["F"] == 1
        assert s["E"] == 1
        assert s["B"] == 1
        assert s["C"] == 1
        assert s["P"] == 1
        assert s["Q"] == 1
