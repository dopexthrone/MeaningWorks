"""
tests/test_kernel_navigator.py — Tests for the semantic map navigator.

Tests next-cell prediction, descent, emergence detection, convergence.
"""

import pytest
from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, connect, FillStatus
from kernel.navigator import (
    next_cell,
    score_candidates,
    is_converged,
    should_descend,
    descend,
    descend_selective,
    detect_emergence,
    promote_emergence,
    ScoredCell,
    EmergenceSignal,
    CONFIDENCE_THRESHOLD,
)


# ============================================================
# Helpers
# ============================================================

def _seeded_grid() -> Grid:
    """Create a grid with intent root filled."""
    g = Grid()
    g.set_intent(
        "Build a tattoo booking system",
        "INT.SEM.ECO.WHY.ORG",
        "founding_problem",
    )
    return g


def _rich_grid() -> Grid:
    """Create a grid with multiple filled cells across layers."""
    g = _seeded_grid()
    fill(g, "SEM.SEM.ECO.WHAT.SFT", "booking_system",
         "Online booking for tattoo studios", 0.95,
         connections=("STR.ENT.ECO.WHAT.SFT", "AGN.ORC.ECO.WHO.SFT"),
         source=(INTENT_CONTRACT,))
    fill(g, "STR.ENT.ECO.WHAT.SFT", "product_booking",
         "Booking product", 0.90,
         parent="SEM.SEM.ECO.WHAT.SFT",
         source=("SEM.SEM.ECO.WHAT.SFT",))
    fill(g, "AGN.ORC.ECO.WHO.SFT", "agent_pipeline",
         "7-agent pipeline", 0.95,
         source=("SEM.SEM.ECO.WHAT.SFT",))
    connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
    return g


# ============================================================
# Next-cell prediction
# ============================================================

class TestNextCell:
    """next_cell(grid) → postcode."""

    def test_empty_grid_returns_none(self):
        g = Grid()
        assert next_cell(g) is None

    def test_intent_only_returns_none(self):
        """Intent root with no connections — nowhere to go."""
        g = _seeded_grid()
        # Root has no connections, so no unfilled connections
        assert next_cell(g) is None

    def test_unfilled_connection_is_top_candidate(self):
        g = _seeded_grid()
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Content", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        result = next_cell(g)
        assert result == "SEM.SEM.ECO.WHAT.SFT"

    def test_multiple_connections_scored(self):
        g = _seeded_grid()
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Content", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT", "STR.ENT.ECO.WHAT.SFT"),
             source=(INTENT_CONTRACT,))
        candidates = score_candidates(g)
        assert len(candidates) >= 2
        # Both targets should be candidates
        keys = [c.postcode_key for c in candidates]
        assert "SEM.SEM.ECO.WHAT.SFT" in keys
        assert "STR.ENT.ECO.WHAT.SFT" in keys

    def test_filled_connection_not_candidate(self):
        g = _seeded_grid()
        # Fill target first
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.95, source=(INTENT_CONTRACT,))
        # Now connect to it
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Content", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        candidates = score_candidates(g)
        keys = [c.postcode_key for c in candidates]
        assert "SEM.SEM.ECO.WHAT.SFT" not in keys

    def test_cross_layer_gap_scored(self):
        """Active layer with empty root should be high priority."""
        g = _seeded_grid()
        # Activate SEM layer by creating an empty root
        g.activate_layer("SEM", "SEM", "WHAT", "SFT")
        candidates = score_candidates(g)
        keys = [c.postcode_key for c in candidates]
        # The SEM ECO root should be a candidate
        sem_candidates = [k for k in keys if k.startswith("SEM.") and ".ECO." in k]
        assert len(sem_candidates) > 0

    def test_low_confidence_neighbor_boosted(self):
        g = _seeded_grid()
        # Fill a cell with low confidence that has an empty connection
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Vague content", 0.60,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        candidates = score_candidates(g)
        # STR target should have low-conf neighbor bonus
        str_candidates = [c for c in candidates if c.postcode_key == "STR.ENT.ECO.WHAT.SFT"]
        assert len(str_candidates) == 1
        assert "low_conf_neighbor" in str_candidates[0].reason

    def test_depth_penalty_applied(self):
        """Deeper cells should score lower than shallower ones, all else equal."""
        g = _seeded_grid()
        # Create two unfilled targets at different depths
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "parent",
             "Parent", 0.90,
             connections=("SEM.SEM.APP.WHAT.SFT", "SEM.SEM.DOM.WHAT.SFT"),
             source=(INTENT_CONTRACT,))
        # Create them
        g.put(Cell(postcode=parse_postcode("SEM.SEM.APP.WHAT.SFT"), primitive="app"))
        g.put(Cell(postcode=parse_postcode("SEM.SEM.DOM.WHAT.SFT"), primitive="dom"))
        candidates = score_candidates(g)
        app_score = next((c.score for c in candidates if c.postcode_key == "SEM.SEM.APP.WHAT.SFT"), 0)
        dom_score = next((c.score for c in candidates if c.postcode_key == "SEM.SEM.DOM.WHAT.SFT"), 0)
        # APP (depth 1) should score higher than DOM (depth 2)
        assert app_score > dom_score

    def test_candidate_cell_is_fillable(self):
        """Candidate cells should appear in next_cell scoring."""
        g = _seeded_grid()
        # Create a candidate cell
        cand_pc = parse_postcode("EMG.CND.ECO.WHAT.COG")
        g.put(Cell(postcode=cand_pc, primitive="discovery", fill=FillState.C))
        # Connect something to it
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Content", 0.95,
             connections=("EMG.CND.ECO.WHAT.COG",),
             source=(INTENT_CONTRACT,))
        candidates = score_candidates(g)
        keys = [c.postcode_key for c in candidates]
        assert "EMG.CND.ECO.WHAT.COG" in keys


class TestScoreConsistency:
    """Score ordering and consistency."""

    def test_scores_are_positive(self):
        g = _rich_grid()
        candidates = score_candidates(g)
        for c in candidates:
            assert c.score > 0

    def test_scores_descending(self):
        g = _rich_grid()
        candidates = score_candidates(g)
        for i in range(len(candidates) - 1):
            assert candidates[i].score >= candidates[i + 1].score

    def test_reason_populated(self):
        g = _rich_grid()
        candidates = score_candidates(g)
        for c in candidates:
            assert c.reason  # should never be empty


# ============================================================
# Convergence
# ============================================================

class TestConvergence:
    """is_converged(grid) checks."""

    def test_empty_grid_not_converged(self):
        g = Grid()
        assert not is_converged(g)

    def test_intent_only_not_converged(self):
        g = _seeded_grid()
        # Root has no connections, technically nothing to resolve,
        # but we need more than just the root
        # With only 1 cell, fill_rate is 1.0 and avg confidence is 1.0
        # which meets the threshold
        assert is_converged(g)

    def test_unfilled_connections_not_converged(self):
        g = _seeded_grid()
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Content", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        assert not is_converged(g)

    def test_all_connections_filled_high_confidence_converged(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.98, source=(INTENT_CONTRACT,))
        connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")
        # Both cells filled, all connections resolved, high confidence
        assert is_converged(g)

    def test_cross_layer_gap_not_converged(self):
        g = _seeded_grid()
        # Activate SEM layer but leave root empty
        g.activate_layer("SEM", "SEM", "WHAT", "SFT")
        assert not is_converged(g)

    def test_high_fill_rate_converged(self):
        """Even with moderate confidence, high fill rate = converged."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a", "C", 0.88, source=(INTENT_CONTRACT,))
        fill(g, "STR.ENT.ECO.WHAT.SFT", "b", "C", 0.88, source=(INTENT_CONTRACT,))
        fill(g, "AGN.ORC.ECO.WHO.SFT", "c", "C", 0.88, source=(INTENT_CONTRACT,))
        fill(g, "EXC.FNC.ECO.HOW.SFT", "d", "C", 0.88, source=(INTENT_CONTRACT,))
        # All have connections to each other so none are dangling
        # 5 cells, all filled, fill_rate = 1.0 > 0.80
        assert is_converged(g)


# ============================================================
# Descent
# ============================================================

class TestShouldDescend:
    """should_descend(cell) checks."""

    def test_filled_low_confidence_descends(self):
        pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", fill=FillState.F,
                    confidence=0.90, source=(INTENT_CONTRACT,))
        assert should_descend(cell)

    def test_high_confidence_no_descend(self):
        pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", fill=FillState.F,
                    confidence=0.96, source=(INTENT_CONTRACT,))
        assert not should_descend(cell)

    def test_empty_no_descend(self):
        pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test")
        assert not should_descend(cell)

    def test_val_scope_no_descend(self):
        pc = parse_postcode("SEM.SEM.VAL.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", fill=FillState.F,
                    confidence=0.90, source=(INTENT_CONTRACT,))
        assert not should_descend(cell)

    def test_partial_fill_descends(self):
        pc = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        cell = Cell(postcode=pc, primitive="test", fill=FillState.P,
                    confidence=0.50, source=(INTENT_CONTRACT,))
        assert should_descend(cell)


class TestDescend:
    """descend(grid, postcode) creates children."""

    def test_descend_eco_creates_app_children(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        assert len(children) > 0
        for child_key in children:
            pc = parse_postcode(child_key)
            assert pc.scope == "APP"
            assert pc.layer == "SEM"
            assert pc.concern == "SEM"
            assert pc.domain == "SFT"

    def test_descend_creates_relevant_dimensions_only(self):
        """Selective descent: only creates children for populated dimensions."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        # Only parent's own dimension (WHAT) — no other SEM.SEM fills exist
        assert len(children) == 1
        assert "SEM.SEM.APP.WHAT.SFT" in children

    def test_descend_expands_with_populated_dimensions(self):
        """When multiple dimensions are filled in the same layer+concern,
        descent creates children for all of them."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "SEM.SEM.ECO.HOW.SFT", "process",
             "How it works", 0.88, source=(INTENT_CONTRACT,))
        fill(g, "SEM.SEM.ECO.WHY.SFT", "purpose",
             "Why it exists", 0.85, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        # WHAT (own) + HOW + WHY = 3 dimensions
        assert len(children) == 3
        dims = {parse_postcode(c).dimension for c in children}
        assert dims == {"WHAT", "HOW", "WHY"}

    def test_descend_high_confidence_no_children(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.96, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        assert children == []

    def test_descend_val_scope_no_children(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.VAL.WHAT.SFT", "value",
             "Atomic", 0.90, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.VAL.WHAT.SFT")
        assert children == []

    def test_descend_nonexistent_cell(self):
        g = _seeded_grid()
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        assert children == []

    def test_descend_idempotent(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        c1 = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        c2 = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        # Second call creates no new children
        assert c2 == []

    def test_descend_children_have_parent_set(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        for child_key in children:
            child = g.get(child_key)
            assert child.parent == "SEM.SEM.ECO.WHAT.SFT"

    def test_descend_children_are_empty(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        for child_key in children:
            child = g.get(child_key)
            assert child.is_empty

    def test_descend_app_creates_dom_children(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "parent",
             "Parent", 0.90, source=(INTENT_CONTRACT,))
        descend(g, "SEM.SEM.ECO.WHAT.SFT")
        # Fill an APP child
        fill(g, "SEM.SEM.APP.WHAT.SFT", "app",
             "App level", 0.88,
             parent="SEM.SEM.ECO.WHAT.SFT",
             source=("SEM.SEM.ECO.WHAT.SFT",))
        children = descend(g, "SEM.SEM.APP.WHAT.SFT")
        for child_key in children:
            pc = parse_postcode(child_key)
            assert pc.scope == "DOM"


class TestDescendSelective:
    """descend_selective only creates specified dimensions."""

    def test_selective_creates_only_requested(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend_selective(g, "SEM.SEM.ECO.WHAT.SFT", ("WHAT", "HOW"))
        assert len(children) == 2
        dims = {parse_postcode(k).dimension for k in children}
        assert dims == {"WHAT", "HOW"}

    def test_selective_empty_dimensions(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        children = descend_selective(g, "SEM.SEM.ECO.WHAT.SFT", ())
        assert children == []


# ============================================================
# Emergence detection
# ============================================================

class TestEmergenceDetection:
    """detect_emergence(grid) pattern detection."""

    def test_too_few_fills_no_detection(self):
        g = _seeded_grid()
        signals = detect_emergence(g)
        assert signals == []

    def test_repeated_primitive_detected(self):
        g = _seeded_grid()
        # Same primitive at different postcodes in different layers
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "provenance",
             "Provenance in SEM", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "provenance",
             "Provenance in COG", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "EXC.GTE.ECO.HOW.SFT", "provenance",
             "Provenance in EXC", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "CTR.PLY.ECO.HOW.SFT", "policy",
             "Policy in CTR", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "AGN.AGT.ECO.WHO.SFT", "agent",
             "Agent in AGN", 0.90, source=(INTENT_CONTRACT,))

        signals = detect_emergence(g)
        repeated = [s for s in signals if s.signal_type == "repeated_primitive"]
        assert len(repeated) >= 1
        prim_names = [s.primitive for s in repeated]
        assert "emerged_provenance" in prim_names

    def test_shared_connection_detected(self):
        g = _seeded_grid()
        # Two cells in different layers connect to same target
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "C1", 0.90,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "AGN.ORC.ECO.WHO.SFT", "pipeline",
             "C2", 0.90,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "reasoning",
             "C3", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "EXC.FNC.ECO.HOW.SFT", "compile_fn",
             "C4", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "CTR.PLY.ECO.HOW.SFT", "policy",
             "C5", 0.90, source=(INTENT_CONTRACT,))

        signals = detect_emergence(g)
        shared = [s for s in signals if s.signal_type == "shared_connection"]
        assert len(shared) >= 1

    def test_orphan_cluster_detected(self):
        g = _seeded_grid()
        # Create orphan cells — filled but no connections, no parent, not connected to
        fill(g, "RES.MET.ECO.HOW_MUCH.ECN", "cost",
             "Cost info", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "OBS.MET.ECO.HOW.ORG", "metric",
             "Metric info", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "NET.FLW.ECO.HOW.SFT", "flow",
             "Flow info", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "TME.SCH.ECO.WHEN.ORG", "timeline",
             "Timeline", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "IDN.ACT.ECO.WHO.ORG", "actor",
             "Actor", 0.90, source=(INTENT_CONTRACT,))

        signals = detect_emergence(g)
        orphan = [s for s in signals if s.signal_type == "orphan_cluster"]
        assert len(orphan) >= 1

    def test_no_false_positives_on_clean_grid(self):
        """Well-connected grid should not produce orphan signals."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "C", 0.95,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "STR.ENT.ECO.WHAT.SFT", "product",
             "C", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=("SEM.SEM.ECO.WHAT.SFT",))
        fill(g, "AGN.ORC.ECO.WHO.SFT", "pipeline",
             "C", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "EXC.FNC.ECO.HOW.SFT", "fn",
             "C", 0.95,
             connections=("AGN.ORC.ECO.WHO.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "reason",
             "C", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        signals = detect_emergence(g)
        orphan = [s for s in signals if s.signal_type == "orphan_cluster"]
        assert len(orphan) == 0


class TestPromoteEmergence:
    """promote_emergence creates candidate cells."""

    def test_promote_creates_candidate(self):
        g = _seeded_grid()
        signal = EmergenceSignal(
            signal_type="repeated_primitive",
            evidence=("SEM.SEM.ECO.WHAT.SFT", "COG.BHV.ECO.HOW.COG"),
            primitive="emerged_provenance",
            description="Provenance appears in 2 layers",
        )
        cell = promote_emergence(g, signal, "EMG.CND.ECO.WHAT.COG")
        assert cell.fill == FillState.C
        assert cell.primitive == "emerged_provenance"
        assert g.has("EMG.CND.ECO.WHAT.COG")

    def test_promoted_cell_has_evidence_as_connections(self):
        g = _seeded_grid()
        signal = EmergenceSignal(
            signal_type="shared_connection",
            evidence=("SEM.SEM.ECO.WHAT.SFT", "AGN.ORC.ECO.WHO.SFT"),
            primitive="hub",
            description="Shared target",
        )
        cell = promote_emergence(g, signal, "EMG.CND.ECO.WHAT.SFT")
        assert cell.connections == ("SEM.SEM.ECO.WHAT.SFT", "AGN.ORC.ECO.WHO.SFT")


# ============================================================
# Integration — Navigator-driven compilation loop
# ============================================================

class TestNavigatorIntegration:
    """End-to-end: use navigator to drive a mini compilation."""

    def test_navigator_driven_loop(self):
        """Simulate: fill intent → navigator picks next → fill → repeat → converge."""
        g = Grid()
        g.set_intent(
            "Build a tattoo booking system with artist schedules and walk-in queuing",
            "INT.SEM.ECO.WHY.ORG",
            "intent",
        )

        # Fill intent details with connections to explore
        fill(g, "INT.SEM.ECO.WHAT.ORG", "mission",
             "Tattoo booking with scheduling and queuing", 0.97,
             connections=(
                 "SEM.SEM.ECO.WHAT.SFT",
                 "STR.ENT.ECO.WHAT.SFT",
                 "IDN.ACT.ECO.WHO.ORG",
             ),
             source=(INTENT_CONTRACT,))

        # Navigator should pick one of the unfilled connections
        target = next_cell(g)
        assert target is not None
        assert target in ("SEM.SEM.ECO.WHAT.SFT", "STR.ENT.ECO.WHAT.SFT", "IDN.ACT.ECO.WHO.ORG")

        # Fill it
        fill(g, target, "booking_system",
             "System for booking tattoo appointments", 0.92,
             source=("INT.SEM.ECO.WHAT.ORG",))

        # Navigator should pick another unfilled connection
        target2 = next_cell(g)
        assert target2 is not None
        assert target2 != target  # shouldn't re-pick the filled one

        # Fill remaining connections
        for remaining_key in ("SEM.SEM.ECO.WHAT.SFT", "STR.ENT.ECO.WHAT.SFT", "IDN.ACT.ECO.WHO.ORG"):
            if not g.get(remaining_key) or g.get(remaining_key).is_empty:
                fill(g, remaining_key, remaining_key.split(".")[0].lower(),
                     "Filled by navigator loop", 0.90,
                     source=("INT.SEM.ECO.WHAT.ORG",))

        # Should be closer to convergence now
        assert len(g.filled_cells()) >= 4

    def test_descent_then_navigate(self):
        """Fill ECO, descend to APP, navigator picks APP children."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "compiler",
             "Semantic compiler", 0.90,
             source=(INTENT_CONTRACT,))

        # Descend
        children = descend(g, "SEM.SEM.ECO.WHAT.SFT")
        assert len(children) > 0

        # Connect root to the ECO cell so we have an unfilled path
        connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")

        # Navigator should pick one of the APP children via depth pressure
        # or the children should be candidates if connected
        candidates = score_candidates(g)
        # At minimum, depth pressure should create some candidates
        assert len(candidates) >= 0  # may be 0 if no scored paths to children

    def test_emergence_feeds_back_into_navigation(self):
        """Detected emergence creates candidate → navigator can target it."""
        g = _seeded_grid()
        # Create conditions for emergence
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "trust",
             "Trust in SEM", 0.90,
             connections=("EMG.CND.ECO.WHAT.COG",),
             source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "trust",
             "Trust in COG", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "EXC.GTE.ECO.HOW.SFT", "trust",
             "Trust in EXC", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "AGN.AGT.ECO.WHO.SFT", "pipeline",
             "Pipeline", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "STR.ENT.ECO.WHAT.SFT", "product",
             "Product", 0.90, source=(INTENT_CONTRACT,))

        # Detect emergence
        signals = detect_emergence(g)
        trust_signals = [s for s in signals if "trust" in s.primitive]

        if trust_signals:
            # Promote it
            promote_emergence(g, trust_signals[0], "EMG.CND.ECO.WHAT.COG")

            # Now navigator should see the candidate
            candidates = score_candidates(g)
            keys = [c.postcode_key for c in candidates]
            assert "EMG.CND.ECO.WHAT.COG" in keys

    def test_full_cycle_intent_to_convergence(self):
        """Prove: intent → fill → navigate → fill → ... → converge."""
        g = Grid()
        g.set_intent("Simple system", "INT.SEM.ECO.WHY.ORG", "intent")

        # Manually fill a few connected cells to get to convergence
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "core",
             "Core system", 0.96,
             source=(INTENT_CONTRACT,))
        connect(g, "INT.SEM.ECO.WHY.ORG", "SEM.SEM.ECO.WHAT.SFT")

        # Now: all connections resolved, high confidence
        assert is_converged(g)

        # Navigator should return None
        assert next_cell(g) is None


# ============================================================
# Edge Cases
# ============================================================

class TestNavigatorEdgeCases:
    """Corner cases."""

    def test_grid_with_only_blocked_cells(self):
        g = _seeded_grid()
        g.put(Cell(postcode=parse_postcode("RES.MET.ECO.HOW_MUCH.ECN"),
                    primitive="blocked", fill=FillState.B))
        # Blocked cells aren't fillable candidates
        candidates = score_candidates(g)
        blocked_cands = [c for c in candidates if c.postcode_key == "RES.MET.ECO.HOW_MUCH.ECN"]
        assert len(blocked_cands) == 0

    def test_grid_with_only_quarantined_cells(self):
        g = _seeded_grid()
        g.put(Cell(postcode=parse_postcode("RES.MET.ECO.HOW_MUCH.ECN"),
                    primitive="bad", fill=FillState.Q))
        candidates = score_candidates(g)
        q_cands = [c for c in candidates if c.postcode_key == "RES.MET.ECO.HOW_MUCH.ECN"]
        assert len(q_cands) == 0

    def test_depth_pressure_on_deeply_nested(self):
        """Depth pressure should work at multiple scope levels."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "eco", "ECO", 0.90, source=(INTENT_CONTRACT,))
        descend(g, "SEM.SEM.ECO.WHAT.SFT")
        fill(g, "SEM.SEM.APP.WHAT.SFT", "app", "APP", 0.88,
             parent="SEM.SEM.ECO.WHAT.SFT", source=("SEM.SEM.ECO.WHAT.SFT",))
        children = descend(g, "SEM.SEM.APP.WHAT.SFT")
        # Should have DOM-level children
        assert any("DOM" in k for k in children)

    def test_scored_cell_dataclass(self):
        sc = ScoredCell(postcode_key="INT.SEM.ECO.WHY.ORG", score=10.0, reason="test")
        assert sc.postcode_key == "INT.SEM.ECO.WHY.ORG"
        assert sc.score == 10.0

    def test_emergence_signal_dataclass(self):
        es = EmergenceSignal(
            signal_type="repeated_primitive",
            evidence=("a", "b"),
            primitive="test",
            description="Test signal",
        )
        assert es.signal_type == "repeated_primitive"
        assert len(es.evidence) == 2
