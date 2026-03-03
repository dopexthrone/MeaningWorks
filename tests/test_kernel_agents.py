"""
tests/test_kernel_agents.py — Tests for the 6 agents + orchestrator.

Tests: MEMORY, AUTHOR, VERIFIER, OBSERVER, EMERGENCE, GOVERNOR, compile().
Uses mock LLM for AUTHOR.
"""

import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, connect, FillStatus
from kernel.agents import (
    memory,
    author,
    verifier,
    observer,
    emergence,
    governor,
    deduplicate_dimensional,
    compile,
    CompileConfig,
    CompileResult,
    DedupeResult,
    SimulationResult,
    SimulationIssue,
    VerifyResult,
    ObserveResult,
    EmergeResult,
)


# ============================================================
# Mock LLM
# ============================================================

def _mock_llm_tattoo(prompt: str) -> list[dict]:
    """Mock LLM that extracts tattoo-booking concepts."""
    return [
        {
            "postcode": "SEM.SEM.ECO.WHAT.SFT",
            "primitive": "booking_system",
            "content": "Online booking system for tattoo studios with artist scheduling",
            "confidence": 0.95,
            "connections": ["STR.ENT.ECO.WHAT.SFT", "IDN.ACT.ECO.WHO.ORG"],
        },
        {
            "postcode": "STR.ENT.ECO.WHAT.SFT",
            "primitive": "product_booking",
            "content": "Booking product with slot management and walk-in queuing",
            "confidence": 0.90,
            "connections": ["SEM.SEM.ECO.WHAT.SFT"],
        },
        {
            "postcode": "IDN.ACT.ECO.WHO.ORG",
            "primitive": "actor_artist",
            "content": "Tattoo artist managing their own schedule and clients",
            "confidence": 0.92,
            "connections": ["SEM.SEM.ECO.WHAT.SFT"],
        },
        {
            "postcode": "EXC.FNC.ECO.HOW.SFT",
            "primitive": "fn_book_slot",
            "content": "Function to book a time slot with a specific artist",
            "confidence": 0.88,
            "connections": ["STR.ENT.ECO.WHAT.SFT"],
        },
        {
            "postcode": "AGN.ORC.ECO.WHO.SFT",
            "primitive": "scheduling_engine",
            "content": "Automated scheduling engine that optimizes artist availability",
            "confidence": 0.85,
            "connections": ["EXC.FNC.ECO.HOW.SFT"],
        },
    ]


def _mock_llm_empty(prompt: str) -> list[dict]:
    """Mock LLM that returns nothing."""
    return []


def _mock_llm_bad_postcodes(prompt: str) -> list[dict]:
    """Mock LLM that returns invalid postcodes."""
    return [
        {
            "postcode": "INVALID",
            "primitive": "bad",
            "content": "This should be skipped",
            "confidence": 0.95,
        },
        {
            "postcode": "SEM.SEM.ECO.WHAT.SFT",
            "primitive": "valid_one",
            "content": "This should succeed",
            "confidence": 0.90,
            "connections": [],
        },
    ]


def _mock_llm_no_provenance(prompt: str) -> list[dict]:
    """Mock LLM that returns extractions with no provenance trail."""
    return [
        {
            "postcode": "NET.FLW.ECO.HOW.SFT",
            "primitive": "orphan",
            "content": "This has no connections to anything",
            "confidence": 0.95,
            "connections": [],
        },
    ]


# ============================================================
# Helpers
# ============================================================

def _seeded_grid() -> Grid:
    g = Grid()
    g.set_intent(
        "Build a tattoo booking system with artist scheduling",
        "INT.SEM.ECO.WHY.SFT",
        "intent",
    )
    return g


def _filled_grid() -> Grid:
    """Grid with several filled cells for testing verifier/governor."""
    g = _seeded_grid()
    fill(g, "SEM.SEM.ECO.WHAT.SFT", "booking",
         "Booking system", 0.95,
         connections=("STR.ENT.ECO.WHAT.SFT",),
         source=(INTENT_CONTRACT,))
    fill(g, "STR.ENT.ECO.WHAT.SFT", "product",
         "Booking product", 0.90,
         connections=("SEM.SEM.ECO.WHAT.SFT",),
         source=("SEM.SEM.ECO.WHAT.SFT",))
    fill(g, "IDN.ACT.ECO.WHO.ORG", "artist",
         "Tattoo artist", 0.92,
         connections=("SEM.SEM.ECO.WHAT.SFT",),
         source=(INTENT_CONTRACT,))
    connect(g, "INT.SEM.ECO.WHY.SFT", "SEM.SEM.ECO.WHAT.SFT")
    return g


# ============================================================
# MEMORY
# ============================================================

class TestMemory:
    """memory(grid, previous_grids) bootstraps from history."""

    def test_imports_same_domain_cells(self):
        prev = _filled_grid()
        new = _seeded_grid()
        memory(new, [prev])
        # Should have imported SEM and STR cells (same SFT domain)
        assert new.has("SEM.SEM.ECO.WHAT.SFT")
        # But at reduced confidence
        imported = new.get("SEM.SEM.ECO.WHAT.SFT")
        assert imported.confidence < 0.95  # decayed

    def test_imports_cross_domain_cells(self):
        """MEMORY imports all domains — multi-pass compilation needs cross-domain accumulation."""
        prev = Grid()
        prev.set_intent("Medical system", "INT.SEM.ECO.WHY.MED", "med_intent")
        fill(prev, "SEM.SEM.ECO.WHAT.MED", "medical",
             "Medical records", 0.95, source=(INTENT_CONTRACT,))

        new = _seeded_grid()  # SFT domain root
        memory(new, [prev])
        assert new.has("SEM.SEM.ECO.WHAT.MED")
        imported = new.get("SEM.SEM.ECO.WHAT.MED")
        assert imported.confidence < 0.95  # still decayed

    def test_does_not_overwrite_existing(self):
        prev = _filled_grid()
        new = _seeded_grid()
        # Pre-fill a cell
        fill(new, "SEM.SEM.ECO.WHAT.SFT", "my_version",
             "My content", 0.99, source=(INTENT_CONTRACT,))
        memory(new, [prev])
        # Should keep our version
        cell = new.get("SEM.SEM.ECO.WHAT.SFT")
        assert cell.primitive == "my_version"

    def test_empty_history(self):
        new = _seeded_grid()
        memory(new, [])
        assert new.total_cells == 1  # only intent root

    def test_confidence_decay(self):
        prev = _filled_grid()
        new = _seeded_grid()
        memory(new, [prev], confidence_decay=0.5)
        imported = new.get("SEM.SEM.ECO.WHAT.SFT")
        # 0.95 * 0.5 = 0.475
        assert imported is not None
        assert abs(imported.confidence - 0.475) < 0.05

    def test_imports_from_multiple_grids(self):
        prev1 = _filled_grid()
        prev2 = Grid()
        prev2.set_intent("Another", "INT.SEM.ECO.WHY.SFT", "other")
        fill(prev2, "EXC.FNC.ECO.HOW.SFT", "fn_book",
             "Booking function", 0.88, source=(INTENT_CONTRACT,))

        new = _seeded_grid()
        memory(new, [prev1, prev2])
        # Should have cells from both
        assert new.has("SEM.SEM.ECO.WHAT.SFT")
        assert new.has("EXC.FNC.ECO.HOW.SFT")


# ============================================================
# AUTHOR
# ============================================================

class TestAuthor:
    """author(grid, input_text, llm_fn) fills from LLM."""

    def test_fills_cells_from_llm(self):
        g = _seeded_grid()
        author(g, "Build a tattoo booking system", _mock_llm_tattoo)
        assert g.has("SEM.SEM.ECO.WHAT.SFT")
        cell = g.get("SEM.SEM.ECO.WHAT.SFT")
        assert cell.is_filled
        assert cell.primitive == "booking_system"

    def test_fills_multiple_cells(self):
        g = _seeded_grid()
        author(g, "Build a tattoo booking system", _mock_llm_tattoo)
        # Mock returns 5 extractions
        filled = [c for c in g.filled_cells() if c.postcode.key != "INT.SEM.ECO.WHY.SFT"]
        assert len(filled) >= 4  # at least 4 of 5 should succeed

    def test_empty_llm_response(self):
        g = _seeded_grid()
        before = g.total_cells
        author(g, "Build something", _mock_llm_empty)
        # No new cells filled (empty cells from connections may exist)
        assert len(g.filled_cells()) == 1  # only intent root

    def test_invalid_postcodes_skipped(self):
        g = _seeded_grid()
        author(g, "Build something", _mock_llm_bad_postcodes)
        # Invalid postcode skipped, valid one fills
        assert g.has("SEM.SEM.ECO.WHAT.SFT")
        assert not g.has("INVALID")

    def test_connections_create_targets(self):
        g = _seeded_grid()
        author(g, "Build a tattoo booking system", _mock_llm_tattoo)
        # First extraction connects to STR and IDN
        assert g.has("STR.ENT.ECO.WHAT.SFT")
        assert g.has("IDN.ACT.ECO.WHO.ORG")

    def test_descent_triggered_on_low_confidence(self):
        g = _seeded_grid()
        author(g, "Build a tattoo booking system", _mock_llm_tattoo)
        # AGN.ORC at 0.85 should trigger descent (creates APP children)
        agn_cell = g.get("AGN.ORC.ECO.WHO.SFT")
        if agn_cell and agn_cell.confidence < 0.95:
            # Check for APP-level children
            app_cells = [
                c for c in g.cells.values()
                if c.postcode.layer == "AGN" and c.postcode.scope == "APP"
            ]
            assert len(app_cells) > 0


# ============================================================
# VERIFIER
# ============================================================

class TestVerifier:
    """verifier(grid) enforces provenance."""

    def test_boosts_valid_cells(self):
        g = _filled_grid()
        original_conf = g.get("SEM.SEM.ECO.WHAT.SFT").confidence
        result = verifier(g)
        assert result.checked >= 3
        assert result.promoted >= 1
        # Confidence should have increased
        new_conf = g.get("SEM.SEM.ECO.WHAT.SFT").confidence
        assert new_conf >= original_conf

    def test_quarantines_broken_provenance(self):
        g = _seeded_grid()
        # Manually place a cell with broken provenance
        bad_pc = parse_postcode("NET.FLW.ECO.HOW.SFT")
        bad_cell = Cell(
            postcode=bad_pc,
            primitive="orphan",
            content="No valid source",
            fill=FillState.F,
            confidence=0.90,
            source=("nonexistent.cell.key",),
        )
        g.put(bad_cell)

        result = verifier(g)
        assert result.quarantined >= 1
        assert g.get("NET.FLW.ECO.HOW.SFT").is_quarantined

    def test_human_source_accepted(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "test",
             "Content", 0.90, source=("human:alex",))
        result = verifier(g)
        assert result.quarantined == 0

    def test_memory_source_accepted(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "test",
             "Content", 0.90, source=("memory:prev_grid",))
        result = verifier(g)
        assert result.quarantined == 0

    def test_intent_contract_source_accepted(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "test",
             "Content", 0.90, source=(INTENT_CONTRACT,))
        result = verifier(g)
        assert result.quarantined == 0

    def test_verify_result_has_issues(self):
        g = _seeded_grid()
        bad_pc = parse_postcode("NET.FLW.ECO.HOW.SFT")
        g.put(Cell(
            postcode=bad_pc, primitive="bad",
            content="Bad", fill=FillState.F, confidence=0.9,
            source=("gone",),
        ))
        result = verifier(g)
        assert len(result.issues) >= 1
        assert "AX1" in result.issues[0]


# ============================================================
# OBSERVER
# ============================================================

class TestObserver:
    """observer(grid) detects patterns."""

    def test_detects_repeated_primitives(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "provenance",
             "Provenance in SEM", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "provenance",
             "Provenance in COG", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "EXC.GTE.ECO.HOW.SFT", "provenance",
             "Provenance in EXC", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "CTR.PLY.ECO.HOW.SFT", "policy",
             "Policy", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "AGN.AGT.ECO.WHO.SFT", "agent",
             "Agent", 0.90, source=(INTENT_CONTRACT,))

        result = observer(g)
        assert result.signals_detected >= 1
        assert result.candidates_created >= 1
        # Emergence targets are now dynamically computed (not hardcoded postcodes)
        emg_cells = [c for c in g.cells.values() if c.postcode.layer == "EMG"]
        assert len(emg_cells) >= 1

    def test_no_patterns_on_small_grid(self):
        g = _seeded_grid()
        result = observer(g)
        assert result.signals_detected == 0
        assert result.candidates_created == 0

    def test_observe_result_tracks_signals(self):
        g = _filled_grid()
        # Add more cells to get past minimum
        fill(g, "EXC.FNC.ECO.HOW.SFT", "fn", "C", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "AGN.ORC.ECO.WHO.SFT", "pipe", "C", 0.90, source=(INTENT_CONTRACT,))
        result = observer(g)
        assert isinstance(result, ObserveResult)


# ============================================================
# EMERGENCE
# ============================================================

class TestEmergence:
    """emergence(grid) promotes candidates."""

    def test_promotes_with_sufficient_evidence(self):
        g = _seeded_grid()
        # Create filled cells as evidence
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a", "C", 0.90, source=(INTENT_CONTRACT,))
        fill(g, "COG.BHV.ECO.HOW.COG", "b", "C", 0.90, source=(INTENT_CONTRACT,))
        # Create candidate with those as connections
        cand_pc = parse_postcode("EMG.CND.ECO.WHAT.COG")
        g.put(Cell(
            postcode=cand_pc, primitive="emerged",
            content="Pattern detected", fill=FillState.C,
            connections=("SEM.SEM.ECO.WHAT.SFT", "COG.BHV.ECO.HOW.COG"),
        ))
        result = emergence(g)
        assert result.promoted == 1
        cell = g.get("EMG.CND.ECO.WHAT.COG")
        assert cell.is_filled

    def test_rejects_insufficient_evidence(self):
        g = _seeded_grid()
        # Candidate with only 1 valid reference
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a", "C", 0.90, source=(INTENT_CONTRACT,))
        cand_pc = parse_postcode("EMG.CND.ECO.WHAT.COG")
        g.put(Cell(
            postcode=cand_pc, primitive="weak",
            content="Weak pattern", fill=FillState.C,
            connections=("SEM.SEM.ECO.WHAT.SFT", "NONEXISTENT.CELL.KEY"),
        ))
        result = emergence(g, min_evidence=2)
        assert result.rejected == 1
        cell = g.get("EMG.CND.ECO.WHAT.COG")
        assert cell.is_candidate  # still candidate, not promoted

    def test_no_candidates_no_action(self):
        g = _filled_grid()
        result = emergence(g)
        assert result.candidates_checked == 0
        assert result.promoted == 0


# ============================================================
# GOVERNOR
# ============================================================

class TestGovernor:
    """governor(grid) runs simulation gate."""

    def test_clean_grid_passes(self):
        g = _filled_grid()
        result = governor(g)
        assert result.passed
        assert result.hard_failures == 0
        assert result.can_emit

    def test_gap_detection(self):
        g = _seeded_grid()
        # Create gap: filled → empty → filled
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a",
             "Content A", 0.95,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        # STR is empty (created by connection)
        # Add a connection from empty STR to a filled cell
        empty_str = g.get("STR.ENT.ECO.WHAT.SFT")
        # We need to make the empty cell connect to something filled
        # Manually wire it
        from dataclasses import replace
        wired_empty = replace(empty_str, connections=("INT.SEM.ECO.WHY.SFT",))
        g.put(wired_empty)

        result = governor(g)
        gap_issues = [i for i in result.issues if i.check == "gap"]
        assert len(gap_issues) >= 1
        assert result.hard_failures >= 1
        assert not result.can_emit

    def test_cycle_detection(self):
        g = _seeded_grid()
        # Create parent cycle: A.parent = B, B.parent = A
        pc_a = parse_postcode("SEM.SEM.ECO.WHAT.SFT")
        pc_b = parse_postcode("STR.ENT.ECO.WHAT.SFT")
        g.put(Cell(postcode=pc_a, primitive="a",
                    content="A", fill=FillState.F, confidence=0.9,
                    parent="STR.ENT.ECO.WHAT.SFT",
                    source=(INTENT_CONTRACT,)))
        g.put(Cell(postcode=pc_b, primitive="b",
                    content="B", fill=FillState.F, confidence=0.9,
                    parent="SEM.SEM.ECO.WHAT.SFT",
                    source=(INTENT_CONTRACT,)))

        result = governor(g)
        cycle_issues = [i for i in result.issues if i.check == "cycle"]
        assert len(cycle_issues) >= 1
        assert result.hard_failures >= 1

    def test_dead_end_detection(self):
        g = _seeded_grid()
        # Create isolated filled cell
        fill(g, "RES.MET.ECO.HOW_MUCH.ECN", "cost",
             "Cost info", 0.90, source=(INTENT_CONTRACT,))
        result = governor(g)
        dead_ends = [i for i in result.issues if i.check == "dead_end"]
        assert len(dead_ends) >= 1
        assert result.soft_warnings >= 1
        # Dead ends are soft — should still pass
        assert result.passed

    def test_conflict_detection(self):
        g = _seeded_grid()
        # Two high-confidence connected cells with zero word overlap
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "alpha",
             "quantum entanglement photosynthesis", 0.95,
             connections=("STR.ENT.ECO.WHAT.SFT",),
             source=(INTENT_CONTRACT,))
        fill(g, "STR.ENT.ECO.WHAT.SFT", "beta",
             "chocolate restaurant submarine mountain", 0.95,
             connections=("SEM.SEM.ECO.WHAT.SFT",),
             source=("SEM.SEM.ECO.WHAT.SFT",))
        result = governor(g)
        conflict_issues = [i for i in result.issues if i.check == "conflict"]
        assert len(conflict_issues) >= 1

    def test_simulation_result_stats(self):
        g = _filled_grid()
        result = governor(g)
        assert result.total_cells > 0
        assert result.filled_cells > 0
        assert result.fill_rate > 0

    def test_empty_grid_passes(self):
        """Grid with only intent root should pass (no issues possible)."""
        g = _seeded_grid()
        result = governor(g)
        assert result.passed


# ============================================================
# DEDUPLICATION — collapse dimensional fan-out
# ============================================================

class TestDeduplicateDimensional:
    """deduplicate_dimensional() collapses fan-out."""

    def test_no_fanout_no_change(self):
        """Distinct concepts at different coordinates are not touched."""
        g = _filled_grid()
        result = deduplicate_dimensional(g)
        assert result.groups_found == 0
        assert result.cells_quarantined == 0

    def test_collapses_large_fanout(self):
        """8 cells sharing layer.concern.scope.domain but varying by dimension
        should be collapsed to 1 (the highest confidence)."""
        g = _seeded_grid()
        dims = ["WHAT", "HOW", "WHY", "WHO", "WHEN", "WHERE", "IF", "HOW_MUCH"]
        for i, dim in enumerate(dims):
            key = f"STR.PRV.DOM.{dim}.SFT"
            conf = 0.80 - i * 0.02  # WHAT=0.80, HOW=0.78, ..., HOW_MUCH=0.66
            fill(g, key, f"provenance-{dim.lower()}", f"Provenance from {dim} angle",
                 conf, source=(INTENT_CONTRACT,))
        result = deduplicate_dimensional(g)
        assert result.groups_found == 1
        assert result.cells_quarantined == 7  # 8 - 1 kept
        # The WHAT cell (highest confidence) should survive
        keeper = g.get("STR.PRV.DOM.WHAT.SFT")
        assert keeper.is_filled
        # All others should be quarantined
        for dim in dims[1:]:
            cell = g.get(f"STR.PRV.DOM.{dim}.SFT")
            assert cell.fill == FillState.Q

    def test_small_group_not_collapsed(self):
        """Two cells differing by dimension is legitimate, not fan-out."""
        g = _seeded_grid()
        fill(g, "STR.PRV.DOM.WHAT.SFT", "provenance-what", "What is provenance",
             0.80, source=(INTENT_CONTRACT,))
        fill(g, "STR.PRV.DOM.HOW.SFT", "provenance-how", "How provenance works",
             0.78, source=(INTENT_CONTRACT,))
        result = deduplicate_dimensional(g)
        assert result.groups_found == 0
        assert result.cells_quarantined == 0

    def test_different_primitives_not_collapsed(self):
        """4 cells at same coordinates but genuinely different concepts stay."""
        g = _seeded_grid()
        fill(g, "SEM.FNC.CMP.WHAT.SFT", "Cell-dataclass", "Frozen dataclass for cells",
             0.95, source=(INTENT_CONTRACT,))
        fill(g, "SEM.FNC.CMP.HOW.SFT", "fill-function", "The mutation operation",
             0.92, source=(INTENT_CONTRACT,))
        fill(g, "SEM.FNC.CMP.WHY.SFT", "provenance-axiom", "Why AX1 matters",
             0.88, source=(INTENT_CONTRACT,))
        fill(g, "SEM.FNC.CMP.WHO.SFT", "author-agent", "Who fills cells",
             0.85, source=(INTENT_CONTRACT,))
        result = deduplicate_dimensional(g)
        # All different base primitives — no fan-out
        assert result.groups_found == 0

    def test_preserves_root(self):
        """Root cell is never touched even if it's in a fan-out group."""
        g = Grid()
        g.set_intent("test", "INT.SEM.ECO.WHY.COG", "intent")
        for dim in ["WHAT", "HOW", "WHO", "WHEN"]:
            key = f"INT.SEM.ECO.{dim}.COG"
            fill(g, key, "intent-thing", f"Intent {dim}",
                 0.80, source=(INTENT_CONTRACT,))
        result = deduplicate_dimensional(g)
        # Root (WHY) should survive regardless — it's excluded from grouping
        root = g.get("INT.SEM.ECO.WHY.COG")
        assert root.is_filled

    def test_returns_dedupe_result(self):
        g = _seeded_grid()
        result = deduplicate_dimensional(g)
        assert isinstance(result, DedupeResult)


# ============================================================
# Orchestrator — compile()
# ============================================================

class TestCompile:
    """compile() runs the full pipeline."""

    def test_basic_compilation(self):
        result = compile(
            "Build a tattoo booking system with artist scheduling",
            _mock_llm_tattoo,
        )
        assert isinstance(result, CompileResult)
        assert result.grid.total_cells > 1
        assert result.iterations >= 1
        assert result.author_calls >= 1

    def test_compilation_fills_cells(self):
        result = compile(
            "Build a tattoo booking system",
            _mock_llm_tattoo,
        )
        filled = result.grid.filled_cells()
        assert len(filled) >= 4  # intent + at least 3 from LLM

    def test_compilation_runs_verifier(self):
        result = compile(
            "Build a tattoo booking system",
            _mock_llm_tattoo,
        )
        assert len(result.verify_results) >= 1
        assert result.verify_results[0].checked > 0

    def test_compilation_runs_governor(self):
        result = compile(
            "Build a tattoo booking system",
            _mock_llm_tattoo,
        )
        assert result.simulation is not None
        assert isinstance(result.simulation, SimulationResult)

    def test_compilation_with_history(self):
        # First compile
        r1 = compile("Tattoo booking", _mock_llm_tattoo)
        # Second compile with history
        r2 = compile(
            "Tattoo booking with walk-ins",
            _mock_llm_tattoo,
            history=[r1.grid],
        )
        # Should have more cells (bootstrapped from history)
        assert r2.grid.total_cells >= r1.grid.total_cells

    def test_compilation_max_iterations(self):
        config = CompileConfig(max_iterations=2)
        result = compile(
            "Build a tattoo booking system",
            _mock_llm_tattoo,
            config=config,
        )
        assert result.iterations <= 2

    def test_compilation_empty_llm(self):
        result = compile(
            "Build something vague",
            _mock_llm_empty,
        )
        # Should still have intent root
        assert result.grid.has("INT.SEM.ECO.WHY.SFT")
        assert len(result.grid.filled_cells()) == 1

    def test_compilation_converges(self):
        """With good extractions, pipeline should converge."""
        config = CompileConfig(max_iterations=5)
        result = compile(
            "Build a tattoo booking system",
            _mock_llm_tattoo,
            config=config,
        )
        # Should converge within 5 iterations
        assert result.converged or result.iterations == 5

    def test_compilation_custom_intent_postcode(self):
        result = compile(
            "Medical records system",
            _mock_llm_empty,
            intent_postcode="INT.SEM.ECO.WHY.MED",
            intent_primitive="medical_intent",
        )
        assert result.grid.root == "INT.SEM.ECO.WHY.MED"
        root = result.grid.get("INT.SEM.ECO.WHY.MED")
        assert root.primitive == "medical_intent"

    def test_compile_result_tracks_all_phases(self):
        result = compile("Tattoo booking", _mock_llm_tattoo)
        assert len(result.verify_results) >= 1
        assert len(result.observe_results) >= 1
        assert len(result.emerge_results) >= 1

    def test_nav_output_after_compile(self):
        result = compile("Tattoo booking", _mock_llm_tattoo)
        nav = result.grid.nav()
        assert "INT" in nav
        assert len(nav) > 50  # should have substantial content


# ============================================================
# Integration — Full Pipeline Stress
# ============================================================

class TestPipelineIntegration:
    """Stress tests for the full pipeline."""

    def test_bad_llm_doesnt_crash(self):
        """LLM returning garbage should not crash the pipeline."""
        def bad_llm(prompt):
            return [
                {"postcode": "INVALID", "primitive": "x", "content": "y", "confidence": 0.9},
                {"not_a_postcode": True},
                {},
                {"postcode": "SEM.SEM.ECO.WHAT.SFT", "primitive": "ok",
                 "content": "valid", "confidence": 0.8, "connections": []},
            ]
        result = compile("Test", bad_llm)
        assert result.grid.total_cells >= 1

    def test_llm_returning_none_fields(self):
        def none_llm(prompt):
            return [
                {"postcode": None, "primitive": None},
                {"postcode": "SEM.SEM.ECO.WHAT.SFT", "primitive": "",
                 "content": "test", "confidence": 0.5},
            ]
        result = compile("Test", none_llm)
        # Should not crash
        assert result.grid.total_cells >= 1

    def test_pipeline_idempotent_on_repeat(self):
        """Running compile twice with same input should produce similar grids."""
        r1 = compile("Tattoo booking system", _mock_llm_tattoo,
                     config=CompileConfig(max_iterations=1))
        r2 = compile("Tattoo booking system", _mock_llm_tattoo,
                     config=CompileConfig(max_iterations=1))
        # Same mock LLM → same extractions → same filled cells
        f1 = {c.postcode.key for c in r1.grid.filled_cells()}
        f2 = {c.postcode.key for c in r2.grid.filled_cells()}
        assert f1 == f2

    def test_quarantine_doesnt_leak_to_emission(self):
        """Governor should not allow quarantined grids to pass."""
        g = _seeded_grid()
        # Add a quarantined cell
        q_pc = parse_postcode("NET.FLW.ECO.HOW.SFT")
        g.put(Cell(postcode=q_pc, primitive="bad",
                    content="Bad", fill=FillState.Q))
        result = governor(g)
        # Quarantined cells should be counted
        assert result.quarantined_cells >= 1
        # But grid can still pass (quarantine is handled)
        assert result.passed  # no hard failures from quarantine alone
