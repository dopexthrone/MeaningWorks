"""
tests/test_kernel_nav_emission.py — Tests for nav format + emission.

Tests: nav serialization/deserialization roundtrip, context budget,
manifest emission, escalation extraction, simulation gate.
"""

import json
import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, connect
from kernel.nav import (
    grid_to_nav,
    nav_to_grid,
    budget_nav,
    estimate_tokens,
    TOKENS_PER_NODE,
)
from kernel.emission import (
    emit,
    Manifest,
    ManifestNode,
    Escalation,
    extract_escalations,
)


# ============================================================
# Helpers
# ============================================================

def _seeded_grid() -> Grid:
    g = Grid()
    g.set_intent(
        "Build a tattoo booking system",
        "INT.SEM.ECO.WHY.SFT",
        "intent",
    )
    return g


def _filled_grid() -> Grid:
    g = _seeded_grid()
    fill(g, "SEM.SEM.ECO.WHAT.SFT", "booking_system",
         "Online booking for tattoo studios", 0.95,
         connections=("STR.ENT.ECO.WHAT.SFT",),
         source=(INTENT_CONTRACT,))
    fill(g, "STR.ENT.ECO.WHAT.SFT", "product_booking",
         "Booking product with slot management", 0.90,
         connections=("SEM.SEM.ECO.WHAT.SFT",),
         source=("SEM.SEM.ECO.WHAT.SFT",))
    fill(g, "IDN.ACT.ECO.WHO.ORG", "artist",
         "Tattoo artist actor", 0.92,
         connections=("SEM.SEM.ECO.WHAT.SFT",),
         source=(INTENT_CONTRACT,))
    fill(g, "EXC.FNC.ECO.HOW.SFT", "fn_book_slot",
         "Book a time slot", 0.88,
         connections=("STR.ENT.ECO.WHAT.SFT",),
         source=(INTENT_CONTRACT,))
    connect(g, "INT.SEM.ECO.WHY.SFT", "SEM.SEM.ECO.WHAT.SFT")
    return g


def _grid_with_escalations() -> Grid:
    g = _filled_grid()
    # Add blocked cell
    blocked_pc = parse_postcode("RES.MET.ECO.HOW_MUCH.ECN")
    g.put(Cell(
        postcode=blocked_pc, primitive="business_model",
        content="", fill=FillState.B, confidence=0.0,
    ))
    # Add low-confidence partial
    fill(g, "TME.SCH.ECO.WHEN.ORG", "timeline",
         "Rough timeline", 0.60, source=(INTENT_CONTRACT,))
    return g


# ============================================================
# Nav Serialization
# ============================================================

class TestGridToNav:
    """grid_to_nav produces correct format."""

    def test_empty_grid(self):
        g = Grid()
        nav = grid_to_nav(g)
        assert "# MAP 0F/0T" in nav

    def test_seeded_grid_has_header(self):
        g = _seeded_grid()
        nav = grid_to_nav(g)
        assert "# MAP 1F/1T" in nav
        assert "# INTENT Build a tattoo booking system" in nav

    def test_filled_grid_has_layers(self):
        g = _filled_grid()
        nav = grid_to_nav(g)
        assert "## INT" in nav
        assert "## SEM" in nav
        assert "## STR" in nav

    def test_cell_format(self):
        g = _seeded_grid()
        nav = grid_to_nav(g)
        assert "INT.SEM.ECO.WHY.SFT | F 1.00 | intent" in nav

    def test_connections_in_nav(self):
        g = _filled_grid()
        nav = grid_to_nav(g)
        assert "-> STR.ENT.ECO.WHAT.SFT" in nav

    def test_excludes_empty_by_default(self):
        g = _filled_grid()
        # There are empty cells created by connections
        nav = grid_to_nav(g, include_empty=False)
        assert "| E " not in nav

    def test_includes_empty_when_requested(self):
        g = _seeded_grid()
        # Create an empty cell
        g.put(Cell(postcode=parse_postcode("SEM.SEM.ECO.WHAT.SFT"), primitive="empty_one"))
        nav = grid_to_nav(g, include_empty=True)
        assert "| E " in nav

    def test_parent_annotation(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "parent",
             "Parent", 0.95, source=(INTENT_CONTRACT,))
        fill(g, "SEM.SEM.APP.WHAT.SFT", "child",
             "Child", 0.90,
             parent="SEM.SEM.ECO.WHAT.SFT",
             source=("SEM.SEM.ECO.WHAT.SFT",))
        nav = grid_to_nav(g)
        assert "^SEM.SEM.ECO.WHAT.SFT" in nav

    def test_blocked_cell_in_nav(self):
        g = _grid_with_escalations()
        nav = grid_to_nav(g)
        assert "| B " in nav

    def test_all_fill_states_represented(self):
        g = _seeded_grid()
        g.put(Cell(postcode=parse_postcode("RES.MET.ECO.HOW_MUCH.ECN"),
                    primitive="b", fill=FillState.B))
        g.put(Cell(postcode=parse_postcode("EMG.CND.ECO.WHAT.COG"),
                    primitive="c", fill=FillState.C))
        fill(g, "NET.FLW.ECO.HOW.SFT", "q", "Bad", 0.95)  # quarantined (no source)
        nav = grid_to_nav(g)
        assert "| F " in nav
        assert "| B " in nav
        assert "| C " in nav
        assert "| Q " in nav


# ============================================================
# Nav Roundtrip
# ============================================================

class TestNavRoundtrip:
    """nav_to_grid(grid_to_nav(grid)) preserves structure."""

    def test_roundtrip_seeded(self):
        g = _seeded_grid()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        assert g2.root == g.root
        assert g2.has("INT.SEM.ECO.WHY.SFT")
        cell = g2.get("INT.SEM.ECO.WHY.SFT")
        assert cell.fill == FillState.F
        assert cell.confidence == 1.0
        assert cell.primitive == "intent"

    def test_roundtrip_filled(self):
        g = _filled_grid()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        # All filled cells should survive
        for cell in g.filled_cells():
            key = cell.postcode.key
            assert g2.has(key), f"Missing cell {key}"
            c2 = g2.get(key)
            assert c2.fill == cell.fill
            assert abs(c2.confidence - cell.confidence) < 0.01
            assert c2.primitive == cell.primitive

    def test_roundtrip_connections(self):
        g = _filled_grid()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        sem_cell = g2.get("SEM.SEM.ECO.WHAT.SFT")
        assert "STR.ENT.ECO.WHAT.SFT" in sem_cell.connections

    def test_roundtrip_parent(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "parent", "P", 0.95, source=(INTENT_CONTRACT,))
        fill(g, "SEM.SEM.APP.WHAT.SFT", "child", "C", 0.90,
             parent="SEM.SEM.ECO.WHAT.SFT", source=("SEM.SEM.ECO.WHAT.SFT",))
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        child = g2.get("SEM.SEM.APP.WHAT.SFT")
        assert child.parent == "SEM.SEM.ECO.WHAT.SFT"

    def test_roundtrip_intent_text(self):
        g = _seeded_grid()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        assert g2.intent_text == "Build a tattoo booking system"

    def test_roundtrip_blocked_cell(self):
        g = _grid_with_escalations()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        blocked = g2.get("RES.MET.ECO.HOW_MUCH.ECN")
        assert blocked is not None
        assert blocked.fill == FillState.B

    def test_content_not_preserved(self):
        """Nav format intentionally doesn't carry content (storage layer concern)."""
        g = _filled_grid()
        nav = grid_to_nav(g)
        g2 = nav_to_grid(nav)
        cell = g2.get("SEM.SEM.ECO.WHAT.SFT")
        assert cell.content == ""  # content is not in nav format


# ============================================================
# Context Budget
# ============================================================

class TestBudget:
    """budget_nav respects token limits."""

    def test_estimate_tokens(self):
        g = _filled_grid()
        tokens = estimate_tokens(g)
        assert tokens > 0
        # 5 filled cells * ~30 tokens + 20 header ≈ 170
        assert tokens < 500

    def test_budget_includes_all_when_sufficient(self):
        g = _filled_grid()
        nav = budget_nav(g, max_tokens=5000)
        # Should include all filled cells
        for cell in g.filled_cells():
            assert cell.primitive in nav

    def test_budget_truncates_when_insufficient(self):
        g = _filled_grid()
        nav = budget_nav(g, max_tokens=100)
        # Should include root at minimum
        assert "intent" in nav
        # Should have truncation notice
        assert "TRUNCATED" in nav

    def test_budget_prioritizes_root(self):
        g = _filled_grid()
        nav = budget_nav(g, max_tokens=60)
        # Root should always be present
        assert "INT.SEM.ECO.WHY.SFT" in nav

    def test_budget_prioritizes_high_confidence(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "high", "H", 0.99, source=(INTENT_CONTRACT,))
        fill(g, "STR.ENT.ECO.WHAT.SFT", "low", "L", 0.50, source=(INTENT_CONTRACT,))
        nav = budget_nav(g, max_tokens=120)
        # High confidence should be included before low
        if "TRUNCATED" in nav:
            # If truncated, high confidence should be there
            assert "high" in nav

    def test_budget_zero_tokens(self):
        g = _filled_grid()
        nav = budget_nav(g, max_tokens=0)
        # Should have at least the header
        assert "# MAP" in nav

    def test_estimate_scales_with_connections(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "many_conns", "C", 0.95,
             connections=("a.b.ECO.WHAT.SFT", "c.d.ECO.WHAT.SFT",
                         "e.f.ECO.WHAT.SFT", "g.h.ECO.WHAT.SFT"),
             source=(INTENT_CONTRACT,))
        tokens = estimate_tokens(g)
        # More connections = more tokens
        assert tokens > TOKENS_PER_NODE * 2


# ============================================================
# Escalation Extraction
# ============================================================

class TestEscalations:
    """extract_escalations finds B and low-confidence P cells."""

    def test_blocked_cell_creates_blocking_escalation(self):
        g = _grid_with_escalations()
        escs = extract_escalations(g)
        blocking = [e for e in escs if e.urgency == "blocking"]
        assert len(blocking) >= 1
        assert blocking[0].postcode == "RES.MET.ECO.HOW_MUCH.ECN"

    def test_low_confidence_partial_creates_non_blocking(self):
        g = _grid_with_escalations()
        escs = extract_escalations(g)
        non_blocking = [e for e in escs if e.urgency == "non-blocking"]
        assert len(non_blocking) >= 1

    def test_high_confidence_partial_no_escalation(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "test", "C", 0.82, source=(INTENT_CONTRACT,))
        escs = extract_escalations(g)
        # 0.82 > 0.80 threshold → no escalation
        non_blocking = [e for e in escs if e.urgency == "non-blocking"]
        assert len(non_blocking) == 0

    def test_filled_cells_no_escalation(self):
        g = _filled_grid()
        escs = extract_escalations(g)
        assert len(escs) == 0

    def test_escalation_has_all_fields(self):
        g = _grid_with_escalations()
        escs = extract_escalations(g)
        for e in escs:
            assert e.id.startswith("ESC-")
            assert e.urgency in ("blocking", "non-blocking")
            assert e.postcode
            assert e.question


# ============================================================
# Emission
# ============================================================

class TestEmit:
    """emit(grid) produces manifests."""

    def test_clean_grid_emits(self):
        g = _filled_grid()
        manifest = emit(g)
        assert manifest is not None
        assert manifest.simulation_passed
        assert manifest.emitted_cells > 0

    def test_only_filled_cells_emitted(self):
        g = _filled_grid()
        manifest = emit(g)
        for node in manifest.nodes:
            assert node.fill_state in ("F", "P")

    def test_root_in_manifest(self):
        g = _filled_grid()
        manifest = emit(g)
        postcodes = [n.postcode for n in manifest.nodes]
        assert "INT.SEM.ECO.WHY.SFT" in postcodes

    def test_dependency_ordering(self):
        """Parents should come before children in the manifest."""
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "parent",
             "Parent content", 0.95, source=(INTENT_CONTRACT,))
        fill(g, "SEM.SEM.APP.WHAT.SFT", "child",
             "Child content", 0.90,
             parent="SEM.SEM.ECO.WHAT.SFT",
             source=("SEM.SEM.ECO.WHAT.SFT",))
        manifest = emit(g)
        postcodes = [n.postcode for n in manifest.nodes]
        parent_idx = postcodes.index("SEM.SEM.ECO.WHAT.SFT")
        child_idx = postcodes.index("SEM.SEM.APP.WHAT.SFT")
        assert parent_idx < child_idx

    def test_escalations_included(self):
        g = _grid_with_escalations()
        manifest = emit(g, force=True)
        assert len(manifest.escalations) >= 1

    def test_simulation_gate_blocks_bad_grid(self):
        g = _seeded_grid()
        # Create gap: filled → empty → filled
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a", "A", 0.95,
             connections=("STR.ENT.ECO.WHAT.SFT",), source=(INTENT_CONTRACT,))
        from dataclasses import replace
        empty = g.get("STR.ENT.ECO.WHAT.SFT")
        wired = replace(empty, connections=("INT.SEM.ECO.WHY.SFT",))
        g.put(wired)
        manifest = emit(g, force=False)
        assert manifest is None  # simulation failed → no emission

    def test_force_emits_despite_failure(self):
        g = _seeded_grid()
        fill(g, "SEM.SEM.ECO.WHAT.SFT", "a", "A", 0.95,
             connections=("STR.ENT.ECO.WHAT.SFT",), source=(INTENT_CONTRACT,))
        from dataclasses import replace
        empty = g.get("STR.ENT.ECO.WHAT.SFT")
        wired = replace(empty, connections=("INT.SEM.ECO.WHY.SFT",))
        g.put(wired)
        manifest = emit(g, force=True)
        assert manifest is not None
        assert not manifest.simulation_passed

    def test_manifest_to_dict(self):
        g = _filled_grid()
        manifest = emit(g)
        d = manifest.to_dict()
        assert d["version"] == "1.0"
        assert d["type"] == "motherlabs_manifest"
        assert "nodes" in d
        assert "escalations" in d
        assert "stats" in d

    def test_manifest_json_serializable(self):
        g = _filled_grid()
        manifest = emit(g)
        d = manifest.to_dict()
        # Should not raise
        json_str = json.dumps(d, indent=2)
        assert len(json_str) > 0
        # Roundtrip
        parsed = json.loads(json_str)
        assert parsed["type"] == "motherlabs_manifest"

    def test_manifest_stats(self):
        g = _filled_grid()
        manifest = emit(g)
        assert manifest.total_grid_cells > 0
        assert manifest.fill_rate > 0
        assert len(manifest.layers_active) > 0

    def test_manifest_intent(self):
        g = _filled_grid()
        manifest = emit(g)
        assert manifest.intent == "Build a tattoo booking system"
        assert manifest.root_postcode == "INT.SEM.ECO.WHY.SFT"

    def test_empty_grid_emits_minimal(self):
        g = _seeded_grid()
        manifest = emit(g)
        assert manifest is not None
        assert manifest.emitted_cells == 1  # just intent root

    def test_node_has_layer_and_scope(self):
        g = _filled_grid()
        manifest = emit(g)
        for node in manifest.nodes:
            assert node.layer
            assert node.scope
            assert node.depth >= 0


# ============================================================
# Integration
# ============================================================

class TestNavEmissionIntegration:
    """Nav + emission working together."""

    def test_nav_then_emit(self):
        """Generate nav for inspection, then emit for execution."""
        g = _filled_grid()
        # Nav for human review
        nav = grid_to_nav(g)
        assert "booking_system" in nav
        # Emission for Claude Code
        manifest = emit(g)
        assert manifest.emitted_cells >= 4

    def test_compile_then_nav_then_emit(self):
        """Full flow: compile → nav → emit."""
        from kernel.agents import compile as kernel_compile

        def mock_llm(prompt):
            return [
                {"postcode": "SEM.SEM.ECO.WHAT.SFT", "primitive": "system",
                 "content": "Booking system", "confidence": 0.95,
                 "connections": ["STR.ENT.ECO.WHAT.SFT"]},
                {"postcode": "STR.ENT.ECO.WHAT.SFT", "primitive": "product",
                 "content": "Product", "confidence": 0.90,
                 "connections": ["SEM.SEM.ECO.WHAT.SFT"]},
            ]

        result = kernel_compile("Tattoo booking", mock_llm)
        # Nav
        nav = grid_to_nav(result.grid)
        assert "SEM" in nav
        # Emit
        manifest = emit(result.grid)
        assert manifest is not None
        # Roundtrip nav
        g2 = nav_to_grid(nav)
        assert g2.has("SEM.SEM.ECO.WHAT.SFT")
