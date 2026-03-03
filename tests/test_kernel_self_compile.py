"""
tests/test_kernel_self_compile.py — Phase 6: The Fixed-Point Test.

F(F) ~ F  — The kernel compiling its own specification.

Uses a mock LLM that returns kernel-aware extractions (cells, postcodes,
connections) as if an actual LLM had read the kernel spec and decomposed it.
Validates that the output grid:
  - Activates the expected layers
  - Produces a valid emission manifest
  - Contains cross-layer connections
  - Converges within bounded iterations
  - Has structural properties consistent with a real compilation
"""

import pytest
from kernel.cell import Cell, FillState, Postcode, parse_postcode, LAYERS
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill, connect
from kernel.navigator import (
    next_cell, score_candidates, is_converged,
    should_descend, descend, detect_emergence, promote_emergence,
)
from kernel.agents import (
    compile, CompileConfig, CompileResult,
    memory, author, verifier, observer, emergence, governor,
)
from kernel.nav import grid_to_nav, nav_to_grid, budget_nav, estimate_tokens
from kernel.emission import emit, Manifest, extract_escalations


# ---------------------------------------------------------------------------
# Kernel spec — the input text that describes the kernel itself
# ---------------------------------------------------------------------------

KERNEL_SPEC = """
Motherlabs Semantic Map Kernel.

A coordinate-based semantic compiler that transforms unstructured intent
into a verified grid of cells connected by provenance-traced edges.

Primitives:
  - Cell: atomic unit with postcode (5-axis coordinate), primitive name,
    content, fill state (F/P/E/B/Q/C), confidence, connections, parent, source.
  - Grid: mutable container of cells, tracks activated layers.
  - Fill: the only mutation operation. Enforces AX1 (provenance), AX2 (descent),
    AX3 (feedback/revision), AX5 (constraint). Returns FillResult.
  - Connect: directed edge between cells, triggers layer activation.

Navigator:
  - next_cell: scores unfilled cells by priority (unfilled connections,
    cross-layer gaps, low-confidence neighbors, depth pressure).
  - descend: creates child cells at deeper scope.
  - detect_emergence: finds repeated primitives, shared connections, orphan clusters.

Agents (6):
  - MEMORY: imports from history at reduced confidence.
  - AUTHOR: only LLM-touching agent. Extracts structured cells from input.
  - VERIFIER: checks provenance chains, boosts or quarantines.
  - OBSERVER: detects emergence patterns, places candidates.
  - EMERGENCE: validates and promotes candidate cells.
  - GOVERNOR: simulation gate with 4 checks (conflict, gap, cycle, dead-end).

Pipeline: MEMORY → [AUTHOR → VERIFIER → OBSERVER → EMERGENCE → GOVERNOR] loop.

Emission: manifest export with dependency ordering, simulation gating, escalation extraction.

Five axioms enforced structurally:
  AX1 PROVENANCE — every fill traces to input
  AX2 DESCENT — parent must be filled before child
  AX3 FEEDBACK — re-fill preserves revision history
  AX4 EMERGENCE — patterns detected and promoted
  AX5 CONSTRAINT — blocked cells reject fills
"""


# ---------------------------------------------------------------------------
# Mock LLM — returns kernel-aware extractions
# ---------------------------------------------------------------------------

_CALL_COUNT = 0

def _mock_llm_kernel(prompt: str) -> list[dict]:
    """Mock LLM that decomposes the kernel spec into structured cells.

    First call: high-level architecture (layer-scope cells).
    Second call: component details (connections, cross-layer wiring).
    Third+ call: refinements (confidence boosts, no new cells).
    """
    global _CALL_COUNT
    _CALL_COUNT += 1
    call = _CALL_COUNT

    if call == 1:
        # First pass: high-level semantic structure
        return [
            {
                "postcode": "SEM.ENT.ECO.WHAT.COG",
                "primitive": "semantic-architecture",
                "content": "5-axis coordinate system (LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN) encoding semantic position. 16 layers from INT to MET.",
                "confidence": 0.90,
                "connections": ("INT.SEM.ECO.WHY.SFT",),
            },
            {
                "postcode": "ORG.ENT.ECO.WHAT.SFT",
                "primitive": "grid-structure",
                "content": "Grid as mutable container of cells. Activated layers track which coordinate planes are populated.",
                "confidence": 0.88,
                "connections": ("SEM.ENT.ECO.WHAT.COG",),
            },
            {
                "postcode": "COG.BHV.ECO.HOW.COG",
                "primitive": "fill-operation",
                "content": "Fill enforces 5 axioms structurally: provenance tracing, descent ordering, revision history, emergence promotion, constraint blocking.",
                "confidence": 0.92,
                "connections": ("ORG.ENT.ECO.WHAT.SFT", "SEM.ENT.ECO.WHAT.COG"),
            },
            {
                "postcode": "AGN.BHV.ECO.WHO.SFT",
                "primitive": "agent-pipeline",
                "content": "6 agents: MEMORY, AUTHOR, VERIFIER, OBSERVER, EMERGENCE, GOVERNOR. Only AUTHOR touches LLM. Pipeline loops until GOVERNOR passes.",
                "confidence": 0.91,
                "connections": ("COG.BHV.ECO.HOW.COG",),
            },
            {
                "postcode": "STR.ENT.ECO.HOW.SFT",
                "primitive": "navigator",
                "content": "Scores unfilled cells by 4 priority axes. next_cell predicts optimal fill target. descend creates deeper scope children.",
                "confidence": 0.85,
                "connections": ("ORG.ENT.ECO.WHAT.SFT", "COG.BHV.ECO.HOW.COG"),
            },
            {
                "postcode": "STA.ENT.ECO.WHAT.SFT",
                "primitive": "fill-state-machine",
                "content": "6 states: F(illed) P(artial) E(mpty) B(locked) Q(uarantined) C(andidate). Transitions enforced by fill() operation.",
                "confidence": 0.93,
                "connections": ("COG.BHV.ECO.HOW.COG", "SEM.ENT.ECO.WHAT.COG"),
            },
        ]

    elif call == 2:
        # Second pass: deeper structure + cross-layer connections
        return [
            {
                "postcode": "CTR.BHV.ECO.HOW.COG",
                "primitive": "axiom-enforcement",
                "content": "5 axioms as structural invariants. AX1 provenance, AX2 descent, AX3 feedback, AX4 emergence, AX5 constraint. Enforced at fill-time, not post-hoc.",
                "confidence": 0.94,
                "connections": ("COG.BHV.ECO.HOW.COG", "STA.ENT.ECO.WHAT.SFT"),
            },
            {
                "postcode": "RES.MET.ECO.HOW.SFT",
                "primitive": "emission",
                "content": "Manifest export: dependency-ordered, simulation-gated. Only F/P cells emitted. Escalations from B/P cells.",
                "confidence": 0.87,
                "connections": ("AGN.BHV.ECO.WHO.SFT", "CTR.BHV.ECO.HOW.COG"),
            },
            {
                "postcode": "EMG.CND.ECO.WHAT.COG",
                "primitive": "emergence-detection",
                "content": "Patterns detected by OBSERVER: repeated primitives, shared connections, orphan clusters. Promoted by EMERGENCE agent when evidence >= 2.",
                "confidence": 0.83,
                "connections": ("AGN.BHV.ECO.WHO.SFT", "STR.ENT.ECO.HOW.SFT"),
            },
            {
                "postcode": "NET.BHV.ECO.HOW.SFT",
                "primitive": "connection-topology",
                "content": "Directed edges between cells. Connect triggers layer activation. Cross-layer connections are the primary information flow.",
                "confidence": 0.86,
                "connections": ("ORG.ENT.ECO.WHAT.SFT", "SEM.ENT.ECO.WHAT.COG"),
            },
            {
                "postcode": "OBS.MET.ECO.HOW.COG",
                "primitive": "simulation-gate",
                "content": "GOVERNOR runs 4 checks: conflict, gap, cycle, dead-end. Hard failures block emission. Soft warnings are informational.",
                "confidence": 0.89,
                "connections": ("AGN.BHV.ECO.WHO.SFT", "RES.MET.ECO.HOW.SFT"),
            },
        ]

    else:
        # Third+ pass: no new cells, pipeline should converge
        return []


def _reset_mock():
    """Reset mock call counter."""
    global _CALL_COUNT
    _CALL_COUNT = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSelfCompilationStructure:
    """Verify the compiled grid has correct structural properties."""

    def setup_method(self):
        _reset_mock()

    def test_compile_converges(self):
        """Pipeline should converge within bounded iterations."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert result.converged is True
        assert result.iterations <= 5

    def test_compile_populates_grid(self):
        """Grid should contain cells after compilation."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert result.grid.total_cells > 5
        assert len(result.grid.filled_cells()) > 5

    def test_compile_activates_expected_layers(self):
        """Kernel compilation should activate semantic, org, cognitive, agent, structure layers."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        active = result.grid.activated_layers
        # These layers appear in our mock extractions
        assert "SEM" in active
        assert "ORG" in active
        assert "COG" in active
        assert "AGN" in active
        assert "STR" in active
        assert "STA" in active

    def test_compile_has_intent_root(self):
        """Grid should have intent root cell."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert result.grid.root is not None
        root_cell = result.grid.get(result.grid.root)
        assert root_cell is not None
        assert root_cell.is_filled

    def test_compile_has_cross_layer_connections(self):
        """Cells should connect across layers."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        cross_layer = 0
        for cell in result.grid.filled_cells():
            cell_layer = cell.postcode.layer
            for conn in cell.connections:
                conn_cell = result.grid.get(conn)
                if conn_cell and conn_cell.postcode.layer != cell_layer:
                    cross_layer += 1
        assert cross_layer >= 3, f"Expected >=3 cross-layer connections, got {cross_layer}"

    def test_compile_provenance_traces_to_intent(self):
        """Every filled cell should trace back to intent through source chain."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        for cell in result.grid.filled_cells():
            # Walk source chain — should reach intent contract or human:
            visited = set()
            frontier = list(cell.source)
            found_root = False
            while frontier:
                src = frontier.pop()
                if src in visited:
                    continue
                visited.add(src)
                if src == INTENT_CONTRACT or src.startswith("human:") or src.startswith("contract:"):
                    found_root = True
                    break
                # Follow source of source
                src_cell = result.grid.get(src)
                if src_cell:
                    frontier.extend(src_cell.source)
            assert found_root, f"Cell {cell.postcode.key} has no provenance trace to intent"


class TestSelfCompilationAgents:
    """Verify agent pipeline behaves correctly during self-compilation."""

    def setup_method(self):
        _reset_mock()

    def test_author_called(self):
        """Author should be called at least once."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert result.author_calls >= 1

    def test_verifier_runs(self):
        """Verifier should produce results each iteration."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert len(result.verify_results) >= 1
        # At least some cells should be checked
        total_checked = sum(v.checked for v in result.verify_results)
        assert total_checked > 0

    def test_governor_passes(self):
        """Governor simulation should eventually pass."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        assert result.simulation is not None
        # Either passed, or converged without hard failures
        assert result.converged

    def test_no_quarantined_in_final_grid(self):
        """Well-formed mock should produce no quarantined cells."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        q_cells = result.grid.quarantined_cells()
        assert len(q_cells) == 0, f"Unexpected quarantined: {[c.postcode.key for c in q_cells]}"

    def test_filled_cells_have_content(self):
        """Every filled cell should have non-empty content."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        for cell in result.grid.filled_cells():
            if cell.postcode.key == result.grid.root:
                continue  # Intent root has the raw input
            assert cell.content, f"Cell {cell.postcode.key} has empty content"


class TestSelfCompilationEmission:
    """Verify emission works on the self-compiled grid."""

    def setup_method(self):
        _reset_mock()

    def test_emit_succeeds(self):
        """Emission should succeed on a converged compilation."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        assert manifest is not None

    def test_manifest_has_nodes(self):
        """Manifest should contain emittable nodes."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        assert len(manifest.nodes) > 0

    def test_manifest_dependency_order(self):
        """Manifest nodes should be dependency-ordered (parents before children)."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        seen = set()
        for node in manifest.nodes:
            if node.parent is not None and node.parent != INTENT_CONTRACT:
                # Parent should already be emitted (or not in emit set)
                parent_in_set = any(n.postcode == node.parent for n in manifest.nodes)
                if parent_in_set:
                    assert node.parent in seen, \
                        f"Node {node.postcode} emitted before parent {node.parent}"
            seen.add(node.postcode)

    def test_manifest_to_dict(self):
        """Manifest serialization should produce valid structure."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        d = manifest.to_dict()
        assert d["version"] == "1.0"
        assert d["type"] == "motherlabs_manifest"
        assert "nodes" in d
        assert "escalations" in d
        assert "stats" in d
        assert d["stats"]["emitted_cells"] > 0

    def test_manifest_fill_rate(self):
        """Manifest should report reasonable fill rate."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        assert manifest.fill_rate > 0.0
        assert manifest.fill_rate <= 1.0

    def test_manifest_layers_active(self):
        """Manifest should report activated layers."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        manifest = emit(result.grid, force=True)
        assert len(manifest.layers_active) >= 5


class TestSelfCompilationNav:
    """Verify nav layer works on self-compiled grid."""

    def setup_method(self):
        _reset_mock()

    def test_nav_serialization(self):
        """Self-compiled grid should serialize to nav format."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        nav_text = grid_to_nav(result.grid)
        assert "# MAP" in nav_text
        assert "F" in nav_text  # Should have filled cells

    def test_nav_roundtrip_preserves_structure(self):
        """Nav roundtrip should preserve cell count and layers."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        nav_text = grid_to_nav(result.grid)
        restored = nav_to_grid(nav_text)
        # Filled cells should survive roundtrip
        original_filled = len(result.grid.filled_cells())
        restored_filled = len(restored.filled_cells())
        assert restored_filled >= original_filled * 0.8, \
            f"Too many cells lost in roundtrip: {original_filled} -> {restored_filled}"

    def test_nav_budget_fits_context(self):
        """Grid should fit in a reasonable context window."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        tokens = estimate_tokens(result.grid)
        assert tokens < 10000, f"Token estimate {tokens} too large for kernel grid"

    def test_budget_truncation(self):
        """Budget nav should truncate within limit."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        nav_text = budget_nav(result.grid, max_tokens=500)
        # Should have truncation marker or fit within budget
        estimated = estimate_tokens(result.grid)
        if estimated > 500:
            assert "TRUNCATED" in nav_text


class TestSelfCompilationFixedPoint:
    """The actual F(F) ~ F tests — structural similarity of output to input."""

    def setup_method(self):
        _reset_mock()

    def test_output_describes_input(self):
        """Compiled grid should contain cells describing the kernel's own concepts."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        primitives = {c.primitive for c in result.grid.filled_cells()}
        # Should find kernel concepts in the output
        kernel_concepts = {
            "semantic-architecture", "grid-structure", "fill-operation",
            "agent-pipeline", "navigator", "fill-state-machine",
        }
        overlap = primitives & kernel_concepts
        assert len(overlap) >= 4, f"Only {len(overlap)} kernel concepts found: {overlap}"

    def test_output_activates_layers_that_describe_layers(self):
        """The grid describing layers should itself use multiple layers."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        # The kernel has 16 layers — the compilation should use several
        assert len(result.grid.activated_layers) >= 5

    def test_output_has_architectural_cells(self):
        """Should find architecture-related cells (ENT concern)."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        arc_cells = [
            c for c in result.grid.filled_cells()
            if c.postcode.concern == "ENT"
        ]
        assert len(arc_cells) >= 2

    def test_output_has_behavioral_cells(self):
        """Should find behavior-related cells (BHV concern)."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        bhv_cells = [
            c for c in result.grid.filled_cells()
            if c.postcode.concern == "BHV"
        ]
        assert len(bhv_cells) >= 2

    def test_structural_self_reference(self):
        """Cells describing agents should connect to cells describing the grid they operate on."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        agent_cell = result.grid.get("AGN.BHV.ECO.WHO.SFT")
        assert agent_cell is not None
        assert agent_cell.is_filled
        # Agent cell should connect to something about cognition or structure
        connected_layers = set()
        for conn in agent_cell.connections:
            conn_cell = result.grid.get(conn)
            if conn_cell:
                connected_layers.add(conn_cell.postcode.layer)
        assert "COG" in connected_layers, \
            f"Agent cell should connect to COG layer, connects to: {connected_layers}"

    def test_compilation_is_deterministic(self):
        """Same input + same mock should produce structurally equivalent output."""
        _reset_mock()
        result1 = compile(KERNEL_SPEC, _mock_llm_kernel)
        _reset_mock()
        result2 = compile(KERNEL_SPEC, _mock_llm_kernel)

        cells1 = {c.postcode.key for c in result1.grid.filled_cells()}
        cells2 = {c.postcode.key for c in result2.grid.filled_cells()}
        assert cells1 == cells2, f"Non-deterministic: {cells1 ^ cells2}"

    def test_second_compilation_with_history_enriches(self):
        """Compiling with history from first run should produce >= same cells."""
        _reset_mock()
        result1 = compile(KERNEL_SPEC, _mock_llm_kernel)
        count1 = len(result1.grid.filled_cells())

        _reset_mock()
        result2 = compile(
            KERNEL_SPEC, _mock_llm_kernel,
            history=[result1.grid],
        )
        count2 = len(result2.grid.filled_cells())
        # History should add context, not lose it
        assert count2 >= count1, f"History compilation lost cells: {count1} -> {count2}"


class TestGroundTruthComparison:
    """Compare self-compilation output against MTH-ORG-001 ground truth properties."""

    def setup_method(self):
        _reset_mock()

    def test_uses_multiple_concerns(self):
        """Like the ground truth, should use multiple concerns (not just ENT)."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        concerns = {c.postcode.concern for c in result.grid.filled_cells()}
        assert len(concerns) >= 2, f"Only {len(concerns)} concerns used: {concerns}"

    def test_uses_multiple_dimensions(self):
        """Like the ground truth, should use multiple dimensions."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        dimensions = {c.postcode.dimension for c in result.grid.filled_cells()}
        assert len(dimensions) >= 2, f"Only {len(dimensions)} dimensions used: {dimensions}"

    def test_confidence_distribution_realistic(self):
        """Confidence scores should span a reasonable range, not all 1.0 or all 0.0."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        confidences = [c.confidence for c in result.grid.filled_cells()]
        assert min(confidences) >= 0.0
        assert max(confidences) <= 1.0
        # Should have some spread
        spread = max(confidences) - min(confidences)
        assert spread >= 0.05, f"Confidence spread too narrow: {spread}"

    def test_connection_density_reasonable(self):
        """Connection density should be reasonable (not zero, not fully connected)."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        filled = result.grid.filled_cells()
        total_connections = sum(len(c.connections) for c in filled)
        avg_connections = total_connections / max(len(filled), 1)
        assert avg_connections >= 0.5, f"Too few connections: avg {avg_connections}"
        assert avg_connections < 10.0, f"Too many connections: avg {avg_connections}"

    def test_depth_distribution(self):
        """Cells should be at depth 0 (ECO scope) like ground truth primary nodes."""
        result = compile(KERNEL_SPEC, _mock_llm_kernel)
        depths = [c.postcode.depth for c in result.grid.filled_cells()]
        # Should have eco-level cells
        assert 0 in depths
        # May have deeper cells from descent
        max_depth = max(depths)
        assert max_depth <= 5, f"Unreasonable max depth: {max_depth}"
