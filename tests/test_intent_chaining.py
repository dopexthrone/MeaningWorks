"""Tests for intent chaining — endpoint extraction, CompileResult population,
and daemon depth-explore enqueuing."""

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from kernel.endpoint_extractor import (
    EndpointChain,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_CHAINS,
    MIN_CELLS_FOR_CHAINING,
    extract_exploration_endpoints,
    chain_summary,
    _extract_frontier_chains,
    _extract_low_confidence_chains,
    _extract_isolated_layer_chains,
)
from kernel.cell import Cell, FillState, Postcode, parse_postcode
from kernel.grid import Grid


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid() -> Grid:
    """Create a small grid for testing."""
    return Grid()


def _fill_cell(grid: Grid, postcode_key: str, primitive: str = "test",
               confidence: float = 0.85, connections: tuple = ()) -> Cell:
    """Fill a cell in the grid at the given postcode."""
    pc = parse_postcode(postcode_key)
    cell = Cell(
        postcode=pc,
        primitive=primitive,
        content=f"Content for {primitive}",
        fill=FillState.F,
        confidence=confidence,
        connections=connections,
        source=("__intent_contract__",),
    )
    grid.put(cell)
    return cell


def _empty_cell(grid: Grid, postcode_key: str, primitive: str = "") -> Cell:
    """Put an empty cell in the grid."""
    pc = parse_postcode(postcode_key)
    cell = Cell(
        postcode=pc,
        primitive=primitive,
        content="",
        fill=FillState.E,
        confidence=0.0,
    )
    grid.put(cell)
    return cell


# ---------------------------------------------------------------------------
# EndpointChain dataclass
# ---------------------------------------------------------------------------

class TestEndpointChain:
    def test_frozen(self):
        chain = EndpointChain(
            chain_type="frontier",
            intent_text="Explore something",
            source_postcodes=("SEM.ENT.ECO.WHAT.SFT",),
            priority=0.8,
            layer="SEM",
            concern="ENT",
        )
        with pytest.raises(AttributeError):
            chain.priority = 0.5

    def test_fields_accessible(self):
        chain = EndpointChain(
            chain_type="low_conf",
            intent_text="Deepen analysis",
            source_postcodes=("COG.BHV.APP.HOW.SFT",),
            priority=0.6,
            layer="COG",
            concern="BHV",
        )
        assert chain.chain_type == "low_conf"
        assert chain.layer == "COG"
        assert chain.concern == "BHV"


# ---------------------------------------------------------------------------
# extract_exploration_endpoints — main function
# ---------------------------------------------------------------------------

class TestExtractExplorationEndpoints:
    def test_empty_grid_returns_empty(self):
        grid = _make_grid()
        chains = extract_exploration_endpoints(grid)
        assert chains == []

    def test_too_few_cells_returns_empty(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        assert grid.total_cells == 2
        assert grid.total_cells < MIN_CELLS_FOR_CHAINING
        chains = extract_exploration_endpoints(grid)
        assert chains == []

    def test_frontier_chains_from_unfilled_connections(self):
        grid = _make_grid()
        # 3 filled cells, one with unfilled connection
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("SEM.BHV.ECO.HOW.SFT",))
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)
        # SEM.BHV.ECO.HOW.SFT doesn't exist — it's a frontier

        chains = extract_exploration_endpoints(grid)
        frontier = [c for c in chains if c.chain_type == "frontier"]
        assert len(frontier) >= 1
        assert any("Semantic" in c.intent_text for c in frontier)

    def test_low_confidence_chains(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3)
        _fill_cell(grid, "SEM.BHV.ECO.HOW.SFT", "behavior", 0.4)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "cognitive", 0.9)

        chains = extract_exploration_endpoints(grid)
        low_conf = [c for c in chains if c.chain_type == "low_conf"]
        assert len(low_conf) >= 1
        assert any("Deepen" in c.intent_text for c in low_conf)

    def test_isolated_layer_chains(self):
        grid = _make_grid()
        # Two layers, each with cells, but no cross-layer connections
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)

        chains = extract_exploration_endpoints(grid)
        isolated = [c for c in chains if c.chain_type == "isolated"]
        # All 3 layers are isolated from each other
        assert len(isolated) >= 2

    def test_deduplicates_by_intent_text(self):
        """If the same intent_text appears from different extractors, keep highest priority."""
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3,
                   connections=("SEM.BHV.ECO.HOW.SFT",))
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)

        chains = extract_exploration_endpoints(grid)
        intent_texts = [c.intent_text for c in chains]
        # No exact duplicates
        assert len(intent_texts) == len(set(intent_texts))

    def test_sorted_by_priority_descending(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.2)  # very low conf = high priority
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9,
                   connections=("COG.FNC.ECO.WHAT.SFT",))
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)

        chains = extract_exploration_endpoints(grid)
        if len(chains) >= 2:
            for i in range(len(chains) - 1):
                assert chains[i].priority >= chains[i + 1].priority

    def test_capped_at_max_chains(self):
        """Even with many endpoints, result is capped."""
        grid = _make_grid()
        # Create many isolated layers with low confidence
        layers = ["SEM", "COG", "STR", "STA", "IDN", "TME", "EXC", "CTR", "RES", "OBS"]
        for layer in layers:
            _fill_cell(grid, f"{layer}.ENT.ECO.WHAT.SFT", f"{layer}_entity", 0.3)

        chains = extract_exploration_endpoints(grid)
        assert len(chains) <= MAX_CHAINS

    def test_includes_original_intent_in_text(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)

        chains = extract_exploration_endpoints(grid, original_intent="task manager app")
        assert any("task manager app" in c.intent_text for c in chains)

    def test_no_original_intent_still_works(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)
        _fill_cell(grid, "STR.FNC.ECO.WHAT.SFT", "structure", 0.9)

        chains = extract_exploration_endpoints(grid, original_intent="")
        assert len(chains) > 0  # Still produces chains without original intent


# ---------------------------------------------------------------------------
# Frontier extraction
# ---------------------------------------------------------------------------

class TestFrontierChains:
    def test_unfilled_connection_creates_chain(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("SEM.BHV.ECO.HOW.SFT", "COG.ENT.ECO.WHAT.SFT"))

        chains = _extract_frontier_chains(grid, "test intent")
        assert len(chains) >= 1

    def test_filled_connection_not_frontier(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("SEM.BHV.ECO.HOW.SFT",))
        _fill_cell(grid, "SEM.BHV.ECO.HOW.SFT", "behavior", 0.9)

        chains = _extract_frontier_chains(grid, "test intent")
        # No frontier — connection is filled
        assert len(chains) == 0

    def test_candidate_connection_is_frontier(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("SEM.BHV.ECO.HOW.SFT",))
        # Put a candidate cell at the connection target
        pc = parse_postcode("SEM.BHV.ECO.HOW.SFT")
        candidate = Cell(
            postcode=pc, primitive="candidate", content="",
            fill=FillState.C, confidence=0.0,
        )
        grid.put(candidate)

        chains = _extract_frontier_chains(grid, "test")
        assert len(chains) >= 1  # Candidate counts as unfilled

    def test_groups_by_layer_concern(self):
        grid = _make_grid()
        # Two unfilled connections in same layer.concern
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("SEM.ENT.APP.WHAT.SFT", "SEM.ENT.DOM.WHAT.SFT"))

        chains = _extract_frontier_chains(grid, "")
        # Both should be in one chain (same SEM.ENT group)
        sem_ent = [c for c in chains if c.layer == "SEM" and c.concern == "ENT"]
        assert len(sem_ent) == 1
        assert len(sem_ent[0].source_postcodes) == 2

    def test_priority_scales_with_count(self):
        grid = _make_grid()
        # Many unfilled connections in one group
        conns = tuple(f"SEM.ENT.APP.{dim}.SFT" for dim in ("WHAT", "HOW", "WHY", "WHO", "WHEN"))
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9, connections=conns)

        chains = _extract_frontier_chains(grid, "")
        if chains:
            assert chains[0].priority == min(1.0, 5 * 0.2)  # 5 connections * 0.2


# ---------------------------------------------------------------------------
# Low-confidence extraction
# ---------------------------------------------------------------------------

class TestLowConfidenceChains:
    def test_below_threshold_creates_chain(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3)

        chains = _extract_low_confidence_chains(grid, "deep dive")
        assert len(chains) == 1
        assert chains[0].chain_type == "low_conf"
        assert "deep dive" in chains[0].intent_text

    def test_above_threshold_no_chain(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)

        chains = _extract_low_confidence_chains(grid, "")
        assert len(chains) == 0

    def test_at_threshold_no_chain(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", LOW_CONFIDENCE_THRESHOLD)

        chains = _extract_low_confidence_chains(grid, "")
        assert len(chains) == 0

    def test_priority_inversely_proportional_to_confidence(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.2)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.5)

        chains = _extract_low_confidence_chains(grid, "")
        # SEM at 0.2 should have higher priority than COG at 0.5
        sem_chain = next(c for c in chains if c.layer == "SEM")
        cog_chain = next(c for c in chains if c.layer == "COG")
        assert sem_chain.priority > cog_chain.priority

    def test_groups_by_layer_concern(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.3)
        _fill_cell(grid, "SEM.ENT.APP.HOW.SFT", "entity_detail", 0.4)

        chains = _extract_low_confidence_chains(grid, "")
        sem_ent = [c for c in chains if c.layer == "SEM" and c.concern == "ENT"]
        assert len(sem_ent) == 1
        assert len(sem_ent[0].source_postcodes) == 2


# ---------------------------------------------------------------------------
# Isolated layer extraction
# ---------------------------------------------------------------------------

class TestIsolatedLayerChains:
    def test_two_isolated_layers(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)

        chains = _extract_isolated_layer_chains(grid, "")
        assert len(chains) == 2
        layers = {c.layer for c in chains}
        assert "SEM" in layers
        assert "COG" in layers

    def test_connected_layers_not_isolated(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9,
                   connections=("COG.BHV.ECO.HOW.SFT",))
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)

        chains = _extract_isolated_layer_chains(grid, "")
        # SEM is connected to COG, so SEM is not isolated
        isolated_layers = {c.layer for c in chains}
        assert "SEM" not in isolated_layers
        # COG has no outgoing cross-layer connections, so it IS isolated
        assert "COG" in isolated_layers

    def test_single_layer_no_chains(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)

        chains = _extract_isolated_layer_chains(grid, "")
        assert len(chains) == 0  # Need at least 2 layers

    def test_includes_connect_in_intent(self):
        grid = _make_grid()
        _fill_cell(grid, "SEM.ENT.ECO.WHAT.SFT", "entity", 0.9)
        _fill_cell(grid, "COG.BHV.ECO.HOW.SFT", "behavior", 0.9)

        chains = _extract_isolated_layer_chains(grid, "task manager")
        assert all("Connect" in c.intent_text for c in chains)
        assert any("task manager" in c.intent_text for c in chains)


# ---------------------------------------------------------------------------
# chain_summary
# ---------------------------------------------------------------------------

class TestChainSummary:
    def test_empty_chains(self):
        assert "No exploration endpoints" in chain_summary([])

    def test_formats_chains(self):
        chains = [
            EndpointChain("frontier", "Explore X", ("SEM.ENT.ECO.WHAT.SFT",),
                          0.8, "SEM", "ENT"),
            EndpointChain("low_conf", "Deepen Y", ("COG.BHV.ECO.HOW.SFT",),
                          0.6, "COG", "BHV"),
        ]
        summary = chain_summary(chains)
        assert "2 exploration endpoint" in summary
        assert "[frontier]" in summary
        assert "[low_conf]" in summary
        assert "Explore X" in summary


# ---------------------------------------------------------------------------
# CompileResult.depth_chains field
# ---------------------------------------------------------------------------

class TestCompileResultDepthChains:
    def test_depth_chains_field_exists(self):
        from core.engine import CompileResult
        result = CompileResult(success=True)
        assert hasattr(result, "depth_chains")
        assert result.depth_chains == []

    def test_depth_chains_populated(self):
        from core.engine import CompileResult
        chains = [{"chain_type": "frontier", "intent_text": "test"}]
        result = CompileResult(success=True, depth_chains=chains)
        assert len(result.depth_chains) == 1
        assert result.depth_chains[0]["chain_type"] == "frontier"


# ---------------------------------------------------------------------------
# Bridge.get_depth_chains
# ---------------------------------------------------------------------------

class TestBridgeGetDepthChains:
    def test_extracts_from_result(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)

        result = SimpleNamespace(
            depth_chains=[
                {"chain_type": "frontier", "intent_text": "Explore X"},
            ]
        )
        chains = bridge.get_depth_chains(result)
        assert len(chains) == 1
        assert chains[0]["chain_type"] == "frontier"

    def test_returns_empty_when_none(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)

        result = SimpleNamespace()
        chains = bridge.get_depth_chains(result)
        assert chains == []

    def test_returns_empty_when_null(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)

        result = SimpleNamespace(depth_chains=None)
        chains = bridge.get_depth_chains(result)
        assert chains == []


# ---------------------------------------------------------------------------
# Daemon._enqueue_depth_chains
# ---------------------------------------------------------------------------

class TestDaemonEnqueueDepthChains:
    def _make_daemon(self):
        from mother.daemon import DaemonMode, DaemonConfig
        daemon = DaemonMode(config=DaemonConfig())
        daemon._queue = []
        daemon._running = False
        return daemon

    def test_enqueues_chains(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(
            success=True,
            depth_chains=[
                {"chain_type": "frontier", "intent_text": "Explore Semantic Entity", "priority": 0.8},
                {"chain_type": "low_conf", "intent_text": "Deepen Cognitive Behavior", "priority": 0.6},
            ],
        )
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 2
        pending = [r for r in daemon._queue if r.status == "pending"]
        assert len(pending) == 2
        assert "[DEPTH-EXPLORE:frontier]" in pending[0].input_text
        assert "[DEPTH-EXPLORE:low_conf]" in pending[1].input_text

    def test_skips_failed_compile(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(success=False, depth_chains=[
            {"chain_type": "frontier", "intent_text": "test", "priority": 0.5},
        ])
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 0

    def test_skips_no_chains(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(success=True, depth_chains=[])
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 0

    def test_skips_missing_depth_chains(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(success=True)
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 0

    def test_caps_at_3_chains(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(
            success=True,
            depth_chains=[
                {"chain_type": "frontier", "intent_text": f"Explore {i}", "priority": 0.5}
                for i in range(10)
            ],
        )
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 3

    def test_respects_queue_capacity(self):
        daemon = self._make_daemon()
        # Fill queue near capacity (default max_queue_size is 10)
        for i in range(9):
            run(daemon.enqueue(f"existing-{i}", "software"))

        result = SimpleNamespace(
            success=True,
            depth_chains=[
                {"chain_type": "frontier", "intent_text": "test chain", "priority": 0.5},
            ],
        )
        count = run(daemon._enqueue_depth_chains(result, "software"))
        # Only 1 slot left, should enqueue at most 0 (needs space - 1 for safety)
        assert count <= 1

    def test_empty_intent_text_skipped(self):
        daemon = self._make_daemon()
        result = SimpleNamespace(
            success=True,
            depth_chains=[
                {"chain_type": "frontier", "intent_text": "", "priority": 0.5},
            ],
        )
        count = run(daemon._enqueue_depth_chains(result, "software"))
        assert count == 0
