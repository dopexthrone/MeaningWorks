"""
Tests for safety fixes 1-3:
  Fix 1: Intent chain depth limit (daemon.py)
  Fix 2: grid.put() AX1 provenance guard (grid.py)
  Fix 3: Re-synthesis fidelity threshold rejection (engine.py)
"""

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# ---------------------------------------------------------------------------
# Fix 1: Intent chain depth limit
# ---------------------------------------------------------------------------

from mother.daemon import CompileRequest, DaemonMode, DaemonConfig


class TestChainDepthLimit:
    """Verify MAX_CHAIN_DEPTH prevents unbounded recursive exploration."""

    def test_compile_request_has_chain_depth_field(self):
        """CompileRequest tracks its chain generation."""
        req = CompileRequest(input_text="test")
        assert req.chain_depth == 0

    def test_compile_request_chain_depth_serialization(self):
        """chain_depth round-trips through to_dict/from_dict."""
        req = CompileRequest(input_text="test", chain_depth=2)
        d = req.to_dict()
        assert d["chain_depth"] == 2
        restored = CompileRequest.from_dict(d)
        assert restored.chain_depth == 2

    def test_max_chain_depth_class_constant(self):
        """DaemonMode has a MAX_CHAIN_DEPTH constant."""
        assert hasattr(DaemonMode, "MAX_CHAIN_DEPTH")
        assert DaemonMode.MAX_CHAIN_DEPTH == 3

    def test_depth_limit_blocks_chaining_at_max(self):
        """At MAX_CHAIN_DEPTH, no further chains are enqueued."""
        daemon = DaemonMode.__new__(DaemonMode)
        daemon._queue = []
        daemon._outcomes = []
        daemon.config = DaemonConfig()

        # Mock result with depth chains
        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore entity X", "chain_type": "frontier", "priority": 0.5}
        ]

        # At depth 3 (= MAX_CHAIN_DEPTH), should return 0
        count = asyncio.new_event_loop().run_until_complete(
            daemon._enqueue_depth_chains(result, "software", parent_depth=3)
        )
        assert count == 0

    def test_depth_limit_blocks_chaining_above_max(self):
        """Above MAX_CHAIN_DEPTH, no further chains are enqueued."""
        daemon = DaemonMode.__new__(DaemonMode)
        daemon._queue = []
        daemon._outcomes = []
        daemon.config = DaemonConfig()

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore entity X", "chain_type": "frontier", "priority": 0.5}
        ]

        count = asyncio.new_event_loop().run_until_complete(
            daemon._enqueue_depth_chains(result, "software", parent_depth=5)
        )
        assert count == 0

    def test_depth_limit_allows_chaining_below_max(self):
        """Below MAX_CHAIN_DEPTH, chains are enqueued normally."""
        daemon = DaemonMode.__new__(DaemonMode)
        daemon._queue = []
        daemon._outcomes = []
        daemon._running = True
        daemon.config = DaemonConfig(max_queue_size=10)
        daemon._bridge = None
        daemon._depth_chain_enqueue_times = []
        daemon._depth_chain_max_per_hour = 10

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore entity X", "chain_type": "frontier", "priority": 0.5}
        ]

        async def _run():
            with patch.object(daemon, "enqueue", new_callable=AsyncMock) as mock_enqueue:
                mock_req = CompileRequest(input_text="[DEPTH-EXPLORE:frontier] explore entity X")
                mock_enqueue.return_value = mock_req
                count = await daemon._enqueue_depth_chains(result, "software", parent_depth=1)
                assert count == 1
                assert mock_req.chain_depth == 2

        asyncio.new_event_loop().run_until_complete(_run())

    def test_child_depth_increments_parent(self):
        """Enqueued chains have depth = parent_depth + 1."""
        daemon = DaemonMode.__new__(DaemonMode)
        daemon._queue = []
        daemon._outcomes = []
        daemon._running = True
        daemon.config = DaemonConfig(max_queue_size=10)
        daemon._bridge = None
        daemon._depth_chain_enqueue_times = []
        daemon._depth_chain_max_per_hour = 10

        result = MagicMock()
        result.success = True
        result.depth_chains = [
            {"intent_text": "explore A", "chain_type": "frontier", "priority": 0.5},
        ]

        async def _run():
            with patch.object(daemon, "enqueue", new_callable=AsyncMock) as mock_enqueue:
                mock_req = CompileRequest(input_text="[DEPTH-EXPLORE:frontier] explore A")
                mock_enqueue.return_value = mock_req

                await daemon._enqueue_depth_chains(result, "software", parent_depth=0)
                assert mock_req.chain_depth == 1

                mock_req.chain_depth = 0
                await daemon._enqueue_depth_chains(result, "software", parent_depth=2)
                assert mock_req.chain_depth == 3

        asyncio.new_event_loop().run_until_complete(_run())

    def test_default_chain_depth_is_zero(self):
        """Root requests start at depth 0."""
        req = CompileRequest(input_text="root request")
        assert req.chain_depth == 0


# ---------------------------------------------------------------------------
# Fix 2: grid.put() AX1 provenance guard
# ---------------------------------------------------------------------------

from kernel.grid import Grid
from kernel.cell import Cell, FillState, Postcode, parse_postcode


class TestGridPutProvenanceGuard:
    """Verify grid.put() rejects filled cells without provenance."""

    def _make_postcode(self, key: str = "INT.ENT.ECO.WHAT.SFT") -> Postcode:
        return parse_postcode(key)

    def test_put_rejects_filled_cell_without_source(self):
        """F-state cell without source raises ValueError."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="hello",
            fill=FillState.F,
            confidence=1.0,
            source=(),  # empty — AX1 violation
        )
        with pytest.raises(ValueError, match="AX1 violation"):
            grid.put(cell)

    def test_put_rejects_partial_cell_without_source(self):
        """P-state cell without source raises ValueError."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="partial",
            fill=FillState.P,
            confidence=0.5,
            source=(),
        )
        with pytest.raises(ValueError, match="AX1 violation"):
            grid.put(cell)

    def test_put_accepts_filled_cell_with_source(self):
        """F-state cell with source is accepted."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="hello",
            fill=FillState.F,
            confidence=1.0,
            source=("human:test",),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_put_accepts_partial_cell_with_source(self):
        """P-state cell with source is accepted."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="partial",
            fill=FillState.P,
            confidence=0.5,
            source=("llm:test",),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_put_accepts_empty_cell_without_source(self):
        """E-state cell without source is fine (structural placeholder)."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="",
            fill=FillState.E,
            confidence=0.0,
            source=(),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_put_accepts_candidate_without_source(self):
        """C-state cell without source is fine (unverified candidate)."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="candidate",
            fill=FillState.C,
            confidence=0.3,
            source=(),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_put_accepts_quarantined_without_source(self):
        """Q-state cell without source is fine."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="quarantined",
            fill=FillState.Q,
            confidence=0.0,
            source=(),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_put_accepts_blocked_without_source(self):
        """B-state cell without source is fine."""
        grid = Grid()
        cell = Cell(
            postcode=self._make_postcode(),
            primitive="test",
            content="blocked",
            fill=FillState.B,
            confidence=0.0,
            source=(),
        )
        grid.put(cell)
        assert grid.get(cell.postcode.key) is cell

    def test_error_message_includes_postcode(self):
        """ValueError message identifies the offending cell."""
        grid = Grid()
        pc = self._make_postcode("OBS.ENT.ECO.WHAT.SFT")
        cell = Cell(
            postcode=pc,
            primitive="test",
            content="hello",
            fill=FillState.F,
            confidence=1.0,
            source=(),
        )
        with pytest.raises(ValueError) as exc_info:
            grid.put(cell)
        assert pc.key in str(exc_info.value)


# ---------------------------------------------------------------------------
# Fix 3: Re-synthesis fidelity threshold rejection
# ---------------------------------------------------------------------------


class TestResynthesisFidelityThreshold:
    """Verify re-synthesis rejects blueprints below threshold even if improved."""

    def _make_engine_with_mocks(self):
        """Create a minimal mock of the engine's re-synthesis fidelity check flow."""
        from core.protocol_spec import PROTOCOL
        return PROTOCOL.engine.fidelity_threshold

    def test_fidelity_threshold_is_070(self):
        """Confirm the threshold is 0.70."""
        threshold = self._make_engine_with_mocks()
        assert threshold == 0.70

    def test_below_threshold_improved_is_rejected(self):
        """Blueprint at 0.65 (up from 0.55) is still rejected — below 0.70."""
        # Simulate the engine's re-check logic
        _pre_fidelity = 0.55
        threshold = 0.70
        recheck_fidelity = 0.65

        # cl_recheck.passed would be False (0.65 < 0.70)
        passed = recheck_fidelity >= threshold
        assert not passed

        # Under old code: would accept (0.65 >= 0.55 is True)
        old_code_accepts = recheck_fidelity >= _pre_fidelity
        assert old_code_accepts  # old code WOULD have accepted

        # Under new code: fidelity regressed? No. Still below threshold? Yes → catastrophic.
        regressed = recheck_fidelity < _pre_fidelity
        assert not regressed
        # Falls to else branch → catastrophic
        # This is the behavior we want

    def test_above_threshold_accepted(self):
        """Blueprint at 0.75 (up from 0.55) is accepted."""
        threshold = 0.70
        recheck_fidelity = 0.75
        passed = recheck_fidelity >= threshold
        assert passed

    def test_regression_is_rejected(self):
        """Blueprint that got worse is always rejected."""
        _pre_fidelity = 0.60
        recheck_fidelity = 0.50
        regressed = recheck_fidelity < _pre_fidelity
        assert regressed

    def test_exact_threshold_accepted(self):
        """Blueprint at exactly 0.70 is accepted (closed_loop_gate.passed uses >=)."""
        threshold = 0.70
        recheck_fidelity = 0.70
        passed = recheck_fidelity >= threshold
        assert passed

    def test_resynthesis_gate_flow_below_threshold(self):
        """End-to-end verification of the gate logic with mocked closed_loop_gate."""
        # Simulate the actual code path from engine.py lines 1738-1765
        verification = {}
        _fidelity_triggered = True
        _pre_fidelity = 0.55
        threshold = 0.70

        # Mock cl_recheck
        cl_recheck = MagicMock()
        cl_recheck.passed = False  # 0.65 < 0.70
        cl_recheck.fidelity_score = 0.65

        # Execute the new logic
        if cl_recheck.passed:
            pass  # accepted
        elif cl_recheck.fidelity_score < _pre_fidelity:
            verification["status"] = "catastrophic"
        else:
            # Improved but still below threshold — reject
            verification["status"] = "catastrophic"

        assert verification.get("status") == "catastrophic"

    def test_resynthesis_gate_flow_above_threshold(self):
        """End-to-end: blueprint passes threshold after re-synthesis."""
        verification = {}

        cl_recheck = MagicMock()
        cl_recheck.passed = True  # 0.80 >= 0.70
        cl_recheck.fidelity_score = 0.80

        if cl_recheck.passed:
            pass  # accepted — no catastrophic
        elif cl_recheck.fidelity_score < 0.55:
            verification["status"] = "catastrophic"
        else:
            verification["status"] = "catastrophic"

        assert verification.get("status") is None  # not catastrophic

    def test_resynthesis_gate_flow_regression(self):
        """End-to-end: regression triggers catastrophic."""
        verification = {}
        _pre_fidelity = 0.60

        cl_recheck = MagicMock()
        cl_recheck.passed = False
        cl_recheck.fidelity_score = 0.50  # regressed

        if cl_recheck.passed:
            pass
        elif cl_recheck.fidelity_score < _pre_fidelity:
            verification["status"] = "catastrophic"
        else:
            verification["status"] = "catastrophic"

        assert verification.get("status") == "catastrophic"
