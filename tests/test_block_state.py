"""Tests for block state activation (Build 2b)."""

import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid
from kernel.ops import fill, block, FillStatus


class TestBlockFunction:
    """block() transitions cells to FillState.B."""

    def _make_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        return grid

    def test_block_filled_cell(self):
        """Blocking a filled cell transitions to B."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=("human:test",))
        result = block(grid, "STR.ENT.CMP.WHAT.SFT", reason="test_block")
        assert result.status == FillStatus.BLOCKED
        assert result.cell.is_blocked
        assert result.violation == "test_block"

    def test_block_preserves_content_in_revisions(self):
        """Blocking preserves current content in revision history."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=("human:test",))
        result = block(grid, "STR.ENT.CMP.WHAT.SFT")
        assert len(result.cell.revisions) == 1
        assert result.cell.revisions[0] == ("user entity", 0.9)

    def test_block_already_blocked(self):
        """Blocking an already-blocked cell is idempotent."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=("human:test",))
        block(grid, "STR.ENT.CMP.WHAT.SFT")
        result = block(grid, "STR.ENT.CMP.WHAT.SFT")
        assert result.status == FillStatus.BLOCKED
        assert result.violation == "already_blocked"

    def test_block_nonexistent_cell(self):
        """Blocking a cell that doesn't exist creates a blocked cell."""
        grid = self._make_grid()
        result = block(grid, "STR.ENT.CMP.WHAT.SFT")
        assert result.status == FillStatus.BLOCKED

    def test_blocked_cell_rejects_fill(self):
        """AX5: fill() on a blocked cell returns BLOCKED."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=("human:test",))
        block(grid, "STR.ENT.CMP.WHAT.SFT")
        result = fill(grid, "STR.ENT.CMP.WHAT.SFT", "user2", "new content",
                      0.9, source=("human:test",))
        assert result.status == FillStatus.BLOCKED

    def test_block_sets_confidence_zero(self):
        """Blocked cells have zero confidence."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=("human:test",))
        result = block(grid, "STR.ENT.CMP.WHAT.SFT")
        assert result.cell.confidence == 0.0


class TestVerifierBlockOnRepeatQuarantine:
    """Verifier blocks cells with 2+ prior quarantines."""

    def _make_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        return grid

    def test_repeat_quarantine_triggers_block(self):
        """Cell with 2+ zero-confidence revisions gets blocked."""
        grid = self._make_grid()
        # Simulate: fill, quarantine (revisions capture), fill again, quarantine again, fill
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "v1",
             0.9, source=("human:test",))
        # Simulate quarantine history by creating cell with revisions
        pc = parse_postcode("STR.ENT.CMP.WHAT.SFT")
        cell_with_quarantine_history = Cell(
            postcode=pc,
            primitive="user",
            content="v3",
            fill=FillState.F,
            confidence=0.9,
            source=("human:test",),
            revisions=(("v1", 0.0), ("v2", 0.0)),  # 2 prior quarantines
        )
        grid.put(cell_with_quarantine_history)

        from kernel.agents import verifier
        result = verifier(grid)
        assert result.blocked >= 1
        cell = grid.get("STR.ENT.CMP.WHAT.SFT")
        assert cell.is_blocked

    def test_single_quarantine_no_block(self):
        """Cell with only 1 zero-confidence revision is NOT blocked."""
        grid = self._make_grid()
        pc = parse_postcode("STR.ENT.CMP.WHAT.SFT")
        cell = Cell(
            postcode=pc,
            primitive="user",
            content="v2",
            fill=FillState.F,
            confidence=0.9,
            source=("human:test",),
            revisions=(("v1", 0.0),),  # Only 1 prior quarantine
        )
        grid.put(cell)

        from kernel.agents import verifier
        result = verifier(grid)
        assert result.blocked == 0


class TestGovernorBlocksCycles:
    """Governor blocks cycle participants."""

    def test_cycle_participants_blocked(self):
        """Cells in a parent cycle get blocked."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        pc_a = parse_postcode("STR.ENT.CMP.WHAT.SFT")
        pc_b = parse_postcode("STR.BHV.CMP.HOW.SFT")

        # Create cycle: A.parent = B, B.parent = A
        cell_a = Cell(
            postcode=pc_a, primitive="a", content="test a",
            fill=FillState.F, confidence=0.9,
            parent="STR.BHV.CMP.HOW.SFT",
            source=("human:test",),
        )
        cell_b = Cell(
            postcode=pc_b, primitive="b", content="test b",
            fill=FillState.F, confidence=0.9,
            parent="STR.ENT.CMP.WHAT.SFT",
            source=("human:test",),
        )
        grid.put(cell_a)
        grid.put(cell_b)

        from kernel.agents import governor
        result = governor(grid)
        assert result.hard_failures > 0
        # At least one participant should be blocked
        a = grid.get("STR.ENT.CMP.WHAT.SFT")
        b = grid.get("STR.BHV.CMP.HOW.SFT")
        assert a.is_blocked or b.is_blocked
