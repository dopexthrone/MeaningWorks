"""Tests for revision history as navigation signal (Build 3)."""

import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill
from kernel.navigator import score_candidates
from kernel.agents import memory


class TestAgentThreading:
    """Agent name is threaded through fill() to grid._agent_map."""

    def _make_grid(self):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        return grid

    def test_fill_records_agent(self):
        """fill() with agent param records in grid._agent_map."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=(INTENT_CONTRACT,), agent="author")
        assert grid._agent_map.get("STR.ENT.CMP.WHAT.SFT") == "author"

    def test_fill_without_agent_no_record(self):
        """fill() without agent param doesn't record."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.9, source=(INTENT_CONTRACT,))
        assert "STR.ENT.CMP.WHAT.SFT" not in grid._agent_map

    def test_verifier_records_agent(self):
        """Verifier fill calls record 'verifier' agent."""
        grid = self._make_grid()
        fill(grid, "STR.ENT.CMP.WHAT.SFT", "user", "user entity",
             0.8, source=(INTENT_CONTRACT,))
        from kernel.agents import verifier
        verifier(grid)
        assert grid._agent_map.get("STR.ENT.CMP.WHAT.SFT") == "verifier"


class TestRevisionPenalty:
    """Navigator penalizes heavily-revised cells."""

    def _make_grid_with_revised_cell(self, revision_count):
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")
        # Create target cell with many revisions
        pc = parse_postcode("STR.ENT.CMP.WHAT.SFT")
        revisions = tuple(("old_content", 0.5) for _ in range(revision_count))
        cell = Cell(
            postcode=pc, primitive="user", content="revised many times",
            fill=FillState.E, confidence=0.0,
            source=(INTENT_CONTRACT,), revisions=revisions,
        )
        grid.put(cell)
        # Create a filled cell that connects to the target
        fill(grid, "STR.REL.APP.HOW.SFT", "link", "connects to user",
             0.9, source=(INTENT_CONTRACT,),
             connections=("STR.ENT.CMP.WHAT.SFT",))
        return grid

    def test_stable_cell_no_penalty(self):
        """Cell with <= 3 revisions gets no revision penalty."""
        grid = self._make_grid_with_revised_cell(2)
        candidates = score_candidates(grid)
        target = [c for c in candidates if c.postcode_key == "STR.ENT.CMP.WHAT.SFT"]
        assert len(target) == 1
        assert "revision_penalty" not in target[0].reason

    def test_unstable_cell_penalized(self):
        """Cell with > 3 revisions gets penalized."""
        grid = self._make_grid_with_revised_cell(5)
        candidates = score_candidates(grid)
        target = [c for c in candidates if c.postcode_key == "STR.ENT.CMP.WHAT.SFT"]
        assert len(target) == 1
        assert "revision_penalty" in target[0].reason

    def test_penalty_scales_with_revisions(self):
        """More revisions = more penalty (may push below zero / off list)."""
        grid_5 = self._make_grid_with_revised_cell(5)
        grid_8 = self._make_grid_with_revised_cell(8)
        candidates_5 = [c for c in score_candidates(grid_5)
                        if c.postcode_key == "STR.ENT.CMP.WHAT.SFT"]
        candidates_8 = [c for c in score_candidates(grid_8)
                        if c.postcode_key == "STR.ENT.CMP.WHAT.SFT"]
        # Either score_8 < score_5, or cell_8 was dropped entirely
        if candidates_8:
            assert candidates_5[0].score > candidates_8[0].score
        else:
            # So heavily penalized it fell off the list entirely
            assert len(candidates_5) > 0


class TestMemoryStabilityFactor:
    """Memory agent applies stability factor from revision history."""

    def test_stable_cell_imported_at_full_decay(self):
        """Cell with no revisions imported at confidence * decay."""
        prev_grid = Grid()
        prev_grid.set_intent("prev", "INT.SEM.ECO.WHY.SFT", "prev-intent")
        fill(prev_grid, "STR.ENT.CMP.WHAT.SFT", "user", "stable entity",
             0.9, source=(INTENT_CONTRACT,))

        new_grid = Grid()
        new_grid.set_intent("new", "INT.SEM.ECO.WHY.SFT", "new-intent")
        memory(new_grid, [prev_grid], confidence_decay=0.7)

        cell = new_grid.get("STR.ENT.CMP.WHAT.SFT")
        assert cell is not None
        # stability=1.0, decay=0.7, original=0.9 → 0.63
        assert abs(cell.confidence - 0.63) < 0.05

    def test_unstable_cell_imported_at_reduced_confidence(self):
        """Cell with revisions imported at lower confidence."""
        prev_grid = Grid()
        prev_grid.set_intent("prev", "INT.SEM.ECO.WHY.SFT", "prev-intent")
        # Create cell with 3 revisions
        pc = parse_postcode("STR.ENT.CMP.WHAT.SFT")
        cell = Cell(
            postcode=pc, primitive="user", content="unstable entity",
            fill=FillState.F, confidence=0.9,
            source=(INTENT_CONTRACT,),
            revisions=(("v1", 0.5), ("v2", 0.6), ("v3", 0.7)),
        )
        prev_grid.put(cell)

        new_grid = Grid()
        new_grid.set_intent("new", "INT.SEM.ECO.WHY.SFT", "new-intent")
        memory(new_grid, [prev_grid], confidence_decay=0.7)

        imported = new_grid.get("STR.ENT.CMP.WHAT.SFT")
        assert imported is not None
        # stability=1/(1+3)=0.25, decay=0.7, original=0.9 → 0.1575
        assert imported.confidence < 0.63  # Less than stable version
