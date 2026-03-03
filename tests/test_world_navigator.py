"""Tests for world-model navigator scoring in kernel/navigator.py."""

import time
import pytest
from kernel.cell import Cell, FillState, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT
from kernel.ops import fill
from kernel.navigator import score_world_candidates, ScoredCell
from kernel.world_grid import bootstrap_world_grid


class TestScoreWorldCandidates:
    """score_world_candidates() tests."""

    def _make_world_grid(self):
        """Create a world grid with some filled cells."""
        grid = bootstrap_world_grid()
        now = time.time()

        # Fill observation cells
        fill(grid, "OBS.ENV.APP.WHAT.USR", "VS Code",
             "Editor open", 0.8,
             source=(f"observation:screen:{now}",))
        fill(grid, "OBS.USR.APP.WHAT.USR", "user speaking",
             "about code", 0.7,
             source=(f"observation:speech:{now}",))

        # Fill a project cell
        fill(grid, "INT.PRJ.APP.WHAT.USR", "my-app",
             "Task manager app", 0.9,
             source=(INTENT_CONTRACT,))

        return grid, now

    def test_returns_scored_cells(self):
        grid, _ = self._make_world_grid()
        candidates = score_world_candidates(grid)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, ScoredCell)

    def test_sorted_by_score_descending(self):
        grid, _ = self._make_world_grid()
        candidates = score_world_candidates(grid)
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                assert candidates[i].score >= candidates[i + 1].score

    def test_staleness_boosts_stale_cells(self):
        grid, now = self._make_world_grid()

        # Make screen observation stale (2 minutes old)
        staleness_map = {
            "OBS.ENV.APP.WHAT.USR": 120.0,
        }

        candidates = score_world_candidates(grid, staleness_map=staleness_map)
        screen_candidates = [c for c in candidates if c.postcode_key == "OBS.ENV.APP.WHAT.USR"]
        assert len(screen_candidates) >= 1
        # Should have a staleness reason
        assert any("stale" in c.reason for c in screen_candidates)

    def test_user_facing_boost(self):
        grid, _ = self._make_world_grid()
        candidates = score_world_candidates(grid)

        # USR/ENV concern cells should get user_facing boost
        user_facing = [c for c in candidates if "user_facing" in c.reason]
        assert len(user_facing) > 0

    def test_active_project_boost(self):
        grid, _ = self._make_world_grid()

        # Mark a cell as belonging to active project
        fill(grid, "INT.ENT.APP.WHAT.SFT", "TaskService",
             "Main service", 0.85,
             source=(INTENT_CONTRACT, "observation:compile:proj-123"))

        candidates = score_world_candidates(grid, active_project="proj-123")
        proj_candidates = [c for c in candidates if "active_project" in c.reason]
        assert len(proj_candidates) > 0

    def test_recently_filled_penalty(self):
        grid, _ = self._make_world_grid()

        # Without penalty
        candidates_before = score_world_candidates(grid)
        screen_before = next(
            (c for c in candidates_before if c.postcode_key == "OBS.ENV.APP.WHAT.USR"),
            None
        )

        # With penalty
        candidates_after = score_world_candidates(
            grid,
            recently_filled=frozenset({"OBS.ENV.APP.WHAT.USR"}),
        )
        screen_after = next(
            (c for c in candidates_after if c.postcode_key == "OBS.ENV.APP.WHAT.USR"),
            None
        )

        # Score should be lower after penalty
        if screen_before and screen_after:
            assert screen_after.score < screen_before.score

    def test_empty_grid_returns_some_candidates(self):
        grid = bootstrap_world_grid()
        candidates = score_world_candidates(grid)
        # Bootstrapped grid has some structure — may have candidates from cross-layer gaps
        assert isinstance(candidates, list)

    def test_no_negative_scores_in_output(self):
        grid, _ = self._make_world_grid()
        candidates = score_world_candidates(grid)
        for c in candidates:
            assert c.score > 0

    def test_includes_base_navigator_results(self):
        grid, _ = self._make_world_grid()

        # Add unfilled connection (base navigator priority 1)
        cell = grid.get("INT.PRJ.APP.WHAT.USR")
        from dataclasses import replace
        connected = replace(cell, connections=("INT.TSK.APP.WHAT.USR",))
        grid.put(connected)

        candidates = score_world_candidates(grid)
        # The unfilled connection target should be in results
        tsk_candidates = [c for c in candidates if c.postcode_key == "INT.TSK.APP.WHAT.USR"]
        assert len(tsk_candidates) >= 1

    def test_staleness_capped_at_3x(self):
        """Staleness score should not grow unboundedly."""
        grid, _ = self._make_world_grid()

        # Very old observation (1 hour)
        staleness_map = {"OBS.ENV.APP.WHAT.USR": 3600.0}
        candidates = score_world_candidates(grid, staleness_map=staleness_map)
        screen = next(
            (c for c in candidates if c.postcode_key == "OBS.ENV.APP.WHAT.USR"),
            None
        )
        # Score should be present but bounded
        if screen:
            # 6.0 * min(3600/60, 3.0) = 6.0 * 3.0 = 18.0 max staleness component
            assert screen.score < 100


class TestScoreWorldCandidatesEdgeCases:
    """Edge cases for score_world_candidates()."""

    def test_none_staleness_map(self):
        grid = bootstrap_world_grid()
        # Should not crash
        candidates = score_world_candidates(grid, staleness_map=None)
        assert isinstance(candidates, list)

    def test_empty_recently_filled(self):
        grid = bootstrap_world_grid()
        candidates = score_world_candidates(grid, recently_filled=frozenset())
        assert isinstance(candidates, list)

    def test_staleness_for_nonexistent_cell(self):
        grid = bootstrap_world_grid()
        staleness_map = {"NONEXISTENT.CELL.KEY.WHAT.USR": 120.0}
        # Should not crash — just skip
        candidates = score_world_candidates(grid, staleness_map=staleness_map)
        assert isinstance(candidates, list)

    def test_no_active_project(self):
        grid = bootstrap_world_grid()
        candidates = score_world_candidates(grid, active_project=None)
        assert isinstance(candidates, list)
