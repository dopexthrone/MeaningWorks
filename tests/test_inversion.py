"""Tests for the L2→L3 self-improvement loop closure (The Inversion).

Verifies:
1. Compilation failures create targeted self-improvement goals
2. Immediate post-compile goal sync fires in both success and failure paths
3. End-to-end: CompilationOutcome → analyze → prompt_patch → state.known
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from mother.goals import GoalStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screen(tmp_path):
    """Create a minimal ChatScreen with mocked internals."""
    from mother.config import MotherConfig
    from mother.screens.chat import ChatScreen

    config = MotherConfig(autonomous_enabled=True)
    screen = ChatScreen(config=config)

    db_path = tmp_path / "test.db"
    screen._bridge = MagicMock()
    screen._bridge.get_learning_context = MagicMock(return_value={
        "trends_line": "",
        "failure_line": "",
        "chronic_weak": [],
        "dimension_averages": {},
    })
    screen._bridge.get_rejection_hints = MagicMock(return_value=[])
    screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)
    screen._bridge.reset_goal_stall = AsyncMock()
    screen._bridge.mark_goal_stuck = AsyncMock()
    screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
    screen._bridge.broadcast_corpus_sync = MagicMock()
    screen._bridge.sync_goals_to_store = MagicMock(return_value=0)

    screen._store = MagicMock()
    screen._store._path = db_path

    screen._route_output = MagicMock()
    screen._journal = None
    screen._compilation_scores = []
    screen._failure_reasons = []
    screen._runway_months = 18
    screen._burn_rate = 0.0
    screen._learning_patterns = {}
    screen._working_memory_summary = ""
    screen._autonomous_actions_count = 0
    screen._config = config

    return screen, db_path


def _goal_dict(goal_id=1, description="Build a dashboard"):
    return {"goal_id": goal_id, "description": description}


# ---------------------------------------------------------------------------
# TestFailureToGoal
# ---------------------------------------------------------------------------

class TestFailureToGoal:
    """Verify that compilation failures create targeted self-improvement goals."""

    def test_failure_creates_targeted_goal(self, tmp_path):
        """A failed compile creates a new goal in GoalStore."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        asyncio.run(screen._compile_goal(db_path, _goal_dict(), Stance.ACT))

        # Verify a goal was created
        gs = GoalStore(db_path)
        active = gs.active(limit=20)
        gs.close()

        descs = [g.description for g in active]
        assert any("compilation failure" in d.lower() or "failed" in d.lower() for d in descs), \
            f"Expected failure goal, got: {descs}"

    def test_failure_goal_references_weakness(self, tmp_path):
        """When chronic weakness exists, the goal description references it."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.get_learning_context = MagicMock(return_value={
            "trends_line": "declining",
            "failure_line": "3 failures in 5 compiles",
            "chronic_weak": ["coherence"],
            "dimension_averages": {"coherence": 42.0},
        })
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        asyncio.run(
            screen._compile_goal(db_path, _goal_dict(), Stance.ACT)
        )

        gs = GoalStore(db_path)
        active = gs.active(limit=20)
        gs.close()

        descs = [g.description for g in active]
        assert any("coherence" in d.lower() for d in descs), \
            f"Expected 'coherence' in goal desc, got: {descs}"
        assert any("42%" in d for d in descs), \
            f"Expected '42%' score in goal desc, got: {descs}"

    def test_failure_goal_deduplicates(self, tmp_path):
        """Same failure twice doesn't create duplicate goals."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        goal = _goal_dict(goal_id=1, description="Build a dashboard")

        # First failure
        asyncio.run(
            screen._compile_goal(db_path, goal, Stance.ACT)
        )
        # Second failure (same goal)
        asyncio.run(
            screen._compile_goal(db_path, goal, Stance.ACT)
        )

        gs = GoalStore(db_path)
        active = gs.active(limit=20)
        gs.close()

        failure_goals = [g for g in active if "failed" in g.description.lower() or "failure" in g.description.lower()]
        assert len(failure_goals) == 1, \
            f"Expected exactly 1 failure goal (dedup), got {len(failure_goals)}: {[g.description for g in failure_goals]}"

    def test_failure_priority_escalates(self, tmp_path):
        """attempt_count=1 → normal priority, attempt_count=2 → high priority."""
        # Test normal priority (attempt 1)
        screen1, db_path1 = _make_screen(tmp_path / "db1")
        (tmp_path / "db1").mkdir(exist_ok=True)
        db_path1 = tmp_path / "db1" / "test.db"
        screen1._store._path = db_path1
        screen1._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen1._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        asyncio.run(
            screen1._compile_goal(db_path1, _goal_dict(goal_id=10), Stance.ACT)
        )

        gs1 = GoalStore(db_path1)
        active1 = gs1.active(limit=20)
        gs1.close()
        failure_goals1 = [g for g in active1 if "failed" in g.description.lower() or "failure" in g.description.lower()]
        assert len(failure_goals1) == 1
        assert failure_goals1[0].priority == "normal"

        # Test high priority (attempt 2)
        screen2, db_path2 = _make_screen(tmp_path / "db2")
        (tmp_path / "db2").mkdir(exist_ok=True)
        db_path2 = tmp_path / "db2" / "test.db"
        screen2._store._path = db_path2
        screen2._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen2._bridge.increment_goal_attempt = AsyncMock(return_value=2)

        asyncio.run(
            screen2._compile_goal(db_path2, _goal_dict(goal_id=20), Stance.ACT)
        )

        gs2 = GoalStore(db_path2)
        active2 = gs2.active(limit=20)
        gs2.close()
        failure_goals2 = [g for g in active2 if "failed" in g.description.lower() or "failure" in g.description.lower()]
        assert len(failure_goals2) == 1
        assert failure_goals2[0].priority == "high"

    def test_stuck_goal_spawns_improvement(self, tmp_path):
        """When max attempts reached, goal gets stuck but improvement goal exists."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=3)
        screen._config.max_goal_attempts = 3

        from mother.stance import Stance
        asyncio.run(
            screen._compile_goal(db_path, _goal_dict(goal_id=5), Stance.ACT)
        )

        # mark_goal_stuck should have been called
        screen._bridge.mark_goal_stuck.assert_called_once()

        # Improvement goal should exist
        gs = GoalStore(db_path)
        active = gs.active(limit=20)
        gs.close()

        failure_goals = [g for g in active if "failed" in g.description.lower() or "failure" in g.description.lower()]
        assert len(failure_goals) >= 1, "Improvement goal should exist even when original is stuck"


# ---------------------------------------------------------------------------
# TestImmediateSync
# ---------------------------------------------------------------------------

class TestImmediateSync:
    """Verify sync_goals_to_store is called immediately after compile."""

    def test_success_triggers_sync(self, tmp_path):
        """Successful compile triggers immediate goal sync."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value={
            "plan_id": 1,
            "step_names": ["step1", "step2"],
            "trust_score": 85.0,
            "total_steps": 2,
            "domain": "software",
        })
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        asyncio.run(
            screen._compile_goal(db_path, _goal_dict(), Stance.ACT)
        )

        screen._bridge.sync_goals_to_store.assert_called()

    def test_failure_triggers_sync(self, tmp_path):
        """Failed compile triggers immediate goal sync."""
        screen, db_path = _make_screen(tmp_path)
        screen._bridge.compile_goal_to_plan = AsyncMock(return_value=None)
        screen._bridge.increment_goal_attempt = AsyncMock(return_value=1)

        from mother.stance import Stance
        asyncio.run(
            screen._compile_goal(db_path, _goal_dict(), Stance.ACT)
        )

        screen._bridge.sync_goals_to_store.assert_called()


# ---------------------------------------------------------------------------
# TestEndToEndLoop
# ---------------------------------------------------------------------------

class TestEndToEndLoop:
    """Verify the full L2→L3 chain: outcome → analysis → prompt patch."""

    def test_outcome_feeds_prompt_patch(self):
        """CompilationOutcome → analyze_outcomes → prompt_patch → non-empty string."""
        from mother.governor_feedback import (
            CompilationOutcome,
            analyze_outcomes,
            generate_compiler_prompt_patch,
        )

        outcomes = [
            CompilationOutcome(
                compile_id="c1",
                input_summary="Build a dashboard",
                trust_score=45.0,
                completeness=60.0,
                consistency=40.0,
                coherence=35.0,
                traceability=50.0,
                actionability=45.0,
                specificity=45.0,
                codegen_readiness=45.0,
                component_count=5,
                rejected=True,
                rejection_reason="Low coherence",
                domain="software",
            ),
            CompilationOutcome(
                compile_id="c2",
                input_summary="Build a login",
                trust_score=80.0,
                completeness=90.0,
                consistency=85.0,
                coherence=75.0,
                traceability=88.0,
                actionability=80.0,
                specificity=80.0,
                codegen_readiness=80.0,
                component_count=3,
                rejected=False,
                domain="software",
            ),
        ]

        report = analyze_outcomes(outcomes)
        patch = generate_compiler_prompt_patch(report)

        assert report.outcomes_analyzed == 2
        assert report.rejection_rate > 0
        assert isinstance(patch, str)
        # The patch should contain self-improvement directives when there are weaknesses
        if report.weaknesses:
            assert "Self-Improvement" in patch or "Compiler" in patch
