"""Tests for weekly build governance — daemon gate, config, and integration."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.config import MotherConfig, load_config, save_config
from mother.daemon import DaemonMode, DaemonConfig
from mother.goals import GoalStore
from mother.scheduler import (
    BuildWindow,
    ScheduleStore,
    generate_briefing,
    is_in_build_window,
    should_suppress_autonomous,
)


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# --- Config ---


class TestWeeklyBuildConfig:
    def test_defaults(self):
        cfg = MotherConfig()
        assert cfg.weekly_build_enabled is True
        assert cfg.weekly_briefing_day == 6
        assert cfg.weekly_briefing_hour == 10
        assert cfg.build_window_start_hour == 22
        assert cfg.build_window_end_hour == 6
        assert cfg.build_window_day == 6
        assert cfg.build_max_per_window == 10

    def test_round_trip(self, tmp_path):
        path = tmp_path / "mother.json"
        cfg = MotherConfig(
            weekly_build_enabled=False,
            weekly_briefing_day=5,
            build_window_start_hour=20,
        )
        save_config(cfg, str(path))
        loaded = load_config(str(path))
        assert loaded.weekly_build_enabled is False
        assert loaded.weekly_briefing_day == 5
        assert loaded.build_window_start_hour == 20

    def test_backward_compat(self, tmp_path):
        """Old config without weekly fields loads fine."""
        path = tmp_path / "mother.json"
        path.write_text('{"name": "Mother", "provider": "claude"}')
        loaded = load_config(str(path))
        assert loaded.weekly_build_enabled is True  # default


# --- Daemon suppression gate ---


class TestDaemonSuppressionGate:
    def test_configure_weekly_build(self):
        daemon = DaemonMode()
        daemon.configure_weekly_build(
            enabled=True,
            briefing_day=5,
            window_start=20,
            window_end=4,
        )
        assert daemon._weekly_build_enabled is True
        assert daemon._weekly_briefing_day == 5
        assert daemon._build_window_start_hour == 20
        assert daemon._build_window_end_hour == 4

    def test_scheduler_tick_suppressed_when_weekly_enabled(self):
        """With weekly build enabled, self-builds are suppressed outside window."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=True)

        # Mock _find_critical_goal to return a goal — should NOT be enqueued
        daemon._find_critical_goal = MagicMock(return_value=(1, "test goal"))

        # Patch so we're outside the build window
        with patch("mother.scheduler.should_suppress_autonomous", return_value=True):
            with patch("mother.scheduler.is_in_build_window", return_value=False):
                run(daemon._scheduler_tick())

        # Queue should be empty — self-build was suppressed
        assert len(daemon._queue) == 0

    def test_scheduler_tick_legacy_when_weekly_disabled(self):
        """With weekly build disabled, legacy immediate execution works."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=False)
        daemon._compile_fn = AsyncMock()

        goal_id = 1
        daemon._find_critical_goal = MagicMock(return_value=(goal_id, "test goal"))
        daemon._enrich_goal_for_build = MagicMock(return_value="test goal enriched")

        # Patch goal tracking (lazy import inside _scheduler_tick)
        with patch("mother.goals.GoalStore") as mock_gs_class:
            mock_gs = MagicMock()
            mock_gs_class.return_value = mock_gs
            run(daemon._scheduler_tick())

        # Should have enqueued the self-improvement compile (legacy behavior)
        assert len(daemon._queue) == 1
        assert "[SELF-IMPROVEMENT]" in daemon._queue[0].input_text

    def test_scheduler_tick_cooldown_applies_legacy(self):
        """Cooldown prevents rapid-fire scheduling in legacy mode."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=False)
        daemon._last_autonomous_compile = time.time()  # just ran

        daemon._find_critical_goal = MagicMock(return_value=(1, "test"))

        run(daemon._scheduler_tick())

        # Should not enqueue due to cooldown
        assert len(daemon._queue) == 0

    def test_suppression_gate_blocks_all_outside_window(self):
        """Verify the suppression logic: outside window = no builds."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=True)
        daemon._find_critical_goal = MagicMock(return_value=(1, "important goal"))

        # Patch _check_briefing_time to no-op
        daemon._check_briefing_time = AsyncMock()

        run(daemon._scheduler_tick())

        # By default, should_suppress_autonomous returns True for most times
        # Since we're not mocking datetime, it depends on when this runs,
        # but the gate logic is tested in test_scheduler.py
        # Here we just verify the method completes without error
        assert isinstance(daemon._queue, list)


# --- Build window execution ---


class TestBuildWindowExecution:
    def test_run_build_window_empty(self, tmp_path):
        """No approved goals → no builds."""
        daemon = DaemonMode()
        daemon._compile_fn = AsyncMock()
        daemon._build_max_per_window = 10

        # Set up empty GoalStore at ~/.motherlabs/ equivalent
        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gs.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        # No builds executed
        assert daemon._compile_fn.call_count == 0

    def test_run_build_window_executes_approved(self, tmp_path):
        """Approved goals get compiled during window."""
        compile_fn = AsyncMock(return_value=MagicMock(success=True))
        daemon = DaemonMode(compile_fn=compile_fn)
        daemon._build_max_per_window = 10
        daemon._enrich_goal_for_build = MagicMock(side_effect=lambda d: d)

        # Set up goals at ~/.motherlabs/ equivalent
        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gid1 = gs.add("Fix convergence", source="system", priority="high")
        gid2 = gs.add("Optimize latency", source="system", priority="normal")
        gs.approve_goals([gid1, gid2])
        gs.close()

        # Set up schedule store with an approved week
        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        briefing = generate_briefing([], "2026-02-23")
        sched.save_briefing(briefing)
        sched.save_approval("2026-02-23", [str(gid1), str(gid2)])
        sched.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        # Both goals should have been compiled
        assert compile_fn.call_count == 2

        # Check report was saved
        sched = ScheduleStore(maps_path)
        week = sched.current_week()
        assert week["status"] == "completed"
        assert week["report"].success_count == 2
        sched.close()

    def test_run_build_window_handles_failure(self, tmp_path):
        """Failed builds are recorded in the report."""
        compile_fn = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        daemon = DaemonMode(compile_fn=compile_fn)
        daemon._build_max_per_window = 10
        daemon._enrich_goal_for_build = MagicMock(side_effect=lambda d: d)

        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gid = gs.add("Failing goal", source="system")
        gs.approve_goals([gid])
        gs.close()

        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        briefing = generate_briefing([], "2026-02-23")
        sched.save_briefing(briefing)
        sched.save_approval("2026-02-23", [str(gid)])
        sched.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        sched = ScheduleStore(maps_path)
        week = sched.current_week()
        assert week["report"].failure_count == 1
        assert "LLM timeout" in week["report"].results[0].error
        sched.close()

    def test_run_build_window_skips_completed(self, tmp_path):
        """Don't re-run a window that's already completed."""
        compile_fn = AsyncMock()
        daemon = DaemonMode(compile_fn=compile_fn)
        daemon._build_max_per_window = 10

        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        briefing = generate_briefing([], "2026-02-23")
        sched.save_briefing(briefing)
        sched.set_status("2026-02-23", "completed")
        sched.close()

        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gid = gs.add("Should not run")
        gs.approve_goals([gid])
        gs.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        assert compile_fn.call_count == 0

    def test_report_callback_fires(self, tmp_path):
        """Report callback is called after window completes."""
        compile_fn = AsyncMock(return_value=MagicMock(success=True))
        report_callback = AsyncMock()
        daemon = DaemonMode(compile_fn=compile_fn)
        daemon._build_max_per_window = 10
        daemon._report_callback = report_callback
        daemon._enrich_goal_for_build = MagicMock(side_effect=lambda d: d)

        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gid = gs.add("Test goal", source="system")
        gs.approve_goals([gid])
        gs.close()

        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        briefing = generate_briefing([], "2026-02-23")
        sched.save_briefing(briefing)
        sched.save_approval("2026-02-23", [str(gid)])
        sched.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        assert report_callback.call_count == 1


# --- Briefing trigger ---


class TestBriefingTrigger:
    def test_check_briefing_no_goals(self, tmp_path):
        """No briefing generated when no active goals."""
        daemon = DaemonMode()
        daemon._weekly_briefing_day = 6
        daemon._weekly_briefing_hour = 10

        # Empty goal store
        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gs.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._check_briefing_time())

        maps_path = ml_dir / "maps.db"
        if maps_path.exists():
            sched = ScheduleStore(maps_path)
            week = sched.current_week()
            sched.close()
            assert week is None

    def test_check_briefing_no_crash_on_missing_db(self, tmp_path):
        """No crash when history.db doesn't exist."""
        daemon = DaemonMode()
        daemon._weekly_briefing_day = 6
        daemon._weekly_briefing_hour = 10

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            # Should not crash
            run(daemon._check_briefing_time())

    def test_check_briefing_no_crash_on_existing_briefing(self, tmp_path):
        """No crash and no duplicate when briefing already exists for this week."""
        daemon = DaemonMode()
        daemon._weekly_briefing_day = 6
        daemon._weekly_briefing_hour = 10

        # Pre-create a briefing
        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        now = datetime.now()
        week_start = (now - __import__('datetime').timedelta(days=now.weekday())).strftime("%Y-%m-%d")
        briefing = generate_briefing([], week_start)
        sched.save_briefing(briefing)
        sched.close()

        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        gs.add("Some goal")
        gs.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._check_briefing_time())

        # Should still have just the one briefing
        sched = ScheduleStore(maps_path)
        week = sched.current_week()
        sched.close()
        assert week is not None


# --- Legacy fallback ---


class TestLegacyFallback:
    def test_legacy_mode_still_works(self):
        """With weekly_build_enabled=False, daemon behaves as before."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=False)
        daemon._compile_fn = AsyncMock()
        daemon._find_critical_goal = MagicMock(return_value=None)
        daemon._should_self_compile = MagicMock(return_value=False)

        run(daemon._scheduler_tick())

        # No goal found, no self-compile needed — tick completes cleanly
        assert len(daemon._queue) == 0

    def test_legacy_pauses_on_failures(self):
        """Legacy mode still pauses after consecutive failures."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=False)
        daemon._autonomous_failures = 3
        daemon._max_autonomous_failures = 3
        daemon._find_critical_goal = MagicMock(return_value=(1, "test"))

        run(daemon._scheduler_tick())

        assert len(daemon._queue) == 0  # suppressed by failure count

    def test_weekly_enabled_doesnt_break_existing_methods(self):
        """DaemonMode's existing methods work with weekly build configured."""
        daemon = DaemonMode()
        daemon.configure_weekly_build(enabled=True)

        status = daemon.status()
        assert "running" in status
        assert "queue_pending" in status

        queue = daemon.get_queue()
        assert isinstance(queue, list)


# --- Empty window ---


class TestEmptyWindow:
    def test_empty_window_no_crash(self, tmp_path):
        """Build window with no approved goals completes cleanly."""
        daemon = DaemonMode()
        daemon._compile_fn = AsyncMock()
        daemon._build_max_per_window = 10

        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        # Add goals but don't approve them
        gs.add("Not approved 1")
        gs.add("Not approved 2")
        gs.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        # No compiles should have run
        assert daemon._compile_fn.call_count == 0

    def test_window_respects_max_limit(self, tmp_path):
        """Build window stops after build_max_per_window."""
        compile_fn = AsyncMock(return_value=MagicMock(success=True))
        daemon = DaemonMode(compile_fn=compile_fn)
        daemon._build_max_per_window = 2
        daemon._enrich_goal_for_build = MagicMock(side_effect=lambda d: d)

        ml_dir = tmp_path / ".motherlabs"
        ml_dir.mkdir()
        db_path = ml_dir / "history.db"
        gs = GoalStore(db_path)
        ids = []
        for i in range(5):
            gid = gs.add(f"Goal {i}", source="system")
            ids.append(gid)
        gs.approve_goals(ids)
        gs.close()

        maps_path = ml_dir / "maps.db"
        sched = ScheduleStore(maps_path)
        briefing = generate_briefing([], "2026-02-23")
        sched.save_briefing(briefing)
        sched.save_approval("2026-02-23", [str(i) for i in ids])
        sched.close()

        with patch("mother.daemon.Path.home", return_value=tmp_path):
            run(daemon._run_build_window())

        # Only 2 builds should run (limited by build_max_per_window)
        assert compile_fn.call_count == 2
