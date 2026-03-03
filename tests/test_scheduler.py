"""Tests for mother/scheduler.py — weekly build governance schedule engine."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mother.scheduler import (
    BriefingItem,
    BuildResult,
    BuildWindow,
    ExecutionReport,
    ScheduleStore,
    WeeklyBriefing,
    format_briefing_markdown,
    format_report_markdown,
    generate_briefing,
    goals_due_for_briefing,
    is_in_build_window,
    next_briefing_time,
    should_suppress_autonomous,
)


# --- Fixtures ---


@pytest.fixture
def sample_goals():
    """Sample goal dicts as they come from GoalStore."""
    return [
        {
            "goal_id": 1,
            "description": "Improve kernel convergence detection accuracy",
            "source": "pattern_detector",
            "priority": "high",
            "status": "active",
            "timestamp": 1000.0,
            "attempt_count": 2,
            "engagement_count": 1,
        },
        {
            "goal_id": 2,
            "description": "Add retry logic to daemon health checks",
            "source": "system",
            "priority": "normal",
            "status": "active",
            "timestamp": 2000.0,
            "attempt_count": 0,
            "engagement_count": 0,
        },
        {
            "goal_id": 3,
            "description": "Optimize chat response latency",
            "source": "user",
            "priority": "urgent",
            "status": "active",
            "timestamp": 3000.0,
            "attempt_count": 0,
            "engagement_count": 3,
        },
    ]


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_schedule.db"


@pytest.fixture
def store(db_path):
    s = ScheduleStore(db_path)
    yield s
    s.close()


# --- Dataclass frozen tests ---


class TestDataclasses:
    def test_briefing_item_frozen(self):
        item = BriefingItem(
            goal_id="1", description="test", source="system",
            priority=2, risk="low", estimated_cost="$0.10", rationale="test",
        )
        with pytest.raises(AttributeError):
            item.description = "changed"

    def test_weekly_briefing_frozen(self):
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="none", estimated_total_cost="$0")
        with pytest.raises(AttributeError):
            b.summary = "changed"

    def test_build_window_defaults(self):
        w = BuildWindow()
        assert w.start_hour == 22
        assert w.end_hour == 6
        assert w.day_of_week == 6

    def test_build_result_frozen(self):
        r = BuildResult(goal_id="1", success=True, duration_seconds=10.0,
                        cost_usd=0.5, summary="done")
        with pytest.raises(AttributeError):
            r.success = False

    def test_execution_report_frozen(self):
        rpt = ExecutionReport(window_date="2026-02-23", results=[], total_cost=0,
                              success_count=0, failure_count=0, skipped_count=0,
                              summary="none")
        with pytest.raises(AttributeError):
            rpt.total_cost = 5.0


# --- Briefing generation ---


class TestGenerateBriefing:
    def test_empty_goals(self):
        b = generate_briefing([], "2026-02-23")
        assert b.total_goals == 0
        assert "No improvement goals" in b.summary
        assert b.items == []

    def test_basic_generation(self, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        assert b.total_goals == 3
        assert b.week_start == "2026-02-23"
        assert len(b.items) == 3
        # Sorted by priority (urgent=0 first)
        assert b.items[0].priority <= b.items[-1].priority

    def test_priority_sorting(self, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        priorities = [item.priority for item in b.items]
        assert priorities == sorted(priorities)

    def test_risk_estimation_high(self):
        goals = [{"goal_id": 1, "description": "Rewrite the engine pipeline",
                  "source": "system", "priority": "high", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert b.items[0].risk == "high"

    def test_risk_estimation_medium(self):
        goals = [{"goal_id": 1, "description": "Refactor daemon scheduling",
                  "source": "system", "priority": "normal", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert b.items[0].risk == "medium"

    def test_risk_estimation_low(self):
        goals = [{"goal_id": 1, "description": "Add a greeting message",
                  "source": "system", "priority": "low", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert b.items[0].risk == "low"

    def test_summary_mentions_counts(self, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        assert "3" in b.summary
        assert "high priority" in b.summary.lower() or "improvement" in b.summary.lower()

    def test_rationale_includes_pattern_source(self):
        goals = [{"goal_id": 1, "description": "Fix token counting",
                  "source": "pattern_detector", "priority": "high", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert "pattern" in b.items[0].rationale.lower()


# --- Build window logic ---


class TestBuildWindow:
    def test_inside_overnight_window_start_day(self):
        # Sunday 23:00
        now = datetime(2026, 3, 1, 23, 0)  # Sunday
        assert now.weekday() == 6
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is True

    def test_inside_overnight_window_next_day(self):
        # Monday 03:00 (overflow from Sunday window)
        now = datetime(2026, 3, 2, 3, 0)  # Monday
        assert now.weekday() == 0
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is True

    def test_outside_window_wrong_day(self):
        # Wednesday 23:00 — not in window
        now = datetime(2026, 2, 25, 23, 0)  # Wednesday
        assert now.weekday() == 2
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is False

    def test_outside_window_right_day_wrong_hour(self):
        # Sunday 15:00 — right day, wrong hour
        now = datetime(2026, 3, 1, 15, 0)
        assert now.weekday() == 6
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is False

    def test_at_window_start_boundary(self):
        # Sunday exactly 22:00
        now = datetime(2026, 3, 1, 22, 0)
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is True

    def test_at_window_end_boundary(self):
        # Monday exactly 06:00 — should be outside (< end_hour, not <=)
        now = datetime(2026, 3, 2, 6, 0)
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is False

    def test_same_day_window(self):
        # Same-day window: 08:00-18:00 on Monday
        now = datetime(2026, 3, 2, 12, 0)  # Monday noon
        window = BuildWindow(start_hour=8, end_hour=18, day_of_week=0)
        assert is_in_build_window(now, window) is True

    def test_same_day_window_outside(self):
        now = datetime(2026, 3, 2, 20, 0)  # Monday 20:00
        window = BuildWindow(start_hour=8, end_hour=18, day_of_week=0)
        assert is_in_build_window(now, window) is False

    def test_midnight_crossing(self):
        # Exactly midnight on Monday (overflow from Sunday 22-06)
        now = datetime(2026, 3, 2, 0, 0)  # Monday 00:00
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=6)
        assert is_in_build_window(now, window) is True

    def test_saturday_window(self):
        # Saturday night window
        now = datetime(2026, 2, 28, 23, 0)  # Saturday
        assert now.weekday() == 5
        window = BuildWindow(start_hour=22, end_hour=6, day_of_week=5)
        assert is_in_build_window(now, window) is True


# --- Suppression logic ---


class TestSuppression:
    def test_suppressed_outside_window(self):
        now = datetime(2026, 2, 25, 14, 0)  # Wednesday afternoon
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is True

    def test_not_suppressed_inside_window(self):
        now = datetime(2026, 3, 1, 23, 0)  # Sunday 23:00
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is False

    def test_suppressed_on_briefing_day_before_window(self):
        now = datetime(2026, 3, 1, 10, 0)  # Sunday morning
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is True

    def test_suppressed_monday_afternoon(self):
        now = datetime(2026, 3, 2, 14, 0)  # Monday afternoon (after window)
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is True

    def test_not_suppressed_monday_early_morning(self):
        # Monday 03:00 — still in overflow window
        now = datetime(2026, 3, 2, 3, 0)
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is False

    def test_suppressed_with_all_day_window(self):
        now = datetime(2026, 2, 24, 12, 0)  # Tuesday
        window = BuildWindow(start_hour=0, end_hour=0, day_of_week=6)
        assert should_suppress_autonomous(now, window, briefing_day=6) is True

    def test_suppressed_wrong_day(self):
        now = datetime(2026, 2, 26, 23, 0)  # Thursday at 23:00
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is True

    def test_not_suppressed_at_window_start(self):
        now = datetime(2026, 3, 1, 22, 0)  # Sunday at exactly 22:00
        window = BuildWindow()
        assert should_suppress_autonomous(now, window, briefing_day=6) is False


# --- Formatting ---


class TestFormatting:
    def test_briefing_markdown_empty(self):
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="Nothing.", estimated_total_cost="$0")
        md = format_briefing_markdown(b)
        assert "Weekly Build Briefing" in md
        assert "No goals to review" in md

    def test_briefing_markdown_with_items(self, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        md = format_briefing_markdown(b)
        assert "Weekly Build Briefing" in md
        assert "| #" in md  # table header
        assert "approve" in md.lower()

    def test_briefing_markdown_contains_all_goals(self, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        md = format_briefing_markdown(b)
        for goal in sample_goals:
            # At least partial description should appear
            assert goal["description"][:30] in md

    def test_report_markdown_empty(self):
        rpt = ExecutionReport(window_date="2026-02-23", results=[], total_cost=0,
                              success_count=0, failure_count=0, skipped_count=0,
                              summary="No builds.")
        md = format_report_markdown(rpt)
        assert "Build Window Report" in md
        assert "No builds executed" in md

    def test_report_markdown_with_results(self):
        results = [
            BuildResult(goal_id="1", success=True, duration_seconds=45.0,
                        cost_usd=0.5, summary="Improved convergence"),
            BuildResult(goal_id="2", success=False, duration_seconds=10.0,
                        cost_usd=0.1, summary="Daemon retry", error="ImportError"),
        ]
        rpt = ExecutionReport(window_date="2026-02-23", results=results,
                              total_cost=0.6, success_count=1, failure_count=1,
                              skipped_count=0, summary="Mixed results.")
        md = format_report_markdown(rpt)
        assert "pass" in md
        assert "FAIL" in md
        assert "Failures" in md
        assert "ImportError" in md

    def test_report_markdown_cost_formatting(self):
        rpt = ExecutionReport(window_date="2026-02-23", results=[], total_cost=3.14,
                              success_count=0, failure_count=0, skipped_count=0,
                              summary="Done.")
        md = format_report_markdown(rpt)
        assert "$3.14" in md


# --- Next briefing time ---


class TestNextBriefingTime:
    def test_monday_to_sunday(self):
        # Monday → next Sunday at 10:00
        now = datetime(2026, 2, 23, 14, 0)  # Monday
        nxt = next_briefing_time(now, briefing_day=6, briefing_hour=10)
        assert nxt.weekday() == 6
        assert nxt.hour == 10
        assert nxt > now

    def test_sunday_before_briefing_hour(self):
        # Sunday 08:00 → same Sunday at 10:00
        now = datetime(2026, 3, 1, 8, 0)  # Sunday
        nxt = next_briefing_time(now, briefing_day=6, briefing_hour=10)
        assert nxt.weekday() == 6
        assert nxt.hour == 10
        assert nxt.day == now.day  # same day

    def test_sunday_after_briefing_hour(self):
        # Sunday 15:00 → next Sunday at 10:00
        now = datetime(2026, 3, 1, 15, 0)
        nxt = next_briefing_time(now, briefing_day=6, briefing_hour=10)
        assert nxt.weekday() == 6
        assert nxt > now
        # 6 days 19 hours difference (next Sunday 10:00 from Sunday 15:00)
        delta = nxt - now
        assert 6 <= delta.days <= 7

    def test_saturday(self):
        # Saturday → next day (Sunday)
        now = datetime(2026, 2, 28, 12, 0)  # Saturday
        nxt = next_briefing_time(now, briefing_day=6, briefing_hour=10)
        assert nxt.weekday() == 6
        assert nxt > now


# --- Goals due for briefing ---


class TestGoalsDueForBriefing:
    def test_no_last_briefing(self, sample_goals):
        result = goals_due_for_briefing(sample_goals, None)
        assert len(result) == 3

    def test_filter_by_timestamp(self, sample_goals):
        # Last briefing covers goal 1 (ts=1000) but not 2 (2000) or 3 (3000)
        last = datetime.fromtimestamp(1500).isoformat()
        result = goals_due_for_briefing(sample_goals, last)
        assert len(result) == 2
        ids = [g["goal_id"] for g in result]
        assert 1 not in ids
        assert 2 in ids
        assert 3 in ids

    def test_all_before_cutoff(self, sample_goals):
        last = datetime.fromtimestamp(999999).isoformat()
        result = goals_due_for_briefing(sample_goals, last)
        assert len(result) == 0

    def test_invalid_last_briefing(self, sample_goals):
        result = goals_due_for_briefing(sample_goals, "not-a-date")
        assert len(result) == 3


# --- Persistence ---


class TestScheduleStore:
    def test_save_and_load_briefing(self, store):
        b = WeeklyBriefing(
            items=[BriefingItem(
                goal_id="1", description="test", source="system",
                priority=2, risk="low", estimated_cost="$0.10", rationale="test",
            )],
            week_start="2026-02-23",
            total_goals=1,
            summary="One goal.",
            estimated_total_cost="$0.10",
        )
        store.save_briefing(b)
        week = store.current_week()
        assert week is not None
        assert week["week_start"] == "2026-02-23"
        assert week["status"] == "awaiting_approval"
        assert week["briefing"].total_goals == 1
        assert len(week["briefing"].items) == 1

    def test_save_approval(self, store):
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="none", estimated_total_cost="$0")
        store.save_briefing(b)
        store.save_approval("2026-02-23", ["1", "3"])
        week = store.current_week()
        assert week["status"] == "approved"
        assert week["approved_goal_ids"] == ["1", "3"]

    def test_save_report(self, store):
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="none", estimated_total_cost="$0")
        store.save_briefing(b)
        rpt = ExecutionReport(
            window_date="2026-02-23",
            results=[BuildResult(goal_id="1", success=True, duration_seconds=30,
                                 cost_usd=0.5, summary="done")],
            total_cost=0.5, success_count=1, failure_count=0, skipped_count=0,
            summary="All passed.",
        )
        store.save_report("2026-02-23", rpt)
        week = store.current_week()
        assert week["status"] == "completed"
        assert week["report"].total_cost == 0.5

    def test_last_report(self, store):
        assert store.last_report() is None
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="none", estimated_total_cost="$0")
        store.save_briefing(b)
        rpt = ExecutionReport(window_date="2026-02-23", results=[], total_cost=0,
                              success_count=0, failure_count=0, skipped_count=0,
                              summary="Empty.")
        store.save_report("2026-02-23", rpt)
        last = store.last_report()
        assert last is not None
        assert last.window_date == "2026-02-23"

    def test_get_specific_week(self, store):
        b = WeeklyBriefing(items=[], week_start="2026-02-16", total_goals=0,
                           summary="w1", estimated_total_cost="$0")
        store.save_briefing(b)
        b2 = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                            summary="w2", estimated_total_cost="$0")
        store.save_briefing(b2)
        week = store.get_week("2026-02-16")
        assert week is not None
        assert week["briefing"].summary == "w1"

    def test_nonexistent_week(self, store):
        assert store.get_week("2099-01-01") is None

    def test_set_status(self, store):
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                           summary="test", estimated_total_cost="$0")
        store.save_briefing(b)
        store.set_status("2026-02-23", "executing")
        week = store.current_week()
        assert week["status"] == "executing"

    def test_upsert_briefing(self, store):
        """Saving briefing twice for same week updates, not duplicates."""
        b1 = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=0,
                            summary="first", estimated_total_cost="$0")
        store.save_briefing(b1)
        b2 = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=5,
                            summary="updated", estimated_total_cost="$5")
        store.save_briefing(b2)
        week = store.current_week()
        assert week["briefing"].summary == "updated"
        assert week["briefing"].total_goals == 5


# --- Edge cases ---


class TestEdgeCases:
    def test_all_goals_rejected(self, store):
        """Approval with empty list is valid — all rejected."""
        b = WeeklyBriefing(items=[], week_start="2026-02-23", total_goals=3,
                           summary="test", estimated_total_cost="$0")
        store.save_briefing(b)
        store.save_approval("2026-02-23", [])
        week = store.current_week()
        assert week["approved_goal_ids"] == []
        assert week["status"] == "approved"

    def test_partial_approval(self, store, sample_goals):
        b = generate_briefing(sample_goals, "2026-02-23")
        store.save_briefing(b)
        store.save_approval("2026-02-23", ["1", "3"])
        week = store.current_week()
        assert len(week["approved_goal_ids"]) == 2

    def test_single_goal_briefing(self):
        goals = [{"goal_id": 1, "description": "Minor tweak", "source": "user",
                  "priority": "low", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert b.total_goals == 1
        assert len(b.items) == 1

    def test_goal_without_optional_fields(self):
        """Goals missing attempt_count/engagement_count should still work."""
        goals = [{"goal_id": 1, "description": "Basic goal", "source": "system",
                  "priority": "normal", "timestamp": 1000.0}]
        b = generate_briefing(goals, "2026-02-23")
        assert b.total_goals == 1
        assert b.items[0].rationale  # should have some rationale

    def test_timezone_aware_window(self):
        """Window check works with naive datetimes (local time)."""
        # This is the expected usage — all times are local
        now = datetime(2026, 3, 1, 23, 30)  # Sunday 23:30
        window = BuildWindow()
        assert is_in_build_window(now, window) is True
