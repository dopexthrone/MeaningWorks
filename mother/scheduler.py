"""
Weekly build governance — schedule engine for self-improvement cadence.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Manages weekly briefings, build windows, and execution reports.
The founder approves which goals to build; approved goals execute
during a designated overnight window.
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional


# --- Frozen dataclasses ---


@dataclass(frozen=True)
class BriefingItem:
    """A single goal presented in the weekly briefing."""

    goal_id: str
    description: str
    source: str           # "grid", "pattern", "feedback", "depth_chain", "user", "system"
    priority: int         # 0=urgent, 1=high, 2=normal, 3=low
    risk: str             # "low", "medium", "high"
    estimated_cost: str   # rough bucket like "$0.10-0.50"
    rationale: str        # why Mother thinks this matters


@dataclass(frozen=True)
class WeeklyBriefing:
    """A complete weekly briefing of accumulated goals."""

    items: list           # list[BriefingItem]
    week_start: str       # ISO date string
    total_goals: int
    summary: str          # 2-3 sentence executive summary
    estimated_total_cost: str


@dataclass(frozen=True)
class BuildWindow:
    """Configuration for the overnight build window."""

    start_hour: int = 22      # 0-23
    end_hour: int = 6         # 0-23
    day_of_week: int = 6      # 0=Mon, 6=Sun


@dataclass(frozen=True)
class BuildResult:
    """Result of a single build during the window."""

    goal_id: str
    success: bool
    duration_seconds: float
    cost_usd: float
    summary: str
    error: Optional[str] = None


@dataclass(frozen=True)
class ExecutionReport:
    """Report generated after a build window completes."""

    window_date: str          # ISO date
    results: list             # list[BuildResult]
    total_cost: float
    success_count: int
    failure_count: int
    skipped_count: int
    summary: str


# --- Priority/risk constants ---

_PRIORITY_LABELS = {0: "urgent", 1: "high", 2: "normal", 3: "low"}
_PRIORITY_FROM_LABEL = {"urgent": 0, "high": 1, "normal": 2, "low": 3}

_RISK_KEYWORDS_HIGH = frozenset({
    "engine", "pipeline", "compiler", "kernel", "core", "llm",
    "bridge", "schema", "migration", "delete", "remove", "rewrite",
})
_RISK_KEYWORDS_MEDIUM = frozenset({
    "daemon", "config", "perception", "goal", "memory", "chat",
    "refactor", "restructure", "optimize", "replace",
})


# --- Pure functions ---


def _estimate_risk(description: str) -> str:
    """Estimate risk level from goal description keywords."""
    words = frozenset(description.lower().split())
    if words & _RISK_KEYWORDS_HIGH:
        return "high"
    if words & _RISK_KEYWORDS_MEDIUM:
        return "medium"
    return "low"


def _estimate_cost(priority: int, risk: str) -> str:
    """Rough cost bucket based on priority and risk."""
    if risk == "high":
        return "$1.00-5.00"
    if risk == "medium" or priority <= 1:
        return "$0.50-2.00"
    return "$0.10-0.50"


def _goal_to_briefing_item(goal: dict) -> BriefingItem:
    """Transform a raw goal dict into a BriefingItem."""
    desc = goal.get("description", "")
    source = goal.get("source", "system")
    priority_label = goal.get("priority", "normal")
    priority = _PRIORITY_FROM_LABEL.get(priority_label, 2)
    risk = _estimate_risk(desc)
    cost = _estimate_cost(priority, risk)

    # Build rationale from available signals
    parts = []
    attempt_count = goal.get("attempt_count", 0)
    engagement = goal.get("engagement_count", 0)
    if attempt_count > 0:
        parts.append(f"{attempt_count} previous attempt(s)")
    if engagement > 0:
        parts.append(f"{engagement} engagement(s)")
    if source == "pattern_detector":
        parts.append("recurring pattern detected")
    elif source == "system":
        parts.append("identified by feedback analysis")
    elif source == "user":
        parts.append("requested by founder")
    rationale = "; ".join(parts) if parts else "accumulated from autonomous analysis"

    return BriefingItem(
        goal_id=str(goal.get("goal_id", "")),
        description=desc,
        source=source,
        priority=priority,
        risk=risk,
        estimated_cost=cost,
        rationale=rationale,
    )


def generate_briefing(goals: list, week_start: str) -> WeeklyBriefing:
    """Transform raw GoalStore goal dicts into a WeeklyBriefing.

    Goals should be dicts with keys matching Goal dataclass fields.
    """
    if not goals:
        return WeeklyBriefing(
            items=[],
            week_start=week_start,
            total_goals=0,
            summary="No improvement goals accumulated this week.",
            estimated_total_cost="$0.00",
        )

    items = [_goal_to_briefing_item(g) for g in goals]
    # Sort: highest priority first, then risk descending
    risk_order = {"high": 0, "medium": 1, "low": 2}
    items = sorted(items, key=lambda i: (i.priority, risk_order.get(i.risk, 2)))

    # Summary
    high_count = sum(1 for i in items if i.priority <= 1)
    risk_count = sum(1 for i in items if i.risk == "high")
    parts = [f"{len(items)} improvement goal(s) accumulated this week."]
    if high_count > 0:
        parts.append(f"{high_count} are high priority.")
    if risk_count > 0:
        parts.append(f"{risk_count} carry high risk and may need careful review.")
    summary = " ".join(parts)

    # Total cost estimate
    if risk_count > 0:
        total_cost = "$5.00-15.00"
    elif high_count > 0:
        total_cost = "$2.00-10.00"
    else:
        total_cost = "$0.50-5.00"

    return WeeklyBriefing(
        items=items,
        week_start=week_start,
        total_goals=len(items),
        summary=summary,
        estimated_total_cost=total_cost,
    )


def is_in_build_window(now: datetime, window: BuildWindow) -> bool:
    """Check if `now` falls inside the build window.

    Handles overnight spans (e.g., 22:00 Sun → 06:00 Mon).
    `now` should be timezone-aware or naive (treated as local).
    """
    current_dow = now.weekday()  # 0=Mon, 6=Sun
    current_hour = now.hour

    if window.start_hour <= window.end_hour:
        # Same-day window (e.g., 08:00-18:00)
        return current_dow == window.day_of_week and window.start_hour <= current_hour < window.end_hour
    else:
        # Overnight window (e.g., 22:00-06:00)
        # Window spans day_of_week at start_hour → day_of_week+1 at end_hour
        next_dow = (window.day_of_week + 1) % 7
        if current_dow == window.day_of_week and current_hour >= window.start_hour:
            return True
        if current_dow == next_dow and current_hour < window.end_hour:
            return True
        return False


def should_suppress_autonomous(now: datetime, window: BuildWindow, briefing_day: int) -> bool:
    """Return True if autonomous self-builds should be blocked.

    Self-builds are suppressed at ALL times when weekly governance is enabled.
    Builds only execute during the approved window, and only for approved goals.
    This function returns False only when inside the build window.
    """
    return not is_in_build_window(now, window)


def format_briefing_markdown(briefing: WeeklyBriefing) -> str:
    """Render a WeeklyBriefing as clean markdown for chat display."""
    lines = [
        f"## Weekly Build Briefing — Week of {briefing.week_start}",
        "",
        briefing.summary,
        "",
        f"**Estimated total cost:** {briefing.estimated_total_cost}",
        "",
    ]

    if not briefing.items:
        lines.append("*No goals to review.*")
        return "\n".join(lines)

    lines.append("| # | Goal | Priority | Risk | Est. Cost |")
    lines.append("|---|------|----------|------|-----------|")

    for i, item in enumerate(briefing.items, 1):
        prio_label = _PRIORITY_LABELS.get(item.priority, "normal")
        lines.append(
            f"| {i} | {item.description[:60]} | {prio_label} | {item.risk} | {item.estimated_cost} |"
        )

    lines.append("")
    lines.append("### Details")
    lines.append("")

    for i, item in enumerate(briefing.items, 1):
        lines.append(f"**{i}. {item.description}**")
        lines.append(f"  - Source: {item.source} | Goal ID: {item.goal_id}")
        lines.append(f"  - {item.rationale}")
        lines.append("")

    lines.append("---")
    lines.append("Approve with: `approve all` or `approve 1,3,5` | Reject with: `reject 2,4`")

    return "\n".join(lines)


def format_report_markdown(report: ExecutionReport) -> str:
    """Render an ExecutionReport as clean markdown."""
    lines = [
        f"## Build Window Report — {report.window_date}",
        "",
        report.summary,
        "",
        f"**Results:** {report.success_count} succeeded, {report.failure_count} failed, {report.skipped_count} skipped",
        f"**Total cost:** ${report.total_cost:.2f}",
        "",
    ]

    if not report.results:
        lines.append("*No builds executed.*")
        return "\n".join(lines)

    lines.append("| Goal | Status | Duration | Cost |")
    lines.append("|------|--------|----------|------|")

    for r in report.results:
        status = "pass" if r.success else "FAIL"
        duration = f"{r.duration_seconds:.0f}s"
        lines.append(
            f"| {r.summary[:50]} | {status} | {duration} | ${r.cost_usd:.2f} |"
        )

    # Detail failures
    failures = [r for r in report.results if not r.success]
    if failures:
        lines.append("")
        lines.append("### Failures")
        for r in failures:
            lines.append(f"- **{r.summary}**: {r.error or 'unknown error'}")

    return "\n".join(lines)


def next_briefing_time(now: datetime, briefing_day: int = 6, briefing_hour: int = 10) -> datetime:
    """Calculate when the next briefing should occur.

    Returns the next occurrence of briefing_day at briefing_hour.
    If now IS that time, returns now (it's briefing time).
    """
    days_ahead = briefing_day - now.weekday()
    if days_ahead < 0:
        days_ahead += 7
    elif days_ahead == 0:
        # Same day — check if we've passed the hour
        if now.hour >= briefing_hour:
            days_ahead = 7  # Next week

    target = now.replace(hour=briefing_hour, minute=0, second=0, microsecond=0)
    target = target + timedelta(days=days_ahead)
    return target


def goals_due_for_briefing(goals: list, last_briefing: Optional[str] = None) -> list:
    """Filter goals accumulated since last briefing.

    Args:
        goals: list of goal dicts with 'timestamp' field
        last_briefing: ISO date of last briefing, or None if never briefed

    Returns goals created after the last briefing date.
    """
    if not last_briefing:
        return list(goals)

    try:
        cutoff = datetime.fromisoformat(last_briefing).timestamp()
    except (ValueError, TypeError):
        return list(goals)

    return [g for g in goals if g.get("timestamp", 0) > cutoff]


# --- Persistence: ScheduleStore ---


_CREATE_SCHEDULE_TABLE = """
CREATE TABLE IF NOT EXISTS weekly_schedule (
    week_start TEXT PRIMARY KEY,
    briefing_json TEXT NOT NULL DEFAULT '{}',
    report_json TEXT NOT NULL DEFAULT '{}',
    approved_goal_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending_briefing'
)
"""


def _briefing_to_dict(b: WeeklyBriefing) -> dict:
    """Serialize a WeeklyBriefing to a JSON-safe dict."""
    return {
        "items": [
            {
                "goal_id": i.goal_id,
                "description": i.description,
                "source": i.source,
                "priority": i.priority,
                "risk": i.risk,
                "estimated_cost": i.estimated_cost,
                "rationale": i.rationale,
            }
            for i in b.items
        ],
        "week_start": b.week_start,
        "total_goals": b.total_goals,
        "summary": b.summary,
        "estimated_total_cost": b.estimated_total_cost,
    }


def _dict_to_briefing(d: dict) -> WeeklyBriefing:
    """Deserialize a dict to WeeklyBriefing."""
    items = [
        BriefingItem(**item)
        for item in d.get("items", [])
    ]
    return WeeklyBriefing(
        items=items,
        week_start=d.get("week_start", ""),
        total_goals=d.get("total_goals", 0),
        summary=d.get("summary", ""),
        estimated_total_cost=d.get("estimated_total_cost", "$0.00"),
    )


def _report_to_dict(r: ExecutionReport) -> dict:
    """Serialize an ExecutionReport to a JSON-safe dict."""
    return {
        "window_date": r.window_date,
        "results": [
            {
                "goal_id": res.goal_id,
                "success": res.success,
                "duration_seconds": res.duration_seconds,
                "cost_usd": res.cost_usd,
                "summary": res.summary,
                "error": res.error,
            }
            for res in r.results
        ],
        "total_cost": r.total_cost,
        "success_count": r.success_count,
        "failure_count": r.failure_count,
        "skipped_count": r.skipped_count,
        "summary": r.summary,
    }


def _dict_to_report(d: dict) -> ExecutionReport:
    """Deserialize a dict to ExecutionReport."""
    results = [
        BuildResult(**res)
        for res in d.get("results", [])
    ]
    return ExecutionReport(
        window_date=d.get("window_date", ""),
        results=results,
        total_cost=d.get("total_cost", 0.0),
        success_count=d.get("success_count", 0),
        failure_count=d.get("failure_count", 0),
        skipped_count=d.get("skipped_count", 0),
        summary=d.get("summary", ""),
    )


class ScheduleStore:
    """SQLite-backed store for weekly schedule state.

    Uses the same maps.db as other kernel stores.
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_SCHEDULE_TABLE)
        self._conn.commit()

    def save_briefing(self, briefing: WeeklyBriefing) -> None:
        """Save or update a weekly briefing."""
        self._conn.execute(
            """INSERT INTO weekly_schedule (week_start, briefing_json, status)
               VALUES (?, ?, 'awaiting_approval')
               ON CONFLICT(week_start) DO UPDATE SET
                 briefing_json = excluded.briefing_json,
                 status = 'awaiting_approval'""",
            (briefing.week_start, json.dumps(_briefing_to_dict(briefing))),
        )
        self._conn.commit()

    def save_approval(self, week_start: str, goal_ids: list) -> None:
        """Record approved goal IDs for a week."""
        self._conn.execute(
            """UPDATE weekly_schedule
               SET approved_goal_ids = ?, status = 'approved'
               WHERE week_start = ?""",
            (json.dumps(goal_ids), week_start),
        )
        self._conn.commit()

    def save_report(self, week_start: str, report: ExecutionReport) -> None:
        """Save execution report after build window."""
        self._conn.execute(
            """UPDATE weekly_schedule
               SET report_json = ?, status = 'completed'
               WHERE week_start = ?""",
            (json.dumps(_report_to_dict(report)), week_start),
        )
        self._conn.commit()

    def current_week(self) -> Optional[dict]:
        """Get the most recent week's schedule entry."""
        row = self._conn.execute(
            """SELECT week_start, briefing_json, report_json, approved_goal_ids, status
               FROM weekly_schedule
               ORDER BY week_start DESC LIMIT 1"""
        ).fetchone()
        if not row:
            return None
        return {
            "week_start": row[0],
            "briefing": _dict_to_briefing(json.loads(row[1])) if row[1] != '{}' else None,
            "report": _dict_to_report(json.loads(row[2])) if row[2] != '{}' else None,
            "approved_goal_ids": json.loads(row[3]),
            "status": row[4],
        }

    def get_week(self, week_start: str) -> Optional[dict]:
        """Get a specific week's schedule entry."""
        row = self._conn.execute(
            """SELECT week_start, briefing_json, report_json, approved_goal_ids, status
               FROM weekly_schedule
               WHERE week_start = ?""",
            (week_start,),
        ).fetchone()
        if not row:
            return None
        return {
            "week_start": row[0],
            "briefing": _dict_to_briefing(json.loads(row[1])) if row[1] != '{}' else None,
            "report": _dict_to_report(json.loads(row[2])) if row[2] != '{}' else None,
            "approved_goal_ids": json.loads(row[3]),
            "status": row[4],
        }

    def last_report(self) -> Optional[ExecutionReport]:
        """Get the most recent execution report."""
        row = self._conn.execute(
            """SELECT report_json FROM weekly_schedule
               WHERE status = 'completed' AND report_json != '{}'
               ORDER BY week_start DESC LIMIT 1"""
        ).fetchone()
        if not row:
            return None
        return _dict_to_report(json.loads(row[0]))

    def set_status(self, week_start: str, status: str) -> None:
        """Update the status of a week's schedule."""
        self._conn.execute(
            "UPDATE weekly_schedule SET status = ? WHERE week_start = ?",
            (status, week_start),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
