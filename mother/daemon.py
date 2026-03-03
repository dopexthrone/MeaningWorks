"""
Daemon mode — long-running unattended operation with compile queue.

NOT a leaf module. Imports from mother/infra_healing.py and mother/config.py.
Enables Mother to run overnight: background compilation queue, periodic
health checks, auto-restart on failure, idle shutdown.

Genome #42: overnight-capable — works while you sleep, reports when you wake.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional

from mother.infra_healing import InfraHealer
from mother.journal import BuildJournal, JournalEntry

logger = logging.getLogger("mother.daemon")


@dataclass
class DaemonConfig:
    """Configuration for daemon mode."""

    health_check_interval: int = 300        # seconds between health checks
    max_queue_size: int = 10                # max pending compile requests
    auto_heal: bool = True                  # auto-heal on degraded infra
    idle_shutdown_hours: float = 0          # 0 = never shutdown
    log_file: Optional[str] = None          # path to daemon log


@dataclass
class CompileRequest:
    """A compilation request in the daemon queue."""

    input_text: str
    domain: str = "software"
    priority: int = 0                       # higher = more urgent
    submitted_at: float = field(default_factory=time.time)
    status: str = "pending"                 # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    chain_depth: int = 0                    # intent chain generation (0 = root, max 3)

    def to_dict(self) -> dict:
        return {
            "input_text": self.input_text,
            "domain": self.domain,
            "priority": self.priority,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "error": self.error,
            "completed_at": self.completed_at,
            "chain_depth": self.chain_depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompileRequest":
        return cls(
            input_text=data["input_text"],
            domain=data.get("domain", "software"),
            priority=data.get("priority", 0),
            submitted_at=data.get("submitted_at", time.time()),
            status=data.get("status", "pending"),
            error=data.get("error"),
            completed_at=data.get("completed_at"),
            chain_depth=data.get("chain_depth", 0),
        )


class DaemonMode:
    """Daemon mode for unattended operation.

    Manages a compile queue, periodic health checks, and auto-healing.
    Does NOT directly call engine.compile() — provides the queue and
    lifecycle management. The bridge or app layer connects the actual
    compilation.
    """

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        config_dir: Optional[Path] = None,
        compile_fn: Any = None,
    ):
        self.config = config or DaemonConfig()
        self._healer = InfraHealer(config_dir=config_dir)
        self._compile_fn = compile_fn      # async callable for actual compilation
        self._queue: List[CompileRequest] = []
        self._running = False
        self._started_at: Optional[float] = None
        self._last_activity: Optional[float] = None
        self._health_task: Optional[asyncio.Task] = None
        self._queue_task: Optional[asyncio.Task] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._scheduler_task: Optional[asyncio.Task] = None
        self._completed_count = 0
        self._failed_count = 0
        # Autonomous scheduler state
        self._autonomous_cooldown = 1800    # 30 min between autonomous compiles
        self._last_autonomous_compile: Optional[float] = None
        self._autonomous_failures = 0
        self._max_autonomous_failures = 3   # pause after this many consecutive failures
        # Weekly build governance (set by chat.py from config)
        self._weekly_build_enabled = False
        self._weekly_briefing_day = 6       # Sunday
        self._weekly_briefing_hour = 10
        self._build_window_start_hour = 22
        self._build_window_end_hour = 6
        self._build_window_day = 6
        self._build_max_per_window = 10
        self._last_briefing_check: Optional[float] = None
        self._briefing_callback: Optional[Any] = None  # async callable for briefing presentation
        self._report_callback: Optional[Any] = None    # async callable for report presentation
        # Depth-chain rate limiting — max enqueues per hour
        self._depth_chain_enqueue_times: List[float] = []
        self._depth_chain_max_per_hour: int = 10

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start daemon mode. Begins health loop and queue processing."""
        if self._running:
            logger.warning("Daemon already running")
            return

        self._running = True
        self._started_at = time.time()
        self._last_activity = time.time()

        logger.info("Daemon mode started")

        # Launch background loops
        self._health_task = asyncio.create_task(self._health_loop())
        self._queue_task = asyncio.create_task(self._process_queue())
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        if self.config.idle_shutdown_hours > 0:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def stop(self) -> None:
        """Stop daemon mode. Cancels background tasks."""
        if not self._running:
            return

        self._running = False
        logger.info("Daemon mode stopping")

        for task in (self._health_task, self._queue_task, self._idle_task, self._scheduler_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._health_task = None
        self._queue_task = None
        self._idle_task = None
        self._scheduler_task = None
        logger.info("Daemon mode stopped")

    async def enqueue(
        self,
        input_text: str,
        domain: str = "software",
        priority: int = 0,
    ) -> CompileRequest:
        """Add a compile request to the queue.

        Raises ValueError if queue is full.
        Returns the created CompileRequest.
        """
        pending = [r for r in self._queue if r.status == "pending"]
        if len(pending) >= self.config.max_queue_size:
            raise ValueError(
                f"Queue full ({self.config.max_queue_size} pending). "
                "Wait for current requests to complete."
            )

        request = CompileRequest(
            input_text=input_text,
            domain=domain,
            priority=priority,
        )
        self._queue.append(request)
        self._last_activity = time.time()
        logger.info(f"Enqueued request: {input_text[:50]}... (priority={priority})")
        return request

    def status(self) -> dict:
        """Return daemon status summary."""
        pending = [r for r in self._queue if r.status == "pending"]
        running = [r for r in self._queue if r.status == "running"]

        return {
            "running": self._running,
            "started_at": self._started_at,
            "uptime_seconds": (
                time.time() - self._started_at if self._started_at else 0
            ),
            "queue_pending": len(pending),
            "queue_running": len(running),
            "completed": self._completed_count,
            "failed": self._failed_count,
            "total_processed": self._completed_count + self._failed_count,
            "last_activity": self._last_activity,
        }

    def save_queue(self, path: Path) -> None:
        """Save pending queue to disk for persistence across restarts."""
        pending = [r for r in self._queue if r.status == "pending"]
        data = [r.to_dict() for r in pending]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(data)} pending request(s) to {path}")

    def load_queue(self, path: Path) -> int:
        """Load queue from disk. Returns number of requests loaded."""
        if not path.exists():
            return 0

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            loaded = 0
            for item in data:
                request = CompileRequest.from_dict(item)
                request.status = "pending"  # reset status on reload
                self._queue.append(request)
                loaded += 1
            logger.info(f"Loaded {loaded} request(s) from {path}")
            return loaded
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load queue from {path}: {e}")
            return 0

    def get_queue(self) -> List[CompileRequest]:
        """Return copy of current queue."""
        return list(self._queue)

    def clear_completed(self) -> int:
        """Remove completed/failed requests from queue. Returns count removed."""
        before = len(self._queue)
        self._queue = [
            r for r in self._queue
            if r.status in ("pending", "running")
        ]
        removed = before - len(self._queue)
        return removed

    # --- Background loops ---

    async def _process_queue(self) -> None:
        """Process compile requests from queue, in priority order."""
        while self._running:
            try:
                # Find highest-priority pending request
                pending = [r for r in self._queue if r.status == "pending"]
                if not pending:
                    await asyncio.sleep(1)
                    continue

                # Sort by priority (descending), then by submission time
                pending.sort(key=lambda r: (-r.priority, r.submitted_at))
                request = pending[0]

                request.status = "running"
                self._last_activity = time.time()
                logger.info(f"Processing: {request.input_text[:50]}...")

                is_autonomous = request.input_text.startswith("[SELF-IMPROVEMENT]")

                if self._compile_fn:
                    try:
                        result = await self._compile_fn(
                            request.input_text,
                            request.domain,
                        )
                        request.result = result
                        request.status = "completed"
                        request.completed_at = time.time()
                        self._completed_count += 1
                        logger.info("Request completed successfully")

                        # Reset autonomous failure counter on success
                        if is_autonomous:
                            self._autonomous_failures = 0

                        # Feedback loop: record outcome + apply observations
                        self._record_feedback(request, result)
                        self._sync_goals()

                        # Intent chaining: enqueue depth-explore for unexplored endpoints
                        await self._enqueue_depth_chains(result, request.domain, request.chain_depth)

                    except Exception as e:
                        request.status = "failed"
                        request.error = str(e)
                        request.completed_at = time.time()
                        self._failed_count += 1
                        logger.error(f"Request failed: {e}")

                        # Track autonomous failures for scheduler pause
                        if is_autonomous:
                            self._autonomous_failures += 1
                            logger.warning(
                                f"Autonomous compile failed ({self._autonomous_failures}/"
                                f"{self._max_autonomous_failures})"
                            )
                else:
                    # No compile function — mark as failed
                    request.status = "failed"
                    request.error = "No compile function configured"
                    request.completed_at = time.time()
                    self._failed_count += 1

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5)

    async def _health_loop(self) -> None:
        """Periodic health checks."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                if not self._running:
                    break

                checks = self._healer.diagnose()
                unhealthy = [c for c in checks if c.status != "healthy"]

                if unhealthy:
                    logger.warning(
                        f"Health check: {len(unhealthy)} issue(s) detected"
                    )
                    if self.config.auto_heal:
                        results = self._healer.heal(checks)
                        healed = [r for r in results if r.success]
                        if healed:
                            logger.info(f"Auto-healed {len(healed)} issue(s)")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Health loop error: {e}")

    async def _idle_loop(self) -> None:
        """Check for idle shutdown."""
        # Check interval: at most 60s, but faster for short idle thresholds
        check_interval = min(60, self.config.idle_shutdown_hours * 3600 / 2)
        check_interval = max(0.5, check_interval)  # floor at 0.5s

        while self._running:
            try:
                await asyncio.sleep(check_interval)
                if not self._running:
                    break

                if self._last_activity is None:
                    continue

                idle_seconds = time.time() - self._last_activity
                idle_hours = idle_seconds / 3600

                if idle_hours >= self.config.idle_shutdown_hours:
                    logger.info(
                        f"Idle shutdown after {idle_hours:.1f}h of inactivity"
                    )
                    await self.stop()
                    break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Idle loop error: {e}")

    # --- Autonomous scheduler ---

    async def _scheduler_loop(self) -> None:
        """Periodically check for goals and enqueue self-improvement compiles."""
        while self._running:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                if not self._running:
                    break
                await self._scheduler_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"Scheduler loop error: {e}")

    # L3: Self-compile staleness thresholds (hours)
    _SELF_COMPILE_STALE_HOURS = 24
    _SELF_COMPILE_CONVERGED_HOURS = 168  # 7 days

    async def _scheduler_tick(self) -> None:
        """Check goals and enqueue a self-improvement compile if warranted.

        When weekly_build_enabled is True, autonomous self-builds are suppressed
        outside the approved build window. Inside the window, only approved goals
        execute. Briefing checks run every tick regardless.
        """
        # Check if weekly briefing is due (runs regardless of governance mode)
        await self._check_briefing_time()

        # Weekly build governance gate
        if self._weekly_build_enabled:
            from datetime import datetime as _dt
            from mother.scheduler import BuildWindow, is_in_build_window, should_suppress_autonomous

            window = BuildWindow(
                start_hour=self._build_window_start_hour,
                end_hour=self._build_window_end_hour,
                day_of_week=self._build_window_day,
            )
            now_dt = _dt.now()

            if should_suppress_autonomous(now_dt, window, self._weekly_briefing_day):
                logger.debug("Weekly governance: self-builds suppressed outside build window")
                return

            # Inside window — run approved goals
            await self._run_build_window()
            return

        # --- Legacy behavior (weekly_build_enabled=False): immediate execution ---

        # Paused after too many consecutive autonomous failures
        if self._autonomous_failures >= self._max_autonomous_failures:
            logger.debug(
                f"Autonomous scheduler paused: {self._autonomous_failures} consecutive failures"
            )
            return

        # Cooldown: don't run too frequently
        now = time.time()
        if self._last_autonomous_compile is not None:
            elapsed = now - self._last_autonomous_compile
            if elapsed < self._autonomous_cooldown:
                return

        # Don't schedule if queue already has pending work
        pending = [r for r in self._queue if r.status == "pending"]
        if pending:
            return

        # Find a critical goal to address
        goal_info = self._find_critical_goal()
        if goal_info:
            goal_id, goal_desc = goal_info
            # Enrich goal with grid context before enqueueing
            enriched = self._enrich_goal_for_build(goal_desc)
            input_text = f"[SELF-IMPROVEMENT] {enriched}"
            # Inject recent feedback hints to improve compile success
            if hasattr(self, '_outcomes') and self._outcomes:
                from mother.governor_feedback import analyze_outcomes, generate_compiler_prompt_patch
                recent_outcomes = self._outcomes[-20:]
                report = analyze_outcomes(recent_outcomes)
                patch = generate_compiler_prompt_patch(report)
                if patch.strip():
                    input_text = f"{patch}\n\n{input_text}"
            self._last_autonomous_compile = now
            try:
                await self.enqueue(input_text, domain="software", priority=-1)
                # Track attempt on the goal
                try:
                    from mother.goals import GoalStore
                    db_path = Path.home() / ".motherlabs" / "history.db"
                    gs = GoalStore(db_path)
                    gs.increment_attempt(goal_id)
                    gs.update_status(goal_id, "in_progress")
                    gs.close()
                except Exception as e:
                    logger.debug(f"Goal attempt tracking skipped: {e}")
                logger.info(f"Autonomous scheduler: enqueued self-improvement compile for goal #{goal_id}: {goal_desc[:80]}")
            except ValueError:
                logger.debug("Autonomous scheduler: queue full, skipping")
            return

        # L3: No critical goal — check if self-compile is due
        if self._should_self_compile():
            self._last_autonomous_compile = now
            try:
                input_text = "[SELF-COMPILE]"
                # Inject recent feedback hints to improve compile success
                if hasattr(self, '_outcomes') and self._outcomes:
                    from mother.governor_feedback import analyze_outcomes, analyze_recent_builds, generate_compiler_prompt_patch
                    db_path = Path.home() / ".motherlabs" / "history.db"
                    report = analyze_recent_builds(str(db_path), 14)
                    patch = generate_compiler_prompt_patch(report)
                    if patch.strip():
                        input_text = f"{patch}\n\n{input_text}"
                await self.enqueue(input_text, domain="software", priority=-2)
                logger.info("Autonomous scheduler: enqueued self-compile (stale or absent)")
            except ValueError:
                logger.debug("Autonomous scheduler: queue full, skipping self-compile")

    def configure_weekly_build(
        self,
        *,
        enabled: bool = True,
        briefing_day: int = 6,
        briefing_hour: int = 10,
        window_start: int = 22,
        window_end: int = 6,
        window_day: int = 6,
        max_per_window: int = 10,
        briefing_callback: Any = None,
        report_callback: Any = None,
    ) -> None:
        """Configure weekly build governance from MotherConfig values."""
        self._weekly_build_enabled = enabled
        self._weekly_briefing_day = briefing_day
        self._weekly_briefing_hour = briefing_hour
        self._build_window_start_hour = window_start
        self._build_window_end_hour = window_end
        self._build_window_day = window_day
        self._build_max_per_window = max_per_window
        self._briefing_callback = briefing_callback
        self._report_callback = report_callback

    async def _run_build_window(self) -> None:
        """Execute approved goals during the build window.

        Pulls approved goals from GoalStore, runs each via the compile function,
        collects BuildResults, and saves an ExecutionReport when done.
        """
        try:
            from mother.goals import GoalStore
            from mother.scheduler import (
                BuildResult, ExecutionReport, ScheduleStore,
                format_report_markdown,
            )
            from datetime import datetime as _dt

            db_path = Path.home() / ".motherlabs" / "history.db"
            maps_path = Path.home() / ".motherlabs" / "maps.db"

            gs = GoalStore(db_path)
            approved_goals = gs.batch_for_window(limit=self._build_max_per_window)

            if not approved_goals:
                gs.close()
                return

            # Check if we already ran this window (avoid re-running)
            sched = ScheduleStore(maps_path)
            week = sched.current_week()
            if week and week["status"] == "completed":
                sched.close()
                gs.close()
                return
            if week and week["status"] == "executing":
                # Already executing — skip (re-entrant protection)
                sched.close()
                gs.close()
                return

            # Mark as executing
            if week:
                sched.set_status(week["week_start"], "executing")
            week_start = week["week_start"] if week else _dt.now().strftime("%Y-%m-%d")

            results: list = []
            for goal in approved_goals:
                start_time = time.time()
                try:
                    gs.update_status(goal.goal_id, "in_progress")
                    gs.increment_attempt(goal.goal_id)

                    if self._compile_fn:
                        enriched = self._enrich_goal_for_build(goal.description)
                        input_text = f"[SELF-IMPROVEMENT] {enriched}"
                        await self._compile_fn(input_text, "software")

                        duration = time.time() - start_time
                        gs.update_status(goal.goal_id, "done",
                                         completion_note="Built in weekly window")
                        results.append(BuildResult(
                            goal_id=str(goal.goal_id),
                            success=True,
                            duration_seconds=duration,
                            cost_usd=0.0,  # cost tracking happens elsewhere
                            summary=goal.description[:80],
                        ))
                        self._autonomous_failures = 0
                        logger.info(f"Build window: completed goal #{goal.goal_id}")
                    else:
                        results.append(BuildResult(
                            goal_id=str(goal.goal_id),
                            success=False,
                            duration_seconds=0.0,
                            cost_usd=0.0,
                            summary=goal.description[:80],
                            error="No compile function configured",
                        ))
                except Exception as e:
                    duration = time.time() - start_time
                    gs.update_status(goal.goal_id, "active",
                                     progress_note=f"Build failed: {str(e)[:100]}")
                    results.append(BuildResult(
                        goal_id=str(goal.goal_id),
                        success=False,
                        duration_seconds=duration,
                        cost_usd=0.0,
                        summary=goal.description[:80],
                        error=str(e)[:200],
                    ))
                    self._autonomous_failures += 1
                    logger.warning(f"Build window: goal #{goal.goal_id} failed: {e}")

            # Generate and save report
            successes = sum(1 for r in results if r.success)
            failures = sum(1 for r in results if not r.success)
            skipped = len(approved_goals) - len(results)
            total_cost = sum(r.cost_usd for r in results)

            report = ExecutionReport(
                window_date=week_start,
                results=results,
                total_cost=total_cost,
                success_count=successes,
                failure_count=failures,
                skipped_count=skipped,
                summary=f"Build window completed: {successes} succeeded, {failures} failed, {skipped} skipped.",
            )
            sched.save_report(week_start, report)
            sched.close()
            gs.close()

            # Notify chat about completed report
            if self._report_callback:
                try:
                    await self._report_callback(report)
                except Exception as e:
                    logger.debug(f"Report callback failed: {e}")

            logger.info(f"Build window complete: {successes}/{len(results)} succeeded")

        except Exception as e:
            logger.debug(f"Build window execution skipped: {e}")

    async def _check_briefing_time(self) -> None:
        """Check if it's time to generate and present the weekly briefing.

        Runs every scheduler tick. Generates a briefing when:
        1. Now is past the next briefing time
        2. No briefing exists for this week yet
        """
        try:
            from datetime import datetime as _dt
            from mother.scheduler import (
                ScheduleStore, generate_briefing, next_briefing_time,
            )
            from mother.goals import GoalStore

            now = _dt.now()
            nxt = next_briefing_time(now, self._weekly_briefing_day, self._weekly_briefing_hour)

            # If next briefing is in the future and more than 6 days away,
            # it means we're past this week's briefing time
            if (nxt - now).days >= 6:
                # We might be past briefing time — check if we already have one
                maps_path = Path.home() / ".motherlabs" / "maps.db"
                sched = ScheduleStore(maps_path)
                week = sched.current_week()

                # Determine this week's start date
                week_start = (now - __import__('datetime').timedelta(days=now.weekday())).strftime("%Y-%m-%d")

                if week and week["week_start"] == week_start:
                    # Already have a briefing for this week
                    sched.close()
                    return

                # Generate briefing from accumulated goals
                db_path = Path.home() / ".motherlabs" / "history.db"
                if not db_path.exists():
                    sched.close()
                    return

                gs = GoalStore(db_path)
                active_goals = gs.pending_briefing_goals()
                gs.close()

                if not active_goals:
                    sched.close()
                    return

                # Convert Goal objects to dicts for generate_briefing
                goal_dicts = [
                    {
                        "goal_id": g.goal_id,
                        "description": g.description,
                        "source": g.source,
                        "priority": g.priority,
                        "status": g.status,
                        "timestamp": g.timestamp,
                        "attempt_count": g.attempt_count,
                        "engagement_count": g.engagement_count,
                    }
                    for g in active_goals
                ]

                briefing = generate_briefing(goal_dicts, week_start)
                sched.save_briefing(briefing)
                sched.close()

                # Notify chat
                if self._briefing_callback:
                    try:
                        await self._briefing_callback(briefing)
                    except Exception as e:
                        logger.debug(f"Briefing callback failed: {e}")

                logger.info(f"Weekly briefing generated: {briefing.total_goals} goals for week of {week_start}")

        except Exception as e:
            logger.debug(f"Briefing check skipped: {e}")

    def _enrich_goal_for_build(self, goal_description: str) -> str:
        """Enrich goal with grid context before enqueueing.

        Constructs a minimal ImprovementGoal and calls goal_to_actionable()
        to expand postcodes into human-readable territory descriptions.
        Falls back to original description on any error.
        """
        try:
            from mother.goal_generator import ImprovementGoal, goal_to_actionable
            from kernel.store import load_grid

            # Try to get grid cells for context
            grid_cells = None
            grid = load_grid("compiler-self-desc")
            if grid is None:
                grid = load_grid("session")
            if grid:
                grid_cells = [
                    (pc_key, cell.confidence, cell.fill.name, cell.primitive)
                    for pc_key, cell in grid.cells.items()
                ]

            # Construct a minimal ImprovementGoal for enrichment
            goal = ImprovementGoal(
                goal_id="sched-0",
                priority="high",
                category="confidence",
                description=goal_description,
                source="scheduler",
            )
            enriched = goal_to_actionable(goal, grid_cells=grid_cells)
            return enriched.description
        except Exception as e:
            logger.debug(f"Goal enrichment skipped: {e}")
            return goal_description

    # Maximum intent chain depth — prevents unbounded recursive exploration.
    # Each successful [DEPTH-EXPLORE] can spawn children, each incrementing depth.
    # At MAX_CHAIN_DEPTH, no further chains are enqueued.
    MAX_CHAIN_DEPTH: int = 3

    async def _enqueue_depth_chains(
        self, result: Any, domain: str = "software", parent_depth: int = 0,
    ) -> int:
        """Enqueue depth-explore compiles for unexplored endpoints from a compilation.

        Checks CompileResult.depth_chains (populated by endpoint_extractor) and
        enqueues the top chains as [DEPTH-EXPLORE] compile requests. Chains are
        capped to avoid flooding the queue.

        parent_depth: chain generation of the parent request (0 = root).
        Children are enqueued at parent_depth + 1.
        Rejects if parent_depth >= MAX_CHAIN_DEPTH.

        Returns count of chains enqueued.
        """
        # Enforce depth limit — no further chaining beyond MAX_CHAIN_DEPTH
        child_depth = parent_depth + 1
        if parent_depth >= self.MAX_CHAIN_DEPTH:
            logger.info(
                f"Intent chaining: depth limit reached ({parent_depth}/{self.MAX_CHAIN_DEPTH}), "
                "suppressing further chains"
            )
            return 0

        chains = getattr(result, "depth_chains", None)
        if not chains:
            return 0

        # Only chain from successful compilations
        if not getattr(result, "success", False):
            return 0

        # Rate limit: max N depth-chain enqueues per hour
        now = time.time()
        one_hour_ago = now - 3600
        self._depth_chain_enqueue_times = [
            t for t in self._depth_chain_enqueue_times if t > one_hour_ago
        ]
        remaining_budget = self._depth_chain_max_per_hour - len(self._depth_chain_enqueue_times)
        if remaining_budget <= 0:
            logger.info(
                f"Intent chaining: rate limit reached "
                f"({self._depth_chain_max_per_hour}/hour), suppressing"
            )
            return 0

        # Don't chain if already at queue capacity
        pending = [r for r in self._queue if r.status == "pending"]
        max_new = max(0, self.config.max_queue_size - len(self._queue) - 1)
        if max_new <= 0:
            return 0

        # Cap at 3 chains per compilation, limited by queue space and rate budget
        max_chains = min(3, max_new, remaining_budget)
        enqueued = 0

        for chain in chains[:max_chains]:
            intent_text = chain.get("intent_text", "")
            if not intent_text:
                continue

            # Don't re-explore if we already have a similar pending entry
            if any(intent_text[:80] in r.input_text for r in pending):
                continue

            try:
                chain_type = chain.get("chain_type", "frontier")
                priority_val = chain.get("priority", 0.5)
                # Depth explores are lower priority than self-improvement
                queue_priority = -3 + priority_val

                req = await self.enqueue(
                    f"[DEPTH-EXPLORE:{chain_type}] {intent_text}",
                    domain=domain,
                    priority=int(queue_priority),
                )
                req.chain_depth = child_depth
                enqueued += 1
                self._depth_chain_enqueue_times.append(time.time())
                logger.info(
                    f"Intent chaining: enqueued {chain_type} explore "
                    f"(depth={child_depth}, priority={priority_val:.2f}): {intent_text[:80]}"
                )
            except ValueError:
                logger.debug("Intent chaining: queue full, stopping chain enqueue")
                break

        return enqueued

    def _find_critical_goal(self) -> Optional[tuple]:
        """Find the highest-priority actionable goal from GoalStore.

        Uses next_actionable_safe() to skip goals conflicting with in-progress work.
        Returns (goal_id, description) tuple or None.
        """
        try:
            from mother.goals import GoalStore
            db_path = Path.home() / ".motherlabs" / "history.db"
            if not db_path.exists():
                return None

            gs = GoalStore(db_path)
            goal = gs.next_actionable_safe(max_attempts=5)
            gs.close()

            if goal is None:
                return None

            return (goal.goal_id, goal.description)
        except Exception as e:
            logger.debug(f"Goal lookup skipped: {e}")
            return None

    def _should_self_compile(self) -> bool:
        """Check if the compiler self-description grid is stale or absent.

        Returns True if a self-compile is warranted. Convergence-gated:
        if the self-desc grid has fill_rate > 0.8, the threshold extends
        from 24h to 7 days (the fixed point narrows the loop).
        """
        try:
            from kernel.store import load_grid, list_maps
            maps = list_maps()
            self_desc = next((m for m in maps if m["id"] == "compiler-self-desc"), None)
            if self_desc is None:
                return True
            age_hours = (time.time() - self_desc["updated"]) / 3600
            grid = load_grid("compiler-self-desc")
            if grid and grid.fill_rate > 0.8:
                return age_hours >= self._SELF_COMPILE_CONVERGED_HOURS
            return age_hours >= self._SELF_COMPILE_STALE_HOURS
        except Exception as e:
            logger.debug(f"Self-compile staleness check failed: {e}")
            return False

    def _record_feedback(self, request: CompileRequest, result: Any) -> None:
        """Record compilation outcome for the feedback loop.

        Extracts trust dimensions from the result and logs the outcome.
        Non-blocking — failures here never affect the compilation result.
        """
        try:
            from mother.governor_feedback import CompilationOutcome, analyze_outcomes, generate_compiler_prompt_patch

            # Extract verification scores from result
            # Handle both CompileResult objects and dict results (headless daemon)
            def _get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            verification = _get(result, "verification", {}) or {}
            trust_score = 0.0

            def _dim(key: str) -> float:
                val = verification.get(key, 0)
                if isinstance(val, dict):
                    return float(val.get("score", 0))
                return float(val) if val else 0.0

            trust_score = sum(_dim(d) for d in (
                "completeness", "consistency", "coherence", "traceability"
            )) / 4.0

            _blueprint = _get(result, "blueprint", {}) or {}
            _components = _blueprint.get("components", []) if isinstance(_blueprint, dict) else []

            outcome = CompilationOutcome(
                compile_id=f"daemon-{int(request.submitted_at)}",
                input_summary=request.input_text[:200],
                trust_score=trust_score,
                completeness=_dim("completeness"),
                consistency=_dim("consistency"),
                coherence=_dim("coherence"),
                traceability=_dim("traceability"),
                component_count=len(_components),
                rejected=not _get(result, "success", False),
                rejection_reason=_get(result, "error", "") or "",
                domain=request.domain,
            )

            if not hasattr(self, "_outcomes"):
                self._outcomes: list = []
            self._outcomes.append(outcome)

            # Keep last 50
            if len(self._outcomes) > 50:
                self._outcomes = self._outcomes[-50:]

            # Record to journal for persistent analysis
            try:
                journal_db = Path.home() / ".motherlabs" / "history.db"
                journal = BuildJournal(journal_db)
                dims_dict = {
                    "completeness": outcome.completeness,
                    "consistency": outcome.consistency,
                    "coherence": outcome.coherence,
                    "traceability": outcome.traceability,
                    "actionability": outcome.actionability,
                    "specificity": outcome.specificity,
                    "codegen_readiness": outcome.codegen_readiness,
                }
                entry = JournalEntry(
                    event_type="compile",
                    description=outcome.input_summary,
                    success=not outcome.rejected,
                    trust_score=outcome.trust_score,
                    component_count=outcome.component_count,
                    cost_usd=0.0,
                    domain=outcome.domain,
                    error_summary=outcome.rejection_reason,
                    duration_seconds=time.time() - request.submitted_at,
                    project_path="daemon",
                    weakest_dimension=min(dims_dict, key=dims_dict.get),
                    dimension_scores=json.dumps(dims_dict),
                )
                journal.record(entry)
                journal.close()
            except Exception as e:
                logger.debug(f"Build journal recording skipped: {e}")

            report = analyze_outcomes(self._outcomes)
            logger.info(
                f"Feedback: {report.outcomes_analyzed} outcomes, "
                f"trend={report.trend}, rejection_rate={report.rejection_rate:.1%}"
            )

            # --- World grid: fill META cells from feedback ---
            try:
                from kernel.store import load_grid, save_grid
                from kernel.ops import fill as grid_fill
                import time as _feedback_time

                world_grid = load_grid("world")
                if world_grid:
                    _now = _feedback_time.time()
                    _meta_fills = 0

                    # Learned patterns → MET.MEM cells
                    try:
                        from kernel.memory import load_patterns
                        patterns = load_patterns(min_confidence=0.5)
                        for p in patterns[:5]:  # cap to avoid flooding
                            grid_fill(
                                world_grid, "MET.MEM.DOM.WHAT.MTH",
                                primitive=p.description[:60],
                                content=p.description + (f". Fix: {p.remediation}" if p.remediation else ""),
                                confidence=min(p.confidence, 0.9),
                                source=(f"observation:compiler:{_now}",),
                            )
                            _meta_fills += 1
                    except Exception as e:
                        logger.debug(f"Pattern→META fill skipped: {e}")

                    # Generated goals → MET.GOL cells
                    try:
                        for w in report.weaknesses[:3]:  # cap to top 3
                            grid_fill(
                                world_grid, "MET.GOL.DOM.WHY.MTH",
                                primitive=f"improve_{w.dimension}"[:60],
                                content=f"Weakness: {w.dimension} at {w.mean_score:.0f}% (severity={w.severity})",
                                confidence=0.5,
                                source=(f"observation:feedback:{_now}",),
                            )
                            _meta_fills += 1
                    except Exception as e:
                        logger.debug(f"Goal→META fill skipped: {e}")

                    if _meta_fills > 0:
                        save_grid(world_grid, "world", name="Mother World Model")
                        logger.debug(f"World grid: filled {_meta_fills} META cell(s) from feedback")
            except Exception as e:
                logger.debug(f"World grid META fill skipped: {e}")

            # Persist structural source facts on self-compile
            if "[SELF-COMPILE]" in request.input_text:
                try:
                    from mother.source_reader import read_codebase, source_snapshot_to_facts
                    from mother.knowledge_base import KnowledgeFact, save_facts
                    import os
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    snapshot = read_codebase(project_root)
                    raw_facts = source_snapshot_to_facts(snapshot)
                    kf_list = [KnowledgeFact(**f) for f in raw_facts]
                    saved = save_facts(kf_list)
                    logger.info(f"Self-compile: persisted {saved} structural source facts")
                except Exception as e:
                    logger.debug(f"Source fact persistence skipped: {e}")

        except Exception as e:
            logger.debug(f"Feedback recording skipped: {e}")

    def _sync_goals(self) -> None:
        """Generate improvement goals from in-memory outcomes and add to GoalStore.

        Uses self._outcomes (populated by _record_feedback) instead of importing
        from core.outcome_store — respects the bridge.py boundary.
        Additive — failures never affect compilation.
        """
        try:
            from mother.governor_feedback import analyze_outcomes
            from mother.goal_generator import (
                ImprovementGoal,
                goals_from_feedback,
                goals_from_compression_losses,
                generate_goal_set,
            )
            from mother.goals import GoalStore

            if not hasattr(self, "_outcomes") or not self._outcomes:
                return

            report = analyze_outcomes(self._outcomes)
            weaknesses = [
                (w.dimension, w.severity, w.mean_score)
                for w in report.weaknesses
            ]
            feedback_goals = goals_from_feedback(
                weaknesses, report.rejection_rate, report.trend
            )

            # Extract compression loss goals from in-memory outcomes
            cat_freq: dict[str, int] = {}
            compilations_with_losses = 0
            for o in self._outcomes:
                cats = getattr(o, "compression_loss_categories", ())
                if cats:
                    compilations_with_losses += 1
                    for cat, _sev in cats:
                        cat_freq[cat] = cat_freq.get(cat, 0) + 1
            compression_goals = goals_from_compression_losses(
                cat_freq, compilations_with_losses
            )

            # Extract pattern-driven goals from learned patterns
            pattern_goals: list = []
            try:
                from kernel.memory import load_patterns
                patterns = load_patterns(min_confidence=0.5)
                for p in patterns:
                    if p.frequency >= 3 and p.confidence >= 0.6:
                        desc = f"Recurring pattern: {p.description}"
                        if p.remediation:
                            desc += f". Suggested fix: {p.remediation}"
                        pattern_goals.append(ImprovementGoal(
                            goal_id=f"pattern-{p.pattern_id}",
                            priority="high" if p.frequency >= 5 else "medium",
                            category="pattern",
                            description=desc,
                            source="pattern_detector",
                        ))
            except Exception as e:
                logger.debug(f"Pattern goal extraction skipped: {e}")

            goal_set = generate_goal_set(
                pattern_goals, feedback_goals, [], compression_goals=compression_goals
            )

            if not goal_set.goals:
                return

            _priority_map = {
                "critical": "urgent",
                "high": "high",
                "medium": "normal",
                "low": "low",
            }

            db_path = Path.home() / ".motherlabs" / "history.db"
            gs = GoalStore(db_path)
            existing = gs.active(limit=50)

            from mother.goals import goal_dedup_key
            existing_keys = [goal_dedup_key(g.description) for g in existing]

            added = 0
            for goal in goal_set.goals:
                key = goal_dedup_key(goal.description)
                if any(key in ek or ek in key for ek in existing_keys):
                    continue
                mapped = _priority_map.get(goal.priority, "normal")
                gs.add(description=goal.description, source="system", priority=mapped)
                existing_keys.append(key)  # prevent intra-batch dupes
                added += 1

            gs.close()

            if added > 0:
                logger.info(f"Goal sync: {added} new goal(s) from feedback")

        except Exception as e:
            logger.debug(f"Goal sync skipped: {e}")
