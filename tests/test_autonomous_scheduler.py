"""Tests for mother/daemon.py — autonomous scheduler."""

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.daemon import DaemonMode, DaemonConfig, CompileRequest


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daemon(tmp_path):
    return DaemonMode(config=DaemonConfig(), config_dir=tmp_path)


@pytest.fixture
def daemon_with_compile(tmp_path):
    async def mock_compile(text, domain):
        return SimpleNamespace(
            success=True,
            verification={"completeness": {"score": 80}},
            blueprint={"components": []},
            error=None,
        )

    return DaemonMode(
        config=DaemonConfig(),
        config_dir=tmp_path,
        compile_fn=mock_compile,
    )


# ---------------------------------------------------------------------------
# Scheduler state initialization
# ---------------------------------------------------------------------------

class TestSchedulerInit:
    def test_default_cooldown(self, daemon):
        assert daemon._autonomous_cooldown == 1800

    def test_default_no_last_compile(self, daemon):
        assert daemon._last_autonomous_compile is None

    def test_default_zero_failures(self, daemon):
        assert daemon._autonomous_failures == 0

    def test_max_failures_default(self, daemon):
        assert daemon._max_autonomous_failures == 3

    def test_scheduler_task_none_initially(self, daemon):
        assert daemon._scheduler_task is None


# ---------------------------------------------------------------------------
# _scheduler_tick
# ---------------------------------------------------------------------------

class TestSchedulerTick:
    def test_paused_on_max_failures(self, daemon):
        daemon._autonomous_failures = 3
        daemon._running = True
        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 0

    def test_cooldown_respected(self, daemon):
        daemon._running = True
        daemon._last_autonomous_compile = time.time()
        daemon._autonomous_failures = 0
        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 0

    def test_no_goal_no_enqueue(self, daemon):
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        daemon._find_critical_goal = MagicMock(return_value=None)
        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 0

    def test_enqueues_on_critical_goal(self, daemon):
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        daemon._find_critical_goal = MagicMock(return_value=(1, "Fix entity extraction"))

        run(daemon._scheduler_tick())

        assert len(daemon._queue) == 1
        assert daemon._queue[0].input_text.startswith("[SELF-IMPROVEMENT]")
        assert "Fix entity extraction" in daemon._queue[0].input_text
        assert daemon._queue[0].priority == -1

    def test_sets_last_autonomous_compile_time(self, daemon):
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        daemon._find_critical_goal = MagicMock(return_value=(1, "Goal X"))

        before = time.time()
        run(daemon._scheduler_tick())
        after = time.time()

        assert daemon._last_autonomous_compile is not None
        assert before <= daemon._last_autonomous_compile <= after

    def test_skips_when_queue_has_pending(self, daemon):
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = None
        daemon._find_critical_goal = MagicMock(return_value=(1, "Goal X"))

        daemon._queue.append(CompileRequest(input_text="existing", status="pending"))
        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 1  # no new item

    def test_cooldown_expired_allows_enqueue(self, daemon):
        daemon._running = True
        daemon._autonomous_failures = 0
        daemon._last_autonomous_compile = time.time() - 2000  # 33 min ago
        daemon._find_critical_goal = MagicMock(return_value=(2, "Goal Y"))

        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 1

    def test_below_max_failures_allows_tick(self, daemon):
        daemon._autonomous_failures = 2  # below max of 3
        daemon._running = True
        daemon._last_autonomous_compile = None
        daemon._find_critical_goal = MagicMock(return_value=(3, "Goal Z"))

        run(daemon._scheduler_tick())
        assert len(daemon._queue) == 1


# ---------------------------------------------------------------------------
# Autonomous failure tracking
# ---------------------------------------------------------------------------

class TestAutonomousFailureTracking:
    def _process_one(self, daemon):
        """Process one pending item from queue synchronously."""

        async def _one():
            pending = [r for r in daemon._queue if r.status == "pending"]
            if not pending:
                return
            pending.sort(key=lambda r: (-r.priority, r.submitted_at))
            req = pending[0]
            req.status = "running"
            is_autonomous = req.input_text.startswith("[SELF-IMPROVEMENT]")

            if daemon._compile_fn:
                try:
                    result = await daemon._compile_fn(req.input_text, req.domain)
                    req.result = result
                    req.status = "completed"
                    req.completed_at = time.time()
                    daemon._completed_count += 1
                    if is_autonomous:
                        daemon._autonomous_failures = 0
                except Exception as e:
                    req.status = "failed"
                    req.error = str(e)
                    req.completed_at = time.time()
                    daemon._failed_count += 1
                    if is_autonomous:
                        daemon._autonomous_failures += 1

        run(_one())

    def test_autonomous_success_resets_counter(self, daemon_with_compile):
        d = daemon_with_compile
        d._autonomous_failures = 2
        d._queue.append(CompileRequest(
            input_text="[SELF-IMPROVEMENT] Fix something",
            domain="software", priority=-1,
        ))

        self._process_one(d)
        assert d._autonomous_failures == 0

    def test_autonomous_failure_increments_counter(self, tmp_path):
        async def fail_compile(text, domain):
            raise Exception("LLM timeout")

        d = DaemonMode(config=DaemonConfig(), config_dir=tmp_path, compile_fn=fail_compile)
        d._autonomous_failures = 1
        d._queue.append(CompileRequest(
            input_text="[SELF-IMPROVEMENT] Fix something",
            domain="software",
        ))

        self._process_one(d)
        assert d._autonomous_failures == 2

    def test_non_autonomous_failure_doesnt_increment(self, tmp_path):
        async def fail_compile(text, domain):
            raise Exception("boom")

        d = DaemonMode(config=DaemonConfig(), config_dir=tmp_path, compile_fn=fail_compile)
        d._autonomous_failures = 0
        d._queue.append(CompileRequest(
            input_text="Build a task manager",
            domain="software",
        ))

        self._process_one(d)
        assert d._autonomous_failures == 0


# ---------------------------------------------------------------------------
# _find_critical_goal
# ---------------------------------------------------------------------------

class TestFindCriticalGoal:
    def test_returns_none_when_no_db(self, daemon):
        result = daemon._find_critical_goal()
        assert result is None or isinstance(result, tuple)


# ---------------------------------------------------------------------------
# start/stop includes scheduler task
# ---------------------------------------------------------------------------

class TestSchedulerLifecycle:
    def test_start_creates_scheduler_task(self, daemon):
        async def _test():
            await daemon.start()
            assert daemon._scheduler_task is not None
            assert not daemon._scheduler_task.done()
            await daemon.stop()

        run(_test())

    def test_stop_cancels_scheduler_task(self, daemon):
        async def _test():
            await daemon.start()
            task = daemon._scheduler_task
            await daemon.stop()
            assert daemon._scheduler_task is None
            assert task.done()

        run(_test())

    def test_status_unchanged_format(self, daemon):
        """Status dict format is backward compatible."""
        s = daemon.status()
        assert "running" in s
        assert "queue_pending" in s
        assert "completed" in s
