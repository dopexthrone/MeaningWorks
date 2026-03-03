"""Tests for mother/daemon.py — daemon/overnight mode."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from mother.daemon import (
    CompileRequest,
    DaemonConfig,
    DaemonMode,
)


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# --- Fixtures ---

@pytest.fixture
def daemon(tmp_path):
    """DaemonMode with temp config dir."""
    return DaemonMode(config_dir=tmp_path)


@pytest.fixture
def daemon_with_compile(tmp_path):
    """DaemonMode with a mock compile function."""
    async def mock_compile(text, domain):
        return {"output": f"compiled: {text[:20]}"}

    return DaemonMode(
        config_dir=tmp_path,
        compile_fn=mock_compile,
    )


# --- DaemonConfig ---

class TestDaemonConfig:
    def test_defaults(self):
        cfg = DaemonConfig()
        assert cfg.health_check_interval == 300
        assert cfg.max_queue_size == 10
        assert cfg.auto_heal is True
        assert cfg.idle_shutdown_hours == 0
        assert cfg.log_file is None

    def test_custom(self):
        cfg = DaemonConfig(
            health_check_interval=60,
            max_queue_size=5,
            auto_heal=False,
            idle_shutdown_hours=8,
        )
        assert cfg.health_check_interval == 60
        assert cfg.max_queue_size == 5
        assert cfg.auto_heal is False
        assert cfg.idle_shutdown_hours == 8


# --- CompileRequest ---

class TestCompileRequest:
    def test_defaults(self):
        req = CompileRequest(input_text="build a thing")
        assert req.input_text == "build a thing"
        assert req.domain == "software"
        assert req.priority == 0
        assert req.status == "pending"
        assert req.result is None
        assert req.error is None
        assert req.submitted_at > 0

    def test_to_dict(self):
        req = CompileRequest(input_text="test", domain="api", priority=5)
        d = req.to_dict()
        assert d["input_text"] == "test"
        assert d["domain"] == "api"
        assert d["priority"] == 5
        assert d["status"] == "pending"

    def test_from_dict(self):
        d = {
            "input_text": "hello",
            "domain": "process",
            "priority": 3,
            "submitted_at": 1000.0,
            "status": "completed",
        }
        req = CompileRequest.from_dict(d)
        assert req.input_text == "hello"
        assert req.domain == "process"
        assert req.priority == 3
        assert req.submitted_at == 1000.0

    def test_roundtrip(self):
        req = CompileRequest(input_text="roundtrip", domain="api", priority=2)
        d = req.to_dict()
        req2 = CompileRequest.from_dict(d)
        assert req2.input_text == req.input_text
        assert req2.domain == req.domain
        assert req2.priority == req.priority

    def test_status_transitions(self):
        req = CompileRequest(input_text="test")
        assert req.status == "pending"
        req.status = "running"
        assert req.status == "running"
        req.status = "completed"
        assert req.status == "completed"


# --- DaemonMode enqueue ---

class TestEnqueue:
    def test_enqueue_basic(self, daemon):
        req = run(daemon.enqueue("build a dashboard"))
        assert req.input_text == "build a dashboard"
        assert req.status == "pending"
        assert len(daemon.get_queue()) == 1

    def test_enqueue_with_params(self, daemon):
        req = run(daemon.enqueue("build api", domain="api", priority=5))
        assert req.domain == "api"
        assert req.priority == 5

    def test_enqueue_respects_max_queue(self, daemon):
        daemon.config.max_queue_size = 2
        run(daemon.enqueue("req1"))
        run(daemon.enqueue("req2"))
        with pytest.raises(ValueError, match="Queue full"):
            run(daemon.enqueue("req3"))

    def test_enqueue_multiple(self, daemon):
        run(daemon.enqueue("req1", priority=1))
        run(daemon.enqueue("req2", priority=5))
        run(daemon.enqueue("req3", priority=3))
        queue = daemon.get_queue()
        assert len(queue) == 3


# --- Queue ordering ---

class TestQueueOrdering:
    def test_priority_ordering(self, daemon_with_compile):
        """Higher priority requests should be processed first."""
        async def _test():
            await daemon_with_compile.enqueue("low", priority=1)
            await daemon_with_compile.enqueue("high", priority=10)
            await daemon_with_compile.enqueue("mid", priority=5)

            await daemon_with_compile.start()
            await asyncio.sleep(0.3)
            await daemon_with_compile.stop()

            queue = daemon_with_compile.get_queue()
            completed = [r for r in queue if r.status == "completed"]
            assert len(completed) >= 1
            # Sort by completed_at — first completed should be "high"
            completed.sort(key=lambda r: r.completed_at or 0)
            assert completed[0].input_text == "high"

        run(_test())


# --- Start/Stop lifecycle ---

class TestLifecycle:
    def test_start_stop(self, daemon):
        async def _test():
            assert not daemon.is_running
            await daemon.start()
            assert daemon.is_running
            await daemon.stop()
            assert not daemon.is_running
        run(_test())

    def test_double_start(self, daemon):
        async def _test():
            await daemon.start()
            await daemon.start()  # should not crash
            assert daemon.is_running
            await daemon.stop()
        run(_test())

    def test_double_stop(self, daemon):
        async def _test():
            await daemon.start()
            await daemon.stop()
            await daemon.stop()  # should not crash
            assert not daemon.is_running
        run(_test())


# --- status ---

class TestStatus:
    def test_status_not_running(self, daemon):
        s = daemon.status()
        assert s["running"] is False
        assert s["queue_pending"] == 0

    def test_status_running(self, daemon):
        async def _test():
            await daemon.start()
            s = daemon.status()
            assert s["running"] is True
            assert s["started_at"] is not None
            assert s["uptime_seconds"] >= 0
            await daemon.stop()
        run(_test())

    def test_status_with_queue(self, daemon):
        run(daemon.enqueue("req1"))
        s = daemon.status()
        assert s["queue_pending"] == 1


# --- save_queue / load_queue ---

class TestQueuePersistence:
    def test_save_queue(self, daemon, tmp_path):
        run(daemon.enqueue("req1"))
        run(daemon.enqueue("req2", priority=5))
        path = tmp_path / "queue.json"
        daemon.save_queue(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["input_text"] == "req1"

    def test_load_queue(self, tmp_path):
        path = tmp_path / "queue.json"
        data = [
            {"input_text": "req1", "domain": "software", "priority": 0},
            {"input_text": "req2", "domain": "api", "priority": 3},
        ]
        path.write_text(json.dumps(data))

        daemon = DaemonMode(config_dir=tmp_path)
        loaded = daemon.load_queue(path)
        assert loaded == 2
        queue = daemon.get_queue()
        assert len(queue) == 2
        assert queue[0].input_text == "req1"
        assert queue[1].domain == "api"

    def test_save_load_roundtrip(self, daemon, tmp_path):
        run(daemon.enqueue("req1", domain="api", priority=3))
        path = tmp_path / "queue.json"
        daemon.save_queue(path)

        daemon2 = DaemonMode(config_dir=tmp_path)
        daemon2.load_queue(path)
        queue = daemon2.get_queue()
        assert len(queue) == 1
        assert queue[0].input_text == "req1"
        assert queue[0].domain == "api"
        assert queue[0].priority == 3
        assert queue[0].status == "pending"

    def test_load_missing_file(self, tmp_path):
        daemon = DaemonMode(config_dir=tmp_path)
        loaded = daemon.load_queue(tmp_path / "nope.json")
        assert loaded == 0

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{bad json!!!")
        daemon = DaemonMode(config_dir=tmp_path)
        loaded = daemon.load_queue(path)
        assert loaded == 0

    def test_save_only_pending(self, daemon, tmp_path):
        req = run(daemon.enqueue("req1"))
        req.status = "completed"
        run(daemon.enqueue("req2"))
        path = tmp_path / "queue.json"
        daemon.save_queue(path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["input_text"] == "req2"


# --- clear_completed ---

class TestClearCompleted:
    def test_clear_completed(self, daemon):
        req1 = run(daemon.enqueue("req1"))
        req1.status = "completed"
        req2 = run(daemon.enqueue("req2"))
        req2.status = "failed"
        run(daemon.enqueue("req3"))  # still pending

        removed = daemon.clear_completed()
        assert removed == 2
        queue = daemon.get_queue()
        assert len(queue) == 1
        assert queue[0].input_text == "req3"


# --- health loop ---

class TestHealthLoop:
    def test_health_loop_calls_diagnose(self, tmp_path):
        config = DaemonConfig(health_check_interval=0)
        daemon = DaemonMode(config=config, config_dir=tmp_path)

        async def _test():
            with patch.object(daemon._healer, "diagnose", return_value=[]) as mock_diag:
                await daemon.start()
                await asyncio.sleep(0.2)
                await daemon.stop()
                assert mock_diag.call_count >= 1
        run(_test())

    def test_auto_heal_on_degraded(self, tmp_path):
        from mother.infra_healing import HealthCheck

        config = DaemonConfig(health_check_interval=0, auto_heal=True)
        daemon = DaemonMode(config=config, config_dir=tmp_path)

        degraded_check = HealthCheck(
            component="config",
            status="degraded",
            message="test",
            auto_healable=True,
        )

        async def _test():
            with patch.object(daemon._healer, "diagnose", return_value=[degraded_check]):
                with patch.object(daemon._healer, "heal", return_value=[]) as mock_heal:
                    await daemon.start()
                    await asyncio.sleep(0.2)
                    await daemon.stop()
                    assert mock_heal.call_count >= 1
        run(_test())


# --- idle shutdown ---

class TestIdleShutdown:
    def test_idle_shutdown(self, tmp_path):
        config = DaemonConfig(idle_shutdown_hours=0.0001)  # ~0.36s
        daemon = DaemonMode(config=config, config_dir=tmp_path)

        async def _test():
            await daemon.start()
            assert daemon.is_running
            daemon._last_activity = time.time() - 3600
            await asyncio.sleep(1.5)
            assert not daemon.is_running
        run(_test())


# --- no compile function ---

class TestNoCompileFn:
    def test_fails_without_compile_fn(self, daemon):
        async def _test():
            await daemon.enqueue("build something")
            await daemon.start()
            await asyncio.sleep(0.3)
            await daemon.stop()
            queue = daemon.get_queue()
            assert queue[0].status == "failed"
            assert "No compile function" in queue[0].error
        run(_test())
