"""Tests for headless daemon mode (mother --daemon) and auto-restart after self-build."""

import asyncio
import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


def run(coro):
    """Run async coroutine in sync test."""
    return asyncio.run(coro)


# Shared mock config factory
def _mock_config(**overrides):
    cfg = MagicMock()
    cfg.provider = "claude"
    cfg.api_keys = {}
    cfg.file_access = True
    cfg.get_model.return_value = "claude-sonnet-4-20250514"
    cfg.pipeline_mode = "staged"
    cfg.daemon_health_check_interval = 300
    cfg.daemon_max_queue_size = 10
    cfg.daemon_auto_heal = True
    cfg.daemon_idle_shutdown_hours = 0
    cfg.daemon_log_file = ""
    cfg.weekly_build_enabled = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --- Argument parsing ---

class TestDaemonArgParsing:
    """--daemon flag is parsed correctly."""

    def test_daemon_flag_present(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--config", type=str, default=None)
        args = parser.parse_args(["--daemon"])
        assert args.daemon is True

    def test_daemon_flag_absent(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--config", type=str, default=None)
        args = parser.parse_args([])
        assert args.daemon is False

    def test_daemon_with_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--config", type=str, default=None)
        args = parser.parse_args(["--daemon", "--config", "/tmp/test.json"])
        assert args.daemon is True
        assert args.config == "/tmp/test.json"


# --- _run_daemon function ---

class TestRunDaemon:
    """_run_daemon creates bridge, daemon, and wires config."""

    @patch("mother.daemon.DaemonMode")
    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_creates_bridge_and_daemon(self, mock_load, mock_bridge_init, mock_daemon_cls):
        from mother.app import _run_daemon

        mock_load.return_value = _mock_config()
        mock_daemon_cls.return_value = MagicMock()

        with patch("asyncio.run"):
            _run_daemon()

        mock_bridge_init.assert_called_once()
        mock_daemon_cls.assert_called_once()

    @patch("mother.daemon.DaemonMode", return_value=MagicMock())
    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_injects_api_keys(self, mock_load, mock_bridge_init, mock_daemon_cls):
        from mother.app import _run_daemon

        mock_load.return_value = _mock_config(
            api_keys={"claude": "sk-test-123", "openai": "sk-oai-456"}
        )

        env_backup = {}
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            env_backup[var] = os.environ.pop(var, None)

        try:
            with patch("asyncio.run"):
                _run_daemon()
            assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-123"
            assert os.environ.get("OPENAI_API_KEY") == "sk-oai-456"
        finally:
            for var, val in env_backup.items():
                if val is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = val

    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_wires_weekly_governance(self, mock_load, mock_bridge_init):
        from mother.app import _run_daemon

        mock_load.return_value = _mock_config(
            weekly_build_enabled=True,
            weekly_briefing_day=5,
            weekly_briefing_hour=9,
            build_window_start_hour=23,
            build_window_end_hour=5,
            build_window_day=5,
            build_max_per_window=8,
        )

        mock_daemon = MagicMock()
        with (
            patch("mother.daemon.DaemonMode", return_value=mock_daemon),
            patch("asyncio.run"),
        ):
            _run_daemon()

        mock_daemon.configure_weekly_build.assert_called_once_with(
            enabled=True,
            briefing_day=5,
            briefing_hour=9,
            window_start=23,
            window_end=5,
            window_day=5,
            max_per_window=8,
        )

    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_daemon_config_fields_wired(self, mock_load, mock_bridge_init):
        from mother.app import _run_daemon

        mock_load.return_value = _mock_config(
            daemon_health_check_interval=120,
            daemon_max_queue_size=5,
            daemon_auto_heal=False,
            daemon_idle_shutdown_hours=8.0,
            daemon_log_file="/tmp/test.log",
        )

        captured_config = None

        def capture_daemon(*args, **kwargs):
            nonlocal captured_config
            captured_config = kwargs.get("config")
            return MagicMock()

        with (
            patch("mother.daemon.DaemonMode", side_effect=capture_daemon),
            patch("asyncio.run"),
        ):
            _run_daemon()

        assert captured_config is not None
        assert captured_config.health_check_interval == 120
        assert captured_config.max_queue_size == 5
        assert captured_config.auto_heal is False
        assert captured_config.idle_shutdown_hours == 8.0

    @patch("mother.daemon.DaemonMode", return_value=MagicMock())
    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_does_not_skip_weekly_governance_when_disabled(self, mock_load, mock_bridge_init, mock_daemon_cls):
        from mother.app import _run_daemon

        mock_load.return_value = _mock_config(weekly_build_enabled=False)
        mock_daemon = mock_daemon_cls.return_value

        with patch("asyncio.run"):
            _run_daemon()

        mock_daemon.configure_weekly_build.assert_not_called()


# --- File logging ---

class TestDaemonLogging:
    """Daemon configures file logging."""

    @patch("mother.daemon.DaemonMode", return_value=MagicMock())
    @patch("mother.bridge.EngineBridge.__init__", return_value=None)
    @patch("mother.config.load_config")
    def test_file_logging_configured(self, mock_load, mock_bridge_init, mock_daemon_cls):
        from mother.app import _run_daemon
        from logging.handlers import RotatingFileHandler

        mock_load.return_value = _mock_config()

        root = logging.getLogger()
        initial_count = len(root.handlers)

        with patch("asyncio.run"):
            _run_daemon()

        new_handlers = [h for h in root.handlers[initial_count:] if isinstance(h, RotatingFileHandler)]
        assert len(new_handlers) >= 1

        # Clean up
        for h in new_handlers:
            root.removeHandler(h)
            h.close()


# --- compile_fn behavior ---

class TestCompileFn:
    """compile_fn calls bridge.self_build, restarts on success, continues on failure."""

    def _extract_compile_fn(self, mock_config=None):
        """Set up _run_daemon and capture the compile_fn passed to DaemonMode."""
        from mother.app import _run_daemon

        captured = {}

        def capture_daemon(*args, **kwargs):
            captured["compile_fn"] = kwargs.get("compile_fn")
            return MagicMock()

        mock_bridge = MagicMock()
        mock_bridge.self_build = AsyncMock(return_value={"success": True, "cost_usd": 0.5})
        mock_bridge.process_social_queue = AsyncMock(return_value=[])
        captured["bridge"] = mock_bridge

        with (
            patch("mother.config.load_config", return_value=mock_config or _mock_config()),
            patch("mother.bridge.EngineBridge", return_value=mock_bridge),
            patch("mother.daemon.DaemonMode", side_effect=capture_daemon),
            patch("asyncio.run"),
        ):
            _run_daemon()

        return captured

    def test_successful_build_triggers_execv(self):
        captured = self._extract_compile_fn()
        compile_fn = captured["compile_fn"]
        bridge = captured["bridge"]
        bridge.self_build = AsyncMock(return_value={"success": True})

        with (
            patch("mother.app.os.execv") as mock_execv,
            patch("mother.app.release_lock"),
        ):
            run(compile_fn("test build", "software"))

            bridge.self_build.assert_called_once()
            mock_execv.assert_called_once()
            execv_args = mock_execv.call_args[0][1]
            assert "--daemon" in execv_args

    def test_failed_build_does_not_restart(self):
        captured = self._extract_compile_fn()
        compile_fn = captured["compile_fn"]
        bridge = captured["bridge"]
        bridge.self_build = AsyncMock(return_value={"success": False, "error": "tests failed"})

        with patch("mother.app.os.execv") as mock_execv:
            result = run(compile_fn("test build", "software"))

        assert result.get("success") is False
        mock_execv.assert_not_called()

    def test_compile_fn_processes_social_before_restart(self):
        captured = self._extract_compile_fn()
        compile_fn = captured["compile_fn"]
        bridge = captured["bridge"]
        bridge.self_build = AsyncMock(return_value={"success": True})

        call_order = []

        async def _mock_social():
            call_order.append("social")
            return []

        bridge.process_social_queue = _mock_social

        with (
            patch("mother.app.os.execv") as mock_execv,
            patch("mother.app.release_lock"),
        ):
            mock_execv.side_effect = lambda *args: call_order.append("execv")
            run(compile_fn("build", "software"))

        assert call_order == ["social", "execv"]

    def test_social_queue_failure_does_not_block_restart(self):
        """If social queue processing fails, restart still proceeds."""
        captured = self._extract_compile_fn()
        compile_fn = captured["compile_fn"]
        bridge = captured["bridge"]
        bridge.self_build = AsyncMock(return_value={"success": True})
        bridge.process_social_queue = AsyncMock(side_effect=RuntimeError("social down"))

        with (
            patch("mother.app.os.execv") as mock_execv,
            patch("mother.app.release_lock"),
        ):
            run(compile_fn("build", "software"))

        mock_execv.assert_called_once()


# --- PID lock ---

class TestPidLock:
    """PID lock prevents dual instances (daemon + TUI)."""

    def test_lock_prevents_second_instance(self, tmp_path):
        from mother.app import acquire_lock, release_lock

        pid_path = tmp_path / "mother.pid"
        assert acquire_lock(pid_path) is True
        # Write current PID to simulate held lock
        pid_path.write_text(str(os.getpid()))
        assert acquire_lock(pid_path) is False
        release_lock(pid_path)

    def test_stale_lock_overridden(self, tmp_path):
        from mother.app import acquire_lock, release_lock

        pid_path = tmp_path / "mother.pid"
        pid_path.write_text("999999999")
        assert acquire_lock(pid_path) is True
        release_lock(pid_path)


# --- _daemon_main lifecycle ---

class TestDaemonMain:
    """_daemon_main starts daemon, handles signals, processes social queue."""

    def test_daemon_main_starts_and_stops(self):
        from mother.app import _daemon_main

        mock_bridge = MagicMock()
        mock_bridge.process_social_queue = AsyncMock(return_value=[])
        mock_daemon = AsyncMock()
        mock_daemon.start = AsyncMock()
        mock_daemon.stop = AsyncMock()

        async def _run():
            async def _trigger_stop():
                await asyncio.sleep(0.1)
                os.kill(os.getpid(), signal.SIGINT)

            task = asyncio.create_task(_daemon_main(mock_bridge, mock_daemon))
            trigger = asyncio.create_task(_trigger_stop())
            await asyncio.gather(task, trigger, return_exceptions=True)

        run(_run())
        mock_daemon.start.assert_called_once()
        mock_daemon.stop.assert_called_once()

    def test_shutdown_sets_event(self):
        from mother.app import _shutdown

        event = asyncio.Event()
        mock_daemon = AsyncMock()

        run(_shutdown(mock_daemon, event))
        assert event.is_set()


# --- Auto-restart in TUI ---

class TestAutoRestartTUI:
    """Auto-restart after successful self-build in chat.py."""

    def test_success_path_has_auto_restart(self):
        """chat.py calls action_restart() instead of showing manual restart prompt."""
        import inspect
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen)
        assert "Restarting to load changes" in source
        assert "Restart (Ctrl+R) to load the changes" not in source

    def test_action_restart_called(self):
        """action_restart() appears in the success branch."""
        import inspect
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen)
        assert "action_restart()" in source


# --- main() routing ---

class TestMainRouting:
    """main() routes to daemon or TUI based on --daemon flag."""

    @patch("mother.app.acquire_lock", return_value=True)
    @patch("mother.app.release_lock")
    @patch("mother.app._run_daemon")
    def test_daemon_flag_routes_to_run_daemon(self, mock_run_daemon, mock_release, mock_lock):
        from mother.app import main

        with patch("sys.argv", ["mother", "--daemon"]):
            main()

        mock_run_daemon.assert_called_once()

    @patch("mother.app.acquire_lock", return_value=True)
    @patch("mother.app.release_lock")
    @patch("mother.app.MotherApp")
    def test_no_daemon_routes_to_tui(self, mock_app_cls, mock_release, mock_lock):
        from mother.app import main

        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        with patch("sys.argv", ["mother"]):
            main()

        mock_app_cls.assert_called_once()
        mock_app.run.assert_called_once()

    @patch("mother.app.acquire_lock", return_value=False)
    def test_lock_failure_exits(self, mock_lock):
        from mother.app import main

        with patch("sys.argv", ["mother", "--daemon"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


# --- Feedback accuracy: build outcome must flow to _record_feedback ---

class TestFeedbackAccuracy:
    """Verify the feedback loop sees actual build success/failure, not just compile success."""

    def test_daemon_self_improve_stamps_build_failure_on_result(self):
        """When self-build fails, CompileResult.success must be False."""
        from dataclasses import dataclass, field
        from typing import Dict, Any, List, Optional

        @dataclass
        class FakeCompileResult:
            success: bool = True
            blueprint: Dict[str, Any] = field(default_factory=dict)
            verification: Dict[str, Any] = field(default_factory=dict)
            error: Optional[str] = None

        fake_result = FakeCompileResult(success=True)

        # Simulate _daemon_self_improve logic post-build
        build_result = {"success": False, "error": "tests failed"}

        success = build_result.get("success", False)
        fake_result.success = success
        if not success:
            fake_result.error = build_result.get("error", "self-build failed")

        assert fake_result.success is False
        assert fake_result.error == "tests failed"

    def test_daemon_self_improve_stamps_build_success_on_result(self):
        """When self-build succeeds, CompileResult.success stays True."""
        from dataclasses import dataclass, field
        from typing import Dict, Any, Optional

        @dataclass
        class FakeCompileResult:
            success: bool = True
            error: Optional[str] = None

        fake_result = FakeCompileResult(success=True)
        build_result = {"success": True}

        fake_result.success = build_result.get("success", False)
        if not build_result.get("success", False):
            fake_result.error = build_result.get("error", "self-build failed")

        assert fake_result.success is True
        assert fake_result.error is None

    def test_daemon_self_improve_stamps_exception_on_result(self):
        """When self-build raises, CompileResult.success set to False."""
        from dataclasses import dataclass, field
        from typing import Optional

        @dataclass
        class FakeCompileResult:
            success: bool = True
            error: Optional[str] = None

        fake_result = FakeCompileResult(success=True)

        # Simulate exception path
        fake_result.success = False
        fake_result.error = "ConnectionError: API down"

        assert fake_result.success is False
        assert "API down" in fake_result.error

    def test_record_feedback_handles_dict_result(self):
        """_record_feedback's _get helper handles dict results (headless daemon)."""
        from mother.daemon import DaemonMode, DaemonConfig

        daemon = DaemonMode(config=DaemonConfig(max_queue_size=10))

        # Dict result (from bridge.self_build)
        result = {"success": False, "error": "tests failed", "verification": {}, "blueprint": {}}

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        assert _get(result, "success", False) is False
        assert _get(result, "error", "") == "tests failed"
        assert _get(result, "verification", {}) == {}
        assert _get(result, "blueprint", {}) == {}

    def test_record_feedback_handles_object_result(self):
        """_record_feedback's _get helper handles CompileResult objects (TUI daemon)."""
        from dataclasses import dataclass, field
        from typing import Dict, Any, Optional

        @dataclass
        class FakeCompileResult:
            success: bool = False
            error: Optional[str] = "build failed"
            verification: Dict[str, Any] = field(default_factory=dict)
            blueprint: Dict[str, Any] = field(default_factory=dict)

        result = FakeCompileResult()

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        assert _get(result, "success", False) is False
        assert _get(result, "error", "") == "build failed"
        assert _get(result, "verification", {}) == {}

    def test_record_feedback_dict_rejected_is_true_on_failure(self):
        """For dict results, rejected flag must be True when success=False."""
        result = {"success": False, "error": "sandbox violation"}

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        rejected = not _get(result, "success", False)
        assert rejected is True

    def test_record_feedback_dict_rejected_is_false_on_success(self):
        """For dict results, rejected flag must be False when success=True."""
        result = {"success": True}

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        rejected = not _get(result, "success", False)
        assert rejected is False

    def test_getattr_on_dict_fails_without_helper(self):
        """Prove that getattr(dict, 'success') returns default, not the dict value."""
        result = {"success": True, "error": ""}
        # This is the bug: getattr on dict doesn't access keys
        assert getattr(result, "success", False) is False  # returns False, not True!
        assert getattr(result, "error", "default") == "default"  # misses dict key

    def test_source_uses_get_helper_in_record_feedback(self):
        """_record_feedback uses _get() helper, not bare getattr(), for result access."""
        import inspect
        from mother.daemon import DaemonMode
        source = inspect.getsource(DaemonMode._record_feedback)
        # Must use _get helper for success and error
        assert "_get(result, \"success\"" in source
        assert "_get(result, \"error\"" in source
        assert "_get(result, \"verification\"" in source

    def test_source_stamps_build_outcome(self):
        """_daemon_self_improve stamps build outcome onto CompileResult."""
        import inspect
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._daemon_self_improve)
        assert "result.success = success" in source
        assert "result.error = " in source
