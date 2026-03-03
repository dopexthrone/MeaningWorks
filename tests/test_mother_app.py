"""
Phase 0: Foundation tests for Mother TUI app.

Tests app instantiation, first-run detection routing, keybindings,
CSS loading, main entry point, and PID lockfile.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from mother.app import (
    MotherApp,
    main,
    acquire_lock,
    release_lock,
    _is_pid_alive,
)
from mother import __version__


class TestMotherAppInstantiation:
    """Test that MotherApp can be created."""

    def test_creates_app(self):
        app = MotherApp()
        assert app.TITLE == "Mother"

    def test_accepts_config_path(self):
        app = MotherApp(config_path="/tmp/test.json")
        assert app._config_path == "/tmp/test.json"

    def test_css_path_set(self):
        app = MotherApp()
        assert app.CSS_PATH in ("mother.tcss", "mother_alien.tcss")

    def test_version_exists(self):
        assert __version__ == "0.1.0"


class TestKeybindings:
    """Test that keybindings are registered."""

    def test_quit_binding_registered(self):
        app = MotherApp()
        keys = {b.key for b in app.BINDINGS}
        assert "ctrl+q" in keys

    def test_settings_binding_registered(self):
        app = MotherApp()
        keys = {b.key for b in app.BINDINGS}
        assert "ctrl+comma" in keys


class TestMainEntryPoint:
    """Test the main() function."""

    @patch("mother.app.acquire_lock", return_value=True)
    @patch("mother.app.release_lock")
    @patch("mother.app.MotherApp")
    @patch("sys.argv", ["mother"])
    def test_main_creates_and_runs_app(self, MockApp, mock_release, mock_acquire):
        instance = MagicMock()
        MockApp.return_value = instance
        main()
        MockApp.assert_called_once_with(config_path=None)
        instance.run.assert_called_once()
        mock_acquire.assert_called_once()
        mock_release.assert_called_once()

    @patch("mother.app.acquire_lock", return_value=True)
    @patch("mother.app.release_lock")
    @patch("mother.app.MotherApp")
    @patch("sys.argv", ["mother", "--config", "/tmp/test.json"])
    def test_main_with_config_arg(self, MockApp, mock_release, mock_acquire):
        instance = MagicMock()
        MockApp.return_value = instance
        main()
        MockApp.assert_called_once_with(config_path="/tmp/test.json")
        instance.run.assert_called_once()

    @patch("sys.argv", ["mother", "--help"])
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("sys.argv", ["mother", "--version"])
    def test_main_version_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0

    @patch("mother.app.acquire_lock", return_value=False)
    @patch("sys.argv", ["mother"])
    def test_main_exits_if_already_running(self, mock_acquire):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("mother.app.acquire_lock", return_value=True)
    @patch("mother.app.release_lock")
    @patch("mother.app.MotherApp")
    @patch("sys.argv", ["mother"])
    def test_main_releases_lock_on_exception(self, MockApp, mock_release, mock_acquire):
        instance = MagicMock()
        instance.run.side_effect = RuntimeError("crash")
        MockApp.return_value = instance
        with pytest.raises(RuntimeError):
            main()
        mock_release.assert_called_once()


class TestPidLockfile:
    """Test PID lockfile acquire/release."""

    def test_acquire_creates_file(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        assert acquire_lock(pid_path) is True
        assert pid_path.exists()
        assert int(pid_path.read_text().strip()) == os.getpid()
        release_lock(pid_path)

    def test_acquire_blocks_second_instance(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        assert acquire_lock(pid_path) is True
        # Same PID — still alive, so second acquire should fail.
        assert acquire_lock(pid_path) is False
        release_lock(pid_path)

    def test_stale_pid_overwritten(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        # Write a PID that definitely doesn't exist.
        pid_path.write_text("9999999")
        assert acquire_lock(pid_path) is True
        assert int(pid_path.read_text().strip()) == os.getpid()
        release_lock(pid_path)

    def test_corrupt_pid_file_overwritten(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        pid_path.write_text("not-a-number")
        assert acquire_lock(pid_path) is True
        release_lock(pid_path)

    def test_release_removes_file(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        acquire_lock(pid_path)
        release_lock(pid_path)
        assert not pid_path.exists()

    def test_release_only_removes_own_pid(self, tmp_path):
        pid_path = tmp_path / "mother.pid"
        pid_path.write_text("12345")
        release_lock(pid_path)
        # File should still exist — PID doesn't match ours.
        assert pid_path.exists()

    def test_release_no_file_is_noop(self, tmp_path):
        pid_path = tmp_path / "nonexistent.pid"
        release_lock(pid_path)  # Should not raise.

    def test_acquire_creates_parent_dirs(self, tmp_path):
        pid_path = tmp_path / "nested" / "dir" / "mother.pid"
        assert acquire_lock(pid_path) is True
        assert pid_path.exists()
        release_lock(pid_path)


class TestIsPidAlive:
    """Test PID liveness check."""

    def test_own_pid_alive(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid_dead(self):
        assert _is_pid_alive(9999999) is False

    def test_zero_pid(self):
        # PID 0 sends signal to every process in the group — PermissionError expected.
        # Either alive (PermissionError) or not, but shouldn't crash.
        result = _is_pid_alive(0)
        assert isinstance(result, bool)


class TestCSSLoads:
    """Test that CSS file exists and is valid."""

    def test_css_file_exists(self):
        css_path = Path(__file__).parent.parent / "mother" / "mother.tcss"
        assert css_path.exists(), f"CSS file not found at {css_path}"

    def test_css_not_empty(self):
        css_path = Path(__file__).parent.parent / "mother" / "mother.tcss"
        content = css_path.read_text()
        assert len(content) > 100, "CSS file seems too small"
