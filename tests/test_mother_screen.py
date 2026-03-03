"""
Tests for Mother screen capture bridge.

All tests mocked. No real screen captures.
"""

import asyncio
import base64
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from mother.config import MotherConfig, save_config, load_config


# ---------------------------------------------------------------------------
# Test availability detection
# ---------------------------------------------------------------------------

class TestScreenCaptureAvailability:
    """Test platform detection for screen capture."""

    def test_is_screen_capture_available_returns_bool(self):
        from mother.screen import is_screen_capture_available
        result = is_screen_capture_available()
        assert isinstance(result, bool)

    @patch("mother.screen.sys")
    def test_available_on_macos(self, mock_sys):
        mock_sys.platform = "darwin"
        # Re-evaluate: can't patch module-level, test the function
        from mother.screen import is_screen_capture_available
        # The function reads sys.platform directly, so patch sys module
        with patch("mother.screen.sys.platform", "darwin"):
            assert is_screen_capture_available() is True

    @patch("mother.screen.sys.platform", "linux")
    def test_not_available_on_linux(self):
        from mother.screen import is_screen_capture_available
        assert is_screen_capture_available() is False

    @patch("mother.screen.sys.platform", "win32")
    def test_not_available_on_windows(self):
        from mother.screen import is_screen_capture_available
        assert is_screen_capture_available() is False


# ---------------------------------------------------------------------------
# Test ScreenCaptureBridge creation
# ---------------------------------------------------------------------------

class TestScreenCaptureBridgeCreation:
    """Test ScreenCaptureBridge instantiation."""

    def test_create_default(self):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge()
        assert bridge._enabled is True

    def test_create_disabled(self):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=False)
        assert bridge._enabled is False


# ---------------------------------------------------------------------------
# Test enabled property
# ---------------------------------------------------------------------------

class TestScreenCaptureBridgeEnabled:
    """Test the multi-gate enabled property."""

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    def test_enabled_when_flag_true_and_macos(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        assert bridge.enabled is True

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    def test_disabled_when_flag_false(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=False)
        assert bridge.enabled is False

    @patch("mother.screen.is_screen_capture_available", return_value=False)
    def test_disabled_when_not_macos(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        assert bridge.enabled is False


# ---------------------------------------------------------------------------
# Test capture (sync)
# ---------------------------------------------------------------------------

class TestScreenCaptureSyncCapture:
    """Test the synchronous _capture_sync method."""

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_success(self, mock_exists, mock_unlink, mock_close,
                             mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        # Setup
        mock_mkstemp.return_value = (5, "/tmp/mother_capture_test.png")
        mock_run.return_value = MagicMock(returncode=0)

        test_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG
        expected_b64 = base64.b64encode(test_data).decode("ascii")

        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=test_data)):
            result = bridge._capture_sync()

        assert result == expected_b64
        mock_run.assert_called_once()
        mock_unlink.assert_called_once_with("/tmp/mother_capture_test.png")

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_with_display_flag(self, mock_exists, mock_unlink, mock_close,
                                       mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_mkstemp.return_value = (5, "/tmp/test.png")
        mock_run.return_value = MagicMock(returncode=0)
        test_data = b"PNG_DATA"

        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=test_data)):
            bridge._capture_sync(display=2)

        cmd = mock_run.call_args[0][0]
        assert "-D" in cmd
        assert "2" in cmd

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_with_region(self, mock_exists, mock_unlink, mock_close,
                                 mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_mkstemp.return_value = (5, "/tmp/test.png")
        mock_run.return_value = MagicMock(returncode=0)
        test_data = b"PNG_DATA"

        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=test_data)):
            bridge._capture_sync(region=(100, 200, 300, 400))

        cmd = mock_run.call_args[0][0]
        assert "-R" in cmd
        assert "100,200,300,400" in cmd

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_subprocess_failure(self, mock_exists, mock_unlink, mock_close,
                                        mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_mkstemp.return_value = (5, "/tmp/test.png")
        mock_run.return_value = MagicMock(returncode=1)

        bridge = ScreenCaptureBridge(enabled=True)
        result = bridge._capture_sync()
        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run", side_effect=Exception("boom"))
    @patch("mother.screen.tempfile.mkstemp", return_value=(5, "/tmp/test.png"))
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_exception_returns_none(self, mock_exists, mock_unlink, mock_close,
                                             mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        result = bridge._capture_sync()
        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_empty_file_returns_none(self, mock_exists, mock_unlink, mock_close,
                                              mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_mkstemp.return_value = (5, "/tmp/test.png")
        mock_run.return_value = MagicMock(returncode=0)

        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=b"")):
            result = bridge._capture_sync()

        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=False)
    def test_capture_disabled_returns_none(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        result = bridge._capture_sync()
        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    def test_capture_flag_disabled_returns_none(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=False)
        result = bridge._capture_sync()
        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("screencapture", 10))
    @patch("mother.screen.tempfile.mkstemp", return_value=(5, "/tmp/test.png"))
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_capture_timeout_returns_none(self, mock_exists, mock_unlink, mock_close,
                                           mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)
        result = bridge._capture_sync()
        assert result is None

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp")
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_temp_file_cleanup(self, mock_exists, mock_close, mock_mkstemp, mock_run, mock_avail):
        """Verify temp file is always cleaned up, even on success."""
        from mother.screen import ScreenCaptureBridge

        mock_mkstemp.return_value = (5, "/tmp/mother_capture_cleanup.png")
        mock_run.return_value = MagicMock(returncode=0)
        test_data = b"PNG"

        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=test_data)), \
             patch("mother.screen.os.unlink") as mock_unlink:
            bridge._capture_sync()
            mock_unlink.assert_called_once_with("/tmp/mother_capture_cleanup.png")


# ---------------------------------------------------------------------------
# Test async capture
# ---------------------------------------------------------------------------

class TestScreenCaptureAsync:
    """Test the async capture_screen method."""

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    def test_async_capture_calls_sync(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)

        async def _run():
            with patch.object(bridge, "_capture_sync", return_value="base64data") as mock_sync:
                result = await bridge.capture_screen()
                assert result == "base64data"
                mock_sync.assert_called_once_with(1, None)

        asyncio.run(_run())

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    def test_async_capture_passes_args(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)

        async def _run():
            with patch.object(bridge, "_capture_sync", return_value="data") as mock_sync:
                await bridge.capture_screen(display=2, region=(0, 0, 100, 100))
                mock_sync.assert_called_once_with(2, (0, 0, 100, 100))

        asyncio.run(_run())

    @patch("mother.screen.is_screen_capture_available", return_value=False)
    def test_async_capture_disabled_returns_none(self, mock_avail):
        from mother.screen import ScreenCaptureBridge
        bridge = ScreenCaptureBridge(enabled=True)

        async def _run():
            result = await bridge.capture_screen()
            assert result is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test config integration
# ---------------------------------------------------------------------------

class TestScreenCaptureConfig:
    """Test screen_capture_enabled config field."""

    def test_config_default_disabled(self):
        config = MotherConfig()
        assert config.screen_capture_enabled is False

    def test_config_roundtrip(self, tmp_path):
        config = MotherConfig(screen_capture_enabled=True)
        path = str(tmp_path / "test.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.screen_capture_enabled is True

    def test_config_disabled_roundtrip(self, tmp_path):
        config = MotherConfig(screen_capture_enabled=False)
        path = str(tmp_path / "test.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.screen_capture_enabled is False


# ---------------------------------------------------------------------------
# Test command construction
# ---------------------------------------------------------------------------

class TestScreenCaptureCommandConstruction:
    """Test the screencapture command flags."""

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp", return_value=(5, "/tmp/test.png"))
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_default_command_has_silent_flag(self, mock_exists, mock_unlink, mock_close,
                                             mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_run.return_value = MagicMock(returncode=0)
        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=b"data")):
            bridge._capture_sync()

        cmd = mock_run.call_args[0][0]
        assert "-x" in cmd  # Silent (no shutter sound)
        assert "-t" in cmd
        assert "png" in cmd

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp", return_value=(5, "/tmp/test.png"))
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_default_no_display_flag(self, mock_exists, mock_unlink, mock_close,
                                      mock_mkstemp, mock_run, mock_avail):
        """Display 1 (main) should NOT add -D flag."""
        from mother.screen import ScreenCaptureBridge

        mock_run.return_value = MagicMock(returncode=0)
        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=b"data")):
            bridge._capture_sync(display=1)

        cmd = mock_run.call_args[0][0]
        assert "-D" not in cmd

    @patch("mother.screen.is_screen_capture_available", return_value=True)
    @patch("mother.screen.subprocess.run")
    @patch("mother.screen.tempfile.mkstemp", return_value=(5, "/tmp/test.png"))
    @patch("mother.screen.os.close")
    @patch("mother.screen.os.unlink")
    @patch("mother.screen.os.path.exists", return_value=True)
    def test_no_region_flag_by_default(self, mock_exists, mock_unlink, mock_close,
                                        mock_mkstemp, mock_run, mock_avail):
        from mother.screen import ScreenCaptureBridge

        mock_run.return_value = MagicMock(returncode=0)
        bridge = ScreenCaptureBridge(enabled=True)

        with patch("builtins.open", mock_open(read_data=b"data")):
            bridge._capture_sync()

        cmd = mock_run.call_args[0][0]
        assert "-R" not in cmd
