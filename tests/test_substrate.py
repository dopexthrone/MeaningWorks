"""Tests for mother/substrate.py — platform portability."""

import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mother.substrate import (
    SubstrateCapabilities,
    SubstrateDetector,
    _command_exists,
    _find_glob,
    _find_spotlight,
    DARWIN,
    LINUX,
    WIN32,
)


# --- Fixtures ---

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset detector cache between tests."""
    SubstrateDetector.reset_cache()
    yield
    SubstrateDetector.reset_cache()


@pytest.fixture
def tmp_search_dir(tmp_path):
    """Create a temp directory with files for search tests."""
    (tmp_path / "hello.txt").write_text("hello world")
    (tmp_path / "data.csv").write_text("a,b,c")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content")
    return tmp_path


# --- SubstrateCapabilities ---

class TestSubstrateCapabilities:
    def test_frozen(self):
        caps = SubstrateDetector.detect()
        with pytest.raises(AttributeError):
            caps.platform = "fake"  # type: ignore[misc]

    def test_all_fields_populated(self):
        caps = SubstrateDetector.detect()
        assert caps.platform in (DARWIN, LINUX, WIN32, "cygwin", "msys")
        assert isinstance(caps.has_spotlight, bool)
        assert isinstance(caps.has_fsevents, bool)
        assert isinstance(caps.has_say, bool)
        assert isinstance(caps.open_command, str)
        assert isinstance(caps.home_dir, Path)
        assert isinstance(caps.config_dir, Path)
        assert isinstance(caps.temp_dir, Path)
        assert isinstance(caps.has_notify, bool)
        assert isinstance(caps.python_version, str)

    def test_home_dir_is_absolute(self):
        caps = SubstrateDetector.detect()
        assert caps.home_dir.is_absolute()

    def test_config_dir_is_absolute(self):
        caps = SubstrateDetector.detect()
        assert caps.config_dir.is_absolute()
        assert caps.config_dir.name == ".motherlabs"

    def test_temp_dir_is_absolute(self):
        caps = SubstrateDetector.detect()
        assert caps.temp_dir.is_absolute()

    def test_python_version_format(self):
        caps = SubstrateDetector.detect()
        parts = caps.python_version.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])


# --- SubstrateDetector.detect() ---

class TestDetect:
    def test_returns_capabilities(self):
        caps = SubstrateDetector.detect()
        assert isinstance(caps, SubstrateCapabilities)

    def test_caching(self):
        caps1 = SubstrateDetector.detect()
        caps2 = SubstrateDetector.detect()
        assert caps1 is caps2

    def test_reset_cache(self):
        caps1 = SubstrateDetector.detect()
        SubstrateDetector.reset_cache()
        caps2 = SubstrateDetector.detect()
        assert caps1 is not caps2
        assert caps1 == caps2  # same values

    def test_darwin_capabilities(self):
        """On macOS, certain capabilities should be true."""
        caps = SubstrateDetector.detect()
        if caps.platform == DARWIN:
            assert caps.open_command == "open"
            # mdfind/say may or may not exist in CI
        elif caps.platform == LINUX:
            assert caps.open_command == "xdg-open"
            assert caps.has_spotlight is False
            assert caps.has_fsevents is False
            assert caps.has_say is False

    @patch("mother.substrate.sys")
    @patch("mother.substrate._command_exists")
    def test_linux_detection(self, mock_cmd, mock_sys):
        mock_sys.platform = LINUX
        mock_cmd.return_value = False
        SubstrateDetector.reset_cache()

        caps = SubstrateDetector.detect()
        assert caps.platform == LINUX
        assert caps.has_spotlight is False
        assert caps.has_fsevents is False
        assert caps.open_command == "xdg-open"


# --- _command_exists ---

class TestCommandExists:
    def test_existing_command(self):
        # python3 should exist in our venv
        assert _command_exists("python3") is True

    def test_nonexistent_command(self):
        assert _command_exists("totally_fake_command_xyz") is False


# --- find_files ---

class TestFindFiles:
    def test_glob_fallback_finds_files(self, tmp_search_dir):
        """With spotlight disabled, glob fallback works."""
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps

            results = SubstrateDetector.find_files("hello", tmp_search_dir)
            assert len(results) >= 1
            assert any("hello.txt" in str(p) for p in results)

    def test_glob_finds_nested(self, tmp_search_dir):
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps

            results = SubstrateDetector.find_files("nested", tmp_search_dir)
            assert len(results) >= 1
            assert any("nested.txt" in str(p) for p in results)

    def test_glob_max_results(self, tmp_search_dir):
        # Create many files
        for i in range(30):
            (tmp_search_dir / f"file_{i}.txt").write_text(f"content {i}")

        results = _find_glob("file_", tmp_search_dir, max_results=5)
        assert len(results) <= 5

    def test_glob_nonexistent_query(self, tmp_search_dir):
        results = _find_glob("zzz_nonexistent_zzz", tmp_search_dir, max_results=10)
        assert results == []

    def test_glob_returns_paths(self, tmp_search_dir):
        results = _find_glob("hello", tmp_search_dir, max_results=10)
        assert all(isinstance(p, Path) for p in results)

    def test_find_files_with_no_directory(self, tmp_search_dir):
        """find_files without directory uses glob on home. Test with mock."""
        with patch("mother.substrate._find_glob", return_value=[]) as mock_glob:
            with patch.object(SubstrateDetector, "detect") as mock_detect:
                caps = SubstrateCapabilities(
                    platform=LINUX,
                    has_spotlight=False,
                    has_fsevents=False,
                    has_say=False,
                    open_command="xdg-open",
                    home_dir=Path.home(),
                    config_dir=Path.home() / ".motherlabs",
                    temp_dir=Path(tempfile.gettempdir()),
                    has_notify=False,
                    python_version=platform.python_version(),
                )
                mock_detect.return_value = caps

                results = SubstrateDetector.find_files(
                    "zzz_nonexistent_substrate_test_zzz",
                    max_results=1,
                )
                assert isinstance(results, list)
                mock_glob.assert_called_once()


# --- _find_spotlight ---

class TestFindSpotlight:
    @patch("mother.substrate.subprocess.run")
    def test_spotlight_success(self, mock_run, tmp_search_dir):
        test_file = tmp_search_dir / "hello.txt"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"{test_file}\n",
            stderr="",
        )
        results = _find_spotlight("hello", tmp_search_dir, 20, 10)
        assert results is not None
        assert len(results) == 1
        assert results[0] == test_file

    @patch("mother.substrate.subprocess.run")
    def test_spotlight_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        results = _find_spotlight("test", None, 20, 10)
        assert results is None

    @patch("mother.substrate.subprocess.run")
    def test_spotlight_timeout(self, mock_run):
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(cmd="mdfind", timeout=10)
        results = _find_spotlight("test", None, 20, 10)
        assert results is None

    @patch("mother.substrate.subprocess.run")
    def test_spotlight_filters_nonexistent(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/nonexistent/path/file.txt\n",
            stderr="",
        )
        results = _find_spotlight("test", None, 20, 10)
        assert results is not None
        assert len(results) == 0  # filtered out


# --- open_file ---

class TestOpenFile:
    def test_open_nonexistent_file(self, tmp_path):
        result = SubstrateDetector.open_file(tmp_path / "nope.txt")
        assert result is False

    @patch("mother.substrate.subprocess.Popen")
    def test_open_existing_file(self, mock_popen, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hi")

        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=DARWIN,
                has_spotlight=True,
                has_fsevents=True,
                has_say=True,
                open_command="open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=True,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            result = SubstrateDetector.open_file(f)
            assert result is True
            mock_popen.assert_called_once()

    def test_open_no_command(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hi")

        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform="unknown",
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            result = SubstrateDetector.open_file(f)
            assert result is False


# --- notify ---

class TestNotify:
    def test_notify_no_support(self):
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform="unknown",
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.notify("Test", "msg") is False

    @patch("mother.substrate.subprocess.run")
    def test_notify_darwin(self, mock_run):
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=DARWIN,
                has_spotlight=True,
                has_fsevents=True,
                has_say=True,
                open_command="open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=True,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.notify("Title", "Body") is True
            mock_run.assert_called_once()

    @patch("mother.substrate.subprocess.run")
    def test_notify_linux(self, mock_run):
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=True,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.notify("Title", "Body") is True
            mock_run.assert_called_once()


# --- say ---

class TestSay:
    def test_say_no_support(self):
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.say("hello") is False


# --- trash_file ---

class TestTrashFile:
    def test_trash_nonexistent(self, tmp_path):
        assert SubstrateDetector.trash_file(tmp_path / "nope.txt") is False

    def test_trash_file_fallback(self, tmp_path):
        f = tmp_path / "delete_me.txt"
        f.write_text("goodbye")

        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.trash_file(f) is True
            assert not f.exists()

    def test_trash_directory_fallback(self, tmp_path):
        d = tmp_path / "delete_dir"
        d.mkdir()
        (d / "file.txt").write_text("hi")

        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=LINUX,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="xdg-open",
                home_dir=Path.home(),
                config_dir=Path.home() / ".motherlabs",
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps
            assert SubstrateDetector.trash_file(d) is True
            assert not d.exists()


# --- ensure_config_dir ---

class TestEnsureConfigDir:
    def test_creates_config_dir(self, tmp_path):
        config_dir = tmp_path / ".motherlabs"
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=sys.platform,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="open",
                home_dir=tmp_path,
                config_dir=config_dir,
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps

            result = SubstrateDetector.ensure_config_dir()
            assert result == config_dir
            assert config_dir.is_dir()

    def test_idempotent(self, tmp_path):
        config_dir = tmp_path / ".motherlabs"
        config_dir.mkdir()
        with patch.object(SubstrateDetector, "detect") as mock_detect:
            caps = SubstrateCapabilities(
                platform=sys.platform,
                has_spotlight=False,
                has_fsevents=False,
                has_say=False,
                open_command="open",
                home_dir=tmp_path,
                config_dir=config_dir,
                temp_dir=Path(tempfile.gettempdir()),
                has_notify=False,
                python_version=platform.python_version(),
            )
            mock_detect.return_value = caps

            result = SubstrateDetector.ensure_config_dir()
            assert result == config_dir
