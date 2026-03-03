"""
Tests for mother/launcher.py — project subprocess lifecycle.
"""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from mother.launcher import LaunchResult, ProjectLauncher, _extract_port


# --- LaunchResult ---


class TestLaunchResult:
    """Test LaunchResult frozen dataclass."""

    def test_construction(self):
        r = LaunchResult(success=True, pid=1234, entry_point="main.py", project_dir="/tmp/test")
        assert r.success is True
        assert r.pid == 1234
        assert r.port is None
        assert r.error is None

    def test_with_port(self):
        r = LaunchResult(success=True, pid=1234, port=8080, entry_point="main.py", project_dir="/tmp")
        assert r.port == 8080

    def test_with_error(self):
        r = LaunchResult(success=False, pid=0, error="not found", entry_point="main.py", project_dir="/tmp")
        assert r.success is False
        assert r.error == "not found"

    def test_frozen(self):
        r = LaunchResult(success=True, pid=1, entry_point="main.py", project_dir="/tmp")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = LaunchResult(success=True, pid=1)
        assert r.entry_point == ""
        assert r.project_dir == ""
        assert r.port is None
        assert r.error is None


# --- _extract_port ---


class TestExtractPort:
    """Test port extraction from log lines."""

    def test_port_keyword(self):
        assert _extract_port("Listening on port 8080") == 8080

    def test_port_with_colon(self):
        assert _extract_port("Server running at http://0.0.0.0:3000") == 3000

    def test_no_port(self):
        assert _extract_port("Hello world") is None

    def test_port_below_range(self):
        assert _extract_port("port 80") is None  # Below 1024

    def test_port_above_range(self):
        assert _extract_port("port 99999") is None  # Above 65535

    def test_port_boundary_low(self):
        assert _extract_port("port 1024") == 1024

    def test_port_boundary_high(self):
        assert _extract_port("port 65535") == 65535

    def test_port_in_url(self):
        assert _extract_port("http://localhost:5000/api") == 5000


# --- ProjectLauncher ---


class TestProjectLauncherInit:
    """Test ProjectLauncher initialization."""

    def test_defaults(self):
        launcher = ProjectLauncher("/tmp/project")
        assert launcher._project_dir == "/tmp/project"
        assert launcher._entry_point == "main.py"
        assert launcher._proc is None
        assert launcher.pid is None
        assert launcher.is_running() is False

    def test_custom_entry_point(self):
        launcher = ProjectLauncher("/tmp/project", entry_point="app.py")
        assert launcher._entry_point == "app.py"


class TestProjectLauncherStart:
    """Test ProjectLauncher.start() with real subprocesses."""

    def test_start_simple_script(self):
        """Start a simple Python script that prints and exits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write('print("Hello from project")\n')

            launcher = ProjectLauncher(tmpdir)
            result = launcher.start(timeout=5.0)

            assert result.success is True
            assert result.pid > 0
            assert result.entry_point == "main.py"
            assert result.project_dir == tmpdir
            assert result.error is None

            # Wait for script to finish
            time.sleep(0.3)
            lines = launcher.drain_output()
            assert any("Hello from project" in l for l in lines)

    def test_start_entry_not_found(self):
        """Entry point doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            launcher = ProjectLauncher(tmpdir, entry_point="missing.py")
            result = launcher.start()

            assert result.success is False
            assert "not found" in result.error

    def test_start_crash(self):
        """Script that immediately crashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write('raise RuntimeError("boom")\n')

            launcher = ProjectLauncher(tmpdir)
            result = launcher.start(timeout=5.0)

            assert result.success is False
            assert result.error is not None

    def test_start_already_running(self):
        """Cannot start twice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    import time
                    print("running")
                    time.sleep(30)
                """))

            launcher = ProjectLauncher(tmpdir)
            result1 = launcher.start(timeout=2.0)
            assert result1.success is True

            result2 = launcher.start(timeout=2.0)
            assert result2.success is False
            assert "already running" in result2.error.lower()

            launcher.stop()

    def test_start_detects_port(self):
        """Detect port from subprocess output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    import time
                    print("Server running on port 8080")
                    time.sleep(30)
                """))

            launcher = ProjectLauncher(tmpdir)
            result = launcher.start(timeout=2.0)
            assert result.success is True

            # Give reader thread time to process
            deadline = time.monotonic() + 3.0
            while launcher.port is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert launcher.port == 8080

            launcher.stop()


class TestProjectLauncherStop:
    """Test ProjectLauncher.stop()."""

    def test_stop_running_process(self):
        """Stop a running process with SIGTERM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    import time
                    print("started")
                    time.sleep(60)
                """))

            launcher = ProjectLauncher(tmpdir)
            result = launcher.start(timeout=2.0)
            assert result.success is True
            assert launcher.is_running() is True

            launcher.stop()
            assert launcher.is_running() is False
            assert launcher.pid is None

    def test_stop_not_running(self):
        """Stop when no process is running — no error."""
        launcher = ProjectLauncher("/tmp/nonexistent")
        launcher.stop()  # Should not raise

    def test_stop_already_exited(self):
        """Stop when process already exited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write('print("done")\n')

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            time.sleep(0.5)  # Let it exit

            launcher.stop()  # Should not raise
            assert launcher.is_running() is False


class TestProjectLauncherOutput:
    """Test output draining."""

    def test_drain_output(self):
        """Drain output lines from subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    for i in range(5):
                        print(f"line {i}")
                """))

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            time.sleep(0.5)

            lines = launcher.drain_output()
            assert len(lines) >= 5
            assert "line 0" in lines[0]
            assert "line 4" in lines[4]

    def test_drain_empty(self):
        """Drain when no output."""
        launcher = ProjectLauncher("/tmp/nonexistent")
        lines = launcher.drain_output()
        assert lines == []

    def test_drain_clears_queue(self):
        """Second drain returns empty after first drain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write('print("hello")\n')

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            time.sleep(0.3)

            lines1 = launcher.drain_output()
            assert len(lines1) > 0

            lines2 = launcher.drain_output()
            assert len(lines2) == 0


class TestProjectLauncherIsRunning:
    """Test is_running() states."""

    def test_not_started(self):
        launcher = ProjectLauncher("/tmp/test")
        assert launcher.is_running() is False

    def test_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write("import time; time.sleep(30)\n")

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            assert launcher.is_running() is True
            launcher.stop()

    def test_after_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write('print("done")\n')

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            time.sleep(0.5)
            assert launcher.is_running() is False

    def test_after_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write("import time; time.sleep(30)\n")

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            launcher.stop()
            assert launcher.is_running() is False


class TestProjectLauncherPid:
    """Test pid property."""

    def test_pid_none_before_start(self):
        launcher = ProjectLauncher("/tmp/test")
        assert launcher.pid is None

    def test_pid_set_while_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write("import time; time.sleep(30)\n")

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            assert launcher.pid is not None
            assert launcher.pid > 0
            launcher.stop()

    def test_pid_none_after_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write("import time; time.sleep(30)\n")

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            launcher.stop()
            assert launcher.pid is None


class TestProjectLauncherLifecycle:
    """Test multiple start/stop cycles."""

    def test_restart(self):
        """Can start, stop, and start again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write("import time; time.sleep(30)\n")

            launcher = ProjectLauncher(tmpdir)

            r1 = launcher.start(timeout=2.0)
            assert r1.success is True
            pid1 = launcher.pid
            launcher.stop()

            r2 = launcher.start(timeout=2.0)
            assert r2.success is True
            pid2 = launcher.pid
            assert pid2 != pid1
            launcher.stop()

    def test_stderr_captured(self):
        """stderr is mixed into stdout output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script = os.path.join(tmpdir, "main.py")
            with open(script, "w") as f:
                f.write(textwrap.dedent("""\
                    import sys
                    print("stdout line")
                    print("stderr line", file=sys.stderr)
                """))

            launcher = ProjectLauncher(tmpdir)
            launcher.start(timeout=2.0)
            time.sleep(0.5)

            lines = launcher.drain_output()
            text = "\n".join(lines)
            assert "stdout line" in text
            assert "stderr line" in text
