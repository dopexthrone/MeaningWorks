"""
Project launcher — subprocess lifecycle for compiled projects.

LEAF module. Stdlib only. No imports from core/ or mother/.

Manages a single child process: start, stop, output streaming.
"""

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of a project launch attempt."""

    success: bool
    pid: int
    port: Optional[int] = None
    entry_point: str = ""
    project_dir: str = ""
    error: Optional[str] = None


class ProjectLauncher:
    """Manages a single child process lifecycle.

    Starts a Python project as a subprocess, captures stdout/stderr
    into a thread-safe queue, and provides stop/drain/status methods.
    """

    def __init__(self, project_dir: str, entry_point: str = "main.py"):
        self._project_dir = project_dir
        self._entry_point = entry_point
        self._proc: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue = queue.Queue(maxsize=2000)
        self._reader_thread: Optional[threading.Thread] = None
        self._detected_port: Optional[int] = None

    def start(self, timeout: float = 15.0) -> LaunchResult:
        """Start the project subprocess.

        Waits up to `timeout` seconds for the process to either:
        - Print a line containing "READY" or a port number
        - Stay alive without crashing

        Stdout/stderr lines are pushed to _output_queue.
        """
        if self._proc is not None and self._proc.poll() is None:
            return LaunchResult(
                success=False,
                pid=self._proc.pid,
                entry_point=self._entry_point,
                project_dir=self._project_dir,
                error="Process already running",
            )

        entry_path = os.path.join(self._project_dir, self._entry_point)
        if not os.path.isfile(entry_path):
            return LaunchResult(
                success=False,
                pid=0,
                entry_point=self._entry_point,
                project_dir=self._project_dir,
                error=f"Entry point not found: {self._entry_point}",
            )

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                [sys.executable, "-u", self._entry_point],
                cwd=self._project_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as e:
            return LaunchResult(
                success=False,
                pid=0,
                entry_point=self._entry_point,
                project_dir=self._project_dir,
                error=str(e),
            )

        self._detected_port = None

        # Start reader thread
        self._reader_thread = threading.Thread(
            target=self._read_output,
            daemon=True,
        )
        self._reader_thread.start()

        # Wait for startup signal or early crash
        deadline = time.monotonic() + min(timeout, 2.0)
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # Process exited early
                rc = self._proc.returncode
                if rc != 0:
                    # Drain remaining output for error context
                    lines = self.drain_output()
                    error_ctx = "\n".join(lines[-5:]) if lines else f"Exit code {rc}"
                    return LaunchResult(
                        success=False,
                        pid=self._proc.pid,
                        entry_point=self._entry_point,
                        project_dir=self._project_dir,
                        error=error_ctx,
                    )
                break
            time.sleep(0.1)

        if self._proc.poll() is not None and self._proc.returncode != 0:
            lines = self.drain_output()
            error_ctx = "\n".join(lines[-5:]) if lines else f"Exit code {self._proc.returncode}"
            return LaunchResult(
                success=False,
                pid=self._proc.pid,
                entry_point=self._entry_point,
                project_dir=self._project_dir,
                error=error_ctx,
            )

        return LaunchResult(
            success=True,
            pid=self._proc.pid,
            port=self._detected_port,
            entry_point=self._entry_point,
            project_dir=self._project_dir,
        )

    def _read_output(self) -> None:
        """Background thread: read stdout lines and push to queue."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                # Detect port numbers in output
                if self._detected_port is None:
                    self._detected_port = _extract_port(line)
                try:
                    self._output_queue.put_nowait(line)
                except queue.Full:
                    # Drop oldest to make room
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._output_queue.put_nowait(line)
                    except queue.Full:
                        pass
        except (ValueError, OSError):
            pass  # Pipe closed

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the child process. SIGTERM first, SIGKILL after timeout."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return
        try:
            self._proc.send_signal(signal.SIGTERM)
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        except OSError:
            pass
        finally:
            self._proc = None

    def is_running(self) -> bool:
        """True if the child process is alive."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def drain_output(self) -> List[str]:
        """Non-blocking: return all queued output lines."""
        lines = []
        while True:
            try:
                lines.append(self._output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    @property
    def pid(self) -> Optional[int]:
        """PID of the child process, or None if not running."""
        if self._proc is not None:
            return self._proc.pid
        return None

    @property
    def port(self) -> Optional[int]:
        """Detected port from process output."""
        return self._detected_port


def _extract_port(line: str) -> Optional[int]:
    """Extract a port number from a log line like 'Listening on port 8080'."""
    lower = line.lower()
    for marker in ("port ", ":"):
        idx = lower.rfind(marker)
        if idx >= 0:
            after = line[idx + len(marker):].strip()
            # Extract leading digits
            digits = ""
            for ch in after:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                port = int(digits)
                if 1024 <= port <= 65535:
                    return port
    return None
