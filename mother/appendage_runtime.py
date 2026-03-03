"""
Appendage runtime — subprocess lifecycle and JSON Lines IPC.

LEAF module. Stdlib only. No imports from core/ or mother/.

Manages a single appendage child process with bidirectional communication:
  Mother → Agent (stdin):   {"id": "req_001", "method": "invoke", "params": {...}}
  Agent → Mother (stdout):  {"id": "req_001", "result": {...}, "error": null}
  Agent startup signal:     {"id": "ready", "result": {"capabilities": [...]}}
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SpawnResult:
    """Outcome of spawning an appendage process."""

    success: bool
    pid: int = 0
    capabilities: List[str] = ()
    error: str = ""


@dataclass(frozen=True)
class InvokeResult:
    """Outcome of invoking an appendage."""

    success: bool
    output: Optional[Dict] = None
    error: str = ""
    duration_seconds: float = 0.0


class AppendageProcess:
    """Manages a single appendage subprocess with JSON Lines IPC.

    Unlike ProjectLauncher (fire-and-forget output), this supports
    bidirectional request/response communication over stdin/stdout.
    """

    def __init__(self, project_dir: str, entry_point: str = "main.py"):
        self._project_dir = project_dir
        self._entry_point = entry_point
        self._proc: Optional[subprocess.Popen] = None
        self._response_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_lines: List[str] = []
        self._request_counter = 0
        self._lock = threading.Lock()

    def spawn(self, timeout: float = 15.0) -> SpawnResult:
        """Start the appendage process and wait for the 'ready' signal.

        Returns SpawnResult with capabilities from the ready message.
        """
        if self._proc is not None and self._proc.poll() is None:
            return SpawnResult(
                success=False,
                pid=self._proc.pid,
                error="Process already running",
            )

        entry_path = os.path.join(self._project_dir, self._entry_point)
        if not os.path.isfile(entry_path):
            return SpawnResult(
                success=False,
                error=f"Entry point not found: {entry_path}",
            )

        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                [sys.executable, "-u", self._entry_point],
                cwd=self._project_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as e:
            return SpawnResult(success=False, error=str(e))

        # Start reader threads
        self._stderr_lines = []
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
        )
        self._reader_thread.start()

        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        # Wait for 'ready' signal
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Check if process died
            if self._proc.poll() is not None:
                stderr_text = "\n".join(self._stderr_lines[-5:]) if self._stderr_lines else ""
                return SpawnResult(
                    success=False,
                    pid=self._proc.pid,
                    error=f"Process exited during startup. {stderr_text}".strip(),
                )

            try:
                msg = self._response_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if msg.get("id") == "ready":
                result = msg.get("result", {})
                caps = result.get("capabilities", [])
                return SpawnResult(
                    success=True,
                    pid=self._proc.pid,
                    capabilities=tuple(caps),
                )

        # Timeout — kill the process
        self.stop(timeout=2.0)
        return SpawnResult(
            success=False,
            error=f"Timeout waiting for ready signal ({timeout}s)",
        )

    def invoke(self, params: dict, timeout: float = 30.0) -> InvokeResult:
        """Send a request and wait for the matching response.

        Returns InvokeResult with the agent's output.
        """
        if self._proc is None or self._proc.poll() is not None:
            return InvokeResult(
                success=False,
                error="Process not running",
            )

        with self._lock:
            self._request_counter += 1
            req_id = f"req_{self._request_counter:04d}"

        request = {"id": req_id, "method": "invoke", "params": params}
        start = time.monotonic()

        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            return InvokeResult(
                success=False,
                error=f"Write failed: {e}",
                duration_seconds=time.monotonic() - start,
            )

        # Wait for matching response
        deadline = time.monotonic() + timeout
        unmatched = []
        try:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self._response_queue.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    if self._proc.poll() is not None:
                        return InvokeResult(
                            success=False,
                            error="Process died during invocation",
                            duration_seconds=time.monotonic() - start,
                        )
                    continue

                if msg.get("id") == req_id:
                    elapsed = time.monotonic() - start
                    err = msg.get("error")
                    if err:
                        return InvokeResult(
                            success=False,
                            output=msg.get("result"),
                            error=str(err),
                            duration_seconds=elapsed,
                        )
                    return InvokeResult(
                        success=True,
                        output=msg.get("result"),
                        duration_seconds=elapsed,
                    )
                else:
                    # Not our response — put back for others
                    unmatched.append(msg)
        finally:
            # Re-queue unmatched messages
            for m in unmatched:
                try:
                    self._response_queue.put_nowait(m)
                except queue.Full:
                    pass

        return InvokeResult(
            success=False,
            error=f"Timeout waiting for response ({timeout}s)",
            duration_seconds=time.monotonic() - start,
        )

    def is_alive(self) -> bool:
        """True if the subprocess is still running."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the subprocess. SIGTERM first, SIGKILL after timeout."""
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

    @property
    def pid(self) -> Optional[int]:
        if self._proc is not None:
            return self._proc.pid
        return None

    def _read_stdout(self) -> None:
        """Background thread: parse JSON lines from stdout into response queue."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if isinstance(msg, dict):
                        self._response_queue.put(msg)
                except json.JSONDecodeError:
                    pass  # Non-JSON output ignored
        except (ValueError, OSError):
            pass  # Pipe closed

    def _read_stderr(self) -> None:
        """Background thread: capture stderr for error context."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr_lines.append(line.rstrip("\n"))
                # Cap at 100 lines
                if len(self._stderr_lines) > 100:
                    self._stderr_lines = self._stderr_lines[-50:]
        except (ValueError, OSError):
            pass
