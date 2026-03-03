"""
Mother Claude Code CLI integration — invoke Claude Code for self-modification.

LEAF module. Stdlib only. No imports from core/ or mother/.

Provides subprocess wrappers for the Claude Code CLI, git snapshotting,
test validation, and rollback. All functions unset CLAUDECODE from the
subprocess environment to prevent nested session blocking.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("mother.claude_code")


@dataclass(frozen=True)
class ClaudeCodeResult:
    """Result from a Claude Code CLI invocation."""

    success: bool = False
    result_text: str = ""
    session_id: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    num_turns: int = 0
    error: str = ""
    is_error: bool = False


def _clean_env(strip_api_key: bool = True) -> dict:
    """Return a copy of os.environ with CLAUDECODE unset.

    Claude Code sets CLAUDECODE=1 in its subprocess environment.
    Nested invocations detect this and refuse to start. Unsetting
    it allows Mother to invoke Claude Code as a child process.

    If strip_api_key=True (default), also removes ANTHROPIC_API_KEY
    so Claude Code CLI uses the user's subscription (Pro/Max) instead
    of consuming API credits. The CLI falls back to subscription auth
    when no API key is present in the environment.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    if strip_api_key:
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def invoke_claude_code(
    prompt: str,
    cwd: str,
    allowed_tools: str = "Read,Edit,Glob,Grep,Bash",
    max_turns: int = 15,
    max_budget_usd: float = 3.0,
    append_system_prompt: Optional[str] = None,
    timeout: int = 600,
    claude_path: str = "",
) -> ClaudeCodeResult:
    """Invoke the Claude Code CLI in non-interactive mode.

    Runs: claude -p <prompt> --output-format json --max-turns N
          --max-budget-usd N --allowedTools "..."

    Returns ClaudeCodeResult with parsed JSON output or plain text fallback.
    """
    if not claude_path:
        claude_path = str(Path.home() / ".local" / "bin" / "claude")

    if not Path(claude_path).exists():
        return ClaudeCodeResult(
            success=False,
            error=f"Claude Code CLI not found at {claude_path}",
            is_error=True,
        )

    cmd = [
        claude_path,
        "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(max_turns),
        "--max-budget-usd", str(max_budget_usd),
        "--allowedTools", allowed_tools,
    ]

    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_clean_env(),
        )
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            # Try to extract clean error from JSON stdout
            error_text = ""
            try:
                data = json.loads(result.stdout.strip())
                error_text = data.get("result", "")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if not error_text:
                error_text = result.stderr.strip() or result.stdout.strip() or "Non-zero exit"
            return ClaudeCodeResult(
                success=False,
                result_text=result.stdout,
                error=error_text,
                is_error=True,
                duration_seconds=elapsed,
            )

        # Try to parse JSON output
        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout)
            # CLI can return exit 0 with is_error=true (e.g. credit exhaustion)
            if data.get("is_error"):
                return ClaudeCodeResult(
                    success=False,
                    result_text=data.get("result", ""),
                    session_id=data.get("session_id", ""),
                    cost_usd=float(data.get("cost_usd", 0.0)),
                    duration_seconds=elapsed,
                    num_turns=int(data.get("num_turns", 0)),
                    error=data.get("result", "Unknown CLI error"),
                    is_error=True,
                )
            return ClaudeCodeResult(
                success=True,
                result_text=data.get("result", stdout),
                session_id=data.get("session_id", ""),
                cost_usd=float(data.get("cost_usd", 0.0)),
                duration_seconds=elapsed,
                num_turns=int(data.get("num_turns", 0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            # Plain text fallback
            return ClaudeCodeResult(
                success=True,
                result_text=stdout,
                duration_seconds=elapsed,
            )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return ClaudeCodeResult(
            success=False,
            error=f"Timed out after {timeout}s",
            is_error=True,
            duration_seconds=elapsed,
        )
    except FileNotFoundError:
        # Could be the binary or the cwd that's missing
        if not Path(claude_path).exists():
            msg = f"Claude Code CLI not found at {claude_path}"
        elif not Path(cwd).is_dir():
            msg = f"Working directory not found: {cwd}"
        else:
            msg = f"File not found (binary: {claude_path}, cwd: {cwd})"
        return ClaudeCodeResult(
            success=False,
            error=msg,
            is_error=True,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return ClaudeCodeResult(
            success=False,
            error=str(e),
            is_error=True,
            duration_seconds=elapsed,
        )


def _kill_proc(proc: subprocess.Popen) -> None:
    """Kill a subprocess and wait for cleanup. Safe to call on already-dead process."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def invoke_claude_code_streaming(
    prompt: str,
    cwd: str,
    on_event: Callable[[Dict], None],
    allowed_tools: str = "Read,Edit,Glob,Grep,Bash",
    max_turns: int = 15,
    max_budget_usd: float = 3.0,
    append_system_prompt: Optional[str] = None,
    timeout: int = 600,
    claude_path: str = "",
) -> ClaudeCodeResult:
    """Invoke Claude Code CLI with real-time streaming output.

    Uses --output-format stream-json and subprocess.Popen to read
    newline-delimited JSON events as they arrive. Each parsed event
    is passed to on_event(). Returns the same ClaudeCodeResult as
    invoke_claude_code().

    on_event runs in the calling thread (the subprocess read loop).
    It must be thread-safe if called from a worker thread, and should
    never raise — exceptions are caught and logged at WARNING.
    """
    if not claude_path:
        claude_path = str(Path.home() / ".local" / "bin" / "claude")

    if not Path(claude_path).exists():
        return ClaudeCodeResult(
            success=False,
            error=f"Claude Code CLI not found at {claude_path}",
            is_error=True,
        )

    cmd = [
        claude_path,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        "--max-budget-usd", str(max_budget_usd),
        "--allowedTools", allowed_tools,
    ]

    if append_system_prompt:
        cmd.extend(["--append-system-prompt", append_system_prompt])

    result_data: Optional[Dict] = None
    start = time.monotonic()
    proc: Optional[subprocess.Popen] = None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=_clean_env(),
        )

        deadline = start + timeout

        # Use a thread to read lines so we can enforce a per-read timeout.
        # proc.stdout iteration blocks on the underlying read(); select()
        # doesn't work reliably on pipes across platforms. Instead, we read
        # in the main loop with a deadline check per line.
        import threading
        import queue as _queue

        line_queue: _queue.Queue = _queue.Queue()
        _SENTINEL = object()

        def _reader():
            """Read stdout lines in a daemon thread, push to queue."""
            try:
                for raw_line in proc.stdout:
                    line_queue.put(raw_line)
            except Exception:
                pass
            finally:
                line_queue.put(_SENTINEL)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_proc(proc)
                elapsed = time.monotonic() - start
                return ClaudeCodeResult(
                    success=False,
                    error=f"Timed out after {timeout}s",
                    is_error=True,
                    duration_seconds=elapsed,
                )

            try:
                # Block up to remaining seconds for next line
                item = line_queue.get(timeout=min(remaining, 60.0))
            except _queue.Empty:
                # No output for 60s but still within deadline — keep waiting
                continue

            if item is _SENTINEL:
                break  # Stream ended (stdout closed)

            line = item.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            # Fire callback — runs in calling thread, must be thread-safe
            try:
                on_event(event)
            except Exception as e:
                logger.warning(f"on_event callback error: {e}")

            # Capture result event for final return
            if event.get("type") == "result":
                result_data = event

        # Wait for process to exit
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Claude Code process did not exit within 30s, killing")
            _kill_proc(proc)

        elapsed = time.monotonic() - start

        if result_data:
            is_err = result_data.get("is_error", False)
            return ClaudeCodeResult(
                success=not is_err and proc.returncode == 0,
                result_text=result_data.get("result", ""),
                session_id=result_data.get("session_id", ""),
                cost_usd=float(result_data.get("cost_usd", 0.0)),
                duration_seconds=elapsed,
                num_turns=int(result_data.get("num_turns", 0)),
                error=result_data.get("result", "") if is_err else "",
                is_error=is_err,
            )

        # No result event — fall back to exit code
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            return ClaudeCodeResult(
                success=False,
                error=stderr.strip() or "Non-zero exit with no result event",
                is_error=True,
                duration_seconds=elapsed,
            )

        return ClaudeCodeResult(
            success=True,
            result_text="",
            duration_seconds=elapsed,
        )

    except FileNotFoundError:
        if not Path(claude_path).exists():
            msg = f"Claude Code CLI not found at {claude_path}"
        elif not Path(cwd).is_dir():
            msg = f"Working directory not found: {cwd}"
        else:
            msg = f"File not found (binary: {claude_path}, cwd: {cwd})"
        return ClaudeCodeResult(
            success=False,
            error=msg,
            is_error=True,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        if proc:
            _kill_proc(proc)
        return ClaudeCodeResult(
            success=False,
            error=str(e),
            is_error=True,
            duration_seconds=elapsed,
        )


def save_build_log(
    events: List[Dict],
    prompt: str,
    repo_dir: str,
    result: ClaudeCodeResult,
    description_slug: str = "",
) -> Optional[Path]:
    """Write a JSONL build log to ~/.motherlabs/build_logs/.

    Returns the log file path, or None on failure.
    """
    log_dir = Path.home() / ".motherlabs" / "build_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = description_slug or "build"
    # Sanitize slug
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug[:40]).strip("_")
    filename = f"{ts}_{slug}.jsonl"
    log_path = log_dir / filename

    try:
        with open(log_path, "w") as f:
            # First line: metadata
            meta = {
                "_type": "metadata",
                "prompt": prompt[:500],
                "repo_dir": repo_dir,
                "timestamp": ts,
            }
            f.write(json.dumps(meta) + "\n")

            # Raw events
            for event in events:
                f.write(json.dumps(event) + "\n")

            # Last line: outcome
            outcome = {
                "_type": "outcome",
                "success": result.success,
                "cost_usd": result.cost_usd,
                "duration_seconds": result.duration_seconds,
                "error": result.error,
                "num_turns": result.num_turns,
            }
            f.write(json.dumps(outcome) + "\n")

        return log_path
    except OSError:
        return None


def git_snapshot(repo_dir: str) -> str:
    """Stage all changes and create a snapshot commit.

    Returns the commit hash, or "" on failure.
    """
    env = _clean_env()
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_dir, capture_output=True, timeout=30, env=env,
        )
        subprocess.run(
            ["git", "commit", "-m", "pre-self-build snapshot", "--allow-empty"],
            cwd=repo_dir, capture_output=True, timeout=30, env=env,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=10, env=env,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def git_rollback(repo_dir: str, commit_hash: str) -> bool:
    """Hard reset to a previous commit. Returns True on success."""
    if not commit_hash:
        return False
    env = _clean_env()
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=repo_dir, capture_output=True, timeout=30, env=env,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_tests(repo_dir: str, timeout: int = 120, sandbox_profile=None) -> bool:
    """Run the project test suite. Returns True if all tests pass.

    If sandbox_profile is provided, wraps the pytest command with sandbox-exec
    to prevent network access and restrict filesystem writes.
    """
    venv_pytest = str(Path(repo_dir) / ".venv" / "bin" / "pytest")
    if not Path(venv_pytest).exists():
        venv_pytest = "pytest"

    env = _clean_env()

    if sandbox_profile is not None:
        try:
            from mother.sandbox import sandbox_command
            command = f"{venv_pytest} tests/ -x -q"
            cmd_list = sandbox_command(command, sandbox_profile, cwd=repo_dir)
            result = subprocess.run(
                cmd_list,
                cwd=repo_dir,
                capture_output=True,
                timeout=timeout,
                env=env,
            )
            return result.returncode == 0
        except ImportError:
            pass  # Fall through to unsandboxed execution

    try:
        result = subprocess.run(
            [venv_pytest, "tests/", "-x", "-q"],
            cwd=repo_dir,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode == 0
    except Exception:
        return False
