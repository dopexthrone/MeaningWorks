"""
Coding agent abstraction — provider-agnostic self-build with failover.

LEAF module. Stdlib only. No imports from core/ or mother/.

Wraps multiple coding CLI agents (Claude Code, Codex, Gemini CLI, Kimi)
behind a uniform interface. Automatic failover: if the preferred provider
fails (credit exhaustion, binary missing, timeout), tries the next.

Each provider is a frozen config + an invoke function that returns
CodingAgentResult. The failover loop tries providers in priority order
until one succeeds or all fail.
"""

import json
import logging
import os
import subprocess
import time
import threading
import queue as _queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("mother.coding_agent")


# ---------------------------------------------------------------------------
# Result type (uniform across all providers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodingAgentResult:
    """Result from any coding agent CLI invocation."""

    success: bool = False
    result_text: str = ""
    session_id: str = ""
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    num_turns: int = 0
    error: str = ""
    is_error: bool = False
    provider: str = ""  # which provider actually ran


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodingAgentProvider:
    """Configuration for a single coding agent CLI."""

    name: str                         # e.g. "claude", "codex", "gemini", "kimi"
    binary: str                       # path or command name
    env_var: str                      # API key env var (checked for availability)
    priority: int = 0                 # lower = tried first
    max_turns: int = 15
    max_budget_usd: float = 3.0
    timeout: int = 600

    def is_available(self) -> bool:
        """Check if the provider can plausibly run (binary exists, key set)."""
        # Check API key
        if self.env_var and not os.environ.get(self.env_var):
            return False
        # Check binary
        binary_path = Path(self.binary)
        if binary_path.is_absolute():
            return binary_path.exists()
        # Relative/bare name — check PATH via shutil.which
        import shutil
        return shutil.which(self.binary) is not None


# ---------------------------------------------------------------------------
# Provider defaults
# ---------------------------------------------------------------------------

def _default_claude_binary() -> str:
    return str(Path.home() / ".local" / "bin" / "claude")


def default_providers() -> List[CodingAgentProvider]:
    """Return all known providers in default priority order."""
    return [
        CodingAgentProvider(
            name="claude",
            binary=_default_claude_binary(),
            env_var="ANTHROPIC_API_KEY",
            priority=0,
            max_turns=15,
            max_budget_usd=3.0,
            timeout=600,
        ),
        CodingAgentProvider(
            name="codex",
            binary="codex",
            env_var="OPENAI_API_KEY",
            priority=1,
            max_turns=15,
            max_budget_usd=3.0,
            timeout=600,
        ),
        CodingAgentProvider(
            name="gemini",
            binary="gemini",
            env_var="GEMINI_API_KEY",
            priority=2,
            max_turns=15,
            max_budget_usd=3.0,
            timeout=600,
        ),
        CodingAgentProvider(
            name="kimi",
            binary="kimi",
            env_var="KIMI_API_KEY",
            priority=3,
            max_turns=15,
            max_budget_usd=3.0,
            timeout=600,
        ),
    ]


def available_providers(
    providers: Optional[List[CodingAgentProvider]] = None,
) -> List[CodingAgentProvider]:
    """Return providers that are currently available, sorted by priority."""
    all_p = providers or default_providers()
    avail = [p for p in all_p if p.is_available()]
    avail.sort(key=lambda p: p.priority)
    return avail


# ---------------------------------------------------------------------------
# Clean env helper (shared)
# ---------------------------------------------------------------------------

def _clean_env(strip_api_key: bool = True) -> dict:
    """Return env copy with CLAUDECODE unset (prevents nested session block).

    If strip_api_key=True (default), also removes ANTHROPIC_API_KEY
    so Claude Code CLI uses the user's subscription instead of API credits.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    if strip_api_key:
        env.pop("ANTHROPIC_API_KEY", None)
    return env


def _kill_proc(proc: subprocess.Popen) -> None:
    """Kill a subprocess and wait for cleanup."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-provider invoke functions
# ---------------------------------------------------------------------------

def _invoke_claude(
    provider: CodingAgentProvider,
    prompt: str,
    cwd: str,
    on_event: Optional[Callable[[Dict], None]] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Invoke Claude Code CLI with streaming."""
    turns = max_turns or provider.max_turns
    budget = max_budget_usd or provider.max_budget_usd
    t_out = timeout or provider.timeout
    binary = provider.binary
    tools = allowed_tools or "Read,Edit,Glob,Grep,Bash"

    cmd = [
        binary,
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(turns),
        "--max-budget-usd", str(budget),
        "--allowedTools", tools,
    ]

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    return _run_streaming_process(
        cmd=cmd, cwd=cwd, timeout=t_out,
        on_event=on_event, provider_name="claude",
        parse_result=_parse_claude_result,
    )


def _invoke_codex(
    provider: CodingAgentProvider,
    prompt: str,
    cwd: str,
    on_event: Optional[Callable[[Dict], None]] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Invoke OpenAI Codex CLI."""
    t_out = timeout or provider.timeout

    cmd = [
        provider.binary,
        "exec",
        "--json",
        "--full-auto",
        prompt,
    ]

    return _run_streaming_process(
        cmd=cmd, cwd=cwd, timeout=t_out,
        on_event=on_event, provider_name="codex",
        parse_result=_parse_codex_result,
    )


def _invoke_gemini(
    provider: CodingAgentProvider,
    prompt: str,
    cwd: str,
    on_event: Optional[Callable[[Dict], None]] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Invoke Google Gemini CLI."""
    t_out = timeout or provider.timeout

    cmd = [
        provider.binary,
        "-p", prompt,
        "--yolo",
    ]

    return _run_streaming_process(
        cmd=cmd, cwd=cwd, timeout=t_out,
        on_event=on_event, provider_name="gemini",
        parse_result=_parse_generic_result,
    )


def _invoke_kimi(
    provider: CodingAgentProvider,
    prompt: str,
    cwd: str,
    on_event: Optional[Callable[[Dict], None]] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Invoke Kimi CLI."""
    turns = max_turns or provider.max_turns
    t_out = timeout or provider.timeout

    cmd = [
        provider.binary,
        "--print",
        "--output-format", "stream-json",
        "--max-turns", str(turns),
        "-p", prompt,
    ]

    return _run_streaming_process(
        cmd=cmd, cwd=cwd, timeout=t_out,
        on_event=on_event, provider_name="kimi",
        parse_result=_parse_kimi_result,
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_INVOKERS: Dict[str, Callable] = {
    "claude": _invoke_claude,
    "codex": _invoke_codex,
    "gemini": _invoke_gemini,
    "kimi": _invoke_kimi,
}


# ---------------------------------------------------------------------------
# Generic streaming subprocess runner
# ---------------------------------------------------------------------------

def _run_streaming_process(
    cmd: List[str],
    cwd: str,
    timeout: int,
    on_event: Optional[Callable[[Dict], None]],
    provider_name: str,
    parse_result: Callable,
) -> CodingAgentResult:
    """Run a subprocess with streaming line-by-line output.

    Reads stdout line by line, tries to parse each as JSON and fires on_event.
    Uses a reader thread + queue pattern (same as claude_code.py) for
    reliable timeout enforcement.
    """
    start = time.monotonic()
    proc: Optional[subprocess.Popen] = None
    collected_lines: List[str] = []
    result_data: Optional[Dict] = None

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_clean_env(),
        )

        deadline = start + timeout
        line_queue: _queue.Queue = _queue.Queue()
        _SENTINEL = object()

        def _reader():
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
                return CodingAgentResult(
                    success=False,
                    error=f"Timed out after {timeout}s",
                    is_error=True,
                    duration_seconds=elapsed,
                    provider=provider_name,
                )

            try:
                item = line_queue.get(timeout=min(remaining, 60.0))
            except _queue.Empty:
                continue

            if item is _SENTINEL:
                break

            line = item.strip()
            if not line:
                continue

            collected_lines.append(line)

            # Try to parse as JSON event
            event = None
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                pass

            if event and on_event:
                try:
                    on_event(event)
                except Exception as e:
                    logger.warning(f"on_event callback error ({provider_name}): {e}")

            # Capture result-type events
            if event and event.get("type") == "result":
                result_data = event

        # Wait for process exit
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning(f"{provider_name} process did not exit within 30s, killing")
            _kill_proc(proc)

        elapsed = time.monotonic() - start

        return parse_result(
            proc=proc,
            elapsed=elapsed,
            result_data=result_data,
            collected_lines=collected_lines,
            provider_name=provider_name,
        )

    except FileNotFoundError:
        return CodingAgentResult(
            success=False,
            error=f"{provider_name} binary not found: {cmd[0]}",
            is_error=True,
            provider=provider_name,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        if proc:
            _kill_proc(proc)
        return CodingAgentResult(
            success=False,
            error=str(e),
            is_error=True,
            duration_seconds=elapsed,
            provider=provider_name,
        )


# ---------------------------------------------------------------------------
# Result parsers (per-provider output format)
# ---------------------------------------------------------------------------

def _parse_claude_result(
    proc: subprocess.Popen,
    elapsed: float,
    result_data: Optional[Dict],
    collected_lines: List[str],
    provider_name: str,
) -> CodingAgentResult:
    """Parse Claude Code stream-json result event."""
    if result_data:
        is_err = result_data.get("is_error", False)
        return CodingAgentResult(
            success=not is_err and proc.returncode == 0,
            result_text=result_data.get("result", ""),
            session_id=result_data.get("session_id", ""),
            cost_usd=float(result_data.get("cost_usd", 0.0)),
            duration_seconds=elapsed,
            num_turns=int(result_data.get("num_turns", 0)),
            error=result_data.get("result", "") if is_err else "",
            is_error=is_err,
            provider=provider_name,
        )

    # No result event — fall back to exit code
    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        return CodingAgentResult(
            success=False,
            error=stderr.strip() or "Non-zero exit with no result event",
            is_error=True,
            duration_seconds=elapsed,
            provider=provider_name,
        )

    return CodingAgentResult(
        success=True,
        result_text="",
        duration_seconds=elapsed,
        provider=provider_name,
    )


def _parse_codex_result(
    proc: subprocess.Popen,
    elapsed: float,
    result_data: Optional[Dict],
    collected_lines: List[str],
    provider_name: str,
) -> CodingAgentResult:
    """Parse Codex JSONL output.

    Codex exec --json emits JSONL events. The last line with
    type=turn.completed or the final stdout text is the result.
    """
    # Find last completed turn or final text
    last_text = ""
    for line in reversed(collected_lines):
        try:
            ev = json.loads(line)
            if ev.get("type") == "turn.completed":
                # Extract agent's final message
                items = ev.get("items", [])
                for item in reversed(items):
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "text":
                                last_text = c.get("text", "")
                                break
                    if last_text:
                        break
            if last_text:
                break
        except (json.JSONDecodeError, ValueError, TypeError):
            # Plain text line — use as fallback
            if not last_text and line.strip():
                last_text = line.strip()

    return CodingAgentResult(
        success=proc.returncode == 0,
        result_text=last_text,
        duration_seconds=elapsed,
        error="" if proc.returncode == 0 else (last_text or "Non-zero exit"),
        is_error=proc.returncode != 0,
        provider=provider_name,
    )


def _parse_kimi_result(
    proc: subprocess.Popen,
    elapsed: float,
    result_data: Optional[Dict],
    collected_lines: List[str],
    provider_name: str,
) -> CodingAgentResult:
    """Parse Kimi CLI stream-json output.

    Similar wire protocol to Claude — look for result event,
    fall back to last text content.
    """
    # Kimi uses similar event structure
    if result_data:
        is_err = result_data.get("is_error", False)
        return CodingAgentResult(
            success=not is_err and proc.returncode == 0,
            result_text=result_data.get("result", ""),
            duration_seconds=elapsed,
            error=result_data.get("result", "") if is_err else "",
            is_error=is_err,
            provider=provider_name,
        )

    # Fall back to last text content from events
    last_text = ""
    for line in reversed(collected_lines):
        try:
            ev = json.loads(line)
            if ev.get("type") == "assistant":
                content = ev.get("message", {}).get("content", [])
                for c in content:
                    if c.get("type") == "text" and c.get("text", "").strip():
                        last_text = c["text"].strip()
                        break
            if last_text:
                break
        except (json.JSONDecodeError, ValueError):
            if not last_text and line.strip():
                last_text = line.strip()

    return CodingAgentResult(
        success=proc.returncode == 0,
        result_text=last_text,
        duration_seconds=elapsed,
        error="" if proc.returncode == 0 else (last_text or "Non-zero exit"),
        is_error=proc.returncode != 0,
        provider=provider_name,
    )


def _parse_generic_result(
    proc: subprocess.Popen,
    elapsed: float,
    result_data: Optional[Dict],
    collected_lines: List[str],
    provider_name: str,
) -> CodingAgentResult:
    """Generic parser for CLIs with plain text output (Gemini, etc.)."""
    # Collect all text output
    text = "\n".join(collected_lines)

    return CodingAgentResult(
        success=proc.returncode == 0,
        result_text=text[-2000:] if len(text) > 2000 else text,
        duration_seconds=elapsed,
        error="" if proc.returncode == 0 else (text[-500:] or "Non-zero exit"),
        is_error=proc.returncode != 0,
        provider=provider_name,
    )


# ---------------------------------------------------------------------------
# Credit/quota error detection
# ---------------------------------------------------------------------------

_CREDIT_ERROR_PATTERNS = (
    "credit balance is too low",
    "credit_balance_too_low",
    "insufficient_quota",
    "rate_limit",
    "billing",
    "quota exceeded",
    "exceeded your current quota",
    "insufficient credits",
    "payment required",
    "402",
)


def _is_credit_error(result: CodingAgentResult) -> bool:
    """Detect if a failure is due to credit/quota exhaustion."""
    check = (result.error + " " + result.result_text).lower()
    return any(pat in check for pat in _CREDIT_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Public API: invoke with failover
# ---------------------------------------------------------------------------

def invoke_coding_agent(
    prompt: str,
    cwd: str,
    on_event: Optional[Callable[[Dict], None]] = None,
    providers: Optional[List[CodingAgentProvider]] = None,
    preferred: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Invoke the best available coding agent with automatic failover.

    Tries providers in priority order. If preferred is set, that provider
    is tried first regardless of priority. On credit/quota errors or
    binary-not-found, falls through to the next provider.

    Returns CodingAgentResult with .provider indicating which one ran.
    """
    all_providers = providers or default_providers()
    avail = [p for p in all_providers if p.is_available()]
    avail.sort(key=lambda p: p.priority)

    # Move preferred to front
    if preferred:
        pref = [p for p in avail if p.name == preferred]
        rest = [p for p in avail if p.name != preferred]
        avail = pref + rest

    if not avail:
        return CodingAgentResult(
            success=False,
            error="No coding agents available. Install one of: claude, codex, gemini, kimi. Set the corresponding API key.",
            is_error=True,
        )

    errors: List[str] = []

    for provider in avail:
        invoker = _INVOKERS.get(provider.name)
        if not invoker:
            logger.debug(f"No invoker for provider {provider.name}, skipping")
            continue

        logger.info(f"Trying coding agent: {provider.name}")

        if on_event:
            try:
                on_event({
                    "_type": "provider_attempt",
                    "provider": provider.name,
                    "binary": provider.binary,
                })
            except Exception:
                pass

        result = invoker(
            provider=provider,
            prompt=prompt,
            cwd=cwd,
            on_event=on_event,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            timeout=timeout,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
        )

        if result.success:
            logger.info(f"Coding agent {provider.name} succeeded (${result.cost_usd:.3f}, {result.duration_seconds:.1f}s)")
            return result

        # Check if this is a credit/quota error — failover
        if _is_credit_error(result):
            msg = f"{provider.name}: credit/quota exhausted — {result.error[:100]}"
            logger.warning(msg)
            errors.append(msg)

            if on_event:
                try:
                    on_event({
                        "_type": "provider_failover",
                        "failed_provider": provider.name,
                        "reason": "credit_exhausted",
                        "error": result.error[:200],
                    })
                except Exception:
                    pass
            continue

        # Binary not found — try next
        if "not found" in result.error.lower():
            msg = f"{provider.name}: binary not found — {result.error[:100]}"
            logger.warning(msg)
            errors.append(msg)
            continue

        # Other error (actual code failure, timeout, etc.) — don't failover,
        # this is a real failure from a working provider
        logger.warning(f"Coding agent {provider.name} failed: {result.error[:200]}")
        return result

    # All providers exhausted
    combined = "; ".join(errors) if errors else "All providers failed"
    return CodingAgentResult(
        success=False,
        error=f"All coding agents exhausted. {combined}",
        is_error=True,
    )


def invoke_coding_agent_streaming(
    prompt: str,
    cwd: str,
    on_event: Callable[[Dict], None],
    providers: Optional[List[CodingAgentProvider]] = None,
    preferred: Optional[str] = None,
    max_turns: Optional[int] = None,
    max_budget_usd: Optional[float] = None,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[str] = None,
) -> CodingAgentResult:
    """Convenience alias — same as invoke_coding_agent with on_event required."""
    return invoke_coding_agent(
        prompt=prompt,
        cwd=cwd,
        on_event=on_event,
        providers=providers,
        preferred=preferred,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        timeout=timeout,
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
    )
