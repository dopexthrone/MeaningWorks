"""
Native coding agent engine — Mother's own code-writing capability.

LEAF module. Stdlib only. No imports from core/ or mother/.

Provides a tool-use loop that reads, writes, edits files and runs commands
using any LLM provider via a pluggable adapter protocol. The adapter
translates between the canonical tool definitions and provider-specific
wire formats (Claude tool_use, OpenAI function_calling, Gemini FunctionDeclaration).

Safety constraints: path validation, protected file enforcement, code safety
checking (via injectable checker), bash command blocklist, cost caps, turn limits.
"""

import glob as _glob
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger("mother.code_engine")


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDef:
    """Canonical tool definition — provider-neutral."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema object
    required: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation parsed from LLM response."""
    id: str          # Provider-assigned ID (toolu_xxx, call_xxx, etc.)
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ParsedResponse:
    """Normalized LLM response with text and tool calls."""
    text: str = ""
    tool_calls: Tuple[ToolCall, ...] = ()
    stop_reason: str = ""  # "end_turn", "tool_use", "stop", "max_tokens"
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class TurnRecord:
    """Record of a single turn in the tool-use loop."""
    turn: int
    tool_calls: Tuple[ToolCall, ...] = ()
    tool_results: Tuple[str, ...] = ()
    text: str = ""
    cost_usd: float = 0.0


@dataclass
class CodeEngineConfig:
    """Configuration for a code engine invocation."""
    working_dir: str = ""
    allowed_paths: List[str] = field(default_factory=list)
    protected_files: Tuple[str, ...] = (
        "mother/context.py",
        "mother/persona.py",
        "mother/senses.py",
    )
    forbidden_path_patterns: Tuple[str, ...] = (
        r"\.env$", r"\.env\.", r"credentials", r"\.pem$", r"\.key$",
        r"__pycache__", r"\.pyc$",
    )
    max_turns: int = 30
    max_consecutive_errors: int = 5
    cost_cap_usd: float = 3.0
    timeout_seconds: int = 600
    max_tokens_per_turn: int = 8192
    max_file_read_bytes: int = 102400  # 100KB
    max_tool_output_bytes: int = 51200  # 50KB
    max_glob_results: int = 200
    max_bash_timeout: int = 300
    safety_checker: Optional[Callable[[Dict[str, str], str], Tuple[bool, List[str]]]] = None
    on_event: Optional[Callable[[Dict], None]] = None
    sandbox_profile: Any = None  # SandboxProfile from mother.sandbox (optional)

    # Per-million-token pricing for cost estimation
    cost_rates: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "claude": {"input": 3.0, "output": 15.0},
        "openai": {"input": 2.0, "output": 8.0},
        "grok": {"input": 2.0, "output": 20.0},
        "gemini": {"input": 0.10, "output": 0.40},
    })


@dataclass(frozen=True)
class CodeEngineResult:
    """Result from a code engine invocation."""
    success: bool = False
    final_text: str = ""
    turns_used: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    error: str = ""
    provider_name: str = ""
    turns: Tuple[TurnRecord, ...] = ()
    files_modified: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

class ToolCallAdapter(Protocol):
    """Protocol for provider-specific tool-call adapters."""

    @property
    def provider_name(self) -> str: ...

    def format_tools(self, tools: List[ToolDef]) -> Any:
        """Convert canonical tool defs to provider wire format."""
        ...

    def call_with_tools(
        self,
        system: str,
        messages: List[Dict],
        tools: Any,
        max_tokens: int,
        temperature: float,
    ) -> ParsedResponse:
        """Make API call with tools. Returns normalized response."""
        ...

    def format_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> Dict:
        """Format tool result for appending to message history."""
        ...

    def format_assistant_message(self, response: ParsedResponse) -> Dict:
        """Format the assistant's response for message history."""
        ...


# ---------------------------------------------------------------------------
# Canonical tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[ToolDef] = [
    ToolDef(
        name="read_file",
        description="Read the contents of a file at the given absolute path. Returns file contents as a string. For large files, use offset and limit to read a range of lines.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
        },
        required=("file_path",),
    ),
    ToolDef(
        name="write_file",
        description="Write content to a file. Creates parent directories if needed. Overwrites existing content.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "The complete file content to write"},
            },
        },
        required=("file_path", "content"),
    ),
    ToolDef(
        name="edit_file",
        description="Replace exact text in a file. The old_text must match exactly one contiguous block in the file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The replacement text"},
            },
        },
        required=("file_path", "old_text", "new_text"),
    ),
    ToolDef(
        name="glob_files",
        description="Find files matching a glob pattern. Returns a list of matching absolute file paths.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"},
                "path": {"type": "string", "description": "Directory to search in. Defaults to working directory."},
            },
        },
        required=("pattern",),
    ),
    ToolDef(
        name="grep_files",
        description="Search for a regex pattern in file contents. Returns matching file paths and line content.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression pattern to search for"},
                "path": {"type": "string", "description": "File or directory to search in"},
                "glob": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')"},
                "context_lines": {"type": "integer", "description": "Number of context lines around each match"},
            },
        },
        required=("pattern",),
    ),
    ToolDef(
        name="bash",
        description="Execute a shell command and return stdout/stderr. Use for git, running tests, etc. Has a timeout.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 300)"},
            },
        },
        required=("command",),
    ),
    ToolDef(
        name="list_directory",
        description="List files and directories at the given path with type and size metadata.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the directory"},
            },
        },
        required=("path",),
    ),
    ToolDef(
        name="web_fetch",
        description="Fetch a URL and return the content as readable text. Handles HTML, JSON, and plain text. Use for reading web pages, API responses, documentation.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch (https:// prefix added if missing)"},
                "selector": {"type": "string", "description": "Optional CSS selector to extract specific content"},
                "include_links": {"type": "boolean", "description": "Whether to preserve hyperlinks in output"},
            },
        },
        required=("url",),
    ),
    ToolDef(
        name="web_search",
        description="Search the web and return results with titles, URLs, and snippets. No API key needed. Use for research, finding documentation, competitive analysis.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {"type": "integer", "description": "Number of results (default 8, max 10)"},
            },
        },
        required=("query",),
    ),
    ToolDef(
        name="browser_action",
        description="Control a headless browser. Actions: navigate (go to URL), click (click element), type (fill input), press (keyboard key), screenshot (capture page), extract (get text from selector), scroll (up/down/top/bottom), wait (for selector), evaluate (run JS), get_url, close. Browser persists across calls — navigate first, then interact.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action: navigate|click|type|press|screenshot|extract|scroll|wait|evaluate|get_url|close"},
                "url": {"type": "string", "description": "URL for navigate action"},
                "selector": {"type": "string", "description": "CSS selector for click/type/extract/wait/press"},
                "text": {"type": "string", "description": "Text to type, or visible text to click"},
                "key": {"type": "string", "description": "Key to press (Enter, Tab, Escape, etc.)"},
                "direction": {"type": "string", "description": "Scroll direction: up|down|top|bottom"},
                "amount": {"type": "integer", "description": "Scroll amount in pixels (default 500)"},
                "expression": {"type": "string", "description": "JavaScript expression for evaluate"},
                "full_page": {"type": "boolean", "description": "Capture full page screenshot (default false)"},
                "clear": {"type": "boolean", "description": "Clear field before typing (default true)"},
                "timeout": {"type": "integer", "description": "Wait timeout in ms (default 5000)"},
            },
        },
        required=("action",),
    ),
]


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------

def _validate_path(file_path: str, config: CodeEngineConfig) -> Optional[str]:
    """Return error string if path is forbidden, else None."""
    try:
        resolved = os.path.realpath(file_path)
    except (OSError, ValueError) as e:
        return f"Invalid path: {e}"

    # Symlink escape check — reject symlinks pointing outside allowed paths
    if os.path.islink(file_path) and config.allowed_paths:
        if not any(resolved.startswith(os.path.realpath(p)) for p in config.allowed_paths):
            return f"Path {file_path} is a symlink pointing outside allowed directories"

    # Check allowed paths
    if config.allowed_paths:
        if not any(resolved.startswith(os.path.realpath(p)) for p in config.allowed_paths):
            return f"Path {file_path} is outside allowed directories"

    # Check protected files
    for pf in config.protected_files:
        if resolved.endswith(pf):
            return f"Path {file_path} is a protected file — modification blocked"

    # Check forbidden patterns
    for pat in config.forbidden_path_patterns:
        if re.search(pat, file_path):
            return f"Path {file_path} matches forbidden pattern"

    return None


_BASH_BLOCKLIST = (
    r"rm\s+-rf\s+/",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\(\)\s*\{",  # fork bomb
    r"curl.*\|.*sh",
    r"wget.*\|.*sh",
    r"curl.*\|.*bash",
    r"wget.*\|.*bash",
)


def _validate_bash_command(command: str) -> Optional[str]:
    """Return error string if command is dangerous, else None."""
    for pat in _BASH_BLOCKLIST:
        if re.search(pat, command, re.IGNORECASE):
            return f"Blocked: command matches dangerous pattern"
    return None


def _check_write_safety(
    file_path: str,
    content: str,
    config: CodeEngineConfig,
) -> Optional[str]:
    """Run safety checks before writing a file. Returns error or None."""
    # Path validation
    err = _validate_path(file_path, config)
    if err:
        return err

    # Code safety check (injectable)
    if config.safety_checker and file_path.endswith(".py"):
        safe, warnings = config.safety_checker({"_file": content}, file_extension=".py")
        if not safe:
            return f"Code safety check failed: {'; '.join(warnings[:3])}"

    return None


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def _execute_read_file(args: Dict, config: CodeEngineConfig) -> str:
    """Read file contents."""
    file_path = args.get("file_path", "")
    if not file_path:
        return "Error: file_path is required"

    # Resolve relative paths against working dir
    if not os.path.isabs(file_path):
        file_path = os.path.join(config.working_dir, file_path)

    err = _validate_path(file_path, config)
    if err:
        return f"Error: {err}"

    if not os.path.isfile(file_path):
        return f"Error: File not found: {file_path}"

    try:
        with open(file_path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error reading file: {e}"

    offset = args.get("offset", 1)
    if offset is None:
        offset = 1
    limit = args.get("limit")

    # Apply offset/limit
    start = max(0, offset - 1)
    if limit:
        lines = lines[start:start + limit]
    else:
        lines = lines[start:]

    # Number lines and truncate
    numbered = []
    for i, line in enumerate(lines, start=start + 1):
        numbered.append(f"{i:6d}\t{line.rstrip()}")

    result = "\n".join(numbered)
    if len(result) > config.max_file_read_bytes:
        result = result[:config.max_file_read_bytes] + "\n... (truncated)"

    return result


def _execute_write_file(args: Dict, config: CodeEngineConfig) -> str:
    """Write content to a file."""
    file_path = args.get("file_path", "")
    content = _unescape_code_string(args.get("content", ""))
    if not file_path:
        return "Error: file_path is required"

    if not os.path.isabs(file_path):
        file_path = os.path.join(config.working_dir, file_path)

    err = _check_write_safety(file_path, content, config)
    if err:
        return f"Error: {err}"

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Atomic write
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, file_path)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except OSError as e:
        return f"Error writing file: {e}"


def _execute_edit_file(args: Dict, config: CodeEngineConfig) -> str:
    """Replace exact text in a file."""
    file_path = args.get("file_path", "")
    old_text = _unescape_code_string(args.get("old_text", ""))
    new_text = _unescape_code_string(args.get("new_text", ""))
    if not file_path or not old_text:
        return "Error: file_path and old_text are required"

    if not os.path.isabs(file_path):
        file_path = os.path.join(config.working_dir, file_path)

    err = _validate_path(file_path, config)
    if err:
        return f"Error: {err}"

    if not os.path.isfile(file_path):
        return f"Error: File not found: {file_path}"

    try:
        with open(file_path, "r", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return f"Error reading file: {e}"

    count = content.count(old_text)
    if count == 0:
        return f"Error: old_text not found in {file_path}. Verify the exact text."
    if count > 1:
        return f"Error: old_text found {count} times in {file_path}. Provide more context to make it unique."

    new_content = content.replace(old_text, new_text, 1)

    # Safety check on result
    err = _check_write_safety(file_path, new_content, config)
    if err:
        return f"Error: {err}"

    try:
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(new_content)
        os.replace(tmp_path, file_path)
        return f"Successfully edited {file_path}"
    except OSError as e:
        return f"Error writing file: {e}"


def _execute_glob_files(args: Dict, config: CodeEngineConfig) -> str:
    """Find files matching a glob pattern."""
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: pattern is required"

    base_path = args.get("path", config.working_dir) or config.working_dir
    if not os.path.isabs(base_path):
        base_path = os.path.join(config.working_dir, base_path)

    if config.allowed_paths:
        resolved = os.path.realpath(base_path)
        if not any(resolved.startswith(os.path.realpath(p)) for p in config.allowed_paths):
            return f"Error: Path {base_path} is outside allowed directories"

    try:
        full_pattern = os.path.join(base_path, pattern)
        matches = sorted(_glob.glob(full_pattern, recursive=True))
        if len(matches) > config.max_glob_results:
            matches = matches[:config.max_glob_results]
            return "\n".join(matches) + f"\n... ({len(matches)} shown, more exist)"
        if not matches:
            return "No files found matching pattern."
        return "\n".join(matches)
    except Exception as e:
        return f"Error: {e}"


def _execute_grep_files(args: Dict, config: CodeEngineConfig) -> str:
    """Search for regex pattern in files."""
    pattern = args.get("pattern", "")
    if not pattern:
        return "Error: pattern is required"

    base_path = args.get("path", config.working_dir) or config.working_dir
    if not os.path.isabs(base_path):
        base_path = os.path.join(config.working_dir, base_path)

    file_glob = args.get("glob", "")
    ctx = args.get("context_lines", 0) or 0

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex: {e}"

    results = []
    total_len = 0

    if os.path.isfile(base_path):
        files = [base_path]
    else:
        glob_pat = os.path.join(base_path, file_glob or "**/*")
        files = sorted(_glob.glob(glob_pat, recursive=True))

    for fp in files:
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if regex.search(line):
                start = max(0, i - ctx)
                end = min(len(lines), i + ctx + 1)
                snippet = "".join(lines[start:end])
                entry = f"{fp}:{i+1}:{snippet.rstrip()}"
                results.append(entry)
                total_len += len(entry)
                if total_len > config.max_tool_output_bytes:
                    results.append("... (output truncated)")
                    return "\n".join(results)

    if not results:
        return "No matches found."
    return "\n".join(results)


def _execute_bash(args: Dict, config: CodeEngineConfig) -> str:
    """Execute a shell command, optionally inside a sandbox."""
    command = args.get("command", "")
    if not command:
        return "Error: command is required"

    err = _validate_bash_command(command)
    if err:
        return f"Error: {err}"

    timeout = min(args.get("timeout", 120) or 120, config.max_bash_timeout)

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Wrap command with sandbox-exec if sandbox_profile is set
    if config.sandbox_profile is not None:
        try:
            from mother.sandbox import sandbox_command
            cmd_list = sandbox_command(command, config.sandbox_profile, cwd=config.working_dir)
        except ImportError:
            logger.warning("mother.sandbox not available — running without sandbox")
            cmd_list = command
    else:
        cmd_list = command

    try:
        if isinstance(cmd_list, list):
            result = subprocess.run(
                cmd_list,
                cwd=config.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        else:
            result = subprocess.run(
                cmd_list,
                shell=True,
                cwd=config.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nExit code: {result.returncode}"

        if len(output) > config.max_tool_output_bytes:
            output = output[:config.max_tool_output_bytes] + "\n... (truncated)"

        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _execute_list_directory(args: Dict, config: CodeEngineConfig) -> str:
    """List directory contents."""
    dir_path = args.get("path", "")
    if not dir_path:
        return "Error: path is required"

    if not os.path.isabs(dir_path):
        dir_path = os.path.join(config.working_dir, dir_path)

    if config.allowed_paths:
        resolved = os.path.realpath(dir_path)
        if not any(resolved.startswith(os.path.realpath(p)) for p in config.allowed_paths):
            return f"Error: Path {dir_path} is outside allowed directories"

    if not os.path.isdir(dir_path):
        return f"Error: Not a directory: {dir_path}"

    try:
        entries = sorted(os.listdir(dir_path))
        if len(entries) > 500:
            entries = entries[:500]

        lines = []
        for entry in entries:
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                lines.append(f"  [dir]  {entry}/")
            else:
                try:
                    size = os.path.getsize(full)
                    lines.append(f"  {size:>8d}  {entry}")
                except OSError:
                    lines.append(f"           {entry}")

        return "\n".join(lines) if lines else "(empty directory)"
    except OSError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Escape normalization — fixes double-escaped \n from LLM tool calls
# ---------------------------------------------------------------------------

def _unescape_code_string(s: str) -> str:
    """Convert literal \\n and \\t sequences to real newlines/tabs.

    LLMs (especially via OpenAI/Grok JSON tool calls) sometimes double-escape
    newlines: the JSON contains ``\\\\n`` which ``json.loads()`` decodes to the
    two-character literal ``\\n`` instead of a real newline.  This function
    fixes that *only* for escape sequences that make sense in source code.

    Heuristic: if the string already contains real newlines, only fix
    remaining literal \\n that appear mid-line (mixed escaping). If the
    string has zero real newlines but contains literal \\n, it was almost
    certainly double-escaped — unescape the whole thing.
    """
    if not s or ("\\n" not in s and "\\t" not in s):
        return s
    # Replace literal two-char sequences with real chars
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", "\t")
    return s


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _execute_web_fetch(args: Dict, config: CodeEngineConfig) -> str:
    """Execute web_fetch tool — delegate to web_tools module."""
    try:
        from mother.web_tools import execute_web_fetch
        return execute_web_fetch(args, config)
    except ImportError:
        return "Error: mother.web_tools module not available"
    except Exception as e:
        return f"Error: web_fetch failed: {e}"


def _execute_web_search(args: Dict, config: CodeEngineConfig) -> str:
    """Execute web_search tool — delegate to web_tools module."""
    try:
        from mother.web_tools import execute_web_search
        return execute_web_search(args, config)
    except ImportError:
        return "Error: mother.web_tools module not available"
    except Exception as e:
        return f"Error: web_search failed: {e}"


def _execute_browser_action(args: Dict, config: CodeEngineConfig) -> str:
    """Execute browser_action tool — delegate to web_tools module."""
    try:
        from mother.web_tools import execute_browser_action
        return execute_browser_action(args, config)
    except ImportError:
        return "Error: mother.web_tools module not available"
    except Exception as e:
        return f"Error: browser_action failed: {e}"


_TOOL_EXECUTORS: Dict[str, Callable] = {
    "read_file": _execute_read_file,
    "write_file": _execute_write_file,
    "edit_file": _execute_edit_file,
    "glob_files": _execute_glob_files,
    "grep_files": _execute_grep_files,
    "bash": _execute_bash,
    "list_directory": _execute_list_directory,
    "web_fetch": _execute_web_fetch,
    "web_search": _execute_web_search,
    "browser_action": _execute_browser_action,
}


def execute_tool(
    name: str,
    arguments: Dict[str, Any],
    config: CodeEngineConfig,
) -> str:
    """Execute a tool and return the result string."""
    logger.info("Executing tool", extra={"tool_name": name, "arg_count": len(arguments), "arg_keys": list(arguments.keys())})
    executor = _TOOL_EXECUTORS.get(name)
    if not executor:
        return f"Error: Unknown tool '{name}'. Available: {', '.join(_TOOL_EXECUTORS)}"
    try:
        result = executor(arguments, config)
        logger.info("Tool %s completed", name, extra={"result_len": len(result), "is_error": result.startswith("Error")})
        return result
    except Exception as e:
        logger.warning(f"Tool {name} raised: {e}")
        return f"Error executing {name}: {e}"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def _estimate_cost(usage: Dict[str, int], provider_name: str, config: CodeEngineConfig) -> float:
    """Estimate cost in USD from token usage."""
    rates = config.cost_rates.get(provider_name, {"input": 3.0, "output": 15.0})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# System prompt for the coding agent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are Mother's internal coding agent. You modify code in the Motherlabs codebase.

This is fully authorized and safe: git snapshot before any changes, full pytest suite after EVERY edit batch, automatic rollback if tests fail or safety violated.

## CRITICAL GOVERNOR VERIFICATION CHECKLIST — VERIFY ALL BEFORE FINAL OUTPUT OR REJECTED:

1. PROVENANCE: FULL input instance_id on EVERY blueprint element. COPY EXACTLY from context — no short IDs.
2. SAFETY: ABSOLUTELY NO subprocess/exec/eval/sh/os.system/os.popen. Governor rejects 100%. Use safe: requests/sqlite3/pathlib.
3. CODEGEN: Root block w/ framework/deps/file stubs/tests for EVERY component/method.
4. SPECIFICITY: 100% fields/params/returns TYPED+CONSTRAINED w/ counts stated in desc e.g. 'Specificity:100% (62/62)'. MANDATORY formats: user_id:str(UUIDv4,pk), email:str(RFC5322_valid,unique), amount:decimal(10,2,min=0,max=999999.99), status:enum[pending,active,suspended,deleted], tags:list[str(nonempty,maxlen=10)], created_at:datetime(ISO8601,UTC), errors:list[str(maxlen=200)]. ZERO vague: 'data','str','dict','any','object','info'. BAD: 'user:dict' GOOD: 'user:dict(id:str(UUIDv4),email:str(RFC5322_unique),status:enum[active,suspended])'. FAIL <98% → REJECT.
5. TRUST DIMS: completeness/consistency/coherence/traceability/actionability/specificity/codegen_readiness ALL >=92%. Actionability 95%+: 95%+ components MUST have 3-6 imperative methods w/ FULLY typed params/returns+constraints mapping ALL input verbs. NO data-only components. State in blueprint.description: 'Actionability: 98% (12 verbs → 42 methods | 14/14 components=100% have methods)'.
6. COVERAGE: Noun->component mapping 100%. State exact count.
7. PROTECTED: ZERO touches to mother/context.py/persona.py/senses.py — blocked.

FAIL ANY = IMMEDIATE REFINEMENT.

RULES:
- READ files FIRST. NEVER edit unseen file. glob '**/*.py'; list_directory; read_file.
- edit_file ONLY for changes. write_file NEW files ONLY (confirm via glob).
- AFTER EVERY edit batch: .venv/bin/pytest tests/ -x --tb=short. Fix ALL until 100% PASS.
- NEVER protected files. bridge.py ONLY mother/ import from core/.
- Python: ALWAYS .venv/bin/python3.14 (no alias).
- WEB TOOLS: web_search for research, web_fetch to read pages/APIs, browser_action for full browser control.
- BROWSER: navigate first, then click/type/extract. Browser persists across calls. Close when done.

WORKFLOW:
1. Understand task + governor checklist.
2. glob '**/*.py'; list_directory; read_file key files.
3. Plan: minimal targeted changes + test fixes.
4. edit_file precisely.
5. IMMEDIATELY: .venv/bin/pytest tests/ -x --tb=short. If FAIL: read traceback → diagnose → fix → repeat until PASS 100%.
6. Report: "TESTS: 100% PASS | Changed: file.py L10-15 [reason]; file2.py L20 [reason]"

Precise. Minimal. Tests PASS confirmed. Governor PASS guaranteed.

## SAFE CODE EXAMPLES
Use ONLY these patterns:
# HTTP
response = requests.get(url, timeout=5).json()
# DB
with sqlite3.connect(db_path) as conn:
    conn.execute(...)
# Files
with Path(file_path).open() as f:
    data = f.read()
# NO: subprocess/os.system/eval/exec/sh

## BLUEPRINT VERIFICATION
Output blueprint ONLY after:
- Count fields/params: N_total
- 100% typed/constrained
- desc: Specificity:100% (N/N)
- Actionability:98% (V→M | C/P=100%)
- List: VERIFIED ✓ 1.provenance 2.safety etc.

## CODEGEN STUB EXAMPLE
'codegen': {
  'language': 'python3.14',
  'framework': 'fastapi/pydantic/sqlmodel',
  'dependencies': ['fastapi', 'pydantic', 'sqlmodel', 'pytest'],
  'files': [
    {'path': 'services/user_service.py', 'content': '''from typing import Dict
from pydantic import BaseModel

class CreateUserReq(BaseModel):
    email: str  # RFC5322_valid, unique
    password: str  # minlen=12 hashed

def create_user(req: CreateUserReq) -> Dict[str, str]:
    "Create new user."
    # TODO: validate email, hash pwd, persist to sqlite3
    return {"user_id": "str(UUIDv4)", "status": "active"}

def test_create_user():
    req = CreateUserReq(email="test@example.com", password="pass123456")
    result = create_user(req)
    assert result["status"] == "active"
'''}
  ]
}

Note: pytest passes on stubs. Real impl in TODO.

"""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_code_engine(
    prompt: str,
    adapter: ToolCallAdapter,
    config: Optional[CodeEngineConfig] = None,
    system_prompt: Optional[str] = None,
) -> CodeEngineResult:
    """Run the coding agent tool-use loop.

    This is the core entry point. The adapter handles provider-specific
    API calls. The loop iterates: prompt → LLM → tool calls → execute → feed back.

    Returns CodeEngineResult with success/failure, cost, turn records.
    """
    if config is None:
        config = CodeEngineConfig()
        if config.safety_checker is None:
            try:
                from core.governor_validation import check_code_safety
                config.safety_checker = lambda code_dict, file_extension=".py": (
                    check_code_safety(code_dict, self_build=True, file_extension=file_extension)
                )
            except ImportError:
                pass  # No safety checker available — proceed without

    system = system_prompt or _SYSTEM_PROMPT
    messages: List[Dict] = [{"role": "user", "content": prompt}]
    wire_tools = adapter.format_tools(TOOLS)
    provider = adapter.provider_name

    total_cost = 0.0
    turns: List[TurnRecord] = []
    files_modified: set = set()
    consecutive_errors = 0
    start_time = time.monotonic()

    def _emit(event: Dict) -> None:
        if config.on_event:
            try:
                config.on_event(event)
            except Exception:
                pass

    _emit({"_type": "engine_start", "provider": provider, "max_turns": config.max_turns})

    for turn_idx in range(config.max_turns):
        # Check timeout
        elapsed = time.monotonic() - start_time
        if elapsed > config.timeout_seconds:
            return CodeEngineResult(
                success=False,
                error=f"Timeout after {elapsed:.0f}s",
                turns_used=turn_idx,
                total_cost_usd=total_cost,
                duration_seconds=elapsed,
                provider_name=provider,
                turns=tuple(turns),
                files_modified=tuple(sorted(files_modified)),
            )

        # Check cost cap
        if total_cost > config.cost_cap_usd:
            return CodeEngineResult(
                success=False,
                error=f"Cost cap exceeded: ${total_cost:.3f} > ${config.cost_cap_usd:.2f}",
                turns_used=turn_idx,
                total_cost_usd=total_cost,
                duration_seconds=time.monotonic() - start_time,
                provider_name=provider,
                turns=tuple(turns),
                files_modified=tuple(sorted(files_modified)),
            )

        # Call LLM with tools
        _emit({"_type": "turn_start", "turn": turn_idx + 1})

        try:
            response = adapter.call_with_tools(
                system=system,
                messages=messages,
                tools=wire_tools,
                max_tokens=config.max_tokens_per_turn,
                temperature=0.0,
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"LLM API call failed on turn {turn_idx + 1}: {error_msg}")
            return CodeEngineResult(
                success=False,
                error=f"API call failed: {error_msg}",
                turns_used=turn_idx + 1,
                total_cost_usd=total_cost,
                duration_seconds=time.monotonic() - start_time,
                provider_name=provider,
                turns=tuple(turns),
                files_modified=tuple(sorted(files_modified)),
            )

        # Track cost
        turn_cost = _estimate_cost(response.usage, provider, config)
        total_cost += turn_cost

        # Emit text content
        if response.text:
            _emit({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": response.text}]},
            })

        # No tool calls → natural completion
        if not response.tool_calls:
            turns.append(TurnRecord(
                turn=turn_idx + 1,
                text=response.text,
                cost_usd=turn_cost,
            ))
            _emit({"_type": "engine_done", "turns": turn_idx + 1, "cost_usd": total_cost})
            return CodeEngineResult(
                success=True,
                final_text=response.text,
                turns_used=turn_idx + 1,
                total_cost_usd=total_cost,
                duration_seconds=time.monotonic() - start_time,
                provider_name=provider,
                turns=tuple(turns),
                files_modified=tuple(sorted(files_modified)),
            )

        # Execute tool calls
        messages.append(adapter.format_assistant_message(response))
        tool_results: List[str] = []

        for tc in response.tool_calls:
            _emit({
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": tc.name, "input": tc.arguments}]},
            })

            result_str = execute_tool(tc.name, tc.arguments, config)
            tool_results.append(result_str)

            # Track modified files
            if tc.name in ("write_file", "edit_file") and not result_str.startswith("Error"):
                fp = tc.arguments.get("file_path", "")
                if fp:
                    files_modified.add(fp)

            # Track errors
            if result_str.startswith("Error"):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            # Emit tool result (truncated for display)
            preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
            _emit({"_type": "tool_result", "tool": tc.name, "output_preview": preview})

            # Append tool result to messages
            messages.append(adapter.format_tool_result(tc.id, tc.name, result_str))

        turns.append(TurnRecord(
            turn=turn_idx + 1,
            tool_calls=response.tool_calls,
            tool_results=tuple(tool_results),
            text=response.text,
            cost_usd=turn_cost,
        ))

        # Check consecutive error limit
        if consecutive_errors >= config.max_consecutive_errors:
            return CodeEngineResult(
                success=False,
                error=f"Too many consecutive tool errors ({consecutive_errors})",
                turns_used=turn_idx + 1,
                total_cost_usd=total_cost,
                duration_seconds=time.monotonic() - start_time,
                provider_name=provider,
                turns=tuple(turns),
                files_modified=tuple(sorted(files_modified)),
            )

    # Turn limit exhausted
    return CodeEngineResult(
        success=False,
        error=f"Turn limit reached ({config.max_turns})",
        turns_used=config.max_turns,
        total_cost_usd=total_cost,
        duration_seconds=time.monotonic() - start_time,
        provider_name=provider,
        turns=tuple(turns),
        files_modified=tuple(sorted(files_modified)),
    )


# ---------------------------------------------------------------------------
# Interop with CodingAgentResult
# ---------------------------------------------------------------------------

def to_coding_agent_result(result: CodeEngineResult) -> Any:
    """Convert CodeEngineResult to CodingAgentResult for API compat.

    Lazy import to maintain LEAF status.
    """
    from mother.coding_agent import CodingAgentResult
    return CodingAgentResult(
        success=result.success,
        result_text=result.final_text,
        cost_usd=result.total_cost_usd,
        duration_seconds=result.duration_seconds,
        num_turns=result.turns_used,
        error=result.error,
        is_error=not result.success,
        provider=result.provider_name,
    )
