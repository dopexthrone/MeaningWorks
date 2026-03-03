"""
Tool runner — execute registered tools as subprocesses.

LEAF module. Stdlib only. No imports from core/ or mother/.

Finds tool projects by name in ~/motherlabs/projects/ and runs them
as subprocesses, capturing output.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ToolRunResult:
    """Outcome of running a tool."""

    success: bool
    output: str = ""
    error: Optional[str] = None
    exit_code: int = 0
    tool_name: str = ""


def _normalize_name(name: str) -> str:
    """Normalize a tool name for directory matching.

    Strips whitespace, lowercases, replaces spaces/underscores with hyphens.
    """
    return name.strip().lower().replace(" ", "-").replace("_", "-")


def find_tool_project(tool_name: str, projects_dir: str = "") -> Optional[str]:
    """Find a project directory by normalized name match.

    Searches ~/motherlabs/projects/ for a directory whose normalized
    name matches the normalized tool_name.

    Args:
        tool_name: Human-readable tool name (e.g. "hello-world")
        projects_dir: Override projects directory (default ~/motherlabs/projects)

    Returns:
        Absolute path to project directory, or None if not found
    """
    if not projects_dir:
        projects_dir = str(Path.home() / "motherlabs" / "projects")

    if not os.path.isdir(projects_dir):
        return None

    target = _normalize_name(tool_name)
    if not target:
        return None

    for entry in os.listdir(projects_dir):
        entry_path = os.path.join(projects_dir, entry)
        if os.path.isdir(entry_path) and _normalize_name(entry) == target:
            return entry_path

    # Substring match as fallback
    for entry in os.listdir(projects_dir):
        entry_path = os.path.join(projects_dir, entry)
        if os.path.isdir(entry_path) and target in _normalize_name(entry):
            return entry_path

    return None


def _find_entry_point(project_dir: str) -> Optional[str]:
    """Find the entry point script in a project directory.

    Checks main.py, then app.py, then any single .py file.
    """
    for candidate in ("main.py", "app.py"):
        path = os.path.join(project_dir, candidate)
        if os.path.isfile(path):
            return candidate

    # Fallback: any .py file at top level
    py_files = [f for f in os.listdir(project_dir) if f.endswith(".py")]
    if len(py_files) == 1:
        return py_files[0]

    return None


def run_tool(
    project_dir: str,
    input_text: str = "",
    timeout: float = 30.0,
) -> ToolRunResult:
    """Run a tool project as a subprocess.

    Finds the entry point, runs it with python3, passes input_text
    via stdin, captures stdout/stderr.

    Args:
        project_dir: Absolute path to the project directory
        input_text: Text to pass via stdin (empty string = no input)
        timeout: Maximum execution time in seconds

    Returns:
        ToolRunResult with output or error
    """
    entry = _find_entry_point(project_dir)
    if not entry:
        return ToolRunResult(
            success=False,
            error="No entry point found (expected main.py or app.py)",
            exit_code=-1,
            tool_name=os.path.basename(project_dir),
        )

    tool_name = os.path.basename(project_dir)
    cmd = [sys.executable, entry]

    try:
        result = subprocess.run(
            cmd,
            input=input_text if input_text else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return ToolRunResult(
                success=True,
                output=output,
                exit_code=0,
                tool_name=tool_name,
            )
        else:
            return ToolRunResult(
                success=False,
                output=output,
                error=stderr or f"Process exited with code {result.returncode}",
                exit_code=result.returncode,
                tool_name=tool_name,
            )

    except subprocess.TimeoutExpired:
        return ToolRunResult(
            success=False,
            error=f"Timed out after {timeout:.0f}s",
            exit_code=-1,
            tool_name=tool_name,
        )
    except FileNotFoundError:
        return ToolRunResult(
            success=False,
            error=f"Python interpreter not found: {sys.executable}",
            exit_code=-1,
            tool_name=tool_name,
        )
    except Exception as e:
        return ToolRunResult(
            success=False,
            error=str(e),
            exit_code=-1,
            tool_name=tool_name,
        )
