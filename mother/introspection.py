"""
Structural self-knowledge — Mother's body map.

LEAF module. Stdlib only. No imports from core/ or mother/.

Computes codebase topology (file counts, modules, LOC, protected files)
and git diff stats after self-build. Both feed into ContextData so the
LLM sees its own structure.

Usage:
    topo = scan_topology("/path/to/repo")
    delta = git_diff_stats("/path/to/repo", "abc1234")
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Directories to exclude from topology scan
_EXCLUDE_DIRS = {".venv", "__pycache__", "output", ".git", "node_modules", ".tox", ".mypy_cache"}

# Architectural invariants — hardcoded, not discovered
_PROTECTED_FILES = [
    "mother/context.py",
    "mother/persona.py",
    "mother/senses.py",
]

_BOUNDARY_RULE = "bridge.py is the only mother→core import path"


@dataclass(frozen=True)
class CodebaseTopology:
    """Static shape of the codebase."""

    total_files: int = 0
    total_test_files: int = 0
    total_tests: int = 0
    modules: Dict[str, int] = field(default_factory=dict)
    total_lines: int = 0
    protected_files: List[str] = field(default_factory=list)
    boundary_rule: str = ""


@dataclass(frozen=True)
class BuildDelta:
    """What changed in last self-build."""

    files_modified: int = 0
    files_added: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    modules_touched: List[str] = field(default_factory=list)


def scan_topology(repo_dir: str) -> CodebaseTopology:
    """Scan the repo for structural metrics.

    Pure function. Walks with pathlib, counts .py files grouped by
    top-level directory. Tries pytest --collect-only for test count.
    """
    root = Path(repo_dir)
    total_files = 0
    total_test_files = 0
    total_lines = 0
    modules: Dict[str, int] = {}

    # Count source .py files (excluding tests/ and excluded dirs)
    for py_file in root.rglob("*.py"):
        # Skip excluded directories
        parts = py_file.relative_to(root).parts
        if any(p in _EXCLUDE_DIRS for p in parts):
            continue

        # Test files
        if parts[0] == "tests":
            total_test_files += 1
            continue

        total_files += 1

        # Group by top-level module directory
        top_level = parts[0] if len(parts) > 1 else "(root)"
        modules[top_level] = modules.get(top_level, 0) + 1

        # Count lines
        try:
            total_lines += len(py_file.read_text(errors="replace").splitlines())
        except (OSError, UnicodeDecodeError):
            pass

    # Try to get test count from pytest --collect-only
    total_tests = 0
    try:
        venv_pytest = str(root / ".venv" / "bin" / "pytest")
        result = subprocess.run(
            [venv_pytest, "tests/", "--collect-only", "-q", "--no-header"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Last non-empty line typically: "4388 tests collected"
            for line in reversed(result.stdout.strip().splitlines()):
                line = line.strip()
                if "test" in line and any(c.isdigit() for c in line):
                    # Extract number before "test"
                    parts_line = line.split()
                    for token in parts_line:
                        if token.isdigit():
                            total_tests = int(token)
                            break
                    break
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass

    return CodebaseTopology(
        total_files=total_files,
        total_test_files=total_test_files,
        total_tests=total_tests,
        modules=modules,
        total_lines=total_lines,
        protected_files=list(_PROTECTED_FILES),
        boundary_rule=_BOUNDARY_RULE,
    )


def git_diff_stats(repo_dir: str, before_hash: str) -> BuildDelta:
    """Compute what changed between before_hash and HEAD.

    Pure function. Runs git diff --numstat and --name-status.
    Returns empty BuildDelta on any failure.
    """
    if not before_hash:
        return BuildDelta()

    try:
        # Get line-level stats
        numstat = subprocess.run(
            ["git", "diff", "--numstat", f"{before_hash}..HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )

        lines_added = 0
        lines_removed = 0
        if numstat.returncode == 0:
            for line in numstat.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        added = int(parts[0]) if parts[0] != "-" else 0
                        removed = int(parts[1]) if parts[1] != "-" else 0
                        lines_added += added
                        lines_removed += removed
                    except ValueError:
                        continue

        # Get file-level status
        namestatus = subprocess.run(
            ["git", "diff", "--name-status", f"{before_hash}..HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=15,
        )

        files_modified = 0
        files_added = 0
        modules_touched: set = set()

        if namestatus.returncode == 0:
            for line in namestatus.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                status = parts[0].strip()
                filepath = parts[1] if len(parts) >= 2 else ""

                if status.startswith("M"):
                    files_modified += 1
                elif status.startswith("A"):
                    files_added += 1

                # Extract top-level module
                path_parts = filepath.split("/")
                if path_parts:
                    modules_touched.add(path_parts[0])

        return BuildDelta(
            files_modified=files_modified,
            files_added=files_added,
            lines_added=lines_added,
            lines_removed=lines_removed,
            modules_touched=sorted(modules_touched),
        )

    except (OSError, subprocess.TimeoutExpired):
        return BuildDelta()
