"""
Project integration utilities.

Extracts metadata from existing projects for packaging and registration.
LEAF module — stdlib only, no core/ or messaging/ imports.
"""

import json
from pathlib import Path
from typing import Any, Dict


def validate_project_structure(project_path: Path) -> tuple[bool, str]:
    """
    Validate that a project has the required structure for integration.

    Returns:
        (valid, error_message) where error_message is empty if valid
    """
    if not project_path.exists():
        return False, f"Path does not exist: {project_path}"

    if not project_path.is_dir():
        return False, f"Path is not a directory: {project_path}"

    # Check for entry point
    entry_candidates = ["main.py", "app.py", "server.py"]
    has_entry = any((project_path / name).exists() for name in entry_candidates)

    # Or single .py file
    if not has_entry:
        py_files = list(project_path.glob("*.py"))
        has_entry = len(py_files) == 1

    if not has_entry:
        return False, "No entry point found (main.py, app.py, server.py, or single .py)"

    # Check for tests
    tests_dir = project_path / "tests"
    if not tests_dir.exists():
        return False, "No tests/ directory found"

    return True, ""


def extract_project_metadata(project_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from project structure.

    Tries blueprint.json first, falls back to pyproject.toml or structure inference.

    Returns:
        {name, description, domain, components, entry_point}
    """
    # Try blueprint.json
    blueprint_path = project_path / "blueprint.json"
    if blueprint_path.exists():
        try:
            with open(blueprint_path) as f:
                bp = json.load(f)

            return {
                "name": bp.get("name", project_path.name),
                "description": bp.get("description", ""),
                "domain": bp.get("domain", "software"),
                "components": bp.get("components", []),
                "entry_point": _find_entry_point(project_path),
            }
        except (json.JSONDecodeError, OSError):
            pass

    # Try pyproject.toml
    pyproject_path = project_path / "pyproject.toml"
    if pyproject_path.exists():
        try:
            # Simple TOML parsing for [project] section
            with open(pyproject_path) as f:
                content = f.read()

            name = None
            description = None

            in_project = False
            for line in content.splitlines():
                line = line.strip()

                if line == "[project]":
                    in_project = True
                    continue

                if in_project and line.startswith("["):
                    break

                if in_project:
                    if line.startswith("name"):
                        name = line.split("=", 1)[1].strip().strip('"\'')
                    elif line.startswith("description"):
                        description = line.split("=", 1)[1].strip().strip('"\'')

            if name or description:
                return {
                    "name": name or project_path.name,
                    "description": description or "",
                    "domain": "software",
                    "components": [],
                    "entry_point": _find_entry_point(project_path),
                }
        except OSError:
            pass

    # Fall back to structure inference
    return {
        "name": project_path.name,
        "description": "",
        "domain": "software",
        "components": [],
        "entry_point": _find_entry_point(project_path),
    }


def extract_code_files(project_path: Path) -> Dict[str, str]:
    """
    Recursively extract all .py files from project.

    Returns:
        {relative_path: file_content}

    Skips __pycache__, .venv, .git directories.
    """
    code_files = {}
    skip_dirs = {"__pycache__", ".venv", ".git", ".pytest_cache", "node_modules"}

    for py_file in project_path.rglob("*.py"):
        # Skip if in excluded directory
        if any(skip in py_file.parts for skip in skip_dirs):
            continue

        try:
            rel_path = py_file.relative_to(project_path)
            with open(py_file, encoding="utf-8") as f:
                code_files[str(rel_path)] = f.read()
        except (OSError, UnicodeDecodeError):
            # Skip files we can't read
            continue

    return code_files


def _find_entry_point(project_path: Path) -> str:
    """Find the entry point file for a project."""
    candidates = ["main.py", "app.py", "server.py"]

    for name in candidates:
        if (project_path / name).exists():
            return name

    # Single .py file
    py_files = list(project_path.glob("*.py"))
    if len(py_files) == 1:
        return py_files[0].name

    return "main.py"  # Default fallback
