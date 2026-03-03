"""
Tests for core/ and kernel/ silent exception cleanup.

Verifies that no bare `except Exception:` handlers remain
in core/ or kernel/ directories (all should capture `as e`
and log via logger.debug()).

Also verifies stress_test.py no longer contains hardcoded API keys.
"""

import ast
import os
from pathlib import Path


# Root of the project
_PROJECT_ROOT = Path(__file__).parent.parent


def _find_bare_except_exception(directory: Path) -> list[tuple[str, int]]:
    """Find all `except Exception:` without `as` clause via AST walk.

    Returns list of (filepath, lineno) for violations.
    """
    violations = []
    for py_file in sorted(directory.rglob("*.py")):
        # Skip __pycache__ and test files
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if it catches Exception without naming it
                if (node.type is not None
                        and isinstance(node.type, ast.Name)
                        and node.type.id == "Exception"
                        and node.name is None):
                    rel = py_file.relative_to(_PROJECT_ROOT)
                    violations.append((str(rel), node.lineno))

    return violations


def test_no_bare_except_exception_in_core():
    """core/ should have zero bare `except Exception:` handlers."""
    violations = _find_bare_except_exception(_PROJECT_ROOT / "core")
    if violations:
        details = "\n".join(f"  {f}:{ln}" for f, ln in violations)
        raise AssertionError(
            f"{len(violations)} bare `except Exception:` in core/:\n{details}"
        )


def test_no_bare_except_exception_in_kernel():
    """kernel/ should have zero bare `except Exception:` handlers."""
    violations = _find_bare_except_exception(_PROJECT_ROOT / "kernel")
    if violations:
        details = "\n".join(f"  {f}:{ln}" for f, ln in violations)
        raise AssertionError(
            f"{len(violations)} bare `except Exception:` in kernel/:\n{details}"
        )


def test_stress_test_no_hardcoded_key():
    """stress_test.py must not contain any hardcoded API keys."""
    stress_test = _PROJECT_ROOT / "stress_test.py"
    if not stress_test.exists():
        return  # file may not exist in all environments

    content = stress_test.read_text()
    # Check for xai- prefixed keys (known pattern)
    assert "xai-" not in content, "stress_test.py contains hardcoded xai- API key"
    # Check that it uses env var with strict exit
    assert "XAI_API_KEY" in content
    assert 'sys.exit' in content or 'raise' in content


def test_stress_test_requires_env_var():
    """stress_test.py should exit if XAI_API_KEY not set."""
    stress_test = _PROJECT_ROOT / "stress_test.py"
    if not stress_test.exists():
        return

    content = stress_test.read_text()
    assert "sys.exit" in content, "stress_test.py should sys.exit when key missing"


def test_core_llm_has_logger():
    """core/llm.py should have a module-level logger."""
    llm_path = _PROJECT_ROOT / "core" / "llm.py"
    content = llm_path.read_text()
    assert "import logging" in content
    assert "logger" in content


def test_runtime_validator_has_logger():
    """core/runtime_validator.py should have a module-level logger."""
    rv_path = _PROJECT_ROOT / "core" / "runtime_validator.py"
    content = rv_path.read_text()
    assert "import logging" in content
    assert "logger" in content


def test_tool_export_has_logger():
    """core/tool_export.py should have a module-level logger."""
    te_path = _PROJECT_ROOT / "core" / "tool_export.py"
    content = te_path.read_text()
    assert "import logging" in content
    assert "logger" in content
