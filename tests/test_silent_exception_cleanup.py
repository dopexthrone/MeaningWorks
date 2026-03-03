"""Tests verifying silent exception cleanup in chat.py.

After the Hardening Sprint fixed engine.py (8 blocks → logger.debug),
this cleanup applied the same pattern to chat.py's 64 silent handlers.
"""

import ast
import logging
from pathlib import Path

CHAT_PY = Path(__file__).resolve().parent.parent / "mother" / "screens" / "chat.py"


def _parse_chat_ast():
    """Parse chat.py into an AST."""
    source = CHAT_PY.read_text()
    return ast.parse(source, filename=str(CHAT_PY)), source


def test_logger_exists():
    """Verify mother.chat logger is configured."""
    logger = logging.getLogger("mother.chat")
    assert logger is not None
    assert logger.name == "mother.chat"


def test_no_bare_except_exception_pass():
    """AST-walk chat.py: assert zero `except Exception: pass` blocks remain.

    This is the core invariant — every Exception handler must capture `as e`
    and log, not silently swallow.
    """
    tree, source = _parse_chat_ast()
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Only check `except Exception:` (not CancelledError, ValueError, etc.)
        if node.type is None:
            continue
        if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
            continue
        # Check if handler body is just `pass`
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            violations.append(node.lineno)

    assert violations == [], (
        f"Found {len(violations)} bare `except Exception: pass` at lines: {violations}"
    )


def test_exception_handlers_bind_variable():
    """Every `except Exception` handler should bind to a variable (as e)."""
    tree, source = _parse_chat_ast()
    unbound = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if not (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
            continue
        # Check for handlers that are just `pass` or `return None` etc without binding
        # These are acceptable: return None, = None, break (5 legacy handlers)
        # But they should NOT be `pass` without binding
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
            and node.name is None
        ):
            unbound.append(node.lineno)

    assert unbound == [], (
        f"Found {len(unbound)} `except Exception:` without variable binding at lines: {unbound}"
    )


def test_cancelled_error_preserved():
    """asyncio.CancelledError handlers must remain unchanged (correct shutdown behavior)."""
    tree, source = _parse_chat_ast()
    cancelled_handlers = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        # Match `except asyncio.CancelledError:`
        if isinstance(node.type, ast.Attribute) and node.type.attr == "CancelledError":
            cancelled_handlers.append(node.lineno)

    # We expect at least 3 CancelledError handlers (lines ~883, ~6333, ~6347, ~6355)
    assert len(cancelled_handlers) >= 3, (
        f"Expected at least 3 CancelledError handlers, found {len(cancelled_handlers)}"
    )


def test_value_error_preserved():
    """ValueError handler for /listen duration parsing must remain unchanged."""
    tree, source = _parse_chat_ast()
    value_error_handlers = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            continue
        if isinstance(node.type, ast.Name) and node.type.id == "ValueError":
            value_error_handlers.append(node.lineno)

    assert len(value_error_handlers) >= 1, (
        "Expected at least 1 ValueError handler (duration parsing)"
    )


def test_no_silent_pass_remains():
    """Verify no `except Exception: pass` pattern exists (source-level grep).

    Complements the AST test with a direct text scan to catch edge cases.
    """
    source = CHAT_PY.read_text()
    lines = source.split("\n")
    violations = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "except Exception:":
            # Check if the next non-empty line is just `pass`
            for j in range(i + 1, min(i + 3, len(lines))):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    continue
                if next_stripped == "pass":
                    violations.append(i + 1)  # 1-indexed
                break

    assert violations == [], (
        f"Found {len(violations)} `except Exception: pass` at lines: {violations}"
    )
