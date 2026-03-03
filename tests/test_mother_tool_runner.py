"""
Tests for mother/tool_runner.py — LEAF module.

Covers: ToolRunResult, find_tool_project, run_tool, _normalize_name, _find_entry_point.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mother.tool_runner import (
    ToolRunResult,
    find_tool_project,
    run_tool,
    _normalize_name,
    _find_entry_point,
)


# =============================================================================
# ToolRunResult
# =============================================================================

class TestToolRunResult:
    def test_frozen(self):
        r = ToolRunResult(success=True, output="hello")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = ToolRunResult(success=True)
        assert r.output == ""
        assert r.error is None
        assert r.exit_code == 0
        assert r.tool_name == ""

    def test_full(self):
        r = ToolRunResult(
            success=False,
            output="out",
            error="err",
            exit_code=1,
            tool_name="my-tool",
        )
        assert not r.success
        assert r.output == "out"
        assert r.error == "err"
        assert r.exit_code == 1
        assert r.tool_name == "my-tool"


# =============================================================================
# _normalize_name
# =============================================================================

class TestNormalizeName:
    def test_basic(self):
        assert _normalize_name("Hello World") == "hello-world"

    def test_underscores(self):
        assert _normalize_name("my_tool") == "my-tool"

    def test_mixed(self):
        assert _normalize_name("  My_Cool Tool  ") == "my-cool-tool"

    def test_already_normalized(self):
        assert _normalize_name("hello-world") == "hello-world"

    def test_empty(self):
        assert _normalize_name("") == ""
        assert _normalize_name("   ") == ""


# =============================================================================
# find_tool_project
# =============================================================================

class TestFindToolProject:
    def test_exact_match(self, tmp_path):
        (tmp_path / "hello-world").mkdir()
        result = find_tool_project("hello-world", str(tmp_path))
        assert result == str(tmp_path / "hello-world")

    def test_normalized_match(self, tmp_path):
        (tmp_path / "hello-world").mkdir()
        result = find_tool_project("Hello World", str(tmp_path))
        assert result == str(tmp_path / "hello-world")

    def test_underscore_match(self, tmp_path):
        (tmp_path / "my-tool").mkdir()
        result = find_tool_project("my_tool", str(tmp_path))
        assert result == str(tmp_path / "my-tool")

    def test_substring_match(self, tmp_path):
        (tmp_path / "booking-system-v2").mkdir()
        result = find_tool_project("booking-system", str(tmp_path))
        assert result == str(tmp_path / "booking-system-v2")

    def test_no_match(self, tmp_path):
        (tmp_path / "other-project").mkdir()
        result = find_tool_project("hello-world", str(tmp_path))
        assert result is None

    def test_empty_name(self, tmp_path):
        result = find_tool_project("", str(tmp_path))
        assert result is None

    def test_nonexistent_dir(self):
        result = find_tool_project("test", "/nonexistent/path")
        assert result is None

    def test_default_dir(self):
        # Should use ~/motherlabs/projects/ by default
        result = find_tool_project("definitely-not-a-real-tool")
        # Either None or a path — no crash
        assert result is None or isinstance(result, str)

    def test_skips_files(self, tmp_path):
        """Only matches directories, not files."""
        (tmp_path / "hello-world").touch()  # file, not dir
        result = find_tool_project("hello-world", str(tmp_path))
        assert result is None

    def test_prefers_exact_over_substring(self, tmp_path):
        (tmp_path / "weather").mkdir()
        (tmp_path / "weather-extended").mkdir()
        result = find_tool_project("weather", str(tmp_path))
        assert result == str(tmp_path / "weather")


# =============================================================================
# _find_entry_point
# =============================================================================

class TestFindEntryPoint:
    def test_main_py(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        assert _find_entry_point(str(tmp_path)) == "main.py"

    def test_app_py(self, tmp_path):
        (tmp_path / "app.py").write_text("print('hi')")
        assert _find_entry_point(str(tmp_path)) == "app.py"

    def test_prefers_main_over_app(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "app.py").write_text("print('hi')")
        assert _find_entry_point(str(tmp_path)) == "main.py"

    def test_single_py_file(self, tmp_path):
        (tmp_path / "server.py").write_text("print('hi')")
        assert _find_entry_point(str(tmp_path)) == "server.py"

    def test_no_py_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("nothing")
        assert _find_entry_point(str(tmp_path)) is None

    def test_multiple_py_files_no_main(self, tmp_path):
        (tmp_path / "foo.py").write_text("1")
        (tmp_path / "bar.py").write_text("2")
        assert _find_entry_point(str(tmp_path)) is None


# =============================================================================
# run_tool
# =============================================================================

class TestRunTool:
    def test_success(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello from tool')")
        result = run_tool(str(tmp_path))
        assert result.success
        assert "hello from tool" in result.output
        assert result.exit_code == 0
        assert result.tool_name == tmp_path.name

    def test_with_input(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "import sys; data = sys.stdin.read(); print(f'Got: {data}')"
        )
        result = run_tool(str(tmp_path), input_text="test input")
        assert result.success
        assert "Got: test input" in result.output

    def test_nonzero_exit(self, tmp_path):
        (tmp_path / "main.py").write_text("import sys; sys.exit(1)")
        result = run_tool(str(tmp_path))
        assert not result.success
        assert result.exit_code == 1

    def test_stderr_on_error(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "import sys; print('err msg', file=sys.stderr); sys.exit(2)"
        )
        result = run_tool(str(tmp_path))
        assert not result.success
        assert "err msg" in result.error

    def test_no_entry_point(self, tmp_path):
        (tmp_path / "readme.txt").write_text("nothing")
        result = run_tool(str(tmp_path))
        assert not result.success
        assert "No entry point" in result.error
        assert result.exit_code == -1

    def test_timeout(self, tmp_path):
        (tmp_path / "main.py").write_text("import time; time.sleep(100)")
        result = run_tool(str(tmp_path), timeout=0.5)
        assert not result.success
        assert "Timed out" in result.error

    def test_tool_name_from_dir(self, tmp_path):
        project = tmp_path / "my-cool-tool"
        project.mkdir()
        (project / "main.py").write_text("print('works')")
        result = run_tool(str(project))
        assert result.tool_name == "my-cool-tool"

    def test_exception_handling(self, tmp_path):
        (tmp_path / "main.py").write_text("raise RuntimeError('boom')")
        result = run_tool(str(tmp_path))
        assert not result.success
        assert result.exit_code != 0
