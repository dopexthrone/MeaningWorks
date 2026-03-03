"""Tests for mother/claude_code.py — Claude Code CLI integration."""

import json
import os
import subprocess
from dataclasses import FrozenInstanceError
from unittest.mock import patch, MagicMock

import pytest

from mother.claude_code import (
    ClaudeCodeResult,
    invoke_claude_code,
    git_snapshot,
    git_rollback,
    run_tests,
    _clean_env,
)


def _make_path_mock(exists=True):
    """Create a mock for pathlib.Path that won't leak filesystem artifacts.

    Bare MagicMock Path objects produce files named '<MagicMock ...>' when
    str() or os.fspath() is called on them (e.g. by subprocess.run(cwd=...)).
    This helper ensures __str__ and __fspath__ return sensible values.
    """
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = exists
    mock_path_instance.__str__ = lambda self: "/mock/path"
    mock_path_instance.__fspath__ = lambda self: "/mock/path"
    # Support / operator (Path.__truediv__)
    mock_path_instance.__truediv__ = lambda self, other: _make_path_instance(exists)

    mock_path_cls = MagicMock()
    mock_path_cls.return_value = mock_path_instance
    mock_path_cls.home.return_value = mock_path_instance
    mock_path_cls.side_effect = lambda x: _make_path_instance(exists)
    return mock_path_cls


def _make_path_instance(exists=True):
    """Create a single mock Path instance with safe str/fspath."""
    inst = MagicMock()
    inst.exists.return_value = exists
    inst.__str__ = lambda self: "/mock/path"
    inst.__fspath__ = lambda self: "/mock/path"
    inst.__truediv__ = lambda self, other: _make_path_instance(exists)
    return inst


# --- ClaudeCodeResult ---

class TestClaudeCodeResult:
    def test_frozen(self):
        r = ClaudeCodeResult(success=True)
        with pytest.raises(FrozenInstanceError):
            r.success = False

    def test_defaults(self):
        r = ClaudeCodeResult()
        assert r.success is False
        assert r.result_text == ""
        assert r.session_id == ""
        assert r.cost_usd == 0.0
        assert r.duration_seconds == 0.0
        assert r.num_turns == 0
        assert r.error == ""
        assert r.is_error is False

    def test_values(self):
        r = ClaudeCodeResult(
            success=True,
            result_text="done",
            session_id="abc123",
            cost_usd=0.42,
            duration_seconds=12.5,
            num_turns=3,
        )
        assert r.success is True
        assert r.result_text == "done"
        assert r.session_id == "abc123"
        assert r.cost_usd == 0.42
        assert r.num_turns == 3


# --- _clean_env ---

class TestCleanEnv:
    def test_removes_claudecode(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1"}):
            env = _clean_env()
            assert "CLAUDECODE" not in env

    def test_preserves_other_vars(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/test"}):
            env = _clean_env()
            assert "PATH" in env
            assert "HOME" in env

    def test_no_claudecode_is_fine(self):
        env_backup = os.environ.copy()
        os.environ.pop("CLAUDECODE", None)
        env = _clean_env()
        assert "CLAUDECODE" not in env
        os.environ.update(env_backup)


# --- invoke_claude_code ---

class TestInvokeClaude:
    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_success_json(self, mock_run, mock_path_cls):
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_path_cls.home.return_value = _make_path_instance(exists=True)

        json_output = json.dumps({
            "result": "Changes applied",
            "session_id": "sess-123",
            "cost_usd": 0.25,
            "num_turns": 5,
        })
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json_output,
            stderr="",
        )

        result = invoke_claude_code(
            "Fix the bug",
            cwd="/repo",
            claude_path="/mock/claude",
        )
        assert result.success is True
        assert result.result_text == "Changes applied"
        assert result.session_id == "sess-123"
        assert result.cost_usd == 0.25
        assert result.num_turns == 5

        # Verify CLAUDECODE is unset in subprocess env
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
        assert "CLAUDECODE" not in env

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_failure_nonzero_exit(self, mock_run, mock_path_cls):
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: something went wrong",
        )

        result = invoke_claude_code(
            "Fix the bug",
            cwd="/repo",
            claude_path="/mock/claude",
        )
        assert result.success is False
        assert result.is_error is True
        assert "something went wrong" in result.error

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_timeout(self, mock_run, mock_path_cls):
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=600)

        result = invoke_claude_code(
            "Fix the bug",
            cwd="/repo",
            claude_path="/mock/claude",
        )
        assert result.success is False
        assert result.is_error is True
        assert "Timed out" in result.error

    def test_cli_not_found(self):
        result = invoke_claude_code(
            "Fix the bug",
            cwd="/repo",
            claude_path="/nonexistent/path/claude",
        )
        assert result.success is False
        assert result.is_error is True
        assert "not found" in result.error

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_plain_text_fallback(self, mock_run, mock_path_cls):
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Just plain text output",
            stderr="",
        )

        result = invoke_claude_code(
            "Fix the bug",
            cwd="/repo",
            claude_path="/mock/claude",
        )
        assert result.success is True
        assert result.result_text == "Just plain text output"
        assert result.session_id == ""

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_env_claudecode_unset(self, mock_run, mock_path_cls):
        """Verify CLAUDECODE is removed from subprocess environment."""
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.return_value = MagicMock(returncode=0, stdout="{}", stderr="")

        with patch.dict(os.environ, {"CLAUDECODE": "1"}):
            invoke_claude_code("test", cwd="/repo", claude_path="/mock/claude")

        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env", {})
        assert "CLAUDECODE" not in env


# --- git_snapshot ---

class TestGitSnapshot:
    @patch("mother.claude_code.subprocess.run")
    def test_returns_hash(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git commit
            MagicMock(returncode=0, stdout="abc123def\n"),  # git rev-parse
        ]
        result = git_snapshot("/repo")
        assert result == "abc123def"

    @patch("mother.claude_code.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.side_effect = Exception("git not available")
        result = git_snapshot("/repo")
        assert result == ""

    @patch("mother.claude_code.subprocess.run")
    def test_env_clean(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\n")
        with patch.dict(os.environ, {"CLAUDECODE": "1"}):
            git_snapshot("/repo")
        for call in mock_run.call_args_list:
            env = call.kwargs.get("env") or call[1].get("env", {})
            assert "CLAUDECODE" not in env


# --- git_rollback ---

class TestGitRollback:
    @patch("mother.claude_code.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert git_rollback("/repo", "abc123") is True

    @patch("mother.claude_code.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert git_rollback("/repo", "abc123") is False

    def test_empty_hash(self):
        assert git_rollback("/repo", "") is False


# --- run_tests ---

class TestRunTests:
    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_pass(self, mock_run, mock_path_cls):
        pytest_inst = _make_path_instance(exists=True)
        pytest_inst.__str__ = lambda self: "/repo/.venv/bin/pytest"
        path_inst = _make_path_instance(exists=True)
        path_inst.__truediv__ = lambda self, other: pytest_inst
        mock_path_cls.return_value = path_inst
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.return_value = MagicMock(returncode=0)
        assert run_tests("/repo") is True

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_fail(self, mock_run, mock_path_cls):
        pytest_inst = _make_path_instance(exists=True)
        pytest_inst.__str__ = lambda self: "/repo/.venv/bin/pytest"
        path_inst = _make_path_instance(exists=True)
        path_inst.__truediv__ = lambda self, other: pytest_inst
        mock_path_cls.return_value = path_inst
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.return_value = MagicMock(returncode=1)
        assert run_tests("/repo") is False

    @patch("mother.claude_code.Path")
    @patch("mother.claude_code.subprocess.run")
    def test_exception(self, mock_run, mock_path_cls):
        pytest_inst = _make_path_instance(exists=True)
        pytest_inst.__str__ = lambda self: "/repo/.venv/bin/pytest"
        path_inst = _make_path_instance(exists=True)
        path_inst.__truediv__ = lambda self, other: pytest_inst
        mock_path_cls.return_value = path_inst
        mock_path_cls.side_effect = lambda x: _make_path_instance(exists=True)
        mock_run.side_effect = Exception("process error")
        assert run_tests("/repo") is False
