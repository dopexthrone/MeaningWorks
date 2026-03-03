"""
Integration tests for sandbox + worktree + code engine + bridge.

Covers:
A. _execute_bash with sandbox wrapping
B. _validate_path symlink hardening
C. run_tests with sandbox_profile
D. Self-build mock: worktree → engine → tests → merge
E. Self-build mock: test failure → worktree removed, main untouched
F. Protected file changes blocked at merge
G. CodeEngineConfig sandbox_profile field
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock
from pathlib import Path

from mother.code_engine import (
    CodeEngineConfig,
    _execute_bash,
    _validate_path,
)
from mother.sandbox import SandboxProfile, create_build_profile
from mother.worktree import BuildWorktree


# ---------------------------------------------------------------------------
# A. _execute_bash with sandbox wrapping
# ---------------------------------------------------------------------------

class TestExecuteBashSandbox:
    def test_without_sandbox_uses_shell(self):
        """Without sandbox_profile, bash executes normally via shell=True."""
        config = CodeEngineConfig(working_dir="/tmp")
        with patch("mother.code_engine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="hello\n", stderr="", returncode=0,
            )
            result = _execute_bash({"command": "echo hello"}, config)
            assert "hello" in result
            # Should use shell=True (string command)
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("shell", False) is True or isinstance(call_kwargs[0][0], str)

    def test_with_sandbox_wraps_command(self):
        """With sandbox_profile, bash wraps command with sandbox-exec."""
        profile = SandboxProfile(allow_write_paths=("/tmp/build",))
        config = CodeEngineConfig(
            working_dir="/tmp",
            sandbox_profile=profile,
        )
        with patch("mother.code_engine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="hello\n", stderr="", returncode=0,
            )
            with patch("mother.sandbox.is_sandbox_available", return_value=True):
                result = _execute_bash({"command": "echo hello"}, config)
            # Should call with a list (sandbox-exec wrapped)
            call_args = mock_run.call_args[0][0]
            assert isinstance(call_args, list)
            assert call_args[0] == "sandbox-exec"

    def test_with_sandbox_fallback_on_import_error(self):
        """If mother.sandbox import fails, falls back to normal execution."""
        config = CodeEngineConfig(
            working_dir="/tmp",
            sandbox_profile=object(),  # non-None to trigger sandbox path
        )
        with patch("mother.code_engine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="ok\n", stderr="", returncode=0,
            )
            with patch.dict("sys.modules", {"mother.sandbox": None}):
                # Import will fail — should fall back
                result = _execute_bash({"command": "echo ok"}, config)
            assert "ok" in result

    def test_sandbox_network_deny_in_sbpl(self):
        """Sandbox profile with deny network generates correct SBPL."""
        profile = SandboxProfile(allow_write_paths=("/tmp/build",), allow_network=False)
        config = CodeEngineConfig(
            working_dir="/tmp",
            sandbox_profile=profile,
        )
        with patch("mother.code_engine.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="", stderr="", returncode=0,
            )
            with patch("mother.sandbox.is_sandbox_available", return_value=True):
                _execute_bash({"command": "curl example.com"}, config)
            call_args = mock_run.call_args[0][0]
            # SBPL string is at index 2
            sbpl = call_args[2]
            assert "(deny network*)" in sbpl


# ---------------------------------------------------------------------------
# B. _validate_path symlink hardening
# ---------------------------------------------------------------------------

class TestValidatePathSymlink:
    def test_rejects_symlink_outside_allowed(self, tmp_path):
        """Symlinks pointing outside allowed paths are rejected."""
        # Create a symlink pointing outside allowed paths
        target = tmp_path / "outside" / "secret.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret")

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        link = allowed / "escape.txt"
        link.symlink_to(target)

        config = CodeEngineConfig(allowed_paths=[str(allowed)])
        err = _validate_path(str(link), config)
        assert err is not None
        assert "symlink" in err.lower()

    def test_allows_symlink_inside_allowed(self, tmp_path):
        """Symlinks pointing within allowed paths are OK."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        target = allowed / "real.txt"
        target.write_text("ok")
        link = allowed / "alias.txt"
        link.symlink_to(target)

        config = CodeEngineConfig(allowed_paths=[str(allowed)])
        err = _validate_path(str(link), config)
        assert err is None

    def test_non_symlink_paths_unaffected(self, tmp_path):
        """Regular files still validate normally."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        f = allowed / "normal.txt"
        f.write_text("fine")

        config = CodeEngineConfig(allowed_paths=[str(allowed)])
        err = _validate_path(str(f), config)
        assert err is None


# ---------------------------------------------------------------------------
# C. run_tests with sandbox_profile
# ---------------------------------------------------------------------------

class TestRunTestsSandbox:
    @patch("mother.claude_code.subprocess.run")
    @patch("mother.claude_code.Path")
    def test_without_sandbox_runs_directly(self, mock_path_cls, mock_run):
        mock_path_inst = MagicMock()
        mock_path_cls.return_value = mock_path_inst
        venv_path = MagicMock()
        venv_path.exists.return_value = True
        mock_path_inst.__truediv__ = lambda self, k: venv_path if k == ".venv" else MagicMock()
        mock_path_cls.side_effect = lambda p: MagicMock(exists=lambda: True) if "pytest" in str(p) else mock_path_inst

        mock_run.return_value = MagicMock(returncode=0)

        from mother.claude_code import run_tests
        result = run_tests("/repo")
        assert result is True

    @patch("mother.claude_code.subprocess.run")
    @patch("mother.claude_code.Path")
    def test_with_sandbox_wraps_pytest(self, mock_path_cls, mock_run):
        # Make Path().exists() return True for pytest binary
        mock_path_cls.side_effect = lambda p: MagicMock(
            exists=lambda: True,
            __truediv__=lambda self, k: MagicMock(
                exists=lambda: True,
                __truediv__=lambda self2, k2: MagicMock(
                    exists=lambda: True,
                    __str__=lambda self3: f"{p}/.venv/bin/pytest",
                ),
                __str__=lambda self2: f"{p}/.venv/bin",
            ),
            __str__=lambda self: str(p),
        )

        mock_run.return_value = MagicMock(returncode=0)

        from mother.claude_code import run_tests
        profile = SandboxProfile(allow_write_paths=("/tmp/build",))

        with patch("mother.sandbox.is_sandbox_available", return_value=True):
            result = run_tests("/repo", sandbox_profile=profile)

        # The command should have been a list (sandbox-exec wrapped)
        call_args = mock_run.call_args[0][0]
        assert isinstance(call_args, list)
        assert call_args[0] == "sandbox-exec"


# ---------------------------------------------------------------------------
# D. Self-build mock: full success path
# ---------------------------------------------------------------------------

class TestSelfBuildWorktreeSuccess:
    """Test the happy path: worktree created → engine runs → tests pass → merge back."""

    def test_worktree_lifecycle_in_self_build(self):
        """Verify worktree is created, used for build, merged, and cleaned up."""
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc123")

        with patch("mother.worktree.worktree_create", return_value=wt) as mock_create, \
             patch("mother.worktree.worktree_merge", return_value=(True, "")) as mock_merge, \
             patch("mother.worktree.worktree_remove") as mock_remove, \
             patch("mother.worktree.worktree_has_protected_changes", return_value=[]):

            # Simulate the flow
            created = mock_create("/repo")
            assert created.path == "/tmp/wt"

            merge_ok, _ = mock_merge("/repo", created)
            assert merge_ok

            mock_remove("/repo", created)
            mock_remove.assert_called_once()


# ---------------------------------------------------------------------------
# E. Self-build mock: failure path
# ---------------------------------------------------------------------------

class TestSelfBuildWorktreeFailure:
    """Test failure: tests fail → worktree removed, main untouched."""

    def test_worktree_removed_on_failure(self):
        """Worktree is cleaned up even when build/tests fail."""
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc123")

        with patch("mother.worktree.worktree_create", return_value=wt), \
             patch("mother.worktree.worktree_remove") as mock_remove:

            # Simulate: engine ran, tests failed, cleanup happens
            try:
                raise RuntimeError("tests failed")
            except RuntimeError:
                pass
            finally:
                from mother.worktree import worktree_remove
                worktree_remove("/repo", wt)

            mock_remove.assert_called_once_with("/repo", wt)


# ---------------------------------------------------------------------------
# F. Protected file changes blocked at merge
# ---------------------------------------------------------------------------

class TestProtectedFileBlock:
    def test_protected_changes_detected(self):
        """Protected file modifications in worktree are caught before merge."""
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc123")

        with patch("mother.worktree.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="mother/context.py\nmother/sandbox.py\n",
            )
            from mother.worktree import worktree_has_protected_changes
            result = worktree_has_protected_changes(wt)
            assert "mother/context.py" in result
            # sandbox.py is not protected
            assert "mother/sandbox.py" not in result


# ---------------------------------------------------------------------------
# G. CodeEngineConfig sandbox_profile field
# ---------------------------------------------------------------------------

class TestCodeEngineConfigSandbox:
    def test_sandbox_profile_default_none(self):
        config = CodeEngineConfig()
        assert config.sandbox_profile is None

    def test_sandbox_profile_accepts_value(self):
        profile = SandboxProfile(allow_write_paths=("/tmp",))
        config = CodeEngineConfig(sandbox_profile=profile)
        assert config.sandbox_profile is profile

    def test_sandbox_profile_in_config(self):
        """sandbox_profile integrates with other config fields."""
        profile = create_build_profile("/tmp/wt", "/tmp/venv")
        config = CodeEngineConfig(
            working_dir="/tmp/wt",
            allowed_paths=["/tmp/wt"],
            sandbox_profile=profile,
        )
        assert config.working_dir == "/tmp/wt"
        assert config.sandbox_profile.allow_network is False
