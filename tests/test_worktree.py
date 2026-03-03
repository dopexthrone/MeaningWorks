"""
Tests for mother/worktree.py — Git worktree lifecycle.

LEAF module. Covers:
A. BuildWorktree frozen dataclass
B. _git_env() environment helper
C. worktree_create() — creates directory, branch, symlinks .venv
D. worktree_merge() — squash merge back to main
E. worktree_remove() — cleanup directory and branch
F. worktree_diff_summary() — diff stat output
G. worktree_has_protected_changes() — detects protected file modifications
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call
from dataclasses import FrozenInstanceError

from mother.worktree import (
    BuildWorktree,
    _git_env,
    worktree_create,
    worktree_merge,
    worktree_remove,
    worktree_diff_summary,
    worktree_has_protected_changes,
)


# ---------------------------------------------------------------------------
# A. BuildWorktree frozen dataclass
# ---------------------------------------------------------------------------

class TestBuildWorktree:
    def test_frozen(self):
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc123")
        with pytest.raises(FrozenInstanceError):
            wt.path = "/other"

    def test_fields(self):
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc123")
        assert wt.path == "/tmp/wt"
        assert wt.branch == "build-abc"
        assert wt.base_commit == "abc123"


# ---------------------------------------------------------------------------
# B. _git_env()
# ---------------------------------------------------------------------------

class TestGitEnv:
    def test_strips_claudecode(self):
        with patch.dict(os.environ, {"CLAUDECODE": "1"}, clear=False):
            env = _git_env()
            assert "CLAUDECODE" not in env

    def test_strips_anthropic_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False):
            env = _git_env()
            assert "ANTHROPIC_API_KEY" not in env

    def test_preserves_other_vars(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}, clear=False):
            env = _git_env()
            assert env.get("MY_VAR") == "hello"


# ---------------------------------------------------------------------------
# C. worktree_create()
# ---------------------------------------------------------------------------

class TestWorktreeCreate:
    @patch("mother.worktree.subprocess.run")
    @patch("mother.worktree.os.makedirs")
    @patch("mother.worktree.os.path.isdir", return_value=True)
    @patch("mother.worktree.os.path.exists", return_value=False)
    @patch("mother.worktree.os.symlink")
    @patch("mother.worktree.os.path.realpath", side_effect=lambda p: p)
    @patch("mother.worktree.uuid.uuid4")
    def test_creates_worktree(self, mock_uuid, mock_realpath, mock_symlink,
                                mock_exists, mock_isdir, mock_makedirs, mock_run):
        mock_uuid.return_value = MagicMock(hex="abcdef123456")
        # git rev-parse HEAD
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n"),  # rev-parse
            MagicMock(returncode=0),  # worktree add
        ]

        wt = worktree_create("/repo")
        assert wt.branch == "build-abcdef123456"
        assert "abcdef123456" in wt.branch
        assert wt.base_commit == "abc123"
        assert ".claude/worktrees/" in wt.path

    @patch("mother.worktree.subprocess.run")
    def test_raises_on_head_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not a git repo")
        with pytest.raises(RuntimeError, match="Failed to get HEAD"):
            worktree_create("/repo")

    @patch("mother.worktree.subprocess.run")
    @patch("mother.worktree.os.makedirs")
    def test_raises_on_worktree_add_failure(self, mock_makedirs, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n"),  # rev-parse
            MagicMock(returncode=1, stderr="worktree add failed"),  # worktree add
        ]
        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            worktree_create("/repo")

    @patch("mother.worktree.subprocess.run")
    @patch("mother.worktree.os.makedirs")
    @patch("mother.worktree.os.path.isdir", return_value=False)  # no .venv
    @patch("mother.worktree.uuid.uuid4")
    def test_no_symlink_without_venv(self, mock_uuid, mock_isdir, mock_makedirs, mock_run):
        mock_uuid.return_value = MagicMock(hex="abcdef123456")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n"),
            MagicMock(returncode=0),
        ]

        with patch("mother.worktree.os.symlink") as mock_symlink:
            wt = worktree_create("/repo")
            mock_symlink.assert_not_called()

    @patch("mother.worktree.subprocess.run")
    @patch("mother.worktree.os.makedirs")
    @patch("mother.worktree.os.path.isdir", return_value=True)  # .venv exists
    @patch("mother.worktree.os.path.exists", return_value=False)
    @patch("mother.worktree.os.symlink")
    @patch("mother.worktree.os.path.realpath", side_effect=lambda p: p)
    @patch("mother.worktree.uuid.uuid4")
    def test_symlinks_venv(self, mock_uuid, mock_realpath, mock_symlink,
                            mock_exists, mock_isdir, mock_makedirs, mock_run):
        mock_uuid.return_value = MagicMock(hex="abcdef123456")
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n"),
            MagicMock(returncode=0),
        ]

        wt = worktree_create("/repo")
        mock_symlink.assert_called_once()

    @patch("mother.worktree.subprocess.run")
    @patch("mother.worktree.os.makedirs")
    @patch("mother.worktree.os.path.isdir", return_value=True)
    @patch("mother.worktree.os.path.exists", return_value=False)
    @patch("mother.worktree.os.symlink")
    @patch("mother.worktree.os.path.realpath", side_effect=lambda p: p)
    @patch("mother.worktree.uuid.uuid4")
    def test_unique_branch_names(self, mock_uuid, mock_realpath, mock_symlink,
                                    mock_exists, mock_isdir, mock_makedirs, mock_run):
        """Each worktree gets a unique branch name."""
        mock_uuid.side_effect = [
            MagicMock(hex="aaa111222333"),
            MagicMock(hex="bbb444555666"),
        ]
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc\n"), MagicMock(returncode=0),
            MagicMock(returncode=0, stdout="def\n"), MagicMock(returncode=0),
        ]

        wt1 = worktree_create("/repo")
        wt2 = worktree_create("/repo")
        assert wt1.branch != wt2.branch


# ---------------------------------------------------------------------------
# D. worktree_merge()
# ---------------------------------------------------------------------------

class TestWorktreeMerge:
    @patch("mother.worktree.subprocess.run")
    def test_merge_success(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="main\n"),  # symbolic-ref
            MagicMock(returncode=0),  # merge --squash
        ]
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        ok, msg = worktree_merge("/repo", wt)
        assert ok is True
        assert msg == ""

    @patch("mother.worktree.subprocess.run")
    def test_merge_conflict(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="main\n"),  # symbolic-ref
            MagicMock(returncode=1, stderr="CONFLICT"),  # merge --squash
            MagicMock(returncode=0),  # reset --hard
        ]
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        ok, msg = worktree_merge("/repo", wt)
        assert ok is False
        assert "CONFLICT" in msg

    @patch("mother.worktree.subprocess.run")
    def test_checkout_main_if_on_other_branch(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="build-abc\n"),  # on build branch
            MagicMock(returncode=0),  # rev-parse --verify main
            MagicMock(returncode=0),  # checkout main
            MagicMock(returncode=0),  # merge --squash
        ]
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        ok, msg = worktree_merge("/repo", wt)
        assert ok is True

    @patch("mother.worktree.subprocess.run")
    def test_checkout_failure(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="other\n"),  # on other branch
            MagicMock(returncode=0),  # rev-parse --verify main
            MagicMock(returncode=1, stderr="checkout failed"),  # checkout main
        ]
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        ok, msg = worktree_merge("/repo", wt)
        assert ok is False
        assert "checkout" in msg.lower() or "failed" in msg.lower()


# ---------------------------------------------------------------------------
# E. worktree_remove()
# ---------------------------------------------------------------------------

class TestWorktreeRemove:
    @patch("mother.worktree.subprocess.run")
    def test_removes_worktree_and_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="main\n")
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        # Should not raise
        worktree_remove("/repo", wt)
        # Check git worktree remove was called
        calls = mock_run.call_args_list
        cmds = [c[0][0] for c in calls]
        assert any("worktree" in cmd and "remove" in cmd for cmd in cmds if isinstance(cmd, list))

    @patch("mother.worktree.subprocess.run", side_effect=Exception("gone"))
    def test_safe_if_already_gone(self, mock_run):
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        # Should not raise even if everything fails
        worktree_remove("/repo", wt)


# ---------------------------------------------------------------------------
# F. worktree_diff_summary()
# ---------------------------------------------------------------------------

class TestWorktreeDiffSummary:
    @patch("mother.worktree.subprocess.run")
    def test_returns_diff_stat(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=" mother/sandbox.py | 100 +++\n 1 file changed\n",
        )
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_diff_summary(wt)
        assert "sandbox.py" in result

    @patch("mother.worktree.subprocess.run")
    def test_returns_empty_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_diff_summary(wt)
        assert result == ""

    @patch("mother.worktree.subprocess.run", side_effect=Exception("fail"))
    def test_returns_empty_on_exception(self, mock_run):
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_diff_summary(wt)
        assert result == ""


# ---------------------------------------------------------------------------
# G. worktree_has_protected_changes()
# ---------------------------------------------------------------------------

class TestWorktreeHasProtectedChanges:
    @patch("mother.worktree.subprocess.run")
    def test_detects_protected_file(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="mother/context.py\nmother/sandbox.py\n",
        )
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt)
        assert "mother/context.py" in result

    @patch("mother.worktree.subprocess.run")
    def test_no_protected_changes(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="mother/sandbox.py\nmother/worktree.py\n",
        )
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt)
        assert result == []

    @patch("mother.worktree.subprocess.run")
    def test_empty_diff(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt)
        assert result == []

    @patch("mother.worktree.subprocess.run")
    def test_multiple_protected_files(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="mother/context.py\nmother/persona.py\nmother/senses.py\n",
        )
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt)
        assert len(result) == 3

    @patch("mother.worktree.subprocess.run", side_effect=Exception("fail"))
    def test_returns_empty_on_exception(self, mock_run):
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt)
        assert result == []

    @patch("mother.worktree.subprocess.run")
    def test_custom_protected_list(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="my/special.py\nother.py\n",
        )
        wt = BuildWorktree(path="/tmp/wt", branch="build-abc", base_commit="abc")
        result = worktree_has_protected_changes(wt, protected=("my/special.py",))
        assert result == ["my/special.py"]
