"""
Tests for mother/github.py -- LEAF module.

Covers: GitHubResult frozen dataclass, gh CLI subprocess wrappers,
push, create_repo, create_release, get_repo_info.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from mother.github import (
    GitHubResult,
    push_to_github,
    create_repo,
    create_release,
    get_repo_info,
)


class TestGitHubResult:
    def test_frozen(self):
        r = GitHubResult(success=True, operation="push")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = GitHubResult(success=True)
        assert r.operation == ""
        assert r.output == ""
        assert r.url == ""
        assert r.error is None


class TestPushToGitHub:
    @patch("mother.github.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="To github.com:user/repo\n   abc123..def456  main -> main",
            stderr="",
        )
        result = push_to_github("/fake/repo")
        assert result.success
        assert result.operation == "push"

    @patch("mother.github.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Permission denied",
        )
        result = push_to_github("/fake/repo")
        assert not result.success
        assert "Permission denied" in result.error

    @patch("mother.github.subprocess.run")
    def test_with_branch(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        push_to_github("/fake/repo", branch="feature")
        # Verify git push origin feature was called
        call_args = mock_run.call_args[0][0]
        assert "origin" in call_args
        assert "feature" in call_args


class TestCreateRepo:
    @patch("mother.github._run_gh")
    def test_extracts_url(self, mock_run_gh):
        mock_run_gh.return_value = GitHubResult(
            success=True,
            operation="repo create",
            output="✓ Created repository user/test-repo on GitHub\nhttps://github.com/user/test-repo",
        )
        result = create_repo("test-repo", description="Test", public=True)
        assert result.success
        assert result.url == "https://github.com/user/test-repo"

    @patch("mother.github._run_gh")
    def test_no_url_fallback(self, mock_run_gh):
        mock_run_gh.return_value = GitHubResult(
            success=True,
            operation="repo create",
            output="Created successfully",
        )
        result = create_repo("test-repo")
        assert result.success
        assert result.url == ""


class TestGetRepoInfo:
    @patch("mother.github._run_gh")
    def test_parses_json(self, mock_run_gh):
        mock_run_gh.return_value = GitHubResult(
            success=True,
            output=json.dumps({
                "nameWithOwner": "user/repo",
                "url": "https://github.com/user/repo",
            }),
        )
        info = get_repo_info("/fake/repo")
        assert info["name"] == "user/repo"
        assert info["url"] == "https://github.com/user/repo"

    @patch("mother.github._run_gh")
    def test_failure_returns_empty(self, mock_run_gh):
        mock_run_gh.return_value = GitHubResult(
            success=False,
            error="Not a git repository",
        )
        info = get_repo_info("/fake/repo")
        assert info == {}


class TestCreateRelease:
    @patch("mother.github._run_gh")
    def test_with_notes(self, mock_run_gh):
        mock_run_gh.return_value = GitHubResult(success=True, operation="release create")
        result = create_release("v1.0.0", title="Release 1.0", notes="Bug fixes")
        assert result.success
        # Verify args passed correctly
        call_args = mock_run_gh.call_args[0][0]
        assert "v1.0.0" in call_args
        assert "--title" in call_args
        assert "--notes" in call_args
