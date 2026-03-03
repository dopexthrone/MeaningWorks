"""Tests for mother/project_publisher.py — project publishing to GitHub."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mother.project_publisher import (
    GitIdentity,
    PublishResult,
    _count_files,
    generate_readme,
    publish_project,
    sanitize_repo_name,
    set_git_identity,
)


# --- sanitize_repo_name ---


class TestSanitizeRepoName:
    def test_simple_name(self):
        assert sanitize_repo_name("my project") == "my-project"

    def test_special_chars(self):
        assert sanitize_repo_name("My @Project! #1") == "my-project-1"

    def test_underscores(self):
        assert sanitize_repo_name("my_cool_project") == "my-cool-project"

    def test_multiple_spaces(self):
        assert sanitize_repo_name("my   project") == "my-project"

    def test_leading_trailing_hyphens(self):
        assert sanitize_repo_name("--my-project--") == "my-project"

    def test_empty_string(self):
        assert sanitize_repo_name("") == "project"

    def test_all_special_chars(self):
        assert sanitize_repo_name("@#$%^&") == "project"

    def test_unicode(self):
        # Unicode chars stripped, only ascii remains
        result = sanitize_repo_name("café-app")
        assert "caf" in result
        assert "-app" in result

    def test_long_name_truncated(self):
        long_name = "a" * 200
        result = sanitize_repo_name(long_name)
        assert len(result) <= 100

    def test_dots_preserved(self):
        assert sanitize_repo_name("my.project.v2") == "my.project.v2"

    def test_leading_dots_stripped(self):
        assert sanitize_repo_name(".hidden") == "hidden"

    def test_case_preserved_lowercase(self):
        assert sanitize_repo_name("MyProject") == "myproject"

    def test_mixed_separators(self):
        assert sanitize_repo_name("my_project name-v2") == "my-project-name-v2"


# --- generate_readme ---


class TestGenerateReadme:
    def test_basic_readme(self):
        readme = generate_readme("Test App", "A test application")
        assert "# Test App" in readme
        assert "A test application" in readme
        assert "Mother" in readme

    def test_with_components_and_trust(self):
        readme = generate_readme("App", "Desc", components=12, trust=87.5)
        assert "12 components" in readme
        assert "88% trust" in readme  # rounded

    def test_no_description(self):
        readme = generate_readme("App", "")
        assert "# App" in readme

    def test_zero_metrics_omitted(self):
        readme = generate_readme("App", "Desc", components=0, trust=0)
        assert "components" not in readme
        assert "trust" not in readme


# --- GitIdentity ---


class TestGitIdentity:
    def test_defaults(self):
        identity = GitIdentity()
        assert identity.name == "Mother"
        assert identity.email == "mother@motherlabs.ai"

    def test_custom(self):
        identity = GitIdentity(name="Bot", email="bot@example.com")
        assert identity.name == "Bot"
        assert identity.email == "bot@example.com"

    def test_frozen(self):
        identity = GitIdentity()
        with pytest.raises(AttributeError):
            identity.name = "Other"


# --- set_git_identity ---


class TestSetGitIdentity:
    @patch("mother.project_publisher._run_git")
    def test_sets_name_and_email(self, mock_git):
        mock_git.return_value = (True, "")
        identity = GitIdentity(name="Test", email="test@test.com")
        result = set_git_identity("/tmp/repo", identity)
        assert result is True
        assert mock_git.call_count == 2

    @patch("mother.project_publisher._run_git")
    def test_returns_false_on_failure(self, mock_git):
        mock_git.return_value = (False, "error")
        identity = GitIdentity()
        result = set_git_identity("/tmp/repo", identity)
        assert result is False


# --- _count_files ---


class TestCountFiles:
    def test_counts_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("z")
        assert _count_files(str(tmp_path)) == 3

    def test_excludes_git(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref")
        assert _count_files(str(tmp_path)) == 1

    def test_empty_dir(self, tmp_path):
        assert _count_files(str(tmp_path)) == 0


# --- publish_project ---


class TestPublishProject:
    def test_missing_directory(self):
        result = publish_project("/nonexistent/path", "test")
        assert result.success is False
        assert "does not exist" in result.error

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = publish_project(str(f), "test")
        assert result.success is False
        assert "Not a directory" in result.error

    def test_empty_directory(self, tmp_path):
        result = publish_project(str(tmp_path), "test")
        assert result.success is False
        assert "empty" in result.error

    @patch("mother.project_publisher._run_git")
    @patch("mother.github.create_repo")
    def test_create_repo_failure(self, mock_create, mock_git, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        from mother.github import GitHubResult
        mock_create.return_value = GitHubResult(
            success=False, error="repo exists"
        )
        result = publish_project(str(tmp_path), "test")
        assert result.success is False
        assert "Failed to create repo" in result.error

    @patch("mother.project_publisher._run_git")
    @patch("mother.github.create_repo")
    def test_happy_path(self, mock_create, mock_git, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        from mother.github import GitHubResult
        mock_create.return_value = GitHubResult(
            success=True,
            operation="repo create",
            output="created",
            url="https://github.com/user/test-project",
        )
        # Mock git calls: rev-parse (not a repo), init, config x2, branch, add, commit, rev-parse, remote remove, remote add, push
        mock_git.side_effect = [
            (False, "not a repo"),   # rev-parse --git-dir
            (True, ""),              # init
            (True, ""),              # config user.name
            (True, ""),              # config user.email
            (True, ""),              # branch -M main
            (True, ""),              # add -A
            (True, ""),              # commit
            (True, "abc123"),        # rev-parse HEAD
            (True, ""),              # remote remove origin
            (True, ""),              # remote add origin
            (True, ""),              # push
        ]

        result = publish_project(str(tmp_path), "Test Project")
        assert result.success is True
        assert result.repo_url == "https://github.com/user/test-project"
        assert result.repo_name == "test-project"
        assert result.commit_hash == "abc123"
        assert result.files_pushed >= 1
        # README should have been generated
        assert (tmp_path / "README.md").exists()

    @patch("mother.project_publisher._run_git")
    @patch("mother.github.create_repo")
    def test_existing_readme_not_overwritten(self, mock_create, mock_git, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("Custom README")
        from mother.github import GitHubResult
        mock_create.return_value = GitHubResult(
            success=True, url="https://github.com/user/test"
        )
        mock_git.return_value = (True, "abc")

        publish_project(str(tmp_path), "test")
        assert (tmp_path / "README.md").read_text() == "Custom README"

    @patch("mother.project_publisher._run_git")
    @patch("mother.github.create_repo")
    def test_push_failure(self, mock_create, mock_git, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        from mother.github import GitHubResult
        mock_create.return_value = GitHubResult(
            success=True, url="https://github.com/user/test"
        )
        # Everything succeeds until push
        calls = [
            (True, ""),              # rev-parse --git-dir (already a repo)
            (True, ""),              # config user.name
            (True, ""),              # config user.email
            (True, ""),              # branch -M main
            (True, ""),              # add -A
            (True, ""),              # commit
            (True, "abc123"),        # rev-parse HEAD
            (True, ""),              # remote remove origin
            (True, ""),              # remote add origin
            (False, "auth failed"), # push fails
        ]
        mock_git.side_effect = calls

        result = publish_project(str(tmp_path), "test")
        assert result.success is False
        assert "push failed" in result.error

    def test_publish_result_frozen(self):
        r = PublishResult(success=True)
        with pytest.raises(AttributeError):
            r.success = False

    @patch("mother.project_publisher._run_git")
    @patch("mother.github.create_repo")
    def test_custom_identity(self, mock_create, mock_git, tmp_path):
        (tmp_path / "app.py").write_text("x")
        from mother.github import GitHubResult
        mock_create.return_value = GitHubResult(
            success=True, url="https://github.com/user/test"
        )
        mock_git.return_value = (True, "hash")

        identity = GitIdentity(name="Bot", email="bot@bot.com")
        publish_project(str(tmp_path), "test", identity=identity)

        # Check that git config was called with custom identity
        config_calls = [c for c in mock_git.call_args_list if "config" in c[0][0]]
        assert len(config_calls) == 2
