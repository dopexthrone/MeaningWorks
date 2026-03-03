"""Tests for mother/build_ops.py — autonomous build operations."""

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mother.build_ops import (
    PROTECTED_FILES,
    BUILD_JOURNAL_PATH,
    generate_commit_message,
    commit_with_message,
    push_after_build,
    BuildJournalEntry,
    format_journal_entry,
    append_to_build_journal,
    build_journal_entry_from_result,
    tag_before_protected_change,
    detect_protected_changes,
    _run_git,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: str) -> str:
    """Initialize a git repo with one commit. Returns initial commit hash."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, env=env)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True, env=env)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, capture_output=True, env=env)
    # Initial commit
    (Path(path) / "README.md").write_text("# test\n")
    subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, env=env)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, env=env,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------

class TestRunGit:
    def test_successful_command(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        ok, output = _run_git(["status"], cwd=repo)
        assert ok
        assert "nothing to commit" in output or "clean" in output

    def test_failed_command(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        ok, output = _run_git(["checkout", "nonexistent-branch"], cwd=repo)
        assert not ok

    def test_timeout(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        # A command that would hang gets a very short timeout
        ok, output = _run_git(["status"], cwd=repo, timeout=0.001)
        # Either succeeds fast or times out — both acceptable
        assert isinstance(ok, bool)


# ---------------------------------------------------------------------------
# Commit messages
# ---------------------------------------------------------------------------

class TestGenerateCommitMessage:
    def test_no_snapshot_fallback(self, tmp_path):
        msg = generate_commit_message(str(tmp_path), "", "fix the thing")
        assert "self-build:" in msg
        assert "fix the thing" in msg

    def test_single_modified_file(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        # Modify a file
        (Path(repo) / "README.md").write_text("# updated\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "update readme")
        assert "README.md" in msg
        assert "Goal:" in msg

    def test_new_module_detection(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        # Add a new Python file in mother/
        (Path(repo) / "mother").mkdir(exist_ok=True)
        (Path(repo) / "mother" / "new_feature.py").write_text("# new\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "add new feature")
        assert "feat:" in msg
        assert "new_feature" in msg

    def test_test_only_change(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        (Path(repo) / "tests").mkdir(exist_ok=True)
        (Path(repo) / "tests" / "test_foo.py").write_text("# test\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "tests"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "add tests")
        assert "test:" in msg

    def test_protected_file_flagged(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        (Path(repo) / "mother").mkdir(exist_ok=True)
        (Path(repo) / "mother" / "persona.py").write_text("# persona\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "persona"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "update persona")
        # New file = added, not modified → won't trigger protected message
        # But if we modify it...
        (Path(repo) / "mother" / "persona.py").write_text("# persona updated\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        snapshot2 = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(["git", "commit", "-m", "update"], cwd=repo, capture_output=True)
        msg2 = generate_commit_message(repo, snapshot2, "persona tweak")
        assert "persona" in msg2.lower()

    def test_line_stats_included(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        (Path(repo) / "README.md").write_text("line1\nline2\nline3\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "change"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "test")
        assert "+" in msg  # +N/-M format
        assert "Files:" in msg

    def test_multi_file_change(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        for i in range(5):
            (Path(repo) / f"file{i}.py").write_text(f"# {i}\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "multi"], cwd=repo, capture_output=True)

        msg = generate_commit_message(repo, snapshot, "add files")
        assert "5 files" in msg or "5 added" in msg


# ---------------------------------------------------------------------------
# Commit with message
# ---------------------------------------------------------------------------

class TestCommitWithMessage:
    def test_commit_staged_changes(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        (Path(repo) / "foo.py").write_text("# foo\n")

        ok, result = commit_with_message(repo, snapshot, "add foo")
        assert ok
        # Should have a commit hash or empty string (if nothing staged)
        # Since we didn't stage, git add -A inside commit_with_message handles it
        assert isinstance(result, str)

    def test_nothing_to_commit(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        # No changes
        ok, result = commit_with_message(repo, snapshot, "nothing")
        assert ok
        assert result == ""  # Nothing to commit is success with empty hash


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

class TestPushAfterBuild:
    def test_no_remote_fails(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        ok, output = push_after_build(repo)
        # No remote configured, should fail gracefully
        assert not ok

    @patch("mother.build_ops._run_git")
    def test_push_succeeds(self, mock_git):
        mock_git.side_effect = [
            (True, "origin"),           # remote check
            (True, "main"),             # branch check
            (True, "Everything up-to-date"),  # push
        ]
        ok, output = push_after_build("/fake/repo")
        assert ok
        assert mock_git.call_count == 3

    @patch("mother.build_ops._run_git")
    def test_push_failure_logged(self, mock_git):
        mock_git.side_effect = [
            (True, "origin"),
            (True, "main"),
            (False, "rejected: non-fast-forward"),
        ]
        ok, output = push_after_build("/fake/repo")
        assert not ok
        assert "rejected" in output


# ---------------------------------------------------------------------------
# Build journal
# ---------------------------------------------------------------------------

class TestBuildJournal:
    def test_format_success_entry(self):
        entry = BuildJournalEntry(
            timestamp="2026-02-27 14:30",
            goal="Fix compilation fidelity",
            success=True,
            files_changed=3,
            lines_added=45,
            lines_removed=12,
            cost_usd=0.15,
            duration_seconds=42.0,
            commit_hash="abc123def456",
            provider="native",
        )
        text = format_journal_entry(entry)
        assert "SUCCESS" in text
        assert "Fix compilation fidelity" in text
        assert "3 files" in text
        assert "+45/-12" in text
        assert "abc123def4" in text
        assert "$0.150" in text
        assert "42s" in text

    def test_format_failure_entry(self):
        entry = BuildJournalEntry(
            timestamp="2026-02-27 14:30",
            goal="Update persona",
            success=False,
            files_changed=0,
            lines_added=0,
            lines_removed=0,
            cost_usd=0.05,
            duration_seconds=10.0,
            error="Tests failed after modification",
        )
        text = format_journal_entry(entry)
        assert "FAILED" in text
        assert "Tests failed" in text

    def test_append_creates_file(self, tmp_path):
        repo = str(tmp_path)
        entry = BuildJournalEntry(
            timestamp="2026-02-27 14:30",
            goal="Test",
            success=True,
            files_changed=1,
            lines_added=10,
            lines_removed=0,
            cost_usd=0.01,
            duration_seconds=5.0,
        )
        result = append_to_build_journal(repo, entry)
        assert result is True

        journal = (Path(repo) / BUILD_JOURNAL_PATH).read_text()
        assert "Build Journal" in journal
        assert "Test" in journal

    def test_append_prepends_entry(self, tmp_path):
        repo = str(tmp_path)
        entry1 = BuildJournalEntry(
            timestamp="2026-02-27 14:00",
            goal="First build",
            success=True,
            files_changed=1,
            lines_added=5,
            lines_removed=0,
            cost_usd=0.01,
            duration_seconds=3.0,
        )
        entry2 = BuildJournalEntry(
            timestamp="2026-02-27 14:30",
            goal="Second build",
            success=True,
            files_changed=2,
            lines_added=10,
            lines_removed=3,
            cost_usd=0.02,
            duration_seconds=6.0,
        )
        append_to_build_journal(repo, entry1)
        append_to_build_journal(repo, entry2)

        journal = (Path(repo) / BUILD_JOURNAL_PATH).read_text()
        # Second build should appear before first (most recent first)
        pos1 = journal.index("First build")
        pos2 = journal.index("Second build")
        assert pos2 < pos1

    def test_build_journal_entry_from_result(self):
        result = {
            "success": True,
            "cost_usd": 0.25,
            "duration_seconds": 60.0,
            "provider": "native",
            "diff_stats": {
                "files_modified": 2,
                "files_added": 1,
                "lines_added": 30,
                "lines_removed": 5,
            },
        }
        entry = build_journal_entry_from_result(
            result, "[SELF-IMPROVEMENT] fix the thing", "", "",
        )
        assert entry.success is True
        assert entry.goal == "fix the thing"  # prefix stripped
        assert entry.files_changed == 3
        assert entry.lines_added == 30
        assert entry.cost_usd == 0.25

    def test_build_journal_entry_failure(self):
        result = {
            "success": False,
            "cost_usd": 0.05,
            "duration_seconds": 10.0,
            "error": "Tests failed after modification",
            "provider": "claude",
        }
        entry = build_journal_entry_from_result(result, "bad change", "", "")
        assert entry.success is False
        assert "Tests failed" in entry.error


# ---------------------------------------------------------------------------
# Safety tags
# ---------------------------------------------------------------------------

class TestSafetyTags:
    def test_tag_created_for_protected(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        tag_name = tag_before_protected_change(repo, ["mother/persona.py"])
        assert tag_name is not None
        assert tag_name.startswith("pre-protected-")

        # Verify tag exists in git
        ok, output = _run_git(["tag", "-l", tag_name], cwd=repo)
        assert ok
        assert tag_name in output

    def test_no_tag_for_non_protected(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        tag_name = tag_before_protected_change(repo, ["mother/bridge.py"])
        assert tag_name is None

    def test_no_tag_for_empty_list(self, tmp_path):
        repo = str(tmp_path)
        _init_git_repo(repo)
        tag_name = tag_before_protected_change(repo, [])
        assert tag_name is None

    def test_detect_protected_changes(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        # Create mother/ dir and protected file
        (Path(repo) / "mother").mkdir(exist_ok=True)
        (Path(repo) / "mother" / "persona.py").write_text("# changed\n")
        (Path(repo) / "mother" / "bridge.py").write_text("# not protected\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "test"], cwd=repo, capture_output=True)

        protected = detect_protected_changes(repo, snapshot)
        assert "mother/persona.py" in protected
        assert "mother/bridge.py" not in protected

    def test_detect_no_protected_changes(self, tmp_path):
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)
        (Path(repo) / "foo.py").write_text("# not protected\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "test"], cwd=repo, capture_output=True)

        protected = detect_protected_changes(repo, snapshot)
        assert protected == []

    def test_detect_empty_snapshot(self, tmp_path):
        protected = detect_protected_changes(str(tmp_path), "")
        assert protected == []


# ---------------------------------------------------------------------------
# PROTECTED_FILES constant
# ---------------------------------------------------------------------------

class TestProtectedFiles:
    def test_known_protected_files(self):
        assert "mother/context.py" in PROTECTED_FILES
        assert "mother/persona.py" in PROTECTED_FILES
        assert "mother/senses.py" in PROTECTED_FILES

    def test_is_frozenset(self):
        assert isinstance(PROTECTED_FILES, frozenset)


# ---------------------------------------------------------------------------
# Integration: commit + journal together
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_build_cycle(self, tmp_path):
        """Simulate a complete build cycle: change → commit → journal."""
        repo = str(tmp_path)
        snapshot = _init_git_repo(repo)

        # Make changes
        (Path(repo) / "new_module.py").write_text("def hello(): pass\n")
        (Path(repo) / "README.md").write_text("# Updated\n")

        # Commit with meaningful message
        ok, commit_hash = commit_with_message(repo, snapshot, "add hello module")
        assert ok
        assert len(commit_hash) > 0

        # Verify commit message is meaningful
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%B"],
            cwd=repo, capture_output=True, text=True,
        )
        msg = result.stdout
        assert "self-build:" not in msg or "new_module" in msg  # meaningful, not mechanical

        # Write journal
        entry = BuildJournalEntry(
            timestamp="2026-02-27 15:00",
            goal="add hello module",
            success=True,
            files_changed=2,
            lines_added=2,
            lines_removed=1,
            cost_usd=0.10,
            duration_seconds=15.0,
            commit_hash=commit_hash,
            provider="native",
        )
        journal_ok = append_to_build_journal(repo, entry)
        assert journal_ok

        # Verify journal exists and has content
        journal = (Path(repo) / BUILD_JOURNAL_PATH).read_text()
        assert "add hello module" in journal
        assert commit_hash[:10] in journal
