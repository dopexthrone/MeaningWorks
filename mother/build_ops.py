"""
Autonomous build operations — commit messages, push, build journal, safety tags.

LEAF module. Stdlib only. No imports from core/ or mother/.

Provides the post-build operations that close the autonomy loop:
- Meaningful commit messages from git diffs
- Auto-push to remote
- In-repo build journal (human-readable)
- Safety tags before protected file changes
"""

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.build_ops")

# Protected files that get safety-tagged before modification
PROTECTED_FILES = frozenset({
    "mother/context.py",
    "mother/persona.py",
    "mother/senses.py",
})

BUILD_JOURNAL_PATH = "BUILD_JOURNAL.md"


def _clean_env() -> dict:
    """Return os.environ without CLAUDECODE to allow nested git."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


def _run_git(args: List[str], cwd: str, timeout: float = 30.0) -> Tuple[bool, str]:
    """Run a git command. Returns (success, output)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_clean_env(),
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return False, err or f"Exit {result.returncode}"
        return True, output
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout:.0f}s"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# 1. Meaningful commit messages
# ---------------------------------------------------------------------------

def generate_commit_message(
    repo_dir: str,
    snapshot_hash: str,
    build_prompt: str = "",
    staged: bool = False,
) -> str:
    """Generate a meaningful commit message from the diff since snapshot.

    Deterministic — no LLM. Analyzes the diff to produce a structured message.
    Falls back to the old mechanical format if diff analysis fails.

    If staged=True, analyzes the staged (cached) diff instead of snapshot..HEAD.
    This is used when generating the message before the commit happens.
    """
    if not snapshot_hash and not staged:
        return f"self-build: {build_prompt[:80]}"

    # Get changed files
    if staged:
        diff_args = ["diff", "--cached", "--name-status"]
        stat_args = ["diff", "--cached", "--numstat"]
    else:
        diff_args = ["diff", "--name-status", f"{snapshot_hash}..HEAD"]
        stat_args = ["diff", "--numstat", f"{snapshot_hash}..HEAD"]

    ok, name_status = _run_git(diff_args, cwd=repo_dir)
    if not ok or not name_status:
        return f"self-build: {build_prompt[:80]}"

    added_files: List[str] = []
    modified_files: List[str] = []
    deleted_files: List[str] = []

    for line in name_status.splitlines():
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        status, filepath = parts[0].strip(), parts[1].strip()
        if status.startswith("A"):
            added_files.append(filepath)
        elif status.startswith("M"):
            modified_files.append(filepath)
        elif status.startswith("D"):
            deleted_files.append(filepath)

    # Get line stats
    ok, numstat = _run_git(stat_args, cwd=repo_dir)
    lines_added = 0
    lines_removed = 0
    if ok and numstat:
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    lines_added += int(parts[0]) if parts[0] != "-" else 0
                    lines_removed += int(parts[1]) if parts[1] != "-" else 0
                except ValueError:
                    continue

    # Classify the change
    all_files = added_files + modified_files + deleted_files
    is_test_only = all(f.startswith("tests/") for f in all_files)
    has_new_module = any(
        f.startswith(("mother/", "kernel/", "core/")) and f.endswith(".py")
        for f in added_files
    )
    touches_protected = any(f in PROTECTED_FILES for f in modified_files)

    # Extract modules touched
    modules = set()
    for f in all_files:
        if "/" in f:
            modules.add(f.split("/")[0])

    # Build subject line
    if is_test_only:
        subject = "test: add/update tests"
    elif has_new_module:
        new_names = [Path(f).stem for f in added_files if f.endswith(".py")]
        subject = f"feat: add {', '.join(new_names[:3])}"
    elif touches_protected:
        protected_names = [Path(f).stem for f in modified_files if f in PROTECTED_FILES]
        subject = f"persona: update {', '.join(protected_names)}"
    elif deleted_files and not added_files and not modified_files:
        subject = f"cleanup: remove {len(deleted_files)} files"
    elif len(modified_files) == 1:
        subject = f"fix: update {modified_files[0]}"
    elif len(all_files) <= 3:
        subject = f"update: {', '.join(Path(f).name for f in all_files[:3])}"
    else:
        subject = f"self-build: {len(all_files)} files across {', '.join(sorted(modules))}"

    # Build body
    body_parts = []
    if build_prompt:
        # Clean prompt — take first meaningful line
        prompt_line = build_prompt.strip().split("\n")[0][:120]
        body_parts.append(f"Goal: {prompt_line}")

    stat_line = f"+{lines_added}/-{lines_removed}"
    file_counts = []
    if added_files:
        file_counts.append(f"{len(added_files)} added")
    if modified_files:
        file_counts.append(f"{len(modified_files)} modified")
    if deleted_files:
        file_counts.append(f"{len(deleted_files)} deleted")
    body_parts.append(f"Files: {', '.join(file_counts)} ({stat_line})")

    if touches_protected:
        body_parts.append("PROTECTED FILES MODIFIED — founder-authorized change")

    body = "\n".join(body_parts)
    return f"{subject}\n\n{body}"


def commit_with_message(
    repo_dir: str,
    snapshot_hash: str,
    build_prompt: str = "",
) -> Tuple[bool, str]:
    """Stage all changes and commit with a meaningful message.

    Returns (success, commit_hash_or_error).
    """
    # Stage
    ok, err = _run_git(["add", "-A"], cwd=repo_dir)
    if not ok:
        return False, f"git add failed: {err}"

    # Check if there's anything to commit
    ok, status = _run_git(["diff", "--cached", "--quiet"], cwd=repo_dir)
    if ok:
        # Exit 0 means no staged changes
        return True, ""  # Nothing to commit, not an error

    # Generate message from staged diff (changes are staged but not yet committed)
    message = generate_commit_message(repo_dir, snapshot_hash, build_prompt, staged=True)

    # Commit
    ok, output = _run_git(["commit", "-m", message], cwd=repo_dir)
    if not ok:
        return False, f"git commit failed: {output}"

    # Get commit hash
    ok, commit_hash = _run_git(["rev-parse", "HEAD"], cwd=repo_dir, timeout=10)
    if ok:
        return True, commit_hash
    return True, ""


# ---------------------------------------------------------------------------
# 2. Auto-push
# ---------------------------------------------------------------------------

def push_after_build(
    repo_dir: str,
    timeout: float = 60.0,
) -> Tuple[bool, str]:
    """Push to remote after a successful build.

    Returns (success, output_or_error).
    """
    # Check if remote exists
    ok, remotes = _run_git(["remote"], cwd=repo_dir)
    if not ok or not remotes.strip():
        return False, "No git remote configured"

    # Get current branch
    ok, branch = _run_git(["branch", "--show-current"], cwd=repo_dir)
    if not ok or not branch:
        return False, "Could not determine current branch"

    # Push
    ok, output = _run_git(
        ["push", "origin", branch],
        cwd=repo_dir,
        timeout=timeout,
    )
    if ok:
        logger.info(f"Auto-push: {branch} pushed to origin")
        return True, output
    else:
        logger.warning(f"Auto-push failed: {output}")
        return False, output


# ---------------------------------------------------------------------------
# 3. Build journal (in-repo, human-readable)
# ---------------------------------------------------------------------------

@dataclass
class BuildJournalEntry:
    """One entry in the build journal."""
    timestamp: str
    goal: str
    success: bool
    files_changed: int
    lines_added: int
    lines_removed: int
    cost_usd: float
    duration_seconds: float
    commit_hash: str = ""
    provider: str = ""
    error: str = ""


def format_journal_entry(entry: BuildJournalEntry) -> str:
    """Format a single journal entry as markdown."""
    status = "SUCCESS" if entry.success else "FAILED"
    lines = [
        f"### [{entry.timestamp}] {status}",
        f"**Goal:** {entry.goal}",
    ]
    if entry.success:
        lines.append(
            f"**Changes:** {entry.files_changed} files, "
            f"+{entry.lines_added}/-{entry.lines_removed}"
        )
        if entry.commit_hash:
            lines.append(f"**Commit:** `{entry.commit_hash[:10]}`")
    else:
        lines.append(f"**Error:** {entry.error}")

    lines.append(
        f"**Cost:** ${entry.cost_usd:.3f} | "
        f"**Duration:** {entry.duration_seconds:.0f}s | "
        f"**Provider:** {entry.provider or 'unknown'}"
    )
    lines.append("")  # blank line separator
    return "\n".join(lines)


def append_to_build_journal(
    repo_dir: str,
    entry: BuildJournalEntry,
) -> bool:
    """Append a build journal entry to BUILD_JOURNAL.md in the repo.

    Creates the file with a header if it doesn't exist.
    Returns True on success.
    """
    journal_path = Path(repo_dir) / BUILD_JOURNAL_PATH

    try:
        if not journal_path.exists():
            header = (
                "# Build Journal\n\n"
                "Autonomous build log — each entry records a self-modification cycle.\n"
                "Most recent builds appear at the top.\n\n"
                "---\n\n"
            )
            journal_path.write_text(header, encoding="utf-8")

        # Read existing content to prepend (most recent first)
        existing = journal_path.read_text(encoding="utf-8")

        # Find the insertion point (after the header/separator)
        separator = "---\n\n"
        sep_idx = existing.find(separator)
        if sep_idx >= 0:
            before = existing[: sep_idx + len(separator)]
            after = existing[sep_idx + len(separator):]
        else:
            before = existing
            after = ""

        entry_text = format_journal_entry(entry)
        new_content = before + entry_text + "\n" + after

        journal_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Build journal updated: {entry.timestamp}")
        return True

    except Exception as e:
        logger.warning(f"Build journal write failed: {e}")
        return False


def build_journal_entry_from_result(
    build_result: dict,
    build_prompt: str,
    snapshot_hash: str = "",
    repo_dir: str = "",
) -> BuildJournalEntry:
    """Create a BuildJournalEntry from a self_build() result dict."""
    # Get diff stats if we have snapshot
    files_changed = 0
    lines_added = 0
    lines_removed = 0

    diff_stats = build_result.get("diff_stats", {})
    if diff_stats:
        files_changed = diff_stats.get("files_modified", 0) + diff_stats.get("files_added", 0)
        lines_added = diff_stats.get("lines_added", 0)
        lines_removed = diff_stats.get("lines_removed", 0)
    elif snapshot_hash and repo_dir and build_result.get("success"):
        # Compute from git
        ok, numstat = _run_git(
            ["diff", "--numstat", f"{snapshot_hash}..HEAD"], cwd=repo_dir
        )
        if ok and numstat:
            for line in numstat.splitlines():
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        lines_added += int(parts[0]) if parts[0] != "-" else 0
                        lines_removed += int(parts[1]) if parts[1] != "-" else 0
                        files_changed += 1
                    except ValueError:
                        continue

    # Get commit hash
    commit_hash = ""
    if repo_dir and build_result.get("success"):
        ok, h = _run_git(["rev-parse", "HEAD"], cwd=repo_dir, timeout=10)
        if ok:
            commit_hash = h

    # Clean prompt for display
    goal = build_prompt.strip().split("\n")[0][:200]
    # Strip [SELF-IMPROVEMENT] prefix if present
    goal = re.sub(r'^\[SELF-IMPROVEMENT\]\s*', '', goal)

    return BuildJournalEntry(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        goal=goal,
        success=build_result.get("success", False),
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        cost_usd=build_result.get("cost_usd", 0.0),
        duration_seconds=build_result.get("duration_seconds", 0.0),
        commit_hash=commit_hash,
        provider=build_result.get("provider", ""),
        error=build_result.get("error", "")[:200] if not build_result.get("success") else "",
    )


# ---------------------------------------------------------------------------
# 4. Safety tags before protected file changes
# ---------------------------------------------------------------------------

def tag_before_protected_change(
    repo_dir: str,
    changed_files: List[str],
) -> Optional[str]:
    """Create a git tag if any protected files are being modified.

    Tag format: pre-protected-YYYYMMDD-HHMMSS
    Returns the tag name if created, None if no protected files touched.
    """
    protected_touched = [f for f in changed_files if f in PROTECTED_FILES]
    if not protected_touched:
        return None

    tag_name = f"pre-protected-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    message = f"Safety tag before modifying: {', '.join(protected_touched)}"

    ok, output = _run_git(
        ["tag", "-a", tag_name, "-m", message],
        cwd=repo_dir,
    )

    if ok:
        logger.info(f"Safety tag created: {tag_name} (files: {protected_touched})")
        return tag_name
    else:
        logger.warning(f"Safety tag failed: {output}")
        return None


def detect_protected_changes(
    repo_dir: str,
    snapshot_hash: str,
) -> List[str]:
    """Check if any protected files were changed since snapshot.

    Returns list of protected file paths that were modified.
    """
    if not snapshot_hash:
        return []

    ok, output = _run_git(
        ["diff", "--name-only", f"{snapshot_hash}..HEAD"],
        cwd=repo_dir,
    )
    if not ok or not output:
        return []

    changed = output.splitlines()
    return [f for f in changed if f in PROTECTED_FILES]
