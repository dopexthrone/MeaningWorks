"""
Git worktree lifecycle for isolated self-build execution.

LEAF module. Stdlib only. No imports from core/ or mother/.

Creates temporary git worktrees under .claude/worktrees/ for each build.
Changes only reach the main repo after tests pass and the worktree merges
cleanly. Failed builds are disposed of without touching main.
"""

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.worktree")


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildWorktree:
    """Represents an active build worktree."""
    path: str           # absolute path to worktree directory
    branch: str         # build-<uuid> branch name
    base_commit: str    # HEAD commit at creation time


# ---------------------------------------------------------------------------
# Environment helper
# ---------------------------------------------------------------------------

def _git_env() -> dict:
    """Clean environment for git commands — strip CLAUDECODE if present."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    # Strip ANTHROPIC_API_KEY to force Max subscription billing
    env.pop("ANTHROPIC_API_KEY", None)
    return env


# ---------------------------------------------------------------------------
# Worktree lifecycle
# ---------------------------------------------------------------------------

def worktree_create(repo_dir: str) -> BuildWorktree:
    """Create a new git worktree for an isolated build.

    Creates: .claude/worktrees/build-<uuid>/
    Symlinks: .venv → main repo .venv for read access.
    Creates the build branch from current HEAD.

    Raises RuntimeError if git worktree creation fails.
    """
    build_id = uuid.uuid4().hex[:12]
    branch_name = f"build-{build_id}"
    worktree_base = os.path.join(repo_dir, ".claude", "worktrees")
    worktree_path = os.path.join(worktree_base, branch_name)

    env = _git_env()

    # Get current HEAD
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir, capture_output=True, text=True, timeout=10, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get HEAD: {result.stderr.strip()}")
    base_commit = result.stdout.strip()

    # Ensure base directory exists
    os.makedirs(worktree_base, exist_ok=True)

    # Create worktree with new branch
    result = subprocess.run(
        ["git", "worktree", "add", worktree_path, "-b", branch_name],
        cwd=repo_dir, capture_output=True, text=True, timeout=30, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")

    # Symlink .venv from main repo for read access
    main_venv = os.path.join(repo_dir, ".venv")
    wt_venv = os.path.join(worktree_path, ".venv")
    if os.path.isdir(main_venv) and not os.path.exists(wt_venv):
        os.symlink(os.path.realpath(main_venv), wt_venv)
        logger.debug(f"Symlinked .venv: {wt_venv} → {main_venv}")

    logger.info(f"Created build worktree: {worktree_path} (branch={branch_name})")
    return BuildWorktree(
        path=worktree_path,
        branch=branch_name,
        base_commit=base_commit,
    )


def worktree_merge(repo_dir: str, wt: BuildWorktree) -> Tuple[bool, str]:
    """Merge worktree changes back to the main branch via squash merge.

    Checkout main, merge --squash the build branch, return (success, message).
    Does NOT commit — caller handles commit with meaningful message.

    Returns (True, "") on success, (False, error_message) on failure.
    """
    env = _git_env()

    # Ensure we're on the main branch
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=repo_dir, capture_output=True, text=True, timeout=10, env=env,
    )
    current_branch = result.stdout.strip() if result.returncode == 0 else ""

    if current_branch != "main" and current_branch != "master":
        # Switch to main
        main_branch = "main"
        # Check if main exists
        check = subprocess.run(
            ["git", "rev-parse", "--verify", "main"],
            cwd=repo_dir, capture_output=True, timeout=10, env=env,
        )
        if check.returncode != 0:
            main_branch = "master"

        result = subprocess.run(
            ["git", "checkout", main_branch],
            cwd=repo_dir, capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            return False, f"Failed to checkout {main_branch}: {result.stderr.strip()}"

    # Squash merge the build branch
    result = subprocess.run(
        ["git", "merge", "--squash", wt.branch],
        cwd=repo_dir, capture_output=True, text=True, timeout=60, env=env,
    )
    if result.returncode != 0:
        # Reset any partial merge state
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo_dir, capture_output=True, timeout=30, env=env,
        )
        return False, f"Merge conflict or failure: {result.stderr.strip()}"

    return True, ""


def worktree_remove(repo_dir: str, wt: BuildWorktree) -> None:
    """Remove a build worktree and its branch. Safe if already gone."""
    env = _git_env()

    # Ensure we're not on the build branch before removing
    try:
        result = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=10, env=env,
        )
        if result.returncode == 0 and result.stdout.strip() == wt.branch:
            subprocess.run(
                ["git", "checkout", "main"],
                cwd=repo_dir, capture_output=True, timeout=30, env=env,
            )
    except Exception:
        pass

    # Remove worktree
    try:
        subprocess.run(
            ["git", "worktree", "remove", wt.path, "--force"],
            cwd=repo_dir, capture_output=True, timeout=30, env=env,
        )
    except Exception as e:
        logger.debug(f"Worktree remove failed (may already be gone): {e}")

    # Delete the branch
    try:
        subprocess.run(
            ["git", "branch", "-D", wt.branch],
            cwd=repo_dir, capture_output=True, timeout=10, env=env,
        )
    except Exception as e:
        logger.debug(f"Branch delete failed (may already be gone): {e}")

    # Prune stale worktree references
    try:
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_dir, capture_output=True, timeout=10, env=env,
        )
    except Exception:
        pass

    logger.info(f"Removed build worktree: {wt.path}")


def worktree_diff_summary(wt: BuildWorktree) -> str:
    """Return git diff --stat from base_commit to current HEAD in the worktree."""
    env = _git_env()
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", f"{wt.base_commit}..HEAD"],
            cwd=wt.path, capture_output=True, text=True, timeout=30, env=env,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def worktree_has_protected_changes(
    wt: BuildWorktree,
    protected: Tuple[str, ...] = (
        "mother/context.py",
        "mother/persona.py",
        "mother/senses.py",
    ),
) -> List[str]:
    """Return list of protected files modified in the worktree.

    Checks git diff --name-only from base_commit to HEAD.
    """
    env = _git_env()
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{wt.base_commit}..HEAD"],
            cwd=wt.path, capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            return []

        changed = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        return [p for p in protected if p in changed]
    except Exception:
        return []
