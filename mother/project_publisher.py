"""
Project publishing — create GitHub repos for emitted projects and push code.

LEAF module. Stdlib only. No imports from core/ or mother/.

Closes the gap between "Mother builds" and "Mother ships publicly."
Takes an emitted project directory, creates a GitHub repo, inits git
with Mother's identity, commits, pushes, and returns the URL.
"""

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.project_publisher")


@dataclass(frozen=True)
class PublishResult:
    """Outcome of a project publish operation."""

    success: bool
    repo_url: str = ""
    repo_name: str = ""
    commit_hash: str = ""
    files_pushed: int = 0
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class GitIdentity:
    """Git author identity for published projects."""

    name: str = "Mother"
    email: str = "mother@motherlabs.ai"


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


def sanitize_repo_name(name: str) -> str:
    """Sanitize a project name for use as a GitHub repository name.

    Rules: lowercase, hyphens for separators, no special chars, max 100 chars.
    """
    if not name:
        return "project"

    # Lowercase
    result = name.lower().strip()

    # Replace spaces and underscores with hyphens
    result = re.sub(r"[\s_]+", "-", result)

    # Remove everything except alphanumeric, hyphens, dots
    result = re.sub(r"[^a-z0-9.\-]", "", result)

    # Collapse multiple hyphens
    result = re.sub(r"-{2,}", "-", result)

    # Strip leading/trailing hyphens and dots
    result = result.strip("-.")

    # Truncate
    if len(result) > 100:
        result = result[:100].rstrip("-.")

    return result or "project"


def set_git_identity(repo_dir: str, identity: GitIdentity) -> bool:
    """Set git user.name and user.email LOCAL to a repo (not global)."""
    ok1, _ = _run_git(["config", "user.name", identity.name], cwd=repo_dir)
    ok2, _ = _run_git(["config", "user.email", identity.email], cwd=repo_dir)
    return ok1 and ok2


def generate_readme(
    name: str,
    description: str,
    components: int = 0,
    trust: float = 0.0,
) -> str:
    """Generate a simple README.md for an emitted project."""
    lines = [f"# {name}", ""]

    if description:
        lines.append(description)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Built by [Mother](https://motherlabs.ai) — a semantic compiler.")

    if components > 0 or trust > 0:
        lines.append("")
        parts = []
        if components > 0:
            parts.append(f"**{components} components**")
        if trust > 0:
            parts.append(f"**{trust:.0f}% trust**")
        lines.append(" | ".join(parts))

    lines.append("")
    return "\n".join(lines)


def _count_files(directory: str) -> int:
    """Count files in a directory (non-recursive excludes .git)."""
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip .git directory
        dirs[:] = [d for d in dirs if d != ".git"]
        count += len(files)
    return count


def publish_project(
    project_dir: str,
    name: str,
    description: str = "",
    public: bool = False,
    identity: Optional[GitIdentity] = None,
    readme_components: int = 0,
    readme_trust: float = 0.0,
    timeout: float = 60.0,
) -> PublishResult:
    """Publish an emitted project to GitHub.

    1. Validates project_dir exists and has files
    2. Sanitizes name for GitHub
    3. Creates repo via gh CLI
    4. Inits git with Mother's identity (local config)
    5. Generates README if none exists
    6. Commits and pushes

    Args:
        project_dir: Path to the emitted project directory
        name: Human-readable project name
        description: Project description
        public: If True, create a public repo (default: private)
        identity: Git identity to use (default: Mother)
        readme_components: Component count for README badge
        readme_trust: Trust score for README badge
        timeout: Timeout for git/gh operations

    Returns:
        PublishResult with repo URL and metadata
    """
    start = time.monotonic()
    identity = identity or GitIdentity()

    # 1. Validate project directory
    project_path = Path(project_dir)
    if not project_path.exists():
        return PublishResult(
            success=False,
            error=f"Project directory does not exist: {project_dir}",
            duration_seconds=time.monotonic() - start,
        )
    if not project_path.is_dir():
        return PublishResult(
            success=False,
            error=f"Not a directory: {project_dir}",
            duration_seconds=time.monotonic() - start,
        )

    file_count = _count_files(project_dir)
    if file_count == 0:
        return PublishResult(
            success=False,
            error="Project directory is empty",
            duration_seconds=time.monotonic() - start,
        )

    # 2. Sanitize repo name
    repo_name = sanitize_repo_name(name)

    # 3. Create GitHub repo via gh CLI
    from mother.github import create_repo

    gh_result = create_repo(
        repo_name,
        description=description[:350] if description else "",
        public=public,
        timeout=timeout,
    )

    if not gh_result.success:
        return PublishResult(
            success=False,
            repo_name=repo_name,
            error=f"Failed to create repo: {gh_result.error}",
            duration_seconds=time.monotonic() - start,
        )

    repo_url = gh_result.url or ""

    # 4. Init git in project directory
    # Check if already a git repo
    ok, _ = _run_git(["rev-parse", "--git-dir"], cwd=project_dir)
    if not ok:
        ok, err = _run_git(["init"], cwd=project_dir)
        if not ok:
            return PublishResult(
                success=False,
                repo_name=repo_name,
                repo_url=repo_url,
                error=f"git init failed: {err}",
                duration_seconds=time.monotonic() - start,
            )

    # Set Mother's identity (LOCAL to this repo)
    set_git_identity(project_dir, identity)

    # Set default branch to main
    _run_git(["branch", "-M", "main"], cwd=project_dir)

    # 5. Generate README if none exists
    readme_path = project_path / "README.md"
    if not readme_path.exists():
        readme_content = generate_readme(
            name, description, readme_components, readme_trust
        )
        readme_path.write_text(readme_content, encoding="utf-8")

    # 6. Add, commit, push
    ok, err = _run_git(["add", "-A"], cwd=project_dir)
    if not ok:
        return PublishResult(
            success=False,
            repo_name=repo_name,
            repo_url=repo_url,
            error=f"git add failed: {err}",
            duration_seconds=time.monotonic() - start,
        )

    ok, err = _run_git(
        ["commit", "-m", f"Initial commit — built by {identity.name}"],
        cwd=project_dir,
    )
    if not ok:
        return PublishResult(
            success=False,
            repo_name=repo_name,
            repo_url=repo_url,
            error=f"git commit failed: {err}",
            duration_seconds=time.monotonic() - start,
        )

    # Get commit hash
    ok, commit_hash = _run_git(["rev-parse", "HEAD"], cwd=project_dir, timeout=10)
    if not ok:
        commit_hash = ""

    # Add remote and push
    if repo_url:
        git_url = repo_url + ".git" if not repo_url.endswith(".git") else repo_url
        _run_git(["remote", "remove", "origin"], cwd=project_dir)  # idempotent
        ok, err = _run_git(["remote", "add", "origin", git_url], cwd=project_dir)
        if not ok:
            return PublishResult(
                success=False,
                repo_name=repo_name,
                repo_url=repo_url,
                commit_hash=commit_hash,
                files_pushed=file_count,
                error=f"git remote add failed: {err}",
                duration_seconds=time.monotonic() - start,
            )

    ok, err = _run_git(
        ["push", "-u", "origin", "main"],
        cwd=project_dir,
        timeout=timeout,
    )
    if not ok:
        return PublishResult(
            success=False,
            repo_name=repo_name,
            repo_url=repo_url,
            commit_hash=commit_hash,
            files_pushed=file_count,
            error=f"git push failed: {err}",
            duration_seconds=time.monotonic() - start,
        )

    logger.info(f"Published project: {repo_name} -> {repo_url}")

    return PublishResult(
        success=True,
        repo_url=repo_url,
        repo_name=repo_name,
        commit_hash=commit_hash,
        files_pushed=file_count,
        duration_seconds=time.monotonic() - start,
    )
