"""
GitHub integration — wrapper for gh CLI.

LEAF module. Stdlib only. No imports from core/ or mother/.

Enables Mother to:
- Push commits to GitHub
- Create repositories
- Manage issues and PRs
- Post releases
- Respond to feedback

Uses gh CLI subprocess calls. All operations return frozen dataclass results.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GitHubResult:
    """Outcome of a GitHub operation."""

    success: bool
    operation: str = ""
    output: str = ""
    url: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _run_gh(
    args: List[str],
    cwd: str = "",
    timeout: float = 30.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Run gh CLI command and return structured result."""
    start = time.monotonic()

    if not args:
        return GitHubResult(
            success=False,
            error="No command provided",
            duration_seconds=0.0,
        )

    cmd = [gh_path] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        elapsed = time.monotonic() - start

        if result.returncode != 0:
            return GitHubResult(
                success=False,
                operation=args[0] if args else "",
                output=result.stdout.strip(),
                error=result.stderr.strip() or f"Exit code {result.returncode}",
                duration_seconds=elapsed,
            )

        return GitHubResult(
            success=True,
            operation=args[0] if args else "",
            output=result.stdout.strip(),
            duration_seconds=elapsed,
        )

    except subprocess.TimeoutExpired:
        return GitHubResult(
            success=False,
            operation=args[0] if args else "",
            error=f"Timed out after {timeout:.0f}s",
            duration_seconds=time.monotonic() - start,
        )
    except FileNotFoundError:
        return GitHubResult(
            success=False,
            operation=args[0] if args else "",
            error=f"gh CLI not found at {gh_path}",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as e:
        return GitHubResult(
            success=False,
            operation=args[0] if args else "",
            error=str(e),
            duration_seconds=time.monotonic() - start,
        )


def push_to_github(
    repo_dir: str,
    branch: str = "",
    timeout: float = 60.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Push commits to GitHub. Uses gh CLI for auth."""
    # Use git push, but verify gh is available for auth
    try:
        env = os.environ.copy()
        start = time.monotonic()

        # Build git push command
        cmd = ["git", "push"]
        if branch:
            cmd.extend(["origin", branch])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=repo_dir,
            env=env,
        )

        elapsed = time.monotonic() - start

        if result.returncode != 0:
            return GitHubResult(
                success=False,
                operation="push",
                output=result.stdout.strip(),
                error=result.stderr.strip(),
                duration_seconds=elapsed,
            )

        return GitHubResult(
            success=True,
            operation="push",
            output=result.stdout.strip() + "\n" + result.stderr.strip(),
            duration_seconds=elapsed,
        )

    except Exception as e:
        return GitHubResult(
            success=False,
            operation="push",
            error=str(e),
            duration_seconds=time.monotonic() - start,
        )


def create_repo(
    name: str,
    description: str = "",
    public: bool = False,
    timeout: float = 30.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Create a new GitHub repository."""
    args = ["repo", "create", name]
    if description:
        args.extend(["--description", description])
    if public:
        args.append("--public")
    else:
        args.append("--private")

    result = _run_gh(args, timeout=timeout, gh_path=gh_path)

    # Extract URL from output if successful
    if result.success and result.output:
        # gh repo create outputs: ✓ Created repository ... on GitHub
        # URL is typically in the format https://github.com/user/repo
        for line in result.output.split("\n"):
            if "github.com" in line:
                import re
                match = re.search(r'https://github\.com/[^\s]+', line)
                if match:
                    return GitHubResult(
                        success=True,
                        operation="repo create",
                        output=result.output,
                        url=match.group(0),
                        duration_seconds=result.duration_seconds,
                    )

    return result


def create_issue(
    title: str,
    body: str = "",
    repo: str = "",
    timeout: float = 30.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Create a GitHub issue in the current or specified repo."""
    args = ["issue", "create", "--title", title]
    if body:
        args.extend(["--body", body])
    if repo:
        args.extend(["--repo", repo])

    return _run_gh(args, timeout=timeout, gh_path=gh_path)


def create_pr(
    title: str,
    body: str = "",
    base: str = "main",
    repo_dir: str = "",
    timeout: float = 30.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Create a pull request from the current branch."""
    args = ["pr", "create", "--title", title, "--base", base]
    if body:
        args.extend(["--body", body])

    return _run_gh(args, cwd=repo_dir, timeout=timeout, gh_path=gh_path)


def create_release(
    tag: str,
    title: str = "",
    notes: str = "",
    repo_dir: str = "",
    timeout: float = 30.0,
    gh_path: str = "gh",
) -> GitHubResult:
    """Create a GitHub release."""
    args = ["release", "create", tag]
    if title:
        args.extend(["--title", title])
    if notes:
        args.extend(["--notes", notes])

    return _run_gh(args, cwd=repo_dir, timeout=timeout, gh_path=gh_path)


def get_repo_info(
    repo_dir: str = "",
    timeout: float = 10.0,
    gh_path: str = "gh",
) -> Dict[str, str]:
    """Get current repository info (owner, name, URL)."""
    result = _run_gh(
        ["repo", "view", "--json", "nameWithOwner,url"],
        cwd=repo_dir,
        timeout=timeout,
        gh_path=gh_path,
    )

    if not result.success:
        return {}

    try:
        data = json.loads(result.output)
        return {
            "name": data.get("nameWithOwner", ""),
            "url": data.get("url", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {}
