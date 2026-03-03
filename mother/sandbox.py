"""
macOS sandbox-exec wrapper for isolated self-build execution.

LEAF module. Stdlib only. No imports from core/ or mother/.

Generates Sandbox Profile Language (SBPL) strings and wraps shell commands
with sandbox-exec to enforce: no network access, restricted filesystem writes,
no persistent state mutation outside the build directory.

Platform-aware: enforced on macOS (sandbox-exec available), graceful no-op on
Linux/other platforms (logs warning, returns unwrapped command).
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.sandbox")


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandboxProfile:
    """Defines what a sandboxed process is allowed to do."""
    allow_read_paths: Tuple[str, ...] = ()   # venv, system libs, worktree
    allow_write_paths: Tuple[str, ...] = ()  # worktree only
    allow_network: bool = False


# ---------------------------------------------------------------------------
# SBPL generation
# ---------------------------------------------------------------------------

def generate_sbpl(profile: SandboxProfile) -> str:
    """Generate a Sandbox Profile Language string for sandbox-exec.

    The profile starts with (allow default) then layers deny rules:
    - (deny network*) unless allow_network is True
    - (deny file-write*) then (allow file-write* (subpath "...")) for each path
    - Always allows /dev/null and /dev/tty writes (pytest logging needs these)
    - Always allows /private/tmp and /private/var/folders for temp files
    """
    parts = ['(version 1)', '(allow default)']

    # Network
    if not profile.allow_network:
        parts.append('(deny network*)')

    # File writes — deny all, then allow specific paths
    if profile.allow_write_paths:
        parts.append('(deny file-write*)')

        for path in profile.allow_write_paths:
            resolved = os.path.realpath(path)
            parts.append(f'(allow file-write* (subpath "{resolved}"))')

        # Device writes required for subprocess stdout/stderr/logging
        parts.append('(allow file-write* (literal "/dev/null"))')
        parts.append('(allow file-write* (literal "/dev/tty"))')
        parts.append('(allow file-write* (regex #"^/dev/fd/"))')

        # Temp directories — pytest and Python stdlib need these
        parts.append('(allow file-write* (subpath "/private/tmp"))')
        parts.append('(allow file-write* (subpath "/private/var/folders"))')

        # Allow writing to stdout/stderr file descriptors
        parts.append('(allow file-write* (literal "/dev/stdout"))')
        parts.append('(allow file-write* (literal "/dev/stderr"))')

    return ''.join(parts)


def sandbox_command(
    command: str,
    profile: SandboxProfile,
    cwd: Optional[str] = None,
) -> List[str]:
    """Wrap a shell command with sandbox-exec.

    Returns the command list: ['sandbox-exec', '-p', <sbpl>, '/bin/bash', '-c', command]
    If sandbox-exec is not available, returns the unwrapped command.
    """
    if not is_sandbox_available():
        logger.warning("sandbox-exec not available — running without sandbox")
        return ['/bin/bash', '-c', command]

    sbpl = generate_sbpl(profile)
    return ['sandbox-exec', '-p', sbpl, '/bin/bash', '-c', command]


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_SANDBOX_EXEC_PATH = '/usr/bin/sandbox-exec'


def is_sandbox_available() -> bool:
    """True if sandbox-exec exists on this system (macOS)."""
    return os.path.isfile(_SANDBOX_EXEC_PATH) and os.access(_SANDBOX_EXEC_PATH, os.X_OK)


# ---------------------------------------------------------------------------
# Standard profiles
# ---------------------------------------------------------------------------

def create_build_profile(
    worktree_dir: str,
    venv_dir: str,
) -> SandboxProfile:
    """Create the standard self-build sandbox profile.

    Allows:
    - Read: everywhere (default allow)
    - Write: worktree directory only (+ /dev/null, /private/tmp)
    - Network: denied

    All paths are resolved via os.path.realpath() before embedding in SBPL
    because macOS requires resolved paths (/tmp → /private/tmp).
    """
    resolved_worktree = os.path.realpath(worktree_dir)
    resolved_venv = os.path.realpath(venv_dir)

    return SandboxProfile(
        allow_read_paths=(resolved_worktree, resolved_venv),
        allow_write_paths=(resolved_worktree,),
        allow_network=False,
    )
