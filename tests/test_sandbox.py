"""
Tests for mother/sandbox.py — macOS sandbox-exec wrapper.

LEAF module. Covers:
A. SandboxProfile frozen dataclass
B. generate_sbpl() SBPL string generation
C. sandbox_command() command wrapping
D. is_sandbox_available() platform detection
E. create_build_profile() standard profile construction
"""

import os
import sys
import pytest
from unittest.mock import patch

from mother.sandbox import (
    SandboxProfile,
    generate_sbpl,
    sandbox_command,
    is_sandbox_available,
    create_build_profile,
)


# ---------------------------------------------------------------------------
# A. SandboxProfile frozen dataclass
# ---------------------------------------------------------------------------

class TestSandboxProfile:
    def test_frozen(self):
        p = SandboxProfile()
        with pytest.raises(AttributeError):
            p.allow_network = True

    def test_defaults(self):
        p = SandboxProfile()
        assert p.allow_read_paths == ()
        assert p.allow_write_paths == ()
        assert p.allow_network is False

    def test_custom_values(self):
        p = SandboxProfile(
            allow_read_paths=("/a", "/b"),
            allow_write_paths=("/c",),
            allow_network=True,
        )
        assert p.allow_read_paths == ("/a", "/b")
        assert p.allow_write_paths == ("/c",)
        assert p.allow_network is True


# ---------------------------------------------------------------------------
# B. generate_sbpl()
# ---------------------------------------------------------------------------

class TestGenerateSbpl:
    def test_version_header(self):
        sbpl = generate_sbpl(SandboxProfile())
        assert "(version 1)" in sbpl

    def test_allow_default(self):
        sbpl = generate_sbpl(SandboxProfile())
        assert "(allow default)" in sbpl

    def test_network_deny_when_false(self):
        sbpl = generate_sbpl(SandboxProfile(allow_network=False))
        assert "(deny network*)" in sbpl

    def test_network_allowed_when_true(self):
        sbpl = generate_sbpl(SandboxProfile(allow_network=True))
        assert "(deny network*)" not in sbpl

    def test_write_deny_with_paths(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        assert "(deny file-write*)" in sbpl

    def test_write_allow_subpath(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        # Path is resolved via realpath — /tmp → /private/tmp on macOS
        resolved = os.path.realpath("/tmp/build")
        assert f'(allow file-write* (subpath "{resolved}"))' in sbpl

    def test_dev_null_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        assert '(allow file-write* (literal "/dev/null"))' in sbpl

    def test_dev_tty_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        assert '(allow file-write* (literal "/dev/tty"))' in sbpl

    def test_private_tmp_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        assert '(allow file-write* (subpath "/private/tmp"))' in sbpl

    def test_private_var_folders_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/build",)))
        assert '(allow file-write* (subpath "/private/var/folders"))' in sbpl

    def test_no_write_deny_without_paths(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=()))
        assert "(deny file-write*)" not in sbpl

    def test_multiple_write_paths(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/a", "/b")))
        resolved_a = os.path.realpath("/a")
        resolved_b = os.path.realpath("/b")
        assert f'(subpath "{resolved_a}")' in sbpl
        assert f'(subpath "{resolved_b}")' in sbpl

    def test_dev_fd_regex_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/x",)))
        assert '(allow file-write* (regex #"^/dev/fd/"))' in sbpl

    def test_stdout_stderr_allowed(self):
        sbpl = generate_sbpl(SandboxProfile(allow_write_paths=("/tmp/x",)))
        assert '(allow file-write* (literal "/dev/stdout"))' in sbpl
        assert '(allow file-write* (literal "/dev/stderr"))' in sbpl


# ---------------------------------------------------------------------------
# C. sandbox_command()
# ---------------------------------------------------------------------------

class TestSandboxCommand:
    def test_returns_list(self):
        profile = SandboxProfile(allow_write_paths=("/tmp/build",))
        with patch("mother.sandbox.is_sandbox_available", return_value=True):
            cmd = sandbox_command("echo hello", profile)
        assert isinstance(cmd, list)
        assert cmd[0] == "sandbox-exec"

    def test_command_structure(self):
        profile = SandboxProfile()
        with patch("mother.sandbox.is_sandbox_available", return_value=True):
            cmd = sandbox_command("echo hello", profile)
        assert cmd[0] == "sandbox-exec"
        assert cmd[1] == "-p"
        # cmd[2] is the SBPL string
        assert cmd[3] == "/bin/bash"
        assert cmd[4] == "-c"
        assert cmd[5] == "echo hello"

    def test_fallback_without_sandbox(self):
        with patch("mother.sandbox.is_sandbox_available", return_value=False):
            cmd = sandbox_command("echo hello", SandboxProfile())
        assert cmd == ["/bin/bash", "-c", "echo hello"]

    def test_sbpl_embedded_in_command(self):
        profile = SandboxProfile(allow_network=False)
        with patch("mother.sandbox.is_sandbox_available", return_value=True):
            cmd = sandbox_command("ls", profile)
        sbpl = cmd[2]
        assert "(deny network*)" in sbpl


# ---------------------------------------------------------------------------
# D. is_sandbox_available()
# ---------------------------------------------------------------------------

class TestIsSandboxAvailable:
    def test_returns_bool(self):
        result = is_sandbox_available()
        assert isinstance(result, bool)

    @patch("os.path.isfile", return_value=True)
    @patch("os.access", return_value=True)
    def test_true_when_exists_and_executable(self, mock_access, mock_isfile):
        assert is_sandbox_available() is True

    @patch("os.path.isfile", return_value=False)
    def test_false_when_not_exists(self, mock_isfile):
        assert is_sandbox_available() is False


# ---------------------------------------------------------------------------
# E. create_build_profile()
# ---------------------------------------------------------------------------

class TestCreateBuildProfile:
    def test_returns_sandbox_profile(self):
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        assert isinstance(profile, SandboxProfile)

    def test_network_denied(self):
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        assert profile.allow_network is False

    def test_write_path_is_worktree(self):
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        resolved = os.path.realpath("/tmp/worktree")
        assert resolved in profile.allow_write_paths

    def test_read_paths_include_both(self):
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        resolved_wt = os.path.realpath("/tmp/worktree")
        resolved_venv = os.path.realpath("/tmp/venv")
        assert resolved_wt in profile.allow_read_paths
        assert resolved_venv in profile.allow_read_paths

    def test_paths_resolved(self):
        """All paths should be resolved via realpath (no symlinks)."""
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        for path in profile.allow_write_paths:
            assert path == os.path.realpath(path)
        for path in profile.allow_read_paths:
            assert path == os.path.realpath(path)

    def test_only_worktree_writable(self):
        profile = create_build_profile("/tmp/worktree", "/tmp/venv")
        assert len(profile.allow_write_paths) == 1
