"""
Tests for mother/introspection.py — structural self-knowledge.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mother.introspection import (
    CodebaseTopology,
    BuildDelta,
    scan_topology,
    git_diff_stats,
)


REPO_DIR = str(Path(__file__).resolve().parent.parent)


# --- CodebaseTopology ---


class TestCodebaseTopology:

    def test_default_construction(self):
        topo = CodebaseTopology()
        assert topo.total_files == 0
        assert topo.total_test_files == 0
        assert topo.total_tests == 0
        assert topo.modules == {}
        assert topo.total_lines == 0
        assert topo.protected_files == []
        assert topo.boundary_rule == ""

    def test_frozen(self):
        topo = CodebaseTopology()
        with pytest.raises(AttributeError):
            topo.total_files = 99


# --- scan_topology ---


class TestScanTopology:

    def test_finds_source_files(self):
        topo = scan_topology(REPO_DIR)
        assert topo.total_files > 0

    def test_finds_test_files(self):
        topo = scan_topology(REPO_DIR)
        assert topo.total_test_files > 0

    def test_modules_non_empty(self):
        topo = scan_topology(REPO_DIR)
        assert len(topo.modules) > 0
        assert "mother" in topo.modules
        assert "core" in topo.modules

    def test_protected_populated(self):
        topo = scan_topology(REPO_DIR)
        assert len(topo.protected_files) == 3
        assert "mother/context.py" in topo.protected_files
        assert "mother/persona.py" in topo.protected_files
        assert "mother/senses.py" in topo.protected_files

    def test_excludes_venv(self):
        """No files from .venv should be counted."""
        topo = scan_topology(REPO_DIR)
        # .venv has thousands of files — if we're counting them,
        # total_files would be enormous
        assert topo.total_files < 500  # sanity bound

    def test_lines_positive(self):
        topo = scan_topology(REPO_DIR)
        assert topo.total_lines > 0

    def test_boundary_rule(self):
        topo = scan_topology(REPO_DIR)
        assert "bridge.py" in topo.boundary_rule


# --- BuildDelta ---


class TestBuildDelta:

    def test_default_construction(self):
        delta = BuildDelta()
        assert delta.files_modified == 0
        assert delta.files_added == 0
        assert delta.lines_added == 0
        assert delta.lines_removed == 0
        assert delta.modules_touched == []

    def test_custom_fields(self):
        delta = BuildDelta(
            files_modified=3,
            files_added=1,
            lines_added=52,
            lines_removed=11,
            modules_touched=["mother", "core"],
        )
        assert delta.files_modified == 3
        assert delta.lines_added == 52
        assert delta.modules_touched == ["mother", "core"]


# --- git_diff_stats ---


class TestGitDiffStats:

    def test_empty_hash_returns_empty(self):
        delta = git_diff_stats(REPO_DIR, "")
        assert delta.files_modified == 0
        assert delta.lines_added == 0

    def test_parse_numstat_output(self):
        """Mock subprocess to test parsing logic."""
        numstat_output = "10\t3\tmother/bridge.py\n42\t8\tcore/engine.py\n"
        namestatus_output = "M\tmother/bridge.py\nA\tcore/engine.py\n"

        mock_numstat = MagicMock()
        mock_numstat.returncode = 0
        mock_numstat.stdout = numstat_output

        mock_namestatus = MagicMock()
        mock_namestatus.returncode = 0
        mock_namestatus.stdout = namestatus_output

        with patch("mother.introspection.subprocess.run", side_effect=[mock_numstat, mock_namestatus]):
            delta = git_diff_stats("/fake/repo", "abc1234")

        assert delta.lines_added == 52
        assert delta.lines_removed == 11
        assert delta.files_modified == 1
        assert delta.files_added == 1
        assert "mother" in delta.modules_touched
        assert "core" in delta.modules_touched

    def test_failure_returns_empty(self):
        """On subprocess error, returns empty BuildDelta."""
        with patch("mother.introspection.subprocess.run", side_effect=OSError("nope")):
            delta = git_diff_stats("/fake/repo", "abc1234")

        assert delta.files_modified == 0
        assert delta.lines_added == 0
        assert delta.modules_touched == []
