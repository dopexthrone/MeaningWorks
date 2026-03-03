"""
Phase 4: Tests for compilation flow in chat screen.

Tests command parsing, pipeline mount, and compile flow logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from mother.screens.chat import ChatScreen, SLASH_COMMANDS
from mother.config import MotherConfig
from mother.widgets.pipeline import PipelinePanel, PIPELINE_STAGES


class TestCompileCommandParsing:
    """Test /compile command routing."""

    def test_compile_in_slash_commands(self):
        assert "/compile" in SLASH_COMMANDS

    def test_build_in_slash_commands(self):
        assert "/build" in SLASH_COMMANDS


class TestPipelinePanelInit:
    """Test pipeline panel initialization."""

    def test_panel_has_seven_stages(self):
        panel = PipelinePanel()
        assert len(panel.stages) == 7

    def test_panel_starts_hidden(self):
        panel = PipelinePanel()
        assert "visible" not in panel.classes


class TestCompileFlowLogic:
    """Test compile flow setup."""

    def test_compilation_count_starts_zero(self):
        screen = ChatScreen()
        assert screen._compilation_count == 0

    def test_config_output_dir_used(self):
        config = MotherConfig(output_dir="/tmp/test-output")
        screen = ChatScreen(config=config)
        assert screen._config.output_dir == "/tmp/test-output"

    def test_pipeline_stages_match_engine(self):
        """Pipeline stages should match the 7-agent engine pipeline."""
        expected = ["Intent", "Persona", "Entity", "Process", "Synthesis", "Verify", "Governor"]
        assert PIPELINE_STAGES == expected
