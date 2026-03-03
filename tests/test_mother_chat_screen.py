"""
Phase 3: Tests for chat screen.
"""

import pytest
from unittest.mock import patch, MagicMock

from mother.screens.chat import (
    ChatScreen,
    GREETING,
    HELP_TEXT,
    SLASH_COMMANDS,
    parse_response,
)
from mother.config import MotherConfig
from mother.persona import build_greeting, narrate_error, inject_personality_bite


class TestChatScreenCreation:
    """Test ChatScreen instantiation."""

    def test_creates_screen(self):
        screen = ChatScreen()
        assert screen._chatting is False

    def test_accepts_config(self):
        config = MotherConfig(name="Athena")
        screen = ChatScreen(config=config)
        assert screen._config.name == "Athena"

    def test_default_config_used(self):
        screen = ChatScreen()
        assert screen._config.name == "Mother"

    def test_api_key_in_config_available(self):
        """Config api_keys should be available for bridge creation."""
        config = MotherConfig(
            provider="grok",
            api_keys={"grok": "xai-test-123"},
        )
        screen = ChatScreen(config=config)
        assert screen._config.api_keys.get("grok") == "xai-test-123"


class TestSlashCommands:
    """Test slash command definitions."""

    def test_help_in_commands(self):
        assert "/help" in SLASH_COMMANDS

    def test_compile_in_commands(self):
        assert "/compile" in SLASH_COMMANDS

    def test_build_in_commands(self):
        assert "/build" in SLASH_COMMANDS

    def test_clear_in_commands(self):
        assert "/clear" in SLASH_COMMANDS

    def test_tools_in_commands(self):
        assert "/tools" in SLASH_COMMANDS

    def test_status_in_commands(self):
        assert "/status" in SLASH_COMMANDS


    def test_search_in_commands(self):
        assert "/search" in SLASH_COMMANDS

    def test_find_in_commands(self):
        assert "/find" in SLASH_COMMANDS


class TestGreeting:
    """Test greeting and help text."""

    def test_greeting_not_empty(self):
        assert len(GREETING) > 20

    def test_help_text_contains_commands(self):
        assert "/compile" in HELP_TEXT
        assert "/build" in HELP_TEXT
        assert "/help" in HELP_TEXT

    def test_help_text_contains_search(self):
        assert "/search" in HELP_TEXT


class TestGreetingIntegration:
    """Test history-aware greeting via build_greeting (C1)."""

    def test_first_visit_uses_default_greeting(self):
        config = MotherConfig()
        greeting = build_greeting(config, memory_summary=None)
        assert "?" in greeting  # First visit greeting is a question

    def test_return_visit_references_topic(self):
        config = MotherConfig()
        summary = {"total_sessions": 2, "topics": ["tattoo studio CRM"], "days_since_last": 0.5}
        greeting = build_greeting(config, memory_summary=summary)
        assert "tattoo studio CRM" in greeting


class TestErrorNarrationIntegration:
    """Test error narration from chat screen perspective (C3)."""

    def test_timeout_narration(self):
        msg = narrate_error(TimeoutError("timed out"), phase="chat")
        assert "timed out" in msg.lower()

    def test_cost_cap_narration(self):
        msg = narrate_error(Exception("cost cap exceeded"), phase="compile")
        assert "cost cap" in msg.lower()


class TestPersonalityBiteIntegration:
    """Test personality bites in compile flow (C5)."""

    def test_playful_first_compile(self):
        bite = inject_personality_bite("playful", "first_compile")
        assert bite is not None
        assert "first" in bite.lower() or "here we go" in bite.lower()

    def test_warm_build_success(self):
        bite = inject_personality_bite("warm", "build_success")
        assert bite is not None


class TestParseResponseFileActions:
    """Test parse_response for file-related action markers."""

    def test_search_action_parsed(self):
        raw = "[ACTION:search]resume[/ACTION][VOICE]Let me look for that.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "search"
        assert parsed["action_arg"] == "resume"
        assert parsed["voice"] == "Let me look for that."

    def test_open_action_parsed(self):
        raw = "[ACTION:open]/Users/test/config.yaml[/ACTION][VOICE]Let me read that.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "open"
        assert parsed["action_arg"] == "/Users/test/config.yaml"

    def test_file_action_move_parsed(self):
        raw = "[ACTION:file]move: ~/a.txt -> ~/b.txt[/ACTION][VOICE]Moving it.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "move:" in parsed["action_arg"]
        assert "->" in parsed["action_arg"]

    def test_file_action_delete_parsed(self):
        raw = "[ACTION:file]delete: ~/old.txt[/ACTION][VOICE]Moving to Trash.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "file"
        assert "delete:" in parsed["action_arg"]

    def test_search_action_display_strips_tags(self):
        raw = "Some text [ACTION:search]query[/ACTION] more text"
        parsed = parse_response(raw)
        assert "[ACTION:" not in parsed["display"]
        assert "Some text" in parsed["display"]
        assert "more text" in parsed["display"]
