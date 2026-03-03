"""
Phase 6: Integration tests for Mother TUI.

Tests full flows: setup → config → chat → compile → results.
All LLM calls are mocked.
"""

import json
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mother.app import MotherApp
from mother.config import MotherConfig, save_config, load_config, detect_first_run
from mother.persona import build_system_prompt
from mother.memory import ConversationStore
from mother.bridge import EngineBridge
from mother.screens.chat import ChatScreen, GREETING, SLASH_COMMANDS
from mother.screens.setup import SetupScreen, STEPS
from mother.screens.settings import SettingsScreen
from mother.widgets.pipeline import PipelinePanel, PIPELINE_STAGES
from mother.widgets.trust_badge import TrustBadge, trust_level


class TestFullFirstRunFlow:
    """Test first-run detection → setup → config saved."""

    def test_first_run_detected(self, tmp_path):
        path = str(tmp_path / "mother.json")
        assert detect_first_run(path) is True

    def test_setup_creates_config(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            name="Athena",
            personality="warm",
            provider="claude",
            setup_complete=True,
        )
        save_config(config, path)
        assert detect_first_run(path) is False
        loaded = load_config(path)
        assert loaded.name == "Athena"

    def test_config_persists_across_restart(self, tmp_path):
        """Simulate restart by save → load cycle."""
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            name="Athena",
            personality="direct",
            provider="openai",
            model="gpt-4o",
            setup_complete=True,
        )
        save_config(config, path)

        # "Restart" — fresh load
        config2 = load_config(path)
        assert config2.name == "Athena"
        assert config2.personality == "direct"
        assert config2.provider == "openai"


class TestChatWithMockedLLM:
    """Test chat flow with mocked bridge."""

    def test_bridge_chat_returns_response(self):
        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "I'm ready to help."
        mock_llm.last_usage = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
        bridge._llm = mock_llm

        response = asyncio.run(bridge.chat(
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="You are Mother.",
        ))
        assert "ready" in response.lower()

    def test_session_cost_tracks(self):
        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Response"
        mock_llm.last_usage = {"input_tokens": 500, "output_tokens": 200, "total_tokens": 700}
        bridge._llm = mock_llm

        asyncio.run(bridge.chat(
            [{"role": "user", "content": "hi"}], "system"
        ))
        assert bridge.get_session_cost() > 0


class TestConversationPersistence:
    """Test message history across sessions."""

    def test_history_persists(self, tmp_path):
        db = tmp_path / "history.db"

        # Session 1
        store1 = ConversationStore(path=db)
        sid = store1.session_id
        store1.add_message("user", "hello", session_id=sid)
        store1.add_message("assistant", "hi there", session_id=sid)
        store1.close()

        # Session 2 — same session_id
        store2 = ConversationStore(path=db)
        history = store2.get_history(session_id=sid)
        assert len(history) == 2
        assert history[0].content == "hello"
        store2.close()


class TestCommandRouting:
    """Test slash command definitions."""

    def test_all_commands_defined(self):
        expected = {"/help", "/clear", "/settings", "/status", "/handoff", "/compile", "/build", "/launch", "/stop", "/tools", "/search", "/find", "/capture", "/camera", "/listen", "/theme", "/ideas"}
        assert SLASH_COMMANDS == expected

    def test_greeting_present(self):
        assert len(GREETING) > 0

    def test_pipeline_stages_count(self):
        assert len(PIPELINE_STAGES) == 7


class TestSystemPromptIntegration:
    """Test persona + config integration."""

    def test_prompt_contains_persona_and_modifier(self):
        config = MotherConfig(personality="warm", name="Athena")
        prompt = build_system_prompt(config)
        assert "Mother" in prompt
        assert "attentiveness" in prompt
        assert "Athena" in prompt

    def test_prompt_without_compile(self):
        config = MotherConfig()
        prompt = build_system_prompt(config, include_compile=False)
        assert "/compile" not in prompt
