"""
Phase 6: Real LLM integration tests for Mother TUI.

These tests require actual API keys and cost real money.
Marked with @pytest.mark.slow — excluded from default test runs.
"""

import os
import asyncio
import pytest

from mother.config import MotherConfig
from mother.bridge import EngineBridge
from mother.persona import build_system_prompt


@pytest.mark.slow
class TestRealLLMChat:
    """Test real chat with an LLM provider."""

    def test_chat_returns_response(self):
        """Send a single message and verify a response comes back."""
        # Auto-detect provider from env
        config = MotherConfig()
        bridge = EngineBridge(provider="auto")
        prompt = build_system_prompt(config)

        response = asyncio.run(bridge.chat(
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
            system_prompt=prompt,
        ))
        assert len(response) > 0
        assert bridge.get_session_cost() > 0


@pytest.mark.slow
class TestRealLLMPersonaConsistency:
    """Test that persona is reflected in responses."""

    def test_persona_shapes_response(self):
        """Verify the response style matches persona traits."""
        config = MotherConfig(personality="direct")
        bridge = EngineBridge(provider="auto")
        prompt = build_system_prompt(config)

        response = asyncio.run(bridge.chat(
            messages=[{"role": "user", "content": "What are you?"}],
            system_prompt=prompt,
        ))
        # Direct persona should not hedge
        assert "I think maybe" not in response
        assert len(response) > 0
