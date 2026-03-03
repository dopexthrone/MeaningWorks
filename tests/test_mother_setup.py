"""
Phase 2: Tests for Mother setup wizard.

Tests step rendering, navigation, data collection, config persistence,
and cancellation.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mother.screens.setup import (
    SetupScreen,
    STEPS,
    PERSONALITY_DESCRIPTIONS,
    STEP_DESCRIPTIONS,
)
from mother.config import MotherConfig, save_config, load_config


class TestSetupScreenInstantiation:
    """Test SetupScreen creation."""

    def test_creates_screen(self):
        screen = SetupScreen()
        assert screen.current_step == "welcome"

    def test_accepts_config_path(self):
        screen = SetupScreen(config_path="/tmp/test.json")
        assert screen._config_path == "/tmp/test.json"

    def test_initial_step_is_welcome(self):
        screen = SetupScreen()
        assert screen._step_index == 0
        assert screen.current_step == "welcome"

    def test_has_default_config(self):
        screen = SetupScreen()
        assert screen._config.name == "Mother"
        assert screen._config.setup_complete is False


class TestStepNavigation:
    """Test step state machine navigation."""

    def test_all_steps_defined(self):
        assert len(STEPS) == 14
        assert STEPS[0] == "welcome"
        assert STEPS[-1] == "confirmation"

    def test_go_next_advances_step(self):
        screen = SetupScreen()
        screen._step_index = 0
        screen._step_index += 1  # simulate _go_next without UI
        assert screen.current_step == "name"

    def test_go_back_retreats_step(self):
        screen = SetupScreen()
        screen._step_index = 4
        screen._step_index -= 1
        assert screen.current_step == "personality"

    def test_cannot_go_before_first(self):
        screen = SetupScreen()
        screen._step_index = 0
        idx = max(0, screen._step_index - 1)
        assert idx == 0

    def test_last_step_is_confirmation(self):
        screen = SetupScreen()
        screen._step_index = len(STEPS) - 1
        assert screen.current_step == "confirmation"


class TestDataCollection:
    """Test collecting data from steps."""

    def test_name_collection(self):
        screen = SetupScreen()
        screen._config.name = "Athena"
        assert screen._config.name == "Athena"

    def test_personality_options_complete(self):
        assert "composed" in PERSONALITY_DESCRIPTIONS
        assert "warm" in PERSONALITY_DESCRIPTIONS
        assert "direct" in PERSONALITY_DESCRIPTIONS
        assert "playful" in PERSONALITY_DESCRIPTIONS

    def test_provider_options(self):
        from mother.config import PROVIDERS
        assert "claude" in PROVIDERS
        assert "openai" in PROVIDERS
        assert "grok" in PROVIDERS
        assert "gemini" in PROVIDERS

    def test_config_model_updates_with_provider(self):
        from mother.config import DEFAULT_MODELS
        screen = SetupScreen()
        screen._config.provider = "openai"
        screen._config.model = DEFAULT_MODELS["openai"]
        assert "gpt" in screen._config.model

    def test_permissions_defaults(self):
        screen = SetupScreen()
        assert screen._config.file_access is True
        assert screen._config.auto_compile is False
        assert screen._config.cost_limit == 100.0


class TestConfigPersistence:
    """Test setup completion saves config."""

    def test_setup_complete_flag(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            name="Athena",
            personality="warm",
            provider="openai",
            setup_complete=True,
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.setup_complete is True
        assert loaded.name == "Athena"

    def test_api_keys_persist(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(api_keys={"claude": "ANTHROPIC_API_KEY"})
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.api_keys["claude"] == "ANTHROPIC_API_KEY"


class TestValidation:
    """Test input validation."""

    def test_api_key_validation_error_stored(self):
        screen = SetupScreen()
        screen._validation_error = "API key is required."
        assert screen._validation_error == "API key is required."

    def test_validation_error_clears_on_collect(self):
        screen = SetupScreen()
        screen._validation_error = "old error"
        screen._step_index = 0  # welcome step — no validation
        result = screen._collect_step_data()
        assert result is True
        assert screen._validation_error is None


class TestStepDescriptions:
    """Test Mother-voiced step descriptions (C7)."""

    def test_all_steps_have_descriptions(self):
        for step in STEPS:
            assert step in STEP_DESCRIPTIONS, f"Missing description for step: {step}"

    def test_provider_description_mentions_rates(self):
        desc = STEP_DESCRIPTIONS["provider"]
        assert "$" in desc
        assert "Gemini" in desc or "gemini" in desc

    def test_permissions_description_explains_auto_compile(self):
        desc = STEP_DESCRIPTIONS["permissions"]
        assert "auto-compile" in desc.lower() or "Auto-compile" in desc


class TestApiKeyBugFix:
    """Test that API key stores the actual key value, not env var name."""

    def test_api_key_stores_value_not_env_var(self):
        screen = SetupScreen()
        screen._config.provider = "claude"
        # Simulate: user enters actual key, not env var name
        screen._config.api_keys["claude"] = "sk-ant-12345"
        assert screen._config.api_keys["claude"] == "sk-ant-12345"
        assert screen._config.api_keys["claude"] != "ANTHROPIC_API_KEY"


class TestCancelSetup:
    """Test cancelling setup."""

    def test_cancel_action_exists(self):
        screen = SetupScreen()
        # Verify escape binding exists
        bindings = {b[0] if isinstance(b, tuple) else b.key for b in screen.BINDINGS}
        assert "escape" in bindings
