"""
Phase 5: Tests for settings screen.
"""

import pytest
from pathlib import Path

from mother.screens.settings import SettingsScreen
from mother.config import MotherConfig, save_config, load_config


class TestSettingsScreenCreation:
    """Test SettingsScreen instantiation."""

    def test_creates_screen(self):
        screen = SettingsScreen()
        assert screen is not None

    def test_accepts_config_path(self, tmp_path):
        path = str(tmp_path / "mother.json")
        save_config(MotherConfig(name="Athena"), path)
        screen = SettingsScreen(config_path=path)
        assert screen._config.name == "Athena"

    def test_loads_default_config_when_no_path(self):
        screen = SettingsScreen(config_path="/tmp/nonexistent_settings.json")
        assert screen._config.name == "Mother"

    def test_has_dismiss_binding(self):
        screen = SettingsScreen()
        bindings = {b[0] if isinstance(b, tuple) else b.key for b in screen.BINDINGS}
        assert "escape" in bindings


class TestSettingsSave:
    """Test saving settings."""

    def test_save_persists_name(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(name="Athena", setup_complete=True)
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.name == "Athena"

    def test_save_persists_provider(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(provider="openai", setup_complete=True)
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.provider == "openai"

    def test_save_persists_permissions(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(file_access=False, auto_compile=True)
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.file_access is False
        assert loaded.auto_compile is True

    def test_config_roundtrip_all_fields(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            name="Athena",
            personality="warm",
            provider="grok",
            model="grok-3",
            file_access=False,
            auto_compile=True,
            cost_limit=10.0,
            setup_complete=True,
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.name == "Athena"
        assert loaded.personality == "warm"
        assert loaded.provider == "grok"
        assert loaded.model == "grok-3"
        assert loaded.cost_limit == 10.0
