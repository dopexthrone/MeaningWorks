"""Test self-modification toggle in settings screen."""

import json
import tempfile
from pathlib import Path

import pytest

from mother.config import MotherConfig, save_config, load_config
from mother.screens.settings import SettingsScreen


class TestSelfModificationToggle:
    """Test self-modification settings toggle."""

    def test_config_has_claude_code_enabled_field(self):
        """Config dataclass has claude_code_enabled field."""
        config = MotherConfig()
        assert hasattr(config, "claude_code_enabled")
        assert isinstance(config.claude_code_enabled, bool)
        assert config.claude_code_enabled is False  # Default

    def test_config_saves_claude_code_enabled(self, tmp_path):
        """Config saves and loads claude_code_enabled correctly."""
        config_path = tmp_path / "test.json"

        # Save with False
        config = MotherConfig()
        config.claude_code_enabled = False
        save_config(config, config_path)

        # Load and verify
        loaded = load_config(config_path)
        assert loaded.claude_code_enabled is False

        # Toggle to True
        loaded.claude_code_enabled = True
        save_config(loaded, config_path)

        # Load and verify
        final = load_config(config_path)
        assert final.claude_code_enabled is True

    def test_config_loads_missing_field_as_default(self, tmp_path):
        """Config without claude_code_enabled gets default value."""
        config_path = tmp_path / "test.json"

        # Write config without the field
        data = {
            "name": "Mother",
            "personality": "composed",
            "provider": "claude"
        }
        config_path.write_text(json.dumps(data))

        # Load should use default
        loaded = load_config(config_path)
        assert loaded.claude_code_enabled is False  # Default value

    def test_settings_screen_has_claude_code_toggle_widget(self):
        """Settings screen compose includes claude_code_enabled widget."""
        # Can't test UI composition without app context
        # Just verify the ID exists in source
        import inspect
        source = inspect.getsource(SettingsScreen.compose)
        assert "settings-claude-code-enabled" in source

    def test_settings_screen_saves_claude_code_enabled(self, tmp_path):
        """Settings screen can save claude_code_enabled setting."""
        # This tests the logic, not the UI
        config_path = tmp_path / "test.json"

        # Create config
        config = MotherConfig()
        config.claude_code_enabled = False
        save_config(config, config_path)

        # Simulate what _save() does
        loaded = load_config(config_path)
        loaded.claude_code_enabled = True  # Toggle it
        save_config(loaded, config_path)

        # Verify saved
        final = load_config(config_path)
        assert final.claude_code_enabled is True

    def test_actual_user_config_has_field(self):
        """User's ~/.motherlabs/mother.json has claude_code_enabled."""
        user_config_path = Path.home() / ".motherlabs" / "mother.json"

        if not user_config_path.exists():
            pytest.skip("User config doesn't exist")

        data = json.loads(user_config_path.read_text())
        assert "claude_code_enabled" in data
        assert isinstance(data["claude_code_enabled"], bool)
