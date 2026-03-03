"""Test /theme command in chat screen."""

import tempfile
from pathlib import Path

from mother.config import MotherConfig, save_config, load_config
from mother.screens.chat import SLASH_COMMANDS


class TestThemeCommand:
    """Test /theme slash command."""

    def test_theme_in_slash_commands(self):
        """Verify /theme is registered as a slash command."""
        assert "/theme" in SLASH_COMMANDS

    def test_theme_in_help_text(self):
        """Verify /theme appears in help text."""
        from mother.screens.chat import HELP_TEXT
        assert "/theme" in HELP_TEXT
        assert "alien" in HELP_TEXT.lower()

    def test_config_theme_field_exists(self):
        """Config has theme field."""
        config = MotherConfig()
        assert hasattr(config, "theme")
        assert config.theme == "default"  # Default value

    def test_theme_saves_to_config(self, tmp_path):
        """Theme can be saved to config."""
        config_path = tmp_path / "test.json"

        # Create config with default theme
        config = MotherConfig()
        config.theme = "default"
        save_config(config, config_path)

        # Load and verify
        loaded = load_config(config_path)
        assert loaded.theme == "default"

        # Change to alien
        loaded.theme = "alien"
        save_config(loaded, config_path)

        # Load and verify
        final = load_config(config_path)
        assert final.theme == "alien"

    def test_app_loads_theme_from_config(self):
        """MotherApp loads correct CSS based on theme."""
        import tempfile
        from mother.app import MotherApp
        from mother.config import MotherConfig, save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"

            # Test default theme
            config = MotherConfig()
            config.theme = "default"
            save_config(config, config_path)

            app = MotherApp(config_path=str(config_path))
            assert app.CSS_PATH == "mother.tcss"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"

            # Test alien theme
            config = MotherConfig()
            config.theme = "alien"
            save_config(config, config_path)

            app = MotherApp(config_path=str(config_path))
            assert app.CSS_PATH == "mother_alien.tcss"

    def test_alien_theme_stylesheet_exists(self):
        """Alien theme CSS file exists."""
        from pathlib import Path
        css_path = Path(__file__).parent.parent / "mother" / "mother_alien.tcss"
        assert css_path.exists()
        assert css_path.stat().st_size > 0  # Not empty
