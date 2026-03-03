"""
Settings screen — modify Mother's configuration.

Sections: Identity, Provider, Permissions, Display.
Changes saved immediately via save_config().
"""

import os
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    Static,
    Button,
    Input,
    RadioButton,
    RadioSet,
    Switch,
    Footer,
)

from mother.config import (
    MotherConfig,
    load_config,
    save_config,
    PROVIDERS,
    DEFAULT_MODELS,
    ENV_VARS,
)
from mother.persona import PERSONALITY_MODIFIERS


class SettingsScreen(Screen):
    """Settings modification screen."""

    BINDINGS = [
        ("escape", "dismiss", "Back"),
    ]

    def __init__(self, config_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._config_path = config_path
        self._config = load_config(config_path)

    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            with Vertical(id="settings-box"):
                yield Static("Settings", id="settings-title")

                # Identity section
                with Vertical(classes="settings-section"):
                    yield Static("Identity", classes="settings-section-title")
                    yield Static("Name:")
                    yield Input(
                        value=self._config.name,
                        id="settings-name",
                    )
                    yield Static("Your Name:")
                    yield Input(
                        value=getattr(self._config, "user_name", "User"),
                        id="settings-user-name",
                    )
                    yield Static("Personality:")
                    radio_set = RadioSet(id="settings-personality")
                    for i, key in enumerate(PERSONALITY_MODIFIERS):
                        btn = RadioButton(
                            key.capitalize(),
                            id=f"settings-personality-{key}",
                            value=(key == self._config.personality),
                        )
                        radio_set.compose_add_child(btn)
                    yield radio_set
                    yield Static("Theme:")
                    theme_radio = RadioSet(id="settings-theme")
                    for theme in ["default", "alien"]:
                        btn = RadioButton(
                            theme.capitalize() if theme == "default" else "Alien (MU-TH-UR 6000)",
                            id=f"settings-theme-{theme}",
                            value=(theme == self._config.theme),
                        )
                        theme_radio.compose_add_child(btn)
                    yield theme_radio

                # Provider section
                with Vertical(classes="settings-section"):
                    yield Static("Provider", classes="settings-section-title")
                    radio_set = RadioSet(id="settings-provider")
                    for provider in PROVIDERS:
                        if provider == "local":
                            try:
                                from core.llm import OllamaClient
                                ollama_up = OllamaClient.is_available()
                            except Exception:
                                ollama_up = False
                            suffix = " (Ollama running)" if ollama_up else " (Ollama not found)"
                        else:
                            env_var = ENV_VARS[provider]
                            has_key = bool(os.environ.get(env_var))
                            suffix = " (key detected)" if has_key else ""
                        btn = RadioButton(
                            f"{provider.capitalize()}{suffix}",
                            id=f"settings-provider-{provider}",
                            value=(provider == self._config.provider),
                        )
                        radio_set.compose_add_child(btn)
                    yield radio_set

                # Permissions section
                with Vertical(classes="settings-section"):
                    yield Static("Permissions", classes="settings-section-title")
                    with Horizontal():
                        yield Static("File access      ", classes="perm-label")
                        yield Switch(
                            value=self._config.file_access,
                            id="settings-file-access",
                        )
                    with Horizontal():
                        yield Static("Auto-compile     ", classes="perm-label")
                        yield Switch(
                            value=self._config.auto_compile,
                            id="settings-auto-compile",
                        )
                    with Horizontal():
                        yield Static("Self-modification", classes="perm-label")
                        yield Switch(
                            value=self._config.claude_code_enabled,
                            id="settings-claude-code-enabled",
                        )

                # Voice section
                with Vertical(classes="settings-section"):
                    yield Static("Voice", classes="settings-section-title")
                    with Horizontal():
                        yield Static("Enable voice ", classes="perm-label")
                        yield Switch(
                            value=self._config.voice_enabled,
                            id="settings-voice-enabled",
                        )
                    yield Static("ElevenLabs API key:")
                    yield Input(
                        value=self._config.api_keys.get("elevenlabs", ""),
                        placeholder="sk_...",
                        password=True,
                        id="settings-voice-key",
                    )

                # Perception section
                with Vertical(classes="settings-section"):
                    yield Static("Perception", classes="settings-section-title")
                    with Horizontal():
                        yield Static("Screen capture    ", classes="perm-label")
                        yield Switch(
                            value=self._config.screen_capture_enabled,
                            id="settings-screen-capture",
                        )
                    with Horizontal():
                        yield Static("Microphone        ", classes="perm-label")
                        yield Switch(
                            value=self._config.microphone_enabled,
                            id="settings-microphone",
                        )
                    with Horizontal():
                        yield Static("Camera            ", classes="perm-label")
                        yield Switch(
                            value=self._config.camera_enabled,
                            id="settings-camera",
                        )
                    with Horizontal():
                        yield Static("Screen monitoring ", classes="perm-label")
                        yield Switch(
                            value=self._config.screen_monitoring,
                            id="settings-screen-monitoring",
                        )
                    with Horizontal():
                        yield Static("Ambient listening ", classes="perm-label")
                        yield Switch(
                            value=self._config.ambient_listening,
                            id="settings-ambient-listening",
                        )
                    yield Static("Perception budget ($/hour):")
                    yield Input(
                        value=str(self._config.perception_budget),
                        placeholder="0.50",
                        id="settings-perception-budget",
                    )

                # Actions
                with Horizontal(id="setup-nav"):
                    yield Button("Cancel", id="settings-cancel")
                    yield Button("Save", id="settings-save", variant="primary")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-save":
            self._save()
        elif event.button.id == "settings-cancel":
            self.dismiss()

    def _save(self) -> None:
        """Collect and save settings."""
        # Name
        try:
            name = self.query_one("#settings-name", Input).value.strip()
            if name:
                self._config.name = name
        except Exception:
            pass

        # User name
        try:
            user_name = self.query_one("#settings-user-name", Input).value.strip()
            if user_name:
                self._config.user_name = user_name
        except Exception:
            pass

        # Personality
        try:
            radio = self.query_one("#settings-personality", RadioSet)
            if radio.pressed_index >= 0:
                keys = list(PERSONALITY_MODIFIERS.keys())
                self._config.personality = keys[radio.pressed_index]
        except Exception:
            pass

        # Theme
        try:
            radio = self.query_one("#settings-theme", RadioSet)
            if radio.pressed_index >= 0:
                themes = ["default", "alien"]
                self._config.theme = themes[radio.pressed_index]
        except Exception:
            pass

        # Provider
        try:
            radio = self.query_one("#settings-provider", RadioSet)
            if radio.pressed_index >= 0:
                self._config.provider = PROVIDERS[radio.pressed_index]
                self._config.model = DEFAULT_MODELS[self._config.provider]
        except Exception:
            pass

        # Permissions
        try:
            self._config.file_access = self.query_one("#settings-file-access", Switch).value
        except Exception:
            pass
        try:
            self._config.auto_compile = self.query_one("#settings-auto-compile", Switch).value
        except Exception:
            pass
        try:
            self._config.claude_code_enabled = self.query_one("#settings-claude-code-enabled", Switch).value
        except Exception:
            pass

        # Voice
        try:
            self._config.voice_enabled = self.query_one("#settings-voice-enabled", Switch).value
        except Exception:
            pass
        try:
            key = self.query_one("#settings-voice-key", Input).value.strip()
            if key:
                self._config.api_keys["elevenlabs"] = key
        except Exception:
            pass

        # Perception
        try:
            self._config.screen_capture_enabled = self.query_one("#settings-screen-capture", Switch).value
        except Exception:
            pass
        try:
            self._config.microphone_enabled = self.query_one("#settings-microphone", Switch).value
        except Exception:
            pass
        try:
            self._config.camera_enabled = self.query_one("#settings-camera", Switch).value
        except Exception:
            pass
        try:
            self._config.screen_monitoring = self.query_one("#settings-screen-monitoring", Switch).value
        except Exception:
            pass
        try:
            self._config.ambient_listening = self.query_one("#settings-ambient-listening", Switch).value
        except Exception:
            pass
        try:
            budget_str = self.query_one("#settings-perception-budget", Input).value.strip()
            if budget_str:
                self._config.perception_budget = float(budget_str)
        except (ValueError, Exception):
            pass

        save_config(self._config, self._config_path)
        self.dismiss()

    def action_dismiss(self) -> None:
        """Go back without saving."""
        self.dismiss()
