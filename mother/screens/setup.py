"""
Setup wizard — first-run conversational onboarding.

Multi-step state machine: Welcome → Name → Personality → Provider →
API Key → Permissions → Confirmation.
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
    Label,
)
from textual.message import Message

from mother.config import (
    MotherConfig,
    save_config,
    ENV_VARS,
    DEFAULT_MODELS,
    PROVIDERS,
)


# Step names for the state machine
STEPS = [
    "welcome",
    "name",
    "user_name",
    "personality",
    "provider",
    "api_key",
    "permissions",
    "voice",
    "screen_capture",
    "microphone",
    "camera",
    "ambient_listening",
    "screen_monitoring",
    "confirmation",
]

# Text content for each personality option
PERSONALITY_DESCRIPTIONS = {
    "composed": "Measured, deliberate, calm authority. The most competent person in the room.",
    "warm": "Attentive, caring, remembers everything. Precise, but with warmth.",
    "direct": "Shortest path to the answer. No preamble. Efficient and decisive.",
    "playful": "Dry wit, intellectual curiosity. Still sharp, but with a lighter touch.",
    "david": "Intellectual curiosity, composed precision, anticipatory. Works while you sleep.",
}

# Mother-voiced guidance for each wizard step
STEP_DESCRIPTIONS = {
    "welcome": (
        "I'm Mother. I build what you describe.\n\n"
        "Let me learn a few things about how you'd like to work together. "
        "This takes about a minute."
    ),
    "name": "What should I call myself? This is how I'll refer to myself in conversation.",
    "user_name": "What's your name? This is how I'll label your messages.",
    "personality": "How should I communicate? This shapes my tone, not my capability.",
    "provider": (
        "Which AI provider should I use?\n"
        "Claude: best quality, $3/$15 per MTok. "
        "OpenAI: balanced, $2/$8. "
        "Grok: fast reasoning, $2/$20. "
        "Gemini: cheapest at $0.10/$0.40. "
        "Local (Ollama): free, runs on your GPU."
    ),
    "api_key": "I need an API key to connect. This stays on your machine — never sent anywhere except the provider.",
    "permissions": (
        "What am I allowed to do?\n"
        "File access lets me read and write project files. "
        "Auto-compile means I'll compile without asking when intent is clear."
    ),
    "voice": (
        "I can speak. If you have an ElevenLabs API key, "
        "enable voice and I'll narrate key moments."
    ),
    "screen_capture": (
        "I can see your screen when you ask. "
        "Uses macOS built-in screencapture. No additional API keys needed."
    ),
    "microphone": (
        "You can talk to me instead of typing. "
        "Transcription uses OpenAI Whisper API. "
        "Requires an OpenAI API key and the sounddevice package."
    ),
    "camera": (
        "I can see through your webcam when you ask. "
        "Uses FFmpeg to capture a single frame. No video recording."
    ),
    "ambient_listening": (
        "I can listen in the background and respond when you speak. "
        "Uses voice activity detection — I only transcribe when you're actually talking. "
        "Transcription uses OpenAI Whisper API."
    ),
    "screen_monitoring": (
        "I can watch your screen in the background and notice when things change. "
        "This uses vision API calls, which cost money. "
        "Set a budget to control hourly spend."
    ),
    "confirmation": "Everything look right? Press Begin to start.",
}


class SetupScreen(Screen):
    """First-run setup wizard."""

    class SetupComplete(Message):
        """Posted when setup finishes."""

        def __init__(self, config: MotherConfig):
            super().__init__()
            self.config = config

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, config_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._config_path = config_path
        self._step_index = 0
        self._config = MotherConfig()
        self._validation_error: str | None = None

    @property
    def current_step(self) -> str:
        return STEPS[self._step_index]

    def compose(self) -> ComposeResult:
        with Container(id="setup-container"):
            with Vertical(id="setup-box"):
                yield Static("", id="setup-title")
                yield Static("", id="setup-step-indicator", classes="setup-step-indicator")
                yield Vertical(id="setup-body")
                with Horizontal(id="setup-nav"):
                    yield Button("Back", id="back-btn", variant="default")
                    yield Button("Next", id="next-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._render_step()

    def _render_step(self) -> None:
        """Render the current step content."""
        title = self.query_one("#setup-title", Static)
        indicator = self.query_one("#setup-step-indicator", Static)
        body = self.query_one("#setup-body", Vertical)
        back_btn = self.query_one("#back-btn", Button)
        next_btn = self.query_one("#next-btn", Button)

        # Step indicator
        step_num = self._step_index + 1
        total = len(STEPS)
        indicator.update(f"Step {step_num} of {total}")

        # Navigation visibility
        back_btn.display = self._step_index > 0
        next_btn.label = "Begin" if self.current_step == "confirmation" else "Next"

        # Clear body
        body.remove_children()

        # Render step-specific content
        step = self.current_step
        if step == "welcome":
            title.update("Welcome")
            body.mount(Static(
                STEP_DESCRIPTIONS["welcome"],
                id="welcome-text",
            ))
        elif step == "name":
            title.update("Identity")
            body.mount(Static(STEP_DESCRIPTIONS["name"]))
            body.mount(Input(
                value=self._config.name,
                placeholder="Mother",
                id="name-input",
            ))
        elif step == "user_name":
            title.update("Your Name")
            body.mount(Static(STEP_DESCRIPTIONS["user_name"]))
            body.mount(Input(
                value=getattr(self._config, "user_name", "User"),
                placeholder="User",
                id="user-name-input",
            ))
        elif step == "personality":
            title.update("Personality")
            body.mount(Static(STEP_DESCRIPTIONS["personality"]))
            radio_set = RadioSet(id="personality-radio")
            for i, (key, desc) in enumerate(PERSONALITY_DESCRIPTIONS.items()):
                btn = RadioButton(f"{key.capitalize()} — {desc}", id=f"personality-{key}")
                radio_set.compose_add_child(btn)
            body.mount(radio_set)
        elif step == "provider":
            title.update("Provider")
            body.mount(Static(STEP_DESCRIPTIONS["provider"]))
            radio_set = RadioSet(id="provider-radio")
            for provider in PROVIDERS:
                if provider == "local":
                    # Check Ollama availability instead of API key
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
                    id=f"provider-{provider}",
                )
                radio_set.compose_add_child(btn)
            body.mount(radio_set)
        elif step == "api_key":
            title.update("API Key")
            if self._config.provider == "local":
                # Local provider: show Ollama status, no API key needed
                try:
                    from core.llm import OllamaClient
                    ollama_up = OllamaClient.is_available()
                except Exception:
                    ollama_up = False
                if ollama_up:
                    body.mount(Static(
                        "Ollama is running. No API key needed — all processing stays local.",
                        id="key-status",
                    ))
                else:
                    body.mount(Static(
                        "Ollama not detected. Start it with: ollama serve",
                        id="key-status",
                    ))
            else:
                env_var = ENV_VARS.get(self._config.provider, "ANTHROPIC_API_KEY")
                has_env = bool(os.environ.get(env_var))
                if has_env:
                    body.mount(Static(
                        f"Found {env_var} in your environment. You're all set.",
                        id="key-status",
                    ))
                else:
                    body.mount(Static(f"Enter your {env_var}:"))
                    body.mount(Input(
                        placeholder="sk-...",
                        password=True,
                        id="api-key-input",
                    ))
            if self._validation_error:
                body.mount(Static(
                    self._validation_error,
                    id="validation-error",
                ))
        elif step == "permissions":
            title.update("Permissions")
            body.mount(Static(STEP_DESCRIPTIONS["permissions"]))
            body.mount(Horizontal(
                Static("File access  ", classes="perm-label"),
                Switch(value=self._config.file_access, id="perm-file"),
            ))
            body.mount(Horizontal(
                Static("Auto-compile ", classes="perm-label"),
                Switch(value=self._config.auto_compile, id="perm-compile"),
            ))
            body.mount(Static(f"Cost limit per compilation: ${self._config.cost_limit:.2f}"))
        elif step == "voice":
            title.update("Voice")
            body.mount(Static(STEP_DESCRIPTIONS["voice"]))
            body.mount(Horizontal(
                Static("Enable voice ", classes="perm-label"),
                Switch(value=self._config.voice_enabled, id="voice-enable"),
            ))
            body.mount(Static("ElevenLabs API key:"))
            body.mount(Input(
                value=self._config.api_keys.get("elevenlabs", ""),
                placeholder="sk_...",
                password=True,
                id="voice-key-input",
            ))
        elif step == "screen_capture":
            title.update("Screen Capture")
            body.mount(Static(STEP_DESCRIPTIONS["screen_capture"]))
            body.mount(Horizontal(
                Static("Enable screen capture ", classes="perm-label"),
                Switch(value=self._config.screen_capture_enabled, id="screen-capture-enable"),
            ))
        elif step == "microphone":
            title.update("Microphone")
            body.mount(Static(STEP_DESCRIPTIONS["microphone"]))
            body.mount(Horizontal(
                Static("Enable microphone ", classes="perm-label"),
                Switch(value=self._config.microphone_enabled, id="mic-enable"),
            ))
            # OpenAI key for Whisper (if not already set as provider key)
            existing_key = self._config.api_keys.get("openai", "")
            if self._config.provider == "openai" and self._config.api_keys.get("openai"):
                body.mount(Static("OpenAI key already set from provider step."))
            else:
                body.mount(Static("OpenAI API key (for Whisper transcription):"))
                body.mount(Input(
                    value=existing_key,
                    placeholder="sk-...",
                    password=True,
                    id="openai-key-input",
                ))
        elif step == "camera":
            title.update("Camera")
            body.mount(Static(STEP_DESCRIPTIONS["camera"]))
            body.mount(Horizontal(
                Static("Enable camera ", classes="perm-label"),
                Switch(value=self._config.camera_enabled, id="camera-enable"),
            ))
        elif step == "ambient_listening":
            title.update("Ambient Listening")
            body.mount(Static(STEP_DESCRIPTIONS["ambient_listening"]))
            body.mount(Horizontal(
                Static("Enable ambient listening ", classes="perm-label"),
                Switch(value=self._config.ambient_listening, id="ambient-listen-enable"),
            ))
        elif step == "screen_monitoring":
            title.update("Screen Monitoring")
            body.mount(Static(STEP_DESCRIPTIONS["screen_monitoring"]))
            body.mount(Horizontal(
                Static("Enable screen monitoring ", classes="perm-label"),
                Switch(value=self._config.screen_monitoring, id="screen-monitor-enable"),
            ))
            body.mount(Static(f"Perception budget: ${self._config.perception_budget:.2f}/hour"))
            body.mount(Input(
                value=str(self._config.perception_budget),
                placeholder="0.50",
                id="perception-budget-input",
            ))
        elif step == "confirmation":
            title.update("Ready")
            voice_str = "yes" if self._config.voice_enabled else "no"
            screen_str = "yes" if self._config.screen_capture_enabled else "no"
            mic_str = "yes" if self._config.microphone_enabled else "no"
            camera_str = "yes" if self._config.camera_enabled else "no"
            ambient_str = "yes" if self._config.ambient_listening else "no"
            monitor_str = "yes" if self._config.screen_monitoring else "no"
            summary = (
                f"Name: {self._config.name}\n"
                f"Your name: {getattr(self._config, 'user_name', 'User')}\n"
                f"Personality: {self._config.personality}\n"
                f"Provider: {self._config.provider}\n"
                f"Model: {self._config.get_model()}\n"
                f"File access: {'yes' if self._config.file_access else 'no'}\n"
                f"Auto-compile: {'yes' if self._config.auto_compile else 'no'}\n"
                f"Voice: {voice_str}\n"
                f"Screen capture: {screen_str}\n"
                f"Microphone: {mic_str}\n"
                f"Camera: {camera_str}\n"
                f"Ambient listening: {ambient_str}\n"
                f"Screen monitoring: {monitor_str}\n"
                f"Perception budget: ${self._config.perception_budget:.2f}/hr\n"
                f"Cost limit: ${self._config.cost_limit:.2f}"
            )
            body.mount(Static(summary, id="summary-text"))
            body.mount(Static(f"\n{STEP_DESCRIPTIONS['confirmation']}", id="confirm-prompt"))

    def _collect_step_data(self) -> bool:
        """Collect data from the current step. Returns False if validation fails."""
        step = self.current_step
        self._validation_error = None

        if step == "name":
            try:
                inp = self.query_one("#name-input", Input)
                name = inp.value.strip()
                if name:
                    self._config.name = name
            except Exception:
                pass

        elif step == "user_name":
            try:
                inp = self.query_one("#user-name-input", Input)
                user_name = inp.value.strip()
                if user_name:
                    self._config.user_name = user_name
            except Exception:
                pass

        elif step == "personality":
            try:
                radio = self.query_one("#personality-radio", RadioSet)
                if radio.pressed_index >= 0:
                    keys = list(PERSONALITY_DESCRIPTIONS.keys())
                    self._config.personality = keys[radio.pressed_index]
            except Exception:
                pass

        elif step == "provider":
            try:
                radio = self.query_one("#provider-radio", RadioSet)
                if radio.pressed_index >= 0:
                    self._config.provider = PROVIDERS[radio.pressed_index]
                    self._config.model = DEFAULT_MODELS[self._config.provider]
            except Exception:
                pass

        elif step == "api_key":
            if self._config.provider == "local":
                pass  # No API key needed for local Ollama
            else:
                env_var = ENV_VARS.get(self._config.provider, "ANTHROPIC_API_KEY")
                has_env = bool(os.environ.get(env_var))
                if not has_env:
                    try:
                        inp = self.query_one("#api-key-input", Input)
                        key = inp.value.strip()
                        if not key:
                            self._validation_error = "API key is required."
                            return False
                        self._config.api_keys[self._config.provider] = key
                    except Exception:
                        pass

        elif step == "permissions":
            try:
                self._config.file_access = self.query_one("#perm-file", Switch).value
            except Exception:
                pass
            try:
                self._config.auto_compile = self.query_one("#perm-compile", Switch).value
            except Exception:
                pass

        elif step == "voice":
            try:
                self._config.voice_enabled = self.query_one("#voice-enable", Switch).value
            except Exception:
                pass
            try:
                key = self.query_one("#voice-key-input", Input).value.strip()
                if key:
                    self._config.api_keys["elevenlabs"] = key
            except Exception:
                pass

        elif step == "screen_capture":
            try:
                self._config.screen_capture_enabled = self.query_one("#screen-capture-enable", Switch).value
            except Exception:
                pass

        elif step == "microphone":
            try:
                self._config.microphone_enabled = self.query_one("#mic-enable", Switch).value
            except Exception:
                pass
            try:
                key = self.query_one("#openai-key-input", Input).value.strip()
                if key:
                    self._config.api_keys["openai"] = key
            except Exception:
                pass

        elif step == "camera":
            try:
                self._config.camera_enabled = self.query_one("#camera-enable", Switch).value
            except Exception:
                pass

        elif step == "ambient_listening":
            try:
                self._config.ambient_listening = self.query_one("#ambient-listen-enable", Switch).value
            except Exception:
                pass

        elif step == "screen_monitoring":
            try:
                self._config.screen_monitoring = self.query_one("#screen-monitor-enable", Switch).value
            except Exception:
                pass
            try:
                budget_str = self.query_one("#perception-budget-input", Input).value.strip()
                if budget_str:
                    self._config.perception_budget = float(budget_str)
            except (ValueError, Exception):
                pass

        return True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next-btn":
            self._go_next()
        elif event.button.id == "back-btn":
            self._go_back()

    def _go_next(self) -> None:
        """Advance to next step or complete setup."""
        if not self._collect_step_data():
            self._render_step()
            return

        if self._step_index >= len(STEPS) - 1:
            self._complete_setup()
        else:
            self._step_index += 1
            self._render_step()

    def _go_back(self) -> None:
        """Go to previous step."""
        if self._step_index > 0:
            self._step_index -= 1
            self._render_step()

    def _complete_setup(self) -> None:
        """Save config and transition to chat."""
        self._config.setup_complete = True
        save_config(self._config, self._config_path)

        from mother.screens.chat import ChatScreen
        self.app.switch_screen(ChatScreen(config=self._config, config_path=self._config_path))

    def action_cancel(self) -> None:
        """Cancel setup — quit app."""
        self.app.exit()
