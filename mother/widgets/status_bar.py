"""
Status bar widget — instance name, state, session cost, modality indicators.

Docked to top of chat screen. Reactive attributes for live updates.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.reactive import reactive


class StatusBar(Horizontal):
    """Top status bar showing instance info."""

    instance_name: reactive[str] = reactive("Mother")
    state: reactive[str] = reactive("ready")
    session_cost: reactive[float] = reactive(0.0)
    provider: reactive[str] = reactive("")
    model: reactive[str] = reactive("")
    mic_active: reactive[bool] = reactive(False)
    screen_active: reactive[bool] = reactive(False)
    camera_active: reactive[bool] = reactive(False)
    voice_active: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(id="status-bar", **kwargs)

    def compose(self) -> ComposeResult:
        yield Static(self.instance_name, id="status-name")
        yield Static("", id="status-modalities")
        yield Static(self.state, id="status-state")
        yield Static("", id="status-cost")

    def _render_modalities(self) -> str:
        """Build compact modality indicator string."""
        indicators = []
        if self.mic_active:
            indicators.append("\U0001f3a4")
        if self.camera_active:
            indicators.append("\U0001f4f7")
        if self.screen_active:
            indicators.append("\U0001f5a5")
        if self.voice_active:
            indicators.append("\U0001f50a")
        return " ".join(indicators)

    def _update_modalities_display(self) -> None:
        try:
            self.query_one("#status-modalities", Static).update(self._render_modalities())
        except Exception:
            pass

    def watch_instance_name(self, value: str) -> None:
        try:
            self.query_one("#status-name", Static).update(value)
        except Exception:
            pass

    def watch_state(self, value: str) -> None:
        try:
            self.query_one("#status-state", Static).update(value)
        except Exception:
            pass

    def watch_session_cost(self, value: float) -> None:
        try:
            cost_str = f"${value:.4f}" if value > 0 else ""
            model_str = self.model or self.provider
            suffix = f" ({model_str})" if model_str else ""
            self.query_one("#status-cost", Static).update(f"{cost_str}{suffix}")
        except Exception:
            pass

    def watch_mic_active(self, value: bool) -> None:
        self._update_modalities_display()

    def watch_screen_active(self, value: bool) -> None:
        self._update_modalities_display()

    def watch_camera_active(self, value: bool) -> None:
        self._update_modalities_display()

    def watch_voice_active(self, value: bool) -> None:
        self._update_modalities_display()
