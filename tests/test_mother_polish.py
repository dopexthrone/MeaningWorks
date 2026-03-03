"""
Phase 5: Tests for polish — error handling, cost caps, empty states, keybindings.
"""

import pytest
from unittest.mock import MagicMock, patch

from mother.app import MotherApp, LoadingScreen, main
from mother.config import MotherConfig
from mother.screens.chat import ChatScreen, GREETING, HELP_TEXT
from mother.bridge import EngineBridge
from mother.widgets.pipeline import PipelinePanel
from mother.widgets.trust_badge import TrustBadge, trust_level


class TestErrorHandling:
    """Test graceful error paths."""

    def test_bridge_handles_empty_usage(self):
        bridge = EngineBridge()
        bridge._track_cost({})
        assert bridge.get_session_cost() == 0.0

    def test_cost_cap_config(self):
        config = MotherConfig(cost_limit=1.0)
        assert config.cost_limit == 1.0

    def test_default_cost_limit(self):
        config = MotherConfig()
        assert config.cost_limit == 100.0


class TestEmptyStates:
    """Test empty/initial states."""

    def test_chat_screen_no_config(self):
        screen = ChatScreen()
        assert screen._config.name == "Mother"

    def test_pipeline_starts_all_pending(self):
        panel = PipelinePanel()
        assert all(s.status == "pending" for s in panel.stages)

    def test_trust_badge_starts_zero(self):
        badge = TrustBadge()
        assert badge.score == 0.0
        assert badge.level == "unverified"


class TestKeybindings:
    """Test global and screen keybindings."""

    def test_app_quit_binding(self):
        app = MotherApp()
        keys = {b.key for b in app.BINDINGS}
        assert "ctrl+q" in keys

    def test_app_settings_binding(self):
        app = MotherApp()
        keys = {b.key for b in app.BINDINGS}
        assert "ctrl+comma" in keys

    def test_chat_screen_quit_binding(self):
        screen = ChatScreen()
        keys = {b[0] if isinstance(b, tuple) else getattr(b, 'key', b) for b in screen.BINDINGS}
        assert "ctrl+q" in keys


class TestLoadingScreen:
    """Test loading screen."""

    def test_loading_screen_creates(self):
        screen = LoadingScreen()
        assert screen is not None


class TestCostTracking:
    """Test session cost tracking edge cases."""

    def test_very_large_token_count(self):
        bridge = EngineBridge()
        bridge._track_cost({"input_tokens": 100000, "output_tokens": 50000})
        assert bridge.get_session_cost() > 0

    def test_negative_tokens_handled(self):
        bridge = EngineBridge()
        bridge._track_cost({"input_tokens": -1, "output_tokens": -1})
        # Should not crash, cost may be negative but no exception
        assert isinstance(bridge.get_session_cost(), float)
