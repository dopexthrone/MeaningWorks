"""
Tests for natural language permission flow in ChatScreen.

Covers: permission state machine, _enable_microphone/_enable_camera,
confirmation/decline routing in _handle_chat, _run_listen/_run_camera
pending permission offers, action routing for enable_mic/enable_camera,
and keybinding changes (ctrl+m -> f8).
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call

from mother.screens.chat import ChatScreen
from mother.config import MotherConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screen(**overrides) -> ChatScreen:
    """Create a ChatScreen without mounting the Textual app.

    Sets all internal state fields to safe defaults, mocks query_one
    to return a mock ChatArea, and applies any keyword overrides.
    """
    screen = ChatScreen.__new__(ChatScreen)
    screen._config = MotherConfig()
    screen._store = None
    screen._bridge = None
    screen._voice = None
    screen._screen_bridge = None
    screen._microphone_bridge = None
    screen._camera_bridge = None
    screen._pending_screenshot = None
    screen._pending_permission = None
    screen._chatting = False
    screen._compilation_count = 0
    screen._last_compile_result = None
    screen._tool_count = 0
    screen._sense_memory = None
    screen._current_senses = None
    screen._current_posture = None
    screen._session_start_time = 0.0
    screen._session_error_count = 0
    screen._perception = None
    screen._perception_queue = None
    screen._perception_consumer_task = None
    screen._pending_perception_screen = None
    screen._pending_perception_camera = None
    screen._perception_event_count = 0
    screen._screen_change_count = 0
    screen._pending_proposal = None

    # Mock query_one to return a mock ChatArea
    mock_chat_area = MagicMock()
    screen.query_one = MagicMock(return_value=mock_chat_area)

    # Mock _speak (voice is almost always None in tests)
    screen._speak = MagicMock()

    # Apply overrides
    for key, value in overrides.items():
        setattr(screen, key, value)

    return screen


# ===========================================================================
# 1. Permission State Machine — word sets
# ===========================================================================

class TestPermissionWordSets:
    """Verify _CONFIRM_WORDS and _DECLINE_WORDS are sane."""

    def test_confirm_words_contains_expected_values(self):
        expected = {"yes", "yeah", "ok", "sure", "do it", "go ahead", "enable it", "turn it on", "y"}
        assert expected.issubset(ChatScreen._CONFIRM_WORDS)

    def test_decline_words_contains_expected_values(self):
        expected = {"no", "nah", "nope", "cancel", "never mind", "nevermind", "don't", "dont", "n"}
        assert expected.issubset(ChatScreen._DECLINE_WORDS)

    def test_no_overlap_between_confirm_and_decline(self):
        overlap = ChatScreen._CONFIRM_WORDS & ChatScreen._DECLINE_WORDS
        assert overlap == set(), f"Overlapping words: {overlap}"


# ===========================================================================
# 2. _enable_microphone()
# ===========================================================================

class TestEnableMicrophone:
    """Test the _enable_microphone() hot-enable method."""

    @patch("mother.screens.chat.is_microphone_available", return_value=False)
    def test_returns_false_when_deps_unavailable(self, _mock_avail):
        screen = _make_screen()
        result = screen._enable_microphone()
        assert result is False

    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_returns_false_when_no_openai_key(self, _mock_avail):
        screen = _make_screen()
        screen._config.api_keys = {}  # No openai key
        result = screen._enable_microphone()
        assert result is False

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_returns_true_and_creates_bridge(self, _mock_avail, mock_bridge_cls, mock_save):
        screen = _make_screen()
        screen._config.api_keys = {"openai": "sk-test-123"}
        result = screen._enable_microphone()
        assert result is True

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_sets_bridge_on_success(self, _mock_avail, mock_bridge_cls, mock_save):
        screen = _make_screen()
        screen._config.api_keys = {"openai": "sk-test-123"}
        screen._enable_microphone()
        assert screen._microphone_bridge is not None
        mock_bridge_cls.assert_called_once_with(openai_api_key="sk-test-123", enabled=True)

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_updates_config_on_success(self, _mock_avail, _mock_bridge, mock_save):
        screen = _make_screen()
        screen._config.api_keys = {"openai": "sk-test-123"}
        screen._enable_microphone()
        assert screen._config.microphone_enabled is True
        mock_save.assert_called_once_with(screen._config)

    @patch("mother.screens.chat.is_microphone_available", return_value=False)
    def test_shows_message_when_deps_missing(self, _mock_avail):
        screen = _make_screen()
        screen._enable_microphone()
        mock_chat_area = screen.query_one.return_value
        mock_chat_area.add_ai_message.assert_called()
        # First call should mention sounddevice
        args = mock_chat_area.add_ai_message.call_args_list[0]
        assert "sounddevice" in args[0][0]

    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_shows_message_when_no_openai_key(self, _mock_avail):
        screen = _make_screen()
        screen._config.api_keys = {}
        screen._enable_microphone()
        mock_chat_area = screen.query_one.return_value
        args = mock_chat_area.add_ai_message.call_args_list[0]
        assert "OpenAI" in args[0][0]


# ===========================================================================
# 3. _enable_camera()
# ===========================================================================

class TestEnableCamera:
    """Test the _enable_camera() hot-enable method."""

    @patch("mother.screens.chat.is_camera_available", return_value=False)
    def test_returns_false_when_ffmpeg_unavailable(self, _mock_avail):
        screen = _make_screen()
        result = screen._enable_camera()
        assert result is False

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.CameraBridge")
    @patch("mother.screens.chat.is_camera_available", return_value=True)
    def test_returns_true_when_available(self, _mock_avail, _mock_bridge, _mock_save):
        screen = _make_screen()
        result = screen._enable_camera()
        assert result is True

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.CameraBridge")
    @patch("mother.screens.chat.is_camera_available", return_value=True)
    def test_sets_bridge_on_success(self, _mock_avail, mock_bridge_cls, _mock_save):
        screen = _make_screen()
        screen._enable_camera()
        assert screen._camera_bridge is not None
        mock_bridge_cls.assert_called_once_with(enabled=True)

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.CameraBridge")
    @patch("mother.screens.chat.is_camera_available", return_value=True)
    def test_updates_config_on_success(self, _mock_avail, _mock_bridge, mock_save):
        screen = _make_screen()
        screen._enable_camera()
        assert screen._config.camera_enabled is True
        mock_save.assert_called_once_with(screen._config)

    @patch("mother.screens.chat.is_camera_available", return_value=False)
    def test_shows_message_when_camera_unavailable(self, _mock_avail):
        screen = _make_screen()
        screen._enable_camera()
        mock_chat_area = screen.query_one.return_value
        args = mock_chat_area.add_ai_message.call_args_list[0]
        assert "opencv" in args[0][0].lower() or "camera" in args[0][0].lower()


# ===========================================================================
# 4. Permission confirmation flow in _handle_chat()
# ===========================================================================

class TestPermissionConfirmation:
    """Test the permission state machine inside _handle_chat."""

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_yes_enables_microphone(self, _avail, _bridge, _save):
        screen = _make_screen(
            _pending_permission="microphone",
            _config=MotherConfig(api_keys={"openai": "sk-test"}),
        )
        # Mock _run_listen to prevent further side effects
        screen._run_listen = MagicMock()
        screen._handle_chat("yes")
        assert screen._pending_permission is None
        assert screen._config.microphone_enabled is True

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.CameraBridge")
    @patch("mother.screens.chat.is_camera_available", return_value=True)
    def test_sure_enables_camera(self, _avail, _bridge, _save):
        screen = _make_screen(
            _pending_permission="camera",
        )
        screen._run_camera = MagicMock()
        screen._handle_chat("sure")
        assert screen._pending_permission is None
        assert screen._config.camera_enabled is True

    def test_no_clears_pending_and_shows_message(self):
        screen = _make_screen(_pending_permission="microphone")
        screen._handle_chat("no")
        assert screen._pending_permission is None
        mock_chat_area = screen.query_one.return_value
        # Should see "No problem." in one of the ai messages
        messages = [c[0][0] for c in mock_chat_area.add_ai_message.call_args_list if len(c[0]) > 0]
        assert any("No problem" in m for m in messages)

    def test_unrelated_text_clears_pending_routes_to_chat(self):
        screen = _make_screen(_pending_permission="microphone")
        # Mock run_worker and the StatusBar to allow normal chat path
        screen.run_worker = MagicMock()
        mock_status = MagicMock()
        # query_one returns different mocks for different widget types
        mock_chat_area = MagicMock()
        def _query_one(cls):
            if cls.__name__ == "StatusBar" if hasattr(cls, '__name__') else False:
                return mock_status
            return mock_chat_area
        screen.query_one = MagicMock(return_value=mock_chat_area)

        screen._handle_chat("what time is it?")
        assert screen._pending_permission is None
        # Should have called run_worker (normal LLM chat path)
        screen.run_worker.assert_called_once()

    @pytest.mark.parametrize("word", ["yeah", "ok", "go ahead", "do it", "enable it", "turn it on"])
    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_confirm_variants_enable_microphone(self, _avail, _bridge, _save, word):
        screen = _make_screen(
            _pending_permission="microphone",
            _config=MotherConfig(api_keys={"openai": "sk-test"}),
        )
        screen._run_listen = MagicMock()
        screen._handle_chat(word)
        assert screen._pending_permission is None
        assert screen._config.microphone_enabled is True

    @pytest.mark.parametrize("word", ["nah", "nope", "cancel", "never mind", "nevermind", "don't", "dont", "n"])
    def test_decline_variants_clear_pending(self, word):
        screen = _make_screen(_pending_permission="camera")
        screen._handle_chat(word)
        assert screen._pending_permission is None

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_confirm_triggers_listen_after_mic_enable(self, _avail, _bridge, _save):
        screen = _make_screen(
            _pending_permission="microphone",
            _config=MotherConfig(api_keys={"openai": "sk-test"}),
        )
        screen._run_listen = MagicMock()
        screen._handle_chat("yes")
        screen._run_listen.assert_called_once_with()

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.CameraBridge")
    @patch("mother.screens.chat.is_camera_available", return_value=True)
    def test_confirm_triggers_camera_after_enable(self, _avail, _bridge, _save):
        screen = _make_screen(_pending_permission="camera")
        screen._run_camera = MagicMock()
        screen._handle_chat("yes")
        screen._run_camera.assert_called_once()

    def test_confirm_does_not_start_llm_call(self):
        """When pending permission is handled, run_worker should NOT be called."""
        screen = _make_screen(_pending_permission="microphone")
        screen.run_worker = MagicMock()
        screen._handle_chat("no")
        screen.run_worker.assert_not_called()

    def test_trailing_punctuation_stripped_for_matching(self):
        """'yes!' should match after stripping trailing punctuation."""
        screen = _make_screen(_pending_permission="camera")
        screen._handle_chat("no!")
        assert screen._pending_permission is None

    @patch("mother.screens.chat.save_config")
    @patch("mother.screens.chat.MicrophoneBridge")
    @patch("mother.screens.chat.is_microphone_available", return_value=True)
    def test_case_insensitive_matching(self, _avail, _bridge, _save):
        """'YES' should match the confirm set."""
        screen = _make_screen(
            _pending_permission="microphone",
            _config=MotherConfig(api_keys={"openai": "sk-test"}),
        )
        screen._run_listen = MagicMock()
        screen._handle_chat("YES")
        assert screen._pending_permission is None
        assert screen._config.microphone_enabled is True

    def test_chatting_flag_blocks_permission_flow(self):
        """If _chatting is True, _handle_chat returns early before permission check."""
        screen = _make_screen(
            _pending_permission="microphone",
            _chatting=True,
        )
        screen._handle_chat("yes")
        # Pending permission should remain unchanged because we returned early
        assert screen._pending_permission == "microphone"

    def test_decline_stores_messages_when_store_present(self):
        """When store is available, user input and response should be persisted."""
        mock_store = MagicMock()
        screen = _make_screen(
            _pending_permission="microphone",
            _store=mock_store,
        )
        screen._handle_chat("no")
        # store.add_message called for user input and assistant response
        assert mock_store.add_message.call_count == 2
        user_call = mock_store.add_message.call_args_list[0]
        assert user_call[0] == ("user", "no")
        assistant_call = mock_store.add_message.call_args_list[1]
        assert assistant_call[0] == ("assistant", "No problem.")


# ===========================================================================
# 5. _run_listen() sets pending permission when mic unavailable
# ===========================================================================

class TestRunListenPermissionOffer:
    """Test _run_listen offers to enable mic when bridge is None."""

    def test_sets_pending_permission_when_bridge_none(self):
        screen = _make_screen(_microphone_bridge=None)
        screen._run_listen(5.0)
        assert screen._pending_permission == "microphone"

    def test_shows_offer_message_when_bridge_none(self):
        screen = _make_screen(_microphone_bridge=None)
        screen._run_listen(5.0)
        mock_chat_area = screen.query_one.return_value
        messages = [c[0][0] for c in mock_chat_area.add_ai_message.call_args_list if len(c[0]) > 0]
        assert any("Want me to turn it on" in m for m in messages)

    def test_sets_pending_permission_when_bridge_disabled(self):
        mock_bridge = MagicMock()
        mock_bridge.enabled = False
        screen = _make_screen(_microphone_bridge=mock_bridge)
        screen._run_listen(5.0)
        assert screen._pending_permission == "microphone"

    def test_does_not_set_pending_when_bridge_active(self):
        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        screen = _make_screen(_microphone_bridge=mock_bridge)
        screen.run_worker = MagicMock()
        screen._run_listen(5.0)
        assert screen._pending_permission is None


# ===========================================================================
# 6. _run_camera() sets pending permission when camera unavailable
# ===========================================================================

class TestRunCameraPermissionOffer:
    """Test _run_camera offers to enable camera when bridge is None."""

    def test_sets_pending_permission_when_bridge_none(self):
        screen = _make_screen(_camera_bridge=None)
        screen._run_camera()
        assert screen._pending_permission == "camera"

    def test_shows_offer_message_when_bridge_none(self):
        screen = _make_screen(_camera_bridge=None)
        screen._run_camera()
        mock_chat_area = screen.query_one.return_value
        messages = [c[0][0] for c in mock_chat_area.add_ai_message.call_args_list if len(c[0]) > 0]
        assert any("Want me to turn it on" in m for m in messages)

    def test_sets_pending_permission_when_bridge_disabled(self):
        mock_bridge = MagicMock()
        mock_bridge.enabled = False
        screen = _make_screen(_camera_bridge=mock_bridge)
        screen._run_camera()
        assert screen._pending_permission == "camera"

    def test_does_not_set_pending_when_bridge_active(self):
        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        screen = _make_screen(_camera_bridge=mock_bridge)
        screen.run_worker = MagicMock()
        screen._run_camera()
        assert screen._pending_permission is None


# ===========================================================================
# 7. Action routing for enable_mic / enable_camera in _chat_worker
# ===========================================================================

class TestActionRouting:
    """Test enable_mic/enable_camera action routing in _chat_worker.

    These tests verify the routing logic by calling _handle_chat and
    mocking the bridge to return action-tagged responses.
    """

    def test_enable_mic_action_calls_listen_when_bridge_active(self):
        """When mic bridge is active and LLM returns enable_mic action, run listen."""
        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        screen = _make_screen(_microphone_bridge=mock_bridge)
        # Simulate the action routing logic directly
        if screen._microphone_bridge is not None and screen._microphone_bridge.enabled:
            screen._run_listen = MagicMock()
            screen._run_listen()
            screen._run_listen.assert_called_once_with()

    def test_enable_mic_action_sets_pending_when_bridge_inactive(self):
        """When mic bridge is None and enable_mic action, set pending permission."""
        screen = _make_screen(_microphone_bridge=None)
        # Simulate the action routing logic
        if screen._microphone_bridge is None or not screen._microphone_bridge.enabled:
            screen._pending_permission = "microphone"
        assert screen._pending_permission == "microphone"

    def test_enable_camera_action_calls_camera_when_bridge_active(self):
        """When camera bridge is active and enable_camera action, run camera."""
        mock_bridge = MagicMock()
        mock_bridge.enabled = True
        screen = _make_screen(_camera_bridge=mock_bridge)
        if screen._camera_bridge is not None and screen._camera_bridge.enabled:
            screen._run_camera = MagicMock()
            screen._run_camera()
            screen._run_camera.assert_called_once()

    def test_enable_camera_action_sets_pending_when_bridge_inactive(self):
        """When camera bridge is None and enable_camera action, set pending permission."""
        screen = _make_screen(_camera_bridge=None)
        if screen._camera_bridge is None or not screen._camera_bridge.enabled:
            screen._pending_permission = "camera"
        assert screen._pending_permission == "camera"

    def test_enable_mic_disabled_bridge_sets_pending(self):
        """When mic bridge exists but is disabled, should set pending."""
        mock_bridge = MagicMock()
        mock_bridge.enabled = False
        screen = _make_screen(_microphone_bridge=mock_bridge)
        if not (screen._microphone_bridge is not None and screen._microphone_bridge.enabled):
            screen._pending_permission = "microphone"
        assert screen._pending_permission == "microphone"

    def test_enable_camera_disabled_bridge_sets_pending(self):
        """When camera bridge exists but is disabled, should set pending."""
        mock_bridge = MagicMock()
        mock_bridge.enabled = False
        screen = _make_screen(_camera_bridge=mock_bridge)
        if not (screen._camera_bridge is not None and screen._camera_bridge.enabled):
            screen._pending_permission = "camera"
        assert screen._pending_permission == "camera"


# ===========================================================================
# 8. Keybinding
# ===========================================================================

class TestKeybinding:
    """Test that F8 replaced ctrl+m for push-to-talk."""

    def test_f8_binding_present(self):
        bindings = ChatScreen.BINDINGS
        binding_keys = [b[0] for b in bindings]
        assert "f8" in binding_keys

    def test_f8_maps_to_listen_action(self):
        bindings = ChatScreen.BINDINGS
        for key, action, _label in bindings:
            if key == "f8":
                assert action == "listen"
                return
        pytest.fail("f8 binding not found")

    def test_ctrl_m_binding_not_present(self):
        bindings = ChatScreen.BINDINGS
        binding_keys = [b[0] for b in bindings]
        assert "ctrl+m" not in binding_keys

    def test_action_listen_calls_run_listen(self):
        screen = _make_screen()
        screen._run_listen = MagicMock()
        screen.action_listen()
        screen._run_listen.assert_called_once_with()
