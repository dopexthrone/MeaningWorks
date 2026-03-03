"""
Tests for Mother voice integration — ElevenLabs TTS.

All tests mocked. No real API calls, no audio playback.
"""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from mother.config import MotherConfig, save_config, load_config


# ---------------------------------------------------------------------------
# Test voice module availability detection
# ---------------------------------------------------------------------------

class TestVoiceAvailability:
    """Test SDK detection and is_voice_available()."""

    def test_is_voice_available_returns_bool(self):
        from mother.voice import is_voice_available
        result = is_voice_available()
        assert isinstance(result, bool)

    def test_default_voice_id_set(self):
        from mother.voice import DEFAULT_VOICE_ID
        assert DEFAULT_VOICE_ID == "2obv5y63xKRNiEZAPxGD"

    def test_default_model_id_set(self):
        from mother.voice import DEFAULT_MODEL_ID
        assert DEFAULT_MODEL_ID == "eleven_v3"


# ---------------------------------------------------------------------------
# Test VoiceBridge creation
# ---------------------------------------------------------------------------

class TestVoiceBridgeCreation:
    """Test VoiceBridge instantiation and defaults."""

    def test_create_with_api_key(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key")
        assert bridge._api_key == "test-key"

    def test_default_voice_id(self):
        from mother.voice import VoiceBridge, DEFAULT_VOICE_ID
        bridge = VoiceBridge(api_key="test-key")
        assert bridge._voice_id == DEFAULT_VOICE_ID

    def test_default_model_id(self):
        from mother.voice import VoiceBridge, DEFAULT_MODEL_ID
        bridge = VoiceBridge(api_key="test-key")
        assert bridge._model_id == DEFAULT_MODEL_ID

    def test_custom_voice_and_model(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(
            api_key="test-key",
            voice_id="custom-voice",
            model_id="custom-model",
        )
        assert bridge._voice_id == "custom-voice"
        assert bridge._model_id == "custom-model"


# ---------------------------------------------------------------------------
# Test VoiceBridge enabled property
# ---------------------------------------------------------------------------

class TestVoiceBridgeEnabled:
    """Test the enabled property logic."""

    def test_disabled_when_flag_false(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=False)
        assert bridge.enabled is False

    def test_disabled_when_no_api_key(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="", enabled=True)
        assert bridge.enabled is False

    @patch("mother.voice._ELEVENLABS_AVAILABLE", False)
    def test_disabled_when_sdk_missing(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        assert bridge.enabled is False


# ---------------------------------------------------------------------------
# Test VoiceBridge speak
# ---------------------------------------------------------------------------

class TestVoiceBridgeSpeak:
    """Test speak() method — async, catches all errors."""

    def test_noop_when_disabled(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=False)
        asyncio.run(bridge.speak("hello"))

    def test_noop_on_empty_text(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        asyncio.run(bridge.speak(""))

    def test_noop_on_whitespace_text(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        asyncio.run(bridge.speak("   "))

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_speak_catches_synthesis_error(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize = MagicMock(side_effect=RuntimeError("API down"))
        # Should not raise
        asyncio.run(bridge.speak("hello"))

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_speak_catches_playback_error(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize = MagicMock(return_value=b"fake audio")
        bridge._play_audio = MagicMock(side_effect=OSError("afplay failed"))
        asyncio.run(bridge.speak("hello"))

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_speak_calls_streaming_method(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize_and_play_streaming = MagicMock()
        asyncio.run(bridge.speak("hello"))
        bridge._synthesize_and_play_streaming.assert_called_once_with("hello", None)

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_speak_with_playback_rate(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize_and_play_streaming = MagicMock()
        asyncio.run(bridge.speak("hello", playback_rate=1.3))
        bridge._synthesize_and_play_streaming.assert_called_once_with("hello", 1.3)

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_speak_catches_connection_error(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize_and_play_streaming = MagicMock(
            side_effect=ConnectionError("no internet")
        )
        asyncio.run(bridge.speak("hello"))


# ---------------------------------------------------------------------------
# Test synthesize() and play() — public pipeline methods
# ---------------------------------------------------------------------------

class TestVoiceBridgePipeline:
    """Test synthesize() and play() — the split API for pipelined speech."""

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_synthesize_returns_audio_bytes(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize = MagicMock(return_value=b"fake audio")
        result = asyncio.run(bridge.synthesize("hello"))
        assert result == b"fake audio"
        bridge._synthesize.assert_called_once_with("hello")

    def test_synthesize_noop_when_disabled(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=False)
        result = asyncio.run(bridge.synthesize("hello"))
        assert result == b""

    def test_synthesize_noop_on_empty_text(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        result = asyncio.run(bridge.synthesize(""))
        assert result == b""

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_synthesize_catches_errors(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize = MagicMock(side_effect=RuntimeError("API down"))
        result = asyncio.run(bridge.synthesize("hello"))
        assert result == b""

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_play_calls_play_audio(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._play_audio = MagicMock()
        asyncio.run(bridge.play(b"audio data", playback_rate=1.2))
        bridge._play_audio.assert_called_once_with(b"audio data", 1.2)

    def test_play_noop_on_empty_bytes(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        # Should not raise
        asyncio.run(bridge.play(b""))

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_play_catches_errors(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._play_audio = MagicMock(side_effect=OSError("afplay died"))
        # Should not raise
        asyncio.run(bridge.play(b"audio data"))

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_pipeline_synthesize_then_play(self):
        """Full pipeline: synthesize then play separately."""
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        bridge._synthesize = MagicMock(return_value=b"sentence audio")
        bridge._play_audio = MagicMock()
        audio = asyncio.run(bridge.synthesize("First sentence."))
        asyncio.run(bridge.play(audio, playback_rate=1.15))
        bridge._synthesize.assert_called_once_with("First sentence.")
        bridge._play_audio.assert_called_once_with(b"sentence audio", 1.15)


# ---------------------------------------------------------------------------
# Test playback internals
# ---------------------------------------------------------------------------

class TestVoiceBridgePlayback:
    """Test _play_audio — temp file + afplay + cleanup."""

    def test_play_noop_on_empty_data(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key")
        bridge._play_audio(b"")  # Should return immediately

    @patch("mother.voice._build_play_cmd", return_value=["/usr/bin/afplay", "/tmp/test.mp3"])
    @patch("mother.voice.subprocess.Popen")
    def test_play_calls_audio_player(self, mock_popen, _mock_cmd):
        from mother.voice import VoiceBridge
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        bridge = VoiceBridge(api_key="test-key")
        bridge._play_audio(b"fake mp3 data")

        mock_popen.assert_called_once()

    @patch("mother.voice._build_play_cmd")
    @patch("mother.voice.subprocess.Popen")
    def test_play_cleans_up_temp_file(self, mock_popen, mock_cmd):
        from mother.voice import VoiceBridge
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Make _build_play_cmd return the actual tmp_path so we can verify cleanup
        def _passthrough_cmd(tmp_path, rate=1.0):
            return ["/usr/bin/afplay", tmp_path]
        mock_cmd.side_effect = _passthrough_cmd

        bridge = VoiceBridge(api_key="test-key")
        bridge._play_audio(b"fake mp3 data")

        # Temp file should be cleaned up
        args = mock_popen.call_args
        tmp_path = next(a for a in args[0][0] if a.endswith(".mp3"))
        assert not os.path.exists(tmp_path)

    @patch("mother.voice._build_play_cmd")
    @patch("mother.voice.subprocess.Popen")
    def test_play_cleans_up_even_on_error(self, mock_popen, mock_cmd):
        from mother.voice import VoiceBridge
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = OSError("afplay crashed")
        mock_popen.return_value = mock_proc

        def _passthrough_cmd(tmp_path, rate=1.0):
            return ["/usr/bin/afplay", tmp_path]
        mock_cmd.side_effect = _passthrough_cmd

        bridge = VoiceBridge(api_key="test-key")
        with pytest.raises(OSError):
            bridge._play_audio(b"fake mp3 data")

        # Temp file still cleaned up via finally
        args = mock_popen.call_args
        tmp_path = next(a for a in args[0][0] if a.endswith(".mp3"))
        assert not os.path.exists(tmp_path)

    @patch("mother.voice._build_play_cmd", return_value=["/usr/bin/afplay", "/tmp/test.mp3"])
    @patch("mother.voice.subprocess.Popen")
    def test_current_process_cleared_after_play(self, mock_popen, _mock_cmd):
        from mother.voice import VoiceBridge
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        bridge = VoiceBridge(api_key="test-key")
        bridge._play_audio(b"fake audio")
        assert bridge._current_process is None


# ---------------------------------------------------------------------------
# Test stop
# ---------------------------------------------------------------------------

class TestVoiceBridgeStop:
    """Test stop() — terminate current afplay process."""

    def test_stop_noop_when_idle(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key")
        asyncio.run(bridge.stop())  # No error

    def test_stop_terminates_process(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key")
        mock_proc = MagicMock()
        bridge._current_process = mock_proc
        asyncio.run(bridge.stop())
        mock_proc.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Test speak_fire_and_forget
# ---------------------------------------------------------------------------

class TestSpeakFireAndForget:
    """Test fire-and-forget background task creation."""

    def test_noop_when_disabled(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=False)
        result = bridge.speak_fire_and_forget("hello")
        assert result is None

    def test_noop_on_empty_text(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        result = bridge.speak_fire_and_forget("")
        assert result is None

    @patch("mother.voice._ELEVENLABS_AVAILABLE", True)
    def test_returns_none_without_event_loop(self):
        from mother.voice import VoiceBridge
        bridge = VoiceBridge(api_key="test-key", enabled=True)
        # No running event loop
        result = bridge.speak_fire_and_forget("hello")
        assert result is None


# ---------------------------------------------------------------------------
# Test voice config fields
# ---------------------------------------------------------------------------

class TestVoiceConfig:
    """Test voice-related config fields."""

    def test_voice_disabled_by_default(self):
        config = MotherConfig()
        assert config.voice_enabled is False

    def test_voice_id_default(self):
        config = MotherConfig()
        assert config.voice_id == "2obv5y63xKRNiEZAPxGD"

    def test_voice_model_default(self):
        config = MotherConfig()
        assert config.voice_model == "eleven_v3"

    def test_voice_config_roundtrip(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(
            voice_enabled=True,
            voice_id="custom-id",
            voice_model="eleven_v2",
            api_keys={"elevenlabs": "sk-test-123"},
        )
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.voice_enabled is True
        assert loaded.voice_id == "custom-id"
        assert loaded.voice_model == "eleven_v2"
        assert loaded.api_keys["elevenlabs"] == "sk-test-123"


# ---------------------------------------------------------------------------
# Test setup wizard voice step
# ---------------------------------------------------------------------------

class TestSetupVoiceStep:
    """Test voice step in setup wizard."""

    def test_steps_count_is_ten(self):
        from mother.screens.setup import STEPS
        assert len(STEPS) == 14

    def test_voice_step_exists(self):
        from mother.screens.setup import STEPS
        assert "voice" in STEPS

    def test_voice_step_after_permissions(self):
        from mother.screens.setup import STEPS
        voice_idx = STEPS.index("voice")
        assert STEPS[voice_idx - 1] == "permissions"
        assert STEPS[voice_idx + 1] == "screen_capture"

    def test_voice_step_description_exists(self):
        from mother.screens.setup import STEP_DESCRIPTIONS
        assert "voice" in STEP_DESCRIPTIONS
        assert "ElevenLabs" in STEP_DESCRIPTIONS["voice"]

    def test_all_steps_have_descriptions(self):
        from mother.screens.setup import STEPS, STEP_DESCRIPTIONS
        for step in STEPS:
            assert step in STEP_DESCRIPTIONS, f"Missing description for step: {step}"


# ---------------------------------------------------------------------------
# Test settings voice section
# ---------------------------------------------------------------------------

class TestSettingsVoice:
    """Test voice section in settings."""

    def test_voice_enabled_persists(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(voice_enabled=True)
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.voice_enabled is True

    def test_voice_api_key_persists(self, tmp_path):
        path = str(tmp_path / "mother.json")
        config = MotherConfig(api_keys={"elevenlabs": "sk-test"})
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.api_keys["elevenlabs"] == "sk-test"


# ---------------------------------------------------------------------------
# Test chat screen voice wiring
# ---------------------------------------------------------------------------

class TestChatVoiceWiring:
    """Test voice integration in ChatScreen."""

    def test_chat_screen_has_voice_attribute(self):
        from mother.screens.chat import ChatScreen
        screen = ChatScreen()
        assert hasattr(screen, "_voice")
        assert screen._voice is None

    def test_chat_screen_has_speak_method(self):
        from mother.screens.chat import ChatScreen
        screen = ChatScreen()
        assert hasattr(screen, "_speak")
        assert callable(screen._speak)

    def test_speak_noop_when_no_voice(self):
        from mother.screens.chat import ChatScreen
        screen = ChatScreen()
        # Should not raise
        screen._speak("hello")

    def test_voice_imports_present(self):
        from mother.screens.chat import ChatScreen
        from mother.voice import VoiceBridge, is_voice_available
        assert VoiceBridge is not None
        assert is_voice_available is not None


# ---------------------------------------------------------------------------
# Test parse_response — voice/action tag extraction
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Test parse_response() envelope parsing."""

    def test_plain_short_text_spoken_via_fallback(self):
        from mother.screens.chat import parse_response
        result = parse_response("I'm operational. What are you working on?")
        assert result["voice"] == "I'm operational. What are you working on?"
        assert result["action"] is None

    def test_voice_tags_extracted(self):
        from mother.screens.chat import parse_response
        result = parse_response("[VOICE]Let me compile that.[/VOICE]")
        assert result["voice"] == "Let me compile that."
        assert result["display"] == "Let me compile that."

    def test_voice_tags_stripped_from_display(self):
        from mother.screens.chat import parse_response
        raw = "[VOICE]Sure, here's the overview.[/VOICE]\n\nReact hooks are functions that..."
        result = parse_response(raw)
        assert result["voice"] == "Sure, here's the overview."
        assert "[VOICE]" not in result["display"]
        assert "[/VOICE]" not in result["display"]
        assert "React hooks are functions that..." in result["display"]

    def test_action_compile_extracted(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:compile]timer application[/ACTION][VOICE]Let me compile that.[/VOICE]"
        result = parse_response(raw)
        assert result["action"] == "compile"
        assert result["action_arg"] == "timer application"
        assert result["voice"] == "Let me compile that."

    def test_action_build_extracted(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:build]todo app[/ACTION][VOICE]Building that now.[/VOICE]"
        result = parse_response(raw)
        assert result["action"] == "build"
        assert result["action_arg"] == "todo app"

    def test_action_tools_extracted(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:tools][/ACTION][VOICE]Let me check.[/VOICE]"
        result = parse_response(raw)
        assert result["action"] == "tools"
        assert result["action_arg"] == ""

    def test_action_status_extracted(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:status][/ACTION][VOICE]Here's where we stand.[/VOICE]"
        result = parse_response(raw)
        assert result["action"] == "status"
        assert result["action_arg"] == ""

    def test_long_text_no_voice(self):
        from mother.screens.chat import parse_response
        long_text = "A" * 450
        result = parse_response(long_text)
        assert result["voice"] is None
        assert result["display"] == long_text

    def test_code_fences_no_voice(self):
        from mother.screens.chat import parse_response
        raw = "Here is code:\n```python\nprint('hi')\n```"
        result = parse_response(raw)
        assert result["voice"] is None

    def test_bullet_list_no_voice(self):
        from mother.screens.chat import parse_response
        raw = "Options:\n- Option A\n- Option B"
        result = parse_response(raw)
        assert result["voice"] is None

    def test_empty_string_graceful(self):
        from mother.screens.chat import parse_response
        result = parse_response("")
        assert result["display"] == ""
        assert result["voice"] is None
        assert result["action"] is None
        assert result["action_arg"] == ""

    def test_combined_action_and_voice(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:compile]calendar app[/ACTION][VOICE]On it.[/VOICE] I'll create a calendar with monthly views."
        result = parse_response(raw)
        assert result["action"] == "compile"
        assert result["action_arg"] == "calendar app"
        assert result["voice"] == "On it."
        assert "calendar app" not in result["display"]
        assert "I'll create a calendar" in result["display"]

    def test_tags_fully_stripped_from_display(self):
        from mother.screens.chat import parse_response
        raw = "[ACTION:compile]x[/ACTION][VOICE]y[/VOICE] rest"
        result = parse_response(raw)
        assert "[ACTION" not in result["display"]
        assert "[/ACTION]" not in result["display"]
        assert "[VOICE]" not in result["display"]
        assert "[/VOICE]" not in result["display"]

    def test_fallback_at_boundary(self):
        from mother.screens.chat import parse_response
        text_400 = "A" * 400
        result = parse_response(text_400)
        assert result["voice"] == text_400  # Exactly at boundary

        text_401 = "A" * 401
        result = parse_response(text_401)
        assert result["voice"] is None  # Over boundary


# ---------------------------------------------------------------------------
# Test INTENT_ROUTING in system prompt
# ---------------------------------------------------------------------------

class TestIntentRouting:
    """Test INTENT_ROUTING constant and system prompt wiring."""

    def test_intent_routing_in_prompt_by_default(self):
        from mother.persona import build_system_prompt, INTENT_ROUTING
        config = MotherConfig()
        prompt = build_system_prompt(config, include_compile=True)
        assert "[VOICE]" in prompt
        assert "[ACTION:" in prompt

    def test_intent_routing_excluded_when_no_compile(self):
        from mother.persona import build_system_prompt, INTENT_ROUTING
        config = MotherConfig()
        prompt = build_system_prompt(config, include_compile=False)
        assert INTENT_ROUTING not in prompt

    def test_intent_routing_contains_marker_docs(self):
        from mother.persona import INTENT_ROUTING
        assert "[VOICE]" in INTENT_ROUTING
        assert "[/VOICE]" in INTENT_ROUTING
        assert "[ACTION:compile]" in INTENT_ROUTING
        assert "[ACTION:build]" in INTENT_ROUTING
        assert "[ACTION:tools]" in INTENT_ROUTING
        assert "[ACTION:status]" in INTENT_ROUTING


class TestIntentRoutingWithIntrospection:
    """Test that voice/action markers survive when introspection is active."""

    def test_voice_and_action_markers_present_with_introspection(self):
        from mother.persona import build_system_prompt, build_introspection_snapshot
        config = MotherConfig()
        snap = build_introspection_snapshot(provider="grok", model="grok-3")
        prompt = build_system_prompt(config, introspection=snap)
        assert "[VOICE]" in prompt
        assert "[ACTION:" in prompt
        assert "[Self-observation]" in prompt
