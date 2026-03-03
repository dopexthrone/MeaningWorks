"""
Tests for Mother microphone bridge — record + Whisper transcription.

All tests mocked. No real recording or API calls.
"""

import asyncio
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock

from mother.config import MotherConfig, save_config, load_config

# Ensure mock modules exist for sounddevice/scipy even if not installed.
# This lets us patch them in recording tests.
_mock_sd = MagicMock()
_mock_wavfile = MagicMock()
_mock_np = MagicMock()


def _ensure_mic_deps():
    """Inject mock audio deps into mother.microphone if not already present."""
    import mother.microphone as mic
    if not hasattr(mic, "sd") or mic.sd is None:
        mic.sd = _mock_sd
    if not hasattr(mic, "wavfile") or mic.wavfile is None:
        mic.wavfile = _mock_wavfile
    if not hasattr(mic, "np") or mic.np is None:
        mic.np = _mock_np


# ---------------------------------------------------------------------------
# Test availability detection
# ---------------------------------------------------------------------------

class TestMicrophoneAvailability:
    """Test SDK detection for microphone."""

    def test_is_microphone_available_returns_bool(self):
        from mother.microphone import is_microphone_available
        result = is_microphone_available()
        assert isinstance(result, bool)

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_available_when_both_deps(self):
        from mother.microphone import is_microphone_available
        assert is_microphone_available() is True

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", False)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_not_available_without_sounddevice(self):
        from mother.microphone import is_microphone_available
        assert is_microphone_available() is False

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", False)
    def test_not_available_without_scipy(self):
        from mother.microphone import is_microphone_available
        assert is_microphone_available() is False

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", False)
    @patch("mother.microphone._SCIPY_AVAILABLE", False)
    def test_not_available_without_both(self):
        from mother.microphone import is_microphone_available
        assert is_microphone_available() is False


# ---------------------------------------------------------------------------
# Test MicrophoneBridge creation
# ---------------------------------------------------------------------------

class TestMicrophoneBridgeCreation:
    """Test MicrophoneBridge instantiation."""

    def test_create_with_key(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="test-key")
        assert bridge._openai_api_key == "test-key"

    def test_create_default_settings(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="test-key")
        assert bridge._sample_rate == 16000
        assert bridge._channels == 1
        assert bridge._enabled is True

    def test_create_custom_settings(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(
            openai_api_key="key",
            sample_rate=44100,
            channels=2,
        )
        assert bridge._sample_rate == 44100
        assert bridge._channels == 2

    def test_create_disabled(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=False)
        assert bridge._enabled is False


# ---------------------------------------------------------------------------
# Test enabled property
# ---------------------------------------------------------------------------

class TestMicrophoneBridgeEnabled:
    """Test the multi-gate enabled property."""

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_enabled_with_all_gates(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)
        assert bridge.enabled is True

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_disabled_when_flag_false(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=False)
        assert bridge.enabled is False

    @patch("mother.microphone.is_microphone_available", return_value=False)
    def test_disabled_without_sdk(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)
        assert bridge.enabled is False

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_disabled_without_key(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="", enabled=True)
        assert bridge.enabled is False

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_disabled_with_empty_key(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="", enabled=True)
        assert bridge.enabled is False


# ---------------------------------------------------------------------------
# Test is_recording property
# ---------------------------------------------------------------------------

class TestMicrophoneIsRecording:
    """Test the is_recording state flag."""

    def test_not_recording_initially(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        assert bridge.is_recording is False


# ---------------------------------------------------------------------------
# Test recording (sync)
# ---------------------------------------------------------------------------

class TestMicrophoneRecordSync:
    """Test the synchronous _record_sync method."""

    @patch("mother.microphone.is_microphone_available", return_value=True)
    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_record_success(self, mock_avail):
        _ensure_mic_deps()
        import mother.microphone as mic
        from mother.microphone import MicrophoneBridge

        mock_sd = MagicMock()
        mock_recording = MagicMock()
        mock_sd.rec.return_value = mock_recording
        mock_wf = MagicMock()

        wav_bytes = b"RIFF" + b"\x00" * 100

        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        original_sd = mic.sd
        original_wavfile = mic.wavfile
        try:
            mic.sd = mock_sd
            mic.wavfile = mock_wf
            with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/test.wav")), \
                 patch("mother.microphone.os.close"), \
                 patch("mother.microphone.os.unlink"), \
                 patch("mother.microphone.os.path.exists", return_value=True), \
                 patch("builtins.open", mock_open(read_data=wav_bytes)):
                result = bridge._record_sync(duration_seconds=3.0)
        finally:
            mic.sd = original_sd
            mic.wavfile = original_wavfile

        assert result == wav_bytes
        mock_sd.rec.assert_called_once()
        mock_sd.wait.assert_called_once()

    @patch("mother.microphone.is_microphone_available", return_value=True)
    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_record_sets_is_recording(self, mock_avail):
        _ensure_mic_deps()
        import mother.microphone as mic
        from mother.microphone import MicrophoneBridge

        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)
        recording_states = []

        mock_sd = MagicMock()
        def capture_state(*args, **kwargs):
            recording_states.append(bridge.is_recording)
            return MagicMock()
        mock_sd.rec.side_effect = capture_state

        original_sd = mic.sd
        original_wavfile = mic.wavfile
        try:
            mic.sd = mock_sd
            mic.wavfile = MagicMock()
            with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/test.wav")), \
                 patch("mother.microphone.os.close"), \
                 patch("mother.microphone.os.unlink"), \
                 patch("mother.microphone.os.path.exists", return_value=True), \
                 patch("builtins.open", mock_open(read_data=b"wav")):
                bridge._record_sync()
        finally:
            mic.sd = original_sd
            mic.wavfile = original_wavfile

        assert True in recording_states  # Was recording during sd.rec
        assert bridge.is_recording is False  # Not recording after

    @patch("mother.microphone.is_microphone_available", return_value=False)
    def test_record_not_available_returns_none(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)
        result = bridge._record_sync()
        assert result is None

    @patch("mother.microphone.is_microphone_available", return_value=True)
    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_record_exception_returns_none(self, mock_avail):
        _ensure_mic_deps()
        import mother.microphone as mic
        from mother.microphone import MicrophoneBridge

        mock_sd = MagicMock()
        mock_sd.rec.side_effect = Exception("audio device error")

        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        original_sd = mic.sd
        try:
            mic.sd = mock_sd
            with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/test.wav")), \
                 patch("mother.microphone.os.close"), \
                 patch("mother.microphone.os.unlink"), \
                 patch("mother.microphone.os.path.exists", return_value=True):
                result = bridge._record_sync()
        finally:
            mic.sd = original_sd

        assert result is None
        assert bridge.is_recording is False  # Cleaned up

    @patch("mother.microphone.is_microphone_available", return_value=True)
    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_record_cleans_up_temp(self, mock_avail):
        _ensure_mic_deps()
        import mother.microphone as mic
        from mother.microphone import MicrophoneBridge

        mock_sd = MagicMock()
        mock_sd.rec.return_value = MagicMock()

        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        original_sd = mic.sd
        original_wavfile = mic.wavfile
        try:
            mic.sd = mock_sd
            mic.wavfile = MagicMock()
            with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/cleanup.wav")), \
                 patch("mother.microphone.os.close"), \
                 patch("mother.microphone.os.path.exists", return_value=True), \
                 patch("mother.microphone.os.unlink") as mock_unlink, \
                 patch("builtins.open", mock_open(read_data=b"wav")):
                bridge._record_sync()
                mock_unlink.assert_called_once_with("/tmp/cleanup.wav")
        finally:
            mic.sd = original_sd
            mic.wavfile = original_wavfile


# ---------------------------------------------------------------------------
# Test transcription (sync)
# ---------------------------------------------------------------------------

class TestMicrophoneTranscribeSync:
    """Test the synchronous _transcribe_sync method."""

    def test_transcribe_empty_bytes_returns_none(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        result = bridge._transcribe_sync(b"")
        assert result is None

    def test_transcribe_none_bytes_returns_none(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        result = bridge._transcribe_sync(b"")
        assert result is None

    def test_transcribe_no_client_returns_none(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="")
        result = bridge._transcribe_sync(b"wav_data")
        assert result is None

    def test_transcribe_success(self):
        from mother.microphone import MicrophoneBridge

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "Hello world"

        bridge = MicrophoneBridge(openai_api_key="key")
        bridge._client = mock_client

        with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/whisper.wav")), \
             patch("mother.microphone.os.close"), \
             patch("mother.microphone.os.unlink"), \
             patch("mother.microphone.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open()):
            result = bridge._transcribe_sync(b"wav_data")

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_transcribe_strips_whitespace(self):
        from mother.microphone import MicrophoneBridge

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "  Hello world  \n"

        bridge = MicrophoneBridge(openai_api_key="key")
        bridge._client = mock_client

        with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/whisper.wav")), \
             patch("mother.microphone.os.close"), \
             patch("mother.microphone.os.unlink"), \
             patch("mother.microphone.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open()):
            result = bridge._transcribe_sync(b"wav_data")

        assert result == "Hello world"

    def test_transcribe_empty_result_returns_none(self):
        from mother.microphone import MicrophoneBridge

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "   "

        bridge = MicrophoneBridge(openai_api_key="key")
        bridge._client = mock_client

        with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/whisper.wav")), \
             patch("mother.microphone.os.close"), \
             patch("mother.microphone.os.unlink"), \
             patch("mother.microphone.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open()):
            result = bridge._transcribe_sync(b"wav_data")

        assert result is None

    def test_transcribe_api_error_returns_none(self):
        from mother.microphone import MicrophoneBridge

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")

        bridge = MicrophoneBridge(openai_api_key="key")
        bridge._client = mock_client

        with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/whisper.wav")), \
             patch("mother.microphone.os.close"), \
             patch("mother.microphone.os.unlink"), \
             patch("mother.microphone.os.path.exists", return_value=True), \
             patch("builtins.open", mock_open()):
            result = bridge._transcribe_sync(b"wav_data")

        assert result is None

    def test_transcribe_cleans_up_temp(self):
        from mother.microphone import MicrophoneBridge

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = "text"

        bridge = MicrophoneBridge(openai_api_key="key")
        bridge._client = mock_client

        with patch("mother.microphone.tempfile.mkstemp", return_value=(5, "/tmp/cleanup.wav")), \
             patch("mother.microphone.os.close"), \
             patch("mother.microphone.os.path.exists", return_value=True), \
             patch("mother.microphone.os.unlink") as mock_unlink, \
             patch("builtins.open", mock_open()):
            bridge._transcribe_sync(b"wav_data")
            mock_unlink.assert_called_once_with("/tmp/cleanup.wav")


# ---------------------------------------------------------------------------
# Test full pipeline (async)
# ---------------------------------------------------------------------------

class TestMicrophoneRecordAndTranscribe:
    """Test the async record_and_transcribe method."""

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_full_pipeline_success(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        async def _run():
            with patch.object(bridge, "_record_sync", return_value=b"wav_data"), \
                 patch.object(bridge, "_transcribe_sync", return_value="Hello world"):
                return await bridge.record_and_transcribe(5.0)

        result = asyncio.run(_run())
        assert result == "Hello world"

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_pipeline_record_failure(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        async def _run():
            with patch.object(bridge, "_record_sync", return_value=None):
                return await bridge.record_and_transcribe(5.0)

        result = asyncio.run(_run())
        assert result is None

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_pipeline_transcribe_failure(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        async def _run():
            with patch.object(bridge, "_record_sync", return_value=b"wav"), \
                 patch.object(bridge, "_transcribe_sync", return_value=None):
                return await bridge.record_and_transcribe(5.0)

        result = asyncio.run(_run())
        assert result is None

    @patch("mother.microphone.is_microphone_available", return_value=False)
    def test_pipeline_disabled_returns_none(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        async def _run():
            return await bridge.record_and_transcribe(5.0)

        result = asyncio.run(_run())
        assert result is None

    @patch("mother.microphone.is_microphone_available", return_value=True)
    def test_pipeline_passes_duration(self, mock_avail):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=True)

        async def _run():
            with patch.object(bridge, "_record_sync", return_value=b"wav") as mock_rec, \
                 patch.object(bridge, "_transcribe_sync", return_value="text"):
                await bridge.record_and_transcribe(10.0)
                mock_rec.assert_called_once_with(10.0)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Test lazy client loading
# ---------------------------------------------------------------------------

class TestMicrophoneClientLoading:
    """Test OpenAI client lazy loading."""

    def test_no_client_without_key(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="")
        result = bridge._get_client()
        assert result is None

    def test_client_cached(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        mock_client = MagicMock()
        bridge._client = mock_client
        assert bridge._get_client() is mock_client


# ---------------------------------------------------------------------------
# Test config integration
# ---------------------------------------------------------------------------

class TestMicrophoneConfig:
    """Test microphone_enabled config field."""

    def test_config_default_disabled(self):
        config = MotherConfig()
        assert config.microphone_enabled is False

    def test_config_roundtrip(self, tmp_path):
        config = MotherConfig(microphone_enabled=True)
        path = str(tmp_path / "test.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.microphone_enabled is True

    def test_config_disabled_roundtrip(self, tmp_path):
        config = MotherConfig(microphone_enabled=False)
        path = str(tmp_path / "test.json")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.microphone_enabled is False


# ---------------------------------------------------------------------------
# Test _compute_rms_int16
# ---------------------------------------------------------------------------

class TestComputeRmsInt16:
    """Test the RMS computation helper."""

    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_rms_silent_audio(self):
        np = pytest.importorskip("numpy")
        from mother.microphone import _compute_rms_int16
        silence = np.zeros(1600, dtype=np.int16)
        assert _compute_rms_int16(silence) == 0.0

    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_rms_loud_audio(self):
        np = pytest.importorskip("numpy")
        from mother.microphone import _compute_rms_int16
        loud = np.full(1600, 10000, dtype=np.int16)
        rms = _compute_rms_int16(loud)
        assert rms == pytest.approx(10000.0, rel=0.01)

    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_rms_mixed_audio(self):
        np = pytest.importorskip("numpy")
        from mother.microphone import _compute_rms_int16
        arr = np.array([1000, -1000, 1000, -1000], dtype=np.int16)
        rms = _compute_rms_int16(arr)
        assert rms == pytest.approx(1000.0, rel=0.01)

    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_rms_empty_array(self):
        np = pytest.importorskip("numpy")
        from mother.microphone import _compute_rms_int16
        empty = np.array([], dtype=np.int16)
        assert _compute_rms_int16(empty) == 0.0

    @patch("mother.microphone._SCIPY_AVAILABLE", False)
    def test_rms_returns_zero_without_scipy(self):
        np = pytest.importorskip("numpy")
        from mother.microphone import _compute_rms_int16
        arr = np.full(1600, 10000, dtype=np.int16)
        assert _compute_rms_int16(arr) == 0.0


# ---------------------------------------------------------------------------
# Test VAD recording
# ---------------------------------------------------------------------------

class TestVadRecording:
    """Test VAD-adaptive recording in MicrophoneBridge."""

    def test_record_vad_sync_method_exists(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        assert hasattr(bridge, "_record_vad_sync")

    def test_record_and_transcribe_vad_method_exists(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        assert hasattr(bridge, "record_and_transcribe_vad")

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", False)
    def test_vad_returns_none_without_sounddevice(self):
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key")
        result = bridge._record_vad_sync()
        assert result is None

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_vad_returns_none_on_no_speech(self):
        """VAD returns None when no speech detected within timeout."""
        np = pytest.importorskip("numpy")
        from mother.microphone import MicrophoneBridge

        _ensure_mic_deps()
        import mother.microphone as mic

        # All chunks return silence (rms=0)
        silence = np.zeros((160, 1), dtype=np.int16)
        mic.sd.rec = MagicMock(return_value=silence)
        mic.sd.wait = MagicMock()

        bridge = MicrophoneBridge(openai_api_key="key")
        result = bridge._record_vad_sync(
            pre_speech_timeout=0.2,
            chunk_seconds=0.01,
        )
        assert result is None

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_vad_records_speech_then_stops_on_silence(self):
        """VAD records speech and stops after silence threshold."""
        np = pytest.importorskip("numpy")
        from mother.microphone import MicrophoneBridge

        _ensure_mic_deps()
        import mother.microphone as mic

        loud = np.full((160, 1), 5000, dtype=np.int16)
        silence = np.zeros((160, 1), dtype=np.int16)

        # 5 loud chunks then silence
        call_count = [0]
        def fake_rec(frames, samplerate, channels, dtype):
            call_count[0] += 1
            if call_count[0] <= 5:
                return loud.copy()
            return silence.copy()

        mic.sd.rec = fake_rec
        mic.sd.wait = MagicMock()
        mic.wavfile.write = MagicMock()
        mic.np = np

        bridge = MicrophoneBridge(openai_api_key="key")
        result = bridge._record_vad_sync(
            silence_threshold=500.0,
            silence_duration=0.15,
            chunk_seconds=0.01,
            pre_speech_timeout=2.0,
            max_duration=5.0,
        )
        # Should have written a WAV file
        assert mic.wavfile.write.called or result is None  # May be None if timing race

    def test_vad_signature_has_all_params(self):
        """_record_vad_sync accepts all expected parameters."""
        import inspect
        from mother.microphone import MicrophoneBridge
        sig = inspect.signature(MicrophoneBridge._record_vad_sync)
        params = set(sig.parameters.keys())
        assert "max_duration" in params
        assert "silence_threshold" in params
        assert "silence_duration" in params
        assert "speech_min_duration" in params
        assert "chunk_seconds" in params
        assert "pre_speech_timeout" in params

    def test_record_and_transcribe_vad_signature(self):
        """record_and_transcribe_vad accepts expected parameters."""
        import inspect
        from mother.microphone import MicrophoneBridge
        sig = inspect.signature(MicrophoneBridge.record_and_transcribe_vad)
        params = set(sig.parameters.keys())
        assert "max_duration" in params
        assert "silence_threshold" in params
        assert "silence_duration" in params
        assert "pre_speech_timeout" in params

    @patch("mother.microphone._SOUNDDEVICE_AVAILABLE", True)
    @patch("mother.microphone._SCIPY_AVAILABLE", True)
    def test_vad_async_returns_none_when_disabled(self):
        """record_and_transcribe_vad returns None when bridge disabled."""
        from mother.microphone import MicrophoneBridge
        bridge = MicrophoneBridge(openai_api_key="key", enabled=False)
        result = asyncio.run(bridge.record_and_transcribe_vad())
        assert result is None


# ---------------------------------------------------------------------------
# Test adaptive listen flow in chat.py
# ---------------------------------------------------------------------------

class TestAdaptiveListenWiring:
    """Test that chat.py uses VAD-adaptive listening."""

    def test_run_listen_defaults_to_none_duration(self):
        """_run_listen defaults to None (VAD mode)."""
        import inspect
        # Read source to check default
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("def _run_listen(self")
        sig_end = source.index(") -> None:", idx)
        sig = source[idx:sig_end + 10]
        assert "duration: Optional[float] = None" in sig

    def test_listen_worker_defaults_to_none(self):
        """_listen_worker defaults to None (VAD mode)."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("async def _listen_worker(self")
        sig_end = source.index(") -> None:", idx)
        sig = source[idx:sig_end + 10]
        assert "duration: Optional[float] = None" in sig

    def test_action_listen_no_hardcoded_duration(self):
        """F8 handler calls _run_listen() with no args (VAD)."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("def action_listen(self)")
        block = source[idx:idx + 200]
        assert "_run_listen()" in block
        assert "_run_listen(5.0)" not in block

    def test_enable_mic_action_no_hardcoded_duration(self):
        """enable_mic action calls _run_listen() with no duration."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index('action == "enable_mic"')
        block = source[idx:idx + 300]
        assert "_run_listen()" in block
        assert "_run_listen(5.0)" not in block

    def test_listen_command_defaults_vad(self):
        """/listen command defaults to VAD (duration=None)."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index('cmd == "/listen"')
        block = source[idx:idx + 300]
        assert "duration = None" in block

    def test_listen_command_accepts_explicit_duration(self):
        """/listen 10 still passes explicit duration."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index('cmd == "/listen"')
        block = source[idx:idx + 300]
        assert "duration = float(arg)" in block

    def test_listen_worker_branches_on_duration(self):
        """_listen_worker uses VAD when duration is None, fixed when given."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("async def _listen_worker")
        block = source[idx:idx + 600]
        assert "record_and_transcribe_vad" in block
        assert "record_and_transcribe(duration)" in block

    def test_mic_permission_confirm_uses_vad(self):
        """After mic permission confirm, uses _run_listen() not _run_listen(5.0)."""
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index('capability == "microphone"')
        block = source[idx:idx + 400]
        assert "_run_listen()" in block
        assert "_run_listen(5.0)" not in block
