"""
Voice bridge — ElevenLabs TTS synthesis + cross-platform playback.

Optional dependency. Voice failure never crashes the TUI.
All synthesis and playback runs in asyncio.to_thread().

Playback backends (checked in order):
- macOS: afplay (built-in, supports playback rate)
- Linux: paplay (PulseAudio), play (SoX), ffplay (FFmpeg)
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger("mother.voice")

# Optional import — graceful degradation
try:
    from elevenlabs import ElevenLabs
    _ELEVENLABS_AVAILABLE = True
except ImportError:
    _ELEVENLABS_AVAILABLE = False

DEFAULT_VOICE_ID = "2obv5y63xKRNiEZAPxGD"
DEFAULT_MODEL_ID = "eleven_v3"
DEFAULT_PLAYBACK_RATE = 1.15  # Slightly quicker than default — alert, not rushed


def _detect_audio_player() -> Tuple[str, str]:
    """Detect available audio playback command.

    Returns (player_name, player_path). player_name is one of:
    "afplay", "paplay", "play", "ffplay", or "" if none found.
    """
    if sys.platform == "darwin":
        path = shutil.which("afplay")
        if path:
            return ("afplay", path)

    # Linux / fallback
    for player in ("paplay", "play", "ffplay"):
        path = shutil.which(player)
        if path:
            return (player, path)

    # macOS fallback if afplay wasn't found (unlikely)
    if sys.platform == "darwin":
        for player in ("play", "ffplay"):
            path = shutil.which(player)
            if path:
                return (player, path)

    return ("", "")


_AUDIO_PLAYER, _AUDIO_PLAYER_PATH = _detect_audio_player()


def _build_play_cmd(tmp_path: str, rate: float = 1.0) -> list:
    """Build the playback command list for the detected player."""
    if _AUDIO_PLAYER == "afplay":
        cmd = [_AUDIO_PLAYER_PATH, tmp_path]
        if rate != 1.0:
            cmd.extend(["-r", str(rate)])
        return cmd
    elif _AUDIO_PLAYER == "paplay":
        # paplay doesn't support rate or mp3 directly, but works for wav
        # For mp3, ffplay or play is better. paplay as last resort.
        return [_AUDIO_PLAYER_PATH, tmp_path]
    elif _AUDIO_PLAYER == "play":
        # SoX 'play' command — supports tempo adjustment
        cmd = [_AUDIO_PLAYER_PATH, "-q", tmp_path]
        if rate != 1.0:
            cmd.extend(["tempo", str(rate)])
        return cmd
    elif _AUDIO_PLAYER == "ffplay":
        cmd = [_AUDIO_PLAYER_PATH, "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path]
        return cmd
    return []


def is_voice_available() -> bool:
    """True if the ElevenLabs SDK is installed and an audio player exists."""
    return _ELEVENLABS_AVAILABLE and bool(_AUDIO_PLAYER)


class VoiceBridge:
    """Async TTS synthesis + playback via ElevenLabs.

    Mirrors bridge.py pattern: wraps blocking calls in to_thread().
    All exceptions caught — voice failure is silent, never fatal.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        enabled: bool = True,
        playback_rate: float = DEFAULT_PLAYBACK_RATE,
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._enabled = enabled
        self._playback_rate = playback_rate
        self._client: Optional[object] = None
        self._current_process: Optional[subprocess.Popen] = None

    @property
    def enabled(self) -> bool:
        """Voice requires both the flag AND the SDK."""
        return self._enabled and _ELEVENLABS_AVAILABLE and bool(self._api_key)

    def _get_client(self):
        """Lazy-load ElevenLabs client."""
        if self._client is None and _ELEVENLABS_AVAILABLE:
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def _synthesize(self, text: str) -> bytes:
        """Call ElevenLabs TTS. Returns raw audio bytes (mp3)."""
        client = self._get_client()
        if client is None:
            return b""

        audio = client.text_to_speech.convert(
            text=text,
            voice_id=self._voice_id,
            model_id=self._model_id,
            output_format="mp3_44100_128",
        )
        # audio is a generator of bytes chunks
        chunks = []
        for chunk in audio:
            chunks.append(chunk)
        return b"".join(chunks)

    def _synthesize_and_play_streaming(self, text: str, rate: Optional[float] = None) -> None:
        """Stream TTS chunks to disk, start playback after first chunk.

        Reduces time-to-first-audio by starting playback while synthesis
        is still streaming. The player reads the mp3 progressively.
        """
        client = self._get_client()
        if client is None:
            return

        effective_rate = rate if rate is not None else self._playback_rate
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="mother_voice_")

            audio = client.text_to_speech.convert(
                text=text,
                voice_id=self._voice_id,
                model_id=self._model_id,
                output_format="mp3_44100_128",
            )

            # Write first chunks, then start playback while continuing to write
            started = False
            for chunk in audio:
                os.write(fd, chunk)
                if not started and len(chunk) > 0:
                    # Flush to disk before starting player
                    os.fsync(fd)
                    cmd = _build_play_cmd(tmp_path, effective_rate)
                    if not cmd:
                        break
                    self._current_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    started = True

            # Done writing — close fd so player can read the rest
            os.close(fd)
            fd = None

            # Wait for playback to finish
            if self._current_process:
                self._current_process.wait()
        except Exception:
            # Close fd if still open
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        finally:
            self._current_process = None
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _play_audio(self, audio_data: bytes, rate: Optional[float] = None) -> None:
        """Write temp mp3, play with detected audio player, cleanup.

        rate: playback speed override. Falls back to self._playback_rate.
        """
        if not audio_data:
            return

        effective_rate = rate if rate is not None else self._playback_rate
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="mother_voice_")
            os.write(fd, audio_data)
            os.close(fd)

            cmd = _build_play_cmd(tmp_path, effective_rate)
            if not cmd:
                return
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_process.wait()
        finally:
            self._current_process = None
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes. Async, runs in thread.

        Returns empty bytes on failure. Callers can pipeline: synthesize
        the next sentence while the current one plays.
        """
        if not self.enabled or not text or not text.strip():
            return b""
        try:
            return await asyncio.to_thread(self._synthesize, text)
        except Exception as e:
            logger.warning(f"Synthesize error (non-fatal): {e}")
            return b""

    async def play(self, audio_data: bytes, playback_rate: Optional[float] = None) -> None:
        """Play pre-synthesized audio. Async, runs in thread."""
        if not audio_data:
            return
        try:
            await asyncio.to_thread(self._play_audio, audio_data, playback_rate)
        except Exception as e:
            logger.warning(f"Playback error (non-fatal): {e}")

    async def speak(self, text: str, playback_rate: Optional[float] = None) -> None:
        """Synthesize and play text with streaming. Async, runs in thread.

        Streams TTS chunks to disk and starts playback after the first chunk,
        so the user hears audio faster. Falls back to non-streaming on error.

        playback_rate: override speed for this utterance. Falls back to default.
        Catches all exceptions — voice failure is silent.
        """
        if not self.enabled or not text or not text.strip():
            return

        try:
            await asyncio.to_thread(
                self._synthesize_and_play_streaming, text, playback_rate
            )
        except Exception as e:
            logger.warning(f"Voice error (non-fatal): {e}")

    def speak_fire_and_forget(self, text: str) -> Optional[asyncio.Task]:
        """Create a background task for speech. Returns the task or None."""
        if not self.enabled or not text or not text.strip():
            return None

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self.speak(text))
            return task
        except RuntimeError:
            logger.debug("No running event loop for fire-and-forget speak")
            return None

    @property
    def is_playing(self) -> bool:
        """True if currently playing audio."""
        return self._current_process is not None

    async def stop(self) -> None:
        """Kill current audio playback process if running."""
        proc = self._current_process
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
