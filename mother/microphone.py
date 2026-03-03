"""
Microphone bridge — record audio + OpenAI Whisper transcription.

Optional dependencies: sounddevice, scipy.
Follows the bridge pattern established by voice.py.
All errors caught — recording/transcription failure returns None, never raises.

Supports two recording modes:
  - Fixed duration: record for exactly N seconds (legacy).
  - VAD-adaptive: record until speech ends (silence detection). Default.
"""

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Optional

logger = logging.getLogger("mother.microphone")

# Optional imports — graceful degradation
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

try:
    import scipy.io.wavfile as wavfile
    import numpy as np
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def is_microphone_available() -> bool:
    """True if both sounddevice and scipy are installed."""
    return _SOUNDDEVICE_AVAILABLE and _SCIPY_AVAILABLE


def _compute_rms_int16(audio_array) -> float:
    """Compute RMS of int16 audio array. Returns 0.0 if scipy unavailable."""
    if not _SCIPY_AVAILABLE:
        return 0.0
    try:
        samples = audio_array.flatten().astype(np.float64)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))
    except Exception:
        return 0.0


class MicrophoneBridge:
    """Async microphone recording + Whisper transcription.

    Records via sounddevice, saves as WAV, sends to OpenAI Whisper API.
    All exceptions caught — failure returns None, never fatal.
    """

    def __init__(
        self,
        openai_api_key: str = "",
        enabled: bool = True,
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self._openai_api_key = openai_api_key
        self._enabled = enabled
        self._sample_rate = sample_rate
        self._channels = channels
        self._client = None
        self._is_recording = False

    @property
    def enabled(self) -> bool:
        """Microphone requires flag + SDK + API key."""
        return (
            self._enabled
            and is_microphone_available()
            and bool(self._openai_api_key)
        )

    @property
    def is_recording(self) -> bool:
        """True while audio is being captured."""
        return self._is_recording

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None and self._openai_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._openai_api_key)
            except ImportError:
                logger.warning("openai package not available")
        return self._client

    def _record_sync(self, duration_seconds: float = 5.0) -> Optional[bytes]:
        """Record audio and return WAV bytes.

        Uses sounddevice for recording, scipy for WAV encoding.
        Returns WAV file bytes or None on failure.
        """
        if not is_microphone_available():
            return None

        self._is_recording = True
        tmp_path = None
        try:
            # Record audio
            frames = int(duration_seconds * self._sample_rate)
            recording = sd.rec(
                frames,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
            )
            sd.wait()  # Block until recording completes

            # Write to temp WAV
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="mother_mic_")
            os.close(fd)
            wavfile.write(tmp_path, self._sample_rate, recording)

            # Read back as bytes
            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            return wav_bytes

        except Exception as e:
            logger.warning(f"Recording error (non-fatal): {e}")
            return None
        finally:
            self._is_recording = False
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _record_vad_sync(
        self,
        max_duration: float = 30.0,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        speech_min_duration: float = 0.3,
        chunk_seconds: float = 0.1,
        pre_speech_timeout: float = 10.0,
    ) -> Optional[bytes]:
        """Record audio with VAD — stops when speech ends (silence detected).

        Listens for up to pre_speech_timeout for speech to begin.
        Once speech is detected, records until silence_duration of quiet,
        up to max_duration total.

        Returns WAV file bytes or None on failure/no speech.
        """
        if not is_microphone_available():
            return None

        self._is_recording = True
        tmp_path = None
        try:
            chunk_frames = int(chunk_seconds * self._sample_rate)
            speech_chunks: list = []
            speech_started = False
            speech_start_time: Optional[float] = None
            silence_start: Optional[float] = None
            listen_start = time.monotonic()

            while True:
                elapsed = time.monotonic() - listen_start
                if elapsed >= max_duration:
                    break

                # Pre-speech timeout — give up if no speech detected
                if not speech_started and elapsed >= pre_speech_timeout:
                    break

                chunk = sd.rec(
                    chunk_frames,
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="int16",
                )
                sd.wait()

                rms = _compute_rms_int16(chunk)
                now = time.monotonic()

                if rms > silence_threshold:
                    # Speech detected
                    silence_start = None
                    if not speech_started:
                        speech_started = True
                        speech_start_time = now
                    speech_chunks.append(chunk)
                else:
                    if speech_started:
                        # Still collecting during brief silence
                        speech_chunks.append(chunk)
                        if silence_start is None:
                            silence_start = now
                        elif (now - silence_start) >= silence_duration:
                            # End of speech
                            break

            if not speech_chunks:
                return None

            # Check minimum speech duration
            speech_dur = (
                (time.monotonic() - speech_start_time)
                if speech_start_time else 0.0
            )
            if speech_dur < speech_min_duration:
                return None

            # Concatenate and write WAV
            recording = np.concatenate(speech_chunks, axis=0)
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="mother_mic_")
            os.close(fd)
            wavfile.write(tmp_path, self._sample_rate, recording)

            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            return wav_bytes

        except Exception as e:
            logger.warning(f"VAD recording error (non-fatal): {e}")
            return None
        finally:
            self._is_recording = False
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _transcribe_sync(self, wav_bytes: bytes) -> Optional[str]:
        """Send WAV bytes to OpenAI Whisper API for transcription.

        Returns transcribed text or None on failure.
        """
        if not wav_bytes:
            return None

        client = self._get_client()
        if client is None:
            return None

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="mother_whisper_")
            os.close(fd)

            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)

            with open(tmp_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            # transcript is a string when response_format="text"
            text = transcript.strip() if isinstance(transcript, str) else str(transcript).strip()
            return text if text else None

        except Exception as e:
            logger.warning(f"Transcription error (non-fatal): {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def record_and_transcribe(
        self, duration_seconds: float = 5.0
    ) -> Optional[str]:
        """Record fixed-duration audio and transcribe. Async, runs in threads.

        Returns transcribed text or None on failure.
        """
        if not self.enabled:
            return None

        wav_bytes = await asyncio.to_thread(self._record_sync, duration_seconds)
        if not wav_bytes:
            return None

        return await asyncio.to_thread(self._transcribe_sync, wav_bytes)

    async def record_and_transcribe_vad(
        self,
        max_duration: float = 30.0,
        silence_threshold: float = 500.0,
        silence_duration: float = 1.5,
        pre_speech_timeout: float = 10.0,
    ) -> Optional[str]:
        """Record with VAD (stops on silence) and transcribe. Async.

        Listens for speech onset up to pre_speech_timeout, then records
        until silence_duration of quiet, capped at max_duration.

        Returns transcribed text or None on failure/no speech.
        """
        if not self.enabled:
            return None

        wav_bytes = await asyncio.to_thread(
            self._record_vad_sync,
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            pre_speech_timeout=pre_speech_timeout,
        )
        if not wav_bytes:
            return None

        return await asyncio.to_thread(self._transcribe_sync, wav_bytes)
