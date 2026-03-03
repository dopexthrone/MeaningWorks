"""
Continuous perception engine — screen monitoring, webcam, ambient mic.

LEAF module. No imports from core/. All bridge callables injected.

Three background loops feed PerceptionEvents into an asyncio.Queue.
Cost-aware via VisionBudget. Change detection via MD5 hashing.
VAD via energy threshold (no ML deps).
"""

import asyncio
import base64
import hashlib
import io
import logging
import math
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional, Tuple

logger = logging.getLogger("mother.perception")

# Optional numpy for VAD — graceful degradation
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PerceptionEventType(Enum):
    """Types of perception events."""
    SCREEN_CHANGED = "screen_changed"
    SPEECH_DETECTED = "speech_detected"
    CAMERA_FRAME = "camera_frame"


@dataclass(frozen=True)
class PerceptionEvent:
    """A single perception event from any modality."""
    event_type: PerceptionEventType
    payload: str  # base64 image or transcribed text
    timestamp: float
    media_type: str = "image/png"  # image/png, image/jpeg, text/plain


@dataclass(frozen=True)
class PerceptionConfig:
    """All tunables for perception loops."""
    screen_enabled: bool = False
    camera_enabled: bool = False
    ambient_mic_enabled: bool = False
    screen_poll_seconds: float = 10.0
    camera_poll_seconds: float = 30.0
    budget_per_hour: float = 5.00
    cost_per_vision_call: float = 0.005
    vad_threshold: float = 500.0
    silence_duration: float = 1.5
    speech_min_duration: float = 0.5
    mic_chunk_seconds: float = 0.1
    mic_sample_rate: int = 16000


# ---------------------------------------------------------------------------
# VisionBudget
# ---------------------------------------------------------------------------

class VisionBudget:
    """Rolling 1-hour spend window for vision API calls."""

    def __init__(self, budget_per_hour: float = 5.00):
        self._budget = budget_per_hour
        self._log: List[Tuple[float, float]] = []  # (timestamp, cost)
        self._lock = asyncio.Lock()

    async def can_spend(self, amount: float) -> bool:
        """True if spending amount won't exceed hourly budget."""
        async with self._lock:
            self._prune()
            current = sum(cost for _, cost in self._log)
            return (current + amount) <= self._budget

    async def record_spend(self, amount: float) -> None:
        """Record a spend event."""
        async with self._lock:
            self._log.append((time.time(), amount))

    async def get_hourly_spend(self) -> float:
        """Current spend in the rolling hour window."""
        async with self._lock:
            self._prune()
            return sum(cost for _, cost in self._log)

    def _prune(self) -> None:
        """Remove entries older than 1 hour."""
        cutoff = time.time() - 3600
        self._log = [(t, c) for t, c in self._log if t > cutoff]


# ---------------------------------------------------------------------------
# Image downscaling (optional Pillow)
# ---------------------------------------------------------------------------

def _downscale_image(b64_data: str, target_width: int = 640, media_type: str = "image/png") -> Tuple[str, str]:
    """Downscale base64 image to target_width using Pillow.

    Returns (new_b64, new_media_type). Falls back to original if Pillow unavailable.
    """
    try:
        from PIL import Image
    except ImportError:
        return b64_data, media_type

    try:
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw))

        if img.width <= target_width:
            return b64_data, media_type

        ratio = target_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((target_width, new_height), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        new_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return new_b64, "image/jpeg"
    except Exception as e:
        logger.debug(f"Downscale failed (non-fatal): {e}")
        return b64_data, media_type


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _compute_hash(data: str) -> str:
    """MD5 hash of a string for change detection."""
    return hashlib.md5(data.encode("ascii", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# VAD (Voice Activity Detection) — energy threshold
# ---------------------------------------------------------------------------

def _compute_rms(audio_bytes: bytes, sample_width: int = 2) -> float:
    """Compute RMS energy of int16 PCM audio bytes.

    Falls back to 0.0 if data is empty or numpy unavailable.
    """
    if not audio_bytes:
        return 0.0

    if _NUMPY_AVAILABLE:
        if len(audio_bytes) < sample_width:
            return 0.0
        # Trim to multiple of sample_width
        usable = len(audio_bytes) - (len(audio_bytes) % sample_width)
        samples = np.frombuffer(audio_bytes[:usable], dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    else:
        # Fallback: pure Python
        if len(audio_bytes) < sample_width:
            return 0.0
        n_samples = len(audio_bytes) // sample_width
        if n_samples == 0:
            return 0.0
        total = 0.0
        for i in range(n_samples):
            sample = struct.unpack_from("<h", audio_bytes, i * sample_width)[0]
            total += sample * sample
        return math.sqrt(total / n_samples)


# ---------------------------------------------------------------------------
# PerceptionEngine
# ---------------------------------------------------------------------------

# Type aliases for injected callables
ScreenCaptureFn = Callable[[], Coroutine[Any, Any, Optional[str]]]
CameraCaptureFn = Callable[[], Coroutine[Any, Any, Optional[str]]]
MicRecordFn = Callable[[float], Coroutine[Any, Any, Optional[bytes]]]
MicTranscribeFn = Callable[[bytes], Coroutine[Any, Any, Optional[str]]]
IsVoicePlayingFn = Callable[[], bool]


class PerceptionEngine:
    """Manages three background perception loops.

    All capture/transcription functions are injected — no direct imports
    from other modules. This keeps PerceptionEngine a pure LEAF.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        event_queue: asyncio.Queue,
        screen_fn: Optional[ScreenCaptureFn] = None,
        camera_fn: Optional[CameraCaptureFn] = None,
        mic_record_fn: Optional[MicRecordFn] = None,
        mic_transcribe_fn: Optional[MicTranscribeFn] = None,
        is_voice_playing_fn: Optional[IsVoicePlayingFn] = None,
    ):
        self._config = config
        self._queue = event_queue
        self._screen_fn = screen_fn
        self._camera_fn = camera_fn
        self._mic_record_fn = mic_record_fn
        self._mic_transcribe_fn = mic_transcribe_fn
        self._is_voice_playing_fn = is_voice_playing_fn or (lambda: False)

        self._budget = VisionBudget(config.budget_per_hour)
        self._tasks: List[asyncio.Task] = []
        self._muted = False
        self._running = False

        # Change detection state
        self._last_screen_hash: Optional[str] = None
        self._last_camera_hash: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """Suppress mic during voice playback."""
        self._muted = True

    def unmute(self) -> None:
        """Resume mic after voice playback."""
        self._muted = False

    async def get_budget_spend(self) -> float:
        """Current hourly spend."""
        return await self._budget.get_hourly_spend()

    async def request_capture(self, modality: str = "screen") -> bool:
        """Force an immediate capture, bypassing poll interval and hash check.

        Returns True if a PerceptionEvent was queued, False otherwise.
        """
        if not self._running:
            return False

        try:
            if modality == "screen" and self._screen_fn:
                img = await self._screen_fn()
                if img is None:
                    return False
                self._last_screen_hash = _compute_hash(img)
                img, media_type = _downscale_image(img, media_type="image/png")
                cost = self._config.cost_per_vision_call
                if not await self._budget.can_spend(cost):
                    return False
                await self._budget.record_spend(cost)
                event = PerceptionEvent(
                    event_type=PerceptionEventType.SCREEN_CHANGED,
                    payload=img,
                    timestamp=time.time(),
                    media_type=media_type,
                )
                await self._queue.put(event)
                return True

            elif modality == "camera" and self._camera_fn:
                img = await self._camera_fn()
                if img is None:
                    return False
                self._last_camera_hash = _compute_hash(img)
                img, media_type = _downscale_image(img, media_type="image/jpeg")
                cost = self._config.cost_per_vision_call
                if not await self._budget.can_spend(cost):
                    return False
                await self._budget.record_spend(cost)
                event = PerceptionEvent(
                    event_type=PerceptionEventType.CAMERA_FRAME,
                    payload=img,
                    timestamp=time.time(),
                    media_type=media_type,
                )
                await self._queue.put(event)
                return True

        except Exception as e:
            logger.debug(f"request_capture({modality}) failed: {e}")

        return False

    def start(self) -> None:
        """Create and start background loop tasks for enabled modalities."""
        if self._running:
            return

        self._running = True

        if self._config.screen_enabled and self._screen_fn:
            self._tasks.append(asyncio.create_task(self._screen_loop()))

        if self._config.camera_enabled and self._camera_fn:
            self._tasks.append(asyncio.create_task(self._camera_loop()))

        if self._config.ambient_mic_enabled and self._mic_record_fn and self._mic_transcribe_fn:
            self._tasks.append(asyncio.create_task(self._ambient_mic_loop()))

    async def stop(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    # --- Screen loop ---

    async def _screen_loop(self) -> None:
        """Periodically capture screen, detect changes, queue events."""
        try:
            while self._running:
                await asyncio.sleep(self._config.screen_poll_seconds)
                if not self._running:
                    break

                try:
                    img = await self._screen_fn()
                    if img is None:
                        continue

                    # Change detection
                    h = _compute_hash(img)
                    if h == self._last_screen_hash:
                        continue
                    self._last_screen_hash = h

                    # Downscale
                    img, media_type = _downscale_image(img, media_type="image/png")

                    # Budget check
                    cost = self._config.cost_per_vision_call
                    if not await self._budget.can_spend(cost):
                        logger.debug("Screen perception: budget exhausted")
                        continue

                    await self._budget.record_spend(cost)

                    event = PerceptionEvent(
                        event_type=PerceptionEventType.SCREEN_CHANGED,
                        payload=img,
                        timestamp=time.time(),
                        media_type=media_type,
                    )
                    await self._queue.put(event)

                except Exception as e:
                    logger.debug(f"Screen loop error (non-fatal): {e}")

        except asyncio.CancelledError:
            return

    # --- Camera loop ---

    async def _camera_loop(self) -> None:
        """Periodically capture webcam frame, detect changes, queue events."""
        try:
            while self._running:
                await asyncio.sleep(self._config.camera_poll_seconds)
                if not self._running:
                    break

                try:
                    img = await self._camera_fn()
                    if img is None:
                        continue

                    # Change detection
                    h = _compute_hash(img)
                    if h == self._last_camera_hash:
                        continue
                    self._last_camera_hash = h

                    # Downscale
                    img, media_type = _downscale_image(img, media_type="image/jpeg")

                    # Budget check
                    cost = self._config.cost_per_vision_call
                    if not await self._budget.can_spend(cost):
                        logger.debug("Camera perception: budget exhausted")
                        continue

                    await self._budget.record_spend(cost)

                    event = PerceptionEvent(
                        event_type=PerceptionEventType.CAMERA_FRAME,
                        payload=img,
                        timestamp=time.time(),
                        media_type=media_type,
                    )
                    await self._queue.put(event)

                except Exception as e:
                    logger.debug(f"Camera loop error (non-fatal): {e}")

        except asyncio.CancelledError:
            return

    # --- Ambient mic loop ---

    async def _ambient_mic_loop(self) -> None:
        """Continuously record short chunks, VAD, buffer speech, transcribe."""
        try:
            speech_buffer: List[bytes] = []
            speech_start_time: Optional[float] = None
            silence_start: Optional[float] = None

            while self._running:
                # Check mute and voice playing
                if self._muted or self._is_voice_playing_fn():
                    speech_buffer.clear()
                    speech_start_time = None
                    silence_start = None
                    await asyncio.sleep(0.2)
                    continue

                try:
                    # Record a short chunk
                    chunk = await self._mic_record_fn(self._config.mic_chunk_seconds)
                    if chunk is None:
                        await asyncio.sleep(0.1)
                        continue

                    # Yield to event loop between chunks
                    await asyncio.sleep(0)

                    rms = _compute_rms(chunk)
                    now = time.time()

                    if rms > self._config.vad_threshold:
                        # Speech detected
                        silence_start = None
                        if speech_start_time is None:
                            speech_start_time = now
                        speech_buffer.append(chunk)
                    else:
                        # Silence
                        if speech_start_time is not None:
                            if silence_start is None:
                                silence_start = now
                            elif (now - silence_start) >= self._config.silence_duration:
                                # Speech segment ended
                                speech_duration = now - speech_start_time
                                if speech_duration >= self._config.speech_min_duration and speech_buffer:
                                    # Concatenate and transcribe
                                    full_audio = b"".join(speech_buffer)
                                    text = await self._mic_transcribe_fn(full_audio)
                                    if text:
                                        event = PerceptionEvent(
                                            event_type=PerceptionEventType.SPEECH_DETECTED,
                                            payload=text,
                                            timestamp=now,
                                            media_type="text/plain",
                                        )
                                        await self._queue.put(event)

                                # Reset
                                speech_buffer.clear()
                                speech_start_time = None
                                silence_start = None

                except Exception as e:
                    logger.debug(f"Ambient mic error (non-fatal): {e}")
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            return
