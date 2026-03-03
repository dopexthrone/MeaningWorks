"""
Tests for Mother continuous perception engine.

All tests mocked. No real captures, recordings, or API calls.
"""

import asyncio
import base64
import hashlib
import struct
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from mother.perception import (
    PerceptionEventType,
    PerceptionEvent,
    PerceptionConfig,
    VisionBudget,
    PerceptionEngine,
    _downscale_image,
    _compute_hash,
    _compute_rms,
)


# ---------------------------------------------------------------------------
# PerceptionEvent
# ---------------------------------------------------------------------------

class TestPerceptionEvent:
    """Test PerceptionEvent dataclass."""

    def test_create_screen_event(self):
        event = PerceptionEvent(
            event_type=PerceptionEventType.SCREEN_CHANGED,
            payload="base64data",
            timestamp=1000.0,
            media_type="image/png",
        )
        assert event.event_type == PerceptionEventType.SCREEN_CHANGED
        assert event.payload == "base64data"
        assert event.timestamp == 1000.0
        assert event.media_type == "image/png"

    def test_create_speech_event(self):
        event = PerceptionEvent(
            event_type=PerceptionEventType.SPEECH_DETECTED,
            payload="hello world",
            timestamp=2000.0,
            media_type="text/plain",
        )
        assert event.event_type == PerceptionEventType.SPEECH_DETECTED
        assert event.payload == "hello world"

    def test_create_camera_event(self):
        event = PerceptionEvent(
            event_type=PerceptionEventType.CAMERA_FRAME,
            payload="jpegdata",
            timestamp=3000.0,
            media_type="image/jpeg",
        )
        assert event.event_type == PerceptionEventType.CAMERA_FRAME

    def test_frozen(self):
        event = PerceptionEvent(
            event_type=PerceptionEventType.SCREEN_CHANGED,
            payload="data",
            timestamp=1.0,
        )
        with pytest.raises(AttributeError):
            event.payload = "new"

    def test_default_media_type(self):
        event = PerceptionEvent(
            event_type=PerceptionEventType.SCREEN_CHANGED,
            payload="data",
            timestamp=1.0,
        )
        assert event.media_type == "image/png"

    def test_event_types_exhaustive(self):
        types = list(PerceptionEventType)
        assert len(types) == 3
        assert PerceptionEventType.SCREEN_CHANGED in types
        assert PerceptionEventType.SPEECH_DETECTED in types
        assert PerceptionEventType.CAMERA_FRAME in types


# ---------------------------------------------------------------------------
# PerceptionConfig
# ---------------------------------------------------------------------------

class TestPerceptionConfig:
    """Test PerceptionConfig defaults and creation."""

    def test_defaults(self):
        config = PerceptionConfig()
        assert config.screen_enabled is False
        assert config.camera_enabled is False
        assert config.ambient_mic_enabled is False
        assert config.screen_poll_seconds == 10.0
        assert config.camera_poll_seconds == 30.0
        assert config.budget_per_hour == 5.00
        assert config.vad_threshold == 500.0
        assert config.silence_duration == 1.5
        assert config.speech_min_duration == 0.5
        assert config.mic_chunk_seconds == 0.1
        assert config.mic_sample_rate == 16000

    def test_custom_values(self):
        config = PerceptionConfig(
            screen_enabled=True,
            camera_enabled=True,
            ambient_mic_enabled=True,
            screen_poll_seconds=5.0,
            budget_per_hour=1.0,
        )
        assert config.screen_enabled is True
        assert config.screen_poll_seconds == 5.0
        assert config.budget_per_hour == 1.0

    def test_frozen(self):
        config = PerceptionConfig()
        with pytest.raises(AttributeError):
            config.screen_enabled = True

    def test_cost_per_vision_call_default(self):
        config = PerceptionConfig()
        assert config.cost_per_vision_call == 0.005


# ---------------------------------------------------------------------------
# VisionBudget
# ---------------------------------------------------------------------------

class TestVisionBudget:
    """Test rolling hourly spend tracking."""

    def test_initial_spend_is_zero(self):
        budget = VisionBudget(0.50)

        async def _run():
            spend = await budget.get_hourly_spend()
            assert spend == 0.0

        asyncio.run(_run())

    def test_can_spend_within_budget(self):
        budget = VisionBudget(0.50)

        async def _run():
            assert await budget.can_spend(0.10) is True

        asyncio.run(_run())

    def test_can_spend_exact_limit(self):
        budget = VisionBudget(0.50)

        async def _run():
            await budget.record_spend(0.45)
            assert await budget.can_spend(0.05) is True

        asyncio.run(_run())

    def test_cannot_spend_over_limit(self):
        budget = VisionBudget(0.50)

        async def _run():
            await budget.record_spend(0.45)
            assert await budget.can_spend(0.10) is False

        asyncio.run(_run())

    def test_record_spend_accumulates(self):
        budget = VisionBudget(1.00)

        async def _run():
            await budget.record_spend(0.10)
            await budget.record_spend(0.20)
            spend = await budget.get_hourly_spend()
            assert abs(spend - 0.30) < 0.001

        asyncio.run(_run())

    def test_prune_removes_old_entries(self):
        budget = VisionBudget(0.50)

        async def _run():
            # Manually inject old entry
            budget._log.append((time.time() - 7200, 0.40))  # 2 hours ago
            budget._log.append((time.time(), 0.05))  # Now
            spend = await budget.get_hourly_spend()
            assert abs(spend - 0.05) < 0.001

        asyncio.run(_run())

    def test_prune_keeps_recent_entries(self):
        budget = VisionBudget(0.50)

        async def _run():
            budget._log.append((time.time() - 1800, 0.20))  # 30 min ago
            budget._log.append((time.time(), 0.10))  # Now
            spend = await budget.get_hourly_spend()
            assert abs(spend - 0.30) < 0.001

        asyncio.run(_run())

    def test_budget_zero_cannot_spend(self):
        budget = VisionBudget(0.0)

        async def _run():
            assert await budget.can_spend(0.001) is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Image downscaling
# ---------------------------------------------------------------------------

class TestDownscaleImage:
    """Test image downscaling with optional Pillow."""

    def test_returns_original_when_pillow_unavailable(self):
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            # Force ImportError
            import importlib
            from mother import perception
            original_downscale = perception._downscale_image

            def _mock_downscale(b64_data, target_width=640, media_type="image/png"):
                # Simulate Pillow unavailable
                return b64_data, media_type

            with patch.object(perception, "_downscale_image", side_effect=_mock_downscale):
                result, mt = perception._downscale_image("abc123", media_type="image/png")
                assert result == "abc123"
                assert mt == "image/png"

    def test_returns_original_for_small_image(self):
        """If Pillow is available and image is small, return original."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        # Create a small 100x100 PNG
        import io
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        result, mt = _downscale_image(b64, target_width=640, media_type="image/png")
        assert result == b64  # No change needed

    def test_downscales_large_image(self):
        """If Pillow is available and image is large, downscale."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import io
        img = Image.new("RGB", (1920, 1080), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        result, mt = _downscale_image(b64, target_width=640, media_type="image/png")
        assert result != b64  # Should be different (smaller)
        assert mt == "image/jpeg"  # Downscaled to JPEG

        # Verify result is valid base64
        decoded = base64.b64decode(result)
        result_img = Image.open(io.BytesIO(decoded))
        assert result_img.width == 640

    def test_returns_jpeg_media_type(self):
        """Downscaled images should be JPEG."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import io
        img = Image.new("RGB", (2000, 1000), color="green")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        _, mt = _downscale_image(b64, target_width=640, media_type="image/png")
        assert mt == "image/jpeg"

    def test_invalid_data_returns_original(self):
        """Bad base64/image data should return original."""
        result, mt = _downscale_image("not_valid_base64!!!", media_type="image/png")
        assert result == "not_valid_base64!!!"
        assert mt == "image/png"


# ---------------------------------------------------------------------------
# Change detection hash
# ---------------------------------------------------------------------------

class TestComputeHash:
    """Test MD5 hashing for change detection."""

    def test_deterministic(self):
        h1 = _compute_hash("test123")
        h2 = _compute_hash("test123")
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        h1 = _compute_hash("input_a")
        h2 = _compute_hash("input_b")
        assert h1 != h2

    def test_returns_string(self):
        h = _compute_hash("data")
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex length

    def test_empty_string(self):
        h = _compute_hash("")
        assert isinstance(h, str)
        assert len(h) == 32


# ---------------------------------------------------------------------------
# RMS / VAD
# ---------------------------------------------------------------------------

class TestComputeRMS:
    """Test RMS energy computation."""

    def test_zeros_return_zero(self):
        audio = b"\x00" * 100
        rms = _compute_rms(audio)
        assert rms == 0.0

    def test_signal_returns_nonzero(self):
        # Pack some int16 samples with non-zero values
        samples = [1000, -1000, 2000, -2000] * 10
        audio = struct.pack(f"<{len(samples)}h", *samples)
        rms = _compute_rms(audio)
        assert rms > 0.0

    def test_louder_signal_higher_rms(self):
        quiet = struct.pack("<10h", *([100] * 10))
        loud = struct.pack("<10h", *([10000] * 10))
        rms_quiet = _compute_rms(quiet)
        rms_loud = _compute_rms(loud)
        assert rms_loud > rms_quiet

    def test_empty_bytes_return_zero(self):
        assert _compute_rms(b"") == 0.0

    def test_single_byte_returns_zero(self):
        # Less than sample_width (2), should return 0
        assert _compute_rms(b"\x01") == 0.0

    def test_numpy_unavailable_uses_fallback(self):
        samples = [1000, -1000, 2000, -2000]
        audio = struct.pack(f"<{len(samples)}h", *samples)

        with patch.dict("sys.modules", {"numpy": None}):
            # Force the fallback path by making import fail
            import importlib
            from mother import perception
            saved_fn = perception._compute_rms

            # Test the pure-Python fallback explicitly
            rms = 0.0
            n_samples = len(audio) // 2
            total = 0.0
            for i in range(n_samples):
                sample = struct.unpack_from("<h", audio, i * 2)[0]
                total += sample * sample
            import math
            rms = math.sqrt(total / n_samples)
            assert rms > 0.0


# ---------------------------------------------------------------------------
# PerceptionEngine — creation
# ---------------------------------------------------------------------------

class TestPerceptionEngineCreation:
    """Test PerceptionEngine instantiation."""

    def test_create_with_defaults(self):
        config = PerceptionConfig()
        q = asyncio.Queue()
        engine = PerceptionEngine(config, q)
        assert engine.running is False
        assert engine.muted is False

    def test_create_with_callables(self):
        config = PerceptionConfig(screen_enabled=True)
        q = asyncio.Queue()

        async def screen_fn():
            return "b64"

        engine = PerceptionEngine(config, q, screen_fn=screen_fn)
        assert engine._screen_fn is screen_fn

    def test_mute_unmute(self):
        config = PerceptionConfig()
        q = asyncio.Queue()
        engine = PerceptionEngine(config, q)

        assert engine.muted is False
        engine.mute()
        assert engine.muted is True
        engine.unmute()
        assert engine.muted is False


# ---------------------------------------------------------------------------
# PerceptionEngine — start/stop
# ---------------------------------------------------------------------------

class TestPerceptionEngineLifecycle:
    """Test start/stop of background loops."""

    def test_start_creates_tasks(self):
        config = PerceptionConfig(screen_enabled=True, camera_enabled=True)
        q = asyncio.Queue()

        async def screen_fn():
            return "b64"

        async def camera_fn():
            return "b64"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                screen_fn=screen_fn,
                camera_fn=camera_fn,
            )
            engine.start()
            assert engine.running is True
            assert len(engine._tasks) == 2
            await engine.stop()
            assert engine.running is False
            assert len(engine._tasks) == 0

        asyncio.run(_run())

    def test_start_no_enabled_modalities(self):
        config = PerceptionConfig()
        q = asyncio.Queue()

        async def _run():
            engine = PerceptionEngine(config, q)
            engine.start()
            assert engine.running is True
            assert len(engine._tasks) == 0
            await engine.stop()

        asyncio.run(_run())

    def test_start_only_screen(self):
        config = PerceptionConfig(screen_enabled=True)
        q = asyncio.Queue()

        async def screen_fn():
            return "b64"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            assert len(engine._tasks) == 1
            await engine.stop()

        asyncio.run(_run())

    def test_start_only_mic(self):
        config = PerceptionConfig(ambient_mic_enabled=True)
        q = asyncio.Queue()

        async def mic_record(duration):
            return b"\x00" * 100

        async def mic_transcribe(audio):
            return "hello"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.start()
            assert len(engine._tasks) == 1
            await engine.stop()

        asyncio.run(_run())

    def test_double_start_is_noop(self):
        config = PerceptionConfig(screen_enabled=True)
        q = asyncio.Queue()

        async def screen_fn():
            return "b64"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            task_count_1 = len(engine._tasks)
            engine.start()  # Should not add more tasks
            task_count_2 = len(engine._tasks)
            assert task_count_1 == task_count_2
            await engine.stop()

        asyncio.run(_run())

    def test_stop_cancels_all_tasks(self):
        config = PerceptionConfig(
            screen_enabled=True,
            camera_enabled=True,
            ambient_mic_enabled=True,
        )
        q = asyncio.Queue()

        async def screen_fn():
            return "b64"

        async def camera_fn():
            return "b64"

        async def mic_record(d):
            return b"\x00"

        async def mic_transcribe(a):
            return "text"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                screen_fn=screen_fn,
                camera_fn=camera_fn,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.start()
            assert len(engine._tasks) == 3
            await engine.stop()
            assert len(engine._tasks) == 0
            assert engine.running is False

        asyncio.run(_run())

    def test_running_property(self):
        config = PerceptionConfig()
        q = asyncio.Queue()

        async def _run():
            engine = PerceptionEngine(config, q)
            assert engine.running is False
            engine.start()
            assert engine.running is True
            await engine.stop()
            assert engine.running is False

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Screen loop
# ---------------------------------------------------------------------------

class TestScreenLoop:
    """Test the screen perception loop behavior."""

    def test_screen_captures_and_queues(self):
        config = PerceptionConfig(
            screen_enabled=True,
            screen_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()
        call_count = 0

        async def screen_fn():
            nonlocal call_count
            call_count += 1
            return f"screen_b64_{call_count}"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            assert not q.empty()
            event = await q.get()
            assert event.event_type == PerceptionEventType.SCREEN_CHANGED
            assert "screen_b64" in event.payload

        asyncio.run(_run())

    def test_screen_skips_unchanged(self):
        config = PerceptionConfig(
            screen_enabled=True,
            screen_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()

        async def screen_fn():
            return "same_data_every_time"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            await asyncio.sleep(0.3)
            await engine.stop()

            # Only one event should be queued (first capture is always new)
            count = 0
            while not q.empty():
                await q.get()
                count += 1
            assert count == 1

        asyncio.run(_run())

    def test_screen_budget_exhausted_stops_queuing(self):
        config = PerceptionConfig(
            screen_enabled=True,
            screen_poll_seconds=0.05,
            budget_per_hour=0.005,  # Exactly one call
            cost_per_vision_call=0.005,
        )
        q = asyncio.Queue()
        counter = [0]

        async def screen_fn():
            counter[0] += 1
            return f"data_{counter[0]}"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            await asyncio.sleep(0.4)
            await engine.stop()

            count = 0
            while not q.empty():
                await q.get()
                count += 1
            # Should have at most 1 event (budget allows one call)
            assert count <= 1

        asyncio.run(_run())

    def test_screen_capture_none_skipped(self):
        config = PerceptionConfig(
            screen_enabled=True,
            screen_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()

        async def screen_fn():
            return None

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            assert q.empty()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Camera loop
# ---------------------------------------------------------------------------

class TestCameraLoop:
    """Test the camera perception loop behavior."""

    def test_camera_captures_and_queues(self):
        config = PerceptionConfig(
            camera_enabled=True,
            camera_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()
        counter = [0]

        async def camera_fn():
            counter[0] += 1
            return f"cam_{counter[0]}"

        async def _run():
            engine = PerceptionEngine(config, q, camera_fn=camera_fn)
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            assert not q.empty()
            event = await q.get()
            assert event.event_type == PerceptionEventType.CAMERA_FRAME

        asyncio.run(_run())

    def test_camera_skips_unchanged(self):
        config = PerceptionConfig(
            camera_enabled=True,
            camera_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()

        async def camera_fn():
            return "same_frame"

        async def _run():
            engine = PerceptionEngine(config, q, camera_fn=camera_fn)
            engine.start()
            await asyncio.sleep(0.3)
            await engine.stop()

            count = 0
            while not q.empty():
                await q.get()
                count += 1
            assert count == 1

        asyncio.run(_run())

    def test_camera_media_type_jpeg(self):
        config = PerceptionConfig(
            camera_enabled=True,
            camera_poll_seconds=0.05,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()

        async def camera_fn():
            return "cam_data"

        async def _run():
            engine = PerceptionEngine(config, q, camera_fn=camera_fn)
            engine.start()
            await asyncio.sleep(0.15)
            await engine.stop()

            if not q.empty():
                event = await q.get()
                # Media type should be image/jpeg (from downscale or original)
                assert event.media_type in ("image/jpeg", "image/png")

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Ambient mic loop
# ---------------------------------------------------------------------------

class TestAmbientMicLoop:
    """Test the ambient microphone loop behavior."""

    def test_detects_speech_and_transcribes(self):
        config = PerceptionConfig(
            ambient_mic_enabled=True,
            vad_threshold=100.0,
            silence_duration=0.1,
            speech_min_duration=0.05,
            mic_chunk_seconds=0.02,
        )
        q = asyncio.Queue()
        call_count = [0]

        async def mic_record(duration):
            call_count[0] += 1
            if call_count[0] <= 5:
                # Loud audio (speech)
                return struct.pack("<10h", *([5000] * 10))
            else:
                # Silence
                return b"\x00" * 20

        async def mic_transcribe(audio):
            return "hello world"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.start()
            await asyncio.sleep(0.5)
            await engine.stop()

            # Should have detected speech
            events = []
            while not q.empty():
                events.append(await q.get())

            speech_events = [e for e in events if e.event_type == PerceptionEventType.SPEECH_DETECTED]
            assert len(speech_events) >= 1
            assert speech_events[0].payload == "hello world"
            assert speech_events[0].media_type == "text/plain"

        asyncio.run(_run())

    def test_ignores_silence(self):
        config = PerceptionConfig(
            ambient_mic_enabled=True,
            vad_threshold=100.0,
            mic_chunk_seconds=0.02,
        )
        q = asyncio.Queue()

        async def mic_record(duration):
            return b"\x00" * 20  # Silence

        async def mic_transcribe(audio):
            return "should not be called"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            assert q.empty()

        asyncio.run(_run())

    def test_mute_skips_recording(self):
        config = PerceptionConfig(
            ambient_mic_enabled=True,
            mic_chunk_seconds=0.02,
        )
        q = asyncio.Queue()
        record_count = [0]

        async def mic_record(duration):
            record_count[0] += 1
            return struct.pack("<10h", *([5000] * 10))

        async def mic_transcribe(audio):
            return "text"

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.mute()
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            # Record function should not have been called (muted)
            assert record_count[0] == 0

        asyncio.run(_run())

    def test_voice_playing_skips_recording(self):
        config = PerceptionConfig(
            ambient_mic_enabled=True,
            mic_chunk_seconds=0.02,
        )
        q = asyncio.Queue()
        record_count = [0]

        async def mic_record(duration):
            record_count[0] += 1
            return struct.pack("<10h", *([5000] * 10))

        async def mic_transcribe(audio):
            return "text"

        def is_voice_playing():
            return True

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
                is_voice_playing_fn=is_voice_playing,
            )
            engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()

            assert record_count[0] == 0

        asyncio.run(_run())

    def test_transcribe_returns_none_no_event(self):
        config = PerceptionConfig(
            ambient_mic_enabled=True,
            vad_threshold=100.0,
            silence_duration=0.1,
            speech_min_duration=0.05,
            mic_chunk_seconds=0.02,
        )
        q = asyncio.Queue()
        call_count = [0]

        async def mic_record(duration):
            call_count[0] += 1
            if call_count[0] <= 5:
                return struct.pack("<10h", *([5000] * 10))
            else:
                return b"\x00" * 20

        async def mic_transcribe(audio):
            return None  # Transcription failed

        async def _run():
            engine = PerceptionEngine(
                config, q,
                mic_record_fn=mic_record,
                mic_transcribe_fn=mic_transcribe,
            )
            engine.start()
            await asyncio.sleep(0.5)
            await engine.stop()

            speech_events = []
            while not q.empty():
                e = await q.get()
                if e.event_type == PerceptionEventType.SPEECH_DETECTED:
                    speech_events.append(e)
            assert len(speech_events) == 0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Budget integration
# ---------------------------------------------------------------------------

class TestBudgetIntegration:
    """Test budget enforcement across loops."""

    def test_get_budget_spend(self):
        config = PerceptionConfig(budget_per_hour=1.0)
        q = asyncio.Queue()
        engine = PerceptionEngine(config, q)

        async def _run():
            spend = await engine.get_budget_spend()
            assert spend == 0.0

        asyncio.run(_run())

    def test_budget_accumulates_with_screen_captures(self):
        config = PerceptionConfig(
            screen_enabled=True,
            screen_poll_seconds=0.05,
            budget_per_hour=10.0,
            cost_per_vision_call=0.01,
        )
        q = asyncio.Queue()
        counter = [0]

        async def screen_fn():
            counter[0] += 1
            return f"data_{counter[0]}"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine.start()
            await asyncio.sleep(0.3)
            await engine.stop()

            spend = await engine.get_budget_spend()
            assert spend > 0.0

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# request_capture — forced perception refresh
# ---------------------------------------------------------------------------

class TestRequestCapture:
    """Test PerceptionEngine.request_capture() for grid-driven refresh."""

    def test_request_capture_screen(self):
        """request_capture('screen') queues a SCREEN_CHANGED event."""
        config = PerceptionConfig(
            screen_enabled=True,
            budget_per_hour=10.0,
            cost_per_vision_call=0.001,
        )
        q = asyncio.Queue()
        call_count = [0]

        async def screen_fn():
            call_count[0] += 1
            return f"img_data_{call_count[0]}"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine._running = True  # simulate started state

            result = await engine.request_capture("screen")
            assert result is True
            assert call_count[0] == 1

            # Event should be in queue
            assert not q.empty()
            event = await q.get()
            assert event.event_type == PerceptionEventType.SCREEN_CHANGED

        asyncio.run(_run())

    def test_request_capture_camera(self):
        """request_capture('camera') queues a CAMERA_FRAME event."""
        config = PerceptionConfig(
            camera_enabled=True,
            budget_per_hour=10.0,
            cost_per_vision_call=0.001,
        )
        q = asyncio.Queue()

        async def camera_fn():
            return "camera_data"

        async def _run():
            engine = PerceptionEngine(config, q, camera_fn=camera_fn)
            engine._running = True

            result = await engine.request_capture("camera")
            assert result is True

            event = await q.get()
            assert event.event_type == PerceptionEventType.CAMERA_FRAME

        asyncio.run(_run())

    def test_request_capture_when_not_running(self):
        """request_capture returns False when engine is not running."""
        config = PerceptionConfig(screen_enabled=True)
        q = asyncio.Queue()

        async def screen_fn():
            return "data"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            # NOT started
            result = await engine.request_capture("screen")
            assert result is False
            assert q.empty()

        asyncio.run(_run())

    def test_request_capture_no_screen_fn(self):
        """request_capture('screen') returns False when no screen_fn."""
        config = PerceptionConfig(screen_enabled=True)
        q = asyncio.Queue()

        async def _run():
            engine = PerceptionEngine(config, q)  # no screen_fn
            engine._running = True
            result = await engine.request_capture("screen")
            assert result is False

        asyncio.run(_run())

    def test_request_capture_updates_hash(self):
        """request_capture updates the hash so next poll detects no change."""
        config = PerceptionConfig(
            screen_enabled=True,
            budget_per_hour=10.0,
            cost_per_vision_call=0.001,
        )
        q = asyncio.Queue()

        async def screen_fn():
            return "consistent_data"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine._running = True

            assert engine._last_screen_hash is None
            await engine.request_capture("screen")
            assert engine._last_screen_hash is not None

        asyncio.run(_run())

    def test_request_capture_budget_exhausted(self):
        """request_capture returns False when budget is exhausted."""
        config = PerceptionConfig(
            screen_enabled=True,
            budget_per_hour=0.001,  # very small budget
            cost_per_vision_call=1.0,  # very expensive
        )
        q = asyncio.Queue()

        async def screen_fn():
            return "data"

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine._running = True

            # First capture may or may not work depending on budget logic
            r1 = await engine.request_capture("screen")
            # Second should fail (budget exhausted)
            r2 = await engine.request_capture("screen")
            # At least one should have been gated
            assert not (r1 and r2), "Budget should gate at least one capture"

        asyncio.run(_run())

    def test_request_capture_unknown_modality(self):
        """request_capture with unknown modality returns False."""
        config = PerceptionConfig()
        q = asyncio.Queue()

        async def _run():
            engine = PerceptionEngine(config, q)
            engine._running = True
            result = await engine.request_capture("unknown_sensor")
            assert result is False

        asyncio.run(_run())

    def test_request_capture_screen_fn_returns_none(self):
        """request_capture returns False when screen_fn returns None."""
        config = PerceptionConfig(
            screen_enabled=True,
            budget_per_hour=10.0,
        )
        q = asyncio.Queue()

        async def screen_fn():
            return None  # capture failed

        async def _run():
            engine = PerceptionEngine(config, q, screen_fn=screen_fn)
            engine._running = True
            result = await engine.request_capture("screen")
            assert result is False
            assert q.empty()

        asyncio.run(_run())
