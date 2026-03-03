"""
Tests for real-time duplex voice — llm_server, voice_duplex, bridge/config additions.

Tests are pure unit tests using mocks. No real audio hardware, no real ElevenLabs API.
All async code is run via asyncio.run() (project convention: no pytest-asyncio).
"""

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Skip entire module if aiohttp is not installed — duplex voice requires it
# for LocalLLMServer (the core dependency for all tests in this file).
pytest.importorskip("aiohttp", reason="aiohttp required for duplex voice tests")


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════
# §1 — LocalLLMServer tests
# ═══════════════════════════════════════════════════════════════════════


class TestLocalLLMServer:
    """Tests for mother/llm_server.py."""

    def test_import(self):
        """Module imports cleanly."""
        from mother.llm_server import LocalLLMServer, is_llm_server_available
        assert is_llm_server_available()  # aiohttp is installed

    def test_default_config(self):
        """Default host/port are localhost:11411."""
        from mother.llm_server import LocalLLMServer
        server = LocalLLMServer()
        assert server.url == "http://127.0.0.1:11411"
        assert server.completions_url == "http://127.0.0.1:11411/v1/chat/completions"
        assert not server.running
        assert server.request_count == 0

    def test_custom_port(self):
        """Custom port is respected."""
        from mother.llm_server import LocalLLMServer
        server = LocalLLMServer(port=12345)
        assert server.url == "http://127.0.0.1:12345"

    def test_set_handlers(self):
        """Handlers can be set."""
        from mother.llm_server import LocalLLMServer
        server = LocalLLMServer()

        def sys_fn():
            return "test prompt"

        async def stream_fn(messages, system_prompt):
            yield "hello"

        server.set_handlers(sys_fn, stream_fn)
        assert server._system_prompt_fn is sys_fn
        assert server._chat_stream_fn is stream_fn

    def test_start_without_handlers_raises(self):
        """Starting without handlers raises RuntimeError."""
        from mother.llm_server import LocalLLMServer

        async def _test():
            server = LocalLLMServer()
            with pytest.raises(RuntimeError, match="set_handlers"):
                await server.start()

        _run(_test())

    def test_start_stop_lifecycle(self):
        """Server starts and stops cleanly."""
        from mother.llm_server import LocalLLMServer

        async def _dummy(messages, system_prompt):
            yield "test"

        async def _test():
            server = LocalLLMServer(port=19411)
            server.set_handlers(lambda: "prompt", _dummy)
            await server.start()
            assert server.running
            await server.stop()
            assert not server.running

        _run(_test())

    def test_double_start_warns(self):
        """Starting twice doesn't crash."""
        from mother.llm_server import LocalLLMServer

        async def _dummy(messages, system_prompt):
            yield "test"

        async def _test():
            server = LocalLLMServer(port=19412)
            server.set_handlers(lambda: "prompt", _dummy)
            await server.start()
            try:
                await server.start()  # Should log warning, not crash
                assert server.running
            finally:
                await server.stop()

        _run(_test())

    def test_stop_when_not_running(self):
        """Stopping when not running is a no-op."""
        from mother.llm_server import LocalLLMServer

        async def _test():
            server = LocalLLMServer()
            await server.stop()  # Should not raise

        _run(_test())

    def test_health_endpoint(self):
        """GET /health returns status."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        async def _dummy(messages, system_prompt):
            yield "test"

        async def _test():
            server = LocalLLMServer(port=19413)
            server.set_handlers(lambda: "prompt", _dummy)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://127.0.0.1:19413/health") as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["status"] == "ok"
            finally:
                await server.stop()

        _run(_test())

    def test_streaming_completions(self):
        """POST /v1/chat/completions with stream=true returns SSE."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        captured_system_prompt = None

        def sys_fn():
            return "You are Mother."

        async def stream_fn(messages, system_prompt):
            nonlocal captured_system_prompt
            captured_system_prompt = system_prompt
            for word in ["Hello", " ", "world"]:
                yield word

        async def _test():
            nonlocal captured_system_prompt
            server = LocalLLMServer(port=19414)
            server.set_handlers(sys_fn, stream_fn)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:19414/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": True,
                        },
                    ) as resp:
                        assert resp.status == 200
                        assert resp.content_type == "text/event-stream"

                        collected = []
                        async for line in resp.content:
                            decoded = line.decode("utf-8").strip()
                            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                                chunk = json.loads(decoded[6:])
                                delta = chunk["choices"][0]["delta"]
                                if "content" in delta:
                                    collected.append(delta["content"])

                        assert "".join(collected) == "Hello world"
                        assert captured_system_prompt == "You are Mother."
            finally:
                await server.stop()

        _run(_test())

    def test_non_streaming_completions(self):
        """POST /v1/chat/completions with stream=false returns JSON."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        async def stream_fn(messages, system_prompt):
            yield "Hello world"

        async def _test():
            server = LocalLLMServer(port=19415)
            server.set_handlers(lambda: "", stream_fn)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:19415/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "Hi"}],
                            "stream": False,
                        },
                    ) as resp:
                        assert resp.status == 200
                        data = await resp.json()
                        assert data["choices"][0]["message"]["content"] == "Hello world"
            finally:
                await server.stop()

        _run(_test())

    def test_malformed_request(self):
        """Malformed JSON returns 400."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        async def _dummy(messages, system_prompt):
            yield "test"

        async def _test():
            server = LocalLLMServer(port=19416)
            server.set_handlers(lambda: "", _dummy)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:19416/v1/chat/completions",
                        data=b"not json",
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        assert resp.status == 400
            finally:
                await server.stop()

        _run(_test())

    def test_empty_messages(self):
        """Empty messages array returns 400."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        async def _dummy(messages, system_prompt):
            yield "test"

        async def _test():
            server = LocalLLMServer(port=19417)
            server.set_handlers(lambda: "", _dummy)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:19417/v1/chat/completions",
                        json={"messages": [], "stream": False},
                    ) as resp:
                        assert resp.status == 400
            finally:
                await server.stop()

        _run(_test())

    def test_request_count_increments(self):
        """Request count increments on each call."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        async def stream_fn(messages, system_prompt):
            yield "ok"

        async def _test():
            server = LocalLLMServer(port=19418)
            server.set_handlers(lambda: "", stream_fn)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    for i in range(3):
                        async with session.post(
                            "http://127.0.0.1:19418/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": "Hi"}]},
                        ) as resp:
                            await resp.read()
                assert server.request_count == 3
            finally:
                await server.stop()

        _run(_test())

    def test_system_prompt_error_resilience(self):
        """Server handles system_prompt_fn errors gracefully."""
        from mother.llm_server import LocalLLMServer
        import aiohttp

        def bad_sys_fn():
            raise ValueError("oops")

        async def stream_fn(messages, system_prompt):
            yield "ok"

        async def _test():
            server = LocalLLMServer(port=19419)
            server.set_handlers(bad_sys_fn, stream_fn)
            await server.start()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "http://127.0.0.1:19419/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "Hi"}]},
                    ) as resp:
                        assert resp.status == 200
            finally:
                await server.stop()

        _run(_test())


# ═══════════════════════════════════════════════════════════════════════
# §2 — SoundDeviceAudioInterface tests
# ═══════════════════════════════════════════════════════════════════════


class TestSoundDeviceAudioInterface:
    """Tests for the sounddevice-based AudioInterface."""

    def test_import(self):
        """Module imports cleanly."""
        from mother.voice_duplex import SoundDeviceAudioInterface, is_duplex_available
        assert isinstance(is_duplex_available(), bool)

    def test_init(self):
        """Constructor sets initial state."""
        from mother.voice_duplex import SoundDeviceAudioInterface
        ai = SoundDeviceAudioInterface()
        assert ai._input_stream is None
        assert ai._output_stream is None
        assert ai._input_callback is None

    @patch("mother.voice_duplex.sd")
    def test_start_opens_streams(self, mock_sd):
        """start() opens input and output streams."""
        from mother.voice_duplex import SoundDeviceAudioInterface

        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_sd.InputStream.return_value = mock_in
        mock_sd.OutputStream.return_value = mock_out

        ai = SoundDeviceAudioInterface()
        callback = MagicMock()
        ai.start(callback)

        mock_sd.InputStream.assert_called_once()
        mock_sd.OutputStream.assert_called_once()
        mock_in.start.assert_called_once()
        mock_out.start.assert_called_once()
        assert ai._input_callback is callback

        ai.stop()

    @patch("mother.voice_duplex.sd")
    def test_stop_closes_streams(self, mock_sd):
        """stop() closes both streams."""
        from mother.voice_duplex import SoundDeviceAudioInterface

        mock_in = MagicMock()
        mock_out = MagicMock()
        mock_sd.InputStream.return_value = mock_in
        mock_sd.OutputStream.return_value = mock_out

        ai = SoundDeviceAudioInterface()
        ai.start(MagicMock())
        ai.stop()

        mock_in.stop.assert_called_once()
        mock_in.close.assert_called_once()
        mock_out.stop.assert_called_once()
        mock_out.close.assert_called_once()
        assert ai._input_callback is None

    def test_output_queues_audio(self):
        """output() puts audio in the queue."""
        from mother.voice_duplex import SoundDeviceAudioInterface
        ai = SoundDeviceAudioInterface()
        ai.output(b"\x00\x01\x02\x03")
        assert not ai._output_queue.empty()
        assert ai._output_queue.get() == b"\x00\x01\x02\x03"

    def test_interrupt_clears_queue(self):
        """interrupt() empties the output queue."""
        from mother.voice_duplex import SoundDeviceAudioInterface
        ai = SoundDeviceAudioInterface()
        for i in range(10):
            ai.output(b"\x00" * 100)
        assert not ai._output_queue.empty()
        ai.interrupt()
        assert ai._output_queue.empty()

    @patch("mother.voice_duplex.sd")
    def test_input_callback_dispatches(self, mock_sd):
        """Input stream callback dispatches bytes to input_callback."""
        from mother.voice_duplex import SoundDeviceAudioInterface

        mock_sd.InputStream.return_value = MagicMock()
        mock_sd.OutputStream.return_value = MagicMock()

        ai = SoundDeviceAudioInterface()
        callback = MagicMock()
        ai.start(callback)

        # Simulate audio input
        test_data = b"\x00\x01" * 4000
        ai._input_stream_callback(test_data, 4000, {}, None)
        callback.assert_called_once()

        ai.stop()

    def test_stop_when_not_started(self):
        """stop() is safe when never started."""
        from mother.voice_duplex import SoundDeviceAudioInterface
        ai = SoundDeviceAudioInterface()
        ai.stop()  # Should not raise

    @patch("mother.voice_duplex.sd")
    def test_double_stop(self, mock_sd):
        """Double stop doesn't crash."""
        from mother.voice_duplex import SoundDeviceAudioInterface

        mock_sd.InputStream.return_value = MagicMock()
        mock_sd.OutputStream.return_value = MagicMock()

        ai = SoundDeviceAudioInterface()
        ai.start(MagicMock())
        ai.stop()
        ai.stop()  # Should not raise

    def test_interrupt_when_empty(self):
        """interrupt() on empty queue is a no-op."""
        from mother.voice_duplex import SoundDeviceAudioInterface
        ai = SoundDeviceAudioInterface()
        ai.interrupt()  # Should not raise

    @patch("mother.voice_duplex.sd")
    def test_callback_ignored_after_stop(self, mock_sd):
        """Input callback doesn't dispatch after stop."""
        from mother.voice_duplex import SoundDeviceAudioInterface

        mock_sd.InputStream.return_value = MagicMock()
        mock_sd.OutputStream.return_value = MagicMock()

        ai = SoundDeviceAudioInterface()
        callback = MagicMock()
        ai.start(callback)
        ai._should_stop.set()

        ai._input_stream_callback(b"\x00" * 100, 50, {}, None)
        callback.assert_not_called()

        ai.stop()


# ═══════════════════════════════════════════════════════════════════════
# §3 — DuplexVoiceSession tests
# ═══════════════════════════════════════════════════════════════════════


class TestDuplexVoiceSession:
    """Tests for the duplex voice session manager."""

    def test_init_state(self):
        """Initial state is IDLE."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexState
        session = DuplexVoiceSession(api_key="test-key")
        assert session.state == DuplexState.IDLE
        assert session.agent_id == ""
        assert not session.active

    def test_custom_config(self):
        """Custom configuration is stored."""
        from mother.voice_duplex import DuplexVoiceSession
        session = DuplexVoiceSession(
            api_key="key",
            server_url="http://localhost:9999/v1/chat/completions",
            agent_id="agent-123",
            voice_id="voice-456",
            language="es",
        )
        assert session._server_url == "http://localhost:9999/v1/chat/completions"
        assert session.agent_id == "agent-123"
        assert session._voice_id == "voice-456"
        assert session._language == "es"

    def test_ensure_agent_creates_new(self):
        """ensure_agent() creates a new agent when none cached."""
        from mother.voice_duplex import DuplexVoiceSession

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.agent_id = "new-agent-id"
        mock_client.conversational_ai.agents.create.return_value = mock_result

        session = DuplexVoiceSession(api_key="test-key")
        session._client = mock_client

        agent_id = _run(session.ensure_agent())
        assert agent_id == "new-agent-id"
        assert session.agent_id == "new-agent-id"

    def test_ensure_agent_reuses_cached(self):
        """ensure_agent() reuses cached agent_id if valid."""
        from mother.voice_duplex import DuplexVoiceSession

        mock_client = MagicMock()
        mock_client.conversational_ai.agents.get.return_value = MagicMock()

        session = DuplexVoiceSession(
            api_key="test-key", agent_id="cached-id"
        )
        session._client = mock_client

        agent_id = _run(session.ensure_agent())
        assert agent_id == "cached-id"
        mock_client.conversational_ai.agents.create.assert_not_called()

    def test_ensure_agent_falls_through_invalid_cache(self):
        """ensure_agent() creates new when cached agent doesn't exist."""
        from mother.voice_duplex import DuplexVoiceSession

        mock_client = MagicMock()
        mock_client.conversational_ai.agents.get.side_effect = Exception("not found")
        mock_result = MagicMock()
        mock_result.agent_id = "fallback-id"
        mock_client.conversational_ai.agents.create.return_value = mock_result

        session = DuplexVoiceSession(
            api_key="test-key", agent_id="stale-id"
        )
        session._client = mock_client

        agent_id = _run(session.ensure_agent())
        assert agent_id == "fallback-id"

    def test_start_session_without_agent_raises(self):
        """start_session() without agent_id raises."""
        from mother.voice_duplex import DuplexVoiceSession

        session = DuplexVoiceSession(api_key="test-key")
        with pytest.raises(RuntimeError, match="ensure_agent"):
            _run(session.start_session())

    def test_end_session_returns_to_idle(self):
        """end_session() returns to IDLE state."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexState

        session = DuplexVoiceSession(api_key="test-key")
        session._state = DuplexState.ACTIVE
        session._conversation = MagicMock()

        async def _test():
            session._loop = asyncio.get_running_loop()
            await session.end_session()

        _run(_test())
        assert session.state == DuplexState.IDLE
        assert session._conversation is None

    def test_end_session_when_idle(self):
        """end_session() when already IDLE is a no-op."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexState

        session = DuplexVoiceSession(api_key="test-key")
        assert session.state == DuplexState.IDLE
        _run(session.end_session())
        assert session.state == DuplexState.IDLE

    def test_emit_event_thread_safe(self):
        """Events are dispatched to queue via call_soon_threadsafe."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexEvent

        loop = asyncio.new_event_loop()
        eq = asyncio.Queue()
        session = DuplexVoiceSession(api_key="test-key", event_queue=eq)

        mock_loop = MagicMock()
        session._loop = mock_loop

        event = DuplexEvent(type="test", text="hello")
        session._emit_event(event)
        mock_loop.call_soon_threadsafe.assert_called_once()
        loop.close()

    def test_emit_event_without_queue(self):
        """_emit_event with no queue is a no-op."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexEvent
        session = DuplexVoiceSession(api_key="test-key")
        session._emit_event(DuplexEvent(type="test"))  # Should not raise

    def test_callbacks_emit_events(self):
        """Transcript callbacks emit correct event types."""
        from mother.voice_duplex import DuplexVoiceSession

        session = DuplexVoiceSession(api_key="test-key")
        session._loop = MagicMock()
        session._event_queue = MagicMock()

        session._on_user_transcript("hello")
        session._on_agent_response("hi there")
        session._on_latency(150)
        session._on_session_ended()

        assert session._loop.call_soon_threadsafe.call_count == 4

    def test_active_property(self):
        """active property reflects ACTIVE state."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexState
        session = DuplexVoiceSession(api_key="test-key")
        assert not session.active
        session._state = DuplexState.ACTIVE
        assert session.active
        session._state = DuplexState.STOPPING
        assert not session.active

    def test_start_session_wrong_state(self):
        """start_session() from non-IDLE/ERROR state is a no-op."""
        from mother.voice_duplex import DuplexVoiceSession, DuplexState

        session = DuplexVoiceSession(api_key="test-key")
        session._agent_id = "test-agent"
        session._state = DuplexState.ACTIVE

        _run(session.start_session())  # Should log warning, not crash
        assert session.state == DuplexState.ACTIVE  # Unchanged


# ═══════════════════════════════════════════════════════════════════════
# §4 — DuplexState & DuplexEvent tests
# ═══════════════════════════════════════════════════════════════════════


class TestDuplexTypes:
    """Tests for DuplexState and DuplexEvent."""

    def test_duplex_state_values(self):
        """All expected states exist."""
        from mother.voice_duplex import DuplexState
        assert DuplexState.IDLE
        assert DuplexState.STARTING
        assert DuplexState.ACTIVE
        assert DuplexState.STOPPING
        assert DuplexState.ERROR
        assert len(DuplexState) == 5

    def test_duplex_event_creation(self):
        """DuplexEvent is a frozen dataclass."""
        from mother.voice_duplex import DuplexEvent
        event = DuplexEvent(type="user_spoke", text="hello")
        assert event.type == "user_spoke"
        assert event.text == "hello"
        assert event.data == {}

    def test_duplex_event_with_data(self):
        """DuplexEvent can carry data dict."""
        from mother.voice_duplex import DuplexEvent
        event = DuplexEvent(type="latency", data={"latency_ms": 150})
        assert event.data["latency_ms"] == 150

    def test_duplex_event_frozen(self):
        """DuplexEvent is immutable."""
        from mother.voice_duplex import DuplexEvent
        event = DuplexEvent(type="test")
        with pytest.raises(AttributeError):
            event.type = "changed"

    def test_duplex_event_defaults(self):
        """DuplexEvent has sensible defaults."""
        from mother.voice_duplex import DuplexEvent
        event = DuplexEvent(type="test")
        assert event.text == ""
        assert event.data == {}


# ═══════════════════════════════════════════════════════════════════════
# §5 — Config tests
# ═══════════════════════════════════════════════════════════════════════


class TestDuplexConfig:
    """Tests for duplex config fields."""

    def test_defaults(self):
        """Duplex config fields have correct defaults."""
        from mother.config import MotherConfig
        config = MotherConfig()
        assert config.voice_duplex_enabled is False
        assert config.voice_duplex_port == 11411
        assert config.voice_duplex_agent_id == ""
        assert config.voice_duplex_language == "en"

    def test_backward_compat(self):
        """Loading config without duplex fields works (backward compat)."""
        from mother.config import MotherConfig
        old_data = {"name": "Mother", "voice_enabled": True}
        known_fields = {f.name for f in MotherConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in old_data.items() if k in known_fields}
        config = MotherConfig(**filtered)
        assert config.voice_duplex_enabled is False

    def test_roundtrip(self):
        """Duplex config survives save/load cycle."""
        import tempfile
        from pathlib import Path
        from mother.config import MotherConfig, save_config, load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            config = MotherConfig(
                voice_duplex_enabled=True,
                voice_duplex_port=12345,
                voice_duplex_agent_id="agent-abc",
                voice_duplex_language="es",
            )
            save_config(config, path)
            loaded = load_config(path)
            assert loaded.voice_duplex_enabled is True
            assert loaded.voice_duplex_port == 12345
            assert loaded.voice_duplex_agent_id == "agent-abc"
            assert loaded.voice_duplex_language == "es"

    def test_duplex_fields_in_dataclass(self):
        """All 4 duplex fields are present in MotherConfig."""
        from mother.config import MotherConfig
        fields = {f.name for f in MotherConfig.__dataclass_fields__.values()}
        assert "voice_duplex_enabled" in fields
        assert "voice_duplex_port" in fields
        assert "voice_duplex_agent_id" in fields
        assert "voice_duplex_language" in fields


# ═══════════════════════════════════════════════════════════════════════
# §6 — Bridge duplex methods tests
# ═══════════════════════════════════════════════════════════════════════


class TestBridgeDuplex:
    """Tests for bridge.py duplex additions."""

    def test_system_prompt_roundtrip(self):
        """get/set_duplex_system_prompt works."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        assert bridge.get_duplex_system_prompt() == ""
        bridge.set_duplex_system_prompt("You are Mother.")
        assert bridge.get_duplex_system_prompt() == "You are Mother."

    def test_system_prompt_update(self):
        """System prompt can be updated."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge()
        bridge.set_duplex_system_prompt("v1")
        bridge.set_duplex_system_prompt("v2")
        assert bridge.get_duplex_system_prompt() == "v2"

    def test_stream_chat_for_duplex_yields_tokens(self):
        """stream_chat_for_duplex yields tokens from LLM."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge()

        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["Hello", " ", "world"])
        mock_llm.last_usage = {}
        bridge._chat_llm = mock_llm
        bridge._chat_model = "test-model"

        async def _test():
            tokens = []
            async for token in bridge.stream_chat_for_duplex(
                [{"role": "user", "content": "Hi"}],
                "You are Mother.",
            ):
                tokens.append(token)
            return tokens

        tokens = _run(_test())
        assert "".join(tokens) == "Hello world"

    def test_stream_chat_for_duplex_empty(self):
        """stream_chat_for_duplex handles empty response."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([])
        mock_llm.last_usage = {}
        bridge._chat_llm = mock_llm
        bridge._chat_model = "test-model"

        async def _test():
            tokens = []
            async for token in bridge.stream_chat_for_duplex([], ""):
                tokens.append(token)
            return tokens

        tokens = _run(_test())
        assert tokens == []

    def test_stream_chat_for_duplex_error_handling(self):
        """stream_chat_for_duplex handles LLM errors gracefully."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.stream.side_effect = RuntimeError("API down")
        mock_llm.last_usage = {}
        bridge._chat_llm = mock_llm
        bridge._chat_model = "test-model"

        async def _test():
            tokens = []
            async for token in bridge.stream_chat_for_duplex(
                [{"role": "user", "content": "Hi"}], ""
            ):
                tokens.append(token)
            return tokens

        tokens = _run(_test())
        assert tokens == []

    def test_stream_chat_for_duplex_uses_chat_llm(self):
        """stream_chat_for_duplex routes through _get_chat_llm."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter(["ok"])
        mock_llm.last_usage = {}
        bridge._chat_llm = mock_llm
        bridge._chat_model = "cheap-model"

        async def _test():
            async for _ in bridge.stream_chat_for_duplex(
                [{"role": "user", "content": "Hi"}], ""
            ):
                pass

        _run(_test())
        mock_llm.stream.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# §7 — Integration / wiring tests
# ═══════════════════════════════════════════════════════════════════════


class TestDuplexWiring:
    """Tests that duplex components wire together correctly."""

    def test_llm_server_is_leaf(self):
        """llm_server.py has no imports from core/."""
        import ast
        from pathlib import Path
        tree = ast.parse(Path("mother/llm_server.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("core."), f"LEAF violation: imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("core."), f"LEAF violation: imports from {node.module}"

    def test_voice_duplex_is_leaf(self):
        """voice_duplex.py has no imports from core/."""
        import ast
        from pathlib import Path
        tree = ast.parse(Path("mother/voice_duplex.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("core."), f"LEAF violation: imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith("core."), f"LEAF violation: imports from {node.module}"

    def test_chat_has_duplex_instance_vars(self):
        """chat.py has duplex instance variable initialization."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert "_duplex_session" in source
        assert "_llm_server" in source
        assert "_duplex_event_queue" in source
        assert "_duplex_consumer_task" in source
        assert "_duplex_active" in source

    def test_chat_has_duplex_methods(self):
        """chat.py has all required duplex methods."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert "def _start_duplex_voice" in source
        assert "def _stop_duplex_voice" in source
        assert "def _duplex_consumer" in source
        assert "def _update_duplex_system_prompt" in source

    def test_chat_unmount_cleans_duplex(self):
        """on_unmount includes duplex cleanup."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        unmount_idx = source.index("async def on_unmount")
        unmount_section = source[unmount_idx:unmount_idx + 1500]
        assert "_stop_duplex_voice" in unmount_section

    def test_hot_reload_handles_duplex(self):
        """_hot_reload_modalities handles duplex toggle."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        reload_idx = source.index("def _hot_reload_modalities")
        reload_section = source[reload_idx:reload_idx + 2000]
        assert "voice_duplex_enabled" in reload_section

    def test_on_mount_starts_duplex_if_enabled(self):
        """on_mount schedules duplex start if config enabled."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        # Duplex startup is in on_mount between mount start and the next method
        mount_start = source.index("def on_mount")
        # Find the next method definition after on_mount
        next_def = source.index("\n    def ", mount_start + 1)
        mount_section = source[mount_start:next_def]
        assert "voice_duplex_enabled" in mount_section
        assert "_start_duplex_voice" in mount_section

    def test_bridge_has_duplex_methods(self):
        """bridge.py has duplex streaming and prompt methods."""
        from pathlib import Path
        source = Path("mother/bridge.py").read_text()
        assert "def stream_chat_for_duplex" in source
        assert "def get_duplex_system_prompt" in source
        assert "def set_duplex_system_prompt" in source


# ═══════════════════════════════════════════════════════════════════════
# §8 — ServerConfig tests
# ═══════════════════════════════════════════════════════════════════════


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_defaults(self):
        from mother.llm_server import ServerConfig
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 11411

    def test_frozen(self):
        from mother.llm_server import ServerConfig
        config = ServerConfig()
        with pytest.raises(AttributeError):
            config.port = 9999

    def test_custom(self):
        from mother.llm_server import ServerConfig
        config = ServerConfig(host="0.0.0.0", port=8080)
        assert config.host == "0.0.0.0"
        assert config.port == 8080


# ═══════════════════════════════════════════════════════════════════════
# §9 — Permission flow tests
# ═══════════════════════════════════════════════════════════════════════


class TestDuplexPermissionFlow:
    """Tests for duplex voice permission-based enable/disable flow."""

    def test_action_dispatch_has_enable_duplex(self):
        """_dispatch_action handles enable_duplex action."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert 'action == "enable_duplex"' in source

    def test_action_dispatch_sets_pending_permission(self):
        """enable_duplex sets _pending_permission = 'duplex_voice'."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        # Find the enable_duplex block
        idx = source.index('action == "enable_duplex"')
        block = source[idx:idx + 300]
        assert '_pending_permission = "duplex_voice"' in block

    def test_permission_confirm_has_duplex_voice(self):
        """Permission confirmation handler has duplex_voice capability."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert 'capability == "duplex_voice"' in source

    def test_permission_confirm_calls_enable_method(self):
        """Permission confirmation calls _enable_duplex_voice()."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        # Find the duplex_voice capability handler
        idx = source.index('capability == "duplex_voice"')
        block = source[idx:idx + 400]
        assert "_enable_duplex_voice()" in block

    def test_enable_duplex_voice_method_exists(self):
        """_enable_duplex_voice method exists in chat.py."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert "def _enable_duplex_voice(self)" in source

    def test_enable_duplex_voice_checks_availability(self):
        """_enable_duplex_voice checks is_duplex_available()."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("def _enable_duplex_voice")
        block = source[idx:idx + 600]
        assert "is_duplex_available" in block

    def test_enable_duplex_voice_checks_api_key(self):
        """_enable_duplex_voice checks for ElevenLabs API key."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("def _enable_duplex_voice")
        block = source[idx:idx + 800]
        assert "elevenlabs" in block.lower() or "ELEVENLABS_API_KEY" in block

    def test_enable_duplex_voice_saves_config(self):
        """_enable_duplex_voice saves config after enabling."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("def _enable_duplex_voice")
        block = source[idx:idx + 1200]
        assert "save_config" in block

    def test_on_mount_uses_feedback_wrapper(self):
        """on_mount uses _start_duplex_voice_with_feedback, not bare _start_duplex_voice."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        mount_start = source.index("def on_mount")
        next_def = source.index("\n    def ", mount_start + 1)
        mount_section = source[mount_start:next_def]
        assert "_start_duplex_voice_with_feedback" in mount_section

    def test_start_duplex_voice_with_feedback_exists(self):
        """_start_duplex_voice_with_feedback method exists."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert "def _start_duplex_voice_with_feedback" in source

    def test_start_duplex_voice_shows_errors(self):
        """_start_duplex_voice shows user-visible messages on failure."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index("async def _start_duplex_voice(self)")
        # Find the next method definition
        next_method = source.index("\n    async def ", idx + 1)
        block = source[idx:next_method]
        # Should use ChatArea for errors, not just logger
        assert "add_ai_message" in block

    def test_persona_intent_routing_has_enable_duplex(self):
        """INTENT_ROUTING includes enable_duplex action examples."""
        from mother.persona import INTENT_ROUTING
        assert "[ACTION:enable_duplex]" in INTENT_ROUTING

    def test_persona_intent_routing_has_talk_example(self):
        """INTENT_ROUTING has 'Talk to me' example for duplex."""
        from mother.persona import INTENT_ROUTING
        assert "Talk to me" in INTENT_ROUTING

    def test_persona_snapshot_has_duplex_param(self):
        """build_introspection_snapshot accepts duplex_voice_active."""
        from mother.persona import build_introspection_snapshot
        import inspect
        sig = inspect.signature(build_introspection_snapshot)
        assert "duplex_voice_active" in sig.parameters

    def test_persona_snapshot_duplex_in_not_available(self):
        """When duplex_voice_active=False, 'real-time voice' in not_available."""
        from mother.persona import build_introspection_snapshot
        snapshot = build_introspection_snapshot(duplex_voice_active=False)
        na = snapshot.get("not_available", [])
        assert any("real-time voice" in item for item in na)

    def test_persona_snapshot_duplex_not_in_not_available_when_active(self):
        """When duplex_voice_active=True, 'real-time voice' NOT in not_available."""
        from mother.persona import build_introspection_snapshot
        snapshot = build_introspection_snapshot(duplex_voice_active=True)
        na = snapshot.get("not_available", [])
        assert not any("real-time voice" in item for item in na)

    def test_chat_threads_duplex_active_to_snapshot(self):
        """chat.py passes duplex_voice_active to build_introspection_snapshot."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        assert "duplex_voice_active=self._duplex_active" in source

    def test_action_dispatch_skips_if_already_active(self):
        """enable_duplex does nothing if _duplex_active is True."""
        from pathlib import Path
        source = Path("mother/screens/chat.py").read_text()
        idx = source.index('action == "enable_duplex"')
        block = source[idx:idx + 300]
        assert "_duplex_active" in block
