"""
Duplex voice session — real-time conversational AI via ElevenLabs WebSocket.

Manages the bidirectional audio pipeline: user speaks → ElevenLabs ASR →
custom LLM (via LocalLLMServer) → ElevenLabs TTS → user hears Mother.
Supports barge-in (user interrupts mid-response), VAD, and streaming audio.

LEAF module. No imports from core/. All external integration via callbacks
and injected configuration.

Audio format: 16kHz, mono, int16 PCM (matches ElevenLabs SDK requirements).
Uses sounddevice (already installed) instead of pyaudio.
"""

import asyncio
import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("mother.voice_duplex")

# Optional imports — graceful degradation
try:
    import sounddevice as sd
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    _SOUNDDEVICE_AVAILABLE = False

try:
    from elevenlabs import ElevenLabs
    from elevenlabs.conversational_ai.conversation import (
        AudioInterface,
        Conversation,
        ConversationInitiationData,
    )
    from elevenlabs.types import (
        AgentConfig,
        ConversationalConfig,
        CustomLlm,
        PromptAgentApiModelOutput,
        TtsConversationalConfigOutput,
    )
    _ELEVENLABS_AVAILABLE = True
except ImportError:
    _ELEVENLABS_AVAILABLE = False


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
INPUT_FRAMES_PER_BUFFER = 4000  # 250ms @ 16kHz
OUTPUT_FRAMES_PER_BUFFER = 1000  # 62.5ms @ 16kHz


def is_duplex_available() -> bool:
    """True if both sounddevice and elevenlabs conversational AI are available."""
    return _SOUNDDEVICE_AVAILABLE and _ELEVENLABS_AVAILABLE


class DuplexState(Enum):
    """State machine for duplex voice session."""
    IDLE = auto()
    STARTING = auto()
    ACTIVE = auto()
    STOPPING = auto()
    ERROR = auto()


@dataclass(frozen=True)
class DuplexEvent:
    """Event emitted from the duplex voice session to the TUI consumer."""
    type: str  # "user_spoke", "agent_spoke", "session_ended", "error", "latency"
    text: str = ""
    data: Dict = field(default_factory=dict)


class SoundDeviceAudioInterface(AudioInterface):
    """AudioInterface implementation using sounddevice instead of pyaudio.

    Input: sounddevice.InputStream with callback dispatching PCM chunks.
    Output: queue-based with dedicated thread writing to OutputStream.
    Interrupt: clears output queue for barge-in support.
    """

    def __init__(self):
        if not _SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is required for SoundDeviceAudioInterface")
        self._input_stream: Optional[object] = None
        self._output_stream: Optional[object] = None
        self._output_queue: queue.Queue = queue.Queue()
        self._output_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        self._input_callback: Optional[Callable[[bytes], None]] = None

    def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Open mic input stream and speaker output stream.

        input_callback receives 250ms PCM chunks (16kHz, mono, int16).
        """
        self._input_callback = input_callback
        self._should_stop.clear()
        self._output_queue = queue.Queue()

        # Input stream — callback dispatches PCM bytes to ElevenLabs
        self._input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=INPUT_FRAMES_PER_BUFFER,
            callback=self._input_stream_callback,
        )
        self._input_stream.start()

        # Output stream — fed by queue from _output_thread
        self._output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=OUTPUT_FRAMES_PER_BUFFER,
        )
        self._output_stream.start()

        # Output thread writes queued audio to output stream
        self._output_thread = threading.Thread(
            target=self._output_writer, daemon=True
        )
        self._output_thread.start()

    def stop(self) -> None:
        """Close all streams and stop output thread."""
        self._should_stop.set()

        if self._output_thread and self._output_thread.is_alive():
            self._output_thread.join(timeout=2.0)
        self._output_thread = None

        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception as e:
                logger.debug(f"Input stream close error: {e}")
            self._input_stream = None

        if self._output_stream:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception as e:
                logger.debug(f"Output stream close error: {e}")
            self._output_stream = None

        self._input_callback = None

    def output(self, audio: bytes) -> None:
        """Queue audio for playback. Non-blocking."""
        self._output_queue.put(audio)

    def interrupt(self) -> None:
        """Clear output queue — barge-in support."""
        try:
            while True:
                self._output_queue.get_nowait()
        except queue.Empty:
            pass

    def _input_stream_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice input callback — dispatches PCM bytes."""
        if status:
            logger.debug(f"Input stream status: {status}")
        if self._input_callback and not self._should_stop.is_set():
            self._input_callback(bytes(indata))

    def _output_writer(self) -> None:
        """Background thread: drain output queue to speaker."""
        while not self._should_stop.is_set():
            try:
                audio = self._output_queue.get(timeout=0.25)
                if self._output_stream and not self._should_stop.is_set():
                    self._output_stream.write(
                        __import__("numpy").frombuffer(audio, dtype="int16")
                    )
            except queue.Empty:
                pass
            except Exception as e:
                logger.debug(f"Output writer error: {e}")


class DuplexVoiceSession:
    """Manages one ElevenLabs Conversational AI session lifecycle.

    Handles agent creation/caching, Conversation lifecycle, audio interface,
    and event dispatch to the TUI via asyncio.Queue.

    Usage:
        session = DuplexVoiceSession(api_key, server_url, event_queue)
        await session.ensure_agent()
        await session.start_session()
        # ... events flow via event_queue
        await session.end_session()
    """

    def __init__(
        self,
        api_key: str,
        server_url: str = "http://127.0.0.1:11411/v1/chat/completions",
        event_queue: Optional[asyncio.Queue] = None,
        agent_id: str = "",
        voice_id: str = "2obv5y63xKRNiEZAPxGD",
        language: str = "en",
    ):
        self._api_key = api_key
        self._server_url = server_url
        self._event_queue = event_queue
        self._agent_id = agent_id
        self._voice_id = voice_id
        self._language = language
        self._state = DuplexState.IDLE
        self._client: Optional[object] = None
        self._conversation: Optional[object] = None
        self._audio_interface: Optional[SoundDeviceAudioInterface] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def state(self) -> DuplexState:
        """Current session state."""
        return self._state

    @property
    def agent_id(self) -> str:
        """Agent ID (empty if not yet created)."""
        return self._agent_id

    @property
    def active(self) -> bool:
        """True if session is actively conversing."""
        return self._state == DuplexState.ACTIVE

    def _get_client(self):
        """Lazy-load ElevenLabs client."""
        if self._client is None and _ELEVENLABS_AVAILABLE:
            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    async def ensure_agent(self) -> str:
        """Create or verify the ElevenLabs agent. Returns agent_id.

        If agent_id is already set, verifies it exists. Otherwise creates
        a new agent configured for custom LLM routing.
        """
        if not _ELEVENLABS_AVAILABLE:
            raise RuntimeError("elevenlabs SDK required for duplex voice")

        client = self._get_client()
        if client is None:
            raise RuntimeError("Failed to create ElevenLabs client")

        # If we have a cached agent_id, try to use it
        if self._agent_id:
            try:
                await asyncio.to_thread(
                    client.conversational_ai.agents.get, agent_id=self._agent_id
                )
                logger.info(f"Reusing existing agent: {self._agent_id}")
                return self._agent_id
            except Exception as e:
                logger.info(f"Cached agent not found ({e}), creating new one")
                self._agent_id = ""

        # Create new agent with custom LLM
        config = ConversationalConfig(
            agent=AgentConfig(
                first_message="",
                language=self._language,
                prompt=PromptAgentApiModelOutput(
                    llm="custom-llm",
                    custom_llm=CustomLlm(
                        url=self._server_url,
                        api_type="chat_completions",
                    ),
                    prompt="You are Mother.",
                    ignore_default_personality=True,
                    temperature=0.7,
                    max_tokens=200,
                ),
            ),
            tts=TtsConversationalConfigOutput(
                voice_id=self._voice_id,
            ),
        )

        result = await asyncio.to_thread(
            client.conversational_ai.agents.create,
            conversation_config=config,
            name="Mother-Duplex",
        )
        self._agent_id = result.agent_id
        logger.info(f"Created duplex agent: {self._agent_id}")
        return self._agent_id

    async def start_session(self) -> None:
        """Start the duplex voice conversation.

        Creates audio interface, connects to ElevenLabs WebSocket,
        and begins streaming audio bidirectionally.
        """
        if self._state not in (DuplexState.IDLE, DuplexState.ERROR):
            logger.warning(f"Cannot start session in state {self._state}")
            return

        if not self._agent_id:
            raise RuntimeError("Call ensure_agent() before start_session()")

        self._state = DuplexState.STARTING
        self._loop = asyncio.get_running_loop()

        try:
            client = self._get_client()
            self._audio_interface = SoundDeviceAudioInterface()

            self._conversation = Conversation(
                client=client,
                agent_id=self._agent_id,
                requires_auth=False,
                audio_interface=self._audio_interface,
                callback_user_transcript=self._on_user_transcript,
                callback_agent_response=self._on_agent_response,
                callback_latency_measurement=self._on_latency,
                callback_end_session=self._on_session_ended,
            )

            # Conversation.start_session() runs in a background thread
            self._conversation.start_session()
            self._state = DuplexState.ACTIVE
            logger.info("Duplex voice session started")

        except Exception as e:
            self._state = DuplexState.ERROR
            logger.error(f"Failed to start duplex session: {e}")
            self._emit_event(DuplexEvent(type="error", text=str(e)))
            raise

    async def end_session(self) -> None:
        """End the duplex voice conversation and clean up."""
        if self._state not in (DuplexState.ACTIVE, DuplexState.STARTING, DuplexState.ERROR):
            return

        self._state = DuplexState.STOPPING
        try:
            if self._conversation:
                self._conversation.end_session()
                self._conversation = None
            self._audio_interface = None
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        finally:
            self._state = DuplexState.IDLE
            self._loop = None
            logger.info("Duplex voice session ended")

    def _emit_event(self, event: DuplexEvent) -> None:
        """Push event to the asyncio queue (thread-safe)."""
        if self._event_queue is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._event_queue.put_nowait, event
            )
        except Exception as e:
            logger.debug(f"Event emit failed: {e}")

    def _on_user_transcript(self, text: str) -> None:
        """Callback from ElevenLabs: user speech transcribed."""
        logger.debug(f"User said: {text}")
        self._emit_event(DuplexEvent(type="user_spoke", text=text))

    def _on_agent_response(self, text: str) -> None:
        """Callback from ElevenLabs: agent response finalized."""
        logger.debug(f"Agent said: {text}")
        self._emit_event(DuplexEvent(type="agent_spoke", text=text))

    def _on_latency(self, latency_ms: int) -> None:
        """Callback from ElevenLabs: latency measurement."""
        logger.debug(f"Duplex latency: {latency_ms}ms")
        self._emit_event(DuplexEvent(
            type="latency", data={"latency_ms": latency_ms}
        ))

    def _on_session_ended(self) -> None:
        """Callback from ElevenLabs: session ended (server-side or error)."""
        logger.info("Duplex session ended by server")
        self._state = DuplexState.IDLE
        self._emit_event(DuplexEvent(type="session_ended"))
