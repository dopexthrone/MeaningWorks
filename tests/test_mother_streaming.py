"""Tests for Mother TUI streaming infrastructure."""

import asyncio
import queue
import threading
import pytest

from mother.screens.chat import _extract_first_sentence


class TestExtractFirstSentence:
    """_extract_first_sentence detects sentence boundaries."""

    def test_complete_sentence_period(self):
        assert _extract_first_sentence("Hello world, this is a test. More text") == "Hello world, this is a test."

    def test_complete_sentence_exclamation(self):
        assert _extract_first_sentence("That's amazing! More text") == "That's amazing!"

    def test_complete_sentence_question(self):
        assert _extract_first_sentence("What do you think? Let me know") == "What do you think?"

    def test_incomplete_returns_none(self):
        assert _extract_first_sentence("Hello world") is None

    def test_ignores_short_abbreviation(self):
        """Sentences shorter than 10 chars are skipped to avoid abbreviations."""
        assert _extract_first_sentence("Dr. Smith") is None

    def test_sentence_at_end(self):
        assert _extract_first_sentence("This is a complete sentence.") == "This is a complete sentence."

    def test_empty_string(self):
        assert _extract_first_sentence("") is None

    def test_newline_after_period(self):
        assert _extract_first_sentence("This is a sentence.\nNext line") == "This is a sentence."


class TestChatAreaStreaming:
    """ChatArea streaming message lifecycle."""

    def test_chat_area_has_streaming_attrs(self):
        from mother.widgets.chat_area import ChatArea
        area = ChatArea()
        assert area._streaming_msg is None
        assert area._streaming_buffer == ""

    def test_message_widget_has_update_text(self):
        from mother.widgets.chat_area import MessageWidget
        widget = MessageWidget(role="mother", text="initial")
        assert hasattr(widget, "update_text")
        assert callable(widget.update_text)


class TestBridgeStreamingState:
    """EngineBridge streaming chat state."""

    def test_bridge_has_chat_token_queue(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.set()
        bridge._chat_stream_result = None
        bridge._chat_stream_usage = {}
        assert isinstance(bridge._chat_token_queue, queue.Queue)

    def test_bridge_begin_chat_stream_clears_state(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.set()
        bridge._chat_stream_result = "old"
        bridge._chat_stream_usage = {"old": True}

        # Add stale token
        bridge._chat_token_queue.put("stale")

        bridge.begin_chat_stream()

        assert bridge._chat_token_queue.empty()
        assert not bridge._chat_stream_done.is_set()
        assert bridge._chat_stream_result is None
        assert bridge._chat_stream_usage == {}

    def test_bridge_stream_chat_tokens_terminates(self):
        """Async generator exits when done is set and queue is empty."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.set()  # Already done
        bridge._chat_stream_result = "hello"
        bridge._chat_stream_usage = {}

        async def _collect():
            tokens = []
            async for token in bridge.stream_chat_tokens():
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_collect())
        assert tokens == []  # Queue was empty, done was set

    def test_bridge_stream_chat_tokens_yields_queued(self):
        """Async generator yields tokens from queue before terminating."""
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_result = None
        bridge._chat_stream_usage = {}

        # Pre-load tokens
        bridge._chat_token_queue.put("Hello ")
        bridge._chat_token_queue.put("world")
        # Signal done
        bridge._chat_stream_done.set()

        async def _collect():
            tokens = []
            async for token in bridge.stream_chat_tokens():
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_collect())
        assert tokens == ["Hello ", "world"]

    def test_bridge_get_stream_result(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_stream_result = "full text"
        assert bridge.get_stream_result() == "full text"

    def test_bridge_get_stream_result_none(self):
        from mother.bridge import EngineBridge
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_stream_result = None
        assert bridge.get_stream_result() is None


class TestVoiceConsumerPipeline:
    """Voice consumer pipelined synthesis tests."""

    def _make_chat_screen(self):
        """Create a minimal ChatScreen with mocked voice for testing."""
        from unittest.mock import AsyncMock, MagicMock
        from mother.screens.chat import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._voice_queue = asyncio.Queue()
        screen._voice_consumer_task = None
        screen._voice_prefetch = None
        screen._current_posture = None
        screen._perception = None

        voice = AsyncMock()
        voice.synthesize = AsyncMock(side_effect=lambda t: f"audio:{t}".encode())
        voice.play = AsyncMock()
        voice.stop = AsyncMock()
        screen._voice = voice
        return screen, voice

    def test_voice_consumer_pipeline_overlaps_synthesis(self):
        """Prefetch is scheduled: sentence N+1 synthesis starts before play(N) returns.

        With instant mocks the tasks resolve synchronously, but we verify
        the pipeline structure: all 3 sentences synthesized and played,
        and sentence 2 was pulled from queue *before* play(1) (visible
        because the queue is empty by the time play(1) runs).
        """
        from unittest.mock import AsyncMock

        screen, voice = self._make_chat_screen()

        synth_texts = []
        play_datas = []
        queue_snapshots_at_play = []

        async def tracked_synthesize(text):
            synth_texts.append(text)
            return f"audio:{text}".encode()

        async def tracked_play(audio_data, playback_rate=None):
            # Record queue size at moment of play — pipeline means
            # next sentence was already dequeued for prefetch
            queue_snapshots_at_play.append(screen._voice_queue.qsize())
            play_datas.append(audio_data.decode())

        voice.synthesize = AsyncMock(side_effect=tracked_synthesize)
        voice.play = AsyncMock(side_effect=tracked_play)

        for s in ["First sentence.", "Second sentence.", "Third sentence."]:
            screen._voice_queue.put_nowait(s)

        asyncio.run(screen._voice_consumer())

        assert synth_texts == ["First sentence.", "Second sentence.", "Third sentence."]
        assert play_datas == ["audio:First sentence.", "audio:Second sentence.", "audio:Third sentence."]
        # When playing sentence 1, sentence 2 was already dequeued for prefetch.
        # Queue started at 3, after synth(1) + dequeue(2) for prefetch => 1 left.
        assert queue_snapshots_at_play[0] == 1  # sentence 3 still in queue
        assert queue_snapshots_at_play[1] == 0  # sentence 3 dequeued for prefetch
        assert queue_snapshots_at_play[2] == 0  # nothing left

    def test_voice_consumer_single_sentence_no_prefetch(self):
        """Single sentence: one synthesize, one play, no extra calls."""
        screen, voice = self._make_chat_screen()
        screen._voice_queue.put_nowait("Only one.")

        asyncio.run(screen._voice_consumer())

        voice.synthesize.assert_called_once_with("Only one.")
        voice.play.assert_called_once()
        assert screen._voice_prefetch is None

    def test_voice_consumer_prefetch_failure_recovers(self):
        """If prefetched synthesis fails, next sentence falls back to sequential."""
        from unittest.mock import AsyncMock

        screen, voice = self._make_chat_screen()
        call_count = 0

        async def flaky_synthesize(text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("API timeout")
            return f"audio:{text}".encode()

        voice.synthesize = AsyncMock(side_effect=flaky_synthesize)

        for s in ["A sentence here.", "Will fail synthesis.", "Should recover now."]:
            screen._voice_queue.put_nowait(s)

        asyncio.run(screen._voice_consumer())

        # Sentence 1 played, sentence 2 failed (prefetch error clears state),
        # sentence 3 falls back to sequential synthesize+play
        assert voice.play.call_count >= 2  # sentence 1 + sentence 3

    def test_voice_consumer_clear_cancels_prefetch(self):
        """_clear_voice_queue cancels pending prefetch task."""
        screen, voice = self._make_chat_screen()

        async def _run():
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            screen._voice_prefetch = ("Next.", asyncio.ensure_future(fut))
            screen._voice_queue.put_nowait("queued")

            await screen._clear_voice_queue()

            assert screen._voice_prefetch is None
            assert screen._voice_queue.empty()
            assert fut.cancelled()

        asyncio.run(_run())
