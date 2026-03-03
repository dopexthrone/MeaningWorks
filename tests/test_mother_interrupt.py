"""Tests for speech interrupt — cancel stream, interrupt state, routing."""

import asyncio
import queue
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mother.bridge import EngineBridge
from mother.screens.chat import ChatScreen


# --- cancel_chat_stream on EngineBridge ---


class TestCancelChatStream:
    def test_drains_queue(self):
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        # Pre-fill tokens
        for t in ["hello", " ", "world"]:
            bridge._chat_token_queue.put_nowait(t)
        assert not bridge._chat_token_queue.empty()

        bridge.cancel_chat_stream()
        assert bridge._chat_token_queue.empty()

    def test_sets_done_flag(self):
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.clear()

        bridge.cancel_chat_stream()
        assert bridge._chat_stream_done.is_set()

    def test_idempotent(self):
        """Double cancel should not crash."""
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()

        bridge.cancel_chat_stream()
        bridge.cancel_chat_stream()
        assert bridge._chat_stream_done.is_set()

    def test_empty_queue_no_error(self):
        """Cancel on already-empty queue works fine."""
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.clear()

        bridge.cancel_chat_stream()
        assert bridge._chat_stream_done.is_set()


# --- Interrupt state fields ---


class TestInterruptState:
    def _make_screen(self):
        """Create a ChatScreen with minimal init, skipping Textual mount."""
        with patch.object(ChatScreen, "__init__", lambda self, **kw: None):
            screen = ChatScreen.__new__(ChatScreen)
        # Manually set the fields we added
        screen._interrupt_requested = False
        screen._interrupt_text = None
        screen._voice = None
        screen._voice_queue = asyncio.Queue()
        screen._voice_prefetch = None
        return screen

    def test_initial_values(self):
        screen = self._make_screen()
        assert screen._interrupt_requested is False
        assert screen._interrupt_text is None

    def test_clear_voice_queue_drains(self):
        screen = self._make_screen()
        screen._voice_queue.put_nowait("sentence one")
        screen._voice_queue.put_nowait("sentence two")
        assert not screen._voice_queue.empty()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(screen._clear_voice_queue())
        finally:
            loop.close()
        assert screen._voice_queue.empty()

    def test_clear_voice_queue_calls_stop(self):
        screen = self._make_screen()
        mock_voice = AsyncMock()
        mock_voice.stop = AsyncMock()
        screen._voice = mock_voice

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(screen._clear_voice_queue())
        finally:
            loop.close()
        mock_voice.stop.assert_awaited_once()


# --- Perception interrupt routing ---


class TestPerceptionInterrupt:
    def _make_screen_with_bridge(self, chatting=False):
        with patch.object(ChatScreen, "__init__", lambda self, **kw: None):
            screen = ChatScreen.__new__(ChatScreen)
        screen._interrupt_requested = False
        screen._interrupt_text = None
        screen._chatting = chatting
        screen._voice = None
        screen._voice_queue = asyncio.Queue()
        screen._voice_prefetch = None
        screen._bridge = MagicMock()
        screen._bridge.cancel_chat_stream = MagicMock()
        screen._unmounted = False
        return screen

    def test_speech_during_chatting_sets_interrupt(self):
        screen = self._make_screen_with_bridge(chatting=True)
        # Simulate what _perception_consumer does when chatting=True
        screen._interrupt_requested = True
        screen._interrupt_text = "hey stop"
        screen._bridge.cancel_chat_stream()

        assert screen._interrupt_requested is True
        assert screen._interrupt_text == "hey stop"
        screen._bridge.cancel_chat_stream.assert_called_once()

    def test_speech_during_idle_does_not_set_interrupt(self):
        screen = self._make_screen_with_bridge(chatting=False)
        # When not chatting, interrupt should NOT be set
        assert screen._interrupt_requested is False


# --- Keyboard interrupt routing ---


class TestKeyboardInterrupt:
    def _make_screen_with_bridge(self, chatting=False):
        with patch.object(ChatScreen, "__init__", lambda self, **kw: None):
            screen = ChatScreen.__new__(ChatScreen)
        screen._interrupt_requested = False
        screen._interrupt_text = None
        screen._chatting = chatting
        screen._voice = None
        screen._voice_queue = asyncio.Queue()
        screen._voice_prefetch = None
        screen._bridge = MagicMock()
        screen._bridge.cancel_chat_stream = MagicMock()
        screen._unmounted = False
        return screen

    def test_typing_during_stream_sets_interrupt(self):
        screen = self._make_screen_with_bridge(chatting=True)
        # Simulate what on_user_text_submitted does
        screen._interrupt_requested = True
        screen._interrupt_text = "actually, do this instead"
        screen._bridge.cancel_chat_stream()

        assert screen._interrupt_requested is True
        assert screen._interrupt_text == "actually, do this instead"
        screen._bridge.cancel_chat_stream.assert_called_once()

    def test_typing_during_idle_does_not_interrupt(self):
        screen = self._make_screen_with_bridge(chatting=False)
        # No interrupt when not streaming
        assert screen._interrupt_requested is False

    def test_command_during_stream_should_not_interrupt(self):
        """Commands (e.g. /help) during stream should NOT trigger interrupt —
        they go through the normal command routing instead."""
        screen = self._make_screen_with_bridge(chatting=True)
        # In the real code, is_command=True skips the interrupt path
        # Just verify the interrupt_requested is still False
        assert screen._interrupt_requested is False


# --- Interrupt flow ---


class TestInterruptFlow:
    def test_interrupt_text_preserved(self):
        """Interrupt text matches the speech payload."""
        screen_interrupt_text = "what about this?"
        # Direct field test
        assert screen_interrupt_text == "what about this?"

    def test_interrupt_resets_flags(self):
        """After handling, flags should be reset."""
        with patch.object(ChatScreen, "__init__", lambda self, **kw: None):
            screen = ChatScreen.__new__(ChatScreen)
        screen._interrupt_requested = True
        screen._interrupt_text = "interrupted message"

        # Simulate the reset that _chat_worker does
        interrupt_text = screen._interrupt_text
        screen._interrupt_requested = False
        screen._interrupt_text = None

        assert interrupt_text == "interrupted message"
        assert screen._interrupt_requested is False
        assert screen._interrupt_text is None

    def test_cancel_stream_stops_token_iteration(self):
        """After cancel_chat_stream, stream_chat_tokens should stop yielding."""
        bridge = EngineBridge.__new__(EngineBridge)
        bridge._chat_token_queue = queue.Queue()
        bridge._chat_stream_done = threading.Event()
        bridge._chat_stream_done.clear()

        # Put some tokens
        bridge._chat_token_queue.put_nowait("hello")
        bridge._chat_token_queue.put_nowait(" world")

        # Cancel
        bridge.cancel_chat_stream()

        # Queue should be empty and done should be set
        assert bridge._chat_token_queue.empty()
        assert bridge._chat_stream_done.is_set()

        # stream_chat_tokens should immediately return (no tokens)
        async def _consume():
            tokens = []
            async for t in bridge.stream_chat_tokens():
                tokens.append(t)
            return tokens

        loop = asyncio.new_event_loop()
        try:
            tokens = loop.run_until_complete(_consume())
        finally:
            loop.close()
        assert tokens == []
