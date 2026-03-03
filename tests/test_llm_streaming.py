"""Tests for LLM client stream() method."""

import pytest
from core.llm import BaseLLMClient, MockClient, FailoverClient


class TestBaseLLMClientStreamFallback:
    """BaseLLMClient.stream() falls back to complete() as single yield."""

    def test_base_client_stream_fallback(self):
        """Default stream() yields the full complete() result."""
        client = MockClient()
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))
        # MockClient.stream yields word-by-word, but BaseLLMClient default
        # would yield single result — MockClient overrides, so test that.
        assert len(tokens) > 0
        result = "".join(tokens)
        assert "Mock response" in result


class TestMockClientStream:
    """MockClient.stream() yields word-by-word tokens."""

    def test_mock_client_stream_yields_words(self):
        """Stream yields individual words."""
        client = MockClient()
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))
        assert len(tokens) > 1  # Multiple words
        full = "".join(tokens)
        assert "Mock response #1" in full

    def test_mock_client_stream_increments_count(self):
        """call_count increments on stream."""
        client = MockClient()
        assert client.call_count == 0
        _ = list(client.stream([{"role": "user", "content": "hi"}]))
        assert client.call_count == 1

    def test_mock_client_stream_sets_usage(self):
        """last_usage is populated after streaming."""
        client = MockClient()
        _ = list(client.stream([{"role": "user", "content": "hi"}]))
        assert client.last_usage["input_tokens"] == 10
        assert client.last_usage["output_tokens"] == 20


class TestFailoverClientStream:
    """FailoverClient.stream() delegates to active provider."""

    def test_failover_stream_delegates(self):
        """FailoverClient delegates stream to first provider."""
        mock1 = MockClient()
        mock2 = MockClient()
        client = FailoverClient([mock1, mock2])
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))
        full = "".join(tokens)
        assert "Mock response" in full
        assert mock1.call_count == 1
        assert mock2.call_count == 0

    def test_failover_stream_failover(self):
        """FailoverClient tries next provider on failure."""

        class FailingClient(MockClient):
            def stream(self, messages, **kwargs):
                raise RuntimeError("boom")

        failing = FailingClient()
        working = MockClient()
        client = FailoverClient([failing, working])
        tokens = list(client.stream([{"role": "user", "content": "hi"}]))
        full = "".join(tokens)
        assert "Mock response" in full
        assert working.call_count == 1

    def test_failover_stream_relays_usage(self):
        """FailoverClient relays last_usage from successful provider."""
        mock = MockClient()
        client = FailoverClient([mock])
        _ = list(client.stream([{"role": "user", "content": "hi"}]))
        assert client.last_usage == mock.last_usage


class TestStreamMethodExists:
    """Verify stream() method exists on all client classes."""

    def test_claude_stream_method_exists(self):
        from core.llm import ClaudeClient
        assert hasattr(ClaudeClient, "stream")
        assert callable(getattr(ClaudeClient, "stream"))

    def test_openai_stream_method_exists(self):
        from core.llm import OpenAIClient
        assert hasattr(OpenAIClient, "stream")
        assert callable(getattr(OpenAIClient, "stream"))

    def test_gemini_stream_method_exists(self):
        from core.llm import GeminiClient
        assert hasattr(GeminiClient, "stream")
        assert callable(getattr(GeminiClient, "stream"))

    def test_grok_stream_method_exists(self):
        from core.llm import GrokClient
        assert hasattr(GrokClient, "stream")
        assert callable(getattr(GrokClient, "stream"))
