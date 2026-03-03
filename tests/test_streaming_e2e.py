"""
End-to-end streaming tests with real LLM API.

Marked @pytest.mark.slow — never run by default.
Invoke explicitly:

    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_streaming_e2e.py -m slow -s -v --timeout=60

Tests Sonnet 4.6 streaming via Claude, MockClient streaming,
and bridge streaming infrastructure with real tokens.
"""

import os
import asyncio
import queue
import threading
import time
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.timeout(60)]


class TestMockClientStreamingE2E:
    """MockClient stream() works end-to-end through bridge infra."""

    def test_mock_stream_through_bridge(self):
        """Full path: MockClient.stream → queue → stream_chat_tokens → collected."""
        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", api_key="fake")
        # Monkey-patch _get_llm to return MockClient
        from core.llm import MockClient
        mock = MockClient()
        bridge._llm = mock

        bridge.begin_chat_stream()

        # Run in thread like real usage
        def _run():
            bridge._stream_chat_sync(
                [{"role": "user", "content": "hello"}],
                system_prompt="You are helpful.",
            )

        thread = threading.Thread(target=_run)
        thread.start()

        # Collect tokens async
        async def _collect():
            tokens = []
            async for token in bridge.stream_chat_tokens():
                tokens.append(token)
            return tokens

        tokens = asyncio.run(_collect())
        thread.join()

        full = "".join(tokens)
        assert "Mock response" in full
        assert len(tokens) > 1  # Word-by-word
        assert bridge.get_stream_result() is not None
        assert "Mock response" in bridge.get_stream_result()


class TestClaudeStreamingE2E:
    """Real Claude API streaming — requires ANTHROPIC_API_KEY."""

    def test_claude_stream_yields_tokens(self):
        """ClaudeClient.stream() yields multiple token chunks."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        from core.llm import ClaudeClient
        client = ClaudeClient(api_key=api_key, model="claude-sonnet-4-5-20250929", deterministic=False)

        tokens = []
        for token in client.stream(
            messages=[{"role": "user", "content": "Say hello in exactly 3 words."}],
            system="Respond concisely.",
            max_tokens=50,
            temperature=0.0,
        ):
            tokens.append(token)

        assert len(tokens) > 1, "Should yield multiple token chunks"
        full = "".join(tokens)
        assert len(full) > 0
        # Usage should be populated
        assert client.last_usage.get("input_tokens", 0) > 0
        assert client.last_usage.get("output_tokens", 0) > 0

    def test_claude_stream_through_bridge(self):
        """Full bridge path with real Claude streaming."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        from mother.bridge import EngineBridge

        bridge = EngineBridge(provider="claude", api_key=api_key, model="claude-sonnet-4-5-20250929")
        bridge.begin_chat_stream()

        def _run():
            bridge._stream_chat_sync(
                [{"role": "user", "content": "Count from 1 to 5."}],
                system_prompt="Be concise.",
            )

        thread = threading.Thread(target=_run)
        thread.start()

        async def _collect():
            tokens = []
            start = time.monotonic()
            first_token_time = None
            async for token in bridge.stream_chat_tokens():
                if first_token_time is None:
                    first_token_time = time.monotonic() - start
                tokens.append(token)
            return tokens, first_token_time

        tokens, first_token_time = asyncio.run(_collect())
        thread.join()

        full = "".join(tokens)
        result = bridge.get_stream_result()

        assert len(tokens) > 1, f"Should yield multiple chunks, got {len(tokens)}"
        assert full == result, "Collected tokens should match stream result"
        assert bridge._chat_stream_done.is_set()
        assert bridge.get_session_cost() > 0

        print(f"\n  Tokens: {len(tokens)}")
        print(f"  First token: {first_token_time:.3f}s")
        print(f"  Full text: {full[:80]}...")
        print(f"  Cost: ${bridge.get_session_cost():.6f}")
