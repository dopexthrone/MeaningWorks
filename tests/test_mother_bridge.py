"""
Phase 3: Tests for engine bridge.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from mother.bridge import EngineBridge, COST_RATES


class TestEngineBridgeCreation:
    """Test bridge instantiation."""

    def test_creates_bridge(self):
        bridge = EngineBridge(provider="claude")
        assert bridge._provider == "claude"

    def test_initial_cost_zero(self):
        bridge = EngineBridge()
        assert bridge.get_session_cost() == 0.0

    def test_provider_info(self):
        bridge = EngineBridge(provider="openai", model="gpt-4o")
        info = bridge.get_provider_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o"


class TestCostTracking:
    """Test session cost accumulation."""

    def test_track_cost(self):
        bridge = EngineBridge()
        bridge._track_cost({"input_tokens": 1000, "output_tokens": 500})
        cost = bridge.get_session_cost()
        assert cost > 0

    def test_cost_accumulates(self):
        bridge = EngineBridge()
        bridge._track_cost({"input_tokens": 1000, "output_tokens": 500})
        first = bridge.get_session_cost()
        bridge._track_cost({"input_tokens": 1000, "output_tokens": 500})
        assert bridge.get_session_cost() > first

    def test_zero_tokens_zero_cost(self):
        bridge = EngineBridge()
        bridge._track_cost({})
        assert bridge.get_session_cost() == 0.0


class TestInsightQueue:
    """Test insight draining."""

    def test_drain_empty_queue(self):
        bridge = EngineBridge()
        insights = asyncio.run(bridge.drain_insights())
        assert insights == []

    def test_drain_returns_insights(self):
        bridge = EngineBridge()
        bridge._insight_queue.put_nowait("insight 1")
        bridge._insight_queue.put_nowait("insight 2")
        insights = asyncio.run(bridge.drain_insights())
        assert len(insights) == 2
        assert "insight 1" in insights


class TestChat:
    """Test chat method with mocked LLM."""

    def test_chat_calls_llm(self):
        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Hello there."
        mock_llm.last_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        bridge._llm = mock_llm

        response = asyncio.run(bridge.chat(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="You are Mother.",
        ))
        assert response == "Hello there."
        mock_llm.complete.assert_called_once()

    def test_chat_tracks_cost(self):
        bridge = EngineBridge()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Response"
        mock_llm.last_usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        bridge._llm = mock_llm

        asyncio.run(bridge.chat([{"role": "user", "content": "hi"}], "system"))
        assert bridge.get_session_cost() > 0


class TestCostRates:
    """Test COST_RATES table and provider-aware costing."""

    def test_all_providers_have_rates(self):
        for provider in ("claude", "openai", "grok", "gemini"):
            assert provider in COST_RATES
            assert "input" in COST_RATES[provider]
            assert "output" in COST_RATES[provider]

    def test_gemini_cheapest(self):
        assert COST_RATES["gemini"]["input"] < COST_RATES["claude"]["input"]

    def test_provider_rate_used(self):
        bridge = EngineBridge(provider="gemini")
        bridge._track_cost({"input_tokens": 1_000_000, "output_tokens": 0})
        cost = bridge.get_session_cost()
        assert abs(cost - 0.10) < 0.001

    def test_claude_rate_used(self):
        bridge = EngineBridge(provider="claude")
        bridge._track_cost({"input_tokens": 1_000_000, "output_tokens": 0})
        cost = bridge.get_session_cost()
        assert abs(cost - 3.0) < 0.001


class TestLastCallCost:
    """Test per-call cost tracking."""

    def test_last_call_cost_initially_zero(self):
        bridge = EngineBridge()
        assert bridge.get_last_call_cost() == 0.0

    def test_last_call_cost_after_track(self):
        bridge = EngineBridge(provider="claude")
        bridge._track_cost({"input_tokens": 1000, "output_tokens": 500})
        assert bridge.get_last_call_cost() > 0

    def test_last_call_cost_resets_per_call(self):
        bridge = EngineBridge(provider="claude")
        bridge._track_cost({"input_tokens": 10000, "output_tokens": 5000})
        first = bridge.get_last_call_cost()
        bridge._track_cost({"input_tokens": 100, "output_tokens": 50})
        second = bridge.get_last_call_cost()
        assert second < first


class TestCostBreakdown:
    """Test cost breakdown method."""

    def test_breakdown_structure(self):
        bridge = EngineBridge(provider="openai", model="gpt-4o")
        breakdown = bridge.get_cost_breakdown()
        assert "session_total" in breakdown
        assert "last_call" in breakdown
        assert "provider" in breakdown
        assert "model" in breakdown
        assert "rates" in breakdown
        assert breakdown["provider"] == "openai"


class TestGetStatus:
    """Test aggregated status."""

    def test_status_structure(self):
        bridge = EngineBridge(provider="claude", model="sonnet")
        status = bridge.get_status()
        assert status["provider"] == "claude"
        assert status["model"] == "sonnet"
        assert "rates" in status
        assert "$" in status["rates"]["input"]


class TestInterpretTrust:
    """Test trust interpretation."""

    def test_high_trust(self):
        bridge = EngineBridge()
        msg = bridge.interpret_trust({"overall_score": 90.0, "clarity": 95.0, "coverage": 85.0})
        assert "High confidence" in msg

    def test_moderate_trust(self):
        bridge = EngineBridge()
        msg = bridge.interpret_trust({"overall_score": 55.0, "clarity": 60.0, "coverage": 30.0})
        assert "Moderate" in msg
        assert "coverage" in msg.lower()

    def test_low_trust(self):
        bridge = EngineBridge()
        msg = bridge.interpret_trust({"overall_score": 20.0, "clarity": 15.0})
        assert "Low confidence" in msg

    def test_nested_verification_dict(self):
        """Real engine verification uses nested dicts with 'score' key."""
        bridge = EngineBridge()
        nested = {
            "overall_score": 72.0,
            "completeness": {"score": 65, "gaps": ["missing auth"]},
            "consistency": {"score": 100, "conflicts": []},
            "coverage": {"score": 50, "uncovered": ["edge cases"]},
        }
        msg = bridge.interpret_trust(nested)
        assert "Moderate" in msg
        assert "coverage" in msg.lower()

    def test_mixed_flat_and_nested(self):
        """Handle mix of flat scores and nested dicts."""
        bridge = EngineBridge()
        mixed = {
            "overall_score": 85.0,
            "clarity": 90.0,
            "completeness": {"score": 60, "gaps": []},
        }
        msg = bridge.interpret_trust(mixed)
        assert "High confidence" in msg
        assert "completeness" in msg.lower()
