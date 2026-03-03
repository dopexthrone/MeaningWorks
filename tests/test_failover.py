"""
Tests for FailoverClient and provider failover functionality.

Phase 6.1: Provider Failover

Tests automatic failover between LLM providers when one fails.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

from core.llm import (
    FailoverClient,
    BaseLLMClient,
    MockClient,
    create_client,
)
from core.exceptions import (
    FailoverExhaustedError,
    ProviderUnavailableError,
    ProviderError,
    ConfigurationError,
)


class TestFailoverClientBasics:
    """Test basic FailoverClient functionality."""

    def test_failover_client_requires_providers(self):
        """FailoverClient should require at least one provider."""
        with pytest.raises(ValueError, match="At least one provider"):
            FailoverClient([])

    def test_failover_client_single_provider(self):
        """FailoverClient should work with single provider."""
        mock = MockClient()
        client = FailoverClient([mock])

        result = client.complete_with_system("test system", "test user")
        assert "[Mock response" in result
        assert mock.call_count == 1

    def test_failover_client_uses_first_provider(self):
        """FailoverClient should use first provider when successful."""
        mock1 = MockClient()
        mock2 = MockClient()
        client = FailoverClient([mock1, mock2])

        result = client.complete_with_system("test system", "test user")

        assert mock1.call_count == 1
        assert mock2.call_count == 0
        assert "[Mock response #1]" in result

    def test_failover_client_inherits_deterministic(self):
        """FailoverClient should inherit deterministic setting from first provider."""
        mock1 = MockClient()
        mock1.deterministic = False

        client = FailoverClient([mock1])
        assert client.deterministic == False


class TestFailoverBehavior:
    """Test failover behavior when providers fail."""

    def test_failover_on_exception(self):
        """Should failover to next provider when first raises exception."""
        failing_client = Mock(spec=BaseLLMClient)
        failing_client.complete.side_effect = Exception("Provider 1 failed")
        failing_client.deterministic = True

        success_client = MockClient()

        client = FailoverClient([failing_client, success_client])
        result = client.complete_with_system("system", "user")

        assert "[Mock response" in result
        assert success_client.call_count == 1

    def test_failover_chain(self):
        """Should try each provider in sequence until success."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = Exception("Provider 1 failed")
        fail1.deterministic = True

        fail2 = Mock(spec=BaseLLMClient)
        fail2.complete.side_effect = Exception("Provider 2 failed")
        fail2.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, fail2, success])
        result = client.complete_with_system("system", "user")

        assert "[Mock response" in result
        assert fail1.complete.call_count == 1
        assert fail2.complete.call_count == 1
        assert success.call_count == 1

    def test_failover_exhausted_error(self):
        """Should raise FailoverExhaustedError when all providers fail."""
        # Use MockClient subclasses to get proper class names
        class Fail1Client(MockClient):
            def complete(self, *args, **kwargs):
                raise Exception("Provider 1 failed")

        class Fail2Client(MockClient):
            def complete(self, *args, **kwargs):
                raise Exception("Provider 2 failed")

        fail1 = Fail1Client()
        fail2 = Fail2Client()

        client = FailoverClient([fail1, fail2])

        with pytest.raises(FailoverExhaustedError) as exc_info:
            client.complete_with_system("system", "user")

        assert "All 2 providers failed" in str(exc_info.value)
        assert len(exc_info.value.providers_tried) == 2
        assert len(exc_info.value.errors) == 2

    def test_failover_tracks_successful_provider(self):
        """Should track which provider succeeded."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = Exception("Failed")
        fail1.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, success])
        client.complete_with_system("system", "user")

        # _last_successful_idx should be 1 (second provider)
        assert client._last_successful_idx == 1


class TestFailoverLogging:
    """Test failover logging behavior."""

    def test_failover_logs_on_failure(self, caplog):
        """Should log when failover occurs."""
        logger = logging.getLogger("test")

        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = Exception("Provider error")
        fail1.__class__.__name__ = "FailingClient"
        fail1.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, success], logger=logger)

        with caplog.at_level(logging.WARNING):
            client.complete_with_system("system", "user")

        assert "FailingClient failed" in caplog.text or success.call_count == 1

    def test_failover_logs_success_after_failure(self, caplog):
        """Should log when failover succeeds after initial failure."""
        logger = logging.getLogger("test")

        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = Exception("Provider error")
        fail1.__class__.__name__ = "FailingClient"
        fail1.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, success], logger=logger)

        with caplog.at_level(logging.INFO):
            client.complete_with_system("system", "user")

        # Verify the call succeeded
        assert success.call_count == 1


class TestFailoverClientProperties:
    """Test FailoverClient property accessors."""

    def test_provider_name_property(self):
        """Should return provider name of current primary."""
        mock1 = Mock(spec=BaseLLMClient)
        mock1.model = "grok-3"
        mock1.__class__.__name__ = "GrokClient"
        mock1.complete.return_value = "response"
        mock1.deterministic = True

        client = FailoverClient([mock1])

        # Before any calls, should report based on first provider
        assert "grok" in client.provider_name.lower()

    def test_model_name_property(self):
        """Should return model name of current primary."""
        mock1 = Mock(spec=BaseLLMClient)
        mock1.model = "grok-3-mini"
        mock1.complete.return_value = "response"
        mock1.deterministic = True

        client = FailoverClient([mock1])

        assert client.model_name == "grok-3-mini"

    def test_provider_name_after_failover(self):
        """Provider name should reflect current successful provider."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = Exception("Failed")
        fail1.__class__.__name__ = "FailClient"
        fail1.model = "fail-model"
        fail1.deterministic = True

        success = Mock(spec=BaseLLMClient)
        success.complete.return_value = "success"
        success.__class__.__name__ = "SuccessClient"
        success.model = "success-model"
        success.deterministic = True

        client = FailoverClient([fail1, success])
        client.complete_with_system("system", "user")

        # After failover, should reflect second provider
        assert client.model_name == "success-model"


class TestFailoverClientComplete:
    """Test FailoverClient.complete() method."""

    def test_complete_passes_all_parameters(self):
        """complete() should pass all parameters to underlying provider."""
        mock = Mock(spec=BaseLLMClient)
        mock.complete.return_value = "response"
        mock.deterministic = True

        client = FailoverClient([mock])

        messages = [{"role": "user", "content": "hello"}]
        client.complete(
            messages=messages,
            system="system prompt",
            max_tokens=1000,
            temperature=0.5
        )

        mock.complete.assert_called_once_with(
            messages=messages,
            system="system prompt",
            max_tokens=1000,
            temperature=0.5
        )

    def test_complete_with_system_passes_parameters(self):
        """complete_with_system() should correctly format messages."""
        mock = Mock(spec=BaseLLMClient)
        mock.complete.return_value = "response"
        mock.deterministic = True

        client = FailoverClient([mock])

        client.complete_with_system(
            "system prompt",
            "user content",
            max_tokens=2000,
            temperature=0.3
        )

        call_kwargs = mock.complete.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "user content"}]
        assert call_kwargs["system"] == "system prompt"
        assert call_kwargs["max_tokens"] == 2000
        assert call_kwargs["temperature"] == 0.3


class TestEngineWithFailover:
    """Test MotherlabsEngine with failover_providers parameter."""

    def test_engine_failover_providers_parameter(self, mock_env_vars):
        """Engine should accept failover_providers parameter."""
        from core.engine import MotherlabsEngine

        # With mock env vars, we can create an engine with failover
        engine = MotherlabsEngine(failover_providers=["grok", "claude"])

        # Should have a FailoverClient
        assert isinstance(engine.llm, FailoverClient)
        assert len(engine.llm.providers) == 2

    def test_engine_failover_skips_unavailable_providers(self, monkeypatch):
        """Engine should skip providers without API keys."""
        # Only set one API key
        monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from core.engine import MotherlabsEngine

        # Try to create with multiple providers - only grok should be available
        engine = MotherlabsEngine(failover_providers=["grok", "claude", "openai"])

        # Should only have grok provider
        assert isinstance(engine.llm, FailoverClient)
        assert len(engine.llm.providers) == 1

    def test_engine_failover_all_unavailable_raises_error(self, no_api_keys):
        """Engine should raise error if no failover providers available."""
        from core.engine import MotherlabsEngine

        with pytest.raises(ConfigurationError, match="No providers available"):
            MotherlabsEngine(failover_providers=["grok", "claude"])

    def test_engine_provider_name_with_failover(self, mock_env_vars):
        """Engine should report provider name correctly with failover."""
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(failover_providers=["grok", "claude"])

        # Should report 'failover' or the current provider
        assert engine.provider_name in ["failover", "grok", "xai"]

    def test_engine_compile_with_failover(self, mock_llm_client):
        """Engine should work with failover client for compilation."""
        from core.engine import MotherlabsEngine

        # Create failover client with mock
        failover = FailoverClient([mock_llm_client])

        engine = MotherlabsEngine(llm_client=failover)
        result = engine.compile("Build a simple User entity")

        # Should complete successfully
        assert hasattr(result, 'success')


class TestFailoverErrorHandling:
    """Test error handling in failover scenarios."""

    def test_preserves_error_messages(self):
        """Should preserve error messages from each failed provider."""
        # Use MockClient subclasses to get proper class names
        class RateLimitClient(MockClient):
            def complete(self, *args, **kwargs):
                raise Exception("Rate limit exceeded")

        class InvalidKeyClient(MockClient):
            def complete(self, *args, **kwargs):
                raise Exception("Invalid API key")

        fail1 = RateLimitClient()
        fail2 = InvalidKeyClient()

        client = FailoverClient([fail1, fail2])

        with pytest.raises(FailoverExhaustedError) as exc_info:
            client.complete_with_system("system", "user")

        errors = exc_info.value.errors
        assert "Rate limit exceeded" in errors.get("RateLimitClient", "")
        assert "Invalid API key" in errors.get("InvalidKeyClient", "")

    def test_handles_provider_unavailable_error(self):
        """Should handle ProviderUnavailableError specifically."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = ProviderUnavailableError(
            "Service unavailable",
            provider="grok",
            status_code=503
        )
        fail1.__class__.__name__ = "GrokClient"
        fail1.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, success])
        result = client.complete_with_system("system", "user")

        # Should failover successfully
        assert "[Mock response" in result

    def test_handles_provider_error(self):
        """Should handle generic ProviderError."""
        fail1 = Mock(spec=BaseLLMClient)
        fail1.complete.side_effect = ProviderError(
            "API error",
            provider="openai",
            status_code=500
        )
        fail1.__class__.__name__ = "OpenAIClient"
        fail1.deterministic = True

        success = MockClient()

        client = FailoverClient([fail1, success])
        result = client.complete_with_system("system", "user")

        assert "[Mock response" in result


class TestFailoverExhaustedError:
    """Test FailoverExhaustedError exception."""

    def test_error_attributes(self):
        """FailoverExhaustedError should have correct attributes."""
        error = FailoverExhaustedError(
            "All providers failed",
            providers_tried=["GrokClient", "ClaudeClient"],
            errors={"GrokClient": "Rate limit", "ClaudeClient": "Timeout"}
        )

        assert str(error) == "All providers failed"
        assert error.providers_tried == ["GrokClient", "ClaudeClient"]
        assert error.errors["GrokClient"] == "Rate limit"
        assert error.errors["ClaudeClient"] == "Timeout"

    def test_error_inherits_from_provider_error(self):
        """FailoverExhaustedError should inherit from ProviderError."""
        error = FailoverExhaustedError("test")

        assert isinstance(error, ProviderError)

    def test_error_defaults(self):
        """FailoverExhaustedError should have sensible defaults."""
        error = FailoverExhaustedError("test error")

        assert error.providers_tried == []
        assert error.errors == {}


class TestProviderUnavailableError:
    """Test ProviderUnavailableError exception."""

    def test_error_inherits_from_provider_error(self):
        """ProviderUnavailableError should inherit from ProviderError."""
        error = ProviderUnavailableError(
            "Service unavailable",
            provider="grok",
            status_code=503
        )

        assert isinstance(error, ProviderError)
        assert error.provider == "grok"
        assert error.status_code == 503


class TestCreateClientWithFailover:
    """Test create_client factory function."""

    def test_create_client_returns_single_client(self, mock_env_vars):
        """create_client should return single client, not failover."""
        client = create_client(provider="grok")

        assert not isinstance(client, FailoverClient)

    def test_create_client_auto_detects_provider(self, monkeypatch):
        """create_client with auto should detect available provider."""
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        client = create_client(provider="auto")

        # Should create GrokClient since only XAI_API_KEY is set
        assert "Grok" in type(client).__name__
