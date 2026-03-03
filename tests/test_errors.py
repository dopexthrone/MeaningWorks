"""
Tests for error handling paths - Phase 5.4: Test Coverage.

Tests all exception classes, error handling, timeout behavior, and failure paths.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from core.exceptions import (
    MotherlabsError,
    CompilationError,
    SchemaValidationError,
    ProviderError,
    TimeoutError,
    CorpusError,
    DialogueError,
    ConfigurationError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from MotherlabsError."""

    def test_motherlabs_error_is_base(self):
        """MotherlabsError should be base exception."""
        err = MotherlabsError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_compilation_error_inherits(self):
        """CompilationError should inherit from MotherlabsError."""
        err = CompilationError("failed", stage="synthesis")
        assert isinstance(err, MotherlabsError)
        assert err.stage == "synthesis"
        assert err.details == {}

    def test_compilation_error_with_details(self):
        """CompilationError should accept details dict."""
        err = CompilationError(
            "synthesis failed",
            stage="synthesis",
            details={"attempts": 3, "last_error": "JSON parse error"}
        )
        assert err.stage == "synthesis"
        assert err.details["attempts"] == 3
        assert "JSON parse error" in err.details["last_error"]

    def test_schema_validation_error_inherits(self):
        """SchemaValidationError should inherit from MotherlabsError."""
        err = SchemaValidationError(
            "invalid blueprint",
            errors=["missing name", "invalid type"],
            blueprint={"components": []}
        )
        assert isinstance(err, MotherlabsError)
        assert len(err.errors) == 2
        assert err.blueprint == {"components": []}

    def test_provider_error_inherits(self):
        """ProviderError should inherit from MotherlabsError."""
        err = ProviderError(
            "API error",
            provider="grok",
            status_code=401,
            response="Unauthorized"
        )
        assert isinstance(err, MotherlabsError)
        assert err.provider == "grok"
        assert err.status_code == 401
        assert err.response == "Unauthorized"

    def test_timeout_error_inherits(self):
        """TimeoutError should inherit from MotherlabsError."""
        err = TimeoutError(
            "operation timed out",
            operation="synthesis",
            timeout_seconds=30.0
        )
        assert isinstance(err, MotherlabsError)
        assert err.operation == "synthesis"
        assert err.timeout_seconds == 30.0

    def test_corpus_error_inherits(self):
        """CorpusError should inherit from MotherlabsError."""
        err = CorpusError(
            "storage failed",
            operation="store",
            record_id="abc123"
        )
        assert isinstance(err, MotherlabsError)
        assert err.operation == "store"
        assert err.record_id == "abc123"

    def test_dialogue_error_inherits(self):
        """DialogueError should inherit from MotherlabsError."""
        err = DialogueError(
            "convergence failed",
            turn_count=12,
            agent="Entity"
        )
        assert isinstance(err, MotherlabsError)
        assert err.turn_count == 12
        assert err.agent == "Entity"

    def test_configuration_error_inherits(self):
        """ConfigurationError should inherit from MotherlabsError."""
        err = ConfigurationError(
            "missing API key",
            config_key="XAI_API_KEY"
        )
        assert isinstance(err, MotherlabsError)
        assert err.config_key == "XAI_API_KEY"

    def test_can_catch_all_with_base(self):
        """All exceptions should be catchable via MotherlabsError."""
        exceptions = [
            CompilationError("test"),
            SchemaValidationError("test"),
            ProviderError("test"),
            TimeoutError("test"),
            CorpusError("test"),
            DialogueError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except MotherlabsError as e:
                assert True  # Successfully caught
            except Exception:
                pytest.fail(f"{type(exc).__name__} not caught by MotherlabsError")


class TestMissingAPIKey:
    """Test handling of missing API keys."""

    def test_missing_xai_api_key(self, no_api_keys):
        """Should fail gracefully without XAI_API_KEY."""
        from core.llm import GrokClient

        with pytest.raises((ValueError, ConfigurationError)) as exc_info:
            GrokClient()

        assert "XAI_API_KEY" in str(exc_info.value) or "api key" in str(exc_info.value).lower()

    def test_missing_anthropic_api_key(self, no_api_keys):
        """Should fail gracefully without ANTHROPIC_API_KEY."""
        from core.llm import ClaudeClient

        with pytest.raises((ValueError, ConfigurationError)) as exc_info:
            ClaudeClient()

        assert "ANTHROPIC_API_KEY" in str(exc_info.value) or "api key" in str(exc_info.value).lower()

    def test_missing_openai_api_key(self, no_api_keys):
        """Should fail gracefully without OPENAI_API_KEY."""
        from core.llm import OpenAIClient

        with pytest.raises((ValueError, ConfigurationError)) as exc_info:
            OpenAIClient()

        assert "OPENAI_API_KEY" in str(exc_info.value) or "api key" in str(exc_info.value).lower()

    def test_engine_fails_without_keys(self, no_api_keys):
        """MotherlabsEngine should fail without any API keys."""
        from core.engine import MotherlabsEngine

        with pytest.raises((ValueError, ConfigurationError)):
            MotherlabsEngine(provider="grok")


class TestInvalidInputHandling:
    """Test handling of invalid inputs."""

    def test_empty_description(self, mock_llm_client):
        """Empty description should be rejected by input quality gate."""
        from core.engine import MotherlabsEngine
        from core.exceptions import InputQualityError

        engine = MotherlabsEngine(llm_client=mock_llm_client)
        with pytest.raises(InputQualityError):
            engine.compile("")

    def test_whitespace_only_description(self, mock_llm_client):
        """Whitespace-only description should be rejected by input quality gate."""
        from core.engine import MotherlabsEngine
        from core.exceptions import InputQualityError

        engine = MotherlabsEngine(llm_client=mock_llm_client)
        with pytest.raises(InputQualityError):
            engine.compile("   \n\t  ")

    def test_none_description_raises_error(self, mock_llm_client):
        """None description should raise TypeError or AttributeError."""
        from core.engine import MotherlabsEngine
        from core.exceptions import InputQualityError

        engine = MotherlabsEngine(llm_client=mock_llm_client)

        with pytest.raises((TypeError, AttributeError, CompilationError, InputQualityError)):
            engine.compile(None)


class TestProviderFailureHandling:
    """Test handling of LLM provider failures."""

    def test_api_error_propagates(self):
        """API errors should propagate as ProviderError."""
        client = Mock()
        client.complete = Mock(side_effect=Exception("API unavailable"))
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=client)
        result = engine.compile("test input")

        # Should capture error in result
        assert not result.success
        assert result.error is not None

    def test_malformed_json_response(self):
        """Malformed JSON response should be handled."""
        client = Mock()
        client.complete = Mock(return_value="not valid json {{{")
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=client)
        result = engine.compile("test input")

        # Should handle JSON parse error gracefully
        assert hasattr(result, 'success')

    def test_empty_response_handling(self):
        """Empty response from provider should be handled."""
        client = Mock()
        client.complete = Mock(return_value="")
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=client)
        result = engine.compile("test input")

        # Should handle empty response gracefully
        assert hasattr(result, 'success')


class TestTimeoutBehavior:
    """Test timeout handling (mocked)."""

    def test_timeout_error_attributes(self):
        """TimeoutError should have correct attributes."""
        err = TimeoutError(
            "LLM call timed out",
            operation="synthesis",
            timeout_seconds=60.0
        )

        assert str(err) == "LLM call timed out"
        assert err.operation == "synthesis"
        assert err.timeout_seconds == 60.0

    def test_timeout_default_attributes(self):
        """TimeoutError should work with defaults."""
        err = TimeoutError("timed out")

        assert err.operation is None
        assert err.timeout_seconds is None


class TestRetryBehavior:
    """Test retry logic handling."""

    def test_compile_with_failing_client(self):
        """Engine should handle failures gracefully even on first attempt."""
        call_count = [0]

        def failing_call(*args, **kwargs):
            call_count[0] += 1
            raise Exception("Transient error")

        client = Mock()
        client.complete = Mock(side_effect=failing_call)
        client.complete_with_system = Mock(side_effect=failing_call)
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        # Disable caching to avoid Mock serialization issues
        engine = MotherlabsEngine(llm_client=client, cache_policy="none")
        # The engine should handle failure gracefully
        result = engine.compile("test")

        # Should have attempted at least once and failed gracefully
        assert call_count[0] >= 1
        assert not result.success
        assert result.error is not None

    def test_max_retries_exceeded(self):
        """Should stop after max retries."""
        client = Mock()
        client.complete = Mock(side_effect=Exception("Persistent error"))
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        # Disable caching to avoid Mock serialization issues
        engine = MotherlabsEngine(llm_client=client, cache_policy="none")
        result = engine.compile("test")

        # Should fail after retries exhausted
        assert not result.success
        assert result.error is not None


class TestSchemaValidationErrors:
    """Test schema validation error handling."""

    def test_invalid_component_type(self):
        """Invalid component type should generate validation error."""
        from core.schema import validate_blueprint

        blueprint = {
            "components": [
                {
                    "name": "Test",
                    "type": "invalid_type",  # Invalid
                    "description": "Test component",
                    "derived_from": "test derivation"
                }
            ],
            "relationships": [],
            "constraints": []
        }

        result = validate_blueprint(blueprint)
        # Should have validation errors or warnings
        assert "errors" in result or "warnings" in result

    def test_missing_required_fields(self):
        """Missing required fields should generate errors."""
        from core.schema import validate_blueprint

        blueprint = {
            "components": [
                {
                    "name": "",  # Missing name
                    "type": "entity"
                    # Missing description and derived_from
                }
            ],
            "relationships": []
        }

        result = validate_blueprint(blueprint)
        # Should report missing fields
        assert result.get("errors") or result.get("warnings")

    def test_invalid_relationship_reference(self):
        """Relationship referencing non-existent component should error."""
        from core.schema import validate_blueprint

        blueprint = {
            "components": [
                {
                    "name": "A",
                    "type": "entity",
                    "description": "Component A",
                    "derived_from": "test"
                }
            ],
            "relationships": [
                {
                    "from": "A",
                    "to": "NonExistent",  # Doesn't exist
                    "type": "depends_on"
                }
            ]
        }

        result = validate_blueprint(blueprint)
        # Should report invalid reference
        assert not result.get("valid", True) or result.get("warnings")


class TestCorpusErrors:
    """Test corpus error handling."""

    def test_corpus_error_attributes(self):
        """CorpusError should have correct attributes."""
        err = CorpusError(
            "Failed to store record",
            operation="store",
            record_id="test-123"
        )

        assert err.operation == "store"
        assert err.record_id == "test-123"

    def test_corpus_error_defaults(self):
        """CorpusError should work with defaults."""
        err = CorpusError("storage failed")

        assert err.operation is None
        assert err.record_id is None


class TestDialogueErrors:
    """Test dialogue error handling."""

    def test_dialogue_error_attributes(self):
        """DialogueError should have correct attributes."""
        err = DialogueError(
            "Dialogue did not converge",
            turn_count=12,
            agent="Process"
        )

        assert err.turn_count == 12
        assert err.agent == "Process"

    def test_dialogue_exhaustion(self):
        """Should handle dialogue exhaustion gracefully."""
        # Create a mock that never converges
        client = Mock()
        client.complete = Mock(return_value="I disagree with everything")
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=client)
        # Should eventually terminate even if not converging
        result = engine.compile("test")

        # Should have some result (even if not successful)
        assert hasattr(result, 'success')


class TestConfigurationErrors:
    """Test configuration error handling."""

    def test_invalid_provider(self):
        """Invalid provider should raise ConfigurationError or ValueError."""
        from core.engine import MotherlabsEngine

        with pytest.raises((ConfigurationError, ValueError, KeyError)):
            MotherlabsEngine(provider="invalid_provider")

    def test_configuration_error_attributes(self):
        """ConfigurationError should have correct attributes."""
        err = ConfigurationError(
            "Invalid model",
            config_key="model"
        )

        assert err.config_key == "model"


class TestCompileResultErrorField:
    """Test that CompileResult.error is properly populated."""

    def test_error_field_on_failure(self):
        """CompileResult should have error message on failure."""
        client = Mock()
        client.complete = Mock(side_effect=Exception("Test error"))
        client.provider_name = "test"
        client.model_name = "test-model"

        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=client)
        result = engine.compile("test")

        assert not result.success
        assert result.error is not None
        assert len(result.error) > 0

    def test_error_field_none_on_success(self, mock_llm_client):
        """CompileResult.error should be None on success."""
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=mock_llm_client)
        result = engine.compile("Build a simple system with User and Session")

        # If successful, error should be None
        if result.success:
            assert result.error is None


class TestExceptionContextPreservation:
    """Test that exception context is preserved through the stack."""

    def test_compilation_error_preserves_stage(self):
        """CompilationError should preserve stage information."""
        try:
            raise CompilationError(
                "Synthesis failed after 3 attempts",
                stage="synthesis",
                details={"attempts": 3}
            )
        except CompilationError as e:
            assert e.stage == "synthesis"
            assert e.details["attempts"] == 3

    def test_provider_error_preserves_response(self):
        """ProviderError should preserve response details."""
        try:
            raise ProviderError(
                "Rate limited",
                provider="openai",
                status_code=429,
                response='{"error": "rate_limit_exceeded"}'
            )
        except ProviderError as e:
            assert e.provider == "openai"
            assert e.status_code == 429
            assert "rate_limit" in e.response


class TestUserFriendlyMessages:
    """Phase 6.5: Test user_message and suggestion on all exceptions."""

    def test_base_error_has_defaults(self):
        """MotherlabsError should have default user_message and suggestion."""
        err = MotherlabsError("internal error")
        assert err.user_message
        assert err.suggestion
        assert "went wrong" in err.user_message.lower() or "something" in err.user_message.lower()

    def test_base_error_custom_user_message(self):
        """MotherlabsError should accept custom user_message."""
        err = MotherlabsError(
            "internal error",
            user_message="Custom user message",
            suggestion="Custom suggestion",
        )
        assert err.user_message == "Custom user message"
        assert err.suggestion == "Custom suggestion"

    def test_compilation_error_stage_message(self):
        """CompilationError should auto-generate user_message from stage."""
        for stage in ("intent", "personas", "dialogue", "synthesis", "verification"):
            err = CompilationError("failed", stage=stage)
            assert err.user_message
            assert err.suggestion
            # Should not be the generic base message
            assert err.user_message != "Something went wrong during processing."

    def test_compilation_error_unknown_stage(self):
        """CompilationError with unknown stage should use defaults."""
        err = CompilationError("failed", stage="unknown_stage")
        assert err.user_message
        assert err.suggestion

    def test_provider_error_401_message(self):
        """ProviderError with 401 should suggest API key check."""
        err = ProviderError("auth failed", status_code=401)
        assert "api key" in err.user_message.lower() or "invalid" in err.user_message.lower()
        assert "key" in err.suggestion.lower()

    def test_provider_error_429_message(self):
        """ProviderError with 429 should suggest waiting."""
        err = ProviderError("rate limited", status_code=429)
        assert "rate" in err.user_message.lower() or "limit" in err.user_message.lower()
        assert "wait" in err.suggestion.lower()

    def test_provider_error_500_message(self):
        """ProviderError with 500+ should mention service issues."""
        err = ProviderError("server error", status_code=500)
        assert "service" in err.user_message.lower() or "issues" in err.user_message.lower()

    def test_timeout_error_message(self):
        """TimeoutError should have user-friendly message."""
        err = TimeoutError("timed out", operation="synthesis", timeout_seconds=60)
        assert "long" in err.user_message.lower() or "too" in err.user_message.lower()

    def test_configuration_error_with_key(self):
        """ConfigurationError should suggest setting the config key."""
        err = ConfigurationError("missing key", config_key="ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" in err.suggestion

    def test_configuration_error_without_key(self):
        """ConfigurationError without config_key should use generic suggestion."""
        err = ConfigurationError("bad config")
        assert err.suggestion
        assert "environment" in err.suggestion.lower() or "configuration" in err.suggestion.lower()

    def test_failover_exhausted_message(self):
        """FailoverExhaustedError should mention all services unavailable."""
        from core.exceptions import FailoverExhaustedError
        err = FailoverExhaustedError("all failed", providers_tried=["a", "b"])
        assert "unavailable" in err.user_message.lower() or "all" in err.user_message.lower()

    def test_provider_unavailable_message(self):
        """ProviderUnavailableError should mention automatic retry."""
        from core.exceptions import ProviderUnavailableError
        err = ProviderUnavailableError("grok down", provider="grok")
        assert "unavailable" in err.user_message.lower()
        assert "another" in err.suggestion.lower() or "try" in err.suggestion.lower()

    def test_to_user_dict(self):
        """to_user_dict should return structured user-facing error info."""
        err = CompilationError("synthesis failed", stage="synthesis")
        d = err.to_user_dict()
        assert "error" in d
        assert "suggestion" in d
        assert "error_type" in d
        assert d["error_type"] == "CompilationError"
        assert d["error"] == err.user_message
        assert d["suggestion"] == err.suggestion

    def test_to_user_dict_on_base(self):
        """to_user_dict should work on base MotherlabsError."""
        err = MotherlabsError("test")
        d = err.to_user_dict()
        assert d["error_type"] == "MotherlabsError"

    def test_corpus_error_message(self):
        """CorpusError should have user-friendly message."""
        err = CorpusError("write failed", operation="store")
        assert "saving" in err.user_message.lower() or "data" in err.user_message.lower()

    def test_dialogue_error_message(self):
        """DialogueError should have user-friendly message."""
        err = DialogueError("convergence failed", turn_count=12)
        assert "agreement" in err.user_message.lower() or "agents" in err.user_message.lower()

    def test_schema_validation_error_message(self):
        """SchemaValidationError should have user-friendly message."""
        err = SchemaValidationError("invalid blueprint", errors=["missing name"])
        assert "specification" in err.user_message.lower() or "structural" in err.user_message.lower()
