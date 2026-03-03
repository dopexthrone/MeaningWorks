"""
Tests for Phase 9.4: Actionable Error Messages with Error Catalog.

Tests that error codes are properly assigned, catalog entries are complete,
and to_user_dict() returns structured diagnostics.
"""

import pytest

from core.error_catalog import (
    CATALOG,
    ErrorEntry,
    get_entry,
    get_code_for_exception,
)
from core.exceptions import (
    MotherlabsError,
    CompilationError,
    SchemaValidationError,
    ProviderError,
    TimeoutError,
    CorpusError,
    DialogueError,
    ConfigurationError,
    ProviderUnavailableError,
    FailoverExhaustedError,
    InputQualityError,
)


# =============================================================================
# ERROR CATALOG STRUCTURE
# =============================================================================


class TestErrorCatalogStructure:
    """Test that the catalog is well-formed."""

    def test_all_entries_have_required_fields(self):
        """Every catalog entry should have code, title, root_cause."""
        for code, entry in CATALOG.items():
            assert entry.code == code, f"{code}: code mismatch"
            assert entry.title, f"{code}: missing title"
            assert entry.root_cause, f"{code}: missing root_cause"

    def test_all_entries_have_fix_examples(self):
        """Every catalog entry should have at least one fix example."""
        for code, entry in CATALOG.items():
            assert len(entry.fix_examples) >= 1, f"{code}: no fix_examples"

    def test_code_categories_exist(self):
        """Catalog should have entries in each category."""
        categories = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"}
        found = {code[:2] for code in CATALOG}
        assert categories == found

    def test_get_entry_existing(self):
        """get_entry should return entry for known code."""
        entry = get_entry("E1001")
        assert entry is not None
        assert entry.code == "E1001"
        assert entry.title == "Input too vague"

    def test_get_entry_missing(self):
        """get_entry should return None for unknown code."""
        assert get_entry("E9999") is None

    def test_entry_is_frozen(self):
        """ErrorEntry should be immutable."""
        entry = get_entry("E1001")
        with pytest.raises(AttributeError):
            entry.code = "MODIFIED"


# =============================================================================
# ERROR CODE INFERENCE
# =============================================================================


class TestErrorCodeInference:
    """Test get_code_for_exception() resolves correct codes."""

    def test_input_quality_default(self):
        """InputQualityError defaults to E1001."""
        code = get_code_for_exception("InputQualityError")
        assert code == "E1001"

    def test_input_quality_short_input(self):
        """InputQualityError with low length_score -> E1002."""
        class FakeScore:
            length_score = 0.05
        code = get_code_for_exception("InputQualityError", quality_score=FakeScore())
        assert code == "E1002"

    def test_input_quality_custom_threshold(self):
        """InputQualityError with custom_threshold -> E1003."""
        code = get_code_for_exception("InputQualityError", custom_threshold=True)
        assert code == "E1003"

    def test_provider_401(self):
        """ProviderError with 401 -> E2001."""
        assert get_code_for_exception("ProviderError", status_code=401) == "E2001"

    def test_provider_429(self):
        """ProviderError with 429 -> E2002."""
        assert get_code_for_exception("ProviderError", status_code=429) == "E2002"

    def test_provider_500(self):
        """ProviderError with 500 -> E2003."""
        assert get_code_for_exception("ProviderError", status_code=500) == "E2003"

    def test_failover_exhausted(self):
        """FailoverExhaustedError -> E2004."""
        assert get_code_for_exception("FailoverExhaustedError") == "E2004"

    def test_provider_unavailable(self):
        """ProviderUnavailableError -> E2005."""
        assert get_code_for_exception("ProviderUnavailableError") == "E2005"

    def test_compilation_intent(self):
        """CompilationError stage=intent -> E3001."""
        assert get_code_for_exception("CompilationError", stage="intent") == "E3001"

    def test_compilation_synthesis(self):
        """CompilationError stage=synthesis -> E3004."""
        assert get_code_for_exception("CompilationError", stage="synthesis") == "E3004"

    def test_compilation_verification(self):
        """CompilationError stage=verification -> E3005."""
        assert get_code_for_exception("CompilationError", stage="verification") == "E3005"

    def test_schema_validation(self):
        """SchemaValidationError -> E3006."""
        assert get_code_for_exception("SchemaValidationError") == "E3006"

    def test_timeout_stage(self):
        """TimeoutError with stage operation -> E4001."""
        assert get_code_for_exception("TimeoutError", operation="synthesis") == "E4001"

    def test_timeout_llm(self):
        """TimeoutError with generic operation -> E4002."""
        assert get_code_for_exception("TimeoutError", operation="llm call") == "E4002"

    def test_configuration_default(self):
        """ConfigurationError default -> E6001."""
        assert get_code_for_exception("ConfigurationError") == "E6001"

    def test_configuration_provider(self):
        """ConfigurationError with provider key -> E6002."""
        assert get_code_for_exception("ConfigurationError", config_key="provider") == "E6002"

    def test_unknown_exception(self):
        """Unknown exception class -> None."""
        assert get_code_for_exception("SomeRandomError") is None


# =============================================================================
# EXCEPTION AUTO-ENRICHMENT
# =============================================================================


class TestExceptionAutoEnrichment:
    """Test that exceptions auto-populate error_code and catalog fields."""

    def test_compilation_error_has_code(self):
        """CompilationError should auto-assign error code from stage."""
        err = CompilationError("failed", stage="synthesis")
        assert err.error_code == "E3004"

    def test_compilation_error_intent_code(self):
        """CompilationError intent stage -> E3001."""
        err = CompilationError("failed", stage="intent")
        assert err.error_code == "E3001"

    def test_provider_error_401_code(self):
        """ProviderError 401 -> E2001."""
        err = ProviderError("auth failed", status_code=401)
        assert err.error_code == "E2001"

    def test_provider_error_429_code(self):
        """ProviderError 429 -> E2002."""
        err = ProviderError("rate limited", status_code=429)
        assert err.error_code == "E2002"

    def test_timeout_error_code(self):
        """TimeoutError with synthesis -> E4001."""
        err = TimeoutError("timed out", operation="synthesis")
        assert err.error_code == "E4001"

    def test_schema_validation_error_code(self):
        """SchemaValidationError -> E3006."""
        err = SchemaValidationError("invalid")
        assert err.error_code == "E3006"

    def test_dialogue_error_code(self):
        """DialogueError -> E3003."""
        err = DialogueError("failed")
        assert err.error_code == "E3003"

    def test_configuration_error_code(self):
        """ConfigurationError -> E6001."""
        err = ConfigurationError("missing key", config_key="ANTHROPIC_API_KEY")
        assert err.error_code == "E6001"

    def test_provider_unavailable_code(self):
        """ProviderUnavailableError -> E2005."""
        err = ProviderUnavailableError("down", provider="grok")
        assert err.error_code == "E2005"

    def test_failover_exhausted_code(self):
        """FailoverExhaustedError -> E2004."""
        err = FailoverExhaustedError("all failed")
        assert err.error_code == "E2004"

    def test_input_quality_error_code(self):
        """InputQualityError -> E1001."""
        err = InputQualityError("too vague")
        assert err.error_code == "E1001"

    def test_explicit_code_overrides_auto(self):
        """Explicit error_code should override auto-inference."""
        err = CompilationError("failed", stage="synthesis", error_code="E9999")
        assert err.error_code == "E9999"

    def test_base_error_no_code_by_default(self):
        """MotherlabsError without error_code should have None."""
        err = MotherlabsError("test")
        assert err.error_code is None

    def test_base_error_with_explicit_code(self):
        """MotherlabsError with explicit error_code should keep it."""
        err = MotherlabsError("test", error_code="E1001")
        assert err.error_code == "E1001"


# =============================================================================
# to_user_dict() WITH ERROR CODES
# =============================================================================


class TestToUserDictWithCodes:
    """Test that to_user_dict includes catalog information."""

    def test_to_user_dict_includes_error_code(self):
        """to_user_dict should include error_code when set."""
        err = CompilationError("synthesis failed", stage="synthesis")
        d = err.to_user_dict()
        assert d["error_code"] == "E3004"

    def test_to_user_dict_includes_root_cause(self):
        """to_user_dict should include root_cause from catalog."""
        err = CompilationError("synthesis failed", stage="synthesis")
        d = err.to_user_dict()
        assert "root_cause" in d
        assert "missing" in d["root_cause"].lower() or "blueprint" in d["root_cause"].lower()

    def test_to_user_dict_includes_fix_examples(self):
        """to_user_dict should include fix_examples from catalog."""
        err = CompilationError("synthesis failed", stage="synthesis")
        d = err.to_user_dict()
        assert "fix_examples" in d
        assert isinstance(d["fix_examples"], list)
        assert len(d["fix_examples"]) >= 1

    def test_to_user_dict_without_code(self):
        """to_user_dict without error_code should not include extra fields."""
        err = MotherlabsError("generic")
        d = err.to_user_dict()
        assert "error_code" not in d
        assert "root_cause" not in d
        assert "fix_examples" not in d

    def test_to_user_dict_preserves_existing_fields(self):
        """to_user_dict should still include error, suggestion, error_type."""
        err = ProviderError("auth failed", status_code=401)
        d = err.to_user_dict()
        assert "error" in d
        assert "suggestion" in d
        assert "error_type" in d
        assert d["error_type"] == "ProviderError"

    def test_to_user_dict_input_quality(self):
        """InputQualityError to_user_dict should include E1001 details."""
        err = InputQualityError("too vague")
        d = err.to_user_dict()
        assert d["error_code"] == "E1001"
        assert "root_cause" in d
        assert "fix_examples" in d

    def test_to_user_dict_timeout(self):
        """TimeoutError to_user_dict should include E4xxx details."""
        err = TimeoutError("timed out", operation="synthesis", timeout_seconds=60)
        d = err.to_user_dict()
        assert d["error_code"] == "E4001"
        assert "root_cause" in d


# =============================================================================
# ENGINE INTEGRATION
# =============================================================================


class TestEngineErrorCodes:
    """Test that engine attaches correct error codes."""

    def test_quality_gate_uses_e1001(self, mock_llm_client):
        """Quality gate rejection should use E1001 or E1002."""
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=mock_llm_client)
        with pytest.raises(InputQualityError) as exc_info:
            engine.compile("Build a thing")
        assert exc_info.value.error_code in ("E1001", "E1002")

    def test_custom_threshold_uses_e1003(self, mock_llm_client):
        """Custom threshold rejection should use E1003."""
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(llm_client=mock_llm_client)
        with pytest.raises(InputQualityError) as exc_info:
            engine.compile("A booking system", min_quality_score=0.99)
        assert exc_info.value.error_code == "E1003"
