"""
Motherlabs Exception Hierarchy.

Phase 5.1: Error Handling & Stability
Phase 6.5: User-Friendly Error Messages
Phase 9.4: Actionable Error Messages with Error Codes

All Motherlabs exceptions inherit from MotherlabsError for consistent
handling throughout the codebase.

Each exception includes:
- message: Technical error message (for logs and developers)
- user_message: Plain-language explanation (for end users)
- suggestion: Actionable next step (for end users)
- error_code: Catalog code (e.g. "E1001") for programmatic handling
"""

from typing import Optional


class MotherlabsError(Exception):
    """
    Base exception for all Motherlabs errors.

    All custom exceptions in the Motherlabs codebase should inherit from this
    class to enable consistent error handling and filtering.

    Phase 6.5: Includes user_message and suggestion for non-technical users.
    Phase 9.4: Includes error_code for programmatic error handling.

    Attributes:
        user_message: Plain-language explanation safe to show to end users
        suggestion: Actionable next step the user can take
        error_code: Catalog error code (e.g. "E1001") or None
    """
    def __init__(self, message: str, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or "Something went wrong during processing."
        self.suggestion = suggestion or "Please try again. If the problem persists, contact support."
        self.error_code = error_code
        self._enrich_from_catalog()

    def _enrich_from_catalog(self):
        """Auto-populate fields from error catalog if error_code is set."""
        if not self.error_code:
            return
        from core.error_catalog import get_entry
        entry = get_entry(self.error_code)
        if not entry:
            return
        # Only fill in fields that weren't explicitly provided
        if self.user_message == "Something went wrong during processing.":
            self.user_message = entry.title
        if self.suggestion == "Please try again. If the problem persists, contact support.":
            self.suggestion = "; ".join(entry.fix_examples) if entry.fix_examples else self.suggestion

    def to_user_dict(self) -> dict:
        """Export user-facing error information as a dictionary."""
        d = {
            "error": self.user_message,
            "suggestion": self.suggestion,
            "error_type": self.__class__.__name__,
        }
        if self.error_code:
            from core.error_catalog import get_entry
            d["error_code"] = self.error_code
            entry = get_entry(self.error_code)
            if entry:
                d["root_cause"] = entry.root_cause
                d["fix_examples"] = list(entry.fix_examples)
        return d


class FractureError(MotherlabsError):
    """Pipeline halted: intent fracture detected."""
    def __init__(self, message: str, stage: str = "", signal=None, **kwargs):
        kwargs.setdefault("user_message", "The intent has competing valid interpretations.")
        kwargs.setdefault("suggestion", "Re-run with the constraint appended to your description.")
        super().__init__(message, **kwargs)
        self.stage = stage
        self.signal = signal


class CompilationError(MotherlabsError):
    """
    Compilation pipeline failed.

    Raised when the semantic compilation process fails at any stage:
    - Intent extraction failure
    - Persona generation failure
    - Dialogue failure
    - Synthesis failure
    - Verification failure

    Attributes:
        stage: The pipeline stage where the error occurred
        details: Additional error context
    """

    _STAGE_MESSAGES = {
        "intent": (
            "We couldn't understand the description you provided.",
            "Try rephrasing your idea with more detail about what the system should do.",
        ),
        "personas": (
            "We couldn't generate the right perspectives for your project.",
            "Try providing more context about the domain or industry.",
        ),
        "dialogue": (
            "The analysis process didn't reach a conclusion.",
            "Try simplifying your description or breaking it into smaller parts.",
        ),
        "synthesis": (
            "We couldn't generate a complete specification from the analysis.",
            "Try again - the AI may produce a better result on retry.",
        ),
        "verification": (
            "The generated specification didn't pass quality checks.",
            "Try providing more specific requirements or constraints.",
        ),
    }

    def __init__(self, message: str, stage: Optional[str] = None,
                 details: Optional[dict] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        # Auto-generate user message from stage if not provided
        if not user_message and stage and stage in self._STAGE_MESSAGES:
            user_message, suggestion = self._STAGE_MESSAGES[stage]
        # Auto-infer error code from stage
        if not error_code and stage:
            from core.error_catalog import get_code_for_exception
            error_code = get_code_for_exception("CompilationError", stage=stage)
        super().__init__(
            message,
            user_message=user_message or "The compilation process encountered an error.",
            suggestion=suggestion or "Please try again with a clearer description.",
            error_code=error_code,
        )
        self.stage = stage
        self.details = details or {}


class SchemaValidationError(MotherlabsError):
    """
    Blueprint failed schema validation.

    Raised when a generated blueprint does not conform to the expected schema:
    - Missing required fields (components, relationships)
    - Invalid component types
    - Invalid relationship types
    - Malformed nested structures

    Attributes:
        errors: List of specific validation errors
        blueprint: The invalid blueprint (for debugging)
    """
    def __init__(self, message: str, errors: Optional[list] = None,
                 blueprint: Optional[dict] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(
            message,
            user_message=user_message or "The generated specification has structural issues.",
            suggestion=suggestion or "Try again - this is usually resolved by retrying the compilation.",
            error_code=error_code or "E3006",
        )
        self.errors = errors or []
        self.blueprint = blueprint


class ProviderError(MotherlabsError):
    """
    LLM provider API error.

    Raised when communication with an LLM provider fails:
    - Authentication failures (invalid API key)
    - Rate limiting
    - Model not found
    - Provider service unavailable
    - Malformed responses

    Attributes:
        provider: Name of the provider (claude, openai, gemini, grok)
        status_code: HTTP status code if applicable
        response: Raw response from provider if available
    """
    def __init__(self, message: str, provider: Optional[str] = None,
                 status_code: Optional[int] = None, response: Optional[str] = None,
                 user_message: Optional[str] = None, suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        # Auto-generate user message from status code
        if not user_message and status_code:
            if status_code == 401:
                user_message = "Your API key is invalid or expired."
                suggestion = "Check your API key configuration and try again."
            elif status_code == 429:
                user_message = "The AI service is temporarily rate-limited."
                suggestion = "Wait a moment and try again."
            elif status_code >= 500:
                user_message = "The AI service is experiencing issues."
                suggestion = "Wait a few minutes and try again."
        # Auto-infer error code from status code
        if not error_code and status_code:
            from core.error_catalog import get_code_for_exception
            error_code = get_code_for_exception("ProviderError", status_code=status_code)
        super().__init__(
            message,
            user_message=user_message or "There was a problem communicating with the AI service.",
            suggestion=suggestion or "Check your internet connection and API key, then try again.",
            error_code=error_code,
        )
        self.provider = provider
        self.status_code = status_code
        self.response = response


class TimeoutError(MotherlabsError):
    """
    Operation timed out.

    Raised when an operation exceeds its configured timeout:
    - LLM API calls
    - Synthesis retries
    - Dialogue turns

    Attributes:
        operation: Description of the operation that timed out
        timeout_seconds: The timeout threshold that was exceeded
    """
    def __init__(self, message: str, operation: Optional[str] = None,
                 timeout_seconds: Optional[float] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        # Auto-infer error code from operation
        if not error_code:
            from core.error_catalog import get_code_for_exception
            error_code = get_code_for_exception("TimeoutError", operation=operation or "")
        super().__init__(
            message,
            user_message=user_message or "The operation took too long and was stopped.",
            suggestion=suggestion or "Try again with a simpler description, or try again later.",
            error_code=error_code,
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class CorpusError(MotherlabsError):
    """
    Corpus storage operation failed.

    Raised when corpus persistence operations fail:
    - Storage write failures
    - Record not found
    - Corrupted data
    - Export failures

    Attributes:
        operation: The corpus operation that failed (store, retrieve, export)
        record_id: The record ID involved if applicable
    """
    def __init__(self, message: str, operation: Optional[str] = None,
                 record_id: Optional[str] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(
            message,
            user_message=user_message or "There was a problem saving or retrieving your data.",
            suggestion=suggestion or "Your compilation may still have succeeded. Try listing your compilations.",
            error_code=error_code,
        )
        self.operation = operation
        self.record_id = record_id


class DialogueError(MotherlabsError):
    """
    Dialogue protocol error.

    Raised when the Entity/Process dialogue encounters issues:
    - Agent registration failures
    - Protocol violations
    - Convergence failures
    - Conflict resolution failures

    Attributes:
        turn_count: Number of turns completed before error
        agent: The agent involved if applicable
    """
    def __init__(self, message: str, turn_count: Optional[int] = None,
                 agent: Optional[str] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(
            message,
            user_message=user_message or "The analysis agents couldn't reach agreement on your specification.",
            suggestion=suggestion or "Try rephrasing your description or adding more specific requirements.",
            error_code=error_code or "E3003",
        )
        self.turn_count = turn_count
        self.agent = agent


class ConfigurationError(MotherlabsError):
    """
    Configuration error.

    Raised when configuration is invalid or missing:
    - Missing required API keys
    - Invalid provider configuration
    - Invalid model specification

    Attributes:
        config_key: The configuration key that is invalid/missing
    """
    def __init__(self, message: str, config_key: Optional[str] = None,
                 user_message: Optional[str] = None, suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        # Auto-generate suggestion from config_key
        if not suggestion and config_key:
            suggestion = f"Set the {config_key} environment variable or pass it as a parameter."
        # Auto-infer error code
        if not error_code:
            from core.error_catalog import get_code_for_exception
            error_code = get_code_for_exception("ConfigurationError", config_key=config_key or "")
        super().__init__(
            message,
            user_message=user_message or "There's a configuration issue preventing operation.",
            suggestion=suggestion or "Check your environment variables and configuration settings.",
            error_code=error_code,
        )
        self.config_key = config_key


class ProviderUnavailableError(ProviderError):
    """
    Specific provider is temporarily unavailable.

    Raised when a provider fails but failover may be possible:
    - Temporary service outage
    - Rate limiting
    - Network issues

    Inherits all attributes from ProviderError.
    """
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("user_message", "One of the AI services is temporarily unavailable.")
        kwargs.setdefault("suggestion", "The system will automatically try another provider.")
        kwargs.setdefault("error_code", "E2005")
        super().__init__(message, **kwargs)


class InputQualityError(MotherlabsError):
    """
    Input quality too low for compilation.

    Phase 9.1: Raised when input text is too vague or short to produce
    a meaningful blueprint. Prevents wasting LLM tokens.

    Attributes:
        quality_score: The QualityScore result
    """
    def __init__(self, message: str, quality_score=None,
                 user_message: Optional[str] = None, suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        if quality_score and not suggestion:
            suggestion = quality_score.suggestion
        # Auto-infer error code from quality score
        if not error_code:
            from core.error_catalog import get_code_for_exception
            error_code = get_code_for_exception(
                "InputQualityError",
                quality_score=quality_score,
            )
        super().__init__(
            message,
            user_message=user_message or "Your description is too vague to compile.",
            suggestion=suggestion or "Add more detail: describe the domain, who uses it, and what they can do.",
            error_code=error_code,
        )
        self.quality_score = quality_score


class GraphError(MotherlabsError):
    """
    Dependency graph error.

    Raised when the component dependency graph has structural issues:
    - Cyclic dependencies
    - Degenerate graph (no nodes)
    - Invalid topology

    Attributes:
        cycle_nodes: Set of node names involved in the cycle
    """
    def __init__(self, message: str, cycle_nodes=None,
                 user_message: Optional[str] = None,
                 suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        super().__init__(
            message,
            user_message=user_message or "The blueprint has circular dependencies between components.",
            suggestion=suggestion or "Review component dependencies and remove circular references.",
            error_code=error_code or "E7001",
        )
        self.cycle_nodes = cycle_nodes or set()


class CostCapExceededError(CompilationError):
    """
    Cost cap exceeded during compilation.

    Phase 21: Raised when accumulated token cost exceeds the configured cap.

    Attributes:
        current_cost: Accumulated cost so far (USD)
        cap: The cap that was exceeded (USD)
    """
    def __init__(self, message: str, current_cost: float = 0.0, cap: float = 0.0,
                 user_message: Optional[str] = None, suggestion: Optional[str] = None,
                 error_code: Optional[str] = None):
        super().__init__(
            message,
            stage="cost_cap",
            user_message=user_message or f"This compilation exceeded the cost limit (${cap:.2f}).",
            suggestion=suggestion or "Try a simpler description or increase the cost cap.",
            error_code=error_code or "E8001",
        )
        self.current_cost = current_cost
        self.cap = cap


class BuildError(MotherlabsError):
    """
    Runtime build loop error.

    Phase 27: Raised when the build loop encounters issues:
    - Dependency installation failures
    - Import check failures
    - Test failures after max iterations
    - Subprocess timeouts

    Attributes:
        iteration: Build loop iteration where the error occurred
        phase: Build phase (install, import, test, smoke)
    """
    def __init__(self, message: str, iteration: int = 0,
                 phase: Optional[str] = None, user_message: Optional[str] = None,
                 suggestion: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(
            message,
            user_message=user_message or "The build loop encountered an error while validating generated code.",
            suggestion=suggestion or "Check the build output for specific errors. The generated code may need manual fixes.",
            error_code=error_code or "E9001",
        )
        self.iteration = iteration
        self.phase = phase


class FailoverExhaustedError(ProviderError):
    """
    All failover providers have been exhausted.

    Raised when every configured provider has failed:
    - All providers returned errors
    - No more fallback options available

    Attributes:
        providers_tried: List of providers that were attempted
        errors: Dict mapping provider name to error message
    """
    def __init__(
        self,
        message: str,
        providers_tried: Optional[list] = None,
        errors: Optional[dict] = None,
        **kwargs
    ):
        kwargs.setdefault(
            "user_message",
            "All AI services are currently unavailable."
        )
        kwargs.setdefault(
            "suggestion",
            "Please wait a few minutes and try again, or check your API keys."
        )
        kwargs.setdefault("error_code", "E2004")
        super().__init__(message, **kwargs)
        self.providers_tried = providers_tried or []
        self.errors = errors or {}
