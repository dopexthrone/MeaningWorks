"""
Motherlabs Error Catalog.

Phase 9.4: Actionable Error Messages

Central registry of error codes with descriptions, root causes, and fix examples.
Each error code maps to a structured entry that can be attached to exceptions
for user-facing diagnostics.

Error code categories:
    E1xxx — Input errors (vague, too short, quality gate)
    E2xxx — Provider errors (auth, rate limit, all failed)
    E3xxx — Compilation errors (intent, dialogue, synthesis, verification)
    E4xxx — Timeout errors
    E5xxx — Codegen errors
    E9xxx — Build loop errors (Phase 27)
    E10xxx — API authentication errors (invalid key, rate limit, budget)
    E11xxx — Tool sharing errors (import validation, export, registry)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class ErrorEntry:
    """A single entry in the error catalog."""
    code: str
    title: str
    root_cause: str
    fix_examples: List[str] = field(default_factory=list)


# =============================================================================
# ERROR CATALOG
# =============================================================================

CATALOG = {
    # -----------------------------------------------------------------
    # E1xxx — Input errors
    # -----------------------------------------------------------------
    "E1001": ErrorEntry(
        code="E1001",
        title="Input too vague",
        root_cause="The description lacks domain-specific nouns, actors, or actions.",
        fix_examples=[
            "Name the domain — what kind of business or system is this for?",
            "Add actors — who uses it? (roles, not generic 'users')",
            "Add actions — what do those actors do? (schedule, manage, track, etc.)",
        ],
    ),
    "E1002": ErrorEntry(
        code="E1002",
        title="Input too short",
        root_cause="The description is too brief to extract meaningful intent.",
        fix_examples=[
            "Aim for at least 10-15 words describing what the system does.",
            "Include the domain, key actors, and the main actions they perform.",
        ],
    ),
    "E1003": ErrorEntry(
        code="E1003",
        title="Input quality below custom threshold",
        root_cause="The description scored below the custom min_quality_score you specified.",
        fix_examples=[
            "Lower the min_quality_score parameter.",
            "Add more detail to your description.",
        ],
    ),

    # -----------------------------------------------------------------
    # E2xxx — Provider errors
    # -----------------------------------------------------------------
    "E2001": ErrorEntry(
        code="E2001",
        title="Invalid API key",
        root_cause="The API key for the LLM provider is missing, expired, or invalid.",
        fix_examples=[
            "Check that your API key environment variable is set (e.g. ANTHROPIC_API_KEY).",
            "Regenerate the key in your provider's dashboard.",
        ],
    ),
    "E2002": ErrorEntry(
        code="E2002",
        title="Rate limited",
        root_cause="The LLM provider is throttling requests due to quota or rate limits.",
        fix_examples=[
            "Wait 30-60 seconds and retry.",
            "Upgrade your provider plan for higher limits.",
        ],
    ),
    "E2003": ErrorEntry(
        code="E2003",
        title="Provider service error",
        root_cause="The LLM provider returned a server error (5xx).",
        fix_examples=[
            "Wait a few minutes and retry.",
            "Check the provider's status page for outages.",
        ],
    ),
    "E2004": ErrorEntry(
        code="E2004",
        title="All providers exhausted",
        root_cause="Every configured failover provider has failed.",
        fix_examples=[
            "Check your API keys for all configured providers.",
            "Wait a few minutes — multiple providers may be having issues simultaneously.",
            "Add more failover providers to improve resilience.",
        ],
    ),
    "E2005": ErrorEntry(
        code="E2005",
        title="Provider temporarily unavailable",
        root_cause="One provider is down but failover to another may succeed.",
        fix_examples=[
            "The system will automatically try the next provider.",
            "If all providers fail, check your configuration.",
        ],
    ),

    # -----------------------------------------------------------------
    # E3xxx — Compilation errors
    # -----------------------------------------------------------------
    "E3001": ErrorEntry(
        code="E3001",
        title="Intent extraction failed",
        root_cause="Could not extract core need, domain, or actors from the description.",
        fix_examples=[
            "Rephrase with clearer structure: who uses the system, what they do, what domain it's in.",
            'Example: "A clinic management system where doctors schedule appointments and nurses manage patient records"',
        ],
    ),
    "E3002": ErrorEntry(
        code="E3002",
        title="Persona generation failed",
        root_cause="Could not generate domain-specific perspectives for the analysis.",
        fix_examples=[
            "Provide more domain context (industry, user roles).",
            "Try again — transient LLM issues may resolve on retry.",
        ],
    ),
    "E3003": ErrorEntry(
        code="E3003",
        title="Dialogue did not converge",
        root_cause="The Entity/Process agents could not reach agreement within the turn limit.",
        fix_examples=[
            "Simplify the description or break it into smaller systems.",
            "Add explicit constraints to reduce ambiguity.",
        ],
    ),
    "E3004": ErrorEntry(
        code="E3004",
        title="Synthesis incomplete",
        root_cause="The blueprint is missing required components or has low canonical coverage.",
        fix_examples=[
            "Try again — synthesis quality varies between attempts.",
            "Provide canonical_components to enforce specific outputs.",
        ],
    ),
    "E3005": ErrorEntry(
        code="E3005",
        title="Verification failed",
        root_cause="The generated blueprint did not meet quality thresholds for completeness or traceability.",
        fix_examples=[
            "Add more specific requirements or constraints to your description.",
            "Try again with a more detailed description.",
        ],
    ),
    "E3006": ErrorEntry(
        code="E3006",
        title="Schema validation failed",
        root_cause="The generated blueprint has structural issues (missing fields, invalid types).",
        fix_examples=[
            "This is usually a transient issue. Try again.",
            "If persistent, the description may be too ambiguous for structured output.",
        ],
    ),

    # -----------------------------------------------------------------
    # E4xxx — Timeout errors
    # -----------------------------------------------------------------
    "E4001": ErrorEntry(
        code="E4001",
        title="Stage timed out",
        root_cause="A pipeline stage exceeded its configured time limit.",
        fix_examples=[
            "Try again — network latency or provider load may have caused the timeout.",
            "Simplify your description to reduce processing time.",
        ],
    ),
    "E4002": ErrorEntry(
        code="E4002",
        title="LLM call timed out",
        root_cause="An individual LLM API call did not respond in time.",
        fix_examples=[
            "Retry — the provider may have been temporarily slow.",
            "Check your network connection.",
        ],
    ),

    # -----------------------------------------------------------------
    # E5xxx — Codegen errors
    # -----------------------------------------------------------------
    "E5001": ErrorEntry(
        code="E5001",
        title="Code generation failed",
        root_cause="Could not generate Python code from the blueprint.",
        fix_examples=[
            "Check that the blueprint has valid components with names and types.",
            "Try compiling again to get a cleaner blueprint.",
        ],
    ),

    # -----------------------------------------------------------------
    # E6xxx — Configuration errors
    # -----------------------------------------------------------------
    "E6001": ErrorEntry(
        code="E6001",
        title="Missing configuration",
        root_cause="A required configuration value is missing.",
        fix_examples=[
            "Set the required environment variable.",
            "Pass the value as a parameter to the constructor.",
        ],
    ),
    "E6002": ErrorEntry(
        code="E6002",
        title="Invalid provider",
        root_cause="The specified LLM provider is not recognized.",
        fix_examples=[
            'Use one of: "claude", "openai", "gemini", "grok", or "auto".',
        ],
    ),

    # -----------------------------------------------------------------
    # E7xxx — Graph / edge case errors
    # -----------------------------------------------------------------
    "E7001": ErrorEntry(
        code="E7001",
        title="Dependency cycle detected",
        root_cause="Components have circular dependencies that prevent ordered materialization.",
        fix_examples=[
            "Review the blueprint relationships for circular depends_on chains.",
            "The system will attempt to break cycles automatically and continue.",
        ],
    ),
    "E7002": ErrorEntry(
        code="E7002",
        title="Blueprint too degraded for materialization",
        root_cause="The blueprint has critical structural issues (0 components, collisions, etc.).",
        fix_examples=[
            "Provide a more detailed description with clear component names.",
            "Ensure the description includes at least one identifiable system component.",
        ],
    ),
    "E7003": ErrorEntry(
        code="E7003",
        title="Materialization plan empty",
        root_cause="No components could be planned for code generation.",
        fix_examples=[
            "Check that the blueprint has named components with types.",
            "Try recompiling with a more specific description.",
        ],
    ),
    "E7004": ErrorEntry(
        code="E7004",
        title="Contradictory constraints detected",
        root_cause="Two or more constraints on the same field are logically incompatible.",
        fix_examples=[
            "Review constraints for conflicting range or enum definitions.",
            "This is a warning — the blueprint was still produced, but results may be incoherent.",
        ],
    ),
    "E7005": ErrorEntry(
        code="E7005",
        title="Layer gate validation failed",
        root_cause="Emission layer output failed gate validation.",
        fix_examples=[
            "Check generated code for syntax errors or unresolved imports.",
        ],
    ),
    "E7006": ErrorEntry(
        code="E7006",
        title="Layered emission fallback to flat",
        root_cause="All components share the same type; layered emission unnecessary.",
        fix_examples=[
            "Informational. Ensure blueprint has diverse component types for layered emission.",
        ],
    ),

    # -----------------------------------------------------------------
    # E8xxx — Cost errors (Phase 21)
    # -----------------------------------------------------------------
    "E8001": ErrorEntry(
        code="E8001",
        title="Per-compilation cost cap exceeded",
        root_cause="The LLM token cost for this compilation exceeded the per-compilation limit.",
        fix_examples=[
            "Simplify the description to reduce the number of LLM calls.",
            "Increase the per_compilation_cap_usd in the cost configuration.",
        ],
    ),
    "E8002": ErrorEntry(
        code="E8002",
        title="Session cost cap exceeded",
        root_cause="The cumulative LLM token cost for this session exceeded the session limit.",
        fix_examples=[
            "Start a new session to reset the cost counter.",
            "Increase the session_cap_usd in the cost configuration.",
        ],
    ),
    # -----------------------------------------------------------------
    # E9xxx — Build loop errors (Phase 27)
    # -----------------------------------------------------------------
    "E9001": ErrorEntry(
        code="E9001",
        title="Build validation failed",
        root_cause="Generated code failed runtime validation (import errors, test failures).",
        fix_examples=[
            "Review the build output for specific import or test errors.",
            "Re-run with --build to attempt automatic fixes.",
            "Check that the blueprint dependencies are correctly specified.",
        ],
    ),
    "E9002": ErrorEntry(
        code="E9002",
        title="Dependency installation failed",
        root_cause="pip install failed for one or more inferred requirements.",
        fix_examples=[
            "Check that the package names in requirements.txt are correct.",
            "Verify network connectivity for pip downloads.",
            "Manually install the failing package and re-run.",
        ],
    ),
    "E9003": ErrorEntry(
        code="E9003",
        title="Build loop max iterations exceeded",
        root_cause="The build loop could not fix all errors within the iteration limit.",
        fix_examples=[
            "Increase max_iterations in BuildSpec if errors are converging.",
            "Review unfixed components — they may have architectural issues.",
            "Simplify the blueprint or add more constraints.",
        ],
    ),

    # -----------------------------------------------------------------
    # E10xxx — API authentication errors
    # -----------------------------------------------------------------
    "E10001": ErrorEntry(
        code="E10001",
        title="Invalid API key",
        root_cause="The provided API key is missing, invalid, or has been revoked.",
        fix_examples=[
            "Check that the X-API-Key header is set correctly.",
            "Generate a new key with: motherlabs keys create --name my-key",
            "Verify the key has not been revoked with: motherlabs keys list",
        ],
    ),
    "E10002": ErrorEntry(
        code="E10002",
        title="Rate limit exceeded",
        root_cause="Too many requests within the rate-limit window for this API key.",
        fix_examples=[
            "Wait for the rate limit window to reset (check X-RateLimit-Reset header).",
            "Request a higher rate limit for your key.",
        ],
    ),
    "E10003": ErrorEntry(
        code="E10003",
        title="Budget exceeded",
        root_cause="The cumulative spend for this API key has reached its budget limit.",
        fix_examples=[
            "Request a budget increase for your key.",
            "Create a new key with a higher budget: motherlabs keys create --name new-key --budget 100",
        ],
    ),
    # -----------------------------------------------------------------
    # E11xxx — Tool sharing errors
    # -----------------------------------------------------------------
    "E11001": ErrorEntry(
        code="E11001",
        title="Import rejected: provenance chain invalid",
        root_cause="The tool package has an empty or malformed provenance chain.",
        fix_examples=[
            "Ensure the .mtool file was exported from a valid Motherlabs instance.",
            "Check that the provenance_chain field is non-empty with valid timestamps.",
        ],
    ),
    "E11002": ErrorEntry(
        code="E11002",
        title="Import rejected: trust score below threshold",
        root_cause="The tool's trust score or verification badge does not meet the import threshold.",
        fix_examples=[
            "Lower the --min-trust threshold if you trust this source.",
            "Re-compile the tool with more detailed input to improve trust score.",
        ],
    ),
    "E11003": ErrorEntry(
        code="E11003",
        title="Import rejected: dangerous code patterns detected",
        root_cause="The generated code contains exec/eval/subprocess or other dangerous patterns.",
        fix_examples=[
            "Inspect the .mtool file's generated_code for suspicious patterns.",
            "Contact the source instance maintainer to review the code.",
        ],
    ),
    "E11004": ErrorEntry(
        code="E11004",
        title="Import rejected: blueprint integrity check failed",
        root_cause="The blueprint has structural issues (no components, duplicate names, invalid references).",
        fix_examples=[
            "Re-export the tool from the source instance.",
            "Inspect the .mtool file's blueprint for structural issues.",
        ],
    ),
    "E11005": ErrorEntry(
        code="E11005",
        title="Export failed: compilation not found in corpus",
        root_cause="The specified compilation ID does not exist in the local corpus.",
        fix_examples=[
            "Run 'motherlabs corpus list' to see available compilations.",
            "Check the compilation ID — it should be a 12-character hex hash.",
        ],
    ),
    "E11006": ErrorEntry(
        code="E11006",
        title="Export failed: compilation not verified",
        root_cause="The compilation exists but has not been verified.",
        fix_examples=[
            "Re-compile with verification enabled.",
            "Run 'motherlabs trust <id>' to check verification status.",
        ],
    ),
    "E11007": ErrorEntry(
        code="E11007",
        title="Tool package deserialization failed",
        root_cause="The .mtool file is corrupted or has an incompatible format version.",
        fix_examples=[
            "Check that the file is valid JSON with format='mtool'.",
            "Re-export the tool from the source instance.",
        ],
    ),
    "E11008": ErrorEntry(
        code="E11008",
        title="Duplicate tool: fingerprint already in registry",
        root_cause="A tool with the same structural fingerprint is already registered.",
        fix_examples=[
            "Run 'motherlabs tools list' to see existing tools.",
            "The existing tool has the same blueprint topology — no import needed.",
        ],
    ),
}


def get_entry(code: str) -> Optional[ErrorEntry]:
    """Look up an error catalog entry by code."""
    return CATALOG.get(code)


def get_code_for_exception(exception_class_name: str, **context) -> Optional[str]:
    """
    Infer an error code from an exception class name and context.

    Args:
        exception_class_name: Name of the exception class
        **context: Additional context (stage, status_code, etc.)

    Returns:
        Error code string or None
    """
    mapping = {
        "InputQualityError": _infer_input_quality_code,
        "ProviderError": _infer_provider_code,
        "ProviderUnavailableError": lambda **ctx: "E2005",
        "FailoverExhaustedError": lambda **ctx: "E2004",
        "CompilationError": _infer_compilation_code,
        "SchemaValidationError": lambda **ctx: "E3006",
        "TimeoutError": _infer_timeout_code,
        "ConfigurationError": _infer_config_code,
        "DialogueError": lambda **ctx: "E3003",
        "GraphError": lambda **ctx: "E7001",
        "CostCapExceededError": lambda **ctx: "E8001",
        "BuildError": lambda **ctx: ctx.get("error_code", "E9001"),
        "CorpusError": lambda **ctx: None,
    }

    resolver = mapping.get(exception_class_name)
    if resolver:
        return resolver(**context)
    return None


def _infer_input_quality_code(**context) -> str:
    """Infer input quality error code from context."""
    quality_score = context.get("quality_score")
    if quality_score:
        # Very short input
        if hasattr(quality_score, "length_score") and quality_score.length_score < 0.1:
            return "E1002"
    custom_threshold = context.get("custom_threshold", False)
    if custom_threshold:
        return "E1003"
    return "E1001"


def _infer_provider_code(**context) -> str:
    """Infer provider error code from status code."""
    status_code = context.get("status_code")
    if status_code == 401:
        return "E2001"
    elif status_code == 429:
        return "E2002"
    elif status_code and status_code >= 500:
        return "E2003"
    return "E2001"  # Default to auth


def _infer_compilation_code(**context) -> str:
    """Infer compilation error code from stage."""
    stage = context.get("stage")
    stage_map = {
        "intent": "E3001",
        "personas": "E3002",
        "dialogue": "E3003",
        "synthesis": "E3004",
        "verification": "E3005",
    }
    return stage_map.get(stage, "E3004")


def _infer_timeout_code(**context) -> str:
    """Infer timeout error code from operation."""
    operation = context.get("operation", "")
    if operation and any(s in operation for s in ("intent", "persona", "synthesis", "dialogue", "verif")):
        return "E4001"
    return "E4002"


def _infer_config_code(**context) -> str:
    """Infer configuration error code."""
    config_key = context.get("config_key", "")
    if "provider" in str(config_key).lower():
        return "E6002"
    return "E6001"
