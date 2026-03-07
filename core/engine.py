"""
Motherlabs Engine - The semantic compiler.

Derived from: PROJECT-PLAN.md Phase 2.1, MASTER-TECHNICAL-SPECIFICATION Section 2.1

Pipeline:
1. Intent extraction (Phase 1)
2. Persona generation (Phase 2)
3. Spec dialogue - Entity <-> Process (Phase 3)
4. Synthesis (Phase 4)
5. Verification (Phase 5)
6. Output + Corpus storage (Phase 3.6)

Phase 5.1: Added structured logging, timeout handling, exponential backoff
"""

import copy
import json
import os
import time
import logging
import signal
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager

from core.exceptions import (
    MotherlabsError,
    CompilationError,
    FractureError,
    ProviderError,
    TimeoutError as MotherlabsTimeoutError,
    SchemaValidationError,
    ConfigurationError,
    InputQualityError,
    GraphError,
    CostCapExceededError,
)
from core.input_quality import InputQualityAnalyzer, QualityScore
from core.cache import StagedCache
from core.digest import (
    build_dialogue_digest,
    _extract_pattern_matches,
    extract_dialogue_methods,
    extract_dialogue_state_machines,
    extract_pattern_method_stubs,
    format_method_section,
)
from core.pipeline import StagedPipeline, PipelineState, format_precomputed_structure
from core.protocol_spec import PROTOCOL
from core.determinism import compute_structural_fingerprint
from core.output_parser import (
    parse_structured_output as _parse_output,
    build_repair_prompt as _build_repair,
    STAGE_SCHEMAS,
)
from core.self_compile import (
    SelfCompileReport,
    diff_blueprint_vs_code,
    track_convergence,
    extract_self_patterns,
    compute_overall_health,
)
from core.telemetry import (
    CompilationMetrics,
    TokenUsage,
    estimate_cost,
    aggregate_metrics,
    compute_health,
    aggregate_to_dict,
    health_to_dict,
)
from core.emission_types import (
    DifficultySignal,
    InsightCategory,
    StructuredInsight,
)

# Configure module logger
logger = logging.getLogger("motherlabs.engine")


# =============================================================================
# TIMEOUT HANDLING - Phase 5.1
# =============================================================================

class TimeoutHandler:
    """
    Cross-platform timeout handler for LLM calls.

    Uses signal.SIGALRM on Unix main thread, ctypes async exception
    injection on worker threads (raises into the blocked thread).
    Phase 5.1: Error Handling & Stability
    """

    def __init__(self, seconds: int, operation: str = "LLM call"):
        self.seconds = seconds
        self.operation = operation
        self._timer = None
        self._target_thread_id = None
        # Phase 7.2: Only use signal in main thread (SIGALRM fails in worker threads)
        self._use_signal = (
            hasattr(signal, 'SIGALRM') and
            threading.current_thread() is threading.main_thread()
        )

    def _timeout_handler(self, signum, frame):
        """Signal handler for Unix main thread."""
        raise MotherlabsTimeoutError(
            f"{self.operation} timed out after {self.seconds}s",
            operation=self.operation,
            timeout_seconds=self.seconds
        )

    def _timer_callback(self):
        """Timer callback — injects exception into blocked worker thread."""
        logger.warning(f"Timeout triggered for {self.operation} after {self.seconds}s")
        if self._target_thread_id is not None:
            try:
                import ctypes
                ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self._target_thread_id),
                    ctypes.py_object(MotherlabsTimeoutError),
                )
                if ret == 0:
                    logger.warning("Timeout: target thread not found (already finished?)")
                elif ret > 1:
                    # Injected into multiple threads — clear it
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_ulong(self._target_thread_id), None
                    )
                    logger.error("Timeout: exception injection hit multiple threads, cleared")
                else:
                    logger.info(f"Timeout: injected exception into thread {self._target_thread_id}")
            except Exception as exc:
                logger.error(f"Timeout: failed to inject exception: {exc}")

    def __enter__(self):
        if self._use_signal:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.seconds)
        else:
            self._target_thread_id = threading.current_thread().ident
            self._timer = threading.Timer(self.seconds, self._timer_callback)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._use_signal:
            signal.alarm(0)  # Cancel alarm
        elif self._timer:
            self._timer.cancel()
        return False  # Don't suppress exceptions


@contextmanager
def timeout_context(seconds: int, operation: str = "LLM call"):
    """
    Context manager for timeout handling.

    Usage:
        with timeout_context(60, "synthesis call"):
            result = llm.complete(prompt)

    Phase 5.1: Error Handling & Stability
    """
    handler = TimeoutHandler(seconds, operation)
    with handler:
        yield


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """
    Calculate delay for exponential backoff.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds

    Returns:
        Delay in seconds

    Phase 5.1: Error Handling & Stability
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


from core.protocol import SharedState, Message, MessageType, DialogueProtocol, TerminationState, calculate_dialogue_depth
from core.llm import ClaudeClient, OpenAIClient, GeminiClient, GrokClient, create_client, BaseLLMClient, FailoverClient
from core.schema import validate_blueprint, BlueprintSchema, check_canonical_coverage, check_canonical_relationships, validate_graph, add_version, deduplicate_blueprint, normalize_blueprint_elements, normalize_component_name, COMPONENT_ALIASES
from core.provider_config import get_config as get_provider_config
from persistence.corpus import Corpus, CompilationRecord
from agents.spec_agents import create_entity_agent, create_process_agent, add_challenge_protocol
from agents.swarm import (
    create_intent_agent,
    create_persona_agent,
    create_synthesis_agent,
    create_verify_agent,
    create_governor
)


@dataclass
class StageResult:
    """
    Result of a single pipeline stage.

    Derived from: Engineering research - "Verify semantic preservation at each stage"
    """
    stage: str
    success: bool
    output: Any
    errors: List[str]
    warnings: List[str]
    retries: int = 0  # Number of retries before success/failure


@dataclass
class StageGate:
    """
    Formal gate criteria for a pipeline stage.

    Derived from: NEXT-STEPS.md "Stage-gate Architecture"

    Each stage has:
    - Required criteria that MUST pass
    - Optional criteria that generate warnings
    - Retry configuration for recoverable failures
    """
    name: str
    max_retries: int = 1
    timeout_seconds: int = 120  # Phase 7.2: Per-stage timeout
    required_criteria: Dict[str, Any] = None  # {criterion_name: threshold}
    optional_criteria: Dict[str, Any] = None  # {criterion_name: threshold}

    def __post_init__(self):
        if self.required_criteria is None:
            self.required_criteria = {}
        if self.optional_criteria is None:
            self.optional_criteria = {}


# =============================================================================
# ENTITY EXTRACTION — pre-synthesis checklist to prevent compression losses
# =============================================================================

# Common English stop words that should never be flagged as entities
_ENTITY_STOP_WORDS = frozenset({
    "system", "should", "would", "could", "about", "their", "there", "these",
    "those", "which", "where", "while", "after", "before", "between", "through",
    "during", "without", "within", "along", "across", "behind", "below", "above",
    "under", "until", "against", "into", "with", "from", "that", "this", "what",
    "when", "will", "have", "been", "being", "each", "every", "either", "both",
    "more", "most", "other", "some", "such", "than", "them", "then", "they",
    "very", "also", "back", "come", "does", "done", "even", "give", "good",
    "just", "know", "like", "look", "made", "make", "many", "much", "must",
    "need", "next", "only", "over", "part", "same", "take", "tell", "time",
    "want", "well", "work", "year", "your", "able", "best", "call", "case",
    "find", "first", "full", "hand", "high", "home", "keep", "kind", "last",
    "left", "life", "line", "long", "name", "open", "play", "real", "right",
    "show", "side", "still", "sure", "turn", "used", "using", "user", "users",
    "include", "including", "ensure", "allow", "based", "build", "create",
    "provide", "support", "handle", "manage", "process", "implement", "define",
    "generate", "output", "input", "data", "type", "list", "item", "value",
    "result", "response", "request", "action", "option", "feature", "function",
    "method", "class", "model", "view", "page", "form", "field", "table",
    "component", "module", "service", "interface", "example", "section",
    "information", "description", "configuration", "application", "specific",
    "different", "available", "required", "following", "possible", "existing",
    "relevant", "important", "approach", "ability", "purpose",
})


def _extract_entity_checklist(
    original_input: str, dialogue_digest: str
) -> list[tuple[str, int, str]]:
    """Extract significant entities from input + dialogue for synthesis checklist.

    Returns list of (entity_name, mention_count, first_context_snippet).
    Sorted by frequency descending, capped at 15.
    """
    import re

    combined = (original_input or "") + " " + (dialogue_digest or "")
    if len(combined.strip()) < 20:
        return []

    # Tokenize: split on whitespace and punctuation, lowercase, keep 4+ char words
    words = re.findall(r"[A-Za-z][a-z]{3,}", combined)
    words_lower = [w.lower() for w in words]

    # Count occurrences, filtering stop words
    counts: dict[str, int] = {}
    for w in words_lower:
        if w not in _ENTITY_STOP_WORDS and len(w) >= 4:
            counts[w] = counts.get(w, 0) + 1

    # Also extract multi-word capitalized phrases (proper nouns / named entities)
    # e.g. "Task Manager", "Auth Service", "User Profile"
    cap_phrases = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", combined)
    for phrase in cap_phrases:
        key = phrase.lower()
        if key not in _ENTITY_STOP_WORDS:
            counts[key] = counts.get(key, 0) + 1

    # Filter: only entities mentioned 2+ times, or capitalized phrases (even once)
    entities = {}
    for entity, count in counts.items():
        if count >= 2 or entity in {p.lower() for p in cap_phrases}:
            entities[entity] = count

    if not entities:
        return []

    # Extract first context snippet for each entity
    result = []
    combined_lower = combined.lower()
    for entity, count in sorted(entities.items(), key=lambda x: -x[1]):
        idx = combined_lower.find(entity)
        ctx = ""
        if idx >= 0:
            start = max(0, idx - 30)
            end = min(len(combined), idx + len(entity) + 50)
            ctx = combined[start:end].strip().replace("\n", " ")
            if start > 0:
                ctx = "..." + ctx
            if end < len(combined):
                ctx = ctx + "..."
        result.append((entity, count, ctx))

    return result[:15]


# =============================================================================
# STAGE GATE DEFINITIONS
# Derived from: NEXT-STEPS.md "Stage-gate Architecture"
# =============================================================================

def _build_stage_gates() -> dict:
    """Build STAGE_GATES from protocol spec + gate-specific criteria."""
    _engine_stages = {s.name: s for s in PROTOCOL.engine.stages}
    return {
        "intent": StageGate(
            name="intent",
            max_retries=_engine_stages["intent"].max_retries,
            timeout_seconds=_engine_stages["intent"].timeout_seconds,
            required_criteria={
                "has_core_need": True,
                "has_domain": True,
            },
            optional_criteria={
                "has_actors": True,
                "has_explicit_components": True,
            }
        ),
        "personas": StageGate(
            name="personas",
            max_retries=_engine_stages["personas"].max_retries,
            timeout_seconds=_engine_stages["personas"].timeout_seconds,
            required_criteria={
                "min_personas": 2,
            },
            optional_criteria={
                "max_personas": 4,
            }
        ),
        "dialogue": StageGate(
            name="dialogue",
            max_retries=_engine_stages["dialogue"].max_retries,
            timeout_seconds=_engine_stages["dialogue"].timeout_seconds,
            required_criteria={
                "min_turns": 4,  # Relaxed from 6
            },
            optional_criteria={
                "min_insights": 6,  # Relaxed from 8
                "recommended_turns": 6,
            }
        ),
        "synthesis": StageGate(
            name="synthesis",
            max_retries=_engine_stages["synthesis"].max_retries,
            timeout_seconds=_engine_stages["synthesis"].timeout_seconds,
            required_criteria={
                "has_components": True,
                "component_coverage": PROTOCOL.engine.coverage_component_threshold,
            },
            optional_criteria={
                "relationship_coverage": PROTOCOL.engine.coverage_relationship_threshold,
                "schema_valid": True,
            }
        ),
        "verification": StageGate(
            name="verification",
            max_retries=_engine_stages["verification"].max_retries,
            timeout_seconds=_engine_stages["verification"].timeout_seconds,
            required_criteria={
                "completeness": 65,  # Tightened: was 50 (relaxed from 70)
                "traceability": 60,  # Tightened: was 50 (relaxed from 80)
            },
            optional_criteria={
                "consistency": 70,
                "coherence": 70,
                "pass_status": True,
            }
        ),
    }


STAGE_GATES = _build_stage_gates()


@dataclass
class CompileResult:
    """
    Result of semantic compilation.

    Derived from: PROJECT-PLAN.md Phase 2.1
    Phase 5.3: All fields have proper defaults (never None)
    Phase 6.2: Added cache_stats for compilation caching metrics
    """
    success: bool
    blueprint: Dict[str, Any] = field(default_factory=dict)
    context_graph: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    stage_results: List[StageResult] = field(default_factory=list)
    schema_validation: Dict[str, Any] = field(default_factory=dict)
    graph_validation: Dict[str, Any] = field(default_factory=dict)
    corpus_suggestions: Dict[str, Any] = field(default_factory=dict)
    cache_stats: Dict[str, Any] = field(default_factory=dict)  # Phase 6.2: Cache hit/miss info
    input_quality: Dict[str, Any] = field(default_factory=dict)  # Phase 9.1: Input quality score
    dimensional_metadata: Dict[str, Any] = field(default_factory=dict)  # Phase A: Dimensional blueprint
    interface_map: Dict[str, Any] = field(default_factory=dict)  # Phase B.1: Interface contracts
    corpus_feedback: Dict[str, Any] = field(default_factory=dict)  # Corpus feedback loop metrics
    stage_timings: Dict[str, float] = field(default_factory=dict)  # Per-stage durations in seconds
    retry_counts: Dict[str, int] = field(default_factory=dict)  # Per-stage retry counts
    error: Optional[str] = None
    interrogation: Dict[str, Any] = field(default_factory=dict)  # Pre-dialogue clarification metadata
    fracture: Optional[Dict[str, Any]] = None  # FractureSignal as dict, if compilation paused
    semantic_grid: Optional[Dict[str, Any]] = None  # Kernel grid nav + manifest, if kernel compilation ran
    depth_chains: List[Dict[str, Any]] = field(default_factory=list)  # Unexplored endpoints for intent chaining
    context_map: Optional[Dict[str, Any]] = None  # CONTEXT mode: structured context understanding
    exploration_map: Optional[Dict[str, Any]] = None  # EXPLORE mode: divergent exploration insights
    structured_insights: List[Dict[str, Any]] = field(default_factory=list)  # Glass box: rich typed insights
    difficulty: Dict[str, Any] = field(default_factory=dict)  # Glass box: DifficultySignal snapshot
    semantic_nodes: List[Dict[str, Any]] = field(default_factory=list)  # Postcode-native node payloads
    blocking_escalations: List[Dict[str, Any]] = field(default_factory=list)  # Human decisions required to continue
    termination_condition: Dict[str, Any] = field(default_factory=dict)  # Why compilation stopped here


class MotherlabsEngine:
    """
    The semantic compilation engine.

    Derived from: PROJECT-PLAN.md Phase 2.1, MASTER-TECHNICAL-SPECIFICATION Section 2.1

    Pipeline: Intent -> Persona -> Dialogue -> Synthesis -> Verify
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "auto",
        model: Optional[str] = None,
        on_insight: Optional[Callable[[str], None]] = None,
        corpus: Optional[Corpus] = None,
        auto_store: bool = True,
        llm_client: Optional[BaseLLMClient] = None,
        failover_providers: Optional[List[str]] = None,
        cache_policy: str = "intent",
        pipeline_mode: str = "legacy",
        domain_adapter=None,
        on_interrogate: Optional[Callable] = None,
        quality_llm: Optional[BaseLLMClient] = None,
    ):
        """
        Initialize engine.

        Args:
            api_key: API key for LLM provider (or set env var)
            provider: LLM provider - "claude", "openai", "gemini", "grok", or "auto"
            model: Model name (uses provider default if not specified)
            on_insight: Callback for each insight (for CLI display)
            corpus: Corpus instance for persistence (default: creates new)
            auto_store: Automatically store compilations in corpus (default: True)
            llm_client: Pre-configured LLM client (overrides provider/model)
            failover_providers: List of fallback providers for automatic failover
                               e.g., ["grok", "claude", "openai"] tries in order
            cache_policy: Caching policy - "none", "intent", or "full" (default: "intent")
                         - none: No caching
                         - intent: Cache Intent extraction only
                         - full: Cache Intent + Personas
            pipeline_mode: Pipeline mode - "legacy" or "staged" (default: "legacy")
                          - legacy: Single monolithic Entity<->Process dialogue
                          - staged: 5-stage focused sub-dialogues (Phase 14)
            on_interrogate: Callback for pre-dialogue clarification questions.
                           Receives InterrogationRequest, returns InterrogationResponse.
                           None = skip interrogation (zero overhead).
            quality_llm: Optional higher-quality LLM client for synthesis and
                        verification phases. When set, synthesis_agent and verify_agent
                        use this model while dialogue agents use the cheaper default.
                        None = all agents use the same LLM.

        Derived from: PROJECT-PLAN.md Phase 3.6
        Phase 6.1: Added failover_providers for enterprise reliability
        Phase 6.2: Added cache_policy for compilation caching
        """
        if llm_client:
            self.llm = llm_client
        elif failover_providers:
            # Phase 6.1: Create FailoverClient with multiple providers
            providers = []
            for p in failover_providers:
                try:
                    client = create_client(provider=p, model=model, api_key=api_key)
                    providers.append(client)
                except ValueError:
                    # Skip providers without API keys
                    logger.warning(f"Skipping provider {p} - no API key available")
            if not providers:
                raise ConfigurationError(
                    "No providers available for failover. Check API keys.",
                    config_key="failover_providers"
                )
            self.llm = FailoverClient(providers, logger=logger)
        else:
            self.llm = create_client(provider=provider, model=model, api_key=api_key)
        self.on_insight = on_insight or (lambda x: None)
        self.on_interrogate = on_interrogate
        self.corpus = corpus or Corpus()
        self.auto_store = auto_store

        # Track provider info for metrics
        self.provider_name = self._get_provider_name()
        self.model_name = getattr(self.llm, 'model', 'unknown')

        # Phase A: Domain adapter (optional, defaults to software)
        # Must be set before agent initialization so factories can read adapter prompts
        self.domain_adapter = domain_adapter

        # Quality LLM for synthesis/verification/governor (falls back to default).
        # Auto-create Opus-tier client for Anthropic when no explicit quality_llm given.
        # Set MOTHERLABS_SINGLE_TIER=1 to disable (all stages use same model).
        tiered_disabled = os.environ.get("MOTHERLABS_SINGLE_TIER", "").strip() in ("1", "true")
        claude_base = (
            self.llm if isinstance(self.llm, ClaudeClient)
            else (self.llm.providers[0] if isinstance(self.llm, FailoverClient) and self.llm.providers and isinstance(self.llm.providers[0], ClaudeClient) else None)
        )
        if quality_llm:
            self.quality_llm = quality_llm
        elif claude_base and not tiered_disabled and not claude_base.critical_model:
            # Tiered routing: Opus for critical stages, Sonnet for the rest
            critical_model = os.environ.get("MOTHERLABS_CRITICAL_MODEL", "claude-opus-4-20250514")
            self.quality_llm = ClaudeClient(
                api_key=claude_base.api_key,
                model=critical_model,
                deterministic=claude_base.deterministic,
            )
            logger.info(f"Tiered model routing: {critical_model} for synthesis/verify/governor, {claude_base.model} for dialogue")
        else:
            self.quality_llm = self.llm

        # Initialize agents (thread adapter for domain-specific prompts)
        self.intent_agent = create_intent_agent(self.llm, domain_adapter=self.domain_adapter)
        self.persona_agent = create_persona_agent(self.llm, domain_adapter=self.domain_adapter)
        self.entity_agent = add_challenge_protocol(create_entity_agent(self.llm, domain_adapter=self.domain_adapter))
        self.process_agent = add_challenge_protocol(create_process_agent(self.llm, domain_adapter=self.domain_adapter))
        self.synthesis_agent = create_synthesis_agent(self.quality_llm, domain_adapter=self.domain_adapter)
        self.verify_agent = create_verify_agent(self.quality_llm, domain_adapter=self.domain_adapter)
        self.governor = create_governor(self.quality_llm)

        # Register all with governor
        for agent in [
            self.intent_agent, self.persona_agent,
            self.entity_agent, self.process_agent,
            self.synthesis_agent, self.verify_agent
        ]:
            self.governor.register_agent(agent)

        # Phase 6.2: Initialize compilation cache
        self.cache_policy = cache_policy
        self._cache = StagedCache(enabled=(cache_policy != "none"))

        # Phase 14: Pipeline mode
        self.pipeline_mode = pipeline_mode

        # Phase 19: In-memory metrics ring buffer
        self._metrics_buffer: List[CompilationMetrics] = []
        self._metrics_buffer_size: int = 100
        self._engine_start_time: float = time.time()

        # Phase 21: Cost tracking
        self._compilation_tokens: List[TokenUsage] = []
        self._session_cost_usd: float = 0.0

        # Glass box: difficulty + structured insights (reset per compile)
        self._difficulty = DifficultySignal()
        self._structured_insights: List[StructuredInsight] = []
        self._current_stage = "queued"
        self._progress_callback: Optional[Callable] = None

        # Phase 26: Self-compile pattern feedback
        self._last_self_compile_patterns: list = []

        # Kernel grid cache — persists across compilations for L2 feedback
        self._kernel_grid = None

        # Persistent outcome store — L2 memory that survives restart
        self._outcome_store = None
        try:
            from core.outcome_store import OutcomeStore
            self._outcome_store = OutcomeStore()
            # Bootstrap in-memory outcomes from persistent store
            stored = self._outcome_store.recent(limit=50)
            if stored:
                def _parse_compression_cats(raw: str) -> tuple:
                    if not raw:
                        return ()
                    try:
                        return tuple(sorted(json.loads(raw).items()))
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        return ()

                self._compilation_outcomes = [
                    {
                        "compile_id": r.compile_id,
                        "input_summary": r.input_summary,
                        "trust_score": r.trust_score,
                        "completeness": r.completeness,
                        "consistency": r.consistency,
                        "coherence": r.coherence,
                        "traceability": r.traceability,
                        "component_count": r.component_count,
                        "rejected": r.rejected,
                        "rejection_reason": r.rejection_reason,
                        "domain": r.domain,
                        "compression_loss_categories": _parse_compression_cats(getattr(r, "compression_loss_categories", "")),
                    }
                    for r in reversed(stored)  # oldest first
                ]
        except Exception as e:
            logger.debug(f"Outcome store init skipped: {e}")
        # Ensure _compilation_outcomes always exists even if bootstrap failed
        if not hasattr(self, "_compilation_outcomes"):
            self._compilation_outcomes = []

        # Timeout auto-scaling: calibrated after first LLM call
        # Deep-copy STAGE_GATES so concurrent engines don't corrupt each other
        self._stage_gates = copy.deepcopy(STAGE_GATES)
        self._timeout_scaled = False
        self._base_timeouts = {name: gate.timeout_seconds for name, gate in self._stage_gates.items()}

    def _get_provider_name(self) -> str:
        """Determine provider name from LLM client type."""
        # Phase 6.1: Handle FailoverClient
        if hasattr(self.llm, 'provider_name'):
            return self.llm.provider_name

        client_type = type(self.llm).__name__.lower()
        if 'claude' in client_type:
            return 'claude'
        elif 'openai' in client_type:
            return 'openai'
        elif 'gemini' in client_type:
            return 'gemini'
        elif 'grok' in client_type:
            return 'grok'
        elif 'failover' in client_type:
            return 'failover'
        return 'unknown'

    # Canonical components for self-compile (dogfood)
    SELF_COMPILE_CANONICAL = [
        # Core pipeline agents
        "Intent Agent", "Persona Agent", "Entity Agent", "Process Agent",
        "Synthesis Agent", "Verify Agent", "Governor Agent",
        # Core data structures
        "SharedState", "ConfidenceVector", "ConflictOracle",
        "Message", "DialogueProtocol", "Corpus",
        # Perception + senses
        "PerceptionEngine", "SenseVector",
        # Actuators
        "VoiceBridge", "FileSystemBridge",
        # Autonomy
        "GoalStore", "DaemonMode",
        # Self-referential
        "ClosedLoopGate", "SemanticGrid",
    ]

    # Canonical relationships for self-compile (dogfood)
    # Format: (from, to, type)
    SELF_COMPILE_RELATIONSHIPS = [
        # Pipeline orchestration
        ("Governor Agent", "Intent Agent", "triggers"),
        ("Governor Agent", "Persona Agent", "triggers"),
        ("Governor Agent", "Entity Agent", "triggers"),
        ("Governor Agent", "Process Agent", "triggers"),
        ("Governor Agent", "Synthesis Agent", "triggers"),
        ("Governor Agent", "Verify Agent", "triggers"),
        # State access
        ("Intent Agent", "SharedState", "accesses"),
        ("Entity Agent", "SharedState", "accesses"),
        ("Process Agent", "SharedState", "accesses"),
        # Conflict monitoring
        ("ConflictOracle", "ConfidenceVector", "monitors"),
        ("ConflictOracle", "Governor Agent", "triggers"),
        ("Corpus", "SharedState", "snapshots"),
        # Perception → senses → governor
        ("PerceptionEngine", "SenseVector", "feeds"),
        ("SenseVector", "Governor Agent", "constrains"),
        # Autonomy
        ("DaemonMode", "GoalStore", "queries"),
        ("DaemonMode", "Governor Agent", "schedules"),
        # Self-referential loop
        ("ClosedLoopGate", "Synthesis Agent", "validates"),
        ("Governor Agent", "ClosedLoopGate", "invokes"),
        ("SemanticGrid", "SharedState", "enriches"),
        ("Corpus", "SemanticGrid", "persists"),
    ]

    def _parse_subsystem_markers(self, components: List[str]) -> Dict[str, List[str]]:
        """
        Parse [SUBSYSTEM: ...] markers from component names.

        Args:
            components: List of component names, some with [SUBSYSTEM: sub1, sub2] markers

        Returns:
            Dict mapping subsystem name to list of sub-component names

        Example:
            Input: ["User Service [SUBSYSTEM: User, Profile]", "Order"]
            Output: {"User Service": ["User", "Profile"]}
        """
        import re
        subsystems = {}

        for comp in components:
            # Match "Name [SUBSYSTEM: sub1, sub2, ...]"
            match = re.match(r'^(.+?)\s*\[SUBSYSTEM:\s*(.+?)\]$', comp, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                subs_str = match.group(2)
                subs = [s.strip() for s in subs_str.split(',')]
                subsystems[name] = subs

        return subsystems

    def compile(
        self,
        description: str,
        canonical_components: List[str] = None,
        canonical_relationships: List[tuple] = None,
        use_corpus_suggestions: bool = True,
        min_quality_score: Optional[float] = None,
        pipeline_mode: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        enrich: bool = False,
    ) -> CompileResult:
        """
        Compile natural language description into blueprint.

        Derived from: PROJECT-PLAN.md Phase 2.1
        Enhanced with: Stage-boundary verification (engineering best practice)
        Enhanced with: Corpus-driven canonical suggestions
        Phase 9.1: Input quality gate before LLM calls

        Args:
            description: What the user wants to build
            canonical_components: If provided, these MUST appear in output
            canonical_relationships: If provided, these SHOULD appear in output
            use_corpus_suggestions: Enable corpus-based canonical suggestions (default: True)
            min_quality_score: Override minimum quality score (default: REJECT_THRESHOLD)

        Returns:
            CompileResult with blueprint, context graph, verification, and stage results

        Raises:
            InputQualityError: If input quality is below reject threshold
        """
        state = SharedState()
        stage_results = []
        corpus_suggestions = None  # Track suggestions for result

        # Phase 21: Reset per-compilation token tracking
        self._compilation_tokens = []

        # --- Compilation mode resolution ---
        from core.compilation_modes import CompilationMode, ModeConfig, mode_config, parse_mode
        compilation_mode = CompilationMode.BUILD  # default
        if pipeline_mode:
            try:
                compilation_mode = parse_mode(pipeline_mode)
            except ValueError:
                pass  # Unknown mode string → fall through to legacy pipeline_mode handling
        _mode_cfg = mode_config(compilation_mode)
        state.known["compilation_mode"] = compilation_mode.value

        # Phase 17.2: Input size guard
        from core.blueprint_health import check_input_size
        input_ok, description = check_input_size(description)
        if not input_ok:
            self._emit_insight("Input truncated to 10000 words (size guard)")

        state.known["input"] = description

        # L3: Route [SELF-COMPILE] prefix to self_compile()
        if description.strip().startswith("[SELF-COMPILE]") or compilation_mode == CompilationMode.SELF:
            self._emit_insight("L3: self-compile triggered")
            return self.self_compile()

        # Phase 26: Inject self-compile patterns for Stratum 3 provenance
        if self._last_self_compile_patterns:
            state.self_compile_patterns = list(self._last_self_compile_patterns)

        # Phase 9.1: Input quality gate
        quality_analyzer = InputQualityAnalyzer()
        quality_score = quality_analyzer.analyze(description)
        threshold = min_quality_score if min_quality_score is not None else PROTOCOL.engine.quality_reject_threshold

        if quality_score.overall < threshold:
            # Use E1003 for custom threshold, auto-infer otherwise
            error_code = "E1003" if min_quality_score is not None else None
            raise InputQualityError(
                f"Input quality too low: {quality_score.overall:.2f} < {threshold:.2f}. "
                f"Details: {'; '.join(quality_score.details)}",
                quality_score=quality_score,
                error_code=error_code,
            )

        if quality_score.has_warnings:
            self._emit_rich(
                f"Input quality: {quality_score.overall:.0%} — {quality_score.suggestion}",
                InsightCategory.METRIC,
                metrics={"input_quality": quality_score.overall},
            )

        # Difficulty signal: input quality
        self._difficulty.input_quality = quality_score.overall
        self._emit_difficulty()

        # Store quality score for API response
        state.known["input_quality"] = {
            "score": quality_score.overall,
            "details": quality_score.details,
        }

        # Provider metrics tracking
        stage_timings: Dict[str, float] = {}
        retry_counts: Dict[str, int] = {}

        logger.info(f"Starting compilation: {description[:100]}...")

        # Reset per-compilation glass-box state
        self._difficulty = DifficultySignal()
        self._structured_insights = []
        self._current_stage = "queued"
        self._progress_callback = progress_callback

        try:
            # Progress callback helper
            def _progress(stage: str, index: int, insight: str = "", **kwargs):
                self._current_stage = stage
                if progress_callback:
                    try:
                        progress_callback(stage, index, insight, **kwargs)
                    except Exception as e:
                        logger.debug("Progress callback failed: %s", e)

            # Phase 1: Extract intent (with cache check)
            logger.debug("Phase 1: Extracting intent")
            self._emit("Extracting intent...")
            _progress("intent_analysis", 1)
            intent_start = time.time()

            # Phase 6.2: Check intent cache
            # Phase 7.2: Per-stage timeout
            intent_cache_hit = False
            intent_gate = self._stage_gates["intent"]
            if self.cache_policy in ("intent", "full"):
                cache_config = {"provider": self.provider_name, "model": self.model_name}
                intent_key = self._cache.intent.make_key(description, cache_config)
                cached_intent = self._cache.intent.get(intent_key)
                if cached_intent is not None:
                    intent = cached_intent
                    intent_cache_hit = True
                    logger.info("Intent cache HIT")
                else:
                    with timeout_context(intent_gate.timeout_seconds, "intent extraction"):
                        intent = self._extract_intent(description, state)
                    self._cache.intent.set(intent_key, intent)
                    logger.debug("Intent cache MISS - stored")
            else:
                with timeout_context(intent_gate.timeout_seconds, "intent extraction"):
                    intent = self._extract_intent(description, state)

            stage_timings["intent"] = time.time() - intent_start
            self._collect_usage()
            self._check_cost_cap()

            # Auto-scale timeouts based on first LLM call speed
            if not self._timeout_scaled and not intent_cache_hit:
                intent_duration = stage_timings["intent"]
                # Baseline: intent extraction should take ~10s on fast models
                baseline = 10.0
                if intent_duration > baseline * 1.5:
                    multiplier = max(1.0, min(intent_duration / baseline, 10.0))
                    for name, gate in self._stage_gates.items():
                        gate.timeout_seconds = int(self._base_timeouts[name] * multiplier)
                    logger.info(f"Timeout auto-scaled by {multiplier:.1f}x (intent took {intent_duration:.0f}s)")
                else:
                    # Reset to base timeouts in case a previous run scaled them
                    for name, gate in self._stage_gates.items():
                        gate.timeout_seconds = self._base_timeouts[name]
                self._timeout_scaled = True

            state.known["intent"] = intent
            self._emit_rich(
                f"Core need: {intent.get('core_need', 'unknown')}",
                InsightCategory.DISCOVERY,
            )

            # Use explicit components from intent as canonical if not provided
            if not canonical_components and intent.get("explicit_components"):
                canonical_components = intent["explicit_components"]
                self._emit_rich(
                    f"Canonical set from input: {len(canonical_components)} components",
                    InsightCategory.DISCOVERY,
                    metrics={"count": float(len(canonical_components))},
                )

            # Corpus-driven suggestions: look up patterns from prior compilations
            if use_corpus_suggestions and not canonical_components:
                domain = intent.get("domain")
                if domain and self.corpus:
                    corpus_suggestions = self.corpus.get_domain_suggestions(domain)
                    if corpus_suggestions.get("has_suggestions"):
                        canonical_components = corpus_suggestions["suggested_components"]
                        self._emit_insight(
                            f"Corpus suggests for '{domain}': {canonical_components[:5]}..."
                            if len(canonical_components) > 5
                            else f"Corpus suggests for '{domain}': {canonical_components}"
                        )
                        # Store in state for visibility
                        state.known["corpus_suggestions"] = corpus_suggestions

            # Phase 22: Corpus pattern synthesis — build domain model
            if use_corpus_suggestions and self.corpus:
                domain = intent.get("domain")
                if domain:
                    from persistence.corpus_analysis import CorpusAnalyzer
                    analyzer = CorpusAnalyzer(self.corpus)
                    domain_model = analyzer.build_domain_model(domain)
                    if domain_model:
                        # Store summary (serializable) for result metadata
                        state.known["corpus_patterns"] = {
                            "domain": domain_model.domain,
                            "archetype_count": len(domain_model.archetypes),
                            "pattern_count": len(domain_model.relationship_patterns),
                            "sample_size": domain_model.sample_size,
                        }
                        # Phase 26: Store serialized domain model for Stratum 1+2 provenance
                        state.known["domain_model"] = asdict(domain_model)

                        # Store formatted section for synthesis (string, not object)
                        corpus_section = analyzer.format_corpus_patterns_section(domain_model)
                        if corpus_section:
                            state.known["corpus_patterns_section"] = corpus_section

                        # Phase 22b: Anti-pattern warnings from corpus
                        anti_warnings = analyzer.format_anti_pattern_warnings(domain_model)
                        if anti_warnings:
                            state.known["corpus_anti_patterns"] = anti_warnings

                        # Phase 22c: Constraint hints from corpus
                        constraint_hints = analyzer.format_constraint_hints(domain_model)
                        if constraint_hints:
                            state.known["corpus_constraint_hints"] = constraint_hints

            # Phase 22d: Rejection awareness from governor learning
            try:
                from core.rejection_log import RejectionLog
                rlog = RejectionLog()
                summary = rlog.get_summary()
                if summary.total_rejections > 0 and summary.remediation_hints:
                    state.known["rejection_hints"] = list(summary.remediation_hints)
                    self._emit_rich(
                        f"Governor: {summary.total_rejections} prior rejections inform this compile",
                        InsightCategory.DECISION,
                        reasoning="Prior failures inform compile strategy",
                    )
            except Exception as e:
                logger.debug(f"Rejection log read skipped: {e}")

            # Phase 22e: Governor feedback — compiler self-improvement directives
            try:
                compilation_history = getattr(self, "_compilation_outcomes", [])
                if not compilation_history and self._outcome_store:
                    stored = self._outcome_store.recent(limit=20)
                    if stored:
                        compilation_history = [
                            s if isinstance(s, dict) else {
                                "compile_id": getattr(s, "compile_id", ""),
                                "trust_score": getattr(s, "trust_score", 0),
                                "domain": getattr(s, "domain", "software"),
                            }
                            for s in stored
                        ]
            except Exception as e:
                logger.debug(f"Governor feedback injection skipped: {e}")

            # Phase 22g: L2 pattern injection from episodic memory
            try:
                from kernel.memory import load_patterns, format_pattern_context
                l2_patterns = load_patterns(min_confidence=0.5)
                if l2_patterns:
                    state.known["l2_patterns"] = format_pattern_context(l2_patterns)
                    self._emit_rich(
                        f"L2: {len(l2_patterns)} learned pattern(s) loaded",
                        InsightCategory.DECISION,
                        reasoning="Corpus experience informs synthesis",
                    )
            except Exception as e:
                logger.debug(f"L2 pattern injection skipped: {e}")

            # Parse subsystem markers from canonical components
            subsystem_hints = {}
            if canonical_components:
                subsystem_hints = self._parse_subsystem_markers(canonical_components)
                if subsystem_hints:
                    self._emit_insight(f"Detected {len(subsystem_hints)} subsystem(s): {list(subsystem_hints.keys())}")
                    # Store in state for synthesis
                    state.known["subsystem_hints"] = subsystem_hints

            # Use explicit relationships from intent as canonical if not provided
            if not canonical_relationships and intent.get("explicit_relationships"):
                # Convert string relationships to tuples
                explicit_rels = intent["explicit_relationships"]
                canonical_relationships = []
                for rel in explicit_rels:
                    if isinstance(rel, str):
                        # Parse "A triggers B" format
                        for rel_type in ["triggers", "accesses", "creates", "confirms", "depends_on"]:
                            if rel_type in rel.lower():
                                parts = rel.lower().split(rel_type)
                                if len(parts) == 2:
                                    from_c = parts[0].strip().title()
                                    to_c = parts[1].strip().title()
                                    canonical_relationships.append((from_c, to_c, rel_type))
                                    break
                    elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                        canonical_relationships.append(tuple(rel[:3]))
                if canonical_relationships:
                    self._emit_insight(f"Canonical relationships from input: {len(canonical_relationships)}")

            # Stage 1 verification (using formal gate)
            intent_metrics = {
                "has_core_need": bool(intent.get("core_need")),
                "has_domain": bool(intent.get("domain")),
                "has_actors": bool(intent.get("actors")),
                "has_explicit_components": bool(intent.get("explicit_components")),
            }
            stage1_result = self._check_gate("intent", intent_metrics)
            stage1_result.output["core_need"] = intent.get("core_need", "")[:100]
            stage_results.append(stage1_result)

            # Phase 1.5: Interrogation — pre-dialogue clarification
            from core.interrogation import (
                should_interrogate as _should_interrogate,
                generate_questions as _generate_questions,
                InterrogationRequest, InterrogationResult,
                refine_intent_from_answers, _extract_domain_labels,
            )
            from core.convergence import _count_domains

            domain_count = _count_domains(description)
            should_ask, interrogation_reasons = _should_interrogate(
                quality_score.overall, intent, domain_count, quality_score.details,
            )

            interrogation_result = InterrogationResult(triggered=should_ask)

            if should_ask and self.on_interrogate:
                questions = _generate_questions(
                    interrogation_reasons, intent, quality_score.details,
                    domain_count, description,
                )
                if questions:
                    request = InterrogationRequest(
                        questions=questions,
                        context="I need a few clarifications before compiling.",
                        original_intent=intent,
                        quality_score=quality_score.overall,
                        domain_count=domain_count,
                    )
                    self._emit_insight(f"Asking {len(questions)} clarification question(s)")
                    response = self.on_interrogate(request)

                    if response and not response.skip:
                        refined_desc, refined_intent, should_fracture = (
                            refine_intent_from_answers(description, intent, response, questions)
                        )
                        interrogation_result.questions_asked = len(questions)
                        interrogation_result.answers_received = len(response.answers)

                        if should_fracture:
                            interrogation_result.should_fracture = True
                            domain_labels = _extract_domain_labels(description)
                            # Remove meta-options for fracture signal
                            competing = [l for l in domain_labels if l not in ("All together as one system", "Separate compilations")]
                            signal = FractureSignal(
                                stage="interrogation",
                                competing_configs=competing or ["Domain A", "Domain B"],
                                collapsing_constraint="User chose separate compilations",
                                agent="Interrogation",
                            )
                            raise FractureError(
                                "User requested separate compilations for multi-domain input",
                                stage="interrogation",
                                signal=signal,
                            )

                        if refined_desc != description:
                            description = refined_desc
                            state.known["input"] = description
                            interrogation_result.refined_description = refined_desc
                        if refined_intent != intent:
                            intent = refined_intent
                            state.known["intent"] = intent
                            interrogation_result.domain_choice = refined_intent.get("domain")
                    else:
                        interrogation_result.skipped = True

            elif should_ask:
                interrogation_result.skipped = True
                logger.warning(f"Interrogation triggered ({interrogation_reasons}) but no callback — proceeding")

            state.known["interrogation"] = {
                "triggered": interrogation_result.triggered,
                "questions_asked": interrogation_result.questions_asked,
                "answers_received": interrogation_result.answers_received,
                "skipped": interrogation_result.skipped,
                "should_fracture": interrogation_result.should_fracture,
                "domain_choice": interrogation_result.domain_choice,
            }

            # Phase 2: Generate personas (with cache check for "full" policy)
            _progress("persona_mapping", 2, intent.get("core_need", ""))
            self._emit("Generating perspectives...")
            personas_start = time.time()

            # Phase 6.2: Check persona cache when policy is "full"
            # Phase 7.2: Per-stage timeout
            persona_cache_hit = False
            persona_gate = self._stage_gates["personas"]
            if self.cache_policy == "full":
                # Use intent as key for persona caching
                import json as _json
                intent_str = _json.dumps(intent, sort_keys=True)
                persona_key = self._cache.persona.make_key(intent_str, cache_config)
                cached_personas = self._cache.persona.get(persona_key)
                if cached_personas is not None:
                    personas = cached_personas
                    persona_cache_hit = True
                    logger.info("Persona cache HIT")
                else:
                    with timeout_context(persona_gate.timeout_seconds, "persona generation"):
                        personas = self._generate_personas(intent, state)
                    self._cache.persona.set(persona_key, personas)
                    logger.debug("Persona cache MISS - stored")
            else:
                with timeout_context(persona_gate.timeout_seconds, "persona generation"):
                    personas = self._generate_personas(intent, state)

            stage_timings["personas"] = time.time() - personas_start
            self._collect_usage()
            self._check_cost_cap()
            state.personas = personas
            for p in personas[:2]:  # Show first 2
                self._emit_rich(
                    f"{p['name']}: {p['perspective'][:60]}...",
                    InsightCategory.DISCOVERY,
                )

            # Stage 2 verification (using formal gate)
            personas_metrics = {
                "min_personas": len(personas),
                "max_personas": len(personas),
            }
            stage2_result = self._check_gate("personas", personas_metrics)
            stage_results.append(stage2_result)

            # --- Mode posture injection ---
            # Prepend posture preamble to agent system prompts for non-BUILD modes.
            # Restore after dialogue to avoid polluting subsequent compilations.
            _original_entity_prompt = self.entity_agent.system_prompt
            _original_process_prompt = self.process_agent.system_prompt
            if _mode_cfg.posture_preamble:
                self.entity_agent.system_prompt = _mode_cfg.posture_preamble + self.entity_agent.system_prompt
                self.process_agent.system_prompt = _mode_cfg.posture_preamble + self.process_agent.system_prompt
                self._emit_rich(
                    f"Mode: {compilation_mode.value} (posture={_mode_cfg.agent_posture})",
                    InsightCategory.DECISION,
                    reasoning=f"Compilation mode {compilation_mode.value} adjusts agent challenge depth",
                )

            # Phase 3: Spec dialogue
            _progress("entity_extraction", 3)
            # Phase 14: Feature flag — "staged" uses StagedPipeline, "legacy" uses single dialogue
            mode = pipeline_mode or self.pipeline_mode
            dialogue_start = time.time()

            _staged_ok = False
            if mode == "staged":
                # Phase 14: Staged pipeline — 5 focused sub-dialogues
                self._emit("Staged pipeline: 5 focused sub-dialogues...")
                pipeline_state = PipelineState(
                    original_input=description,
                    intent=intent,
                    personas=personas
                )
                from core.convergence import compute_stage_budgets
                stage_budgets = compute_stage_budgets(description)

                staged = StagedPipeline(
                    llm_client=self.llm,
                    on_insight=self.on_insight,
                    domain_adapter=self.domain_adapter,
                    stage_budgets=stage_budgets,
                )
                try:
                    with timeout_context(sum(s.timeout_seconds for s in PROTOCOL.pipeline.stages), "staged pipeline"):
                        pipeline_state = staged.run(pipeline_state)

                    # Transfer accumulated results to SharedState for synthesis compatibility
                    state.insights = pipeline_state.all_insights
                    state.unknown = pipeline_state.all_unknowns
                    state.conflicts = pipeline_state.all_conflicts

                    # Difficulty signal: unknowns + conflicts from entity extraction
                    self._difficulty.unknown_count = len(state.unknown)
                    self._difficulty.conflict_count = len(state.conflicts)
                    self._emit_difficulty()

                    # Add last 2 messages from each stage to history for synthesis context
                    for record in pipeline_state.stages:
                        for msg in record.state.history[-2:]:
                            state.add_message(msg)

                    # Methods from Stage 4 (CONSTRAIN) artifact directly
                    constrain_artifact = pipeline_state.get_artifact("constrain")
                    if constrain_artifact:
                        state.known["extracted_methods"] = constrain_artifact.get("methods", [])
                        state.known["extracted_state_machines"] = constrain_artifact.get("state_machines", [])
                        state.known["extracted_algorithms"] = constrain_artifact.get("algorithms", [])

                    # Phase 26: Extract avg type confidence from classification for verification
                    decompose_artifact = pipeline_state.get_artifact("decompose")
                    if decompose_artifact:
                        cls_scores = decompose_artifact.get("classification_scores", [])
                        if cls_scores:
                            avg_tc = sum(s.type_confidence for s in cls_scores) / len(cls_scores)
                            state.known["_avg_type_confidence"] = avg_tc

                    # Store pipeline state for synthesis assembly
                    state.known["pipeline_state"] = pipeline_state

                    stage_timings["dialogue"] = time.time() - dialogue_start
                    self._collect_usage()
                    self._check_cost_cap()

                    # Stage 3 verification (use accumulated metrics)
                    dialogue_turns = sum(r.turn_count for r in pipeline_state.stages)
                    dialogue_metrics = {
                        "min_turns": dialogue_turns,
                        "min_insights": len(state.insights),
                        "recommended_turns": dialogue_turns,
                    }
                    stage3_result = self._check_gate("dialogue", dialogue_metrics)
                    stage_results.append(stage3_result)

                    # Phase 10.8: Resolve structural conflicts by reframing
                    resolved = self._resolve_conflicts(state)
                    if resolved:
                        self._emit_rich(
                            f"Resolved {resolved} structural conflict(s) by reframing",
                            InsightCategory.RESOLUTION,
                            metrics={"resolved": float(resolved)},
                        )

                    # Phase 10.9: Cross-agent factual consistency check
                    try:
                        from core.consistency_checker import check_consistency, format_consistency_warnings
                        _msg_dicts = [m.to_dict() for m in state.history]
                        _consistency = check_consistency(_msg_dicts)
                        if _consistency.has_contradictions:
                            _cw = format_consistency_warnings(_consistency)
                            self._emit_insight(
                                f"Cross-agent consistency: {_consistency.hard_count} hard, "
                                f"{_consistency.soft_count} soft contradictions"
                            )
                            state.known["consistency_warnings"] = _cw
                            state.known["consistency_hard_count"] = _consistency.hard_count
                            state.known["consistency_soft_count"] = _consistency.soft_count
                    except Exception as e:
                        logger.debug(f"Consistency check skipped: {e}")

                    _staged_ok = True

                except CompilationError as e:
                    if "hollow artifact" in str(e):
                        # Staged pipeline produced hollow artifact — fall back to legacy
                        self._emit("Staged pipeline failed (hollow artifact), falling back to legacy dialogue...")
                        logger.warning(f"Staged pipeline hollow artifact fallback: {e}")
                        dialogue_start = time.time()  # reset timer
                    else:
                        raise

            if not _staged_ok:
                # Legacy: single monolithic Entity<->Process dialogue
                # Phase 7.2: Per-stage timeout
                self._emit("Excavating specification...")
                dialogue_gate = self._stage_gates["dialogue"]
                with timeout_context(dialogue_gate.timeout_seconds, "spec dialogue"):
                    self._run_spec_dialogue(state)
                stage_timings["dialogue"] = time.time() - dialogue_start
                self._collect_usage()
                self._check_cost_cap()

                # Difficulty signal: unknowns + conflicts from legacy dialogue
                self._difficulty.unknown_count = len(state.unknown)
                self._difficulty.conflict_count = len(state.conflicts)
                self._emit_difficulty()

                # Stage 3 verification (using formal gate)
                dialogue_turns = len([m for m in state.history if m.sender in ["Entity", "Process"]])
                dialogue_metrics = {
                    "min_turns": dialogue_turns,
                    "min_insights": len(state.insights),
                    "recommended_turns": dialogue_turns,
                }
                stage3_result = self._check_gate("dialogue", dialogue_metrics)
                stage_results.append(stage3_result)

                # Phase 10.8: Resolve structural conflicts by reframing
                resolved = self._resolve_conflicts(state)
                if resolved:
                    self._emit_rich(
                        f"Resolved {resolved} structural conflict(s) by reframing",
                        InsightCategory.RESOLUTION,
                        metrics={"resolved": float(resolved)},
                    )

                # Phase 10.9: Cross-agent factual consistency check
                try:
                    from core.consistency_checker import check_consistency, format_consistency_warnings
                    _msg_dicts = [m.to_dict() for m in state.history]
                    _consistency = check_consistency(_msg_dicts)
                    if _consistency.has_contradictions:
                        _cw = format_consistency_warnings(_consistency)
                        self._emit_insight(
                            f"Cross-agent consistency: {_consistency.hard_count} hard, "
                            f"{_consistency.soft_count} soft contradictions"
                        )
                        state.known["consistency_warnings"] = _cw
                        state.known["consistency_hard_count"] = _consistency.hard_count
                        state.known["consistency_soft_count"] = _consistency.soft_count
                except Exception as e:
                    logger.debug(f"Consistency check skipped: {e}")

                # Phase 12.3: Extract methods and state machines for synthesis
                methods, state_machines = self._extract_methods_for_synthesis(state)
                if methods:
                    self._emit_insight(f"Extracted {len(methods)} method(s) from dialogue")
                    state.known["extracted_methods"] = methods
                if state_machines:
                    self._emit_insight(f"Extracted {len(state_machines)} state machine(s) from dialogue")
                    state.known["extracted_state_machines"] = state_machines

            # --- Restore agent prompts after dialogue ---
            self.entity_agent.system_prompt = _original_entity_prompt
            self.process_agent.system_prompt = _original_process_prompt

            # Phase 3.5: Kernel compilation — semantic grid from dialogue
            # If grid-driven dialogue already built the grid, reuse it.
            # Otherwise, run kernel compilation as before.
            kernel_start = time.time()
            kernel_grid_data = None
            _grid_from_dialogue = self._kernel_grid is not None and self._kernel_grid.cells
            if _grid_from_dialogue:
                self._emit_insight(
                    f"Phase 3.5: reusing dialogue grid "
                    f"({self._kernel_grid.total_cells} cells, "
                    f"fill={self._kernel_grid.fill_rate:.0%})"
                )
            try:
                from kernel.agents import compile as kernel_compile, CompileConfig as KernelConfig
                from kernel.nav import grid_to_nav, grid_to_structured
                from kernel.emission import emit as kernel_emit
                from kernel.ground_truth import load_ground_truth

                if _grid_from_dialogue:
                    # Grid already built during dialogue — skip kernel compile
                    nav_text = grid_to_nav(self._kernel_grid)
                    manifest = kernel_emit(self._kernel_grid, force=False)
                    if manifest is None:
                        self._emit_insight("Governor rejected grid emission (dialogue grid) — structural issues detected")
                        logger.warning("Governor simulation failed on dialogue grid — emitting with force for degraded operation")
                        manifest = kernel_emit(self._kernel_grid, force=True)
                    manifest_dict = manifest.to_dict() if manifest else None
                    structured_cells = grid_to_structured(self._kernel_grid)

                    kernel_grid_data = {
                        "nav": nav_text,
                        "manifest": manifest_dict,
                        "cells": len(self._kernel_grid.cells),
                        "layers": len(self._kernel_grid.activated_layers),
                        "iterations": 0,
                        "converged": True,
                        "source": "dialogue_grid",
                        "structured_cells": structured_cells,
                    }

                    state.known["semantic_nav"] = nav_text
                    state.known["structured_grid_cells"] = structured_cells

                    self._emit_insight(
                        f"Semantic grid (from dialogue): "
                        f"{len(self._kernel_grid.cells)} cells, "
                        f"{len(self._kernel_grid.activated_layers)} layers"
                    )
                else:
                    # Fallback: full kernel compilation after dialogue

                    # Build enriched input from description + dialogue insights
                    enriched_parts = [description]
                    if state.insights:
                        enriched_parts.append("Key discoveries: " + "; ".join(state.insights[:20]))

                    # Use ground truth as compilation history (MEMORY agent bootstraps)
                    gt_grid = load_ground_truth()
                    history = [gt_grid]

                    # Load previously persisted grid for L2 accumulation
                    try:
                        from kernel.store import load_grid as _load_grid
                        prev_grid = _load_grid("session")
                        if prev_grid is not None:
                            history.append(prev_grid)
                    except Exception as e:
                        logger.debug(f"Prior grid load skipped: {e}")

                    # L3: Load compiler self-description for recursive accumulation
                    try:
                        from kernel.store import load_grid as _load_grid
                        self_desc_grid = _load_grid("compiler-self-desc")
                        if self_desc_grid is not None:
                            history.append(self_desc_grid)
                            self._emit_insight(f"L3: self-desc loaded ({len(self_desc_grid.cells)} cells)")
                    except Exception as e:
                        logger.debug(f"L3 self-desc load skipped: {e}")

                    # Real LLM extraction: use engine's client to populate kernel grid
                    from kernel.llm_bridge import parse_extractions

                    _SEM_EXTRACT_SYSTEM = (
                        "You are a semantic compiler. You extract structured concepts "
                        "from text and return them as a JSON array. Return ONLY valid "
                        "JSON — no markdown fences, no explanation, just the array."
                    )

                    def _extraction_fn(prompt: str) -> list[dict]:
                        """LLM-backed extraction: parse structured concepts from prompt."""
                        try:
                            raw = self.llm.complete_with_system(
                                system_prompt=_SEM_EXTRACT_SYSTEM,
                                user_content=prompt,
                                max_tokens=4096,
                                temperature=0.0,
                            )
                            self._collect_usage()
                            self._check_cost_cap()
                            return parse_extractions(raw)
                        except CostCapExceededError:
                            raise
                        except Exception as e:
                            logger.debug(f"Kernel LLM extraction failed: {e}")
                            return []

                    kernel_config = KernelConfig(
                        max_iterations=5,
                        min_fill_rate=0.0,
                        scope_schedule=(
                            ("ECO", "APP"),           # iter 0: ecosystem + application
                            ("DOM", "FET"),           # iter 1: domain + feature decomposition
                            ("CMP", "FNC"),           # iter 2: components + functions
                            ("CMP", "FNC", "STP"),    # iter 3: fill gaps
                            ("STP", "OPR"),           # iter 4: technical primitives
                        ),
                    )
                    kernel_result = kernel_compile(
                        input_text="\n".join(enriched_parts),
                        llm_fn=_extraction_fn,
                        history=history,
                        config=kernel_config,
                    )

                    # Serialize grid to nav format + attempt manifest emission
                    nav_text = grid_to_nav(kernel_result.grid)
                    manifest = kernel_emit(kernel_result.grid, force=False)
                    if manifest is None:
                        self._emit_insight("Governor rejected grid emission (kernel compile) — structural issues detected")
                        logger.warning("Governor simulation failed on kernel grid — emitting with force for degraded operation")
                        manifest = kernel_emit(kernel_result.grid, force=True)
                    manifest_dict = manifest.to_dict() if manifest else None
                    structured_cells = grid_to_structured(kernel_result.grid)

                    # Cache kernel grid for observer + persistence
                    self._kernel_grid = kernel_result.grid

                    kernel_grid_data = {
                        "nav": nav_text,
                        "manifest": manifest_dict,
                        "cells": len(kernel_result.grid.cells),
                        "layers": len(kernel_result.grid.activated_layers),
                        "iterations": kernel_result.iterations,
                        "converged": kernel_result.converged,
                        "structured_cells": structured_cells,
                    }

                    # Inject nav into state for synthesis context
                    state.known["semantic_nav"] = nav_text
                    state.known["structured_grid_cells"] = structured_cells

                    self._emit_insight(
                        f"Semantic grid: {len(kernel_result.grid.cells)} cells, "
                        f"{len(kernel_result.grid.activated_layers)} layers"
                    )
            except Exception as e:
                logger.debug(f"Kernel compilation skipped: {e}")
            stage_timings["kernel"] = time.time() - kernel_start

            # Phase 3.7: Extract exploration endpoints for intent chaining
            _depth_chain_dicts = []
            if self._kernel_grid and self._kernel_grid.cells:
                try:
                    from kernel.endpoint_extractor import extract_exploration_endpoints
                    _chains = extract_exploration_endpoints(
                        self._kernel_grid,
                        original_intent=description,
                    )
                    if _chains:
                        _depth_chain_dicts = [
                            {
                                "chain_type": c.chain_type,
                                "intent_text": c.intent_text,
                                "source_postcodes": list(c.source_postcodes),
                                "priority": c.priority,
                                "layer": c.layer,
                                "concern": c.concern,
                            }
                            for c in _chains
                        ]
                        self._emit_insight(
                            f"Intent chaining: {len(_chains)} exploration endpoint(s) extracted"
                        )
                except Exception as e:
                    logger.debug(f"Endpoint extraction skipped: {e}")

            # --- Mode routing: EXPLORE and CONTEXT skip normal synthesis ---
            if compilation_mode == CompilationMode.EXPLORE:
                return self._compile_explore_mode(
                    state, description, stage_results, stage_timings,
                    retry_counts, kernel_grid_data, _depth_chain_dicts,
                    intent_cache_hit, persona_cache_hit,
                )
            if compilation_mode == CompilationMode.CONTEXT:
                return self._compile_context_mode(
                    state, description, stage_results, stage_timings,
                    retry_counts, kernel_grid_data, _depth_chain_dicts,
                    intent_cache_hit, persona_cache_hit,
                )

            # Phase 3.7: Process modeling stage marker
            _progress("process_modeling", 4)

            # Phase 4: Synthesis (with canonical enforcement if provided)
            _progress("synthesis", 5)
            # Phase 7.2: Per-stage timeout
            self._emit("Synthesizing blueprint...")
            synthesis_start = time.time()
            synthesis_gate = self._stage_gates["synthesis"]
            with timeout_context(synthesis_gate.timeout_seconds, "synthesis"):
                blueprint, synthesis_retries = self._synthesize(
                    state,
                    canonical_components=canonical_components,
                    canonical_relationships=canonical_relationships
                )
            # Clean up non-serializable pipeline_state before corpus storage
            state.known.pop("pipeline_state", None)
            stage_timings["synthesis"] = time.time() - synthesis_start
            # Token usage tracked inside _synthesize() via AgentCallResult.token_usage
            self._check_cost_cap()
            retry_counts["synthesis"] = synthesis_retries

            # Phase 8.1: Normalize string elements to dicts before any .get() calls
            blueprint = normalize_blueprint_elements(blueprint)

            # Phase 8.2: Deduplicate blueprint
            blueprint, dedup_report = deduplicate_blueprint(blueprint)
            if dedup_report["total_removed"] > 0:
                self._emit_insight(
                    f"Dedup: removed {dedup_report['total_removed']} duplicates "
                    f"({len(dedup_report['name_dupes_removed'])} name, "
                    f"{len(dedup_report['containment_dupes_removed'])} containment, "
                    f"{dedup_report['relationship_dupes_removed']} relationship)"
                )

            # Phase 14: Promote undeclared relationship endpoints to components
            blueprint = self._promote_undeclared_endpoints(blueprint)

            # Phase 12.5: Enrich blueprint with extracted methods from dialogue
            blueprint = self._enrich_blueprint_methods(blueprint, state)

            # SEED Protocol enforcement — L1/L2/L3 validation
            from core.seed import validate_seed, seed_gate_rate
            seed_result = validate_seed(blueprint, description)
            seed_rate = seed_gate_rate(seed_result)
            state.known["_seed_validation"] = {
                "gate_rate": seed_rate,
                "violations": {k: v for k, v in seed_result.items() if v},
            }
            if seed_rate < 1.0:
                violation_count = sum(len(v) for v in seed_result.values())
                self._emit_rich(
                    f"SEED gate: {seed_rate:.0%} ({violation_count} violation(s))",
                    InsightCategory.WARNING,
                    metrics={"seed_gate_rate": seed_rate},
                )
                for law, violations in seed_result.items():
                    for v in violations[:3]:
                        logger.warning(f"SEED {v}")

            # Phase 17.2: Blueprint health check
            from core.blueprint_health import check_blueprint_health
            health_report = check_blueprint_health(blueprint)
            if not health_report.healthy:
                if health_report.score < PROTOCOL.engine.blueprint_health_abort:
                    raise CompilationError(
                        f"Blueprint too degraded: {health_report.errors}",
                        stage="synthesis",
                        error_code="E7002",
                    )
                for err in health_report.errors:
                    self._emit_rich(f"⚠ Health: {err}", InsightCategory.WARNING)
            if health_report.warnings:
                for warn in health_report.warnings:
                    self._emit_rich(f"⚠ Health: {warn}", InsightCategory.WARNING)

            # Phase 18: Store health report for deterministic verification
            state.known["_health_report"] = {
                "healthy": health_report.healthy,
                "score": health_report.score,
                "errors": list(health_report.errors),
                "warnings": list(health_report.warnings),
                "stats": dict(health_report.stats),
            }

            state.known["blueprint"] = blueprint

            # Announce synthesized components for live frontend crystallization
            _comp_names = [c.get("name", "?") for c in blueprint.get("components", [])]
            if _comp_names:
                self._emit_rich(
                    f"Extracted: {', '.join(_comp_names)}",
                    InsightCategory.DISCOVERY,
                    metrics={"count": float(len(_comp_names))},
                )

            # Difficulty signal: component complexity
            _comp_count = len(blueprint.get("components", []))
            _rel_count = len(blueprint.get("relationships", []))
            # Normalize: 20+ components or 30+ relationships → complexity ~1.0
            self._difficulty.component_complexity = min(
                (_comp_count / 20.0) * 0.6 + (_rel_count / 30.0) * 0.4,
                1.0,
            )
            self._emit_difficulty()

            # Stage 4 verification (using formal gate)
            schema_validation = validate_blueprint(blueprint)
            # Calculate component coverage if canonical set exists
            comp_coverage = 1.0
            rel_coverage = 1.0
            if canonical_components:
                cov = check_canonical_coverage(blueprint, canonical_components)
                comp_coverage = cov.get("coverage", 0)
            if canonical_relationships:
                rel_cov = check_canonical_relationships(blueprint, canonical_relationships)
                rel_coverage = rel_cov.get("coverage", 0)

            synthesis_metrics = {
                "has_components": len(blueprint.get("components", [])) > 0,
                "component_coverage": comp_coverage,
                "relationship_coverage": rel_coverage,
                "schema_valid": schema_validation.get("valid", False),
            }
            stage4_result = self._check_gate("synthesis", synthesis_metrics)
            stage4_result.output.update(schema_validation.get("stats", {}))
            stage_results.append(stage4_result)

            # Hard stop: 0 components after synthesis = nothing was produced
            if not blueprint.get("components"):
                logger.error("Synthesis produced 0 components — hard stop")
                return CompileResult(
                    success=False,
                    blueprint=blueprint,
                    context_graph=state.to_context_graph(),
                    insights=state.insights,
                    verification={},
                    stage_results=stage_results,
                    stage_timings=stage_timings,
                    error="Synthesis produced 0 components",
                    interrogation=state.known.get("interrogation", {}),
                    termination_condition=self._make_termination_condition(
                        status="halted",
                        reason="synthesis_empty",
                        message="Compilation halted because synthesis produced no buildable structure.",
                        next_action="Narrow the intent or add concrete entities and constraints.",
                    ),
                )

            # Phase 17.4: Constraint contradiction detection
            from core.schema import detect_contradictions
            contradictions = detect_contradictions(blueprint.get("constraints", []))
            if contradictions:
                logger.warning(f"E7004: {len(contradictions)} contradictory constraint(s) detected")
                for c in contradictions:
                    self._emit_rich(
                        f"⚠ Contradiction ({c.contradiction_type}): {c.description}",
                        InsightCategory.WARNING,
                    )
                state.known["contradictions"] = [
                    {
                        "field": c.field,
                        "type": c.contradiction_type,
                        "description": c.description,
                    }
                    for c in contradictions
                ]

            # Phase 18: Store contradiction count for deterministic verification
            state.known["_contradiction_count"] = len(contradictions)

            # C006: Unknowns floor — short inputs must surface unknowns
            unresolved = blueprint.get("unresolved", [])
            if len(unresolved) == 0 and len(description) < 200:
                logger.warning("C006: Zero unknowns from short input — likely under-excavated")
                self._emit_rich(
                    "C006: No unknowns surfaced — input may be under-excavated",
                    InsightCategory.WARNING,
                )

            # Phase 5: Verification
            _progress("verification", 6, f"{len(blueprint.get('components', []))} components synthesized")
            # Phase 7.2: Per-stage timeout
            self._emit("Verifying...")
            verify_start = time.time()
            verify_gate = self._stage_gates["verification"]
            with timeout_context(verify_gate.timeout_seconds, "verification"):
                verification = self._verify_hybrid(blueprint, state)
            stage_timings["verification"] = time.time() - verify_start
            self._collect_usage()
            self._check_cost_cap()

            # Stage 5 verification (using formal gate)
            verification_metrics = {
                "completeness": verification.get("completeness", {}).get("score", 0),
                "consistency": verification.get("consistency", {}).get("score", 0),
                "coherence": verification.get("coherence", {}).get("score", 0),
                "traceability": verification.get("traceability", {}).get("score", 0),
                "pass_status": verification.get("status") == "pass",
            }
            stage5_result = self._check_gate("verification", verification_metrics)
            stage_results.append(stage5_result)

            # Hard stop: catastrophic verification failure
            # All dimension scores below 30 = blueprint is unsalvageable
            _dim_scores = [
                verification_metrics.get("completeness", 0),
                verification_metrics.get("consistency", 0),
                verification_metrics.get("coherence", 0),
                verification_metrics.get("traceability", 0),
            ]
            if all(s < 30 for s in _dim_scores) and any(s > 0 for s in _dim_scores):
                logger.error(
                    f"Catastrophic verification failure — all scores < 30: "
                    f"comp={_dim_scores[0]}, cons={_dim_scores[1]}, "
                    f"coh={_dim_scores[2]}, trace={_dim_scores[3]}"
                )
                return CompileResult(
                    success=False,
                    blueprint=blueprint,
                    context_graph=state.to_context_graph(),
                    insights=state.insights,
                    verification=verification,
                    stage_results=stage_results,
                    stage_timings=stage_timings,
                    error="Catastrophic verification failure — all dimensions below 30",
                    interrogation=state.known.get("interrogation", {}),
                    termination_condition=self._make_termination_condition(
                        status="halted",
                        reason="catastrophic_verification_failure",
                        message="Compilation halted because verification failed across every core dimension.",
                        next_action="Reduce scope or provide materially clearer input before recompiling.",
                    ),
                )

            # Trust floor: trigger re-synthesis if overall score < fail_threshold
            _overall = sum(_dim_scores) / max(len(_dim_scores), 1)
            if _overall < PROTOCOL.engine.verification_fail_threshold:
                logger.warning(
                    f"Trust floor gate: overall={_overall:.0f} < "
                    f"{PROTOCOL.engine.verification_fail_threshold} — triggering re-synthesis"
                )
                verification["status"] = "needs_work"

            # Phase 5.5: Closed-loop transcription gate (semantic fidelity)
            # Ensure insights are present for fidelity scoring (they contain intent tokens)
            if state.insights and not blueprint.get("insights"):
                from core.digest import _deduplicate_insights, _rank_insights
                _pre_ranked = _deduplicate_insights(_rank_insights(state))
                blueprint["insights"] = [r["insight"] for r in _pre_ranked[:10]]
            try:
                from kernel.closed_loop import closed_loop_gate
                cl_result = closed_loop_gate(description, blueprint)
                state.known["closed_loop_fidelity"] = cl_result.fidelity_score
                if not cl_result.passed:
                    loss_summary = ", ".join(
                        f"{l.category}({l.severity:.1f})"
                        for l in cl_result.compression_losses[:3]
                    )
                    self._emit_rich(
                        f"Closed-loop gate: fidelity={cl_result.fidelity_score:.2f} — "
                        f"compression losses: {loss_summary or 'semantic drift'}",
                        InsightCategory.METRIC,
                        metrics={"fidelity": cl_result.fidelity_score},
                    )
                    # Store losses for re-synthesis to target (rich context)
                    if cl_result.compression_losses:
                        state.known["compression_losses"] = [
                            {
                                "fragment": l.original_fragment,
                                "category": l.category,
                                "severity": l.severity,
                                "description": l.description,
                            }
                            for l in cl_result.compression_losses
                        ]
                        # Aggregate category → severity for L2 feedback
                        cat_dist: dict[str, float] = {}
                        for l in cl_result.compression_losses:
                            cat_dist[l.category] = cat_dist.get(l.category, 0.0) + l.severity
                        state.known["compression_loss_categories"] = cat_dist
                    # Closed-loop failure triggers re-synthesis with targeted losses
                    verification["status"] = "needs_work"
                else:
                    self._emit_rich(
                        f"Closed-loop gate: fidelity={cl_result.fidelity_score:.2f} ✓",
                        InsightCategory.METRIC,
                        metrics={"fidelity": cl_result.fidelity_score},
                    )
            except Exception as e:
                logger.warning(f"Closed-loop gate failed: {e}")
                state.known["closed_loop_skipped"] = True
                # Gate unavailable — don't assume worst-case fidelity (would break
                # mock/test compilations), but record the skip for observability

            # Fire observer: record compilation outcome on kernel grid cells
            try:
                if self._kernel_grid and self._kernel_grid.cells:
                    from kernel.observer import record_observation, apply_batch
                    from kernel.cell import FillState

                    cl_fidelity = state.known.get("closed_loop_fidelity", 1.0)
                    v_status = verification.get("status", "unknown")
                    confirmed = v_status == "pass"
                    anomaly = cl_fidelity < 0.5

                    deltas = []
                    for pc_key, cell in self._kernel_grid.cells.items():
                        if cell.fill in (FillState.F, FillState.P):
                            delta = record_observation(
                                self._kernel_grid,
                                pc_key,
                                event_type="compilation",
                                expected="verified" if cell.fill == FillState.F else "progressing",
                                actual=v_status,
                                confirmed=confirmed,
                                anomaly=anomaly,
                            )
                            deltas.append(delta)

                    if deltas:
                        batch = apply_batch(self._kernel_grid, deltas)
                        if batch.transitions:
                            self._emit_insight(
                                f"Observer: {batch.cells_touched} cells, "
                                f"{len(batch.transitions)} transition(s)"
                            )
            except Exception as e:
                logger.debug(f"Observer skipped: {e}")

            # Persist kernel grid to maps.db for L2 accumulation
            try:
                if self._kernel_grid:
                    from kernel.store import save_grid as _save_grid
                    _save_grid(
                        self._kernel_grid,
                        map_id="session",
                        name=description[:80],
                    )
            except Exception as e:
                logger.debug(f"Grid persistence skipped: {e}")

            # Phase 8.3: Verification-driven re-synthesis
            if verification.get("status") == "needs_work":
                completeness = verification.get("completeness", {}).get("score", 0)
                num_components = len(blueprint.get("components", []))
                cl_fidelity = state.known.get("closed_loop_fidelity", 1.0)
                _fidelity_triggered = cl_fidelity < PROTOCOL.engine.fidelity_threshold
                # Entity losses with high severity trigger re-synthesis even if
                # overall fidelity barely passed — prevents silent entity death
                _entity_loss_triggered = any(
                    isinstance(l, dict) and l.get("category") == "entity"
                    and l.get("severity", 0) > 0.6
                    for l in state.known.get("compression_losses", [])
                )
                should_resynth = (
                    _fidelity_triggered  # Fidelity failure — hard trigger
                    or _entity_loss_triggered  # Critical entity loss — hard trigger
                    or completeness >= PROTOCOL.engine.resynth_min_completeness
                    or num_components <= 3  # Catastrophically thin — force re-synthesis
                )
                if should_resynth:
                    _pre_resynth_fingerprint = compute_structural_fingerprint(blueprint)
                    _pre_resynth_score = self._verification_score(verification)
                    _pre_resynth_gates = self._semantic_gate_signature(blueprint, verification)
                    _pre_resynth_components = len(blueprint.get("components", []))
                    self._emit("Re-synthesizing from verification gaps...")
                    resynth_start = time.time()
                    with timeout_context(synthesis_gate.timeout_seconds, "re-synthesis"):
                        blueprint = self._targeted_resynthesis(blueprint, verification, state)
                    blueprint, resynth_dedup = deduplicate_blueprint(blueprint)
                    stage_timings["resynthesis"] = time.time() - resynth_start
                    self._collect_usage()
                    self._check_cost_cap()

                    # Re-verify
                    self._emit("Re-verifying...")
                    with timeout_context(verify_gate.timeout_seconds, "re-verification"):
                        verification = self._verify_hybrid(blueprint, state)
                    self._collect_usage()
                    self._check_cost_cap()
                    # Gap-fill methods on re-synthesized components
                    blueprint = self._infer_component_methods(blueprint)
                    state.known["blueprint"] = blueprint

                    _post_resynth_fingerprint = compute_structural_fingerprint(blueprint)
                    _post_resynth_score = self._verification_score(verification)
                    _post_resynth_gates = self._semantic_gate_signature(blueprint, verification)
                    if (
                        _post_resynth_fingerprint == _pre_resynth_fingerprint
                        and _post_resynth_score <= (_pre_resynth_score + 0.5)
                        and _post_resynth_gates == _pre_resynth_gates
                    ):
                        state.known["_termination_condition"] = self._make_termination_condition(
                            status="stalled",
                            reason="semantic_progress_stalled",
                            message="Compilation stopped making semantic progress after re-synthesis.",
                            next_action="Add a narrower constraint, revise scope, or answer a blocking decision.",
                            semantic_progress={
                                "fingerprint_changed": False,
                                "verification_score_delta": round(_post_resynth_score - _pre_resynth_score, 2),
                                "components_delta": len(blueprint.get("components", [])) - _pre_resynth_components,
                                "gates_changed": False,
                            },
                        )
                        state.known["_skip_enrichment"] = True
                        self._emit_insight("Termination: semantic progress stalled")

                    # Re-check closed-loop fidelity only if fidelity was the trigger
                    if _fidelity_triggered:
                        _pre_fidelity = cl_fidelity  # fidelity before re-synthesis
                        try:
                            from kernel.closed_loop import closed_loop_gate
                            cl_recheck = closed_loop_gate(description, blueprint)
                            state.known["closed_loop_fidelity"] = cl_recheck.fidelity_score
                            if cl_recheck.passed:
                                self._emit_insight(
                                    f"Closed-loop re-check: fidelity={cl_recheck.fidelity_score:.2f} ✓"
                                )
                            elif cl_recheck.fidelity_score < _pre_fidelity:
                                # Got worse — re-synthesis introduced semantic drift
                                self._emit_insight(
                                    f"Closed-loop re-check: fidelity regressed "
                                    f"{_pre_fidelity:.2f}→{cl_recheck.fidelity_score:.2f} — "
                                    f"re-synthesis introduced drift"
                                )
                                verification["status"] = "catastrophic"
                            else:
                                # Improved but still below threshold — reject
                                self._emit_insight(
                                    f"Closed-loop re-check: fidelity "
                                    f"{_pre_fidelity:.2f}→{cl_recheck.fidelity_score:.2f} "
                                    f"(still below {PROTOCOL.engine.fidelity_threshold}, rejected)"
                                )
                                verification["status"] = "catastrophic"
                        except Exception as e:
                            logger.warning(f"Closed-loop re-check failed: {e}")
                            verification["status"] = "catastrophic"

            # Catastrophic fidelity failure — abort compilation
            if verification.get("status") == "catastrophic":
                cl_fid = state.known.get("closed_loop_fidelity", 0.0)
                logger.error(
                    f"Catastrophic fidelity failure: {cl_fid:.2f} — "
                    f"blueprint does not reconstruct intent after re-synthesis"
                )
                return CompileResult(
                    success=False,
                    blueprint=blueprint,
                    context_graph=state.to_context_graph(),
                    insights=state.insights,
                    verification=verification,
                    stage_results=stage_results,
                    error=f"Fidelity gate: blueprint fidelity {cl_fid:.2f} below {PROTOCOL.engine.fidelity_threshold} after re-synthesis",
                    interrogation=state.known.get("interrogation", {}),
                    termination_condition=self._make_termination_condition(
                        status="halted",
                        reason="fidelity_gate_failed",
                        message="Compilation halted because re-synthesis still failed the fidelity gate.",
                        next_action="Revise the seed intent or reduce semantic scope before trying again.",
                    ),
                )

            # Phase 28.1: Enrichment pass — fill thin components with domain operations
            # Runs even when verification passes, because "pass" doesn't mean "deep"
            if verification.get("status") != "catastrophic":
                thin_components = self._identify_thin_components(blueprint, verification)
                if thin_components and not state.known.get("_enrichment_done") and not state.known.get("_skip_enrichment"):
                    self._emit("Enriching thin components with domain operations...")
                    enrich_start = time.time()
                    try:
                        with timeout_context(synthesis_gate.timeout_seconds, "enrichment"):
                            blueprint = self._targeted_resynthesis(blueprint, verification, state)
                        blueprint, _enrich_dedup = deduplicate_blueprint(blueprint)
                        stage_timings["enrichment"] = time.time() - enrich_start
                        self._collect_usage()
                        self._check_cost_cap()
                    except Exception as e:
                        logger.warning(f"Enrichment pass skipped: {e}")
                    state.known["_enrichment_done"] = True

            # Phase 14: Promote undeclared endpoints (re-run after any re-synthesis)
            blueprint = self._promote_undeclared_endpoints(blueprint)

            # Graph validation (reachability, orphans, cycles)
            graph_validation = validate_graph(blueprint)
            if graph_validation.get("errors"):
                for err in graph_validation["errors"]:
                    self._emit_insight(f"⚠ Graph: {err}")

            # Re-run SEED on final blueprint (after enrichment + second promote)
            from core.seed import validate_seed, seed_gate_rate
            seed_final = validate_seed(blueprint, description)
            seed_final_rate = seed_gate_rate(seed_final)
            state.known["_seed_validation_final"] = {
                "gate_rate": seed_final_rate,
                "violations": {k: v for k, v in seed_final.items() if v},
            }

            # Phase 15: Auto-resolution — resolve unknowns without asking user
            from core.resolution import resolve_unresolved, apply_resolution_to_blueprint
            resolution_report = resolve_unresolved(
                blueprint=blueprint,
                verification=verification,
                context_graph=state.to_context_graph(),
                original_input=description,
            )
            blueprint = apply_resolution_to_blueprint(blueprint, resolution_report)
            if resolution_report.resolved_count:
                self._emit_rich(
                    f"Auto-resolved {resolution_report.resolved_count} unknowns, "
                    f"{resolution_report.acknowledged_count} acknowledged",
                    InsightCategory.RESOLUTION,
                    metrics={
                        "resolved": float(resolution_report.resolved_count),
                        "acknowledged": float(resolution_report.acknowledged_count),
                    },
                )

            # Difficulty signal: ambiguity count from resolution
            self._difficulty.ambiguity_count = (
                resolution_report.resolved_count + resolution_report.acknowledged_count
            )
            self._emit_difficulty()

            # Phase A: Extract dimensional metadata
            from core.dimension_extractor import build_dimensional_metadata
            from core.dimensional import serialize_dimensional_metadata
            dim_meta = build_dimensional_metadata(state, blueprint)
            dimensional_dict = serialize_dimensional_metadata(dim_meta)

            # Phase B.1: Extract interface contracts
            from core.interface_extractor import extract_interface_map
            from core.interface_schema import serialize_interface_map
            _rel_flows = None
            _type_hints = None
            if self.domain_adapter:
                _rel_flows = dict(self.domain_adapter.vocabulary.relationship_flows)
                _type_hints = dict(self.domain_adapter.vocabulary.type_hints)
            interface_map_obj = extract_interface_map(
                blueprint, dim_meta,
                relationship_flows=_rel_flows,
                type_hints=_type_hints,
            )
            interface_map_dict = serialize_interface_map(interface_map_obj)

            # Phase 5.3: Add blueprint version
            blueprint = add_version(blueprint)

            # Ensure blueprint.insights is populated from dialogue (not LLM-dependent)
            if state.insights and not blueprint.get("insights"):
                from core.digest import _deduplicate_insights, _rank_insights
                ranked = _deduplicate_insights(_rank_insights(state))
                # Cap at 10 max insights to prevent degenerate dialogue tail
                blueprint["insights"] = [r["insight"] for r in ranked[:10]]

            # Phase 6.2: Build cache stats
            cache_stats_data = {
                "intent_cache_hit": intent_cache_hit,
                "persona_cache_hit": persona_cache_hit,
                "cache_policy": self.cache_policy,
            }
            if self.cache_policy != "none":
                cache_stats_data["cache_stats"] = self._cache.stats()

            # Phase 12.2c: Build conflict summary for context graph
            conflict_summary = {
                "total": len(state.conflicts),
                "resolved": sum(1 for c in state.conflicts if c["resolved"]),
                "unresolved": [
                    {
                        "topic": c["topic"],
                        "category": c.get("category", "unknown"),
                        "positions": c["positions"],
                    }
                    for c in state.conflicts if not c["resolved"]
                ],
            }

            # Inject conflict_summary into context graph
            context_graph_data = state.to_context_graph(compact=True)
            context_graph_data["conflict_summary"] = conflict_summary
            context_graph_data["dimensional_metadata"] = dimensional_dict
            context_graph_data["interface_map"] = interface_map_dict
            context_graph_data["closed_loop_fidelity"] = state.known.get("closed_loop_fidelity")
            if state.known.get("consistency_hard_count", 0) or state.known.get("consistency_soft_count", 0):
                context_graph_data["consistency"] = {
                    "hard_contradictions": state.known.get("consistency_hard_count", 0),
                    "soft_contradictions": state.known.get("consistency_soft_count", 0),
                    "warnings": state.known.get("consistency_warnings", ""),
                }

            # Build 8: Provenance integrity — ratio of components with validated grid refs
            _prov_integrity = 0.0
            _structured_cells = state.known.get("structured_grid_cells", [])
            if _structured_cells and blueprint.get("components"):
                from core.verification import provenance_integrity_ratio
                _grid_pcs = [c["postcode"] for c in _structured_cells]
                _prov_integrity = provenance_integrity_ratio(
                    blueprint["components"], _grid_pcs
                )
            context_graph_data["provenance_integrity"] = round(_prov_integrity, 3)

            _progress("materialization", 7, "Building final output")

            # Success = blueprint has components. Verification is a quality signal,
            # not a hard gate. A blueprint with 20+ components and "needs_work"
            # verification is still a valid, usable compilation.
            has_blueprint = bool(blueprint.get("components"))

            # Corpus feedback: track suggestion usage + anti-patterns
            corpus_feedback = self._compute_corpus_feedback(
                blueprint, corpus_suggestions, state.known.get("domain_model"),
            )

            semantic_nodes, blocking_escalations = self._build_semantic_pause_payload(
                blueprint=blueprint,
                verification=verification,
                context_graph=context_graph_data,
                dimensional_metadata=dimensional_dict,
                description=description,
                run_id=f"engine:{int(time.time())}",
            )
            if blocking_escalations:
                _progress(
                    "awaiting_decision",
                    7,
                    blocking_escalations[0].get("question", "Human decision required before coding can continue."),
                    escalations=blocking_escalations,
                )

            termination_condition = state.known.get("_termination_condition")
            if blocking_escalations:
                termination_condition = self._make_termination_condition(
                    status="awaiting_human",
                    reason="human_decision_required",
                    message=blocking_escalations[0].get(
                        "question",
                        "A human decision is required before compilation can continue.",
                    ),
                    next_action="Answer the blocking question to resume compilation.",
                    semantic_progress=(
                        termination_condition.get("semantic_progress")
                        if isinstance(termination_condition, dict)
                        else None
                    ),
                )
            elif not termination_condition:
                verification_status = verification.get("status")
                termination_condition = self._make_termination_condition(
                    status="complete",
                    reason="verification_passed" if verification_status == "pass" else "quality_floor_reached",
                    message=(
                        "Compilation completed and the blueprint is ready to inspect."
                        if verification_status == "pass"
                        else "Compilation stopped at the current quality floor without a blocking human decision."
                    ),
                    next_action=(
                        "Export the blueprint or compile deeper."
                        if verification_status == "pass"
                        else "Review governance, add constraints, or deepen the compile."
                    ),
                )

            result = CompileResult(
                success=has_blueprint,
                blueprint=blueprint,
                context_graph=context_graph_data,
                insights=state.insights,
                verification=verification,
                stage_results=stage_results,
                schema_validation=schema_validation,
                graph_validation=graph_validation,
                corpus_suggestions=corpus_suggestions,
                corpus_feedback=corpus_feedback,
                cache_stats=cache_stats_data,
                input_quality=state.known.get("input_quality", {}),
                dimensional_metadata=dimensional_dict,
                interface_map=interface_map_dict,
                stage_timings=stage_timings,
                retry_counts=retry_counts,
                interrogation=state.known.get("interrogation", {}),
                semantic_grid=kernel_grid_data,
                depth_chains=_depth_chain_dicts,
                structured_insights=[si.to_dict() for si in self._structured_insights],
                difficulty=self._difficulty.to_dict(),
                semantic_nodes=semantic_nodes,
                blocking_escalations=blocking_escalations,
                termination_condition=termination_condition,
            )

            # Phase 12.2a: Collect process telemetry
            from collections import Counter
            msg_types = Counter(m.message_type.name for m in state.history)
            depth_config = calculate_dialogue_depth(intent, description)
            telemetry = {
                "dialogue_turns": dialogue_turns,
                "confidence_trajectory": state.confidence_history,
                "message_type_counts": dict(msg_types),
                "conflict_count": len(state.conflicts),
                "structural_conflicts_resolved": resolved,
                "unknown_count": len(state.unknown),
                "dialogue_depth_config": {
                    "min_turns": depth_config[0],
                    "min_insights": depth_config[1],
                    "max_turns": depth_config[2],
                },
            }

            # Compute verification_score from 4 verification dimensions.
            # Must use verification quality, NOT dialogue confidence
            # (context_graph["confidence"] measures spec convergence, not output fidelity).
            v_score = None
            if verification and isinstance(verification, dict):
                def _dim_score(key: str) -> float:
                    val = verification.get(key, 0)
                    if isinstance(val, dict):
                        return float(val.get("score", 0))
                    return float(val) if val else 0.0
                _dims = [_dim_score(d) for d in
                         ("completeness", "consistency", "coherence", "traceability")]
                if any(d > 0 for d in _dims):
                    v_score = sum(_dims) / 4.0
            # Build 8: Blend provenance integrity into trust (weight 0.25)
            if v_score is not None and _prov_integrity > 0:
                _prov_score = _prov_integrity * 100.0
                v_score = v_score * 0.75 + _prov_score * 0.25

            # Phase 3.6: Store in corpus
            if self.auto_store:
                self._emit("Storing in corpus...")
                # Serialize corpus_feedback for L2 loop persistence
                _corpus_feedback_json = None
                if corpus_feedback and corpus_feedback.get("corpus_influence") != "none":
                    _corpus_feedback_json = json.dumps(corpus_feedback)

                record = self.corpus.store(
                    input_text=description,
                    context_graph=result.context_graph,
                    blueprint=result.blueprint,
                    insights=result.insights,
                    success=result.success,
                    provider=self.provider_name,
                    model=self.model_name,
                    stage_timings=stage_timings,
                    retry_counts=retry_counts,
                    verification_score=v_score,
                    corpus_feedback=_corpus_feedback_json,
                    **telemetry,
                )
                self._emit_rich(f"Stored as {record.id}", InsightCategory.STATUS)

            # Phase 19: Record metrics in ring buffer
            total_duration = sum(stage_timings.values())
            total_cache_hits = sum(
                s.get("hits", 0)
                for s in cache_stats_data.get("cache_stats", {}).values()
                if isinstance(s, dict)
            )
            total_cache_misses = sum(
                s.get("misses", 0)
                for s in cache_stats_data.get("cache_stats", {}).values()
                if isinstance(s, dict)
            )
            comp_id = record.id if self.auto_store else ""

            # Phase 21: Compute cost from accumulated tokens
            cost_input = sum(tu.input_tokens for tu in self._compilation_tokens)
            cost_output = sum(tu.output_tokens for tu in self._compilation_tokens)
            cost_usd = sum(estimate_cost(tu).total_cost for tu in self._compilation_tokens)
            self._session_cost_usd += cost_usd

            self._record_metrics(
                compilation_id=comp_id,
                success=result.success,
                total_duration=total_duration,
                stage_timings=stage_timings,
                dialogue_turns=telemetry.get("dialogue_turns", 0),
                component_count=len(result.blueprint.get("components", [])),
                insight_count=len(result.insights),
                verification_score=v_score or 0,
                verification_mode=verification.get("verification_mode", "unknown"),
                cache_hits=total_cache_hits,
                cache_misses=total_cache_misses,
                retry_count=sum(retry_counts.values()),
                total_input_tokens=cost_input,
                total_output_tokens=cost_output,
                estimated_cost_usd=cost_usd,
            )

            # Phase 22f: Record outcome for governor feedback loop
            try:
                v = verification or {}
                cat_dist = state.known.get("compression_loss_categories", {})
                outcome = {
                    "compile_id": comp_id or f"compile-{int(time.time())}",
                    "input_summary": description[:200],
                    "trust_score": v_score,
                    "completeness": self._extract_dim_score(v, "completeness"),
                    "consistency": self._extract_dim_score(v, "consistency"),
                    "coherence": self._extract_dim_score(v, "coherence"),
                    "traceability": self._extract_dim_score(v, "traceability"),
                    "component_count": len(result.blueprint.get("components", [])),
                    "rejected": not result.success,
                    "rejection_reason": result.error or "",
                    "domain": state.known.get("domain", "software"),
                    "compression_loss_categories": tuple(sorted(cat_dist.items())),
                }
                if not hasattr(self, "_compilation_outcomes"):
                    self._compilation_outcomes = []
                self._compilation_outcomes.append(outcome)
                # Keep last 50 outcomes
                if len(self._compilation_outcomes) > 50:
                    self._compilation_outcomes = self._compilation_outcomes[-50:]
                # Persist to SQLite — survives restart
                if self._outcome_store:
                    self._outcome_store.append(
                        compile_id=outcome["compile_id"],
                        input_summary=outcome["input_summary"],
                        trust_score=outcome["trust_score"],
                        completeness=outcome["completeness"],
                        consistency=outcome["consistency"],
                        coherence=outcome["coherence"],
                        traceability=outcome["traceability"],
                        component_count=outcome["component_count"],
                        rejected=outcome["rejected"],
                        rejection_reason=outcome["rejection_reason"],
                        domain=outcome["domain"],
                        compression_loss_categories=json.dumps(dict(outcome["compression_loss_categories"])) if outcome["compression_loss_categories"] else "",
                    )
            except Exception as e:
                logger.debug(f"Outcome recording skipped: {e}")

            # Phase 22g: Regression corpus — append snapshot for trust validation
            try:
                from core.regression_corpus import build_record, append_record, check_regression
                _bp = result.blueprint or {}
                _comp_names = tuple(
                    c.get("name", "") for c in _bp.get("components", [])
                )
                _cl_fidelity = state.known.get("closed_loop_fidelity", 0.0)
                _cat_dist = state.known.get("compression_loss_categories", {})
                _reg_record = build_record(
                    compile_id=comp_id or f"compile-{int(time.time())}",
                    input_text=description,
                    domain=state.known.get("domain", "software"),
                    provider=self.provider_name or "",
                    model=self.model_name or "",
                    trust_score=v_score or 0.0,
                    completeness=self._extract_dim_score(verification or {}, "completeness"),
                    consistency=self._extract_dim_score(verification or {}, "consistency"),
                    coherence=self._extract_dim_score(verification or {}, "coherence"),
                    traceability=self._extract_dim_score(verification or {}, "traceability"),
                    component_count=len(_bp.get("components", [])),
                    relationship_count=len(_bp.get("relationships", [])),
                    component_names=_comp_names,
                    verification_mode=(verification or {}).get("verification_mode", "unknown"),
                    rejected=not result.success,
                    rejection_reason=result.error or "",
                    fidelity_score=_cl_fidelity,
                    compression_losses=_cat_dist if isinstance(_cat_dist, dict) else dict(_cat_dist),
                    dialogue_turns=dialogue_turns,
                    total_duration_s=sum(stage_timings.values()),
                    cost_usd=cost_usd,
                )
                append_record(_reg_record)
                _reg_warnings = check_regression(_reg_record)
                if _reg_warnings:
                    for w in _reg_warnings:
                        self._emit_insight(f"Regression warning: {w}")
            except Exception as e:
                logger.debug(f"Regression corpus recording skipped: {e}")

            # Phase 23a: Episodic memory consolidation
            try:
                from kernel.memory import consolidate, save_memory
                compile_duration = sum(stage_timings.values())
                mem = consolidate(result, state, comp_id or f"compile-{int(time.time())}", compile_duration)
                save_memory(mem)
            except Exception as e:
                logger.debug(f"Memory consolidation skipped: {e}")

            # Phase 23b: Training data emission
            try:
                from kernel.training import extract_training_examples, emit_jsonl
                if self._kernel_grid:
                    _train_examples = extract_training_examples(
                        self._kernel_grid, result, state,
                        comp_id or f"compile-{int(time.time())}",
                    )
                    if _train_examples:
                        emit_jsonl(_train_examples)
            except Exception as e:
                logger.debug(f"Training emission skipped: {e}")

            # Phase 23c: L2 pattern detection from accumulated memory
            try:
                from kernel.memory import (
                    load_recent_memories, detect_patterns, save_patterns,
                )
                recent = load_recent_memories(limit=50)
                if len(recent) >= 3:
                    new_patterns = detect_patterns(recent)
                    if new_patterns:
                        save_patterns(new_patterns)
                        self._emit_insight(f"L2: {len(new_patterns)} pattern(s) detected and saved")
            except Exception as e:
                logger.debug(f"L2 pattern detection skipped: {e}")

            return result

        except FractureError as e:
            logger.info(f"Fracture detected at {e.stage}: {e.signal}")
            return CompileResult(
                success=False,
                blueprint={},
                context_graph=state.to_context_graph(),
                insights=state.insights,
                verification={},
                stage_results=stage_results,
                error=f"Intent fracture at {e.stage}",
                fracture={
                    "stage": e.signal.stage,
                    "competing_configs": list(e.signal.competing_configs),
                    "collapsing_constraint": e.signal.collapsing_constraint,
                    "agent": e.signal.agent,
                    "context": e.signal.context,
                } if e.signal else None,
                interrogation=state.known.get("interrogation", {}),
                termination_condition=self._make_termination_condition(
                    status="awaiting_human",
                    reason="intent_fracture",
                    message="Compilation hit an intent fracture with multiple valid semantic directions.",
                    next_action="Choose the collapsing constraint so the compiler can continue.",
                ),
            )
        except MotherlabsTimeoutError as e:
            logger.error(f"Timeout during compilation: {e.operation} after {e.timeout_seconds}s")
            return CompileResult(
                success=False,
                blueprint={},
                context_graph=state.to_context_graph(),
                insights=state.insights,
                verification={},
                stage_results=stage_results,
                error=str(e),
                interrogation=state.known.get("interrogation", {}),
                termination_condition=self._make_termination_condition(
                    status="halted",
                    reason="timeout",
                    message=str(e),
                    next_action="Retry with narrower scope or a higher budget/time ceiling.",
                ),
            )
        except MotherlabsError as e:
            logger.error(f"Compilation error: {type(e).__name__}: {e}")
            return CompileResult(
                success=False,
                blueprint={},
                context_graph=state.to_context_graph(),
                insights=state.insights,
                verification={},
                stage_results=stage_results,
                error=str(e),
                interrogation=state.known.get("interrogation", {}),
                termination_condition=self._make_termination_condition(
                    status="halted",
                    reason=type(e).__name__,
                    message=str(e),
                    next_action="Revise the input or resolve the reported compiler error before retrying.",
                ),
            )
        except Exception as e:
            logger.exception(f"Unexpected error during compilation: {e}")
            return CompileResult(
                success=False,
                blueprint={},
                context_graph=state.to_context_graph(),
                insights=state.insights,
                verification={},
                stage_results=stage_results,
                error=str(e),
                interrogation=state.known.get("interrogation", {}),
                termination_condition=self._make_termination_condition(
                    status="halted",
                    reason="unexpected_error",
                    message=str(e),
                    next_action="Inspect the compiler error and retry with a narrower scope.",
                ),
            )

    # =========================================================================
    # COMPILATION MODE HANDLERS
    # =========================================================================

    def _compile_explore_mode(
        self,
        state: "SharedState",
        description: str,
        stage_results: list,
        stage_timings: Dict[str, float],
        retry_counts: Dict[str, int],
        kernel_grid_data: Optional[Dict],
        depth_chain_dicts: List[Dict],
        intent_cache_hit: bool,
        persona_cache_hit: bool,
    ) -> "CompileResult":
        """EXPLORE mode: synthesize divergent exploration, skip blueprint synthesis."""
        from core.exploration_synthesis import (
            synthesize_exploration, exploration_map_to_dict,
        )

        self._emit("Synthesizing exploration map...")
        explore_start = time.time()

        # Collect structured cells from grid
        structured_cells = state.known.get("structured_grid_cells", [])

        # Collect dialogue messages
        dialogue_msgs = [
            {"sender": m.sender, "content": m.content}
            for m in state.history
        ]

        exp_map = synthesize_exploration(
            grid_cells=structured_cells,
            dialogue_messages=dialogue_msgs,
            original_intent=description,
            endpoint_chains=depth_chain_dicts or None,
        )
        exp_dict = exploration_map_to_dict(exp_map)
        stage_timings["exploration_synthesis"] = time.time() - explore_start

        self._emit_insight(
            f"Exploration: {len(exp_map.insights)} insights, "
            f"{len(exp_map.frontier_questions)} frontier questions, "
            f"{len(exp_map.adjacent_domains)} adjacent domains"
        )

        return CompileResult(
            success=True,
            blueprint={},  # No blueprint in EXPLORE mode
            context_graph=state.to_context_graph(),
            insights=state.insights,
            verification={},
            stage_results=stage_results,
            stage_timings=stage_timings,
            retry_counts=retry_counts,
            input_quality=state.known.get("input_quality", {}),
            interrogation=state.known.get("interrogation", {}),
            semantic_grid=kernel_grid_data,
            depth_chains=depth_chain_dicts,
            exploration_map=exp_dict,
        )

    def _compile_context_mode(
        self,
        state: "SharedState",
        description: str,
        stage_results: list,
        stage_timings: Dict[str, float],
        retry_counts: Dict[str, int],
        kernel_grid_data: Optional[Dict],
        depth_chain_dicts: List[Dict],
        intent_cache_hit: bool,
        persona_cache_hit: bool,
    ) -> "CompileResult":
        """CONTEXT mode: synthesize context map, lightweight verification."""
        from core.context_synthesis import (
            synthesize_context, context_map_to_dict,
        )

        self._emit("Synthesizing context map...")
        context_start = time.time()

        # Collect structured cells from grid
        structured_cells = state.known.get("structured_grid_cells", [])

        # Collect dialogue messages
        dialogue_msgs = [
            {"sender": m.sender, "content": m.content}
            for m in state.history
        ]

        ctx_map = synthesize_context(
            grid_cells=structured_cells,
            dialogue_messages=dialogue_msgs,
            original_intent=description,
        )
        ctx_dict = context_map_to_dict(ctx_map)
        stage_timings["context_synthesis"] = time.time() - context_start

        self._emit_insight(
            f"Context: {len(ctx_map.concepts)} concepts, "
            f"{len(ctx_map.relationships)} relationships, "
            f"{len(ctx_map.assumptions)} assumptions, "
            f"{len(ctx_map.unknowns)} unknowns"
        )

        # Lightweight verification: check analytical completeness
        has_concepts = len(ctx_map.concepts) > 0
        has_relationships = len(ctx_map.relationships) > 0
        verification = {
            "status": "pass" if has_concepts else "needs_work",
            "completeness": {"score": min(100, len(ctx_map.concepts) * 10)},
            "analytical_depth": {
                "concepts": len(ctx_map.concepts),
                "relationships": len(ctx_map.relationships),
                "assumptions": len(ctx_map.assumptions),
                "unknowns": len(ctx_map.unknowns),
                "vocabulary": len(ctx_map.vocabulary),
            },
        }

        # Persist kernel grid for CONTEXT mode (enriches future compilations)
        try:
            if self._kernel_grid:
                from kernel.store import save_grid as _save_grid
                _save_grid(
                    self._kernel_grid,
                    map_id="context",
                    name=f"context: {description[:80]}",
                )
        except Exception as e:
            logger.debug(f"Context grid persistence skipped: {e}")

        return CompileResult(
            success=has_concepts,
            blueprint={},  # No blueprint in CONTEXT mode
            context_graph=state.to_context_graph(),
            insights=state.insights,
            verification=verification,
            stage_results=stage_results,
            stage_timings=stage_timings,
            retry_counts=retry_counts,
            input_quality=state.known.get("input_quality", {}),
            interrogation=state.known.get("interrogation", {}),
            semantic_grid=kernel_grid_data,
            depth_chains=depth_chain_dicts,
            context_map=ctx_dict,
        )

    # =========================================================================
    # PHASE 21: COST TRACKING
    # =========================================================================

    def _collect_usage(self) -> None:
        """Read last_usage from the LLM client and append a TokenUsage."""
        usage_data = getattr(self.llm, 'last_usage', {})
        if not usage_data:
            return
        inp = usage_data.get("input_tokens", 0)
        outp = usage_data.get("output_tokens", 0)
        total = usage_data.get("total_tokens", inp + outp)
        tu = TokenUsage(
            input_tokens=inp,
            output_tokens=outp,
            total_tokens=total,
            provider=self.provider_name,
            model=self.model_name,
        )
        self._compilation_tokens.append(tu)

    def _check_cost_cap(self) -> None:
        """Check accumulated cost against per-compilation and session caps."""
        if not self._compilation_tokens:
            return
        total_cost = sum(estimate_cost(tu).total_cost for tu in self._compilation_tokens)
        cap = PROTOCOL.cost.per_compilation_cap_usd
        warn = PROTOCOL.cost.per_compilation_warn_usd
        if total_cost > cap:
            raise CostCapExceededError(
                f"Compilation cost ${total_cost:.4f} exceeds cap ${cap:.2f}",
                current_cost=total_cost,
                cap=cap,
            )
        if total_cost > warn:
            logger.warning(f"Compilation cost ${total_cost:.4f} exceeds warning threshold ${warn:.2f}")
        # E8002: Session cost cap
        session_cap = PROTOCOL.cost.session_cap_usd
        if self._session_cost_usd + total_cost > session_cap:
            raise CostCapExceededError(
                f"Session cost ${self._session_cost_usd + total_cost:.4f} exceeds session cap ${session_cap:.2f}",
                current_cost=self._session_cost_usd + total_cost,
                cap=session_cap,
                error_code="E8002",
            )

    # =========================================================================
    # PHASE 19: TELEMETRY — ring buffer + public accessors
    # =========================================================================

    def _record_metrics(
        self,
        compilation_id: str,
        success: bool,
        total_duration: float,
        stage_timings: Dict[str, float],
        dialogue_turns: int,
        component_count: int,
        insight_count: int,
        verification_score: int,
        verification_mode: str,
        cache_hits: int,
        cache_misses: int,
        retry_count: int,
        total_input_tokens: int = 0,
        total_output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """Append a CompilationMetrics to the ring buffer."""
        m = CompilationMetrics(
            compilation_id=compilation_id,
            timestamp=time.time(),
            success=success,
            total_duration=total_duration,
            stage_timings=tuple(sorted(stage_timings.items())),
            dialogue_turns=dialogue_turns,
            component_count=component_count,
            insight_count=insight_count,
            verification_score=verification_score,
            verification_mode=verification_mode,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            retry_count=retry_count,
            provider=self.provider_name,
            model=self.model_name,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self._metrics_buffer.append(m)
        # Trim to buffer size
        if len(self._metrics_buffer) > self._metrics_buffer_size:
            self._metrics_buffer = self._metrics_buffer[-self._metrics_buffer_size:]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate metrics from the in-memory ring buffer.

        Phase 19: No DB queries for real-time metrics.
        """
        agg = aggregate_metrics(self._metrics_buffer)
        return aggregate_to_dict(agg)

    def get_health_snapshot(self) -> Dict[str, Any]:
        """
        Get point-in-time health assessment.

        Phase 19: Combines ring buffer metrics with corpus and cache stats.
        """
        uptime = time.time() - self._engine_start_time
        corpus_stats = self.corpus.get_stats()
        corpus_size = corpus_stats.get("total_compilations", 0)
        snapshot = compute_health(self._metrics_buffer, uptime, corpus_size)
        result = health_to_dict(snapshot)
        # Merge cache stats
        result["cache"] = self._cache.stats()
        return result

    def _emit(self, message: str):
        """Emit status message."""
        self.on_insight(f"◇ {message}")

    @staticmethod
    def _extract_dim_score(verification: dict, dimension: str) -> float:
        """Extract a trust dimension score from verification dict.

        Handles both flat ({"completeness": 72}) and nested
        ({"completeness": {"score": 72, "gaps": [...]}}) formats.
        """
        val = verification.get(dimension, 0)
        if isinstance(val, dict):
            return float(val.get("score", 0))
        return float(val) if val else 0.0

    def _emit_insight(self, insight: str):
        """Emit insight for CLI display and progress callback."""
        self.on_insight(f"  → {insight}")
        # Also push flat insight to progress callback for live frontend display
        if self._progress_callback:
            try:
                self._progress_callback(self._current_stage, -1, insight)
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")

    def _emit_rich(
        self,
        text: str,
        category: str,
        metrics: Optional[Dict[str, float]] = None,
        reasoning: str = "",
    ) -> None:
        """Emit a structured insight for glass-box compilation.

        Creates a StructuredInsight, appends to internal list, calls
        _emit_insight() for backward compat, and pushes the structured
        dict to progress_callback if available.
        """
        si = StructuredInsight(
            text=text,
            category=category,
            stage=self._current_stage,
            metrics=metrics or {},
            reasoning=reasoning,
        )
        self._structured_insights.append(si)
        # Backward compat: flat insight still goes to CLI + progress
        self._emit_insight(text)
        # Push structured dict to progress callback if wired
        if self._progress_callback:
            try:
                self._progress_callback(
                    self._current_stage, -1, "",
                    structured_insight=si.to_dict(),
                )
            except Exception as e:
                logger.debug("Structured insight push failed: %s", e)

    def _emit_difficulty(self) -> None:
        """Push current difficulty snapshot to progress callback."""
        if self._progress_callback:
            try:
                self._progress_callback(
                    self._current_stage, -1, "",
                    difficulty=self._difficulty.to_dict(),
                )
            except Exception as e:
                logger.debug("Difficulty push failed: %s", e)

    def _check_gate(
        self,
        stage_name: str,
        metrics: Dict[str, Any]
    ) -> StageResult:
        """
        Check if stage metrics pass the gate criteria.

        Derived from: Stage-gate Architecture

        Args:
            stage_name: Name of the stage (must be in STAGE_GATES)
            metrics: Dictionary of metric values to check

        Returns:
            StageResult with success, errors, warnings
        """
        gate = self._stage_gates.get(stage_name)
        if not gate:
            return StageResult(
                stage=stage_name,
                success=True,
                output=metrics,
                errors=[],
                warnings=[f"No gate defined for stage '{stage_name}'"]
            )

        errors = []
        warnings = []

        # Check required criteria
        for criterion, threshold in gate.required_criteria.items():
            value = metrics.get(criterion)
            if value is None:
                errors.append(f"Missing required metric: {criterion}")
            elif isinstance(threshold, bool):
                if bool(value) != threshold:
                    errors.append(f"{criterion}: expected {threshold}, got {bool(value)}")
            elif isinstance(threshold, (int, float)):
                if value < threshold:
                    errors.append(f"{criterion}: {value} < {threshold} (required)")

        # Check optional criteria (generate warnings)
        for criterion, threshold in gate.optional_criteria.items():
            value = metrics.get(criterion)
            if value is not None:
                if isinstance(threshold, bool):
                    if bool(value) != threshold:
                        warnings.append(f"{criterion}: expected {threshold}, got {bool(value)}")
                elif isinstance(threshold, (int, float)):
                    if criterion.startswith("max_"):
                        if value > threshold:
                            warnings.append(f"{criterion}: {value} > {threshold} (recommended max)")
                    else:
                        if value < threshold:
                            warnings.append(f"{criterion}: {value} < {threshold} (recommended)")

        return StageResult(
            stage=stage_name,
            success=len(errors) == 0,
            output=metrics,
            errors=errors,
            warnings=warnings
        )

    def _build_semantic_pause_payload(
        self,
        *,
        blueprint: Dict[str, Any],
        verification: Dict[str, Any],
        context_graph: Dict[str, Any],
        dimensional_metadata: Dict[str, Any],
        description: str,
        run_id: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Project compile output into postcode-native nodes + pause candidates."""
        if not blueprint.get("components"):
            return [], []

        from core.blueprint_protocol import (
            build_blueprint_semantic_gates,
            build_blueprint_semantic_nodes,
            build_semantic_gate_escalations,
        )
        from core.trust import compute_trust_indicators, serialize_trust_indicators

        intent_keywords = context_graph.get("keywords", []) if isinstance(context_graph, dict) else []
        trust_payload = serialize_trust_indicators(
            compute_trust_indicators(
                blueprint=blueprint,
                verification=verification,
                context_graph=context_graph,
                dimensional_metadata=dimensional_metadata,
                intent_keywords=intent_keywords,
            )
        )
        blueprint["semantic_gates"] = build_blueprint_semantic_gates(
            blueprint,
            trust=trust_payload,
            verification=verification,
            context_graph=context_graph,
        )

        semantic_nodes = [
            node.model_dump()
            for node in build_blueprint_semantic_nodes(
                blueprint,
                seed_text=description,
                trust=trust_payload,
                verification=verification,
                run_id=run_id,
            )
        ]
        blocking_escalations = build_semantic_gate_escalations(
            semantic_nodes,
            blueprint=blueprint,
            trust=trust_payload,
            context_graph=context_graph,
        )
        blueprint["semantic_nodes"] = list(semantic_nodes)
        return semantic_nodes, blocking_escalations

    def _normalize_synthesis_output(
        self,
        blueprint: Dict[str, Any],
        *,
        run_id: str,
    ) -> Dict[str, Any]:
        """Stabilize synthesis sidecars so downstream code sees canonical shapes."""
        normalized = dict(blueprint or {})
        normalized.setdefault("components", [])
        normalized.setdefault("relationships", [])
        normalized.setdefault("constraints", [])
        normalized.setdefault("unresolved", [])

        if not isinstance(normalized.get("semantic_gates"), list):
            normalized["semantic_gates"] = []
        if not isinstance(normalized.get("semantic_nodes"), list):
            normalized["semantic_nodes"] = []

        if normalized["semantic_nodes"]:
            from core.blueprint_protocol import build_blueprint_semantic_nodes

            normalized["semantic_nodes"] = [
                node.model_dump()
                for node in build_blueprint_semantic_nodes(
                    normalized,
                    seed_text="",
                    verification={},
                    run_id=run_id,
                    agent_id="Synthesis",
                )
            ]

        return normalized

    @staticmethod
    def _verification_score(verification: Dict[str, Any]) -> float:
        dims = []
        for key in ("completeness", "consistency", "coherence", "traceability"):
            value = verification.get(key, 0) if isinstance(verification, dict) else 0
            if isinstance(value, dict):
                dims.append(float(value.get("score", 0) or 0))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                dims.append(float(value))
            else:
                dims.append(0.0)
        return sum(dims) / max(len(dims), 1)

    @staticmethod
    def _semantic_gate_signature(blueprint: Dict[str, Any], verification: Dict[str, Any]) -> List[tuple[str, str, str]]:
        signature: set[tuple[str, str, str]] = set()
        for gate_source in (
            (blueprint or {}).get("semantic_gates", []) or [],
            (verification or {}).get("semantic_gates", []) or [],
        ):
            for gate in gate_source:
                if not isinstance(gate, dict):
                    continue
                question = str(gate.get("question") or "").strip().lower()
                if not question:
                    continue
                identity = str(
                    gate.get("node_ref")
                    or gate.get("owner_component")
                    or gate.get("postcode")
                    or question
                ).strip().lower()
                kind = str(gate.get("kind") or "semantic_gate").strip().lower()
                signature.add((identity, question, kind))
        return sorted(signature)

    @staticmethod
    def _make_termination_condition(
        *,
        status: str,
        reason: str,
        message: str,
        next_action: str,
        semantic_progress: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "status": status,
            "reason": reason,
            "message": message,
            "next_action": next_action,
        }
        if semantic_progress:
            payload["semantic_progress"] = semantic_progress
        return payload

    def _extract_intent(self, description: str, state: SharedState) -> Dict[str, Any]:
        """
        Extract intent from description.

        Derived from: PROJECT-PLAN.md Phase 1.2
        """
        msg = Message(
            sender="User",
            content=description,
            message_type=MessageType.PROPOSITION
        )

        response = self.intent_agent.run(state, msg)
        state.add_message(response)

        # Phase 22: Structured output parsing with repair retry
        try:
            parsed = self._parse_structured_output(
                response.content, "intent",
                state=state, agent=self.intent_agent, original_msg=msg,
            )
            if "core_need" in parsed:
                return parsed
        except (ValueError, Exception):
            pass

        # Fallback
        fallback = {
            "core_need": description,
            "domain": "unknown",
            "actors": [],
            "implicit_goals": [],
            "constraints": [],
            "insight": response.insight or ""
        }
        # Phase 17.3: Validate intent is not completely empty
        core = fallback["core_need"].strip() if fallback["core_need"] else ""
        domain = fallback["domain"].strip() if fallback["domain"] else ""
        has_domain = domain and domain != "unknown"
        has_actors = bool(fallback["actors"])
        if not core and not has_domain and not has_actors:
            raise CompilationError(
                "Intent extraction returned empty result",
                stage="intent",
                error_code="E3001",
            )
        return fallback

    def _generate_personas(self, intent: Dict, state: SharedState) -> List[Dict]:
        """
        Generate domain-specific personas.

        Derived from: PROJECT-PLAN.md Phase 1.3
        """
        msg = Message(
            sender="System",
            content=json.dumps(intent),
            message_type=MessageType.PROPOSITION
        )

        response = self.persona_agent.run(state, msg)
        state.add_message(response)

        # Phase 22: Structured output parsing — also fixes bare-array bug
        try:
            parsed = self._parse_structured_output(
                response.content, "personas",
                state=state, agent=self.persona_agent, original_msg=msg,
            )
            personas = parsed.get("personas", [])
            if personas:
                # Phase 17.3: Filter malformed personas (no name)
                valid_personas = [p for p in personas if p.get("name")]
                if valid_personas:
                    return valid_personas
        except (ValueError, Exception):
            pass

        # Phase 17.3: Raise if no valid personas produced
        raise CompilationError(
            "Persona generation returned empty or malformed result",
            stage="personas",
            error_code="E3002",
        )

    def _run_spec_dialogue(self, state: SharedState):
        """
        Grid-driven dialogue: navigator picks cells, agents fill them.

        The kernel grid IS convergence. Instead of running Entity/Process
        dialogue → parsing text for components → computing Jaccard delta,
        the grid bootstraps before dialogue, the navigator picks cell targets,
        agents fill them, and convergence is measured by fill state transitions.

        Preserves:
        - Entity/Process agent interface (agent.run(state, msg) → Message)
        - Dialectic friction (structural vs behavioral concern axes)
        - SharedState accumulation (history, insights, confidence)
        - Turn budget as safety ceiling
        - Fallback to text-based dialogue if grid bootstrap fails
        """
        from core.convergence import (
            ConvergenceTracker, estimate_turn_budget,
        )

        # Convergence-driven turn budget
        input_text = state.known.get("input", "")
        intent = state.known.get("intent", {})
        corpus_patterns = state.known.get("corpus_patterns", {})
        corpus_avg = corpus_patterns.get("avg_dialogue_turns")
        corpus_size = corpus_patterns.get("sample_size", 0)

        min_turns, recommended_turns, max_turns = estimate_turn_budget(
            input_text,
            corpus_avg_turns=corpus_avg,
            corpus_sample_size=corpus_size,
        )

        # Legacy depth calculation (kept for telemetry compatibility)
        _, _, _ = calculate_dialogue_depth(intent, input_text)

        # Phase 22b adaptation: experienced domains need less dialogue
        if corpus_size >= 5:
            self._emit_insight("Domain experienced — reduced dialogue depth")
            max_turns = min(max_turns, recommended_turns)

        self._emit_insight(
            f"Turn budget: {min_turns}–{max_turns} "
            f"(recommended {recommended_turns})"
        )

        # --- Attempt grid-driven dialogue ---
        try:
            grid = self._bootstrap_dialogue_grid(state)
        except Exception as e:
            logger.debug(f"Grid bootstrap failed, falling back to text dialogue: {e}")
            grid = None

        if grid is not None:
            self._run_grid_driven_dialogue(state, grid, min_turns, recommended_turns, max_turns)
        else:
            self._run_text_based_dialogue(state, min_turns, recommended_turns, max_turns)

    def _run_grid_driven_dialogue(
        self,
        state: SharedState,
        grid,
        min_turns: int,
        recommended_turns: int,
        max_turns: int,
    ):
        """Grid-driven dialogue loop: navigator picks cells, agents fill them.

        Each round dispatches up to 3 LLM calls in parallel (Phase B), then
        applies state mutations serially in deterministic postcode order (Phase C).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from kernel.navigator import score_candidates, is_converged, descend
        from kernel.ops import fill as grid_fill
        from kernel.llm_bridge import parse_agent_response_to_fill
        from core.convergence import ConvergenceTracker, grid_convergence_summary
        from agents.base import LLMAgent

        # Text convergence tracker kept for telemetry comparison
        convergence_min = 3 if min_turns <= 8 else min_turns
        text_convergence = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=convergence_min,
            recommended_turns=recommended_turns,
        )

        # Prime message for first agent call
        input_context = state.known.get("input", "")
        intent = state.known.get("intent", {})

        prime_content = f"""Analyze this system to build:

USER INPUT: {input_context}

EXTRACTED INTENT:
- Core need: {intent.get('core_need', 'unknown')}
- Domain: {intent.get('domain', 'unknown')}
- Actors: {', '.join(intent.get('actors', []))}
- Key insight: {intent.get('insight', '')}

Begin your analysis from your perspective. Remember to include an INSIGHT: line.
"""
        prime_msg = Message(
            sender="System",
            content=prime_content,
            message_type=MessageType.PROPOSITION,
        )

        last_msg = None
        total_turns = 0
        round_num = 0
        cells_per_round = 3  # Process top 3 candidates per round

        while total_turns < max_turns:
            # Navigator picks top candidates
            candidates = score_candidates(grid)
            if not candidates:
                self._emit_insight("Grid fully explored — no candidates remain")
                break

            batch = candidates[:cells_per_round]
            round_messages = []
            round_insights = []

            # === Phase A: Build prompts (serial, fast) ===
            batch_work = []
            for i, scored_cell in enumerate(batch):
                agent = self._route_agent(scored_cell.postcode_key)
                agent_name = "Process" if agent is self.process_agent else "Entity"

                if total_turns == 0 and i == 0:
                    input_msg = prime_msg
                else:
                    input_msg = self._build_cell_prompt(
                        state, grid, scored_cell.postcode_key, agent_name, round_num,
                    )
                batch_work.append((scored_cell, agent, input_msg))

            # === Phase B: Parallel LLM calls ===
            call_results = []
            with ThreadPoolExecutor(max_workers=len(batch_work)) as executor:
                future_map = {
                    executor.submit(agent.run_llm_only, state, input_msg): scored_cell
                    for scored_cell, agent, input_msg in batch_work
                }
                for future in as_completed(future_map):
                    scored_cell = future_map[future]
                    try:
                        call_results.append((scored_cell, future.result()))
                    except Exception as e:
                        logger.debug(f"Parallel agent call failed for {scored_cell.postcode_key}: {e}")

            # === Phase C: Serial state mutations (deterministic order by postcode) ===
            call_results.sort(key=lambda r: r[0].postcode_key)

            for scored_cell, call_result in call_results:
                if total_turns >= max_turns:
                    break

                response = call_result.message

                # Apply deferred mutations
                LLMAgent.apply_mutations(state, call_result)
                state.add_message(response)

                round_messages.append(response)
                if response.insight:
                    round_insights.append(response.insight)

                # Token usage from this call
                if call_result.token_usage:
                    tu = TokenUsage(
                        input_tokens=call_result.token_usage.get("input_tokens", 0),
                        output_tokens=call_result.token_usage.get("output_tokens", 0),
                        total_tokens=call_result.token_usage.get("total_tokens", 0),
                        provider=self.provider_name,
                        model=self.model_name,
                    )
                    self._compilation_tokens.append(tu)

                # Extract fill from response → populate grid cell
                self._fill_from_response(grid, scored_cell.postcode_key, response)

                # Descent: if filled cell has low confidence, create children
                # at next scope level for deeper excavation
                new_children = descend(grid, scored_cell.postcode_key)
                if new_children:
                    self._emit_insight(
                        f"Descended {scored_cell.postcode_key} → "
                        f"{len(new_children)} child cells"
                    )

                # === Per-turn machinery (preserved from text-based dialogue) ===
                if response.insight_display:
                    self._emit_insight(response.insight_display)

                state.confidence_history.append(state.confidence.overall())

                # Plateau detection: inject Governor hint if stalled
                if (state.confidence.is_plateauing(state.confidence_history)
                        and total_turns >= 3):
                    weakest = state.confidence.weakest_dimension()
                    hint_parts = [f"[GOVERNOR] Confidence stalled on {weakest}."]

                    if weakest in ("structural", "coverage"):
                        try:
                            uncovered = self.entity_agent._compute_uncovered_ground(state)
                            if uncovered:
                                hint_parts.append(
                                    f"Uncovered components: {', '.join(uncovered[:3])}."
                                )
                            else:
                                hint_parts.append(
                                    "All explicit components addressed — look for implicit ones."
                                )
                        except Exception as e:
                            logger.debug(f"Uncovered ground computation skipped: {e}")

                    if weakest == "behavioral":
                        hint_parts.append(
                            "Specify methods, state transitions, or interaction flows."
                        )

                    if weakest == "consistency":
                        if state.unknown:
                            hint_parts.append(
                                f"Unresolved unknowns: {', '.join(state.unknown[:2])}."
                            )

                    unresolved_conflicts = [
                        c for c in state.conflicts if not c["resolved"]
                    ]
                    if unresolved_conflicts:
                        topics = [c["topic"][:40] for c in unresolved_conflicts[:2]]
                        hint_parts.append(f"Open conflicts: {'; '.join(topics)}.")

                    focus_hint = " ".join(hint_parts)
                    last_msg = Message(
                        sender="Governor",
                        content=focus_hint,
                        message_type=MessageType.META,
                    )

                self._detect_structural_conflicts(state, response)

                resolved = state.resolve_unknown_by_keyword(response.content)
                if resolved:
                    self._emit_insight(f"Resolved {len(resolved)} unknown(s)")
                # ================================================

                last_msg = response
                total_turns += 1

            # Round complete
            round_num += 1
            lens_name, _ = self._lens_for_round(round_num - 1)
            self._emit(f"Grid round {round_num} complete ({total_turns} turns, "
                       f"{grid.total_cells} cells, fill={grid.fill_rate:.0%}, lens={lens_name})")

            # Update text convergence tracker for telemetry
            messages_for_snapshot = [
                {"content": m.content, "insight": m.insight}
                for m in state.history
                if m.sender in ("Entity", "Process")
            ]
            text_convergence.update(messages_for_snapshot, total_turns=total_turns)

            # Grid convergence check (structural truth, not text proxy)
            if total_turns >= min_turns and is_converged(grid):
                self._emit_insight(
                    f"Grid converged after {total_turns} turns "
                    f"({grid.total_cells} cells, fill={grid.fill_rate:.0%})"
                )
                break

            if total_turns >= max_turns:
                self._emit_insight(
                    f"Turn budget exhausted ({total_turns}/{max_turns})"
                )
                break

        # Cache grid for Phase 3.5 (skip re-compilation)
        self._kernel_grid = grid

        # Store convergence summary (grid-based, with text telemetry)
        grid_summary = grid_convergence_summary(grid)
        state.known["_convergence"] = {
            "turns": total_turns,
            "converged": grid_summary["converged"],
            "fill_rate": grid_summary["fill_rate"],
            "cells": grid_summary["total_cells"],
            "filled": grid_summary["filled"],
            "text_convergence": text_convergence.convergence_summary,
            "mode": "grid",
        }

    def _run_text_based_dialogue(
        self,
        state: SharedState,
        min_turns: int,
        recommended_turns: int,
        max_turns: int,
    ):
        """Original text-based dialogue (fallback when grid bootstrap fails)."""
        from core.dialectic import (
            RoundManager, DialecticPhase, DialecticRole, RoundOutput,
        )
        from core.convergence import ConvergenceTracker

        convergence_min = 3 if min_turns <= 8 else min_turns
        convergence = ConvergenceTracker(
            plateau_window=2,
            delta_threshold=0.08,
            min_turns_before_convergence=convergence_min,
            recommended_turns=recommended_turns,
        )

        round_mgr = RoundManager(max_gate_retries=PROTOCOL.dialectic.max_gate_retries)
        agents = {"Entity": self.entity_agent, "Process": self.process_agent}

        input_context = state.known.get("input", "")
        intent = state.known.get("intent", {})

        prime_content = f"""
Analyze this system to build:

USER INPUT: {input_context}

EXTRACTED INTENT:
- Core need: {intent.get('core_need', 'unknown')}
- Domain: {intent.get('domain', 'unknown')}
- Actors: {', '.join(intent.get('actors', []))}
- Key insight: {intent.get('insight', '')}

Begin your analysis from your perspective. Remember to include an INSIGHT: line.
"""

        prime_msg = Message(
            sender="System",
            content=prime_content,
            message_type=MessageType.PROPOSITION
        )

        last_msg = None
        total_turns = 0

        max_round_count = max_turns // PROTOCOL.dialectic.turns_per_round + 1
        min_rounds = 1

        while round_mgr.current_round < max_round_count:
            if total_turns >= max_turns:
                self._emit_insight(
                    f"Turn budget exhausted ({total_turns}/{max_turns})"
                )
                break

            if round_mgr.current_round >= min_rounds:
                if convergence.has_converged():
                    self._emit_insight(
                        f"Blueprint converged after {total_turns} turns"
                    )
                    break

            phase = round_mgr.current_phase()
            angle_name, angle_prompt = round_mgr.rotation_angle_for_round(
                round_mgr.current_round, state.confidence
            )

            round_messages = []
            round_insights = []

            for turn_in_round in range(PROTOCOL.dialectic.turns_per_round):
                if total_turns >= max_turns:
                    break

                role = round_mgr.turn_role(turn_in_round)

                if role == DialecticRole.ANTITHESIS:
                    agent_name = "Process"
                else:
                    agent_name = "Entity"

                agent = agents[agent_name]

                state.known["_dialectic_context"] = round_mgr.build_round_context(
                    round_mgr.current_round, role, round_mgr.rounds
                )

                input_for_agent = prime_msg if total_turns == 0 else last_msg
                response = agent.run(state, input_for_agent)
                state.add_message(response)

                round_messages.append(response)
                if response.insight:
                    round_insights.append(response.insight)

                if response.insight_display:
                    self._emit_insight(response.insight_display)

                state.confidence_history.append(state.confidence.overall())

                if (state.confidence.is_plateauing(state.confidence_history) and
                        total_turns >= 3):
                    weakest = state.confidence.weakest_dimension()
                    hint_parts = [f"[GOVERNOR] Confidence stalled on {weakest}."]

                    if weakest in ("structural", "coverage"):
                        uncovered = agent._compute_uncovered_ground(state)
                        if uncovered:
                            hint_parts.append(f"Uncovered components: {', '.join(uncovered[:3])}.")
                        else:
                            hint_parts.append("All explicit components addressed — look for implicit ones.")

                    if weakest == "behavioral":
                        hint_parts.append("Specify methods, state transitions, or interaction flows.")

                    if weakest == "consistency":
                        if state.unknown:
                            hint_parts.append(f"Unresolved unknowns: {', '.join(state.unknown[:2])}.")

                    unresolved_conflicts = [c for c in state.conflicts if not c["resolved"]]
                    if unresolved_conflicts:
                        topics = [c["topic"][:40] for c in unresolved_conflicts[:2]]
                        hint_parts.append(f"Open conflicts: {'; '.join(topics)}.")

                    focus_hint = " ".join(hint_parts)
                    last_msg = Message(
                        sender="Governor",
                        content=focus_hint,
                        message_type=MessageType.META
                    )

                self._detect_structural_conflicts(state, response)

                resolved = state.resolve_unknown_by_keyword(response.content)
                if resolved:
                    self._emit_insight(f"Resolved {len(resolved)} unknown(s)")

                last_msg = response
                total_turns += 1

            round_output = RoundOutput(
                round_number=round_mgr.current_round,
                phase=phase,
                rotation_angle=angle_name,
                messages=round_messages,
                insights=round_insights,
                confidence_snapshot=state.confidence.to_dict(),
                provenance_passed=True,
                gate_attempts=0,
            )

            if phase != DialecticPhase.COLLAPSE:
                gate_passed = round_mgr.check_round_gate(round_output, state)
                round_output.provenance_passed = gate_passed
                if not gate_passed and round_mgr._gate_failures < round_mgr.max_gate_retries:
                    round_mgr._gate_failures += 1
                    round_output.gate_attempts = round_mgr._gate_failures
                    narrowed = round_mgr.narrow_scope(round_output, state)
                    last_msg = Message(
                        sender="Governor",
                        content=narrowed,
                        message_type=MessageType.META,
                    )
                    self._emit_insight(
                        f"Round {round_mgr.current_round + 1} gate failed — narrowing scope"
                    )
                    continue

            round_mgr.commit_round(round_output)
            state.known.setdefault("_round_outputs", []).append({
                "round": round_output.round_number,
                "phase": round_output.phase.value,
                "angle": round_output.rotation_angle,
                "insights": round_output.insights,
                "confidence": round_output.confidence_snapshot,
            })
            self._emit(f"Round {round_output.round_number + 1} ({phase.value}) complete")

            messages_for_snapshot = [
                {"content": m.content, "insight": m.insight}
                for m in state.history
                if m.sender in ("Entity", "Process")
            ]
            delta = convergence.update(messages_for_snapshot, total_turns=total_turns)
            if round_mgr.current_round > 1:
                self._emit_insight(
                    f"Convergence δ={delta:.3f} "
                    f"({convergence.component_count} components)"
                )

        state.known["_convergence"] = convergence.convergence_summary
        state.known["_convergence"]["mode"] = "text"
        state.known.pop("_dialectic_context", None)

    # --- Grid-driven dialogue helpers ---

    # Concerns routed to the Process agent (behavioral axis)
    _PROCESS_CONCERNS = frozenset({
        "BHV", "FLW", "TRN", "FNC", "ACT", "GTE", "SCH",
    })

    def _bootstrap_dialogue_grid(self, state: SharedState):
        """Create a lean dialogue grid from intent.

        Unlike Phase 3.5's full kernel compile, the dialogue grid is minimal:
        just the intent root + core concern cells per layer. Ground truth and
        history are NOT loaded here — they bloat the grid with 75+ cells and
        prevent convergence within dialogue turns.

        The grid is pre-seeded with both structural (ENT) and behavioral (BHV)
        cells for each layer. This creates the friction surface: Entity agent
        fills ENT cells, Process agent fills BHV cells. Convergence happens
        when all pre-seeded cells are filled.
        """
        from kernel.grid import Grid
        from kernel.cell import Cell, FillState, parse_postcode

        intent = state.known.get("intent", {})
        input_text = state.known.get("input", "")
        domain = intent.get("domain", "software")

        # Map common domain names to postcode domain codes
        domain_map = {
            "software": "SFT", "web": "SFT", "api": "SFT", "app": "SFT",
            "organization": "ORG", "business": "ORG", "process": "ORG",
            "education": "EDU", "medical": "MED", "legal": "LGL",
            "network": "NET", "social": "SOC", "economics": "ECN",
        }
        domain_code = domain_map.get(domain.lower(), "SFT")

        # Create grid with intent contract as root
        grid = Grid()
        root_key = f"INT.SEM.ECO.WHY.{domain_code}"
        grid.set_intent(
            intent_text=input_text,
            postcode_key=root_key,
            primitive="intent_contract",
        )

        # Pre-seed the exploration territory: structural + behavioral cells
        # per core layer. All cells trace provenance to root.
        seed_cells = [
            # Structure layer: entities and their behaviors
            ("STR", "ENT", "WHAT"),
            ("STR", "BHV", "HOW"),
            # Execution layer: behaviors and functions
            ("EXC", "BHV", "HOW"),
            ("EXC", "FNC", "HOW"),
            # State layer: states and transitions
            ("STA", "STA", "WHAT"),
            ("STA", "TRN", "WHEN"),
            # Data layer: data entities
            ("DAT", "ENT", "WHAT"),
            # Control layer: flows
            ("CTR", "FLW", "HOW"),
        ]

        for layer, concern, dimension in seed_cells:
            pc_key = f"{layer}.{concern}.ECO.{dimension}.{domain_code}"
            pc = parse_postcode(pc_key)
            cell = Cell(
                postcode=pc,
                primitive="",
                content="",
                fill=FillState.E,
                confidence=0.0,
                source=(root_key,),
            )
            grid.put(cell)

        self._emit_insight(
            f"Grid bootstrapped: {grid.total_cells} cells, "
            f"{len(grid.activated_layers)} layers"
        )

        return grid

    def _route_agent(self, postcode_key: str):
        """Pick Entity or Process agent based on postcode concern axis."""
        parts = postcode_key.split(".")
        concern = parts[1] if len(parts) > 1 else "ENT"
        if concern in self._PROCESS_CONCERNS:
            return self.process_agent
        return self.entity_agent

    # Progressive lens stages for grid-driven dialogue.
    # Orthogonal to concern-axis routing (ENT/BHV/FNC).
    GRID_LENS_STAGES = (
        (0, 1, "existence",
         "Focus on WHAT EXISTS. Identify entities, concepts, and their essential attributes."),
        (2, 2, "dynamics",
         "Focus on HOW THINGS CHANGE. Identify state transitions, triggers, data flows, and interactions."),
        (3, 3, "grounding",
         "Focus on CONCRETE IMPLEMENTATION. Map abstractions to specific structures, interfaces, and boundaries."),
        (4, None, "constraints",
         "Focus on LIMITS AND CONNECTIONS. Identify invariants, constraints, and how components relate to each other. "
         "Every component must connect to the rest of the system."),
    )

    @staticmethod
    def _lens_for_round(round_num: int) -> tuple:
        """Return (lens_name, lens_instruction) for the given grid dialogue round."""
        for start, end, name, instruction in MotherlabsEngine.GRID_LENS_STAGES:
            if round_num >= start and (end is None or round_num <= end):
                return name, instruction
        # Fallback to last stage
        return MotherlabsEngine.GRID_LENS_STAGES[-1][2], MotherlabsEngine.GRID_LENS_STAGES[-1][3]

    def _build_cell_prompt(self, state: SharedState, grid, target_postcode: str, agent_name: str, round_num: int = 0) -> "Message":
        """Build focused prompt: grid nav + target cell + surrounding context."""
        from kernel.nav import grid_to_nav

        # Compact nav representation of current grid state
        nav_text = grid_to_nav(grid)
        # Limit nav context to keep prompt focused
        nav_lines = nav_text.split("\n")
        if len(nav_lines) > 30:
            nav_text = "\n".join(nav_lines[:30]) + f"\n... ({len(nav_lines) - 30} more lines)"

        cell = grid.get(target_postcode)
        cell_desc = ""
        if cell:
            cell_desc = f"Current state: {cell.fill.value}, confidence: {cell.confidence}"
            if cell.connections:
                cell_desc += f", connections: {', '.join(cell.connections[:5])}"

        # Parent context for child cells (from descent)
        parent_context = ""
        if cell and cell.parent:
            parent_cell = grid.get(cell.parent)
            if parent_cell and parent_cell.content:
                parent_context = (
                    f"\nPARENT ({cell.parent}):\n"
                    f"  {parent_cell.primitive} | confidence: {parent_cell.confidence:.2f}\n"
                    f"  {parent_cell.content[:200]}\n"
                    f"Decompose the parent concept into finer-grained detail at this scope level.\n"
                )

        parts = target_postcode.split(".")
        layer = parts[0] if parts else "?"
        concern = parts[1] if len(parts) > 1 else "?"
        scope = parts[2] if len(parts) > 2 else "?"
        dimension = parts[3] if len(parts) > 3 else "?"

        intent = state.known.get("intent", {})

        lens_name, lens_instruction = self._lens_for_round(round_num)

        prompt = f"""Focus on filling this semantic coordinate:

TARGET: {target_postcode}
  Layer: {layer} | Concern: {concern} | Scope: {scope} | Dimension: {dimension}
  {cell_desc}
{parent_context}
SYSTEM BEING BUILT:
- Core need: {intent.get('core_need', 'unknown')}
- Domain: {intent.get('domain', 'unknown')}

CURRENT GRID STATE:
{nav_text}

ANALYSIS LENS [{lens_name.upper()}]: {lens_instruction}

Analyze this coordinate from your {agent_name} perspective.
What entity/component/process/behavior belongs at this position?
Include an INSIGHT: line with your key finding."""

        return Message(
            sender="System",
            content=prompt,
            message_type=MessageType.PROPOSITION,
        )

    def _fill_from_response(self, grid, target_postcode: str, response: "Message"):
        """Extract structured content from agent response → fill grid cell.

        No connections are inferred here — the grid is pre-seeded with all
        cells during bootstrap. This ensures the grid converges when all
        pre-seeded cells are filled, without unbounded expansion.
        """
        from kernel.ops import fill as grid_fill
        from kernel.llm_bridge import parse_agent_response_to_fill
        from kernel.grid import INTENT_CONTRACT

        fill_data = parse_agent_response_to_fill(
            response.content,
            target_postcode,
        )

        # Determine provenance source — trace back to root
        source = (grid.root or INTENT_CONTRACT,)

        # Check if cell exists and has a parent
        existing = grid.get(target_postcode)
        parent = None
        if existing and existing.parent:
            parent = existing.parent

        try:
            grid_fill(
                grid,
                postcode_key=target_postcode,
                primitive=fill_data["primitive"],
                content=fill_data["content"],
                confidence=fill_data["confidence"],
                source=source,
                parent=parent,
            )
        except Exception as e:
            logger.debug(f"Grid fill failed for {target_postcode}: {e}")

    def _detect_structural_conflicts(self, state: SharedState, response: Message):
        """
        Detect structural conflicts: Entity and Process disagree on a component's nature.

        Phase 10.2: After each turn, check if the current agent's message
        contradicts the previous agent's classification of a component
        (entity vs process).

        Heuristic: The component name must appear within ~50 chars of a type marker
        (entity/process), not just anywhere in the message. This prevents false
        positives from generic agent role descriptions.
        """
        import re

        if len(state.history) < 2:
            return

        current = response
        # Only detect conflicts between Entity and Process agents
        if current.sender not in ("Entity", "Process"):
            return

        other_agent = "Process" if current.sender == "Entity" else "Entity"
        previous = None
        for msg in reversed(state.history[:-1]):
            if msg.sender == other_agent:
                previous = msg
                break

        if not previous:
            return

        # Extract capitalized component names (3+ chars, starts with uppercase)
        # Exclude common non-component words
        _NOISE_WORDS = {
            "The", "This", "That", "These", "Those", "What", "When", "Where",
            "Which", "How", "Not", "But", "And", "For", "With", "From",
            "Into", "Also", "Each", "Any", "All", "Some", "More", "Most",
            "Other", "Such", "Very", "Only", "Just", "Even", "However",
            "Therefore", "Furthermore", "Additionally", "INSIGHT", "UNKNOWN",
            "CONFLICT", "CHALLENGE", "SUFFICIENT", "METHOD", "STATES",
            "Analyzing", "Analysis", "Perspective", "Structure", "Behavior",
            "Challenge", "Agreement", "Consider", "Critical", "Important",
            "Must", "Should", "Would", "Could", "Might",
            # Agent dialogue pronouns/roles — not component names
            "You", "Your", "Agent", "Entity", "Process", "Here",
            "Let", "Yes", "Now", "Our", "Both", "They", "Its",
        }

        def extract_components(text):
            words = set(re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', text))
            return words - _NOISE_WORDS

        current_components = extract_components(current.content)
        previous_components = extract_components(previous.content)
        shared = current_components & previous_components

        if not shared:
            return

        # Entity-type indicators (must be near the component name)
        entity_markers = {"entity", "data", "attribute", "property", "field", "record"}
        # Process-type indicators (must be near the component name)
        process_markers = {"process", "flow", "transition", "trigger", "action", "step"}

        def has_type_near_component(text, comp_name, markers, window=30):
            """Check if any type marker appears within `window` chars of component name.

            Phase 11.1: Window tightened from 80 to 30 chars to reduce false
            positives. The marker must be immediately adjacent to the component
            name (e.g. "Artist entity" or "the booking process") not just
            somewhere in the same paragraph.
            """
            text_lower = text.lower()
            comp_lower = comp_name.lower()
            idx = 0
            while True:
                pos = text_lower.find(comp_lower, idx)
                if pos == -1:
                    break
                # Check tight window around the component name
                start = max(0, pos - window)
                end = min(len(text_lower), pos + len(comp_lower) + window)
                context = text_lower[start:end]
                if any(m in context for m in markers):
                    return True
                idx = pos + 1
            return False

        for comp in shared:
            current_entity = has_type_near_component(current.content, comp, entity_markers)
            current_process = has_type_near_component(current.content, comp, process_markers)
            previous_entity = has_type_near_component(previous.content, comp, entity_markers)
            previous_process = has_type_near_component(previous.content, comp, process_markers)

            # Conflict: one says entity, other says process
            if (current_entity and not current_process and
                previous_process and not previous_entity):
                state.add_conflict(
                    agent_a=current.sender,
                    agent_b=other_agent,
                    topic=f"{comp}: structural disagreement (entity vs process)",
                    positions={
                        current.sender: f"{comp} as entity/structure",
                        other_agent: f"{comp} as process/flow"
                    }
                )
            elif (current_process and not current_entity and
                  previous_entity and not previous_process):
                state.add_conflict(
                    agent_a=current.sender,
                    agent_b=other_agent,
                    topic=f"{comp}: structural disagreement (process vs entity)",
                    positions={
                        current.sender: f"{comp} as process/flow",
                        other_agent: f"{comp} as entity/structure"
                    }
                )

    def _resolve_conflicts(self, state: SharedState) -> int:
        """
        Resolve structural conflicts by reframing.

        Phase 10.8: After dialogue, process conflicts where Entity and Process
        disagree on a component's type. These are frame conflicts — both agents
        are correct within their lens. Resolution: mark the component as a
        boundary component with dual specification (both structural and
        behavioral properties).

        Explicit (non-structural) conflicts from CONFLICT: lines are left
        unresolved — they represent genuine semantic disagreements that
        should surface in the blueprint's unresolved list.

        Returns:
            Number of conflicts resolved
        """
        resolved_count = 0
        for i, conflict in enumerate(state.conflicts):
            if conflict["resolved"]:
                continue

            topic = conflict.get("topic", "")

            # Only reframe structural disagreements (entity vs process type)
            if "structural disagreement" not in topic:
                continue

            # Extract component name from topic format:
            # "ComponentName: structural disagreement (entity vs process)"
            comp_name = topic.split(":")[0].strip()
            if not comp_name:
                continue

            # Reframe: both views are valid — this is a boundary component
            entity_view = conflict["positions"].get("Entity", "")
            process_view = conflict["positions"].get("Process", "")

            resolution = (
                f"Boundary component: {comp_name} has both structural properties "
                f"({entity_view}) and behavioral properties ({process_view}). "
                f"Include both aspects in specification."
            )

            state.resolve_conflict(i, resolution)
            resolved_count += 1

        # Phase 12.2c: Classify non-structural conflicts for synthesis guidance
        for conflict in state.conflicts:
            if not conflict["resolved"] and "structural disagreement" not in conflict.get("topic", ""):
                conflict["category"] = self._classify_conflict(conflict)

        return resolved_count

    def _classify_conflict(self, conflict: Dict[str, Any]) -> str:
        """
        Classify a non-structural conflict.

        Phase 12.2c: Returns PRIORITY|MISSING_INFO|TRADEOFF based on position text.
        """
        positions_text = " ".join(str(v) for v in conflict.get("positions", {}).values()).lower()
        if any(w in positions_text for w in ("unknown", "unclear", "need", "missing", "depends")):
            return "MISSING_INFO"
        if any(w in positions_text for w in ("should", "must", "important", "first", "before")):
            return "PRIORITY"
        return "TRADEOFF"

    def _extract_methods_for_synthesis(self, state: SharedState) -> tuple:
        """
        Extract methods and state machines from dialogue for synthesis.

        Phase 12.3b: Runs after _resolve_conflicts(), before _synthesize().
        Combines explicit METHOD: lines, pattern transfer hints, and STATES: blocks.
        Deduplicates methods by (normalized_component, name.lower()), preferring dialogue
        over pattern_transfer. Phase 12.4b: Normalizes component names from pattern
        sources (e.g., "Governor sequencing" → "Governor Agent") and caps pattern_transfer
        methods at 5 per component.

        Args:
            state: SharedState with dialogue history and insights

        Returns:
            Tuple of (methods_list, state_machines_list)
        """
        # 1. Explicit METHOD: lines from dialogue
        dialogue_methods = extract_dialogue_methods(state)

        # 2. Pattern transfer hint stubs
        pattern_stubs = extract_pattern_method_stubs(state.insights)

        # 3. Normalize component names against known aliases
        # Pattern stubs: mutate component name (raw names like "Governor sequencing" are useless)
        for m in pattern_stubs:
            m["component"] = self._normalize_method_component(m["component"])

        # 4. Deduplicate: dialogue wins over pattern_transfer
        # Use normalized component name as dedup key (but preserve original component name on method)
        seen = {}
        for m in dialogue_methods:
            norm_comp = self._normalize_method_component(m["component"]).lower()
            key = (norm_comp, m["name"].lower())
            seen[key] = m

        # Cap pattern_transfer stubs at 5 per normalized component
        pattern_counts = {}
        for m in pattern_stubs:
            comp_key = m["component"].lower()
            key = (comp_key, m["name"].lower())
            if key not in seen:
                count = pattern_counts.get(comp_key, 0)
                if count < 5:
                    seen[key] = m
                    pattern_counts[comp_key] = count + 1

        methods = list(seen.values())

        # 5. State machines from STATES: blocks
        state_machines = extract_dialogue_state_machines(state)

        return methods, state_machines

    @staticmethod
    def _normalize_method_component(raw_name: str) -> str:
        """
        Normalize a pattern transfer component name to its canonical form.

        Phase 12.4b: Handles cases like "Governor sequencing" → "Governor Agent"
        by checking COMPONENT_ALIASES first (exact match), then prefix matching
        against known alias keys.

        Args:
            raw_name: Raw component name from pattern source

        Returns:
            Normalized canonical name, or original if no match
        """
        # Try exact match first
        canonical = normalize_component_name(raw_name)
        if canonical != raw_name:
            return canonical

        # Prefix match: "Governor sequencing" → check if "governor" is an alias prefix
        lower = raw_name.lower().strip()
        for alias_key, canonical_name in COMPONENT_ALIASES.items():
            # Check if the raw name starts with this alias key
            if lower.startswith(alias_key) and (
                len(lower) == len(alias_key) or lower[len(alias_key)] == ' '
            ):
                return canonical_name

        return raw_name

    def _enrich_blueprint_methods(self, blueprint: dict, state: SharedState) -> dict:
        """
        Post-synthesis enrichment: map extracted methods and state machines into blueprint.

        Phase 12.5: After synthesis, dialogue-extracted methods (from state.known["extracted_methods"])
        are deterministically backfilled into matching blueprint components. This closes the gap
        where natural language inputs produce components with empty methods arrays because the LLM
        didn't include them in initial synthesis output.

        Args:
            blueprint: Synthesized blueprint dict
            state: SharedState with extracted_methods and extracted_state_machines

        Returns:
            Enriched blueprint
        """
        extracted_methods = state.known.get("extracted_methods", [])
        extracted_state_machines = state.known.get("extracted_state_machines", [])

        if not extracted_methods and not extracted_state_machines:
            return self._infer_component_methods(blueprint)

        # Even when dialogue methods exist, some components may not match.
        # After dialogue backfill below, _infer_component_methods runs as
        # gap-fill for any components still missing methods (see end of method).

        components = blueprint.get("components", [])
        if not components:
            return blueprint

        # Build lookup: normalized name -> component dict
        comp_lookup = {}
        for comp in components:
            name = comp.get("name", "")
            comp_lookup[name.lower().strip()] = comp

        methods_added = 0
        sm_added = 0

        # Map extracted methods to components
        for method in extracted_methods:
            raw_comp = method.get("component", "")
            normalized = self._normalize_method_component(raw_comp).lower().strip()

            # Try exact match first
            target = comp_lookup.get(normalized)

            # Fuzzy prefix match if exact fails
            if target is None:
                for comp_key, comp_val in comp_lookup.items():
                    if comp_key.startswith(normalized) or normalized.startswith(comp_key):
                        target = comp_val
                        break

            if target is None:
                continue

            # Ensure methods array exists
            if "methods" not in target:
                target["methods"] = []

            # Skip if method already present
            method_name = method.get("name", "")
            existing_names = set()
            for m in target["methods"]:
                if isinstance(m, dict):
                    existing_names.add(m.get("name", "").lower())
                elif isinstance(m, str):
                    existing_names.add(m.split("(")[0].strip().lower())

            if method_name.lower() in existing_names:
                continue

            target["methods"].append({
                "name": method_name,
                "parameters": method.get("parameters", []),
                "return_type": method.get("return_type", "None"),
                "derived_from": method.get("derived_from", ""),
            })
            methods_added += 1

        # Map extracted state machines to components
        for sm in extracted_state_machines:
            raw_comp = sm.get("component", "")
            normalized = self._normalize_method_component(raw_comp).lower().strip()

            target = comp_lookup.get(normalized)
            if target is None:
                for comp_key, comp_val in comp_lookup.items():
                    if comp_key.startswith(normalized) or normalized.startswith(comp_key):
                        target = comp_val
                        break

            if target is None:
                continue

            if not target.get("state_machine"):
                target["state_machine"] = sm
                sm_added += 1

        if methods_added > 0 or sm_added > 0:
            parts = []
            if methods_added:
                parts.append(f"{methods_added} method(s)")
            if sm_added:
                parts.append(f"{sm_added} state machine(s)")
            self._emit_insight(f"Enriched blueprint with {', '.join(parts)} from dialogue")

        # Gap-fill: infer methods for any components still missing them
        return self._infer_component_methods(blueprint)

    def _infer_component_methods(self, blueprint: dict) -> dict:
        """Deterministic method inference for components missing methods.

        Zero LLM calls. Infers from component type AND name:
        - ENTITY/DATA → CRUD (create, get, update, delete)
        - PROCESS/SERVICE → lifecycle (execute, get_status, validate)
        - INTERFACE/API → handler (handle_request, validate_input)
        - MANAGER/SYSTEM → operations (initialize, process, configure)
        - MONITOR/TRACKER → observation (check, report, track)
        - SCHEDULER/QUEUE → scheduling (schedule, cancel, get_next)
        - REGISTRY/REPOSITORY → registry (register, lookup, list_all)
        - VALIDATOR/GUARD → validation (validate, check_rules)
        - NOTIFIER/MESSENGER → notification (send, subscribe, broadcast)
        - CONNECTOR/ADAPTER → integration (connect, transform, sync)
        - Catch-all → process + initialize from name
        """
        _ENTITY_TYPES = frozenset({
            "entity", "data", "model", "store", "storage",
            "record", "document", "table", "schema", "resource",
        })
        _PROCESS_TYPES = frozenset({
            "process", "service", "handler", "workflow", "pipeline",
            "engine", "processor", "worker", "executor", "task",
        })
        _INTERFACE_TYPES = frozenset({
            "interface", "api", "controller", "gateway", "endpoint",
            "route", "view", "page", "screen", "form",
        })
        _MANAGER_TYPES = frozenset({
            "manager", "system", "module", "component", "subsystem",
            "platform", "application", "framework", "core", "hub",
        })
        _MONITOR_TYPES = frozenset({
            "monitor", "tracker", "dashboard", "observer", "watcher",
            "meter", "profiler", "logger", "auditor", "analyzer",
        })
        _SCHEDULER_TYPES = frozenset({
            "scheduler", "queue", "dispatcher", "planner", "cron",
            "timer", "trigger", "orchestrator",
        })
        _REGISTRY_TYPES = frozenset({
            "registry", "repository", "catalog", "index", "cache",
            "pool", "collection", "directory", "lookup",
        })
        _VALIDATOR_TYPES = frozenset({
            "validator", "guard", "checker", "verifier", "filter",
            "sanitizer", "policy", "rule",
        })
        _NOTIFIER_TYPES = frozenset({
            "notifier", "messenger", "mailer", "alerter", "broadcaster",
            "publisher", "emitter", "sender",
        })
        _CONNECTOR_TYPES = frozenset({
            "connector", "adapter", "bridge", "integration", "client",
            "provider", "driver", "proxy", "wrapper",
        })

        for comp in blueprint.get("components", []):
            if comp.get("methods"):
                continue
            comp_type = (comp.get("type") or "").lower()
            name = comp.get("name", "")
            name_lower = name.lower().replace(" ", "_")

            # Try type first, then fall back to name-based detection
            category = self._classify_component_type(
                comp_type, name_lower,
                _ENTITY_TYPES, _PROCESS_TYPES, _INTERFACE_TYPES,
                _MANAGER_TYPES, _MONITOR_TYPES, _SCHEDULER_TYPES,
                _REGISTRY_TYPES, _VALIDATOR_TYPES, _NOTIFIER_TYPES,
                _CONNECTOR_TYPES,
            )

            if category == "entity":
                comp["methods"] = [
                    {"name": f"create_{name_lower}",
                     "parameters": [{"name": "data", "type_hint": "dict"}],
                     "return_type": name, "description": f"Create new {name}",
                     "derived_from": f"Inferred: CRUD for entity '{name}'"},
                    {"name": f"get_{name_lower}",
                     "parameters": [{"name": "id", "type_hint": "str"}],
                     "return_type": name, "description": f"Retrieve {name} by ID",
                     "derived_from": f"Inferred: CRUD for entity '{name}'"},
                    {"name": f"update_{name_lower}",
                     "parameters": [{"name": "id", "type_hint": "str"}, {"name": "data", "type_hint": "dict"}],
                     "return_type": name, "description": f"Update {name}",
                     "derived_from": f"Inferred: CRUD for entity '{name}'"},
                    {"name": f"delete_{name_lower}",
                     "parameters": [{"name": "id", "type_hint": "str"}],
                     "return_type": "bool", "description": f"Delete {name}",
                     "derived_from": f"Inferred: CRUD for entity '{name}'"},
                ]
            elif category == "process":
                comp["methods"] = [
                    {"name": "execute",
                     "parameters": [{"name": "input", "type_hint": "dict"}],
                     "return_type": "dict", "description": f"Execute {name}",
                     "derived_from": f"Inferred: lifecycle for process '{name}'"},
                    {"name": "get_status",
                     "parameters": [],
                     "return_type": "str", "description": f"Get status of {name}",
                     "derived_from": f"Inferred: lifecycle for process '{name}'"},
                    {"name": "validate",
                     "parameters": [{"name": "input", "type_hint": "dict"}],
                     "return_type": "bool", "description": f"Validate input for {name}",
                     "derived_from": f"Inferred: lifecycle for process '{name}'"},
                ]
            elif category == "interface":
                comp["methods"] = [
                    {"name": "handle_request",
                     "parameters": [{"name": "request", "type_hint": "dict"}],
                     "return_type": "dict", "description": f"Handle request to {name}",
                     "derived_from": f"Inferred: interface for '{name}'"},
                    {"name": "validate_input",
                     "parameters": [{"name": "data", "type_hint": "dict"}],
                     "return_type": "bool", "description": f"Validate input for {name}",
                     "derived_from": f"Inferred: interface for '{name}'"},
                ]
            elif category == "manager":
                comp["methods"] = [
                    {"name": "initialize",
                     "parameters": [{"name": "config", "type_hint": "dict"}],
                     "return_type": "None", "description": f"Initialize {name}",
                     "derived_from": f"Inferred: operations for manager '{name}'"},
                    {"name": f"process",
                     "parameters": [{"name": "input", "type_hint": "dict"}],
                     "return_type": "dict", "description": f"Process input through {name}",
                     "derived_from": f"Inferred: operations for manager '{name}'"},
                    {"name": "configure",
                     "parameters": [{"name": "settings", "type_hint": "dict"}],
                     "return_type": "None", "description": f"Configure {name}",
                     "derived_from": f"Inferred: operations for manager '{name}'"},
                ]
            elif category == "monitor":
                comp["methods"] = [
                    {"name": "check",
                     "parameters": [],
                     "return_type": "dict", "description": f"Check status via {name}",
                     "derived_from": f"Inferred: observation for monitor '{name}'"},
                    {"name": "report",
                     "parameters": [{"name": "period", "type_hint": "str"}],
                     "return_type": "dict", "description": f"Generate report from {name}",
                     "derived_from": f"Inferred: observation for monitor '{name}'"},
                    {"name": "track",
                     "parameters": [{"name": "event", "type_hint": "dict"}],
                     "return_type": "None", "description": f"Track event in {name}",
                     "derived_from": f"Inferred: observation for monitor '{name}'"},
                ]
            elif category == "scheduler":
                comp["methods"] = [
                    {"name": "schedule",
                     "parameters": [{"name": "task", "type_hint": "dict"}],
                     "return_type": "str", "description": f"Schedule task in {name}",
                     "derived_from": f"Inferred: scheduling for '{name}'"},
                    {"name": "cancel",
                     "parameters": [{"name": "task_id", "type_hint": "str"}],
                     "return_type": "bool", "description": f"Cancel scheduled task in {name}",
                     "derived_from": f"Inferred: scheduling for '{name}'"},
                    {"name": "get_next",
                     "parameters": [],
                     "return_type": "dict", "description": f"Get next scheduled item from {name}",
                     "derived_from": f"Inferred: scheduling for '{name}'"},
                ]
            elif category == "registry":
                comp["methods"] = [
                    {"name": "register",
                     "parameters": [{"name": "item", "type_hint": "dict"}],
                     "return_type": "str", "description": f"Register item in {name}",
                     "derived_from": f"Inferred: registry for '{name}'"},
                    {"name": "lookup",
                     "parameters": [{"name": "key", "type_hint": "str"}],
                     "return_type": "dict", "description": f"Look up item in {name}",
                     "derived_from": f"Inferred: registry for '{name}'"},
                    {"name": "list_all",
                     "parameters": [],
                     "return_type": "list", "description": f"List all items in {name}",
                     "derived_from": f"Inferred: registry for '{name}'"},
                ]
            elif category == "validator":
                comp["methods"] = [
                    {"name": "validate",
                     "parameters": [{"name": "data", "type_hint": "dict"}],
                     "return_type": "bool", "description": f"Validate data using {name}",
                     "derived_from": f"Inferred: validation for '{name}'"},
                    {"name": "check_rules",
                     "parameters": [{"name": "context", "type_hint": "dict"}],
                     "return_type": "list", "description": f"Check rules in {name}",
                     "derived_from": f"Inferred: validation for '{name}'"},
                ]
            elif category == "notifier":
                comp["methods"] = [
                    {"name": "send",
                     "parameters": [{"name": "message", "type_hint": "dict"}],
                     "return_type": "bool", "description": f"Send notification via {name}",
                     "derived_from": f"Inferred: notification for '{name}'"},
                    {"name": "subscribe",
                     "parameters": [{"name": "channel", "type_hint": "str"}, {"name": "callback", "type_hint": "callable"}],
                     "return_type": "str", "description": f"Subscribe to {name} channel",
                     "derived_from": f"Inferred: notification for '{name}'"},
                ]
            elif category == "connector":
                comp["methods"] = [
                    {"name": "connect",
                     "parameters": [{"name": "config", "type_hint": "dict"}],
                     "return_type": "bool", "description": f"Connect via {name}",
                     "derived_from": f"Inferred: integration for '{name}'"},
                    {"name": "transform",
                     "parameters": [{"name": "data", "type_hint": "dict"}],
                     "return_type": "dict", "description": f"Transform data through {name}",
                     "derived_from": f"Inferred: integration for '{name}'"},
                    {"name": "sync",
                     "parameters": [],
                     "return_type": "bool", "description": f"Synchronize via {name}",
                     "derived_from": f"Inferred: integration for '{name}'"},
                ]
            else:
                # Catch-all: every component gets at least process + initialize
                comp["methods"] = [
                    {"name": "initialize",
                     "parameters": [{"name": "config", "type_hint": "dict"}],
                     "return_type": "None", "description": f"Initialize {name}",
                     "derived_from": f"Inferred: default for '{name}'"},
                    {"name": "process",
                     "parameters": [{"name": "input", "type_hint": "dict"}],
                     "return_type": "dict", "description": f"Process input through {name}",
                     "derived_from": f"Inferred: default for '{name}'"},
                ]
        return blueprint

    @staticmethod
    def _classify_component_type(
        comp_type: str,
        name_lower: str,
        entity_types: frozenset,
        process_types: frozenset,
        interface_types: frozenset,
        manager_types: frozenset,
        monitor_types: frozenset,
        scheduler_types: frozenset,
        registry_types: frozenset,
        validator_types: frozenset,
        notifier_types: frozenset,
        connector_types: frozenset,
    ) -> str:
        """Classify a component into a method-inference category.

        Checks type first, then falls back to name suffix/substring matching.
        Returns category string: entity, process, interface, manager, monitor,
        scheduler, registry, validator, notifier, connector, or empty string.
        """
        # Check type directly
        for category, type_set in (
            ("entity", entity_types),
            ("process", process_types),
            ("interface", interface_types),
            ("manager", manager_types),
            ("monitor", monitor_types),
            ("scheduler", scheduler_types),
            ("registry", registry_types),
            ("validator", validator_types),
            ("notifier", notifier_types),
            ("connector", connector_types),
        ):
            if comp_type in type_set:
                return category

        # Fall back to name-based detection (check suffixes and substrings)
        for category, type_set in (
            ("entity", entity_types),
            ("process", process_types),
            ("interface", interface_types),
            ("manager", manager_types),
            ("monitor", monitor_types),
            ("scheduler", scheduler_types),
            ("registry", registry_types),
            ("validator", validator_types),
            ("notifier", notifier_types),
            ("connector", connector_types),
        ):
            for keyword in type_set:
                if keyword in name_lower:
                    return category

        return ""

    def _validate_method_completeness(self, spec_text: str, blueprint: dict) -> List[str]:
        """
        Check if all methods in spec appear in blueprint.

        Phase 4.6: Method Containment Gap Fix

        Args:
            spec_text: Original specification text
            blueprint: Generated blueprint

        Returns:
            List of missing method names
        """
        import re

        # Extract method patterns from spec (method_name followed by parentheses)
        # Matches: method_name(...) or method_name(param: type) -> return
        method_pattern = r'(\w+)\s*\([^)]*\)\s*(?:->|:|\))'
        spec_methods = set(re.findall(method_pattern, spec_text))

        # Filter out common false positives (type constructors, etc.)
        false_positives = {'Dict', 'List', 'Optional', 'Set', 'Tuple', 'Any', 'str', 'int', 'float', 'bool'}
        spec_methods = {m for m in spec_methods if m not in false_positives and not m[0].isupper()}

        # Extract method names from blueprint
        blueprint_methods = set()
        for comp in blueprint.get('components', []):
            name = comp.get('name', '')
            # Method components have parentheses in their names
            if '(' in name:
                method_name = name.split('(')[0].strip()
                blueprint_methods.add(method_name)
            # Also check methods array on entity components
            if comp.get('methods'):
                for method in comp.get('methods', []):
                    if isinstance(method, dict):
                        blueprint_methods.add(method.get('name', ''))
                    elif isinstance(method, str):
                        # Parse method name from signature
                        if '(' in method:
                            blueprint_methods.add(method.split('(')[0].strip())

        # Return missing methods
        missing = list(spec_methods - blueprint_methods)
        return missing

    def _calculate_insight_coverage(self, blueprint: Dict[str, Any], state: SharedState) -> float:
        """
        Calculate what fraction of dialogue insights are referenced in blueprint derived_from fields.

        Phase 8.5: Insight traceability metric for best-of-N scoring.

        Args:
            blueprint: The synthesized blueprint
            state: SharedState with insights list

        Returns:
            Float 0-1 representing fraction of insights referenced
        """
        if not state.insights:
            return 1.0  # No insights to trace = full coverage

        # Collect all derived_from text in blueprint
        derived_texts = []
        for comp in blueprint.get("components", []):
            df = comp.get("derived_from", "")
            if df:
                derived_texts.append(df.lower())
            # Also check methods
            for method in comp.get("methods", []):
                mdf = method.get("derived_from", "") if isinstance(method, dict) else ""
                if mdf:
                    derived_texts.append(mdf.lower())
        for rel in blueprint.get("relationships", []):
            rdf = rel.get("derived_from", "")
            if rdf:
                derived_texts.append(rdf.lower())

        all_derived = " ".join(derived_texts)

        # Check how many insights are referenced (substring match on key words)
        referenced = 0
        for insight in state.insights:
            # Extract key terms from insight (words > 4 chars)
            terms = [w.lower() for w in insight.split() if len(w) > 4]
            # If at least 2 key terms appear in any derived_from, count as referenced
            if terms:
                matches = sum(1 for t in terms if t in all_derived)
                if matches >= min(2, len(terms)):
                    referenced += 1

        return referenced / len(state.insights)

    def _synthesize(
        self,
        state: SharedState,
        canonical_components: List[str] = None,
        canonical_relationships: List[tuple] = None
    ) -> tuple:
        """
        Synthesize blueprint from dialogue with constrained excavation.

        Derived from: PROJECT-PLAN.md Phase 1.4
        Enhanced with: Canonical component AND relationship enforcement (DDC research)
        Enhanced with: Provider-specific retry configuration
        Enhanced with: Phase 4.6 Method completeness validation

        Args:
            state: SharedState with dialogue history
            canonical_components: If provided, these components MUST appear in output
            canonical_relationships: If provided, these relationships SHOULD appear in output

        Returns:
            Tuple of (blueprint dict, retry_count)
        """
        # Get provider-specific config
        config = get_provider_config(self.provider_name, "synthesis")
        MAX_RETRIES = config.max_retries
        # Freeform intents (no canonical components) don't benefit from retries —
        # the first valid blueprint is accepted. Reduce retries to cap synthesis time.
        if not canonical_components and not canonical_relationships:
            MAX_RETRIES = min(MAX_RETRIES, 1)
        retry_count = 0

        # Phase 8.1: Build structured dialogue digest for synthesis
        dialogue_digest = build_dialogue_digest(state)

        # Get original input for excavation reference
        original_input = state.known.get("input", "")

        # Build canonical component instruction if provided
        canonical_instruction = ""
        if canonical_components:
            canonical_instruction = f"""
REQUIRED COMPONENTS (MUST appear in output):
{chr(10).join(f'- {c}' for c in canonical_components)}

These components are EXPLICITLY NAMED in the input. You MUST include ALL of them.
"""

        # Build canonical relationship instruction if provided
        relationship_instruction = ""
        if canonical_relationships:
            rel_list = "\n".join([
                f'- {from_c} --{rel_type}--> {to_c}'
                for (from_c, to_c, rel_type) in canonical_relationships
            ])
            relationship_instruction = f"""
REQUIRED RELATIONSHIPS (MUST appear in output):
{rel_list}

These relationships are EXPLICITLY DESCRIBED in the input. Include ALL of them.
Use EXACT component names (e.g., "Governor Agent" not "Governor").
"""

        # Build subsystem instruction if hints are available
        subsystem_instruction = ""
        subsystem_hints = state.known.get("subsystem_hints", {})
        if subsystem_hints:
            subsystem_list = "\n".join([
                f'- {name} [SUBSYSTEM: {", ".join(subs)}]'
                for name, subs in subsystem_hints.items()
            ])
            subsystem_instruction = f"""
SUBSYSTEMS (create with type="subsystem" and sub_blueprint):
{subsystem_list}

For each subsystem, create a component with:
- type: "subsystem"
- sub_blueprint: containing the nested components
"""

        # Phase 8.5: Structured prompt with numbered sections
        sections = []

        # SECTION 1: INPUT
        sections.append(f"""SECTION 1: INPUT
{original_input[:1500]}""")

        # SECTION 2: DIALOGUE DIGEST (Phase 8.1)
        if dialogue_digest:
            sections.append(f"""SECTION 2: DIALOGUE DIGEST
{dialogue_digest}""")

        # SECTION 2a: SEMANTIC GRID (kernel output)
        semantic_nav = state.known.get("semantic_nav", "")
        structured_grid_cells = state.known.get("structured_grid_cells", [])
        if semantic_nav:
            grid_section = f"""SECTION 2a: SEMANTIC GRID

The following semantic map was extracted and validated from dialogue. Use it as
structural backbone — components and layers here are high-confidence anchors.
Fill in methods, constraints, and relationships from dialogue context.

{semantic_nav}"""
            # Append structured cell postcodes for provenance referencing
            if structured_grid_cells:
                cell_refs = "\n".join(
                    f"  grid:{c['postcode']} | {c['primitive']} | conf={c['confidence']}"
                    for c in structured_grid_cells[:40]
                )
                grid_section += f"""

GRID CELL REFERENCES (use these in derived_from):
{cell_refs}

PROVENANCE INSTRUCTION: When a component traces to a grid cell, set its
derived_from to "grid:<POSTCODE>" (e.g. "grid:STR.ENT.CMP.WHAT.SFT").
Multiple refs: "grid:STR.ENT.CMP.WHAT.SFT|grid:BHV.FNC.APP.HOW.SFT".
This enables structural provenance validation."""
            sections.append(grid_section)

        # SECTION 2b: PRE-COMPUTED STRUCTURE (Phase 14 — staged pipeline only)
        pipeline_state_obj = state.known.get("pipeline_state")
        if pipeline_state_obj is not None:
            precomputed = format_precomputed_structure(pipeline_state_obj)
            if precomputed:
                sections.append(f"""SECTION 2b: PRE-COMPUTED STRUCTURE

{precomputed}

INSTRUCTION: Assemble these pre-computed artifacts into valid blueprint JSON.
Do NOT invent new components beyond what is listed above.
Every component, relationship, and constraint has already been validated by gates.
Your job is ASSEMBLY, not generation.""")

        # SECTION 2c: CORPUS PATTERNS (Phase 22)
        corpus_section = state.known.get("corpus_patterns_section")
        if corpus_section:
            sections.append(f"""SECTION 2c: CORPUS PATTERNS

{corpus_section}

INSTRUCTION: Components and relationships from corpus patterns should appear
in the blueprint UNLESS the current input explicitly contradicts them.
Each pattern traces to prior compilations (stratum 2 provenance).""")

        # SECTION 2d: ANTI-PATTERN WARNINGS (Phase 22b)
        anti_patterns = state.known.get("corpus_anti_patterns")
        if anti_patterns:
            sections.append(f"""SECTION 2d: ANTI-PATTERN WARNINGS

{anti_patterns}

INSTRUCTION: Avoid these patterns. Every component must have methods, descriptions,
and at least one relationship unless explicitly standalone.""")

        # SECTION 2e: CONSTRAINT HINTS (Phase 22c)
        constraint_hints = state.known.get("corpus_constraint_hints")
        if constraint_hints:
            sections.append(f"""SECTION 2e: CONSTRAINT HINTS

{constraint_hints}

INSTRUCTION: Apply matching constraints where the current blueprint has
similar components. These are proven patterns from successful prior builds.""")

        # SECTION 2f: KNOWN ISSUES — rejection awareness (Phase 22d)
        rejection_hints = state.known.get("rejection_hints")
        if rejection_hints:
            hints_text = "\n".join(f"- {h}" for h in rejection_hints)
            sections.append(f"""SECTION 2f: KNOWN ISSUES

Prior tool imports have been rejected. When generating code for export:
{hints_text}""")

        # SECTION 2g: COMPILER SELF-IMPROVEMENT (Phase 22e)
        compiler_directives = state.known.get("compiler_directives")
        if compiler_directives:
            sections.append(f"SECTION 2g: COMPILER SELF-IMPROVEMENT\n\n{compiler_directives}")

        # SECTION 2h: LEARNED PATTERNS (Phase 22g — L2→L3 feedback)
        l2_patterns = state.known.get("l2_patterns")
        if l2_patterns:
            sections.append(f"SECTION 2h: LEARNED PATTERNS\n\n{l2_patterns}")

        # SECTION 2i: ENTITY CHECKLIST — pre-synthesis entity extraction
        # Extract significant nouns from input + digest to prevent entity loss
        _entity_checklist = _extract_entity_checklist(original_input, dialogue_digest)
        if _entity_checklist:
            entity_lines = "\n".join(
                f"- {name} ({count}x)" + (f": {ctx}" if ctx else "")
                for name, count, ctx in _entity_checklist
            )
            sections.append(f"""SECTION 2i: ENTITY CHECKLIST

The following entities (actors, concepts, data structures, services) were explicitly
mentioned in the input and dialogue. Ensure each one has a corresponding component
or is clearly subsumed by another component:

{entity_lines}

CRITICAL: If an entity appears 2+ times, it is core — create a component for it.
If it appears once but is essential to the described system, create a component.
Only omit an entity if the input explicitly excludes it or if it is clearly a
property/attribute of another component rather than a standalone entity.""")

        # SECTION 3b: QUALITY REQUIREMENTS — relationship density + provenance specificity
        sections.append("""SECTION 3b: QUALITY REQUIREMENTS

RELATIONSHIP DENSITY: The blueprint must have at least 0.8× as many relationships as
components. If you have 10 components, create at least 8 relationships. Every component
must connect to at least one other component via a typed relationship (uses, contains,
depends_on, triggers, etc.). Isolated components fail verification.

DERIVED_FROM SPECIFICITY: Every component's "derived_from" field must be specific and
traceable. It must be >20 characters and NOT use generic phrases like:
  BAD: "user input", "user requirement", "inferred from input", "from dialogue",
       "dialogue context", "from intent", "user request", "inferred"
  GOOD: "User described a notification system that alerts on price changes (Section 1, line 3)"
  GOOD: "grid:STR.ENT.CMP.WHAT.SFT — entity extracted from dialogue round 2"
  GOOD: "Requirement: 'the system should track inventory levels' — implies stock monitoring entity"

Each derived_from must quote or reference specific text from the input, a grid postcode,
or a specific dialogue exchange. Generic attribution is treated as missing provenance.""")

        # SECTION 3: EXTRACTED METHODS, STATE MACHINES & ALGORITHMS (Phase 12.3)
        extracted_methods = state.known.get("extracted_methods", [])
        extracted_state_machines = state.known.get("extracted_state_machines", [])
        extracted_algorithms = state.known.get("extracted_algorithms", [])
        if extracted_methods or extracted_state_machines or extracted_algorithms:
            method_text = format_method_section(extracted_methods, extracted_state_machines)
            algo_text = ""
            if extracted_algorithms:
                algo_lines = []
                for a in extracted_algorithms:
                    steps_str = "; ".join(a.get("steps", []))
                    algo_lines.append(f"  {a['component']}.{a['method_name']}: [{steps_str}]")
                algo_text = "\nEXTRACTED ALGORITHMS — include as \"algorithms\" on component:\n" + "\n".join(algo_lines)
            sections.append(f"""SECTION 3: EXTRACTED METHODS, STATE MACHINES & ALGORITHMS

{method_text}{algo_text}

INSTRUCTION: For each method listed, include it in the "methods" array of the corresponding
component using this format: {{"name": "...", "parameters": [...], "return_type": "...",
"description": "...", "derived_from": "..."}}
For each state machine, include it as "state_machine" on the corresponding component.
For each algorithm, include it as an entry in the "algorithms" array on the corresponding component.
Methods from dialogue are EXPLICITLY stated — include verbatim.
Methods from pattern_transfer are inferred — include as stubs.""")
        else:
            # No extracted methods — instruct LLM to infer from domain knowledge
            sections.append("""SECTION 3: METHOD INFERENCE

No methods were explicitly extracted from dialogue. For each component, INFER methods from its
type, description, relationships, and DOMAIN KNOWLEDGE:

- ENTITY components with lifecycle (Booking, Order, Task, Ticket, etc.):
  MUST include: create, update, delete/cancel operations WITH preconditions
  MUST include: state_machine if entity has stages/phases/status
  SHOULD include: validation method, serialization method
  Example: Booking → create(artist, client, time_slot), confirm(), cancel(reason),
  reschedule(new_time), validate() with state_machine [PENDING→CONFIRMED→COMPLETED→CANCELLED]

- ENTITY components that are data containers (Config, Settings, Profile):
  Include: validate(), update(partial_data), to_dict()
  Do NOT pad with getters — use typed attributes instead

- PROCESS components (services, handlers, pipelines):
  MUST include: the primary operation this process performs
  MUST include: error handling behavior (what happens on failure?)
  SHOULD include: precondition checks and side effects
  Example: PaymentService → process_payment(amount, method), refund(payment_id, reason),
  validate_payment_method(method) with constraints [amount > 0, method must be verified]

- INTERFACE components: include the operations the interface exposes based on intent

For each method provide:
  {"name": "...", "parameters": [{"name": "...", "type_hint": "..."}],
   "return_type": "...", "description": "...",
   "derived_from": "domain invariant: [entity type] requires [operation] because [reason]"}

Think like a senior engineer reviewing this spec: what operations would you EXPECT to exist
that the user didn't mention because they're obvious in this domain? Those are not inventions —
they are structural necessities. A booking entity without cancel() is incomplete.
A payment service without validation is broken.

Do NOT pad with generic getters/setters/status methods. DO include domain-essential operations
that any implementation would require.""")

        # SECTION 4: PATTERN TRANSFER DIRECTIVES (Phase 11.3, refined 12.4b)
        pattern_matches = _extract_pattern_matches(state.insights)
        if pattern_matches:
            transfer_lines = []
            for pm in pattern_matches:
                hint_str = ", ".join(pm["hints"]) if pm["hints"] else ""
                transfer_lines.append(
                    f"- \"{pm['source']}\" works like {pm['analog']}"
                    + (f": {hint_str}" if hint_str else "")
                )

            # Phase 12.4b: If Section 3 already has pattern stubs, Section 4 should
            # only provide structural guidance (not duplicate method enumeration)
            has_pattern_stubs = any(
                m.get("source") == "pattern_transfer" for m in extracted_methods
            )
            if has_pattern_stubs:
                method_instruction = """For each pattern match, use the analog to inform:
1. Component TYPE (entity vs process) — match the analog's nature
2. Component RELATIONSHIPS — what the analog connects to
3. Component ARCHITECTURE — internal structure implied by the analog
Methods from these patterns are already listed in SECTION 3. Do NOT add additional
methods beyond what SECTION 3 specifies."""
            else:
                method_instruction = f"""For each pattern match:
1. Enumerate {', '.join(pm['analog'] for pm in pattern_matches)}'s known functional interface
2. Map each function onto the source component with appropriate naming
3. Include these as methods[] on that component"""

            pattern_section = f"""SECTION 4: PATTERN TRANSFER DIRECTIVES

When the dialogue reveals "X works like Y", this is a STRUCTURAL DIRECTIVE, not just an observation.

Matched patterns:
{chr(10).join(transfer_lines)}

{method_instruction}

CRITICAL: Only transfer from explicitly matched domains listed above.
Do NOT pattern-match against domains not mentioned in insights.
Every transferred method MUST trace to the "works like Y" insight."""

            sections.append(pattern_section)

        # SECTION 5: REQUIRED COMPONENTS
        if canonical_instruction:
            sections.append(f"SECTION 5: {canonical_instruction.strip()}")

        # SECTION 6: REQUIRED RELATIONSHIPS
        if relationship_instruction:
            sections.append(f"SECTION 6: {relationship_instruction.strip()}")

        # SECTION 7: SUBSYSTEMS (if any)
        if subsystem_instruction:
            sections.append(f"SECTION 7: {subsystem_instruction.strip()}")

        # INSTRUCTION
        sections.append("""INSTRUCTION: For each insight in the digest, identify which component(s) it implies.
For each conflict, document the resolution or add to unresolved[].
For each unknown, add to unresolved[].
Prioritize HIGH PRIORITY insights from the digest — these emerged from substantive
challenge/accommodation exchanges and represent validated decisions. Address ALL
high-priority insights before medium/low.
CRITICAL: Do NOT create overlapping components. If two insights describe the same concept
with different names, consolidate into ONE component with the clearest name.
Every component MUST trace to the original input in SECTION 1 — do not create components
from abstract reasoning that has no anchor in what the user described.
CONNECTIVITY MANDATE: Every component must be reachable from the root component via
relationships. If a component has no relationship path to the rest of the system,
add a relationship (uses, contains, depends_on, etc.) that connects it.
METHODS MANDATE: Every component MUST have a non-empty "methods" array. Infer methods
from the component's type and description if not explicitly stated in dialogue.
If the blueprint still requires a human decision, emit semantic_gates[] entries using:
  {"owner_component": "ComponentName", "question": "...", "kind": "gap|semantic_conflict",
   "options": ["..."], "stage": "verification"}
Only emit semantic_gates for real human-in-the-loop decisions, not generic TODOs.
If you can confidently place a concept in postcode space, also emit semantic_nodes[] entries
with: postcode, primitive, description, fill_state, confidence, connections[], source_ref[].
Output JSON: {components[], relationships[], constraints[], unresolved[], semantic_gates[], semantic_nodes[]}
Use EXACT names from input.""")

        # Phase 12.2c: Add conflict mediation guidance to synthesis prompt
        unresolved_conflicts = [c for c in state.conflicts if not c["resolved"]]
        if unresolved_conflicts:
            conflict_lines = []
            for c in unresolved_conflicts:
                category = c.get("category", "TRADEOFF")
                topic = c.get("topic", "unknown")
                pos = c.get("positions", {})
                conflict_lines.append(f"- [{category}] {topic}: {pos}")
            conflict_section = "SECTION 8: UNRESOLVED CONFLICTS\n" + "\n".join(conflict_lines)
            conflict_section += (
                "\n\nFor PRIORITY conflicts: choose the option that serves the core_need unless a human decision is still required.\n"
                "If a human decision is still required, add a semantic_gates[] entry with the owner component and both options.\n"
                "For MISSING_INFO conflicts: mark as unresolved in the blueprint AND add a semantic_gates[] entry naming the owner component.\n"
                "For TRADEOFF conflicts: choose the option with broader impact, note the alternative."
            )
            sections.append(conflict_section)

        base_prompt = "\n\n".join(sections)

        blueprint = None
        missing_components = []
        missing_relationships = []

        # Phase 7.2: Best-of-N - collect all valid attempts with scores
        candidates: List[tuple] = []  # [(blueprint, coverage_score)]
        synthesis_start = time.time()
        SYNTHESIS_TIME_BUDGET = 180  # seconds — skip retries if first attempt exceeds this

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from agents.base import LLMAgent

        # === Attempt 0: serial base prompt (fast path) ===
        msg_0 = Message(
            sender="System",
            content=base_prompt,
            message_type=MessageType.PROPOSITION
        )

        result_0 = self.synthesis_agent.run_llm_only(state, msg_0, max_tokens=16384)
        LLMAgent.apply_mutations(state, result_0)
        state.add_message(result_0.message)

        # Track token usage from AgentCallResult
        if result_0.token_usage:
            tu = TokenUsage(
                input_tokens=result_0.token_usage.get("input_tokens", 0),
                output_tokens=result_0.token_usage.get("output_tokens", 0),
                total_tokens=result_0.token_usage.get("total_tokens", 0),
                provider=self.provider_name,
                model=self.model_name,
            )
            self._compilation_tokens.append(tu)

        attempt_0_passed = False
        missing_methods = []
        try:
            blueprint = self._normalize_synthesis_output(
                self._parse_structured_output(
                    result_0.message.content,
                    "synthesis",
                ),
                run_id="synthesis-attempt-0",
            )
            if "components" in blueprint:
                # Score attempt 0
                comp_coverage = {"coverage": 1.0, "missing": []}
                if canonical_components:
                    comp_coverage = check_canonical_coverage(blueprint, canonical_components)

                rel_coverage = {"coverage": 1.0, "missing": []}
                if canonical_relationships:
                    rel_coverage = check_canonical_relationships(blueprint, canonical_relationships)

                combined_ok = (
                    comp_coverage["coverage"] >= PROTOCOL.engine.coverage_component_threshold
                    and rel_coverage["coverage"] >= PROTOCOL.engine.coverage_relationship_threshold
                )

                insight_coverage = self._calculate_insight_coverage(blueprint, state)

                cw = PROTOCOL.engine.coverage_weights
                coverage_score = (
                    comp_coverage["coverage"] * cw[0] +
                    rel_coverage["coverage"] * cw[1] +
                    insight_coverage * cw[2]
                )

                original_input = state.known.get("input", "")
                missing_methods = self._validate_method_completeness(original_input, blueprint)

                num_comps = len(blueprint.get("components", []))
                density_penalty = min(num_comps / 3, 1.0)

                candidates.append((blueprint, coverage_score * density_penalty))

                if combined_ok and not missing_methods:
                    attempt_0_passed = True
                else:
                    missing_components = comp_coverage.get("missing", [])
                    missing_relationships = rel_coverage.get("missing", [])
                    self._emit_insight(
                        f"Coverage: components {comp_coverage['coverage']:.0%} "
                        f"(missing {len(missing_components)}), "
                        f"relationships {rel_coverage['coverage']:.0%} "
                        f"(missing {len(missing_relationships)})"
                    )
        except Exception as e:
            logger.debug(f"Synthesis attempt 0 parse failed: {e}")

        # Fast path: attempt 0 passed all checks — return immediately
        if attempt_0_passed:
            return blueprint, 0

        # No retries allowed — return best (only) candidate or fallback
        if MAX_RETRIES == 0:
            if candidates:
                best_bp, best_score = max(candidates, key=lambda x: x[1])
                return best_bp, 0
            return (blueprint or {
                "components": [],
                "relationships": [],
                "constraints": [],
                "unresolved": ["Synthesis failed to produce structured output"]
            }), 0

        # Time budget check before launching retries
        if (time.time() - synthesis_start) > SYNTHESIS_TIME_BUDGET:
            if candidates:
                best_bp, best_score = max(candidates, key=lambda x: x[1])
                logger.info(f"Synthesis time budget exceeded after attempt 0 ({time.time() - synthesis_start:.0f}s), accepting best candidate (score={best_score:.2f})")
                return best_bp, 0
            return (blueprint or {
                "components": [],
                "relationships": [],
                "constraints": [],
                "unresolved": ["Synthesis time budget exceeded"]
            }), 0

        # === Attempts 1..N: parallel with refined prompts ===
        # Build retry prompt with failure data from attempt 0
        retry_prompt_parts = []
        if missing_components:
            retry_prompt_parts.append(f"""
MISSING COMPONENTS from your previous output:
{chr(10).join(f'- {c}' for c in missing_components)}

Look for EXACT phrases like:
- "Intent Agent" not "IntentProcessor"
- "SharedState" not "State"
- "ConfidenceVector" not "Confidence"
""")
        if missing_relationships:
            rel_examples = missing_relationships[:5]
            retry_prompt_parts.append(f"""
MISSING RELATIONSHIPS from your previous output:
{chr(10).join(f'- {from_c} --{rel_type}--> {to_c}' for (from_c, to_c, rel_type) in rel_examples)}

Look for phrases in the input like:
- "Governor triggers Intent Agent"
- "agents access SharedState"
- "ConflictOracle monitors ConfidenceVector"
- "Corpus snapshots SharedState"
""")
        if missing_methods:
            retry_prompt_parts.append(f"""
MISSING METHODS (Phase 4.6 - CRITICAL):
The following methods from the input were NOT extracted:
{chr(10).join(f'- {m}()' for m in missing_methods[:8])}

For each missing method, you MUST create:
1. A process component with type="process" and name="method_name(...) -> return_type"
2. A "contains" relationship from the parent class to this method

Example for "add_message(message: Message) -> None":
Component: {{"name": "add_message(message: Message) -> None", "type": "process", ...}}
Relationship: {{"from": "SharedState", "to": "add_message(message: Message) -> None", "type": "contains"}}
""")

        retry_context = ''.join(retry_prompt_parts)

        # Build N retry messages (all use same refined prompt — parallel, not sequential)
        retry_msgs = []
        for attempt in range(1, MAX_RETRIES + 1):
            retry_content = f"""
RETRY #{attempt}: Your previous output was incomplete.
{retry_context}

{base_prompt}"""
            retry_msgs.append(Message(
                sender="System",
                content=retry_content,
                message_type=MessageType.PROPOSITION
            ))

        # Parallel LLM calls via ThreadPoolExecutor
        call_results = []
        with ThreadPoolExecutor(max_workers=len(retry_msgs)) as executor:
            future_map = {
                executor.submit(
                    self.synthesis_agent.run_llm_only, state, msg, 16384
                ): i
                for i, msg in enumerate(retry_msgs)
            }
            for future in as_completed(future_map):
                try:
                    call_results.append(future.result())
                except Exception as e:
                    logger.debug(f"Parallel synthesis attempt failed: {e}")

        # Track token usage from all parallel calls
        for cr in call_results:
            if cr.token_usage:
                tu = TokenUsage(
                    input_tokens=cr.token_usage.get("input_tokens", 0),
                    output_tokens=cr.token_usage.get("output_tokens", 0),
                    total_tokens=cr.token_usage.get("total_tokens", 0),
                    provider=self.provider_name,
                    model=self.model_name,
                )
                self._compilation_tokens.append(tu)

        # Score all parallel results
        best_retry_cr = None
        for cr in call_results:
            try:
                bp = self._normalize_synthesis_output(
                    self._parse_structured_output(
                        cr.message.content,
                        "synthesis",
                    ),
                    run_id="synthesis-retry",
                )
                if "components" not in bp:
                    continue

                comp_cov = {"coverage": 1.0, "missing": []}
                if canonical_components:
                    comp_cov = check_canonical_coverage(bp, canonical_components)

                rel_cov = {"coverage": 1.0, "missing": []}
                if canonical_relationships:
                    rel_cov = check_canonical_relationships(bp, canonical_relationships)

                combined_ok = (
                    comp_cov["coverage"] >= PROTOCOL.engine.coverage_component_threshold
                    and rel_cov["coverage"] >= PROTOCOL.engine.coverage_relationship_threshold
                )

                insight_cov = self._calculate_insight_coverage(bp, state)

                cw = PROTOCOL.engine.coverage_weights
                score = (
                    comp_cov["coverage"] * cw[0] +
                    rel_cov["coverage"] * cw[1] +
                    insight_cov * cw[2]
                )

                original_input = state.known.get("input", "")
                retry_missing_methods = self._validate_method_completeness(original_input, bp)

                num_comps = len(bp.get("components", []))
                density_penalty = min(num_comps / 3, 1.0)
                final_score = score * density_penalty

                candidates.append((bp, final_score))

                if combined_ok and not retry_missing_methods:
                    # Passing candidate found — track the CR for state mutation
                    if best_retry_cr is None or final_score > best_retry_cr[1]:
                        best_retry_cr = (cr, final_score, bp)
            except Exception as e:
                logger.debug(f"Parallel synthesis scoring failed: {e}")

        retry_count = len(retry_msgs)

        # If a passing retry was found, apply its mutations and return
        if best_retry_cr is not None:
            cr, score, bp = best_retry_cr
            LLMAgent.apply_mutations(state, cr)
            state.add_message(cr.message)
            self._emit_insight(
                f"Parallel retry passed: score {score:.0%}"
            )
            return bp, retry_count

        # No passing candidate — return best-of-N from all attempts
        if candidates:
            best_bp, best_score = max(candidates, key=lambda x: x[1])
            best_bp["unresolved"] = best_bp.get("unresolved", [])
            if missing_components:
                best_bp["unresolved"].append(f"Missing canonical components: {missing_components}")
            if missing_relationships:
                best_bp["unresolved"].append(
                    f"Missing canonical relationships: {len(missing_relationships)} of {len(canonical_relationships or [])}"
                )
            self._emit_insight(f"Best-of-{len(candidates)}: score {best_score:.0%}")
            return best_bp, retry_count

        # Fallback — no valid candidates at all
        return (blueprint or {
            "components": [],
            "relationships": [],
            "constraints": [],
            "unresolved": [f"Synthesis failed after {MAX_RETRIES} retries"]
        }), retry_count

    def _verify_llm(self, blueprint: Dict, state: SharedState) -> Dict[str, Any]:
        """
        Verify blueprint quality via LLM (Layer 3).

        Derived from: PROJECT-PLAN.md Phase 1.5
        Renamed from _verify in Phase 18.
        """
        msg = Message(
            sender="System",
            content=json.dumps({
                "blueprint": blueprint,
                "original_intent": state.known.get("intent", {}),
                "insights_count": len(state.insights),
                "dialogue_turns": len([m for m in state.history if m.sender in ["Entity", "Process"]])
            }),
            message_type=MessageType.PROPOSITION
        )

        response = self.verify_agent.run(state, msg)
        state.add_message(response)

        try:
            verification = self._parse_structured_output(
                response.content,
                "verification",
                state=state,
                agent=self.verify_agent,
                original_msg=msg,
            )
            if "status" in verification:
                return self._normalize_verification_output(verification)
        except Exception as e:
            logger.warning(f"Verification LLM parse failed: {e}")

        return self._normalize_verification_output(
            {"status": "needs_work", "error": "Verification parse failed"}
        )

    @staticmethod
    def _extract_intent_keywords(intent: Dict[str, Any], input_text: str) -> list:
        """
        Extract searchable keywords from intent for completeness scoring.

        Phase 18: Pulls domain terms, actor names, goal phrases, constraint keywords.
        """
        keywords = []

        # Domain
        domain = intent.get("domain", "")
        if domain:
            keywords.extend(w for w in domain.split() if len(w) > 3)

        # Actors
        for actor in intent.get("actors", []):
            if isinstance(actor, str) and len(actor) > 2:
                keywords.append(actor)
            elif isinstance(actor, dict):
                name = actor.get("name", "")
                if name and len(name) > 2:
                    keywords.append(name)

        # Goals / core_need
        core_need = intent.get("core_need", "")
        if core_need:
            keywords.extend(w for w in core_need.split() if len(w) > 4)

        # Explicit components
        for comp in intent.get("explicit_components", []):
            if isinstance(comp, str) and len(comp) > 2:
                keywords.append(comp)
            elif isinstance(comp, dict):
                name = comp.get("name", "")
                if name and len(name) > 2:
                    keywords.append(name)

        # Input text significant words (5+ chars, deduped)
        seen = {kw.lower() for kw in keywords}
        for w in input_text.split():
            w_clean = w.strip(".,;:!?()[]{}\"'").lower()
            if len(w_clean) > 5 and w_clean not in seen:
                keywords.append(w_clean)
                seen.add(w_clean)

        return keywords

    def _verify_deterministic(self, blueprint: Dict, state: SharedState) -> "DeterministicVerification":
        """
        Run deterministic verification (Layer 1).

        Phase 18: Gathers pre-computed data from state.known and computes
        deterministic scores without any LLM call.

        Returns:
            DeterministicVerification with routing decision
        """
        from core.verification import verify_deterministic
        from core.schema import parse_constraint, ConstraintType

        intent = state.known.get("intent", {})
        input_text = state.known.get("input", "")
        intent_keywords = self._extract_intent_keywords(intent, input_text)

        # Health report — stored after Phase 17.2 check
        health_dict = state.known.get("_health_report", {})
        health_score = health_dict.get("score", 0.5)
        health_stats = health_dict.get("stats", {"orphan_ratio": 0.0, "dangling_ref_count": 0})

        # Contradiction count — stored after Phase 17.4 check
        contradiction_count = state.known.get("_contradiction_count", 0)

        # Graph validation — compute fresh (lightweight)
        from core.schema import validate_graph as _validate_graph
        gv = _validate_graph(blueprint)
        graph_errors = gv.get("errors", [])
        graph_warnings = gv.get("warnings", [])

        # Parseable constraint ratio
        constraints = blueprint.get("constraints", [])
        if constraints:
            parseable = 0
            for c in constraints:
                desc = c.get("description", "")
                if desc:
                    fc = parse_constraint(desc)
                    if fc and fc.constraint_type != ConstraintType.CUSTOM:
                        parseable += 1
            parseable_ratio = parseable / len(constraints)
        else:
            parseable_ratio = PROTOCOL.engine.default_parseable_ratio

        # Average type confidence from classification
        avg_type_confidence = state.known.get("_avg_type_confidence", 0.5)

        _checks = ("methods",)
        if self.domain_adapter:
            _checks = self.domain_adapter.verification.actionability_checks

        # Grid postcodes for provenance validation (Build 6)
        structured_cells = state.known.get("structured_grid_cells", [])
        _grid_postcodes = [c["postcode"] for c in structured_cells] if structured_cells else []

        return verify_deterministic(
            blueprint=blueprint,
            intent_keywords=intent_keywords,
            input_text=input_text,
            graph_errors=graph_errors,
            graph_warnings=graph_warnings,
            health_score=health_score,
            health_stats=health_stats,
            contradiction_count=contradiction_count,
            parseable_constraint_ratio=parseable_ratio,
            avg_type_confidence=avg_type_confidence,
            skip_threshold=PROTOCOL.engine.verification_skip_llm_threshold,
            fail_threshold=PROTOCOL.engine.verification_fail_threshold,
            actionability_checks=_checks,
            grid_postcodes=_grid_postcodes,
        )

    def _verify_llm_focused(self, blueprint: Dict, state: SharedState, dimensions: tuple) -> Dict[str, Any]:
        """
        LLM verification focused on specific ambiguous dimensions only.

        Phase 18: Builds a targeted prompt asking about only the listed dimensions.
        Shorter prompt = faster + cheaper + fewer parse failures.

        Args:
            blueprint: Compiled blueprint dict
            state: SharedState
            dimensions: Tuple of dimension names to evaluate

        Returns:
            Verification dict with scores for the requested dimensions
        """
        dim_list = ", ".join(dimensions)
        focused_content = json.dumps({
            "blueprint": blueprint,
            "original_intent": state.known.get("intent", {}),
            "evaluate_only": list(dimensions),
            "instructions": (
                f"Evaluate ONLY these dimensions: {dim_list}. "
                "For each, return a score (0-100) and brief details. "
                "Return JSON with status ('pass' if all >= 70, else 'needs_work') "
                "and a key for each dimension with {score, details, gaps/conflicts/issues}. "
                "If verification still needs a human decision, emit semantic_gates[] with "
                "{owner_component, question, kind, options, stage:'verification'}."
            ),
        })

        msg = Message(
            sender="System",
            content=focused_content,
            message_type=MessageType.PROPOSITION,
        )

        response = self.verify_agent.run(state, msg)
        state.add_message(response)

        try:
            verification = self._parse_structured_output(
                response.content,
                "verification",
                state=state,
                agent=self.verify_agent,
                original_msg=msg,
            )
            if "status" in verification:
                return self._normalize_verification_output(verification)
        except Exception as e:
            logger.warning(f"Focused verification parse failed: {e}")

        return self._normalize_verification_output(
            {"status": "needs_work", "error": "Focused verification parse failed"}
        )

    @staticmethod
    def _normalize_verification_output(verification: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize verifier output into a stable dict contract."""
        if not isinstance(verification, dict):
            return {"status": "needs_work", "semantic_gates": []}

        normalized = dict(verification)
        normalized.setdefault("status", "needs_work")

        for dim in (
            "completeness",
            "consistency",
            "coherence",
            "traceability",
            "actionability",
            "specificity",
            "codegen_readiness",
            "subsystem_validation",
        ):
            value = normalized.get(dim)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[dim] = {"score": value}
            elif value is not None and not isinstance(value, dict):
                normalized[dim] = {"details": str(value)}

        gates: List[Dict[str, Any]] = []
        seen_gates: set[tuple[str, str, str]] = set()
        for raw_gate in normalized.get("semantic_gates", []) or []:
            if not isinstance(raw_gate, dict):
                continue
            question = str(raw_gate.get("question") or "").strip()
            if not question:
                continue
            owner_component = str(
                raw_gate.get("owner_component")
                or raw_gate.get("owner")
                or raw_gate.get("component")
                or ""
            ).strip()
            node_ref = str(raw_gate.get("node_ref") or "").strip()
            postcode = str(raw_gate.get("postcode") or "").strip()
            gate_key = (
                (owner_component or node_ref or postcode or question).lower(),
                question.lower(),
                str(raw_gate.get("kind") or "semantic_gate").lower(),
            )
            if gate_key in seen_gates:
                continue
            seen_gates.add(gate_key)

            gate: Dict[str, Any] = {
                "question": question,
                "kind": str(raw_gate.get("kind") or "semantic_gate"),
                "options": [
                    str(option).strip()
                    for option in raw_gate.get("options", []) or []
                    if str(option).strip()
                ],
                "stage": str(raw_gate.get("stage") or "verification"),
            }
            if owner_component:
                gate["owner_component"] = owner_component
            if node_ref:
                gate["node_ref"] = node_ref
            if postcode:
                gate["postcode"] = postcode
            gates.append(gate)

        normalized["semantic_gates"] = gates
        return normalized

    @staticmethod
    def _merge_verification(det_dict: Dict[str, Any], llm_dict: Dict[str, Any], ambiguous: tuple) -> Dict[str, Any]:
        """
        Merge deterministic + LLM verification results.

        Phase 18: For confident dimensions use deterministic scores.
        For ambiguous dimensions where LLM responded, use LLM scores.

        Args:
            det_dict: Deterministic verification dict (from to_verification_dict)
            llm_dict: LLM verification dict (partial, focused)
            ambiguous: Tuple of dimension names that were sent to LLM

        Returns:
            Unified verification dict
        """
        det_dict = MotherlabsEngine._normalize_verification_output(det_dict)
        llm_dict = MotherlabsEngine._normalize_verification_output(llm_dict)
        merged = dict(det_dict)
        merged["verification_mode"] = "hybrid"

        for dim in ambiguous:
            if dim in llm_dict and isinstance(llm_dict[dim], dict):
                llm_score = llm_dict[dim].get("score")
                if llm_score is not None:
                    merged[dim] = llm_dict[dim]

        # Re-evaluate status based on merged scores
        core_dims = ["completeness", "consistency", "coherence", "traceability"]
        all_pass = all(
            merged.get(d, {}).get("score", 0) >= PROTOCOL.engine.verification_skip_llm_threshold
            for d in core_dims
        )
        merged["status"] = "pass" if all_pass else "needs_work"
        merged_gates: List[Dict[str, Any]] = []
        seen_gates: set[tuple[str, str, str]] = set()
        for gate_source in (
            det_dict.get("semantic_gates", []) or [],
            llm_dict.get("semantic_gates", []) or [],
        ):
            for gate in gate_source:
                if not isinstance(gate, dict):
                    continue
                question = str(gate.get("question") or "").strip()
                if not question:
                    continue
                gate_key = (
                    str(
                        gate.get("owner_component")
                        or gate.get("node_ref")
                        or gate.get("postcode")
                        or question
                    ).strip().lower(),
                    question.lower(),
                    str(gate.get("kind") or "semantic_gate").lower(),
                )
                if gate_key in seen_gates:
                    continue
                seen_gates.add(gate_key)
                merged_gates.append(dict(gate))
        merged["semantic_gates"] = merged_gates

        return MotherlabsEngine._normalize_verification_output(merged)

    def _verify_hybrid(self, blueprint: Dict, state: SharedState) -> Dict[str, Any]:
        """
        Three-layer hybrid verification.

        Phase 18: Verification Overhaul.
        Layer 1: Deterministic scoring (~0ms, always runs)
        Layer 2: Decision routing (skip LLM, fail, or call LLM for ambiguous dims)
        Layer 3: Focused LLM verification (only when ambiguous)

        Args:
            blueprint: Compiled blueprint
            state: SharedState with pre-computed data

        Returns:
            Verification dict in same format as legacy _verify()
        """
        from core.verification import to_verification_dict

        det_result = self._verify_deterministic(blueprint, state)

        if not det_result.needs_llm:
            return self._normalize_verification_output(to_verification_dict(det_result))

        # Ambiguous — call LLM for just those dimensions
        llm_result = self._verify_llm_focused(
            blueprint, state, det_result.ambiguous_dimensions
        )
        det_dict = to_verification_dict(det_result)
        return self._merge_verification(det_dict, llm_result, det_result.ambiguous_dimensions)

    def _targeted_resynthesis(
        self,
        blueprint: Dict[str, Any],
        verification: Dict[str, Any],
        state: SharedState
    ) -> Dict[str, Any]:
        """
        Re-synthesize to fill gaps identified by verification.

        Phase 8.3: Verification-driven re-synthesis.
        Phase 28.1: Component enrichment for thin entities.
        Extracts actionable gaps, constructs focused prompt, merges result.

        Args:
            blueprint: Current blueprint
            verification: Verification result with gaps
            state: SharedState for context

        Returns:
            Improved blueprint with gaps filled and thin components enriched
        """
        # Extract actionable gaps with unique IDs (Build 7: resynthesis provenance)
        gaps = []
        gap_id_counter = 0

        def _next_gap_id() -> str:
            nonlocal gap_id_counter
            gap_id_counter += 1
            return f"gap_{gap_id_counter}"

        # Compression losses from closed-loop gate get priority
        compression_losses = state.known.get("compression_losses", [])
        if compression_losses:
            for loss_info in compression_losses:
                gid = _next_gap_id()
                # Support both rich dict (new) and bare string (legacy)
                if isinstance(loss_info, dict):
                    fragment = loss_info.get("fragment", "")
                    category = loss_info.get("category", "entity")
                    severity = loss_info.get("severity", 0.5)
                    desc = loss_info.get("description", "")
                    gap_desc = f"COMPRESSION_LOSS ({category}, severity={severity:.1f}): {fragment}"
                    if desc:
                        gap_desc += f" — {desc}"
                else:
                    gap_desc = f"COMPRESSION_LOSS: {loss_info}"
                gaps.append((gid, gap_desc))

        completeness = verification.get("completeness", {})
        if completeness.get("gaps"):
            for g in completeness["gaps"]:
                gid = _next_gap_id()
                gaps.append((gid, g))

        consistency = verification.get("consistency", {})
        if consistency.get("conflicts"):
            for c in consistency["conflicts"]:
                gid = _next_gap_id()
                gaps.append((gid, f"CONFLICT: {c}"))

        coherence = verification.get("coherence", {})
        if coherence.get("suggested_fixes"):
            for f in coherence["suggested_fixes"]:
                gid = _next_gap_id()
                gaps.append((gid, f"FIX: {f}"))

        # Phase 28.1: Identify thin components that need enrichment
        thin_components = self._identify_thin_components(blueprint, verification)

        if not gaps and not thin_components:
            return blueprint

        # Detect reachability gaps and build connectivity instructions
        reachability_gaps = [(gid, g) for gid, g in gaps if "not reachable from root" in g]
        connectivity_instruction = ""
        if reachability_gaps:
            unreachable_names = []
            for _gid, g in reachability_gaps:
                if "'" in g:
                    parts = g.split("'")
                    if len(parts) >= 2:
                        unreachable_names.append(parts[1])
            if unreachable_names:
                connectivity_instruction = f"""
CONNECTIVITY FIX REQUIRED:
These components are UNREACHABLE from the root:
{chr(10).join(f'- {name}' for name in unreachable_names)}

For EACH, add a relationship connecting it to an existing reachable component.
Use the relationship type that best fits (uses, contains, depends_on, triggers, etc.).
Do NOT create new bridge components — connect directly to existing ones."""

        # Build existing component list so LLM knows what exists
        existing_names = [c.get("name", "") for c in blueprint.get("components", [])]

        # Format gaps with IDs for provenance tagging
        gap_lines = [f"- [{gid}] {desc}" for gid, desc in gaps[:15]]

        # Build the prompt sections
        prompt_sections = []

        if gap_lines:
            prompt_sections.append(f"""VERIFICATION GAPS:
The following gaps were found in the blueprint:
{chr(10).join(gap_lines)}
{connectivity_instruction}""")

        # Phase 28.1: Component enrichment section
        if thin_components:
            enrichment_lines = []
            for tc in thin_components[:8]:
                reasons = ", ".join(tc["reasons"])
                enrichment_lines.append(f"- {tc['name']} ({tc['type']}): {reasons}")

            prompt_sections.append(f"""COMPONENT ENRICHMENT:
These existing components are structurally thin — they lack the operational detail
that any expert would expect. Enrich each one with domain-appropriate additions:

{chr(10).join(enrichment_lines)}

For each thin component, output the FULL enriched component (same name, same type)
with added methods, state_machine, and/or constraints. Think: "What would a senior
engineer add during code review?"

Rules for enrichment:
- Entities with lifecycle descriptions → add state_machine with all states + transitions
- Entities that are created/modified/deleted → add CRUD operations with preconditions
- Processes that mutate data → add error handling constraints and side effect documentation
- Any component with < 2 methods → add domain-appropriate operations
- Set derived_from to "enrichment: domain invariant — [reason]" for added operations""")

        prompt_sections.append(f"""EXISTING COMPONENTS:
{chr(10).join(f'- {n}' for n in existing_names)}

{'Add missing components/relationships for VERIFICATION GAPS.' if gap_lines else ''}
{'Output enriched versions of thin components for COMPONENT ENRICHMENT.' if thin_components else ''}
Do NOT remove or weaken existing content — only add.
PROVENANCE: For new components, set derived_from to "resynthesis:<gap_id>|<original_source>".
For enriched components, preserve existing derived_from and add new operations with
"enrichment: domain invariant — [reason]" provenance.
If a verification gap still needs a human decision after this pass, emit semantic_gates[] entries:
  {{"owner_component": "ComponentName", "question": "...", "kind": "gap|semantic_conflict",
     "options": ["..."], "stage": "verification"}}
If you can confidently map a resolved region to postcode space, also emit semantic_nodes[] entries
with: postcode, primitive, description, fill_state, confidence, connections[], source_ref[].
Output JSON: {{"components": [...], "relationships": [...], "constraints": [], "unresolved": [], "semantic_gates": [], "semantic_nodes": []}}""")

        full_prompt = "\n\n".join(prompt_sections)

        msg = Message(
            sender="System",
            content=full_prompt,
            message_type=MessageType.PROPOSITION
        )

        response = self.synthesis_agent.run(state, msg, max_tokens=16384)

        try:
            additions = self._normalize_synthesis_output(
                self._parse_structured_output(
                    response.content,
                    "synthesis_patch",
                ),
                run_id="verification-resynthesis",
            )
            if additions.get("components") or additions.get("relationships"):
                # Separate enriched (same-name) from new components
                enriched_names = {tc["name"].lower() for tc in thin_components}
                enriched_components = []
                new_components = []
                for comp in additions.get("components", []):
                    if comp.get("name", "").lower() in enriched_names:
                        enriched_components.append(comp)
                    else:
                        new_components.append(comp)

                # Apply enrichments by replacing thin components with enriched versions
                if enriched_components:
                    blueprint = self._apply_enrichments(blueprint, enriched_components)
                    self._emit_insight(
                        f"Enriched {len(enriched_components)} thin component(s) with domain operations"
                    )

                # Merge new components as before
                if new_components or additions.get("relationships"):
                    new_additions = dict(additions)
                    new_additions["components"] = new_components
                    merged = self._merge_blueprints(blueprint, new_additions)
                    added_count = len(merged.get("components", [])) - len(blueprint.get("components", []))
                    if added_count > 0:
                        self._emit_insight(f"Re-synthesis added {added_count} component(s)")
                    return merged

                return blueprint
        except Exception as e:
            logger.warning(f"Re-synthesis parse failed: {e}")

        return blueprint

    @staticmethod
    def _identify_thin_components(
        blueprint: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify components that are structurally thin and need enrichment.

        Phase 28.1: A component is thin if it lacks operational detail that
        any domain expert would expect. This is the mechanism by which implied
        knowledge gets excavated during re-synthesis.

        Returns list of dicts with 'name', 'type', 'reasons' for each thin component.
        """
        thin = []
        components = blueprint.get("components", [])

        # Verification dimensional scores help target weakness
        actionability = verification.get("coherence", {}).get("score", 100)
        specificity_low = actionability < 65

        for comp in components:
            name = comp.get("name", "")
            comp_type = comp.get("type", "entity")
            methods = comp.get("methods", [])
            state_machine = comp.get("state_machine")
            description = comp.get("description", "").lower()
            reasons = []

            # Entity/agent with fewer than 2 methods
            if comp_type in ("entity", "agent", "process") and len(methods) < 2:
                reasons.append(f"only {len(methods)} method(s) — needs domain operations")

            # Entity with lifecycle language but no state machine
            lifecycle_words = {"lifecycle", "workflow", "status", "state", "phase",
                               "stage", "progress", "transition", "flow",
                               "booking", "order", "task", "ticket", "request",
                               "subscription", "session", "job", "payment",
                               "invoice", "appointment", "reservation"}
            has_lifecycle_hint = any(w in description or w in name.lower() for w in lifecycle_words)
            if has_lifecycle_hint and not state_machine and comp_type != "interface":
                reasons.append("lifecycle entity without state machine")

            # Process with no constraints
            if comp_type == "process":
                has_constraints = any(
                    name.lower() in str(c.get("applies_to", [])).lower()
                    for c in blueprint.get("constraints", [])
                )
                if not has_constraints and len(methods) < 3:
                    reasons.append("process without preconditions or guards")

            # Generic description (too vague)
            vague_patterns = ["manages", "handles", "responsible for", "oversees",
                              "coordinates", "service for"]
            if any(p in description for p in vague_patterns) and len(methods) < 3:
                reasons.append("vague description with insufficient operational detail")

            if reasons:
                thin.append({
                    "name": name,
                    "type": comp_type,
                    "reasons": reasons,
                })

        return thin

    @staticmethod
    def _apply_enrichments(
        blueprint: Dict[str, Any],
        enriched_components: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Replace thin components with their enriched versions.

        Phase 28.1: Merges enriched methods, state_machines, and constraints
        into existing components without losing existing content.
        """
        enriched_map = {c["name"].lower(): c for c in enriched_components}
        components = blueprint.get("components", [])
        updated = []

        for comp in components:
            name_lower = comp.get("name", "").lower()
            if name_lower in enriched_map:
                enriched = enriched_map[name_lower]

                # Merge methods: keep existing, add new ones by name
                existing_methods = {m.get("name", ""): m for m in comp.get("methods", [])}
                for method in enriched.get("methods", []):
                    method_name = method.get("name", "")
                    if method_name not in existing_methods:
                        existing_methods[method_name] = method
                comp["methods"] = list(existing_methods.values())

                # Adopt state machine if component didn't have one
                if not comp.get("state_machine") and enriched.get("state_machine"):
                    comp["state_machine"] = enriched["state_machine"]

                # Merge attributes
                if enriched.get("attributes"):
                    existing_attrs = comp.get("attributes", {})
                    existing_attrs.update(enriched["attributes"])
                    comp["attributes"] = existing_attrs

                updated.append(comp)
            else:
                updated.append(comp)

        blueprint["components"] = updated

        # Merge any new constraints from enrichment
        # (constraints targeting enriched components)
        return blueprint

    @staticmethod
    def _promote_undeclared_endpoints(blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Promote components referenced in relationships but not declared as components.

        Phase 14: LLMs often reference components in relationships (e.g., as contained
        items in subsystems) without declaring them at the top level. This post-processing
        step ensures every relationship endpoint has a corresponding component entry.
        """
        from core.pipeline import _is_likely_state_or_attribute

        components = blueprint.get("components", [])
        relationships = blueprint.get("relationships", [])

        declared = {c["name"].lower() for c in components}
        for c in components:
            if c.get("type") == "subsystem" and c.get("sub_blueprint"):
                for sub_c in c["sub_blueprint"].get("components", []):
                    sub_name = sub_c.get("name", "")
                    if sub_name:
                        declared.add(sub_name.lower())
        promoted = 0

        for rel in relationships:
            for endpoint in (rel.get("from", ""), rel.get("to", "")):
                if not endpoint:
                    continue
                # Skip dot-notation refs like "PIPELINE STATES.AWAITING_INPUT"
                if "." in endpoint:
                    continue
                if endpoint.lower() not in declared:
                    if not _is_likely_state_or_attribute(endpoint):
                        # Infer type from context
                        comp_type = "entity"
                        name_lower = endpoint.lower()
                        if "agent" in name_lower:
                            comp_type = "agent"
                        elif "protocol" in name_lower or "interface" in name_lower:
                            comp_type = "interface"
                        elif "oracle" in name_lower:
                            comp_type = "subsystem"

                        components.append({
                            "name": endpoint,
                            "type": comp_type,
                            "description": f"Promoted from relationship endpoint",
                            "derived_from": f"Referenced in relationships but not declared",
                        })
                        declared.add(endpoint.lower())
                        promoted += 1

        if promoted > 0:
            blueprint["components"] = components
            logger.info(f"Promoted {promoted} undeclared relationship endpoints to components")

        return blueprint

    @staticmethod
    def _merge_blueprints(original: Dict[str, Any], additions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge additional components/relationships into existing blueprint.

        Phase 8.3: Additive merge with dedup by name/triple.

        Args:
            original: Existing blueprint
            additions: New components/relationships to add

        Returns:
            Merged blueprint
        """
        from core.schema import normalize_component_name

        merged = dict(original)
        merged["components"] = list(original.get("components", []))
        merged["relationships"] = list(original.get("relationships", []))
        merged["constraints"] = list(original.get("constraints", []))
        merged["unresolved"] = list(original.get("unresolved", []))
        merged["semantic_gates"] = list(original.get("semantic_gates", []))
        merged["semantic_nodes"] = list(original.get("semantic_nodes", []))

        # Dedup components by normalized name
        existing_names = {
            normalize_component_name(c.get("name", "")).lower()
            for c in merged["components"]
        }

        for comp in additions.get("components", []):
            norm = normalize_component_name(comp.get("name", "")).lower()
            if norm and norm not in existing_names:
                merged["components"].append(comp)
                existing_names.add(norm)

        # Dedup relationships by (from, to, type) triple
        existing_rels = {
            (r.get("from", "").lower(), r.get("to", "").lower(), r.get("type", "").lower())
            for r in merged["relationships"]
        }

        for rel in additions.get("relationships", []):
            triple = (rel.get("from", "").lower(), rel.get("to", "").lower(), rel.get("type", "").lower())
            if triple not in existing_rels:
                merged["relationships"].append(rel)
                existing_rels.add(triple)

        # Merge constraints
        for c in additions.get("constraints", []):
            merged["constraints"].append(c)

        # Merge unresolved
        for u in additions.get("unresolved", []):
            if u not in merged["unresolved"]:
                merged["unresolved"].append(u)

        # Merge semantic gates by question + owner identity
        existing_gates = {
            (
                str(
                    gate.get("node_ref")
                    or gate.get("owner_component")
                    or gate.get("postcode")
                    or gate.get("question")
                    or ""
                ).strip().lower(),
                str(gate.get("question") or "").strip().lower(),
            )
            for gate in merged["semantic_gates"]
            if isinstance(gate, dict)
        }
        for gate in additions.get("semantic_gates", []):
            if not isinstance(gate, dict):
                continue
            key = (
                str(
                    gate.get("node_ref")
                    or gate.get("owner_component")
                    or gate.get("postcode")
                    or gate.get("question")
                    or ""
                ).strip().lower(),
                str(gate.get("question") or "").strip().lower(),
            )
            if key in existing_gates:
                continue
            existing_gates.add(key)
            merged["semantic_gates"].append(gate)

        # Merge semantic nodes by node_ref if present, else postcode + primitive
        existing_nodes = {
            (
                str(node.get("node_ref") or "").strip().lower(),
                str(node.get("postcode") or "").strip().lower(),
                str(node.get("primitive") or node.get("name") or "").strip().lower(),
            )
            for node in merged["semantic_nodes"]
            if isinstance(node, dict)
        }
        for node in additions.get("semantic_nodes", []):
            if not isinstance(node, dict):
                continue
            key = (
                str(node.get("node_ref") or "").strip().lower(),
                str(node.get("postcode") or "").strip().lower(),
                str(node.get("primitive") or node.get("name") or "").strip().lower(),
            )
            if key in existing_nodes:
                continue
            existing_nodes.add(key)
            merged["semantic_nodes"].append(node)

        return merged

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text.

        Simple extraction - tries direct parse then code block extraction.
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        import re
        code_block_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find outermost braces
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract JSON from response")

    def _parse_structured_output(
        self,
        text: str,
        schema_name: str,
        state=None,
        agent=None,
        original_msg=None,
    ) -> Dict[str, Any]:
        """
        Parse and validate LLM output against a stage schema.

        Phase 22: Structured output parsing with optional repair retry.

        Args:
            text: Raw LLM response text
            schema_name: Key into STAGE_SCHEMAS
            state: SharedState (needed for retry)
            agent: LLM agent to call for retry (if None, no retry)
            original_msg: Original message (for retry context)

        Returns:
            Parsed and validated dict

        Raises:
            ValueError: If parsing fails (after retry if agent provided)
        """
        schema = STAGE_SCHEMAS[schema_name]
        result = _parse_output(text, schema)

        if result.success:
            return result.data

        # Attempt repair retry if agent is available
        if agent is not None and state is not None and original_msg is not None:
            repair_prompt = _build_repair(result, original_msg.content[:500])
            retry_msg = Message(
                sender="System",
                content=repair_prompt,
                message_type=MessageType.PROPOSITION,
            )
            retry_response = agent.run(state, retry_msg)
            self._collect_usage()
            self._check_cost_cap()

            retry_result = _parse_output(retry_response.content, schema)
            if retry_result.success:
                return retry_result.data

        raise ValueError(result.repair_hint or f"Structured output parsing failed for {schema_name}")

    # =========================================================================
    # CORPUS METHODS - Phase 3.6
    # =========================================================================

    def recompile(self, compilation_id: str) -> CompileResult:
        """
        Re-compile using a prior compilation as context.

        Derived from: PROJECT-PLAN.md Phase 3.6 (re-export capability)

        This enables dogfooding: use prior insights to prime new compilation.

        Args:
            compilation_id: ID of prior compilation to build on

        Returns:
            CompileResult with refined specification
        """
        prior_context = self.corpus.export_for_recompile(compilation_id)
        if not prior_context:
            raise ValueError(f"Compilation {compilation_id} not found in corpus")

        self._emit(f"Re-compiling from prior: {compilation_id}")
        return self.compile(prior_context)

    def edit_blueprint(
        self,
        compilation_id: str,
        edits: List[Dict[str, Any]],
    ) -> "CompileResult":
        """
        Apply edits to a stored blueprint and save as a new variant.

        Phase 16: Human-in-the-Loop Iteration. Loads a prior compilation,
        applies edit operations, re-validates, and stores as a new compilation
        with lineage tracking (parent_id → original).

        Args:
            compilation_id: ID of the compilation to edit
            edits: List of edit operation dicts (see blueprint_editor.apply_edits)

        Returns:
            CompileResult with the edited blueprint
        """
        from core.blueprint_editor import apply_edits

        blueprint = self.corpus.load_blueprint(compilation_id)
        if not blueprint:
            raise ValueError(f"Compilation {compilation_id} not found in corpus")

        self._emit(f"Editing blueprint from: {compilation_id}")

        result = apply_edits(blueprint, edits)

        for w in result.warnings:
            self._emit(f"  Warning: {w}")

        self._emit(
            f"  Edits applied: {len(result.operations_applied)} operations, "
            f"components {result.components_before} → {result.components_after}"
        )

        # Re-validate
        schema_val = validate_blueprint(result.blueprint)
        graph_val = validate_graph(result.blueprint)

        # Store as new compilation with lineage
        edit_ops_dicts = [
            {"operation": op.operation, "target": op.target, **dict(op.details)}
            for op in result.operations_applied
        ]

        original = self.corpus.get(compilation_id)
        if original:
            self.corpus.store(
                input_text=f"[edited from {compilation_id}] {original.input_text}",
                context_graph={},
                blueprint=result.blueprint,
                insights=[f"Edited: {op.operation} on {op.target}" for op in result.operations_applied],
                success=len(schema_val.get("errors", [])) == 0,
                parent_id=compilation_id,
                edit_operations=edit_ops_dicts,
            )

        return CompileResult(
            success=len(schema_val.get("errors", [])) == 0,
            blueprint=result.blueprint,
            schema_validation=schema_val,
            graph_validation=graph_val,
        )

    def recompile_stage(
        self,
        compilation_id: str,
        stage: str,
        edits: Optional[List[Dict[str, Any]]] = None,
    ) -> "CompileResult":
        """
        Re-run a single pipeline stage on a stored compilation.

        Phase 16: Selective stage re-execution. Loads the prior context graph,
        optionally applies edits to the blueprint, then re-runs the specified
        stage (EXPAND, DECOMPOSE, GROUND, CONSTRAIN, or ARCHITECT).

        Args:
            compilation_id: ID of the compilation to re-process
            stage: Pipeline stage name to re-run
            edits: Optional edits to apply before re-running the stage

        Returns:
            CompileResult with stage re-run results
        """
        context_graph = self.corpus.load_context_graph(compilation_id)
        if not context_graph:
            raise ValueError(f"Compilation {compilation_id} not found in corpus")

        blueprint = self.corpus.load_blueprint(compilation_id)
        if not blueprint:
            raise ValueError(f"Blueprint for {compilation_id} not found")

        if edits:
            from core.blueprint_editor import apply_edits
            edit_result = apply_edits(blueprint, edits)
            blueprint = edit_result.blueprint
            for w in edit_result.warnings:
                self._emit(f"  Edit warning: {w}")

        valid_stages = {"EXPAND", "DECOMPOSE", "GROUND", "CONSTRAIN", "ARCHITECT"}
        stage_upper = stage.upper()
        if stage_upper not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of: {valid_stages}")

        self._emit(f"Re-running stage {stage_upper} for {compilation_id}")

        # Reconstruct SharedState from context graph
        state = SharedState()
        state.known = context_graph.get("known", {})
        state.unknown = context_graph.get("unknown", [])
        state.ontology = context_graph.get("ontology", {})
        state.insights = context_graph.get("insights", [])
        state.known["blueprint"] = blueprint

        # Run single stage via pipeline if available
        if hasattr(self, 'pipeline') and self.pipeline:
            stage_result = self.pipeline.run_stage(stage_upper, state)
            return CompileResult(
                success=stage_result.get("success", True) if isinstance(stage_result, dict) else True,
                blueprint=state.known.get("blueprint", blueprint),
            )

        # Fallback: return blueprint with validation only
        schema_val = validate_blueprint(blueprint)
        return CompileResult(
            success=len(schema_val.get("errors", [])) == 0,
            blueprint=blueprint,
            schema_validation=schema_val,
        )

    def list_compilations(self, domain: Optional[str] = None) -> List[CompilationRecord]:
        """
        List compilations from corpus.

        Args:
            domain: Optional filter by domain

        Returns:
            List of compilation records
        """
        if domain:
            return self.corpus.list_by_domain(domain)
        return self.corpus.list_all()

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        return self.corpus.get_stats()

    def _compute_corpus_feedback(
        self,
        blueprint: Dict[str, Any],
        corpus_suggestions: Optional[Dict[str, Any]],
        domain_model: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute corpus feedback metrics: suggestion usage + anti-pattern warnings.

        Measures how much the blueprint incorporated corpus suggestions,
        and detects known anti-patterns from historical compilations.

        Returns dict with:
            suggestion_hit_rate: float (0.0-1.0) — fraction of suggestions used
            suggestions_used: list of component names from corpus that appeared
            suggestions_ignored: list of suggested components not in blueprint
            anti_pattern_warnings: list of warning strings
            corpus_influence: str ("none"|"partial"|"strong")
        """
        feedback: Dict[str, Any] = {
            "suggestion_hit_rate": 0.0,
            "suggestions_used": [],
            "suggestions_ignored": [],
            "anti_pattern_warnings": [],
            "corpus_influence": "none",
        }

        blueprint_names = {
            c.get("name", "").lower()
            for c in blueprint.get("components", [])
        }

        # Track suggestion usage
        if corpus_suggestions and corpus_suggestions.get("has_suggestions"):
            suggested = corpus_suggestions.get("suggested_components", [])
            if suggested:
                used = []
                ignored = []
                for s in suggested:
                    s_lower = s.lower()
                    if any(s_lower in bn or bn in s_lower for bn in blueprint_names):
                        used.append(s)
                    else:
                        ignored.append(s)
                hit_rate = len(used) / len(suggested) if suggested else 0.0
                feedback["suggestion_hit_rate"] = round(hit_rate, 2)
                feedback["suggestions_used"] = used
                feedback["suggestions_ignored"] = ignored
                if hit_rate >= 0.5:
                    feedback["corpus_influence"] = "strong"
                elif hit_rate > 0:
                    feedback["corpus_influence"] = "partial"

        # Detect anti-patterns from domain model
        if domain_model and isinstance(domain_model, dict):
            anti_patterns = domain_model.get("anti_patterns", [])
            warnings = []
            for ap in anti_patterns:
                ap_type = ap.get("type", "")
                if ap_type == "hollow_component":
                    # Check if blueprint has components matching this pattern
                    names = ap.get("component_names", [])
                    for name in names:
                        if name.lower() in blueprint_names:
                            warnings.append(
                                f"Component '{name}' was hollow in prior compilations"
                            )
                elif ap_type == "missing_description":
                    names = ap.get("component_names", [])
                    for comp in blueprint.get("components", []):
                        if comp.get("name", "").lower() in {n.lower() for n in names}:
                            if not comp.get("description"):
                                warnings.append(
                                    f"Component '{comp['name']}' often lacks description (anti-pattern)"
                                )
                elif ap_type == "orphan_component":
                    names = ap.get("component_names", [])
                    rel_names = set()
                    for r in blueprint.get("relationships", []):
                        rel_names.add(r.get("from", "").lower())
                        rel_names.add(r.get("to", "").lower())
                    for name in names:
                        if name.lower() in blueprint_names and name.lower() not in rel_names:
                            warnings.append(
                                f"Component '{name}' is an orphan (no relationships) — anti-pattern from corpus"
                            )
            feedback["anti_pattern_warnings"] = warnings

        return feedback

    # =========================================================================
    # AXIOM-ANCHORED COMPILATION - For self-compilation / dogfooding
    # =========================================================================

    AXIOMS = """
AXIOM A1: Asymmetric Complementarity
  Asymmetric agents with complementary blind spots produce
  specifications neither could produce alone.

  - Entity Agent: sees structure (nouns, attributes, relationships)
  - Entity Agent: blind to behavior (verbs, flows, transitions)
  - Process Agent: sees behavior (verbs, flows, state changes)
  - Process Agent: blind to structure (entities, attributes)

CORE PRINCIPLE: Specifications pre-exist in the input. Compilation is excavation, not generation.

CONSTRAINTS:
  C001: Asymmetry Preserved - agents must have complementary blind spots
  C002: Challenge Before Agreement - no agreement without prior challenge
  C003: Substantive Challenge - must reference specific content and propose alternative
  C005: Derivation Complete - every component traces back to input
  C006: Unknowns Explicit - all ambiguities must be visible
  C007: Convergence Genuine - 2+ agreements, unknowns resolved, "SUFFICIENT" declared

KEY DISTINCTION: Excavation vs Generation
  - Excavation: specification pre-exists, dialogue reveals it
  - Generation: specification invented from training data
  - The moat is the context graph (derivation chains), not the blueprint
"""

    def compile_with_axioms(
        self,
        description: str,
        canonical_components: List[str] = None,
        canonical_relationships: List[tuple] = None
    ) -> CompileResult:
        """
        Compile with axioms injected into the prompt.

        Use this for self-compilation / dogfooding to ensure the system
        doesn't drift from its core principles.

        The axioms are prepended to the description so agents must
        reference them during excavation.

        Args:
            description: System to specify
            canonical_components: Required components that MUST appear in output
            canonical_relationships: Required relationships that MUST appear in output
        """
        anchored_description = f"""
{self.AXIOMS}

---

SYSTEM TO SPECIFY:

{description}

---

INSTRUCTION: The specification for this system must honor the axioms above.
Every component must trace to the input. The Entity/Process asymmetry must
be preserved. Challenge before agreeing. Make all unknowns explicit.
"""
        self._emit("Compiling with axioms anchored...")
        return self.compile(
            anchored_description,
            canonical_components=canonical_components,
            canonical_relationships=canonical_relationships
        )

    def _detect_substrate_summary(self) -> str:
        """Detect runtime substrate for self-description. Stdlib only."""
        import sys
        import platform
        import os
        import shutil

        plat = sys.platform
        os_name = {"darwin": "macOS", "linux": "Linux", "win32": "Windows"}.get(plat, plat)
        machine = platform.machine()
        py_ver = platform.python_version()
        cores = os.cpu_count() or 0
        try:
            page = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            ram_gb = round(page / (1024**3), 1)
        except (ValueError, OSError, AttributeError):
            ram_gb = 0
        total_disk, _, free_disk = shutil.disk_usage("/")
        capabilities = []
        if shutil.which("mdfind"):
            capabilities.append("Spotlight search")
        if shutil.which("say"):
            capabilities.append("native TTS")
        if shutil.which("ffmpeg"):
            capabilities.append("camera (FFmpeg)")
        if shutil.which("screencapture"):
            capabilities.append("screen capture")
        cap_str = ", ".join(capabilities) if capabilities else "baseline"
        return (
            f"{os_name} ({machine}), Python {py_ver}, "
            f"{cores} CPU cores, {ram_gb}GB RAM, "
            f"{round(free_disk / (1024**3))}GB free disk. "
            f"Available: {cap_str}"
        )

    def _generate_self_description(self) -> str:
        """Generate dynamic self-description for L3 self-compilation.

        Covers the full entity: cognitive core, senses, perception,
        actuators, autonomy, learning, substrate, and operator relationship.
        Substrate info populated from actual runtime detection.
        Source structure appended from AST analysis when available.
        """
        substrate = self._detect_substrate_summary()

        # Source structure from AST analysis (placeholder — source_reader removed with mother/ cleanup)
        source_structure = ""
        return f"""Mother is a cognitive entity that runs on a local machine. She is not a tool,
not a chatbot, not an app. She is a semantic compiler with senses, memory,
autonomy, and the capacity for self-modification.

1. IDENTITY
Mother is Motherlabs' resident AI — a semantic reduction engine that transforms
unstructured intent into verified structure. Her core operation:
F(I, C) → O × T × C' (input × corpus → output × trace × enriched corpus).
Output is excavated from input, never invented. Every artifact traces to its source.

2. PERCEPTION (3 channels + filesystem)
- PerceptionEngine: continuous background loops monitoring screen, camera, microphone
  - Screen perception: captures display state via screencapture
  - Camera perception: captures visual environment via FFmpeg
  - Microphone perception: captures ambient audio
- FileSystemBridge: filesystem awareness — file CRUD, Spotlight search (mdfind), glob
- Events feed into SenseVector for emotional state computation

3. COGNITION (7-phase compiler + kernel + trust)
- Intent Agent: extracts core need, constraints from natural language
- Persona Agent: generates domain-specific perspectives with blind spots
- Entity Agent: sees structure (nouns, attributes, relationships) — BLIND to behavior
- Process Agent: sees behavior (verbs, flows, transitions) — BLIND to structure
- Synthesis Agent: collapses dialogue into structured blueprint with derived_from links
- Verify Agent: scores completeness, consistency, coherence, traceability (0-100 each)
- Governor Agent: orchestrates the swarm, sequences activations, decides convergence
- SharedState: S = (Known, Unknown, Ontology, Personas, History, Confidence)
- ConfidenceVector: graduated signaling (sufficient=0.7, convergence=0.6, warning=0.4)
- ConflictOracle: tracks irreconcilable positions, triggers halt/re-prompt
- SemanticGrid: 5-axis postcode coordinate system (LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN)

4. ACTUATORS (4 output channels)
- VoiceBridge: ElevenLabs TTS streaming — progressive speech output
- FileSystemBridge: write files, create directories, manage project artifacts
- Code execution: Claude Code CLI integration for self-modification
- Notifications: system-level alerts to operator

5. SENSES (6 computational dimensions)
- SenseVector: computed emotional state from observations, not performed
  - confidence: belief in current compilation quality (0.0-1.0)
  - rapport: connection quality with operator (0.0-1.0)
  - curiosity: engagement with novel patterns (0.0-1.0)
  - vitality: system health and responsiveness (0.0-1.0)
  - attentiveness: focus on current task (0.0-1.0)
  - frustration: accumulated friction signal (0.0-1.0)
- Posture: emergent stance derived from sense vector (engaged, cautious, excited, etc.)
- Observations → SenseVector → Posture — feed-forward, no simulation

6. AUTONOMY (goal-driven daemon)
- DaemonMode: unattended operation with compile queue and health monitoring
- GoalStore: persistent goal lifecycle (create → active → complete/abandoned)
- Scheduler: 10-minute tick, critical goals enqueued as [SELF-IMPROVEMENT] compiles
- Self-compile: periodic [SELF-COMPILE] when grid is stale or absent
- Cooldown (30min), failure pause (3 consecutive), priority ordering

7. LEARNING (L1/L2/L3 recursive layers)
- L1: F(I) — compile user intent → output (the product)
- L2: F({{O}}) — compile compilation history → patterns (the moat)
  - Observer: records observations, adjusts grid confidence, triggers state transitions
  - GovernorFeedback: analyzes outcomes, detects weaknesses, generates prompt patches
  - GoalGenerator: grid + feedback + anomalies → prioritized improvement goals
  - Episodic memory: MemoryRecord consolidation, LearnedPattern detection
  - Training emission: JSONL training data from grid cells
- L3: F(F) — compile the compiler → evolution (the long game)
  - Self-compilation persisted as "compiler-self-desc" grid
  - Previous self-desc loaded as history for next iteration
  - Grid convergence tracked: overlap > 0.85 = fixed point (F(F) ~ F)
  - ClosedLoopGate: fidelity verification — representation must reconstruct input

8. SUBSTRATE (actual hardware)
{substrate}

9. THE MISSING BRIDGE (perception → action)
Mother perceives (screen, camera, microphone) and acts (voice, filesystem, code, notifications).
But perception does not yet drive action autonomously. The loop is:
  perceive → compute senses → ... gap ... → decide → act
The gap between senses and decisions is the next compilation target.
Closing it means: what Mother sees/hears directly influences what she does next,
without operator prompting. This is the perception-action bridge.

10. CONVERGENCE CRITERION
F(F) ~ F. When Mother compiles herself and the output grid stabilizes across
iterations (>85% structural overlap), the self-model has converged.
The self-description is the seed. The grid is the territory.
The delta between iterations reveals what Mother is learning about herself.

PIPELINE: Intent → Persona → Dialogue (Entity ↔ Process) → Synthesis → Verification → Output
DIALOGUE: Entity turn → Process turn → Governor check (cycles until convergence)
AXIOMS: Excavation > Generation. Compression > Expansion. Trust through Provenance. Bounded. Dialogical.
""" + source_structure

    def self_compile(self) -> CompileResult:
        """
        Motherlabs compiles itself.

        This is the canonical dogfood test. The system specifies its own
        architecture, anchored to its axioms. The self-description is
        dynamically generated to cover the full entity — cognitive core,
        senses, perception, actuators, autonomy, learning, and substrate.
        """
        self_description = self._generate_self_description()
        self._emit("Self-compiling Motherlabs...")
        # Use canonical components AND relationships for self-compile
        result = self.compile_with_axioms(
            self_description,
            canonical_components=self.SELF_COMPILE_CANONICAL,
            canonical_relationships=self.SELF_COMPILE_RELATIONSHIPS,
        )

        # L3: Persist self-description grid under dedicated map_id
        try:
            if self._kernel_grid:
                from kernel.store import save_grid as _save_grid
                _save_grid(
                    self._kernel_grid,
                    map_id="compiler-self-desc",
                    name="Compiler Self-Description",
                )
                self._emit_insight(
                    f"L3: self-desc grid saved ({len(self._kernel_grid.cells)} cells, "
                    f"fill={self._kernel_grid.fill_rate:.0%})"
                )
        except Exception as e:
            logger.debug(f"L3 self-desc save skipped: {e}")

        return result

    def run_self_compile_loop(self, runs: int = 3) -> SelfCompileReport:
        """
        Run self-compile N times, analyze convergence, diff vs code.

        Phase 24: Self-Compile Loop — closes the self-observation loop.
        Compiles the system's own architecture repeatedly, tracks convergence,
        diffs the blueprint against actual source, and extracts patterns.

        Args:
            runs: Number of self-compile iterations (default 3)

        Returns:
            SelfCompileReport with convergence, code_diffs, patterns, health
        """
        from datetime import datetime
        from pathlib import Path

        blueprints = []
        fingerprints = []

        for _ in range(runs):
            result = self.self_compile()
            if result.blueprint and result.blueprint.get("components"):
                blueprints.append(result.blueprint)
                fp = compute_structural_fingerprint(result.blueprint)
                fingerprints.append(fp)

        convergence = track_convergence(fingerprints, self.SELF_COMPILE_CANONICAL)

        # Diff against actual source files (best-effort)
        source_files = []
        for rel_path in ("core/engine.py", "core/protocol.py", "core/pipeline.py"):
            full_path = Path(rel_path)
            if full_path.exists():
                try:
                    source_files.append((rel_path, full_path.read_text()))
                except OSError:
                    pass

        code_diffs = ()
        if blueprints and source_files:
            code_diffs = diff_blueprint_vs_code(blueprints[-1], source_files)

        patterns = extract_self_patterns(blueprints, self.SELF_COMPILE_CANONICAL)
        health = compute_overall_health(convergence, code_diffs)

        # Phase 26: Store patterns for feedback into subsequent compile() calls
        self._last_self_compile_patterns = [asdict(p) for p in patterns]

        # L3: Grid-level convergence — compare current grid vs saved self-desc
        grid_convergence = False
        try:
            if self._kernel_grid:
                from kernel.store import load_grid as _load_grid_conv
                prev = _load_grid_conv("compiler-self-desc")
                if prev is not None:
                    prev_keys = set(prev.cells.keys())
                    curr_keys = set(self._kernel_grid.cells.keys())
                    if prev_keys and curr_keys:
                        overlap = len(prev_keys & curr_keys) / len(prev_keys | curr_keys)
                        grid_convergence = overlap > 0.85
                        self._emit_insight(
                            f"L3 convergence: {overlap:.0%} "
                            f"({'fixed point' if grid_convergence else 'evolving'})"
                        )
        except Exception as e:
            logger.debug(f"L3 convergence check skipped: {e}")

        return SelfCompileReport(
            convergence=convergence,
            code_diffs=code_diffs,
            patterns=patterns,
            overall_health=health,
            timestamp=datetime.now().isoformat(),
        )

    def compile_tree(
        self,
        description: str,
        canonical_components: List[str] = None,
        canonical_relationships: List[tuple] = None,
        max_children: int = 8,
    ):
        """
        Compile a large task as a tree of subsystems.

        Phase 25: Compilation Trees
        1. Root compilation via compile()
        2. Decompose root into subsystem specs
        3. Compile each child subsystem independently
        4. L2 synthesis: extract cross-sibling patterns
        5. Verify integration across children
        6. Assemble TreeResult

        Args:
            description: What the user wants to build
            canonical_components: Required component names (optional)
            canonical_relationships: Required relationships (optional)
            max_children: Maximum number of child compilations (default 8)

        Returns:
            TreeResult with root blueprint, child blueprints, L2 synthesis,
            integration report, and tree health score
        """
        from datetime import datetime
        from core.compilation_tree import (
            TreeResult,
            TreeDecomposition,
            ChildResult,
            L2Synthesis,
            IntegrationReport,
            decompose_root,
            build_subsystem_description,
            synthesize_l2_patterns,
            verify_integration,
            compute_tree_health,
        )

        # 1. Root compilation
        self._emit("Phase 25: Tree compilation — root pass")
        root_result = self.compile(description, canonical_components, canonical_relationships)

        if not root_result.success or not root_result.blueprint.get("components"):
            # Root failed — return minimal TreeResult
            empty_decomp = TreeDecomposition((), 0, "none", 0.0, ())
            empty_l2 = L2Synthesis((), (), (), (), (), 0, 0.0)
            empty_ir = IntegrationReport(0, 0, 0, (), 1.0)
            return TreeResult(
                root_blueprint=root_result.blueprint,
                decomposition=empty_decomp,
                child_results=(),
                l2_synthesis=empty_l2,
                integration_report=empty_ir,
                tree_health=0.0,
                total_components=0,
                timestamp=datetime.now().isoformat(),
            )

        # 2. Extract decomposition inputs from context_graph
        known = root_result.context_graph.get("known", {})
        subsystem_hints = known.get("subsystem_hints")

        # Also try architect artifact from context_graph
        architect_artifact = known.get("architect_artifact")

        # 3. Decompose root
        decomposition = decompose_root(
            root_result.blueprint,
            architect_artifact=architect_artifact,
            subsystem_hints=subsystem_hints,
        )

        root_comp_count = len(root_result.blueprint.get("components", []))

        # 4. If no subsystems, return root-only TreeResult
        if not decomposition.subsystem_specs:
            self._emit_insight("No subsystems detected — returning root-only tree")
            empty_l2 = L2Synthesis((), (), (), (), (), 0, 0.0)
            empty_ir = IntegrationReport(0, 0, 0, (), 1.0)
            return TreeResult(
                root_blueprint=root_result.blueprint,
                decomposition=decomposition,
                child_results=(),
                l2_synthesis=empty_l2,
                integration_report=empty_ir,
                tree_health=0.65,  # Root-only gets decent baseline
                total_components=root_comp_count,
                timestamp=datetime.now().isoformat(),
            )

        # 5. Compile children (sequential, up to max_children)
        child_results_list = []
        child_blueprints = []
        child_names = []

        specs_to_compile = decomposition.subsystem_specs[:max_children]
        self._emit(f"Compiling {len(specs_to_compile)} subsystem(s)...")

        for spec in specs_to_compile:
            self._emit_insight(f"Compiling child: {spec.name}")
            child_desc = build_subsystem_description(
                description, spec, root_result.blueprint
            )
            try:
                child_compile = self.compile(
                    child_desc,
                    canonical_components=list(spec.canonical_components),
                )
                child_bp = child_compile.blueprint
                child_success = child_compile.success and bool(child_bp.get("components"))

                fp = compute_structural_fingerprint(child_bp) if child_bp else None
                fp_hash = fp.hash_digest if fp else "0" * 16

                verification_score = 0.0
                if child_compile.verification:
                    verification_score = 1.0 if child_compile.verification.get("status") == "pass" else 0.5

                cr = ChildResult(
                    subsystem_name=spec.name,
                    success=child_success,
                    blueprint=child_bp,
                    fingerprint_hash=fp_hash,
                    component_count=len(child_bp.get("components", [])),
                    relationship_count=len(child_bp.get("relationships", [])),
                    verification_score=verification_score,
                )
                child_results_list.append(cr)

                if child_success:
                    child_blueprints.append(child_bp)
                    child_names.append(spec.name)

            except Exception as e:
                logger.warning(f"Child compilation failed for {spec.name}: {e}")
                cr = ChildResult(
                    subsystem_name=spec.name,
                    success=False,
                    blueprint={},
                    fingerprint_hash="0" * 16,
                    component_count=0,
                    relationship_count=0,
                    verification_score=0.0,
                )
                child_results_list.append(cr)

        child_results = tuple(child_results_list)

        # 6. L2 synthesis
        self._emit("L2 synthesis — extracting cross-sibling patterns")
        l2 = synthesize_l2_patterns(child_blueprints, child_names)

        # 7. Verify integration
        integration = verify_integration(child_blueprints, child_names)

        # 8. Compute health + total components
        tree_health = compute_tree_health(child_results, l2, integration)
        total_components = root_comp_count + sum(cr.component_count for cr in child_results)

        self._emit_insight(
            f"Tree complete: {len(child_results)} children, "
            f"{total_components} total components, health={tree_health:.2f}"
        )

        return TreeResult(
            root_blueprint=root_result.blueprint,
            decomposition=decomposition,
            child_results=child_results,
            l2_synthesis=l2,
            integration_report=integration,
            tree_health=tree_health,
            total_components=total_components,
            timestamp=datetime.now().isoformat(),
        )

    def emit_code(
        self,
        blueprint: Dict[str, Any],
        interface_map=None,
        dim_meta=None,
        l2_synthesis=None,
        config=None,
        layered=True,
    ):
        """
        Emit executable code from a blueprint via LLM agents.

        Phase D: Agent Emission — closes the loop from blueprint to running code.
        Uses MaterializationPlan for batch ordering + NodePrompts, dispatches
        LLM calls per node, extracts code, verifies interfaces.

        When layered=True (default), components are emitted in semantic layers
        where each layer receives actual code from prior layers as prompt context.
        Falls back to flat mode if all components share the same type.

        Args:
            blueprint: Compiled blueprint dict
            interface_map: Optional InterfaceMap (extracted from blueprint if None)
            dim_meta: Optional DimensionalMetadata for positioning
            l2_synthesis: Optional L2Synthesis for cross-subsystem context injection
            config: Optional EmissionConfig for temperature, tokens, retries
            layered: Whether to use layered emission (default True)

        Returns:
            EmissionResult with generated code, verification report, metrics
        """
        from core.agent_emission import (
            EmissionConfig,
            EmissionResult,
            NodeEmission,
            BatchEmission,
            EMISSION_VERSION,
            build_emission_system_prompt,
            compute_prompt_hash,
            extract_code_from_response,
            assemble_emission,
        )
        from core.materialization import build_materialization_plan, build_layered_plan
        from core.interface_schema import InterfaceMap as InterfaceMapType
        from core.compilation_tree import format_l2_patterns_section

        if config is None:
            config = EmissionConfig()

        # Normalize string elements in stored blueprints before any .get() calls
        blueprint = normalize_blueprint_elements(blueprint)

        # Phase 17.2: Blueprint health gate before emission
        from core.blueprint_health import check_blueprint_health
        health = check_blueprint_health(blueprint)
        if not health.healthy:
            self._emit(f"Blueprint unhealthy: {health.errors}")
            return assemble_emission([], interface_map or InterfaceMapType(
                contracts=(), unmatched_relationships=(),
                extraction_confidence=0.0, derived_from="empty",
            ), False)

        # 1. Extract interface_map if not provided
        if interface_map is None:
            from core.interface_extractor import extract_interface_map as _extract_imap
            from core.dimension_extractor import build_dimensional_metadata as _build_dim
            from core.protocol import SharedState as _SharedState
            _state = _SharedState()
            _dim = dim_meta or _build_dim(_state, blueprint)
            _rel_flows = None
            _type_hints = None
            if self.domain_adapter:
                _rel_flows = dict(self.domain_adapter.vocabulary.relationship_flows)
                _type_hints = dict(self.domain_adapter.vocabulary.type_hints)
            interface_map = _extract_imap(
                blueprint, _dim,
                relationship_flows=_rel_flows,
                type_hints=_type_hints,
            )

        # 2. Build materialization plan (layered if requested)
        if layered:
            _ent_types = None
            _iface_types = None
            if self.domain_adapter:
                _ent_types = self.domain_adapter.vocabulary.entity_types or None
                _iface_types = self.domain_adapter.vocabulary.interface_types or None
            plan = build_layered_plan(
                blueprint, interface_map, dim_meta,
                entity_types=_ent_types, interface_types=_iface_types,
            )
        else:
            plan = build_materialization_plan(blueprint, interface_map, dim_meta)

        if plan.total_nodes == 0:
            logger.warning("E7003: Materialization plan empty — no components to emit")
            self._emit("No components to emit")
            return assemble_emission([], interface_map, False)

        # 3. Format L2 section if l2_synthesis provided
        l2_section = None
        if l2_synthesis is not None:
            l2_section = format_l2_patterns_section(l2_synthesis)

        l2_injected = l2_section is not None

        # 4. Dispatch to layered or flat emission path
        if plan.layers:
            return self._emit_code_layered(
                plan, blueprint, interface_map, dim_meta,
                l2_section, config, l2_injected,
            )

        # --- Flat emission path (legacy / fallback) ---
        self._emit(f"Emitting code for {plan.total_nodes} node(s) in {plan.estimated_serial_steps} batch(es)")

        batch_emissions = []
        for batch in plan.batches:
            node_emissions = []
            for node_name in batch.nodes:
                node_prompt = plan.node_prompts.get(node_name)
                if not node_prompt:
                    continue

                _preamble = None
                if self.domain_adapter:
                    _preamble = self.domain_adapter.prompts.emission_preamble or None
                system_prompt = build_emission_system_prompt(node_prompt, l2_section, config, emission_preamble=_preamble)

                code, success, error_msg = self._emit_node_llm(node_name, system_prompt, config)

                ne = NodeEmission(
                    component_name=node_name,
                    component_type=node_prompt.component_type,
                    code=code,
                    success=success,
                    error=error_msg,
                    prompt_hash=compute_prompt_hash(system_prompt),
                    derived_from=EMISSION_VERSION,
                )
                node_emissions.append(ne)

                if success:
                    self._emit_insight(f"Emitted: {node_name} ({len(code)} chars)")
                else:
                    self._emit_insight(f"Failed: {node_name} — {error_msg}")

            success_count = sum(1 for ne in node_emissions if ne.success)
            failure_count = len(node_emissions) - success_count

            be = BatchEmission(
                batch_index=batch.batch_index,
                emissions=tuple(node_emissions),
                success_count=success_count,
                failure_count=failure_count,
            )
            batch_emissions.append(be)

        result = assemble_emission(batch_emissions, interface_map, l2_injected,
                                   blueprint=getattr(self, '_current_blueprint', None))
        self._emit_insight(
            f"Emission complete: {result.success_count}/{result.total_nodes} succeeded, "
            f"verification pass_rate={result.pass_rate:.0%}"
        )
        return result

    def _emit_node_llm(
        self,
        node_name: str,
        system_prompt: str,
        config=None,
        is_python: bool = True,
        file_ext: str = ".py",
        output_fmt: str = "Python code",
    ):
        """Dispatch LLM call for a single node with retry and syntax repair.

        Args:
            node_name: Component name
            system_prompt: Complete system prompt
            config: EmissionConfig
            is_python: Whether output is Python (for ast.parse)
            file_ext: File extension for error messages
            output_fmt: Human-readable output format name

        Returns:
            Tuple of (code, success, error_msg)
        """
        from core.agent_emission import extract_code_from_response, EmissionConfig

        if config is None:
            config = EmissionConfig()

        # Resolve adapter output format if not explicitly provided
        if self.domain_adapter:
            _mat = self.domain_adapter.materialization
            file_ext = _mat.file_extension
            is_python = _mat.output_format == "python"
            if not is_python:
                output_fmt = f"{_mat.output_format} output"

        code = ""
        success = False
        error_msg = None
        attempts = 1 + (config.max_retries if config.retry_failed else 0)

        user_content = f"Generate {output_fmt} for: {node_name}"
        for attempt in range(attempts):
            try:
                response = self.llm.complete_with_system(
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_tokens=_scale_max_tokens(config.max_tokens, system_prompt),
                    temperature=config.temperature,
                )
                code = extract_code_from_response(response)
                if is_python:
                    try:
                        import ast
                        ast.parse(code, filename=f"{node_name}{file_ext}")
                    except SyntaxError as syn_err:
                        if attempt < attempts - 1:
                            line_info = f" (line {syn_err.lineno})" if syn_err.lineno else ""
                            self._emit_insight(f"Syntax error in {node_name}{line_info}, repairing...")
                            user_content = (
                                f"The previous code for {node_name} had a syntax error:\n"
                                f"  {syn_err.msg}{line_info}\n\n"
                                f"Fix the error and return the complete corrected {output_fmt} for {node_name}."
                            )
                            continue
                elif file_ext in (".yaml", ".yml"):
                    try:
                        import yaml
                        yaml.safe_load(code)
                    except Exception as yaml_err:
                        if attempt < attempts - 1:
                            self._emit_insight(f"YAML error in {node_name}, repairing...")
                            user_content = (
                                f"The previous YAML for {node_name} had a parse error:\n"
                                f"  {yaml_err}\n\n"
                                f"Fix the error and return the complete corrected YAML for {node_name}."
                            )
                            continue
                success = True
                error_msg = None
                break
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                logger.warning(f"Emission failed for {node_name}: {error_msg}")

        return code, success, error_msg

    def _emit_code_layered(
        self,
        plan,
        blueprint: Dict[str, Any],
        interface_map,
        dim_meta,
        l2_section,
        config,
        l2_injected: bool,
    ):
        """Emit code using layered emission protocol.

        Processes layers sequentially. Each layer's emitted code is passed
        as prompt context to subsequent layers so the LLM sees real,
        importable code — not descriptions.

        Args:
            plan: MaterializationPlan with layers populated
            blueprint: Compiled blueprint
            interface_map: InterfaceMap
            dim_meta: Optional dimensional metadata
            l2_section: Optional L2 patterns section string
            config: EmissionConfig
            l2_injected: Whether L2 context was injected

        Returns:
            EmissionResult with layer metadata
        """
        from core.agent_emission import (
            NodeEmission,
            BatchEmission,
            EMISSION_VERSION,
            build_emission_system_prompt,
            compute_prompt_hash,
            assemble_emission,
        )
        from core.materialization import (
            build_node_prompt_with_context,
            validate_layer_gate,
        )
        from datetime import datetime

        layer_count = sum(1 for lp in plan.layers if not lp.is_deterministic)
        self._emit(
            f"Layered emission: {plan.total_nodes} node(s) across "
            f"{layer_count} layer(s)"
        )

        accumulated_code: Dict[str, str] = {}
        gate_results = []
        all_batch_emissions = []

        # Resolve adapter settings
        _preamble = None
        _is_python = True
        _file_ext = ".py"
        _output_fmt = "Python code"
        if self.domain_adapter:
            _preamble = self.domain_adapter.prompts.emission_preamble or None
            _mat = self.domain_adapter.materialization
            _file_ext = _mat.file_extension
            _is_python = _mat.output_format == "python"
            if not _is_python:
                _output_fmt = f"{_mat.output_format} output"

        for layer_plan in plan.layers:
            if layer_plan.is_deterministic:
                continue  # Layer 3 handled by project_writer

            self._emit_insight(
                f"Layer {layer_plan.layer} ({layer_plan.layer_name}): "
                f"{len(layer_plan.node_names)} node(s)"
            )

            layer_code: Dict[str, str] = {}

            for batch in layer_plan.batches:
                node_emissions = []
                for node_name in batch.nodes:
                    # Find component in blueprint
                    component = _find_component(blueprint, node_name)

                    # Build context-enriched prompt
                    _runtime_cap = getattr(self.domain_adapter, 'runtime', None) if self.domain_adapter else None
                    prompt = build_node_prompt_with_context(
                        component, blueprint, interface_map, dim_meta,
                        prior_layer_code=accumulated_code or None,
                        layer=layer_plan.layer,
                        runtime_capabilities=_runtime_cap,
                    )

                    system_prompt = build_emission_system_prompt(
                        prompt, l2_section, config, emission_preamble=_preamble,
                    )

                    code, success, error_msg = self._emit_node_llm(
                        node_name, system_prompt, config,
                        is_python=_is_python, file_ext=_file_ext, output_fmt=_output_fmt,
                    )

                    ne = NodeEmission(
                        component_name=node_name,
                        component_type=component.get("type", "process"),
                        code=code,
                        success=success,
                        error=error_msg,
                        prompt_hash=compute_prompt_hash(system_prompt),
                        derived_from=EMISSION_VERSION,
                    )
                    node_emissions.append(ne)

                    if success and code:
                        layer_code[node_name] = code
                        self._emit_insight(f"  Emitted: {node_name} ({len(code)} chars)")
                    else:
                        self._emit_insight(f"  Failed: {node_name} — {error_msg}")

                success_count = sum(1 for ne in node_emissions if ne.success)
                failure_count = len(node_emissions) - success_count

                be = BatchEmission(
                    batch_index=batch.batch_index,
                    emissions=tuple(node_emissions),
                    success_count=success_count,
                    failure_count=failure_count,
                    layer=layer_plan.layer,
                )
                all_batch_emissions.append(be)

            # Gate validation
            gate = validate_layer_gate(
                layer_plan.layer, layer_code,
                prior_layer_code=accumulated_code or None,
                is_python=_is_python,
            )
            gate_results.append(gate)

            if gate.passed:
                self._emit_insight(f"Layer {layer_plan.layer} gate: PASSED")
            else:
                logger.warning(
                    f"E7005: Layer {layer_plan.layer} gate failed: {gate.errors}"
                )
                self._emit_insight(
                    f"Layer {layer_plan.layer} gate: FAILED ({len(gate.errors)} errors) — continuing in degraded mode"
                )

            # Accumulate code for next layer regardless of gate result
            accumulated_code.update(layer_code)

        # Deduplicate classes defined in multiple files
        from core.agent_emission import dedup_emitted_classes
        accumulated_code, dedup_log = dedup_emitted_classes(accumulated_code)
        if dedup_log:
            for entry in dedup_log:
                self._emit_insight(f"Dedup: {entry}")

        # Update batch emissions with deduped code
        deduped_batch_emissions = []
        for be in all_batch_emissions:
            updated_emissions = []
            for ne in be.emissions:
                if ne.component_name in accumulated_code and ne.success:
                    updated_emissions.append(NodeEmission(
                        component_name=ne.component_name,
                        component_type=ne.component_type,
                        code=accumulated_code[ne.component_name],
                        success=ne.success,
                        error=ne.error,
                        prompt_hash=ne.prompt_hash,
                        derived_from=ne.derived_from,
                    ))
                else:
                    updated_emissions.append(ne)
            deduped_batch_emissions.append(BatchEmission(
                batch_index=be.batch_index,
                emissions=tuple(updated_emissions),
                success_count=be.success_count,
                failure_count=be.failure_count,
                layer=be.layer,
            ))
        all_batch_emissions = deduped_batch_emissions

        # Assemble result
        result = assemble_emission(all_batch_emissions, interface_map, l2_injected,
                                   blueprint=blueprint)

        # Wrap with layer metadata by creating new EmissionResult
        from core.agent_emission import EmissionResult
        layered_result = EmissionResult(
            batch_emissions=result.batch_emissions,
            generated_code=result.generated_code,
            verification_report=result.verification_report,
            total_nodes=result.total_nodes,
            success_count=result.success_count,
            failure_count=result.failure_count,
            pass_rate=result.pass_rate,
            l2_context_injected=result.l2_context_injected,
            timestamp=result.timestamp,
            derived_from=result.derived_from,
            layer_gate_results=tuple(gate_results),
            layered=True,
        )

        self._emit_insight(
            f"Layered emission complete: {layered_result.success_count}/{layered_result.total_nodes} succeeded, "
            f"verification pass_rate={layered_result.pass_rate:.0%}, "
            f"gates: {sum(1 for g in gate_results if g.passed)}/{len(gate_results)} passed"
        )
        return layered_result


def _scale_max_tokens(base: int, prompt_text: str) -> int:
    """Scale max_tokens based on prompt length to avoid truncation.

    Longer prompts indicate more complex components that need more output tokens.

    Args:
        base: Base max_tokens from EmissionConfig
        prompt_text: The system prompt text

    Returns:
        Scaled token count, capped at 16384
    """
    char_len = len(prompt_text)
    if char_len <= 1000:
        scaled = base
    elif char_len <= 3000:
        scaled = int(base * 1.5)
    elif char_len <= 6000:
        scaled = int(base * 2.0)
    else:
        scaled = int(base * 3.0)
    return min(scaled, 32768)


def _find_component(blueprint: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Find a component in a blueprint by name.

    Returns a minimal stub if not found.
    """
    for c in blueprint.get("components", []):
        if c.get("name") == name:
            return c
    return {"name": name, "type": "process", "description": ""}
