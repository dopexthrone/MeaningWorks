"""
Motherlabs Protocol Specification — single source of truth for all protocol constants.

This is a LEAF MODULE: it imports ONLY from stdlib (dataclasses, typing).
No project imports allowed. This ensures zero circular dependency risk.

Every magic number, threshold, marker list, and tuning constant that governs
compilation behavior lives here. All other modules read from PROTOCOL.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, FrozenSet


# =============================================================================
# SUB-SPECS
# =============================================================================


@dataclass(frozen=True)
class ConfidenceSpec:
    """Thresholds and boost values for the confidence vector system."""

    # Thresholds
    sufficient: float = 0.7        # All dimensions must reach this for is_sufficient()
    convergence: float = 0.6       # Lower threshold for dialogue convergence check
    warning: float = 0.4           # Below this triggers ConflictOracle alert

    # Message type boosts
    boost_agreement: float = 0.15
    boost_accommodation: float = 0.08
    boost_proposition_with_insight: float = 0.06
    boost_proposition_without_insight: float = 0.02
    boost_challenge_with_insight: float = 0.03
    boost_challenge_without_insight: float = -0.02

    # Content marker boosts
    boost_positive_marker: float = 0.03
    penalty_self_negative: float = -0.05
    boost_discovery_negative: float = 0.02

    # Coverage calculation
    coverage_per_insight: float = 0.08
    unknown_penalty_per: float = 0.05
    unknown_penalty_cap: float = 0.3

    # Insight-grounded blending
    insight_grounding_factor: float = 0.08
    blending_grounded_weight: float = 0.5
    blending_accumulated_weight: float = 0.5

    # Consistency window
    consistency_window: int = 6

    # Plateau detection
    plateau_window: int = 3
    plateau_threshold: float = 0.02


@dataclass(frozen=True)
class FractureSignal:
    """Mid-pipeline signal: intent resolves to competing configurations."""
    stage: str                      # Which pipeline stage fractured
    competing_configs: List[str]    # The valid alternatives (minimum 2)
    collapsing_constraint: str      # What information would resolve it
    agent: str = "Entity"           # Which agent detected the fracture
    context: str = ""               # Additional context for the user


@dataclass(frozen=True)
class DialogueSpec:
    """Dialogue depth, turn limits, and convergence rules."""

    # Base defaults
    default_max_turns: int = 64
    default_min_turns: int = 6
    default_min_insights: int = 8

    # Depth calculation bonuses
    component_divisor: int = 3
    component_bonus_cap: int = 3
    actor_divisor: int = 2
    actor_bonus_cap: int = 2
    relationship_divisor: int = 4
    relationship_bonus_cap: int = 2
    description_length_threshold: int = 2000
    constraint_count_threshold: int = 5

    # Depth caps
    min_turns_cap: int = 48
    max_turns_offset: int = 16  # max_turns = min(min_turns + offset, ceiling)
    max_turns_ceiling: int = 64

    # Convergence
    convergence_window: int = 4
    convergence_agreement_threshold: int = 2


@dataclass(frozen=True)
class GovernorSpec:
    """Governor agent termination heuristics."""

    low_dim_threshold: float = 0.4   # WARNING threshold reused
    low_dim_extra_turns: int = 2
    spread_threshold: float = 0.4
    spread_extra_turns: int = 3


@dataclass(frozen=True)
class DialecticSpec:
    """Dialectic round parameters."""
    turns_per_round: int = 3          # thesis + antithesis + synthesis
    max_rounds: int = 3               # THESIS, STRESS_TEST, COLLAPSE
    max_gate_retries: int = 4         # re-entry attempts per round
    total_turn_budget: int = 30       # hard ceiling: 3×3 + retries
    collapse_no_challenge: bool = True  # round 3 suppresses new challenges


@dataclass(frozen=True)
class PipelineStageSpec:
    """Configuration for a single pipeline stage."""

    name: str
    max_turns: int
    min_turns: int
    timeout_seconds: int


@dataclass(frozen=True)
class PipelineSpec:
    """Staged pipeline configuration."""

    stages: Tuple[PipelineStageSpec, ...] = (
        PipelineStageSpec("expand", 4, 2, 300),
        PipelineStageSpec("decompose", 8, 3, 600),
        PipelineStageSpec("ground", 8, 3, 600),
        PipelineStageSpec("constrain", 6, 2, 300),
        PipelineStageSpec("architect", 6, 2, 300),
    )

    early_termination_agreements: int = 2


@dataclass(frozen=True)
class EngineStageSpec:
    """Configuration for an engine pipeline phase."""

    name: str
    timeout_seconds: int
    max_retries: int


@dataclass(frozen=True)
class EngineSpec:
    """Engine-level constants."""

    stages: Tuple[EngineStageSpec, ...] = (
        EngineStageSpec("intent", 300, 2),
        EngineStageSpec("personas", 300, 2),
        EngineStageSpec("dialogue", 1800, 1),
        EngineStageSpec("synthesis", 600, 3),
        EngineStageSpec("verification", 300, 1),
    )

    # Quality thresholds (reads from InputQualityAnalyzer defaults)
    quality_reject_threshold: float = 0.15
    quality_warn_threshold: float = 0.35

    # Re-synthesis
    resynth_min_completeness: int = 30

    # Closed-loop fidelity threshold (must match kernel/closed_loop.py default)
    fidelity_threshold: float = 0.70

    # Phase 18: Verification hybrid thresholds
    verification_skip_llm_threshold: int = 70
    verification_fail_threshold: int = 40
    verification_codegen_ready_threshold: int = 50

    # Blueprint health abort threshold (below this, skip emission)
    blueprint_health_abort: float = 0.2

    # Re-synthesis coverage thresholds
    coverage_component_threshold: float = 0.8
    coverage_relationship_threshold: float = 0.5
    coverage_weights: Tuple[float, ...] = (0.4, 0.3, 0.3)  # component, relationship, insight

    # Default parseable constraint ratio when no constraints present
    default_parseable_ratio: float = 0.5


@dataclass(frozen=True)
class ContextSpec:
    """Context building parameters for _build_context."""

    recent_messages: int = 3
    truncation_length: int = 150
    core_need_truncation: int = 100

    # Phase thresholds for phase_hint
    explore_threshold: int = 4
    challenge_threshold: int = 8

    # Persona limits
    max_personas: int = 3
    max_priorities: int = 3
    max_blind_spot_length: int = 80
    max_key_questions: int = 2

    # Discovered/uncovered limits
    max_discovered: int = 8
    max_uncovered: int = 4
    max_constraints_display: int = 3
    max_unknowns_display: int = 3


@dataclass(frozen=True)
class MessageDetectionSpec:
    """Marker lists for message type detection."""

    agreement_markers: Tuple[str, ...] = (
        "sufficient",
        "comprehensive enough",
        "complete enough",
        "adequately covered",
        "well covered",
        "thoroughly addressed",
        "agree with",
        "i agree",
        "agreed",
        "consensus",
        "converge",
        "aligned",
    )

    strong_challenge_markers: Tuple[str, ...] = (
        "but what about",
        "you missed",
        "what happens when",
        "challenge:",
        "overlooked",
        "didn't address",
        "failed to consider",
        "gap in",
        "doesn't account for",
    )

    weak_challenge_markers: Tuple[str, ...] = (
        "however,",
        "what about",
        "missing",
    )

    accommodation_markers: Tuple[str, ...] = (
        "you're right",
        "good point",
        "valid point",
        "i missed",
        "adding",
        "revising",
        "updating",
        "incorporating",
        "your insight is valuable",
        "your insight is crucial",
        "your insight is important",
        "excellent observation",
        "good observation",
        "fair point",
    )

    positive_markers: Tuple[str, ...] = (
        "sufficient", "complete", "comprehensive", "covered", "addressed",
        "valuable", "crucial", "important", "excellent", "good point",
        "well structured", "thorough",
    )

    negative_markers: Tuple[str, ...] = (
        "missing", "missed", "overlooked", "unclear",
        "ambiguous", "gap", "failed",
    )

    self_markers: Tuple[str, ...] = (
        "i missed", "i failed", "i overlooked", "my analysis missed",
        "i didn't", "i haven't",
    )


@dataclass(frozen=True)
class CostSpec:
    """Phase 21: Cost optimization thresholds."""
    per_compilation_cap_usd: float = 5.0
    per_compilation_warn_usd: float = 2.5
    session_cap_usd: float = 50.0
    health_cost_warn_threshold: float = 2.0


@dataclass(frozen=True)
class CompactionSpec:
    """Context graph compaction parameters for export."""
    max_known_value_length: int = 500
    exclude_keys: Tuple[str, ...] = ("pipeline_state", "_domain_adapter", "_stage_config")
    max_insights: int = 50
    max_decision_trace: int = 20


@dataclass(frozen=True)
class BuildSpec:
    """Phase 27: Runtime build loop parameters."""
    max_iterations: int = 10
    max_fixes_per_component: int = 5
    subprocess_timeout_seconds: int = 300
    pip_install_timeout_seconds: int = 300
    smoke_test_timeout_seconds: int = 30
    create_venv: bool = True


@dataclass(frozen=True)
class InterrogationSpec:
    """Pre-dialogue interrogation parameters."""
    multi_domain_threshold: int = 3       # Trigger if _count_domains() >= this
    quality_interrogate_threshold: float = 0.35  # Below this = interrogate
    max_questions: int = 4
    skip_on_no_callback: bool = True


@dataclass(frozen=True)
class ProvenanceSpec:
    """Insight provenance gate parameters."""

    stem_length: int = 5
    min_matches: int = 1

    common_words: FrozenSet[str] = frozenset({
        "about", "above", "after", "again", "also", "been", "before",
        "being", "below", "between", "both", "could", "does", "doing",
        "each", "from", "further", "have", "having", "here", "itself",
        "just", "more", "most", "much", "must", "need", "only", "other",
        "over", "same", "should", "some", "such", "system", "than",
        "that", "their", "them", "then", "there", "these", "they",
        "this", "those", "through", "under", "very", "want", "well",
        "were", "what", "when", "where", "which", "while", "will",
        "with", "would", "your", "however", "still", "also", "make",
        "many", "like", "take", "come", "good", "look", "think",
        # Meta-vocabulary: words about structure/architecture, not domain
        "entity", "entities", "process", "processes", "component",
        "components", "require", "requires", "required", "flows",
        "persistent", "trigger", "triggers", "missing",
        # Function words that pass 5-char threshold but aren't domain terms
        "needs", "needed", "every", "everything", "nothing",
        "someone", "something", "enough", "already", "first",
    })


# =============================================================================
# TOP-LEVEL CONTAINER
# =============================================================================


@dataclass(frozen=True)
class ProtocolSpec:
    """Top-level container holding all protocol sub-specs."""

    confidence: ConfidenceSpec = ConfidenceSpec()
    dialogue: DialogueSpec = DialogueSpec()
    governor: GovernorSpec = GovernorSpec()
    dialectic: DialecticSpec = DialecticSpec()
    pipeline: PipelineSpec = PipelineSpec()
    engine: EngineSpec = EngineSpec()
    context: ContextSpec = ContextSpec()
    message_detection: MessageDetectionSpec = MessageDetectionSpec()
    provenance: ProvenanceSpec = ProvenanceSpec()
    cost: CostSpec = CostSpec()
    compaction: CompactionSpec = CompactionSpec()
    build: BuildSpec = BuildSpec()
    interrogation: InterrogationSpec = InterrogationSpec()


# =============================================================================
# SINGLETON — the one true protocol spec
# =============================================================================

PROTOCOL = ProtocolSpec()
