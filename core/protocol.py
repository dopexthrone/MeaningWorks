"""
Motherlabs Protocol - Core data structures and dialogue mechanics.

Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.3, 5.1

SharedState: S = (K, U, O, P, H)
  K = Known (resolved specifications)
  U = Unknown (explicit ambiguities)
  O = Ontology (shared vocabulary)
  P = Personas (domain perspectives)
  H = History (dialogue trace)
"""

import threading

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

from core.protocol_spec import PROTOCOL


# =============================================================================
# ONTOLOGY STRUCTURE
# Derived from: Dogfood v5 gap analysis - "Internal structure/format of Ontology"
# =============================================================================

@dataclass
class OntologyTerm:
    """
    A term in the shared ontology vocabulary.

    Derived from: Dogfood blueprint - "Ontology: shared vocabulary"
    """
    term: str                           # The canonical term name
    definition: str                     # What it means in this context
    aliases: List[str] = field(default_factory=list)  # Alternative names
    source: str = ""                    # Where this term came from (input/dialogue)
    relationships: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"is_a": ["Entity"], "has": ["attributes"]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "definition": self.definition,
            "aliases": self.aliases,
            "source": self.source,
            "relationships": self.relationships
        }


@dataclass
class Ontology:
    """
    Shared vocabulary for the compilation.

    Derived from: Dogfood blueprint - "O = Ontology (shared vocabulary)"

    Structure:
    - terms: Dict[str, OntologyTerm] - canonical term -> definition
    - aliases: Dict[str, str] - alias -> canonical term
    """
    terms: Dict[str, OntologyTerm] = field(default_factory=dict)
    _aliases: Dict[str, str] = field(default_factory=dict)

    def add_term(self, term: str, definition: str, aliases: List[str] = None, source: str = ""):
        """Add a term to the ontology."""
        canonical = term.lower().strip()
        self.terms[canonical] = OntologyTerm(
            term=term,
            definition=definition,
            aliases=aliases or [],
            source=source
        )
        for alias in (aliases or []):
            self._aliases[alias.lower().strip()] = canonical

    def get_term(self, name: str) -> Optional[OntologyTerm]:
        """Get a term by name or alias."""
        key = name.lower().strip()
        if key in self.terms:
            return self.terms[key]
        if key in self._aliases:
            return self.terms[self._aliases[key]]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for context graph."""
        return {
            "terms": {k: v.to_dict() for k, v in self.terms.items()},
            "alias_count": len(self._aliases)
        }


class MessageType(Enum):
    """
    Message types in the dialogue protocol.

    Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.3
    """
    PROPOSITION = "proposition"
    CHALLENGE = "challenge"
    ACCOMMODATION = "accommodation"
    AGREEMENT = "agreement"
    META = "meta"
    INSIGHT = "insight"


class TerminationState(Enum):
    """
    Termination conditions for dialogue.

    Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.1 Phase 3
    """
    SUCCESS = "success"           # Convergence achieved
    EXHAUSTION = "exhaustion"     # Max turns reached
    AMBIGUOUS = "ambiguous"       # Irreconcilable conflict (ConflictOracle)
    USER_FLAG = "user_flag"       # User flagged an insight


# =============================================================================
# CONFIDENCE THRESHOLDS
# Derived from: Dogfood v5 gap analysis - "Specific numerical thresholds needed"
# =============================================================================

CONFIDENCE_THRESHOLD_SUFFICIENT = PROTOCOL.confidence.sufficient
CONFIDENCE_THRESHOLD_CONVERGENCE = PROTOCOL.confidence.convergence
CONFIDENCE_THRESHOLD_WARNING = PROTOCOL.confidence.warning
CONFIDENCE_BOOST_AGREEMENT = PROTOCOL.confidence.boost_agreement
CONFIDENCE_BOOST_ACCOMMODATION = PROTOCOL.confidence.boost_accommodation
CONFIDENCE_BOOST_POSITIVE_MARKER = PROTOCOL.confidence.boost_positive_marker


@dataclass
class ConfidenceVector:
    """
    Graduated confidence signal for convergence.

    Derived from: Dogfood blueprint - "signals = confidence vectors + dependency maps, not flags"

    Replaces binary agreement flags with graduated scoring.

    Thresholds:
    - SUFFICIENT (0.7): Ready for synthesis
    - CONVERGENCE (0.6): Can end dialogue
    - WARNING (0.4): ConflictOracle alerts Governor
    """
    structural: float = 0.0    # Entity Agent's confidence (0.0-1.0)
    behavioral: float = 0.0    # Process Agent's confidence (0.0-1.0)
    coverage: float = 0.0      # How much of the input has been addressed
    consistency: float = 0.0   # Internal consistency of current spec

    def overall(self) -> float:
        """Weighted average confidence."""
        return (self.structural + self.behavioral + self.coverage + self.consistency) / 4

    def is_sufficient(self, threshold: float = PROTOCOL.confidence.sufficient) -> bool:
        """Check if all dimensions meet threshold."""
        return all([
            self.structural >= threshold,
            self.behavioral >= threshold,
            self.coverage >= threshold,
            self.consistency >= threshold
        ])

    def needs_attention(self) -> List[str]:
        """Return dimensions below warning threshold."""
        threshold = PROTOCOL.confidence.warning
        issues = []
        if self.structural < threshold:
            issues.append("structural")
        if self.behavioral < threshold:
            issues.append("behavioral")
        if self.coverage < threshold:
            issues.append("coverage")
        if self.consistency < threshold:
            issues.append("consistency")
        return issues

    def weakest_dimension(self) -> str:
        """
        Return the name of the lowest confidence dimension.

        Phase 10.4: Used by Governor for focus hints.
        """
        dims = {
            "structural": self.structural,
            "behavioral": self.behavioral,
            "coverage": self.coverage,
            "consistency": self.consistency,
        }
        return min(dims, key=dims.get)

    def dimension_spread(self) -> float:
        """
        Return max - min across dimensions.

        Phase 10.4: Large spread indicates uneven exploration.
        """
        vals = [self.structural, self.behavioral, self.coverage, self.consistency]
        return max(vals) - min(vals)

    def is_plateauing(self, history: list,
                      window: int = PROTOCOL.confidence.plateau_window,
                      threshold: float = PROTOCOL.confidence.plateau_threshold) -> bool:
        """
        Check if overall confidence has plateaued over recent snapshots.

        Phase 10.4: If confidence hasn't changed by more than `threshold`
        over the last `window` snapshots, it's plateauing.

        Args:
            history: List of overall confidence floats (one per turn)
            window: Number of recent snapshots to check
            threshold: Minimum change to not be considered plateau
        """
        if len(history) < window:
            return False
        recent = history[-window:]
        return (max(recent) - min(recent)) < threshold

    def to_dict(self) -> Dict[str, float]:
        """Serialize for context graph."""
        return {
            "structural": self.structural,
            "behavioral": self.behavioral,
            "coverage": self.coverage,
            "consistency": self.consistency,
            "overall": self.overall()
        }


@dataclass
class Message:
    """
    Atomic unit of dialogue.

    Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.3

    Insight fields:
    - insight: Full insight text, never truncated (for corpus/derivation)
    - insight_display: Truncated <60 chars for CLI display
    """
    sender: str
    content: str
    message_type: MessageType
    references: List[int] = field(default_factory=list)  # Indices of prior messages
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    insight: Optional[str] = None  # Full insight, never truncated
    insight_display: Optional[str] = None  # <60 char version for CLI
    insight_stratum: int = 0  # Phase 23: provenance stratum (0=input, 1=domain, 2=corpus)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for context graph."""
        return {
            "sender": self.sender,
            "content": self.content,
            "type": self.message_type.value,
            "references": self.references,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "insight": self.insight,  # Full insight preserved
            "insight_display": self.insight_display,
            "insight_stratum": self.insight_stratum,
        }


@dataclass
class SharedState:
    """
    Shared state across agents: S = (K, U, O, P, H, C)

    Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.3

    Key Properties:
    - Monotonic unknowns: U can only shrink or move items to K
    - H is append-only (immutable trace)
    - Insights are extracted from each message for CLI display
    - C (confidence) tracks graduated convergence signals
    """
    known: Dict[str, Any] = field(default_factory=dict)       # K: resolved specs
    unknown: List[str] = field(default_factory=list)          # U: ambiguities
    ontology: Dict[str, str] = field(default_factory=dict)    # O: vocabulary
    personas: List[Dict[str, str]] = field(default_factory=list)  # P: perspectives
    history: List[Message] = field(default_factory=list)      # H: dialogue trace
    insights: List[str] = field(default_factory=list)         # CLI display chain
    insight_strata: Dict[int, int] = field(default_factory=dict)  # Phase 23: insight index → stratum level (0/1/2/3)
    self_compile_patterns: List[Dict[str, Any]] = field(default_factory=list)  # Phase 24: patterns from prior self-compile loops
    flags: List[int] = field(default_factory=list)            # User-flagged indices
    confidence: ConfidenceVector = field(default_factory=ConfidenceVector)  # C: convergence signals
    confidence_history: List[float] = field(default_factory=list)  # Phase 10.4: Per-turn overall confidence snapshots
    conflicts: List[Dict[str, Any]] = field(default_factory=list)  # ConflictOracle tracking
    fractures: List[Dict[str, Any]] = field(default_factory=list)  # Fracture signals

    def __post_init__(self):
        # Thread-safety lock for concurrent agent access (Phase B/C boundary)
        object.__setattr__(self, "_lock", threading.Lock())

    def add_message(self, message: Message):
        """Add message to history, extract insight if present."""
        self.history.append(message)
        if message.insight:
            self.insights.append(message.insight)
            if message.insight_stratum > 0:
                self.insight_strata[len(self.insights) - 1] = message.insight_stratum

    def add_insight(self, insight: str, stratum: int = 0):
        """Add standalone insight with optional stratum label."""
        self.insights.append(insight)
        if stratum > 0:
            self.insight_strata[len(self.insights) - 1] = stratum

    def get_insight_stratum(self, index: int) -> int:
        """Get stratum level for an insight (default 0 = user input)."""
        return self.insight_strata.get(index, 0)

    def flag_current(self):
        """Flag the most recent insight for review."""
        if self.insights:
            self.flags.append(len(self.insights) - 1)

    def get_recent(self, n: int = 5) -> List[Message]:
        """Get last n messages from history."""
        return self.history[-n:] if len(self.history) >= n else self.history

    def add_conflict(self, agent_a: str, agent_b: str, topic: str, positions: Dict[str, str]):
        """
        Record an irreconcilable conflict between agents.

        Derived from: Dogfood blueprint - ConflictOracle component
        """
        self.conflicts.append({
            "agents": [agent_a, agent_b],
            "topic": topic,
            "positions": positions,
            "turn": len(self.history),
            "resolved": False
        })

    def add_fracture(self, stage: str, configs: List[str], constraint: str, agent: str = "Entity"):
        """Record an intent fracture: competing valid configurations detected."""
        self.fractures.append({
            "stage": stage,
            "competing_configs": configs,
            "collapsing_constraint": constraint,
            "agent": agent,
        })

    def resolve_conflict(self, index: int, resolution: str):
        """Mark a conflict as resolved with given resolution."""
        if 0 <= index < len(self.conflicts):
            self.conflicts[index]["resolved"] = True
            self.conflicts[index]["resolution"] = resolution

    def has_unresolved_conflicts(self) -> bool:
        """Check if any conflicts remain unresolved."""
        return any(not c["resolved"] for c in self.conflicts)

    def add_unknown(self, unknown: str):
        """
        Add an unknown/ambiguity to the tracking list.

        Phase 10.3: K↑/U↓ spiral — unknowns are explicitly tracked.
        Deduplicates by normalized text.
        """
        normalized = unknown.strip().lower()
        existing = [u.strip().lower() for u in self.unknown]
        if normalized and normalized not in existing:
            self.unknown.append(unknown.strip())

    def resolve_unknown(self, unknown: str):
        """
        Remove a specific unknown by exact match (case-insensitive).

        Phase 10.3: K↑/U↓ spiral — resolved unknowns are removed.
        """
        normalized = unknown.strip().lower()
        self.unknown = [u for u in self.unknown if u.strip().lower() != normalized]

    def resolve_unknown_by_keyword(self, response_text: str, min_matches: int = 2) -> List[str]:
        """
        Resolve unknowns addressed by a response (keyword matching).

        Phase 10.3: After each turn, check if existing unknowns are addressed
        by the response content. An unknown is considered resolved if 2+ of its
        significant words (>3 chars) appear in the response.

        Args:
            response_text: The agent response text to check against
            min_matches: Minimum keyword matches needed (default 2)

        Returns:
            List of resolved unknown strings
        """
        response_lower = response_text.lower()
        resolved = []

        for unknown in list(self.unknown):
            # Extract significant words (>3 chars) from the unknown
            words = [w.lower() for w in unknown.split() if len(w) > 3]
            if not words:
                continue

            matches = sum(1 for w in words if w in response_lower)
            threshold = min(min_matches, len(words))

            if matches >= threshold:
                resolved.append(unknown)

        # Remove resolved unknowns
        for r in resolved:
            self.resolve_unknown(r)

        return resolved

    def compact_known(self, spec=None) -> Dict[str, Any]:
        """
        Return a compacted copy of self.known, stripping large intermediates.

        Args:
            spec: CompactionSpec (uses PROTOCOL.compaction if None)

        Returns:
            Dict with excluded keys removed and long string values truncated.
        """
        if spec is None:
            spec = PROTOCOL.compaction
        result = {}
        for key, value in self.known.items():
            if key in spec.exclude_keys:
                continue
            if isinstance(value, str) and len(value) > spec.max_known_value_length:
                result[key] = value[:spec.max_known_value_length] + "..."
            else:
                result[key] = value
        return result

    def to_context_graph(self, compact: bool = False) -> Dict[str, Any]:
        """
        Export as context graph.

        Derived from: MASTER-TECHNICAL-SPECIFICATION Section 3.1

        Args:
            compact: If True, strip large intermediates from known, cap insights
                     and decision_trace. Default False preserves full export.
        """
        spec = PROTOCOL.compaction

        if compact:
            known_data = self.compact_known(spec)
            insights_data = self.insights[:spec.max_insights]
            trace_data = [m.to_dict() for m in self.history[-spec.max_decision_trace:]]
        else:
            known_data = self.known
            insights_data = self.insights
            trace_data = [m.to_dict() for m in self.history]

        return {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "version": "0.3.0"
            },
            "known": known_data,
            "unknown": self.unknown,
            "ontology": self.ontology,
            "personas": self.personas,
            "decision_trace": trace_data,
            "insights": insights_data,
            "insight_strata": {str(k): v for k, v in self.insight_strata.items()},
            "self_compile_patterns": list(self.self_compile_patterns),
            "flags": self.flags,
            "confidence": self.confidence.to_dict(),
            "conflicts": self.conflicts
        }


class DialogueProtocol:
    """
    Protocol for agent dialogue.

    Derived from: MASTER-TECHNICAL-SPECIFICATION Section 2.4

    Rules (Constraints C002, C003):
    1. Obligatory challenge before agreement
    2. Substantive challenges must reference specific content
    3. Accommodation required after challenge
    4. Meta-level authority (can adjust depth/focus)

    Convergence (Constraint C007):
    - SUCCESS: 2+ agreements in last 4 turns AND U empty AND depth requirements met
    - EXHAUSTION: max_turns reached
    - USER_FLAG: user flagged an insight

    Depth Controls:
    - min_turns: Minimum dialogue turns before convergence allowed
    - min_insights: Minimum insights extracted before convergence allowed
    """

    def __init__(
        self,
        max_turns: int = PROTOCOL.dialogue.default_max_turns,
        min_turns: int = PROTOCOL.dialogue.default_min_turns,
        min_insights: int = PROTOCOL.dialogue.default_min_insights
    ):
        self.max_turns = max_turns
        self.min_turns = min_turns
        self.min_insights = min_insights
        self.turn_count = 0
        self.agents: List[str] = []
        self.current_idx = 0

    def register(self, agents: List[str]):
        """Register agents for dialogue."""
        self.agents = agents

    def next_turn(self) -> str:
        """Get next agent in rotation."""
        agent = self.agents[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.agents)
        self.turn_count += 1
        return agent

    def should_terminate(self, state: SharedState) -> Optional[TerminationState]:
        """
        Check termination conditions.

        Derived from: MASTER-TECHNICAL-SPECIFICATION Section 5.3

        Detection order: [EXHAUSTION, USER_FLAG, AMBIGUOUS, SUCCESS]

        Depth requirements must be met before SUCCESS:
        - min_turns reached
        - min_insights extracted

        ConflictOracle: Detects irreconcilable positions (AMBIGUOUS)
        ConvergenceSignaling: Uses confidence vectors for graduated assessment
        """
        # Budget exhausted
        if self.turn_count >= self.max_turns:
            return TerminationState.EXHAUSTION

        # User flagged an insight
        if state.flags:
            return TerminationState.USER_FLAG

        # Check depth requirements first
        depth_met = (
            self.turn_count >= self.min_turns and
            len(state.insights) >= self.min_insights
        )

        # ConflictOracle: Check for irreconcilable conflicts
        # Only trigger AMBIGUOUS after depth requirements met —
        # early conflicts should extend dialogue, not kill it
        if depth_met and state.has_unresolved_conflicts():
            unresolved = [c for c in state.conflicts if not c["resolved"]]
            if len(unresolved) >= 2:
                return TerminationState.AMBIGUOUS

        # Check for convergence (only if depth requirements met)
        conv_window = PROTOCOL.dialogue.convergence_window
        if depth_met and len(state.history) >= conv_window:
            recent = state.get_recent(conv_window)

            # Count agreements in last 4 turns
            agreements = sum(
                1 for m in recent
                if m.message_type == MessageType.AGREEMENT
            )

            # Check for explicit "sufficient" signal
            has_sufficient = any(
                "sufficient" in m.content.lower()
                for m in recent
            )

            # ConvergenceSignaling: Check confidence vector threshold
            confidence_sufficient = state.confidence.is_sufficient(threshold=PROTOCOL.confidence.convergence)

            # Success: depth met + 2+ agreements + sufficient signal + unknowns resolved
            # OR confidence vector indicates sufficient across all dimensions
            if agreements >= PROTOCOL.dialogue.convergence_agreement_threshold and has_sufficient and not state.unknown:
                return TerminationState.SUCCESS

            # Alternative: confidence-based convergence (graduated signaling)
            if confidence_sufficient and not state.unknown and not state.has_unresolved_conflicts():
                return TerminationState.SUCCESS

        return None

    def reset(self):
        """Reset protocol state for new dialogue."""
        self.turn_count = 0
        self.current_idx = 0


def calculate_dialogue_depth(intent: Dict[str, Any], description: str) -> tuple:
    """
    Calculate dialogue depth based on input complexity.

    Phase 8.4: Complex inputs deserve more dialogue turns.
    Phase 12.1a: Returns 3-tuple with adaptive max_turns ceiling.

    Args:
        intent: Extracted intent dict with explicit_components, actors, constraints
        description: Original user description text

    Returns:
        Tuple of (min_turns, min_insights, max_turns)

    Complexity factors:
    - base: 6 turns
    - +1 per 3 explicit components (max +3)
    - +1 per 2 actors (max +2)
    - +1 per 4 explicit relationships (max +2)
    - +1 if description > 2000 chars
    - +1 if > 5 constraints
    - min_turns cap at 18, max_turns = min(min_turns + 6, 24)
    """
    spec = PROTOCOL.dialogue
    base_turns = spec.default_min_turns

    # Component complexity
    explicit_components = intent.get("explicit_components", [])
    component_bonus = min(len(explicit_components) // spec.component_divisor, spec.component_bonus_cap)

    # Actor complexity
    actors = intent.get("actors", [])
    actor_bonus = min(len(actors) // spec.actor_divisor, spec.actor_bonus_cap)

    # Relationship complexity (Phase 12.1a)
    explicit_relationships = intent.get("explicit_relationships", [])
    relationship_bonus = min(len(explicit_relationships) // spec.relationship_divisor, spec.relationship_bonus_cap)

    # Length complexity
    length_bonus = 1 if len(description) > spec.description_length_threshold else 0

    # Constraint complexity
    constraints = intent.get("constraints", [])
    constraint_bonus = 1 if len(constraints) > spec.constraint_count_threshold else 0

    total_bonus = (component_bonus + actor_bonus + relationship_bonus
                   + length_bonus + constraint_bonus)
    min_turns = min(base_turns + total_bonus, spec.min_turns_cap)
    max_turns = min(min_turns + spec.max_turns_offset, spec.max_turns_ceiling)
    min_insights = max(min_turns + 2, spec.default_min_insights)

    return (min_turns, min_insights, max_turns)
