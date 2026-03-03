"""
Motherlabs Base Agent - Foundation for all agents.

Derived from: PROJECT-PLAN.md Phase 0.3, MASTER-TECHNICAL-SPECIFICATION Section 5.2

Responsibilities:
- BaseAgent abstract class with run(state, input_msg) -> Message
- LLMAgent with system prompt + insight extraction
- Context building from SharedState
- Message type detection and insight extraction
- Confidence extraction and state updates (ConvergenceSignaling)
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from core.protocol import Message, MessageType, SharedState, ConfidenceVector
from core.protocol_spec import PROTOCOL

logger = logging.getLogger("motherlabs.agents")


# ---------------------------------------------------------------------------
# Agent roles — structural identity, not name-string-based
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    """Structural role that determines what an agent sees and produces."""
    STRUCTURAL = "structural"         # Entity axis: what things ARE
    BEHAVIORAL = "behavioral"         # Process axis: what things DO
    INTEGRATIVE = "integrative"       # Synthesis: merges both axes
    EVALUATIVE = "evaluative"         # Verifier/Governor: judges output
    ORCHESTRATIVE = "orchestrative"   # Intent/Persona: orchestrates pipeline


# Vocabulary that structural agents should not see
_BEHAVIORAL_VOCABULARY = frozenset({
    "workflow", "sequence", "pipeline", "trigger", "event",
    "transition", "state machine", "process flow", "step",
    "orchestration", "dispatch", "handler", "callback",
})

# Vocabulary that behavioral agents should not see
_STRUCTURAL_VOCABULARY = frozenset({
    "schema", "entity", "attribute", "data model", "type system",
    "class hierarchy", "inheritance", "composition", "record",
    "field", "property", "interface", "struct",
})

# Expected output vocabulary by role (for output validation)
_STRUCTURAL_OUTPUT_VOCAB = frozenset({
    "entity", "attribute", "schema", "model", "type", "component",
    "structure", "class", "object", "record", "field", "table",
    "property", "relationship", "hierarchy", "composition",
    "interface", "data", "definition", "identifier",
})

_BEHAVIORAL_OUTPUT_VOCAB = frozenset({
    "process", "flow", "step", "sequence", "transition", "event",
    "trigger", "action", "handler", "callback", "workflow",
    "pipeline", "dispatch", "state", "behavior", "operation",
    "execute", "invoke", "response", "interaction",
})


@dataclass(frozen=True)
class OutputValidation:
    """Result of agent output validation."""
    role_match_score: float            # 0.0-1.0: how well output matches role
    vocabulary_found: tuple[str, ...]  # Role-expected vocabulary found
    vocabulary_missing: tuple[str, ...] = ()  # Expected but absent


@dataclass(frozen=True)
class AgentCallResult:
    """Immutable result from run_llm_only() — captures everything needed for deferred state mutations."""
    agent_name: str
    response_text: str
    message: "Message"
    conflicts: tuple        # Extracted CONFLICT: lines (not yet applied)
    unknowns: tuple         # Extracted UNKNOWN: lines (not yet applied)
    fractures: tuple        # Extracted FRACTURE: dicts (not yet applied)
    confidence_boost: float
    agent_dimension: str    # "structural" | "behavioral" | ""
    has_insight: bool
    token_usage: dict       # Captured immediately after LLM call (from thread-local)
    output_warnings: tuple[str, ...] = ()  # Output validation warnings


class BaseAgent(ABC):
    """
    Base class for all Motherlabs agents.

    Derived from: PROJECT-PLAN.md Phase 0.3
    """

    def __init__(self, name: str, perspective: str, llm_client=None,
                 role: Optional["AgentRole"] = None):
        """
        Initialize agent.

        Args:
            name: Agent identifier (e.g., "Entity", "Process")
            perspective: What this agent sees (for context)
            llm_client: ClaudeClient instance
            role: Structural role (determines context filtering)
        """
        self.name = name
        self.perspective = perspective
        self.llm = llm_client
        self.role = role

    @abstractmethod
    def run(self, state: SharedState, input_msg: Optional[Message] = None) -> Message:
        """Execute agent logic, return message."""
        pass

    def _build_context(self, state: SharedState) -> str:
        """
        Build three-level anchored context from shared state.

        Phase 10.1: Semantic Anchoring — agents stay grounded at all levels.
        - L1 (Core): core_need, domain, key constraints — computed once, immutable
        - L2 (Evolving): discovered components, confidence, conflict count — per turn
        - L3 (Immediate): uncovered ground, active unknowns, phase hint — adaptive

        Token budget: ~310 tokens (~1250 chars)
        """
        ctx = PROTOCOL.context
        # L1: Core anchoring (never changes)
        intent = state.known.get("intent", {})
        core_need = intent.get("core_need", state.known.get("input", "")[:ctx.core_need_truncation])
        domain = intent.get("domain", "unknown")
        constraints = intent.get("constraints", [])
        constraints_str = ", ".join(constraints[:ctx.max_constraints_display]) if constraints else "none stated"

        l1 = f"L1-CORE: {core_need} [{domain}] constraints: {constraints_str}"

        # L2: Evolving state
        discovered = self._extract_discovered_components(state)
        discovered_str = ", ".join(discovered[:ctx.max_discovered]) if discovered else "none yet"
        conf = state.confidence
        conf_str = f"S={conf.structural:.1f} B={conf.behavioral:.1f} C={conf.coverage:.1f}"
        conflict_count = len([c for c in state.conflicts if not c["resolved"]])

        l2 = f"L2-STATE: discovered=[{discovered_str}] confidence=({conf_str}) conflicts={conflict_count}"

        # L3: Immediate context
        uncovered = self._compute_uncovered_ground(state)
        uncovered_str = ", ".join(uncovered[:ctx.max_uncovered]) if uncovered else "none"
        unknowns_str = "; ".join(state.unknown[:ctx.max_unknowns_display]) if state.unknown else "none"
        turn_count = len([m for m in state.history if m.sender in ("Entity", "Process")])
        if turn_count < ctx.explore_threshold:
            phase_hint = "EXPLORE"
        elif turn_count < ctx.challenge_threshold:
            phase_hint = "CHALLENGE"
        else:
            phase_hint = "CONVERGE"

        l3 = f"L3-NOW: uncovered=[{uncovered_str}] unknowns=[{unknowns_str}] phase={phase_hint}"

        # Dialectic round context injection
        dialectic_ctx = state.known.get("_dialectic_context", "")
        if dialectic_ctx:
            l3 += f"\n{dialectic_ctx}"

        # Personas (structured injection — Phase 12.1b)
        personas_str = ""
        if state.personas:
            persona_parts = []
            for p in state.personas[:ctx.max_personas]:
                name = p.get('name', 'Unknown')
                priorities = "; ".join(str(pr) for pr in p.get('priorities', [])[:ctx.max_priorities])
                blind_spots = str(p.get('blind_spots', ''))[:ctx.max_blind_spot_length]
                key_qs = "; ".join(str(q) for q in p.get('key_questions', [])[:ctx.max_key_questions])
                persona_parts.append(
                    f"{name}: priorities=[{priorities}] blind_spots=[{blind_spots}] questions=[{key_qs}]"
                )
            personas_str = "\n  ".join(persona_parts)

        # Recent messages
        recent = state.get_recent(ctx.recent_messages)
        history_str = "\n".join([
            f"[{m.sender}] {m.content[:ctx.truncation_length]}"
            for m in recent
        ])

        return f"""
--- SYSTEM INSTRUMENTATION (not part of the specification domain) ---
{l1}
{l2}
{l3}
--- END SYSTEM INSTRUMENTATION ---

PERSPECTIVE: {self.perspective}
PERSONAS: {personas_str or "None yet"}

RECENT:
{history_str or "Starting fresh"}
"""

    def _build_filtered_context(self, state: SharedState) -> str:
        """Build role-filtered context from shared state.

        For STRUCTURAL agents: excludes Process messages, strips behavioral
        vocabulary, hides confidence.behavioral.
        For BEHAVIORAL agents: excludes Entity messages, strips structural
        vocabulary, hides confidence.structural.
        Other roles get unfiltered context.
        """
        if self.role is None or self.role not in (AgentRole.STRUCTURAL, AgentRole.BEHAVIORAL):
            return self._build_context(state)

        ctx = PROTOCOL.context
        intent = state.known.get("intent", {})
        core_need = intent.get("core_need", state.known.get("input", "")[:ctx.core_need_truncation])
        domain = intent.get("domain", "unknown")
        constraints = intent.get("constraints", [])
        constraints_str = ", ".join(constraints[:ctx.max_constraints_display]) if constraints else "none stated"
        l1 = f"L1-CORE: {core_need} [{domain}] constraints: {constraints_str}"

        # L2: Filtered confidence
        discovered = self._extract_discovered_components(state)
        discovered_str = ", ".join(discovered[:ctx.max_discovered]) if discovered else "none yet"
        conf = state.confidence
        if self.role == AgentRole.STRUCTURAL:
            conf_str = f"S={conf.structural:.1f} C={conf.coverage:.1f}"
        else:
            conf_str = f"B={conf.behavioral:.1f} C={conf.coverage:.1f}"
        conflict_count = len([c for c in state.conflicts if not c["resolved"]])
        l2 = f"L2-STATE: discovered=[{discovered_str}] confidence=({conf_str}) conflicts={conflict_count}"

        # L3: Filtered unknowns
        uncovered = self._compute_uncovered_ground(state)
        uncovered_str = ", ".join(uncovered[:ctx.max_uncovered]) if uncovered else "none"
        unknowns_str = "; ".join(state.unknown[:ctx.max_unknowns_display]) if state.unknown else "none"
        turn_count = len([m for m in state.history if m.sender in ("Entity", "Process")])
        if turn_count < ctx.explore_threshold:
            phase_hint = "EXPLORE"
        elif turn_count < ctx.challenge_threshold:
            phase_hint = "CHALLENGE"
        else:
            phase_hint = "CONVERGE"
        l3 = f"L3-NOW: uncovered=[{uncovered_str}] unknowns=[{unknowns_str}] phase={phase_hint}"

        dialectic_ctx = state.known.get("_dialectic_context", "")
        if dialectic_ctx:
            l3 += f"\n{dialectic_ctx}"

        # Personas
        personas_str = ""
        if state.personas:
            persona_parts = []
            for p in state.personas[:ctx.max_personas]:
                name = p.get('name', 'Unknown')
                priorities = "; ".join(str(pr) for pr in p.get('priorities', [])[:ctx.max_priorities])
                blind_spots = str(p.get('blind_spots', ''))[:ctx.max_blind_spot_length]
                key_qs = "; ".join(str(q) for q in p.get('key_questions', [])[:ctx.max_key_questions])
                persona_parts.append(
                    f"{name}: priorities=[{priorities}] blind_spots=[{blind_spots}] questions=[{key_qs}]"
                )
            personas_str = "\n  ".join(persona_parts)

        # Filtered recent messages
        if self.role == AgentRole.STRUCTURAL:
            excluded_sender = "Process"
            strip_vocab = _BEHAVIORAL_VOCABULARY
        else:
            excluded_sender = "Entity"
            strip_vocab = _STRUCTURAL_VOCABULARY

        recent = state.get_recent(ctx.recent_messages)
        filtered_messages = [m for m in recent if m.sender != excluded_sender]
        history_lines = []
        for m in filtered_messages:
            text = m.content[:ctx.truncation_length]
            # Strip opposing vocabulary
            for vocab in strip_vocab:
                text = text.replace(vocab, "[...]")
            history_lines.append(f"[{m.sender}] {text}")

        history_str = "\n".join(history_lines)

        return f"""
--- SYSTEM INSTRUMENTATION (not part of the specification domain) ---
{l1}
{l2}
{l3}
--- END SYSTEM INSTRUMENTATION ---

PERSPECTIVE: {self.perspective}
PERSONAS: {personas_str or "None yet"}

RECENT:
{history_str or "Starting fresh"}
"""

    # System meta-vocabulary that must never leak into discovered components.
    # These are context-header artifacts, not domain entities.
    _SYSTEM_ARTIFACTS = frozenset({
        "INSIGHT", "UNKNOWN", "CONFLICT", "FRACTURE", "METHOD",
        "The", "This", "And", "That", "With", "From", "Into", "When",
        # Dialectic meta-vocabulary
        "THESIS", "ANTITHESIS", "SYNTHESIS", "COLLAPSE", "CONVERGE",
        "CONVERGENCE", "EXPLORE", "CHALLENGE", "SUFFICIENT",
        # System instrumentation headers
        "CORE", "STATE", "NOW", "GOVERNOR",
        # Agent role names (system, not domain)
        "Entity", "Process", "Intent", "Persona", "Synthesis", "Verify",
        "EntityAgent", "ProcessAgent", "IntentAgent",
        # Round/phase vocabulary
        "Round", "Phase", "Turn", "Budget", "Gate",
        # Confidence dimensions
        "Structural", "Behavioral", "Coverage", "Consistency",
        # Common protocol terms
        "PERSPECTIVE", "PERSONAS", "RECENT", "SYSTEM", "INSTRUMENTATION",
        "PROTOCOL", "DIALECTIC",
    })

    def _extract_discovered_components(self, state: SharedState) -> list:
        """
        Extract component names discovered so far from dialogue insights.

        Phase 10.1: Looks for capitalized multi-word names in insights
        that likely represent system components.

        Filters out system meta-vocabulary (agent names, dialectic terms,
        context header artifacts) to prevent context contamination.
        """
        components = set()
        for insight in state.insights:
            # Extract capitalized words (likely component names)
            words = re.findall(r'\b([A-Z][a-zA-Z]{2,})\b', insight)
            for w in words:
                # Filter system artifacts and noise
                if w not in self._SYSTEM_ARTIFACTS:
                    components.add(w)
        return sorted(components)

    def _compute_uncovered_ground(self, state: SharedState) -> list:
        """
        Compute what hasn't been addressed yet.

        Phase 10.1: Compare explicit_components from intent against
        discovered components to find gaps.
        """
        intent = state.known.get("intent", {})
        explicit = intent.get("explicit_components", [])
        if not explicit:
            return []

        discovered = set(w.lower() for w in self._extract_discovered_components(state))
        uncovered = []
        for comp in explicit:
            # Check if any discovered component matches (case-insensitive, partial)
            comp_lower = comp.lower()
            if not any(comp_lower in d or d in comp_lower for d in discovered):
                uncovered.append(comp)
        return uncovered


class LLMAgent(BaseAgent):
    """
    Agent powered by LLM.

    Derived from: PROJECT-PLAN.md Phase 0.3

    Features:
    - System prompt customization
    - Insight extraction from response
    - Message type detection
    """

    def __init__(
        self,
        name: str,
        perspective: str,
        system_prompt: str,
        llm_client,
        role: Optional[AgentRole] = None,
    ):
        super().__init__(name, perspective, llm_client, role=role)
        self.system_prompt = system_prompt

    def run(self, state: SharedState, input_msg: Optional[Message] = None,
            max_tokens: int = 4096) -> Message:
        """
        Execute LLM agent (backward-compatible: LLM call + state mutations).

        Derived from: PROJECT-PLAN.md Phase 0.3 acceptance criteria

        Args:
            state: SharedState context
            input_msg: Input message to respond to
            max_tokens: Maximum response tokens (default 4096, increase for synthesis)
        """
        result = self.run_llm_only(state, input_msg, max_tokens)
        self.apply_mutations(state, result)
        return result.message

    def run_llm_only(self, state: SharedState, input_msg: Optional[Message] = None,
                     max_tokens: int = 4096) -> AgentCallResult:
        """
        LLM call only — no state mutations. Safe for concurrent execution.

        Reads state as a snapshot, calls LLM, parses response, extracts
        conflicts/unknowns/fractures as data, computes confidence boost.
        Returns frozen AgentCallResult for deferred application via apply_mutations().
        """
        if not self.llm:
            raise ValueError(f"Agent {self.name} has no LLM client")

        context = self._build_filtered_context(state)
        full_system = f"{self.system_prompt}\n\n{context}"

        if input_msg:
            user_content = f"[{input_msg.sender}]: {input_msg.content}"
        else:
            user_content = "Begin. Analyze the known information and provide your perspective."

        # Phase 13.3: Explicit temperature=0.0 for C008 determinism enforcement
        if hasattr(self.llm, 'deterministic') and not self.llm.deterministic:
            logger.warning(
                "Agent '%s' using non-deterministic LLM client — "
                "C008 determinism not enforced", self.name
            )

        response = self.llm.complete_with_system(
            system_prompt=full_system,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Capture thread-local usage immediately (before another thread overwrites)
        token_usage = {}
        tl = getattr(self.llm, '_thread_local', None)
        if tl is not None:
            token_usage = getattr(tl, 'last_usage', {})
        if not token_usage:
            token_usage = dict(getattr(self.llm, 'last_usage', {}) or {})

        msg_type, insight_full, insight_display = self._parse_response(response, input_msg)

        # Phase 23: Stratified provenance gate
        _provenance_stratum = -1
        if insight_full:
            passed, stratum = self._check_insight_provenance(insight_full, state)
            if not passed:
                insight_full = None
                insight_display = None
            else:
                _provenance_stratum = stratum

        # Extract conflicts (pure string parsing — no state mutation)
        conflicts = self._extract_conflicts_pure(response)

        # Extract unknowns (pure string parsing)
        unknowns = self._extract_unknowns_pure(response)

        # Extract fractures (pure string parsing)
        fractures = self._extract_fractures_pure(response)

        # Compute confidence boost (reads state snapshot, doesn't write)
        confidence_boost, agent_dimension = self._compute_confidence_boost(
            state, response, msg_type
        )

        has_insight = any(
            line.strip().lower().startswith("insight:")
            for line in response.split("\n")
        )

        message = Message(
            sender=self.name,
            content=response,
            message_type=msg_type,
            insight=insight_full,
            insight_display=insight_display,
            insight_stratum=_provenance_stratum if insight_full else 0,
        )

        return AgentCallResult(
            agent_name=self.name,
            response_text=response,
            message=message,
            conflicts=tuple(conflicts),
            unknowns=tuple(unknowns),
            fractures=tuple(fractures),
            confidence_boost=confidence_boost,
            agent_dimension=agent_dimension,
            has_insight=has_insight,
            token_usage=token_usage,
        )

    @staticmethod
    def apply_mutations(state: SharedState, result: AgentCallResult) -> None:
        """Apply deferred state mutations from an AgentCallResult. Must be called serially.

        Acquires state._lock if available to ensure thread-safe mutation.
        """
        lock = getattr(state, "_lock", None)
        if lock:
            lock.acquire()
        try:
            LLMAgent._apply_mutations_unlocked(state, result)
        finally:
            if lock:
                lock.release()

    @staticmethod
    def _apply_mutations_unlocked(state: SharedState, result: AgentCallResult) -> None:
        """Internal mutation logic — caller must hold lock."""
        # Apply conflicts
        for conflict_text in result.conflicts:
            state.add_conflict(
                agent_a=result.agent_name,
                agent_b="other",
                topic=conflict_text,
                positions={result.agent_name: conflict_text},
            )

        # Apply unknowns
        for unknown_text in result.unknowns:
            state.add_unknown(unknown_text)

        # Apply fractures
        for fracture in result.fractures:
            state.add_fracture(
                stage=fracture["stage"],
                configs=fracture["configs"],
                constraint=fracture["constraint"],
            )

        # Apply confidence update
        cs = PROTOCOL.confidence
        if result.agent_dimension == "structural":
            accumulated = max(0.0, state.confidence.structural + result.confidence_boost)
            entity_insights = sum(1 for m in state.history
                                  if m.sender == "Entity" and m.insight)
            grounded = min(1.0, entity_insights * cs.insight_grounding_factor)
            state.confidence.structural = min(
                1.0, grounded * cs.blending_grounded_weight + accumulated * cs.blending_accumulated_weight
            )
        elif result.agent_dimension == "behavioral":
            accumulated = max(0.0, state.confidence.behavioral + result.confidence_boost)
            process_insights = sum(1 for m in state.history
                                   if m.sender == "Process" and m.insight)
            grounded = min(1.0, process_insights * cs.insight_grounding_factor)
            state.confidence.behavioral = min(
                1.0, grounded * cs.blending_grounded_weight + accumulated * cs.blending_accumulated_weight
            )

        # Update coverage based on insight extraction
        if state.insights:
            base_coverage = min(1.0, len(state.insights) * cs.coverage_per_insight)
            if state.unknown:
                unknown_penalty = min(cs.unknown_penalty_cap, len(state.unknown) * cs.unknown_penalty_per)
                base_coverage = max(0.0, base_coverage - unknown_penalty)
            state.confidence.coverage = base_coverage

        # Consistency update
        recent = state.get_recent(cs.consistency_window)
        if recent:
            productive = 0
            unproductive = 0
            for m in recent:
                if m.message_type in (
                        MessageType.AGREEMENT, MessageType.ACCOMMODATION,
                        MessageType.PROPOSITION):
                    productive += 1
                elif m.message_type == MessageType.CHALLENGE:
                    if m.insight:
                        productive += 1
                    else:
                        unproductive += 1
            total = productive + unproductive
            if total > 0:
                state.confidence.consistency = productive / (total + 1)

    def validate_agent_output(self, response: str) -> OutputValidation:
        """Validate whether agent output matches its declared role.

        Keyword classification (no LLM) scores whether output contains
        vocabulary expected from the role. STRUCTURAL should produce
        entity/schema language, BEHAVIORAL should produce process/flow
        language.

        Returns OutputValidation with role_match_score.
        """
        if self.role not in (AgentRole.STRUCTURAL, AgentRole.BEHAVIORAL):
            return OutputValidation(role_match_score=1.0, vocabulary_found=())

        if self.role == AgentRole.STRUCTURAL:
            expected = _STRUCTURAL_OUTPUT_VOCAB
        else:
            expected = _BEHAVIORAL_OUTPUT_VOCAB

        response_lower = response.lower()
        words = set(response_lower.split())
        # Normalize words
        normalized = {w.strip(".,;:!?\"'()[]{}") for w in words}

        found = []
        missing = []
        for vocab_word in expected:
            if vocab_word in normalized or vocab_word in response_lower:
                found.append(vocab_word)
            else:
                missing.append(vocab_word)

        score = len(found) / max(1, len(expected)) if expected else 0.0
        score = min(1.0, score)

        return OutputValidation(
            role_match_score=round(score, 4),
            vocabulary_found=tuple(sorted(found)),
            vocabulary_missing=tuple(sorted(missing)),
        )

    def _extract_conflicts(self, state: SharedState, response: str):
        """
        Extract CONFLICT: lines from agent response and record in state.

        Phase 10.2: Agents output `CONFLICT:` lines to flag disagreements.
        Format: CONFLICT: <agent_a> vs <agent_b> on <topic>
        Simplified: CONFLICT: <topic description>
        """
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("CONFLICT:"):
                conflict_text = line_stripped[9:].strip()
                if conflict_text:
                    state.add_conflict(
                        agent_a=self.name,
                        agent_b="other",
                        topic=conflict_text,
                        positions={self.name: conflict_text}
                    )

    def _extract_unknowns(self, state: SharedState, response: str):
        """
        Extract UNKNOWN: lines from agent response and add to state.

        Phase 10.3: Agents output `UNKNOWN:` lines to flag ambiguities.
        Format: UNKNOWN: <description of what's unclear>
        """
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("UNKNOWN:"):
                unknown_text = line_stripped[8:].strip()
                if unknown_text:
                    state.add_unknown(unknown_text)

    def _extract_fractures(self, state: SharedState, response: str):
        """
        Extract FRACTURE: lines from agent response.

        Format: FRACTURE: config_a | config_b : what_would_resolve_it
        Requires 2+ competing configs to register.
        """
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("FRACTURE:"):
                fracture_text = line_stripped[9:].strip()
                if fracture_text:
                    parts = fracture_text.split(":")
                    configs_part = parts[0].strip()
                    constraint = parts[1].strip() if len(parts) > 1 else "needs clarification"
                    configs = [c.strip() for c in configs_part.split("|") if c.strip()]
                    if len(configs) >= 2:
                        state.add_fracture(
                            stage="unknown",
                            configs=configs,
                            constraint=constraint,
                        )

    # --- Pure extraction methods (no state mutation) for parallel dispatch ---

    def _extract_conflicts_pure(self, response: str) -> list:
        """Extract CONFLICT: lines as strings without mutating state."""
        results = []
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("CONFLICT:"):
                conflict_text = line_stripped[9:].strip()
                if conflict_text:
                    results.append(conflict_text)
        return results

    def _extract_unknowns_pure(self, response: str) -> list:
        """Extract UNKNOWN: lines as strings without mutating state."""
        results = []
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("UNKNOWN:"):
                unknown_text = line_stripped[8:].strip()
                if unknown_text:
                    results.append(unknown_text)
        return results

    def _extract_fractures_pure(self, response: str) -> list:
        """Extract FRACTURE: lines as dicts without mutating state."""
        results = []
        for line in response.split("\n"):
            line_stripped = line.strip()
            if line_stripped.upper().startswith("FRACTURE:"):
                fracture_text = line_stripped[9:].strip()
                if fracture_text:
                    parts = fracture_text.split(":")
                    configs_part = parts[0].strip()
                    constraint = parts[1].strip() if len(parts) > 1 else "needs clarification"
                    configs = [c.strip() for c in configs_part.split("|") if c.strip()]
                    if len(configs) >= 2:
                        results.append({
                            "stage": "unknown",
                            "configs": configs,
                            "constraint": constraint,
                        })
        return results

    def _compute_confidence_boost(self, state: SharedState, response: str,
                                  msg_type: MessageType) -> tuple:
        """Compute confidence boost and dimension without mutating state.

        Returns:
            (confidence_boost: float, agent_dimension: str)
            agent_dimension is "structural", "behavioral", or ""
        """
        response_lower = response.lower()
        cs = PROTOCOL.confidence
        md = PROTOCOL.message_detection

        confidence_boost = 0.0

        if msg_type == MessageType.AGREEMENT:
            confidence_boost += cs.boost_agreement
        elif msg_type == MessageType.ACCOMMODATION:
            confidence_boost += cs.boost_accommodation
        elif msg_type == MessageType.PROPOSITION:
            has_insight = any(
                line.strip().lower().startswith("insight:")
                for line in response.split("\n")
            )
            if has_insight:
                confidence_boost += cs.boost_proposition_with_insight
            else:
                confidence_boost += cs.boost_proposition_without_insight
        elif msg_type == MessageType.CHALLENGE:
            has_insight = any(
                line.strip().lower().startswith("insight:")
                for line in response.split("\n")
            )
            if has_insight:
                confidence_boost += cs.boost_challenge_with_insight
            else:
                confidence_boost += cs.boost_challenge_without_insight

        for marker in md.positive_markers:
            if marker in response_lower:
                confidence_boost += cs.boost_positive_marker
                break

        for marker in md.negative_markers:
            if marker in response_lower:
                if any(sm in response_lower for sm in md.self_markers):
                    confidence_boost += cs.penalty_self_negative
                elif msg_type == MessageType.CHALLENGE:
                    pass
                else:
                    confidence_boost += cs.boost_discovery_negative
                break

        # Dispatch on role first, fall back to name for backward compat
        agent_dimension = ""
        if self.role == AgentRole.STRUCTURAL:
            agent_dimension = "structural"
        elif self.role == AgentRole.BEHAVIORAL:
            agent_dimension = "behavioral"
        elif self.name == "Entity":
            agent_dimension = "structural"
        elif self.name == "Process":
            agent_dimension = "behavioral"

        return confidence_boost, agent_dimension

    def _update_confidence(self, state: SharedState, response: str, msg_type: MessageType):
        """
        Update state confidence vector based on response analysis.

        Phase 10.6: Reworked to recognize progress from all message types.
        Propositions with insights are the primary driver in early/mid dialogue.
        Agreements confirm, but discoveries build.

        Derived from: Dogfood blueprint - ConvergenceSignaling component
        """
        response_lower = response.lower()
        cs = PROTOCOL.confidence
        md = PROTOCOL.message_detection

        # Calculate confidence indicators from response
        confidence_boost = 0.0

        # --- Message type signals ---
        if msg_type == MessageType.AGREEMENT:
            confidence_boost += cs.boost_agreement
        elif msg_type == MessageType.ACCOMMODATION:
            confidence_boost += cs.boost_accommodation
        elif msg_type == MessageType.PROPOSITION:
            # Propositions are productive turns — they build understanding
            # Check if this turn produced an insight (new discovery)
            has_insight = any(
                line.strip().lower().startswith("insight:")
                for line in response.split("\n")
            )
            if has_insight:
                confidence_boost += cs.boost_proposition_with_insight
            else:
                confidence_boost += cs.boost_proposition_without_insight
        elif msg_type == MessageType.CHALLENGE:
            # Substantive challenges reveal territory — net positive if with insight
            has_insight = any(
                line.strip().lower().startswith("insight:")
                for line in response.split("\n")
            )
            if has_insight:
                confidence_boost += cs.boost_challenge_with_insight
            else:
                confidence_boost += cs.boost_challenge_without_insight

        # Content-based positive signals
        for marker in md.positive_markers:
            if marker in response_lower:
                confidence_boost += cs.boost_positive_marker
                break  # Only count once

        # Negative signals — semantic split (Phase 10.6b)
        # Case 1: Self-directed ("I missed", "I failed") → penalize own dimension
        # Case 2: Territory-claiming ("there is a gap", "missing X") → discovery, boost
        # Case 3: Other-directed ("you overlooked") → already handled by CHALLENGE type
        for marker in md.negative_markers:
            if marker in response_lower:
                # Check if self-directed
                if any(sm in response_lower for sm in md.self_markers):
                    confidence_boost += cs.penalty_self_negative
                elif msg_type == MessageType.CHALLENGE:
                    pass  # Other-directed, already scored by message type
                else:
                    # Territory-claiming: agent found a gap = discovery
                    confidence_boost += cs.boost_discovery_negative
                break  # Only count once

        # Update appropriate dimension based on agent name.
        # Blend per-turn boost (noisy) with insight-grounded anchor (deterministic)
        # to stabilize structural/behavioral across runs.
        if self.name == "Entity":
            accumulated = max(0.0, state.confidence.structural + confidence_boost)
            entity_insights = sum(1 for m in state.history
                                  if m.sender == "Entity" and m.insight)
            grounded = min(1.0, entity_insights * cs.insight_grounding_factor)
            state.confidence.structural = min(1.0, grounded * cs.blending_grounded_weight + accumulated * cs.blending_accumulated_weight)
        elif self.name == "Process":
            accumulated = max(0.0, state.confidence.behavioral + confidence_boost)
            process_insights = sum(1 for m in state.history
                                   if m.sender == "Process" and m.insight)
            grounded = min(1.0, process_insights * cs.insight_grounding_factor)
            state.confidence.behavioral = min(1.0, grounded * cs.blending_grounded_weight + accumulated * cs.blending_accumulated_weight)

        # Update coverage based on insight extraction
        if state.insights:
            # Coverage increases with each insight, diminishing returns
            base_coverage = min(1.0, len(state.insights) * cs.coverage_per_insight)
            # Phase 10.3: Penalize coverage for unresolved unknowns
            if state.unknown:
                unknown_penalty = min(cs.unknown_penalty_cap, len(state.unknown) * cs.unknown_penalty_per)
                base_coverage = max(0.0, base_coverage - unknown_penalty)
            state.confidence.coverage = base_coverage

        # Consistency: ratio of productive turns over all classified turns.
        # Productive = agreement, accommodation, proposition, OR challenge-with-insight.
        # In productive friction, a challenge that yields an insight advances K↑.
        # Only empty challenges (no insight) count as unproductive.
        recent = state.get_recent(cs.consistency_window)
        if recent:
            productive = 0
            unproductive = 0
            for m in recent:
                if m.message_type in (
                        MessageType.AGREEMENT, MessageType.ACCOMMODATION,
                        MessageType.PROPOSITION):
                    productive += 1
                elif m.message_type == MessageType.CHALLENGE:
                    if m.insight:
                        productive += 1  # Challenge that yielded insight
                    else:
                        unproductive += 1
            total = productive + unproductive
            if total > 0:
                state.confidence.consistency = productive / (total + 1)

    def _parse_response(
        self,
        content: str,
        input_msg: Optional[Message]
    ) -> Tuple[MessageType, Optional[str], Optional[str]]:
        """
        Parse response to determine type and extract insight.

        Derived from: PROJECT-PLAN.md Phase 0.3 acceptance criteria
        - Message type detection (proposition, challenge, agreement, etc.)
        - Insight extraction (INSIGHT: lines, symbolic patterns)

        Returns:
            (message_type, insight_full, insight_display)
            - insight_full: Complete insight for corpus/derivation
            - insight_display: Truncated <60 chars for CLI
        """
        content_lower = content.lower()

        # Determine message type
        msg_type = self._detect_message_type(content_lower, input_msg)

        # Extract insight (full and display versions)
        insight_full, insight_display = self._extract_insight(content)

        return msg_type, insight_full, insight_display

    # Phase 10.5: Common words excluded from reference matching
    _COMMON_WORDS = PROTOCOL.provenance.common_words

    def _detect_message_type(
        self,
        content_lower: str,
        input_msg: Optional[Message]
    ) -> MessageType:
        """
        Detect message type from content.

        Phase 10.5: Anti-gaming + substance check.
        - Challenge markers split into strong and weak
        - If agreement markers AND only weak challenge markers → AGREEMENT
        - Strong challenges still require substance check
        - Generic pushback demoted to PROPOSITION
        """
        md = PROTOCOL.message_detection
        # Agreement signals
        agreement_markers = md.agreement_markers

        # Strong challenge signals (always meaningful)
        strong_challenge_markers = md.strong_challenge_markers

        # Weak challenge signals (often used as transition words)
        weak_challenge_markers = md.weak_challenge_markers

        agreement_count = sum(1 for m in agreement_markers if m in content_lower)
        strong_challenge_count = sum(1 for m in strong_challenge_markers if m in content_lower)
        weak_challenge_count = sum(1 for m in weak_challenge_markers if m in content_lower)
        total_challenge_count = strong_challenge_count + weak_challenge_count

        # Phase 10.7b: Explicit challenge declaration overrides everything.
        # If a line starts with "challenge:" or "challenge back:" or "challenge to",
        # the agent is declaratively stating intent — this beats any amount of
        # polite acknowledgment ("your analysis is solid, BUT challenge: ...")
        for line in content_lower.split("\n"):
            stripped = line.strip().lstrip("*").strip()
            if (stripped.startswith("challenge:") or
                    stripped.startswith("challenge back:") or
                    stripped.startswith("challenge to")):
                if self._is_substantive_challenge(content_lower, input_msg):
                    return MessageType.CHALLENGE
                return MessageType.PROPOSITION

        # Phase 10.5: Anti-gaming logic
        if agreement_count > 0 and total_challenge_count > 0:
            # Agreement + only weak challenges → AGREEMENT wins
            if strong_challenge_count == 0:
                return MessageType.AGREEMENT
            # Strong challenges present → they dominate when >= agreement count
            if strong_challenge_count >= agreement_count:
                if self._is_substantive_challenge(content_lower, input_msg):
                    return MessageType.CHALLENGE
                return MessageType.PROPOSITION
            # Agreement clearly dominates
            return MessageType.AGREEMENT

        # Pure agreement
        if agreement_count > 0:
            return MessageType.AGREEMENT

        # Pure challenge — with substance check
        if total_challenge_count > 0:
            if self._is_substantive_challenge(content_lower, input_msg):
                return MessageType.CHALLENGE
            return MessageType.PROPOSITION

        # Accommodation (responding to challenge with acknowledgment)
        accommodation_markers = md.accommodation_markers
        if any(m in content_lower for m in accommodation_markers):
            if input_msg and input_msg.message_type == MessageType.CHALLENGE:
                return MessageType.ACCOMMODATION
            return MessageType.PROPOSITION

        # Default to proposition
        return MessageType.PROPOSITION

    def _is_substantive_challenge(
        self,
        content_lower: str,
        input_msg: Optional[Message]
    ) -> bool:
        """
        Check if a challenge is substantive (references specific content).

        Phase 10.5: A challenge must either:
        1. Reference significant words from the prior message, OR
        2. Contain a strong challenge marker (already filtered upstream)

        The threshold adapts: with few significant prior words, 1 match suffices.
        With many, 2+ matches required.

        This prevents "However, I completely agree" from being classified
        as CHALLENGE just because of the "however," marker.
        """
        if input_msg is None:
            return True

        prior_content = input_msg.content.lower()
        # Extract significant words from prior message (>4 chars, not common)
        prior_words = set(
            w for w in re.findall(r'\b\w{5,}\b', prior_content)
            if w not in self._COMMON_WORDS
        )

        if not prior_words:
            return True  # Can't check reference, allow it

        # Count how many prior words appear in current content
        matches = sum(1 for w in prior_words if w in content_lower)

        # Adaptive threshold: need 1 match if few prior words, 2 if many
        threshold = 1 if len(prior_words) <= 3 else 2
        return matches >= threshold

    def _extract_insight(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract insight from response.

        Derived from: PROJECT-PLAN.md Phase 0.3
        Patterns: INSIGHT: lines, symbolic operators (=, ->, etc.)

        Insight patterns from AGENTS.md:
        - Decomposition: X = Y + Z
        - Implication: X -> Y
        - Contrast: X != Y
        - Resolution: conflict: X -> Y
        - Discovery: hidden: X
        - Connection: X <-> Y

        Returns:
            (insight_full, insight_display)
            - insight_full: Complete insight for corpus/derivation (never truncated)
            - insight_display: Truncated <60 chars for CLI display
        """
        lines = content.split("\n")

        # Strategy 1: Explicit INSIGHT: line
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.lower().startswith("insight:"):
                raw_insight = line_stripped[8:].strip()
                return self._clean_insight(raw_insight)

        # Strategy 2: Lines with symbolic patterns
        for line in lines:
            line_stripped = line.strip().lstrip("- ").lstrip("* ")

            # Check for insight patterns
            if " = " in line_stripped and "+" in line_stripped:  # Decomposition
                if len(line_stripped) < 200:  # Reasonable upper bound
                    return self._clean_insight(line_stripped)

            if " → " in line_stripped or " -> " in line_stripped:  # Implication
                if len(line_stripped) < 200:
                    return self._clean_insight(line_stripped)

            if " ≠ " in line_stripped or " != " in line_stripped:  # Contrast
                if len(line_stripped) < 200:
                    return self._clean_insight(line_stripped)

            if line_stripped.lower().startswith("conflict:"):  # Resolution
                return self._clean_insight(line_stripped)

            if line_stripped.lower().startswith("hidden:"):  # Discovery
                return self._clean_insight(line_stripped)

        return None, None

    def _clean_insight(self, insight: str) -> Tuple[str, str]:
        """
        Clean and normalize insight text.

        Returns:
            (insight_full, insight_display)
            - insight_full: Cleaned but NOT truncated (for corpus)
            - insight_display: Truncated <60 chars (for CLI)

        Derived from: AGENTS.md - display should be <60 chars, but full preserved
        """
        # Remove markdown formatting
        insight = re.sub(r'\*\*(.+?)\*\*', r'\1', insight)
        insight = re.sub(r'\*(.+?)\*', r'\1', insight)
        insight = re.sub(r'`(.+?)`', r'\1', insight)

        # Remove leading markers
        insight_full = insight.lstrip("→ ").lstrip("- ").lstrip("• ").strip()

        # Create display version (truncated for CLI)
        if len(insight_full) > 60:
            truncated = insight_full[:57]
            last_space = truncated.rfind(" ")
            if last_space > 40:
                insight_display = truncated[:last_space] + "..."
            else:
                insight_display = truncated + "..."
        else:
            insight_display = insight_full

        return insight_full, insight_display

    def _check_insight_provenance(self, insight: str, state: SharedState) -> tuple:
        """
        Check that an insight traces back via stratified provenance gates.

        Phase 23: Stratified Provenance — three levels of trust.

        Stratum 0 (User Input): Insight stems match original input stems.
            Highest trust. Always checked first.

        Stratum 1 (Domain Entailment): Insight references corpus vocabulary
            terms AND at least 1 input stem. Single-hop from input via corpus
            knowledge. Requires domain_model in state.known.

        Stratum 2 (Corpus Patterns): Insight matches corpus archetype or
            relationship pattern names. Multi-hop, pattern must have ≥3
            backing compilations. Requires domain_model in state.known.

        Immutability: Higher strata cannot override lower. An insight accepted
        at stratum 0 stays stratum 0 even if it also matches stratum 1/2.

        Returns:
            (passed: bool, stratum: int) — stratum is -1 if rejected
        """
        original_input = state.known.get("input", "")
        if not original_input:
            return (True, 0)  # No input to check against, allow at stratum 0

        prov = PROTOCOL.provenance
        input_stems = {
            w[:prov.stem_length] for w in re.findall(r'\b\w{5,}\b', original_input.lower())
            if w not in self._COMMON_WORDS
        }

        if not input_stems:
            return (True, 0)  # No checkable terms in input, allow at stratum 0

        insight_stems = [
            w[:prov.stem_length] for w in re.findall(r'\b\w{5,}\b', insight.lower())
            if w not in self._COMMON_WORDS
        ]

        if not insight_stems:
            return (True, 0)  # No checkable terms in insight, allow at stratum 0

        # --- Stratum 0: User input stems ---
        input_matches = sum(1 for s in insight_stems if s in input_stems)
        if input_matches >= prov.min_matches:
            return (True, 0)

        # --- Stratum 1: Domain entailment (input + corpus vocabulary) ---
        domain_model = state.known.get("domain_model")
        if domain_model and self._check_stratum_1(insight, insight_stems, input_stems, domain_model):
            return (True, 1)

        # --- Stratum 2: Corpus patterns (archetype/pattern matching) ---
        if domain_model and self._check_stratum_2(insight, domain_model):
            return (True, 2)

        # --- Stratum 3: Self-observation patterns (self-compile loop) ---
        if self._check_stratum_3(insight, state):
            return (True, 3)

        return (False, -1)

    def _check_stratum_1(
        self,
        insight: str,
        insight_stems: list,
        input_stems: set,
        domain_model: dict,
    ) -> bool:
        """
        Stratum 1: Domain entailment gate.

        Accepts if the insight references a corpus vocabulary term AND has at
        least one connection to the original input. This is a single-hop
        extension: input → corpus vocabulary → insight.

        The vocabulary term must appear in ≥3 compilations (enforced by
        corpus extraction, not re-checked here).

        Args:
            insight: The candidate insight text
            insight_stems: Pre-extracted stems from the insight
            input_stems: Pre-extracted stems from the original input
            domain_model: Serialized DomainModel dict from state.known
        """
        vocabulary = domain_model.get("vocabulary", {})
        if not vocabulary:
            return False

        insight_lower = insight.lower()
        prov = PROTOCOL.provenance

        # Find vocabulary terms that appear in the insight.
        # For multi-word vocab terms, also check if any individual word appears.
        matching_vocab_terms = []
        for term in vocabulary:
            term_lower = term.lower()
            if term_lower in insight_lower:
                matching_vocab_terms.append(term_lower)
            else:
                # Check individual words of multi-word terms
                for word in term_lower.split():
                    if len(word) >= 5 and word in insight_lower:
                        matching_vocab_terms.append(term_lower)
                        break

        if not matching_vocab_terms:
            return False

        # Must also have at least 1 connection to input:
        # Either a direct insight stem match, or a matching vocab term
        # whose own stems trace back to input.

        # Direct insight stem match
        for s in insight_stems:
            if s in input_stems:
                return True

        # Vocab term stem tracing: does the matched vocab term contain
        # a word whose stem appears in the input?
        for term in matching_vocab_terms:
            term_stems = [
                w[:prov.stem_length] for w in re.findall(r'\b\w{5,}\b', term)
                if w not in self._COMMON_WORDS
            ]
            for ts in term_stems:
                if ts in input_stems:
                    return True

        return False

    def _check_stratum_2(
        self,
        insight: str,
        domain_model: dict,
    ) -> bool:
        """
        Stratum 2: Corpus pattern gate.

        Accepts if the insight matches a known corpus archetype name or
        relationship pattern. These patterns come from ≥3 prior compilations
        (enforced by corpus extraction).

        This is the lowest trust level — no input connection required,
        but the pattern must have strong corpus backing.

        Args:
            insight: The candidate insight text
            domain_model: Serialized DomainModel dict from state.known
        """
        insight_lower = insight.lower()

        # Check archetype names
        archetypes = domain_model.get("archetypes", [])
        for arch in archetypes:
            canonical = arch.get("canonical_name", "")
            if not canonical:
                continue
            # Archetype must have sufficient backing
            source_ids = arch.get("source_ids", [])
            if len(source_ids) < 3:
                continue
            # Check if canonical name (or any variant) appears in insight
            if canonical.lower() in insight_lower:
                return True
            for variant in arch.get("variants", []):
                if variant.lower() in insight_lower:
                    return True

        # Check relationship pattern component names
        patterns = domain_model.get("relationship_patterns", [])
        for pat in patterns:
            source_ids = pat.get("source_ids", [])
            if len(source_ids) < 3:
                continue
            components = pat.get("components", [])
            # At least 2 pattern components must appear in insight
            matches = sum(1 for c in components if c.lower() in insight_lower)
            if matches >= 2:
                return True

        return False

    def _check_stratum_3(
        self,
        insight: str,
        state: SharedState,
    ) -> bool:
        """
        Stratum 3: Self-observation pattern gate.

        Phase 24: Accepts if insight matches a SelfPattern.name from a prior
        self-compile loop. Only stable patterns qualify:
        - pattern_type must be "stable_component" or "stable_relationship"
        - Pattern must come from >=2 self-compile runs (frequency > 0)

        Drift points and canonical gaps are NOT trusted enough for stratum 3.

        Args:
            insight: The candidate insight text
            state: SharedState with self_compile_patterns

        Returns:
            True if insight matches a stable self-compile pattern
        """
        patterns = state.self_compile_patterns
        if not patterns:
            return False

        insight_lower = insight.lower()

        for pattern in patterns:
            ptype = pattern.get("pattern_type", "")
            # Only stable patterns qualify
            if ptype not in ("stable_component", "stable_relationship"):
                continue

            name = pattern.get("name", "")
            if not name:
                continue

            # Check if pattern name appears in insight (case-insensitive)
            if name.lower() in insight_lower:
                return True

        return False


# =============================================================================
# PHASE 2: Method Signature Extraction
# Derived from: ROADMAP.md Phase 2 - treat natural language as source code
# =============================================================================

# Pattern for method signatures: word(params) or word() -> type
# Examples:
#   add_known(name: str, value: Any)
#   snapshot() -> Dict
#   __init__(input_text: str)
METHOD_SIGNATURE_PATTERN = re.compile(
    r'(\w+)\s*\(([^)]*)\)(?:\s*(?:->|→)\s*(\w+(?:\[[\w\s,]+\])?))?'
)


def extract_method_signatures(text: str) -> list:
    """
    Extract method signatures from natural language text.

    Phase 2: Treat natural language as source code to be parsed.
    Only extract what is explicitly stated - never invent.

    Patterns recognized:
    - add_known(name: str, value: Any)
    - snapshot() -> Dict
    - __init__(input_text: str)
    - method_name(param1, param2) -> return_type

    Args:
        text: Natural language text potentially containing method signatures

    Returns:
        List of dicts with keys: name, parameters, return_type, derived_from
    """
    methods = []

    for match in METHOD_SIGNATURE_PATTERN.finditer(text):
        method_name = match.group(1)
        params_str = match.group(2)
        return_type = match.group(3) or "None"

        # Skip common false positives (single words that look like methods)
        false_positives = {'if', 'for', 'while', 'with', 'try', 'except', 'def', 'class', 'return', 'print'}
        if method_name.lower() in false_positives:
            continue

        # Parse parameters
        parameters = []
        if params_str.strip():
            for param in params_str.split(','):
                param = param.strip()
                if not param:
                    continue

                if ':' in param:
                    # Typed parameter: name: type
                    parts = param.split(':', 1)
                    param_name = parts[0].strip()
                    param_type = parts[1].strip()
                else:
                    # Untyped parameter
                    param_name = param
                    param_type = "Any"

                parameters.append({
                    "name": param_name,
                    "type_hint": param_type,
                    "default": None,
                    "derived_from": param
                })

        methods.append({
            "name": method_name,
            "parameters": parameters,
            "return_type": return_type,
            "description": "",
            "derived_from": match.group(0)  # Exact matched text
        })

    return methods


def extract_state_transitions(text: str) -> dict:
    """
    Extract state machine transitions from natural language text.

    Phase 2: Parse state transition patterns.

    Patterns recognized:
    - INIT -> ACTIVE -> HALTED
    - STATE_A → STATE_B
    - state1 -> state2 on event

    Args:
        text: Natural language text potentially containing state transitions

    Returns:
        Dict with keys: states, initial_state, transitions, derived_from
        Or None if no state machine found
    """
    # Pattern for state transitions: WORD -> WORD or WORD → WORD
    state_pattern = re.compile(r'([A-Z_]+)\s*(?:->|→)\s*([A-Z_]+)')

    states = set()
    transitions = []

    for match in state_pattern.finditer(text):
        from_state = match.group(1)
        to_state = match.group(2)

        states.add(from_state)
        states.add(to_state)

        transitions.append({
            "from_state": from_state,
            "to_state": to_state,
            "trigger": "",
            "derived_from": match.group(0)
        })

    if not states:
        return None

    # Determine initial state (first one mentioned, or one that's never a target)
    target_states = {t["to_state"] for t in transitions}
    initial_candidates = states - target_states
    initial_state = list(initial_candidates)[0] if initial_candidates else list(states)[0]

    return {
        "states": sorted(list(states)),
        "initial_state": initial_state,
        "transitions": transitions,
        "derived_from": text
    }
