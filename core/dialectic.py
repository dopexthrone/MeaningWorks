"""
Motherlabs Dialectic Rounds — structured thesis/antithesis/synthesis for spec dialogue.

LEAF MODULE: imports only from stdlib and core.protocol_spec.
No imports from core.protocol, agents/, or mother/.

The spec dialogue runs as 3 structured rounds, each with 3 turns:
  1. THESIS   — Entity stakes structural position
  2. ANTITHESIS — Process challenges from behavioral lens
  3. SYNTHESIS — Entity accommodates or holds with justification

Between rounds, a provenance gate fires on the synthesis insight.
Pass → commit round, advance. Fail → re-enter with narrowed scope (max 2 retries).

Three rounds:
  Round 0 (THESIS): Establish positions. Angle: existence.
  Round 1 (STRESS_TEST): Test from weakest dimension angle. Adaptive.
  Round 2 (COLLAPSE): Final synthesis. No new challenges. Governor reads all 3 rounds.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.protocol_spec import PROTOCOL


# =============================================================================
# ENUMS
# =============================================================================


class DialecticRole(Enum):
    """Role within a single round's 3-turn sequence."""
    THESIS = "thesis"          # stake position
    ANTITHESIS = "antithesis"  # challenge position
    SYNTHESIS = "synthesis"    # accommodate or hold


class DialecticPhase(Enum):
    """Phase across rounds."""
    THESIS = "thesis"          # round 0: establish
    STRESS_TEST = "stress"     # round 1: test from new angle
    COLLAPSE = "collapse"      # round 2: final synthesis only


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass
class RoundOutput:
    """Captures the output of a single dialectic round."""
    round_number: int
    phase: DialecticPhase
    rotation_angle: str
    messages: list              # 3 Messages from this round
    insights: list              # insights extracted this round
    confidence_snapshot: dict   # ConfidenceVector.to_dict() at round end
    provenance_passed: bool = True
    gate_attempts: int = 0


# =============================================================================
# ROTATION ANGLES
# =============================================================================


ROTATION_ANGLES = (
    ("existence",    "What entities/concepts exist? What are their essential attributes?"),
    ("dynamics",     "How do things change? What are the state transitions, triggers, and flows?"),
    ("grounding",    "How do abstractions map to concrete implementation?"),
    ("constraints",  "What limits, rules, and invariants govern the system?"),
    ("state",        "What state is tracked? How does it evolve over time?"),
)

# Maps weakest confidence dimension → rotation angle index for round 1
_WEAKNESS_TO_ANGLE = {
    "structural": 2,   # grounding
    "behavioral": 1,   # dynamics
    "coverage": 4,     # state
    "consistency": 3,  # constraints
}


# =============================================================================
# ROUND PROMPT FRAGMENTS
# =============================================================================


ROUND_ROLE_PROMPTS = {
    DialecticRole.THESIS: (
        "ROUND ROLE: THESIS — Stake your position on the current angle. "
        "Be specific. Name components, attributes, and relationships. "
        "End with an INSIGHT: line summarizing your key structural claim."
    ),
    DialecticRole.ANTITHESIS: (
        "ROUND ROLE: ANTITHESIS — Challenge the thesis. "
        "Identify what was missed, assumed, or contradicted. "
        "Reference specific claims from the thesis. "
        "End with an INSIGHT: line exposing the gap or tension."
    ),
    DialecticRole.SYNTHESIS: (
        "ROUND ROLE: SYNTHESIS — Resolve the tension between thesis and antithesis. "
        "Accommodate valid challenges. Justify positions you hold. "
        "The synthesis must trace to both prior turns. "
        "End with an INSIGHT: line capturing the resolved position."
    ),
}

COLLAPSE_PROMPT = (
    "ROUND 3: FINAL SYNTHESIS — No new challenges permitted. "
    "Integrate insights from all prior rounds into a unified position. "
    "Every claim must trace to a prior round's insight. "
    "End with an INSIGHT: line capturing the final specification state."
)


# =============================================================================
# ROUND MANAGER
# =============================================================================


class RoundManager:
    """Controls the dialectic round structure for spec dialogue."""

    def __init__(self, max_gate_retries: int = PROTOCOL.dialectic.max_gate_retries):
        self.rounds: List[RoundOutput] = []
        self.current_round: int = 0
        self._gate_failures: int = 0
        self.max_gate_retries = max_gate_retries

    def current_phase(self) -> DialecticPhase:
        """Return the phase for the current round."""
        if self.current_round == 0:
            return DialecticPhase.THESIS
        elif self.current_round == 1:
            return DialecticPhase.STRESS_TEST
        else:
            return DialecticPhase.COLLAPSE

    def turn_role(self, turn_in_round: int) -> DialecticRole:
        """Map turn index within a round to a role."""
        if turn_in_round == 0:
            return DialecticRole.THESIS
        elif turn_in_round == 1:
            return DialecticRole.ANTITHESIS
        else:
            return DialecticRole.SYNTHESIS

    def rotation_angle_for_round(
        self, round_num: int, confidence=None
    ) -> Tuple[str, str]:
        """
        Return (angle_name, angle_prompt) for the given round.

        Round 0: always "existence" (Entity's natural lens).
        Round 1: adaptive — maps weakest confidence dimension to angle.
        Round 2+: "collapse" with synthesis-only prompt.
        """
        if round_num == 0:
            return ROTATION_ANGLES[0]

        if round_num >= 2:
            # Convergence-driven: rounds beyond initial 3 cycle through
            # exploration angles to cover uncovered territory
            if round_num >= 3:
                angle_idx = (round_num - 3) % len(ROTATION_ANGLES)
                angle_name, angle_prompt = ROTATION_ANGLES[angle_idx]
                return (
                    f"convergence_{angle_name}",
                    f"[CONVERGENCE ROUND] Revisit from {angle_name} lens — "
                    f"what was missed? {angle_prompt}",
                )
            return ("collapse", COLLAPSE_PROMPT)

        # Round 1: adaptive based on weakest dimension
        if confidence is not None and hasattr(confidence, 'weakest_dimension'):
            weakest = confidence.weakest_dimension()
            idx = _WEAKNESS_TO_ANGLE.get(weakest, 1)
            return ROTATION_ANGLES[idx]

        # Fallback: dynamics
        return ROTATION_ANGLES[1]

    def check_round_gate(self, round_output: RoundOutput, state) -> bool:
        """
        Provenance gate between rounds.

        Checks that the synthesis turn (message index 2) has an insight
        that passes provenance. Reuses existing stratum 0-3 checks via
        the insight field on the Message.

        Returns True if gate passes.
        """
        if len(round_output.messages) < 3:
            return False

        synthesis_msg = round_output.messages[2]

        # Gate requirement: synthesis must have produced a provenance-passing insight
        if synthesis_msg.insight is None:
            return False

        # If the insight exists on the message, it already passed provenance
        # (LLMAgent._check_insight_provenance strips it if it fails).
        return True

    def commit_round(self, round_output: RoundOutput):
        """Commit a completed round and advance."""
        self.rounds.append(round_output)
        self.current_round += 1
        self._gate_failures = 0

    def build_round_context(
        self, round_num: int, role: DialecticRole, prior_rounds: List[RoundOutput]
    ) -> str:
        """
        Build context string injected into agent state for the current turn.

        Includes: round number, phase, role prompt, rotation angle focus,
        prior round summaries.
        """
        phase = self.current_phase()
        _, angle_prompt = self.rotation_angle_for_round(round_num, None)

        parts = []
        parts.append(f"[DIALECTIC ROUND {round_num + 1}/3 — {phase.value.upper()}]")

        # Role prompt
        if phase == DialecticPhase.COLLAPSE:
            parts.append(COLLAPSE_PROMPT)
        else:
            parts.append(ROUND_ROLE_PROMPTS.get(role, ""))

        # Angle focus (for non-collapse rounds)
        if phase != DialecticPhase.COLLAPSE:
            parts.append(f"FOCUS: {angle_prompt}")

        # Prior round summaries
        if prior_rounds:
            parts.append("")
            parts.append("PRIOR ROUNDS:")
            for pr in prior_rounds:
                insights_str = "; ".join(pr.insights[:3]) if pr.insights else "none"
                conf = pr.confidence_snapshot
                conf_str = (
                    f"S={conf.get('structural', 0):.1f} "
                    f"B={conf.get('behavioral', 0):.1f} "
                    f"C={conf.get('coverage', 0):.1f}"
                ) if conf else "n/a"
                parts.append(
                    f"  Round {pr.round_number + 1} ({pr.phase.value}): "
                    f"angle={pr.rotation_angle}, "
                    f"insights=[{insights_str}], "
                    f"confidence=({conf_str})"
                )

        return "\n".join(parts)

    def narrow_scope(self, round_output: RoundOutput, state) -> str:
        """
        Generate a Governor message narrowing scope after a gate failure.

        Extracts uncovered ground and unresolved unknowns from state.
        """
        parts = ["RETRY: Synthesis failed provenance gate."]

        # Uncovered ground from intent
        intent = state.known.get("intent", {}) if hasattr(state, 'known') else {}
        explicit = intent.get("explicit_components", [])
        if explicit:
            parts.append(f"Uncovered components: {', '.join(explicit[:3])}.")

        # Unresolved unknowns
        unknowns = getattr(state, 'unknown', [])
        if unknowns:
            parts.append(f"Unresolved unknowns: {', '.join(unknowns[:3])}.")

        parts.append(
            "Narrow focus to these specifics. "
            "Trace every claim to the original input."
        )

        return " ".join(parts)
