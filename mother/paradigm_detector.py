"""
Paradigm shift detection — stagnation, regression, and failure pattern analysis.

LEAF module. Genome #185 (paradigm-shifting).

All functions are pure — no external API calls, no LLM invocations.
Statistical analysis over compilation score sequences and failure logs.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_STAGNATION_THRESHOLD = 5       # consecutive similar scores
_REGRESSION_THRESHOLD = 3       # consecutive declining scores
_STAGNATION_VARIANCE = 0.02     # max variance for plateau detection

_SHIFT_CATEGORIES: Dict[str, FrozenSet[str]] = {
    "technical": frozenset({"rewrite", "refactor", "architecture", "framework", "language", "platform"}),
    "process": frozenset({"workflow", "pipeline", "methodology", "approach", "strategy", "paradigm"}),
    "market": frozenset({"pivot", "market", "customer", "audience", "positioning", "segment"}),
    "capability": frozenset({"limitation", "gap", "missing", "unable", "insufficient", "inadequate"}),
}


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParadigmShiftSignal:
    """A single signal that a paradigm shift may be needed."""
    signal_type: str       # "stagnation", "regression", "pattern"
    category: str          # from _SHIFT_CATEGORIES or "general"
    severity: str          # "low", "moderate", "high"
    evidence: str
    recommendation: str
    summary: str


@dataclass(frozen=True)
class ParadigmAssessment:
    """Complete paradigm shift assessment."""
    signals: Tuple[ParadigmShiftSignal, ...]
    shift_recommended: bool
    urgency: str           # "none", "low", "moderate", "high"
    assessment_summary: str


# ---------------------------------------------------------------------------
# #185 — Paradigm-shifting: detection functions
# ---------------------------------------------------------------------------

def detect_stagnation(scores: List[float], window: int = _STAGNATION_THRESHOLD) -> bool:
    """Detect score stagnation (plateau).

    Returns True if variance of last `window` scores < _STAGNATION_VARIANCE.
    """
    if len(scores) < window:
        return False

    recent = scores[-window:]
    mean = sum(recent) / len(recent)
    variance = sum((s - mean) ** 2 for s in recent) / len(recent)
    return variance < _STAGNATION_VARIANCE


def detect_regression(scores: List[float], window: int = _REGRESSION_THRESHOLD) -> bool:
    """Detect score regression (monotonic decline).

    Returns True if last `window` scores are monotonically decreasing.
    """
    if len(scores) < window:
        return False

    recent = scores[-window:]
    for i in range(1, len(recent)):
        if recent[i] >= recent[i - 1]:
            return False
    return True


def classify_failure_patterns(failure_reasons: List[str]) -> List[str]:
    """Categorize failure strings into shift categories.

    Uses keyword intersection with _SHIFT_CATEGORIES.
    Returns list of category names detected in failure reasons.
    """
    if not failure_reasons:
        return []

    combined = " ".join(r.lower() for r in failure_reasons)
    words = frozenset(combined.split())

    detected: List[str] = []
    for category, keywords in _SHIFT_CATEGORIES.items():
        if words & keywords:
            detected.append(category)

    return detected


def assess_paradigm_shift(
    compilation_scores: List[float],
    failure_reasons: List[str],
    goal_completion_rate: float = 1.0,
    stuck_count: int = 0,
) -> ParadigmAssessment:
    """Assess whether a paradigm shift is needed.

    Combines stagnation detection, regression detection, failure pattern
    classification, and goal/stuck metrics.

    Returns frozen ParadigmAssessment.
    """
    signals: List[ParadigmShiftSignal] = []

    # Check stagnation
    if detect_stagnation(compilation_scores):
        mean_score = sum(compilation_scores[-_STAGNATION_THRESHOLD:]) / _STAGNATION_THRESHOLD
        signals.append(ParadigmShiftSignal(
            signal_type="stagnation",
            category="general",
            severity="moderate" if mean_score < 0.7 else "low",
            evidence=f"Last {_STAGNATION_THRESHOLD} scores show plateau (variance < {_STAGNATION_VARIANCE})",
            recommendation="Consider restructuring approach — current method has plateaued",
            summary=f"Score plateau detected at ~{mean_score:.2f}",
        ))

    # Check regression
    if detect_regression(compilation_scores):
        drop = compilation_scores[-_REGRESSION_THRESHOLD] - compilation_scores[-1]
        signals.append(ParadigmShiftSignal(
            signal_type="regression",
            category="general",
            severity="high" if drop > 0.2 else "moderate",
            evidence=f"Last {_REGRESSION_THRESHOLD} scores are monotonically declining (drop: {drop:.2f})",
            recommendation="Halt current approach — quality is degrading",
            summary=f"Score regression: {drop:.2f} decline over {_REGRESSION_THRESHOLD} compilations",
        ))

    # Classify failure patterns
    failure_categories = classify_failure_patterns(failure_reasons)
    for category in failure_categories:
        signals.append(ParadigmShiftSignal(
            signal_type="pattern",
            category=category,
            severity="moderate",
            evidence=f"Failure pattern detected in category: {category}",
            recommendation=f"Address {category} failures before continuing",
            summary=f"Recurring {category} failures",
        ))

    # Stuck count signal
    if stuck_count >= 3:
        signals.append(ParadigmShiftSignal(
            signal_type="stagnation",
            category="capability",
            severity="high" if stuck_count >= 5 else "moderate",
            evidence=f"{stuck_count} goals are stuck",
            recommendation="Re-evaluate approach — multiple goals cannot progress",
            summary=f"{stuck_count} stuck goals",
        ))

    # Goal completion rate signal
    if goal_completion_rate < 0.3:
        signals.append(ParadigmShiftSignal(
            signal_type="regression",
            category="process",
            severity="high",
            evidence=f"Goal completion rate: {goal_completion_rate:.0%}",
            recommendation="Current process is not producing results — fundamental change needed",
            summary=f"Low completion rate: {goal_completion_rate:.0%}",
        ))

    # Determine urgency
    signal_count = len(signals)
    if signal_count >= 3:
        urgency = "high"
    elif signal_count >= 2:
        urgency = "moderate"
    elif signal_count >= 1:
        urgency = "low"
    else:
        urgency = "none"

    shift_recommended = urgency in ("moderate", "high")

    # Summary
    if not signals:
        assessment_summary = "No paradigm shift signals detected. Current approach is progressing."
    else:
        categories = set(s.category for s in signals if s.category != "general")
        cat_str = ", ".join(sorted(categories)) if categories else "general"
        assessment_summary = (
            f"{signal_count} paradigm shift signal(s) detected in: {cat_str}. "
            f"Urgency: {urgency}. Shift {'recommended' if shift_recommended else 'not yet recommended'}."
        )

    return ParadigmAssessment(
        signals=tuple(signals),
        shift_recommended=shift_recommended,
        urgency=urgency,
        assessment_summary=assessment_summary,
    )
