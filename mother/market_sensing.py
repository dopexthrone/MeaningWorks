"""
Market sensing — heuristic market timing and demand detection.

LEAF module. Genome #35 (market-sensing).

All functions are pure — no external API calls, no LLM invocations.
Keyword-intersection analysis over description text.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_TREND_SIGNALS: FrozenSet[str] = frozenset({
    "growing", "emerging", "declining", "saturated", "disrupted",
    "trending", "booming", "stagnant", "volatile", "shifting",
})

_DEMAND_SIGNALS: FrozenSet[str] = frozenset({
    "demand", "need", "want", "shortage", "gap",
    "underserved", "opportunity", "unmet", "waiting", "requesting",
})

_RISK_SIGNALS: FrozenSet[str] = frozenset({
    "risk", "volatile", "uncertain", "recession", "downturn",
    "regulation", "compliance", "lawsuit", "competitor", "threat",
})

_TIMING_SIGNALS: Dict[str, FrozenSet[str]] = {
    "early": frozenset({"emerging", "new", "novel", "pioneering", "first"}),
    "growth": frozenset({"growing", "expanding", "scaling", "accelerating", "booming"}),
    "mature": frozenset({"established", "mature", "stable", "commoditized", "legacy"}),
    "decline": frozenset({"declining", "shrinking", "dying", "obsolete", "contracting"}),
}


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSignal:
    """Market timing and demand assessment."""
    trend_direction: str          # "up", "down", "flat", "unknown"
    market_phase: str             # "early", "growth", "mature", "decline", "unknown"
    demand_strength: str          # "strong", "moderate", "weak"
    risk_level: str               # "high", "moderate", "low"
    signal_count: int
    signals_detected: Tuple[str, ...]
    timing_recommendation: str
    summary: str


# ---------------------------------------------------------------------------
# #35 — Market-sensing: signal detection and timing assessment
# ---------------------------------------------------------------------------

def detect_market_signals(description: str) -> List[str]:
    """Detect market signals from a text description.

    Uses frozenset intersection on trend, demand, and risk signal sets.
    Returns list of detected signal strings.
    """
    if not description:
        return []

    words = frozenset(description.lower().split())
    detected: List[str] = []

    for signal_set in (_TREND_SIGNALS, _DEMAND_SIGNALS, _RISK_SIGNALS):
        detected.extend(sorted(words & signal_set))

    return detected


def assess_market_timing(
    description: str,
    competitors: Optional[List[str]] = None,
) -> MarketSignal:
    """Assess market timing from description and competitor landscape.

    Counts trend/demand/risk signals, classifies market phase,
    and generates a timing recommendation.

    Returns frozen MarketSignal.
    """
    if not description:
        return MarketSignal(
            trend_direction="unknown",
            market_phase="unknown",
            demand_strength="weak",
            risk_level="low",
            signal_count=0,
            signals_detected=(),
            timing_recommendation="Insufficient data for timing assessment",
            summary="No description provided.",
        )

    words = frozenset(description.lower().split())

    # Count signals by category
    trend_hits = words & _TREND_SIGNALS
    demand_hits = words & _DEMAND_SIGNALS
    risk_hits = words & _RISK_SIGNALS

    all_signals = sorted(trend_hits | demand_hits | risk_hits)
    signal_count = len(all_signals)

    # Trend direction
    up_words = {"growing", "emerging", "booming", "trending"}
    down_words = {"declining", "saturated", "stagnant", "disrupted"}
    up_count = len(trend_hits & up_words)
    down_count = len(trend_hits & down_words)

    if up_count > down_count:
        trend_direction = "up"
    elif down_count > up_count:
        trend_direction = "down"
    elif trend_hits:
        trend_direction = "flat"
    else:
        trend_direction = "unknown"

    # Market phase from timing signals
    market_phase = "unknown"
    best_phase_score = 0
    for phase, phase_words in _TIMING_SIGNALS.items():
        phase_score = len(words & phase_words)
        if phase_score > best_phase_score:
            best_phase_score = phase_score
            market_phase = phase

    # Demand strength
    demand_count = len(demand_hits)
    if demand_count >= 3:
        demand_strength = "strong"
    elif demand_count >= 1:
        demand_strength = "moderate"
    else:
        demand_strength = "weak"

    # Risk level
    risk_count = len(risk_hits)
    if risk_count >= 3:
        risk_level = "high"
    elif risk_count >= 1:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Timing recommendation based on phase + competitors
    competitor_count = len(competitors) if competitors else 0

    if market_phase == "early":
        if competitor_count <= 2:
            timing_recommendation = "First-mover advantage — enter now before competition increases"
        else:
            timing_recommendation = "Early market with existing competition — differentiate strongly"
    elif market_phase == "growth":
        if competitor_count <= 5:
            timing_recommendation = "Growth phase with room — enter with proven execution"
        else:
            timing_recommendation = "Crowded growth market — find underserved niche"
    elif market_phase == "mature":
        timing_recommendation = "Mature market — compete on cost, quality, or disruption"
    elif market_phase == "decline":
        timing_recommendation = "Declining market — avoid unless pivoting the category"
    else:
        timing_recommendation = "Market phase unclear — gather more data before committing"

    # Summary
    summary = (
        f"Market phase: {market_phase}, trend: {trend_direction}, "
        f"demand: {demand_strength}, risk: {risk_level}. "
        f"{signal_count} signal(s) detected."
    )

    return MarketSignal(
        trend_direction=trend_direction,
        market_phase=market_phase,
        demand_strength=demand_strength,
        risk_level=risk_level,
        signal_count=signal_count,
        signals_detected=tuple(all_signals),
        timing_recommendation=timing_recommendation,
        summary=summary,
    )
