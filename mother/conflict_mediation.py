"""
Conflict mediation — multi-party conflict detection and resolution strategy generation.

LEAF module. Genome #139 (conflict-mediating).

All functions are pure — no external API calls, no LLM invocations.
Heuristic keyword analysis over structured inputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_CONFLICT_SIGNALS: Dict[str, List[str]] = {
    "priority": ["urgent", "critical", "blocker", "must-have", "non-negotiable", "deadline", "asap"],
    "resource": ["budget", "headcount", "bandwidth", "capacity", "allocation", "overloaded", "stretched"],
    "technical": ["incompatible", "breaking change", "migration", "deprecated", "rewrite", "refactor"],
    "scope": ["scope creep", "out of scope", "additional", "expanded", "requirements changed", "moving target"],
    "interpersonal": ["disagree", "conflict", "tension", "frustrated", "miscommunication", "blame", "ownership"],
    "architectural": ["monolith", "microservice", "framework", "language choice", "database", "api design"],
}

_CONFLICT_SEVERITY: Dict[str, int] = {
    "priority": 4,
    "resource": 3,
    "technical": 3,
    "scope": 2,
    "interpersonal": 4,
    "architectural": 3,
}

_RESOLUTION_STRATEGIES: Dict[str, Dict[str, str]] = {
    "priority": {
        "strategy": "structured-triage",
        "description": "Use impact/effort matrix to objectively rank priorities. Separate must-haves from nice-to-haves with stakeholder alignment.",
        "facilitation": "Run a time-boxed prioritization session with all parties. Use voting or weighted scoring to depersonalize the ranking.",
    },
    "resource": {
        "strategy": "constraint-based-planning",
        "description": "Map actual capacity against commitments. Make trade-offs visible — cutting scope is better than burning out teams.",
        "facilitation": "Present capacity data transparently. Ask each party to propose what they'd cut if forced to reduce scope by 30%.",
    },
    "technical": {
        "strategy": "spike-and-evaluate",
        "description": "Run a time-boxed technical investigation before committing. Let evidence resolve the disagreement, not authority.",
        "facilitation": "Propose a 2-day spike with clear evaluation criteria agreed upfront. The data decides.",
    },
    "scope": {
        "strategy": "scope-freeze-and-review",
        "description": "Freeze current scope, document all change requests, and batch-review with cost/impact analysis.",
        "facilitation": "Establish a change request process. Every addition must answer: what does this replace, or what timeline does it extend?",
    },
    "interpersonal": {
        "strategy": "structured-dialogue",
        "description": "Separate positions from interests. Each party states what they need (not what they want) and why.",
        "facilitation": "Use 'I need X because Y' framing. Find shared interests before negotiating positions.",
    },
    "architectural": {
        "strategy": "decision-record",
        "description": "Document options with trade-offs using ADR (Architecture Decision Record) format. Time-box the decision.",
        "facilitation": "Each side writes a one-page proposal with pros/cons/risks. Review together, decide, and commit — no revisiting without new evidence.",
    },
}


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConflictAnalysis:
    """Structured analysis of a detected conflict."""
    conflict_types: Tuple[str, ...]  # detected conflict categories
    severity: str  # "low", "moderate", "high", "critical"
    severity_score: int  # 0-10
    parties_involved: Tuple[str, ...]
    resolution_strategies: Tuple[Dict[str, str], ...]
    recommended_strategy: str
    summary: str


# ---------------------------------------------------------------------------
# #139 — Conflict-mediating: detection and resolution
# ---------------------------------------------------------------------------

def detect_conflicts(
    statements: List[str],
    parties: Optional[List[str]] = None,
) -> List[str]:
    """Detect conflict types from a list of statements.

    Returns list of detected conflict category names.
    """
    if not statements:
        return []

    combined = " ".join(s.lower() for s in statements)
    detected: List[str] = []

    for category, signals in _CONFLICT_SIGNALS.items():
        if any(signal in combined for signal in signals):
            detected.append(category)

    return detected


def classify_conflict_type(statements: List[str]) -> Dict[str, object]:
    """Classify the primary conflict type and its characteristics.

    Returns dict with: primary_type, all_types, signal_count, confidence.
    """
    if not statements:
        return {
            "primary_type": "none",
            "all_types": [],
            "signal_count": 0,
            "confidence": "n/a",
        }

    combined = " ".join(s.lower() for s in statements)

    type_scores: List[Tuple[str, int]] = []
    total_signals = 0

    for category, signals in _CONFLICT_SIGNALS.items():
        count = sum(1 for signal in signals if signal in combined)
        if count > 0:
            type_scores.append((category, count))
            total_signals += count

    if not type_scores:
        return {
            "primary_type": "none",
            "all_types": [],
            "signal_count": 0,
            "confidence": "low",
        }

    type_scores.sort(key=lambda x: x[1], reverse=True)
    primary = type_scores[0][0]
    all_types = [t[0] for t in type_scores]

    confidence = "high" if total_signals >= 4 else "medium" if total_signals >= 2 else "low"

    return {
        "primary_type": primary,
        "all_types": all_types,
        "signal_count": total_signals,
        "confidence": confidence,
    }


def generate_resolution_strategy(
    statements: List[str],
    parties: Optional[List[str]] = None,
    context: str = "",
) -> ConflictAnalysis:
    """Generate a complete conflict analysis with resolution strategies.

    Detects conflict types, scores severity, and recommends resolution approaches.

    Returns frozen ConflictAnalysis.
    """
    involved = tuple(parties or [])

    # Detect conflicts
    conflict_types = detect_conflicts(statements, parties)

    if not conflict_types:
        return ConflictAnalysis(
            conflict_types=(),
            severity="low",
            severity_score=0,
            parties_involved=involved,
            resolution_strategies=(),
            recommended_strategy="none",
            summary="No conflicts detected in the provided statements.",
        )

    # Score severity
    severity_score = sum(
        _CONFLICT_SEVERITY.get(ct, 1) for ct in conflict_types
    )
    severity_score = min(10, severity_score)

    if severity_score >= 8:
        severity = "critical"
    elif severity_score >= 5:
        severity = "high"
    elif severity_score >= 3:
        severity = "moderate"
    else:
        severity = "low"

    # Get resolution strategies for each detected type
    strategies: List[Dict[str, str]] = []
    for ct in conflict_types:
        if ct in _RESOLUTION_STRATEGIES:
            strategies.append({
                "type": ct,
                **_RESOLUTION_STRATEGIES[ct],
            })

    # Recommend the strategy for the highest-severity conflict type
    ranked = sorted(conflict_types, key=lambda t: _CONFLICT_SEVERITY.get(t, 0), reverse=True)
    recommended = _RESOLUTION_STRATEGIES.get(ranked[0], {}).get("strategy", "structured-dialogue")

    # Context boost
    context_lower = context.lower()
    if "deadline" in context_lower or "launch" in context_lower:
        severity_score = min(10, severity_score + 1)
        if severity_score >= 8 and severity != "critical":
            severity = "critical"

    # Summary
    type_list = ", ".join(conflict_types)
    party_note = f" between {', '.join(involved)}" if involved else ""
    summary = (
        f"Detected {len(conflict_types)} conflict type(s): {type_list}{party_note}. "
        f"Severity: {severity} ({severity_score}/10). "
        f"Recommended approach: {recommended}."
    )

    return ConflictAnalysis(
        conflict_types=tuple(conflict_types),
        severity=severity,
        severity_score=severity_score,
        parties_involved=involved,
        resolution_strategies=tuple(strategies),
        recommended_strategy=recommended,
        summary=summary,
    )
