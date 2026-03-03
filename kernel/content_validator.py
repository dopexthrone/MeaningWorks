"""
kernel/content_validator.py — Content-postcode alignment validator.

LEAF module. Uses the 5-axis postcode to validate whether cell content
matches its declared coordinate. The DIMENSION axis is the primary
discriminator: WHAT expects definitional language, HOW expects procedural,
WHY expects teleological, etc.

Returns ContentFit — a measurement instrument, not a gate. Score < 0.3
produces warnings but does not block fill().
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ContentFit:
    """Result of content-postcode validation."""
    score: float           # 0.0-1.0: how well content matches postcode
    dimension_match: float # 0.0-1.0: dimension-axis fit
    concern_match: float   # 0.0-1.0: concern-axis fit
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Dimension keyword sets — what vocabulary each dimension expects
# ---------------------------------------------------------------------------

_DIMENSION_KEYWORDS: dict[str, frozenset[str]] = {
    "WHAT": frozenset({
        "is", "defines", "represents", "contains", "consists",
        "entity", "structure", "component", "attribute", "type",
        "class", "object", "schema", "model", "definition",
        "data", "record", "field", "property", "interface",
        "named", "called", "known", "identifier", "identity",
    }),
    "HOW": frozenset({
        "process", "step", "method", "algorithm", "procedure",
        "execute", "performs", "runs", "calls", "invokes",
        "flow", "pipeline", "sequence", "workflow", "transforms",
        "converts", "computes", "calculates", "iterates", "loops",
        "sends", "receives", "dispatches", "handles", "routes",
    }),
    "WHY": frozenset({
        "purpose", "reason", "because", "enables", "ensures",
        "goal", "objective", "rationale", "motivation", "intent",
        "benefit", "value", "solves", "addresses", "prevents",
        "allows", "supports", "facilitates", "drives", "justifies",
    }),
    "WHO": frozenset({
        "user", "actor", "agent", "role", "owner",
        "responsible", "performs", "operator", "admin", "client",
        "service", "consumer", "producer", "sender", "receiver",
        "stakeholder", "participant", "authority", "team", "system",
    }),
    "WHEN": frozenset({
        "before", "after", "during", "triggers", "schedule",
        "timeout", "interval", "sequence", "order", "first",
        "then", "next", "finally", "until", "while",
        "event", "signal", "phase", "stage", "lifecycle",
    }),
    "WHERE": frozenset({
        "location", "module", "layer", "package", "directory",
        "endpoint", "path", "route", "scope", "boundary",
        "region", "zone", "namespace", "context", "environment",
        "server", "client", "local", "remote", "distributed",
    }),
    "IF": frozenset({
        "condition", "constraint", "requires", "unless", "when",
        "guard", "check", "validate", "assert", "threshold",
        "rule", "policy", "permission", "limit", "boundary",
        "precondition", "invariant", "assumption", "depends", "only",
    }),
    "HOW_MUCH": frozenset({
        "cost", "budget", "limit", "count", "quantity",
        "rate", "capacity", "threshold", "maximum", "minimum",
        "percentage", "ratio", "metric", "measure", "scale",
        "tokens", "bytes", "seconds", "latency", "throughput",
    }),
}

# ---------------------------------------------------------------------------
# Concern keyword sets — what vocabulary each concern expects
# ---------------------------------------------------------------------------

_CONCERN_KEYWORDS: dict[str, frozenset[str]] = {
    "ENT": frozenset({"entity", "object", "class", "model", "record", "schema", "type", "struct"}),
    "BHV": frozenset({"behavior", "action", "process", "flow", "transition", "event", "trigger"}),
    "FNC": frozenset({"function", "method", "call", "return", "parameter", "argument", "compute"}),
    "REL": frozenset({"relationship", "connection", "link", "depends", "references", "associates"}),
    "STA": frozenset({"state", "status", "current", "active", "idle", "pending", "transition"}),
    "PLN": frozenset({"plan", "schedule", "phase", "milestone", "goal", "strategy", "roadmap"}),
    "MEM": frozenset({"memory", "cache", "store", "recall", "history", "persist", "retrieve"}),
    "ORC": frozenset({"orchestrate", "coordinate", "pipeline", "dispatch", "route", "sequence"}),
    "AGT": frozenset({"agent", "autonomous", "decision", "observe", "act", "perceive"}),
    "GTE": frozenset({"gate", "check", "validate", "guard", "pass", "fail", "threshold"}),
    "PLY": frozenset({"policy", "rule", "permission", "access", "enforce", "govern"}),
    "MET": frozenset({"metric", "measure", "score", "count", "rate", "ratio", "monitor"}),
    "CNS": frozenset({"constraint", "limit", "bound", "require", "restrict", "invariant"}),
    "FLW": frozenset({"flow", "stream", "pipe", "channel", "route", "direction"}),
    "TRN": frozenset({"transition", "change", "move", "shift", "convert", "transform"}),
    "PRV": frozenset({"provenance", "source", "origin", "trace", "lineage", "derived"}),
}


# ---------------------------------------------------------------------------
# Stemmer — reuse pattern from closed_loop.py
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """Lightweight suffix-stripping stem."""
    prev = word
    for _ in range(3):
        stemmed = _stem_once(prev)
        if stemmed == prev:
            break
        prev = stemmed
    return prev


def _stem_once(word: str) -> str:
    """Single pass of suffix stripping."""
    if len(word) <= 4:
        return word
    for suffix in ("ation", "tion", "sion", "ment", "ness", "able", "ible",
                   "ful", "ing", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    for suffix in ("ed", "er", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("s") and len(word) == 4:
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    """Tokenize and lowercase text."""
    return {w.lower().strip(".,;:!?\"'()[]{}") for w in text.split() if len(w) >= 2}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_content(
    content: str,
    dimension: str,
    concern: str = "",
) -> ContentFit:
    """Validate content against its postcode axes.

    Args:
        content: Cell content text.
        dimension: Dimension axis value (WHAT, HOW, WHY, etc.).
        concern: Concern axis value (ENT, BHV, FNC, etc.). Optional.

    Returns:
        ContentFit with scores and warnings.
    """
    if not content or not content.strip():
        return ContentFit(
            score=0.0,
            dimension_match=0.0,
            concern_match=0.0,
            warnings=("empty content",),
        )

    tokens = _tokenize(content)
    stemmed_tokens = {_stem(t) for t in tokens}

    # --- Dimension match ---
    dim_keywords = _DIMENSION_KEYWORDS.get(dimension, frozenset())
    dim_match = _keyword_overlap(stemmed_tokens, dim_keywords)

    # --- Concern match ---
    concern_match = 0.0
    if concern:
        concern_keywords = _CONCERN_KEYWORDS.get(concern, frozenset())
        if concern_keywords:
            concern_match = _keyword_overlap(stemmed_tokens, concern_keywords)
        else:
            concern_match = 0.5  # Unknown concern — neutral

    # --- Composite score ---
    if concern and concern in _CONCERN_KEYWORDS:
        score = 0.7 * dim_match + 0.3 * concern_match
    else:
        score = dim_match

    # --- Warnings ---
    warnings: list[str] = []
    if dim_match < 0.3:
        warnings.append(f"low dimension fit: content doesn't match {dimension} vocabulary")
    if concern and concern in _CONCERN_KEYWORDS and concern_match < 0.2:
        warnings.append(f"low concern fit: content doesn't match {concern} vocabulary")

    return ContentFit(
        score=round(score, 4),
        dimension_match=round(dim_match, 4),
        concern_match=round(concern_match, 4),
        warnings=tuple(warnings),
    )


def _keyword_overlap(tokens: set[str], keywords: frozenset[str]) -> float:
    """Compute overlap between content tokens and keyword set.

    Uses stemmed matching: each keyword is stemmed and checked against
    stemmed content tokens. Scoring: each match contributes 0.15, saturating
    at 1.0. This means ~7 keyword matches = full score. Even 2-3 matches
    give a meaningful signal (0.3-0.45).
    """
    if not tokens or not keywords:
        return 0.0

    stemmed_keywords = {_stem(k) for k in keywords}
    matches = tokens & stemmed_keywords

    if not matches:
        return 0.0

    # Each match contributes 0.15, saturates at 1.0
    return min(1.0, len(matches) * 0.15)
