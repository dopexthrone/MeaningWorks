"""
Motherlabs Component Classification — deterministic classification engine.

Phase 15: Component Classification Algorithm
Derived from: ROADMAP.md Phase 15 — explicit rules replace LLM black box.

Problem: DECOMPOSE stage relies on LLM to assign component types. "FlashDesign"
(1 mention) gets the same treatment as "Artist" (5+ mentions). This module
provides deterministic scoring to classify and filter components BEFORE synthesis.

Signals:
1. Mention frequency — how many times a term appears in input + dialogue
2. Grammatical role — subject (actor) vs object (data) vs modifier (attribute)
3. Semantic centrality — relationship density in the current blueprint graph
4. Type confidence — how confidently we can assign entity/process/interface/etc.

LLM fallback: only when confidence < 0.6.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set, FrozenSet


@dataclass(frozen=True)
class ClassificationScore:
    """
    Classification assessment for a single component candidate.

    All scores in [0, 1]. Combined into overall confidence.
    """
    name: str
    mention_frequency: float     # 0-1 normalized frequency
    grammatical_role: str        # "subject" | "object" | "modifier" | "unknown"
    semantic_centrality: float   # 0-1 based on relationship density
    inferred_type: str           # "entity" | "process" | "interface" | "event" | "subsystem" | "agent"
    type_confidence: float       # 0-1 how sure we are about the type
    is_component: bool           # True if likely a real component, not an attribute
    overall_confidence: float    # 0-1 combined score
    reasoning: str               # Why this classification


# =============================================================================
# MENTION FREQUENCY
# =============================================================================

def compute_mention_frequency(
    name: str,
    input_text: str,
    dialogue_history: List[str],
) -> float:
    """
    Compute normalized mention frequency of a name in input + dialogue.

    Higher frequency = more likely a real component (not incidental mention).

    Args:
        name: Component candidate name
        input_text: Original user input
        dialogue_history: List of dialogue message content strings

    Returns:
        Normalized frequency in [0, 1]
    """
    name_lower = name.lower()
    # Also search for individual words in multi-word names
    words = name_lower.split()

    # Count in input (weight 2x — user mentioned it)
    input_lower = input_text.lower()
    input_count = input_lower.count(name_lower)
    # Also count individual words for multi-word names
    if len(words) > 1:
        input_count += sum(input_lower.count(w) for w in words if len(w) > 3) * 0.5

    # Count in dialogue
    dialogue_count = 0
    for msg in dialogue_history:
        msg_lower = msg.lower()
        dialogue_count += msg_lower.count(name_lower)
        if len(words) > 1:
            dialogue_count += sum(msg_lower.count(w) for w in words if len(w) > 3) * 0.3

    # Weighted total: input mentions count double
    total = input_count * 2.0 + dialogue_count

    # Normalize: 1 mention = 0.1, 5+ mentions = 0.5, 10+ = 0.8, 20+ = 1.0
    if total <= 0:
        return 0.0
    elif total <= 1:
        return 0.1
    elif total <= 3:
        return 0.3
    elif total <= 5:
        return 0.5
    elif total <= 10:
        return 0.7
    elif total <= 20:
        return 0.85
    else:
        return 1.0


# =============================================================================
# GRAMMATICAL ROLE
# =============================================================================

# Patterns indicating actor/subject role
_SUBJECT_PATTERNS = [
    re.compile(r'\b{}\s+(?:handles?|manages?|processes?|triggers?|monitors?|creates?|generates?|orchestrates?|controls?)\b', re.IGNORECASE),
    re.compile(r'\b{}\s+(?:is responsible|should|must|will|can)\b', re.IGNORECASE),
    re.compile(r'\b{}\s+agent\b', re.IGNORECASE),
]

# Patterns indicating object/data role
_OBJECT_PATTERNS = [
    re.compile(r'\b(?:stores?|contains?|holds?|tracks?)\s+{}\b', re.IGNORECASE),
    re.compile(r'\b{}\s+(?:data|record|entry|item|object|instance)\b', re.IGNORECASE),
    re.compile(r'\b(?:the|each|every|a|an)\s+{}\b', re.IGNORECASE),
]

# Patterns indicating modifier/attribute role
_MODIFIER_PATTERNS = [
    re.compile(r'\b{}\s+(?:value|field|property|attribute|flag|status|state|count|level|score)\b', re.IGNORECASE),
    re.compile(r'\b(?:of|in|from)\s+{}\b', re.IGNORECASE),
]


def detect_grammatical_role(
    name: str,
    input_text: str,
    dialogue_history: List[str],
    subject_patterns: Optional[Tuple] = None,
    object_patterns: Optional[Tuple] = None,
) -> str:
    """
    Detect whether a name is used as subject (actor), object (data), or modifier (attribute).

    Heuristic-based. Looks at surrounding context in input and dialogue.

    Args:
        name: Component name to check
        input_text: Original user input
        dialogue_history: List of dialogue strings
        subject_patterns: Optional domain-specific subject regex strings (with {} placeholder)
        object_patterns: Optional domain-specific object regex strings (with {} placeholder)

    Returns:
        "subject" | "object" | "modifier" | "unknown"
    """
    all_text = input_text + " " + " ".join(dialogue_history)
    name_escaped = re.escape(name)

    subject_score = 0
    object_score = 0
    modifier_score = 0

    # Use adapter patterns if provided, else default compiled patterns
    if subject_patterns:
        for pat_str in subject_patterns:
            compiled = re.compile(pat_str.format(name_escaped), re.IGNORECASE)
            subject_score += len(compiled.findall(all_text))
    else:
        for pattern in _SUBJECT_PATTERNS:
            compiled = re.compile(pattern.pattern.format(name_escaped), re.IGNORECASE)
            subject_score += len(compiled.findall(all_text))

    if object_patterns:
        for pat_str in object_patterns:
            compiled = re.compile(pat_str.format(name_escaped), re.IGNORECASE)
            object_score += len(compiled.findall(all_text))
    else:
        for pattern in _OBJECT_PATTERNS:
            compiled = re.compile(pattern.pattern.format(name_escaped), re.IGNORECASE)
            object_score += len(compiled.findall(all_text))

    for pattern in _MODIFIER_PATTERNS:
        compiled = re.compile(pattern.pattern.format(name_escaped), re.IGNORECASE)
        modifier_score += len(compiled.findall(all_text))

    # Type-name heuristics
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("agent", "service", "handler", "controller", "manager", "orchestrator", "protocol")):
        subject_score += 2
    if any(kw in name_lower for kw in ("state", "data", "record", "message", "event", "vector", "corpus")):
        object_score += 2
    if any(kw in name_lower for kw in ("count", "flag", "level", "score", "status", "mode")):
        modifier_score += 2

    if subject_score > object_score and subject_score > modifier_score:
        return "subject"
    elif object_score > modifier_score:
        return "object"
    elif modifier_score > 0:
        return "modifier"
    return "unknown"


# =============================================================================
# SEMANTIC CENTRALITY
# =============================================================================

def compute_semantic_centrality(
    name: str,
    relationships: List[Dict[str, Any]],
    total_components: int,
) -> float:
    """
    Compute centrality of a component based on relationship density.

    A component referenced in many relationships is more likely to be
    architecturally significant than one with zero or one connection.

    Returns:
        Centrality score in [0, 1]
    """
    if total_components <= 1:
        return 0.5

    name_lower = name.lower()
    count = 0
    for rel in relationships:
        from_c = rel.get("from", "").lower()
        to_c = rel.get("to", "").lower()
        if name_lower in from_c or name_lower in to_c:
            count += 1

    # Normalize: 0 relationships = 0, max reasonable = total_components
    max_possible = max(total_components, 1)
    return min(count / max_possible, 1.0)


# =============================================================================
# TYPE INFERENCE
# =============================================================================

_TYPE_KEYWORDS = {
    "agent": {"agent", "handler", "worker", "processor", "service", "manager", "orchestrator"},
    "entity": {"state", "data", "record", "model", "store", "repository", "corpus", "vector", "oracle"},
    "process": {"protocol", "pipeline", "flow", "workflow", "algorithm", "compiler", "engine"},
    "interface": {"api", "interface", "contract", "boundary", "endpoint", "gateway"},
    "event": {"event", "signal", "trigger", "notification", "message", "callback"},
    "subsystem": {"subsystem", "module", "system", "layer", "tier"},
}


def infer_component_type(
    name: str,
    grammatical_role: str,
    llm_assigned_type: str = "",
    type_keywords: Optional[Dict[str, FrozenSet[str]]] = None,
) -> Tuple[str, float]:
    """
    Infer component type from name keywords and grammatical role.

    Args:
        name: Component name
        grammatical_role: Result from detect_grammatical_role
        llm_assigned_type: Type assigned by LLM (used as tiebreaker)
        type_keywords: Optional domain-specific type keywords (default: software)

    Returns:
        (inferred_type, confidence) where confidence in [0, 1]
    """
    kw_map = type_keywords if type_keywords is not None else _TYPE_KEYWORDS
    name_lower = name.lower()
    scores: Dict[str, float] = {}

    # Keyword matching
    for comp_type, keywords in kw_map.items():
        for kw in keywords:
            if kw in name_lower:
                scores[comp_type] = scores.get(comp_type, 0) + 1.0

    # Grammatical role hints
    if grammatical_role == "subject":
        scores["agent"] = scores.get("agent", 0) + 0.5
        scores["process"] = scores.get("process", 0) + 0.3
    elif grammatical_role == "object":
        scores["entity"] = scores.get("entity", 0) + 0.5
    elif grammatical_role == "modifier":
        # Modifiers are unlikely to be real components
        pass

    # LLM type as tiebreaker (0.3 weight)
    if llm_assigned_type:
        normalized = llm_assigned_type.lower()
        if normalized in kw_map:
            scores[normalized] = scores.get(normalized, 0) + 0.3

    if not scores:
        # No signals — default to entity with low confidence
        return ("entity", 0.3)

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    total = sum(scores.values())

    # Confidence: proportion of signal in the winning type
    confidence = best_score / total if total > 0 else 0.3

    return (best_type, min(confidence, 1.0))


# =============================================================================
# COMPONENT VS ATTRIBUTE DETECTION
# =============================================================================

_GENERIC = frozenset({
    "data", "input", "output", "result", "value", "type", "name",
    "config", "settings", "options", "params", "args", "info",
})


def is_likely_component(
    name: str,
    mention_freq: float,
    grammatical_role: str,
    centrality: float,
    generic_terms: Optional[FrozenSet[str]] = None,
    has_relationships: bool = True,
) -> Tuple[bool, str]:
    """
    Determine if a candidate is a real component or an attribute/state.

    Uses multiple signals: frequency, role, centrality, name patterns.

    Args:
        name: Component name
        mention_freq: Normalized mention frequency
        grammatical_role: Grammatical role string
        centrality: Semantic centrality score
        generic_terms: Optional domain-specific generic terms (default: software)
        has_relationships: Whether relationship data exists (False at DECOMPOSE stage)

    Returns:
        (is_component, reasoning)
    """
    # Hard reject: known attribute/state patterns
    name_lower = name.lower().replace(" ", "_")

    # Single-word generic terms
    terms = generic_terms if generic_terms is not None else _GENERIC
    if name_lower in terms:
        return (False, f"'{name}' is a generic term, not a specific component")

    # Too short (< 3 chars)
    if len(name.replace(" ", "")) < 3:
        return (False, f"'{name}' too short to be a component")

    # ALL_CAPS with underscores = enum/state value
    if name.isupper() and "_" in name:
        return (False, f"'{name}' looks like a state/enum value")

    # Modifier role + low frequency + low centrality = attribute
    # Only use centrality signal when relationship data exists
    if grammatical_role == "modifier" and mention_freq < 0.3 and (centrality < 0.2 if has_relationships else False):
        return (False, f"'{name}' is a modifier with low frequency and centrality")

    # Zero mentions in input + zero centrality = likely noise
    # Skip when relationships are structurally absent (e.g. DECOMPOSE stage)
    if mention_freq == 0.0 and centrality == 0.0 and has_relationships:
        return (False, f"'{name}' has no mentions and no relationships")

    return (True, f"'{name}' accepted: freq={mention_freq:.1f}, role={grammatical_role}, centrality={centrality:.1f}")


# =============================================================================
# MAIN CLASSIFIER
# =============================================================================

def classify_components(
    candidates: List[Dict[str, Any]],
    input_text: str,
    dialogue_history: List[str],
    relationships: List[Dict[str, Any]],
    type_keywords: Optional[Dict[str, FrozenSet[str]]] = None,
    generic_terms: Optional[FrozenSet[str]] = None,
    subject_patterns: Optional[Tuple] = None,
    object_patterns: Optional[Tuple] = None,
) -> List[ClassificationScore]:
    """
    Classify a list of component candidates.

    Main entry point for Phase 15. Takes raw candidates from DECOMPOSE
    and returns scored classifications.

    Args:
        candidates: List of {"name": str, "type": str, "derived_from": str}
        input_text: Original user input
        dialogue_history: List of dialogue message strings
        relationships: Current relationships from blueprint/pipeline
        type_keywords: Optional domain-specific type keywords (default: software)
        generic_terms: Optional domain-specific generic terms (default: software)
        subject_patterns: Optional domain-specific subject regex strings
        object_patterns: Optional domain-specific object regex strings

    Returns:
        List of ClassificationScore, one per candidate, sorted by confidence desc
    """
    total_components = len(candidates)
    results = []

    for candidate in candidates:
        name = candidate.get("name", "")
        llm_type = candidate.get("type", "")

        if not name:
            continue

        # 1. Mention frequency
        freq = compute_mention_frequency(name, input_text, dialogue_history)

        # 2. Grammatical role
        role = detect_grammatical_role(
            name, input_text, dialogue_history,
            subject_patterns=subject_patterns,
            object_patterns=object_patterns,
        )

        # 3. Semantic centrality
        centrality = compute_semantic_centrality(name, relationships, total_components)

        # 4. Type inference
        inferred_type, type_confidence = infer_component_type(name, role, llm_type, type_keywords)

        # 5. Component vs attribute
        is_comp, reasoning = is_likely_component(
            name, freq, role, centrality, generic_terms,
            has_relationships=len(relationships) > 0,
        )

        # 6. Overall confidence
        # Weighted: freq 0.3, centrality 0.3, type_conf 0.2, role 0.2
        role_score = {"subject": 0.9, "object": 0.7, "modifier": 0.3, "unknown": 0.5}.get(role, 0.5)
        overall = (
            freq * 0.3
            + centrality * 0.3
            + type_confidence * 0.2
            + role_score * 0.2
        )

        if not is_comp:
            overall *= 0.3  # Heavy penalty for non-components

        results.append(ClassificationScore(
            name=name,
            mention_frequency=freq,
            grammatical_role=role,
            semantic_centrality=centrality,
            inferred_type=inferred_type,
            type_confidence=type_confidence,
            is_component=is_comp,
            overall_confidence=min(overall, 1.0),
            reasoning=reasoning,
        ))

    # Sort by confidence descending
    results.sort(key=lambda s: s.overall_confidence, reverse=True)
    return results


def filter_by_confidence(
    scores: List[ClassificationScore],
    threshold: float = 0.3,
) -> Tuple[List[ClassificationScore], List[ClassificationScore]]:
    """
    Split classifications into accepted and rejected based on confidence threshold.

    Args:
        scores: Classification results
        threshold: Minimum confidence to accept (default 0.3)

    Returns:
        (accepted, rejected) — both sorted by confidence desc
    """
    accepted = [s for s in scores if s.is_component and s.overall_confidence >= threshold]
    rejected = [s for s in scores if not s.is_component or s.overall_confidence < threshold]
    return accepted, rejected


def needs_llm_fallback(score: ClassificationScore) -> bool:
    """
    Determine if a classification needs LLM review.

    Returns True if confidence is in the uncertain zone (0.3-0.6).
    Below 0.3 = confident rejection. Above 0.6 = confident acceptance.
    """
    return score.is_component and 0.3 <= score.overall_confidence < 0.6
