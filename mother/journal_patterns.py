"""
Journal pattern compiler — L2 from operational history.

LEAF module. Stdlib + json only. No imports from core/ or mother/.

Compiles journal entry dicts into dimension-level patterns:
averages, trajectories, chronic weaknesses, failure co-occurrences,
domain weaknesses. Returns frozen dataclass with pre-formatted
context strings for injection into Mother's situational awareness.

Usage:
    entries = [asdict(e) for e in journal.recent(limit=20)]
    patterns = extract_patterns(entries)
    # patterns.trends_line -> "Weak: traceability (42%). Declining: specificity (-12%)."
    # patterns.failure_line -> "Low-trust pattern: weak actionability + specificity (4/6 failures)."
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# The 7 verification dimensions (canonical names)
_DIMENSIONS = (
    "completeness",
    "consistency",
    "specificity",
    "actionability",
    "traceability",
    "modularity",
    "testability",
)

_WEAK_THRESHOLD = 55
_FAILING_THRESHOLD = 50
_TRAJECTORY_THRESHOLD = 5
_LOW_TRUST_THRESHOLD = 55
_DOMAIN_MIN_ENTRIES = 3

# Formatting caps
_MAX_WEAK = 3
_MAX_IMPROVING = 2
_MAX_DECLINING = 2
_MAX_FAILURE_PATTERNS = 1


@dataclass(frozen=True)
class JournalPatterns:
    """Compiled patterns from journal entries. Frozen, pure data."""

    dimension_averages: Dict[str, float] = field(default_factory=dict)
    dimension_trajectories: Dict[str, float] = field(default_factory=dict)
    chronic_weak: List[str] = field(default_factory=list)
    failure_co_occurrences: List[Tuple[str, str, int, int]] = field(
        default_factory=list
    )  # (dim_a, dim_b, co_count, total_failures)
    domain_weaknesses: Dict[str, List[str]] = field(
        default_factory=dict
    )  # domain -> weakest dims
    trends_line: str = ""
    failure_line: str = ""


def _parse_dimension_scores(raw: str) -> Dict[str, float]:
    """Parse dimension_scores JSON string. Returns empty dict on any error."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


def _compute_averages(
    entries_with_scores: List[Dict[str, float]],
) -> Dict[str, float]:
    """Per-dimension average scores."""
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for scores in entries_with_scores:
        for dim, val in scores.items():
            totals[dim] += val
            counts[dim] += 1
    return {dim: round(totals[dim] / counts[dim], 1) for dim in totals if counts[dim] > 0}


def _compute_trajectories(
    entries_with_scores: List[Dict[str, float]],
) -> Dict[str, float]:
    """Per-dimension trajectory: second-half avg minus first-half avg.

    Entries should be in chronological order (oldest first).
    Returns only dimensions with |delta| > threshold.
    """
    n = len(entries_with_scores)
    if n < 3:
        return {}

    mid = n // 2
    first_half = entries_with_scores[:mid]
    second_half = entries_with_scores[mid:]

    first_avgs = _compute_averages(first_half)
    second_avgs = _compute_averages(second_half)

    trajectories: Dict[str, float] = {}
    all_dims = set(first_avgs) | set(second_avgs)
    for dim in all_dims:
        if dim in first_avgs and dim in second_avgs:
            delta = second_avgs[dim] - first_avgs[dim]
            if abs(delta) > _TRAJECTORY_THRESHOLD:
                trajectories[dim] = round(delta, 1)
    return trajectories


def _find_failure_co_occurrences(
    entries: List[dict],
    entries_with_scores: List[Dict[str, float]],
) -> List[Tuple[str, str, int, int]]:
    """Find dimension pairs that co-occur below threshold in failed/low-trust entries."""
    # Build list of (scores, entry) for failures
    failure_scores: List[Dict[str, float]] = []
    for entry, scores in zip(entries, entries_with_scores):
        if not scores:
            continue
        is_failure = (
            not entry.get("success", True)
            or (entry.get("trust_score", 100) > 0 and entry.get("trust_score", 100) < _LOW_TRUST_THRESHOLD)
        )
        if is_failure:
            failure_scores.append(scores)

    total_failures = len(failure_scores)
    if total_failures < 2:
        return []

    # Count co-occurrences of weak dimension pairs
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    dims = sorted(set().union(*(s.keys() for s in failure_scores)))

    for scores in failure_scores:
        weak_dims = [d for d in dims if scores.get(d, 100) < _FAILING_THRESHOLD]
        for i in range(len(weak_dims)):
            for j in range(i + 1, len(weak_dims)):
                pair = (weak_dims[i], weak_dims[j])
                pair_counts[pair] += 1

    # Filter: pair must appear in at least 2 failures
    results = [
        (a, b, count, total_failures)
        for (a, b), count in sorted(pair_counts.items(), key=lambda x: -x[1])
        if count >= 2
    ]
    return results


def _find_domain_weaknesses(
    entries: List[dict],
    entries_with_scores: List[Dict[str, float]],
) -> Dict[str, List[str]]:
    """Per-domain weakest dimensions (avg < threshold, min 3 entries per domain)."""
    domain_scores: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for entry, scores in zip(entries, entries_with_scores):
        if not scores:
            continue
        domain = entry.get("domain", "")
        if domain:
            domain_scores[domain].append(scores)

    result: Dict[str, List[str]] = {}
    for domain, score_list in domain_scores.items():
        if len(score_list) < _DOMAIN_MIN_ENTRIES:
            continue
        avgs = _compute_averages(score_list)
        weak = sorted(
            [(d, v) for d, v in avgs.items() if v < _WEAK_THRESHOLD],
            key=lambda x: x[1],
        )
        if weak:
            result[domain] = [d for d, _ in weak]
    return result


def _format_trends_line(
    averages: Dict[str, float],
    trajectories: Dict[str, float],
    chronic_weak: List[str],
) -> str:
    """Format the trends context line. Empty string when nothing to report."""
    parts: List[str] = []

    # Weak dimensions
    if chronic_weak:
        weak_strs = [
            f"{d} ({averages.get(d, 0):.0f}%)" for d in chronic_weak[:_MAX_WEAK]
        ]
        parts.append(f"Weak: {', '.join(weak_strs)}")

    # Improving
    improving = sorted(
        [(d, v) for d, v in trajectories.items() if v > 0],
        key=lambda x: -x[1],
    )[:_MAX_IMPROVING]
    if improving:
        imp_strs = [f"{d} (+{v:.0f}%)" for d, v in improving]
        parts.append(f"Improving: {', '.join(imp_strs)}")

    # Declining
    declining = sorted(
        [(d, v) for d, v in trajectories.items() if v < 0],
        key=lambda x: x[1],
    )[:_MAX_DECLINING]
    if declining:
        dec_strs = [f"{d} ({v:.0f}%)" for d, v in declining]
        parts.append(f"Declining: {', '.join(dec_strs)}")

    if not parts:
        return ""
    return ". ".join(parts) + "."


def _format_failure_line(
    co_occurrences: List[Tuple[str, str, int, int]],
) -> str:
    """Format failure co-occurrence line. Empty string when nothing to report."""
    if not co_occurrences:
        return ""

    # Take the top pattern only
    a, b, count, total = co_occurrences[0]
    return f"Low-trust pattern: weak {a} + {b} ({count}/{total} failures)."


def estimate_success_probability(
    historical_success_rate: float,
    attempt_count: int,
    avg_trust_score: float = 50.0,
) -> float:
    """Estimate probability of next compile succeeding. Pure function.
    Combines historical rate with attempt decay and trust signal.
    Returns 0.0-1.0.
    """
    if attempt_count <= 0:
        # No history — use historical rate directly
        base = max(0.0, min(1.0, historical_success_rate))
    else:
        # Decay with repeated failures: each attempt without success drops confidence
        base = max(0.0, min(1.0, historical_success_rate))
        decay = 0.85 ** attempt_count  # exponential decay
        base *= decay
    # Trust signal: high average trust lifts probability
    trust_factor = avg_trust_score / 100.0
    adjusted = base * 0.7 + trust_factor * 0.3
    return max(0.05, min(0.95, round(adjusted, 3)))


def compute_unit_economics(
    total_cost: float,
    total_compiles: int,
    successful_compiles: int,
    total_components: int,
) -> dict:
    """Compute unit economics from journal aggregates. Pure function.
    Returns dict with cost_per_compile, cost_per_component, cost_per_success, waste_ratio.
    """
    result = {
        "cost_per_compile": 0.0,
        "cost_per_component": 0.0,
        "cost_per_success": 0.0,
        "waste_ratio": 0.0,
    }
    if total_compiles > 0:
        result["cost_per_compile"] = round(total_cost / total_compiles, 4)
    if total_components > 0:
        result["cost_per_component"] = round(total_cost / total_components, 4)
    if successful_compiles > 0:
        result["cost_per_success"] = round(total_cost / successful_compiles, 4)
    if total_compiles > 0:
        failed = total_compiles - successful_compiles
        result["waste_ratio"] = round(failed / total_compiles, 3)
    return result


def detect_opportunities(patterns: JournalPatterns) -> list:
    """Detect improvement opportunities from trajectory data. Pure function."""
    if not patterns.dimension_trajectories:
        return []
    improving = sorted(
        [(d, v) for d, v in patterns.dimension_trajectories.items() if v > 0],
        key=lambda x: -x[1],
    )
    opportunities = []
    for dim, delta in improving[:2]:
        avg = patterns.dimension_averages.get(dim, 0)
        opportunities.append(
            f"{dim} improving (+{delta:.0f}% trend, now {avg:.0f}%) "
            f"— consider formalizing this strength"
        )
    return opportunities


_COUNTERFACTUAL_STRATEGIES = {
    "specificity": "What if the spec were more concrete? Try adding exact field names, types, and constraints.",
    "modularity": "What if this were split into smaller independent pieces? Try decomposing before compiling.",
    "actionability": "What if each component had explicit build instructions? Try adding implementation hints.",
    "completeness": "What if you addressed the missing pieces first? Try listing what's undefined.",
    "consistency": "What if you resolved the contradictions? Try aligning conflicting requirements.",
    "traceability": "What if every output linked back to a specific input? Try adding provenance markers.",
    "testability": "What if this were designed for testing first? Try specifying acceptance criteria upfront.",
}


def generate_counterfactual(weakest_dimension: str, attempt_count: int) -> str:
    """Generate a 'what if' alternative based on weakest dimension. Pure function."""
    if attempt_count < 2 or not weakest_dimension:
        return ""
    return _COUNTERFACTUAL_STRATEGIES.get(
        weakest_dimension.lower(),
        f"What if you approached {weakest_dimension} differently?",
    )


_SKILL_GAP_HINTS = {
    "specificity": "Specs aren't specific enough — try including exact field names, types, and constraints before compiling.",
    "modularity": "Designs aren't decomposed well — try breaking into smaller, independent pieces.",
    "actionability": "Output isn't actionable — try adding explicit build instructions to each component.",
    "completeness": "Specs have gaps — try listing every required feature before starting.",
    "consistency": "Requirements contradict each other — try resolving conflicts before compiling.",
    "traceability": "Output doesn't trace back to input — try tagging each component with its source requirement.",
    "testability": "Output isn't testable — try defining acceptance criteria upfront.",
}


def detect_skill_gap(chronic_weak: list, domain_weaknesses: dict) -> str:
    """Detect the most impactful skill gap from journal patterns. Pure function.
    Returns a human-readable suggestion or empty string.
    """
    if not chronic_weak:
        return ""
    weakest = chronic_weak[0]
    hint = _SKILL_GAP_HINTS.get(weakest, "")
    if not hint:
        return ""
    # Domain-specific amplification
    for domain, dims in domain_weaknesses.items():
        if weakest in dims:
            hint += f" (Especially in {domain} projects.)"
            break
    return hint


def challenge_assumptions(goal_description: str, attempt_count: int) -> str:
    """Generate assumption challenge for a struggling goal. Pure function.
    Inverts the most likely implicit assumption in the goal description.
    """
    if attempt_count < 2 or not goal_description:
        return ""
    text = goal_description.lower()
    # Detect implicit assumptions and invert them
    challenges = []
    if "single" in text or "one" in text:
        challenges.append("What if this needs to handle multiple, not just one?")
    if "simple" in text or "basic" in text:
        challenges.append("What if 'simple' is masking hidden complexity?")
    if "fast" in text or "quick" in text:
        challenges.append("What if speed isn't the right priority here?")
    if "user" in text and "users" not in text:
        challenges.append("What if there are multiple user types with different needs?")
    if "api" in text:
        challenges.append("What if the API contract is wrong or incomplete?")
    if "database" in text or "db" in text:
        challenges.append("What if the data model needs restructuring first?")
    if not challenges:
        challenges.append(f"What assumptions are baked into '{goal_description[:60]}'? Try stating them explicitly.")
    return challenges[0]


def extract_patterns(entries: List[dict]) -> JournalPatterns:
    """Extract dimension-level patterns from journal entry dicts.

    Args:
        entries: List of dicts with keys: success, trust_score, domain,
                 dimension_scores (JSON string), timestamp.
                 Newest-first order (as returned by journal.recent()).

    Returns:
        JournalPatterns with computed fields and formatted context lines.
    """
    if not entries:
        return JournalPatterns()

    # Reverse to chronological order (oldest first) for trajectory computation
    chrono = list(reversed(entries))

    # Parse dimension scores for each entry
    scores_list = [_parse_dimension_scores(e.get("dimension_scores", "")) for e in chrono]

    # Filter to entries that actually have dimension scores
    has_scores = [(e, s) for e, s in zip(chrono, scores_list) if s]
    if not has_scores:
        return JournalPatterns()

    entries_with_data = [e for e, _ in has_scores]
    scores_with_data = [s for _, s in has_scores]

    # Compute all patterns
    averages = _compute_averages(scores_with_data)
    trajectories = _compute_trajectories(scores_with_data)

    chronic_weak = sorted(
        [d for d, v in averages.items() if v < _WEAK_THRESHOLD],
        key=lambda d: averages[d],
    )

    co_occurrences = _find_failure_co_occurrences(entries_with_data, scores_with_data)
    domain_weak = _find_domain_weaknesses(entries_with_data, scores_with_data)

    # Format context lines
    trends_line = _format_trends_line(averages, trajectories, chronic_weak)
    failure_line = _format_failure_line(co_occurrences[:_MAX_FAILURE_PATTERNS])

    return JournalPatterns(
        dimension_averages=averages,
        dimension_trajectories=trajectories,
        chronic_weak=chronic_weak,
        failure_co_occurrences=co_occurrences,
        domain_weaknesses=domain_weak,
        trends_line=trends_line,
        failure_line=failure_line,
    )


# --- Reframe-capable (#103) ---

_REFRAME_TRIGGERS = {
    "build": "What if the problem isn't building this, but defining what it actually needs to do?",
    "create": "What if the problem isn't building this, but defining what it actually needs to do?",
    "fix": "What if this isn't broken — the system around it changed?",
    "repair": "What if this isn't broken — the system around it changed?",
    "improve": "What if this needs replacement, not improvement?",
    "optimize": "What if this needs replacement, not improvement?",
    "scale": "What if scaling is the wrong goal — what if simplification would solve the real problem?",
    "migrate": "What if the destination is wrong? Validate the target before moving.",
    "integrate": "What if these shouldn't be integrated? What if separation is the better design?",
    "automate": "What if manual is better here? Not everything benefits from automation.",
}


def generate_reframe(goal_description: str, chronic_weak: list, attempt_count: int) -> str:
    """Reframe the problem itself when stuck. Pure function.
    Unlike challenge_assumptions (questions hidden assumptions) and
    generate_counterfactual (dimension-specific strategies), this
    offers structural reframing of the problem.
    """
    if attempt_count < 2 or not goal_description:
        return ""
    words = goal_description.lower().split()
    for word in words:
        if word in _REFRAME_TRIGGERS:
            return _REFRAME_TRIGGERS[word]
    # Dimension-informed fallback
    if chronic_weak:
        dim = chronic_weak[0]
        return (
            f"Reframe: the recurring weakness in '{dim}' suggests the problem "
            f"structure itself may need rethinking, not just the solution."
        )
    return "What if you're solving the wrong problem? Restate what success looks like without mentioning the current approach."


# --- Serendipity-engineering (#106) ---

_SERENDIPITY_STOPWORDS = frozenset({
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "is",
    "it", "that", "this", "with", "from", "by", "at", "be", "as", "are",
    "was", "were", "been", "do", "does", "did", "has", "have", "had",
    "will", "would", "could", "should", "may", "might", "can", "test",
    "build", "create", "make", "use", "get", "set", "new", "add",
})


def find_cross_topic_connections(topics: list, recent_subjects: list) -> str:
    """Find unexpected connections between current topics and recurring subjects.
    Pure function. Returns a connection insight or empty string.
    """
    if not topics or not recent_subjects:
        return ""
    topic_words = set()
    for t in topics:
        topic_words |= set(t.lower().split()) - _SERENDIPITY_STOPWORDS
    subject_words = {}
    for s in recent_subjects:
        words = set(s.lower().split()) - _SERENDIPITY_STOPWORDS
        subject_words[s] = words
    best_subject = ""
    best_overlap = set()
    for s, sw in subject_words.items():
        overlap = topic_words & sw
        if len(overlap) > len(best_overlap):
            best_overlap = overlap
            best_subject = s
    if not best_overlap:
        return ""
    shared = ", ".join(sorted(best_overlap)[:3])
    return (
        f"Unexpected connection: your current work shares themes ({shared}) "
        f"with a recurring pattern: '{best_subject[:60]}'. Worth exploring?"
    )


# --- Teaching-mode (#154) ---


def generate_teaching_summary(compile_result: dict, learning_context: dict) -> str:
    """Generate a 'here's what happened' teaching narrative. Pure function."""
    if not compile_result:
        return ""
    parts = []
    trust = compile_result.get("trust_score", 0)
    count = compile_result.get("component_count", 0)
    steps = compile_result.get("step_names", [])
    dim_scores = compile_result.get("dimension_scores", {})
    if count > 0:
        parts.append(f"Decomposed into {count} component{'s' if count != 1 else ''}")
    if trust > 0:
        parts.append(f"trust score {trust:.0f}%")
    if dim_scores:
        weakest = min(dim_scores, key=dim_scores.get)
        strongest = max(dim_scores, key=dim_scores.get)
        if weakest != strongest:
            parts.append(f"strongest in {strongest}, weakest in {weakest}")
    chronic = learning_context.get("chronic_weak", [])
    if chronic:
        parts.append(f"recurring weakness: {chronic[0]}")
    if steps:
        parts.append(f"steps: {', '.join(steps[:5])}")
    if not parts:
        return ""
    return "What happened: " + ". ".join(parts) + "."


# --- Skill-transfer-capable (#156) ---


def extract_methodology(patterns_dict: dict, goal_count: int, compile_count: int) -> str:
    """Summarize Mother's working methodology. Pure function.
    Needs >= 3 compiles to have meaningful data.
    """
    if compile_count < 3:
        return ""
    parts = []
    chronic = patterns_dict.get("chronic_weak", [])
    trajectories = patterns_dict.get("dimension_trajectories", {})
    averages = patterns_dict.get("dimension_averages", {})
    improving = [d for d, v in trajectories.items() if v > 0]
    declining = [d for d, v in trajectories.items() if v < 0]
    if improving:
        parts.append(f"Improving: {', '.join(improving[:3])}")
    if declining:
        parts.append(f"Declining: {', '.join(declining[:3])}")
    if chronic:
        parts.append(f"Persistent weakness: {', '.join(chronic[:2])}")
    if averages:
        top = sorted(averages, key=averages.get, reverse=True)[:2]
        top_strs = [f"{d} ({averages[d]:.0f}%)" for d in top]
        parts.append(f"Strengths: {', '.join(top_strs)}")
    parts.append(f"Over {compile_count} compiles, {goal_count} active goals")
    return ". ".join(parts) + "."
