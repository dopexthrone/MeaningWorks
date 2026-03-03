"""
Transparent compilation output formatting.

LEAF module. Stdlib only. No imports from core/ or mother/.

Takes raw verification dicts, component lists, timings — returns formatted strings.
Separates text-channel from voice-channel output.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("mother.compile_render")


# The 7 verification dimensions in display order
DIMENSION_ORDER = [
    "completeness",
    "consistency",
    "coherence",
    "traceability",
    "actionability",
    "specificity",
    "codegen_readiness",
]

# Longest dimension name for alignment
_MAX_DIM_LEN = max(len(d) for d in ["overall"] + DIMENSION_ORDER)


@dataclass(frozen=True)
class CompileDisplay:
    """Formatted compilation output, split by channel."""

    dimension_block: str  # 7-line dimension breakdown with bars
    component_summary: str  # compact component list
    timing_line: str  # per-stage durations
    gap_lines: List[str]  # gaps for weak dimensions only
    voice_verdict: str  # one evaluative sentence (Mother's take)
    voice_notable: List[str]  # retries, slow stages (voice-worthy events)
    personality_voice_only: bool  # True when voice is on


def format_dimension_bar(name: str, score: float, bar_width: int = 20) -> str:
    """Format a single dimension score bar.

    Returns: '  completeness       [========    ] 67%'
    """
    filled = int((score / 100.0) * bar_width)
    filled = max(0, min(bar_width, filled))
    bar = f"[{'=' * filled}{' ' * (bar_width - filled)}]"
    padded_name = name.ljust(_MAX_DIM_LEN)
    return f"  {padded_name} {bar} {score:.0f}%"


def _extract_dimension_scores(verification: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric scores from verification dict.

    Handles both flat (score: 72) and nested ({"score": 72, "gaps": [...]}) formats.
    """
    scores: Dict[str, float] = {}
    for dim in DIMENSION_ORDER:
        v = verification.get(dim)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            scores[dim] = float(v)
        elif isinstance(v, dict) and "score" in v:
            scores[dim] = float(v["score"])
    return scores


def format_dimension_breakdown(verification: Dict[str, Any]) -> str:
    """Format all 7 dimensions as aligned score bars.

    Returns multi-line string, one line per dimension.
    """
    scores = _extract_dimension_scores(verification)
    if not scores:
        return ""
    lines = []
    for dim in DIMENSION_ORDER:
        if dim in scores:
            lines.append(format_dimension_bar(dim, scores[dim]))
    return "\n".join(lines)


def format_gaps(verification: Dict[str, Any], threshold: float = 70.0) -> List[str]:
    """Extract gap details for dimensions scoring below threshold.

    Returns list of strings like:
      'actionability 41%: missing method signatures; no typed attributes'
    """
    scores = _extract_dimension_scores(verification)
    lines: List[str] = []
    for dim in DIMENSION_ORDER:
        score = scores.get(dim)
        if score is None or score >= threshold:
            continue
        v = verification.get(dim)
        gap_texts: List[str] = []
        if isinstance(v, dict):
            for key in ("gaps", "conflicts", "issues"):
                raw = v.get(key, [])
                if isinstance(raw, list):
                    gap_texts.extend(str(g) for g in raw if g)
        if gap_texts:
            joined = "; ".join(gap_texts[:3])  # cap at 3 gaps per dim
            lines.append(f"  {dim} {score:.0f}%: {joined}")
        else:
            lines.append(f"  {dim} {score:.0f}%")
    return lines


def format_component_summary(
    components: List[Any], max_show: int = 8
) -> str:
    """Compact component list: 'AuthService (service), UserModel (entity) +12 more'"""
    if not components:
        return ""
    parts: List[str] = []
    for comp in components[:max_show]:
        if isinstance(comp, dict):
            name = comp.get("name", "?")
            ctype = comp.get("type", "")
            if ctype:
                parts.append(f"{name} ({ctype})")
            else:
                parts.append(name)
        else:
            parts.append(str(comp))
    summary = ", ".join(parts)
    remaining = len(components) - max_show
    if remaining > 0:
        summary += f" +{remaining} more"
    return f"Components: {summary}"


def format_timing_line(stage_timings: Dict[str, float]) -> str:
    """Format stage durations as a single line.

    Returns: 'intent 2.1s | synth 3.2s | verify 1.1s | total 16.2s'
    """
    if not stage_timings:
        return ""
    parts: List[str] = []
    # Short aliases for stage names
    aliases = {
        "intent": "intent",
        "personas": "persona",
        "dialogue": "dialogue",
        "synthesis": "synth",
        "verification": "verify",
        "resynthesis": "resynth",
    }
    for stage, duration in stage_timings.items():
        label = aliases.get(stage, stage)
        parts.append(f"{label} {duration:.1f}s")
    total = sum(stage_timings.values())
    parts.append(f"total {total:.1f}s")
    return " | ".join(parts)


def format_voice_verdict(
    overall_score: float,
    verification: Dict[str, Any],
    component_count: int,
    retry_counts: Dict[str, int],
) -> str:
    """One evaluative sentence for voice channel. Mother's take, not data."""
    scores = _extract_dimension_scores(verification)

    # Find weakest
    weakest_dim = ""
    weakest_score = 101.0
    for dim, score in scores.items():
        if score < weakest_score:
            weakest_dim = dim
            weakest_score = score

    count_str = f"{component_count} components."

    if overall_score >= 80:
        return f"{count_str} This one's solid."
    elif overall_score >= 60:
        if weakest_dim:
            return f"{count_str} {weakest_dim.replace('_', ' ')} is the weak spot at {weakest_score:.0f}%."
        return f"{count_str} Decent but has gaps."
    elif overall_score >= 40:
        if weakest_dim:
            return f"{count_str} {weakest_dim.replace('_', ' ')} is the weak spot at {weakest_score:.0f}%."
        return f"{count_str} Needs work."
    else:
        return f"{count_str} This needs significant refinement."


def collect_notable_events(
    retry_counts: Dict[str, int],
    stage_timings: Dict[str, float],
    slow_threshold: float = 10.0,
) -> List[str]:
    """Collect voice-worthy events: retries, unusually slow stages."""
    events: List[str] = []
    total_retries = sum(retry_counts.values())
    if total_retries > 0:
        stages = ", ".join(f"{k} ({v})" for k, v in retry_counts.items() if v > 0)
        events.append(f"Retried {total_retries} times: {stages}.")
    for stage, duration in stage_timings.items():
        if duration >= slow_threshold:
            events.append(f"{stage} took {duration:.1f}s.")
    return events


def render_compile_output(
    verification: Dict[str, Any],
    components: List[Any],
    stage_timings: Dict[str, float],
    retry_counts: Dict[str, int],
    voice_enabled: bool = False,
) -> CompileDisplay:
    """Build the full CompileDisplay from raw compilation data."""
    logger.info("Rendering compile output", extra={"component_count": len(components), "voice_enabled": voice_enabled})
    overall = 0.0
    if isinstance(verification, dict):
        overall = verification.get("overall_score", 0.0)

    component_count = len(components) if isinstance(components, list) else 0

    return CompileDisplay(
        dimension_block=format_dimension_breakdown(verification),
        component_summary=format_component_summary(components),
        timing_line=format_timing_line(stage_timings),
        gap_lines=format_gaps(verification),
        voice_verdict=format_voice_verdict(
            overall, verification, component_count, retry_counts
        ),
        voice_notable=collect_notable_events(retry_counts, stage_timings),
        personality_voice_only=voice_enabled,
    )
