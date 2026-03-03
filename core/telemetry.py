"""
Motherlabs Telemetry - In-memory metrics collection and aggregation.

Phase 19: Monitoring & Observability
Phase 21: Cost Optimization — TokenUsage, CostEstimate, PRICING_TABLE, estimate_cost()

LEAF MODULE — zero project imports, only stdlib.
All dataclasses frozen. All functions pure.
"""

import time
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Tuple, List


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class TokenUsage:
    """Token usage from a single LLM call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    provider: str
    model: str


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate from a single LLM call."""
    input_cost: float       # USD
    output_cost: float      # USD
    total_cost: float       # USD
    token_usage: TokenUsage


# Pricing per 1M tokens: (input_per_1M, output_per_1M) in USD
PRICING_TABLE: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4": (3.0, 15.0),
    "claude-opus-4": (15.0, 75.0),
    "claude-haiku": (0.25, 1.25),
    "gpt-5": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "grok-3": (3.0, 15.0),
    "grok-4": (2.0, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.0),
}


def estimate_cost(usage: TokenUsage) -> CostEstimate:
    """
    Estimate cost from token usage using prefix-match pricing.

    Args:
        usage: TokenUsage with provider and model.

    Returns:
        CostEstimate. Zero cost if model not in pricing table.
    """
    model = usage.model or ""
    input_rate = 0.0
    output_rate = 0.0

    # Prefix match: longest prefix wins
    best_prefix = ""
    for prefix, (inp, outp) in PRICING_TABLE.items():
        if model.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            input_rate = inp
            output_rate = outp

    input_cost = (usage.input_tokens / 1_000_000) * input_rate
    output_cost = (usage.output_tokens / 1_000_000) * output_rate

    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        token_usage=usage,
    )


@dataclass(frozen=True)
class CompilationMetrics:
    """Metrics captured from a single compilation."""
    compilation_id: str
    timestamp: float                                    # time.time()
    success: bool
    total_duration: float                               # seconds
    stage_timings: Tuple[Tuple[str, float], ...]        # (stage_name, seconds)
    dialogue_turns: int
    component_count: int
    insight_count: int
    verification_score: int                             # overall 0-100
    verification_mode: str                              # "deterministic" | "hybrid" | "llm_only"
    cache_hits: int
    cache_misses: int
    retry_count: int
    provider: str
    model: str
    # Phase 21: Cost tracking (defaults preserve backward compat)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregated metrics over a window of compilations."""
    window_size: int
    success_rate: float                                 # 0.0-1.0
    avg_duration: float                                 # seconds
    avg_components: float
    avg_insights: float
    avg_verification_score: float
    p50_duration: float                                 # median
    p95_duration: float                                 # 95th percentile
    verification_mode_counts: Tuple[Tuple[str, int], ...]
    provider_counts: Tuple[Tuple[str, int], ...]
    cache_hit_rate: float                               # 0.0-1.0
    total_retries: int
    # Phase 21: Cost aggregation (defaults preserve backward compat)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0
    avg_tokens_per_compilation: float = 0.0


@dataclass(frozen=True)
class HealthSnapshot:
    """Point-in-time health assessment."""
    status: str                                         # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    compilations_total: int
    compilations_recent: int
    recent_success_rate: float
    recent_avg_duration: float
    cache_hit_rate: float
    corpus_size: int
    last_compilation_age: float                         # seconds since last compilation
    issues: Tuple[str, ...]
    # Phase 21: Cost in health
    recent_total_cost_usd: float = 0.0


# =============================================================================
# PURE FUNCTIONS
# =============================================================================


def percentile(values: Sequence[float], p: float) -> float:
    """
    Compute the p-th percentile of a sequence of values.

    Args:
        values: Sequence of numeric values (need not be sorted).
        p: Percentile in range 0-100.

    Returns:
        The interpolated percentile value. 0.0 for empty input.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Rank index (0-based)
    k = (p / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def aggregate_metrics(metrics: Sequence[CompilationMetrics]) -> AggregateMetrics:
    """
    Compute aggregate statistics over a window of compilation metrics.

    Args:
        metrics: Sequence of CompilationMetrics (e.g. ring buffer contents).

    Returns:
        AggregateMetrics with all computed fields. Zero-valued for empty input.
    """
    n = len(metrics)
    if n == 0:
        return AggregateMetrics(
            window_size=0,
            success_rate=0.0,
            avg_duration=0.0,
            avg_components=0.0,
            avg_insights=0.0,
            avg_verification_score=0.0,
            p50_duration=0.0,
            p95_duration=0.0,
            verification_mode_counts=(),
            provider_counts=(),
            cache_hit_rate=0.0,
            total_retries=0,
        )

    successes = sum(1 for m in metrics if m.success)
    durations = [m.total_duration for m in metrics]
    total_hits = sum(m.cache_hits for m in metrics)
    total_misses = sum(m.cache_misses for m in metrics)
    total_cache = total_hits + total_misses

    # Mode counts
    mode_counts: Dict[str, int] = {}
    for m in metrics:
        mode_counts[m.verification_mode] = mode_counts.get(m.verification_mode, 0) + 1

    provider_counts: Dict[str, int] = {}
    for m in metrics:
        provider_counts[m.provider] = provider_counts.get(m.provider, 0) + 1

    # Phase 21: Cost aggregation
    ti = sum(m.total_input_tokens for m in metrics)
    to = sum(m.total_output_tokens for m in metrics)
    tc = sum(m.estimated_cost_usd for m in metrics)

    return AggregateMetrics(
        window_size=n,
        success_rate=successes / n,
        avg_duration=sum(durations) / n,
        avg_components=sum(m.component_count for m in metrics) / n,
        avg_insights=sum(m.insight_count for m in metrics) / n,
        avg_verification_score=sum(m.verification_score for m in metrics) / n,
        p50_duration=percentile(durations, 50),
        p95_duration=percentile(durations, 95),
        verification_mode_counts=tuple(sorted(mode_counts.items())),
        provider_counts=tuple(sorted(provider_counts.items())),
        cache_hit_rate=total_hits / total_cache if total_cache > 0 else 0.0,
        total_retries=sum(m.retry_count for m in metrics),
        total_input_tokens=ti,
        total_output_tokens=to,
        total_cost_usd=tc,
        avg_cost_usd=tc / n,
        avg_tokens_per_compilation=(ti + to) / n,
    )


def compute_health(
    metrics: Sequence[CompilationMetrics],
    uptime: float,
    corpus_size: int,
    cost_warn_threshold: float = 3.0,
) -> HealthSnapshot:
    """
    Compute a point-in-time health assessment.

    Status rules:
        - "healthy": success_rate >= 0.8 AND avg_duration < 600s
        - "degraded": success_rate >= 0.5 OR avg_duration < 900s
        - "unhealthy": otherwise

    Args:
        metrics: Recent compilation metrics (ring buffer).
        uptime: Seconds since engine start.
        corpus_size: Total compilations in corpus.
        cost_warn_threshold: Per-compilation cost threshold for health warning (USD).

    Returns:
        HealthSnapshot with status and issues list.
    """
    agg = aggregate_metrics(metrics)
    n = len(metrics)

    issues: List[str] = []
    now = time.time()
    last_age = (now - metrics[-1].timestamp) if n > 0 else 0.0

    # Determine status
    if n == 0:
        status = "healthy"
        issues.append("No compilations recorded yet")
    else:
        success_ok = agg.success_rate >= 0.8
        duration_ok = agg.avg_duration < 600.0

        if success_ok and duration_ok:
            status = "healthy"
        elif agg.success_rate >= 0.5 or agg.avg_duration < 900.0:
            status = "degraded"
            if not success_ok:
                issues.append(f"Success rate below 80%: {agg.success_rate:.0%}")
            if not duration_ok:
                issues.append(f"Avg duration above 600s: {agg.avg_duration:.1f}s")
        else:
            status = "unhealthy"
            issues.append(f"Success rate: {agg.success_rate:.0%}")
            issues.append(f"Avg duration: {agg.avg_duration:.1f}s")

    # Phase 21: Cost health issue
    if n > 0 and agg.avg_cost_usd > cost_warn_threshold:
        issues.append(f"Avg cost above ${cost_warn_threshold:.2f}: ${agg.avg_cost_usd:.2f}")

    return HealthSnapshot(
        status=status,
        uptime_seconds=uptime,
        compilations_total=corpus_size,
        compilations_recent=n,
        recent_success_rate=agg.success_rate,
        recent_avg_duration=agg.avg_duration,
        cache_hit_rate=agg.cache_hit_rate,
        corpus_size=corpus_size,
        last_compilation_age=last_age,
        issues=tuple(issues),
        recent_total_cost_usd=agg.total_cost_usd,
    )


# =============================================================================
# SERIALIZATION
# =============================================================================


def token_usage_to_dict(usage: TokenUsage) -> Dict[str, Any]:
    """Serialize a TokenUsage to a plain dict."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "provider": usage.provider,
        "model": usage.model,
    }


def cost_estimate_to_dict(estimate: CostEstimate) -> Dict[str, Any]:
    """Serialize a CostEstimate to a plain dict."""
    return {
        "input_cost": estimate.input_cost,
        "output_cost": estimate.output_cost,
        "total_cost": estimate.total_cost,
        "token_usage": token_usage_to_dict(estimate.token_usage),
    }


def metrics_to_dict(metrics: CompilationMetrics) -> Dict[str, Any]:
    """Serialize a CompilationMetrics to a plain dict."""
    return {
        "compilation_id": metrics.compilation_id,
        "timestamp": metrics.timestamp,
        "success": metrics.success,
        "total_duration": metrics.total_duration,
        "stage_timings": {k: v for k, v in metrics.stage_timings},
        "dialogue_turns": metrics.dialogue_turns,
        "component_count": metrics.component_count,
        "insight_count": metrics.insight_count,
        "verification_score": metrics.verification_score,
        "verification_mode": metrics.verification_mode,
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,
        "retry_count": metrics.retry_count,
        "provider": metrics.provider,
        "model": metrics.model,
        "total_input_tokens": metrics.total_input_tokens,
        "total_output_tokens": metrics.total_output_tokens,
        "estimated_cost_usd": metrics.estimated_cost_usd,
    }


def aggregate_to_dict(agg: AggregateMetrics) -> Dict[str, Any]:
    """Serialize an AggregateMetrics to a plain dict."""
    return {
        "window_size": agg.window_size,
        "success_rate": agg.success_rate,
        "avg_duration": agg.avg_duration,
        "avg_components": agg.avg_components,
        "avg_insights": agg.avg_insights,
        "avg_verification_score": agg.avg_verification_score,
        "p50_duration": agg.p50_duration,
        "p95_duration": agg.p95_duration,
        "verification_mode_counts": {k: v for k, v in agg.verification_mode_counts},
        "provider_counts": {k: v for k, v in agg.provider_counts},
        "cache_hit_rate": agg.cache_hit_rate,
        "total_retries": agg.total_retries,
        "total_input_tokens": agg.total_input_tokens,
        "total_output_tokens": agg.total_output_tokens,
        "total_cost_usd": agg.total_cost_usd,
        "avg_cost_usd": agg.avg_cost_usd,
        "avg_tokens_per_compilation": agg.avg_tokens_per_compilation,
    }


def health_to_dict(health: HealthSnapshot) -> Dict[str, Any]:
    """Serialize a HealthSnapshot to a plain dict."""
    return {
        "status": health.status,
        "uptime_seconds": health.uptime_seconds,
        "compilations_total": health.compilations_total,
        "compilations_recent": health.compilations_recent,
        "recent_success_rate": health.recent_success_rate,
        "recent_avg_duration": health.recent_avg_duration,
        "cache_hit_rate": health.cache_hit_rate,
        "corpus_size": health.corpus_size,
        "last_compilation_age": health.last_compilation_age,
        "issues": list(health.issues),
        "recent_total_cost_usd": health.recent_total_cost_usd,
    }
