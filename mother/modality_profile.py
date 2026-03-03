"""
mother/modality_profile.py — Per-modality configuration profiles.

LEAF module. No imports from core/ or mother/. Stdlib only.

Encodes per-modality characteristics (reliability, cost, latency,
information density) enabling adaptive attention thresholds and
budget allocation. Pure functions, no side effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ModalityProfile",
    "ModalityBudget",
    "default_profiles",
    "should_process",
    "allocate_budget",
    "adjust_threshold",
    "format_modality_context",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalityProfile:
    """Configuration profile for a single perception modality."""
    name: str               # "screen" | "speech" | "camera"
    reliability: float      # 0.0-1.0 (screen=0.95, speech=0.7, camera=0.6)
    cost_per_event: float   # USD (screen=0.005, speech=0.001, camera=0.005)
    latency_ms: float       # typical processing time
    information_density: float  # relative bits-per-event (0.0-1.0)
    enabled: bool = True
    attention_threshold: float = 0.3  # minimum significance to attend


@dataclass(frozen=True)
class ModalityBudget:
    """Budget tracking for a single modality within an hour."""
    modality: str
    hourly_limit: float     # USD
    events_this_hour: int
    cost_this_hour: float
    remaining: float


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict] = {
    "screen": {
        "reliability": 0.95,
        "cost_per_event": 0.005,
        "latency_ms": 200.0,
        "information_density": 0.8,
        "attention_threshold": 0.3,
    },
    "speech": {
        "reliability": 0.70,
        "cost_per_event": 0.001,
        "latency_ms": 500.0,
        "information_density": 0.9,
        "attention_threshold": 0.2,
    },
    "camera": {
        "reliability": 0.60,
        "cost_per_event": 0.005,
        "latency_ms": 300.0,
        "information_density": 0.5,
        "attention_threshold": 0.4,
    },
}


def default_profiles() -> dict[str, ModalityProfile]:
    """Create default profiles for screen, speech, camera."""
    return {
        name: ModalityProfile(name=name, **params)
        for name, params in _DEFAULTS.items()
    }


# ---------------------------------------------------------------------------
# Processing gate
# ---------------------------------------------------------------------------

def should_process(
    profile: ModalityProfile,
    attention_score: float,
    budget: ModalityBudget | None = None,
) -> bool:
    """Determine if an event should be processed.

    Returns False if:
    - Modality is disabled
    - Attention score below threshold
    - Budget exhausted (remaining <= 0)
    """
    if not profile.enabled:
        return False

    if attention_score < profile.attention_threshold:
        return False

    if budget is not None and budget.remaining <= 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------

def allocate_budget(
    profiles: dict[str, ModalityProfile],
    total_hourly_budget: float,
) -> dict[str, ModalityBudget]:
    """Allocate hourly budget across modalities proportional to information density.

    Disabled modalities get zero budget. Their share is redistributed.
    """
    enabled = {n: p for n, p in profiles.items() if p.enabled}

    if not enabled:
        return {
            name: ModalityBudget(
                modality=name,
                hourly_limit=0.0,
                events_this_hour=0,
                cost_this_hour=0.0,
                remaining=0.0,
            )
            for name in profiles
        }

    total_density = sum(p.information_density for p in enabled.values())
    if total_density <= 0:
        # Equal split if all densities are zero
        per_modality = total_hourly_budget / len(enabled)
        total_density = 1.0  # avoid division by zero

    budgets: dict[str, ModalityBudget] = {}
    for name, profile in profiles.items():
        if not profile.enabled:
            budgets[name] = ModalityBudget(
                modality=name,
                hourly_limit=0.0,
                events_this_hour=0,
                cost_this_hour=0.0,
                remaining=0.0,
            )
        else:
            share = (profile.information_density / total_density) * total_hourly_budget
            budgets[name] = ModalityBudget(
                modality=name,
                hourly_limit=share,
                events_this_hour=0,
                cost_this_hour=0.0,
                remaining=share,
            )

    return budgets


def update_budget_after_event(
    budget: ModalityBudget,
    cost: float,
) -> ModalityBudget:
    """Return a new budget with updated cost and event count."""
    return ModalityBudget(
        modality=budget.modality,
        hourly_limit=budget.hourly_limit,
        events_this_hour=budget.events_this_hour + 1,
        cost_this_hour=budget.cost_this_hour + cost,
        remaining=max(0.0, budget.remaining - cost),
    )


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

def adjust_threshold(
    profile: ModalityProfile,
    recent_signal_rate: float,
    target_rate: float = 5.0,
    min_threshold: float = 0.1,
    max_threshold: float = 0.9,
) -> ModalityProfile:
    """Adjust attention threshold based on recent signal rate.

    If signal rate exceeds target (too noisy), raise threshold.
    If signal rate is below target (too quiet), lower threshold.
    """
    if target_rate <= 0:
        return profile

    ratio = recent_signal_rate / target_rate
    current = profile.attention_threshold

    if ratio > 1.5:
        # Too noisy — raise threshold
        new_threshold = min(max_threshold, current + 0.05)
    elif ratio < 0.5:
        # Too quiet — lower threshold
        new_threshold = max(min_threshold, current - 0.05)
    else:
        # In range — no change
        return profile

    return ModalityProfile(
        name=profile.name,
        reliability=profile.reliability,
        cost_per_event=profile.cost_per_event,
        latency_ms=profile.latency_ms,
        information_density=profile.information_density,
        enabled=profile.enabled,
        attention_threshold=new_threshold,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_modality_context(profiles: dict[str, ModalityProfile]) -> str:
    """Render active modalities for prompt awareness.

    Returns empty string if no modalities enabled.
    """
    enabled = [p for p in profiles.values() if p.enabled]
    if not enabled:
        return ""

    lines = ["[Active Modalities]"]
    for p in sorted(enabled, key=lambda x: -x.information_density):
        lines.append(
            f"  {p.name}: reliability={p.reliability:.0%}, "
            f"density={p.information_density:.1f}, "
            f"threshold={p.attention_threshold:.2f}"
        )

    return "\n".join(lines)
