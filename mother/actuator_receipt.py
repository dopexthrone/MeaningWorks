"""
mother/actuator_receipt.py — Uniform action feedback receipts.

LEAF module. No imports from core/ or mother/. Stdlib only.

Wraps any actuator outcome (compilation, file write, search, self-build,
voice, web) into a uniform frozen receipt with success/failure, duration,
cost, and modalities affected.

Ring buffer (ActuatorStore) keeps bounded recent history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ActuatorReceipt",
    "ActuatorLog",
    "ActuatorStore",
    "create_receipt",
    "format_actuator_context",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActuatorReceipt:
    """Frozen receipt for a single actuator action."""
    action_type: str        # "compile" | "file_write" | "search" | "self_build" | "web" | "voice"
    success: bool
    started_at: float
    completed_at: float
    duration_seconds: float  # completed_at - started_at
    cost_usd: float = 0.0
    output_summary: str = ""
    error: str = ""
    modalities_affected: tuple[str, ...] = ()  # ("screen", "filesystem", "voice")


@dataclass(frozen=True)
class ActuatorLog:
    """Aggregate stats over recent receipts."""
    receipts: tuple[ActuatorReceipt, ...]
    total_actions: int
    success_rate: float
    total_cost: float
    avg_duration: float


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_receipt(
    action_type: str,
    success: bool,
    started_at: float,
    completed_at: float,
    cost_usd: float = 0.0,
    output_summary: str = "",
    error: str = "",
    modalities_affected: tuple[str, ...] = (),
) -> ActuatorReceipt:
    """Create receipt with auto-computed duration."""
    duration = max(0.0, completed_at - started_at)
    return ActuatorReceipt(
        action_type=action_type,
        success=success,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        cost_usd=cost_usd,
        output_summary=output_summary,
        error=error,
        modalities_affected=modalities_affected,
    )


# ---------------------------------------------------------------------------
# Store (ring buffer)
# ---------------------------------------------------------------------------

class ActuatorStore:
    """Bounded ring buffer of actuator receipts.

    Oldest receipts are evicted when max_receipts is exceeded.
    """

    def __init__(self, max_receipts: int = 100) -> None:
        self._max = max(1, max_receipts)
        self._receipts: list[ActuatorReceipt] = []

    def record(self, receipt: ActuatorReceipt) -> None:
        """Append receipt, evict oldest if full."""
        self._receipts.append(receipt)
        if len(self._receipts) > self._max:
            self._receipts = self._receipts[-self._max:]

    def recent(self, n: int = 10) -> tuple[ActuatorReceipt, ...]:
        """Most recent n receipts, newest last."""
        return tuple(self._receipts[-n:])

    def by_type(self, action_type: str) -> tuple[ActuatorReceipt, ...]:
        """Filter receipts by action_type."""
        return tuple(r for r in self._receipts if r.action_type == action_type)

    def summary(self) -> ActuatorLog:
        """Aggregate stats over all stored receipts."""
        receipts = tuple(self._receipts)
        total = len(receipts)
        if total == 0:
            return ActuatorLog(
                receipts=(),
                total_actions=0,
                success_rate=0.0,
                total_cost=0.0,
                avg_duration=0.0,
            )

        successes = sum(1 for r in receipts if r.success)
        total_cost = sum(r.cost_usd for r in receipts)
        avg_dur = sum(r.duration_seconds for r in receipts) / total

        return ActuatorLog(
            receipts=receipts,
            total_actions=total,
            success_rate=successes / total,
            total_cost=total_cost,
            avg_duration=avg_dur,
        )

    def count(self) -> int:
        """Number of stored receipts."""
        return len(self._receipts)

    def clear(self) -> None:
        """Remove all receipts."""
        self._receipts.clear()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_actuator_context(log: ActuatorLog, recent_n: int = 5) -> str:
    """Render recent actions for prompt context injection.

    Returns empty string if no actions recorded.
    """
    if log.total_actions == 0:
        return ""

    lines = ["[Recent Actions]"]
    lines.append(
        f"  Total: {log.total_actions}, "
        f"Success: {log.success_rate:.0%}, "
        f"Cost: ${log.total_cost:.4f}, "
        f"Avg duration: {log.avg_duration:.1f}s"
    )

    # Show most recent entries
    recent = log.receipts[-recent_n:] if len(log.receipts) > recent_n else log.receipts
    for r in reversed(recent):
        status = "ok" if r.success else "FAIL"
        summary = r.output_summary[:60] if r.output_summary else ""
        line = f"  [{status}] {r.action_type} ({r.duration_seconds:.1f}s)"
        if r.cost_usd > 0:
            line += f" ${r.cost_usd:.4f}"
        if summary:
            line += f" — {summary}"
        if r.error:
            line += f" ERR: {r.error[:40]}"
        lines.append(line)

    return "\n".join(lines)
