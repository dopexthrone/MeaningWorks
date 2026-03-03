"""
Mother selective attention — significance filter for perception events.

LEAF module. Stdlib only. No imports from core/ or mother/.

Provides AttentionFilter that scores perception events by significance
and decides whether Mother should attend to them. Rate-limited to
prevent attention overload.
"""

import time
from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class AttentionScore:
    """Result of evaluating a single perception event."""

    should_attend: bool = False
    significance: float = 0.0    # 0.0-1.0
    rationale: str = ""


@dataclass(frozen=True)
class AttentionState:
    """Aggregate attention state for observability."""

    events_seen: int = 0
    events_attended: int = 0
    load: float = 0.0            # attended / capacity
    last_attended_time: float = 0.0


class AttentionFilter:
    """Deterministic significance filter for perception events.

    Scores events based on type, context, and rate limits.
    Tracks attended timestamps for rate limiting.
    """

    def __init__(self, capacity_per_minute: int = 5):
        self._capacity_per_minute = capacity_per_minute
        self._events_seen: int = 0
        self._events_attended: int = 0
        self._attended_timestamps: List[float] = []
        self._last_attended_time: float = 0.0

    def evaluate(
        self,
        event_type: str,               # "speech"/"screen"/"camera"
        payload_size: int,
        elapsed_since_last_event: float,
        senses_attentiveness: float,
        conversation_active: bool,
        now: Optional[float] = None,
    ) -> AttentionScore:
        """Score a perception event and decide whether to attend.

        Significance scoring (deterministic):
        - Speech + idle: 0.9 | Speech + chatting: 0.3
        - Screen + silence >30s: 0.5 | Screen + active: 0.2
        - Camera: 0.1 (+ 0.3 if attentiveness > 0.7)
        - Rate limit: >capacity/min → halve significance
        - Threshold: sig < 0.25 → should_attend=False
        """
        now = now if now is not None else time.time()
        self._events_seen += 1

        # Prune timestamps older than 120s
        self._attended_timestamps = [
            t for t in self._attended_timestamps if (now - t) <= 120.0
        ]

        # Base significance by event type
        if event_type == "speech":
            if conversation_active:
                sig = 0.3
                rationale = "speech during active conversation"
            else:
                sig = 0.9
                rationale = "speech while idle"
        elif event_type == "screen":
            if elapsed_since_last_event > 30.0 and not conversation_active:
                sig = 0.5
                rationale = "screen change after silence"
            else:
                sig = 0.2
                rationale = "screen change during activity"
        elif event_type == "camera":
            sig = 0.1
            rationale = "camera frame"
            if senses_attentiveness > 0.7:
                sig += 0.3
                rationale = "camera frame, high attentiveness"
        elif event_type == "health":
            sig = 0.8
            rationale = "process health event"
        else:
            sig = 0.1
            rationale = f"unknown event type: {event_type}"

        # Rate limiting: count attended in last 60s
        recent_attended = sum(
            1 for t in self._attended_timestamps if (now - t) <= 60.0
        )
        if recent_attended >= self._capacity_per_minute:
            sig *= 0.5
            rationale += " (rate-limited)"

        # Threshold
        should_attend = sig >= 0.25

        if should_attend:
            self._events_attended += 1
            self._attended_timestamps.append(now)
            self._last_attended_time = now

        return AttentionScore(
            should_attend=should_attend,
            significance=round(sig, 3),
            rationale=rationale,
        )

    @property
    def state(self) -> AttentionState:
        """Current aggregate attention state."""
        # Count attended in last 60s for load calculation
        now = time.time()
        recent = sum(1 for t in self._attended_timestamps if (now - t) <= 60.0)
        load = recent / self._capacity_per_minute if self._capacity_per_minute > 0 else 0.0
        return AttentionState(
            events_seen=self._events_seen,
            events_attended=self._events_attended,
            load=round(min(load, 1.0), 3),
            last_attended_time=self._last_attended_time,
        )
