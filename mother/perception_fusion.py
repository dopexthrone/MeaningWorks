"""
mother/perception_fusion.py — Cross-modal correlation detector.

LEAF module. No imports from core/ or mother/. Stdlib only.

Detects co-occurrence patterns across perception modalities within
configurable time windows. Produces composite FusionSignals like
"presenting", "away", "focused", "multitasking", "idle".

All detection is deterministic — zero LLM calls.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "FusionEvent",
    "FusionSignal",
    "PerceptionFusion",
    "format_fusion_context",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionEvent:
    """A single perception event for fusion processing."""
    event_type: str         # "screen_changed" | "speech_detected" | "camera_frame"
    timestamp: float
    summary: str = ""


@dataclass(frozen=True)
class FusionSignal:
    """A detected cross-modal pattern."""
    pattern: str            # "presenting" | "away" | "focused" | "multitasking" | "idle"
    confidence: float       # 0.0-1.0
    evidence: tuple[str, ...] # modalities contributing
    detected_at: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAPID_SCREEN_THRESHOLD = 3  # screen events within window = "frequent"


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class PerceptionFusion:
    """Sliding-window cross-modal co-occurrence detector.

    Ingests perception events and detects composite patterns based on
    which modalities have recent activity within the time window.
    """

    def __init__(self, window_seconds: float = 15.0, idle_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._idle_seconds = idle_seconds
        self._events: list[FusionEvent] = []
        self._last_event_time: float = 0.0

    @property
    def window_seconds(self) -> float:
        return self._window

    def ingest(self, event: FusionEvent) -> None:
        """Add event to sliding window, prune expired."""
        self._events.append(event)
        self._last_event_time = max(self._last_event_time, event.timestamp)
        self._prune(event.timestamp)

    def _prune(self, now: float) -> None:
        """Remove events outside the window."""
        cutoff = now - self._window
        self._events = [e for e in self._events if e.timestamp >= cutoff]

    def _modality_counts(self, now: float) -> dict[str, int]:
        """Count events per modality type within window."""
        self._prune(now)
        counts: dict[str, int] = defaultdict(int)
        for e in self._events:
            counts[e.event_type] += 1
        return dict(counts)

    def detect(self, now: float | None = None) -> list[FusionSignal]:
        """Check co-occurrence rules and return detected patterns.

        Rules:
        - screen_changed + speech_detected → "presenting" (0.8)
        - frequent screen_changed + speech_detected → "multitasking" (0.7)
        - screen_changed + no speech → "focused" (0.6)
        - no screen + no speech + camera empty/absent → "away" (0.9)
        - no events for >idle_seconds → "idle" (0.5)
        """
        ts = now if now is not None else time.time()
        self._prune(ts)
        counts = self._modality_counts(ts)

        has_screen = counts.get("screen_changed", 0) > 0
        has_speech = counts.get("speech_detected", 0) > 0
        has_camera = counts.get("camera_frame", 0) > 0
        frequent_screen = counts.get("screen_changed", 0) >= _RAPID_SCREEN_THRESHOLD

        signals: list[FusionSignal] = []

        # Check idle first (no events for extended period)
        if not self._events and self._last_event_time > 0:
            gap = ts - self._last_event_time
            if gap >= self._idle_seconds:
                signals.append(FusionSignal(
                    pattern="idle",
                    confidence=0.5,
                    evidence=(),
                    detected_at=ts,
                ))
                return signals

        # Multitasking: frequent screen changes + speech (more specific, checked first)
        if frequent_screen and has_speech:
            signals.append(FusionSignal(
                pattern="multitasking",
                confidence=0.7,
                evidence=("screen", "speech"),
                detected_at=ts,
            ))
        # Presenting: screen change + speech (less specific)
        elif has_screen and has_speech:
            signals.append(FusionSignal(
                pattern="presenting",
                confidence=0.8,
                evidence=("screen", "speech"),
                detected_at=ts,
            ))

        # Focused: screen activity but no speech
        if has_screen and not has_speech:
            signals.append(FusionSignal(
                pattern="focused",
                confidence=0.6,
                evidence=("screen",),
                detected_at=ts,
            ))

        # "Away" previously fired at 0.9 confidence whenever no events were in
        # window — even when modalities are simply disabled. This caused false
        # "away" during active chat (camera off ≠ user absent). Now subsumed by
        # the idle check above which correctly requires gap >= idle_seconds.

        return signals

    def clear(self) -> None:
        """Reset all state."""
        self._events.clear()
        self._last_event_time = 0.0

    def event_count(self) -> int:
        """Number of events currently in window."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_fusion_context(signals: list[FusionSignal]) -> str:
    """Render fusion signals for prompt injection.

    Returns empty string if no signals detected.
    """
    if not signals:
        return ""

    lines = ["[Activity State]"]
    for sig in signals:
        evidence_str = ", ".join(sig.evidence) if sig.evidence else "inferred"
        lines.append(
            f"  {sig.pattern} (conf={sig.confidence:.1f}, from: {evidence_str})"
        )

    return "\n".join(lines)
