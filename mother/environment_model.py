"""
mother/environment_model.py — Decaying world state model.

LEAF module. No imports from core/ or mother/. Stdlib only.

Accumulates perception observations into a time-decaying queryable state.
Each modality has a configurable half-life — entries decay exponentially
and are pruned when confidence drops below threshold.

Provides format_environment_context() for prompt injection.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "EnvironmentEntry",
    "EnvironmentSnapshot",
    "EnvironmentModel",
    "create_model",
    "format_environment_context",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentEntry:
    """A single observation in the world model."""
    modality: str           # "screen" | "speech" | "camera"
    summary: str            # "VS Code - main.py" | "user speaking" | "user at desk"
    confidence: float       # 0.0-1.0, decays over time
    observed_at: float      # time.time()
    raw_hash: str = ""      # MD5 of payload for dedup


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Frozen point-in-time view of the environment."""
    entries: tuple[EnvironmentEntry, ...]
    taken_at: float
    dominant_context: str   # highest-confidence summary
    staleness: float        # avg age of entries in seconds


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HALF_LIVES: dict[str, float] = {
    "screen": 60.0,
    "speech": 30.0,
    "camera": 120.0,
}

_PRUNE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EnvironmentModel:
    """Time-decaying world state accumulator.

    Entries decay exponentially based on per-modality half-lives.
    Deduplicates by raw_hash — same hash updates confidence, doesn't duplicate.
    """

    def __init__(self, half_lives: dict[str, float] | None = None) -> None:
        self._half_lives = dict(half_lives) if half_lives else dict(_DEFAULT_HALF_LIVES)
        self._entries: dict[str, EnvironmentEntry] = {}  # key = raw_hash or auto-key

    @property
    def half_lives(self) -> dict[str, float]:
        return dict(self._half_lives)

    def observe(
        self,
        modality: str,
        summary: str,
        confidence: float = 1.0,
        raw_hash: str = "",
        now: float | None = None,
    ) -> None:
        """Add or update an observation.

        If raw_hash matches an existing entry, updates it (refreshes timestamp
        and takes max confidence). Otherwise adds a new entry.
        """
        ts = now if now is not None else time.time()
        confidence = max(0.0, min(1.0, confidence))

        if raw_hash:
            key = raw_hash
        else:
            # Auto-key from modality + summary hash
            key = hashlib.md5(f"{modality}:{summary}".encode()).hexdigest()

        entry = EnvironmentEntry(
            modality=modality,
            summary=summary,
            confidence=confidence,
            observed_at=ts,
            raw_hash=raw_hash,
        )
        self._entries[key] = entry

    def snapshot(self, now: float | None = None) -> EnvironmentSnapshot:
        """Decay confidences, prune below threshold, return frozen snapshot."""
        ts = now if now is not None else time.time()
        decayed: list[EnvironmentEntry] = []

        for entry in self._entries.values():
            age = ts - entry.observed_at
            half_life = self._half_lives.get(entry.modality, 60.0)
            if half_life <= 0:
                decay_factor = 0.0
            else:
                decay_factor = math.pow(0.5, age / half_life)
            new_conf = entry.confidence * decay_factor

            if new_conf >= _PRUNE_THRESHOLD:
                decayed.append(EnvironmentEntry(
                    modality=entry.modality,
                    summary=entry.summary,
                    confidence=new_conf,
                    observed_at=entry.observed_at,
                    raw_hash=entry.raw_hash,
                ))

        # Update internal state — prune dead entries
        new_entries: dict[str, EnvironmentEntry] = {}
        for e in decayed:
            key = e.raw_hash if e.raw_hash else hashlib.md5(
                f"{e.modality}:{e.summary}".encode()
            ).hexdigest()
            new_entries[key] = e
        self._entries = {
            k: self._entries[k] for k in self._entries if k in new_entries
        }

        # Compute snapshot fields
        entries_tuple = tuple(sorted(decayed, key=lambda e: -e.confidence))
        dominant = entries_tuple[0].summary if entries_tuple else ""
        if entries_tuple:
            staleness = sum(ts - e.observed_at for e in entries_tuple) / len(entries_tuple)
        else:
            staleness = 0.0

        return EnvironmentSnapshot(
            entries=entries_tuple,
            taken_at=ts,
            dominant_context=dominant,
            staleness=staleness,
        )

    def query(self, modality: str, now: float | None = None) -> list[EnvironmentEntry]:
        """Filter entries by modality, applying decay."""
        snap = self.snapshot(now=now)
        return [e for e in snap.entries if e.modality == modality]

    def dominant(self, now: float | None = None) -> str:
        """Highest-confidence summary across all modalities."""
        snap = self.snapshot(now=now)
        return snap.dominant_context

    def entry_count(self) -> int:
        """Current number of live entries (before decay)."""
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(half_lives: dict[str, float] | None = None) -> EnvironmentModel:
    """Factory with defaults: screen=60s, speech=30s, camera=120s."""
    return EnvironmentModel(half_lives=half_lives)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_environment_context(snapshot: EnvironmentSnapshot) -> str:
    """Render environment snapshot for prompt injection.

    Returns empty string if snapshot has no entries.
    """
    if not snapshot.entries:
        return ""

    lines = ["[Environment State]"]
    for entry in snapshot.entries[:5]:  # cap at 5 most confident
        age = snapshot.taken_at - entry.observed_at
        if age < 60:
            age_str = f"{age:.0f}s ago"
        else:
            age_str = f"{age / 60:.1f}m ago"
        lines.append(
            f"  {entry.modality}: {entry.summary} "
            f"(conf={entry.confidence:.2f}, {age_str})"
        )

    if snapshot.dominant_context:
        lines.append(f"  Dominant: {snapshot.dominant_context}")
    lines.append(f"  Staleness: {snapshot.staleness:.1f}s avg")

    return "\n".join(lines)
