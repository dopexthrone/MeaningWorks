"""
kernel/perception_bridge.py — Perception → grid fill translation.

LEAF module. Imports only kernel/cell.py types.

Deterministic mapping from perception modalities to grid postcodes.
No LLM calls. Same perception event always maps to the same cell
(re-fill = AX3 revision, not new cell).
"""

from __future__ import annotations

from kernel.cell import parse_postcode

__all__ = [
    "MODALITY_POSTCODES",
    "perception_to_fill",
    "environment_snapshot_to_fills",
    "fusion_signal_to_fill",
]


# ---------------------------------------------------------------------------
# Modality → postcode mapping
# ---------------------------------------------------------------------------

MODALITY_POSTCODES: dict[str, str] = {
    "screen": "OBS.ENV.APP.WHAT.USR",
    "speech": "OBS.USR.APP.WHAT.USR",
    "camera": "OBS.USR.APP.WHERE.USR",
    "fusion": "OBS.ENV.APP.HOW.USR",
}

# Reverse: for looking up modality from a postcode key
_POSTCODE_TO_MODALITY: dict[str, str] = {v: k for k, v in MODALITY_POSTCODES.items()}


def perception_to_fill(
    modality: str,
    summary: str,
    confidence: float,
    timestamp: float,
) -> dict:
    """Convert a perception event to kwargs for kernel/ops.fill().

    Args:
        modality: "screen" | "speech" | "camera" | "fusion"
        summary: Human-readable summary of the observation
        confidence: 0.0-1.0, from attention filter significance
        timestamp: time.time() when observed

    Returns:
        Dict of kwargs for ops.fill():
            postcode_key, primitive, content, confidence, source

    Raises:
        ValueError: Unknown modality
    """
    postcode_key = MODALITY_POSTCODES.get(modality)
    if postcode_key is None:
        raise ValueError(
            f"Unknown modality {modality!r}. "
            f"Valid: {sorted(MODALITY_POSTCODES.keys())}"
        )

    # Validate the postcode is well-formed
    parse_postcode(postcode_key)

    # Truncate summary for primitive (short label)
    primitive = summary[:60].strip() if summary else modality
    content = summary[:500].strip() if summary else ""
    confidence = max(0.0, min(1.0, confidence))

    return {
        "postcode_key": postcode_key,
        "primitive": primitive,
        "content": content,
        "confidence": confidence,
        "source": (f"observation:{modality}:{timestamp}",),
    }


def environment_snapshot_to_fills(
    snapshot_entries: list[tuple[str, str, float, float]],
) -> list[dict]:
    """Batch convert environment snapshot entries to fill params.

    Args:
        snapshot_entries: List of (modality, summary, confidence, observed_at)
            tuples — the caller extracts these from EnvironmentSnapshot.entries.

    Returns:
        List of fill kwarg dicts. One per valid modality entry.
        Entries with unknown modalities are silently skipped.
    """
    fills: list[dict] = []
    for modality, summary, confidence, observed_at in snapshot_entries:
        if modality not in MODALITY_POSTCODES:
            continue
        try:
            fill_params = perception_to_fill(modality, summary, confidence, observed_at)
            fills.append(fill_params)
        except ValueError:
            continue
    return fills


def fusion_signal_to_fill(
    pattern: str,
    confidence: float,
    evidence: tuple[str, ...],
    timestamp: float,
) -> dict:
    """Convert a FusionSignal to fill params for the world grid.

    Fusion signals always map to OBS.ENV.APP.HOW.USR (the "fusion" modality).

    Args:
        pattern: "presenting" | "away" | "focused" | "multitasking" | "idle"
        confidence: 0.0-1.0
        evidence: Modalities contributing to the signal
        timestamp: time.time() when detected

    Returns:
        Dict of kwargs for ops.fill()
    """
    postcode_key = MODALITY_POSTCODES["fusion"]

    evidence_str = ", ".join(evidence) if evidence else "inferred"
    primitive = f"{pattern} ({evidence_str})"[:60]
    content = f"Fusion pattern: {pattern}. Evidence: {evidence_str}."

    confidence = max(0.0, min(1.0, confidence))

    return {
        "postcode_key": postcode_key,
        "primitive": primitive,
        "content": content,
        "confidence": confidence,
        "source": (f"fusion:{pattern}:{timestamp}",),
    }


def modality_for_postcode(postcode_key: str) -> str | None:
    """Look up which modality a postcode belongs to, or None."""
    return _POSTCODE_TO_MODALITY.get(postcode_key)
