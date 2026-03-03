"""Discord-specific formatting — embeds with trust scores.

Reuses patterns from mother/discord.py:build_embed().
"""

from typing import List, Optional
from bots.shared.formatter import FormattedResult

PURPLE = 0x7C3AED
RED = 0xEF4444
GREEN = 0x22C55E


def build_embed(fr: FormattedResult) -> dict:
    """Convert FormattedResult to a Discord embed dict."""
    if fr.error:
        return {
            "title": "Compilation Failed",
            "description": fr.error[:4096],
            "color": RED,
        }

    fields = []

    # Trust
    fields.append({
        "name": "Trust Score",
        "value": f"{fr.trust_score:.0f}%",
        "inline": True,
    })
    fields.append({
        "name": "Badge",
        "value": fr.trust_badge,
        "inline": True,
    })

    # Duration
    if fr.duration > 0:
        fields.append({
            "name": "Duration",
            "value": f"{fr.duration:.1f}s",
            "inline": True,
        })

    # Components
    if fr.components:
        comp_text = "\n".join(f"- {c}" for c in fr.components[:10])
        if len(fr.components) > 10:
            comp_text += f"\n... and {len(fr.components) - 10} more"
        fields.append({
            "name": f"Components ({len(fr.components)})",
            "value": comp_text[:1024],
            "inline": False,
        })

    # Gaps
    if fr.gaps:
        gap_text = "\n".join(f"- {g}" for g in fr.gaps[:5])
        fields.append({
            "name": f"Gaps ({len(fr.gaps)})",
            "value": gap_text[:1024],
            "inline": False,
        })

    color = GREEN if fr.trust_score >= 70 else PURPLE

    embed = {
        "title": fr.title[:256],
        "color": color,
        "fields": fields[:25],
    }
    return embed
