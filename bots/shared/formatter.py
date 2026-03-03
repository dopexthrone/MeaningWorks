"""Platform-agnostic result formatting.

Converts CompileBotResult into a FormattedResult with structured text
that can be adapted to Telegram Markdown or Discord embeds.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bots.shared.api_client import CompileBotResult


@dataclass
class FormattedResult:
    """Structured compilation result for display."""

    title: str = ""
    trust_score: float = 0.0
    trust_badge: str = "unverified"
    components: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    duration: float = 0.0
    error: Optional[str] = None


def format_result(result: CompileBotResult) -> FormattedResult:
    """Convert a CompileBotResult to a FormattedResult."""
    if not result.success:
        return FormattedResult(
            title="Compilation Failed",
            error=result.error or "Unknown error",
        )

    blueprint = result.blueprint or {}
    trust = result.trust or {}

    # Extract components
    components = []
    for comp in blueprint.get("components", []):
        name = comp.get("name", "")
        if name:
            components.append(name)

    # Extract gaps from trust report
    gaps = trust.get("gap_report", [])

    return FormattedResult(
        title=blueprint.get("core_need", "Compilation Complete"),
        trust_score=trust.get("overall_score", 0.0),
        trust_badge=trust.get("verification_badge", "unverified"),
        components=components,
        gaps=gaps,
        duration=result.duration_seconds,
    )


def format_as_text(fr: FormattedResult) -> str:
    """Render FormattedResult as plain text (works for Telegram Markdown)."""
    if fr.error:
        return f"*Compilation Failed*\n\n{fr.error}"

    lines = [f"*{fr.title}*"]
    lines.append(f"Trust: {fr.trust_score:.0f}% ({fr.trust_badge})")

    if fr.components:
        lines.append(f"\nComponents ({len(fr.components)}):")
        for c in fr.components[:10]:
            lines.append(f"  - {c}")
        if len(fr.components) > 10:
            lines.append(f"  ... and {len(fr.components) - 10} more")

    if fr.gaps:
        lines.append(f"\nGaps ({len(fr.gaps)}):")
        for g in fr.gaps[:5]:
            lines.append(f"  - {g}")

    if fr.duration > 0:
        lines.append(f"\nCompleted in {fr.duration:.1f}s")

    return "\n".join(lines)
