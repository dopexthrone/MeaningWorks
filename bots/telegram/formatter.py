"""Telegram-specific formatting — Markdown output with trust scores."""

from bots.shared.formatter import FormattedResult


def format_telegram(fr: FormattedResult) -> str:
    """Render FormattedResult as Telegram MarkdownV2-safe text."""
    if fr.error:
        return f"*Compilation Failed*\n\n`{_escape(fr.error)}`"

    lines = [f"*{_escape(fr.title)}*"]
    lines.append(f"Trust: `{fr.trust_score:.0f}%` \\({_escape(fr.trust_badge)}\\)")

    if fr.components:
        lines.append(f"\n*Components* \\({len(fr.components)}\\):")
        for c in fr.components[:10]:
            lines.append(f"  \\- {_escape(c)}")
        if len(fr.components) > 10:
            lines.append(f"  \\.\\.\\. and {len(fr.components) - 10} more")

    if fr.gaps:
        lines.append(f"\n*Gaps* \\({len(fr.gaps)}\\):")
        for g in fr.gaps[:5]:
            lines.append(f"  \\- {_escape(g)}")

    if fr.duration > 0:
        lines.append(f"\nCompleted in `{fr.duration:.1f}s`")

    return "\n".join(lines)


def _escape(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    chars = r"_*[]()~`>#+-=|{}.!"
    for c in chars:
        text = text.replace(c, f"\\{c}")
    return text
