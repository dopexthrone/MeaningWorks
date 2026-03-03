"""
Trust badge widget — displays compilation trust indicators.

Shows VERIFIED/PARTIAL/UNVERIFIED badge with score bars.
Compact and expanded modes.
"""

from textual.widgets import Static


# Trust thresholds
VERIFIED_THRESHOLD = 80.0
PARTIAL_THRESHOLD = 40.0


def trust_level(score: float) -> str:
    """Determine trust level from score."""
    if score >= VERIFIED_THRESHOLD:
        return "verified"
    elif score >= PARTIAL_THRESHOLD:
        return "partial"
    return "unverified"


def trust_label(score: float) -> str:
    """Human-readable trust label."""
    level = trust_level(score)
    return {
        "verified": "VERIFIED",
        "partial": "PARTIAL",
        "unverified": "UNVERIFIED",
    }[level]


def format_score_bar(score: float, width: int = 20) -> str:
    """Render a text-based score bar."""
    filled = int((score / 100.0) * width)
    filled = max(0, min(width, filled))
    return f"[{'=' * filled}{' ' * (width - filled)}] {score:.0f}%"


class TrustBadge(Static):
    """Displays trust indicators for a compilation result."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._score: float = 0.0
        self._expanded: bool = False
        self._dimensions: dict = {}

    def set_trust(
        self,
        score: float,
        dimensions: dict | None = None,
    ) -> None:
        """Update trust display."""
        self._score = score
        self._dimensions = dimensions or {}
        self._render()

    def toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._render()

    def _render(self) -> None:
        level = trust_level(self._score)
        label = trust_label(self._score)
        css_class = f"trust-badge trust-{level}"
        self.set_classes(css_class)

        lines = [f"{label}  {format_score_bar(self._score)}"]

        if self._expanded and self._dimensions:
            for name, value in self._dimensions.items():
                bar = format_score_bar(value, width=15)
                lines.append(f"  {name}: {bar}")

        self.update("\n".join(lines))

    @property
    def score(self) -> float:
        return self._score

    @property
    def level(self) -> str:
        return trust_level(self._score)

    def interpret(self) -> str:
        """Mother-voiced trust interpretation."""
        level = self.level
        weakest = None
        weakest_score = 101.0
        for name, value in self._dimensions.items():
            if isinstance(value, (int, float)) and name != "overall_score":
                if value < weakest_score:
                    weakest = name
                    weakest_score = value

        if level == "verified":
            msg = "High confidence."
            if weakest and weakest_score < 70:
                msg += f" Weakest area: {weakest}."
            return msg
        elif level == "partial":
            msg = "Moderate confidence — review the gaps."
            if weakest:
                msg += f" Weakest: {weakest}."
            return msg
        else:
            msg = "Low confidence. I'd recommend refining the input."
            if weakest:
                msg += f" {weakest} needs attention."
            return msg
