"""
Design system tokens — single source of truth for the Mother panel.

LEAF module. Stdlib only. No imports from core/.

Colors, typography, spacing, dimensions, pipeline states, and animation
parameters. Shared between the Python backend and the SwiftUI panel
via exported JSON at ~/.motherlabs/design-tokens.json.
"""

import json
from pathlib import Path


# --- Colors ---

COLORS = {
    # Accent (Anthropic warm orange — "Crail")
    "accent": "#D97757",
    "accent_deep": "#C15F3C",
    "accent_subtle": "rgba(217,119,87,0.15)",

    # Status (Apple system colors)
    "status_green": "#34C759",
    "status_amber": "#FF9F0A",
    "status_red": "#FF3B30",
    "status_blue": "#007AFF",
    "status_purple": "#AF52DE",

    # Pipeline states
    "pipeline_idle": "#B0AEA5",
    "pipeline_listening": "#788C5D",
    "pipeline_intent": "#D97757",
    "pipeline_compiling": "#6A9BCC",
    "pipeline_error": "#C15F3C",
    "pipeline_success": "#788C5D",

    # Surfaces (dark mode)
    "dark_bg": "#141413",
    "dark_surface": "rgba(30,29,27,0.90)",
    "dark_elevated": "#2A2926",
    "dark_separator": "rgba(250,249,245,0.08)",

    # Surfaces (light mode)
    "light_bg": "#FAF9F5",
    "light_surface": "rgba(244,243,238,0.88)",
    "light_elevated": "#F4F3EE",
    "light_separator": "rgba(20,20,19,0.06)",
}

# --- Typography ---

FONTS = {
    "system": '-apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif',
    "mono": '"SF Mono", ui-monospace, "Menlo", monospace',
}

# --- Spacing (px) ---

SPACING = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}

# --- Panel dimensions (pt) ---

PANEL_DIMENSIONS = {
    "pill":   {"w": 172, "h": 40,  "r": 20},
    "voice":  {"w": 260, "h": 196, "r": 22},
    "chat":   {"w": 304, "h": 404, "r": 22},
    "screen": {"w": 340, "h": 260, "r": 22},
}

# --- Pipeline states ---

PIPELINE_STATES = {
    "idle":      {"color": "pipeline_idle",      "label": "Idle",      "pulse": False},
    "listening": {"color": "pipeline_listening",  "label": "Listening", "pulse": True},
    "intent":    {"color": "pipeline_intent",     "label": "Intent",    "pulse": False},
    "compiling": {"color": "pipeline_compiling",  "label": "Compiling", "pulse": False},
    "error":     {"color": "pipeline_error",      "label": "Error",     "pulse": True},
    "success":   {"color": "pipeline_success",    "label": "Complete",  "pulse": False},
}

# --- Status dot ---

STATUS_DOT = {"size": 7, "glow_radius_factor": 0.8}

# --- Animation ---

ANIMATION = {
    "morph_duration": 0.5,
    "morph_curve": "cubic-bezier(0.2, 0.9, 0.3, 1.0)",
    "micro_duration": 0.25,
}


# --- Export functions ---

def get_tokens() -> dict:
    """Return the full token set as a nested dict."""
    return {
        "colors": COLORS,
        "fonts": FONTS,
        "spacing": SPACING,
        "panel_dimensions": PANEL_DIMENSIONS,
        "pipeline_states": PIPELINE_STATES,
        "status_dot": STATUS_DOT,
        "animation": ANIMATION,
    }


def export_json(path: str) -> None:
    """Write the full token set to a JSON file.

    Creates parent directories if needed.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(get_tokens(), indent=2))


def export_css(path: str) -> None:
    """Write CSS custom properties for web appendage UIs.

    Generates :root { --token-name: value; } for colors and spacing.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [":root {"]

    # Colors
    for key, value in COLORS.items():
        lines.append(f"  --color-{key.replace('_', '-')}: {value};")

    # Spacing
    for key, value in SPACING.items():
        lines.append(f"  --space-{key}: {value}px;")

    # Status dot
    lines.append(f"  --status-dot-size: {STATUS_DOT['size']}px;")

    # Animation
    lines.append(f"  --morph-duration: {ANIMATION['morph_duration']}s;")
    lines.append(f"  --morph-curve: {ANIMATION['morph_curve']};")
    lines.append(f"  --micro-duration: {ANIMATION['micro_duration']}s;")

    lines.append("}")
    p.write_text("\n".join(lines) + "\n")
