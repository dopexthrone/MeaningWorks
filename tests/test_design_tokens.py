"""Tests for mother.design_tokens — design system codification."""

import json
import pytest
from pathlib import Path

from mother.design_tokens import (
    COLORS, FONTS, SPACING, PANEL_DIMENSIONS,
    PIPELINE_STATES, STATUS_DOT, ANIMATION,
    get_tokens, export_json, export_css,
)


class TestColorKeys:
    """All required color keys present."""

    def test_accent_colors(self):
        assert "accent" in COLORS
        assert "accent_deep" in COLORS
        assert "accent_subtle" in COLORS

    def test_status_colors(self):
        for key in ("status_green", "status_amber", "status_red", "status_blue", "status_purple"):
            assert key in COLORS

    def test_pipeline_colors(self):
        for key in ("pipeline_idle", "pipeline_listening", "pipeline_intent",
                     "pipeline_compiling", "pipeline_error", "pipeline_success"):
            assert key in COLORS

    def test_dark_surfaces(self):
        for key in ("dark_bg", "dark_surface", "dark_elevated", "dark_separator"):
            assert key in COLORS

    def test_light_surfaces(self):
        for key in ("light_bg", "light_surface", "light_elevated", "light_separator"):
            assert key in COLORS


class TestPanelDimensions:
    """All panel modes defined with w, h, r."""

    def test_all_modes(self):
        for mode in ("pill", "voice", "chat", "screen"):
            assert mode in PANEL_DIMENSIONS
            dims = PANEL_DIMENSIONS[mode]
            assert "w" in dims
            assert "h" in dims
            assert "r" in dims

    def test_pill_is_smallest(self):
        assert PANEL_DIMENSIONS["pill"]["h"] < PANEL_DIMENSIONS["chat"]["h"]


class TestPipelineStates:
    """Pipeline states match protocol expectations."""

    def test_all_states_present(self):
        for state in ("idle", "listening", "intent", "compiling", "error", "success"):
            assert state in PIPELINE_STATES

    def test_state_structure(self):
        for state, spec in PIPELINE_STATES.items():
            assert "color" in spec
            assert "label" in spec
            assert "pulse" in spec
            # Color references must exist in COLORS
            assert spec["color"] in COLORS

    def test_error_pulses(self):
        assert PIPELINE_STATES["error"]["pulse"] is True
        assert PIPELINE_STATES["idle"]["pulse"] is False


class TestGetTokens:
    """get_tokens() returns complete dict."""

    def test_all_sections(self):
        tokens = get_tokens()
        for key in ("colors", "fonts", "spacing", "panel_dimensions",
                     "pipeline_states", "status_dot", "animation"):
            assert key in tokens

    def test_json_serializable(self):
        tokens = get_tokens()
        serialized = json.dumps(tokens)
        restored = json.loads(serialized)
        assert restored == tokens


class TestExportJson:
    """export_json() produces valid JSON file."""

    def test_writes_valid_json(self, tmp_path):
        out = tmp_path / "tokens.json"
        export_json(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert "colors" in data
        assert "panel_dimensions" in data

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "sub" / "deep" / "tokens.json"
        export_json(str(out))
        assert out.exists()


class TestExportCss:
    """export_css() produces valid CSS custom properties."""

    def test_writes_css(self, tmp_path):
        out = tmp_path / "tokens.css"
        export_css(str(out))
        content = out.read_text()
        assert content.startswith(":root {")
        assert content.strip().endswith("}")

    def test_contains_color_vars(self, tmp_path):
        out = tmp_path / "tokens.css"
        export_css(str(out))
        content = out.read_text()
        assert "--color-accent:" in content
        assert "--color-dark-bg:" in content

    def test_contains_spacing_vars(self, tmp_path):
        out = tmp_path / "tokens.css"
        export_css(str(out))
        content = out.read_text()
        assert "--space-sm: 8px;" in content
        assert "--space-xl: 24px;" in content
