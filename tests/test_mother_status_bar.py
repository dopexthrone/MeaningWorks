"""
Phase 3: Tests for status bar widget.
"""

import pytest

from mother.widgets.status_bar import StatusBar


class TestStatusBar:
    """Test StatusBar widget."""

    def test_default_name(self):
        bar = StatusBar()
        assert bar.instance_name == "Mother"

    def test_default_state(self):
        bar = StatusBar()
        assert bar.state == "ready"

    def test_default_cost(self):
        bar = StatusBar()
        assert bar.session_cost == 0.0

    def test_default_model(self):
        bar = StatusBar()
        assert bar.model == ""
