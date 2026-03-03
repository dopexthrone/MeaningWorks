"""
Phase 3: Tests for input area widget.
"""

import pytest

from mother.widgets.input_area import InputArea, InputSubmitted, UserTextSubmitted


class TestInputSubmitted:
    """Test InputSubmitted message."""

    def test_text_stored(self):
        msg = InputSubmitted(text="hello")
        assert msg.text == "hello"

    def test_is_command_false_by_default(self):
        msg = InputSubmitted(text="hello")
        assert msg.is_command is False

    def test_is_command_true_for_slash(self):
        msg = InputSubmitted(text="/help", is_command=True)
        assert msg.is_command is True

    def test_alias_is_same_class(self):
        assert InputSubmitted is UserTextSubmitted


class TestInputArea:
    """Test InputArea widget."""

    def test_creates_widget(self):
        area = InputArea()
        assert area.id == "input-area"
