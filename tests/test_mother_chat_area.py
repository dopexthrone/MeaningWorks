"""
Phase 3: Tests for chat area widget.
"""

import pytest

from mother.widgets.chat_area import ChatArea, MessageWidget


class TestChatArea:
    """Test ChatArea widget."""

    def test_initial_count_zero(self):
        area = ChatArea()
        assert area.message_count == 0

    def test_message_widget_stores_role(self):
        widget = MessageWidget(role="user", text="hello")
        assert widget._role == "user"
        assert widget._text == "hello"

    def test_message_widget_mother_role(self):
        widget = MessageWidget(role="mother", text="hi")
        assert widget._role == "mother"


class TestMessageWidget:
    """Test individual message widget."""

    def test_stores_content(self):
        widget = MessageWidget(role="assistant", text="test content")
        assert widget._text == "test content"

    def test_stores_role(self):
        widget = MessageWidget(role="system", text="info")
        assert widget._role == "system"

    def test_different_roles(self):
        for role in ("user", "mother", "system", "assistant"):
            w = MessageWidget(role=role, text="x")
            assert w._role == role
