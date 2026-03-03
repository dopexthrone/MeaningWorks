"""
Chat area widget — scrollable message display.

Each message is a MessageWidget with role-based CSS class.
Mother messages: left-aligned, muted cyan border.
User messages: left-aligned, white border.
"""

from typing import Optional

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static


class MessageWidget(Static):
    """A single chat message."""

    def __init__(self, role: str, text: str, **kwargs):
        self._role = role
        self._text = text
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        role_label = self._role.capitalize()
        role_class = f"{self._role}-role"
        yield Static(role_label, classes=f"message-role {role_class}")
        yield Static(escape_markup(self._text))

    def update_text(self, new_content: str) -> None:
        """Update the text child of this message (for streaming)."""
        children = list(self.children)
        if len(children) >= 2:
            children[1].update(escape_markup(new_content))


class ChatArea(ScrollableContainer):
    """Scrollable chat display area."""

    def __init__(self, **kwargs):
        super().__init__(id="chat-area", **kwargs)
        self._message_count = 0
        self._streaming_msg: Optional[MessageWidget] = None
        self._streaming_buffer: str = ""
        self.instance_name: str = "Mother"
        self.user_name: str = "User"

    def add_ai_message(self, content: str) -> None:
        """Add a message from the AI using the configured instance name."""
        self.add_message(self.instance_name.lower(), content)

    def add_user_message(self, content: str) -> None:
        """Add a message from the user using the configured user name."""
        self.add_message(self.user_name.lower(), content)

    def add_message(self, role: str, content: str) -> None:
        """Add a message and auto-scroll to bottom."""
        css_class = f"message {role}-message"
        widget = MessageWidget(role=role, text=content, classes=css_class)
        self.mount(widget)
        self._message_count += 1
        widget.scroll_visible()

    def begin_streaming_message(self, role: str) -> "MessageWidget":
        """Mount an empty message widget for incremental updates."""
        css_class = f"message {role}-message"
        widget = MessageWidget(role=role, text="", classes=css_class)
        self.mount(widget)
        self._message_count += 1
        self._streaming_msg = widget
        self._streaming_buffer = ""
        widget.scroll_visible()
        return widget

    def append_streaming_text(self, text: str) -> None:
        """Append token to the streaming message."""
        if self._streaming_msg:
            self._streaming_buffer += text
            self._streaming_msg.update_text(self._streaming_buffer)
            self._streaming_msg.scroll_visible()

    def finish_streaming_message(self, final_text: str) -> None:
        """Swap raw streamed text with parsed/cleaned final text."""
        if self._streaming_msg:
            self._streaming_msg.update_text(final_text)
            self._streaming_msg = None
            self._streaming_buffer = ""

    def clear_messages(self) -> None:
        """Remove all messages."""
        self.remove_children()
        self._message_count = 0
        self._streaming_msg = None
        self._streaming_buffer = ""

    @property
    def message_count(self) -> int:
        return self._message_count
