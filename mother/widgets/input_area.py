"""
Input area widget — multiline text input with slash command detection.

Enter sends, Shift+Enter inserts newline. Supports pasting multi-paragraph text.
"""

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import TextArea
from textual.message import Message
from textual import events


class UserTextSubmitted(Message):
    """Posted when user submits text via the input area."""

    def __init__(self, text: str, is_command: bool = False):
        super().__init__()
        self.text = text
        self.is_command = is_command


# Keep old name as alias for backwards compatibility in tests
InputSubmitted = UserTextSubmitted


class ChatTextArea(TextArea):
    """TextArea subclass: Enter sends, Shift+Enter inserts newline.

    System clipboard paste (Cmd+V / terminal bracketed paste) works
    via the inherited TextArea._on_paste handler.
    """

    async def _on_key(self, event: events.Key) -> None:
        """Enter submits. Shift+Enter inserts newline. Everything else normal."""
        if event.key == "enter":
            # Submit the text
            event.stop()
            event.prevent_default()
            text = self.text.strip()
            if not text:
                return
            self.clear()
            is_command = text.startswith("/")
            self.post_message(UserTextSubmitted(text=text, is_command=is_command))
            return
        if event.key == "shift+enter":
            # Insert newline — rewrite as plain enter for TextArea
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        # Everything else: typing, Cmd+V paste, arrows, etc.
        await super()._on_key(event)


class InputArea(Container):
    """Chat input area with command detection. Multiline, Enter to send."""

    def __init__(self, **kwargs):
        super().__init__(id="input-area", **kwargs)

    def compose(self) -> ComposeResult:
        yield ChatTextArea(id="chat-input")

    def focus_input(self) -> None:
        """Focus the input widget."""
        try:
            self.query_one("#chat-input", TextArea).focus()
        except Exception:
            pass
