"""
Panel IPC protocol — message types and serialization.

LEAF module. Stdlib only. No imports from core/.

Defines the wire format between Mother's Python backend and
the native macOS SwiftUI panel (or any other IPC client).
"""

import json
import uuid
from dataclasses import dataclass, field


# --- Message types ---

class MessageType:
    """String constants for panel IPC messages."""

    # Client -> Server
    CHAT = "chat"
    CANCEL_STREAM = "cancel_stream"
    CAPTURE_SCREEN = "capture_screen"
    RECORD_AUDIO = "record_audio"
    SUBSCRIBE = "subscribe"
    AUTH = "auth"

    # Server -> Client
    CHAT_TOKEN = "chat_token"
    CHAT_DONE = "chat_done"
    SENSES_UPDATE = "senses_update"
    POSTURE_UPDATE = "posture_update"
    COMPILE_INSIGHT = "compile_insight"
    BUILD_PHASE = "build_phase"
    PERCEPTION_EVENT = "perception_event"
    ERROR = "error"
    READY = "ready"


class Channel:
    """Subscription channels for push updates."""

    SENSES = "senses"
    PERCEPTION = "perception"
    GOALS = "goals"
    PIPELINE = "pipeline"
    APPENDAGES = "appendages"

    ALL = ("senses", "perception", "goals", "pipeline", "appendages")


# --- Message dataclass ---

@dataclass(frozen=True)
class PanelMessage:
    """Single IPC message between panel client and server.

    Frozen for thread safety across async boundaries.
    """

    msg_type: str
    msg_id: str = ""
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON wire format."""
        return json.dumps({
            "type": self.msg_type,
            "id": self.msg_id,
            "payload": self.payload,
        })

    @classmethod
    def from_json(cls, raw: str) -> "PanelMessage":
        """Deserialize from JSON wire format.

        Raises ValueError on invalid JSON or missing 'type' field.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object")

        msg_type = data.get("type")
        if not msg_type:
            raise ValueError("Message missing 'type' field")

        return cls(
            msg_type=msg_type,
            msg_id=data.get("id", ""),
            payload=data.get("payload", {}),
        )

    @classmethod
    def new(cls, msg_type: str, payload: dict = None) -> "PanelMessage":
        """Create a new message with auto-generated ID."""
        return cls(
            msg_type=msg_type,
            msg_id=uuid.uuid4().hex[:12],
            payload=payload or {},
        )
