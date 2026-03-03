"""Tests for mother.panel_protocol — message types and serialization."""

import json
import pytest

from mother.panel_protocol import PanelMessage, MessageType, Channel


class TestMessageType:
    """MessageType constants exist and are strings."""

    def test_client_message_types(self):
        assert MessageType.CHAT == "chat"
        assert MessageType.CANCEL_STREAM == "cancel_stream"
        assert MessageType.CAPTURE_SCREEN == "capture_screen"
        assert MessageType.RECORD_AUDIO == "record_audio"
        assert MessageType.SUBSCRIBE == "subscribe"
        assert MessageType.AUTH == "auth"

    def test_server_message_types(self):
        assert MessageType.CHAT_TOKEN == "chat_token"
        assert MessageType.CHAT_DONE == "chat_done"
        assert MessageType.SENSES_UPDATE == "senses_update"
        assert MessageType.POSTURE_UPDATE == "posture_update"
        assert MessageType.COMPILE_INSIGHT == "compile_insight"
        assert MessageType.BUILD_PHASE == "build_phase"
        assert MessageType.PERCEPTION_EVENT == "perception_event"
        assert MessageType.ERROR == "error"
        assert MessageType.READY == "ready"


class TestChannel:
    """Channel constants exist."""

    def test_all_channels(self):
        assert Channel.SENSES == "senses"
        assert Channel.PERCEPTION == "perception"
        assert Channel.GOALS == "goals"
        assert Channel.PIPELINE == "pipeline"
        assert Channel.APPENDAGES == "appendages"

    def test_all_tuple(self):
        assert set(Channel.ALL) == {"senses", "perception", "goals", "pipeline", "appendages"}


class TestPanelMessage:
    """PanelMessage construction, frozen check, serialization."""

    def test_construction(self):
        msg = PanelMessage(msg_type="chat", msg_id="abc", payload={"text": "hi"})
        assert msg.msg_type == "chat"
        assert msg.msg_id == "abc"
        assert msg.payload == {"text": "hi"}

    def test_defaults(self):
        msg = PanelMessage(msg_type="ready")
        assert msg.msg_id == ""
        assert msg.payload == {}

    def test_frozen(self):
        msg = PanelMessage(msg_type="chat")
        with pytest.raises(AttributeError):
            msg.msg_type = "error"

    def test_to_json(self):
        msg = PanelMessage(msg_type="chat_token", msg_id="x1", payload={"token": "Hello"})
        data = json.loads(msg.to_json())
        assert data["type"] == "chat_token"
        assert data["id"] == "x1"
        assert data["payload"]["token"] == "Hello"

    def test_from_json_roundtrip(self):
        original = PanelMessage(msg_type="senses_update", msg_id="s1", payload={"confidence": 0.8})
        wire = original.to_json()
        restored = PanelMessage.from_json(wire)
        assert restored.msg_type == original.msg_type
        assert restored.msg_id == original.msg_id
        assert restored.payload == original.payload

    def test_from_json_invalid(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            PanelMessage.from_json("not json")

    def test_from_json_missing_type(self):
        with pytest.raises(ValueError, match="missing 'type'"):
            PanelMessage.from_json('{"id": "x", "payload": {}}')

    def test_from_json_not_object(self):
        with pytest.raises(ValueError, match="JSON object"):
            PanelMessage.from_json('"just a string"')

    def test_new_generates_id(self):
        msg = PanelMessage.new("chat", {"text": "hi"})
        assert msg.msg_type == "chat"
        assert len(msg.msg_id) == 12
        assert msg.payload == {"text": "hi"}

    def test_new_unique_ids(self):
        ids = {PanelMessage.new("chat").msg_id for _ in range(100)}
        assert len(ids) == 100
