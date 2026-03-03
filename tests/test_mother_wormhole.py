"""
Tests for mother/wormhole.py -- LEAF module.

Covers: WormholeMessage, WormholeConnection, message serialization/parsing.

Full integration tests require websockets library and are skipped if unavailable.
"""

import json
import pytest

from mother.wormhole import WormholeMessage


class TestWormholeMessage:
    def test_frozen(self):
        msg = WormholeMessage(message_id="123", message_type="handshake")
        with pytest.raises(AttributeError):
            msg.message_type = "other"

    def test_defaults(self):
        msg = WormholeMessage(message_id="123", message_type="test")
        assert msg.payload == {}
        assert msg.timestamp == 0.0
        assert msg.reply_to == ""


class TestMessageSerialization:
    def test_serialize_parse_roundtrip(self):
        from mother.wormhole import Wormhole

        original = WormholeMessage(
            message_id="msg123",
            message_type="compile_request",
            payload={"description": "build X", "domain": "software"},
            timestamp=1234567890.0,
            reply_to="",
        )

        serialized = Wormhole._serialize_message(original)
        parsed = Wormhole._parse_message(serialized)

        assert parsed is not None
        assert parsed.message_id == original.message_id
        assert parsed.message_type == original.message_type
        assert parsed.payload == original.payload
        assert parsed.timestamp == original.timestamp

    def test_parse_invalid_json(self):
        from mother.wormhole import Wormhole

        result = Wormhole._parse_message("{bad json")
        assert result is None

    def test_parse_missing_fields(self):
        from mother.wormhole import Wormhole

        result = Wormhole._parse_message('{"message_id": "123"}')
        assert result is not None
        assert result.message_id == "123"
        assert result.message_type == ""


@pytest.mark.skipif(
    not pytest.importorskip("websockets", reason="websockets not installed"),
    reason="websockets not available",
)
class TestWormholeIntegration:
    """Full integration tests. Requires websockets."""

    @pytest.mark.slow
    def test_server_integration_placeholder(self):
        """Wormhole server integration (requires pytest-asyncio, marked slow)."""
        # Full async integration test deferred to real-LLM/integration suite
        # Testing the imports and class construction here
        from mother.wormhole import Wormhole

        wormhole = Wormhole(
            instance_id="test123",
            instance_name="Test Mother",
            port=9999,
        )
        assert wormhole.instance_id == "test123"
        assert wormhole.port == 9999
        assert not wormhole._running
