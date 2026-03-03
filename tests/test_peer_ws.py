"""
Tests for PeerWSServer and PeerWSClient — real-time peer events.

Tests use mocked websockets — no actual server needed.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# --- PeerEvent ---

class TestPeerEvent:
    """Test PeerEvent dataclass and serialization."""

    def test_create_event(self):
        from motherlabs_platform.peer_ws import PeerEvent

        event = PeerEvent(
            event_type="tool_published",
            instance_id="abc123",
            payload={"package_id": "pkg-1", "name": "weather"},
            timestamp="2026-02-15T10:00:00",
        )
        assert event.event_type == "tool_published"
        assert event.instance_id == "abc123"

    def test_event_to_json(self):
        from motherlabs_platform.peer_ws import PeerEvent

        event = PeerEvent(
            event_type="peer_joined",
            instance_id="abc",
            payload={"name": "Mac Mother"},
            timestamp="2026-02-15T10:00:00",
        )
        data = json.loads(event.to_json())
        assert data["event_type"] == "peer_joined"
        assert data["payload"]["name"] == "Mac Mother"

    def test_event_from_json(self):
        from motherlabs_platform.peer_ws import PeerEvent

        raw = json.dumps({
            "event_type": "compilation_complete",
            "instance_id": "xyz",
            "payload": {"trust": 85.0},
            "timestamp": "2026-02-15T10:00:00",
        })
        event = PeerEvent.from_json(raw)
        assert event.event_type == "compilation_complete"
        assert event.payload["trust"] == 85.0

    def test_event_roundtrip(self):
        from motherlabs_platform.peer_ws import PeerEvent

        original = PeerEvent(
            event_type="peer_left",
            instance_id="def456",
            payload={},
            timestamp="now",
        )
        restored = PeerEvent.from_json(original.to_json())
        assert restored == original


# --- PeerWSServer ---

class TestPeerWSServer:
    """Test PeerWSServer broadcast logic."""

    def test_broadcast_to_clients(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSServer, PeerEvent

            server = PeerWSServer()

            # Add mock WebSocket clients
            ws1 = AsyncMock()
            ws1.send = AsyncMock()
            ws2 = AsyncMock()
            ws2.send = AsyncMock()

            server._clients = {ws1, ws2}

            event = PeerEvent(
                event_type="tool_published",
                instance_id="abc",
                payload={"name": "weather"},
                timestamp="now",
            )

            sent = await server.broadcast(event)
            assert sent == 2
            ws1.send.assert_called_once()
            ws2.send.assert_called_once()

            # Verify the message content
            msg = ws1.send.call_args[0][0]
            data = json.loads(msg)
            assert data["event_type"] == "tool_published"
        asyncio.run(_run())

    def test_broadcast_removes_disconnected(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSServer, PeerEvent

            server = PeerWSServer()

            ws_good = AsyncMock()
            ws_good.send = AsyncMock()
            ws_bad = AsyncMock()
            ws_bad.send = AsyncMock(side_effect=Exception("closed"))

            server._clients = {ws_good, ws_bad}

            event = PeerEvent(
                event_type="peer_left",
                instance_id="abc",
                payload={},
                timestamp="now",
            )

            sent = await server.broadcast(event)
            assert sent == 1
            # Bad client should be removed
            assert ws_bad not in server._clients
            assert ws_good in server._clients
        asyncio.run(_run())

    def test_broadcast_no_clients(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSServer, PeerEvent

            server = PeerWSServer()

            event = PeerEvent(
                event_type="peer_joined",
                instance_id="abc",
                payload={},
                timestamp="now",
            )

            sent = await server.broadcast(event)
            assert sent == 0
        asyncio.run(_run())

    def test_client_count(self):
        from motherlabs_platform.peer_ws import PeerWSServer

        server = PeerWSServer()
        assert server.client_count == 0

        server._clients = {MagicMock(), MagicMock()}
        assert server.client_count == 2


# --- PeerWSServer Stop ---

class TestPeerWSServerStop:
    """Test PeerWSServer stop and cleanup."""

    def test_stop_closes_clients(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSServer

            server = PeerWSServer()

            ws1 = AsyncMock()
            ws1.close = AsyncMock()
            ws2 = AsyncMock()
            ws2.close = AsyncMock()

            server._clients = {ws1, ws2}
            server._server = None  # Skip actual server close

            await server.stop()
            ws1.close.assert_called_once()
            ws2.close.assert_called_once()
            assert len(server._clients) == 0
        asyncio.run(_run())


# --- PeerWSClient ---

class TestPeerWSClient:
    """Test PeerWSClient connection and event handling."""

    def test_initial_state(self):
        from motherlabs_platform.peer_ws import PeerWSClient

        client = PeerWSClient("ws://100.1.2.3:8766")
        assert client.is_connected is False

    def test_disconnect_when_not_connected(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSClient

            client = PeerWSClient("ws://100.1.2.3:8766")
            # Should not raise
            await client.disconnect()
            assert client.is_connected is False
        asyncio.run(_run())

    def test_event_callback(self):
        async def _run():
            from motherlabs_platform.peer_ws import PeerWSClient, PeerEvent

            received = []

            def on_event(event):
                received.append(event)

            client = PeerWSClient("ws://100.1.2.3:8766", on_event=on_event)

            # Simulate receiving a message by calling the callback directly
            event = PeerEvent(
                event_type="tool_published",
                instance_id="abc",
                payload={"name": "weather"},
                timestamp="now",
            )
            client._on_event(event)
            assert len(received) == 1
            assert received[0].event_type == "tool_published"
        asyncio.run(_run())


# --- Integration: Context + Persona ---

class TestPeerContextIntegration:
    """Test peer fields in ContextData and synthesize_situation."""

    def test_context_data_peer_fields(self):
        from mother.context import ContextData

        data = ContextData(
            connected_peers=[
                {"instance_id": "mac-1", "name": "Mac Mother", "host": "192.168.1.2", "port": "8000"},
                {"instance_id": "srv-1", "name": "Server", "host": "192.168.1.3", "port": "8000"},
            ],
        )
        assert len(data.connected_peers) == 2
        assert data.connected_peers[0]["name"] == "Mac Mother"

    def test_synthesize_situation_shows_peers(self):
        from mother.context import ContextData, synthesize_situation

        data = ContextData(
            connected_peers=[
                {"instance_id": "mac-1", "name": "Mac Mother", "host": "192.168.1.2"},
            ],
        )
        result = synthesize_situation(data)
        assert "Mac Mother" in result

    def test_synthesize_situation_no_peers(self):
        from mother.context import ContextData, synthesize_situation

        data = ContextData(connected_peers=[])
        result = synthesize_situation(data)
        assert "peer" not in result.lower()

    def test_persona_intent_routing_has_peer_examples(self):
        from mother.persona import INTENT_ROUTING

        assert "delegate" in INTENT_ROUTING
        assert "discover_peers" in INTENT_ROUTING
        assert "list_peers" in INTENT_ROUTING
