"""Tests for mother.panel_server — HTTP routes, WebSocket, auth, PID lock."""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from mother.panel_server import (
    create_app, acquire_lock, release_lock, ServerState,
    _is_pid_alive,
)
from mother.panel_protocol import MessageType


# --- Fixtures ---

@pytest.fixture
def mock_bridge():
    """Create a mock EngineBridge with common methods stubbed."""
    bridge = MagicMock()
    bridge.get_status.return_value = {
        "provider": "claude",
        "model": "claude-sonnet-4-20250514",
        "session_cost": 0.0,
        "rates": {"input": "$3.00/MTok", "output": "$15.00/MTok"},
    }
    bridge.get_session_cost.return_value = 0.0
    bridge.get_last_call_cost.return_value = 0.01
    bridge.chat = AsyncMock(return_value="Hello from Mother")
    bridge.list_tools = AsyncMock(return_value=[])
    bridge.get_active_appendages = AsyncMock(return_value=[])
    bridge.get_active_goals = AsyncMock(return_value=[])
    bridge.add_goal = AsyncMock(return_value={"id": 1, "description": "test"})
    bridge.search_files = AsyncMock(return_value=[])
    bridge.read_file = AsyncMock(return_value={"content": "hello", "path": "/tmp/test"})
    bridge.compile = AsyncMock()

    # Streaming
    bridge.begin_chat_stream = MagicMock()
    bridge.stream_chat = AsyncMock()
    bridge.cancel_chat_stream = MagicMock()
    bridge.get_stream_result.return_value = "streamed response"

    return bridge


@pytest.fixture
def auth_token():
    return "test_token_abc123"


@pytest.fixture
def app(mock_bridge, auth_token):
    """Create test app with mock bridge and known auth token."""
    return create_app(bridge=mock_bridge, auth_token=auth_token)


@pytest.fixture
def client(app):
    return TestClient(app)


def auth_header(token="test_token_abc123"):
    return {"Authorization": f"Bearer {token}"}


# --- Health (no auth required) ---

class TestHealth:
    def test_health_returns_alive(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["alive"] is True
        assert "uptime_s" in data

    def test_health_no_auth_needed(self, client):
        r = client.get("/health")
        assert r.status_code == 200


# --- Auth ---

class TestAuth:
    def test_status_requires_auth(self, client):
        r = client.get("/status")
        assert r.status_code == 401

    def test_status_wrong_token(self, client):
        r = client.get("/status", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_status_correct_token(self, client):
        r = client.get("/status", headers=auth_header())
        assert r.status_code == 200

    def test_tools_requires_auth(self, client):
        r = client.get("/tools")
        assert r.status_code == 401


# --- Status ---

class TestStatus:
    def test_returns_provider_info(self, client):
        r = client.get("/status", headers=auth_header())
        data = r.json()
        assert data["provider"] == "claude"
        assert "session_cost" in data


# --- Config ---

class TestConfig:
    def test_get_config(self, client):
        r = client.get("/config", headers=auth_header())
        assert r.status_code == 200
        data = r.json()
        assert "provider" in data

    def test_post_config(self, client):
        with patch("mother.panel_server.save_config") as mock_save:
            r = client.post(
                "/config",
                json={"voice_enabled": True},
                headers=auth_header(),
            )
            assert r.status_code == 200
            assert r.json()["ok"] is True


# --- Chat ---

class TestChat:
    def test_post_chat(self, client, mock_bridge):
        r = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hi"}], "system_prompt": "be nice"},
            headers=auth_header(),
        )
        assert r.status_code == 200
        data = r.json()
        assert data["text"] == "Hello from Mother"
        assert "cost" in data


# --- Tools ---

class TestTools:
    def test_get_tools(self, client, mock_bridge):
        r = client.get("/tools", headers=auth_header())
        assert r.status_code == 200
        assert r.json() == []


# --- Files ---

class TestFiles:
    def test_file_search(self, client, mock_bridge):
        r = client.post(
            "/file/search",
            json={"query": "test.py", "path": "/tmp"},
            headers=auth_header(),
        )
        assert r.status_code == 200

    def test_file_read(self, client, mock_bridge):
        r = client.post(
            "/file/read",
            json={"path": "/tmp/test.txt"},
            headers=auth_header(),
        )
        assert r.status_code == 200
        assert r.json()["content"] == "hello"


# --- Goals ---

class TestGoals:
    def test_get_goals(self, client, mock_bridge):
        r = client.get("/goals", headers=auth_header())
        assert r.status_code == 200

    def test_post_goal(self, client, mock_bridge):
        r = client.post(
            "/goal",
            json={"description": "build something cool"},
            headers=auth_header(),
        )
        assert r.status_code == 200
        assert r.json()["id"] == 1


# --- WebSocket ---

class TestWebSocket:
    def test_ws_requires_auth(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "type": "auth", "id": "", "payload": {"token": "wrong"},
            }))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"

    def test_ws_connect_and_ready(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "type": "auth", "id": "", "payload": {"token": "test_token_abc123"},
            }))
            data = json.loads(ws.receive_text())
            assert data["type"] == "ready"
            assert data["payload"]["version"] == "1.0"

    def test_ws_subscribe(self, client):
        with client.websocket_connect("/ws") as ws:
            # Auth
            ws.send_text(json.dumps({
                "type": "auth", "id": "", "payload": {"token": "test_token_abc123"},
            }))
            ws.receive_text()  # ready

            # Subscribe
            ws.send_text(json.dumps({
                "type": "subscribe", "id": "s1",
                "payload": {"channels": ["senses", "perception"]},
            }))
            # No error response expected — subscribe is silent

    def test_ws_unknown_type(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({
                "type": "auth", "id": "", "payload": {"token": "test_token_abc123"},
            }))
            ws.receive_text()  # ready

            ws.send_text(json.dumps({
                "type": "bogus", "id": "b1", "payload": {},
            }))
            data = json.loads(ws.receive_text())
            assert data["type"] == "error"
            assert "Unknown" in data["payload"]["message"]


# --- PID lockfile ---

class TestPIDLock:
    def test_acquire_and_release(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        assert acquire_lock(pid_path) is True
        assert pid_path.exists()
        assert int(pid_path.read_text()) == os.getpid()

        release_lock(pid_path)
        assert not pid_path.exists()

    def test_acquire_stale_lock(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        # Write a PID that doesn't exist
        pid_path.write_text("999999999")
        assert acquire_lock(pid_path) is True
        assert int(pid_path.read_text()) == os.getpid()

    def test_acquire_live_lock(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        # Write our own PID (it's alive)
        pid_path.write_text(str(os.getpid()))
        assert acquire_lock(pid_path) is False

    def test_release_wrong_pid(self, tmp_path):
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("999999999")
        release_lock(pid_path)
        # Should not have been deleted (different PID)
        assert pid_path.exists()
