"""Tests for messaging bridge — protocol translation and TCP client."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from messaging.bridge import TCPClient, MessageBridge


# =============================================================================
# CONCRETE BRIDGE FOR TESTING
# =============================================================================

class StubBridge(MessageBridge):
    """Concrete bridge subclass for testing abstract base."""

    async def start(self):
        pass

    async def stop(self):
        pass


# =============================================================================
# MESSAGE TRANSLATION
# =============================================================================

class TestTranslateToRuntime:
    """Tests for translate_to_runtime() — platform message → runtime JSON."""

    def test_plain_message(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("Hello, how are you?")
        assert msg["target"] == "Chat Agent"
        assert msg["payload"]["action"] == "chat"
        assert msg["payload"]["message"] == "Hello, how are you?"

    def test_learn_command(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/learn check the weather")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "learn"
        assert msg["payload"]["skill"] == "check the weather"

    def test_status_command(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/status")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "status"

    def test_health_command(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/health")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "health"

    def test_custom_default_target(self):
        bridge = StubBridge(default_target="Support Bot")
        msg = bridge.translate_to_runtime("help me")
        assert msg["target"] == "Support Bot"

    def test_whitespace_stripped(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("  /status  ")
        assert msg["target"] == "_system"

    def test_learn_with_whitespace(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/learn   summarize articles  ")
        assert msg["payload"]["skill"] == "summarize articles"


# =============================================================================
# RESPONSE TRANSLATION
# =============================================================================

class TestTranslateFromRuntime:
    """Tests for translate_from_runtime() — runtime response → human text."""

    def test_error_response(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"error": "Unknown component"})
        assert "Error" in text
        assert "Unknown component" in text

    def test_content_response(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"content": "Hello!"})
        assert text == "Hello!"

    def test_response_key(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"response": "Done"})
        assert text == "Done"

    def test_components_list(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"components": ["Chat Agent", "Search Tool"]})
        assert "Chat Agent" in text
        assert "Search Tool" in text

    def test_health_response(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"uptime": 3600.5, "messages": 42})
        assert "3600" in text
        assert "42" in text

    def test_generic_json(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"custom": "data"})
        assert "custom" in text


# =============================================================================
# TCP CLIENT
# =============================================================================

class TestTCPClient:
    """Tests for TCPClient — connection and message sending."""

    def test_defaults(self):
        client = TCPClient()
        assert client.host == "127.0.0.1"
        assert client.port == 8080
        assert not client.connected

    def test_custom_address(self):
        client = TCPClient(host="10.0.0.1", port=9090)
        assert client.host == "10.0.0.1"
        assert client.port == 9090

    def test_send_requires_connection(self):
        async def _run():
            client = TCPClient()
            with pytest.raises(ConnectionError, match="Not connected"):
                await client.send("target", {"action": "chat"})
        asyncio.run(_run())


# =============================================================================
# HANDLE MESSAGE — end-to-end with mock TCP
# =============================================================================

class TestHandleMessage:
    """Test handle_message() with mocked TCP."""

    def test_chat_message_flow(self):
        async def _run():
            bridge = StubBridge()
            bridge.tcp.send = AsyncMock(return_value={"content": "Hi there!"})
            result = await bridge.handle_message("Hello")
            assert result == "Hi there!"
            bridge.tcp.send.assert_called_once_with("Chat Agent", {"action": "chat", "message": "Hello"})
        asyncio.run(_run())

    def test_system_command_flow(self):
        async def _run():
            bridge = StubBridge()
            bridge.tcp.send = AsyncMock(return_value={"components": ["A", "B"]})
            result = await bridge.handle_message("/status")
            assert "A" in result
            bridge.tcp.send.assert_called_once_with("_system", {"action": "status"})
        asyncio.run(_run())

    def test_connection_error_handled(self):
        async def _run():
            bridge = StubBridge()
            bridge.tcp.send = AsyncMock(side_effect=ConnectionError("refused"))
            result = await bridge.handle_message("hello")
            assert "Cannot reach agent" in result
        asyncio.run(_run())


# =============================================================================
# MOTHER AGENT COMMANDS
# =============================================================================

class TestMotherAgentCommands:
    """Tests for /compile, /tools, /instance commands."""

    def test_compile_command(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/compile a simple calculator")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "compile"
        assert msg["payload"]["skill"] == "a simple calculator"

    def test_compile_with_domain(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/compile:api a REST endpoint for users")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "compile"
        assert msg["payload"]["domain"] == "api"
        assert msg["payload"]["skill"] == "a REST endpoint for users"

    def test_tools_list(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/tools")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "tools"
        assert msg["payload"]["subaction"] == "list"

    def test_tools_search(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/tools search calculator")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "tools"
        assert msg["payload"]["subaction"] == "search"
        assert msg["payload"]["query"] == "calculator"

    def test_tools_export(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/tools export abc-123")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "tools"
        assert msg["payload"]["subaction"] == "export"
        assert msg["payload"]["compilation_id"] == "abc-123"

    def test_tools_import(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/tools import /tmp/tool.mtool")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "tools"
        assert msg["payload"]["subaction"] == "import"
        assert msg["payload"]["file_path"] == "/tmp/tool.mtool"

    def test_instance_command(self):
        bridge = StubBridge()
        msg = bridge.translate_to_runtime("/instance")
        assert msg["target"] == "_system"
        assert msg["payload"]["action"] == "instance"


# =============================================================================
# MOTHER AGENT RESPONSE FORMATTING
# =============================================================================

class TestMotherAgentResponseFormatting:
    """Tests for translate_from_runtime with mother agent responses."""

    def test_compilation_success(self):
        bridge = StubBridge()
        response = {
            "compilation_id": "abc-123",
            "status": "compiled",
            "trust_score": 85.5,
            "verification_badge": "verified",
            "component_count": 3,
        }
        text = bridge.translate_from_runtime(response)
        assert "abc-123" in text
        assert "85.5" in text
        assert "verified" in text

    def test_compilation_failed_with_error_key(self):
        """When 'error' key is present, error branch takes precedence."""
        bridge = StubBridge()
        response = {"compilation_id": "x", "status": "failed", "error": "timeout"}
        text = bridge.translate_from_runtime(response)
        assert "Error" in text
        assert "timeout" in text

    def test_compilation_failed_without_error_key(self):
        """When only status=failed, uses compilation result formatting."""
        bridge = StubBridge()
        response = {"compilation_id": "x", "status": "failed"}
        text = bridge.translate_from_runtime(response)
        assert "Failed" in text

    def test_tool_list_formatting(self):
        bridge = StubBridge()
        response = {
            "tools": [
                {"name": "calculator", "domain": "software", "trust_score": 90.0},
                {"name": "converter", "domain": "software", "trust_score": 75.0},
            ]
        }
        text = bridge.translate_from_runtime(response)
        assert "calculator" in text
        assert "converter" in text

    def test_empty_tool_list(self):
        bridge = StubBridge()
        text = bridge.translate_from_runtime({"tools": []})
        assert "No tools found" in text

    def test_instance_info_formatting(self):
        bridge = StubBridge()
        response = {
            "instance_id": "inst-001",
            "tool_count": 5,
            "corpus_path": "/some/path",
            "status": "active",
        }
        text = bridge.translate_from_runtime(response)
        assert "inst-001" in text
        assert "5" in text

    def test_export_success(self):
        bridge = StubBridge()
        response = {"compilation_id": "x", "status": "exported", "file_path": "/tmp/x.mtool"}
        text = bridge.translate_from_runtime(response)
        assert "Exported" in text
        assert "/tmp/x.mtool" in text

    def test_import_success(self):
        bridge = StubBridge()
        response = {"compilation_id": "x", "status": "imported", "file_path": "/tmp/x.mtool"}
        text = bridge.translate_from_runtime(response)
        assert "Imported" in text

    def test_import_rejected(self):
        bridge = StubBridge()
        response = {"compilation_id": "x", "status": "rejected", "reason": "low trust"}
        text = bridge.translate_from_runtime(response)
        assert "rejected" in text
