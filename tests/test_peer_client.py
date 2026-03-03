"""
Tests for PeerClient — async HTTP client for remote Mother instances.

All tests mock httpx — no actual remote server needed.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# --- Health Check ---

class TestPeerClientHealthCheck:
    """Test PeerClient.health_check() method."""

    def test_health_check_reachable(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.status_code = 200

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            status = await client.health_check("test-peer-id")
            assert status.reachable is True
            assert status.instance_id == "test-peer-id"
            assert status.latency_ms > 0
            mock_client.get.assert_called_once_with("/v1/health")
        asyncio.run(_run())

    def test_health_check_unreachable(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            status = await client.health_check("test-peer-id")
            assert status.reachable is False
            assert "Connection refused" in status.error
        asyncio.run(_run())

    def test_health_check_bad_status(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.status_code = 503

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            status = await client.health_check("test-peer-id")
            assert status.reachable is False
            assert "503" in status.error
        asyncio.run(_run())


# --- Digest ---

class TestPeerClientDigest:
    """Test PeerClient.get_digest() method."""

    def test_get_digest(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "instance_id": "abc123",
                "tool_count": 5,
                "verified_tool_count": 3,
            }
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            digest = await client.get_digest()
            assert digest["instance_id"] == "abc123"
            assert digest["tool_count"] == 5
        asyncio.run(_run())


# --- List Tools ---

class TestPeerClientListTools:
    """Test PeerClient.list_tools() method."""

    def test_list_tools(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "tools": [
                    {"package_id": "t1", "name": "weather", "domain": "api"},
                    {"package_id": "t2", "name": "counter", "domain": "software"},
                ]
            }
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            tools = await client.list_tools()
            assert len(tools) == 2
            assert tools[0]["name"] == "weather"
        asyncio.run(_run())

    def test_list_tools_with_domain_filter(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"tools": []}
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            await client.list_tools(domain="api")
            mock_client.get.assert_called_with("/v2/tools/export", params={"domain": "api"})
        asyncio.run(_run())


# --- Search Tools ---

class TestPeerClientSearchTools:
    """Test PeerClient.search_tools() method."""

    def test_search_tools(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "results": [{"package_id": "t1", "name": "weather"}]
            }
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            results = await client.search_tools("weather")
            assert len(results) == 1
            mock_client.get.assert_called_with("/v2/tools/search", params={"q": "weather"})
        asyncio.run(_run())


# --- Get Tool ---

class TestPeerClientGetTool:
    """Test PeerClient.get_tool() method."""

    def test_get_tool(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "package_id": "pkg-abc",
                "name": "weather-api",
                "domain": "api",
                "trust_score": 85.0,
            }
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            tool = await client.get_tool("pkg-abc")
            assert tool["package_id"] == "pkg-abc"
            assert tool["trust_score"] == 85.0
        asyncio.run(_run())


# --- Register Self ---

class TestPeerClientRegisterSelf:
    """Test PeerClient.register_self() method."""

    def test_register_success(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_resp = MagicMock()
            mock_resp.status_code = 200

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            result = await client.register_self("my-id", "Ubuntu Mother", "http://me:8000")
            assert result is True
        asyncio.run(_run())

    def test_register_failure(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            result = await client.register_self("my-id", "Ubuntu Mother", "http://me:8000")
            assert result is False
        asyncio.run(_run())


# --- Close ---

class TestPeerClientClose:
    """Test PeerClient.close() method."""

    def test_close(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()

            client = PeerClient("http://100.1.2.3:8000")
            client._client = mock_client

            await client.close()
            mock_client.aclose.assert_called_once()
            assert client._client is None
        asyncio.run(_run())

    def test_close_when_not_connected(self):
        async def _run():
            from motherlabs_platform.peer_client import PeerClient

            client = PeerClient("http://100.1.2.3:8000")
            # Should not raise
            await client.close()
        asyncio.run(_run())


# --- PeerStatus ---

class TestPeerStatus:
    """Test PeerStatus frozen dataclass."""

    def test_peer_status_creation(self):
        from motherlabs_platform.peer_client import PeerStatus

        status = PeerStatus(
            instance_id="abc",
            reachable=True,
            latency_ms=42.5,
            last_checked="2026-02-15T10:00:00",
            tool_count=5,
        )
        assert status.instance_id == "abc"
        assert status.reachable is True
        assert status.tool_count == 5

    def test_peer_status_defaults(self):
        from motherlabs_platform.peer_client import PeerStatus

        status = PeerStatus(
            instance_id="abc",
            reachable=False,
            latency_ms=0.0,
            last_checked="now",
        )
        assert status.tool_count == 0
        assert status.error == ""
