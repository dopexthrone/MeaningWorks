"""Tests for EngineBridge.run_self_test() — boot-time self-test."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from mother.bridge import EngineBridge


@pytest.fixture
def bridge():
    return EngineBridge(provider="claude")


class TestRunSelfTest:

    def test_pass(self, bridge):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"42 passed in 3.2s\n", None))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(bridge.run_self_test())

        assert result["passed"] is True
        assert "42 passed" in result["summary"]
        assert result["duration_seconds"] >= 0

    def test_fail(self, bridge):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"FAILED tests/test_foo.py\n1 failed\n", None))
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(bridge.run_self_test())

        assert result["passed"] is False
        assert "1 failed" in result["summary"]

    def test_timeout(self, bridge):
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = asyncio.run(bridge.run_self_test())

        assert result["passed"] is False
        assert result["summary"] == "timeout"

    def test_error(self, bridge):
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError("no pytest")):
            result = asyncio.run(bridge.run_self_test())

        assert result["passed"] is False
        assert "error" in result["summary"]
