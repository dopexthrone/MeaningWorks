"""Discord bridge — Gateway websocket adapter.

Uses websockets for Discord Gateway connection and httpx for REST API calls.
No discord.py dependency. Implements the minimal Gateway handshake for message events.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from messaging.bridge import MessageBridge

logger = logging.getLogger(__name__)

# Discord Gateway opcodes
DISPATCH = 0
HEARTBEAT = 1
IDENTIFY = 2
HELLO = 10
HEARTBEAT_ACK = 11

GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"


class DiscordBridge(MessageBridge):
    """Discord Gateway bridge using websockets.

    Requires a bot token from Discord Developer Portal.
    Implements minimal Gateway v10 protocol: identify, heartbeat, message dispatch.
    """

    API_BASE = "https://discord.com/api/v10"

    def __init__(
        self,
        token: str,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 8080,
        default_target: str = "Chat Agent",
    ):
        super().__init__(tcp_host, tcp_port, default_target)
        self.token = token
        self._ws = None
        self._running = False
        self._heartbeat_interval = 45.0
        self._sequence: Optional[int] = None
        self._session_id: Optional[str] = None

    async def _send_ws(self, op: int, d: Any = None) -> None:
        """Send a Gateway payload."""
        if self._ws:
            await self._ws.send(json.dumps({"op": op, "d": d}))

    async def _heartbeat_loop(self) -> None:
        """Send heartbeats at the interval specified by HELLO."""
        while self._running:
            await asyncio.sleep(self._heartbeat_interval)
            await self._send_ws(HEARTBEAT, self._sequence)

    async def _identify(self) -> None:
        """Send IDENTIFY payload to authenticate."""
        await self._send_ws(IDENTIFY, {
            "token": self.token,
            "intents": 512 | 32768,  # GUILD_MESSAGES | MESSAGE_CONTENT
            "properties": {
                "os": "linux",
                "browser": "motherlabs",
                "device": "motherlabs",
            },
        })

    async def _send_channel_message(self, channel_id: str, content: str) -> None:
        """Send a message to a Discord channel via REST API."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required: pip install httpx")

        if len(content) > 2000:
            content = content[:1995] + "\n..."

        url = f"{self.API_BASE}/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self.token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(url, headers=headers, json={"content": content})

    async def start(self) -> None:
        """Start the Discord bridge — connect to Gateway and runtime."""
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets required: pip install websockets")

        await self.tcp.connect()
        self._running = True
        logger.info("Discord bridge starting")

        async with websockets.connect(GATEWAY_URL) as ws:
            self._ws = ws

            # Wait for HELLO
            hello = json.loads(await ws.recv())
            if hello.get("op") == HELLO:
                self._heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000.0
                logger.info("Gateway HELLO, heartbeat_interval=%.1fs", self._heartbeat_interval)

            # Start heartbeat
            hb_task = asyncio.create_task(self._heartbeat_loop())

            # Identify
            await self._identify()

            try:
                async for raw_msg in ws:
                    if not self._running:
                        break
                    payload = json.loads(raw_msg)
                    op = payload.get("op")
                    seq = payload.get("s")
                    if seq is not None:
                        self._sequence = seq

                    if op == DISPATCH:
                        event = payload.get("t")
                        data = payload.get("d", {})
                        if event == "READY":
                            self._session_id = data.get("session_id")
                            logger.info("Discord READY, session=%s", self._session_id)
                        elif event == "MESSAGE_CREATE":
                            await self._handle_discord_message(data)
                    elif op == HEARTBEAT_ACK:
                        pass  # Expected
            finally:
                hb_task.cancel()
                self._ws = None

    async def _handle_discord_message(self, data: Dict[str, Any]) -> None:
        """Process an incoming Discord message."""
        # Ignore bot messages
        if data.get("author", {}).get("bot"):
            return

        content = data.get("content", "")
        channel_id = data.get("channel_id")
        if not content or not channel_id:
            return

        logger.info("Discord message in %s: %s", channel_id, content[:50])
        response = await self.handle_message(content)
        await self._send_channel_message(channel_id, response)

    async def stop(self) -> None:
        """Stop the Discord bridge."""
        self._running = False
        if self._ws:
            await self._ws.close()
        await self.tcp.disconnect()
        logger.info("Discord bridge stopped")
