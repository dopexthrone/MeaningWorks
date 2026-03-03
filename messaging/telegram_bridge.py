"""Telegram bridge — long-polling Bot API adapter.

Uses httpx for HTTP calls to the Telegram Bot API. No telegram library dependency.
Implements long-polling via getUpdates with timeout.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from messaging.bridge import MessageBridge

logger = logging.getLogger(__name__)


class TelegramBridge(MessageBridge):
    """Telegram Bot API bridge using long-polling.

    Requires a bot token from @BotFather.
    Translates Telegram messages to runtime TCP messages and back.
    """

    API_BASE = "https://api.telegram.org/bot{token}"

    def __init__(
        self,
        token: str,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 8080,
        default_target: str = "Chat Agent",
        poll_timeout: int = 30,
    ):
        super().__init__(tcp_host, tcp_port, default_target)
        self.token = token
        self.api_url = self.API_BASE.format(token=token)
        self.poll_timeout = poll_timeout
        self._offset = 0
        self._running = False

    async def _api_call(self, method: str, **params) -> Dict[str, Any]:
        """Make a Telegram Bot API call."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError("httpx required: pip install httpx")

        url = f"{self.api_url}/{method}"
        async with httpx.AsyncClient(timeout=self.poll_timeout + 10) as client:
            resp = await client.post(url, json=params)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
            return data.get("result", {})

    async def _send_message(self, chat_id: int, text: str) -> None:
        """Send a text message to a Telegram chat."""
        # Telegram has a 4096 char limit
        if len(text) > 4096:
            text = text[:4090] + "\n..."
        await self._api_call("sendMessage", chat_id=chat_id, text=text)

    async def _poll_updates(self):
        """Long-poll for new updates."""
        try:
            updates = await self._api_call(
                "getUpdates",
                offset=self._offset,
                timeout=self.poll_timeout,
            )
            return updates if isinstance(updates, list) else []
        except Exception as e:
            logger.warning("Poll failed: %s", e)
            await asyncio.sleep(2)
            return []

    async def start(self) -> None:
        """Start the Telegram bridge — connect to runtime and poll for messages."""
        await self.tcp.connect()
        self._running = True
        logger.info("Telegram bridge started")

        while self._running:
            updates = await self._poll_updates()
            for update in updates:
                self._offset = update.get("update_id", 0) + 1
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = message.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue

                logger.info("Telegram message from %s: %s", chat_id, text[:50])
                response = await self.handle_message(text)
                await self._send_message(chat_id, response)

    async def stop(self) -> None:
        """Stop the Telegram bridge."""
        self._running = False
        await self.tcp.disconnect()
        logger.info("Telegram bridge stopped")
