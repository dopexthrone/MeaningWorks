"""Telegram bot — long-polling with raw httpx. No library dependency.

Reuses patterns from messaging/telegram_bridge.py.
Run as: python -m bots.telegram.bot
"""

import asyncio
import logging
import os
import sys

import httpx

from bots.shared.api_client import MotherlabsAPIClient
from bots.shared.user_state import UserStateStore
from bots.telegram.handlers import TelegramHandlers

logger = logging.getLogger("motherlabs.bots.telegram")

API_BASE = "https://api.telegram.org/bot{token}"
POLL_TIMEOUT = 30
TYPING_INTERVAL = 5


class TelegramBot:
    """Long-polling Telegram bot."""

    def __init__(self, token: str, api_url: str):
        self.token = token
        self.api_url_tg = API_BASE.format(token=token)
        self._offset = 0
        self._running = False

        api_client = MotherlabsAPIClient(api_url=api_url)
        user_state = UserStateStore()
        self.handlers = TelegramHandlers(api_client, user_state)

    async def _tg_call(self, method: str, **params) -> dict:
        """Make a Telegram Bot API call."""
        url = f"{self.api_url_tg}/{method}"
        async with httpx.AsyncClient(timeout=POLL_TIMEOUT + 10) as client:
            resp = await client.post(url, json=params)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
            return data.get("result", {})

    async def _send_message(self, chat_id: int, text: str, parse_mode: str = None):
        """Send a text message (4096 char limit)."""
        if len(text) > 4096:
            text = text[:4090] + "\n..."
        params = {"chat_id": chat_id, "text": text}
        if parse_mode:
            params["parse_mode"] = parse_mode
        await self._tg_call("sendMessage", **params)

    async def _send_typing(self, chat_id: int):
        """Send typing indicator."""
        try:
            await self._tg_call("sendChatAction", chat_id=chat_id, action="typing")
        except Exception:
            pass

    async def _typing_loop(self, chat_id: int, stop_event: asyncio.Event):
        """Send typing indicator every TYPING_INTERVAL seconds until stopped."""
        while not stop_event.is_set():
            await self._send_typing(chat_id)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=TYPING_INTERVAL)
            except asyncio.TimeoutError:
                pass

    async def _handle_update(self, update: dict):
        """Route a single update to the appropriate handler."""
        message = update.get("message", {})
        text = message.get("text", "").strip()
        chat_id = message.get("chat", {}).get("id")
        user_id = str(message.get("from", {}).get("id", ""))

        if not text or not chat_id:
            return

        # Route commands
        if text == "/start":
            reply = await self.handlers.handle_start(chat_id)
        elif text == "/help":
            reply = await self.handlers.handle_help(chat_id)
        elif text == "/status":
            reply = await self.handlers.handle_status(chat_id)
        elif text.startswith("/compile"):
            description = text[len("/compile"):].strip()
            # Start typing indicator
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_loop(chat_id, stop_typing))
            try:
                reply = await self.handlers.handle_compile(
                    chat_id, user_id, description, self._send_typing
                )
            finally:
                stop_typing.set()
                await typing_task
        else:
            # Treat bare text as implicit /compile
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_loop(chat_id, stop_typing))
            try:
                reply = await self.handlers.handle_compile(
                    chat_id, user_id, text, self._send_typing
                )
            finally:
                stop_typing.set()
                await typing_task

        await self._send_message(chat_id, reply)

    async def run(self):
        """Main polling loop."""
        self._running = True
        logger.info("Telegram bot started")

        while self._running:
            try:
                updates = await self._tg_call(
                    "getUpdates",
                    offset=self._offset,
                    timeout=POLL_TIMEOUT,
                )
                if not isinstance(updates, list):
                    updates = []

                for update in updates:
                    self._offset = update.get("update_id", 0) + 1
                    try:
                        await self._handle_update(update)
                    except Exception:
                        logger.exception("Error handling update %s", update.get("update_id"))

            except Exception:
                logger.exception("Poll error")
                await asyncio.sleep(2)

    def stop(self):
        self._running = False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    api_url = os.environ.get("MOTHERLABS_API_URL", "http://api:8000")
    bot = TelegramBot(token=token, api_url=api_url)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
