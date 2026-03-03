"""Telegram command handlers.

Commands: /start, /compile <description>, /status, /help
"""

import logging
from typing import Optional

from bots.shared.api_client import MotherlabsAPIClient
from bots.shared.user_state import UserStateStore
from bots.shared.formatter import format_result, format_as_text

logger = logging.getLogger("motherlabs.bots.telegram")


class TelegramHandlers:
    """Stateless command handler — delegates to API client and user state."""

    def __init__(self, api_client: MotherlabsAPIClient, user_state: UserStateStore):
        self.api = api_client
        self.state = user_state

    async def handle_start(self, chat_id: int) -> str:
        return (
            "Welcome to Motherlabs.\n\n"
            "I compile natural language into verified software specifications.\n\n"
            "Commands:\n"
            "/compile <description> — Compile your idea\n"
            "/status — Check API status\n"
            "/help — Show this message"
        )

    async def handle_help(self, chat_id: int) -> str:
        return await self.handle_start(chat_id)

    async def handle_status(self, chat_id: int) -> str:
        try:
            health = await self.api.health()
            status = health.get("status", "unknown")
            domains = ", ".join(health.get("domains_available", []))
            queue = health.get("worker_queue_depth", 0)
            return f"Status: {status}\nDomains: {domains}\nQueue depth: {queue}"
        except Exception as e:
            return f"API unreachable: {e}"

    async def handle_compile(
        self, chat_id: int, user_id: str, description: str, send_typing: callable
    ) -> str:
        """Run a compilation. send_typing is called every 5s during compilation."""
        if not description.strip():
            return "Usage: /compile <description of what you want to build>"

        # Concurrency gate
        if not self.state.try_start_compilation(user_id, "telegram", ""):
            return "You already have a compilation running. Please wait for it to finish."

        try:
            result = await self.api.compile(description=description)
            self.state.finish_compilation(user_id, "telegram", "done")
            formatted = format_result(result)
            return format_as_text(formatted)
        except Exception as e:
            self.state.finish_compilation(user_id, "telegram", "failed")
            logger.exception("Compilation failed for user %s", user_id)
            return f"Compilation failed: {e}"
