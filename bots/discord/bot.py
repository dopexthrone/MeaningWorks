"""Discord bot — slash commands via discord.py.

Run as: python -m bots.discord.bot
"""

import logging
import os
import sys

import discord
from discord.ext import commands

from bots.shared.api_client import MotherlabsAPIClient
from bots.shared.user_state import UserStateStore
from bots.discord.cogs.compile import CompileCog
from bots.discord.cogs.info import InfoCog

logger = logging.getLogger("motherlabs.bots.discord")


class MotherlabsDiscordBot(commands.Bot):
    """Discord bot with slash commands for compilation."""

    def __init__(self, api_url: str):
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

        self.api_client = MotherlabsAPIClient(api_url=api_url)
        self.user_state = UserStateStore()

    async def setup_hook(self):
        """Register cogs and sync slash commands."""
        await self.add_cog(CompileCog(self, self.api_client, self.user_state))
        await self.add_cog(InfoCog(self, self.api_client))
        await self.tree.sync()
        logger.info("Slash commands synced")

    async def on_ready(self):
        logger.info("Discord bot ready as %s", self.user)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.error("DISCORD_BOT_TOKEN not set")
        sys.exit(1)

    api_url = os.environ.get("MOTHERLABS_API_URL", "http://api:8000")
    bot = MotherlabsDiscordBot(api_url=api_url)
    bot.run(token)


if __name__ == "__main__":
    main()
