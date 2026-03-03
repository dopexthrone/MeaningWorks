"""Discord /compile slash command — deferred response with threaded results."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from bots.shared.api_client import MotherlabsAPIClient
from bots.shared.user_state import UserStateStore
from bots.shared.formatter import format_result
from bots.discord.formatter import build_embed

logger = logging.getLogger("motherlabs.bots.discord.compile")


class CompileCog(commands.Cog):
    """Slash commands for compilation."""

    def __init__(self, bot: commands.Bot, api_client: MotherlabsAPIClient, user_state: UserStateStore):
        self.bot = bot
        self.api = api_client
        self.state = user_state

    @app_commands.command(name="compile", description="Compile a natural language description into a specification")
    @app_commands.describe(description="What do you want to build?")
    async def compile(self, interaction: discord.Interaction, description: str):
        user_id = str(interaction.user.id)

        # Concurrency gate
        if not self.state.try_start_compilation(user_id, "discord", ""):
            await interaction.response.send_message(
                "You already have a compilation running. Please wait for it to finish.",
                ephemeral=True,
            )
            return

        # Defer — compilation takes 30-120s
        await interaction.response.defer(thinking=True)

        try:
            result = await self.api.compile(description=description)
            self.state.finish_compilation(user_id, "discord", "done")

            formatted = format_result(result)
            embed_data = build_embed(formatted)
            embed = discord.Embed.from_dict(embed_data)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            self.state.finish_compilation(user_id, "discord", "failed")
            logger.exception("Compilation failed for user %s", user_id)
            await interaction.followup.send(f"Compilation failed: {e}")


async def setup(bot: commands.Bot, api_client: MotherlabsAPIClient, user_state: UserStateStore):
    await bot.add_cog(CompileCog(bot, api_client, user_state))
