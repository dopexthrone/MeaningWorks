"""Discord /status and /help slash commands."""

import logging

import discord
from discord import app_commands
from discord.ext import commands

from bots.shared.api_client import MotherlabsAPIClient

logger = logging.getLogger("motherlabs.bots.discord.info")


class InfoCog(commands.Cog):
    """Informational slash commands."""

    def __init__(self, bot: commands.Bot, api_client: MotherlabsAPIClient):
        self.bot = bot
        self.api = api_client

    @app_commands.command(name="status", description="Check Motherlabs API status")
    async def status(self, interaction: discord.Interaction):
        try:
            health = await self.api.health()
            embed = discord.Embed(
                title="Motherlabs Status",
                color=0x22C55E if health.get("status") == "ok" else 0xEF4444,
            )
            embed.add_field(name="Status", value=health.get("status", "unknown"), inline=True)
            embed.add_field(
                name="Domains",
                value=", ".join(health.get("domains_available", [])) or "none",
                inline=True,
            )
            embed.add_field(
                name="Queue Depth",
                value=str(health.get("worker_queue_depth", 0)),
                inline=True,
            )
            await interaction.response.send_message(embed=embed)
        except Exception as e:
            await interaction.response.send_message(f"API unreachable: {e}", ephemeral=True)

    @app_commands.command(name="help", description="Show Motherlabs bot commands")
    async def help_cmd(self, interaction: discord.Interaction):
        embed = discord.Embed(
            title="Motherlabs",
            description="Semantic compiler — natural language to verified specifications.",
            color=0x7C3AED,
        )
        embed.add_field(
            name="/compile <description>",
            value="Compile your idea into a specification with trust scores",
            inline=False,
        )
        embed.add_field(name="/status", value="Check API status", inline=False)
        embed.add_field(name="/help", value="Show this message", inline=False)
        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot, api_client: MotherlabsAPIClient):
    await bot.add_cog(InfoCog(bot, api_client))
