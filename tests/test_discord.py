"""Tests for mother/discord.py — Discord webhook integration."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mother.discord import (
    DiscordResult,
    build_embed,
    format_build_announcement,
    post_webhook,
)


# --- build_embed ---


class TestBuildEmbed:
    def test_basic_embed(self):
        embed = build_embed("Title", "Description")
        assert embed["title"] == "Title"
        assert embed["description"] == "Description"
        assert embed["color"] == 0x7C3AED

    def test_with_url(self):
        embed = build_embed("T", "D", url="https://example.com")
        assert embed["url"] == "https://example.com"

    def test_custom_color(self):
        embed = build_embed("T", "D", color=0xFF0000)
        assert embed["color"] == 0xFF0000

    def test_with_fields(self):
        fields = [
            {"name": "Field1", "value": "Value1", "inline": True},
            {"name": "Field2", "value": "Value2"},
        ]
        embed = build_embed("T", "D", fields=fields)
        assert len(embed["fields"]) == 2
        assert embed["fields"][0]["name"] == "Field1"
        assert embed["fields"][0]["inline"] is True
        assert embed["fields"][1]["inline"] is False  # default

    def test_title_truncated(self):
        embed = build_embed("x" * 300, "D")
        assert len(embed["title"]) == 256

    def test_description_truncated(self):
        embed = build_embed("T", "x" * 5000)
        assert len(embed["description"]) == 4096

    def test_fields_capped_at_25(self):
        fields = [{"name": f"F{i}", "value": f"V{i}"} for i in range(30)]
        embed = build_embed("T", "D", fields=fields)
        assert len(embed["fields"]) == 25

    def test_no_url_no_fields(self):
        embed = build_embed("T", "D")
        assert "url" not in embed
        assert "fields" not in embed


# --- format_build_announcement ---


class TestFormatBuildAnnouncement:
    def test_basic_announcement(self):
        embed = format_build_announcement("MyApp", "A cool app")
        assert embed["title"] == "New Build: MyApp"
        assert "A cool app" in embed["description"]

    def test_with_metrics(self):
        embed = format_build_announcement(
            "App", "Desc", repo_url="https://github.com/x/y",
            components=15, trust=92.5,
        )
        assert embed["url"] == "https://github.com/x/y"
        assert any(f["name"] == "Components" for f in embed["fields"])
        assert any(f["name"] == "Trust" for f in embed["fields"])

    def test_no_metrics(self):
        embed = format_build_announcement("App", "Desc")
        assert "fields" not in embed or embed.get("fields") is None

    def test_empty_description(self):
        embed = format_build_announcement("App", "")
        assert "built by Mother" in embed["description"]


# --- post_webhook ---


class TestPostWebhook:
    def test_no_url(self):
        result = post_webhook("", content="hello")
        assert result.success is False
        assert "No webhook URL" in result.error

    def test_no_content_no_embed(self):
        result = post_webhook("https://example.com/webhook", content="", embed=None)
        assert result.success is False
        assert "Must provide" in result.error

    @patch("mother.discord.urllib.request.urlopen")
    def test_happy_path_204(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""
        mock_urlopen.return_value = mock_response

        result = post_webhook(
            "https://discord.com/api/webhooks/123/abc",
            content="Hello Discord!",
        )
        assert result.success is True
        assert result.operation == "post"

    @patch("mother.discord.urllib.request.urlopen")
    def test_happy_path_200_with_id(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"id": "msg123"}).encode()
        mock_urlopen.return_value = mock_response

        result = post_webhook(
            "https://discord.com/api/webhooks/123/abc",
            content="Hello",
        )
        assert result.success is True
        assert result.message_id == "msg123"

    @patch("mother.discord.urllib.request.urlopen")
    def test_with_embed(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""
        mock_urlopen.return_value = mock_response

        embed = build_embed("Test", "Embed content")
        result = post_webhook(
            "https://discord.com/api/webhooks/123/abc",
            embed=embed,
        )
        assert result.success is True

        # Verify the request payload included embeds
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert "embeds" in body

    @patch("mother.discord.urllib.request.urlopen")
    def test_content_truncated(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 204
        mock_response.read.return_value = b""
        mock_urlopen.return_value = mock_response

        long_content = "x" * 3000
        post_webhook("https://example.com/webhook", content=long_content)

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        assert len(body["content"]) <= 2000
        assert body["content"].endswith("...")

    @patch("mother.discord.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 429, "Too Many Requests", {}, None
        )
        result = post_webhook("https://example.com/webhook", content="hi")
        assert result.success is False
        assert "429" in result.error

    @patch("mother.discord.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = post_webhook("https://example.com/webhook", content="hi", timeout=5.0)
        assert result.success is False
        assert "Timed out" in result.error

    @patch("mother.discord.urllib.request.urlopen")
    def test_url_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("DNS failed")
        result = post_webhook("https://example.com/webhook", content="hi")
        assert result.success is False
        assert "URL error" in result.error

    def test_custom_username(self):
        with patch("mother.discord.urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 204
            mock_response.read.return_value = b""
            mock_urlopen.return_value = mock_response

            post_webhook("https://example.com/webhook", content="hi", username="Bot")

            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            body = json.loads(req.data.decode("utf-8"))
            assert body["username"] == "Bot"

    def test_result_frozen(self):
        r = DiscordResult(success=True)
        with pytest.raises(AttributeError):
            r.success = False
