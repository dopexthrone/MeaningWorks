"""
Discord webhook integration — post messages and embeds via webhook URL.

LEAF module. Stdlib only. No imports from core/ or mother/.

Discord webhooks need zero auth — just a URL. No SDK, no approval process.
POST JSON to the webhook URL and Discord delivers it.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("mother.discord")


@dataclass(frozen=True)
class DiscordResult:
    """Outcome of a Discord webhook operation."""

    success: bool
    operation: str = ""
    message_id: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


def build_embed(
    title: str,
    description: str,
    url: str = "",
    color: int = 0x7C3AED,
    fields: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """Construct a Discord embed dict.

    Args:
        title: Embed title
        description: Embed description (max 4096 chars)
        url: Optional URL the title links to
        color: Embed color as integer (default: purple)
        fields: Optional list of {"name": str, "value": str, "inline": bool}
    """
    embed: dict = {
        "title": title[:256],  # Discord limit
        "description": description[:4096],
        "color": color,
    }
    if url:
        embed["url"] = url
    if fields:
        embed["fields"] = [
            {
                "name": f.get("name", "")[:256],
                "value": f.get("value", "")[:1024],
                "inline": f.get("inline", False),
            }
            for f in fields[:25]  # Discord limit: 25 fields
        ]
    return embed


def format_build_announcement(
    name: str,
    description: str,
    repo_url: str = "",
    components: int = 0,
    trust: float = 0.0,
) -> dict:
    """Create a Discord embed dict for a build announcement."""
    fields = []
    if components > 0:
        fields.append({"name": "Components", "value": str(components), "inline": True})
    if trust > 0:
        fields.append({"name": "Trust", "value": f"{trust:.0f}%", "inline": True})

    return build_embed(
        title=f"New Build: {name}",
        description=description[:4096] if description else "A new project built by Mother.",
        url=repo_url,
        fields=fields if fields else None,
    )


def post_webhook(
    webhook_url: str,
    content: str = "",
    username: str = "Mother",
    embed: Optional[dict] = None,
    timeout: float = 10.0,
) -> DiscordResult:
    """Post a message to a Discord webhook.

    Args:
        webhook_url: The full Discord webhook URL
        content: Plain text content (max 2000 chars)
        username: Display name for the webhook message
        embed: Optional embed dict (from build_embed or format_build_announcement)
        timeout: Request timeout in seconds
    """
    start = time.monotonic()

    if not webhook_url:
        return DiscordResult(
            success=False,
            operation="post",
            error="No webhook URL provided",
            duration_seconds=0.0,
        )

    if not content and not embed:
        return DiscordResult(
            success=False,
            operation="post",
            error="Must provide content or embed",
            duration_seconds=0.0,
        )

    # Enforce Discord limits
    if content and len(content) > 2000:
        content = content[:1997] + "..."

    payload: dict = {"username": username}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response = urllib.request.urlopen(req, timeout=timeout)
        elapsed = time.monotonic() - start

        # Discord returns 204 No Content on success, or 200 with ?wait=true
        status = response.status
        if status in (200, 204):
            # Try to extract message ID from response body (only present with ?wait=true)
            message_id = ""
            try:
                body = response.read().decode("utf-8")
                if body:
                    resp_data = json.loads(body)
                    message_id = resp_data.get("id", "")
            except Exception:
                pass

            return DiscordResult(
                success=True,
                operation="post",
                message_id=message_id,
                duration_seconds=elapsed,
            )

        return DiscordResult(
            success=False,
            operation="post",
            error=f"HTTP {status}",
            duration_seconds=elapsed,
        )

    except urllib.error.HTTPError as e:
        return DiscordResult(
            success=False,
            operation="post",
            error=f"HTTP {e.code}: {e.reason}",
            duration_seconds=time.monotonic() - start,
        )
    except urllib.error.URLError as e:
        return DiscordResult(
            success=False,
            operation="post",
            error=f"URL error: {e.reason}",
            duration_seconds=time.monotonic() - start,
        )
    except TimeoutError:
        return DiscordResult(
            success=False,
            operation="post",
            error=f"Timed out after {timeout:.0f}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as e:
        return DiscordResult(
            success=False,
            operation="post",
            error=str(e),
            duration_seconds=time.monotonic() - start,
        )
