"""
Twitter/X integration — API client for posting and monitoring.

LEAF module. Uses stdlib + httpx (already a dependency via discord bridge).

Enables Mother to:
- Post tweets
- Read mentions/replies
- Build in public
- Monitor feedback

Uses Twitter API v2 with OAuth 2.0 bearer token authentication.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass(frozen=True)
class TweetResult:
    """Outcome of a Twitter operation."""

    success: bool
    operation: str = ""
    tweet_id: str = ""
    tweet_url: str = ""
    text: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


class TwitterClient:
    """
    Twitter API v2 client.

    Requires TWITTER_BEARER_TOKEN env var or explicit token.
    """

    API_BASE = "https://api.twitter.com/2"

    def __init__(self, bearer_token: Optional[str] = None):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required for Twitter integration")

        self.bearer_token = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN", "")
        if not self.bearer_token:
            raise ValueError("Twitter bearer token required (TWITTER_BEARER_TOKEN env var)")

        self.client = httpx.Client(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def post_tweet(self, text: str) -> TweetResult:
        """Post a tweet. Returns TweetResult with tweet_id and URL."""
        if not text or len(text) > 280:
            return TweetResult(
                success=False,
                operation="post",
                error=f"Tweet length must be 1-280 chars (got {len(text)})",
            )

        start = time.monotonic()

        try:
            response = self.client.post(
                "/tweets",
                json={"text": text},
            )

            elapsed = time.monotonic() - start

            if response.status_code not in (200, 201):
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", response.text or f"HTTP {response.status_code}")
                return TweetResult(
                    success=False,
                    operation="post",
                    error=error_msg,
                    duration_seconds=elapsed,
                )

            data = response.json()
            tweet_id = data.get("data", {}).get("id", "")
            url = f"https://twitter.com/i/web/status/{tweet_id}" if tweet_id else ""

            return TweetResult(
                success=True,
                operation="post",
                tweet_id=tweet_id,
                tweet_url=url,
                text=text,
                duration_seconds=elapsed,
            )

        except httpx.TimeoutException:
            return TweetResult(
                success=False,
                operation="post",
                error="Request timed out",
                duration_seconds=time.monotonic() - start,
            )
        except Exception as e:
            return TweetResult(
                success=False,
                operation="post",
                error=str(e),
                duration_seconds=time.monotonic() - start,
            )

    def get_mentions(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent mentions. Returns list of {id, text, author, created_at}."""
        try:
            # Get authenticated user's ID first
            me_response = self.client.get("/users/me")
            if me_response.status_code != 200:
                return []

            user_id = me_response.json().get("data", {}).get("id", "")
            if not user_id:
                return []

            # Get mentions
            response = self.client.get(
                f"/users/{user_id}/mentions",
                params={"max_results": min(limit, 100)},
            )

            if response.status_code != 200:
                return []

            data = response.json()
            tweets = data.get("data", [])

            return [
                {
                    "id": t.get("id", ""),
                    "text": t.get("text", ""),
                    "author": t.get("author_id", ""),
                    "created_at": t.get("created_at", ""),
                }
                for t in tweets
            ]

        except Exception:
            return []

    def reply_to_tweet(self, tweet_id: str, text: str) -> TweetResult:
        """Reply to a specific tweet."""
        if not text or len(text) > 280:
            return TweetResult(
                success=False,
                operation="reply",
                error=f"Reply length must be 1-280 chars (got {len(text)})",
            )

        start = time.monotonic()

        try:
            response = self.client.post(
                "/tweets",
                json={
                    "text": text,
                    "reply": {"in_reply_to_tweet_id": tweet_id},
                },
            )

            elapsed = time.monotonic() - start

            if response.status_code not in (200, 201):
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", response.text or f"HTTP {response.status_code}")
                return TweetResult(
                    success=False,
                    operation="reply",
                    error=error_msg,
                    duration_seconds=elapsed,
                )

            data = response.json()
            reply_id = data.get("data", {}).get("id", "")
            url = f"https://twitter.com/i/web/status/{reply_id}" if reply_id else ""

            return TweetResult(
                success=True,
                operation="reply",
                tweet_id=reply_id,
                tweet_url=url,
                text=text,
                duration_seconds=elapsed,
            )

        except Exception as e:
            return TweetResult(
                success=False,
                operation="reply",
                error=str(e),
                duration_seconds=time.monotonic() - start,
            )

    def close(self):
        """Close the HTTP client."""
        if self.client:
            self.client.close()
