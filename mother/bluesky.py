"""
Bluesky AT Protocol client — post to Bluesky via app password auth.

LEAF module. Stdlib only. No imports from core/ or mother/.

Free, open API. Growing dev audience. Uses AT Protocol with app password.
"""

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger("mother.bluesky")


@dataclass(frozen=True)
class BlueskyResult:
    """Outcome of a Bluesky operation."""

    success: bool
    operation: str = ""
    post_uri: str = ""
    post_url: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _detect_links(text: str) -> List[Tuple[int, int, str]]:
    """Detect URLs in text. Returns list of (byte_start, byte_end, url).

    Byte positions are for the UTF-8 encoded text, as required by AT Protocol facets.
    """
    pattern = re.compile(r"https?://[^\s\)\]\>,]+")
    text_bytes = text.encode("utf-8")
    results = []

    for match in pattern.finditer(text):
        url = match.group(0)
        # Strip trailing punctuation that's not part of the URL
        while url and url[-1] in ".,:;!?":
            url = url[:-1]

        # Find byte positions
        prefix = text[: match.start()].encode("utf-8")
        url_bytes = url.encode("utf-8")
        byte_start = len(prefix)
        byte_end = byte_start + len(url_bytes)

        results.append((byte_start, byte_end, url))

    return results


def _build_facets(text: str) -> List[dict]:
    """Build AT Protocol facets for links in text."""
    links = _detect_links(text)
    facets = []
    for byte_start, byte_end, url in links:
        facets.append({
            "index": {
                "byteStart": byte_start,
                "byteEnd": byte_end,
            },
            "features": [
                {
                    "$type": "app.bsky.richtext.facet#link",
                    "uri": url,
                }
            ],
        })
    return facets


class BlueskyClient:
    """Bluesky AT Protocol client using app password authentication."""

    PDS_URL = "https://bsky.social/xrpc"

    def __init__(self, handle: str = "", app_password: str = ""):
        self._handle = handle or os.environ.get("BLUESKY_HANDLE", "")
        self._app_password = app_password or os.environ.get("BLUESKY_APP_PASSWORD", "")
        self._access_jwt: str = ""
        self._did: str = ""

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> Tuple[bool, dict]:
        """Make an authenticated request to the PDS. Returns (success, response_data)."""
        url = f"{self.PDS_URL}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        if self._access_jwt:
            headers["Authorization"] = f"Bearer {self._access_jwt}"

        try:
            body = json.dumps(data).encode("utf-8") if data else None
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            response = urllib.request.urlopen(req, timeout=timeout)
            resp_body = response.read().decode("utf-8")
            return True, json.loads(resp_body) if resp_body else {}
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            return False, {"error": f"HTTP {e.code}", "message": error_body}
        except urllib.error.URLError as e:
            return False, {"error": f"URL error: {e.reason}"}
        except TimeoutError:
            return False, {"error": f"Timed out after {timeout:.0f}s"}
        except Exception as e:
            return False, {"error": str(e)}

    def _ensure_session(self) -> Optional[str]:
        """Create a session if not already authenticated. Returns error string or None."""
        if self._access_jwt:
            return None

        if not self._handle or not self._app_password:
            return "Bluesky credentials not configured (BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)"

        ok, data = self._request(
            "POST",
            "com.atproto.server.createSession",
            {"identifier": self._handle, "password": self._app_password},
        )

        if not ok:
            return f"Auth failed: {data.get('error', 'unknown')}"

        self._access_jwt = data.get("accessJwt", "")
        self._did = data.get("did", "")

        if not self._access_jwt:
            return "Auth response missing accessJwt"

        return None

    def post(self, text: str, timeout: float = 10.0) -> BlueskyResult:
        """Post to Bluesky.

        Args:
            text: Post text (max 300 graphemes per AT Protocol spec)
            timeout: Request timeout
        """
        start = time.monotonic()

        if not text:
            return BlueskyResult(
                success=False,
                operation="post",
                error="Empty post text",
                duration_seconds=0.0,
            )

        # Bluesky limit is 300 graphemes (characters)
        if len(text) > 300:
            return BlueskyResult(
                success=False,
                operation="post",
                error=f"Post exceeds 300 char limit (got {len(text)})",
                duration_seconds=0.0,
            )

        # Authenticate
        auth_error = self._ensure_session()
        if auth_error:
            return BlueskyResult(
                success=False,
                operation="post",
                error=auth_error,
                duration_seconds=time.monotonic() - start,
            )

        # Build record
        record: dict = {
            "$type": "app.bsky.feed.post",
            "text": text,
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }

        # Auto-detect links and add facets
        facets = _build_facets(text)
        if facets:
            record["facets"] = facets

        # Create post
        ok, data = self._request(
            "POST",
            "com.atproto.repo.createRecord",
            {
                "repo": self._did,
                "collection": "app.bsky.feed.post",
                "record": record,
            },
            timeout=timeout,
        )

        elapsed = time.monotonic() - start

        if not ok:
            return BlueskyResult(
                success=False,
                operation="post",
                error=data.get("error", "Unknown error"),
                duration_seconds=elapsed,
            )

        post_uri = data.get("uri", "")

        # Construct web URL from AT URI
        # at://did:plc:xxx/app.bsky.feed.post/yyy -> https://bsky.app/profile/handle/post/yyy
        post_url = ""
        if post_uri and self._handle:
            parts = post_uri.split("/")
            if len(parts) >= 5:
                rkey = parts[-1]
                post_url = f"https://bsky.app/profile/{self._handle}/post/{rkey}"

        return BlueskyResult(
            success=True,
            operation="post",
            post_uri=post_uri,
            post_url=post_url,
            duration_seconds=elapsed,
        )

    def close(self):
        """Clear session state."""
        self._access_jwt = ""
        self._did = ""
