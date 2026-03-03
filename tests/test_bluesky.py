"""Tests for mother/bluesky.py — Bluesky AT Protocol client."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mother.bluesky import (
    BlueskyClient,
    BlueskyResult,
    _build_facets,
    _detect_links,
)


# --- _detect_links ---


class TestDetectLinks:
    def test_simple_url(self):
        links = _detect_links("Check https://example.com for details")
        assert len(links) == 1
        assert links[0][2] == "https://example.com"

    def test_multiple_urls(self):
        links = _detect_links("See https://a.com and https://b.com")
        assert len(links) == 2

    def test_no_urls(self):
        links = _detect_links("No links here")
        assert len(links) == 0

    def test_trailing_punctuation_stripped(self):
        links = _detect_links("Visit https://example.com.")
        assert links[0][2] == "https://example.com"

    def test_http_url(self):
        links = _detect_links("Old http://example.com link")
        assert len(links) == 1
        assert links[0][2] == "http://example.com"

    def test_url_with_path(self):
        links = _detect_links("See https://github.com/user/repo")
        assert links[0][2] == "https://github.com/user/repo"

    def test_byte_positions_ascii(self):
        text = "Visit https://example.com now"
        links = _detect_links(text)
        start, end, url = links[0]
        # Verify byte positions match
        assert text.encode("utf-8")[start:end].decode() == url


# --- _build_facets ---


class TestBuildFacets:
    def test_link_facet(self):
        facets = _build_facets("See https://example.com")
        assert len(facets) == 1
        assert facets[0]["features"][0]["$type"] == "app.bsky.richtext.facet#link"
        assert facets[0]["features"][0]["uri"] == "https://example.com"

    def test_no_links(self):
        facets = _build_facets("No links")
        assert len(facets) == 0

    def test_byte_index_present(self):
        facets = _build_facets("See https://example.com")
        assert "byteStart" in facets[0]["index"]
        assert "byteEnd" in facets[0]["index"]


# --- BlueskyClient ---


class TestBlueskyClient:
    def test_init_from_env(self):
        with patch.dict("os.environ", {"BLUESKY_HANDLE": "user.bsky.social", "BLUESKY_APP_PASSWORD": "pass123"}):
            client = BlueskyClient()
            assert client._handle == "user.bsky.social"
            assert client._app_password == "pass123"

    def test_init_from_params(self):
        client = BlueskyClient(handle="test.bsky.social", app_password="pw")
        assert client._handle == "test.bsky.social"

    def test_post_empty_text(self):
        client = BlueskyClient(handle="test", app_password="pw")
        result = client.post("")
        assert result.success is False
        assert "Empty" in result.error

    def test_post_too_long(self):
        client = BlueskyClient(handle="test", app_password="pw")
        result = client.post("x" * 301)
        assert result.success is False
        assert "300 char limit" in result.error

    def test_post_missing_credentials(self):
        client = BlueskyClient(handle="", app_password="")
        result = client.post("Hello Bluesky!")
        assert result.success is False
        assert "credentials" in result.error.lower()

    @patch("mother.bluesky.urllib.request.urlopen")
    def test_auth_failure(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://bsky.social", 401, "Unauthorized", {}, MagicMock(read=lambda: b'{"error":"AuthFailed"}')
        )
        client = BlueskyClient(handle="test.bsky.social", app_password="bad")
        result = client.post("Hello!")
        assert result.success is False

    @patch("mother.bluesky.urllib.request.urlopen")
    def test_happy_path(self, mock_urlopen):
        # First call: createSession
        auth_response = MagicMock()
        auth_response.status = 200
        auth_response.read.return_value = json.dumps({
            "accessJwt": "jwt123",
            "did": "did:plc:abc123",
        }).encode()

        # Second call: createRecord
        post_response = MagicMock()
        post_response.status = 200
        post_response.read.return_value = json.dumps({
            "uri": "at://did:plc:abc123/app.bsky.feed.post/xyz789",
            "cid": "bafyrei...",
        }).encode()

        mock_urlopen.side_effect = [auth_response, post_response]

        client = BlueskyClient(handle="test.bsky.social", app_password="pw123")
        result = client.post("Hello Bluesky!")

        assert result.success is True
        assert result.post_uri == "at://did:plc:abc123/app.bsky.feed.post/xyz789"
        assert "test.bsky.social" in result.post_url
        assert "xyz789" in result.post_url

    @patch("mother.bluesky.urllib.request.urlopen")
    def test_session_reused(self, mock_urlopen):
        # Pre-set auth
        client = BlueskyClient(handle="test.bsky.social", app_password="pw")
        client._access_jwt = "existing_jwt"
        client._did = "did:plc:existing"

        post_response = MagicMock()
        post_response.status = 200
        post_response.read.return_value = json.dumps({
            "uri": "at://did:plc:existing/app.bsky.feed.post/abc",
        }).encode()
        mock_urlopen.return_value = post_response

        result = client.post("Reuse session")
        assert result.success is True
        # Only 1 call (createRecord), no createSession
        assert mock_urlopen.call_count == 1

    def test_close(self):
        client = BlueskyClient(handle="test", app_password="pw")
        client._access_jwt = "jwt"
        client._did = "did"
        client.close()
        assert client._access_jwt == ""
        assert client._did == ""

    def test_result_frozen(self):
        r = BlueskyResult(success=True)
        with pytest.raises(AttributeError):
            r.success = False
