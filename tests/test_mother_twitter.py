"""
Tests for mother/twitter.py -- LEAF module.

Covers: TweetResult frozen dataclass, TwitterClient with mocked httpx,
post_tweet, get_mentions, reply_to_tweet.
"""

from unittest.mock import MagicMock, patch

import pytest

# httpx is optional — skip tests if not available
pytest.importorskip("httpx")

from mother.twitter import TweetResult, TwitterClient


class TestTweetResult:
    def test_frozen(self):
        r = TweetResult(success=True, operation="post")
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = TweetResult(success=True)
        assert r.operation == ""
        assert r.tweet_id == ""
        assert r.tweet_url == ""


class TestTwitterClient:
    @patch("mother.twitter.httpx.Client")
    def test_requires_token(self, mock_client_cls):
        with pytest.raises(ValueError, match="bearer token required"):
            TwitterClient(bearer_token="")

    @patch.dict("os.environ", {"TWITTER_BEARER_TOKEN": "test_token"})
    @patch("mother.twitter.httpx.Client")
    def test_loads_token_from_env(self, mock_client_cls):
        client = TwitterClient()
        assert client.bearer_token == "test_token"

    @patch("mother.twitter.httpx.Client")
    def test_post_tweet_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "123456"}}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = TwitterClient(bearer_token="test_token")
        result = client.post_tweet("Hello world")

        assert result.success
        assert result.tweet_id == "123456"
        assert "123456" in result.tweet_url

    @patch("mother.twitter.httpx.Client")
    def test_post_tweet_too_long(self, mock_client_cls):
        client = TwitterClient(bearer_token="test_token")
        result = client.post_tweet("x" * 281)
        assert not result.success
        assert "280 chars" in result.error

    @patch("mother.twitter.httpx.Client")
    def test_post_tweet_empty(self, mock_client_cls):
        client = TwitterClient(bearer_token="test_token")
        result = client.post_tweet("")
        assert not result.success

    @patch("mother.twitter.httpx.Client")
    def test_post_tweet_api_error(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {"detail": "Forbidden"}
        mock_response.text = '{"detail": "Forbidden"}'
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = TwitterClient(bearer_token="test_token")
        result = client.post_tweet("Hello")

        assert not result.success
        assert "Forbidden" in result.error

    @patch("mother.twitter.httpx.Client")
    def test_get_mentions(self, mock_client_cls):
        mock_client = MagicMock()
        # Mock user ID lookup
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"data": {"id": "user123"}}
        # Mock mentions response
        mentions_response = MagicMock()
        mentions_response.status_code = 200
        mentions_response.json.return_value = {
            "data": [
                {"id": "t1", "text": "Hey @user", "author_id": "a1", "created_at": "2026-01-01"},
            ]
        }
        mock_client.get.side_effect = [me_response, mentions_response]
        mock_client_cls.return_value = mock_client

        client = TwitterClient(bearer_token="test_token")
        mentions = client.get_mentions(limit=10)

        assert len(mentions) == 1
        assert mentions[0]["id"] == "t1"
        assert mentions[0]["text"] == "Hey @user"

    @patch("mother.twitter.httpx.Client")
    def test_reply_to_tweet(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "reply123"}}
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = TwitterClient(bearer_token="test_token")
        result = client.reply_to_tweet("original123", "Thanks!")

        assert result.success
        assert result.tweet_id == "reply123"
