"""Tests for mother/social_queue.py — cross-platform social post queue."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from mother.social_queue import (
    DEFAULT_COOLDOWNS,
    MAX_RETRIES,
    PostResult,
    SocialPost,
    SocialQueue,
)


@pytest.fixture
def queue(tmp_path):
    """Create a SocialQueue with a temporary database."""
    db = str(tmp_path / "test.db")
    return SocialQueue(db_path=db, cooldowns={"twitter": 0, "discord": 0, "bluesky": 0})


@pytest.fixture
def dispatchers():
    """Mock dispatchers that succeed."""
    return {
        "twitter": lambda content, embed: {"success": True, "url": "https://twitter.com/123", "error": ""},
        "discord": lambda content, embed: {"success": True, "url": "", "error": ""},
        "bluesky": lambda content, embed: {"success": True, "url": "https://bsky.app/post/123", "error": ""},
    }


# --- Enqueue/Dequeue ---


class TestEnqueue:
    def test_enqueue_returns_id(self, queue):
        post = SocialPost(platform="twitter", content="Hello")
        row_id = queue.enqueue(post)
        assert row_id > 0

    def test_enqueue_multiple(self, queue):
        id1 = queue.enqueue(SocialPost(platform="twitter", content="A"))
        id2 = queue.enqueue(SocialPost(platform="discord", content="B"))
        assert id2 > id1

    def test_pending_count(self, queue):
        assert queue.pending_count() == 0
        queue.enqueue(SocialPost(platform="twitter", content="A"))
        queue.enqueue(SocialPost(platform="discord", content="B"))
        assert queue.pending_count() == 2


# --- Process Pending ---


class TestProcessPending:
    def test_process_empty_queue(self, queue, dispatchers):
        results = queue.process_pending(dispatchers)
        assert results == []

    def test_process_single_post(self, queue, dispatchers):
        queue.enqueue(SocialPost(platform="twitter", content="Hello"))
        results = queue.process_pending(dispatchers)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].platform == "twitter"
        assert queue.pending_count() == 0

    def test_process_multi_platform(self, queue, dispatchers):
        queue.enqueue(SocialPost(platform="twitter", content="T"))
        queue.enqueue(SocialPost(platform="discord", content="D"))
        queue.enqueue(SocialPost(platform="bluesky", content="B"))
        results = queue.process_pending(dispatchers)
        assert len(results) == 3
        platforms = {r.platform for r in results}
        assert platforms == {"twitter", "discord", "bluesky"}

    def test_failed_post_retries(self, queue):
        queue.enqueue(SocialPost(platform="twitter", content="fail"))
        fail_dispatcher = {
            "twitter": lambda c, e: {"success": False, "url": "", "error": "rate limited"},
        }
        # First failure
        results = queue.process_pending(fail_dispatcher)
        assert len(results) == 1
        assert results[0].success is False
        assert queue.pending_count() == 1  # Still pending for retry

        # Second failure
        results = queue.process_pending(fail_dispatcher)
        assert queue.pending_count() == 1

        # Third failure → marked as failed
        results = queue.process_pending(fail_dispatcher)
        assert queue.pending_count() == 0
        assert len(queue.failed_posts()) == 1

    def test_no_dispatcher_skipped(self, queue):
        queue.enqueue(SocialPost(platform="mastodon", content="hi"))
        results = queue.process_pending({"twitter": lambda c, e: {"success": True, "url": "", "error": ""}})
        assert results == []
        assert queue.pending_count() == 1

    def test_dispatcher_exception(self, queue):
        queue.enqueue(SocialPost(platform="twitter", content="crash"))
        bad_dispatchers = {
            "twitter": lambda c, e: (_ for _ in ()).throw(RuntimeError("boom")),
        }
        # Use a proper function that raises
        def _raise(c, e):
            raise RuntimeError("boom")
        bad_dispatchers["twitter"] = _raise

        results = queue.process_pending(bad_dispatchers)
        assert len(results) == 1
        assert results[0].success is False
        assert "boom" in results[0].error


# --- Scheduling ---


class TestScheduling:
    def test_future_post_skipped(self, queue, dispatchers):
        future = time.time() + 3600  # 1 hour from now
        queue.enqueue(SocialPost(platform="twitter", content="later", scheduled_at=future))
        results = queue.process_pending(dispatchers)
        assert results == []
        assert queue.pending_count() == 1

    def test_past_post_processed(self, queue, dispatchers):
        past = time.time() - 10
        queue.enqueue(SocialPost(platform="twitter", content="now", scheduled_at=past))
        results = queue.process_pending(dispatchers)
        assert len(results) == 1
        assert results[0].success is True

    def test_immediate_post_processed(self, queue, dispatchers):
        queue.enqueue(SocialPost(platform="twitter", content="now", scheduled_at=0))
        results = queue.process_pending(dispatchers)
        assert len(results) == 1


# --- Rate Limiting ---


class TestRateLimiting:
    def test_rate_limiting(self, tmp_path):
        db = str(tmp_path / "test.db")
        q = SocialQueue(db_path=db, cooldowns={"twitter": 9999})  # Very long cooldown
        q.enqueue(SocialPost(platform="twitter", content="first"))
        q.enqueue(SocialPost(platform="twitter", content="second"))

        dispatcher = {"twitter": lambda c, e: {"success": True, "url": "", "error": ""}}

        results = q.process_pending(dispatcher)
        # Only first should be processed, second rate-limited
        assert len(results) == 1


# --- Enqueue Announcement ---


class TestEnqueueAnnouncement:
    def test_all_platforms(self, queue):
        row_ids = queue.enqueue_announcement(
            "MyApp", "A cool app", "https://github.com/x/y", 12, 87.5,
        )
        assert len(row_ids) == 3
        assert queue.pending_count() == 3

    def test_specific_platforms(self, queue):
        row_ids = queue.enqueue_announcement(
            "App", "Desc", platforms=["discord"],
        )
        assert len(row_ids) == 1

    def test_twitter_content_fits(self, queue):
        queue.enqueue_announcement("App", "Short desc", "https://github.com/x/y", 5, 90)
        # Verify content length by processing with a spy
        contents = []
        def spy(content, embed):
            contents.append(content)
            return {"success": True, "url": "", "error": ""}
        queue.process_pending({"twitter": spy, "discord": spy, "bluesky": spy})
        twitter_content = [c for c in contents if c and len(c) <= 280]
        assert len(twitter_content) >= 1

    def test_discord_uses_embed(self, queue):
        queue.enqueue_announcement("App", "Desc", "https://github.com/x/y", 5, 90)
        embeds = []
        def spy(content, embed):
            embeds.append(embed)
            return {"success": True, "url": "", "error": ""}
        queue.process_pending({"twitter": spy, "discord": spy, "bluesky": spy})
        # Discord should have embed
        discord_embeds = [e for e in embeds if e is not None]
        assert len(discord_embeds) >= 1


# --- Persistence ---


class TestPersistence:
    def test_survives_reinit(self, tmp_path):
        db = str(tmp_path / "test.db")
        q1 = SocialQueue(db_path=db, cooldowns={"twitter": 0})
        q1.enqueue(SocialPost(platform="twitter", content="persistent"))
        assert q1.pending_count() == 1

        # Create new instance pointing to same DB
        q2 = SocialQueue(db_path=db, cooldowns={"twitter": 0})
        assert q2.pending_count() == 1


# --- Recent Posts / Failed Posts ---


class TestRecentPosts:
    def test_recent_posts(self, queue, dispatchers):
        queue.enqueue(SocialPost(platform="twitter", content="hello"))
        queue.process_pending(dispatchers)
        recent = queue.recent_posts(limit=10)
        assert len(recent) == 1
        assert recent[0]["status"] == "sent"

    def test_failed_posts(self, queue):
        # Enqueue and fail 3 times
        queue.enqueue(SocialPost(platform="twitter", content="fail"))
        fail = {"twitter": lambda c, e: {"success": False, "url": "", "error": "err"}}
        for _ in range(MAX_RETRIES):
            queue.process_pending(fail)
        failed = queue.failed_posts()
        assert len(failed) == 1
        assert failed[0]["error"] == "err"

    def test_recent_posts_empty(self, queue):
        assert queue.recent_posts() == []


# --- Edge Cases ---


class TestEdgeCases:
    def test_close_is_noop(self, queue):
        queue.close()  # Should not raise

    def test_post_result_fields(self):
        r = PostResult(platform="twitter", success=True, url="https://x.com/123")
        assert r.platform == "twitter"
        assert r.url == "https://x.com/123"

    def test_social_post_defaults(self):
        p = SocialPost(platform="twitter", content="hi")
        assert p.scheduled_at == 0.0
        assert p.priority == 0
        assert p.retry_count == 0
        assert p.embed is None
