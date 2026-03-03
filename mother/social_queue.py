"""
Social queue — cross-platform post scheduling with SQLite persistence.

LEAF module. Stdlib only. No imports from core/ or mother/.

Rate limiting, scheduling, retry. Persists to SQLite so queued posts
survive restarts.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("mother.social_queue")


@dataclass(frozen=True)
class SocialPost:
    """A post to be dispatched to a social platform."""

    platform: str  # "twitter", "discord", "bluesky"
    content: str
    embed: Optional[str] = None  # JSON string for discord embeds
    scheduled_at: float = 0.0  # unix timestamp (0 = immediate)
    priority: int = 0  # higher = sooner
    retry_count: int = 0


@dataclass(frozen=True)
class PostResult:
    """Outcome of dispatching a single post."""

    platform: str
    success: bool
    url: str = ""
    error: str = ""


# Per-platform cooldown in seconds
DEFAULT_COOLDOWNS: Dict[str, float] = {
    "twitter": 60.0,
    "discord": 5.0,
    "bluesky": 30.0,
}

MAX_RETRIES = 3


class SocialQueue:
    """Cross-platform social post queue with SQLite persistence.

    Posts are enqueued, then processed in priority/schedule order.
    Rate limiting per platform. Failed posts retry up to MAX_RETRIES times.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        cooldowns: Optional[Dict[str, float]] = None,
    ):
        self._db_path = db_path or str(Path.home() / ".motherlabs" / "maps.db")
        self._cooldowns = cooldowns or DEFAULT_COOLDOWNS.copy()
        self._last_post_time: Dict[str, float] = {}
        self._init_db()

    def _init_db(self):
        """Create the social_queue table if it doesn't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS social_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embed TEXT,
                    scheduled_at REAL NOT NULL DEFAULT 0,
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    result_url TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(self._db_path)

    def enqueue(self, post: SocialPost) -> int:
        """Add a post to the queue. Returns the row ID."""
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO social_queue
                    (platform, content, embed, scheduled_at, priority, status, retry_count, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    post.platform,
                    post.content,
                    post.embed,
                    post.scheduled_at,
                    post.priority,
                    post.retry_count,
                    time.time(),
                ),
            )
            conn.commit()
            row_id = cursor.lastrowid or 0
            logger.debug(f"Enqueued {post.platform} post (id={row_id})")
            return row_id
        finally:
            conn.close()

    def enqueue_announcement(
        self,
        name: str,
        description: str,
        repo_url: str = "",
        components: int = 0,
        trust: float = 0.0,
        platforms: Optional[List[str]] = None,
    ) -> List[int]:
        """Enqueue platform-appropriate announcements for all given platforms.

        Args:
            name: Project name
            description: Project description
            repo_url: GitHub repo URL
            components: Number of components
            trust: Trust score (0-100)
            platforms: List of platforms to announce on (default: all)

        Returns:
            List of enqueued row IDs
        """
        platforms = platforms or ["twitter", "discord", "bluesky"]
        row_ids = []

        for platform in platforms:
            if platform == "twitter":
                # 280 char limit
                tweet = f"Just shipped {name}: {description[:100]}"
                if components > 0:
                    tweet += f" — {components} components"
                if trust > 0:
                    tweet += f", {trust:.0f}% trust"
                if repo_url:
                    tweet += f"\n{repo_url}"
                if len(tweet) > 280:
                    tweet = tweet[:277] + "..."
                row_ids.append(self.enqueue(SocialPost(
                    platform="twitter",
                    content=tweet,
                )))

            elif platform == "discord":
                from mother.discord import format_build_announcement
                embed = format_build_announcement(
                    name, description, repo_url, components, trust
                )
                row_ids.append(self.enqueue(SocialPost(
                    platform="discord",
                    content="",
                    embed=json.dumps(embed),
                )))

            elif platform == "bluesky":
                # 300 char limit
                post_text = f"Just shipped {name}: {description[:120]}"
                if components > 0:
                    post_text += f" — {components} components"
                if trust > 0:
                    post_text += f", {trust:.0f}% trust"
                if repo_url:
                    post_text += f"\n{repo_url}"
                if len(post_text) > 300:
                    post_text = post_text[:297] + "..."
                row_ids.append(self.enqueue(SocialPost(
                    platform="bluesky",
                    content=post_text,
                )))

        return row_ids

    def process_pending(
        self,
        dispatchers: Dict[str, Callable],
    ) -> List[PostResult]:
        """Process due posts. Dispatchers are platform -> callable.

        Each dispatcher receives (content: str, embed: Optional[dict]) and
        returns a dict with {"success": bool, "url": str, "error": str}.

        Rate-limited per platform. Posts scheduled in the future are skipped.
        Failed posts are retried up to MAX_RETRIES times.
        """
        now = time.time()
        results = []

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, platform, content, embed, retry_count
                FROM social_queue
                WHERE status = 'pending' AND scheduled_at <= ?
                ORDER BY priority DESC, created_at ASC
                """,
                (now,),
            ).fetchall()
        finally:
            conn.close()

        for row_id, platform, content, embed_json, retry_count in rows:
            # Rate limiting
            cooldown = self._cooldowns.get(platform, 10.0)
            last_time = self._last_post_time.get(platform, 0.0)
            if now - last_time < cooldown:
                continue

            dispatcher = dispatchers.get(platform)
            if not dispatcher:
                logger.debug(f"No dispatcher for platform: {platform}")
                continue

            # Parse embed
            embed = None
            if embed_json:
                try:
                    embed = json.loads(embed_json)
                except json.JSONDecodeError:
                    pass

            # Dispatch
            try:
                result = dispatcher(content, embed)
                success = result.get("success", False)
                url = result.get("url", "")
                error = result.get("error", "")
            except Exception as e:
                success = False
                url = ""
                error = str(e)

            self._last_post_time[platform] = time.time()

            # Update status
            conn = self._connect()
            try:
                if success:
                    conn.execute(
                        "UPDATE social_queue SET status = 'sent', result_url = ? WHERE id = ?",
                        (url, row_id),
                    )
                    conn.commit()
                else:
                    new_retry = retry_count + 1
                    if new_retry >= MAX_RETRIES:
                        conn.execute(
                            "UPDATE social_queue SET status = 'failed', error = ?, retry_count = ? WHERE id = ?",
                            (error, new_retry, row_id),
                        )
                    else:
                        conn.execute(
                            "UPDATE social_queue SET retry_count = ?, error = ? WHERE id = ?",
                            (new_retry, error, row_id),
                        )
                    conn.commit()
            finally:
                conn.close()

            results.append(PostResult(
                platform=platform,
                success=success,
                url=url,
                error=error,
            ))

        return results

    def pending_count(self) -> int:
        """Count pending posts."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM social_queue WHERE status = 'pending'"
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def recent_posts(self, limit: int = 20) -> List[dict]:
        """Get recent posts (all statuses)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, platform, content, status, result_url, error, created_at
                FROM social_queue
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "platform": r[1],
                    "content": r[2][:200],
                    "status": r[3],
                    "url": r[4],
                    "error": r[5],
                    "created_at": r[6],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def failed_posts(self) -> List[dict]:
        """Get failed posts for inspection/retry."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, platform, content, error, retry_count, created_at
                FROM social_queue
                WHERE status = 'failed'
                ORDER BY created_at DESC
                """,
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "platform": r[1],
                    "content": r[2][:200],
                    "error": r[3],
                    "retry_count": r[4],
                    "created_at": r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def close(self):
        """No-op — connections are per-operation."""
        pass
