"""
Episodic memory — compress raw conversation sessions into retrievable episodes.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Transforms raw message sequences into structured episode records via
extractive heuristics (no LLM required). Episodes persist to history.db
and are searchable by topic, decision, and artifact keywords.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


DEFAULT_DB_PATH = Path.home() / ".motherlabs" / "history.db"

# Heuristic patterns for extraction
_DECISION_PATTERNS = re.compile(
    r"(?:let'?s|we should|i'?ll|going to|decided to|chose|picking|using|switching to|"
    r"proceed with|build|implement|create|add|enable|disable|set|configure)\s+(.{5,80})",
    re.IGNORECASE,
)

_ARTIFACT_PATTERNS = re.compile(
    r"(?:"
    r"[a-zA-Z_][\w/]*\.(?:py|js|ts|rs|go|md|json|yaml|toml|sql|sh|css|html)"  # file paths
    r"|`[a-zA-Z_][\w.]*(?:\(\))?`"  # inline code references
    r"|created?\s+(?:a\s+)?(?:new\s+)?(?:file|module|class|function|table|endpoint)"
    r"|wrote\s+(?:a\s+)?(?:new\s+)?(?:test|spec|config)"
    r")",
    re.IGNORECASE,
)

_QUESTION_PATTERNS = re.compile(
    r"(?:^|\n)\s*(?:what|how|why|should|can|could|would|is|are|do|does)\s+.{5,80}\?",
    re.IGNORECASE,
)

# Stopwords for topic extraction
_TOPIC_STOPS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
    "and", "or", "but", "not", "for", "with", "this", "that",
    "was", "are", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "can", "may", "i",
    "you", "he", "she", "we", "they", "me", "my", "your", "just",
    "about", "some", "also", "very", "really", "quite", "much",
    "so", "if", "then", "there", "here", "all", "any", "each",
    "no", "yes", "ok", "okay", "sure", "right", "well", "now",
    "get", "got", "let", "make", "made", "want", "need", "like",
    "know", "think", "see", "look", "good", "new", "one", "two",
})


@dataclass(frozen=True)
class Episode:
    """A compressed summary of a conversation session."""

    episode_id: str           # Unique identifier (session_id or generated)
    session_id: str           # Source session
    start_time: float         # First message timestamp
    end_time: float           # Last message timestamp
    summary: str              # 1-3 sentence extractive summary
    topics: tuple             # Key topics discussed (frozen for hashability)
    decisions: tuple          # Decisions made during session
    artifacts: tuple          # Files/modules created or modified
    questions: tuple          # Key questions raised
    message_count: int        # Total messages in session
    user_turns: int           # User message count
    duration_seconds: float   # Session wall-clock time


def _extract_topics(messages: List[Dict[str, str]], max_topics: int = 8) -> List[str]:
    """Extract key topics from messages via noun frequency analysis."""
    word_freq: Dict[str, int] = {}
    for msg in messages:
        content = msg.get("content", "")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        for w in words:
            if w not in _TOPIC_STOPS and len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1

    # Filter: words that appear in multiple messages are topics
    # Weight by frequency but cap single-message words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    topics = []
    for word, freq in sorted_words:
        if freq >= 2 and word not in topics:
            topics.append(word)
        if len(topics) >= max_topics:
            break
    return topics


def _extract_decisions(messages: List[Dict[str, str]], max_decisions: int = 5) -> List[str]:
    """Extract decisions from messages via action verb patterns."""
    decisions = []
    seen = set()
    for msg in messages:
        content = msg.get("content", "")
        matches = _DECISION_PATTERNS.findall(content)
        for match in matches:
            # Clean and deduplicate
            clean = match.strip().rstrip(".,;:!?")
            norm = clean.lower()[:40]
            if norm not in seen and len(clean) > 5:
                decisions.append(clean)
                seen.add(norm)
            if len(decisions) >= max_decisions:
                return decisions
    return decisions


def _extract_artifacts(messages: List[Dict[str, str]], max_artifacts: int = 10) -> List[str]:
    """Extract file paths and code references from messages."""
    artifacts = []
    seen = set()
    for msg in messages:
        content = msg.get("content", "")
        matches = _ARTIFACT_PATTERNS.findall(content)
        for match in matches:
            clean = match.strip().strip("`")
            norm = clean.lower()
            if norm not in seen and len(clean) > 2:
                artifacts.append(clean)
                seen.add(norm)
            if len(artifacts) >= max_artifacts:
                return artifacts
    return artifacts


def _extract_questions(messages: List[Dict[str, str]], max_questions: int = 5) -> List[str]:
    """Extract key questions from user messages."""
    questions = []
    seen = set()
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        matches = _QUESTION_PATTERNS.findall(content)
        for match in matches:
            clean = match.strip()
            norm = clean.lower()[:40]
            if norm not in seen and len(clean) > 10:
                questions.append(clean)
                seen.add(norm)
            if len(questions) >= max_questions:
                return questions
    return questions


def _build_summary(
    messages: List[Dict[str, str]],
    topics: List[str],
    decisions: List[str],
    artifacts: List[str],
) -> str:
    """Build a 1-3 sentence extractive summary."""
    parts = []

    # First user message = session opener
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if user_msgs:
        opener = user_msgs[0].get("content", "")[:100]
        if len(opener) > 80:
            opener = opener[:80] + "..."
        parts.append(f"Started with: \"{opener}\"")

    # Topics summary
    if topics:
        parts.append(f"Discussed: {', '.join(topics[:5])}")

    # Decisions summary
    if decisions:
        parts.append(f"Decided: {decisions[0]}")

    # Artifacts summary
    if artifacts:
        count = len(artifacts)
        if count == 1:
            parts.append(f"Touched: {artifacts[0]}")
        else:
            parts.append(f"Touched {count} files/artifacts")

    return ". ".join(parts) + "." if parts else "Empty session."


def compress_session(
    messages: List[Dict[str, str]],
    session_id: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> Episode:
    """Compress a sequence of messages into an Episode.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
                  Optionally include "timestamp" key.
        session_id: Source session identifier.
        start_time: Explicit start time. If 0, derived from messages.
        end_time: Explicit end time. If 0, derived from messages.

    Returns:
        Frozen Episode dataclass.
    """
    if not messages:
        return Episode(
            episode_id=session_id or "empty",
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            summary="Empty session.",
            topics=(),
            decisions=(),
            artifacts=(),
            questions=(),
            message_count=0,
            user_turns=0,
            duration_seconds=0.0,
        )

    # Derive timestamps if not provided
    if start_time == 0.0:
        start_time = messages[0].get("timestamp", time.time())
    if end_time == 0.0:
        end_time = messages[-1].get("timestamp", time.time())

    topics = _extract_topics(messages)
    decisions = _extract_decisions(messages)
    artifacts = _extract_artifacts(messages)
    questions = _extract_questions(messages)
    summary = _build_summary(messages, topics, decisions, artifacts)

    user_turns = sum(1 for m in messages if m.get("role") == "user")

    return Episode(
        episode_id=session_id or f"ep-{int(start_time)}",
        session_id=session_id,
        start_time=start_time,
        end_time=end_time,
        summary=summary,
        topics=tuple(topics),
        decisions=tuple(decisions),
        artifacts=tuple(artifacts),
        questions=tuple(questions),
        message_count=len(messages),
        user_turns=user_turns,
        duration_seconds=max(0.0, end_time - start_time),
    )


def _ensure_episodes_table(conn: sqlite3.Connection) -> None:
    """Create episodes table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            topics_json TEXT NOT NULL DEFAULT '[]',
            decisions_json TEXT NOT NULL DEFAULT '[]',
            artifacts_json TEXT NOT NULL DEFAULT '[]',
            questions_json TEXT NOT NULL DEFAULT '[]',
            message_count INTEGER NOT NULL DEFAULT 0,
            user_turns INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL NOT NULL DEFAULT 0.0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_episodes_start
        ON episodes (start_time)
    """)
    conn.commit()


def save_episode(episode: Episode, db_path: Optional[Path] = None) -> int:
    """Persist an episode to history.db. Returns row id.

    Upserts by episode_id — safe to call multiple times for same session.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_episodes_table(conn)
        cursor = conn.execute(
            """INSERT INTO episodes
               (episode_id, session_id, start_time, end_time, summary,
                topics_json, decisions_json, artifacts_json, questions_json,
                message_count, user_turns, duration_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(episode_id) DO UPDATE SET
                   summary = excluded.summary,
                   topics_json = excluded.topics_json,
                   decisions_json = excluded.decisions_json,
                   artifacts_json = excluded.artifacts_json,
                   questions_json = excluded.questions_json,
                   message_count = excluded.message_count,
                   user_turns = excluded.user_turns,
                   duration_seconds = excluded.duration_seconds
            """,
            (
                episode.episode_id,
                episode.session_id,
                episode.start_time,
                episode.end_time,
                episode.summary,
                json.dumps(list(episode.topics)),
                json.dumps(list(episode.decisions)),
                json.dumps(list(episode.artifacts)),
                json.dumps(list(episode.questions)),
                episode.message_count,
                episode.user_turns,
                episode.duration_seconds,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def _row_to_episode(row: Tuple) -> Episode:
    """Convert a SQLite row to an Episode."""
    return Episode(
        episode_id=row[0],
        session_id=row[1],
        start_time=row[2],
        end_time=row[3],
        summary=row[4],
        topics=tuple(json.loads(row[5])),
        decisions=tuple(json.loads(row[6])),
        artifacts=tuple(json.loads(row[7])),
        questions=tuple(json.loads(row[8])),
        message_count=row[9],
        user_turns=row[10],
        duration_seconds=row[11],
    )


def load_episodes(
    limit: int = 50,
    since: float = 0.0,
    db_path: Optional[Path] = None,
) -> List[Episode]:
    """Load recent episodes, optionally filtered by time."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_episodes_table(conn)
        rows = conn.execute(
            """SELECT episode_id, session_id, start_time, end_time, summary,
                      topics_json, decisions_json, artifacts_json, questions_json,
                      message_count, user_turns, duration_seconds
               FROM episodes
               WHERE start_time >= ?
               ORDER BY start_time DESC
               LIMIT ?""",
            (since, limit),
        ).fetchall()
        return [_row_to_episode(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def search_episodes(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = None,
) -> List[Episode]:
    """Search episodes by keyword match across summary, topics, decisions, artifacts.

    Uses LIKE matching across multiple fields. Returns relevance-ordered results.
    """
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    # Extract search keywords
    words = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in _TOPIC_STOPS]
    if not words:
        return []

    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_episodes_table(conn)
        # Build WHERE clause: any keyword matches any field
        conditions = []
        params = []
        for word in words[:5]:  # Cap at 5 keywords
            pattern = f"%{word}%"
            conditions.append(
                "(LOWER(summary) LIKE ? OR LOWER(topics_json) LIKE ? "
                "OR LOWER(decisions_json) LIKE ? OR LOWER(artifacts_json) LIKE ? "
                "OR LOWER(questions_json) LIKE ?)"
            )
            params.extend([pattern] * 5)

        where = " OR ".join(conditions)
        rows = conn.execute(
            f"""SELECT episode_id, session_id, start_time, end_time, summary,
                       topics_json, decisions_json, artifacts_json, questions_json,
                       message_count, user_turns, duration_seconds
                FROM episodes
                WHERE {where}
                ORDER BY start_time DESC
                LIMIT ?""",
            params + [limit],
        ).fetchall()
        return [_row_to_episode(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def episode_count(db_path: Optional[Path] = None) -> int:
    """Count total episodes stored."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return 0
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_episodes_table(conn)
        row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def format_episode_context(episodes: List[Episode], max_tokens: int = 400) -> str:
    """Format episodes as a context block for prompt injection.

    Returns a [EPISODIC MEMORY] block fitting within the token budget.
    """
    if not episodes:
        return ""

    char_budget = max_tokens * 4
    lines = ["[EPISODIC MEMORY]"]
    used = len(lines[0])

    for ep in episodes:
        # Format: date - summary (topics)
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(ep.start_time)
            date_str = dt.strftime("%b %d")
        except (ValueError, OSError):
            date_str = "unknown"

        topic_str = ", ".join(ep.topics[:4]) if ep.topics else ""
        line = f"- {date_str}: {ep.summary}"
        if topic_str:
            line += f" [{topic_str}]"

        if used + len(line) + 1 > char_budget:
            break
        lines.append(line)
        used += len(line) + 1

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
