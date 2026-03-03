"""
Memory indexer — background extraction of facts and episodes from messages.

LEAF module. Imports only from mother/ LEAF modules (episodic_memory,
knowledge_base). No imports from core/.

Called after every message to extract knowledge facts, and on session
close to compress the session into an episode. Also provides one-time
backfill for existing conversation history.
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional

from mother.episodic_memory import compress_session, save_episode, Episode
from mother.knowledge_base import extract_facts, save_facts, KnowledgeFact

logger = logging.getLogger("mother.memory_indexer")

DEFAULT_DB_PATH = Path.home() / ".motherlabs" / "history.db"


class MemoryIndexer:
    """Extracts and indexes memory from conversation messages.

    Usage:
        indexer = MemoryIndexer(db_path)

        # After each message:
        indexer.index_message(content, role, session_id)

        # When session ends:
        indexer.close_session(session_id)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        # Buffer: session_id -> list of message dicts
        self._session_buffers: Dict[str, List[Dict]] = {}
        self._session_start_times: Dict[str, float] = {}
        self._facts_extracted_count = 0
        self._episodes_saved_count = 0

    def index_message(
        self,
        content: str,
        role: str,
        session_id: str = "",
        timestamp: float = 0.0,
    ) -> List[KnowledgeFact]:
        """Index a single message: extract facts and buffer for episode compression.

        Returns list of extracted facts (may be empty).
        """
        ts = timestamp or time.time()

        # Buffer the message for later episode compression
        if session_id:
            if session_id not in self._session_buffers:
                self._session_buffers[session_id] = []
                self._session_start_times[session_id] = ts
            self._session_buffers[session_id].append({
                "role": role,
                "content": content,
                "timestamp": ts,
            })

        # Extract and save facts
        facts = extract_facts(content, role, session_id=session_id, timestamp=ts)
        if facts:
            try:
                save_facts(facts, db_path=self._db_path)
                self._facts_extracted_count += len(facts)
                logger.debug(f"Indexed {len(facts)} facts from {role} message")
            except Exception as e:
                logger.debug(f"Fact save skipped: {e}")

        return facts

    def close_session(self, session_id: str) -> Optional[Episode]:
        """Compress buffered messages into an episode and persist.

        Returns the Episode if created, None if session too short.
        """
        messages = self._session_buffers.pop(session_id, [])
        start_time = self._session_start_times.pop(session_id, 0.0)

        if len(messages) < 3:
            # Too short to be meaningful
            return None

        end_time = messages[-1].get("timestamp", time.time()) if messages else time.time()

        episode = compress_session(
            messages=messages,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
        )

        try:
            save_episode(episode, db_path=self._db_path)
            self._episodes_saved_count += 1
            logger.info(
                f"Episode saved: {episode.episode_id} "
                f"({episode.message_count} msgs, {len(episode.topics)} topics)"
            )
        except Exception as e:
            logger.debug(f"Episode save skipped: {e}")
            return None

        return episode

    @property
    def stats(self) -> Dict:
        """Current indexer statistics."""
        return {
            "facts_extracted": self._facts_extracted_count,
            "episodes_saved": self._episodes_saved_count,
            "active_session_buffers": len(self._session_buffers),
            "buffered_messages": sum(
                len(msgs) for msgs in self._session_buffers.values()
            ),
        }


def index_context_map(
    context_map: Dict,
    intent: str,
    db_path: Optional[Path] = None,
) -> Dict:
    """Bulk-index concepts and vocabulary from a CONTEXT compilation.

    Creates KnowledgeFact entries for each concept and vocabulary term,
    linking them to the original intent as source.

    Args:
        context_map: Dict from context_map_to_dict() with concepts, vocabulary, etc.
        intent: The original user intent that produced this context.
        db_path: Optional override for the database path.

    Returns:
        Stats dict with counts of what was indexed.
    """
    path = db_path or DEFAULT_DB_PATH
    ts = time.time()
    source = f"context:{intent[:60]}"
    facts = []

    # Index concepts as facts
    concepts = context_map.get("concepts", [])
    for concept in concepts:
        name = concept.get("name", "")
        desc = concept.get("description", "")
        conf = concept.get("confidence", 0.5)
        if name:
            facts.append(KnowledgeFact(
                fact_id=f"ctx:{name.lower()}:{intent[:30]}",
                category="decision",
                subject=name,
                predicate="is concept in",
                value=desc[:200] if desc else intent[:200],
                confidence=conf,
                source=source,
                first_seen=ts,
                last_confirmed=ts,
                access_count=0,
            ))

    # Index vocabulary terms
    vocabulary = context_map.get("vocabulary", [])
    for term in vocabulary:
        if term and len(term) > 1:
            facts.append(KnowledgeFact(
                fact_id=f"vocab:{term.lower()}",
                category="tool",
                subject=term,
                predicate="is domain term",
                value=f"From context: {intent[:120]}",
                confidence=0.6,
                source=source,
                first_seen=ts,
                last_confirmed=ts,
                access_count=0,
            ))

    # Index assumptions
    assumptions = context_map.get("assumptions", [])
    for assumption in assumptions[:10]:
        text = assumption.get("text", "")
        cat = assumption.get("category", "")
        if text:
            facts.append(KnowledgeFact(
                fact_id=f"assumption:{hash(text) & 0xFFFFFFFF:08x}",
                category="constraint",
                subject=cat or "assumption",
                predicate="assumes",
                value=text[:200],
                confidence=assumption.get("confidence", 0.5),
                source=source,
                first_seen=ts,
                last_confirmed=ts,
                access_count=0,
            ))

    # Persist
    saved = 0
    if facts:
        try:
            saved = save_facts(facts, db_path=path)
        except Exception as e:
            logger.debug(f"Context map fact save skipped: {e}")

    return {
        "concepts_indexed": len(concepts),
        "vocabulary_indexed": len(vocabulary),
        "assumptions_indexed": min(len(assumptions), 10),
        "facts_saved": saved,
    }


def reindex_history(
    db_path: Optional[Path] = None,
    batch_size: int = 100,
) -> Dict:
    """One-time backfill: compress all existing sessions into episodes + extract facts.

    Reads from the messages table, groups by session_id, compresses each
    session into an episode, and extracts facts from each message.

    Returns statistics about what was indexed.
    """
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return {"sessions_processed": 0, "episodes_created": 0, "facts_extracted": 0}

    conn = sqlite3.connect(str(path), timeout=10)
    stats = {
        "sessions_processed": 0,
        "episodes_created": 0,
        "facts_extracted": 0,
        "messages_processed": 0,
    }

    try:
        # Get all sessions
        sessions = conn.execute(
            """SELECT session_id, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts,
                      COUNT(*) as msg_count
               FROM messages
               GROUP BY session_id
               ORDER BY first_ts ASC"""
        ).fetchall()

        for session_id, first_ts, last_ts, msg_count in sessions:
            if msg_count < 3:
                continue

            # Fetch messages for this session
            rows = conn.execute(
                """SELECT role, content, timestamp
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ?""",
                (session_id, batch_size),
            ).fetchall()

            messages = [
                {"role": r[0], "content": r[1], "timestamp": r[2]}
                for r in rows
            ]

            # Compress to episode
            episode = compress_session(
                messages=messages,
                session_id=session_id,
                start_time=first_ts,
                end_time=last_ts,
            )
            try:
                save_episode(episode, db_path=path)
                stats["episodes_created"] += 1
            except Exception as e:
                logger.debug(f"Episode backfill skipped for {session_id}: {e}")

            # Extract facts from each message
            for msg in messages:
                facts = extract_facts(
                    msg["content"],
                    msg["role"],
                    session_id=session_id,
                    timestamp=msg.get("timestamp", 0.0),
                )
                if facts:
                    try:
                        save_facts(facts, db_path=path)
                        stats["facts_extracted"] += len(facts)
                    except Exception as e:
                        logger.debug(f"Fact backfill skipped: {e}")
                stats["messages_processed"] += 1

            stats["sessions_processed"] += 1

    finally:
        conn.close()

    return stats
