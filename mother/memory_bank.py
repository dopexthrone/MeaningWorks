"""
Memory bank — unified retrieval across all memory systems.

LEAF module. Imports only from mother/ LEAF modules (episodic_memory,
knowledge_base, recall). No imports from core/.

Provides a single query interface that fans out across:
- Conversation recall (FTS5)
- Episodic memory (compressed sessions)
- Knowledge facts (structured facts)
- Learned patterns (compilation patterns)
- Goals (active and completed)

Results are ranked by recency × relevance, deduped, and formatted
as a unified context block for LLM injection.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mother.episodic_memory import search_episodes, format_episode_context, Episode
from mother.knowledge_base import search_facts, query_facts, format_knowledge_context, KnowledgeFact


DEFAULT_DB_PATH = Path.home() / ".motherlabs" / "history.db"

# Source weights for ranking
_SOURCE_WEIGHTS = {
    "recall": 1.0,       # Direct keyword matches in conversation
    "episode": 0.9,      # Compressed session matches
    "knowledge": 0.85,   # Structured fact matches
    "pattern": 0.8,      # Learned compilation patterns
    "goal": 0.7,         # Goal descriptions
}

# Recency half-life: relevance halves every N days
_RECENCY_HALF_LIFE_DAYS = 14.0


@dataclass(frozen=True)
class MemoryResult:
    """A single result from unified memory retrieval."""

    source: str          # "recall" | "episode" | "knowledge" | "pattern" | "goal"
    content: str         # Formatted text for this result
    relevance: float     # 0.0-1.0 composite score
    timestamp: float     # When the memory was created/last confirmed
    category: str        # Sub-category within source


def _recency_score(timestamp: float, now: float = 0.0) -> float:
    """Compute recency score with exponential decay.

    Returns 1.0 for now, ~0.5 after _RECENCY_HALF_LIFE_DAYS, approaches 0.
    """
    if now == 0.0:
        now = time.time()
    age_days = max(0.0, (now - timestamp) / 86400)
    import math
    return math.exp(-0.693 * age_days / _RECENCY_HALF_LIFE_DAYS)


def _score_result(source: str, base_relevance: float, timestamp: float) -> float:
    """Compute composite relevance score: source_weight × base_relevance × recency."""
    weight = _SOURCE_WEIGHTS.get(source, 0.5)
    recency = _recency_score(timestamp)
    return weight * base_relevance * recency


def _results_from_episodes(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
) -> List[MemoryResult]:
    """Search episodic memory and convert to MemoryResults."""
    episodes = search_episodes(query, limit=limit, db_path=db_path)
    results = []
    for ep in episodes:
        # Build content from episode
        parts = [ep.summary]
        if ep.decisions:
            parts.append(f"Decisions: {', '.join(ep.decisions[:3])}")
        if ep.artifacts:
            parts.append(f"Artifacts: {', '.join(ep.artifacts[:3])}")

        content = " | ".join(parts)

        # Relevance: based on topic overlap with query
        query_words = set(query.lower().split())
        topic_words = set(w.lower() for w in ep.topics)
        overlap = len(query_words & topic_words)
        base_rel = min(1.0, 0.5 + overlap * 0.15)

        results.append(MemoryResult(
            source="episode",
            content=content,
            relevance=_score_result("episode", base_rel, ep.end_time),
            timestamp=ep.end_time,
            category="session",
        ))
    return results


def _results_from_knowledge(
    query: str,
    limit: int = 10,
    db_path: Optional[Path] = None,
) -> List[MemoryResult]:
    """Search knowledge base and convert to MemoryResults."""
    facts = search_facts(query, limit=limit, db_path=db_path)
    results = []
    for fact in facts:
        results.append(MemoryResult(
            source="knowledge",
            content=f"{fact.subject}: {fact.value}",
            relevance=_score_result("knowledge", fact.confidence, fact.last_confirmed),
            timestamp=fact.last_confirmed,
            category=fact.category,
        ))
    return results


def _results_from_patterns(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
) -> List[MemoryResult]:
    """Search learned patterns and convert to MemoryResults."""
    try:
        from kernel.memory import load_patterns
        maps_db = (db_path or DEFAULT_DB_PATH).parent / "maps.db"
        patterns = load_patterns(min_confidence=0.3, db_dir=maps_db.parent)
    except Exception:
        return []

    results = []
    query_lower = query.lower()
    for pat in patterns:
        # Check if pattern description matches query
        desc_lower = pat.description.lower()
        if any(w in desc_lower for w in query_lower.split() if len(w) > 2):
            results.append(MemoryResult(
                source="pattern",
                content=f"[{pat.category}] {pat.description} (freq={pat.frequency}). Fix: {pat.remediation}",
                relevance=_score_result("pattern", pat.confidence, time.time()),
                timestamp=time.time(),
                category=pat.category,
            ))

    return results[:limit]


def _results_from_goals(
    query: str,
    limit: int = 5,
    db_path: Optional[Path] = None,
) -> List[MemoryResult]:
    """Search goals and convert to MemoryResults."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    import sqlite3
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        # Check if goals table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='goals'"
        ).fetchone()
        if not tables:
            return []

        # Search goals by description
        words = [w.lower() for w in query.split() if len(w) > 2]
        if not words:
            return []

        conditions = []
        params = []
        for word in words[:5]:
            conditions.append("LOWER(description) LIKE ?")
            params.append(f"%{word}%")

        where = " OR ".join(conditions)
        params.append(limit)

        rows = conn.execute(
            f"""SELECT description, status, priority, timestamp
                FROM goals
                WHERE {where}
                ORDER BY timestamp DESC
                LIMIT ?""",
            params,
        ).fetchall()

        results = []
        for row in rows:
            desc, status, priority, ts = row
            content = f"Goal ({status}, priority={priority}): {desc}"
            base_rel = 0.6 if status == "active" else 0.4
            results.append(MemoryResult(
                source="goal",
                content=content,
                relevance=_score_result("goal", base_rel, ts),
                timestamp=ts,
                category=status,
            ))
        return results
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def query(
    text: str,
    recall_engine=None,
    limit: int = 15,
    db_path: Optional[Path] = None,
) -> List[MemoryResult]:
    """Unified memory query across all stores.

    Args:
        text: Query string.
        recall_engine: Optional RecallEngine instance for FTS5 search.
                      If None, skips conversation recall.
        limit: Maximum total results to return.
        db_path: Path to history.db. Defaults to ~/.motherlabs/history.db.

    Returns:
        List of MemoryResult, ranked by composite relevance score.
    """
    if not text or not text.strip():
        return []

    all_results: List[MemoryResult] = []

    # Fan out across all memory sources
    # 1. Conversation recall (FTS5)
    if recall_engine is not None:
        try:
            recall_results = recall_engine.search(text, limit=5)
            for r in recall_results:
                snippet = r.content[:150]
                if len(r.content) > 150:
                    snippet += "..."
                all_results.append(MemoryResult(
                    source="recall",
                    content=f"{r.role}: \"{snippet}\"",
                    relevance=_score_result("recall", 1.0 - (r.relevance_rank * 0.08), r.timestamp),
                    timestamp=r.timestamp,
                    category=r.role,
                ))
        except Exception:
            pass

    # 2. Episodic memory
    try:
        all_results.extend(_results_from_episodes(text, limit=5, db_path=db_path))
    except Exception:
        pass

    # 3. Knowledge facts
    try:
        all_results.extend(_results_from_knowledge(text, limit=8, db_path=db_path))
    except Exception:
        pass

    # 4. Learned patterns
    try:
        all_results.extend(_results_from_patterns(text, limit=3, db_path=db_path))
    except Exception:
        pass

    # 5. Goals
    try:
        all_results.extend(_results_from_goals(text, limit=3, db_path=db_path))
    except Exception:
        pass

    # Rank by composite relevance
    all_results.sort(key=lambda r: r.relevance, reverse=True)

    # Deduplicate by content similarity (exact prefix match)
    seen_prefixes = set()
    unique = []
    for r in all_results:
        prefix = r.content[:60].lower()
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            unique.append(r)

    return unique[:limit]


def format_memory_context(
    results: List[MemoryResult],
    max_tokens: int = 500,
) -> str:
    """Format unified memory results as a context block.

    Replaces the old [Recalled] block with a richer [MEMORY] block
    that includes episodes, facts, patterns, and conversation recall.
    """
    if not results:
        return ""

    char_budget = max_tokens * 4
    lines = ["[MEMORY]"]
    used = len(lines[0])

    # Group by source for organized display
    source_order = ["knowledge", "episode", "recall", "pattern", "goal"]
    by_source = {}
    for r in results:
        by_source.setdefault(r.source, []).append(r)

    for source in source_order:
        items = by_source.get(source, [])
        if not items:
            continue

        # Source header
        headers = {
            "knowledge": "Known:",
            "episode": "Past sessions:",
            "recall": "From conversation:",
            "pattern": "Learned:",
            "goal": "Goals:",
        }
        header = headers.get(source, f"{source}:")
        if used + len(header) + 1 > char_budget:
            break
        lines.append(header)
        used += len(header) + 1

        for item in items[:4]:  # Max 4 per source
            line = f"  - {item.content}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

    if len(lines) == 1:
        return ""

    return "\n".join(lines)


def memory_stats(db_path: Optional[Path] = None) -> dict:
    """Return statistics about the memory bank."""
    from mother.episodic_memory import episode_count
    from mother.knowledge_base import fact_count

    path = db_path or DEFAULT_DB_PATH
    return {
        "episodes": episode_count(db_path=path),
        "facts": fact_count(db_path=path),
        "db_path": str(path),
    }
