"""
Knowledge base — structured facts extracted from conversation and compilation.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Extracts and stores discrete knowledge facts (preferences, decisions,
capabilities, project info, people) via heuristic pattern matching.
Facts have confidence scores and decay over time if not confirmed.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple


DEFAULT_DB_PATH = Path.home() / ".motherlabs" / "history.db"

# Fact categories
CATEGORIES = frozenset({
    "preference",    # "I prefer X", "always use X"
    "decision",      # "let's use X", "we chose X"
    "capability",    # "Mother can X", learned abilities
    "project",       # project names, descriptions, states
    "person",        # people mentioned, roles, relationships
    "constraint",    # "never do X", "must always Y"
    "tool",          # tools, frameworks, libraries in use
    "pattern",       # recurring patterns, conventions
})

# Extraction patterns
_PREFERENCE_PATTERNS = re.compile(
    r"(?:i\s+(?:prefer|like|want|always|usually|never)\s+(.{5,80})"
    r"|(?:always|never)\s+(.{5,60})"
    r"|(?:my\s+(?:preferred|favorite|default)\s+(?:\w+\s+)?(?:is|are)\s+(.{3,60})))",
    re.IGNORECASE,
)

_DECISION_PATTERNS = re.compile(
    r"(?:(?:let'?s|we(?:'?ll|\s+should))\s+"
    r"(?:use|go\s+with|pick|switch\s+to|implement|build|create|enable|disable)\s+(.{5,80}))",
    re.IGNORECASE,
)

_CONSTRAINT_PATTERNS = re.compile(
    r"(?:(?:don'?t|do\s+not|never|must\s+not)\s+"
    r"(?:use|modify|change|delete|remove|touch|edit|add|create|push|commit|deploy|run)\s+(.{5,80})"
    r"|(?:must\s+always|always\s+(?:need|should|have)\s+to)\s+(.{5,60}))",
    re.IGNORECASE,
)

# Tool patterns: only high-precision matches (install commands, explicit integrations)
_TOOL_INSTALL_PATTERN = re.compile(
    r"(?:pip|npm|brew|cargo|go)\s+install\s+([a-zA-Z][\w.-]{2,40})",
    re.IGNORECASE,
)
_TOOL_INTEGRATION_PATTERN = re.compile(
    r"(?:switched?\s+to|integrated?\s+with|deployed?\s+(?:on|to)|configured)\s+"
    r"([A-Z][a-zA-Z]{2,30})",
)

_PERSON_PATTERNS = re.compile(
    r"(?:my|our)\s+(?:friend|colleague|boss|partner|co-founder|team\s*mate)\s+"
    r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KnowledgeFact:
    """A single discrete fact extracted from conversation."""

    fact_id: str              # Unique identifier
    category: str             # From CATEGORIES
    subject: str              # What the fact is about
    predicate: str            # The relationship or action
    value: str                # The fact content
    confidence: float         # 0.0-1.0 (decays over time if not confirmed)
    source: str               # "conversation:session_id" | "compilation:run_id"
    first_seen: float         # Unix timestamp
    last_confirmed: float     # Unix timestamp of last confirmation
    access_count: int         # How many times retrieved


def _normalize_fact_id(category: str, subject: str, value: str) -> str:
    """Generate a deterministic fact_id for deduplication."""
    norm = f"{category}:{subject}:{value}".lower()
    # Simple hash to keep IDs short
    norm = re.sub(r'[^a-z0-9:]+', '_', norm)
    return norm[:120]


def extract_facts(
    content: str,
    role: str,
    session_id: str = "",
    timestamp: float = 0.0,
) -> List[KnowledgeFact]:
    """Extract knowledge facts from a single message.

    Uses heuristic pattern matching — no LLM required.
    Only extracts from user messages (role='user') for preferences/constraints.
    Extracts from both roles for decisions, tools, projects.

    Returns list of KnowledgeFact (may be empty).
    """
    facts = []
    ts = timestamp or time.time()
    source = f"conversation:{session_id}" if session_id else "conversation:unknown"

    # Preferences — user messages only
    if role == "user":
        for match in _PREFERENCE_PATTERNS.finditer(content):
            groups = [g for g in match.groups() if g]
            if groups:
                value = groups[0].strip().rstrip(".,;:!?")
                if len(value) > 4:
                    fid = _normalize_fact_id("preference", "user", value)
                    facts.append(KnowledgeFact(
                        fact_id=fid,
                        category="preference",
                        subject="user",
                        predicate="prefers",
                        value=value,
                        confidence=0.8,
                        source=source,
                        first_seen=ts,
                        last_confirmed=ts,
                        access_count=0,
                    ))

    # Constraints — user messages only
    if role == "user":
        for match in _CONSTRAINT_PATTERNS.finditer(content):
            groups = [g for g in match.groups() if g]
            if groups:
                value = groups[0].strip().rstrip(".,;:!?")
                if len(value) > 4:
                    fid = _normalize_fact_id("constraint", "user", value)
                    facts.append(KnowledgeFact(
                        fact_id=fid,
                        category="constraint",
                        subject="user",
                        predicate="constrains",
                        value=value,
                        confidence=0.9,
                        source=source,
                        first_seen=ts,
                        last_confirmed=ts,
                        access_count=0,
                    ))

    # Decisions — both roles
    for match in _DECISION_PATTERNS.finditer(content):
        value = match.group(1).strip().rstrip(".,;:!?")
        if len(value) > 4:
            fid = _normalize_fact_id("decision", "session", value)
            facts.append(KnowledgeFact(
                fact_id=fid,
                category="decision",
                subject="session",
                predicate="decided",
                value=value,
                confidence=0.7,
                source=source,
                first_seen=ts,
                last_confirmed=ts,
                access_count=0,
            ))

    # Tools — both roles
    _TOOL_NOISE = {
        "the", "and", "for", "with", "from", "this", "that", "not", "you", "your",
        "our", "their", "its", "his", "her", "some", "any", "all", "what", "which",
        "who", "how", "when", "where", "why", "there", "here", "something", "anything",
        "everything", "nothing", "someone", "anyone", "everyone", "things", "stuff",
        "them", "those", "these", "other", "another", "more", "less", "most", "many",
        "much", "very", "really", "just", "only", "also", "too", "well", "good",
        "bad", "new", "old", "same", "different", "right", "wrong", "first", "last",
        "next", "each", "every", "both", "few", "several", "lot", "way", "time",
        "long", "sure", "about", "over", "into", "through", "after", "before", "but",
        "problems", "issues", "changes", "work", "working", "works", "make", "making",
        "made", "want", "need", "like", "think", "know", "see", "look", "come",
        "take", "give", "find", "tell", "ask", "try", "use", "keep", "let", "put",
        "set", "run", "read", "help", "show", "turn", "move", "play", "part",
        "point", "place", "case", "fact", "idea", "kind", "sort", "type", "bit",
        "full", "one", "two", "three", "four", "five", "ten", "first", "second",
        "start", "end", "top", "bottom", "left", "right", "same", "different",
        "real", "main", "own", "whole", "open", "close", "free", "public", "private",
        "local", "global", "simple", "complex", "small", "large", "big", "high", "low",
        "back", "down", "off", "out", "number", "name", "line", "file", "data",
        "code", "test", "build", "call", "check", "note", "form", "list", "text",
        "page", "view", "link", "home", "user", "account", "system", "state",
        "process", "task", "tasks", "event", "action", "result", "value", "error",
        "issue", "problem", "step", "level", "role", "rule", "command",
        "internet", "apps", "voice", "screen", "camera", "module", "function",
        "class", "method", "object", "table", "field", "column", "row",
        "window", "button", "input", "output", "server", "client", "port",
        "path", "directory", "folder", "api", "url", "key", "token", "base",
        "model", "config", "settings", "option", "flag", "mode", "tool",
        "feature", "service", "platform", "microphone", "human",
        "natural", "actual", "multiple", "consistent", "genuine", "better",
        "worse", "inside", "outside", "above", "below", "real", "descriptions",
        "language", "experience", "specific", "general", "basic", "advanced",
        "standard", "custom", "personal", "special", "common", "normal",
        "current", "previous", "next", "recent", "early", "late", "quick",
        "slow", "fast", "hard", "easy", "wrong", "correct", "true", "false",
        "possible", "available", "required", "important", "necessary",
        "customers", "whatever", "mic", "words",
    }
    # Combine results from both tool patterns
    _tool_matches = []
    for match in _TOOL_INSTALL_PATTERN.finditer(content):
        _tool_matches.append(match.group(1))
    for match in _TOOL_INTEGRATION_PATTERN.finditer(content):
        _tool_matches.append(match.group(1))
    for tool_raw in _tool_matches:
        tool = tool_raw.strip().rstrip(".,;:!?")
        # Filter noise: skip short words and common English words
        if len(tool) > 2 and tool.lower() not in _TOOL_NOISE:
            fid = _normalize_fact_id("tool", tool, "in_use")
            facts.append(KnowledgeFact(
                fact_id=fid,
                category="tool",
                subject=tool,
                predicate="in_use",
                value=f"Uses {tool}",
                confidence=0.6,
                source=source,
                first_seen=ts,
                last_confirmed=ts,
                access_count=0,
            ))

    # People — user messages only
    if role == "user":
        for match in _PERSON_PATTERNS.finditer(content):
            groups = [g for g in match.groups() if g]
            if groups:
                name = groups[0].strip()
                if len(name) > 1:
                    fid = _normalize_fact_id("person", name, "mentioned")
                    facts.append(KnowledgeFact(
                        fact_id=fid,
                        category="person",
                        subject=name,
                        predicate="mentioned",
                        value=f"Person: {name}",
                        confidence=0.5,
                        source=source,
                        first_seen=ts,
                        last_confirmed=ts,
                        access_count=0,
                    ))

    # Deduplicate within this extraction
    seen_ids = set()
    unique = []
    for f in facts:
        if f.fact_id not in seen_ids:
            seen_ids.add(f.fact_id)
            unique.append(f)

    return unique


def _ensure_knowledge_table(conn: sqlite3.Connection) -> None:
    """Create knowledge_facts table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            source TEXT NOT NULL DEFAULT '',
            first_seen REAL NOT NULL,
            last_confirmed REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_category
        ON knowledge_facts (category)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_subject
        ON knowledge_facts (subject)
    """)
    conn.commit()


def save_fact(fact: KnowledgeFact, db_path: Optional[Path] = None) -> int:
    """Persist a fact. Upserts by fact_id — confirms existing facts.

    On conflict: bumps confidence (min 1.0), updates last_confirmed,
    increments access_count.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        cursor = conn.execute(
            """INSERT INTO knowledge_facts
               (fact_id, category, subject, predicate, value, confidence,
                source, first_seen, last_confirmed, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fact_id) DO UPDATE SET
                   confidence = MIN(1.0, knowledge_facts.confidence + 0.1),
                   last_confirmed = excluded.last_confirmed,
                   access_count = knowledge_facts.access_count + 1
            """,
            (
                fact.fact_id,
                fact.category,
                fact.subject,
                fact.predicate,
                fact.value,
                fact.confidence,
                fact.source,
                fact.first_seen,
                fact.last_confirmed,
                fact.access_count,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_facts(facts: List[KnowledgeFact], db_path: Optional[Path] = None) -> int:
    """Persist multiple facts in a single transaction. Returns count saved."""
    if not facts:
        return 0
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        count = 0
        for fact in facts:
            conn.execute(
                """INSERT INTO knowledge_facts
                   (fact_id, category, subject, predicate, value, confidence,
                    source, first_seen, last_confirmed, access_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_id) DO UPDATE SET
                       confidence = MIN(1.0, knowledge_facts.confidence + 0.1),
                       last_confirmed = excluded.last_confirmed,
                       access_count = knowledge_facts.access_count + 1
                """,
                (
                    fact.fact_id,
                    fact.category,
                    fact.subject,
                    fact.predicate,
                    fact.value,
                    fact.confidence,
                    fact.source,
                    fact.first_seen,
                    fact.last_confirmed,
                    fact.access_count,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def _row_to_fact(row: Tuple) -> KnowledgeFact:
    """Convert a SQLite row to a KnowledgeFact."""
    return KnowledgeFact(
        fact_id=row[0],
        category=row[1],
        subject=row[2],
        predicate=row[3],
        value=row[4],
        confidence=row[5],
        source=row[6],
        first_seen=row[7],
        last_confirmed=row[8],
        access_count=row[9],
    )


def query_facts(
    subject: Optional[str] = None,
    category: Optional[str] = None,
    min_confidence: float = 0.3,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[KnowledgeFact]:
    """Query facts by subject and/or category."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        conditions = ["confidence >= ?"]
        params: list = [min_confidence]

        if subject:
            conditions.append("LOWER(subject) LIKE ?")
            params.append(f"%{subject.lower()}%")

        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions)
        params.append(limit)

        rows = conn.execute(
            f"""SELECT fact_id, category, subject, predicate, value,
                       confidence, source, first_seen, last_confirmed, access_count
                FROM knowledge_facts
                WHERE {where}
                ORDER BY confidence DESC, last_confirmed DESC
                LIMIT ?""",
            params,
        ).fetchall()
        return [_row_to_fact(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def search_facts(
    query: str,
    min_confidence: float = 0.3,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[KnowledgeFact]:
    """Search facts by keyword across subject and value fields."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words:
        return []

    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        conditions = []
        params: list = []
        for word in words[:5]:
            pattern = f"%{word}%"
            conditions.append(
                "(LOWER(subject) LIKE ? OR LOWER(value) LIKE ?)"
            )
            params.extend([pattern, pattern])

        where = " OR ".join(conditions)
        params.extend([min_confidence, limit])

        rows = conn.execute(
            f"""SELECT fact_id, category, subject, predicate, value,
                       confidence, source, first_seen, last_confirmed, access_count
                FROM knowledge_facts
                WHERE ({where}) AND confidence >= ?
                ORDER BY confidence DESC, last_confirmed DESC
                LIMIT ?""",
            params,
        ).fetchall()
        return [_row_to_fact(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def decay_stale_facts(
    max_age_days: float = 30.0,
    decay_rate: float = 0.1,
    db_path: Optional[Path] = None,
) -> int:
    """Decay confidence of facts not confirmed within max_age_days.

    Returns count of facts decayed. Facts below 0.1 confidence are deleted.
    """
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return 0

    cutoff = time.time() - (max_age_days * 86400)
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        # Decay stale facts
        conn.execute(
            """UPDATE knowledge_facts
               SET confidence = confidence - ?
               WHERE last_confirmed < ? AND confidence > 0.1""",
            (decay_rate, cutoff),
        )
        # Delete facts that dropped below threshold
        cursor = conn.execute(
            "DELETE FROM knowledge_facts WHERE confidence <= 0.1"
        )
        deleted = cursor.rowcount
        conn.commit()
        # Count remaining decayed
        decayed = conn.execute(
            "SELECT changes()"
        ).fetchone()
        return deleted
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def fact_count(db_path: Optional[Path] = None) -> int:
    """Count total facts stored."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return 0
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        _ensure_knowledge_table(conn)
        row = conn.execute("SELECT COUNT(*) FROM knowledge_facts").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def format_knowledge_context(
    facts: List[KnowledgeFact],
    max_tokens: int = 300,
) -> str:
    """Format facts as a context block for prompt injection.

    Returns a [KNOWN FACTS] block organized by category.
    """
    if not facts:
        return ""

    char_budget = max_tokens * 4
    lines = ["[KNOWN FACTS]"]
    used = len(lines[0])

    # Group by category
    by_category: Dict[str, List[KnowledgeFact]] = {}
    for f in facts:
        by_category.setdefault(f.category, []).append(f)

    # Priority order for display
    priority = ["preference", "constraint", "decision", "project", "tool", "person", "pattern", "capability"]

    for cat in priority:
        cat_facts = by_category.get(cat, [])
        if not cat_facts:
            continue

        header = f"  {cat.title()}:"
        if used + len(header) + 1 > char_budget:
            break
        lines.append(header)
        used += len(header) + 1

        for f in cat_facts[:5]:  # Max 5 per category
            line = f"    - {f.value} (conf={f.confidence:.1f})"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
