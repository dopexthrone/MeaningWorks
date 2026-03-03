"""
Mother goals — persistent store for actionable goals.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

Goals survive across sessions. Mother wakes up with pending work.
Priority-ordered, status-tracked, source-tagged. Follows the
mother/idea_journal.py pattern exactly.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class Goal:
    """A persistent, actionable goal."""

    goal_id: int = 0
    timestamp: float = 0.0
    description: str = ""
    source: str = "user"            # "user" | "mother" | "system"
    priority: str = "normal"        # "urgent" | "high" | "normal" | "low"
    status: str = "active"          # "active" | "in_progress" | "done" | "dismissed"
    progress_note: str = ""
    last_worked: float = 0.0
    completion_note: str = ""
    engagement_count: int = 0
    redirect_count: int = 0
    stall_count: int = 0         # consecutive ticks with no action emitted
    attempt_count: int = 0       # total compilation/action attempts
    due_timestamp: float = 0.0   # 0 = no deadline


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS goals (
    goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'active',
    progress_note TEXT NOT NULL DEFAULT '',
    last_worked REAL NOT NULL DEFAULT 0.0,
    completion_note TEXT NOT NULL DEFAULT '',
    engagement_count INTEGER NOT NULL DEFAULT 0,
    redirect_count INTEGER NOT NULL DEFAULT 0,
    stall_count INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0
)
"""

_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}


def compute_goal_health(goal: Goal, now: float = 0.0) -> float:
    """Pure health score [0.0-1.0]. Decays with age, inactivity, redirects."""
    if now <= 0.0:
        now = time.time()

    age_hours = (now - goal.timestamp) / 3600
    age_decay = max(0.0, 1.0 - (age_hours / (7 * 24)))  # Linear decay over 7 days

    if goal.last_worked > 0:
        idle_hours = (now - goal.last_worked) / 3600
    else:
        idle_hours = age_hours
    idle_decay = max(0.0, 1.0 - (idle_hours / 48))  # Stale after 48h no work

    redirect_penalty = max(0.0, 1.0 - (goal.redirect_count * 0.2))
    engagement_bonus = min(0.3, goal.engagement_count * 0.05)
    stall_decay = max(0.0, 1.0 - (goal.stall_count * 0.25))

    # Deadline pressure: overdue goals get urgency boost (health stays high)
    deadline_boost = 0.0
    if goal.due_timestamp > 0:
        seconds_left = goal.due_timestamp - now
        if seconds_left <= 0:
            deadline_boost = 0.3  # Overdue — keep health high to prevent pruning
        else:
            days_left = seconds_left / 86400
            if days_left < 3:
                deadline_boost = 0.2 * (1.0 - days_left / 3)  # Increasing urgency

    raw = (age_decay * 0.3 + idle_decay * 0.4 + redirect_penalty * 0.3) + engagement_bonus + deadline_boost
    return max(0.0, min(1.0, raw * stall_decay))


_ANTONYMS = {
    "build": "remove", "add": "remove", "enable": "disable",
    "increase": "decrease", "expand": "reduce", "create": "delete",
    "start": "stop", "open": "close", "scale": "shrink",
}
_CONFLICT_STOPWORDS = {"the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "is", "it"}


def goal_dedup_key(description: str) -> str:
    """Normalize a goal description for deduplication.

    Strips all numbers and percentages so "5 quarantined cells" and
    "71 quarantined cells" produce the same key. Used by daemon, bridge,
    and anywhere goals are persisted to prevent duplicates.
    """
    import re
    return re.sub(r'\d+[\.\d]*%?', '', description.lower()).strip()


def detect_goal_conflicts(goals) -> list:
    """Detect conflicting goals via antonym and overlap heuristics. Pure function."""
    conflicts = []
    desc_words = {}
    for g in goals:
        desc_words[g.goal_id] = set(g.description.lower().split()) - _CONFLICT_STOPWORDS
    goal_list = list(goals)
    for i, ga in enumerate(goal_list):
        for gb in goal_list[i + 1:]:
            wa, wb = desc_words[ga.goal_id], desc_words[gb.goal_id]
            overlap = wa & wb
            found_antonym = False
            for word in wa:
                antonym = _ANTONYMS.get(word)
                if antonym and antonym in wb:
                    conflicts.append({"goal_a": ga.goal_id, "goal_b": gb.goal_id,
                                      "reason": f"'{word}' vs '{antonym}'"})
                    found_antonym = True
                    break
            if not found_antonym and len(overlap) >= 3:
                conflicts.append({"goal_a": ga.goal_id, "goal_b": gb.goal_id,
                                  "reason": f"overlap: {', '.join(list(overlap)[:3])}"})
    return conflicts


def detect_goal_bias(goals) -> list:
    """Detect cognitive biases in goal distribution. Pure function."""
    if len(goals) < 3:
        return []
    biases = []
    n = len(goals)
    # Urgency bias
    urgent_count = sum(1 for g in goals if g.priority in ("urgent", "high"))
    if urgent_count / n > 0.6:
        biases.append(f"Urgency bias: {urgent_count}/{n} goals are urgent/high — everything can't be urgent")
    # Source bias
    from collections import Counter
    source_counts = Counter(g.source for g in goals)
    top_source, top_count = source_counts.most_common(1)[0]
    if top_count / n > 0.8 and n >= 4:
        biases.append(f"Source bias: {top_count}/{n} goals from '{top_source}' — consider diversifying")
    # Stagnation
    zero_engagement = sum(1 for g in goals if g.engagement_count == 0)
    if zero_engagement / n > 0.5 and n >= 4:
        biases.append(f"Stagnation: {zero_engagement}/{n} goals have zero engagement — still relevant?")
    return biases[:2]


def batch_compatible_goals(goals, max_batch: int = 3) -> list:
    """Group compatible goals for batch compilation. Pure function.
    Compatible = same priority, same source, no word overlap (independent).
    Returns list of goal groups (each group is a list of goals).
    """
    if len(goals) < 2:
        return [goals] if goals else []
    # Sort by priority for grouping
    _prio_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
    sorted_goals = sorted(goals, key=lambda g: _prio_order.get(g.priority, 4))
    batches = []
    used = set()
    for i, ga in enumerate(sorted_goals):
        if ga.goal_id in used:
            continue
        batch = [ga]
        used.add(ga.goal_id)
        wa = set(ga.description.lower().split())
        for gb in sorted_goals[i + 1:]:
            if gb.goal_id in used:
                continue
            if gb.priority != ga.priority:
                continue
            wb = set(gb.description.lower().split())
            if len(wa & wb) <= 2 and len(batch) < max_batch:
                batch.append(gb)
                used.add(gb.goal_id)
                wa |= wb
        batches.append(batch)
    return batches


class GoalStore:
    """SQLite-backed goal store. Stores in existing history.db."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        # Add columns if missing (safe idempotent migration)
        try:
            self._conn.execute(
                "ALTER TABLE goals ADD COLUMN engagement_count INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE goals ADD COLUMN redirect_count INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE goals ADD COLUMN stall_count INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE goals ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE goals ADD COLUMN due_timestamp REAL NOT NULL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            pass

    def add(
        self,
        description: str,
        source: str = "user",
        priority: str = "normal",
        due_timestamp: float = 0.0,
        dedup: bool = False,
    ) -> int:
        """Record a new goal. Returns the goal_id, or -1 if duplicate (when dedup=True)."""
        if dedup:
            key = goal_dedup_key(description)
            existing = self.active(limit=50)
            for g in existing:
                if goal_dedup_key(g.description) == key:
                    return -1
        ts = time.time()
        cursor = self._conn.execute(
            """INSERT INTO goals
               (timestamp, description, source, priority, status,
                progress_note, last_worked, completion_note,
                engagement_count, redirect_count, stall_count, attempt_count,
                due_timestamp)
               VALUES (?, ?, ?, ?, 'active', '', 0.0, '', 0, 0, 0, 0, ?)""",
            (ts, description, source, priority, due_timestamp),
        )
        self._conn.commit()
        return cursor.lastrowid

    def active(self, limit: int = 20) -> List[Goal]:
        """Return active goals, ordered by priority then recency."""
        rows = self._conn.execute(
            """SELECT * FROM goals
               WHERE status IN ('active', 'in_progress')
               ORDER BY
                 CASE priority
                   WHEN 'urgent' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'normal' THEN 2
                   WHEN 'low' THEN 3
                   ELSE 4
                 END,
                 timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def update_status(
        self,
        goal_id: int,
        status: str,
        progress_note: str = "",
        completion_note: str = "",
    ) -> None:
        """Update status and optional notes. Touches last_worked."""
        now = time.time()
        self._conn.execute(
            """UPDATE goals
               SET status = ?, progress_note = ?, completion_note = ?,
                   last_worked = ?
               WHERE goal_id = ?""",
            (status, progress_note, completion_note, now, goal_id),
        )
        self._conn.commit()

    def next_actionable(self, max_attempts: int = 5) -> Optional[Goal]:
        """Return highest-priority active goal not recently worked.

        Prefers goals never worked (last_worked=0), then oldest last_worked.
        Skips goals that have exceeded max_attempts to prevent infinite cycling.
        """
        row = self._conn.execute(
            """SELECT * FROM goals
               WHERE status IN ('active', 'in_progress')
                 AND attempt_count < ?
               ORDER BY
                 CASE priority
                   WHEN 'urgent' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'normal' THEN 2
                   WHEN 'low' THEN 3
                   ELSE 4
                 END,
                 last_worked ASC,
                 timestamp DESC
               LIMIT 1""",
            (max_attempts,),
        ).fetchone()
        return self._row_to_goal(row) if row else None

    def next_actionable_safe(self, max_attempts: int = 5) -> Optional[Goal]:
        """Return highest-priority active goal that doesn't conflict with in-progress work.

        Iterates candidates from next_actionable logic, skips any that conflict
        with currently in-progress goals via detect_goal_conflicts().
        """
        in_progress = self._conn.execute(
            "SELECT * FROM goals WHERE status = 'in_progress'"
        ).fetchall()
        in_progress_goals = [self._row_to_goal(r) for r in in_progress]

        rows = self._conn.execute(
            """SELECT * FROM goals
               WHERE status IN ('active', 'in_progress')
                 AND attempt_count < ?
               ORDER BY
                 CASE priority
                   WHEN 'urgent' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'normal' THEN 2
                   WHEN 'low' THEN 3
                   ELSE 4
                 END,
                 last_worked ASC,
                 timestamp DESC
               LIMIT 20""",
            (max_attempts,),
        ).fetchall()
        candidates = [self._row_to_goal(r) for r in rows]

        for candidate in candidates:
            if candidate.status == "in_progress":
                continue
            if not in_progress_goals:
                return candidate
            conflicts = detect_goal_conflicts([candidate] + in_progress_goals)
            involves_candidate = any(
                c["goal_a"] == candidate.goal_id or c["goal_b"] == candidate.goal_id
                for c in conflicts
            )
            if not involves_candidate:
                return candidate
        return None

    def count_active(self) -> int:
        """Return count of active + in_progress goals."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM goals WHERE status IN ('active', 'in_progress')"
        ).fetchone()
        return row[0] if row else 0

    def get(self, goal_id: int) -> Optional[Goal]:
        """Get a single goal by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM goals WHERE goal_id = ?",
            (goal_id,),
        ).fetchone()
        return self._row_to_goal(row) if row else None

    def all_goals(self, limit: int = 50) -> List[Goal]:
        """Return all goals, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM goals ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def increment_engagement(self, goal_id: int) -> None:
        """Record user engagement with a goal."""
        self._conn.execute(
            "UPDATE goals SET engagement_count = engagement_count + 1 WHERE goal_id = ?",
            (goal_id,),
        )
        self._conn.commit()

    def increment_redirect(self, goal_id: int) -> None:
        """Record user redirecting away from a goal."""
        self._conn.execute(
            "UPDATE goals SET redirect_count = redirect_count + 1 WHERE goal_id = ?",
            (goal_id,),
        )
        self._conn.commit()

    def increment_stall(self, goal_id: int) -> int:
        """Bump stall_count, touch last_worked. Returns new count."""
        now = time.time()
        self._conn.execute(
            "UPDATE goals SET stall_count = stall_count + 1, last_worked = ? WHERE goal_id = ?",
            (now, goal_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT stall_count FROM goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        return row[0] if row else 0

    def increment_attempt(self, goal_id: int) -> int:
        """Bump attempt_count, touch last_worked. Returns new count."""
        now = time.time()
        self._conn.execute(
            "UPDATE goals SET attempt_count = attempt_count + 1, last_worked = ? WHERE goal_id = ?",
            (now, goal_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT attempt_count FROM goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        return row[0] if row else 0

    def reset_stall(self, goal_id: int) -> None:
        """Zero stall_count — progress was made."""
        self._conn.execute(
            "UPDATE goals SET stall_count = 0 WHERE goal_id = ?",
            (goal_id,),
        )
        self._conn.commit()

    def score_and_prune(self, threshold: float = 0.1) -> int:
        """Auto-dismiss goals with health below threshold. Returns count dismissed."""
        now = time.time()
        goals = self.active()
        dismissed = 0
        for g in goals:
            if compute_goal_health(g, now) < threshold:
                self.update_status(g.goal_id, "dismissed", progress_note="Auto-dismissed: stale")
                dismissed += 1
        return dismissed

    def approve_goals(self, goal_ids: List[int]) -> int:
        """Batch-set goals to 'approved' status. Returns count updated."""
        if not goal_ids:
            return 0
        now = time.time()
        updated = 0
        for gid in goal_ids:
            cursor = self._conn.execute(
                """UPDATE goals SET status = 'approved', last_worked = ?
                   WHERE goal_id = ? AND status IN ('active', 'in_progress')""",
                (now, gid),
            )
            updated += cursor.rowcount
        self._conn.commit()
        return updated

    def approved(self, limit: int = 20) -> List[Goal]:
        """Return all goals with status 'approved', ordered by priority."""
        rows = self._conn.execute(
            """SELECT * FROM goals
               WHERE status = 'approved'
               ORDER BY
                 CASE priority
                   WHEN 'urgent' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'normal' THEN 2
                   WHEN 'low' THEN 3
                   ELSE 4
                 END,
                 timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def batch_for_window(self, limit: int = 10) -> List[Goal]:
        """Return approved goals sorted by priority, capped at limit.

        Used by the daemon to pull approved goals during build window.
        """
        return self.approved(limit=limit)

    def pending_briefing_goals(self, since_timestamp: float = 0.0) -> List[Goal]:
        """Return active goals not yet briefed (accumulated since timestamp).

        If since_timestamp is 0, returns all active goals.
        """
        if since_timestamp > 0:
            rows = self._conn.execute(
                """SELECT * FROM goals
                   WHERE status = 'active' AND timestamp > ?
                   ORDER BY
                     CASE priority
                       WHEN 'urgent' THEN 0
                       WHEN 'high' THEN 1
                       WHEN 'normal' THEN 2
                       WHEN 'low' THEN 3
                       ELSE 4
                     END,
                     timestamp DESC""",
                (since_timestamp,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM goals
                   WHERE status = 'active'
                   ORDER BY
                     CASE priority
                       WHEN 'urgent' THEN 0
                       WHEN 'high' THEN 1
                       WHEN 'normal' THEN 2
                       WHEN 'low' THEN 3
                       ELSE 4
                     END,
                     timestamp DESC"""
            ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_goal(row) -> Goal:
        """Convert a database row tuple to Goal. Handles 9/11/13/14-col rows."""
        if len(row) >= 14:
            return Goal(
                goal_id=row[0],
                timestamp=row[1],
                description=row[2],
                source=row[3],
                priority=row[4],
                status=row[5],
                progress_note=row[6],
                last_worked=row[7],
                completion_note=row[8],
                engagement_count=row[9],
                redirect_count=row[10],
                stall_count=row[11],
                attempt_count=row[12],
                due_timestamp=row[13],
            )
        if len(row) >= 13:
            return Goal(
                goal_id=row[0],
                timestamp=row[1],
                description=row[2],
                source=row[3],
                priority=row[4],
                status=row[5],
                progress_note=row[6],
                last_worked=row[7],
                completion_note=row[8],
                engagement_count=row[9],
                redirect_count=row[10],
                stall_count=row[11],
                attempt_count=row[12],
            )
        if len(row) >= 11:
            return Goal(
                goal_id=row[0],
                timestamp=row[1],
                description=row[2],
                source=row[3],
                priority=row[4],
                status=row[5],
                progress_note=row[6],
                last_worked=row[7],
                completion_note=row[8],
                engagement_count=row[9],
                redirect_count=row[10],
            )
        return Goal(
            goal_id=row[0],
            timestamp=row[1],
            description=row[2],
            source=row[3],
            priority=row[4],
            status=row[5],
            progress_note=row[6],
            last_worked=row[7],
            completion_note=row[8],
        )
