"""
Progress storage for live compilation view (Feature 6).

Stores stage-by-stage progress for async compilation tasks
so the frontend can show real-time pipeline advancement.

Extended: structured_insights + difficulty for glass-box compilation.
Extended again: escalations + human_decisions for a durable task ledger.
Extended again: termination_condition for stable guarded stop states.
"""

import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("motherlabs.worker.progress")

_PROGRESS_DB = os.path.join(
    os.environ.get("MOTHERLABS_DATA_DIR", os.path.expanduser("~/.motherlabs")),
    "progress.db",
)

_TOTAL_STAGES = 8  # queued, intent, persona, entity, process, synthesis, verify, materialize

STAGE_NAMES = [
    "queued",
    "intent_analysis",
    "persona_mapping",
    "entity_extraction",
    "process_modeling",
    "synthesis",
    "verification",
    "materialization",
]


def _get_connection() -> sqlite3.Connection:
    """Get SQLite connection, creating table on first use."""
    os.makedirs(os.path.dirname(_PROGRESS_DB), exist_ok=True)
    conn = sqlite3.connect(_PROGRESS_DB, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            task_id TEXT PRIMARY KEY,
            current_stage TEXT NOT NULL,
            stage_index INTEGER NOT NULL DEFAULT 0,
            total_stages INTEGER NOT NULL DEFAULT 8,
            insights TEXT NOT NULL DEFAULT '[]',
            updated_at REAL NOT NULL
        )
    """)
    conn.commit()
    # Idempotent schema migration: add glass-box + ledger columns
    _migrate_columns(conn)
    return conn


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add glass-box, ledger, and termination columns if missing."""
    try:
        cursor = conn.execute("PRAGMA table_info(progress)")
        existing = {row["name"] for row in cursor.fetchall()}
        if "structured_insights" not in existing:
            conn.execute("ALTER TABLE progress ADD COLUMN structured_insights TEXT DEFAULT '[]'")
        if "difficulty" not in existing:
            conn.execute("ALTER TABLE progress ADD COLUMN difficulty TEXT DEFAULT '{}'")
        if "escalations" not in existing:
            conn.execute("ALTER TABLE progress ADD COLUMN escalations TEXT DEFAULT '[]'")
        if "human_decisions" not in existing:
            conn.execute("ALTER TABLE progress ADD COLUMN human_decisions TEXT DEFAULT '[]'")
        if "termination_condition" not in existing:
            conn.execute("ALTER TABLE progress ADD COLUMN termination_condition TEXT DEFAULT '{}'")
        conn.commit()
    except Exception as e:
        logger.debug("Column migration skipped: %s", e)


def _merge_json_records(
    existing: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in [*existing, *incoming]:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def write_progress(
    task_id: str,
    stage: str,
    index: int,
    insight: str = "",
    structured_insight: Optional[Dict[str, Any]] = None,
    difficulty: Optional[Dict[str, Any]] = None,
    escalations: Optional[List[Dict[str, Any]]] = None,
    human_decisions: Optional[List[Dict[str, Any]]] = None,
    termination_condition: Optional[Dict[str, Any]] = None,
) -> None:
    """Write or update progress for a compilation task.

    Args:
        task_id: Huey task ID
        stage: Current stage name
        index: Stage index (0-based)
        insight: Optional flat insight text (backward compat)
        structured_insight: Optional StructuredInsight dict to append
        difficulty: Optional DifficultySignal snapshot (replaces stored value)
        escalations: Optional escalation ledger entries to persist
        human_decisions: Optional human decision ledger entries to persist
        termination_condition: Optional stop condition to persist
    """
    try:
        conn = _get_connection()
        try:
            # Read existing data
            row = conn.execute(
                """
                SELECT insights, structured_insights, difficulty, escalations, human_decisions, termination_condition
                FROM progress WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()

            if row:
                existing_insights = json.loads(row["insights"])
                existing_structured = json.loads(row["structured_insights"] or "[]")
                existing_difficulty = row["difficulty"] or "{}"
                existing_escalations = json.loads(row["escalations"] or "[]")
                existing_human_decisions = json.loads(row["human_decisions"] or "[]")
                existing_termination = row["termination_condition"] or "{}"
            else:
                existing_insights = []
                existing_structured = []
                existing_difficulty = "{}"
                existing_escalations = []
                existing_human_decisions = []
                existing_termination = "{}"

            if insight:
                existing_insights.append(insight)

            if structured_insight:
                existing_structured.append(structured_insight)

            difficulty_json = json.dumps(difficulty) if difficulty else existing_difficulty
            escalations_json = json.dumps(
                _merge_json_records(existing_escalations, escalations or [])
            )
            human_decisions_json = json.dumps(
                _merge_json_records(existing_human_decisions, human_decisions or [])
            )
            termination_json = json.dumps(termination_condition) if termination_condition else existing_termination

            conn.execute(
                """
                INSERT OR REPLACE INTO progress
                    (task_id, current_stage, stage_index, total_stages, insights,
                     structured_insights, difficulty, escalations, human_decisions, termination_condition, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, stage, index, _TOTAL_STAGES,
                    json.dumps(existing_insights),
                    json.dumps(existing_structured),
                    difficulty_json,
                    escalations_json,
                    human_decisions_json,
                    termination_json,
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.debug("Progress write failed (non-critical): %s", e)


def write_task_ledger(
    task_id: str,
    *,
    escalations: Optional[List[Dict[str, Any]]] = None,
    human_decisions: Optional[List[Dict[str, Any]]] = None,
    termination_condition: Optional[Dict[str, Any]] = None,
    stage: str = "complete",
    index: int = _TOTAL_STAGES - 1,
) -> None:
    """Persist ledger entries without needing a fresh progress event."""
    existing = read_progress(task_id) or {}
    write_progress(
        task_id=task_id,
        stage=str(existing.get("current_stage") or stage),
        index=int(existing.get("stage_index") or index),
        insight="",
        structured_insight=None,
        difficulty=existing.get("difficulty"),
        escalations=escalations,
        human_decisions=human_decisions,
        termination_condition=termination_condition,
    )


def append_human_decision(task_id: str, decision: Dict[str, Any]) -> None:
    """Append a single human decision to the task ledger."""
    write_task_ledger(task_id, human_decisions=[decision])


def write_task_termination(task_id: str, termination_condition: Dict[str, Any]) -> None:
    """Persist a stable stop condition for a task."""
    existing = read_progress(task_id) or {}
    write_task_ledger(
        task_id,
        termination_condition=termination_condition,
        stage=str(existing.get("current_stage") or "termination_guard"),
        index=int(existing.get("stage_index") or (_TOTAL_STAGES - 1)),
    )


def read_progress(task_id: str) -> Optional[Dict[str, Any]]:
    """Read current progress for a compilation task.

    Args:
        task_id: Huey task ID

    Returns:
        Dict with current_stage, stage_index, total_stages, insights,
        structured_insights, difficulty, escalations, human_decisions, termination_condition
        — or None if no progress recorded
    """
    try:
        conn = _get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM progress WHERE task_id = ?",
                (task_id,),
            ).fetchone()

            if not row:
                return None

            result = {
                "current_stage": row["current_stage"],
                "stage_index": row["stage_index"],
                "total_stages": row["total_stages"],
                "insights": json.loads(row["insights"]),
            }

            # Glass-box fields (graceful if columns missing on old DB)
            try:
                result["structured_insights"] = json.loads(row["structured_insights"] or "[]")
                result["difficulty"] = json.loads(row["difficulty"] or "{}")
                result["escalations"] = json.loads(row["escalations"] or "[]")
                result["human_decisions"] = json.loads(row["human_decisions"] or "[]")
                result["termination_condition"] = json.loads(row["termination_condition"] or "{}")
            except (KeyError, IndexError):
                result["structured_insights"] = []
                result["difficulty"] = {}
                result["escalations"] = []
                result["human_decisions"] = []
                result["termination_condition"] = {}

            return result
        finally:
            conn.close()
    except Exception:
        return None


def clear_progress(task_id: str) -> None:
    """Remove progress record after task completes."""
    try:
        conn = _get_connection()
        try:
            conn.execute("DELETE FROM progress WHERE task_id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
