"""
Mother executive — compiler-driven goal execution plans.

LEAF module. Stdlib + sqlite3 only. No imports from core/ or mother/.

A goal is intent. The compiler reduces intent to structure. Therefore:
the compiler applied to Mother's own goals produces the action plan.

GoalPlan and PlanStep persist the compiled plan. PlanStore manages
SQLite tables that coexist with history.db. classify_goal determines
whether a goal is compilable. extract_steps_from_blueprint reduces
a blueprint into an ordered step sequence.
"""

import json
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PlanStep:
    """A single executable step in a goal plan."""

    step_id: int = 0
    plan_id: int = 0
    position: int = 0
    name: str = ""
    description: str = ""
    action_type: str = ""        # "compile" | "build" | "file" | "search" | "reason" | "prepare"
    action_arg: str = ""
    status: str = "pending"      # "pending" | "in_progress" | "done" | "failed" | "skipped"
    result_note: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass(frozen=True)
class GoalPlan:
    """A compiled execution plan for a goal."""

    plan_id: int = 0
    goal_id: int = 0
    created_at: float = 0.0
    blueprint_json: str = ""     # provenance: full blueprint
    trust_score: float = 0.0
    status: str = "active"       # "active" | "executing" | "done" | "failed" | "abandoned"
    total_steps: int = 0
    completed_steps: int = 0
    steps: List[PlanStep] = field(default_factory=list)


_CREATE_PLANS = """
CREATE TABLE IF NOT EXISTS goal_plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    created_at REAL NOT NULL,
    blueprint_json TEXT NOT NULL DEFAULT '',
    trust_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    total_steps INTEGER NOT NULL DEFAULT 0,
    completed_steps INTEGER NOT NULL DEFAULT 0
)
"""

_CREATE_STEPS = """
CREATE TABLE IF NOT EXISTS plan_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    name TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    action_type TEXT NOT NULL DEFAULT 'reason',
    action_arg TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    result_note TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL DEFAULT 0.0,
    completed_at REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (plan_id) REFERENCES goal_plans(plan_id)
)
"""


# --- Classification heuristic ---

_BUILDABLE_VERBS = frozenset({
    "build", "create", "implement", "deploy", "develop", "design",
    "construct", "make", "set up", "setup", "scaffold", "generate",
    "write", "code", "program", "engineer", "add", "integrate",
    "launch", "ship", "configure", "install", "migrate", "automate",
    "acquire", "learn", "record", "monitor", "watch", "track", "capture",
    "fix", "improve", "reduce", "optimize", "enhance", "strengthen",
    "refactor", "tune", "patch", "upgrade", "resolve",
})

_BUILDABLE_NOUNS = frozenset({
    "app", "application", "system", "service", "api", "pipeline",
    "website", "site", "page", "dashboard", "server", "database",
    "tool", "bot", "script", "module", "component", "feature",
    "endpoint", "interface", "platform", "plugin", "extension",
    "workflow", "integration", "bridge", "cli", "ui", "frontend",
    "backend", "microservice", "webhook", "scheduler", "worker",
    "capability", "agent", "appendage", "recorder", "tracker", "monitor",
    "compiler", "engine", "synthesis", "verification", "fidelity",
    "rejection", "trust", "completeness", "consistency", "coherence",
    "compression", "entity", "quality", "threshold", "gate",
})


def classify_goal(description: str) -> str:
    """Classify a goal as 'compilable' or 'conversational'.

    Keyword heuristic. No LLM call. Checks for buildable verbs
    and buildable nouns in the description.
    """
    words = description.lower().split()
    word_set = frozenset(words)

    has_verb = bool(word_set & _BUILDABLE_VERBS)
    has_noun = bool(word_set & _BUILDABLE_NOUNS)

    # Also check two-word phrases for "set up"
    text_lower = description.lower()
    if "set up" in text_lower or "setup" in text_lower:
        has_verb = True

    if has_verb and has_noun:
        return "compilable"
    return "conversational"


# --- Self-build detection ---

_SELF_BUILD_SIGNALS = frozenset({
    "confidence", "coverage", "capability", "implement", "compiler",
    "kernel", "quality", "resilience", "quarantined", "strengthen",
    "self-improvement", "self-build", "grid", "postcode", "cell",
    "layer", "concern", "semantic", "entity", "behavior",
})


def _is_self_build_goal(description: str) -> bool:
    """Detect if a goal targets self-modification. 2+ signal words = True."""
    words = frozenset(description.lower().split())
    # Also check substrings for hyphenated tokens
    text_lower = description.lower()
    matches = sum(1 for sig in _SELF_BUILD_SIGNALS if sig in text_lower)
    return matches >= 2


# --- Blueprint step extraction ---



def _topological_order(
    components: List[Dict],
    relationships: List[Dict],
) -> List[Dict]:
    """Order components by dependency graph. Kahn's algorithm.

    Falls back to original order on cycle or empty relationships.
    """
    if not relationships or not components:
        return list(components)

    # Build name -> component map
    name_map = {}
    for c in components:
        name = c.get("name", "")
        if name:
            name_map[name] = c

    # Build adjacency + in-degree
    in_degree: Dict[str, int] = {c.get("name", ""): 0 for c in components}
    adjacency: Dict[str, List[str]] = {c.get("name", ""): [] for c in components}

    for rel in relationships:
        source = rel.get("source", rel.get("from", ""))
        target = rel.get("target", rel.get("to", ""))
        if source in adjacency and target in in_degree:
            adjacency[source].append(target)
            in_degree[target] = in_degree.get(target, 0) + 1

    # Kahn's algorithm
    queue: deque = deque()
    for name, deg in in_degree.items():
        if deg == 0:
            queue.append(name)

    ordered_names: List[str] = []
    while queue:
        name = queue.popleft()
        ordered_names.append(name)
        for neighbor in adjacency.get(name, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Cycle detection: if not all nodes visited, fall back
    if len(ordered_names) != len(components):
        return list(components)

    # Map back to components
    result = []
    for name in ordered_names:
        if name in name_map:
            result.append(name_map[name])
    return result


def extract_steps_from_blueprint(
    blueprint: Dict,
    goal_description: str = "",
) -> List[Dict]:
    """Reduce blueprint to executable plan steps.

    Produces steps that map to Mother's actual action vocabulary:
    compile, build, search, file, goal_done. Each action_arg is
    a natural language description that _run_compile/_run_build
    can consume directly.

    Two strategies:
    - Build plan (default): compile → build → goal_done
    - Multi-phase: per-subsystem compile+build when blueprint has
      multiple components with sub_blueprint fields
    """
    components = blueprint.get("components", [])
    if not components:
        return []

    # Check for subsystems (nested blueprints)
    subsystems = [c for c in components if c.get("sub_blueprint")]

    if subsystems and len(subsystems) > 1:
        return _multi_phase_plan(subsystems, goal_description)

    return _build_plan(blueprint, goal_description)


_DESTRUCTIVE_KEYWORDS = frozenset({
    "delete", "drop", "remove", "purge", "migrate", "overwrite",
    "destroy", "truncate", "reset", "wipe", "erase", "rollback",
    "force", "revert",
})
_MODIFY_KEYWORDS = frozenset({
    "update", "modify", "change", "alter", "replace", "patch",
    "rename", "move", "merge",
})


_ONE_WAY_KEYWORDS = frozenset({
    "delete", "drop", "purge", "destroy", "truncate", "wipe",
    "erase", "deploy", "publish", "send", "migrate", "ship",
    "release", "push", "broadcast",
})
_TWO_WAY_KEYWORDS = frozenset({
    "test", "draft", "review", "plan", "preview", "simulate",
    "check", "verify", "validate", "lint", "scan", "analyze",
    "prepare", "stage", "sketch",
})


def estimate_step_risk(step_description: str) -> str:
    """Estimate consequence level of a plan step. Pure function.
    Returns "high", "medium", or "low".
    """
    words = frozenset(step_description.lower().split())
    if words & _DESTRUCTIVE_KEYWORDS:
        return "high"
    if words & _MODIFY_KEYWORDS:
        return "medium"
    return "low"


def classify_reversibility(action_description: str) -> str:
    """Classify action as 'one-way' or 'two-way' door. Pure function.
    One-way: destructive/deployment actions that can't easily be undone.
    Two-way: review/test/draft actions that are safe to retry.
    """
    words = frozenset(action_description.lower().split())
    if words & _ONE_WAY_KEYWORDS:
        return "one-way"
    if words & _TWO_WAY_KEYWORDS:
        return "two-way"
    return "two-way"


def _needs_preparation(blueprint: Dict, goal_description: str) -> bool:
    """Heuristic: does this goal need a readiness-staging step?

    True when the goal involves research, data gathering, or integration
    with external resources that should be resolved before compilation.
    """
    text = (goal_description + " " + blueprint.get("description", "")).lower()
    prep_signals = frozenset({
        "research", "gather", "collect", "fetch", "download", "import",
        "investigate", "analyze", "audit", "review", "survey", "explore",
        "integrate", "migrate", "connect", "sync", "ingest", "load",
    })
    words = frozenset(text.split())
    return bool(words & prep_signals)


def _build_plan(blueprint: Dict, goal_description: str) -> List[Dict]:
    """Standard plan: [prepare →] compile → build/self_build → verify.

    Inserts a readiness-staging step when the goal involves material
    that needs pre-resolution (research, data, integration).
    Uses self_build action_type when the goal targets self-modification.
    The engine handles component decomposition internally.
    The build loop handles validation and fixes.
    """
    desc = goal_description or blueprint.get("description", "project")
    steps: List[Dict] = []

    # Readiness staging: prepare materials before compile
    if _needs_preparation(blueprint, desc):
        steps.append({
            "name": "prepare",
            "description": f"Stage required materials and resolve dependencies for: {desc}",
            "action_type": "prepare",
            "action_arg": desc,
        })

    # Determine build action type
    build_action = "self_build" if _is_self_build_goal(desc) else "build"

    steps.extend([
        {
            "name": "compile",
            "description": f"Run full compilation pipeline on: {desc}",
            "action_type": "compile",
            "action_arg": desc,
        },
        {
            "name": "build",
            "description": f"Build compiled blueprint into project: {desc}",
            "action_type": build_action,
            "action_arg": desc,
        },
        {
            "name": "verify",
            "description": "Verify output and mark goal complete",
            "action_type": "goal_done",
            "action_arg": "",
        },
    ])
    return steps


def _multi_phase_plan(
    subsystems: List[Dict],
    goal_description: str,
) -> List[Dict]:
    """Per-subsystem compile+build, then final verify step.

    Each subsystem with a sub_blueprint becomes a build unit.
    Ordering uses topological sort on subsystem relationships.
    """
    steps: List[Dict] = []
    for i, sub in enumerate(subsystems):
        name = sub.get("name", f"subsystem_{i}")
        desc = sub.get("description", name)

        steps.append({
            "name": f"compile_{name}",
            "description": f"Compile subsystem: {desc}",
            "action_type": "compile",
            "action_arg": desc,
        })
        steps.append({
            "name": f"build_{name}",
            "description": f"Build subsystem: {desc}",
            "action_type": "build",
            "action_arg": desc,
        })

    steps.append({
        "name": "verify",
        "description": "Verify all subsystems built and mark goal complete",
        "action_type": "goal_done",
        "action_arg": "",
    })
    return steps


# --- PlanStore ---

class PlanStore:
    """SQLite-backed plan store. Coexists with history.db tables."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_PLANS)
        self._conn.execute(_CREATE_STEPS)
        self._conn.commit()

    def create_plan(
        self,
        goal_id: int,
        blueprint_json: str,
        trust_score: float,
        steps: List[Dict],
    ) -> int:
        """Create a plan with steps. Returns plan_id."""
        now = time.time()
        cursor = self._conn.execute(
            """INSERT INTO goal_plans
               (goal_id, created_at, blueprint_json, trust_score,
                status, total_steps, completed_steps)
               VALUES (?, ?, ?, ?, 'active', ?, 0)""",
            (goal_id, now, blueprint_json, trust_score, len(steps)),
        )
        plan_id = cursor.lastrowid

        for i, step in enumerate(steps):
            self._conn.execute(
                """INSERT INTO plan_steps
                   (plan_id, position, name, description,
                    action_type, action_arg, status)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    plan_id,
                    i,
                    step.get("name", ""),
                    step.get("description", ""),
                    step.get("action_type", "reason"),
                    step.get("action_arg", ""),
                ),
            )

        self._conn.commit()
        return plan_id

    def get_plan_for_goal(self, goal_id: int) -> Optional[GoalPlan]:
        """Get the active/executing plan for a goal. Returns None if none."""
        row = self._conn.execute(
            """SELECT * FROM goal_plans
               WHERE goal_id = ? AND status IN ('active', 'executing')
               ORDER BY created_at DESC LIMIT 1""",
            (goal_id,),
        ).fetchone()
        if not row:
            return None

        plan = self._row_to_plan(row)
        steps = self._get_steps(plan.plan_id)
        return GoalPlan(
            plan_id=plan.plan_id,
            goal_id=plan.goal_id,
            created_at=plan.created_at,
            blueprint_json=plan.blueprint_json,
            trust_score=plan.trust_score,
            status=plan.status,
            total_steps=plan.total_steps,
            completed_steps=plan.completed_steps,
            steps=steps,
        )

    def next_step(self, plan_id: int) -> Optional[PlanStep]:
        """Get next pending step, or stale in_progress step (>5 min)."""
        row = self._conn.execute(
            """SELECT * FROM plan_steps
               WHERE plan_id = ? AND status = 'pending'
               ORDER BY position ASC LIMIT 1""",
            (plan_id,),
        ).fetchone()
        if row:
            return self._row_to_step(row)

        # Fallback: stale in_progress steps (async work likely finished/failed)
        stale_cutoff = time.time() - 300  # 5 minutes
        row = self._conn.execute(
            """SELECT * FROM plan_steps
               WHERE plan_id = ? AND status = 'in_progress'
                 AND started_at > 0 AND started_at < ?
               ORDER BY position ASC LIMIT 1""",
            (plan_id, stale_cutoff),
        ).fetchone()
        return self._row_to_step(row) if row else None

    def update_step(
        self,
        step_id: int,
        status: str,
        result_note: str = "",
    ) -> None:
        """Update step status and optional result note."""
        now = time.time()
        if status == "in_progress":
            self._conn.execute(
                """UPDATE plan_steps
                   SET status = ?, started_at = ?
                   WHERE step_id = ?""",
                (status, now, step_id),
            )
        elif status in ("done", "failed", "skipped"):
            self._conn.execute(
                """UPDATE plan_steps
                   SET status = ?, result_note = ?, completed_at = ?
                   WHERE step_id = ?""",
                (status, result_note, now, step_id),
            )
        else:
            self._conn.execute(
                """UPDATE plan_steps SET status = ? WHERE step_id = ?""",
                (status, step_id),
            )
        self._conn.commit()

    def update_plan_progress(self, plan_id: int) -> None:
        """Recompute completed_steps and update plan status."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM plan_steps WHERE plan_id = ? AND status = 'done'",
            (plan_id,),
        ).fetchone()
        completed = row[0] if row else 0

        total_row = self._conn.execute(
            "SELECT total_steps FROM goal_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        total = total_row[0] if total_row else 0

        status = "executing"
        if completed >= total and total > 0:
            status = "done"

        # Check for failures
        failed_row = self._conn.execute(
            "SELECT COUNT(*) FROM plan_steps WHERE plan_id = ? AND status = 'failed'",
            (plan_id,),
        ).fetchone()
        if failed_row and failed_row[0] > 0:
            # If all remaining are failed, mark plan as failed
            pending_row = self._conn.execute(
                "SELECT COUNT(*) FROM plan_steps WHERE plan_id = ? AND status = 'pending'",
                (plan_id,),
            ).fetchone()
            if pending_row and pending_row[0] == 0 and completed < total:
                status = "failed"

        self._conn.execute(
            "UPDATE goal_plans SET completed_steps = ?, status = ? WHERE plan_id = ?",
            (completed, status, plan_id),
        )
        self._conn.commit()

    def abandon_plan(self, plan_id: int) -> None:
        """Mark a plan as abandoned."""
        self._conn.execute(
            "UPDATE goal_plans SET status = 'abandoned' WHERE plan_id = ?",
            (plan_id,),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _get_steps(self, plan_id: int) -> List[PlanStep]:
        """Get all steps for a plan, ordered by position."""
        rows = self._conn.execute(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY position ASC",
            (plan_id,),
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    @staticmethod
    def _row_to_plan(row) -> GoalPlan:
        return GoalPlan(
            plan_id=row[0],
            goal_id=row[1],
            created_at=row[2],
            blueprint_json=row[3],
            trust_score=row[4],
            status=row[5],
            total_steps=row[6],
            completed_steps=row[7],
        )

    @staticmethod
    def _row_to_step(row) -> PlanStep:
        return PlanStep(
            step_id=row[0],
            plan_id=row[1],
            position=row[2],
            name=row[3],
            description=row[4],
            action_type=row[5],
            action_arg=row[6],
            status=row[7],
            result_note=row[8],
            started_at=row[9],
            completed_at=row[10],
        )


# --- Risk Register (#59) ---

@dataclass(frozen=True)
class RiskEntry:
    """A persistent risk entry."""
    risk_id: int = 0
    timestamp: float = 0.0
    description: str = ""
    severity: str = "medium"
    source: str = ""
    status: str = "active"
    mitigation: str = ""
    goal_id: int = 0


_CREATE_RISK_TABLE = """
CREATE TABLE IF NOT EXISTS risk_register (
    risk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    severity TEXT NOT NULL DEFAULT 'medium',
    source TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    mitigation TEXT NOT NULL DEFAULT '',
    goal_id INTEGER NOT NULL DEFAULT 0
)
"""

_RISK_CRITICAL = frozenset({"data loss", "security breach", "production down", "pii exposed", "credential leak"})
_RISK_HIGH = frozenset({"delete", "destroy", "deploy", "migrate", "credential", "overwrite", "purge"})
_RISK_MEDIUM = frozenset({"modify", "update", "integration", "external", "permission", "access"})


def classify_risk_severity(description: str) -> str:
    """Classify risk severity from description. Pure function."""
    lower = description.lower()
    for phrase in _RISK_CRITICAL:
        if phrase in lower:
            return "critical"
    words = frozenset(lower.split())
    if words & _RISK_HIGH:
        return "high"
    if words & _RISK_MEDIUM:
        return "medium"
    return "low"


class RiskRegister:
    """SQLite-backed risk register. Follows GoalStore pattern."""

    def __init__(self, db_path):
        self._db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_RISK_TABLE)
        self._conn.commit()

    def add_risk(self, description: str, severity: str = "medium",
                 source: str = "", goal_id: int = 0) -> int:
        ts = time.time()
        cursor = self._conn.execute(
            """INSERT INTO risk_register
               (timestamp, description, severity, source, status, mitigation, goal_id)
               VALUES (?, ?, ?, ?, 'active', '', ?)""",
            (ts, description, severity, source, goal_id),
        )
        self._conn.commit()
        return cursor.lastrowid

    def active_risks(self, limit: int = 20):
        rows = self._conn.execute(
            """SELECT * FROM risk_register
               WHERE status = 'active'
               ORDER BY
                 CASE severity
                   WHEN 'critical' THEN 0
                   WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2
                   WHEN 'low' THEN 3
                   ELSE 4
                 END,
                 timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def update_status(self, risk_id: int, status: str, mitigation: str = "") -> None:
        self._conn.execute(
            "UPDATE risk_register SET status = ?, mitigation = ? WHERE risk_id = ?",
            (status, mitigation, risk_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entry(row) -> 'RiskEntry':
        return RiskEntry(
            risk_id=row[0], timestamp=row[1], description=row[2],
            severity=row[3], source=row[4], status=row[5],
            mitigation=row[6], goal_id=row[7],
        )
