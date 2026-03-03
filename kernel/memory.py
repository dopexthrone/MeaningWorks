"""
kernel/memory.py — Episodic memory consolidation for compilation history.

LEAF module. After each compile, persists a structured MemoryRecord to
~/.motherlabs/maps.db (memory_records table). Enables L2 pattern detection
from accumulated compilation experience.

No imports from core/ or mother/. Uses only stdlib + kernel types.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Database location (shared with kernel/store.py)
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".motherlabs"
_DEFAULT_DB_NAME = "maps.db"


def _db_path(db_dir: Optional[Path] = None) -> Path:
    d = db_dir or _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_DB_NAME


_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    timestamp REAL NOT NULL,
    intent_summary TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT 'software',
    trust_score REAL NOT NULL DEFAULT 0.0,
    fidelity_score REAL NOT NULL DEFAULT 0.0,
    compression_losses_json TEXT NOT NULL DEFAULT '{}',
    gate_results_json TEXT NOT NULL DEFAULT '{}',
    learnings_json TEXT NOT NULL DEFAULT '[]',
    gaps_json TEXT NOT NULL DEFAULT '[]',
    cell_count INTEGER NOT NULL DEFAULT 0,
    fill_rate REAL NOT NULL DEFAULT 0.0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    duration_seconds REAL NOT NULL DEFAULT 0.0
);
"""


def _ensure_memory_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_MEMORY_SCHEMA)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryRecord:
    run_id: str
    timestamp: float
    intent_summary: str
    domain: str
    trust_score: float
    fidelity_score: float
    compression_losses: dict
    gate_results: dict
    learnings: list
    gaps: list
    cell_count: int
    fill_rate: float
    cost_usd: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# Consolidation — extract a MemoryRecord from compile artifacts
# ---------------------------------------------------------------------------

def consolidate(compile_result, state, run_id: str, duration: float) -> MemoryRecord:
    """Build a MemoryRecord from a CompileResult + SharedState.

    Args:
        compile_result: CompileResult dataclass from engine.compile()
        state: SharedState from the compilation
        run_id: Unique compilation identifier
        duration: Wall-clock seconds for the compilation

    Returns:
        Frozen MemoryRecord ready for persistence.
    """
    # Intent summary — first 200 chars of the input
    intent_summary = (state.known.get("input", "") or "")[:200]

    # Domain
    domain = state.known.get("domain", "software")

    # Trust score from verification
    v = getattr(compile_result, "verification", {}) or {}
    trust_score = _extract_trust(v)

    # Fidelity score — closed-loop gate result if present
    fidelity_score = 0.0
    cl = state.known.get("closed_loop_result", {})
    if isinstance(cl, dict):
        fidelity_score = float(cl.get("fidelity", 0.0))

    # Compression losses
    compression_losses = dict(state.known.get("compression_loss_categories", {}))

    # Gate results
    gate_results = _extract_gate_results(compile_result, state)

    # Auto-extract learnings and gaps from insights
    insights = getattr(compile_result, "insights", []) or []
    learnings = _extract_learnings(insights)
    gaps = _extract_gaps(insights, compile_result)

    # Grid stats
    grid = getattr(compile_result, "semantic_grid", None) or {}
    cell_count = 0
    fill_rate = 0.0
    if isinstance(grid, dict):
        cell_count = grid.get("cells", 0)
        fill_rate = grid.get("fill_rate", 0.0)

    # Cost
    cost_usd = 0.0
    stage_timings = getattr(compile_result, "stage_timings", {}) or {}
    # Cost comes from telemetry if available
    if hasattr(compile_result, "cache_stats"):
        cs = compile_result.cache_stats or {}
        cost_usd = float(cs.get("estimated_cost_usd", 0.0))

    return MemoryRecord(
        run_id=run_id,
        timestamp=time.time(),
        intent_summary=intent_summary,
        domain=domain,
        trust_score=trust_score,
        fidelity_score=fidelity_score,
        compression_losses=compression_losses,
        gate_results=gate_results,
        learnings=learnings,
        gaps=gaps,
        cell_count=cell_count,
        fill_rate=fill_rate,
        cost_usd=cost_usd,
        duration_seconds=duration,
    )


def _extract_trust(verification: dict) -> float:
    """Extract overall trust score from verification dict."""
    if not verification or not isinstance(verification, dict):
        return 0.0

    dims = ["completeness", "consistency", "coherence", "traceability"]
    scores = []
    for d in dims:
        val = verification.get(d, 0)
        if isinstance(val, dict):
            scores.append(float(val.get("score", 0)))
        elif isinstance(val, (int, float)):
            scores.append(float(val))
    return sum(scores) / len(scores) if scores else 0.0


def _extract_gate_results(compile_result, state) -> dict:
    """Extract pass/fail from known quality gates."""
    gates = {}

    # Success gate
    gates["compilation"] = "pass" if getattr(compile_result, "success", False) else "fail"

    # Closed-loop gate
    cl = state.known.get("closed_loop_result", {})
    if isinstance(cl, dict) and "passed" in cl:
        gates["closed_loop"] = "pass" if cl["passed"] else "fail"

    # Verification gate
    v = getattr(compile_result, "verification", {}) or {}
    if v:
        gates["verification"] = "pass"
    else:
        gates["verification"] = "fail"

    return gates


def _extract_learnings(insights: list) -> list:
    """Extract learning-relevant insights."""
    learnings = []
    keywords = ["pattern", "reuse", "improved", "optimization", "corpus", "feedback", "learned"]
    for insight in insights:
        if not isinstance(insight, str):
            continue
        lower = insight.lower()
        if any(kw in lower for kw in keywords):
            learnings.append(insight)
    return learnings[:10]  # Cap at 10


def _extract_gaps(insights: list, compile_result) -> list:
    """Extract gap indicators from compilation artifacts."""
    gaps = []

    # From insights
    gap_keywords = ["missing", "gap", "incomplete", "low confidence", "empty", "sparse"]
    for insight in insights:
        if not isinstance(insight, str):
            continue
        lower = insight.lower()
        if any(kw in lower for kw in gap_keywords):
            gaps.append(insight)

    # From verification failures
    v = getattr(compile_result, "verification", {}) or {}
    for dim in ["completeness", "consistency", "coherence", "traceability"]:
        val = v.get(dim, {})
        score = 0.0
        if isinstance(val, dict):
            score = float(val.get("score", 0))
        elif isinstance(val, (int, float)):
            score = float(val)
        if 0 < score < 60:
            gaps.append(f"Low {dim}: {score:.0f}/100")

    return gaps[:10]  # Cap at 10


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_memory(record: MemoryRecord, db_dir: Optional[Path] = None) -> int:
    """Persist a MemoryRecord. Returns the row id."""
    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    _ensure_memory_schema(conn)

    try:
        with conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO memory_records
                   (run_id, timestamp, intent_summary, domain, trust_score,
                    fidelity_score, compression_losses_json, gate_results_json,
                    learnings_json, gaps_json, cell_count, fill_rate,
                    cost_usd, duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.run_id,
                    record.timestamp,
                    record.intent_summary,
                    record.domain,
                    record.trust_score,
                    record.fidelity_score,
                    json.dumps(record.compression_losses),
                    json.dumps(record.gate_results),
                    json.dumps(record.learnings),
                    json.dumps(record.gaps),
                    record.cell_count,
                    record.fill_rate,
                    record.cost_usd,
                    record.duration_seconds,
                ),
            )
            return cursor.lastrowid
    finally:
        conn.close()


def load_recent_memories(
    limit: int = 50,
    domain: Optional[str] = None,
    db_dir: Optional[Path] = None,
) -> list[MemoryRecord]:
    """Load most recent memory records, optionally filtered by domain."""
    d = db_dir or _DEFAULT_DB_DIR
    path = d / _DEFAULT_DB_NAME
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_memory_schema(conn)

    try:
        if domain:
            rows = conn.execute(
                "SELECT * FROM memory_records WHERE domain = ? ORDER BY timestamp DESC LIMIT ?",
                (domain, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memory_records ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [_row_to_record(r) for r in rows]
    finally:
        conn.close()


def memory_stats(db_dir: Optional[Path] = None) -> dict:
    """Aggregate statistics over all memory records."""
    d = db_dir or _DEFAULT_DB_DIR
    path = d / _DEFAULT_DB_NAME
    if not path.exists():
        return {"total": 0}

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_memory_schema(conn)

    try:
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                AVG(trust_score) as avg_trust,
                AVG(fidelity_score) as avg_fidelity,
                AVG(duration_seconds) as avg_duration,
                SUM(cost_usd) as total_cost,
                AVG(fill_rate) as avg_fill_rate,
                AVG(cell_count) as avg_cell_count
            FROM memory_records"""
        ).fetchone()

        if not row or row["total"] == 0:
            return {"total": 0}

        # Domain breakdown
        domain_rows = conn.execute(
            "SELECT domain, COUNT(*) as cnt FROM memory_records GROUP BY domain"
        ).fetchall()
        domains = {r["domain"]: r["cnt"] for r in domain_rows}

        return {
            "total": row["total"],
            "avg_trust": round(row["avg_trust"] or 0, 2),
            "avg_fidelity": round(row["avg_fidelity"] or 0, 2),
            "avg_duration": round(row["avg_duration"] or 0, 1),
            "total_cost": round(row["total_cost"] or 0, 4),
            "avg_fill_rate": round(row["avg_fill_rate"] or 0, 3),
            "avg_cell_count": round(row["avg_cell_count"] or 0, 1),
            "domains": domains,
        }
    finally:
        conn.close()


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        run_id=row["run_id"],
        timestamp=row["timestamp"],
        intent_summary=row["intent_summary"],
        domain=row["domain"],
        trust_score=row["trust_score"],
        fidelity_score=row["fidelity_score"],
        compression_losses=json.loads(row["compression_losses_json"]),
        gate_results=json.loads(row["gate_results_json"]),
        learnings=json.loads(row["learnings_json"]),
        gaps=json.loads(row["gaps_json"]),
        cell_count=row["cell_count"],
        fill_rate=row["fill_rate"],
        cost_usd=row["cost_usd"],
        duration_seconds=row["duration_seconds"],
    )


# ===========================================================================
# L2 Pattern Detection — detect recurring patterns from memory records
# ===========================================================================

_PATTERNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    frequency INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.0,
    affected_postcodes_json TEXT NOT NULL DEFAULT '[]',
    remediation TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);
"""


def _ensure_patterns_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_PATTERNS_SCHEMA)


@dataclass(frozen=True)
class LearnedPattern:
    pattern_id: str
    category: str           # "recurring_gap", "domain_confidence", "layer_rejection"
    description: str
    frequency: int
    confidence: float
    affected_postcodes: tuple[str, ...]
    remediation: str


def detect_patterns(
    memories: list[MemoryRecord],
    min_occurrences: int = 3,
) -> list[LearnedPattern]:
    """Detect recurring patterns from a list of memory records.

    Analyzes:
    - Recurring compression loss categories
    - Domain-specific trust trends
    - Recurring gaps across compilations
    - Gate failure patterns
    """
    if len(memories) < min_occurrences:
        return []

    patterns = []

    # 1. Recurring compression loss categories
    loss_freq: dict[str, int] = {}
    for mem in memories:
        for cat in mem.compression_losses:
            loss_freq[cat] = loss_freq.get(cat, 0) + 1

    for cat, freq in loss_freq.items():
        if freq >= min_occurrences:
            conf = min(1.0, freq / len(memories))
            patterns.append(LearnedPattern(
                pattern_id=f"recurring_loss_{cat}",
                category="recurring_gap",
                description=f"Compression loss '{cat}' occurred in {freq}/{len(memories)} compilations",
                frequency=freq,
                confidence=conf,
                affected_postcodes=(),
                remediation=f"Strengthen {cat} extraction in synthesis phase",
            ))

    # 2. Domain-specific trust trends
    domain_scores: dict[str, list[float]] = {}
    for mem in memories:
        domain_scores.setdefault(mem.domain, []).append(mem.trust_score)

    for domain, scores in domain_scores.items():
        if len(scores) >= min_occurrences:
            avg = sum(scores) / len(scores)
            if avg < 60.0:
                patterns.append(LearnedPattern(
                    pattern_id=f"low_trust_{domain}",
                    category="domain_confidence",
                    description=f"Domain '{domain}' has low average trust ({avg:.1f}/100) across {len(scores)} compilations",
                    frequency=len(scores),
                    confidence=min(1.0, len(scores) / len(memories)),
                    affected_postcodes=(),
                    remediation=f"Review domain adapter and verification calibration for '{domain}'",
                ))

    # 3. Recurring gap strings
    gap_freq: dict[str, int] = {}
    for mem in memories:
        for gap in mem.gaps:
            # Normalize: strip numbers, lowercase
            key = gap.lower().split(":")[0].strip()
            gap_freq[key] = gap_freq.get(key, 0) + 1

    for gap_key, freq in gap_freq.items():
        if freq >= min_occurrences:
            patterns.append(LearnedPattern(
                pattern_id=f"recurring_gap_{gap_key.replace(' ', '_')}",
                category="recurring_gap",
                description=f"Gap '{gap_key}' appeared in {freq}/{len(memories)} compilations",
                frequency=freq,
                confidence=min(1.0, freq / len(memories)),
                affected_postcodes=(),
                remediation=f"Address recurring gap: {gap_key}",
            ))

    # 4. Gate failure patterns
    gate_failures: dict[str, int] = {}
    for mem in memories:
        for gate, result in mem.gate_results.items():
            if result == "fail":
                gate_failures[gate] = gate_failures.get(gate, 0) + 1

    for gate, freq in gate_failures.items():
        if freq >= min_occurrences:
            patterns.append(LearnedPattern(
                pattern_id=f"gate_failure_{gate}",
                category="layer_rejection",
                description=f"Gate '{gate}' failed in {freq}/{len(memories)} compilations",
                frequency=freq,
                confidence=min(1.0, freq / len(memories)),
                affected_postcodes=(),
                remediation=f"Investigate persistent {gate} gate failures",
            ))

    return patterns


def save_patterns(
    patterns: list[LearnedPattern],
    db_dir: Optional[Path] = None,
) -> int:
    """Persist learned patterns. Returns count saved."""
    if not patterns:
        return 0

    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    _ensure_patterns_schema(conn)

    saved = 0
    try:
        with conn:
            for p in patterns:
                conn.execute(
                    """INSERT OR REPLACE INTO learned_patterns
                       (pattern_id, category, description, frequency,
                        confidence, affected_postcodes_json, remediation, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p.pattern_id,
                        p.category,
                        p.description,
                        p.frequency,
                        p.confidence,
                        json.dumps(list(p.affected_postcodes)),
                        p.remediation,
                        time.time(),
                    ),
                )
                saved += 1
    finally:
        conn.close()

    return saved


def load_patterns(
    min_confidence: float = 0.5,
    db_dir: Optional[Path] = None,
) -> list[LearnedPattern]:
    """Load learned patterns above confidence threshold."""
    d = db_dir or _DEFAULT_DB_DIR
    path = d / _DEFAULT_DB_NAME
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_patterns_schema(conn)

    try:
        rows = conn.execute(
            "SELECT * FROM learned_patterns WHERE confidence >= ? ORDER BY frequency DESC",
            (min_confidence,),
        ).fetchall()
        return [
            LearnedPattern(
                pattern_id=r["pattern_id"],
                category=r["category"],
                description=r["description"],
                frequency=r["frequency"],
                confidence=r["confidence"],
                affected_postcodes=tuple(json.loads(r["affected_postcodes_json"])),
                remediation=r["remediation"],
            )
            for r in rows
        ]
    finally:
        conn.close()


def format_pattern_context(patterns: list[LearnedPattern]) -> str:
    """Format learned patterns for prompt injection."""
    if not patterns:
        return ""

    lines = ["[LEARNED PATTERNS FROM PRIOR COMPILATIONS]"]
    for p in patterns[:10]:  # Cap at 10 for prompt size
        lines.append(
            f"- [{p.category}] {p.description} (confidence={p.confidence:.2f}). "
            f"Remediation: {p.remediation}"
        )
    return "\n".join(lines)
