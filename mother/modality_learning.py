"""
mother/modality_learning.py — Multimodal pattern learning.

LEAF module. No imports from core/ or mother/. Stdlib only.

Tracks which modality combinations correlate with positive interaction
outcomes. Detects patterns, generates insights, and recommends weight
adjustments for fusion confidence multipliers.

Persistence via SQLite (modality_insights table in maps.db).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

__all__ = [
    "InteractionRecord",
    "ModalityInsight",
    "LearningReport",
    "ModalityPatternDetector",
    "save_modality_insights",
    "load_modality_insights",
    "format_learning_context",
]


# ---------------------------------------------------------------------------
# Database location (shared with kernel/store.py, kernel/memory.py)
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".motherlabs"
_DEFAULT_DB_NAME = "maps.db"


def _db_path(db_dir: Optional[Path] = None) -> Path:
    d = db_dir or _DEFAULT_DB_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / _DEFAULT_DB_NAME


_INSIGHTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS modality_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    pattern TEXT NOT NULL,
    correlation REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    recommendation TEXT NOT NULL DEFAULT 'maintain'
);
"""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_INSIGHTS_SCHEMA)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionRecord:
    """Record of a single user interaction with active modality context."""
    patterns_active: tuple[str, ...]  # fusion patterns at time of interaction
    modalities_active: tuple[str, ...]  # which modalities had recent events
    response_quality: float  # 0.0-1.0
    timestamp: float


@dataclass(frozen=True)
class ModalityInsight:
    """Insight about a specific fusion pattern's usefulness."""
    pattern: str            # e.g. "presenting"
    correlation: float      # with positive outcomes (-1.0 to 1.0)
    sample_count: int
    recommendation: str     # "boost" | "maintain" | "suppress"


@dataclass(frozen=True)
class LearningReport:
    """Summary of multimodal pattern learning."""
    insights: tuple[ModalityInsight, ...]
    total_interactions: int
    most_useful_pattern: str
    least_useful_pattern: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ModalityPatternDetector:
    """Accumulates interaction records and computes pattern correlations.

    Analyzes which fusion patterns (presenting, focused, etc.) correlate
    with positive user interaction outcomes.
    """

    def __init__(self, min_samples: int = 10) -> None:
        self._min_samples = max(1, min_samples)
        self._records: list[InteractionRecord] = []

    def record_interaction(
        self,
        patterns_active: tuple[str, ...],
        modalities_active: tuple[str, ...],
        response_quality: float,
        timestamp: float | None = None,
    ) -> None:
        """Record an interaction outcome with active pattern context."""
        ts = timestamp if timestamp is not None else time.time()
        quality = max(0.0, min(1.0, response_quality))
        self._records.append(InteractionRecord(
            patterns_active=patterns_active,
            modalities_active=modalities_active,
            response_quality=quality,
            timestamp=ts,
        ))

    def record_count(self) -> int:
        """Number of stored interaction records."""
        return len(self._records)

    def analyze(self) -> LearningReport:
        """Compute correlations and generate insights.

        For each observed pattern, computes the average response quality
        when the pattern was active vs when it wasn't. The correlation
        is the difference: (avg_with - avg_without), bounded to [-1, 1].

        Patterns with fewer than min_samples get recommendation="maintain".
        """
        if not self._records:
            return LearningReport(
                insights=(),
                total_interactions=0,
                most_useful_pattern="",
                least_useful_pattern="",
            )

        # Collect all observed patterns
        all_patterns: set[str] = set()
        for rec in self._records:
            all_patterns.update(rec.patterns_active)

        if not all_patterns:
            return LearningReport(
                insights=(),
                total_interactions=len(self._records),
                most_useful_pattern="",
                least_useful_pattern="",
            )

        # Global average
        global_avg = sum(r.response_quality for r in self._records) / len(self._records)

        insights: list[ModalityInsight] = []
        for pattern in sorted(all_patterns):
            with_pattern = [r for r in self._records if pattern in r.patterns_active]
            without_pattern = [r for r in self._records if pattern not in r.patterns_active]

            sample_count = len(with_pattern)

            if sample_count < self._min_samples:
                insights.append(ModalityInsight(
                    pattern=pattern,
                    correlation=0.0,
                    sample_count=sample_count,
                    recommendation="maintain",
                ))
                continue

            avg_with = sum(r.response_quality for r in with_pattern) / len(with_pattern)
            avg_without = (
                sum(r.response_quality for r in without_pattern) / len(without_pattern)
                if without_pattern else global_avg
            )

            correlation = avg_with - avg_without
            correlation = max(-1.0, min(1.0, correlation))

            # Determine recommendation
            if abs(correlation) < 0.05:
                recommendation = "maintain"
            elif correlation > 0:
                recommendation = "boost"
            else:
                recommendation = "suppress"

            insights.append(ModalityInsight(
                pattern=pattern,
                correlation=correlation,
                sample_count=sample_count,
                recommendation=recommendation,
            ))

        insights_tuple = tuple(insights)

        # Find most/least useful
        scoreable = [i for i in insights if i.sample_count >= self._min_samples]
        if scoreable:
            most_useful = max(scoreable, key=lambda i: i.correlation).pattern
            least_useful = min(scoreable, key=lambda i: i.correlation).pattern
        else:
            most_useful = ""
            least_useful = ""

        return LearningReport(
            insights=insights_tuple,
            total_interactions=len(self._records),
            most_useful_pattern=most_useful,
            least_useful_pattern=least_useful,
        )

    def recommend_weights(self) -> dict[str, float]:
        """Pattern → weight multiplier for fusion confidence.

        Maps correlation to weight: 0.5 (suppress) to 2.0 (boost).
        Patterns below min_samples get weight 1.0 (neutral).
        """
        report = self.analyze()
        weights: dict[str, float] = {}
        for insight in report.insights:
            if insight.sample_count < self._min_samples:
                weights[insight.pattern] = 1.0
            else:
                # Map correlation [-1, 1] to weight [0.5, 2.0]
                # correlation=0 → 1.0, correlation=1 → 2.0, correlation=-1 → 0.5
                weight = 1.0 + insight.correlation * 0.75
                weight = max(0.5, min(2.0, weight))
                weights[insight.pattern] = weight
        return weights

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_modality_insights(
    report: LearningReport,
    db_dir: Optional[Path] = None,
) -> None:
    """Persist learning report insights to maps.db."""
    path = _db_path(db_dir)
    conn = sqlite3.connect(str(path))
    try:
        _ensure_schema(conn)
        ts = time.time()
        for insight in report.insights:
            conn.execute(
                "INSERT INTO modality_insights (timestamp, pattern, correlation, sample_count, recommendation) "
                "VALUES (?, ?, ?, ?, ?)",
                (ts, insight.pattern, insight.correlation, insight.sample_count, insight.recommendation),
            )
        conn.commit()
    finally:
        conn.close()


def load_modality_insights(
    db_dir: Optional[Path] = None,
    limit: int = 50,
) -> LearningReport:
    """Load most recent insights from maps.db."""
    path = _db_path(db_dir)
    if not path.exists():
        return LearningReport(insights=(), total_interactions=0,
                              most_useful_pattern="", least_useful_pattern="")

    conn = sqlite3.connect(str(path))
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT pattern, correlation, sample_count, recommendation "
            "FROM modality_insights ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        if not rows:
            return LearningReport(insights=(), total_interactions=0,
                                  most_useful_pattern="", least_useful_pattern="")

        insights = tuple(
            ModalityInsight(
                pattern=row[0],
                correlation=row[1],
                sample_count=row[2],
                recommendation=row[3],
            )
            for row in rows
        )

        total = sum(i.sample_count for i in insights)
        scoreable = [i for i in insights if i.sample_count > 0]
        most_useful = max(scoreable, key=lambda i: i.correlation).pattern if scoreable else ""
        least_useful = min(scoreable, key=lambda i: i.correlation).pattern if scoreable else ""

        return LearningReport(
            insights=insights,
            total_interactions=total,
            most_useful_pattern=most_useful,
            least_useful_pattern=least_useful,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_learning_context(report: LearningReport) -> str:
    """Render learning insights for prompt injection.

    Returns empty string if no insights available.
    """
    if not report.insights:
        return ""

    lines = ["[Modality Learning]"]
    lines.append(f"  Interactions analyzed: {report.total_interactions}")

    for insight in report.insights:
        if insight.recommendation == "maintain":
            continue  # skip neutral
        direction = "+" if insight.correlation > 0 else "-"
        lines.append(
            f"  {insight.pattern}: {direction}{abs(insight.correlation):.2f} "
            f"({insight.recommendation}, n={insight.sample_count})"
        )

    if report.most_useful_pattern:
        lines.append(f"  Most useful: {report.most_useful_pattern}")
    if report.least_useful_pattern and report.least_useful_pattern != report.most_useful_pattern:
        lines.append(f"  Least useful: {report.least_useful_pattern}")

    return "\n".join(lines)
