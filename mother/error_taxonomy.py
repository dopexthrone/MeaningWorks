"""
Mother error taxonomy — structured error classification.

LEAF module. Stdlib only. No imports from core/ or mother/.

Classifies exceptions into operational categories with severity,
retriability, and user-actionability for senses and context injection.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ErrorClassification:
    """Structured classification of an error."""

    category: str = "unknown"     # auth|rate_limit|cost|timeout|compile_failure|build_failure|launch_failure|connection|validation|unknown
    severity: float = 0.5         # 0.0-1.0
    retriable: bool = False
    user_actionable: bool = False
    phase: str = ""               # chat|compile|build|launch|perception
    fingerprint: str = ""         # "auth:401" | "timeout:compile"


def classify_error(error: Exception, phase: str = "chat") -> ErrorClassification:
    """Classify an exception into an operational category.

    Classification rules (checked in order):
    1. ConnectionError or "connect" → connection
    2. "401"/"auth"/"unauthorized" → auth
    3. "429"/"rate" → rate_limit
    4. "cost"/"cap"/"budget" → cost
    5. TimeoutError/"timeout" → timeout
    6. phase=compile (fallthrough) → compile_failure
    7. phase=build → build_failure
    8. phase=launch → launch_failure
    9. "validation"/"invalid" → validation
    10. fallback → unknown
    """
    msg = str(error).lower()
    err_type = type(error).__name__

    # 1. Connection errors
    if isinstance(error, ConnectionError) or "connect" in msg:
        return ErrorClassification(
            category="connection",
            severity=0.3,
            retriable=True,
            user_actionable=False,
            phase=phase,
            fingerprint=f"connection:{phase}",
        )

    # 2. Auth errors
    if "401" in msg or "auth" in msg or "unauthorized" in msg:
        return ErrorClassification(
            category="auth",
            severity=0.2,
            retriable=False,
            user_actionable=True,
            phase=phase,
            fingerprint="auth:401",
        )

    # 3. Rate limiting
    if "429" in msg or "rate" in msg:
        return ErrorClassification(
            category="rate_limit",
            severity=0.1,
            retriable=True,
            user_actionable=False,
            phase=phase,
            fingerprint="rate_limit:429",
        )

    # 4. Cost / budget
    if "cost" in msg or "cap" in msg or "budget" in msg:
        return ErrorClassification(
            category="cost",
            severity=0.4,
            retriable=False,
            user_actionable=True,
            phase=phase,
            fingerprint=f"cost:{phase}",
        )

    # 5. Timeout
    if isinstance(error, TimeoutError) or "timeout" in msg:
        return ErrorClassification(
            category="timeout",
            severity=0.3,
            retriable=True,
            user_actionable=False,
            phase=phase,
            fingerprint=f"timeout:{phase}",
        )

    # 6-8. Phase-specific fallthrough
    if phase == "compile":
        return ErrorClassification(
            category="compile_failure",
            severity=0.5,
            retriable=False,
            user_actionable=False,
            phase=phase,
            fingerprint=f"compile_failure:{err_type}",
        )

    if phase == "build":
        return ErrorClassification(
            category="build_failure",
            severity=0.6,
            retriable=False,
            user_actionable=False,
            phase=phase,
            fingerprint=f"build_failure:{err_type}",
        )

    if phase == "launch":
        return ErrorClassification(
            category="launch_failure",
            severity=0.7,
            retriable=False,
            user_actionable=False,
            phase=phase,
            fingerprint=f"launch_failure:{err_type}",
        )

    # 9. Validation
    if "validation" in msg or "invalid" in msg:
        return ErrorClassification(
            category="validation",
            severity=0.4,
            retriable=False,
            user_actionable=True,
            phase=phase,
            fingerprint=f"validation:{phase}",
        )

    # 10. Unknown fallback
    return ErrorClassification(
        category="unknown",
        severity=0.5,
        retriable=False,
        user_actionable=False,
        phase=phase,
        fingerprint=f"unknown:{err_type}",
    )


def compute_error_impact(classifications: List[ErrorClassification]) -> Dict[str, float]:
    """Compute aggregate error impact metrics.

    Returns:
        total_severity: sum of all severities
        max_severity: highest single severity
        retriable_fraction: fraction of errors that are retriable
        user_actionable_fraction: fraction that user can fix
    """
    if not classifications:
        return {
            "total_severity": 0.0,
            "max_severity": 0.0,
            "retriable_fraction": 0.0,
            "user_actionable_fraction": 0.0,
        }

    total = len(classifications)
    return {
        "total_severity": round(sum(c.severity for c in classifications), 4),
        "max_severity": round(max(c.severity for c in classifications), 4),
        "retriable_fraction": round(
            sum(1 for c in classifications if c.retriable) / total, 4
        ),
        "user_actionable_fraction": round(
            sum(1 for c in classifications if c.user_actionable) / total, 4
        ),
    }


def summarize_errors(classifications: List[ErrorClassification]) -> str:
    """Produce a human-readable summary of error classifications.

    Returns empty string for empty list.
    """
    if not classifications:
        return ""

    # Count by category
    counts: Dict[str, int] = {}
    for c in classifications:
        counts[c.category] = counts.get(c.category, 0) + 1

    parts = []
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count == 1:
            parts.append(f"1 {cat}")
        else:
            parts.append(f"{count} {cat}")

    total = len(classifications)
    retriable = sum(1 for c in classifications if c.retriable)
    actionable = sum(1 for c in classifications if c.user_actionable)

    summary = f"Errors: {', '.join(parts)}."
    if retriable > 0:
        summary += f" {retriable} retriable."
    if actionable > 0:
        summary += f" {actionable} user-actionable."

    return summary
