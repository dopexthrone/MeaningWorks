"""
Motherlabs Metering — usage tracking and rate limiting.

Phase D: V2 API + Platform Layer

Tracks per-domain compilation metrics for the platform layer.
Thread-safe, in-memory. Production would use Redis/DB.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class DomainMetrics:
    """Per-domain usage metrics."""
    compilation_count: int = 0
    total_duration_seconds: float = 0.0
    total_cost_usd: float = 0.0
    last_compilation_time: float = 0.0
    errors: int = 0


class MeteringTracker:
    """Thread-safe usage tracker for multi-domain compilations.

    Tracks per-domain metrics: compilation count, duration, cost.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._domains: Dict[str, DomainMetrics] = {}
        self._key_metrics: Dict[str, DomainMetrics] = {}
        self._start_time = time.time()

    def record_compilation(
        self,
        domain: str,
        duration_seconds: float,
        cost_usd: float,
        success: bool = True,
        key_id: Optional[str] = None,
    ) -> None:
        """Record a compilation event.

        Args:
            domain: Domain adapter name
            duration_seconds: Compilation duration
            cost_usd: Estimated cost
            success: Whether compilation succeeded
            key_id: Optional API key ID for per-key tracking
        """
        with self._lock:
            if domain not in self._domains:
                self._domains[domain] = DomainMetrics()
            m = self._domains[domain]
            m.compilation_count += 1
            m.total_duration_seconds += duration_seconds
            m.total_cost_usd += cost_usd
            m.last_compilation_time = time.time()
            if not success:
                m.errors += 1

            # Per-key tracking
            if key_id is not None:
                if key_id not in self._key_metrics:
                    self._key_metrics[key_id] = DomainMetrics()
                km = self._key_metrics[key_id]
                km.compilation_count += 1
                km.total_duration_seconds += duration_seconds
                km.total_cost_usd += cost_usd
                km.last_compilation_time = time.time()
                if not success:
                    km.errors += 1

    def record_error(self, domain: str) -> None:
        """Record a compilation error."""
        self.record_compilation(domain, 0.0, 0.0, success=False)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a JSON-safe dict.

        Returns:
            Dict with per_domain metrics, totals, and uptime
        """
        with self._lock:
            per_domain = {}
            total_compilations = 0
            total_cost = 0.0

            for domain, m in self._domains.items():
                avg_duration = (
                    m.total_duration_seconds / m.compilation_count
                    if m.compilation_count > 0 else 0.0
                )
                per_domain[domain] = {
                    "compilation_count": m.compilation_count,
                    "total_duration_seconds": round(m.total_duration_seconds, 2),
                    "avg_duration_seconds": round(avg_duration, 2),
                    "total_cost_usd": round(m.total_cost_usd, 4),
                    "errors": m.errors,
                    "success_rate": round(
                        (m.compilation_count - m.errors) / m.compilation_count
                        if m.compilation_count > 0 else 0.0,
                        3,
                    ),
                }
                total_compilations += m.compilation_count
                total_cost += m.total_cost_usd

            return {
                "per_domain": per_domain,
                "total_compilations": total_compilations,
                "total_cost_usd": round(total_cost, 4),
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }

    def get_domain_count(self, domain: str) -> int:
        """Get compilation count for a specific domain."""
        with self._lock:
            m = self._domains.get(domain)
            return m.compilation_count if m else 0

    def get_key_metrics(self, key_id: str) -> Dict[str, Any]:
        """Get metrics for a specific API key.

        Returns:
            Dict with compilation_count, total_duration, total_cost, errors, success_rate
        """
        with self._lock:
            m = self._key_metrics.get(key_id)
            if not m:
                return {
                    "compilation_count": 0,
                    "total_duration_seconds": 0.0,
                    "total_cost_usd": 0.0,
                    "errors": 0,
                    "success_rate": 0.0,
                }
            avg_duration = (
                m.total_duration_seconds / m.compilation_count
                if m.compilation_count > 0 else 0.0
            )
            return {
                "compilation_count": m.compilation_count,
                "total_duration_seconds": round(m.total_duration_seconds, 2),
                "avg_duration_seconds": round(avg_duration, 2),
                "total_cost_usd": round(m.total_cost_usd, 4),
                "errors": m.errors,
                "success_rate": round(
                    (m.compilation_count - m.errors) / m.compilation_count
                    if m.compilation_count > 0 else 0.0,
                    3,
                ),
            }

    def reset(self) -> None:
        """Reset all metrics. Used in tests."""
        with self._lock:
            self._domains.clear()
            self._key_metrics.clear()
            self._start_time = time.time()
