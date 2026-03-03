"""
Phase 19: Telemetry module tests.

Tests for core/telemetry.py — frozen dataclasses, aggregation, health, serialization.
"""

import time
import pytest

from core.telemetry import (
    CompilationMetrics,
    AggregateMetrics,
    HealthSnapshot,
    aggregate_metrics,
    compute_health,
    percentile,
    metrics_to_dict,
    aggregate_to_dict,
    health_to_dict,
)


# =============================================================================
# HELPERS
# =============================================================================


def make_metrics(
    compilation_id: str = "abc123",
    success: bool = True,
    total_duration: float = 10.0,
    component_count: int = 5,
    insight_count: int = 3,
    verification_score: int = 80,
    verification_mode: str = "deterministic",
    cache_hits: int = 1,
    cache_misses: int = 2,
    retry_count: int = 0,
    provider: str = "mock",
    model: str = "mock-1",
    timestamp: float = None,
) -> CompilationMetrics:
    return CompilationMetrics(
        compilation_id=compilation_id,
        timestamp=timestamp or time.time(),
        success=success,
        total_duration=total_duration,
        stage_timings=(("intent", 1.0), ("synthesis", 5.0), ("verification", 2.0)),
        dialogue_turns=4,
        component_count=component_count,
        insight_count=insight_count,
        verification_score=verification_score,
        verification_mode=verification_mode,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        retry_count=retry_count,
        provider=provider,
        model=model,
    )


# =============================================================================
# FROZEN DATACLASS TESTS
# =============================================================================


class TestCompilationMetrics:
    """CompilationMetrics frozen dataclass tests."""

    def test_frozen(self):
        """CompilationMetrics is immutable."""
        m = make_metrics()
        with pytest.raises(AttributeError):
            m.success = False

    def test_all_fields_stored(self):
        """All fields are accessible."""
        m = make_metrics(
            compilation_id="xyz",
            success=False,
            total_duration=42.0,
            provider="claude",
        )
        assert m.compilation_id == "xyz"
        assert m.success is False
        assert m.total_duration == 42.0
        assert m.provider == "claude"
        assert isinstance(m.stage_timings, tuple)


class TestAggregateMetrics:
    """AggregateMetrics frozen dataclass tests."""

    def test_frozen(self):
        """AggregateMetrics is immutable."""
        agg = AggregateMetrics(
            window_size=0, success_rate=0.0, avg_duration=0.0,
            avg_components=0.0, avg_insights=0.0, avg_verification_score=0.0,
            p50_duration=0.0, p95_duration=0.0,
            verification_mode_counts=(), provider_counts=(),
            cache_hit_rate=0.0, total_retries=0,
        )
        with pytest.raises(AttributeError):
            agg.window_size = 5

    def test_computed_fields(self):
        """AggregateMetrics stores computed fields."""
        agg = AggregateMetrics(
            window_size=10, success_rate=0.9, avg_duration=15.0,
            avg_components=5.0, avg_insights=3.0, avg_verification_score=75.0,
            p50_duration=12.0, p95_duration=30.0,
            verification_mode_counts=(("deterministic", 8), ("hybrid", 2)),
            provider_counts=(("mock", 10),),
            cache_hit_rate=0.5, total_retries=2,
        )
        assert agg.window_size == 10
        assert agg.success_rate == 0.9
        assert agg.p95_duration == 30.0


class TestHealthSnapshot:
    """HealthSnapshot frozen dataclass tests."""

    def test_frozen(self):
        """HealthSnapshot is immutable."""
        h = HealthSnapshot(
            status="healthy", uptime_seconds=100.0,
            compilations_total=10, compilations_recent=5,
            recent_success_rate=1.0, recent_avg_duration=10.0,
            cache_hit_rate=0.5, corpus_size=10,
            last_compilation_age=5.0, issues=(),
        )
        with pytest.raises(AttributeError):
            h.status = "degraded"

    def test_status_values(self):
        """HealthSnapshot accepts all valid status values."""
        for status in ("healthy", "degraded", "unhealthy"):
            h = HealthSnapshot(
                status=status, uptime_seconds=0.0,
                compilations_total=0, compilations_recent=0,
                recent_success_rate=0.0, recent_avg_duration=0.0,
                cache_hit_rate=0.0, corpus_size=0,
                last_compilation_age=0.0, issues=(),
            )
            assert h.status == status


# =============================================================================
# PERCENTILE TESTS
# =============================================================================


class TestPercentile:
    """percentile() function tests."""

    def test_median_odd(self):
        """Median of odd-length list."""
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0

    def test_median_even(self):
        """Median of even-length list."""
        assert percentile([1, 2, 3, 4], 50) == 2.5

    def test_p95(self):
        """95th percentile computation."""
        vals = list(range(1, 101))  # 1..100
        p95 = percentile(vals, 95)
        assert 95.0 <= p95 <= 96.0

    def test_empty(self):
        """Empty input returns 0.0."""
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value returns that value for any percentile."""
        assert percentile([42.0], 0) == 42.0
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 100) == 42.0

    def test_unsorted_input(self):
        """Works correctly with unsorted input."""
        assert percentile([5, 1, 3, 2, 4], 50) == 3.0


# =============================================================================
# AGGREGATE METRICS TESTS
# =============================================================================


class TestAggregateMetricsFunction:
    """aggregate_metrics() function tests."""

    def test_empty(self):
        """Empty input produces zero-valued aggregate."""
        agg = aggregate_metrics([])
        assert agg.window_size == 0
        assert agg.success_rate == 0.0
        assert agg.avg_duration == 0.0
        assert agg.cache_hit_rate == 0.0
        assert agg.total_retries == 0

    def test_single(self):
        """Single metric produces correct aggregate."""
        m = make_metrics(
            success=True, total_duration=10.0,
            component_count=5, insight_count=3,
            verification_score=80,
            cache_hits=1, cache_misses=2,
        )
        agg = aggregate_metrics([m])
        assert agg.window_size == 1
        assert agg.success_rate == 1.0
        assert agg.avg_duration == 10.0
        assert agg.avg_components == 5.0
        assert agg.avg_insights == 3.0
        assert agg.avg_verification_score == 80.0
        assert agg.p50_duration == 10.0
        assert agg.p95_duration == 10.0

    def test_multiple_mixed(self):
        """Multiple metrics with mixed success rates."""
        metrics = [
            make_metrics(success=True, total_duration=10.0, cache_hits=2, cache_misses=0),
            make_metrics(success=False, total_duration=20.0, cache_hits=0, cache_misses=3),
            make_metrics(success=True, total_duration=15.0, cache_hits=1, cache_misses=1),
        ]
        agg = aggregate_metrics(metrics)
        assert agg.window_size == 3
        assert abs(agg.success_rate - 2 / 3) < 0.01
        assert abs(agg.avg_duration - 15.0) < 0.01
        # Cache: 3 hits / (3+4) = 3/7
        assert abs(agg.cache_hit_rate - 3 / 7) < 0.01

    def test_percentiles_correct(self):
        """P50 and P95 computed correctly for multiple values."""
        metrics = [make_metrics(total_duration=float(i)) for i in range(1, 11)]
        agg = aggregate_metrics(metrics)
        assert agg.p50_duration == 5.5
        assert agg.p95_duration > 9.0

    def test_verification_mode_counts(self):
        """Verification mode counts are accumulated correctly."""
        metrics = [
            make_metrics(verification_mode="deterministic"),
            make_metrics(verification_mode="deterministic"),
            make_metrics(verification_mode="hybrid"),
        ]
        agg = aggregate_metrics(metrics)
        mode_dict = dict(agg.verification_mode_counts)
        assert mode_dict["deterministic"] == 2
        assert mode_dict["hybrid"] == 1

    def test_provider_counts(self):
        """Provider counts are accumulated correctly."""
        metrics = [
            make_metrics(provider="claude"),
            make_metrics(provider="claude"),
            make_metrics(provider="grok"),
        ]
        agg = aggregate_metrics(metrics)
        prov_dict = dict(agg.provider_counts)
        assert prov_dict["claude"] == 2
        assert prov_dict["grok"] == 1

    def test_total_retries(self):
        """Total retries are summed."""
        metrics = [
            make_metrics(retry_count=1),
            make_metrics(retry_count=3),
        ]
        agg = aggregate_metrics(metrics)
        assert agg.total_retries == 4


# =============================================================================
# HEALTH TESTS
# =============================================================================


class TestComputeHealth:
    """compute_health() function tests."""

    def test_healthy(self):
        """All good metrics → healthy status."""
        metrics = [make_metrics(success=True, total_duration=10.0) for _ in range(10)]
        h = compute_health(metrics, uptime=3600.0, corpus_size=100)
        assert h.status == "healthy"
        assert len(h.issues) == 0

    def test_degraded_low_success_rate(self):
        """60% success rate → degraded."""
        metrics = (
            [make_metrics(success=True, total_duration=10.0) for _ in range(6)]
            + [make_metrics(success=False, total_duration=10.0) for _ in range(4)]
        )
        h = compute_health(metrics, uptime=3600.0, corpus_size=100)
        assert h.status == "degraded"
        assert any("Success rate" in i for i in h.issues)

    def test_degraded_slow_duration(self):
        """All success but slow (700s avg) → degraded."""
        metrics = [make_metrics(success=True, total_duration=700.0) for _ in range(10)]
        h = compute_health(metrics, uptime=3600.0, corpus_size=100)
        assert h.status == "degraded"
        assert any("duration" in i.lower() for i in h.issues)

    def test_unhealthy(self):
        """Low success + slow → unhealthy."""
        metrics = [make_metrics(success=False, total_duration=1000.0) for _ in range(10)]
        h = compute_health(metrics, uptime=3600.0, corpus_size=100)
        assert h.status == "unhealthy"
        assert len(h.issues) >= 2

    def test_issues_populated(self):
        """Issues list describes reasons for non-healthy status."""
        metrics = [make_metrics(success=False, total_duration=10.0) for _ in range(10)]
        h = compute_health(metrics, uptime=3600.0, corpus_size=100)
        assert len(h.issues) > 0
        assert any("Success rate" in i or "success" in i.lower() for i in h.issues)

    def test_no_compilations(self):
        """No compilations → healthy with message."""
        h = compute_health([], uptime=10.0, corpus_size=0)
        assert h.status == "healthy"
        assert h.compilations_recent == 0
        assert any("No compilations" in i for i in h.issues)

    def test_uptime_and_corpus_size(self):
        """Uptime and corpus_size are passed through."""
        h = compute_health([], uptime=42.0, corpus_size=99)
        assert h.uptime_seconds == 42.0
        assert h.corpus_size == 99


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestMetricsToDict:
    """metrics_to_dict() function tests."""

    def test_round_trip_fields(self):
        """All fields present in dict output."""
        m = make_metrics(compilation_id="test123", provider="claude")
        d = metrics_to_dict(m)
        assert d["compilation_id"] == "test123"
        assert d["provider"] == "claude"
        assert isinstance(d["stage_timings"], dict)
        assert "intent" in d["stage_timings"]

    def test_stage_timings_as_dict(self):
        """stage_timings are converted from tuples to dict."""
        m = make_metrics()
        d = metrics_to_dict(m)
        assert isinstance(d["stage_timings"], dict)
        assert d["stage_timings"]["intent"] == 1.0


class TestAggregateToDict:
    """aggregate_to_dict() function tests."""

    def test_correct_shape(self):
        """Output dict has all expected keys."""
        agg = aggregate_metrics([make_metrics()])
        d = aggregate_to_dict(agg)
        assert "window_size" in d
        assert "success_rate" in d
        assert "avg_duration" in d
        assert "p50_duration" in d
        assert "p95_duration" in d
        assert isinstance(d["verification_mode_counts"], dict)
        assert isinstance(d["provider_counts"], dict)

    def test_empty_aggregate(self):
        """Empty aggregate produces correct dict."""
        agg = aggregate_metrics([])
        d = aggregate_to_dict(agg)
        assert d["window_size"] == 0
        assert d["success_rate"] == 0.0


class TestHealthToDict:
    """health_to_dict() function tests."""

    def test_correct_shape(self):
        """Output dict has all expected keys."""
        h = compute_health([], uptime=10.0, corpus_size=5)
        d = health_to_dict(h)
        assert d["status"] == "healthy"
        assert d["uptime_seconds"] == 10.0
        assert d["corpus_size"] == 5
        assert isinstance(d["issues"], list)
        assert "compilations_recent" in d
        assert "recent_success_rate" in d
        assert "last_compilation_age" in d


# =============================================================================
# LEAF MODULE TEST
# =============================================================================


class TestLeafModule:
    """Verify telemetry.py has no engine/protocol/pipeline imports."""

    def test_no_project_imports(self):
        """telemetry.py imports only stdlib modules."""
        import inspect
        import core.telemetry as mod
        source = inspect.getsource(mod)
        # Should not import any project modules
        for forbidden in [
            "from core.engine",
            "from core.protocol",
            "from core.pipeline",
            "from core.llm",
            "from persistence",
            "from api",
        ]:
            assert forbidden not in source, f"Leaf module imports {forbidden}"


# =============================================================================
# JSON LOGGING TESTS (Phase 19)
# =============================================================================


class TestJsonLogFormatter:
    """Phase 19: Structured JSON logging formatter."""

    def test_json_log_format_valid_json(self):
        """JsonLogFormatter produces valid JSON."""
        import json
        import logging
        from cli.main import JsonLogFormatter

        formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_log_includes_required_fields(self):
        """JSON log output includes timestamp, level, logger, message."""
        import json
        import logging
        from cli.main import JsonLogFormatter

        formatter = JsonLogFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            name="motherlabs.test", level=logging.WARNING, pathname="", lineno=0,
            msg="Something happened", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "motherlabs.test"
        assert parsed["message"] == "Something happened"
        assert "timestamp" in parsed
