"""Tests for mother/actuator_receipt.py — uniform action feedback."""

import pytest

from mother.actuator_receipt import (
    ActuatorReceipt,
    ActuatorLog,
    ActuatorStore,
    create_receipt,
    format_actuator_context,
)


# ---------------------------------------------------------------------------
# ActuatorReceipt dataclass
# ---------------------------------------------------------------------------

class TestActuatorReceipt:
    def test_frozen(self):
        r = create_receipt("compile", True, 100.0, 110.0)
        with pytest.raises(AttributeError):
            r.success = False

    def test_defaults(self):
        r = create_receipt("compile", True, 100.0, 110.0)
        assert r.cost_usd == 0.0
        assert r.output_summary == ""
        assert r.error == ""
        assert r.modalities_affected == ()


# ---------------------------------------------------------------------------
# create_receipt factory
# ---------------------------------------------------------------------------

class TestCreateReceipt:
    def test_computed_duration(self):
        r = create_receipt("compile", True, 100.0, 115.5)
        assert r.duration_seconds == pytest.approx(15.5)

    def test_negative_duration_clamped(self):
        r = create_receipt("compile", True, 110.0, 100.0)
        assert r.duration_seconds == 0.0

    def test_all_fields(self):
        r = create_receipt(
            action_type="self_build",
            success=False,
            started_at=100.0,
            completed_at=200.0,
            cost_usd=0.15,
            output_summary="Built 5 modules",
            error="Test failure",
            modalities_affected=("screen", "filesystem"),
        )
        assert r.action_type == "self_build"
        assert r.success is False
        assert r.duration_seconds == pytest.approx(100.0)
        assert r.cost_usd == pytest.approx(0.15)
        assert r.output_summary == "Built 5 modules"
        assert r.error == "Test failure"
        assert r.modalities_affected == ("screen", "filesystem")

    def test_various_action_types(self):
        for action in ("compile", "file_write", "search", "self_build", "web", "voice"):
            r = create_receipt(action, True, 0.0, 1.0)
            assert r.action_type == action


# ---------------------------------------------------------------------------
# ActuatorStore — basic operations
# ---------------------------------------------------------------------------

class TestActuatorStoreBasic:
    def test_empty_store(self):
        s = ActuatorStore()
        assert s.count() == 0
        assert s.recent() == ()

    def test_record_and_count(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        assert s.count() == 1

    def test_record_multiple(self):
        s = ActuatorStore()
        for i in range(5):
            s.record(create_receipt("compile", True, float(i), float(i+1)))
        assert s.count() == 5

    def test_clear(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        s.clear()
        assert s.count() == 0


# ---------------------------------------------------------------------------
# ActuatorStore — ring buffer eviction
# ---------------------------------------------------------------------------

class TestRingBuffer:
    def test_max_receipts_enforced(self):
        s = ActuatorStore(max_receipts=5)
        for i in range(10):
            s.record(create_receipt("compile", True, float(i), float(i+1)))
        assert s.count() == 5

    def test_oldest_evicted(self):
        s = ActuatorStore(max_receipts=3)
        for i in range(5):
            s.record(create_receipt("compile", True, float(i), float(i+1),
                                    output_summary=f"run{i}"))
        recent = s.recent(n=10)
        assert len(recent) == 3
        # Should have runs 2, 3, 4 (oldest 0, 1 evicted)
        assert recent[0].output_summary == "run2"
        assert recent[2].output_summary == "run4"

    def test_101st_evicts_first(self):
        s = ActuatorStore(max_receipts=100)
        for i in range(101):
            s.record(create_receipt("compile", True, float(i), float(i+1),
                                    output_summary=f"run{i}"))
        assert s.count() == 100
        all_recent = s.recent(n=100)
        assert all_recent[0].output_summary == "run1"  # run0 evicted
        assert all_recent[-1].output_summary == "run100"  # newest

    def test_min_max_is_1(self):
        s = ActuatorStore(max_receipts=0)
        s.record(create_receipt("compile", True, 0.0, 1.0))
        s.record(create_receipt("compile", True, 1.0, 2.0))
        assert s.count() == 1


# ---------------------------------------------------------------------------
# ActuatorStore — recent
# ---------------------------------------------------------------------------

class TestRecent:
    def test_recent_returns_newest_last(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0, output_summary="first"))
        s.record(create_receipt("compile", True, 1.0, 2.0, output_summary="second"))
        recent = s.recent(n=10)
        assert recent[0].output_summary == "first"
        assert recent[1].output_summary == "second"

    def test_recent_n_limits(self):
        s = ActuatorStore()
        for i in range(10):
            s.record(create_receipt("compile", True, float(i), float(i+1)))
        recent = s.recent(n=3)
        assert len(recent) == 3

    def test_recent_n_more_than_stored(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        recent = s.recent(n=10)
        assert len(recent) == 1


# ---------------------------------------------------------------------------
# ActuatorStore — by_type
# ---------------------------------------------------------------------------

class TestByType:
    def test_filter_by_type(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        s.record(create_receipt("file_write", True, 1.0, 2.0))
        s.record(create_receipt("compile", False, 2.0, 3.0))
        compiles = s.by_type("compile")
        assert len(compiles) == 2
        assert all(r.action_type == "compile" for r in compiles)

    def test_filter_empty_type(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        assert s.by_type("voice") == ()


# ---------------------------------------------------------------------------
# ActuatorStore — summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        s = ActuatorStore()
        log = s.summary()
        assert log.total_actions == 0
        assert log.success_rate == 0.0
        assert log.total_cost == 0.0
        assert log.avg_duration == 0.0
        assert log.receipts == ()

    def test_summary_aggregation(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 10.0, cost_usd=0.10))
        s.record(create_receipt("compile", False, 10.0, 15.0, cost_usd=0.05))
        s.record(create_receipt("file_write", True, 15.0, 16.0, cost_usd=0.00))
        log = s.summary()
        assert log.total_actions == 3
        assert log.success_rate == pytest.approx(2/3)
        assert log.total_cost == pytest.approx(0.15)
        assert log.avg_duration == pytest.approx((10 + 5 + 1) / 3)

    def test_summary_all_success(self):
        s = ActuatorStore()
        for i in range(5):
            s.record(create_receipt("compile", True, float(i), float(i+1)))
        log = s.summary()
        assert log.success_rate == pytest.approx(1.0)

    def test_summary_all_failure(self):
        s = ActuatorStore()
        for i in range(5):
            s.record(create_receipt("compile", False, float(i), float(i+1)))
        log = s.summary()
        assert log.success_rate == pytest.approx(0.0)

    def test_summary_receipts_tuple(self):
        s = ActuatorStore()
        s.record(create_receipt("compile", True, 0.0, 1.0))
        log = s.summary()
        assert isinstance(log.receipts, tuple)


# ---------------------------------------------------------------------------
# ActuatorLog dataclass
# ---------------------------------------------------------------------------

class TestActuatorLog:
    def test_frozen(self):
        log = ActuatorLog(receipts=(), total_actions=0, success_rate=0.0, total_cost=0.0, avg_duration=0.0)
        with pytest.raises(AttributeError):
            log.total_actions = 5


# ---------------------------------------------------------------------------
# format_actuator_context
# ---------------------------------------------------------------------------

class TestFormatActuatorContext:
    def test_empty_log_returns_empty_string(self):
        log = ActuatorLog(receipts=(), total_actions=0, success_rate=0.0, total_cost=0.0, avg_duration=0.0)
        assert format_actuator_context(log) == ""

    def test_contains_header(self):
        r = create_receipt("compile", True, 0.0, 10.0)
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=1.0, total_cost=0.0, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "[Recent Actions]" in result

    def test_contains_summary_stats(self):
        r = create_receipt("compile", True, 0.0, 10.0, cost_usd=0.15)
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=1.0, total_cost=0.15, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "Total: 1" in result
        assert "100%" in result

    def test_success_shown_as_ok(self):
        r = create_receipt("compile", True, 0.0, 10.0)
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=1.0, total_cost=0.0, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "[ok]" in result

    def test_failure_shown_as_fail(self):
        r = create_receipt("compile", False, 0.0, 10.0, error="timeout")
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=0.0, total_cost=0.0, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "[FAIL]" in result
        assert "timeout" in result

    def test_cost_shown(self):
        r = create_receipt("compile", True, 0.0, 10.0, cost_usd=0.15)
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=1.0, total_cost=0.15, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "$0.15" in result or "0.1500" in result

    def test_output_summary_shown(self):
        r = create_receipt("compile", True, 0.0, 10.0, output_summary="Built 5 components")
        log = ActuatorLog(receipts=(r,), total_actions=1, success_rate=1.0, total_cost=0.0, avg_duration=10.0)
        result = format_actuator_context(log)
        assert "Built 5 components" in result

    def test_recent_n_limit(self):
        receipts = tuple(
            create_receipt("compile", True, float(i), float(i+1), output_summary=f"r{i}")
            for i in range(10)
        )
        log = ActuatorLog(receipts=receipts, total_actions=10, success_rate=1.0, total_cost=0.0, avg_duration=1.0)
        result = format_actuator_context(log, recent_n=3)
        # Should show at most 3 individual entries + header + summary line
        lines = [l for l in result.split("\n") if l.strip().startswith("[ok]")]
        assert len(lines) == 3
