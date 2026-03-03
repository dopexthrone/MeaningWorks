"""
Tests for core/rejection_log.py — governor rejection event store.
"""

import os
import tempfile

import pytest

from core.rejection_log import (
    RejectionEvent,
    RejectionLog,
    RejectionSummary,
)


# --- RejectionEvent ---


class TestRejectionEvent:
    """Test RejectionEvent frozen dataclass."""

    def test_construction(self):
        event = RejectionEvent(
            timestamp="2026-02-14T10:00:00Z",
            package_id="pkg-123",
            source_instance="inst-abc",
            rejection_reason="Trust score too low",
            failed_check="trust",
            trust_score=45.0,
            provenance_depth=2,
        )
        assert event.timestamp == "2026-02-14T10:00:00Z"
        assert event.package_id == "pkg-123"
        assert event.failed_check == "trust"
        assert event.trust_score == 45.0

    def test_frozen(self):
        event = RejectionEvent(
            timestamp="2026-02-14T10:00:00Z",
            package_id="pkg-123",
            source_instance="inst-abc",
            rejection_reason="test",
            failed_check="trust",
            trust_score=45.0,
            provenance_depth=2,
        )
        with pytest.raises(AttributeError):
            event.trust_score = 100.0


# --- RejectionSummary ---


class TestRejectionSummary:
    """Test RejectionSummary frozen dataclass."""

    def test_construction(self):
        summary = RejectionSummary(
            total_rejections=5,
            by_check={"trust": 3, "provenance": 2},
            recent_reasons=("reason1", "reason2"),
            remediation_hints=("hint1",),
        )
        assert summary.total_rejections == 5
        assert summary.by_check["trust"] == 3
        assert len(summary.recent_reasons) == 2
        assert len(summary.remediation_hints) == 1

    def test_frozen(self):
        summary = RejectionSummary(
            total_rejections=0,
            by_check={},
            recent_reasons=(),
            remediation_hints=(),
        )
        with pytest.raises(AttributeError):
            summary.total_rejections = 10

    def test_empty_summary(self):
        summary = RejectionSummary(
            total_rejections=0,
            by_check={},
            recent_reasons=(),
            remediation_hints=(),
        )
        assert summary.total_rejections == 0
        assert summary.by_check == {}


# --- RejectionLog ---


def _make_event(
    failed_check: str = "trust",
    trust_score: float = 45.0,
    reason: str = "Trust score too low",
) -> RejectionEvent:
    return RejectionEvent(
        timestamp="2026-02-14T10:00:00Z",
        package_id="pkg-test",
        source_instance="inst-test",
        rejection_reason=reason,
        failed_check=failed_check,
        trust_score=trust_score,
        provenance_depth=1,
    )


class TestRejectionLogInit:
    """Test RejectionLog initialization."""

    def test_creates_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            assert os.path.isfile(db_path)

    def test_default_path(self):
        log = RejectionLog()
        assert log._db_path.endswith("rejections.db")
        assert ".motherlabs" in log._db_path


class TestRejectionLogRecord:
    """Test recording rejection events."""

    def test_record_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            event = _make_event()
            log.record(event)

            summary = log.get_summary()
            assert summary.total_rejections == 1

    def test_record_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            log.record(_make_event(failed_check="trust"))
            log.record(_make_event(failed_check="provenance"))
            log.record(_make_event(failed_check="trust"))

            summary = log.get_summary()
            assert summary.total_rejections == 3
            assert summary.by_check["trust"] == 2
            assert summary.by_check["provenance"] == 1


class TestRejectionLogGetSummary:
    """Test summary aggregation."""

    def test_empty_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            summary = log.get_summary()
            assert summary.total_rejections == 0
            assert summary.by_check == {}
            assert summary.recent_reasons == ()
            assert summary.remediation_hints == ()

    def test_by_check_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            log.record(_make_event(failed_check="trust"))
            log.record(_make_event(failed_check="trust"))
            log.record(_make_event(failed_check="code_safety"))

            summary = log.get_summary()
            assert summary.by_check == {"trust": 2, "code_safety": 1}

    def test_recent_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            for i in range(7):
                log.record(_make_event(reason=f"reason_{i}"))

            summary = log.get_summary()
            assert len(summary.recent_reasons) == 5
            # Most recent should be last inserted
            assert summary.recent_reasons[0] == "reason_6"

    def test_remediation_hints_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            log.record(_make_event(failed_check="trust"))
            log.record(_make_event(failed_check="code_safety"))

            summary = log.get_summary()
            assert len(summary.remediation_hints) >= 2
            # Trust hint should mention verification scores
            trust_hints = [h for h in summary.remediation_hints if "verification" in h.lower() or "trust" in h.lower()]
            assert len(trust_hints) >= 1
            # Code safety hint should mention exec/eval
            safety_hints = [h for h in summary.remediation_hints if "exec" in h.lower()]
            assert len(safety_hints) >= 1


class TestRejectionLogGetRecent:
    """Test recent event retrieval."""

    def test_get_recent_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            events = log.get_recent()
            assert events == []

    def test_get_recent_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            log.record(_make_event(reason="first"))
            log.record(_make_event(reason="second"))
            log.record(_make_event(reason="third"))

            events = log.get_recent(limit=2)
            assert len(events) == 2
            assert events[0].rejection_reason == "third"
            assert events[1].rejection_reason == "second"

    def test_get_recent_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            for i in range(20):
                log.record(_make_event(reason=f"event_{i}"))

            events = log.get_recent(limit=5)
            assert len(events) == 5

    def test_get_recent_returns_full_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            log = RejectionLog(db_path=db_path)
            log.record(RejectionEvent(
                timestamp="2026-02-14T12:00:00Z",
                package_id="pkg-abc",
                source_instance="inst-xyz",
                rejection_reason="Bad provenance",
                failed_check="provenance",
                trust_score=30.0,
                provenance_depth=0,
            ))

            events = log.get_recent(limit=1)
            assert len(events) == 1
            e = events[0]
            assert e.timestamp == "2026-02-14T12:00:00Z"
            assert e.package_id == "pkg-abc"
            assert e.source_instance == "inst-xyz"
            assert e.failed_check == "provenance"
            assert e.trust_score == 30.0
            assert e.provenance_depth == 0


class TestRemediationHints:
    """Test pattern-based remediation hint generation."""

    def test_trust_hint(self):
        hints = RejectionLog.generate_remediation_hints({"trust": 3})
        assert len(hints) == 1
        assert "verification" in hints[0].lower() or "trust" in hints[0].lower()

    def test_provenance_hint(self):
        hints = RejectionLog.generate_remediation_hints({"provenance": 2})
        assert len(hints) == 1
        assert "provenance" in hints[0].lower()

    def test_code_safety_hint(self):
        hints = RejectionLog.generate_remediation_hints({"code_safety": 1})
        assert len(hints) == 1
        assert "exec" in hints[0].lower() or "eval" in hints[0].lower()

    def test_multiple_check_types(self):
        hints = RejectionLog.generate_remediation_hints(
            {"trust": 5, "provenance": 2, "code_safety": 1}
        )
        assert len(hints) == 3
        # Most common first
        assert "trust" in hints[0].lower() or "verification" in hints[0].lower()

    def test_empty_checks(self):
        hints = RejectionLog.generate_remediation_hints({})
        assert hints == ()

    def test_none_checks(self):
        hints = RejectionLog.generate_remediation_hints(None)
        assert hints == ()

    def test_unknown_check_type(self):
        hints = RejectionLog.generate_remediation_hints({"unknown_check": 1})
        assert hints == ()

    def test_blueprint_hint(self):
        hints = RejectionLog.generate_remediation_hints({"blueprint": 2})
        assert len(hints) == 1
        assert "blueprint" in hints[0].lower()

    def test_fingerprint_hint(self):
        hints = RejectionLog.generate_remediation_hints({"fingerprint": 1})
        assert len(hints) == 1
        assert "fingerprint" in hints[0].lower() or "duplicate" in hints[0].lower()


# --- tool_export integration ---


class TestToolExportRejectionLogging:
    """Test that tool_export logs rejections via _identify_failed_check."""

    def test_identify_trust_failure(self):
        from core.tool_export import _identify_failed_check
        from unittest.mock import MagicMock

        result = MagicMock()
        result.provenance_valid = True
        result.trust_sufficient = False
        result.code_safe = True
        result.rejection_reason = "Trust score 45 < 60"

        assert _identify_failed_check(result) == "trust"

    def test_identify_provenance_failure(self):
        from core.tool_export import _identify_failed_check
        from unittest.mock import MagicMock

        result = MagicMock()
        result.provenance_valid = False
        result.trust_sufficient = True
        result.code_safe = True
        result.rejection_reason = "Missing provenance"

        assert _identify_failed_check(result) == "provenance"

    def test_identify_code_safety_failure(self):
        from core.tool_export import _identify_failed_check
        from unittest.mock import MagicMock

        result = MagicMock()
        result.provenance_valid = True
        result.trust_sufficient = True
        result.code_safe = False
        result.rejection_reason = "Unsafe code"

        assert _identify_failed_check(result) == "code_safety"

    def test_identify_fingerprint_from_reason(self):
        from core.tool_export import _identify_failed_check
        from unittest.mock import MagicMock

        result = MagicMock()
        result.provenance_valid = True
        result.trust_sufficient = True
        result.code_safe = True
        result.rejection_reason = "Duplicate fingerprint: already registered"

        assert _identify_failed_check(result) == "fingerprint"

    def test_identify_unknown(self):
        from core.tool_export import _identify_failed_check
        from unittest.mock import MagicMock

        result = MagicMock()
        result.provenance_valid = True
        result.trust_sufficient = True
        result.code_safe = True
        result.rejection_reason = "Something weird happened"

        assert _identify_failed_check(result) == "unknown"


# --- Context rendering ---


class TestContextRejectionRendering:
    """Test rejection count in context synthesis."""

    def test_rejection_count_shown(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(rejection_count=3)
        result = synthesize_situation(data)
        assert "3 tool imports rejected" in result

    def test_singular_rejection(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(rejection_count=1)
        result = synthesize_situation(data)
        assert "1 tool import rejected" in result
        assert "imports" not in result

    def test_zero_rejections_hidden(self):
        from mother.context import ContextData, synthesize_situation
        data = ContextData(rejection_count=0)
        result = synthesize_situation(data)
        assert "rejected" not in result
