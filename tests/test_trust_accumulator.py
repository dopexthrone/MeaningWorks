"""Tests for mother/trust_accumulator.py — cross-session trust accumulation."""

import pytest

from mother.trust_accumulator import (
    TrustSnapshot,
    update_trust,
    save_trust_snapshot,
    load_trust_snapshot,
    format_trust_context,
)


# ---------------------------------------------------------------------------
# TrustSnapshot dataclass
# ---------------------------------------------------------------------------

class TestTrustSnapshot:
    def test_frozen(self):
        s = TrustSnapshot()
        with pytest.raises(AttributeError):
            s.total_compilations = 5

    def test_defaults(self):
        s = TrustSnapshot()
        assert s.total_compilations == 0
        assert s.successful_compilations == 0
        assert s.session_success_rate == 0.0
        assert s.rolling_success_rate == 0.0
        assert s.avg_fidelity == 0.0
        assert s.avg_trust_score == 0.0
        assert s.consecutive_failures == 0

    def test_should_pause_autonomous_false(self):
        s = TrustSnapshot(consecutive_failures=2)
        assert not s.should_pause_autonomous

    def test_should_pause_autonomous_true(self):
        s = TrustSnapshot(consecutive_failures=3)
        assert s.should_pause_autonomous

    def test_should_pause_autonomous_above(self):
        s = TrustSnapshot(consecutive_failures=5)
        assert s.should_pause_autonomous


# ---------------------------------------------------------------------------
# update_trust — pure function
# ---------------------------------------------------------------------------

class TestUpdateTrust:
    def test_first_success(self):
        s = update_trust(TrustSnapshot(), success=True, fidelity=0.8, trust_score=85.0)
        assert s.total_compilations == 1
        assert s.successful_compilations == 1
        assert s.session_success_rate == 1.0
        assert s.rolling_success_rate == 1.0
        assert s.consecutive_failures == 0
        assert s.last_fidelity == 0.8
        assert s.last_trust_score == 85.0

    def test_first_failure(self):
        s = update_trust(TrustSnapshot(), success=False, fidelity=0.3, trust_score=40.0)
        assert s.total_compilations == 1
        assert s.successful_compilations == 0
        assert s.session_success_rate == 0.0
        assert s.rolling_success_rate == 0.0
        assert s.consecutive_failures == 1

    def test_consecutive_failures_count(self):
        s = TrustSnapshot()
        for _ in range(4):
            s = update_trust(s, success=False)
        assert s.consecutive_failures == 4
        assert s.should_pause_autonomous

    def test_success_resets_consecutive(self):
        s = TrustSnapshot(consecutive_failures=5)
        s = update_trust(s, success=True, fidelity=0.9, trust_score=90.0)
        assert s.consecutive_failures == 0

    def test_rolling_average(self):
        s = TrustSnapshot()
        # First success
        s = update_trust(s, success=True, fidelity=0.8, trust_score=80.0)
        assert s.rolling_success_rate == 1.0
        # Second failure
        s = update_trust(s, success=False, fidelity=0.4, trust_score=30.0)
        # rolling = 0.3 * 0.0 + 0.7 * 1.0 = 0.7
        assert s.rolling_success_rate == pytest.approx(0.7, abs=0.01)

    def test_session_success_rate(self):
        s = TrustSnapshot()
        s = update_trust(s, success=True)
        s = update_trust(s, success=True)
        s = update_trust(s, success=False)
        assert s.session_success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_fidelity_clamped(self):
        s = update_trust(TrustSnapshot(), success=True, fidelity=1.5, trust_score=200.0)
        assert s.last_fidelity == 1.0
        assert s.last_trust_score == 100.0

    def test_fidelity_negative_clamped(self):
        s = update_trust(TrustSnapshot(), success=True, fidelity=-0.5, trust_score=-10.0)
        assert s.last_fidelity == 0.0
        assert s.last_trust_score == 0.0

    def test_avg_fidelity_tracks(self):
        s = TrustSnapshot()
        s = update_trust(s, success=True, fidelity=0.8)
        assert s.avg_fidelity == 0.8
        s = update_trust(s, success=True, fidelity=0.6)
        # 0.3 * 0.6 + 0.7 * 0.8 = 0.18 + 0.56 = 0.74
        assert s.avg_fidelity == pytest.approx(0.74, abs=0.01)

    def test_ten_compilations(self):
        s = TrustSnapshot()
        for i in range(10):
            s = update_trust(s, success=(i % 3 != 0), fidelity=0.7, trust_score=75.0)
        assert s.total_compilations == 10
        # i=0 fail, i=1 success, i=2 success, i=3 fail, i=4 success, i=5 success,
        # i=6 fail, i=7 success, i=8 success, i=9 fail
        # 6 successes, 4 failures
        assert s.successful_compilations == 6

    def test_empty_snapshot_preserves(self):
        s = TrustSnapshot()
        s2 = update_trust(s, success=True)
        # Original unchanged (frozen)
        assert s.total_compilations == 0
        assert s2.total_compilations == 1


# ---------------------------------------------------------------------------
# Persistence — save/load
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        s = TrustSnapshot(
            total_compilations=10,
            successful_compilations=8,
            session_success_rate=0.8,
            rolling_success_rate=0.85,
            avg_fidelity=0.75,
            avg_trust_score=82.0,
            consecutive_failures=0,
            last_fidelity=0.78,
            last_trust_score=85.0,
        )
        save_trust_snapshot(s, db_dir=tmp_path)
        loaded = load_trust_snapshot(db_dir=tmp_path)
        assert loaded.total_compilations == 10
        assert loaded.successful_compilations == 8
        assert loaded.rolling_success_rate == pytest.approx(0.85)
        assert loaded.avg_fidelity == pytest.approx(0.75)
        assert loaded.consecutive_failures == 0

    def test_load_empty_db(self, tmp_path):
        s = load_trust_snapshot(db_dir=tmp_path)
        assert s.total_compilations == 0

    def test_load_nonexistent_dir(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        s = load_trust_snapshot(db_dir=nonexistent)
        assert s.total_compilations == 0

    def test_multiple_saves_returns_latest(self, tmp_path):
        s1 = TrustSnapshot(total_compilations=5, successful_compilations=4)
        s2 = TrustSnapshot(total_compilations=10, successful_compilations=8)
        save_trust_snapshot(s1, db_dir=tmp_path)
        save_trust_snapshot(s2, db_dir=tmp_path)
        loaded = load_trust_snapshot(db_dir=tmp_path)
        assert loaded.total_compilations == 10

    def test_round_trip_preserves_all_fields(self, tmp_path):
        s = TrustSnapshot(
            total_compilations=15,
            successful_compilations=12,
            session_success_rate=0.8,
            rolling_success_rate=0.82,
            avg_fidelity=0.71,
            avg_trust_score=78.5,
            consecutive_failures=2,
            last_fidelity=0.65,
            last_trust_score=70.0,
        )
        save_trust_snapshot(s, db_dir=tmp_path)
        loaded = load_trust_snapshot(db_dir=tmp_path)
        assert loaded.total_compilations == s.total_compilations
        assert loaded.successful_compilations == s.successful_compilations
        assert loaded.session_success_rate == pytest.approx(s.session_success_rate)
        assert loaded.rolling_success_rate == pytest.approx(s.rolling_success_rate)
        assert loaded.avg_fidelity == pytest.approx(s.avg_fidelity)
        assert loaded.avg_trust_score == pytest.approx(s.avg_trust_score)
        assert loaded.consecutive_failures == s.consecutive_failures
        assert loaded.last_fidelity == pytest.approx(s.last_fidelity)
        assert loaded.last_trust_score == pytest.approx(s.last_trust_score)


# ---------------------------------------------------------------------------
# format_trust_context
# ---------------------------------------------------------------------------

class TestFormatTrustContext:
    def test_empty_returns_empty_string(self):
        s = TrustSnapshot()
        assert format_trust_context(s) == ""

    def test_contains_header(self):
        s = TrustSnapshot(total_compilations=1, successful_compilations=1,
                          rolling_success_rate=1.0, session_success_rate=1.0)
        result = format_trust_context(s)
        assert "[Trust Accumulator]" in result

    def test_contains_compilation_count(self):
        s = TrustSnapshot(total_compilations=10, successful_compilations=8,
                          rolling_success_rate=0.8, session_success_rate=0.8)
        result = format_trust_context(s)
        assert "10" in result
        assert "8 successful" in result

    def test_success_rate_shown(self):
        s = TrustSnapshot(total_compilations=5, successful_compilations=4,
                          rolling_success_rate=0.85, session_success_rate=0.8)
        result = format_trust_context(s)
        assert "85%" in result or "80%" in result

    def test_fidelity_shown(self):
        s = TrustSnapshot(total_compilations=5, avg_fidelity=0.78)
        result = format_trust_context(s)
        assert "0.78" in result

    def test_consecutive_failures_shown(self):
        s = TrustSnapshot(total_compilations=5, consecutive_failures=2)
        result = format_trust_context(s)
        assert "2" in result

    def test_pause_warning_shown(self):
        s = TrustSnapshot(total_compilations=5, consecutive_failures=3)
        result = format_trust_context(s)
        assert "paused" in result.lower() or "Autonomous" in result

    def test_zero_fidelity_hidden(self):
        s = TrustSnapshot(total_compilations=1, avg_fidelity=0.0)
        result = format_trust_context(s)
        assert "fidelity" not in result.lower()

    def test_zero_trust_hidden(self):
        s = TrustSnapshot(total_compilations=1, avg_trust_score=0.0)
        result = format_trust_context(s)
        assert "trust:" not in result.lower() or "Avg trust" not in result
