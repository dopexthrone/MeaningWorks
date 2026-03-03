"""Tests for mother/environment_model.py — decaying world state model."""

import math
import time

import pytest

from mother.environment_model import (
    EnvironmentEntry,
    EnvironmentSnapshot,
    EnvironmentModel,
    create_model,
    format_environment_context,
)


# ---------------------------------------------------------------------------
# EnvironmentEntry
# ---------------------------------------------------------------------------

class TestEnvironmentEntry:
    def test_frozen(self):
        e = EnvironmentEntry(modality="screen", summary="VS Code", confidence=1.0, observed_at=100.0)
        with pytest.raises(AttributeError):
            e.modality = "camera"

    def test_defaults(self):
        e = EnvironmentEntry(modality="screen", summary="x", confidence=0.5, observed_at=0.0)
        assert e.raw_hash == ""

    def test_with_hash(self):
        e = EnvironmentEntry(modality="speech", summary="hello", confidence=0.9, observed_at=1.0, raw_hash="abc123")
        assert e.raw_hash == "abc123"


# ---------------------------------------------------------------------------
# EnvironmentSnapshot
# ---------------------------------------------------------------------------

class TestEnvironmentSnapshot:
    def test_frozen(self):
        s = EnvironmentSnapshot(entries=(), taken_at=0.0, dominant_context="", staleness=0.0)
        with pytest.raises(AttributeError):
            s.taken_at = 1.0

    def test_empty(self):
        s = EnvironmentSnapshot(entries=(), taken_at=100.0, dominant_context="", staleness=0.0)
        assert s.dominant_context == ""
        assert s.staleness == 0.0


# ---------------------------------------------------------------------------
# create_model factory
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_default_half_lives(self):
        m = create_model()
        assert m.half_lives == {"screen": 60.0, "speech": 30.0, "camera": 120.0}

    def test_custom_half_lives(self):
        m = create_model(half_lives={"screen": 10.0})
        assert m.half_lives == {"screen": 10.0}

    def test_returns_environment_model(self):
        m = create_model()
        assert isinstance(m, EnvironmentModel)


# ---------------------------------------------------------------------------
# EnvironmentModel.observe
# ---------------------------------------------------------------------------

class TestObserve:
    def test_observe_adds_entry(self):
        m = create_model()
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        assert m.entry_count() == 1

    def test_observe_multiple_modalities(self):
        m = create_model()
        m.observe("screen", "Terminal", now=100.0)
        m.observe("speech", "user talking", now=100.0)
        m.observe("camera", "user at desk", now=100.0)
        assert m.entry_count() == 3

    def test_observe_dedup_by_hash(self):
        m = create_model()
        m.observe("screen", "VS Code v1", raw_hash="h1", now=100.0)
        m.observe("screen", "VS Code v2", raw_hash="h1", now=110.0)
        # Same hash → update, not duplicate
        assert m.entry_count() == 1

    def test_observe_different_hashes_no_dedup(self):
        m = create_model()
        m.observe("screen", "VS Code", raw_hash="h1", now=100.0)
        m.observe("screen", "Terminal", raw_hash="h2", now=100.0)
        assert m.entry_count() == 2

    def test_observe_auto_key_dedup(self):
        m = create_model()
        m.observe("screen", "VS Code", now=100.0)
        m.observe("screen", "VS Code", now=110.0)
        # Same modality+summary → same auto-key → update
        assert m.entry_count() == 1

    def test_observe_clamps_confidence(self):
        m = create_model()
        m.observe("screen", "x", confidence=1.5, now=100.0)
        snap = m.snapshot(now=100.0)
        assert snap.entries[0].confidence <= 1.0

    def test_observe_clamps_negative_confidence(self):
        m = create_model()
        m.observe("screen", "x", confidence=-0.5, now=100.0)
        snap = m.snapshot(now=100.0)
        # Negative clamped to 0.0, which is below prune threshold
        assert len(snap.entries) == 0

    def test_observe_default_confidence_is_1(self):
        m = create_model()
        m.observe("screen", "x", now=100.0)
        snap = m.snapshot(now=100.0)
        assert snap.entries[0].confidence == 1.0


# ---------------------------------------------------------------------------
# EnvironmentModel.snapshot — decay
# ---------------------------------------------------------------------------

class TestDecay:
    def test_no_decay_at_observation_time(self):
        m = create_model(half_lives={"screen": 60.0})
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        snap = m.snapshot(now=100.0)
        assert len(snap.entries) == 1
        assert snap.entries[0].confidence == pytest.approx(1.0)

    def test_half_decay_at_half_life(self):
        m = create_model(half_lives={"screen": 60.0})
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        snap = m.snapshot(now=160.0)  # 60s later
        assert len(snap.entries) == 1
        assert snap.entries[0].confidence == pytest.approx(0.5, abs=0.01)

    def test_quarter_at_two_half_lives(self):
        m = create_model(half_lives={"screen": 60.0})
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        snap = m.snapshot(now=220.0)  # 120s = 2 half-lives
        assert snap.entries[0].confidence == pytest.approx(0.25, abs=0.01)

    def test_prune_below_threshold(self):
        m = create_model(half_lives={"screen": 10.0})
        m.observe("screen", "VS Code", confidence=0.1, now=100.0)
        # After ~4.3 half-lives (43s), 0.1 * 0.5^4.3 ≈ 0.005 < 0.05
        snap = m.snapshot(now=143.0)
        assert len(snap.entries) == 0

    def test_different_modalities_different_half_lives(self):
        m = create_model(half_lives={"screen": 60.0, "speech": 30.0})
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        m.observe("speech", "user talking", confidence=1.0, now=100.0)
        snap = m.snapshot(now=130.0)  # 30s later
        screen_entry = [e for e in snap.entries if e.modality == "screen"][0]
        speech_entry = [e for e in snap.entries if e.modality == "speech"][0]
        # Screen at 30s/60s half-life ≈ 0.707
        assert screen_entry.confidence == pytest.approx(0.707, abs=0.01)
        # Speech at 30s/30s half-life = 0.5
        assert speech_entry.confidence == pytest.approx(0.5, abs=0.01)

    def test_unknown_modality_uses_default_60s(self):
        m = create_model()
        m.observe("lidar", "something", confidence=1.0, now=100.0)
        snap = m.snapshot(now=160.0)  # 60s
        assert snap.entries[0].confidence == pytest.approx(0.5, abs=0.01)

    def test_zero_half_life_decays_immediately(self):
        m = create_model(half_lives={"screen": 0.0})
        m.observe("screen", "x", confidence=1.0, now=100.0)
        snap = m.snapshot(now=100.1)
        assert len(snap.entries) == 0


# ---------------------------------------------------------------------------
# EnvironmentModel.snapshot — ordering & fields
# ---------------------------------------------------------------------------

class TestSnapshotFields:
    def test_entries_sorted_by_confidence_descending(self):
        m = create_model(half_lives={"screen": 60.0, "speech": 30.0})
        m.observe("speech", "talking", confidence=0.5, now=100.0)
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        snap = m.snapshot(now=100.0)
        assert snap.entries[0].modality == "screen"  # 1.0 > 0.5
        assert snap.entries[1].modality == "speech"

    def test_dominant_context_is_highest_confidence(self):
        m = create_model()
        m.observe("screen", "Terminal", confidence=0.8, now=100.0)
        m.observe("speech", "user speaking", confidence=0.3, now=100.0)
        snap = m.snapshot(now=100.0)
        assert snap.dominant_context == "Terminal"

    def test_staleness_computation(self):
        m = create_model()
        m.observe("screen", "VS Code", now=90.0)
        m.observe("speech", "talking", now=100.0)
        snap = m.snapshot(now=110.0)
        # Ages: 20s and 10s, avg = 15s
        assert snap.staleness == pytest.approx(15.0, abs=0.5)

    def test_taken_at(self):
        m = create_model()
        m.observe("screen", "x", now=100.0)
        snap = m.snapshot(now=150.0)
        assert snap.taken_at == 150.0


# ---------------------------------------------------------------------------
# EnvironmentModel.query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_filters_by_modality(self):
        m = create_model()
        m.observe("screen", "VS Code", now=100.0)
        m.observe("speech", "hello", now=100.0)
        m.observe("camera", "desk", now=100.0)
        results = m.query("screen", now=100.0)
        assert len(results) == 1
        assert results[0].modality == "screen"

    def test_query_empty_modality(self):
        m = create_model()
        m.observe("screen", "x", now=100.0)
        results = m.query("speech", now=100.0)
        assert results == []

    def test_query_applies_decay(self):
        m = create_model(half_lives={"screen": 10.0})
        m.observe("screen", "x", confidence=0.1, now=100.0)
        # After enough time, entry decays below threshold
        results = m.query("screen", now=200.0)
        assert results == []


# ---------------------------------------------------------------------------
# EnvironmentModel.dominant
# ---------------------------------------------------------------------------

class TestDominant:
    def test_dominant_returns_best_summary(self):
        m = create_model()
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        m.observe("speech", "whispering", confidence=0.3, now=100.0)
        assert m.dominant(now=100.0) == "VS Code"

    def test_dominant_empty_model(self):
        m = create_model()
        assert m.dominant(now=100.0) == ""


# ---------------------------------------------------------------------------
# EnvironmentModel misc
# ---------------------------------------------------------------------------

class TestMisc:
    def test_clear(self):
        m = create_model()
        m.observe("screen", "x", now=100.0)
        m.observe("speech", "y", now=100.0)
        m.clear()
        assert m.entry_count() == 0
        snap = m.snapshot(now=100.0)
        assert len(snap.entries) == 0

    def test_empty_snapshot(self):
        m = create_model()
        snap = m.snapshot(now=100.0)
        assert len(snap.entries) == 0
        assert snap.dominant_context == ""
        assert snap.staleness == 0.0

    def test_multiple_snapshots_progressive_decay(self):
        m = create_model(half_lives={"screen": 60.0})
        m.observe("screen", "VS Code", confidence=1.0, now=100.0)
        snap1 = m.snapshot(now=130.0)
        snap2 = m.snapshot(now=160.0)
        # Confidence should decrease over time
        assert snap1.entries[0].confidence > snap2.entries[0].confidence

    def test_observe_refreshes_timestamp(self):
        m = create_model()
        m.observe("screen", "VS Code", raw_hash="h1", now=100.0)
        m.observe("screen", "VS Code updated", raw_hash="h1", now=200.0)
        snap = m.snapshot(now=200.0)
        # Entry was refreshed at t=200, so no decay
        assert snap.entries[0].confidence == pytest.approx(1.0)
        assert snap.entries[0].observed_at == 200.0


# ---------------------------------------------------------------------------
# format_environment_context
# ---------------------------------------------------------------------------

class TestFormatEnvironmentContext:
    def test_empty_snapshot_returns_empty_string(self):
        snap = EnvironmentSnapshot(entries=(), taken_at=100.0, dominant_context="", staleness=0.0)
        assert format_environment_context(snap) == ""

    def test_contains_header(self):
        e = EnvironmentEntry(modality="screen", summary="VS Code", confidence=0.9, observed_at=95.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="VS Code", staleness=5.0)
        result = format_environment_context(snap)
        assert "[Environment State]" in result

    def test_contains_modality_and_summary(self):
        e = EnvironmentEntry(modality="screen", summary="VS Code - main.py", confidence=0.9, observed_at=95.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="VS Code - main.py", staleness=5.0)
        result = format_environment_context(snap)
        assert "screen" in result
        assert "VS Code - main.py" in result

    def test_contains_confidence(self):
        e = EnvironmentEntry(modality="screen", summary="x", confidence=0.75, observed_at=90.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="x", staleness=10.0)
        result = format_environment_context(snap)
        assert "0.75" in result

    def test_age_in_seconds(self):
        e = EnvironmentEntry(modality="screen", summary="x", confidence=0.9, observed_at=70.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="x", staleness=30.0)
        result = format_environment_context(snap)
        assert "30s ago" in result

    def test_age_in_minutes(self):
        e = EnvironmentEntry(modality="screen", summary="x", confidence=0.9, observed_at=0.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=120.0, dominant_context="x", staleness=120.0)
        result = format_environment_context(snap)
        assert "2.0m ago" in result

    def test_dominant_shown(self):
        e = EnvironmentEntry(modality="screen", summary="Terminal", confidence=0.9, observed_at=95.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="Terminal", staleness=5.0)
        result = format_environment_context(snap)
        assert "Dominant: Terminal" in result

    def test_staleness_shown(self):
        e = EnvironmentEntry(modality="screen", summary="x", confidence=0.9, observed_at=85.0)
        snap = EnvironmentSnapshot(entries=(e,), taken_at=100.0, dominant_context="x", staleness=15.0)
        result = format_environment_context(snap)
        assert "Staleness: 15.0s" in result

    def test_caps_at_5_entries(self):
        entries = tuple(
            EnvironmentEntry(modality="screen", summary=f"entry{i}", confidence=1.0 - i*0.1, observed_at=100.0)
            for i in range(8)
        )
        snap = EnvironmentSnapshot(entries=entries, taken_at=100.0, dominant_context="entry0", staleness=0.0)
        result = format_environment_context(snap)
        # Should show at most 5 modality lines
        assert result.count("screen:") == 5

    def test_multiple_modalities(self):
        entries = (
            EnvironmentEntry(modality="screen", summary="VS Code", confidence=0.9, observed_at=95.0),
            EnvironmentEntry(modality="speech", summary="talking", confidence=0.7, observed_at=90.0),
        )
        snap = EnvironmentSnapshot(entries=entries, taken_at=100.0, dominant_context="VS Code", staleness=7.5)
        result = format_environment_context(snap)
        assert "screen" in result
        assert "speech" in result
