"""Tests for mother/perception_fusion.py — cross-modal correlation."""

import time

import pytest

from mother.perception_fusion import (
    FusionEvent,
    FusionSignal,
    PerceptionFusion,
    format_fusion_context,
)


# ---------------------------------------------------------------------------
# FusionEvent / FusionSignal dataclasses
# ---------------------------------------------------------------------------

class TestFusionEvent:
    def test_frozen(self):
        e = FusionEvent(event_type="screen_changed", timestamp=100.0)
        with pytest.raises(AttributeError):
            e.event_type = "x"

    def test_defaults(self):
        e = FusionEvent(event_type="screen_changed", timestamp=100.0)
        assert e.summary == ""

    def test_with_summary(self):
        e = FusionEvent(event_type="speech_detected", timestamp=100.0, summary="hello world")
        assert e.summary == "hello world"


class TestFusionSignal:
    def test_frozen(self):
        s = FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen",), detected_at=100.0)
        with pytest.raises(AttributeError):
            s.pattern = "away"

    def test_evidence_tuple(self):
        s = FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen", "speech"), detected_at=100.0)
        assert len(s.evidence) == 2


# ---------------------------------------------------------------------------
# PerceptionFusion — basic operations
# ---------------------------------------------------------------------------

class TestFusionBasics:
    def test_empty_returns_no_signals(self):
        f = PerceptionFusion()
        signals = f.detect(now=100.0)
        assert signals == []

    def test_ingest_adds_event(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        assert f.event_count() == 1

    def test_window_property(self):
        f = PerceptionFusion(window_seconds=20.0)
        assert f.window_seconds == 20.0

    def test_clear(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.clear()
        assert f.event_count() == 0

    def test_prune_expired(self):
        f = PerceptionFusion(window_seconds=10.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=115.0))
        # First event (t=100) should be pruned at t=115 with 10s window
        assert f.event_count() == 1


# ---------------------------------------------------------------------------
# Single-modality → no cross-modal signal
# ---------------------------------------------------------------------------

class TestSingleModality:
    def test_screen_only_produces_focused(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        signals = f.detect(now=100.0)
        patterns = [s.pattern for s in signals]
        assert "focused" in patterns
        assert "presenting" not in patterns

    def test_speech_only_no_signal(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=100.0))
        signals = f.detect(now=100.0)
        patterns = [s.pattern for s in signals]
        assert "presenting" not in patterns
        assert "focused" not in patterns

    def test_camera_only_no_fusion_signal(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="camera_frame", timestamp=100.0))
        signals = f.detect(now=100.0)
        patterns = [s.pattern for s in signals]
        assert "presenting" not in patterns
        assert "focused" not in patterns


# ---------------------------------------------------------------------------
# Cross-modal patterns
# ---------------------------------------------------------------------------

class TestCrossModalPatterns:
    def test_screen_plus_speech_is_presenting(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=105.0))
        signals = f.detect(now=105.0)
        patterns = [s.pattern for s in signals]
        assert "presenting" in patterns

    def test_presenting_confidence_is_0_8(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=105.0))
        signals = f.detect(now=105.0)
        presenting = [s for s in signals if s.pattern == "presenting"][0]
        assert presenting.confidence == pytest.approx(0.8)

    def test_presenting_evidence(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=105.0))
        signals = f.detect(now=105.0)
        presenting = [s for s in signals if s.pattern == "presenting"][0]
        assert "screen" in presenting.evidence
        assert "speech" in presenting.evidence

    def test_frequent_screen_plus_speech_is_multitasking(self):
        f = PerceptionFusion()
        # 3+ screen events = frequent
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=102.0))
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=104.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=105.0))
        signals = f.detect(now=105.0)
        patterns = [s.pattern for s in signals]
        assert "multitasking" in patterns
        # Multitasking replaces presenting (more specific)
        assert "presenting" not in patterns

    def test_multitasking_confidence_is_0_7(self):
        f = PerceptionFusion()
        for i in range(3):
            f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0 + i))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=103.0))
        signals = f.detect(now=103.0)
        mt = [s for s in signals if s.pattern == "multitasking"][0]
        assert mt.confidence == pytest.approx(0.7)

    def test_focused_pattern(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        signals = f.detect(now=100.0)
        focused = [s for s in signals if s.pattern == "focused"]
        assert len(focused) == 1
        assert focused[0].confidence == pytest.approx(0.6)
        assert "screen" in focused[0].evidence


# ---------------------------------------------------------------------------
# Idle pattern
# ---------------------------------------------------------------------------

class TestIdlePattern:
    def test_idle_after_timeout(self):
        f = PerceptionFusion(idle_seconds=60.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        # Window expires, then idle timeout passes
        signals = f.detect(now=200.0)
        patterns = [s.pattern for s in signals]
        assert "idle" in patterns

    def test_idle_confidence_is_0_5(self):
        f = PerceptionFusion(idle_seconds=60.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        signals = f.detect(now=200.0)
        idle_sig = [s for s in signals if s.pattern == "idle"][0]
        assert idle_sig.confidence == pytest.approx(0.5)

    def test_no_idle_if_recent_events(self):
        f = PerceptionFusion(idle_seconds=60.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        signals = f.detect(now=105.0)
        patterns = [s.pattern for s in signals]
        assert "idle" not in patterns

    def test_no_idle_if_never_had_events(self):
        f = PerceptionFusion(idle_seconds=60.0)
        signals = f.detect(now=200.0)
        assert signals == []


# ---------------------------------------------------------------------------
# Away pattern
# ---------------------------------------------------------------------------

class TestAwayPattern:
    def test_no_away_before_idle_gap(self):
        """No 'away' signal when gap < idle_seconds — sensors off ≠ user absent."""
        f = PerceptionFusion(window_seconds=10.0, idle_seconds=30.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        # At t=115: gap=15 < idle_seconds=30 → no signal at all
        signals = f.detect(now=115.0)
        patterns = [s.pattern for s in signals]
        assert "away" not in patterns

    def test_idle_replaces_away_after_gap(self):
        """After idle_seconds gap, 'idle' fires (not 'away' at 0.9)."""
        f = PerceptionFusion(window_seconds=10.0, idle_seconds=30.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        # At t=135: gap=35 >= idle_seconds=30 → idle at 0.5, NOT away at 0.9
        signals = f.detect(now=135.0)
        patterns = [s.pattern for s in signals]
        assert "idle" in patterns
        assert "away" not in patterns

    def test_idle_confidence_is_moderate(self):
        """Idle confidence is 0.5 — not the false 0.9 from old away pattern."""
        f = PerceptionFusion(window_seconds=10.0, idle_seconds=30.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        signals = f.detect(now=135.0)
        idle = [s for s in signals if s.pattern == "idle"]
        assert len(idle) == 1
        assert idle[0].confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Window expiry
# ---------------------------------------------------------------------------

class TestWindowExpiry:
    def test_events_outside_window_dont_fuse(self):
        f = PerceptionFusion(window_seconds=10.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=115.0))
        # Screen event at t=100 is outside window at t=115 (window=10)
        signals = f.detect(now=115.0)
        patterns = [s.pattern for s in signals]
        assert "presenting" not in patterns

    def test_both_within_window_fuse(self):
        f = PerceptionFusion(window_seconds=10.0)
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=108.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=115.0))
        signals = f.detect(now=115.0)
        patterns = [s.pattern for s in signals]
        assert "presenting" in patterns

    def test_rapid_screen_detection_exact_threshold(self):
        f = PerceptionFusion()
        # Exactly 3 = threshold for "frequent"
        for i in range(3):
            f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0 + i))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=103.0))
        signals = f.detect(now=103.0)
        patterns = [s.pattern for s in signals]
        assert "multitasking" in patterns

    def test_two_screen_events_not_frequent(self):
        f = PerceptionFusion()
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=100.0))
        f.ingest(FusionEvent(event_type="screen_changed", timestamp=102.0))
        f.ingest(FusionEvent(event_type="speech_detected", timestamp=103.0))
        signals = f.detect(now=103.0)
        patterns = [s.pattern for s in signals]
        # 2 < 3 threshold, so presenting, not multitasking
        assert "presenting" in patterns
        assert "multitasking" not in patterns


# ---------------------------------------------------------------------------
# format_fusion_context
# ---------------------------------------------------------------------------

class TestFormatFusionContext:
    def test_empty_returns_empty_string(self):
        assert format_fusion_context([]) == ""

    def test_contains_header(self):
        sig = FusionSignal(pattern="focused", confidence=0.6, evidence=("screen",), detected_at=100.0)
        result = format_fusion_context([sig])
        assert "[Activity State]" in result

    def test_contains_pattern(self):
        sig = FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen", "speech"), detected_at=100.0)
        result = format_fusion_context([sig])
        assert "presenting" in result

    def test_contains_confidence(self):
        sig = FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen", "speech"), detected_at=100.0)
        result = format_fusion_context([sig])
        assert "0.8" in result

    def test_contains_evidence(self):
        sig = FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen", "speech"), detected_at=100.0)
        result = format_fusion_context([sig])
        assert "screen" in result
        assert "speech" in result

    def test_empty_evidence_shows_inferred(self):
        sig = FusionSignal(pattern="idle", confidence=0.5, evidence=(), detected_at=100.0)
        result = format_fusion_context([sig])
        assert "inferred" in result

    def test_multiple_signals(self):
        sigs = [
            FusionSignal(pattern="focused", confidence=0.6, evidence=("screen",), detected_at=100.0),
            FusionSignal(pattern="presenting", confidence=0.8, evidence=("screen", "speech"), detected_at=100.0),
        ]
        result = format_fusion_context(sigs)
        assert "focused" in result
        assert "presenting" in result
