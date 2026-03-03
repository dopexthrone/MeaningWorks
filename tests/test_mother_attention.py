"""
Tests for mother/attention.py — selective attention filter.
"""

import time

import pytest

from mother.attention import AttentionScore, AttentionState, AttentionFilter


class TestAttentionScore:

    def test_defaults(self):
        s = AttentionScore()
        assert s.should_attend is False
        assert s.significance == 0.0
        assert s.rationale == ""

    def test_frozen(self):
        s = AttentionScore()
        with pytest.raises(AttributeError):
            s.should_attend = True


class TestAttentionState:

    def test_defaults(self):
        s = AttentionState()
        assert s.events_seen == 0
        assert s.events_attended == 0
        assert s.load == 0.0
        assert s.last_attended_time == 0.0

    def test_frozen(self):
        s = AttentionState()
        with pytest.raises(AttributeError):
            s.events_seen = 5


class TestAttentionFilter:

    def test_speech_idle_high_significance(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="speech",
            payload_size=100,
            elapsed_since_last_event=60.0,
            senses_attentiveness=0.5,
            conversation_active=False,
            now=1000.0,
        )
        assert score.should_attend is True
        assert score.significance == 0.9

    def test_speech_active_lower_significance(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="speech",
            payload_size=100,
            elapsed_since_last_event=5.0,
            senses_attentiveness=0.5,
            conversation_active=True,
            now=1000.0,
        )
        assert score.should_attend is True
        assert score.significance == 0.3

    def test_screen_silence_medium(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="screen",
            payload_size=50000,
            elapsed_since_last_event=45.0,
            senses_attentiveness=0.5,
            conversation_active=False,
            now=1000.0,
        )
        assert score.should_attend is True
        assert score.significance == 0.5

    def test_screen_active_low(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="screen",
            payload_size=50000,
            elapsed_since_last_event=5.0,
            senses_attentiveness=0.5,
            conversation_active=True,
            now=1000.0,
        )
        assert score.should_attend is False
        assert score.significance == 0.2

    def test_camera_default_low(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="camera",
            payload_size=10000,
            elapsed_since_last_event=30.0,
            senses_attentiveness=0.3,
            conversation_active=False,
            now=1000.0,
        )
        assert score.should_attend is False
        assert score.significance == 0.1

    def test_camera_high_attentiveness_higher(self):
        f = AttentionFilter()
        score = f.evaluate(
            event_type="camera",
            payload_size=10000,
            elapsed_since_last_event=30.0,
            senses_attentiveness=0.8,
            conversation_active=False,
            now=1000.0,
        )
        assert score.should_attend is True
        assert score.significance == 0.4

    def test_rate_limiting_halves_significance(self):
        f = AttentionFilter(capacity_per_minute=3)
        now = 1000.0
        # Fill capacity with speech events
        for i in range(3):
            f.evaluate(
                event_type="speech",
                payload_size=100,
                elapsed_since_last_event=60.0,
                senses_attentiveness=0.5,
                conversation_active=False,
                now=now + i,
            )
        # 4th event should be rate-limited
        score = f.evaluate(
            event_type="speech",
            payload_size=100,
            elapsed_since_last_event=60.0,
            senses_attentiveness=0.5,
            conversation_active=False,
            now=now + 3,
        )
        assert score.significance == 0.45  # 0.9 * 0.5

    def test_should_attend_threshold(self):
        f = AttentionFilter()
        # Camera with low attentiveness → 0.1 < 0.25 → should_attend=False
        score = f.evaluate(
            event_type="camera",
            payload_size=10000,
            elapsed_since_last_event=10.0,
            senses_attentiveness=0.3,
            conversation_active=False,
            now=1000.0,
        )
        assert score.should_attend is False

    def test_state_tracks_events(self):
        f = AttentionFilter()
        f.evaluate("speech", 100, 60.0, 0.5, False, now=1000.0)
        f.evaluate("camera", 100, 10.0, 0.3, False, now=1001.0)
        st = f.state
        assert st.events_seen == 2
        assert st.events_attended == 1  # only speech attended

    def test_state_load_computation(self):
        f = AttentionFilter(capacity_per_minute=2)
        now = time.time()
        f.evaluate("speech", 100, 60.0, 0.5, False, now=now)
        f.evaluate("speech", 100, 60.0, 0.5, False, now=now + 1)
        st = f.state
        assert st.load == 1.0  # 2 attended / 2 capacity

    def test_timestamp_pruning(self):
        f = AttentionFilter(capacity_per_minute=5)
        # Add events at t=0
        for i in range(5):
            f.evaluate("speech", 100, 60.0, 0.5, False, now=1000.0 + i)
        # At t=200 (>120s later), old timestamps should be pruned
        score = f.evaluate("speech", 100, 60.0, 0.5, False, now=1200.0)
        # No rate limiting because old timestamps pruned
        assert score.significance == 0.9

    def test_unknown_event_type(self):
        f = AttentionFilter()
        score = f.evaluate("unknown", 100, 60.0, 0.5, False, now=1000.0)
        assert score.significance == 0.1
        assert score.should_attend is False

    def test_screen_silence_not_active(self):
        """Screen change + silence >30s + NOT conversation_active → 0.5"""
        f = AttentionFilter()
        score = f.evaluate("screen", 50000, 45.0, 0.5, False, now=1000.0)
        assert score.significance == 0.5
        assert score.should_attend is True

    def test_screen_silence_but_conversation_active(self):
        """Screen change + silence >30s but conversation IS active → 0.2"""
        f = AttentionFilter()
        score = f.evaluate("screen", 50000, 45.0, 0.5, True, now=1000.0)
        assert score.significance == 0.2
        assert score.should_attend is False

    def test_health_event_significance(self):
        """Health events should have high significance (0.8)."""
        f = AttentionFilter()
        score = f.evaluate("health", 0, 60.0, 0.5, False, now=1000.0)
        assert score.significance == 0.8
        assert score.should_attend is True
        assert score.rationale == "process health event"

    def test_health_event_rate_limited(self):
        """Health events should respect rate limiting."""
        f = AttentionFilter(capacity_per_minute=2)
        now = 1000.0
        # Fill capacity
        f.evaluate("speech", 100, 60.0, 0.5, False, now=now)
        f.evaluate("speech", 100, 60.0, 0.5, False, now=now + 1)
        # Health event should be rate-limited: 0.8 * 0.5 = 0.4
        score = f.evaluate("health", 0, 60.0, 0.5, False, now=now + 2)
        assert score.significance == 0.4
        assert score.should_attend is True  # 0.4 >= 0.25
