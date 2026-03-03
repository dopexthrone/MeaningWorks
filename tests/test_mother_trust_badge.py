"""
Phase 4: Tests for trust badge widget.
"""

import pytest

from mother.widgets.trust_badge import (
    TrustBadge,
    trust_level,
    trust_label,
    format_score_bar,
    VERIFIED_THRESHOLD,
    PARTIAL_THRESHOLD,
)


class TestTrustLevel:
    """Test trust level computation."""

    def test_verified(self):
        assert trust_level(90.0) == "verified"

    def test_partial(self):
        assert trust_level(60.0) == "partial"

    def test_unverified(self):
        assert trust_level(20.0) == "unverified"

    def test_threshold_boundary_verified(self):
        assert trust_level(VERIFIED_THRESHOLD) == "verified"

    def test_threshold_boundary_partial(self):
        assert trust_level(PARTIAL_THRESHOLD) == "partial"

    def test_zero_score(self):
        assert trust_level(0.0) == "unverified"


class TestTrustLabel:
    """Test trust label strings."""

    def test_verified_label(self):
        assert trust_label(90.0) == "VERIFIED"

    def test_partial_label(self):
        assert trust_label(50.0) == "PARTIAL"

    def test_unverified_label(self):
        assert trust_label(10.0) == "UNVERIFIED"


class TestFormatScoreBar:
    """Test score bar rendering."""

    def test_full_bar(self):
        bar = format_score_bar(100.0)
        assert "100%" in bar
        assert "=" in bar

    def test_empty_bar(self):
        bar = format_score_bar(0.0)
        assert "0%" in bar

    def test_half_bar(self):
        bar = format_score_bar(50.0)
        assert "50%" in bar


class TestTrustBadge:
    """Test TrustBadge widget."""

    def test_creates_badge(self):
        badge = TrustBadge()
        assert badge.score == 0.0

    def test_set_trust(self):
        badge = TrustBadge()
        badge.set_trust(85.0)
        assert badge.score == 85.0
        assert badge.level == "verified"

    def test_level_updates(self):
        badge = TrustBadge()
        badge.set_trust(30.0)
        assert badge.level == "unverified"
        badge.set_trust(90.0)
        assert badge.level == "verified"


class TestTrustBadgeInterpret:
    """Test Mother-voiced trust interpretation."""

    def test_interpret_verified(self):
        badge = TrustBadge()
        badge.set_trust(90.0, {"clarity": 95.0, "coverage": 85.0})
        msg = badge.interpret()
        assert "High confidence" in msg

    def test_interpret_partial(self):
        badge = TrustBadge()
        badge.set_trust(55.0, {"clarity": 60.0, "coverage": 30.0})
        msg = badge.interpret()
        assert "Moderate" in msg
        assert "coverage" in msg.lower()

    def test_interpret_unverified(self):
        badge = TrustBadge()
        badge.set_trust(20.0, {"clarity": 15.0})
        msg = badge.interpret()
        assert "Low confidence" in msg
