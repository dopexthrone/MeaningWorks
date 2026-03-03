"""Tests for self-monitoring — degradation detection triggers throttle."""

import pytest

from mother.senses import SenseVector, compute_posture


class TestDegradationDetection:

    def test_low_vitality_high_frustration_is_degraded(self):
        """Vitality < 0.2 + frustration >= 0.5 = degradation threshold."""
        v = SenseVector(vitality=0.15, frustration=0.6, confidence=0.3)
        # This state should trigger throttling in chat.py
        assert v.vitality < 0.2
        assert v.frustration >= 0.5

    def test_healthy_state_not_degraded(self):
        """Normal vitality and low frustration = no degradation."""
        v = SenseVector(vitality=0.8, frustration=0.1)
        assert not (v.vitality < 0.2 and v.frustration >= 0.5)

    def test_low_vitality_alone_not_degraded(self):
        """Low vitality without frustration doesn't trigger degradation."""
        v = SenseVector(vitality=0.1, frustration=0.2)
        assert not (v.vitality < 0.2 and v.frustration >= 0.5)

    def test_high_frustration_alone_not_degraded(self):
        """High frustration without low vitality doesn't trigger degradation."""
        v = SenseVector(vitality=0.5, frustration=0.8)
        assert not (v.vitality < 0.2 and v.frustration >= 0.5)
