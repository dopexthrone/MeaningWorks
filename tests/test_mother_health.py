"""
Tests for Mother health probe — process and port monitoring.

Covers: HealthStatus/HealthEvent frozen dataclasses, HealthProbe state
transitions, PID checking, port checking, grace period, uptime, counters.
"""

import os
from unittest.mock import patch

import pytest

from mother.health import HealthStatus, HealthEvent, HealthProbe


class TestHealthStatusDefaults:

    def test_defaults(self):
        s = HealthStatus()
        assert s.alive is False
        assert s.port_responsive is False
        assert s.port == 0
        assert s.pid == 0
        assert s.uptime_seconds == 0.0
        assert s.check_count == 0
        assert s.consecutive_failures == 0
        assert s.last_check_time == 0.0

    def test_frozen(self):
        s = HealthStatus()
        with pytest.raises(AttributeError):
            s.alive = True


class TestHealthEventDefaults:

    def test_defaults(self):
        e = HealthEvent()
        assert e.event_type == ""
        assert e.severity == 0.0
        assert e.message == ""
        assert e.timestamp == 0.0

    def test_frozen(self):
        e = HealthEvent()
        with pytest.raises(AttributeError):
            e.event_type = "died"


class TestHealthProbeAlive:

    def test_check_alive_process(self):
        """Check own PID — should be alive."""
        probe = HealthProbe(pid=os.getpid(), start_time=1000.0)
        status, event = probe.check(now=1005.0)
        assert status.alive is True
        assert status.pid == os.getpid()

    def test_check_dead_process(self):
        """Check nonexistent PID — should be dead."""
        # PID 999999 is very unlikely to exist
        probe = HealthProbe(pid=999999, start_time=1000.0)
        status, event = probe.check(now=1005.0)
        assert status.alive is False
        assert event is not None
        assert event.event_type == "died"
        assert event.severity == 1.0


class TestHealthProbePort:

    @patch.object(HealthProbe, "is_port_open", return_value=True)
    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    def test_port_responsive(self, mock_pid, mock_port):
        probe = HealthProbe(pid=123, port=8080, start_time=1000.0, grace_period=0.0)
        status, event = probe.check(now=1005.0)
        assert status.alive is True
        assert status.port_responsive is True
        assert event is not None
        assert event.event_type == "port_up"

    @patch.object(HealthProbe, "is_port_open", return_value=False)
    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    def test_port_unresponsive_after_grace(self, mock_pid, mock_port):
        probe = HealthProbe(pid=123, port=8080, start_time=1000.0, grace_period=2.0)
        # After grace period
        status, event = probe.check(now=1003.0)
        assert status.alive is True
        assert status.port_responsive is False
        assert event is not None
        assert event.event_type == "port_down"

    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    def test_no_port_configured(self, mock_pid):
        probe = HealthProbe(pid=123, port=0, start_time=1000.0)
        status, event = probe.check(now=1005.0)
        assert status.alive is True
        assert status.port_responsive is False
        assert event is None  # No port = no port events


class TestHealthProbeTransitions:

    def test_alive_to_dead_emits_died(self):
        probe = HealthProbe(pid=123, start_time=1000.0)

        with patch.object(HealthProbe, "is_pid_alive", return_value=True):
            status1, event1 = probe.check(now=1001.0)
            assert status1.alive is True
            assert event1 is None  # No transition (was alive, still alive)

        with patch.object(HealthProbe, "is_pid_alive", return_value=False):
            status2, event2 = probe.check(now=1002.0)
            assert status2.alive is False
            assert event2 is not None
            assert event2.event_type == "died"

    def test_dead_stays_dead_no_event(self):
        probe = HealthProbe(pid=123, start_time=1000.0)

        with patch.object(HealthProbe, "is_pid_alive", return_value=False):
            # First check: alive→dead transition
            _, event1 = probe.check(now=1001.0)
            assert event1 is not None
            assert event1.event_type == "died"

            # Second check: still dead, no new event
            _, event2 = probe.check(now=1002.0)
            assert event2 is None

    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    def test_port_up_to_down(self, mock_pid):
        probe = HealthProbe(pid=123, port=8080, start_time=1000.0, grace_period=0.0)

        with patch.object(HealthProbe, "is_port_open", return_value=True):
            _, event1 = probe.check(now=1001.0)
            assert event1 is not None
            assert event1.event_type == "port_up"

        with patch.object(HealthProbe, "is_port_open", return_value=False):
            _, event2 = probe.check(now=1002.0)
            assert event2 is not None
            assert event2.event_type == "port_down"

    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    def test_port_down_to_up(self, mock_pid):
        probe = HealthProbe(pid=123, port=8080, start_time=1000.0, grace_period=0.0)

        with patch.object(HealthProbe, "is_port_open", return_value=False):
            _, event1 = probe.check(now=1001.0)
            # After grace, first check shows port_down
            assert event1 is not None
            assert event1.event_type == "port_down"

        with patch.object(HealthProbe, "is_port_open", return_value=True):
            _, event2 = probe.check(now=1002.0)
            assert event2 is not None
            assert event2.event_type == "port_up"


class TestHealthProbeMetrics:

    def test_uptime_computation(self):
        probe = HealthProbe(pid=os.getpid(), start_time=1000.0)
        status, _ = probe.check(now=1030.0)
        assert status.uptime_seconds == 30.0

    def test_check_count_increments(self):
        probe = HealthProbe(pid=os.getpid(), start_time=1000.0)
        probe.check(now=1001.0)
        probe.check(now=1002.0)
        probe.check(now=1003.0)
        status, _ = probe.check(now=1004.0)
        assert status.check_count == 4

    def test_consecutive_failures_count_and_reset(self):
        probe = HealthProbe(pid=123, start_time=1000.0)

        with patch.object(HealthProbe, "is_pid_alive", return_value=False):
            status1, _ = probe.check(now=1001.0)
            assert status1.consecutive_failures == 1
            status2, _ = probe.check(now=1002.0)
            assert status2.consecutive_failures == 2

        with patch.object(HealthProbe, "is_pid_alive", return_value=True):
            status3, _ = probe.check(now=1003.0)
            assert status3.consecutive_failures == 0


class TestHealthProbeStaticMethods:

    def test_is_pid_alive_self(self):
        assert HealthProbe.is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_nonexistent(self):
        assert HealthProbe.is_pid_alive(999999) is False

    def test_is_port_open_closed(self):
        # Port 1 should not be open on any normal system
        assert HealthProbe.is_port_open(1, timeout=0.1) is False


class TestHealthProbeGracePeriod:

    @patch.object(HealthProbe, "is_pid_alive", return_value=True)
    @patch.object(HealthProbe, "is_port_open", return_value=False)
    def test_grace_period_suppresses_early_port_failures(self, mock_port, mock_pid):
        probe = HealthProbe(pid=123, port=8080, start_time=1000.0, grace_period=10.0)
        # Check during grace period — port not up, but no event
        status, event = probe.check(now=1005.0)
        assert status.alive is True
        assert status.port_responsive is False
        assert event is None  # Suppressed by grace period
