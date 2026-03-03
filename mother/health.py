"""
Mother health probe — process and port monitoring for launched projects.

LEAF module. Stdlib only (os, socket, time). No imports from core/ or mother/.

Provides HealthProbe that checks if a launched process is alive and
its port is responsive. Emits HealthEvent on state transitions only.
"""

import os
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class HealthStatus:
    """Current health state of a monitored process."""

    alive: bool = False
    port_responsive: bool = False
    port: int = 0
    pid: int = 0
    uptime_seconds: float = 0.0
    check_count: int = 0
    consecutive_failures: int = 0
    last_check_time: float = 0.0


@dataclass(frozen=True)
class HealthEvent:
    """State transition event emitted by health checks."""

    event_type: str = ""          # "died" | "port_down" | "port_up" | "recovered"
    severity: float = 0.0         # 0.0-1.0
    message: str = ""
    timestamp: float = 0.0


# Event severities
_SEVERITIES = {
    "died": 1.0,
    "port_down": 0.7,
    "port_up": 0.3,
    "recovered": 0.6,
}


class HealthProbe:
    """Monitor a launched process by PID and optional port.

    check() returns (status, event_or_None). Events fire only on
    state transitions — not on every check.
    """

    def __init__(
        self,
        pid: int,
        port: int = 0,
        start_time: float = 0.0,
        grace_period: float = 5.0,
    ):
        self._pid = pid
        self._port = port
        self._start_time = start_time or time.time()
        self._grace_period = grace_period

        # State tracking
        self._check_count: int = 0
        self._consecutive_failures: int = 0
        self._last_alive: bool = True     # Assume alive at start
        self._last_port_up: bool = False  # Unknown at start
        self._port_ever_checked: bool = False
        self._last_check_time: float = 0.0

    def check(self, now: Optional[float] = None) -> Tuple[HealthStatus, Optional[HealthEvent]]:
        """Check process health. Returns (status, event_or_None).

        Events only fire on state transitions:
        - alive → dead: "died" event
        - port_up → port_down: "port_down" event
        - port_down → port_up: "port_up" event
        - dead → alive: "recovered" event (unlikely but handled)
        """
        now = now if now is not None else time.time()
        self._check_count += 1
        self._last_check_time = now

        alive = self.is_pid_alive(self._pid)
        uptime = now - self._start_time

        # Port check
        port_responsive = False
        if self._port > 0 and alive:
            port_responsive = self.is_port_open(self._port)

        # Build status
        if not alive:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        status = HealthStatus(
            alive=alive,
            port_responsive=port_responsive,
            port=self._port,
            pid=self._pid,
            uptime_seconds=round(uptime, 2),
            check_count=self._check_count,
            consecutive_failures=self._consecutive_failures,
            last_check_time=now,
        )

        # Determine event (state transitions only)
        event = None

        if self._last_alive and not alive:
            # Process died
            event = HealthEvent(
                event_type="died",
                severity=_SEVERITIES["died"],
                message=f"Process {self._pid} is no longer running.",
                timestamp=now,
            )
        elif not self._last_alive and alive:
            # Process recovered (unexpected but handled)
            event = HealthEvent(
                event_type="recovered",
                severity=_SEVERITIES["recovered"],
                message=f"Process {self._pid} recovered.",
                timestamp=now,
            )
        elif alive and self._port > 0:
            # Grace period: don't report port failures during startup
            in_grace = (uptime < self._grace_period)

            if self._port_ever_checked and self._last_port_up and not port_responsive:
                # Port went down
                event = HealthEvent(
                    event_type="port_down",
                    severity=_SEVERITIES["port_down"],
                    message=f"Port {self._port} stopped responding.",
                    timestamp=now,
                )
            elif self._port_ever_checked and not self._last_port_up and port_responsive:
                # Port came up
                event = HealthEvent(
                    event_type="port_up",
                    severity=_SEVERITIES["port_up"],
                    message=f"Port {self._port} is now responsive.",
                    timestamp=now,
                )
            elif not self._port_ever_checked and port_responsive:
                # First successful port check
                event = HealthEvent(
                    event_type="port_up",
                    severity=_SEVERITIES["port_up"],
                    message=f"Port {self._port} is now responsive.",
                    timestamp=now,
                )
            elif not self._port_ever_checked and not port_responsive and not in_grace:
                # First check after grace period — port still not up
                event = HealthEvent(
                    event_type="port_down",
                    severity=_SEVERITIES["port_down"],
                    message=f"Port {self._port} not responding after grace period.",
                    timestamp=now,
                )

            if port_responsive or (not in_grace and not self._port_ever_checked):
                self._port_ever_checked = True

        # Update state for next check
        self._last_alive = alive
        if self._port > 0 and alive:
            self._last_port_up = port_responsive

        return status, event

    @staticmethod
    def is_pid_alive(pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    @staticmethod
    def is_port_open(port: int, timeout: float = 1.0) -> bool:
        """Check if a TCP port is accepting connections on localhost."""
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False
