"""
Mother temporal awareness — felt duration and time-of-day.

LEAF module. Stdlib only. No imports from core/ or mother/.

Provides TemporalState (frozen) and TemporalEngine that ticks
each conversation turn to produce session age, idle duration,
conversation tempo, and time-of-day classification.
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TemporalState:
    """Snapshot of temporal context. All fields derived, no side effects."""

    session_age_seconds: float = 0.0
    idle_seconds: float = 0.0          # since last user message
    wall_clock_hour: int = -1          # 0-23
    conversation_tempo: float = 0.0    # msgs/min, rolling
    is_typical_time: bool = False      # matches relationship preferred_time
    time_of_day: str = ""              # morning/afternoon/evening/night
    flow_state: str = ""               # deep/shallow/idle
    session_pattern: str = ""           # weekday/weekend


def classify_flow(tempo: float, idle: float, session_age: float) -> str:
    """Classify user engagement into flow states. Pure function.

    Args:
        tempo: messages per minute (rolling)
        idle: seconds since last user message
        session_age: total session duration in seconds
    """
    if tempo >= 2.0 and idle < 60 and session_age > 300:
        return "deep"
    if tempo >= 0.5 and idle < 120:
        return "shallow"
    return "idle"


def _classify_time_of_day(hour: int) -> str:
    """Classify hour (0-23) into time-of-day label."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def classify_session_pattern(now: float = 0.0) -> str:
    """Classify weekday vs weekend. Pure function. Injectable now."""
    _now = now if now > 0 else time.time()
    wday = time.localtime(_now).tm_wday  # 0=Mon, 6=Sun
    return "weekend" if wday >= 5 else "weekday"


def is_stale(timestamp: float, max_age_hours: float = 24.0, now: float = 0.0) -> bool:
    """Check if a timestamp is older than max_age_hours. Pure function."""
    if timestamp <= 0:
        return True
    _now = now if now > 0 else time.time()
    return (_now - timestamp) / 3600.0 > max_age_hours


class TemporalEngine:
    """Stateless temporal state computer. Pure function wrapped in a class."""

    def tick(
        self,
        last_user_message_time: float,
        messages_this_session: int,
        session_start_time: float,
        preferred_time: str = "",
        now: Optional[float] = None,
    ) -> TemporalState:
        """Compute temporal state from session facts.

        Args:
            last_user_message_time: epoch of last user message (0 if never)
            messages_this_session: total messages in this session
            session_start_time: epoch when session started
            preferred_time: user's preferred time-of-day from relationship
            now: injectable epoch for testing (defaults to time.time())
        """
        now = now if now is not None else time.time()

        session_age = max(0.0, now - session_start_time)
        idle = max(0.0, now - last_user_message_time) if last_user_message_time > 0 else 0.0

        # Tempo: msgs/min, clamped when session is very young
        age_minutes = session_age / 60.0
        if age_minutes < 0.5:
            tempo = float(messages_this_session) * 2.0  # extrapolate from <30s
        else:
            tempo = messages_this_session / age_minutes

        hour = time.localtime(now).tm_hour
        tod = _classify_time_of_day(hour)
        is_typical = preferred_time != "" and tod == preferred_time

        flow = classify_flow(tempo, idle, session_age)
        pattern = classify_session_pattern(now)

        return TemporalState(
            session_age_seconds=round(session_age, 2),
            idle_seconds=round(idle, 2),
            wall_clock_hour=hour,
            conversation_tempo=round(tempo, 3),
            is_typical_time=is_typical,
            time_of_day=tod,
            flow_state=flow,
            session_pattern=pattern,
        )
