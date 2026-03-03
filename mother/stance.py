"""
Mother stance — when to act vs stay quiet.

LEAF module. Stdlib only. No imports from core/ or mother/.

Stance determines Mother's initiative level for GOAL EXECUTION.
SILENT is the default — Mother must earn the right to act through
goal health, user idle time, and posture state.

For DIALOGUE INITIATIVE (curiosity, observations, reflections),
see impulse.py. Stance and Impulse are independent axes:
- Stance gates goal work (compile, build, execute plan steps)
- Impulse gates conversation (ask questions, share observations)

The autonomous tick checks both. If Stance says SILENT but Impulse
says SPEAK, Mother talks without executing goals.
"""

from dataclasses import dataclass
from enum import Enum


class Stance(Enum):
    ACT = "act"        # Execute autonomously
    WAIT = "wait"      # Conditions approaching but not yet
    ASK = "ask"        # Propose action, don't execute
    SILENT = "silent"  # Do nothing (default)
    REFUSE = "refuse"  # Deny low-value action


@dataclass(frozen=True)
class StanceContext:
    has_active_goals: bool = False
    highest_goal_health: float = 0.0
    user_idle_seconds: float = 0.0
    conversation_active: bool = False
    posture_state: str = "steady"       # from Posture.state_label
    session_messages: int = 0
    autonomous_actions_this_session: int = 0
    flow_state: str = ""                # from TemporalState.flow_state
    frustration: float = 0.0            # from SenseVector.frustration
    goal_source: str = ""               # "user" | "mother" | "system"
    domain_trust: float = 0.5           # 0.0-1.0 domain permission level
    is_typical_time: bool = False       # from TemporalState
    time_of_day: str = ""              # morning/afternoon/evening/night
    session_pattern: str = ""          # weekday/weekend


def compute_stance(ctx: StanceContext) -> Stance:
    """Pure, deterministic stance computation for goal execution. SILENT is default."""
    if ctx.conversation_active:
        return Stance.SILENT

    if ctx.flow_state == "deep":
        return Stance.SILENT  # user in deep flow — protect momentum

    if not ctx.has_active_goals:
        return Stance.SILENT

    if ctx.highest_goal_health < 0.2:
        return Stance.SILENT          # stale goals — don't bother

    # Frustration + stale goals → refuse (don't add noise)
    if ctx.frustration >= 0.6 and ctx.highest_goal_health < 0.3:
        return Stance.REFUSE

    # Dynamic budget based on session state
    budget = 5  # base
    if ctx.frustration >= 0.4:
        budget = max(2, budget - 2)   # frustrated → fewer actions
    if ctx.posture_state == "energized":
        budget += 2                   # energized → more room
    if ctx.time_of_day == "night" and not ctx.is_typical_time:
        budget = max(1, budget - 2)   # off-hours: reduce initiative
    if ctx.session_pattern == "weekend" and not ctx.is_typical_time:
        budget = max(1, budget - 1)   # weekend off-pattern: reduce initiative

    if ctx.autonomous_actions_this_session >= budget:
        return Stance.SILENT          # enough initiative for one session

    # Minimum idle before any action — domain trust adjusts
    min_idle = 60.0
    if ctx.domain_trust >= 0.7:
        min_idle = 40.0               # trusted domain — respond sooner
    elif ctx.domain_trust < 0.3:
        min_idle = 90.0               # untrusted domain — wait longer
    if ctx.user_idle_seconds < min_idle:
        return Stance.SILENT          # user recently active

    # Mother-sourced goals can auto-execute within budget
    if ctx.goal_source == "mother" and ctx.autonomous_actions_this_session < budget:
        if ctx.highest_goal_health >= 0.3 and ctx.user_idle_seconds >= 60:
            return Stance.ACT

    # Graduated response — domain trust adjusts idle thresholds
    # High domain_trust (>0.7): user trusts Mother here → shorter wait
    # Low domain_trust (<0.3): unfamiliar domain → longer wait, always ASK
    act_idle = 120.0
    ask_idle = 60.0
    if ctx.domain_trust >= 0.7:
        act_idle = 90.0    # trusted domain — act sooner
        ask_idle = 45.0
    elif ctx.domain_trust < 0.3:
        act_idle = 240.0   # untrusted domain — wait longer, prefer ASK
        ask_idle = 120.0

    if ctx.highest_goal_health >= 0.5 and ctx.user_idle_seconds >= act_idle:
        # Low domain trust: always propose first, never auto-execute
        if ctx.domain_trust < 0.3:
            return Stance.ASK
        return Stance.ACT             # healthy goal + idle — go

    if ctx.highest_goal_health >= 0.3 and ctx.user_idle_seconds >= ask_idle:
        return Stance.ASK             # medium goal + idle — propose

    return Stance.WAIT                # not yet


def explain_stance_tradeoff(ctx: StanceContext, stance: Stance) -> str:
    """Produce a brief explanation of why this stance was chosen. Pure function."""
    if stance == Stance.SILENT:
        return ""
    parts = []
    if stance == Stance.ACT:
        parts.append(f"Acting — goal health {ctx.highest_goal_health:.0%}, idle {ctx.user_idle_seconds:.0f}s")
        if ctx.domain_trust >= 0.7:
            parts.append("strong domain trust")
    elif stance == Stance.ASK:
        parts.append("Proposing rather than acting")
        if ctx.domain_trust < 0.3:
            parts.append("unfamiliar domain")
        elif ctx.highest_goal_health < 0.5:
            parts.append(f"moderate goal health ({ctx.highest_goal_health:.0%})")
    elif stance == Stance.WAIT:
        parts.append(f"Waiting — not ready (health {ctx.highest_goal_health:.0%}, idle {ctx.user_idle_seconds:.0f}s)")
    elif stance == Stance.REFUSE:
        parts.append(f"Refusing — frustration {ctx.frustration:.0%}, goal health {ctx.highest_goal_health:.0%}")
    return ". ".join(parts) + "." if parts else ""
