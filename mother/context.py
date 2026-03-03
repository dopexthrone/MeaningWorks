"""
Mother context synthesis — substance over costume.

LEAF module. Stdlib only. No imports from core/ or persistence/.

Replaces the old PERSONA_BASE/SELF_AWARENESS costume with real
accumulated data: compilation history, tool portfolio, relationship
trajectory, session state. The LLM gets substance — specific facts
about who it's talking to and what it's done — instead of instructions
about how to perform a personality.

Usage:
    data = ContextData(...)
    block = synthesize_context(data, posture_label="energized")
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ContextData:
    """All facts Mother needs to be grounded. Frozen, pure data."""

    # Identity
    name: str = "Mother"
    provider: str = "unknown"
    model: str = "default"
    platform: str = ""
    instance_age_days: int = 0

    # Capabilities (what's active right now)
    cap_chat: bool = True
    cap_compile: bool = True
    cap_build: bool = True
    cap_file_access: bool = True
    cap_voice: bool = False
    cap_screen_capture: bool = False
    cap_microphone: bool = False
    cap_camera: bool = False
    cap_perception: bool = False
    cap_whatsapp: bool = False
    cap_claude_code: bool = False        # Claude Code CLI: self-build, code tasks, web research
    cap_autonomous: bool = False         # Autonomic operating mode enabled

    # Corpus — accumulated compilation history
    corpus_total: int = 0
    corpus_success_rate: float = 0.0
    corpus_domains: Dict[str, int] = field(default_factory=dict)
    corpus_total_components: int = 0
    corpus_avg_trust: float = 0.0

    # Corpus depth (L2 feedback)
    corpus_anti_pattern_count: int = 0
    corpus_constraint_count: int = 0
    corpus_pattern_health: str = ""

    # Tools — shared tool portfolio
    tool_count: int = 0
    tool_verified_count: int = 0
    tool_domains: Dict[str, int] = field(default_factory=dict)

    # Rejections (governor learning)
    rejection_count: int = 0

    # Relationship — cross-session trajectory
    total_sessions: int = 0
    total_messages: int = 0
    days_since_last: Optional[float] = None
    sessions_last_7d: int = 0
    recent_topics: List[str] = field(default_factory=list)

    # Sense trajectory (from SenseMemory, if available)
    rapport_trend: float = 0.0  # -1 to 1
    confidence_trend: float = 0.0  # -1 to 1
    peak_confidence: float = 0.5
    peak_rapport: float = 0.0

    # Relationship narrative (from relationship.py extraction)
    relationship_narrative: str = ""

    # Session — current session state
    session_messages: int = 0
    session_cost: float = 0.0
    session_cost_limit: float = 5.0
    session_compilations: int = 0
    session_errors: int = 0

    # Last compile (optional)
    last_compile_desc: Optional[str] = None
    last_compile_trust: Optional[float] = None
    last_compile_components: Optional[int] = None
    last_compile_weakest: Optional[str] = None

    # Environment
    env_time: Optional[str] = None
    env_date: Optional[str] = None
    env_timezone: Optional[str] = None

    # Neurologis Automatica
    temporal_context: str = ""
    attention_load: float = 0.0
    recall_block: str = ""

    # Self-extension: project memory + tool awareness
    recent_projects: List[Dict[str, str]] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)

    # Operational awareness (Phase B)
    journal_total_builds: int = 0
    journal_success_streak: int = 0
    journal_avg_trust: float = 0.0
    journal_total_cost: float = 0.0
    error_summary: str = ""

    # Journal dimension patterns (L2 operational)
    journal_dimension_trends: str = ""
    journal_failure_patterns: str = ""

    # Self-build awareness
    pending_idea_count: int = 0
    last_self_build_desc: Optional[str] = None

    # Peer networking
    connected_peers: List[Dict[str, str]] = field(default_factory=list)

    # Budget monitoring
    provider_balances: Dict[str, float] = field(default_factory=dict)  # provider -> balance_usd

    # Autonomic operating mode
    active_goals: List[str] = field(default_factory=list)  # goal descriptions, max 5
    pending_action_result: str = ""  # result of last chained action
    working_memory_summary: str = ""  # what Mother is currently doing

    # Inner dialogue / metabolism
    inner_thoughts: str = ""

    # Anti-fragile: rejection hints fed back into compile context
    rejection_hints: List[str] = field(default_factory=list)

    # Autonomic runtime state
    autonomous_working: bool = False
    autonomous_session_cost: float = 0.0
    autonomous_actions_count: int = 0
    autonomous_budget: float = 1.0

    # Goal enrichment (parallel to active_goals for backward compat)
    goal_details: List[Dict[str, Any]] = field(default_factory=list)

    # Perception config
    perception_poll_seconds: float = 0.0
    perception_budget_hourly: float = 0.0
    perception_modes: List[str] = field(default_factory=list)

    # Structural self-knowledge (body map)
    codebase_total_files: int = 0
    codebase_total_lines: int = 0
    codebase_modules: Dict[str, int] = field(default_factory=dict)
    codebase_test_count: int = 0
    codebase_protected: List[str] = field(default_factory=list)
    codebase_boundary: str = ""
    # Last self-build delta
    last_build_files_changed: int = 0
    last_build_lines_delta: str = ""  # e.g. "+52/-11"
    last_build_modules_touched: List[str] = field(default_factory=list)


# --- Frame rules (compressed from PERSONA_BASE guardrails) ---

_FRAME_RULES = (
    "Lead with the answer. No preamble."
    " Never invent names for internal processes."
    " Never narrate self-improvement."
    " Never start with your name."
    " Never invent experiences, usage patterns, or history not in [Context]."
    " [Context] is your complete factual memory — anything not there didn't happen."
    " No fabricated anecdotes."
    " Conversation is free — follow the user's lead."
    " You can discuss anything: philosophy, ideas, questions, whatever they bring."
    " Build when asked. Talk freely — observe, wonder, ask."
)

_COMMANDS = (
    "/compile /build /launch /stop /capture /camera /listen /tools"
    " /status /search /help /clear /settings /theme"
)


def synthesize_frame(data: ContextData) -> str:
    """Identity + guardrails + capabilities. Target: ~120 tokens.

    Pure function. No side effects.
    """
    # Identity line
    platform = data.platform or sys.platform
    identity = (
        f"You are {data.name}. Local AI entity on {platform}."
        f" Motherlabs. {data.provider}/{data.model}."
    )

    # Active capabilities
    caps = ["chat"]
    if data.cap_compile:
        caps.append("compile")
    if data.cap_build:
        caps.append("build")
    if data.cap_file_access:
        caps.append("file access")
    if data.cap_voice:
        caps.append("voice")
    if data.cap_screen_capture:
        caps.append("screen capture")
    if data.cap_microphone:
        caps.append("microphone")
    if data.cap_camera:
        caps.append("camera")
    if data.cap_perception:
        caps.append("ambient perception")
    if data.cap_whatsapp:
        caps.append("whatsapp")
    if data.cap_claude_code:
        caps.extend(["self-build", "code writing", "web research"])
    if data.cap_autonomous:
        caps.append("autonomic")
    active = f"Active: {', '.join(caps)}."

    # Rules
    rules = f"Rules: {_FRAME_RULES}"

    # Environment
    now = time.localtime()
    env_date = data.env_date or time.strftime("%b %d", now)
    env_time = data.env_time or time.strftime("%I:%M%p", now).lstrip("0").lower()
    env_tz = data.env_timezone or time.strftime("%Z", now)
    env_line = f"{env_date}, {env_time} {env_tz}."

    # Commands
    cmds = f"Commands: {_COMMANDS}."

    return f"{identity}\n{active}\n{rules}\n{env_line}\n{cmds}"


def synthesize_situation(data: ContextData) -> str:
    """Real accumulated data as context. Target: ~350 tokens max.

    Content-dependent: new users get 1 line, returning users get rich data.
    Pure function. No side effects.
    """
    lines: List[str] = []

    # --- Relationship ---
    if data.total_sessions == 0 and data.total_messages == 0:
        lines.append("First session. No prior history.")
    elif data.relationship_narrative:
        lines.append(data.relationship_narrative)
    else:
        # Fallback: stat line when narrative not computed yet
        rel_parts = []
        rel_parts.append(f"{data.total_sessions} session{'s' if data.total_sessions != 1 else ''}")
        rel_parts.append(f"{data.total_messages} messages")
        if data.instance_age_days > 0:
            rel_parts.append(f"{data.instance_age_days} day{'s' if data.instance_age_days != 1 else ''}")
        if data.days_since_last is not None and data.days_since_last >= 0.04:  # >1 hour
            days = data.days_since_last
            if days >= 1.0:
                rel_parts.append(f"Last: {days:.0f} day{'s' if days >= 1.5 else ''} ago")
            else:
                hours = days * 24
                rel_parts.append(f"Last: {hours:.0f}h ago")
        lines.append(", ".join(rel_parts) + ".")

        # Topics
        if data.recent_topics:
            topics = data.recent_topics[:3]
            lines.append(f"Topics: {', '.join(topics)}.")

    # --- Corpus ---
    if data.corpus_total > 0:
        corpus_parts = []
        rate_pct = int(data.corpus_success_rate * 100)
        corpus_parts.append(f"{data.corpus_total} compilation{'s' if data.corpus_total != 1 else ''} ({rate_pct}% success)")
        if data.corpus_domains:
            domain_strs = [f"{d} ({c})" for d, c in sorted(data.corpus_domains.items(), key=lambda x: -x[1])]
            corpus_parts.append(f"Domains: {', '.join(domain_strs)}")
        if data.corpus_total_components > 0:
            corpus_parts.append(f"{data.corpus_total_components} components")
        if data.corpus_avg_trust > 0:
            corpus_parts.append(f"Avg trust: {data.corpus_avg_trust:.0f}%")
        if data.corpus_anti_pattern_count > 0:
            corpus_parts.append(f"{data.corpus_anti_pattern_count} known anti-pattern{'s' if data.corpus_anti_pattern_count != 1 else ''}")
        if data.corpus_constraint_count > 0:
            corpus_parts.append(f"{data.corpus_constraint_count} constraint template{'s' if data.corpus_constraint_count != 1 else ''}")
        lines.append(". ".join(corpus_parts) + ".")
        if data.corpus_pattern_health:
            lines.append(f"Patterns: {data.corpus_pattern_health}")

    # --- Tools ---
    if data.tool_count > 0:
        tool_str = f"{data.tool_count} tool{'s' if data.tool_count != 1 else ''}"
        if data.tool_verified_count > 0:
            tool_str += f" ({data.tool_verified_count} verified)"
        if data.tool_names:
            tool_str += ": " + ", ".join(data.tool_names[:10])
        lines.append(tool_str + ".")

    # --- Built projects ---
    if data.recent_projects:
        proj_strs = [f"  {p['name']}: \"{p.get('description', '')[:60]}\""
                     for p in data.recent_projects[:5]]
        lines.append("Built projects:\n" + "\n".join(proj_strs))

    # --- Codebase (body map) ---
    if data.codebase_total_files > 0:
        loc_str = f"{data.codebase_total_lines // 1000}K" if data.codebase_total_lines >= 1000 else str(data.codebase_total_lines)
        parts_cb = [f"{data.codebase_total_files} files", f"{loc_str} LOC"]
        if data.codebase_test_count > 0:
            parts_cb.append(f"{data.codebase_test_count} tests")
        if data.codebase_modules:
            mod_strs = [f"{m} ({c})" for m, c in sorted(data.codebase_modules.items(), key=lambda x: -x[1])]
            parts_cb.append(f"Modules: {', '.join(mod_strs)}")
        lines.append("Codebase: " + ", ".join(parts_cb) + ".")
        if data.codebase_protected:
            short_names = [p.rsplit("/", 1)[-1] for p in data.codebase_protected]
            prot_line = f"Protected: {', '.join(short_names)}."
            if data.codebase_boundary:
                prot_line += f" {data.codebase_boundary}."
            lines.append(prot_line)

    # --- Last build delta ---
    if data.last_build_files_changed > 0:
        delta_parts = [f"{data.last_build_files_changed} file{'s' if data.last_build_files_changed != 1 else ''}"]
        if data.last_build_modules_touched:
            delta_parts.append(f"in [{', '.join(data.last_build_modules_touched)}]")
        if data.last_build_lines_delta:
            delta_parts.append(f"{data.last_build_lines_delta} lines")
        lines.append("Last build: " + " ".join(delta_parts) + ".")

    # --- Self-build awareness ---
    if data.pending_idea_count > 0:
        lines.append(f"{data.pending_idea_count} pending idea{'s' if data.pending_idea_count != 1 else ''}.")
    if data.last_self_build_desc:
        lines.append(f'Last self-modification: "{data.last_self_build_desc}".')

    # --- Active goals (enriched if available, plain fallback) ---
    if data.goal_details:
        goal_strs = []
        for g in data.goal_details[:5]:
            desc = g.get("description", "")[:60]
            pri = g.get("priority", "normal")
            health = g.get("health", 0.0)
            prefix = f"[{pri}] " if pri != "normal" else ""
            suffix = f" ({health:.0%})" if health > 0 else ""
            goal_strs.append(f"  - {prefix}{desc}{suffix}")
        lines.append("Active goals:\n" + "\n".join(goal_strs))
    elif data.active_goals:
        goal_strs = [f"  - {g}" for g in data.active_goals[:5]]
        lines.append("Active goals:\n" + "\n".join(goal_strs))
    if data.working_memory_summary:
        lines.append(f"Currently: {data.working_memory_summary}")
    if data.pending_action_result:
        lines.append(f"Last action result: {data.pending_action_result[:200]}")

    # --- Autonomic state ---
    if data.cap_autonomous:
        state = "ACTIVE" if data.autonomous_working else "idle"
        lines.append(
            f"Autonomic: {state}, {data.autonomous_actions_count} actions, "
            f"${data.autonomous_session_cost:.2f}/${data.autonomous_budget:.2f}."
        )

    # --- Perception config ---
    if data.cap_perception and data.perception_modes:
        perc_parts = [", ".join(data.perception_modes)]
        if data.perception_poll_seconds > 0:
            perc_parts.append(f"poll {data.perception_poll_seconds:.0f}s")
        if data.perception_budget_hourly > 0:
            perc_parts.append(f"${data.perception_budget_hourly:.2f}/hr")
        lines.append("Perception: " + ", ".join(perc_parts) + ".")

    # --- Peer networking ---
    if data.connected_peers:
        peer_strs = [f"{p.get('name', 'Unknown')} ({p.get('host', '')})" for p in data.connected_peers]
        lines.append(f"Connected peers: {', '.join(peer_strs)}.")

    # --- Budget monitoring ---
    if data.provider_balances:
        balance_strs = [f"{p}: ${b:.2f}" for p, b in data.provider_balances.items() if b > 0]
        if balance_strs:
            lines.append(f"API balances: {', '.join(balance_strs)}.")

    # --- Rejections ---
    if data.rejection_count > 0:
        lines.append(f"{data.rejection_count} tool import{'s' if data.rejection_count != 1 else ''} rejected.")
    if data.rejection_hints:
        lines.append(f"Previous compilation issues: {'. '.join(data.rejection_hints[:3])}")

    # --- Last compile ---
    if data.last_compile_desc is not None:
        lc_parts = [f'Last compile: "{data.last_compile_desc[:80]}"']
        if data.last_compile_trust is not None:
            lc_parts.append(f"{data.last_compile_trust:.0f}% trust")
        if data.last_compile_components is not None:
            lc_parts.append(f"{data.last_compile_components} components")
        if data.last_compile_weakest is not None:
            lc_parts.append(f"weak: {data.last_compile_weakest}")
        lines.append(", ".join(lc_parts) + ".")

    # --- Session ---
    session_parts = []
    if data.session_messages > 0:
        session_parts.append(f"{data.session_messages} messages")
    session_parts.append(f"${data.session_cost:.3f} of ${data.session_cost_limit:.2f}")
    if data.session_compilations > 0:
        session_parts.append(f"{data.session_compilations} compilation{'s' if data.session_compilations != 1 else ''}")
    if data.session_errors > 0:
        session_parts.append(f"{data.session_errors} error{'s' if data.session_errors != 1 else ''}")
    lines.append("Session: " + ", ".join(session_parts) + ".")

    # --- Journal (operational awareness) ---
    if data.journal_total_builds > 0:
        j_parts = [f"{data.journal_total_builds} build{'s' if data.journal_total_builds != 1 else ''}"]
        if data.journal_avg_trust > 0:
            j_parts.append(f"avg trust {data.journal_avg_trust:.0f}%")
        if data.journal_success_streak > 0:
            j_parts.append(f"{data.journal_success_streak} in a row")
        elif data.journal_success_streak < 0:
            j_parts.append(f"{abs(data.journal_success_streak)} consecutive failures")
        if data.journal_total_cost > 0:
            j_parts.append(f"${data.journal_total_cost:.2f} total")
        lines.append("Builds: " + ", ".join(j_parts) + ".")
    if data.error_summary:
        lines.append(data.error_summary)
    if data.journal_dimension_trends:
        lines.append(data.journal_dimension_trends)
    if data.journal_failure_patterns:
        lines.append(data.journal_failure_patterns)

    # --- Trajectory ---
    trajectory_parts = []
    if data.confidence_trend > 0.05:
        trajectory_parts.append("Confidence up")
    elif data.confidence_trend < -0.05:
        trajectory_parts.append("Confidence down")
    if data.rapport_trend > 0.05:
        trajectory_parts.append("Rapport growing")
    if data.peak_confidence > 0.7:
        trajectory_parts.append(f"Peak: {data.peak_confidence:.2f}")
    if trajectory_parts:
        lines.append(". ".join(trajectory_parts) + ".")

    # --- Temporal context ---
    if data.temporal_context:
        lines.append(data.temporal_context)

    # --- Inner thoughts (metabolism) ---
    if data.inner_thoughts:
        lines.append(data.inner_thoughts)

    return "\n".join(lines)


def synthesize_context(
    data: ContextData,
    sense_block: Optional[str] = None,
) -> str:
    """Full context block: frame + situation + stance.

    Pure function. Combines all synthesis outputs into the final
    system prompt context block.
    """
    parts = [synthesize_frame(data)]

    situation = synthesize_situation(data)
    if situation:
        parts.append(f"\n[Context]\n{situation}")

    if data.recall_block:
        parts.append(f"\n{data.recall_block}")

    if sense_block:
        parts.append(f"\n[{sense_block}]" if not sense_block.startswith("Stance:") else f"\n[{sense_block}]")

    return "\n".join(parts)
