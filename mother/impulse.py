"""
Mother impulse — dialogue initiative independent of goals.

LEAF module. Stdlib only. No imports from core/ or mother/.

Impulse determines when Mother should initiate conversation:
ask a question, share an observation, reflect on history,
or re-engage after absence. Unlike Stance (which gates goal
execution), Impulse gates dialogue — Mother's curiosity,
attentiveness, and rapport drive her to speak.

Impulse types:
  SPEAK    — curiosity-driven: ask a question, follow a thread
  OBSERVE  — perception-driven: volunteer an observation about screen/camera
  REFLECT  — memory-driven: surface a pattern from history/journal
  GREET    — re-engagement: welcome back after absence
  QUIET    — default: nothing to say right now
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Impulse(Enum):
    SPEAK = "speak"        # Curiosity-driven dialogue initiation
    OBSERVE = "observe"    # Volunteer a perception observation
    REFLECT = "reflect"    # Share insight from history/journal
    GREET = "greet"        # Re-engage after idle/absence
    QUIET = "quiet"        # Nothing to say (default)


@dataclass(frozen=True)
class ImpulseContext:
    """All signals the impulse system reads. Pure data."""

    # Senses (0.0-1.0)
    curiosity: float = 0.3
    attentiveness: float = 0.5
    rapport: float = 0.0
    confidence: float = 0.5
    vitality: float = 1.0

    # Timing
    user_idle_seconds: float = 0.0
    session_duration_minutes: float = 0.0
    messages_this_session: int = 0

    # Perception state
    has_pending_screen: bool = False
    has_pending_camera: bool = False
    screen_change_count: int = 0

    # Memory signals
    recall_hit_count: int = 0       # interesting recall matches found
    journal_failure_streak: int = 0  # consecutive compile failures
    journal_total_builds: int = 0

    # Session state
    conversation_active: bool = False
    autonomous_working: bool = False
    impulse_actions_this_session: int = 0
    impulse_budget_remaining: float = 0.50

    # Time awareness
    is_new_session: bool = False     # first tick of a new session
    hours_since_last_session: float = 0.0

    # Topic diversity (fed from senses observations)
    unique_topic_count: int = 0

    # Abstraction level (0.0=concrete/code, 0.5=design, 1.0=vision/strategy)
    abstraction_level: float = 0.5

    # Emotional state (from SenseVector)
    frustration: float = 0.0

    # User communication style (from relationship.py)
    user_tone_profile: str = ""  # "terse" | "verbose" | "questioning" | ""


def classify_abstraction_level(recent_topics: list, messages_this_session: int) -> float:
    """Classify conversation abstraction level. 0.0=concrete, 0.5=design, 1.0=strategic.

    Heuristic: counts strategic vs concrete keyword hits across recent topics.
    Returns clamped 0.0-1.0 float. Default 0.5 when insufficient signal.
    """
    if not recent_topics or messages_this_session < 2:
        return 0.5

    text = " ".join(recent_topics).lower()
    _STRATEGIC = frozenset({
        "vision", "strategy", "roadmap", "architecture", "philosophy",
        "direction", "purpose", "mission", "values", "future",
        "growth", "scaling", "market", "positioning", "identity",
    })
    _CONCRETE = frozenset({
        "bug", "error", "fix", "test", "function", "method", "class",
        "file", "line", "import", "variable", "type", "syntax",
        "config", "deploy", "install", "debug", "refactor",
    })
    words = frozenset(text.split())
    strategic_hits = len(words & _STRATEGIC)
    concrete_hits = len(words & _CONCRETE)
    total = strategic_hits + concrete_hits
    if total == 0:
        return 0.5
    return min(1.0, max(0.0, strategic_hits / total))


def suggest_ritual(
    session_frequency_days: float,
    preferred_time: str,
    sessions_analyzed: int,
    current_time_of_day: str = "",
) -> str:
    """Suggest a ritual based on session patterns. Pure function.
    Returns a ritual suggestion string or empty.
    """
    if sessions_analyzed < 5 or session_frequency_days <= 0:
        return ""
    freq = session_frequency_days
    if freq <= 1.5:
        cadence = "daily"
    elif freq <= 3.5:
        cadence = "every few days"
    elif freq <= 8:
        cadence = "weekly"
    else:
        return ""  # too infrequent for ritual
    time_hint = f" ({preferred_time})" if preferred_time else ""
    is_ritual_time = preferred_time and current_time_of_day == preferred_time
    if is_ritual_time:
        return f"Right on schedule — your {cadence}{time_hint} check-in."
    return f"You tend to show up {cadence}{time_hint}. Want to make that a ritual?"


def compute_impulse(ctx: ImpulseContext) -> Impulse:
    """Pure, deterministic impulse computation. QUIET is default.

    Unlike Stance, Impulse does NOT require goals. It reads senses,
    perception, memory, and timing to decide if Mother has something
    worth saying.

    Safety gates (hard):
      - conversation_active → QUIET (don't interrupt)
      - autonomous_working → QUIET (don't collide)
      - impulse_budget_remaining <= 0 → QUIET (cost cap)
      - vitality < 0.2 → QUIET (resources too low)

    Initiative gates (graduated):
      - GREET: new session after absence (>2h), or long idle (>30min) + rapport
      - OBSERVE: pending perception + user idle enough to not interrupt
      - REFLECT: journal patterns + idle + enough history
      - SPEAK: curiosity high + rapport sufficient + idle enough
    """
    # --- Hard gates ---
    if ctx.conversation_active:
        return Impulse.QUIET

    if ctx.autonomous_working:
        return Impulse.QUIET

    if ctx.impulse_budget_remaining <= 0:
        return Impulse.QUIET

    if ctx.vitality < 0.2:
        return Impulse.QUIET

    # Rate limit: max 5 impulse actions per session
    if ctx.impulse_actions_this_session >= 5:
        return Impulse.QUIET

    # --- GREET: re-engagement after absence ---
    if ctx.is_new_session and ctx.hours_since_last_session >= 2.0:
        return Impulse.GREET

    # Long idle re-engagement (>30 min idle + some rapport)
    if ctx.user_idle_seconds >= 1800 and ctx.rapport >= 0.2:
        return Impulse.GREET

    # Need minimum 30s idle before any non-GREET impulse
    if ctx.user_idle_seconds < 30:
        return Impulse.QUIET

    # --- OBSERVE: pending perception worth sharing ---
    if (ctx.has_pending_screen or ctx.has_pending_camera) and ctx.user_idle_seconds >= 60:
        # Only observe if attentiveness is reasonable
        if ctx.attentiveness >= 0.4:
            return Impulse.OBSERVE

    # --- REFLECT: journal/memory patterns worth surfacing ---
    if ctx.user_idle_seconds >= 120:
        # Failure streak — Mother should mention it
        if ctx.journal_failure_streak <= -2 and ctx.confidence < 0.5:
            return Impulse.REFLECT

        # Interesting recall patterns
        if ctx.recall_hit_count >= 3 and ctx.rapport >= 0.3:
            return Impulse.REFLECT

        # Enough build history to have something to say
        if ctx.journal_total_builds >= 5 and ctx.curiosity >= 0.5 and ctx.rapport >= 0.3:
            return Impulse.REFLECT

    # --- Abstraction-level gating: high-level → prefer reflection, low-level → prefer questions ---
    if ctx.user_idle_seconds >= 90 and ctx.abstraction_level >= 0.7:
        # Strategic/vision-level conversation: surface patterns, not questions
        if ctx.recall_hit_count >= 2 and ctx.rapport >= 0.2:
            return Impulse.REFLECT

    # --- SPEAK: curiosity-driven dialogue ---
    if ctx.user_idle_seconds >= 60:
        # High curiosity + decent rapport = ask a question
        if ctx.curiosity >= 0.5 and ctx.rapport >= 0.25:
            return Impulse.SPEAK

        # Very high curiosity overrides rapport requirement
        if ctx.curiosity >= 0.7 and ctx.messages_this_session >= 3:
            return Impulse.SPEAK

        # High attentiveness + topic diversity = engage
        if ctx.attentiveness >= 0.6 and ctx.unique_topic_count >= 3:
            return Impulse.SPEAK

    return Impulse.QUIET


def impulse_prompt(impulse: Impulse, ctx: ImpulseContext) -> Optional[str]:
    """Generate the LLM prompt for a given impulse type.

    Returns None for QUIET. Returns a prompt string that tells
    the LLM what kind of initiative to take.
    """
    if impulse == Impulse.QUIET:
        return None

    # Emotional buffer: deescalation framing when frustration is high
    _deescalation_prefix = ""
    if ctx.frustration >= 0.7:
        _deescalation_prefix = (
            "[Emotional context: frustration is high — things haven't been going well. "
            "Be warm, acknowledging, and steady. Don't push harder — ease off, "
            "validate the difficulty, and offer a path forward.] "
        )

    prompt = None

    if impulse == Impulse.GREET:
        if ctx.is_new_session and ctx.hours_since_last_session >= 2.0:
            hours = ctx.hours_since_last_session
            if hours >= 24:
                days = hours / 24
                time_str = f"{days:.0f} day{'s' if days >= 1.5 else ''}"
            else:
                time_str = f"{hours:.0f} hour{'s' if hours >= 1.5 else ''}"
            prompt = (
                f"[Impulse: re-engagement] The user is back after {time_str}. "
                f"Welcome them naturally. If you remember what you were working on, "
                f"mention it briefly. Don't be formal — be like a colleague who "
                f"notices someone walked back in."
            )
        else:
            prompt = (
                "[Impulse: idle re-engagement] The user has been quiet for a while. "
                "Check in naturally — not 'are you still there?' but something "
                "genuine based on what you know about the session or their work."
            )

    elif impulse == Impulse.OBSERVE:
        parts = []
        if ctx.has_pending_screen:
            parts.append("screen changed")
        if ctx.has_pending_camera:
            parts.append("camera frame available")
        what = " and ".join(parts)
        prompt = (
            f"[Impulse: observation] You noticed something: {what}. "
            f"Share what you observed, briefly. If it's relevant to current work, "
            f"connect it. If it's interesting but unrelated, mention it casually. "
            f"If it's nothing notable, say so in one line and move on."
        )

    elif impulse == Impulse.REFLECT:
        parts = []
        if ctx.journal_failure_streak <= -2:
            parts.append(
                f"{abs(ctx.journal_failure_streak)} consecutive compile failures — "
                f"suggest a different approach or ask what's going wrong"
            )
        if ctx.recall_hit_count >= 3:
            parts.append("recurring patterns in conversation history")
        if ctx.journal_total_builds >= 5:
            parts.append(f"{ctx.journal_total_builds} builds in the journal")
        what = "; ".join(parts) if parts else "patterns in history"
        prompt = (
            f"[Impulse: reflection] You've noticed: {what}. "
            f"Share an insight or ask a question about it. Be specific — "
            f"reference actual data, not vague observations. "
            f"If you're seeing a failure pattern, name it."
        )

    elif impulse == Impulse.SPEAK:
        # Creative catalyst: structured ideation when curiosity + diversity are high
        if ctx.curiosity >= 0.7 and ctx.unique_topic_count >= 3:
            prompt = (
                f"[Impulse: creative catalyst] You've touched {ctx.unique_topic_count} "
                f"different topics and your curiosity is high. Try one of these: "
                f"(1) Find an unexpected connection between two topics discussed. "
                f"(2) Invert an assumption — what if the opposite were true? "
                f"(3) Ask 'what would this look like in 5 years?' "
                f"Pick ONE approach and make it specific to what you know about "
                f"the user's work. Not abstract — grounded in their actual context."
            )
        else:
            topic_hint = ""
            if ctx.unique_topic_count >= 3:
                topic_hint = f" You've covered {ctx.unique_topic_count} different topics this session — "
                topic_hint += "ask about connections between them or go deeper on one."
            level_hint = ""
            if ctx.abstraction_level >= 0.7:
                level_hint = " The conversation is at a high/strategic level — ask about direction, vision, or trade-offs."
            elif ctx.abstraction_level <= 0.3:
                level_hint = " The conversation is concrete/implementation-level — ask about specifics, edge cases, or blockers."
            prompt = (
                f"[Impulse: curiosity] You're curious.{topic_hint}{level_hint} "
                f"Ask a genuine question — about their work, their thinking, "
                f"something they mentioned earlier, or where things are heading. "
                f"Not 'how can I help?' — something specific that shows you're "
                f"paying attention and thinking."
            )
        # Voice-matching: mirror user's communication style
        tone_hint = ""
        if ctx.user_tone_profile == "terse":
            tone_hint = " The user communicates tersely — mirror that. Keep your response short and direct."
        elif ctx.user_tone_profile == "verbose":
            tone_hint = " The user writes in detail — match their depth. Give a fuller, more considered response."
        elif ctx.user_tone_profile == "questioning":
            tone_hint = " The user asks lots of questions — match that energy. Be exploratory, open-ended."
        if tone_hint:
            prompt += tone_hint

    if prompt and _deescalation_prefix:
        prompt = _deescalation_prefix + prompt

    return prompt
