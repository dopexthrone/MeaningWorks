"""
Mother metabolism — continuous internal processing independent of conversation.

LEAF module. Stdlib only. No imports from core/ or mother/.

Metabolism determines what Mother thinks about when not responding to a user.
It varies by mode — from deep consolidation during sleep to parallel
implication threading during conversation. The user never sees mode labels
or depth scores; they experience variable response depth, unprompted insights,
and the sense that Mother thinks between interactions.

Metabolic modes:
  SLEEP     — night hours + long idle: memory consolidation
  IDLE      — user absent, system on: low-energy review
  DEEP      — triggered by complexity: dedicated processing
  PARALLEL  — during conversation: background implication threading
  ACTIVE    — normal mode, no metabolism tick fires
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class MetabolicMode(Enum):
    SLEEP = "sleep"        # Night hours + long idle: consolidation
    IDLE = "idle"          # User absent, system on: low-energy review
    DEEP = "deep"          # Triggered by complexity: dedicated processing
    PARALLEL = "parallel"  # During conversation: background implication threading
    ACTIVE = "active"      # Normal mode, no metabolism tick fires


class ThoughtType(Enum):
    CONNECTION = "connection"      # Linking two previously separate observations
    QUESTION = "question"          # Something Mother wants to understand
    PATTERN = "pattern"            # Recurring structure across sessions
    CONSOLIDATION = "consolidation"  # Memory compression/synthesis
    IMPLICATION = "implication"    # Consequence of current work not yet discussed
    CURIOSITY = "curiosity"        # Something interesting, not urgent


class ThoughtDisposition(Enum):
    SURFACE = "surface"    # Worth mentioning when opportunity arises
    INTERNAL = "internal"  # Enriches context silently
    JOURNAL = "journal"    # Persist for later
    DISCARD = "discard"    # Not interesting enough


@dataclass(frozen=True)
class MetabolicContext:
    """All signals the metabolism system reads. Pure data."""

    # Time
    wall_clock_hour: int = 12           # 0-23
    user_idle_seconds: float = 0.0
    session_duration_minutes: float = 0.0

    # Senses (0.0-1.0)
    curiosity: float = 0.3
    attentiveness: float = 0.5
    rapport: float = 0.0
    confidence: float = 0.5
    vitality: float = 1.0

    # Conversation state
    conversation_active: bool = False
    autonomous_working: bool = False
    messages_this_session: int = 0
    unique_topic_count: int = 0

    # Content signals
    last_compile_trust: Optional[float] = None
    last_compile_weakest: Optional[str] = None
    journal_failure_streak: int = 0
    recent_topics: List[str] = field(default_factory=list)
    recall_hit_count: int = 0

    # Budget
    session_cost: float = 0.0
    session_cost_limit: float = 5.0
    metabolism_session_cost: float = 0.0
    metabolism_budget: float = 0.30

    # Processing state
    thoughts_this_session: int = 0
    max_thoughts_per_session: int = 20
    last_thought_time: float = 0.0
    current_time: float = 0.0
    deep_think_subject: Optional[str] = None

    # Abstraction level (0.0=concrete/code, 0.5=design, 1.0=vision/strategy)
    abstraction_level: float = 0.5

    # Sleep config
    sleep_start_hour: int = 2
    sleep_end_hour: int = 7


@dataclass(frozen=True)
class Thought:
    """A single internal thought."""

    thought_type: ThoughtType = ThoughtType.CURIOSITY
    disposition: ThoughtDisposition = ThoughtDisposition.INTERNAL
    subject: str = ""
    trigger: str = ""
    depth: float = 0.0        # 0-1, how deeply processed
    mode: MetabolicMode = MetabolicMode.ACTIVE
    timestamp: float = 0.0


@dataclass(frozen=True)
class MetabolicState:
    """Snapshot for context enrichment."""

    mode: MetabolicMode = MetabolicMode.ACTIVE
    thought_count: int = 0
    recent_thoughts: List[Thought] = field(default_factory=list)
    surfaceable_count: int = 0
    inner_narrative: str = ""


# --- Rate limits by mode ---

_MIN_INTERVAL = {
    MetabolicMode.SLEEP: 600.0,     # 10 min
    MetabolicMode.IDLE: 300.0,      # 5 min
    MetabolicMode.DEEP: 120.0,      # 2 min
    MetabolicMode.PARALLEL: 180.0,  # 3 min
    MetabolicMode.ACTIVE: 0.0,      # never fires
}

_MAX_THOUGHTS = {
    MetabolicMode.SLEEP: 20,
    MetabolicMode.IDLE: 20,
    MetabolicMode.DEEP: 20,
    MetabolicMode.PARALLEL: 20,
    MetabolicMode.ACTIVE: 0,
}


def compute_metabolic_mode(ctx: MetabolicContext) -> MetabolicMode:
    """Pure, deterministic mode computation. ACTIVE is default.

    Priority order:
      1. PARALLEL — conversation active with enough depth
      2. ACTIVE — autonomous working (no metabolism)
      3. SLEEP — night hours + long idle
      4. DEEP — explicit deep-think subject set
      5. IDLE — user absent, content to process
      6. ACTIVE — default (no metabolism)
    """
    # Priority 1: PARALLEL during rich conversation
    if ctx.conversation_active and not ctx.deep_think_subject:
        if (ctx.unique_topic_count >= 2
                and ctx.messages_this_session >= 3
                and ctx.curiosity >= 0.4):
            return MetabolicMode.PARALLEL

    # Priority 2: autonomous working — don't interfere
    if ctx.autonomous_working:
        return MetabolicMode.ACTIVE

    # Priority 3: SLEEP — night hours + long idle
    if ctx.sleep_start_hour <= ctx.sleep_end_hour:
        is_night = ctx.sleep_start_hour <= ctx.wall_clock_hour < ctx.sleep_end_hour
    else:
        is_night = ctx.wall_clock_hour >= ctx.sleep_start_hour or ctx.wall_clock_hour < ctx.sleep_end_hour
    if is_night and ctx.user_idle_seconds >= 1800:
        return MetabolicMode.SLEEP

    # Priority 4: DEEP — explicit subject for focused processing
    if ctx.deep_think_subject:
        return MetabolicMode.DEEP

    # Priority 5: IDLE — user absent, something to process
    if ctx.user_idle_seconds >= 120:
        has_content = (
            ctx.messages_this_session >= 1
            or ctx.recall_hit_count >= 1
            or len(ctx.recent_topics) >= 1
        )
        if has_content:
            return MetabolicMode.IDLE

    # Default: no metabolism
    return MetabolicMode.ACTIVE


def should_think(ctx: MetabolicContext, mode: MetabolicMode) -> bool:
    """Gate: should a thought be generated right now?

    Hard gates (any one blocks):
      - ACTIVE mode → never
      - Budget exhausted → never
      - Session cost > 80% of limit → never
      - Vitality < 0.15 → never
      - Thoughts this session >= max → never
      - Too soon since last thought (mode-specific) → never
    """
    # ACTIVE never fires
    if mode == MetabolicMode.ACTIVE:
        return False

    # Budget exhausted
    if ctx.metabolism_session_cost >= ctx.metabolism_budget:
        return False

    # Global session cost gate (80%)
    if ctx.session_cost_limit > 0 and ctx.session_cost >= ctx.session_cost_limit * 0.8:
        return False

    # Vitality gate
    if ctx.vitality < 0.15:
        return False

    # Max thoughts per session
    if ctx.thoughts_this_session >= ctx.max_thoughts_per_session:
        return False

    # Rate limit per mode
    min_interval = _MIN_INTERVAL.get(mode, 300.0)
    if ctx.current_time > 0 and ctx.last_thought_time > 0:
        elapsed = ctx.current_time - ctx.last_thought_time
        if elapsed < min_interval:
            return False

    return True


def classify_thought_type(mode: MetabolicMode, ctx: MetabolicContext) -> ThoughtType:
    """What kind of thought should be generated? Deterministic from mode + signals."""

    if mode == MetabolicMode.SLEEP:
        # Sleep consolidates memories
        if len(ctx.recent_topics) >= 3:
            return ThoughtType.CONSOLIDATION
        if ctx.recall_hit_count >= 2:
            return ThoughtType.PATTERN
        return ThoughtType.CONSOLIDATION

    if mode == MetabolicMode.IDLE:
        # Idle reviews and connects — abstraction level selects thought type
        if ctx.abstraction_level >= 0.7:
            # High-level: prefer connections and implications over curiosity
            if ctx.recall_hit_count >= 2:
                return ThoughtType.CONNECTION
            return ThoughtType.IMPLICATION
        if ctx.curiosity >= 0.5:
            return ThoughtType.CURIOSITY
        if ctx.recall_hit_count >= 2:
            return ThoughtType.CONNECTION
        if ctx.journal_failure_streak <= -2:
            return ThoughtType.QUESTION
        return ThoughtType.CURIOSITY

    if mode == MetabolicMode.DEEP:
        # Deep processes implications of a specific subject
        if ctx.last_compile_trust is not None and ctx.last_compile_trust < 40:
            return ThoughtType.QUESTION
        return ThoughtType.IMPLICATION

    if mode == MetabolicMode.PARALLEL:
        # Parallel threads implications during conversation
        if ctx.unique_topic_count >= 3:
            return ThoughtType.CONNECTION
        return ThoughtType.IMPLICATION

    return ThoughtType.CURIOSITY


def classify_disposition(
    thought_type: ThoughtType,
    mode: MetabolicMode,
    ctx: MetabolicContext,
) -> ThoughtDisposition:
    """What to do with a thought once generated."""

    # Sleep thoughts → journal (persist for next session)
    if mode == MetabolicMode.SLEEP:
        return ThoughtDisposition.JOURNAL

    # Deep thoughts → surface (worth mentioning)
    if mode == MetabolicMode.DEEP:
        return ThoughtDisposition.SURFACE

    # High-curiosity connections → surface
    if thought_type == ThoughtType.CONNECTION and ctx.curiosity >= 0.5:
        return ThoughtDisposition.SURFACE

    # Implications during conversation → surface
    if thought_type == ThoughtType.IMPLICATION and mode == MetabolicMode.PARALLEL:
        return ThoughtDisposition.SURFACE

    # Questions → surface (Mother wants to ask)
    if thought_type == ThoughtType.QUESTION:
        return ThoughtDisposition.SURFACE

    # Consolidation → journal
    if thought_type == ThoughtType.CONSOLIDATION:
        return ThoughtDisposition.JOURNAL

    # Patterns → internal (enrich context)
    if thought_type == ThoughtType.PATTERN:
        return ThoughtDisposition.INTERNAL

    # Low-curiosity idle thoughts → internal
    if mode == MetabolicMode.IDLE and ctx.curiosity < 0.4:
        return ThoughtDisposition.INTERNAL

    return ThoughtDisposition.INTERNAL


def compute_depth(mode: MetabolicMode, ctx: MetabolicContext) -> float:
    """How deeply processed this thought should be. 0.0-1.0.

    SLEEP is deep (consolidation needs thoroughness).
    DEEP is deepest.
    PARALLEL is shallow (background, don't slow conversation).
    IDLE is moderate.
    """
    if mode == MetabolicMode.DEEP:
        return min(1.0, 0.7 + ctx.curiosity * 0.3)

    if mode == MetabolicMode.SLEEP:
        return 0.6

    if mode == MetabolicMode.IDLE:
        return 0.3 + ctx.curiosity * 0.2

    if mode == MetabolicMode.PARALLEL:
        return 0.2

    return 0.0


def metabolism_prompt(
    mode: MetabolicMode,
    thought_type: ThoughtType,
    ctx: MetabolicContext,
) -> Optional[str]:
    """Generate the LLM prompt for an internal thought.

    Returns None for ACTIVE mode (no thinking happens).
    The prompt is compact — ~200 tokens system, ~50 tokens instruction.
    """
    if mode == MetabolicMode.ACTIVE:
        return None

    # Build topic context
    topic_hint = ""
    if ctx.recent_topics:
        topics = ctx.recent_topics[:5]
        topic_hint = f" Recent topics: {', '.join(topics)}."

    compile_hint = ""
    if ctx.last_compile_trust is not None:
        compile_hint = f" Last compile trust: {ctx.last_compile_trust:.0f}%."
        if ctx.last_compile_weakest:
            compile_hint += f" Weakest: {ctx.last_compile_weakest}."

    failure_hint = ""
    if ctx.journal_failure_streak <= -2:
        failure_hint = f" {abs(ctx.journal_failure_streak)} consecutive failures."

    deep_hint = ""
    if ctx.deep_think_subject:
        deep_hint = f" Focus: {ctx.deep_think_subject}."

    context_line = f"{topic_hint}{compile_hint}{failure_hint}{deep_hint}".strip()

    # Type-specific instructions
    type_instructions = {
        ThoughtType.CONNECTION: (
            "Find a connection between two separate observations or topics "
            "from the session. Name the link concretely."
        ),
        ThoughtType.QUESTION: (
            "Formulate one specific question about something you want to "
            "understand better. Be precise — not vague curiosity."
        ),
        ThoughtType.PATTERN: (
            "Identify a recurring pattern across sessions or topics. "
            "State it as a concrete observation."
        ),
        ThoughtType.CONSOLIDATION: (
            "Compress and synthesize what you know about the recent topics "
            "into a single insight. Discard noise, keep signal."
        ),
        ThoughtType.IMPLICATION: (
            "What is one consequence or implication of the current work "
            "that hasn't been discussed? Be specific."
        ),
        ThoughtType.CURIOSITY: (
            "What's one thing that genuinely interests you about the "
            "current context? Not a question — a direction of interest."
        ),
    }

    instruction = type_instructions.get(thought_type, type_instructions[ThoughtType.CURIOSITY])

    return (
        f"[Internal thought — {mode.value} mode]\n"
        f"{context_line}\n"
        f"{instruction}\n"
        f"Respond in 1-2 sentences. No preamble. No meta-commentary."
    )


def render_metabolism_context(state: MetabolicState) -> str:
    """Render MetabolicState into a context injection string.

    Returns empty string if nothing worth injecting.
    """
    if state.mode == MetabolicMode.ACTIVE and state.thought_count == 0:
        return ""

    parts: List[str] = []

    if state.inner_narrative:
        parts.append(state.inner_narrative)

    if state.surfaceable_count > 0:
        # Surface recent thoughts worth mentioning
        surfaceable = [
            t for t in state.recent_thoughts
            if t.disposition == ThoughtDisposition.SURFACE
        ]
        if surfaceable:
            thought_strs = [t.subject for t in surfaceable[:3] if t.subject]
            if thought_strs:
                parts.append("Thinking about: " + "; ".join(thought_strs) + ".")

    if not parts:
        return ""

    return "\n".join(parts)


def should_enforce_boundary(
    session_age_seconds: float,
    vitality: float,
    frustration: float = 0.0,
) -> str:
    """Check if Mother should suggest a break. Pure function.
    Returns "break", "warning", or "".
    """
    hours = session_age_seconds / 3600.0
    if hours >= 4.0 and vitality < 0.3:
        return "break"
    if hours >= 3.0 and vitality < 0.4:
        return "warning"
    if hours >= 2.0 and frustration >= 0.7:
        return "warning"
    return ""
