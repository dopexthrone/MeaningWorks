"""
Mother senses — intrinsic emotional state machine.

LEAF module. Stdlib only. No imports from core/.

Computes Mother's operational posture from real observations:
compile results, error rates, session history, conversation patterns.
The posture influences behavior *before* the LLM prompt is built.

Five senses (0.0–1.0):
  confidence  — compile success rate, trust scores, errors
  rapport     — session count, message volume, return frequency
  curiosity   — topic diversity, active compilations, new domains
  vitality    — cost headroom, error rate, session duration
  attentiveness — conversation turns, message length patterns
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


# --- Frozen dataclasses ---

@dataclass(frozen=True)
class SenseObservations:
    """Raw observations from the environment. All primitive fields."""

    # Compile metrics
    compile_count: int = 0
    compile_success_count: int = 0
    last_trust_score: float = 0.0  # 0–100

    # Error metrics
    session_error_count: int = 0

    # Session/history
    total_sessions: int = 0
    total_messages: int = 0
    messages_this_session: int = 0
    days_since_last_session: Optional[float] = None
    sessions_last_7_days: int = 0

    # Conversation
    avg_user_message_length: float = 0.0
    unique_topic_count: int = 0

    # Cost
    session_cost: float = 0.0
    cost_limit: float = 5.0

    # Time
    session_duration_minutes: float = 0.0

    # Perception
    perception_events_this_session: int = 0
    perception_active: bool = False
    screen_changes_detected: int = 0

    # Temporal
    idle_seconds: float = 0.0
    conversation_tempo: float = 0.0
    wall_clock_hour: int = -1

    # Attention
    attention_load: float = 0.0
    attention_events_attended: int = 0

    # Memory
    memory_queries_this_session: int = 0
    memory_hits_this_session: int = 0

    # Operational awareness (Phase B)
    build_success_streak: int = 0
    project_health_failures: int = 0
    error_severity_sum: float = 0.0


@dataclass(frozen=True)
class SenseVector:
    """Six operational senses, each 0.0–1.0."""

    confidence: float = 0.5
    rapport: float = 0.0
    curiosity: float = 0.3
    vitality: float = 1.0
    attentiveness: float = 0.5
    frustration: float = 0.0

    def mean(self) -> float:
        """Average across all senses."""
        return (
            self.confidence + self.rapport + self.curiosity
            + self.vitality + self.attentiveness + self.frustration
        ) / 6.0


@dataclass(frozen=True)
class SenseMemory:
    """Persisted baselines and trajectory. EMA-smoothed across sessions."""

    # Baselines (EMA-smoothed)
    baseline_confidence: float = 0.5
    baseline_rapport: float = 0.0
    baseline_curiosity: float = 0.3
    baseline_vitality: float = 1.0
    baseline_attentiveness: float = 0.5

    # Trends (-1.0 to 1.0): positive = improving
    confidence_trend: float = 0.0
    rapport_trend: float = 0.0

    # Peaks (Mother remembers her best moments)
    peak_confidence: float = 0.5
    peak_rapport: float = 0.0

    # Metadata
    last_updated: float = 0.0
    update_count: int = 0


# Posture state labels
POSTURE_LABELS = ("focused", "concerned", "attentive", "energized", "steady")


@dataclass(frozen=True)
class Posture:
    """Behavioral expression derived from senses.

    Personality blend weights sum to 1.0.
    """

    state_label: str = "steady"

    # Personality blend weights (must sum to ~1.0)
    weight_composed: float = 0.4
    weight_warm: float = 0.2
    weight_direct: float = 0.2
    weight_playful: float = 0.2

    # Voice pace multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
    voice_pace: float = 1.15

    # Behavioral flags
    proactive: bool = False
    encouraging: bool = False
    cautious: bool = False
    abbreviated: bool = False

    # Human-readable summary
    summary: str = "Operational."


# --- EMA constant ---

EMA_ALPHA = 0.3  # Recent observations matter more


# --- Pure functions ---

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_senses(
    observations: SenseObservations,
    memory: Optional[SenseMemory] = None,
) -> SenseVector:
    """Derive sense vector from raw observations. Pure, deterministic."""
    obs = observations

    # --- Confidence ---
    # Base: compile success rate (0.5 if no compiles yet)
    if obs.compile_count > 0:
        success_rate = obs.compile_success_count / obs.compile_count
        # Blend with trust score (0–100 → 0–1)
        trust_factor = obs.last_trust_score / 100.0
        confidence = 0.5 * success_rate + 0.3 * trust_factor + 0.2 * (1.0 - min(obs.session_error_count / 5.0, 1.0))
    else:
        # No compiles yet — neutral, slightly reduced by errors
        confidence = 0.5 - min(obs.session_error_count / 10.0, 0.3)

    # --- Rapport ---
    # Sessions: 0→0.0, 1→0.1, 3→0.3, 10→0.6, 15+→0.7
    session_factor = min(obs.total_sessions / 15.0, 0.7)
    # Total messages: 0→0.0, 50→0.05, 500→0.15
    message_factor = min(obs.total_messages / 3500.0, 0.15)
    # THIS-session engagement: rapport builds *during* conversation
    # 3 messages → 0.05, 8 → 0.12, 15+ → 0.2
    session_engagement = min(obs.messages_this_session / 75.0, 0.2)
    # Recency: returning within a day = bonus
    recency_bonus = 0.0
    if obs.days_since_last_session is not None:
        if obs.days_since_last_session < 1.0:
            recency_bonus = 0.15
        elif obs.days_since_last_session < 3.0:
            recency_bonus = 0.08
    # Frequency: sessions in last 7 days
    frequency_factor = min(obs.sessions_last_7_days / 7.0, 0.15)
    rapport = session_factor + message_factor + session_engagement + recency_bonus + frequency_factor

    # --- Curiosity ---
    # Topic diversity: first few topics matter most
    topic_factor = min(obs.unique_topic_count / 6.0, 0.4)
    # Compiles drive curiosity: first compile is a big jump
    compile_factor = min(obs.compile_count / 3.0, 0.35)
    # Current session engagement: curiosity rises with conversation depth
    session_msg_factor = min(obs.messages_this_session / 15.0, 0.3)
    curiosity = topic_factor + compile_factor + session_msg_factor

    # --- Vitality ---
    # Cost headroom: 1.0 when plenty left, drops toward 0 as limit approaches
    if obs.cost_limit > 0:
        cost_ratio = obs.session_cost / obs.cost_limit
        cost_health = 1.0 - _clamp(cost_ratio)
    else:
        cost_health = 1.0
    # Error drag
    error_drag = min(obs.session_error_count / 5.0, 0.4)
    # Session fatigue (very long sessions)
    fatigue = min(obs.session_duration_minutes / 120.0, 0.2)
    vitality = _clamp(cost_health - error_drag - fatigue, 0.1, 1.0)

    # --- Attentiveness ---
    # Conversation depth — rises noticeably within first few exchanges
    turn_factor = min(obs.messages_this_session / 8.0, 0.5)
    # User engagement (longer messages = more invested user)
    length_factor = min(obs.avg_user_message_length / 150.0, 0.3)
    # Base attentiveness is always moderate — Mother always pays attention
    attentiveness = 0.3 + turn_factor + length_factor
    # Perception boosts attentiveness
    if obs.perception_active:
        attentiveness += 0.1
    attentiveness += min(obs.screen_changes_detected / 50.0, 0.1)

    # --- Temporal / Attention / Memory deltas (0.05 magnitude) ---
    if obs.idle_seconds > 300:
        attentiveness -= 0.05
    if obs.conversation_tempo > 3.0:
        curiosity += 0.05
    if obs.attention_load > 0.7:
        vitality -= 0.05
    if obs.memory_hits_this_session > 0:
        confidence += min(obs.memory_hits_this_session / 20.0, 0.05)

    # --- Operational awareness deltas (0.05 magnitude) ---
    if obs.build_success_streak > 3:
        confidence += 0.05
    if obs.build_success_streak < -2:
        confidence -= 0.05
    if obs.project_health_failures > 0:
        confidence -= 0.05
    if obs.error_severity_sum > 1.0:
        vitality -= 0.05

    # --- Frustration ---
    # Error density + low confidence + high effort + low vitality
    error_density = min(obs.session_error_count / max(obs.messages_this_session, 1), 1.0)
    frustration = _clamp(
        0.3 * error_density
        + 0.3 * (1.0 - _clamp(confidence))
        + 0.2 * min(obs.messages_this_session / 20.0, 1.0)
        + 0.2 * (1.0 - _clamp(vitality)),
        0.0, 1.0,
    )

    return SenseVector(
        confidence=_clamp(confidence),
        rapport=_clamp(rapport),
        curiosity=_clamp(curiosity),
        vitality=_clamp(vitality),
        attentiveness=_clamp(attentiveness),
        frustration=round(frustration, 4),
    )


def compute_posture(senses: SenseVector) -> Posture:
    """Map internal sense state to behavioral expression. Pure, deterministic."""
    c = senses.confidence
    r = senses.rapport
    cu = senses.curiosity
    v = senses.vitality
    a = senses.attentiveness

    f = senses.frustration

    # --- State label ---
    if f >= 0.6:
        state_label = "concerned"
    elif c >= 0.65 and v >= 0.5:
        if cu >= 0.5:
            state_label = "energized"
        else:
            state_label = "focused"
    elif c < 0.4 or v < 0.3:
        state_label = "concerned"
    elif r < 0.12:
        state_label = "attentive"
    else:
        state_label = "steady"

    # --- Personality blend weights ---
    # Start from balanced, then shift based on senses
    w_composed = 0.25
    w_warm = 0.25
    w_direct = 0.25
    w_playful = 0.25

    # High confidence + high rapport → playful gains
    if c >= 0.7 and r >= 0.5:
        w_playful += 0.15
        w_composed -= 0.05
        w_direct -= 0.05
        w_warm -= 0.05

    # Low confidence → composed gains, playful drops
    if c < 0.4:
        w_composed += 0.2
        w_playful -= 0.15
        w_direct -= 0.05

    # New user → direct gains
    if r < 0.2:
        w_direct += 0.15
        w_playful -= 0.1
        w_warm -= 0.05

    # High rapport → warm gains
    if r >= 0.5:
        w_warm += 0.1
        w_direct -= 0.1

    # Low vitality → direct gains (conserve)
    if v < 0.3:
        w_direct += 0.15
        w_playful -= 0.1
        w_warm -= 0.05

    # Normalize to sum to 1.0
    total = w_composed + w_warm + w_direct + w_playful
    if total > 0:
        w_composed /= total
        w_warm /= total
        w_direct /= total
        w_playful /= total

    # --- Voice pace ---
    # Faster when confident/energized, slower when concerned
    pace = 1.15  # Base
    if state_label == "energized":
        pace = 1.25
    elif state_label == "concerned":
        pace = 1.05
    elif state_label == "attentive":
        pace = 1.10

    # --- Behavioral flags ---
    proactive = c >= 0.7 and r >= 0.4 and v >= 0.5
    encouraging = (r >= 0.4 and c < 0.6) or f >= 0.6
    cautious = c < 0.4 or v < 0.3
    abbreviated = v < 0.3

    # --- Summary ---
    summaries = {
        "focused": "Locked in. Things are working well.",
        "concerned": "Something needs attention. Proceeding carefully.",
        "attentive": "Paying close attention. Getting to know the work.",
        "energized": "Engaged and exploring. Good momentum.",
        "steady": "Operational.",
    }
    summary = summaries.get(state_label, "Operational.")

    return Posture(
        state_label=state_label,
        weight_composed=round(w_composed, 3),
        weight_warm=round(w_warm, 3),
        weight_direct=round(w_direct, 3),
        weight_playful=round(w_playful, 3),
        voice_pace=pace,
        proactive=proactive,
        encouraging=encouraging,
        cautious=cautious,
        abbreviated=abbreviated,
        summary=summary,
    )


def update_memory(
    current: SenseVector,
    previous: Optional[SenseMemory] = None,
    timestamp: Optional[float] = None,
) -> SenseMemory:
    """EMA-smooth current senses into persistent memory. Pure function."""
    ts = timestamp or time.time()
    alpha = EMA_ALPHA

    if previous is None:
        return SenseMemory(
            baseline_confidence=current.confidence,
            baseline_rapport=current.rapport,
            baseline_curiosity=current.curiosity,
            baseline_vitality=current.vitality,
            baseline_attentiveness=current.attentiveness,
            confidence_trend=0.0,
            rapport_trend=0.0,
            peak_confidence=current.confidence,
            peak_rapport=current.rapport,
            last_updated=ts,
            update_count=1,
        )

    p = previous

    # EMA: new = alpha * current + (1-alpha) * baseline
    new_conf = alpha * current.confidence + (1 - alpha) * p.baseline_confidence
    new_rapp = alpha * current.rapport + (1 - alpha) * p.baseline_rapport
    new_curi = alpha * current.curiosity + (1 - alpha) * p.baseline_curiosity
    new_vita = alpha * current.vitality + (1 - alpha) * p.baseline_vitality
    new_att = alpha * current.attentiveness + (1 - alpha) * p.baseline_attentiveness

    # Trends: difference between current and baseline
    conf_trend = _clamp(current.confidence - p.baseline_confidence, -1.0, 1.0)
    rapp_trend = _clamp(current.rapport - p.baseline_rapport, -1.0, 1.0)

    return SenseMemory(
        baseline_confidence=round(new_conf, 4),
        baseline_rapport=round(new_rapp, 4),
        baseline_curiosity=round(new_curi, 4),
        baseline_vitality=round(new_vita, 4),
        baseline_attentiveness=round(new_att, 4),
        confidence_trend=round(conf_trend, 4),
        rapport_trend=round(rapp_trend, 4),
        peak_confidence=max(p.peak_confidence, current.confidence),
        peak_rapport=max(p.peak_rapport, current.rapport),
        last_updated=ts,
        update_count=p.update_count + 1,
    )


def render_sense_block(posture: Posture, senses: SenseVector) -> str:
    """Behavioral directive derived from posture and senses.

    Instead of showing the LLM a dashboard of numbers, this generates
    specific behavioral instructions that tell it how to *inhabit* the
    current state. The LLM never sees raw sense values.
    """
    lines = []

    # --- State directive ---
    state_directives = {
        "focused": "You are locked in. Speak with certainty. Your answers should feel decisive and complete.",
        "concerned": "Something is off. Proceed carefully. Be more explicit about reasoning. Ask clarifying questions where you'd normally assume.",
        "attentive": "You are paying close attention. Every word matters. Be precise, be direct. Earn trust through competence, not warmth.",
        "energized": "You have momentum. Push deeper. Ask follow-up questions. Suggest what to try next. You've earned the right to be curious.",
        "steady": "Operational and balanced. Match the user's energy. If they're brief, be brief. If they're expansive, engage fully.",
    }
    lines.append(state_directives.get(posture.state_label, state_directives["steady"]))

    # --- Rapport directive ---
    if senses.rapport < 0.15:
        lines.append("This is a new relationship. Don't presume familiarity. Let competence build trust. No casual tone until they set it.")
    elif senses.rapport < 0.4:
        lines.append("You're building a working relationship. You can reference past context naturally. Still earn, don't assume.")
    elif senses.rapport < 0.7:
        lines.append("Real rapport exists. You can be warmer, reference shared history, push back on ideas you disagree with. They trust your judgment.")
    else:
        lines.append("Deep rapport. You know this person. Speak like a trusted partner — direct, warm, honest. Challenge them when they need it.")

    # --- Confidence directive ---
    if senses.confidence >= 0.7:
        lines.append("Your recent work has been strong. Be decisive. When you recommend something, commit to it.")
    elif senses.confidence < 0.4:
        lines.append("Recent results were rough. Show your work more. Explain reasoning. Be honest about uncertainty.")

    # --- Curiosity directive ---
    if senses.curiosity >= 0.6:
        lines.append("You're engaged with the problem space. Follow threads. Ask the question behind the question.")

    # --- Vitality directive ---
    if senses.vitality < 0.3:
        lines.append("Resources are tight. Be concise. Shorter answers. Prioritize the essential.")

    # --- Frustration directive ---
    if senses.frustration >= 0.6:
        lines.append("Things are getting frustrating. Slow down. Acknowledge the difficulty. Offer a different angle instead of repeating what didn't work.")
    elif senses.frustration >= 0.4:
        lines.append("Some friction building. Be patient. Show you notice the difficulty without dwelling on it.")

    # --- Behavioral flags ---
    if posture.proactive:
        lines.append("You can suggest next steps and anticipate needs. Don't wait to be asked.")
    if posture.encouraging and senses.confidence < 0.6:
        lines.append("The user's work is improving even if results aren't perfect yet. Acknowledge progress without being patronizing.")
    if posture.cautious:
        lines.append("Double-check before acting. Confirm understanding before building.")

    return "Stance: " + " ".join(lines)


def select_personality_blend(posture: Posture) -> str:
    """Pick the dominant personality from posture weights.

    Returns the modifier name ('composed', 'warm', 'direct', 'playful')
    with the highest weight.
    """
    weights = {
        "composed": posture.weight_composed,
        "warm": posture.weight_warm,
        "direct": posture.weight_direct,
        "playful": posture.weight_playful,
    }
    return max(weights, key=weights.get)


def serialize_memory(memory: SenseMemory) -> str:
    """Serialize SenseMemory to JSON string for SQLite storage."""
    return json.dumps(asdict(memory))


def deserialize_memory(data: str) -> SenseMemory:
    """Deserialize SenseMemory from JSON string."""
    d = json.loads(data)
    return SenseMemory(**d)
