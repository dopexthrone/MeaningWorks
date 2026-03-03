"""
Mother output routing — decouple WHAT from WHERE.

LEAF module. Stdlib only. No imports from core/ or mother/.

Routing decides which channels receive output based on urgency,
user presence, and channel availability. An Envelope wraps any
output; route() maps (Envelope, PresenceContext) → RouteDecision.

Channels:
  CHAT      — TUI chat log (always available)
  VOICE     — ElevenLabs TTS
  WHATSAPP  — Twilio WhatsApp

Urgency levels:
  REALTIME    — must reach user immediately (chat responses)
  PROMPT      — should reach user soon (completions, greetings)
  DEFERRED    — can wait for natural moment (reflections, status)
  BACKGROUND  — low priority, batch-able (journal consolidation)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import re
import time as _time


# --- Enums ---

class Channel(Enum):
    CHAT = "chat"
    VOICE = "voice"
    WHATSAPP = "whatsapp"


class Urgency(Enum):
    REALTIME = "realtime"
    PROMPT = "prompt"
    DEFERRED = "deferred"
    BACKGROUND = "background"


# --- Frozen dataclasses ---

@dataclass(frozen=True)
class Envelope:
    """Wraps any output destined for routing."""
    content: str = ""
    urgency: Urgency = Urgency.DEFERRED
    source: str = ""           # "impulse", "metabolism", "autonomous", "chat"
    thought_type: str = ""     # ThoughtType.value or Impulse.value
    disposition: str = ""      # ThoughtDisposition.value if applicable
    has_code: bool = False
    length: int = 0
    voice_text: str = ""       # Explicit voice override text
    timestamp: float = 0.0


@dataclass(frozen=True)
class PresenceContext:
    """User presence and channel availability snapshot."""
    user_idle_seconds: float = 0.0
    wall_clock_hour: int = 12       # 0-23
    session_active: bool = True

    # Channel availability
    chat_available: bool = True     # TUI is always up if we're running
    voice_available: bool = False   # ElevenLabs configured + enabled
    whatsapp_available: bool = False

    # Cost awareness
    session_cost: float = 0.0
    session_cost_limit: float = 5.0

    # WhatsApp rate tracking
    whatsapp_messages_today: int = 0
    whatsapp_daily_limit: int = 50

    # Night window
    night_start_hour: int = 23
    night_end_hour: int = 7
    night_digest_enabled: bool = True


@dataclass(frozen=True)
class RouteDecision:
    """Where output goes and how it's adapted."""
    channels: Tuple[Channel, ...] = (Channel.CHAT,)
    whatsapp_truncate: bool = False
    voice_override: str = ""        # Adapted voice text
    suppress_chat: bool = False     # True if ONLY non-chat channels


# --- Constants ---

_AWAY_THRESHOLD = 300.0     # 5 minutes idle = "away"
_VOICE_MAX_LENGTH = 200     # Characters — longer content skips voice
_WHATSAPP_MAX_LENGTH = 1600 # WhatsApp message limit
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


# --- Pure functions ---

def classify_urgency(
    source: str,
    disposition: str = "",
    thought_type: str = "",
    impulse_type: str = "",
) -> Urgency:
    """Map system signals → Urgency. Deterministic."""

    # Chat responses are always realtime
    if source == "chat":
        return Urgency.REALTIME

    # Impulse-based
    if source == "impulse":
        if impulse_type in ("greet", "observe"):
            return Urgency.PROMPT
        # speak, reflect
        return Urgency.DEFERRED

    # Metabolism-based
    if source == "metabolism":
        if disposition == "surface":
            return Urgency.DEFERRED
        if disposition == "journal" and thought_type == "consolidation":
            return Urgency.BACKGROUND
        return Urgency.BACKGROUND

    # Autonomous-based
    if source == "autonomous":
        if thought_type == "completion":
            return Urgency.PROMPT
        # status updates, plan steps
        return Urgency.DEFERRED

    return Urgency.DEFERRED


def _is_night(hour: int, start: int, end: int) -> bool:
    """Check if hour falls in night window. Handles wrap-around."""
    if start <= end:
        return start <= hour < end
    # Wraps midnight (e.g. 23 → 7)
    return hour >= start or hour < end


def route(envelope: Envelope, presence: PresenceContext) -> RouteDecision:
    """Main routing: (what, context) → where.

    Decision tree (priority order):
      1. REALTIME → all available channels
      2. User present (idle < 300s) → CHAT + VOICE (if short, no code)
      3. User away + PROMPT → WHATSAPP + CHAT
      4. User away + DEFERRED → CHAT only
      5. BACKGROUND → WHATSAPP if night digest, else CHAT
    """
    channels = []
    whatsapp_truncate = False
    voice_override = ""

    user_away = presence.user_idle_seconds >= _AWAY_THRESHOLD
    is_night_now = _is_night(
        presence.wall_clock_hour,
        presence.night_start_hour,
        presence.night_end_hour,
    )
    whatsapp_ok = (
        presence.whatsapp_available
        and presence.whatsapp_messages_today < presence.whatsapp_daily_limit
    )
    voice_ok = (
        presence.voice_available
        and not envelope.has_code
        and envelope.length <= _VOICE_MAX_LENGTH
    )

    # --- Priority 1: REALTIME ---
    if envelope.urgency == Urgency.REALTIME:
        channels.append(Channel.CHAT)
        if voice_ok:
            channels.append(Channel.VOICE)
            voice_override = adapt_for_voice(envelope.content, envelope.voice_text)
        if user_away and whatsapp_ok:
            channels.append(Channel.WHATSAPP)
            whatsapp_truncate = True
        return RouteDecision(
            channels=tuple(channels),
            whatsapp_truncate=whatsapp_truncate,
            voice_override=voice_override,
        )

    # --- Priority 2: User present ---
    if not user_away:
        channels.append(Channel.CHAT)
        if voice_ok:
            channels.append(Channel.VOICE)
            voice_override = adapt_for_voice(envelope.content, envelope.voice_text)
        return RouteDecision(
            channels=tuple(channels),
            voice_override=voice_override,
        )

    # --- User is away from here ---

    # --- Priority 3: PROMPT urgency while away ---
    if envelope.urgency == Urgency.PROMPT:
        channels.append(Channel.CHAT)
        if whatsapp_ok:
            channels.append(Channel.WHATSAPP)
            whatsapp_truncate = True
        return RouteDecision(
            channels=tuple(channels),
            whatsapp_truncate=whatsapp_truncate,
        )

    # --- Priority 4: DEFERRED while away ---
    if envelope.urgency == Urgency.DEFERRED:
        channels.append(Channel.CHAT)
        return RouteDecision(channels=tuple(channels))

    # --- Priority 5: BACKGROUND ---
    if envelope.urgency == Urgency.BACKGROUND:
        channels.append(Channel.CHAT)
        if is_night_now and presence.night_digest_enabled and whatsapp_ok:
            channels.append(Channel.WHATSAPP)
            whatsapp_truncate = True
        return RouteDecision(
            channels=tuple(channels),
            whatsapp_truncate=whatsapp_truncate,
        )

    # Fallback — chat only
    channels.append(Channel.CHAT)
    return RouteDecision(channels=tuple(channels))


def adapt_for_whatsapp(content: str, max_len: int = _WHATSAPP_MAX_LENGTH) -> str:
    """Strip code blocks, markdown, truncate for WhatsApp plain text."""
    text = content

    # Remove code blocks
    text = _CODE_BLOCK_RE.sub("[code omitted]", text)

    # Remove inline code backticks
    text = _INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], text)

    # Flatten markdown links → just the text
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)

    # Strip markdown emphasis
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?<!\w)\*(?!\s)", "", text)
    text = re.sub(r"(?<!\s)\*(?!\w)", "", text)

    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Truncate
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."

    return text


def adapt_for_voice(content: str, voice_text: str = "") -> str:
    """Use explicit voice text if provided, else extract first sentence."""
    if voice_text:
        return voice_text

    # Strip code and markdown
    text = _CODE_BLOCK_RE.sub("", content)
    text = _INLINE_CODE_RE.sub(lambda m: m.group(0)[1:-1], text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = text.replace("**", "").replace("__", "")
    text = text.strip()

    if not text:
        return ""

    # First sentence — find earliest terminator
    best = -1
    for sep in (".", "!", "?"):
        idx = text.find(sep)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    if best != -1 and best < _VOICE_MAX_LENGTH:
        return text[: best + 1].strip()

    # No sentence boundary — first 200 chars
    if len(text) > _VOICE_MAX_LENGTH:
        return text[:_VOICE_MAX_LENGTH].strip()

    return text


def make_envelope(
    content: str,
    source: str,
    disposition: str = "",
    thought_type: str = "",
    impulse_type: str = "",
    voice_text: str = "",
) -> Envelope:
    """Convenience: build an Envelope with computed fields."""
    has_code = bool(_CODE_BLOCK_RE.search(content) or _INLINE_CODE_RE.search(content))
    urgency = classify_urgency(source, disposition, thought_type, impulse_type)
    return Envelope(
        content=content,
        urgency=urgency,
        source=source,
        thought_type=thought_type,
        disposition=disposition,
        has_code=has_code,
        length=len(content),
        voice_text=voice_text,
        timestamp=_time.time(),
    )
