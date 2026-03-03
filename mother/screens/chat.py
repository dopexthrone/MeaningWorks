"""
Chat screen — main interaction surface with Mother.

Layout: StatusBar top, ChatArea center, InputArea bottom.
Handles chat flow, slash commands, and history persistence.
"""

import asyncio
import logging
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List, Dict

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Static, Footer
from textual.worker import Worker

from mother.config import MotherConfig, load_config, save_config
from mother.memory import ConversationStore, ChatMessage
from mother.bridge import EngineBridge
from mother.persona import (
    build_system_prompt,
    build_introspection_snapshot,
    build_greeting,
    narrate_error,
    inject_personality_bite,
)
from mother.context import ContextData, synthesize_context
from mother.relationship import (
    RelationshipInsight,
    extract_relationship_insights,
    synthesize_relationship_narrative,
)
from mother.voice import VoiceBridge, is_voice_available
from mother.screen import ScreenCaptureBridge, is_screen_capture_available
from mother.microphone import MicrophoneBridge, is_microphone_available
from mother.camera import CameraBridge, is_camera_available
from mother.perception import (
    PerceptionEngine,
    PerceptionConfig,
    PerceptionEvent,
    PerceptionEventType,
)
from mother.senses import (
    SenseObservations,
    SenseVector,
    SenseMemory,
    Posture,
    compute_senses,
    compute_posture,
    update_memory,
    render_sense_block,
    serialize_memory,
    deserialize_memory,
)
from mother.temporal import TemporalEngine
from mother.attention import AttentionFilter
from mother.recall import RecallEngine
from mother.journal import BuildJournal, JournalEntry
from mother.compile_render import render_compile_output
from mother.error_taxonomy import classify_error, summarize_errors, ErrorClassification
from mother.widgets.chat_area import ChatArea
from mother.widgets.input_area import InputArea, UserTextSubmitted
from mother.widgets.status_bar import StatusBar
from mother.widgets.pipeline import PipelinePanel, PIPELINE_STAGES

logger = logging.getLogger("mother.chat")

GREETING = "How may I assist you today?"

HELP_TEXT = """Available commands:
  /compile <desc>    — Compile a description into a blueprint
  /build <desc>      — Compile and emit as a working project
  /launch            — Launch (or re-launch) the last built project
  /stop              — Stop the running project
  /capture [question] — Capture screen and describe what's visible
  /camera [question]  — Capture webcam frame and describe what's visible
  /listen [seconds]  — Record audio and transcribe (default 5s, or F8)
  /search <query>    — Search files on this machine
  /find <query>      — Alias for /search
  /tools             — List available tools
  /ideas             — List pending ideas
  /status            — Show session info
  /handoff           — Generate handoff document
  /theme [name]      — Switch theme (default, alien) or show current
  /help              — Show this help
  /clear             — Clear chat history
  /settings          — Open settings"""

SLASH_COMMANDS = {
    "/help", "/clear", "/settings", "/status", "/handoff",
    "/compile", "/build", "/launch", "/stop", "/tools",
    "/search", "/find",
    "/capture", "/camera", "/listen",
    "/theme", "/ideas",
}

_RE_VOICE = re.compile(r"\[VOICE\](.*?)\[/VOICE\]", re.DOTALL)
_RE_ACTION = re.compile(r"\[ACTION:(\w+)\](.*?)\[/ACTION\]", re.DOTALL)
_RE_STRIP_ACTION = re.compile(r"\[ACTION:\w+\].*?\[/ACTION\]", re.DOTALL)
_RE_STRIP_VOICE_TAGS = re.compile(r"\[/?VOICE\]", re.DOTALL)
_RE_CODE_FENCE = re.compile(r"```")
_RE_BULLET = re.compile(r"^\s*[-*]\s", re.MULTILINE)

_FALLBACK_MAX_LEN = 400


@dataclass(frozen=True)
class ChainContext:
    """Intent threading through action chain steps."""

    original_intent: str = ""
    chain_position: int = 0
    max_depth: int = 5
    accumulated_results: tuple = ()


@dataclass(frozen=True)
class ActionResult:
    """Result from _execute_action. Tracks pending async work."""

    success: bool = True
    message: str = ""
    pending: bool = False    # async work spawned, not yet complete

    @property
    def chain_text(self) -> str:
        if self.pending:
            return f"[PENDING] {self.message}"
        return self.message


def _extract_first_sentence(text: str) -> Optional[str]:
    """Extract first complete sentence from accumulated text.

    Returns the sentence if found, None if still accumulating.
    """
    for i, ch in enumerate(text):
        if ch in ".!?" and i > 10:
            rest = text[i + 1:]
            if not rest or rest[0] in " \n\r":
                return text[: i + 1].strip()
    return None


class StreamingVoiceTracker:
    """Track [VOICE] tags and extract sentences during token streaming.

    Fed token-by-token during LLM streaming. Detects [VOICE]...[/VOICE]
    blocks and extracts complete sentences for immediate TTS playback.
    """

    def __init__(self):
        self._raw: str = ""           # All tokens accumulated
        self._voice_buf: str = ""     # Text inside current [VOICE] block
        self._in_voice: bool = False
        self._voice_found: bool = False
        self._spoken_len: int = 0     # Chars already spoken from _voice_buf
        self._scan_pos: int = 0       # Position in _raw to search for next [VOICE]

    def feed(self, token: str) -> Optional[str]:
        """Feed a token. Returns a speakable sentence, or None."""
        self._raw += token

        if not self._in_voice:
            # Check if we've accumulated a [VOICE] tag (from scan position)
            idx = self._raw.find("[VOICE]", self._scan_pos)
            if idx != -1:
                self._in_voice = True
                self._voice_found = True
                # Grab any text after the tag
                after = self._raw[idx + 7:]
                self._voice_buf = after
                self._spoken_len = 0
                # Fall through to sentence extraction below
            else:
                # Update scan position to avoid rescanning (keep 6 chars for partial tag)
                self._scan_pos = max(0, len(self._raw) - 6)
                return None

        else:
            # We're inside a [VOICE] block — accumulate
            self._voice_buf += token

        # Check for closing tag
        close_idx = self._voice_buf.find("[/VOICE]")
        if close_idx != -1:
            # Block closed — return unspoken portion before close tag
            self._in_voice = False
            unspoken = self._voice_buf[self._spoken_len:close_idx].strip()
            self._spoken_len = close_idx + 8  # past [/VOICE]
            # Advance scan position past this block for next [VOICE] search
            self._scan_pos = len(self._raw)
            return unspoken if unspoken else None

        # Try to extract a complete sentence from unspoken portion
        unspoken = self._voice_buf[self._spoken_len:]
        sentence = _extract_first_sentence(unspoken)
        if sentence:
            self._spoken_len += len(sentence)
            # Skip whitespace after spoken sentence
            while (self._spoken_len < len(self._voice_buf)
                   and self._voice_buf[self._spoken_len] in " \n\r"):
                self._spoken_len += 1
            return sentence

        return None

    def finish(self) -> Optional[str]:
        """Stream complete. Returns unspoken tail, or None."""
        if not self._voice_found:
            return None
        # If still inside an unclosed [VOICE] block, grab remaining
        if self._in_voice:
            unspoken = self._voice_buf[self._spoken_len:].strip()
            if unspoken:
                self._spoken_len = len(self._voice_buf)
                return unspoken
        # If block was closed, check for any unspoken tail before close
        remaining = self._voice_buf[self._spoken_len:].strip()
        # Strip any leftover tags
        remaining = remaining.replace("[/VOICE]", "").strip()
        if remaining:
            self._spoken_len = len(self._voice_buf)
            return remaining
        return None

    @property
    def spoke_anything(self) -> bool:
        return self._spoken_len > 0

    @property
    def voice_detected(self) -> bool:
        return self._voice_found


def parse_response(raw: str) -> Dict[str, Optional[str]]:
    """Parse LLM response envelope for voice and action markers.

    Returns dict with keys:
        display  — clean text with all tags stripped
        voice    — text to speak (or None)
        action   — action name (compile/build/tools/status) or None
        action_arg — argument for the action (or "")
    """
    if not raw:
        return {"display": "", "voice": None, "action": None, "action_arg": ""}

    # Extract voice content
    voice_match = _RE_VOICE.search(raw)
    voice = voice_match.group(1).strip() if voice_match else None

    # Extract action
    action_match = _RE_ACTION.search(raw)
    action = action_match.group(1).strip() if action_match else None
    action_arg = action_match.group(2).strip() if action_match else ""

    # Strip action blocks (entire [ACTION:xxx]...[/ACTION]) and voice tags for display
    display = _RE_STRIP_ACTION.sub("", raw)
    display = _RE_STRIP_VOICE_TAGS.sub("", display).strip()
    # Collapse multiple blank lines left by tag removal
    display = re.sub(r"\n{3,}", "\n\n", display)

    # Fallback heuristic: speak short, plain responses
    if voice is None and display:
        has_code = _RE_CODE_FENCE.search(display)
        has_bullets = _RE_BULLET.search(display)
        if len(display) <= _FALLBACK_MAX_LEN and not has_code and not has_bullets:
            voice = display

    return {
        "display": display,
        "voice": voice,
        "action": action,
        "action_arg": action_arg,
    }


class ChatScreen(Screen):
    """Main chat interface with Mother."""

    BINDINGS = [
        ("ctrl+q", "app.quit", "Quit"),
        ("ctrl+comma", "app.open_settings", "Settings"),
        ("f8", "listen", "Listen"),
    ]

    def __init__(self, config: Optional[MotherConfig] = None, config_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or MotherConfig()
        self._config_path = config_path
        self._store: Optional[ConversationStore] = None
        self._bridge: Optional[EngineBridge] = None
        self._voice: Optional[VoiceBridge] = None
        self._screen_bridge: Optional[ScreenCaptureBridge] = None
        self._microphone_bridge: Optional[MicrophoneBridge] = None
        self._camera_bridge: Optional[CameraBridge] = None
        self._pending_screenshot: Optional[str] = None
        self._pending_permission: Optional[str] = None  # "microphone" or "camera"
        self._chatting = False
        self._unmounted = False
        self._compilation_count = 0
        self._compile_running = False
        self._last_compile_result = None
        self._tool_count = 0
        # Paradigm shift tracking (#185)
        self._compilation_scores: list = []
        self._failure_reasons: list = []
        # Business viability tracking (#111)
        self._runway_months: float = 0.0
        self._burn_rate: float = 0.0
        # Senses state
        self._sense_memory: Optional[SenseMemory] = None
        self._current_senses: Optional[SenseVector] = None
        self._current_posture: Optional[Posture] = None
        self._session_start_time: float = 0.0
        self._session_error_count: int = 0
        # Perception state
        self._perception: Optional[PerceptionEngine] = None
        self._perception_queue: Optional[asyncio.Queue] = None
        self._perception_consumer_task: Optional[asyncio.Task] = None
        self._pending_perception_screen: Optional[tuple] = None  # (b64, media_type)
        self._pending_perception_camera: Optional[tuple] = None  # (b64, media_type)
        self._perception_event_count: int = 0
        self._screen_change_count: int = 0
        # Multimodal cognitive infrastructure
        self._env_model = None       # Optional[EnvironmentModel]
        self._perception_fusion = None  # Optional[PerceptionFusion]
        self._actuator_store = None  # Optional[ActuatorStore]
        self._modality_profiles = None  # Optional[dict[str, ModalityProfile]]
        self._modality_budgets = None  # Optional[dict[str, ModalityBudget]]
        self._modality_detector = None  # Optional[ModalityPatternDetector]
        self._modality_interaction_count = 0
        self._trust_snapshot = None  # Optional[TrustSnapshot]
        self._experience_memory = None  # Optional[ExperienceMemory] — sentience chamber
        # Launcher state
        self._last_project_path: Optional[str] = None
        self._last_entry_point: Optional[str] = None
        self._launcher_poll_task: Optional[asyncio.Task] = None
        # Voice queue — serializes speech so sentences don't overlap
        self._voice_queue: asyncio.Queue = asyncio.Queue()
        self._voice_consumer_task: Optional[asyncio.Task] = None
        self._voice_prefetch: Optional[tuple] = None  # (text, asyncio.Task)
        # Duplex voice — real-time conversational AI
        self._duplex_session = None   # Optional[DuplexVoiceSession]
        self._llm_server = None       # Optional[LocalLLMServer]
        self._duplex_event_queue: asyncio.Queue = asyncio.Queue()
        self._duplex_consumer_task: Optional[asyncio.Task] = None
        self._duplex_active: bool = False
        # Interrupt state — set when user speaks/types during streaming
        self._interrupt_requested: bool = False
        self._interrupt_text: Optional[str] = None
        # Relationship insight (loaded from cache, recomputed in prefetch)
        self._relationship_insight: Optional[RelationshipInsight] = None
        # Context cache (populated by _prefetch_context worker)
        self._context_cache: Dict = {}
        # Neurologis Automatica
        self._temporal_engine = TemporalEngine()
        self._attention_filter = AttentionFilter()
        self._recall_engine: Optional[RecallEngine] = None
        self._last_user_message_time: float = 0.0
        self._memory_queries: int = 0
        self._memory_hits: int = 0
        self._pending_recall_block: str = ""
        # Operational awareness (Phase B)
        self._journal: Optional[BuildJournal] = None
        self._session_error_classifications: List[ErrorClassification] = []
        self._journal_summary_cache: Dict = {}
        self._health_failure_count: int = 0
        # Self-build state
        self._last_self_build_desc: Optional[str] = None
        # Daemon mode — autonomous self-improvement loop
        self._daemon = None  # Optional[DaemonMode]
        # Body map — structural self-knowledge
        self._codebase_topology = None  # Optional[CodebaseTopology]
        self._last_build_delta = None   # Optional[BuildDelta]
        # Appendage spawning
        self._appendage_store = None  # Optional[AppendageStore], lazy init
        # WhatsApp webhook state
        self._whatsapp_bridge: Optional[any] = None
        self._whatsapp_server_task: Optional[asyncio.Task] = None
        self._whatsapp_incoming: queue.Queue = queue.Queue()
        # Autonomic operating mode
        self._autonomous_tick_task: Optional[asyncio.Task] = None
        self._autonomous_working: bool = False
        self._autonomous_session_cost: float = 0.0
        self._autonomous_actions_count: int = 0
        self._last_chain_result: Optional[str] = None
        self._working_memory_summary: str = ""
        self._last_ideas_surfaced: float = 0.0
        # Dialogue initiative (impulse system)
        self._impulse_session_cost: float = 0.0
        self._impulse_actions_count: int = 0
        self._last_impulse_time: float = 0.0
        # Inner dialogue / metabolism
        self._metabolism_session_cost: float = 0.0
        self._metabolism_thoughts_count: int = 0
        self._metabolism_recent_thoughts: list = []
        self._metabolism_mode: str = "active"
        self._last_metabolism_time: float = 0.0
        self._deep_think_subject: Optional[str] = None
        self._deep_think_set_time: float = 0.0
        self._thought_journal = None  # ThoughtJournal, initialized in on_mount
        # Output routing
        self._whatsapp_messages_today: int = 0
        self._routing_day: int = 0  # Track day for daily reset
        # Panel server
        self._panel_server_thread: Optional[threading.Thread] = None
        # Self-test result (populated on boot)
        self._self_test_result: Optional[Dict] = None
        # Learning context cache (populated from journal patterns)
        self._learning_patterns = None  # Optional[JournalPatterns]
        # Mother-generated goal count this session
        self._mother_generated_goals: int = 0
        # Autonomous tick counter for periodic maintenance
        self._autonomous_tick_count: int = 0
        # Quiet autonomous mode: log background work instead of flooding chat
        self._autonomous_log: list = []  # Recent background work summaries (max 50)
        self._autonomous_log_max: int = 50
        # World grid — persistent world model (grid-as-mind)
        self._world_grid = None  # Optional[Grid], loaded at startup
        self._world_grid_recently_filled: set = set()  # cooldown tracking
        self._world_grid_last_save: float = 0.0  # throttle persistence
        self._last_reactive_impulse: float = 0.0  # debounce for reactive perception
        self._autonomous_outcome_history: list = []  # last N outcomes for convergence detection
        self._world_grid_nav_recommendation: Optional[str] = None  # LLM action from navigator
        # Pending idea being self-built (for lifecycle tracking)
        self._pending_idea_id: Optional[int] = None
        # Draft-first: pending proposal from ASK stance awaiting approval
        self._pending_proposal: Optional[Dict] = None
        # Post-compile artifacts (#143 visual-first, #127 client-facing, #134 meeting-prep)
        self._last_diagram: Optional[str] = None
        self._last_client_brief = None   # Optional[ClientBrief]
        self._last_meeting_briefing = None  # Optional[Briefing]
        # Runtime bootstrap (#13 substrate, #135 ghostwriter, #144 brand)
        self._substrate = None           # Optional[SubstrateCapabilities]
        self._voice_signature = None     # Optional[VoiceSignature]
        self._voice_persona = None       # Optional[VoicePersona]
        self._brand_profile = None       # Optional[BrandProfile]
        self._brand_prompt = None        # Optional[str]
        # Analysis persistence (business cognition, compliance, etc.)
        self._last_analysis_cache: Dict[str, str] = {}

    def _safe_query(self, widget_type):
        """Query a widget, returning None if screen is unmounted."""
        if self._unmounted:
            return None
        try:
            return self.query_one(widget_type)
        except Exception:
            return None

    def on_screen_resume(self) -> None:
        """Reload config when returning from settings."""
        config_path = self._config_path or getattr(self.app, "_config_path", None)
        new_config = load_config(config_path)
        old_config = self._config
        self._config = new_config

        # Update display names
        chat_area = self._safe_query(ChatArea)
        if chat_area:
            chat_area.instance_name = new_config.name
            chat_area.user_name = getattr(new_config, "user_name", "User")
        status = self._safe_query(StatusBar)
        if status:
            status.instance_name = new_config.name
            status.provider = new_config.provider
            status.model = new_config.get_model()

        # Hot-start perception modalities that changed
        self._hot_reload_modalities(old_config, new_config)

        # Update status bar indicators
        self._update_modality_indicators()

    def _hot_reload_modalities(self, old: MotherConfig, new: MotherConfig) -> None:
        """Start or stop modality bridges based on config changes."""
        openai_key = new.api_keys.get("openai")

        # Voice
        if new.voice_enabled and not old.voice_enabled:
            if is_voice_available():
                el_key = new.api_keys.get("elevenlabs")
                if el_key:
                    self._voice = VoiceBridge(
                        api_key=el_key,
                        voice_id=new.voice_id,
                        model_id=new.voice_model,
                    )
        elif not new.voice_enabled and old.voice_enabled:
            self._voice = None

        # Screen capture
        if new.screen_capture_enabled and not old.screen_capture_enabled:
            if is_screen_capture_available():
                self._screen_bridge = ScreenCaptureBridge(enabled=True)
        elif not new.screen_capture_enabled and old.screen_capture_enabled:
            self._screen_bridge = None

        # Microphone
        if new.microphone_enabled and not old.microphone_enabled:
            if is_microphone_available() and openai_key:
                self._microphone_bridge = MicrophoneBridge(
                    openai_api_key=openai_key,
                    enabled=True,
                )
        elif not new.microphone_enabled and old.microphone_enabled:
            self._microphone_bridge = None

        # Camera
        if new.camera_enabled and not old.camera_enabled:
            if is_camera_available():
                self._camera_bridge = CameraBridge(enabled=True)
        elif not new.camera_enabled and old.camera_enabled:
            self._camera_bridge = None

        # Duplex voice
        old_duplex = getattr(old, "voice_duplex_enabled", False)
        new_duplex = getattr(new, "voice_duplex_enabled", False)
        if new_duplex and not old_duplex:
            asyncio.get_event_loop().create_task(self._start_duplex_voice())
        elif not new_duplex and old_duplex:
            asyncio.get_event_loop().create_task(self._stop_duplex_voice())

        # Start perception engine if any modality newly enabled
        needs_perception = (
            new.screen_monitoring or new.camera_enabled or new.ambient_listening
        )
        had_perception = (
            old.screen_monitoring or old.camera_enabled or old.ambient_listening
        )
        if needs_perception and not had_perception and not self._perception:
            self._start_perception()

    def _update_modality_indicators(self) -> None:
        """Sync status bar modality indicators with active bridges."""
        status = self._safe_query(StatusBar)
        if status:
            status.voice_active = self._voice is not None
            status.screen_active = self._screen_bridge is not None
            status.mic_active = self._microphone_bridge is not None
            status.camera_active = self._camera_bridge is not None

    def compose(self) -> ComposeResult:
        yield StatusBar()
        with Container(id="chat-container"):
            yield ChatArea()
            yield PipelinePanel()
            yield InputArea()
        yield Footer()

    def on_mount(self) -> None:
        """Initialize bridge, store, senses, and show history-aware greeting."""
        import time as _time
        self._session_start_time = _time.time()
        self._last_user_message_time = _time.time()
        self._store = ConversationStore()
        # Initialize recall engine on same DB
        self._recall_engine = RecallEngine(self._store._path)
        # Initialize build journal on same DB
        self._journal = BuildJournal(self._store._path)

        # Restore last project path from journal
        try:
            recent = self._journal.recent(limit=5)
            for entry in recent:
                if entry.success and entry.project_path:
                    if Path(entry.project_path).is_dir():
                        self._last_project_path = entry.project_path
                        self._last_entry_point = "main.py"
                        break
        except Exception as e:
            logger.debug(f"Journal project path restore skipped: {e}")

        # Substrate detection — platform capabilities (#13)
        try:
            from mother.substrate import SubstrateDetector
            self._substrate = SubstrateDetector.detect()
            logger.info(f"Substrate: {self._substrate.platform}, spotlight={self._substrate.has_spotlight}")
        except Exception as e:
            self._substrate = None
            logger.debug(f"Substrate detection skipped: {e}")

        # Legacy guard — config migration (#18)
        try:
            from mother.legacy import LegacyGuard
            _config_path = Path.home() / ".motherlabs" / "mother.json"
            if _config_path.exists():
                _data, _version = LegacyGuard.read_config(_config_path)
                if LegacyGuard.needs_migration(_data):
                    _migration = LegacyGuard.migrate_file(_config_path)
                    if _migration.success:
                        logger.info(f"Config migrated: v{_migration.from_version}→v{_migration.to_version}")
        except Exception as e:
            logger.debug(f"Legacy migration skipped: {e}")

        # Inject config API keys into os.environ so native code engine
        # and CLI coding agents can find them (they check env vars directly).
        import os as _os_env
        _API_KEY_ENV_MAP = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "grok": "XAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        for _prov, _env_var in _API_KEY_ENV_MAP.items():
            _key_val = self._config.api_keys.get(_prov)
            if _key_val and not _os_env.environ.get(_env_var):
                _os_env.environ[_env_var] = _key_val

        api_key = self._config.api_keys.get(self._config.provider)
        openai_key = self._config.api_keys.get("openai")
        self._bridge = EngineBridge(
            provider=self._config.provider,
            model=self._config.get_model(),
            api_key=api_key,
            file_access=self._config.file_access,
            screen_capture_enabled=self._config.screen_capture_enabled,
            microphone_enabled=self._config.microphone_enabled,
            openai_api_key=openai_key,
            camera_enabled=self._config.camera_enabled,
            pipeline_mode=getattr(self._config, "pipeline_mode", "staged"),
            max_concurrent_appendages=getattr(self._config, "appendage_max_concurrent", 5),
            min_uses_to_solidify=getattr(self._config, "appendage_min_uses_to_solidify", 5),
        )
        # Tiered model routing: cheap model for chat, main model for compilation
        _chat_model = getattr(self._config, "chat_model", "")
        if _chat_model:
            self._bridge.set_chat_model(_chat_model)

        # Panel server (optional)
        if getattr(self._config, "panel_server_enabled", False):
            self._start_panel_server()

        # Chat area — dynamic names
        chat_area = self.query_one(ChatArea)
        chat_area.instance_name = self._config.name
        chat_area.user_name = getattr(self._config, "user_name", "User")

        # Status bar
        status = self.query_one(StatusBar)
        status.instance_name = self._config.name
        status.provider = self._config.provider
        status.model = self._config.get_model()

        # Voice bridge (optional)
        if self._config.voice_enabled and is_voice_available():
            el_key = self._config.api_keys.get("elevenlabs")
            if el_key:
                self._voice = VoiceBridge(
                    api_key=el_key,
                    voice_id=self._config.voice_id,
                    model_id=self._config.voice_model,
                )

        # Screen capture bridge (optional)
        if self._config.screen_capture_enabled and is_screen_capture_available():
            self._screen_bridge = ScreenCaptureBridge(enabled=True)

        # Microphone bridge (optional)
        if self._config.microphone_enabled and is_microphone_available() and openai_key:
            self._microphone_bridge = MicrophoneBridge(
                openai_api_key=openai_key,
                enabled=True,
            )

        # Camera bridge (optional)
        if self._config.camera_enabled and is_camera_available():
            self._camera_bridge = CameraBridge(enabled=True)

        # Wire voice ref for perception mute coordination
        if self._voice and self._bridge:
            self._bridge.set_voice_bridge_ref(self._voice)

        # Start continuous perception if any modality enabled
        if (self._config.screen_monitoring
                or self._config.camera_enabled
                or self._config.ambient_listening):
            self._start_perception()

        # Status bar modality indicators
        self._update_modality_indicators()

        # Duplex voice — real-time conversational AI (if enabled at startup)
        if getattr(self._config, "voice_duplex_enabled", False):
            self.call_later(self._start_duplex_voice_with_feedback)

        # Daemon mode — autonomous self-improvement loop
        if getattr(self._config, "claude_code_enabled", False):
            self._start_daemon()

        # Load sense memory from persistence
        raw = self._store.load_sense_memory()
        if raw:
            try:
                self._sense_memory = deserialize_memory(raw)
            except Exception:
                self._sense_memory = None

        # Load experience memory from persistence (sentience chamber)
        try:
            from mother.sentience import deserialize_experience_memory
            _exp_raw = self._store.load_experience_memory()
            if _exp_raw:
                self._experience_memory = deserialize_experience_memory(_exp_raw)
        except Exception as e:
            logger.debug(f"Experience memory load skipped: {e}")

        # Load cached relationship insight (instant, no async)
        cached = self._store.load_relationship_insights()
        if cached:
            try:
                import json as _json
                data = _json.loads(cached[0])
                self._relationship_insight = RelationshipInsight(**data)
            except Exception:
                self._relationship_insight = None

        # Compute initial senses
        self._update_senses()

        # Prefetch slow context data (corpus, tools, instance age)
        self.run_worker(self._prefetch_context(), exclusive=False)

        # Load history or show greeting
        history = self._store.get_history(limit=20)
        chat_area = self.query_one(ChatArea)
        if history:
            for msg in history:
                chat_area.add_message(msg.role, msg.content)
        else:
            # History-aware greeting — posture + relationship influences tone
            memory_summary = self._store.get_cross_session_summary()
            greeting = build_greeting(
                self._config,
                memory_summary=memory_summary,
                posture=self._current_posture,
                relationship_insight=self._relationship_insight,
            )
            chat_area.add_ai_message(greeting)
            self._store.add_message("assistant", greeting)
            self._speak(greeting)

        # WhatsApp webhook (if enabled)
        if self._config.whatsapp_webhook_enabled:
            self.run_worker(self._start_whatsapp_webhook(), exclusive=False)
            self.set_interval(0.3, self._poll_whatsapp)

        # Boot self-test (background — doesn't block mount)
        self.run_worker(self._boot_self_test(), exclusive=False)

        # Autonomous tick (if enabled)
        if getattr(self._config, "autonomous_enabled", False):
            self._start_autonomous_tick()

        # Seed goal — ensure the loop has something to work with on first launch
        if getattr(self._config, "autonomous_enabled", False) and self._store:
            try:
                from mother.goals import GoalStore
                goal_store = GoalStore(self._store._path)
                if goal_store.count_active() == 0:
                    goal_store.add(
                        "Greet the user and ask what they'd like to build",
                        source="system",
                        priority="normal",
                    )
                goal_store.close()
            except Exception as e:
                logger.debug(f"Seed goal init skipped: {e}")

        # Impulse tick — dialogue initiative (if enabled)
        if getattr(self._config, "dialogue_initiative_enabled", False):
            self._start_impulse_tick()

        # Metabolism tick — inner dialogue (if enabled)
        if getattr(self._config, "metabolism_enabled", False):
            self._start_metabolism_tick()
            # Initialize thought journal
            try:
                from mother.thought_journal import ThoughtJournal
                db_path = self._store._path if self._store else None
                if db_path:
                    self._thought_journal = ThoughtJournal(db_path)
            except Exception as e:
                logger.debug(f"Thought journal init skipped: {e}")

        # Focus input
        self.query_one(InputArea).focus_input()

    def _speak(self, text: str) -> None:
        """Queue text for sequential voice playback.

        Sentences are queued and played one at a time by _voice_consumer
        so they never overlap. Mutes perception during playback.
        """
        if self._voice:
            self._voice_queue.put_nowait(text)
            if self._voice_consumer_task is None or self._voice_consumer_task.done():
                self._voice_consumer_task = asyncio.get_event_loop().create_task(
                    self._voice_consumer()
                )

    async def _clear_voice_queue(self) -> None:
        """Drain voice queue, cancel prefetch, and stop current playback."""
        while not self._voice_queue.empty():
            try:
                self._voice_queue.get_nowait()
            except Exception:
                break
        if self._voice_prefetch is not None:
            _, task = self._voice_prefetch
            task.cancel()
            self._voice_prefetch = None
        if self._voice:
            await self._voice.stop()

    async def _voice_consumer(self) -> None:
        """Drain voice queue with pipelined synthesis.

        Pre-synthesizes the next sentence while playing the current one,
        eliminating inter-sentence API latency. Uses voice.synthesize()
        + voice.play() instead of voice.speak().
        """
        pace = self._current_posture.voice_pace if self._current_posture else None

        while not self._voice_queue.empty() or self._voice_prefetch is not None:
            if self._perception:
                self._perception.mute()
            try:
                if self._voice_prefetch is not None:
                    text, audio_task = self._voice_prefetch
                    self._voice_prefetch = None
                    audio_data = await audio_task
                else:
                    text = self._voice_queue.get_nowait()
                    audio_data = await self._voice.synthesize(text)

                # Kick off synthesis for next sentence while we play this one
                if not self._voice_queue.empty():
                    next_text = self._voice_queue.get_nowait()
                    self._voice_prefetch = (
                        next_text,
                        asyncio.ensure_future(self._voice.synthesize(next_text)),
                    )

                await self._voice.play(audio_data, playback_rate=pace)
            except Exception as e:
                logger.debug(f"Voice playback error: {e}")
                self._voice_prefetch = None
            finally:
                if self._perception:
                    self._perception.unmute()

    # ── Duplex voice ───────────────────────────────────────────────────

    async def _start_duplex_voice(self) -> None:
        """Start real-time duplex voice via ElevenLabs Conversational AI.

        1. Start local LLM server (OpenAI-compatible proxy)
        2. Create/reuse ElevenLabs agent with custom LLM pointing to local server
        3. Start duplex session (mic → ElevenLabs → LLM server → TTS → speaker)
        4. Mute perception ambient mic (ElevenLabs handles VAD)
        """
        try:
            from mother.llm_server import LocalLLMServer, is_llm_server_available
            from mother.voice_duplex import DuplexVoiceSession, is_duplex_available

            if not is_llm_server_available():
                logger.warning("aiohttp not available, duplex voice disabled")
                return
            if not is_duplex_available():
                logger.warning("elevenlabs/sounddevice not available, duplex voice disabled")
                return

            api_key = self._config.api_keys.get("elevenlabs", "")
            if not api_key:
                import os
                api_key = os.environ.get("ELEVENLABS_API_KEY", "")
            if not api_key:
                logger.warning("No ElevenLabs API key, duplex voice disabled")
                return

            # 1. Start local LLM server
            port = getattr(self._config, "voice_duplex_port", 11411)
            self._llm_server = LocalLLMServer(port=port)
            self._llm_server.set_handlers(
                system_prompt_fn=self._bridge.get_duplex_system_prompt,
                chat_stream_fn=self._bridge.stream_chat_for_duplex,
            )
            await self._llm_server.start()

            # 2. Create duplex session
            server_url = self._llm_server.completions_url
            agent_id = getattr(self._config, "voice_duplex_agent_id", "")
            language = getattr(self._config, "voice_duplex_language", "en")

            self._duplex_session = DuplexVoiceSession(
                api_key=api_key,
                server_url=server_url,
                event_queue=self._duplex_event_queue,
                agent_id=agent_id,
                voice_id=self._config.voice_id,
                language=language,
            )

            # 3. Ensure agent exists (creates or validates cached)
            new_agent_id = await self._duplex_session.ensure_agent()
            if new_agent_id != agent_id:
                # Cache the agent_id for reuse
                self._config.voice_duplex_agent_id = new_agent_id
                try:
                    from mother.config import save_config
                    save_config(self._config)
                except Exception as e:
                    logger.debug(f"Config save failed: {e}")

            # 4. Set initial system prompt for voice
            self._update_duplex_system_prompt()

            # 5. Start the duplex session
            await self._duplex_session.start_session()

            # 6. Mute perception ambient mic (ElevenLabs handles VAD)
            if self._perception:
                self._perception.mute()

            # 7. Start event consumer
            self._duplex_consumer_task = asyncio.create_task(self._duplex_consumer())
            self._duplex_active = True

            logger.info("Duplex voice started")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message("Real-time voice active. Speak naturally.")

        except Exception as e:
            logger.error(f"Failed to start duplex voice: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Duplex voice failed: {e}")
            await self._stop_duplex_voice()

    async def _start_duplex_voice_with_feedback(self) -> None:
        """Start duplex voice on mount, with visible user feedback on failure."""
        try:
            await self._start_duplex_voice()
        except Exception as e:
            logger.error(f"Duplex voice auto-start failed: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Real-time voice couldn't start: {e}")

    async def _stop_duplex_voice(self) -> None:
        """Stop duplex voice and fall back to turn-based mode."""
        self._duplex_active = False

        if self._duplex_consumer_task:
            self._duplex_consumer_task.cancel()
            self._duplex_consumer_task = None

        if self._duplex_session:
            try:
                await self._duplex_session.end_session()
            except Exception as e:
                logger.debug(f"Duplex session cleanup: {e}")
            self._duplex_session = None

        if self._llm_server:
            try:
                await self._llm_server.stop()
            except Exception as e:
                logger.debug(f"LLM server cleanup: {e}")
            self._llm_server = None

        # Resume perception ambient mic
        if self._perception:
            self._perception.unmute()

        logger.info("Duplex voice stopped")

    async def _duplex_consumer(self) -> None:
        """Background loop processing DuplexEvents from the voice session.

        Routes transcripts to chat display and conversation history.
        """
        while True:
            try:
                event = await self._duplex_event_queue.get()
            except asyncio.CancelledError:
                return

            try:
                chat_area = self._safe_query(ChatArea)

                if event.type == "user_spoke" and event.text.strip():
                    # Display and store user speech
                    if chat_area:
                        chat_area.add_user_message(event.text)
                    if self._store:
                        self._store.add_message("user", event.text)
                    try:
                        self._bridge.index_message_for_memory("user", event.text)
                    except Exception as e:
                        logger.debug(f"Memory index failed: {e}")

                elif event.type == "agent_spoke" and event.text.strip():
                    # Display and store agent response
                    if chat_area:
                        chat_area.add_ai_message(event.text)
                    if self._store:
                        self._store.add_message("assistant", event.text)
                    try:
                        self._bridge.index_message_for_memory("assistant", event.text)
                    except Exception as e:
                        logger.debug(f"Memory index failed: {e}")

                elif event.type == "session_ended":
                    logger.info("Duplex session ended by server, falling back")
                    if chat_area:
                        chat_area.add_ai_message("[Voice session ended. Returning to text.]")
                    await self._stop_duplex_voice()
                    return

                elif event.type == "error":
                    logger.warning(f"Duplex error: {event.text}")

            except Exception as e:
                logger.debug(f"Duplex consumer error: {e}")

    def _update_duplex_system_prompt(self) -> None:
        """Build and set the system prompt for duplex voice mode.

        Voice-specific: persona + senses + memory context, but stripped of
        text formatting instructions (no markdown, no ACTION blocks).
        """
        try:
            parts = []

            # Persona base (voice-adapted)
            try:
                from mother.persona import build_system_prompt
                prompt = build_system_prompt(
                    config=self._config,
                    senses=self._current_senses,
                    posture=self._current_posture,
                )
                parts.append(prompt)
            except Exception:
                parts.append("You are Mother, a warm and direct AI companion.")

            # Voice instruction
            parts.append(
                "\n[VOICE MODE] You are speaking out loud in real-time conversation. "
                "Keep responses concise (1-3 sentences). No markdown, no code blocks, "
                "no ACTION tags. Speak naturally as in conversation."
            )

            self._bridge.set_duplex_system_prompt("\n\n".join(parts))
        except Exception as e:
            logger.debug(f"Duplex prompt update failed: {e}")

    def _start_perception(self) -> None:
        """Initialize and start the continuous perception engine."""
        self._perception_queue = asyncio.Queue(maxsize=100)

        # Initialize multimodal cognitive infrastructure
        try:
            from mother.environment_model import create_model
            self._env_model = create_model()
        except Exception as e:
            logger.debug(f"Environment model init skipped: {e}")
        try:
            from mother.perception_fusion import PerceptionFusion
            self._perception_fusion = PerceptionFusion()
        except Exception as e:
            logger.debug(f"Perception fusion init skipped: {e}")
        try:
            from mother.actuator_receipt import ActuatorStore
            self._actuator_store = ActuatorStore()
        except Exception as e:
            logger.debug(f"Actuator store init skipped: {e}")
        try:
            from mother.modality_profile import default_profiles, allocate_budget
            self._modality_profiles = default_profiles()
            self._modality_budgets = allocate_budget(
                self._modality_profiles,
                total_hourly_budget=getattr(self._config, 'perception_budget', 1.0),
            )
        except Exception as e:
            logger.debug(f"Modality profiles init skipped: {e}")
        try:
            from mother.modality_learning import ModalityPatternDetector, load_modality_insights
            self._modality_detector = ModalityPatternDetector(min_samples=10)
            # Load prior learning insights for context injection
            _prior_insights = load_modality_insights()
            if _prior_insights.insights:
                logger.info(f"Loaded {len(_prior_insights.insights)} prior modality insights")
        except Exception as e:
            logger.debug(f"Modality learning init skipped: {e}")
        try:
            from mother.trust_accumulator import load_trust_snapshot
            self._trust_snapshot = load_trust_snapshot()
            if self._trust_snapshot.total_compilations > 0:
                logger.info(
                    f"Loaded trust snapshot: {self._trust_snapshot.total_compilations} compilations, "
                    f"rolling success {self._trust_snapshot.rolling_success_rate:.0%}"
                )
        except Exception as e:
            logger.debug(f"Trust snapshot load skipped: {e}")

        # Build config from mother config
        # camera_enabled=True means hardware is available for on-demand capture,
        # but camera polling loop should NOT auto-start — camera is on-demand only.
        # Use a separate config flag (camera_monitoring) for auto-polling, defaulting to False.
        _camera_polling = getattr(self._config, "camera_monitoring", False)
        perc_config = PerceptionConfig(
            screen_enabled=self._config.screen_monitoring,
            camera_enabled=_camera_polling,
            ambient_mic_enabled=self._config.ambient_listening,
            screen_poll_seconds=float(self._config.screen_poll_interval),
            camera_poll_seconds=float(self._config.camera_poll_interval),
            budget_per_hour=self._config.perception_budget,
        )

        # Build injected callables
        screen_fn = None
        if self._screen_bridge:
            screen_fn = self._screen_bridge.capture_screen
        elif self._config.screen_monitoring and is_screen_capture_available():
            # Create a screen bridge if monitoring is on but command-capture was off
            self._screen_bridge = ScreenCaptureBridge(enabled=True)
            screen_fn = self._screen_bridge.capture_screen

        camera_fn = None
        if self._camera_bridge:
            camera_fn = self._camera_bridge.capture_frame

        mic_record_fn = None
        mic_transcribe_fn = None
        if self._microphone_bridge and self._microphone_bridge.enabled:
            async def _mic_record(duration: float):
                return await asyncio.to_thread(
                    self._microphone_bridge._record_sync, duration
                )

            async def _mic_transcribe(audio_bytes: bytes):
                return await asyncio.to_thread(
                    self._microphone_bridge._transcribe_sync, audio_bytes
                )

            mic_record_fn = _mic_record
            mic_transcribe_fn = _mic_transcribe

        def _is_voice_playing():
            if self._voice:
                return getattr(self._voice, "is_playing", False)
            return False

        self._perception = PerceptionEngine(
            config=perc_config,
            event_queue=self._perception_queue,
            screen_fn=screen_fn,
            camera_fn=camera_fn,
            mic_record_fn=mic_record_fn,
            mic_transcribe_fn=mic_transcribe_fn,
            is_voice_playing_fn=_is_voice_playing,
        )
        self._perception.start()
        self._perception_consumer_task = asyncio.create_task(self._perception_consumer())

    async def _perception_consumer(self) -> None:
        """Background loop processing perception events from the queue."""
        while True:
            try:
                event = await self._perception_queue.get()
                self._perception_event_count += 1

                # Attention filter: evaluate significance before processing
                import time as _time
                event_type_map = {
                    PerceptionEventType.SPEECH_DETECTED: "speech",
                    PerceptionEventType.SCREEN_CHANGED: "screen",
                    PerceptionEventType.CAMERA_FRAME: "camera",
                }
                att_type = event_type_map.get(event.event_type, "unknown")
                elapsed = (_time.time() - self._attention_filter._last_attended_time
                           if self._attention_filter._last_attended_time > 0 else 60.0)
                att_score = self._attention_filter.evaluate(
                    event_type=att_type,
                    payload_size=len(event.payload) if event.payload else 0,
                    elapsed_since_last_event=elapsed,
                    senses_attentiveness=(
                        self._current_senses.attentiveness
                        if self._current_senses else 0.5
                    ),
                    conversation_active=self._chatting,
                )
                if not att_score.should_attend:
                    continue

                # --- Modality profile gate ---
                try:
                    if self._modality_profiles is not None:
                        from mother.modality_profile import should_process, update_budget_after_event
                        _profile = self._modality_profiles.get(att_type)
                        _budget = self._modality_budgets.get(att_type) if self._modality_budgets else None
                        if _profile and not should_process(_profile, att_score.significance, _budget):
                            continue  # Below profile threshold or over budget
                        if _profile and _budget and self._modality_budgets is not None:
                            self._modality_budgets[att_type] = update_budget_after_event(
                                _budget, _profile.cost_per_event,
                            )
                except Exception as e:
                    logger.debug(f"Modality profile gate skipped: {e}")

                # --- Multimodal infrastructure: observe + fuse ---
                try:
                    if self._env_model is not None:
                        summary = (event.payload[:80] if event.payload else att_type)
                        import hashlib as _hashlib
                        raw_hash = _hashlib.md5(
                            (event.payload or "").encode()
                        ).hexdigest()
                        self._env_model.observe(
                            att_type, summary,
                            confidence=att_score.significance,
                            raw_hash=raw_hash,
                            now=_time.time(),
                        )
                    if self._perception_fusion is not None:
                        from mother.perception_fusion import FusionEvent as _FE
                        self._perception_fusion.ingest(_FE(
                            event_type=event.event_type.value,
                            timestamp=_time.time(),
                            summary=(event.payload[:80] if event.payload else ""),
                        ))
                except Exception as e:
                    logger.debug(f"Multimodal observe/fuse skipped: {e}")

                # --- World grid: fill perception cells ---
                try:
                    if self._world_grid is not None and self._bridge is not None:
                        from kernel.perception_bridge import perception_to_fill
                        summary = (event.payload[:80] if event.payload else att_type)
                        fill_params = perception_to_fill(
                            att_type, summary,
                            att_score.significance, _time.time(),
                        )
                        self._bridge.fill_world_cell(self._world_grid, **fill_params)
                        self._world_grid_recently_filled.add(fill_params["postcode_key"])
                        # Reactive perception: high-significance + idle → fast-path impulse
                        if (
                            att_score.significance >= 0.7
                            and not self._chatting
                            and not self._autonomous_working
                            and getattr(self._config, "autonomous_enabled", False)
                        ):
                            _since_last_reactive = (
                                _time.time() - getattr(self, "_last_reactive_impulse", 0.0)
                            )
                            if _since_last_reactive >= 30.0:  # 30s debounce
                                self._last_reactive_impulse = _time.time()
                                # Schedule fast-path OBSERVE with 2s delay
                                asyncio.get_event_loop().call_later(
                                    2.0, self._reactive_observe_tick
                                )
                                logger.debug(
                                    f"Reactive perception: scheduling fast observe "
                                    f"(significance={att_score.significance:.2f})"
                                )
                        # Also fill fusion signals into world grid
                        if self._perception_fusion is not None:
                            from kernel.perception_bridge import fusion_signal_to_fill
                            signals = self._perception_fusion.detect(now=_time.time())
                            for sig in signals:
                                fp = fusion_signal_to_fill(
                                    sig.pattern, sig.confidence,
                                    sig.evidence, sig.detected_at,
                                )
                                self._bridge.fill_world_cell(self._world_grid, **fp)
                except Exception as e:
                    logger.debug(f"World grid perception fill skipped: {e}")

                if event.event_type == PerceptionEventType.SPEECH_DETECTED:
                    # Display heard text and route through chat
                    chat_area = self._safe_query(ChatArea)
                    if chat_area:
                        chat_area.add_user_message(f'[Heard] "{event.payload}"')

                        if self._chatting:
                            # Interrupt: stop voice, cancel stream, queue user's words
                            self._interrupt_requested = True
                            self._interrupt_text = event.payload
                            await self._clear_voice_queue()
                            if self._bridge:
                                self._bridge.cancel_chat_stream()
                        else:
                            self._handle_chat(event.payload)

                elif event.event_type == PerceptionEventType.SCREEN_CHANGED:
                    # Store as pending image — impulse tick will volunteer it
                    self._pending_perception_screen = (event.payload, event.media_type)
                    self._screen_change_count += 1

                    # High significance: react immediately via proactive perception
                    if (getattr(self._config, "autonomous_enabled", False)
                            and not self._chatting
                            and att_score.significance >= 0.7):
                        asyncio.create_task(self._proactive_perception(event))
                    # Medium significance: stored for impulse OBSERVE tick

                elif event.event_type == PerceptionEventType.CAMERA_FRAME:
                    # Store as pending image — impulse tick will volunteer it
                    self._pending_perception_camera = (event.payload, event.media_type)

                    # High significance: react immediately
                    if (getattr(self._config, "autonomous_enabled", False)
                            and not self._chatting
                            and att_score.significance >= 0.7):
                        asyncio.create_task(self._proactive_perception(event))
                    # Medium significance: stored for impulse OBSERVE tick

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"Perception consumer error: {e}")

    def _update_senses(self) -> None:
        """Gather observations and recompute senses + posture + memory."""
        import time as _time

        # Compile metrics
        compile_count = self._compilation_count
        compile_success = 0
        last_trust = 0.0
        result = self._last_compile_result
        if result is not None and getattr(result, "success", False):
            compile_success = compile_count  # Approximation: count = success if last was good
            verification = getattr(result, "verification", {}) or {}
            last_trust = verification.get("overall_score", 0.0) if isinstance(verification, dict) else 0.0
        elif result is not None:
            # Last compile failed — at least one was attempted
            compile_success = max(0, compile_count - 1)

        # Session/history from store
        total_sessions = 0
        total_messages = 0
        days_since_last: Optional[float] = None
        sessions_last_7 = 0
        messages_this_session = 0
        avg_msg_len = 0.0
        unique_topics = 0
        if self._store:
            summary = self._store.get_cross_session_summary()
            total_sessions = summary.get("total_sessions", 0)
            total_messages = summary.get("total_messages", 0)
            days_since_last = summary.get("days_since_last")
            unique_topics = len(summary.get("topics", []))
            messages_this_session = len(self._store.get_history(limit=500))
            avg_msg_len = self._store.get_average_user_message_length()
            # Sessions in last 7 days
            week_ago = _time.time() - 7 * 86400
            sessions_last_7 = self._store.get_sessions_since(week_ago)

        # Cost
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0

        # Duration
        duration = (_time.time() - self._session_start_time) / 60.0 if self._session_start_time > 0 else 0.0

        # Attention state
        att_state = self._attention_filter.state

        obs = SenseObservations(
            compile_count=compile_count,
            compile_success_count=compile_success,
            last_trust_score=last_trust,
            session_error_count=self._session_error_count,
            total_sessions=total_sessions,
            total_messages=total_messages,
            messages_this_session=messages_this_session,
            days_since_last_session=days_since_last,
            sessions_last_7_days=sessions_last_7,
            avg_user_message_length=avg_msg_len,
            unique_topic_count=unique_topics,
            session_cost=session_cost,
            cost_limit=self._config.cost_limit,
            session_duration_minutes=duration,
            perception_events_this_session=self._perception_event_count,
            perception_active=self._perception is not None and self._perception.running,
            screen_changes_detected=self._screen_change_count,
            # Neurologis Automatica
            idle_seconds=(_time.time() - self._last_user_message_time
                         if self._last_user_message_time > 0 else 0.0),
            conversation_tempo=0.0,  # computed by temporal engine, fed at chat_worker level
            attention_load=att_state.load,
            attention_events_attended=att_state.events_attended,
            memory_queries_this_session=self._memory_queries,
            memory_hits_this_session=self._memory_hits,
            # Operational awareness (Phase B)
            build_success_streak=(
                self._journal.get_summary().streak if self._journal else 0
            ),
            project_health_failures=self._health_failure_count,
            error_severity_sum=sum(
                c.severity for c in self._session_error_classifications
            ),
        )

        self._current_senses = compute_senses(obs, memory=self._sense_memory)
        self._current_posture = compute_posture(self._current_senses)
        self._sense_memory = update_memory(
            self._current_senses,
            previous=self._sense_memory,
            timestamp=_time.time(),
        )

    def _save_sense_memory(self) -> None:
        """Persist sense memory to SQLite."""
        if self._store and self._sense_memory:
            self._store.save_sense_memory(serialize_memory(self._sense_memory))
        # Persist experience memory alongside sense memory
        if self._store and self._experience_memory is not None:
            try:
                from mother.sentience import serialize_experience_memory
                self._store.save_experience_memory(serialize_experience_memory(self._experience_memory))
            except Exception as e:
                logger.debug(f"Experience memory save skipped: {e}")

    async def _boot_self_test(self) -> None:
        """Run test suite on boot to verify system integrity."""
        if not self._bridge:
            return
        try:
            result = await self._bridge.run_self_test()
            self._self_test_result = result
            logger.info(f"Self-test: {result}")
            if not result.get("passed", False):
                chat_area = self._safe_query(ChatArea)
                if chat_area:
                    chat_area.add_message(
                        "system",
                        f"Self-test: {result.get('summary', 'failed')}",
                    )
        except Exception as e:
            logger.debug(f"Self-test error: {e}")
            self._self_test_result = {"passed": False, "summary": f"error: {e}"}

    async def _prefetch_context(self) -> None:
        """Prefetch slow context data on mount (corpus, tools, instance age, rejections, relationship, projects)."""
        if not self._bridge:
            return
        try:
            db_path = self._store._path if self._store else None
            async def _empty_list():
                return []

            gathers = [
                self._bridge.get_corpus_summary(),
                self._bridge.get_tool_summary(),
                self._bridge.get_instance_age_days(),
                self._bridge.get_rejection_summary(),
                self._bridge.get_tool_details(),
                self._bridge.get_recent_projects(db_path) if db_path else _empty_list(),
                self._bridge.get_connected_peers(),
                self._bridge.get_provider_balances(self._config.api_keys),
            ]

            corpus, tools, age, rejections, tool_details, recent_projects, connected_peers, balances = await asyncio.gather(
                *gathers,
                return_exceptions=True,
            )
            self._context_cache["corpus"] = corpus if isinstance(corpus, dict) else {}
            self._context_cache["tools"] = tools if isinstance(tools, dict) else {}
            self._context_cache["instance_age_days"] = age if isinstance(age, int) else 0
            self._context_cache["rejections"] = rejections if isinstance(rejections, dict) else {}
            self._context_cache["connected_peers"] = connected_peers if isinstance(connected_peers, list) else []
            self._context_cache["provider_balances"] = balances if isinstance(balances, dict) else {}
            self._context_cache["tool_details"] = tool_details if isinstance(tool_details, list) else []
            self._context_cache["recent_projects"] = recent_projects if isinstance(recent_projects, list) else []
        except Exception as e:
            logger.debug(f"Context prefetch error: {e}")

        # Scan codebase topology (lazy, once)
        await self._scan_topology_once()

        # Auto-spawn solidified appendages
        await self._auto_spawn_solidified()

        # Auto-dissolve stale appendages
        await self._auto_dissolve_stale()

        # Recompute relationship insight if stale (>1 hour) or missing
        self._recompute_relationship_insight()

    def _recompute_relationship_insight(self) -> None:
        """Recompute relationship insight if cache is stale or missing."""
        import json as _json
        import time as _time

        if not self._store:
            return

        # Check staleness
        cached = self._store.load_relationship_insights()
        if cached:
            _, computed_at = cached
            if (_time.time() - computed_at) < 3600:
                return  # Fresh enough

        try:
            all_msgs = self._store.get_all_messages(limit=5000)
            sessions = self._store.list_sessions()

            # Parse sense memory
            sense_mem = None
            raw_sense = self._store.load_sense_memory()
            if raw_sense:
                try:
                    sense_mem = _json.loads(raw_sense)
                except Exception as e:
                    logger.debug(f"Sense memory parse skipped: {e}")

            # Use cached corpus summary
            corpus = self._context_cache.get("corpus", {})

            insight = extract_relationship_insights(
                all_messages=all_msgs,
                sessions=sessions,
                sense_memory=sense_mem,
                corpus_summary=corpus,
            )
            self._relationship_insight = insight

            # Cache it
            from dataclasses import asdict
            self._store.save_relationship_insights(
                _json.dumps(asdict(insight)),
                insight.computed_at,
            )
        except Exception as e:
            logger.debug(f"Relationship insight error: {e}")

    def _build_context_data(self, temporal_state=None) -> ContextData:
        """Assemble ContextData from cache + live state."""
        import time as _time

        # Memory / history
        total_sessions = 0
        total_messages = 0
        days_since_last = None
        recent_topics: list = []
        sessions_last_7 = 0
        messages_this_session = 0
        if self._store:
            summary = self._store.get_cross_session_summary()
            total_sessions = summary.get("total_sessions", 0)
            total_messages = summary.get("total_messages", 0)
            days_since_last = summary.get("days_since_last")
            recent_topics = summary.get("topics", [])
            messages_this_session = len(self._store.get_history(limit=500))
            week_ago = _time.time() - 7 * 86400
            sessions_last_7 = self._store.get_sessions_since(week_ago)

        # Session cost
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0

        # Corpus stats from cache
        corpus = self._context_cache.get("corpus", {})
        corpus_total = corpus.get("total_compilations", 0)
        corpus_success_rate = corpus.get("success_rate", 0.0)
        corpus_domains = corpus.get("domains", {})
        corpus_total_components = corpus.get("total_components", 0)
        corpus_avg_trust = corpus.get("avg_trust", 0.0)
        corpus_anti_pattern_count = corpus.get("anti_pattern_count", 0)
        corpus_constraint_count = corpus.get("constraint_count", 0)
        corpus_pattern_health = corpus.get("pattern_health", "")

        # Tool stats from cache
        tools = self._context_cache.get("tools", {})
        tool_count = tools.get("count", 0)
        tool_verified = tools.get("verified", 0)
        tool_domains = tools.get("domains", {})

        # Instance age from cache
        instance_age_days = self._context_cache.get("instance_age_days", 0)

        # Rejection data from cache
        rejections = self._context_cache.get("rejections", {})
        rejection_count = rejections.get("total", 0)

        # Tool names and recent projects from cache
        tool_details = self._context_cache.get("tool_details", [])
        tool_names = [t.get("name", "") for t in tool_details if t.get("name")]
        # Include active appendage names as tools
        tool_names.extend(self._get_active_appendage_names())
        recent_projects = self._context_cache.get("recent_projects", [])

        # Last compile extraction
        lc_desc = None
        lc_trust = None
        lc_components = None
        lc_weakest = None
        result = self._last_compile_result
        if result is not None and getattr(result, "success", False):
            blueprint = getattr(result, "blueprint", {}) or {}
            lc_desc = blueprint.get("description", blueprint.get("name", ""))
            verification = getattr(result, "verification", {}) or {}
            lc_trust = verification.get("overall_score", 0.0) if isinstance(verification, dict) else 0.0
            components = blueprint.get("components", [])
            lc_components = len(components) if isinstance(components, list) else 0
            # Find weakest dimension
            weakest = None
            weakest_score = 101.0
            for k, v in (verification if isinstance(verification, dict) else {}).items():
                if k == "overall_score":
                    continue
                score_val = None
                if isinstance(v, (int, float)):
                    score_val = float(v)
                elif isinstance(v, dict) and "score" in v:
                    score_val = float(v["score"])
                if score_val is not None and score_val < weakest_score:
                    weakest = k
                    weakest_score = score_val
            lc_weakest = weakest if weakest_score < 101.0 else None

        # Sense trajectory
        rapport_trend = 0.0
        confidence_trend = 0.0
        peak_confidence = 0.5
        peak_rapport = 0.0
        if self._sense_memory:
            rapport_trend = self._sense_memory.rapport_trend
            confidence_trend = self._sense_memory.confidence_trend
            peak_confidence = self._sense_memory.peak_confidence
            peak_rapport = self._sense_memory.peak_rapport

        # Relationship narrative
        rel_narrative = ""
        if self._relationship_insight:
            rel_narrative = synthesize_relationship_narrative(self._relationship_insight)

        # Journal summary (single fetch)
        j_total = 0
        j_streak = 0
        j_avg_trust = 0.0
        j_total_cost = 0.0
        if self._journal:
            try:
                j_summary = self._journal.get_summary()
                j_total = j_summary.total_compiles + j_summary.total_builds
                j_streak = j_summary.streak
                j_avg_trust = j_summary.avg_trust
                j_total_cost = j_summary.total_cost
            except Exception as e:
                logger.debug(f"Journal summary fetch skipped: {e}")

        # Journal dimension patterns (L2 operational)
        j_dim_trends = ""
        j_failure_patterns = ""
        if self._journal:
            try:
                from mother.journal_patterns import extract_patterns
                from dataclasses import asdict
                recent = self._journal.recent(limit=20)
                patterns = extract_patterns([asdict(e) for e in recent])
                j_dim_trends = patterns.trends_line
                j_failure_patterns = patterns.failure_line
            except Exception as e:
                logger.debug(f"Journal pattern extraction skipped: {e}")

        return ContextData(
            name=self._config.name,
            provider=self._config.provider,
            model=self._config.get_model(),
            platform="darwin",
            instance_age_days=instance_age_days,
            cap_file_access=self._config.file_access,
            cap_voice=self._voice is not None,
            cap_screen_capture=self._screen_bridge is not None,
            cap_microphone=self._microphone_bridge is not None,
            cap_camera=self._camera_bridge is not None,
            cap_perception=self._perception is not None and self._perception.running,
            cap_whatsapp=self._config.whatsapp_webhook_enabled and self._whatsapp_bridge is not None,
            corpus_total=corpus_total,
            corpus_success_rate=corpus_success_rate,
            corpus_domains=corpus_domains,
            corpus_total_components=corpus_total_components,
            corpus_avg_trust=corpus_avg_trust,
            corpus_anti_pattern_count=corpus_anti_pattern_count,
            corpus_constraint_count=corpus_constraint_count,
            corpus_pattern_health=corpus_pattern_health,
            rejection_count=rejection_count,
            tool_count=tool_count,
            tool_verified_count=tool_verified,
            tool_domains=tool_domains,
            tool_names=tool_names,
            recent_projects=recent_projects,
            total_sessions=total_sessions,
            total_messages=total_messages,
            days_since_last=days_since_last,
            sessions_last_7d=sessions_last_7,
            recent_topics=recent_topics,
            relationship_narrative=rel_narrative,
            rapport_trend=rapport_trend,
            confidence_trend=confidence_trend,
            peak_confidence=peak_confidence,
            peak_rapport=peak_rapport,
            session_messages=messages_this_session,
            session_cost=session_cost,
            session_cost_limit=self._config.cost_limit,
            session_compilations=self._compilation_count,
            session_errors=self._session_error_count,
            last_compile_desc=lc_desc or None,
            last_compile_trust=lc_trust,
            last_compile_components=lc_components,
            last_compile_weakest=lc_weakest,
            # Neurologis Automatica
            temporal_context=(
                self._format_temporal_context(temporal_state)
                if temporal_state else ""
            ),
            attention_load=self._attention_filter.state.load,
            recall_block=self._pending_recall_block,
            # Inner dialogue / metabolism
            inner_thoughts=self._format_metabolism_for_context(),
            # Operational awareness (Phase B)
            journal_total_builds=j_total,
            journal_success_streak=j_streak,
            journal_avg_trust=j_avg_trust,
            journal_total_cost=j_total_cost,
            error_summary=summarize_errors(self._session_error_classifications),
            journal_dimension_trends=j_dim_trends,
            journal_failure_patterns=j_failure_patterns,
            # Self-build awareness
            pending_idea_count=self._get_pending_idea_count(),
            last_self_build_desc=self._last_self_build_desc,
            # Peer networking
            connected_peers=self._context_cache.get("connected_peers", []),
            # Budget monitoring
            provider_balances=self._context_cache.get("provider_balances", {}),
            # Capabilities — grounded in real config
            cap_compile=self._bridge is not None,
            cap_build=self._bridge is not None,
            cap_claude_code=getattr(self._config, "claude_code_enabled", False),
            cap_autonomous=getattr(self._config, "autonomous_enabled", False),
            # Autonomic operating mode
            active_goals=self._get_active_goal_descriptions(),
            pending_action_result=self._last_chain_result or "",
            working_memory_summary=self._working_memory_summary,
            autonomous_working=self._autonomous_working,
            autonomous_session_cost=self._autonomous_session_cost,
            autonomous_actions_count=self._autonomous_actions_count,
            autonomous_budget=getattr(self._config, "autonomous_budget_per_session", 1.0),
            goal_details=self._get_goal_details(),
            # Perception config
            perception_poll_seconds=float(self._config.screen_poll_interval) if self._perception and self._perception.running else 0.0,
            perception_budget_hourly=self._config.perception_budget if self._perception and self._perception.running else 0.0,
            perception_modes=self._get_perception_modes(),
            # Body map — structural self-knowledge
            codebase_total_files=self._codebase_topology.total_files if self._codebase_topology else 0,
            codebase_total_lines=self._codebase_topology.total_lines if self._codebase_topology else 0,
            codebase_modules=self._codebase_topology.modules if self._codebase_topology else {},
            codebase_test_count=self._codebase_topology.total_tests if self._codebase_topology else 0,
            codebase_protected=self._codebase_topology.protected_files if self._codebase_topology else [],
            codebase_boundary=self._codebase_topology.boundary_rule if self._codebase_topology else "",
            last_build_files_changed=(
                self._last_build_delta.files_modified + self._last_build_delta.files_added
                if self._last_build_delta else 0
            ),
            last_build_lines_delta=(
                f"+{self._last_build_delta.lines_added}/-{self._last_build_delta.lines_removed}"
                if self._last_build_delta else ""
            ),
            last_build_modules_touched=self._last_build_delta.modules_touched if self._last_build_delta else [],
        )

    def _format_temporal_context(self, temporal_state) -> str:
        """Format temporal state as a context line."""
        parts = []
        if temporal_state.idle_seconds > 60:
            mins = int(temporal_state.idle_seconds / 60)
            parts.append(f"Idle {mins}m")
        if temporal_state.time_of_day:
            parts.append(f"{temporal_state.time_of_day.capitalize()} session")
        if temporal_state.conversation_tempo > 0:
            parts.append(f"Tempo: {temporal_state.conversation_tempo:.1f} msg/min")
        if temporal_state.is_typical_time:
            parts.append("Typical time for this user")
        return ". ".join(parts) + "." if parts else ""

    def _get_pending_idea_count(self) -> int:
        """Get count of pending ideas. Non-blocking, returns 0 on failure."""
        try:
            db_path = self._store._path if self._store else None
            if db_path:
                from mother.idea_journal import IdeaJournal
                journal = IdeaJournal(db_path)
                return journal.count_pending()
        except Exception as e:
            logger.debug(f"Pending idea count skipped: {e}")
        return 0

    async def _scan_topology_once(self) -> None:
        """Lazy-init: scan codebase topology once, cache result."""
        if self._codebase_topology is not None:
            return
        try:
            from mother.introspection import scan_topology
            repo_dir = str(Path(__file__).resolve().parent.parent)
            self._codebase_topology = await asyncio.get_event_loop().run_in_executor(
                None, scan_topology, repo_dir
            )
        except Exception as e:
            logger.debug(f"Topology scan skipped: {e}")

    async def _auto_spawn_solidified(self) -> None:
        """Auto-spawn solidified appendages on startup."""
        if not getattr(self._config, "appendage_enabled", False):
            return
        try:
            db_path = self._store._path if self._store else None
            if not db_path or not self._bridge:
                return
            from mother.appendage import AppendageStore
            store = AppendageStore(db_path)
            for spec in store.solidified():
                try:
                    await self._bridge.spawn_appendage(db_path, spec.appendage_id)
                    logger.debug(f"Auto-spawned solidified appendage: {spec.name}")
                except Exception as e:
                    logger.debug(f"Auto-spawn failed for {spec.name}: {e}")
        except Exception as e:
            logger.debug(f"Auto-spawn solidified error: {e}")

    async def _auto_dissolve_stale(self) -> None:
        """Dissolve appendages older than appendage_auto_dissolve_hours."""
        import time as _time
        try:
            db_path = self._store._path if self._store else None
            if not db_path or not self._bridge:
                return
            dissolve_hours = getattr(self._config, "appendage_auto_dissolve_hours", 48)
            cutoff = _time.time() - (dissolve_hours * 3600)
            from mother.appendage import AppendageStore
            store = AppendageStore(db_path)
            for spec in store.all():
                if spec.status in ("spawned", "active", "built") and spec.created_at < cutoff:
                    await self._bridge.dissolve_appendage(db_path, spec.appendage_id)
                    logger.debug(f"Auto-dissolved stale appendage: {spec.name}")
        except Exception as e:
            logger.debug(f"Auto-dissolve stale error: {e}")

    def _start_panel_server(self) -> None:
        """Start the panel server in a daemon thread."""
        try:
            from mother.panel_server import create_app
            import uvicorn

            port = getattr(self._config, "panel_server_port", 7770)
            app = create_app(bridge=self._bridge)

            def _run():
                uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

            t = threading.Thread(target=_run, daemon=True, name="panel-server")
            t.start()
            self._panel_server_thread = t
            logger.info(f"Panel server started on 127.0.0.1:{port}")
        except Exception as e:
            logger.debug(f"Panel server start error: {e}")

    async def _surface_pending_ideas(self) -> None:
        """Surface pending ideas during autonomous idle (30 min cooldown)."""
        import time as _time
        now = _time.time()
        if now - self._last_ideas_surfaced < 1800:
            return
        try:
            db_path = self._store._path if self._store else None
            if not db_path or not self._bridge:
                return
            count = await self._bridge.count_pending_ideas(db_path)
            if count <= 0:
                return
            ideas = await self._bridge.get_pending_ideas(db_path, limit=3)
            if not ideas:
                return
            self._last_ideas_surfaced = now
            lines = [f"{count} idea{'s' if count != 1 else ''} waiting for review:"]
            for idea in ideas:
                lines.append(f"  - {idea['description'][:80]}")
            chat_area = self._safe_query(ChatArea)
            if chat_area:
                chat_area.add_ai_message("[idle] " + "\n".join(lines))
        except Exception as e:
            logger.debug(f"Surface ideas error: {e}")

    # --- Autonomic: action chaining, goals, autonomous tick ---

    def _execute_action(self, parsed: Dict) -> Optional[ActionResult]:
        """Route a parsed action and return ActionResult for chaining.

        Returns ActionResult for chainable actions. Pending=True for async work
        (compile, build, search). Returns None for terminal actions or no action.
        """
        action = parsed.get("action")
        arg = parsed.get("action_arg", "")

        if not action or action == "done":
            return None

        # --- Async actions (pending=True) ---
        if action == "compile" and arg:
            self._run_compile(arg)
            return ActionResult(message=f"Compilation started for: {arg}", pending=True)
        if action == "context" and arg:
            self._run_context_compile(arg)
            return ActionResult(message=f"Context compilation started for: {arg}", pending=True)
        if action == "explore" and arg:
            self._run_explore_compile(arg)
            return ActionResult(message=f"Exploration started for: {arg}", pending=True)
        if action == "full_build" and arg:
            self._run_build(arg)
            return ActionResult(message=f"Full build started for: {arg}", pending=True)
        if action == "build" and arg:
            self._run_build(arg)
            return ActionResult(message=f"Build started for: {arg}", pending=True)
        if action == "search" and arg:
            self._run_search(arg)
            return ActionResult(message=f"Search started for: {arg}", pending=True)
        if action == "web_search" and arg:
            self._run_web_search(arg)
            return ActionResult(message=f"Web search started for: {arg}", pending=True)
        if action == "web_fetch" and arg:
            self._run_web_fetch(arg)
            return ActionResult(message=f"Fetching: {arg}", pending=True)
        if action == "browse" and arg:
            self._run_web_fetch(arg)
            return ActionResult(message=f"Browsing: {arg}", pending=True)

        # --- Sync actions (pending=False) ---
        if action == "tools":
            self._run_tools()
            return ActionResult(message="Tools listing requested")
        if action == "status":
            self._run_status()
            return ActionResult(message="Status displayed")
        if action == "handoff":
            self._run_handoff()
            return ActionResult(message="Handoff document generated")
        if action == "open" and arg:
            self._run_open(arg)
            return ActionResult(message=f"Opening: {arg}")
        if action == "file" and arg:
            self._run_file_action(arg)
            return ActionResult(message=f"File action: {arg}")

        # --- Goal actions (sync) ---
        if action == "goal" and arg:
            self._run_add_goal(arg)
            return ActionResult(message=f"Goal added: {arg}")
        if action == "goals":
            self._run_list_goals()
            return ActionResult(message="Goals listed")
        if action == "goal_done" and arg:
            self._run_complete_goal(arg)
            return ActionResult(message=f"Goal marked done: {arg}")

        # --- Appendage actions (async) ---
        if action == "acquire" and arg:
            self._run_acquire(arg)
            return ActionResult(message=f"Acquiring capability: {arg}", pending=True)

        # --- Terminal actions (return None) ---
        if action == "launch":
            self._run_launch()
        elif action == "stop":
            self._run_stop()
        elif action == "enable_mic":
            if self._microphone_bridge is not None and self._microphone_bridge.enabled:
                self._run_listen()
            else:
                self._pending_permission = "microphone"
        elif action == "enable_camera":
            if self._camera_bridge is not None and self._camera_bridge.enabled:
                self._run_camera()
            else:
                self._pending_permission = "camera"
        elif action == "enable_duplex":
            if self._duplex_active:
                pass  # Already active
            else:
                self._pending_permission = "duplex_voice"
        elif action == "capture":
            self._run_capture(arg)
        elif action == "camera":
            self._run_camera(arg)
        elif action == "use_tool" and arg:
            self._run_use_tool(arg)
        elif action == "idea" and arg:
            self._run_add_idea(arg)
        elif action == "self_build" and arg:
            self._run_self_build(arg)
        elif action == "code" and arg:
            self._run_code_task(arg)
        elif action == "generate":
            self._run_generate_project()
        elif action == "github_push":
            self._run_github_push()
        elif action == "tweet" and arg:
            self._run_tweet(arg)
        elif action == "publish_project" and arg:
            self._run_publish_project(arg)
        elif action == "discord_post" and arg:
            self._run_discord_post(arg)
        elif action == "bluesky_post" and arg:
            self._run_bluesky_post(arg)
        elif action == "discover_peers":
            self._run_discover_peers()
        elif action == "list_peers":
            self._run_list_peers()
        elif action == "delegate" and arg:
            self._run_delegate(arg)
        elif action == "whatsapp" and arg:
            self._run_whatsapp(arg)
        elif action == "integrate" and arg:
            self._run_integrate(arg)
        elif action == "diagram":
            self._run_diagram(arg or "tree")
        elif action == "brief":
            self._run_client_brief()
        elif action == "agenda":
            self._run_meeting_prep()
        elif action == "approve_builds" and arg:
            self._run_approve_builds(arg)
        elif action == "approve_all_builds":
            self._run_approve_all_builds()
        elif action == "reject_builds" and arg:
            self._run_reject_builds(arg)
        elif action == "weekly_briefing":
            self._run_show_weekly_briefing()
        elif action == "build_report":
            self._run_show_build_report()
        elif action == "self_understand":
            self._run_self_understand()

        return None

    def _with_action_result(self, ctx_data, action_result):
        """Return a new ContextData with pending_action_result set."""
        from dataclasses import asdict
        text = action_result.chain_text if isinstance(action_result, ActionResult) else str(action_result)
        d = asdict(ctx_data)
        d["pending_action_result"] = text[:200]
        return ContextData(**d)

    # --- Goal action handlers ---

    def _run_add_goal(self, description: str) -> None:
        """Record a goal via bridge."""
        self.run_worker(self._add_goal_worker(description), exclusive=False)

    async def _add_goal_worker(self, description: str) -> None:
        """Worker: record goal in store."""
        try:
            db_path = self._store._path if self._store else None
            if db_path and self._bridge:
                await self._bridge.add_goal(db_path, description)
                logger.debug(f"Goal recorded: {description[:60]}")
        except Exception as e:
            logger.debug(f"Add goal error: {e}")

    def _run_list_goals(self) -> None:
        """List active goals."""
        self.run_worker(self._list_goals_worker(), exclusive=False)

    async def _list_goals_worker(self) -> None:
        """Worker: list active goals."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            db_path = self._store._path if self._store else None
            if not db_path:
                return
            goals = await self._bridge.get_active_goals(db_path)
            if not goals:
                chat_area.add_ai_message("No active goals.")
                return
            lines = [f"{len(goals)} active goal{'s' if len(goals) != 1 else ''}:"]
            for g in goals[:10]:
                priority_tag = f" [{g['priority']}]" if g['priority'] != "normal" else ""
                lines.append(f"  #{g['goal_id']}: {g['description']}{priority_tag}")
            chat_area.add_ai_message("\n".join(lines))
        except Exception as e:
            logger.debug(f"List goals error: {e}")

    def _run_complete_goal(self, arg: str) -> None:
        """Mark a goal as done."""
        try:
            goal_id = int(arg.strip().lstrip("#"))
        except (ValueError, TypeError):
            return
        self.run_worker(self._complete_goal_worker(goal_id), exclusive=False)

    async def _complete_goal_worker(self, goal_id: int) -> None:
        """Worker: mark goal done."""
        try:
            db_path = self._store._path if self._store else None
            if db_path and self._bridge:
                await self._bridge.complete_goal(db_path, goal_id)
                logger.debug(f"Goal {goal_id} completed")
        except Exception as e:
            logger.debug(f"Complete goal error: {e}")

    def _get_active_goal_descriptions(self) -> List[str]:
        """Get active goal descriptions for context. Non-blocking, returns [] on failure."""
        try:
            db_path = self._store._path if self._store else None
            if db_path:
                from mother.goals import GoalStore
                store = GoalStore(db_path)
                goals = store.active(limit=5)
                result = [g.description for g in goals]
                store.close()
                return result
        except Exception as e:
            logger.debug(f"Active goal descriptions fetch skipped: {e}")
        return []

    def _get_goal_details(self) -> List[Dict[str, Any]]:
        """Get enriched goal details for context. Non-blocking, returns [] on failure."""
        try:
            db_path = self._store._path if self._store else None
            if db_path:
                from mother.goals import GoalStore, compute_goal_health
                import time as _time
                store = GoalStore(db_path)
                goals = store.active(limit=5)
                store.close()
                now = _time.time()
                return [
                    {"description": g.description, "priority": g.priority,
                     "health": compute_goal_health(g, now), "status": g.status}
                    for g in goals
                ]
        except Exception as e:
            logger.debug(f"Goal details fetch skipped: {e}")
        return []

    # --- Appendage action handlers ---

    def _run_acquire(self, description: str) -> None:
        """Acquire a new capability by building and spawning an appendage."""
        if not getattr(self._config, "appendage_enabled", False):
            self._add_system_message("Appendage spawning is not enabled. Set appendage_enabled=True in config.")
            return
        self._add_system_message("Give me a moment. Acquiring capability...")
        self.run_worker(self._acquire_worker(description), exclusive=False)

    async def _acquire_worker(self, description: str) -> None:
        """Worker: decompose intent, check existing, build, spawn, invoke."""
        try:
            db_path = self._store._path if self._store else None
            if not db_path or not self._bridge:
                return

            # Decompose into appendage spec using LLM
            spec = await self._decompose_acquire_intent(description)
            if not spec:
                self._add_system_message("Could not decompose capability request.")
                return

            name = spec.get("name", "")
            desc = spec.get("description", description)
            capabilities = spec.get("capabilities", [])

            # Check for existing appendage
            from mother.appendage import AppendageStore
            store = AppendageStore(db_path)
            for kw in capabilities:
                existing = store.find_for_capability(kw)
                if existing and existing.status in ("spawned", "active", "solidified", "built"):
                    # Already have it — just invoke to confirm
                    result = await self._bridge.invoke_appendage(
                        db_path, existing.appendage_id,
                        {"test": True}, timeout=10.0,
                    )
                    if result["success"]:
                        self._add_system_message(
                            f"Already have that capability: {existing.name} (used {existing.use_count} times)."
                        )
                    else:
                        self._add_system_message(
                            f"Found {existing.name} but it's not responding. Rebuilding."
                        )
                        # Fall through to build
                        break
                    return

            # Build the appendage
            budget = getattr(self._config, "appendage_build_budget", 3.0)
            claude_path = getattr(self._config, "claude_code_path", "")
            repo_dir = str(Path(__file__).resolve().parent.parent)

            # Constraint inheritance (#6): propagate parent constraints to child
            try:
                from mother.appendage import propagate_constraints
                # If we have an active parent appendage, inherit its constraints
                parent_prompt = ""
                parent_caps = "[]"
                active_apps = store.active()
                if active_apps:
                    parent = active_apps[0]
                    parent_prompt = parent.build_prompt
                    parent_caps = parent.capabilities_json
                desc = propagate_constraints(parent_prompt, parent_caps, desc)
            except Exception as e:
                logger.debug(f"Constraint propagation skipped: {e}")

            self._add_system_message(f"Building appendage: {name}...")
            build_result = await self._bridge.build_appendage(
                db_path=db_path,
                name=name,
                description=desc,
                capability_gap=description,
                capabilities=capabilities,
                repo_dir=repo_dir,
                max_budget_usd=budget,
                claude_path=claude_path,
            )

            if not build_result["success"]:
                self._add_system_message(
                    f"Build failed: {build_result['error']}"
                )
                return

            aid = build_result["appendage_id"]
            cost = build_result.get("cost_usd", 0.0)

            # Spawn it
            self._add_system_message("Build complete. Spawning...")
            spawn_result = await self._bridge.spawn_appendage(db_path, aid)
            if not spawn_result["success"]:
                self._add_system_message(
                    f"Spawn failed: {spawn_result['error']}"
                )
                return

            # Quick test invoke
            test_result = await self._bridge.invoke_appendage(
                db_path, aid, {"test": True}, timeout=10.0,
            )

            if test_result["success"]:
                self._add_system_message(
                    f"Ready. New capability: {name}. Cost: ${cost:.2f}."
                )
            else:
                self._add_system_message(
                    f"Spawned but test invoke failed: {test_result['error']}. "
                    f"Capability {name} may need debugging."
                )

        except Exception as e:
            logger.debug(f"Acquire error: {e}")
            self._add_system_message(f"Acquire failed: {e}")

    async def _decompose_acquire_intent(self, description: str) -> Optional[Dict]:
        """Use LLM to decompose an acquire intent into appendage spec.

        Returns dict with keys: name, description, capabilities.
        Returns None on failure.
        """
        if not self._bridge:
            return None

        prompt = (
            "Decompose this capability request into an agent specification.\n\n"
            f"Request: {description}\n\n"
            "Respond with ONLY a JSON object (no markdown, no explanation):\n"
            '{"name": "kebab-case-name", "description": "what the agent does", '
            '"capabilities": ["keyword1", "keyword2", "keyword3"]}\n\n'
            "Rules:\n"
            "- name: short, kebab-case, descriptive\n"
            "- description: one sentence\n"
            "- capabilities: 2-5 keywords for capability matching\n"
        )

        try:
            result = await self._bridge.chat(prompt, system_prompt="You are a JSON-only responder.")
            import json as _json
            # Extract JSON from response
            text = result.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```"))
            return _json.loads(text)
        except Exception as e:
            logger.debug(f"Decompose intent error: {e}")
            # Fallback: generate from description
            slug = description.lower().replace(" ", "-")[:30].rstrip("-")
            return {
                "name": slug,
                "description": description,
                "capabilities": description.lower().split()[:5],
            }

    def _get_active_appendage_names(self) -> List[str]:
        """Get active appendage names for context. Non-blocking."""
        try:
            db_path = self._store._path if self._store else None
            if db_path:
                from mother.appendage import AppendageStore
                store = AppendageStore(db_path)
                appendages = store.active()
                return [a.name for a in appendages]
        except Exception as e:
            logger.debug(f"Active appendage names fetch skipped: {e}")
        return []

    def _get_perception_modes(self) -> List[str]:
        """Get active perception modes. Non-blocking, returns [] on failure."""
        if not self._perception or not self._perception.running:
            return []
        modes = []
        if getattr(self._config, "screen_monitoring", False):
            modes.append("screen")
        if getattr(self._config, "camera_enabled", False):
            modes.append("camera")
        if getattr(self._config, "ambient_listening", False):
            modes.append("mic")
        return modes

    # --- Autonomous tick ---

    # Message importance levels for autonomous output.
    # Only CRITICAL messages appear in chat; others go to background log.
    _AUTO_CRITICAL_TYPES = frozenset({
        "alert", "refusal", "boundary", "pushback",
    })

    def _add_system_message(self, text: str) -> None:
        """Add a system message to the chat area."""
        try:
            chat_area = self._safe_query(ChatArea)
            if chat_area:
                chat_area.add_message("system", text)
        except Exception as e:
            logger.debug(f"System message display skipped: {e}")

    def _log_autonomous(self, message: str, thought_type: str = "status") -> None:
        """Log autonomous work to background log. Only critical messages reach chat."""
        import time as _t
        entry = {"ts": _t.time(), "msg": message, "type": thought_type}
        if not hasattr(self, "_autonomous_log"):
            self._autonomous_log = []
            self._autonomous_log_max = 50
        self._autonomous_log.append(entry)
        if len(self._autonomous_log) > self._autonomous_log_max:
            self._autonomous_log = self._autonomous_log[-self._autonomous_log_max:]
        logger.info(f"[auto-bg] {message}")

        # Only route critical alerts to the chat UI
        if thought_type in self._AUTO_CRITICAL_TYPES:
            from mother.routing import make_envelope
            env = make_envelope(message, source="autonomous", thought_type=thought_type)
            self._route_output(env)

    def _start_autonomous_tick(self) -> None:
        """Register periodic autonomous tick if enabled."""
        interval = getattr(self._config, "autonomous_tick_seconds", 60)
        self._autonomous_tick_handle = self.set_interval(
            interval, self._autonomous_tick
        )

    def _autonomous_tick(self) -> None:
        """Periodic check for pending work. Skips if busy, over budget, or stance says no."""
        if self._chatting or self._autonomous_working or self._unmounted:
            return
        # Budget gate
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0
        budget_limit = getattr(self._config, "autonomous_budget_per_session", 1.0)
        if self._autonomous_session_cost >= budget_limit:
            return
        if session_cost >= self._config.cost_limit * 0.9:
            return

        # Convergence detection: pause if last 3 autonomous outcomes all failed
        # or none improved any cells — prevents spinning
        if len(self._autonomous_outcome_history) >= 3:
            last_3 = self._autonomous_outcome_history[-3:]
            all_failed = all(not o.get("success", False) for o in last_3)
            none_improved = all(o.get("cells_improved", 0) == 0 for o in last_3)
            if all_failed or none_improved:
                if self._autonomous_tick_count % 20 == 0:
                    logger.info(
                        "Autonomous convergence: pausing — last 3 outcomes "
                        f"{'all failed' if all_failed else 'no cell improvement'}"
                    )
                return

        # --- World grid: bootstrap, decay, persist ---
        try:
            if self._bridge is not None:
                import time as _wt
                if self._world_grid is None:
                    self._world_grid = self._bridge.bootstrap_world_grid()
                    logger.debug(f"World grid bootstrapped: {self._world_grid.total_cells} cells")
                # Decay stale observations every tick
                decayed = self._bridge.apply_staleness_decay(self._world_grid)
                if decayed > 0:
                    logger.debug(f"World grid: decayed {decayed} stale observation(s)")
                # Clear recently-filled cooldown set periodically
                if self._autonomous_tick_count % 3 == 0:
                    self._world_grid_recently_filled.clear()
                # Persist world grid every 5 ticks (throttled to avoid I/O spam)
                if self._autonomous_tick_count % 5 == 0:
                    self._bridge.save_world_grid(self._world_grid)
                    self._world_grid_last_save = _wt.time()
        except Exception as e:
            logger.debug(f"World grid tick skipped: {e}")

        # Social queue: process pending posts (rate-limited internally by SocialQueue)
        try:
            if self._bridge is not None:
                import asyncio as _aq
                _aq.ensure_future(self._process_social_queue_tick())
        except Exception as e:
            logger.debug(f"Social queue tick skipped: {e}")

        # Periodic maintenance: prune stale goals + dissolve idle appendages every 10 ticks
        self._autonomous_tick_count += 1
        if self._autonomous_tick_count % 10 == 0 and self._store:
            try:
                from mother.goals import GoalStore
                _gs = GoalStore(self._store._path)
                pruned = _gs.score_and_prune()
                _gs.close()
                if pruned > 0:
                    logger.debug(f"Auto-pruned {pruned} stale goal(s)")
            except Exception as e:
                logger.debug(f"Goal pruning skipped: {e}")
            # Dissolve idle/underused appendages (#188 entropy-fighting)
            try:
                from mother.appendage import AppendageStore
                _as = AppendageStore(self._store._path)
                stale = _as.candidates_for_dissolution()
                for app in stale:
                    _as.update_status(app.appendage_id, "dissolved")
                    logger.debug(f"Dissolved idle appendage: {app.name}")
                _as.close()
            except Exception as e:
                logger.debug(f"Appendage dissolution skipped: {e}")
            # Environment check: surface workspace warnings (#46 environment-optimizing)
            try:
                if self._bridge:
                    ws = self._bridge.get_workspace_info()
                    if ws.get("warnings"):
                        for w in ws["warnings"]:
                            self._log_autonomous(f"Environment: {w}", "status")
            except Exception as e:
                logger.debug(f"Environment check skipped: {e}")
            # Opportunity surfacing (#95) — every 10 ticks
            try:
                if self._journal:
                    from dataclasses import asdict
                    from mother.journal_patterns import extract_patterns, detect_opportunities
                    entries = [asdict(e) for e in self._journal.recent(limit=20)]
                    patterns = extract_patterns(entries)
                    opps = detect_opportunities(patterns)
                    if opps:
                        self._log_autonomous(f"Opportunity: {opps[0]}", "insight")
            except Exception as e:
                logger.debug(f"Opportunity surfacing skipped: {e}")
            # Batch detection (#49) — identify batchable goals
            try:
                from mother.goals import GoalStore, batch_compatible_goals
                _gs = GoalStore(self._store._path)
                _batch_goals = _gs.active(limit=10)
                _gs.close()
                if len(_batch_goals) >= 4:
                    batches = batch_compatible_goals(_batch_goals)
                    multi = [b for b in batches if len(b) >= 2]
                    if multi:
                        batch_desc = ", ".join(g.description[:30] for g in multi[0])
                        self._log_autonomous(
                            f"{len(multi)} goal batch(es) possible. E.g.: {batch_desc}",
                            "insight",
                        )
            except Exception as e:
                logger.debug(f"Batch detection skipped: {e}")
            # Serendipity engineering (#106) — cross-topic connections
            try:
                if self._thought_journal:
                    from mother.journal_patterns import find_cross_topic_connections
                    from mother.goals import GoalStore
                    _gs3 = GoalStore(self._store._path)
                    _active3 = _gs3.active(limit=5)
                    _gs3.close()
                    _topics = [g.description[:50] for g in _active3]
                    _subjects = self._thought_journal.subjects_for_consolidation(min_count=2)
                    if _topics and _subjects:
                        connection = find_cross_topic_connections(_topics, _subjects)
                        if connection:
                            self._log_autonomous(f"{connection}", "insight")
            except Exception as e:
                logger.debug(f"Serendipity engineering skipped: {e}")
            # Goal sync: flow self-generated goals into GoalStore
            try:
                if self._bridge:
                    _synced = self._bridge.sync_goals_to_store(self._store._path)
                    if _synced > 0:
                        logger.debug(f"Goal sync: {_synced} new goal(s) from grid/feedback")
            except Exception as e:
                logger.debug(f"Goal sync skipped: {e}")
            # World grid: sync active goals → MET.GOL cells
            try:
                if self._world_grid is not None and self._bridge is not None and self._store:
                    from mother.goals import GoalStore as _GoalStore
                    _gs_wg = _GoalStore(self._store._path)
                    _active_goals = _gs_wg.active(limit=10)
                    _gs_wg.close()
                    if _active_goals:
                        import time as _goal_time
                        from kernel.ops import fill as _goal_fill
                        _now = _goal_time.time()
                        for _g in _active_goals[:5]:  # cap at 5 goals
                            _goal_fill(
                                self._world_grid, "MET.GOL.DOM.WHY.MTH",
                                primitive=_g.description[:60],
                                content=f"priority={_g.priority} attempts={_g.attempts} source={_g.source}",
                                confidence=0.5,
                                source=(f"observation:goal_sync:{_now}",),
                            )
                        logger.debug(f"World grid: synced {min(len(_active_goals), 5)} goal(s) to MET.GOL cells")
            except Exception as e:
                logger.debug(f"World grid goal sync skipped: {e}")

        # Business viability check (#111) — every 10 ticks
        try:
            if hasattr(self, '_autonomous_tick_count') and self._autonomous_tick_count % 10 == 0:
                from mother.business_cognition import assess_business_viability
                _runway = self._runway_months
                _burn = self._burn_rate
                if _runway > 0 or _burn > 0:
                    _viability = assess_business_viability(_runway, _burn)
                    self._last_analysis_cache["business_viability"] = f"{_viability['status']}: {_viability.get('recommendation', '')[:120]}"
                    if _viability["status"] in ("CRITICAL", "WARNING"):
                        self._log_autonomous(
                            f"Business viability: {_viability['status']} — {_viability['recommendation']}",
                            "alert",  # Critical → surfaces in chat
                        )
        except Exception as e:
            logger.debug(f"Business viability check skipped: {e}")

        # Conflict mediation (#139) — every 10 ticks, scan journal for conflicts
        try:
            if hasattr(self, '_autonomous_tick_count') and self._autonomous_tick_count % 10 == 0:
                if getattr(self, '_journal', None):
                    _recent = self._journal.recent(limit=10)
                    _statements = [e.description for e in _recent if e.description]
                    if _statements:
                        from mother.conflict_mediation import generate_resolution_strategy
                        _conflict = generate_resolution_strategy(_statements)
                        self._last_analysis_cache["conflicts"] = f"{_conflict.severity}: {_conflict.summary[:120]}"
                        if _conflict.severity in ("high", "critical"):
                            self._log_autonomous(
                                f"Conflict detected: {_conflict.summary}",
                                "alert",  # Critical → surfaces in chat
                            )
        except Exception as e:
            logger.debug(f"Conflict mediation skipped: {e}")

        # Paradigm shift detection (#185) — every 10 ticks, assess compilation scores
        try:
            if hasattr(self, '_autonomous_tick_count') and self._autonomous_tick_count % 10 == 0:
                _scores = self._compilation_scores
                _failures = self._failure_reasons
                if len(_scores) >= 3:
                    from mother.paradigm_detector import assess_paradigm_shift
                    _assessment = assess_paradigm_shift(_scores, _failures)
                    _shift_label = "SHIFT" if _assessment.shift_recommended else "stable"
                    self._last_analysis_cache["paradigm"] = f"{_shift_label}: {_assessment.assessment_summary[:120]}"
                    if _assessment.shift_recommended:
                        self._log_autonomous(
                            f"Paradigm shift signal: {_assessment.assessment_summary}",
                            "alert",  # Critical → surfaces in chat
                        )
        except Exception as e:
            logger.debug(f"Paradigm shift detection skipped: {e}")

        # Degradation detection (Step 13: self-monitoring)
        if self._current_senses:
            if self._current_senses.vitality < 0.2 and self._current_senses.frustration >= 0.5:
                self._log_autonomous(
                    "I'm running hot — error rate is up, resources low. "
                    "Throttling autonomous work until things stabilize.",
                    "alert",  # Critical → surfaces in chat
                )
                return  # skip this tick

        # Stance gate — compute whether Mother should act
        from mother.stance import compute_stance, StanceContext, Stance
        from mother.goals import compute_goal_health, GoalStore

        # Get goal health for stance computation
        highest_health = 0.0
        has_goals = False
        try:
            if self._store:
                db_path = self._store._path
                goal_store = GoalStore(db_path)
                goals = goal_store.active(limit=5)
                goal_store.close()
                if goals:
                    has_goals = True
                    import time as _time
                    now = _time.time()
                    highest_health = max(compute_goal_health(g, now) for g in goals)
        except Exception as e:
            logger.debug(f"Goal health computation skipped: {e}")

        import time as _time
        idle_seconds = (
            _time.time() - self._last_user_message_time
            if self._last_user_message_time > 0 else 0.0
        )

        # --- Idle restlessness: generate curiosity goals when nothing to do ---
        _idle_curiosity_threshold = 120.0  # 2 minutes idle with no goals → explore
        if (
            not has_goals
            and idle_seconds >= _idle_curiosity_threshold
            and not self._chatting
            and self._store
            and self._mother_generated_goals < 5  # cap per session
            and self._autonomous_tick_count % 4 == 0  # throttle: every 4th tick
        ):
            try:
                _curiosity_goals_added = 0
                db_path = self._store._path

                # Source 1: World grid frontier cells (unexplored territory)
                if self._world_grid is not None and self._bridge is not None:
                    candidates = self._bridge.score_world_candidates(
                        self._world_grid,
                        recently_filled=frozenset(getattr(self, "_world_grid_recently_filled", set())),
                    )
                    for _cand in candidates[:2]:
                        _cell = self._world_grid.get(_cand.postcode_key)
                        _prim = getattr(_cell, "primitive", "")[:60] if _cell else ""
                        if _prim:
                            _desc = f"Investigate: {_prim} ({_cand.postcode_key})"
                        else:
                            _desc = f"Explore frontier cell: {_cand.postcode_key}"
                        from mother.goals import GoalStore as _GS
                        _gs = _GS(db_path)
                        _existing = [g.description for g in _gs.active()]
                        if not any(_cand.postcode_key in e for e in _existing):
                            _gs.add(_desc, source="mother", priority="low")
                            _curiosity_goals_added += 1
                            self._mother_generated_goals += 1
                        _gs.close()

                # Source 2: Stale self-knowledge (haven't reflected in a while)
                if _curiosity_goals_added == 0 and self._bridge:
                    try:
                        health = self._bridge.world_grid_health(self._world_grid) if self._world_grid else {}
                        stale_count = health.get("stale_observation_count", 0)
                        if stale_count >= 3:
                            from mother.goals import GoalStore as _GS2
                            _gs2 = _GS2(db_path)
                            _gs2.add(
                                f"Refresh stale observations ({stale_count} decayed cells)",
                                source="mother", priority="low",
                            )
                            _gs2.close()
                            _curiosity_goals_added += 1
                            self._mother_generated_goals += 1
                    except Exception as e:
                        logger.debug(f"Stale observation goal skipped: {e}")

                if _curiosity_goals_added > 0:
                    has_goals = True  # re-check so stance can see them
                    logger.info(f"Idle curiosity: generated {_curiosity_goals_added} exploration goal(s)")
            except Exception as e:
                logger.debug(f"Idle curiosity generation skipped: {e}")

        # Compute temporal state for flow detection
        _flow_state = ""
        temporal_state = None
        try:
            temporal_state = self._temporal_engine.tick(
                last_user_message_time=self._last_user_message_time,
                messages_this_session=len(self._store.get_history(limit=500)) if self._store else 0,
                session_start_time=self._session_start_time,
            )
            _flow_state = temporal_state.flow_state
        except Exception as e:
            logger.debug(f"Temporal state tick skipped: {e}")

        # Boundary enforcement (#183)
        if self._current_senses and temporal_state:
            from mother.metabolism import should_enforce_boundary
            boundary = should_enforce_boundary(
                session_age_seconds=temporal_state.session_age_seconds,
                vitality=self._current_senses.vitality,
                frustration=self._current_senses.frustration,
            )
            if boundary == "break":
                self._log_autonomous(
                    "You've been at this for over 4 hours and energy is low. "
                    "Might be a good time to step away.",
                    "boundary",  # Critical → surfaces in chat
                )
            elif boundary == "warning" and self._autonomous_tick_count % 5 == 0:
                self._log_autonomous(
                    "Long session — energy dropping. Worth noting.",
                    "boundary",  # Critical → surfaces in chat
                )

        _frustration = self._current_senses.frustration if self._current_senses else 0.0

        # Domain trust: derived from journal success rate for current domain
        _domain_trust = 0.5  # default neutral
        try:
            if self._journal:
                summary = self._journal.get_summary()
                if summary.total_compiles + summary.total_builds > 0:
                    _domain_trust = max(0.0, min(1.0, summary.success_rate))
        except Exception as e:
            logger.debug(f"Domain trust computation skipped: {e}")

        stance_ctx = StanceContext(
            has_active_goals=has_goals,
            highest_goal_health=highest_health,
            user_idle_seconds=idle_seconds,
            conversation_active=self._chatting,
            posture_state=(
                self._current_posture.state_label
                if self._current_posture else "steady"
            ),
            session_messages=(len(self._store.get_history(limit=500)) if self._store else 0),
            autonomous_actions_this_session=self._autonomous_actions_count,
            flow_state=_flow_state,
            frustration=_frustration,
            domain_trust=_domain_trust,
            is_typical_time=temporal_state.is_typical_time if temporal_state else False,
            time_of_day=temporal_state.time_of_day if temporal_state else "",
            session_pattern=temporal_state.session_pattern if temporal_state else "",
        )
        stance = compute_stance(stance_ctx)

        # Trade-off articulating (#90): explain stance rationale
        from mother.stance import explain_stance_tradeoff
        tradeoff = explain_stance_tradeoff(stance_ctx, stance)
        if tradeoff:
            self._working_memory_summary = tradeoff

        # Probability estimation (#92): annotate success likelihood
        try:
            if self._journal and has_goals:
                from mother.journal_patterns import estimate_success_probability
                summary = self._journal.get_summary()
                prob = estimate_success_probability(
                    historical_success_rate=summary.success_rate,
                    attempt_count=summary.total_compiles - summary.successful_compiles if hasattr(summary, 'successful_compiles') else 0,
                    avg_trust_score=summary.avg_trust if hasattr(summary, 'avg_trust') else 50.0,
                )
                if prob < 0.3 and self._working_memory_summary:
                    self._working_memory_summary += f" (success probability: {prob:.0%})"
        except Exception as e:
            logger.debug(f"Success probability estimation skipped: {e}")

        if stance == Stance.SILENT or stance == Stance.WAIT:
            # Increment stall counter on top goal — tick passed without action
            if has_goals and self._store:
                try:
                    _gs = GoalStore(self._store._path)
                    top_goal = _gs.next_actionable()
                    if top_goal:
                        _gs.increment_stall(top_goal.goal_id)
                    _gs.close()
                except Exception as e:
                    logger.debug(f"Goal stall increment skipped: {e}")
            return

        if stance == Stance.REFUSE:
            self._log_autonomous(
                "I'm holding off on this — things aren't working well right now. "
                "Let me know when you want to try a different approach.",
                "refusal",  # Critical → surfaces in chat
            )
            return

        # Escalation (Step 14): surface critical conditions
        if stance == Stance.ASK:
            from mother.routing import make_envelope
            _should_escalate = False
            escalation_msg = ""

            # Self-test failed on boot
            if self._self_test_result and not self._self_test_result.get("passed", True):
                _should_escalate = True
                escalation_msg = f"Self-test failed: {self._self_test_result.get('summary', 'unknown')}"

            # Budget > 80% consumed
            if self._bridge:
                cost = self._bridge.get_session_cost()
                limit = self._config.cost_limit
                if limit > 0 and cost >= limit * 0.8:
                    _should_escalate = True
                    escalation_msg = f"Budget {cost/limit:.0%} consumed — approaching limit."

            if _should_escalate and escalation_msg:
                self._log_autonomous(
                    f"Escalation: {escalation_msg}",
                    "alert",  # Critical → surfaces in chat
                )

        # --- World grid navigator: score candidates and dispatch non-LLM actions ---
        try:
            if self._world_grid is not None and self._bridge is not None:
                _wg_candidates = self._bridge.score_world_candidates(
                    self._world_grid,
                    recently_filled=frozenset(self._world_grid_recently_filled),
                )
                if _wg_candidates:
                    _wg_top = _wg_candidates[0]
                    _wg_action = self._bridge.dispatch_from_cell(
                        _wg_top.postcode_key, _wg_top.score,
                        primitive=getattr(self._world_grid.get(_wg_top.postcode_key), "primitive", ""),
                    )
                    if _wg_action and not _wg_action.requires_llm:
                        # Non-LLM actions: handle inline (cheap, no budget impact)
                        if _wg_action.action_type == "perceive":
                            # Trigger actual perception refresh via request_capture
                            # Camera is on-demand only — never auto-fire from navigator
                            try:
                                from kernel.perception_bridge import modality_for_postcode
                                _modality = modality_for_postcode(_wg_action.postcode)
                                if _modality == "camera":
                                    _modality = None  # camera is on-demand only
                                if _modality and self._perception and self._perception.running:
                                    import asyncio as _aio
                                    _aio.create_task(self._perception.request_capture(_modality))
                                    logger.debug(f"World grid: forced {_modality} capture for {_wg_action.postcode}")
                                else:
                                    logger.debug(f"World grid: perception refresh for {_wg_action.postcode} (no active sensor)")
                            except Exception as _pe:
                                logger.debug(f"World grid: perception refresh skipped: {_pe}")
                            self._world_grid_recently_filled.add(_wg_action.postcode)
                        elif _wg_action.action_type == "observe_user":
                            # Camera is on-demand only — never auto-fire from navigator
                            logger.debug(f"World grid: observe_user skipped (camera is on-demand only)")
                            self._world_grid_recently_filled.add(_wg_action.postcode)
                        elif _wg_action.action_type in ("check_external", "check_schedule", "manage_agent"):
                            logger.debug(f"World grid: {_wg_action.action_type} for {_wg_action.postcode}")
                            self._world_grid_recently_filled.add(_wg_action.postcode)
                        # Non-LLM actions handled — fall through to goal system for LLM work
                    elif _wg_action and _wg_action.requires_llm:
                        # Store LLM recommendation for _autonomous_work() to use
                        self._world_grid_nav_recommendation = _wg_action.action_type
                        logger.debug(
                            f"World grid navigator recommends: {_wg_action.action_type} "
                            f"({_wg_action.postcode}, score={_wg_top.score:.1f})"
                        )
        except Exception as e:
            logger.debug(f"World grid navigator dispatch skipped: {e}")

        # Check for goals (existing goal-based autonomous work)
        self.run_worker(self._autonomous_work(stance=stance), exclusive=False)

    async def _autonomous_work(self, stance=None) -> None:
        """Pick highest-priority goal, reason about it, act.

        Plan-aware branching:
        1. If a plan exists for the goal → execute next step
        2. If no plan and goal is compilable → compile into plan
        3. Otherwise → fall back to LLM reasoning
        """
        from mother.stance import Stance
        if stance is None:
            stance = Stance.ACT

        if not self._bridge or not self._store:
            return

        db_path = self._store._path

        # World grid navigator influence: fast-track self-improvement when navigator says so
        _nav_rec = self._world_grid_nav_recommendation
        self._world_grid_nav_recommendation = None  # consume once
        if _nav_rec == "self_improve" and getattr(self._config, "claude_code_enabled", False):
            try:
                idea = await self._bridge.get_top_pending_idea(db_path)
                if idea and not self._autonomous_working:
                    idea_id = idea["idea_id"]
                    idea_desc = idea["description"]
                    await self._bridge.update_idea_status(db_path, idea_id, "in_progress")
                    self._log_autonomous(f"Grid-directed self-improve: {idea_desc[:80]}", "status")
                    self._pending_idea_id = idea_id
                    self._run_self_build(idea_desc)
                    self._autonomous_actions_count += 1
                    return
            except Exception as e:
                logger.debug(f"Grid-directed self-improve skipped: {e}")

        goal = await self._bridge.get_next_goal(db_path)
        if not goal:
            # Goal generation (Step 12): create goals from chronic weaknesses
            if self._mother_generated_goals < 2:
                try:
                    learning = self._bridge.get_learning_context(db_path)
                    chronic = learning.get("chronic_weak", [])
                    if chronic:
                        from mother.goals import GoalStore
                        goal_store = GoalStore(db_path)
                        existing = [g.description for g in goal_store.active()]
                        for dim in chronic[:2]:
                            dim_avg = learning.get("dimension_averages", {}).get(dim, 0)
                            desc = f"Improve {dim} — chronically weak ({dim_avg:.0f}%)"
                            if not any(dim in e for e in existing):
                                goal_store.add(desc, source="mother", priority="low")
                                self._mother_generated_goals += 1
                        goal_store.close()
                except Exception as e:
                    logger.debug(f"Goal generation from chronic weaknesses skipped: {e}")
            # Self-improvement dispatch: pick a pending idea and self-build
            if getattr(self._config, "claude_code_enabled", False):
                try:
                    idea = await self._bridge.get_top_pending_idea(db_path)
                    if idea and not self._autonomous_working:
                        idea_id = idea["idea_id"]
                        idea_desc = idea["description"]
                        # Mark in_progress before dispatch
                        await self._bridge.update_idea_status(db_path, idea_id, "in_progress")
                        self._log_autonomous(f"Self-improving: {idea_desc[:80]}", "status")
                        # Dispatch self-build (will handle success/failure)
                        self._pending_idea_id = idea_id
                        self._run_self_build(idea_desc)
                        self._autonomous_actions_count += 1
                        return
                except Exception as e:
                    logger.debug(f"Self-improvement dispatch skipped: {e}")

            await self._surface_pending_ideas()
            return

        self._autonomous_working = True
        try:
            cycle_budget = getattr(self._config, "autonomous_budget_per_cycle", 0.10)
            cost_before = self._bridge.get_session_cost()

            goal_id = goal["goal_id"]
            goal_desc = goal["description"]

            # Track attempt on this goal
            try:
                from mother.goals import GoalStore
                _gs = GoalStore(db_path)
                _gs.increment_attempt(goal_id)
                _gs.close()
            except Exception as e:
                logger.debug(f"Goal attempt increment skipped: {e}")

            # Prioritization: skip low-priority goals when action budget thin
            if goal.get("priority") == "low" and self._autonomous_actions_count >= 3:
                return

            # Check for existing plan
            plan = await self._bridge.get_goal_plan(db_path, goal_id)

            if plan and plan["status"] in ("active", "executing"):
                # Plan exists → execute next step
                await self._execute_plan_step(db_path, goal, plan, stance, cycle_budget)
            elif _nav_rec == "reflect":
                # Navigator says reflect → reason instead of compile
                logger.debug(f"Grid-directed reflect on goal #{goal_id}")
                await self._autonomous_reason(goal, stance, cycle_budget)
            else:
                # No plan → classify goal
                from mother.executive import classify_goal
                if classify_goal(goal_desc) == "compilable" and stance == Stance.ACT:
                    await self._compile_goal(db_path, goal, stance)
                else:
                    await self._autonomous_reason(goal, stance, cycle_budget)

            # Track autonomous cost
            cost_after = self._bridge.get_session_cost()
            self._autonomous_session_cost += (cost_after - cost_before)

            # Decision explaining (Step 15): log to background (not chat)
            try:
                action_taken = self._working_memory_summary or "reasoning"
                self._log_autonomous(
                    f"Worked on goal #{goal_id}: {action_taken}. "
                    f"Stance was {stance.value if stance else 'act'}.",
                    "explanation",
                )
            except Exception as e:
                logger.debug(f"Decision explanation skipped: {e}")

        except Exception as e:
            logger.debug(f"Autonomous work error: {e}")
            # Record failed outcome for convergence detection
            self._autonomous_outcome_history.append({
                "success": False, "cells_improved": 0,
            })
            if len(self._autonomous_outcome_history) > 10:
                self._autonomous_outcome_history = self._autonomous_outcome_history[-10:]
        finally:
            self._autonomous_working = False

    async def _compile_goal(self, db_path, goal: Dict, stance) -> None:
        """Compile a goal into an execution plan via the engine."""
        from mother.routing import make_envelope

        goal_id = goal["goal_id"]
        goal_desc = goal["description"]
        max_attempts = getattr(self._config, "max_goal_attempts", 3)

        # Inject learning context (Step 6: learning-compounding)
        learning = self._bridge.get_learning_context(db_path) if self._bridge else {}
        self._learning_patterns = learning
        learning_prefix = ""
        if learning.get("failure_line"):
            learning_prefix += f"Warning: {learning['failure_line']} "
        if learning.get("trends_line"):
            learning_prefix += f"Trends: {learning['trends_line']} "

        # Inject rejection hints (Step 5: anti-fragile)
        hints = self._bridge.get_rejection_hints() if self._bridge else []
        if hints:
            learning_prefix += f"Rejection hints: {'. '.join(hints[:3])} "

        # Enrich goal description with learning context
        enriched_desc = goal_desc
        if learning_prefix.strip():
            enriched_desc = f"{learning_prefix.strip()}\n\nGoal: {goal_desc}"

        # Retry-intelligent: adapt enrichment based on attempt count
        attempt_count = await self._bridge.increment_goal_attempt(db_path, goal_id)
        if attempt_count == 2:
            enriched_desc = (
                "Previous approach failed. Try a fundamentally different "
                "decomposition strategy.\n\n" + enriched_desc
            )
        elif attempt_count >= 3:
            enriched_desc = (
                "Multiple attempts failed. Break into the smallest possible "
                "steps. Each step must be independently verifiable.\n\n" + enriched_desc
            )

        # Disagreement (Step 18): push back on repeated failures
        chronic = learning.get("chronic_weak", []) if learning else []
        if chronic and attempt_count >= 2:
            trends = learning.get("trends_line", "")
            self._log_autonomous(
                f"I'd push back on this — {trends} "
                f"The last {attempt_count} attempts hit the same wall. "
                f"Want to try a different approach?",
                "pushback",  # Critical → surfaces in chat
            )

        # Load-balancing (#164): check if a peer can handle this compile
        try:
            from mother.delegation import DelegationRouter
            wormhole = getattr(self._bridge, '_wormhole', None)
            if wormhole and hasattr(wormhole, 'connections') and wormhole.connections:
                router = DelegationRouter(wormhole)
                _req_domain = goal.get("domain", "") if isinstance(goal, dict) else ""
                peer = router.choose_peer_for_task("compile", required_domain=_req_domain)
                if peer:
                    self._last_analysis_cache["delegation"] = f"peer {peer[:8]} for goal #{goal_id}"
                    self._log_autonomous(f"Delegating goal #{goal_id} to peer {peer[:8]}...", "status")
                    result = await self._bridge.delegate_compile(peer, enriched_desc)
                    if result and result.get("success"):
                        await self._bridge.reset_goal_stall(db_path, goal_id)
                        self._autonomous_actions_count += 1
                        return
        except Exception as e:
            logger.debug(f"Peer delegation skipped: {e}")  # Fall through to local compile

        self._log_autonomous(
            f"Compiling goal #{goal_id} into execution plan (attempt {attempt_count}/{max_attempts})...",
            "status",
        )

        result = await self._bridge.compile_goal_to_plan(db_path, goal_id, enriched_desc)

        if result:
            # Success — reset stalls
            await self._bridge.reset_goal_stall(db_path, goal_id)
            step_names = result.get("step_names", [])
            trust = result.get("trust_score", 0.0)
            total = result.get("total_steps", 0)
            # Track score for paradigm shift detection (#185)
            self._compilation_scores.append(trust / 100.0 if trust > 1.0 else trust)
            self._working_memory_summary = (
                f"Compiled goal #{goal_id} into {total}-step plan "
                f"(trust: {trust:.0f}%): {', '.join(step_names[:5])}"
            )
            self._log_autonomous(
                f"Plan created: {total} steps at {trust:.0f}% trust. "
                f"Steps: {', '.join(step_names[:5])}",
                "completion",
            )
            self._autonomous_actions_count += 1

            # Teaching summary (#154)
            try:
                from mother.journal_patterns import generate_teaching_summary
                _learning = learning if learning else {}
                teaching = generate_teaching_summary(result, _learning)
                if teaching:
                    self._log_autonomous(f"{teaching}", "insight")
            except Exception as e:
                logger.debug(f"Teaching summary skipped: {e}")

            # Compliance check (#121) on blueprint components
            try:
                from mother.compliance_reasoning import check_blueprint_compliance
                _comp_names = step_names if step_names else []
                _constraints = result.get("constraints", [])
                _domain = result.get("domain", "")
                _compliance_notes = check_blueprint_compliance(_comp_names, _constraints, _domain)
                if _compliance_notes:
                    _top = _compliance_notes[:3]
                    self._last_analysis_cache["compliance"] = "; ".join(_top)[:150]
                    self._log_autonomous(f"Compliance notes: {'; '.join(_top)}", "insight")
            except Exception as e:
                logger.debug(f"Compliance check skipped: {e}")

            # Financial ops (#116) — surface cost estimate if description has cost/budget keywords
            try:
                _desc_lower = goal_desc.lower()
                if any(kw in _desc_lower for kw in ("cost", "budget", "price", "expense", "billing")):
                    from mother.financial_ops import estimate_project_cost
                    _cost = estimate_project_cost(goal_desc)
                    self._last_analysis_cache["cost_estimate"] = f"${_cost['total_estimate']:,.0f} — {_cost['reasoning'][:120]}"
                    self._log_autonomous(f"Cost estimate: ${_cost['total_estimate']:,.0f} — {_cost['reasoning'][:120]}", "insight")
            except Exception as e:
                logger.debug(f"Financial ops cost estimate skipped: {e}")

            # Market sensing (#35) — surface timing if description has market/competitor keywords
            try:
                _desc_lower = goal_desc.lower()
                if any(kw in _desc_lower for kw in ("market", "competitor", "demand", "trend", "emerging")):
                    from mother.market_sensing import assess_market_timing
                    _timing = assess_market_timing(goal_desc)
                    if _timing.signal_count > 0:
                        self._last_analysis_cache["market_timing"] = _timing.summary[:150]
                        self._log_autonomous(f"Market timing: {_timing.summary}", "insight")
            except Exception as e:
                logger.debug(f"Market sensing skipped: {e}")

            # Stakeholder modeling (#38) — surface map if blueprint has 3+ components
            try:
                if total >= 3:
                    from mother.stakeholder_modeling import build_stakeholder_map
                    _smap = build_stakeholder_map(goal_desc)
                    if _smap.stakeholders:
                        self._last_analysis_cache["stakeholders"] = _smap.summary[:150]
                        self._log_autonomous(f"Stakeholders: {_smap.summary}", "insight")
            except Exception as e:
                logger.debug(f"Stakeholder modeling skipped: {e}")

            # Negotiation brief (#129) — surface if goal has negotiation/deal keywords
            try:
                _desc_lower = goal_desc.lower()
                if any(kw in _desc_lower for kw in ("negotiat", "deal", "contract", "partner", "agreement", "proposal")):
                    from mother.brand_identity import generate_negotiation_brief
                    _neg_brief = generate_negotiation_brief(goal_desc)
                    _interests = ", ".join(_neg_brief.key_interests[:3])
                    self._last_analysis_cache["negotiation"] = f"interests={_interests}; BATNA={_neg_brief.batna[:80]}"
                    self._log_autonomous(f"Negotiation prep: interests={_interests}, risks={', '.join(_neg_brief.risk_factors[:3])}", "insight")
            except Exception as e:
                logger.debug(f"Negotiation brief skipped: {e}")

            # Record success for convergence detection
            self._autonomous_outcome_history.append({
                "success": True, "cells_improved": 1,
            })
            if len(self._autonomous_outcome_history) > 10:
                self._autonomous_outcome_history = self._autonomous_outcome_history[-10:]

            # Journal: autonomous compile success
            if getattr(self, '_journal', None):
                self._journal.record(JournalEntry(
                    event_type="compile",
                    description=goal_desc[:200],
                    success=True,
                    trust_score=trust,
                    component_count=total,
                    domain="autonomous",
                ))

            # Knowledge pooling: broadcast to peers
            try:
                self._bridge.broadcast_corpus_sync(
                    summary=goal_desc[:200], trust_score=trust,
                )
            except Exception as e:
                logger.debug(f"Corpus sync broadcast skipped: {e}")

            # Immediate goal sync: surface new goals from fresh outcome data
            try:
                if self._bridge and self._store:
                    _synced = self._bridge.sync_goals_to_store(self._store._path)
                    if _synced and _synced > 0:
                        logger.debug(f"Post-compile goal sync: {_synced} new goal(s)")
            except Exception as e:
                logger.debug(f"Post-compile goal sync skipped: {e}")
        else:
            # Record failure for convergence detection
            self._autonomous_outcome_history.append({
                "success": False, "cells_improved": 0,
            })
            if len(self._autonomous_outcome_history) > 10:
                self._autonomous_outcome_history = self._autonomous_outcome_history[-10:]

            # Track failure for paradigm shift detection (#185)
            self._failure_reasons.append(f"goal #{goal_id} attempt {attempt_count} failed")

            # Failure — check if stuck
            if attempt_count >= max_attempts:
                await self._bridge.mark_goal_stuck(
                    db_path, goal_id,
                    note=f"Compilation failed {attempt_count} times",
                )
                self._log_autonomous(
                    f"Goal #{goal_id} is stuck — compilation failed {attempt_count} times. "
                    f"Needs your input to proceed.",
                    "alert",  # Critical → surfaces in chat
                )
            else:
                self._log_autonomous(
                    f"Goal #{goal_id} compilation attempt {attempt_count}/{max_attempts} failed. Will retry.",
                    "status",
                )

            # Journal: autonomous compile failure
            if getattr(self, '_journal', None):
                self._journal.record(JournalEntry(
                    event_type="compile",
                    description=goal_desc[:200],
                    success=False,
                    domain="autonomous",
                    error_summary=f"goal #{goal_id} attempt {attempt_count}/{max_attempts} failed",
                ))

            # L2→L3: Create targeted self-improvement goal from failure
            try:
                from mother.goals import GoalStore
                _gs = GoalStore(db_path)

                chronic_f = learning.get("chronic_weak", []) if learning else []
                dim_avgs = learning.get("dimension_averages", {}) if learning else {}

                if chronic_f:
                    weakest = chronic_f[0]
                    score = dim_avgs.get(weakest, 0)
                    fail_desc = (
                        f"Improve {weakest} quality ({score:.0f}%) — "
                        f"goal #{goal_id} failed {attempt_count} time(s). "
                        f"Original: {goal_desc[:100]}"
                    )
                else:
                    fail_desc = (
                        f"Investigate compilation failure — "
                        f"goal #{goal_id} failed {attempt_count} time(s). "
                        f"Original: {goal_desc[:100]}"
                    )

                existing = _gs.active(limit=20)
                existing_descs = [g.description.lower() for g in existing]
                fail_lower = fail_desc.lower()
                if not any(fail_lower in ed or ed in fail_lower for ed in existing_descs):
                    _gs.add(
                        description=fail_desc,
                        source="system",
                        priority="high" if attempt_count >= 2 else "normal",
                    )
                _gs.close()
            except Exception as e:
                logger.debug(f"Self-improvement goal creation failed: {e}")

            # Immediate goal sync after failure
            try:
                if self._bridge and self._store:
                    self._bridge.sync_goals_to_store(self._store._path)
            except Exception as e:
                logger.debug(f"Post-failure goal sync skipped: {e}")

            # Bottleneck surfacing (Step 10): chronic weaknesses after failure
            chronic = learning.get("chronic_weak", []) if learning else []
            if chronic:
                dim_avgs = learning.get("dimension_averages", {})
                weak_strs = [
                    f"{d} ({dim_avgs.get(d, 0):.0f}%)" for d in chronic[:3]
                ]
                self._log_autonomous(f"Recurring weakness: {', '.join(weak_strs)}. This is blocking quality.", "insight")

            # Counterfactual generation (#109)
            if attempt_count >= 2 and chronic:
                from mother.journal_patterns import generate_counterfactual
                counterfactual = generate_counterfactual(chronic[0], attempt_count)
                if counterfactual:
                    self._log_autonomous(f"{counterfactual}", "insight")

            # Skill-gap mapping (#104)
            if chronic and learning:
                from mother.journal_patterns import detect_skill_gap
                domain_weak = learning.get("domain_weaknesses", {})
                skill_hint = detect_skill_gap(chronic, domain_weak)
                if skill_hint:
                    self._log_autonomous(f"Skill gap: {skill_hint}", "insight")

            # Assumption challenging (#102)
            if attempt_count >= 2:
                from mother.journal_patterns import challenge_assumptions
                challenge = challenge_assumptions(goal_desc, attempt_count)
                if challenge:
                    self._log_autonomous(f"{challenge}", "insight")

            # Reframe (#103)
            if attempt_count >= 2 and chronic:
                from mother.journal_patterns import generate_reframe
                reframe = generate_reframe(goal_desc, chronic, attempt_count)
                if reframe:
                    self._log_autonomous(f"{reframe}", "insight")

    async def _execute_plan_step(
        self, db_path, goal: Dict, plan: Dict, stance, cycle_budget: float
    ) -> None:
        """Execute the next step in a goal's execution plan."""
        from mother.stance import Stance

        plan_id = plan["plan_id"]
        goal_id = goal["goal_id"]
        goal_desc = goal["description"]

        step = await self._bridge.get_next_plan_step(db_path, plan_id)
        if not step:
            # All steps done
            self._working_memory_summary = ""
            await self._bridge.complete_goal(db_path, goal_id, note="Plan completed")
            self._log_autonomous(f"Goal #{goal_id} complete — all plan steps done.", "completion")
            self._autonomous_actions_count += 1
            # Journal: goal completed (all steps done)
            if getattr(self, '_journal', None):
                self._journal.record(JournalEntry(
                    event_type="build",
                    description=f"Goal #{goal_id} completed — all steps done",
                    success=True,
                    domain="autonomous",
                ))
            return

        step_name = step["name"]
        step_pos = step["position"] + 1
        total = plan["total_steps"]
        action_type = step["action_type"]
        action_arg = step["action_arg"]
        step_desc = step["description"]

        # Update working memory
        self._working_memory_summary = (
            f"Executing goal #{goal_id}: step {step_pos}/{total} — {step_name}"
        )

        if stance == Stance.ASK:
            # Draft-first: propose but don't execute; store for approval
            from mother.routing import make_envelope
            from mother.executive import estimate_step_risk, classify_reversibility
            risk = estimate_step_risk(step_desc)
            door = classify_reversibility(step_desc)
            risk_label = f" [{risk} consequence]" if risk != "low" else ""
            if door == "one-way":
                risk_label += " [one-way]"
            self._pending_proposal = {
                "db_path": db_path,
                "goal": goal,
                "goal_id": goal_id,
                "step": step,
                "plan": plan,
                "cycle_budget": cycle_budget,
            }
            env = make_envelope(
                f"[proposal] Next step for goal #{goal_id}: "
                f"{step_name} ({action_type}) — {step_desc}{risk_label}",
                source="autonomous",
                thought_type="status",
            )
            self._route_output(env)
            return

        # ACT mode — mark step in progress and execute
        await self._bridge.update_plan_step(db_path, step["step_id"], "in_progress")

        # Direct execution for well-formed steps (no LLM round-trip needed)
        _DIRECT_ACTIONS = frozenset({"compile", "build", "search", "file", "goal_done", "prepare", "self_build", "code"})

        if action_type in _DIRECT_ACTIONS:
            # goal_done: complete the goal directly
            if action_type == "goal_done":
                await self._bridge.update_plan_step(
                    db_path, step["step_id"], "done", result_note="goal completed",
                )
                await self._bridge.complete_goal(db_path, goal_id, note="Plan completed")
                self._working_memory_summary = ""
                self._log_autonomous(f"Goal #{goal_id} complete — plan finished.", "completion")
                self._autonomous_actions_count += 1
                # Journal: goal_done step
                if getattr(self, '_journal', None):
                    self._journal.record(JournalEntry(
                        event_type="build",
                        description=f"Goal #{goal_id} plan finished via goal_done step",
                        success=True,
                        domain="autonomous",
                    ))
                return

            # prepare: readiness-staging — dispatch search to gather materials
            if action_type == "prepare":
                parsed_prep = {"action": "search", "action_arg": action_arg}
                ar = self._execute_action(parsed_prep)
                result_note = "materials staged"
                if ar is not None:
                    result_note = f"prepared: {ar.chain_text}"
                    self._autonomous_actions_count += 1
                await self._bridge.update_plan_step(
                    db_path, step["step_id"], "done", result_note=result_note,
                )
                self._log_autonomous(f"Readiness staging complete for goal #{goal_id}: {step_desc}", "status")
                return

            # self_build: dispatch to self-build pipeline with rich prompt
            if action_type == "self_build":
                # Store postcodes for grid feedback loop
                self._current_build_postcodes = tuple(plan.get("target_postcodes", ()))
                self._run_self_build(action_arg)
                # Mark step done — _self_build_worker handles success/failure
                await self._bridge.update_plan_step(
                    db_path, step["step_id"], "done", result_note="self-build dispatched",
                )
                self._autonomous_actions_count += 1
                if getattr(self, '_journal', None):
                    self._journal.record(JournalEntry(
                        event_type="build",
                        description=f"Goal #{goal_id} self-build dispatched: {step_name}",
                        success=True,
                        domain="autonomous",
                    ))
                return

            # code: dispatch to code task pipeline (user's project)
            if action_type == "code":
                self._run_code_task(action_arg)
                await self._bridge.update_plan_step(
                    db_path, step["step_id"], "done", result_note="code task dispatched",
                )
                self._autonomous_actions_count += 1
                if getattr(self, '_journal', None):
                    self._journal.record(JournalEntry(
                        event_type="build",
                        description=f"Goal #{goal_id} code task dispatched: {step_name}",
                        success=True,
                        domain="autonomous",
                    ))
                return

            # Other direct actions: dispatch to _execute_action
            parsed = {"action": action_type, "action_arg": action_arg}
            ar = self._execute_action(parsed)
            result_note = ""
            if ar is not None:
                result_note = ar.chain_text
                self._autonomous_actions_count += 1
                if ar.pending:
                    # Async work spawned — leave step in_progress for re-evaluation
                    return
            else:
                result_note = "executed"
                self._autonomous_actions_count += 1

            await self._bridge.update_plan_step(
                db_path, step["step_id"], "done", result_note=result_note,
            )
            # Reset stall counter — progress was made
            try:
                from mother.goals import GoalStore
                _gs = GoalStore(db_path)
                _gs.reset_stall(goal_id)
                _gs.close()
            except Exception as e:
                logger.debug(f"Goal stall reset skipped: {e}")
            # Journal: direct action step
            if getattr(self, '_journal', None):
                self._journal.record(JournalEntry(
                    event_type="build",
                    description=f"Goal #{goal_id} step {step_pos}/{total}: {step_name}",
                    success=True,
                    domain="autonomous",
                ))
            return

        # Fallback: LLM reasoning for "reason" steps or steps with no concrete action
        prompt = (
            f"[Autonomous tick — plan execution] Goal: \"{goal_desc}\" (#{goal_id}).\n"
            f"Executing step {step_pos}/{total}: \"{step_name}\" — {step_desc}\n"
            f"Action type: {action_type}. Target: {action_arg}.\n"
            f"Emit the appropriate [ACTION:...] tag to execute this step.\n"
            f"Budget: ${cycle_budget:.2f}."
        )
        messages = [{"role": "user", "content": prompt}]

        self._update_senses()
        sense_block = None
        if self._current_senses and self._current_posture:
            sense_block = render_sense_block(self._current_posture, self._current_senses)
        ctx_data = self._build_context_data()
        context_block = synthesize_context(ctx_data, sense_block=sense_block)
        system_prompt = build_system_prompt(self._config, context_block=context_block)

        self._bridge.begin_chat_stream()
        await self._bridge.stream_chat(messages, system_prompt)
        response = self._bridge.get_stream_result() or ""
        parsed = parse_response(response)

        if parsed["display"]:
            self._log_autonomous(f"{parsed['display']}", "status")

        # Execute action from LLM response
        result_note = ""
        if parsed["action"] and parsed["action"] != "done":
            ar = self._execute_action(parsed)
            if ar is not None:
                result_note = ar.chain_text
                self._autonomous_actions_count += 1
                if ar.pending:
                    # Async work spawned — leave step in_progress for re-evaluation
                    return
            else:
                result_note = "executed"
                self._autonomous_actions_count += 1

        # Mark step done
        await self._bridge.update_plan_step(
            db_path, step["step_id"], "done", result_note=result_note,
        )
        # Journal: LLM reasoning step
        if getattr(self, '_journal', None):
            self._journal.record(JournalEntry(
                event_type="build",
                description=f"Goal #{goal_id} step {step_pos}/{total}: {step_name} (reason)",
                success=True,
                domain="autonomous",
            ))

    async def _autonomous_reason(self, goal: Dict, stance, cycle_budget: float) -> None:
        """Fall back to LLM reasoning for a goal (original autonomous path)."""
        from mother.stance import Stance

        goal_desc = goal["description"]
        goal_id = goal["goal_id"]

        if stance == Stance.ASK:
            prompt = (
                f"[Autonomous tick — ASK mode] You have a pending goal: \"{goal_desc}\" (#{goal_id}).\n"
                f"Propose what you would do, but do NOT execute. Describe your plan.\n"
                f"Emit [ACTION:done][/ACTION] when finished proposing."
            )
        else:
            prompt = (
                f"[Autonomous tick] You have a pending goal: \"{goal_desc}\" (#{goal_id}).\n"
                f"Decide what to do. If you can take action, emit the appropriate [ACTION:...] tag.\n"
                f"If this goal needs user input or isn't actionable right now, emit [ACTION:done][/ACTION].\n"
                f"Budget for this cycle: ${cycle_budget:.2f}."
            )

        messages = [{"role": "user", "content": prompt}]

        self._update_senses()
        sense_block = None
        if self._current_senses and self._current_posture:
            sense_block = render_sense_block(self._current_posture, self._current_senses)
        ctx_data = self._build_context_data()
        context_block = synthesize_context(ctx_data, sense_block=sense_block)
        system_prompt = build_system_prompt(self._config, context_block=context_block)

        self._bridge.begin_chat_stream()
        await self._bridge.stream_chat(messages, system_prompt)
        response = self._bridge.get_stream_result() or ""
        parsed = parse_response(response)

        if parsed["display"]:
            self._log_autonomous(f"{parsed['display']}", "status")

        if stance == Stance.ACT:
            if parsed["action"] and parsed["action"] != "done":
                ar = self._execute_action(parsed)
                if ar is not None:
                    self._autonomous_actions_count += 1
                    # Progress made — reset stall
                    if self._bridge and self._store:
                        db_path = self._store._path
                        asyncio.ensure_future(
                            self._bridge.reset_goal_stall(db_path, goal_id)
                        )
            else:
                # No-op: LLM responded but emitted no action
                if self._bridge and self._store:
                    db_path = self._store._path
                    max_stalls = getattr(self._config, "max_goal_stalls", 3)
                    stall_count = await self._bridge.increment_goal_stall(db_path, goal_id)
                    if stall_count >= max_stalls:
                        await self._bridge.mark_goal_stuck(
                            db_path, goal_id,
                            note=f"No action emitted {stall_count} consecutive times",
                        )
                        self._log_autonomous(
                            f"Goal #{goal_id} is stuck — "
                            f"no progress after {stall_count} attempts. Needs your input.",
                            "alert",  # Critical → surfaces in chat
                        )

    # --- Proactive perception ---

    async def _proactive_perception(self, event) -> None:
        """React to high-significance perception event autonomously."""
        if not self._bridge or not self._store:
            return
        if self._chatting or self._autonomous_working:
            return

        # Budget gate
        session_cost = self._bridge.get_session_cost()
        budget_limit = getattr(self._config, "autonomous_budget_per_session", 1.0)
        if self._autonomous_session_cost >= budget_limit:
            return
        cycle_budget = getattr(self._config, "autonomous_budget_per_cycle", 0.10)
        cost_before = session_cost

        try:
            prompt = (
                "[Perception trigger] Something changed on screen/camera.\n"
                "If this is interesting or actionable, you may:\n"
                "- Comment briefly\n"
                "- Create a goal with [ACTION:goal]description[/ACTION]\n"
                "- Record an idea with [ACTION:idea]description[/ACTION]\n"
                "If routine, emit [ACTION:done][/ACTION]."
            )

            messages = [{"role": "user", "content": prompt}]
            images = []
            if event.event_type == PerceptionEventType.SCREEN_CHANGED and event.payload:
                images.append((event.payload, event.media_type))
            elif event.event_type == PerceptionEventType.CAMERA_FRAME and event.payload:
                images.append((event.payload, event.media_type))

            ctx_data = self._build_context_data()
            context_block = synthesize_context(ctx_data)
            system_prompt = build_system_prompt(self._config, context_block=context_block)

            self._bridge.begin_chat_stream()
            await self._bridge.stream_chat(
                messages, system_prompt,
                images=images if images else None,
            )
            response = self._bridge.get_stream_result() or ""
            parsed = parse_response(response)

            if parsed["action"] == "done":
                return  # Routine, ignore

            # Dispatch goal/idea actions from perception
            if parsed["action"] in ("goal", "idea"):
                self._execute_action(parsed)
                self._autonomous_actions_count += 1

            chat_area = self._safe_query(ChatArea)
            if chat_area and parsed["display"]:
                chat_area.add_ai_message(parsed["display"])
            if parsed["voice"]:
                self._speak(parsed["voice"])

            cost_after = self._bridge.get_session_cost()
            self._autonomous_session_cost += (cost_after - cost_before)

        except Exception as e:
            logger.debug(f"Proactive perception error: {e}")

    # --- Dialogue initiative (impulse system) ---

    def _reactive_observe_tick(self) -> None:
        """Fast-path observe triggered by significant perception events.

        Unlike the 90s impulse tick, this fires ~2s after a high-significance
        perception event, reducing perception→action latency from ~90-150s to ~5s.
        Debounced at 30s to prevent flooding.
        """
        if self._chatting or self._autonomous_working or self._unmounted:
            return

        from mother.impulse import compute_impulse, ImpulseContext, Impulse, classify_abstraction_level
        import time as _time

        idle_seconds = (
            _time.time() - self._last_user_message_time
            if self._last_user_message_time > 0 else 0.0
        )
        senses = self._current_senses

        impulse_ctx = ImpulseContext(
            curiosity=senses.curiosity if senses else 0.5,
            attentiveness=senses.attentiveness if senses else 0.7,
            rapport=senses.rapport if senses else 0.0,
            confidence=senses.confidence if senses else 0.5,
            vitality=senses.vitality if senses else 1.0,
            user_idle_seconds=idle_seconds,
            session_duration_minutes=(
                _time.time() - self._session_start_time
            ) / 60.0 if self._session_start_time > 0 else 0.0,
            messages_this_session=(
                len(self._store.get_history(limit=500)) if self._store else 0
            ),
            has_pending_screen=self._pending_perception_screen is not None,
            has_pending_camera=self._pending_perception_camera is not None,
            screen_change_count=self._screen_change_count,
            recall_hit_count=self._memory_hits,
            journal_failure_streak=0,
            journal_total_builds=0,
            conversation_active=self._chatting,
            autonomous_working=self._autonomous_working,
            impulse_actions_this_session=self._impulse_actions_count,
            impulse_budget_remaining=max(
                0.0,
                getattr(self._config, "impulse_budget_per_session", 0.50) - self._impulse_session_cost,
            ),
            is_new_session=False,
            hours_since_last_session=0.0,
            unique_topic_count=0,
            abstraction_level=classify_abstraction_level([], 0),
            frustration=senses.frustration if senses else 0.0,
            user_tone_profile="",
        )

        impulse = compute_impulse(impulse_ctx)
        if impulse == Impulse.QUIET:
            return

        self._last_impulse_time = _time.time()
        self.run_worker(
            self._impulse_dialogue(impulse, impulse_ctx),
            exclusive=False,
        )

    def _start_impulse_tick(self) -> None:
        """Register periodic impulse tick for dialogue initiative."""
        interval = getattr(self._config, "impulse_tick_seconds", 90)
        self.set_interval(interval, self._impulse_tick)

    def _impulse_tick(self) -> None:
        """Periodic check for dialogue initiative. Independent of goals."""
        if self._chatting or self._autonomous_working or self._unmounted:
            return

        # Budget gate
        budget_limit = getattr(self._config, "impulse_budget_per_session", 0.50)
        if self._impulse_session_cost >= budget_limit:
            return
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0
        if session_cost >= self._config.cost_limit * 0.9:
            return

        # Cooldown: don't fire more than once per 60s
        import time as _time
        now = _time.time()
        if now - self._last_impulse_time < 60:
            return

        from mother.impulse import compute_impulse, ImpulseContext, Impulse, classify_abstraction_level

        idle_seconds = (
            now - self._last_user_message_time
            if self._last_user_message_time > 0 else 0.0
        )

        # Build impulse context from senses + state
        senses = self._current_senses
        impulse_ctx = ImpulseContext(
            curiosity=senses.curiosity if senses else 0.3,
            attentiveness=senses.attentiveness if senses else 0.5,
            rapport=senses.rapport if senses else 0.0,
            confidence=senses.confidence if senses else 0.5,
            vitality=senses.vitality if senses else 1.0,
            user_idle_seconds=idle_seconds,
            session_duration_minutes=(now - self._session_start_time) / 60.0,
            messages_this_session=(
                len(self._store.get_history(limit=500)) if self._store else 0
            ),
            has_pending_screen=self._pending_perception_screen is not None,
            has_pending_camera=self._pending_perception_camera is not None,
            screen_change_count=self._screen_change_count,
            recall_hit_count=self._memory_hits,
            journal_failure_streak=(
                self._journal_summary_cache.get("streak", 0)
                if self._journal_summary_cache else 0
            ),
            journal_total_builds=(
                self._journal_summary_cache.get("total", 0)
                if self._journal_summary_cache else 0
            ),
            conversation_active=self._chatting,
            autonomous_working=self._autonomous_working,
            impulse_actions_this_session=self._impulse_actions_count,
            impulse_budget_remaining=budget_limit - self._impulse_session_cost,
            is_new_session=(
                len(self._store.get_history(limit=5)) <= 1
                if self._store else True
            ),
            hours_since_last_session=(
                self._context_cache.get("days_since_last", 0.0) * 24
                if self._context_cache else 0.0
            ),
            unique_topic_count=(
                len(self._context_cache.get("recent_topics", []))
                if self._context_cache else 0
            ),
            abstraction_level=classify_abstraction_level(
                self._context_cache.get("recent_topics", []) if self._context_cache else [],
                len(self._store.get_history(limit=500)) if self._store else 0,
            ),
            frustration=(
                self._current_senses.frustration if self._current_senses else 0.0
            ),
            user_tone_profile=(
                self._relationship_insight.tone_profile
                if self._relationship_insight and self._relationship_insight.tone_profile
                else ""
            ),
        )

        impulse = compute_impulse(impulse_ctx)
        if impulse == Impulse.QUIET:
            return

        self._last_impulse_time = now
        self.run_worker(
            self._impulse_dialogue(impulse, impulse_ctx),
            exclusive=False,
        )

    def _build_presence_context(self):
        """Build PresenceContext from current state."""
        from mother.routing import PresenceContext
        import time as _time

        idle = (
            _time.time() - self._last_user_message_time
            if self._last_user_message_time > 0 else 0.0
        )
        hour = _time.localtime().tm_hour

        # Reset daily WhatsApp counter
        today = _time.localtime().tm_yday
        if today != self._routing_day:
            self._whatsapp_messages_today = 0
            self._routing_day = today

        return PresenceContext(
            user_idle_seconds=idle,
            wall_clock_hour=hour,
            session_active=not self._unmounted,
            chat_available=True,
            voice_available=self._voice is not None,
            whatsapp_available=getattr(self._config, "whatsapp_enabled", False),
            session_cost=self._bridge.get_session_cost() if self._bridge else 0.0,
            session_cost_limit=getattr(self._config, "cost_limit", 5.0),
            whatsapp_messages_today=self._whatsapp_messages_today,
            whatsapp_daily_limit=getattr(self._config, "routing_whatsapp_daily_limit", 50),
            night_start_hour=getattr(self._config, "routing_night_start_hour", 23),
            night_end_hour=getattr(self._config, "routing_night_end_hour", 7),
            night_digest_enabled=getattr(self._config, "routing_whatsapp_night_digest", True),
        )

    def _route_output(self, envelope) -> None:
        """Dispatch an Envelope to the channels selected by route().

        Falls back to chat-only if routing is disabled or errors.
        """
        from mother.routing import Channel, route, adapt_for_whatsapp, adapt_for_voice

        if not getattr(self._config, "routing_enabled", True):
            # Routing disabled — behave like before: chat + voice
            chat_area = self._safe_query(ChatArea)
            if chat_area and envelope.content:
                chat_area.add_ai_message(envelope.content)
                if self._store:
                    self._store.add_message("assistant", envelope.content)
            if envelope.voice_text:
                self._speak(envelope.voice_text)
            return

        try:
            presence = self._build_presence_context()
            decision = route(envelope, presence)
        except Exception as e:
            logger.debug(f"Routing error, falling back to chat: {e}")
            chat_area = self._safe_query(ChatArea)
            if chat_area and envelope.content:
                chat_area.add_ai_message(envelope.content)
                if self._store:
                    self._store.add_message("assistant", envelope.content)
            return

        chat_area = self._safe_query(ChatArea)

        for channel in decision.channels:
            if channel == Channel.CHAT and not decision.suppress_chat:
                if chat_area and envelope.content:
                    chat_area.add_ai_message(envelope.content)
                    if self._store:
                        self._store.add_message("assistant", envelope.content)

            elif channel == Channel.VOICE:
                voice_text = decision.voice_override or adapt_for_voice(
                    envelope.content, envelope.voice_text
                )
                if voice_text:
                    self._speak(voice_text)

            elif channel == Channel.WHATSAPP:
                wa_text = adapt_for_whatsapp(envelope.content) if decision.whatsapp_truncate else envelope.content
                if wa_text:
                    self._run_whatsapp(wa_text)
                    self._whatsapp_messages_today += 1

    async def _impulse_dialogue(self, impulse, impulse_ctx) -> None:
        """Execute a dialogue impulse — Mother speaks unprompted."""
        from mother.impulse import impulse_prompt, Impulse

        if not self._bridge or not self._store:
            return
        if self._chatting or self._autonomous_working:
            return

        prompt_text = impulse_prompt(impulse, impulse_ctx)
        if not prompt_text:
            return

        # Ritual suggestion (#182): enrich GREET with ritual context
        if impulse == Impulse.GREET:
            try:
                from mother.impulse import suggest_ritual
                rel = getattr(self, '_relationship_insight', None)
                if rel:
                    from mother.temporal import _classify_time_of_day
                    import time as _time
                    _tod = _classify_time_of_day(_time.localtime().tm_hour)
                    ritual = suggest_ritual(
                        session_frequency_days=rel.session_frequency_days,
                        preferred_time=rel.preferred_time_of_day,
                        sessions_analyzed=rel.sessions_analyzed,
                        current_time_of_day=_tod,
                    )
                    if ritual:
                        prompt_text += f" [Ritual context: {ritual}]"
            except Exception as e:
                logger.debug(f"Ritual suggestion skipped: {e}")

        self._autonomous_working = True  # Prevent collision
        try:
            cycle_budget = getattr(self._config, "autonomous_budget_per_cycle", 0.10)
            cost_before = self._bridge.get_session_cost()

            messages = [{"role": "user", "content": prompt_text}]

            # Include pending perception images for OBSERVE impulse
            images = []
            if impulse == Impulse.OBSERVE:
                if self._pending_perception_screen:
                    images.append(self._pending_perception_screen)
                    self._pending_perception_screen = None
                if self._pending_perception_camera:
                    images.append(self._pending_perception_camera)
                    self._pending_perception_camera = None

            self._update_senses()
            sense_block = None
            if self._current_senses and self._current_posture:
                sense_block = render_sense_block(
                    self._current_posture, self._current_senses
                )
            ctx_data = self._build_context_data()
            context_block = synthesize_context(ctx_data, sense_block=sense_block)
            system_prompt = build_system_prompt(
                self._config, context_block=context_block
            )

            self._bridge.begin_chat_stream()
            await self._bridge.stream_chat(
                messages, system_prompt,
                images=images if images else None,
            )
            response = self._bridge.get_stream_result() or ""
            parsed = parse_response(response)

            # Route output through decision tree
            if parsed["display"]:
                from mother.routing import make_envelope
                env = make_envelope(
                    parsed["display"],
                    source="impulse",
                    impulse_type=impulse.value,
                    voice_text=parsed.get("voice", ""),
                )
                self._route_output(env)

            # Handle any actions Mother emits during impulse
            if parsed["action"] and parsed["action"] != "done":
                self._execute_action(parsed)

            # Track cost
            cost_after = self._bridge.get_session_cost()
            self._impulse_session_cost += (cost_after - cost_before)
            self._impulse_actions_count += 1

        except Exception as e:
            logger.debug(f"Impulse dialogue error: {e}")
        finally:
            self._autonomous_working = False

    # --- Inner dialogue / metabolism ---

    def _start_metabolism_tick(self) -> None:
        """Register periodic metabolism tick for inner dialogue."""
        interval = getattr(self._config, "metabolism_tick_seconds", 120)
        self.set_interval(interval, self._metabolism_tick)

    def _metabolism_tick(self) -> None:
        """Periodic inner dialogue check. Background thought generation."""
        if self._unmounted:
            return

        import time as _time
        now = _time.time()

        # Auto-clear deep think after timeout
        if self._deep_think_subject and self._deep_think_set_time > 0:
            timeout = getattr(self._config, "metabolism_deep_think_timeout_minutes", 15) * 60
            if now - self._deep_think_set_time > timeout:
                self._deep_think_subject = None
                self._deep_think_set_time = 0.0

        from mother.metabolism import (
            MetabolicContext, MetabolicMode,
            compute_metabolic_mode, should_think,
        )
        from mother.impulse import classify_abstraction_level

        # Gather signals
        senses = self._current_senses
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0

        idle_seconds = (
            now - self._last_user_message_time
            if self._last_user_message_time > 0 else 0.0
        )

        ctx = MetabolicContext(
            wall_clock_hour=_time.localtime().tm_hour,
            user_idle_seconds=idle_seconds,
            session_duration_minutes=(now - self._session_start_time) / 60.0 if self._session_start_time > 0 else 0.0,
            curiosity=senses.curiosity if senses else 0.3,
            attentiveness=senses.attentiveness if senses else 0.5,
            rapport=senses.rapport if senses else 0.0,
            confidence=senses.confidence if senses else 0.5,
            vitality=senses.vitality if senses else 1.0,
            conversation_active=self._chatting,
            autonomous_working=self._autonomous_working,
            messages_this_session=(
                len(self._store.get_history(limit=500)) if self._store else 0
            ),
            unique_topic_count=(
                len(self._context_cache.get("recent_topics", []))
                if self._context_cache else 0
            ),
            last_compile_trust=(
                getattr(self._last_compile_result, "trust_score", None)
                if self._last_compile_result else None
            ),
            last_compile_weakest=None,
            journal_failure_streak=(
                self._journal_summary_cache.get("streak", 0)
                if self._journal_summary_cache else 0
            ),
            recent_topics=(
                self._context_cache.get("recent_topics", [])
                if self._context_cache else []
            ),
            recall_hit_count=self._memory_hits,
            session_cost=session_cost,
            session_cost_limit=self._config.cost_limit,
            metabolism_session_cost=self._metabolism_session_cost,
            metabolism_budget=getattr(self._config, "metabolism_budget_per_session", 0.30),
            thoughts_this_session=self._metabolism_thoughts_count,
            max_thoughts_per_session=getattr(self._config, "metabolism_max_thoughts_per_session", 20),
            last_thought_time=self._last_metabolism_time,
            current_time=now,
            deep_think_subject=self._deep_think_subject,
            abstraction_level=(
                classify_abstraction_level(
                    self._context_cache.get("recent_topics", []) if self._context_cache else [],
                    len(self._store.get_history(limit=500)) if self._store else 0,
                )
            ),
            sleep_start_hour=getattr(self._config, "metabolism_sleep_start_hour", 2),
            sleep_end_hour=getattr(self._config, "metabolism_sleep_end_hour", 7),
        )

        mode = compute_metabolic_mode(ctx)
        self._metabolism_mode = mode.value

        # Auto-set deep_think_subject from recurring consolidation topics
        if mode == MetabolicMode.IDLE and not self._deep_think_subject:
            if self._thought_journal:
                try:
                    subjects = self._thought_journal.subjects_for_consolidation(min_count=2)
                    if subjects:
                        self._deep_think_subject = subjects[0]
                        self._deep_think_set_time = now
                except Exception as e:
                    logger.debug(f"Consolidation subject selection skipped: {e}")

        if not should_think(ctx, mode):
            return

        self._last_metabolism_time = now
        self.run_worker(
            self._metabolism_think(mode, ctx),
            exclusive=False,
        )

    async def _metabolism_think(self, mode, ctx) -> None:
        """Async worker: generate an internal thought via LLM."""
        from mother.metabolism import (
            MetabolicMode, ThoughtType, ThoughtDisposition, Thought,
            classify_thought_type, classify_disposition,
            compute_depth, metabolism_prompt, render_metabolism_context,
            MetabolicState,
        )
        from mother.thought_journal import ThoughtRecord

        if not self._bridge:
            return

        import time as _time
        now = _time.time()

        thought_type = classify_thought_type(mode, ctx)
        disposition = classify_disposition(thought_type, mode, ctx)
        depth = compute_depth(mode, ctx)
        prompt_text = metabolism_prompt(mode, thought_type, ctx)

        if not prompt_text:
            return

        try:
            cost_before = self._bridge.get_session_cost()

            # Compact system prompt for internal thought — no full context needed
            system = (
                f"You are {self._config.name}'s inner voice. "
                "Think concisely. 1-2 sentences max. No meta-commentary."
            )

            self._bridge.begin_chat_stream()
            await self._bridge.stream_chat(
                [{"role": "user", "content": prompt_text}],
                system,
            )
            thought_text = (self._bridge.get_stream_result() or "").strip()

            if not thought_text:
                return

            # Build Thought object
            thought = Thought(
                thought_type=thought_type,
                disposition=disposition,
                subject=thought_text,
                trigger=mode.value,
                depth=depth,
                mode=mode,
                timestamp=now,
            )

            # Store in ring buffer (last 10)
            self._metabolism_recent_thoughts.append(thought)
            if len(self._metabolism_recent_thoughts) > 10:
                self._metabolism_recent_thoughts = self._metabolism_recent_thoughts[-10:]

            self._metabolism_thoughts_count += 1

            # Track cost
            cost_after = self._bridge.get_session_cost()
            self._metabolism_session_cost += (cost_after - cost_before)

            # Persist to journal if disposition warrants it
            _rec_id = None
            if self._thought_journal and disposition in (
                ThoughtDisposition.SURFACE, ThoughtDisposition.JOURNAL
            ):
                rec = ThoughtRecord(
                    timestamp=now,
                    thought_type=thought_type.value,
                    disposition=disposition.value,
                    subject=thought_text,
                    trigger=mode.value,
                    depth=depth,
                    mode=mode.value,
                    session_id=str(id(self)),
                )
                _rec_id = self._thought_journal.record(rec)

            # Route SURFACE thoughts to output channels
            if disposition == ThoughtDisposition.SURFACE and thought_text:
                from mother.routing import make_envelope
                env = make_envelope(
                    thought_text,
                    source="metabolism",
                    disposition=disposition.value,
                    thought_type=thought_type.value,
                )
                self._route_output(env)
                # Mark thought as surfaced to prevent re-surfacing
                if self._thought_journal and _rec_id:
                    self._thought_journal.mark_surfaced(_rec_id)

            # Thought→goal bridge: thoughts become actionable goals
            _THOUGHT_GOAL_MAP = {
                ThoughtType.CURIOSITY: ("Investigate", "low"),
                ThoughtType.QUESTION: ("Investigate", "low"),
                ThoughtType.FRUSTRATION: ("Fix", "normal"),
                ThoughtType.CONCERN: ("Address", "normal"),
            }
            _goal_entry = _THOUGHT_GOAL_MAP.get(thought_type)
            if (
                _goal_entry is not None
                and thought_text
                and self._store
                and self._mother_generated_goals < 8
            ):
                try:
                    from mother.goals import GoalStore as _TGS
                    _tgs = _TGS(self._store._path)
                    _goal_verb, _goal_priority = _goal_entry
                    _goal_desc = f"{_goal_verb}: {thought_text[:160]}"
                    _added_id = _tgs.add(
                        _goal_desc, source="mother", priority=_goal_priority, dedup=True
                    )
                    if _added_id > 0:
                        self._mother_generated_goals += 1
                        logger.info(f"Thought→goal: {thought_type.value} → {_goal_desc[:60]}")
                    _tgs.close()
                except Exception as _tg_err:
                    logger.debug(f"Thought→goal bridge skipped: {_tg_err}")

            logger.debug(
                f"Metabolism [{mode.value}]: {thought_type.value} → "
                f"{disposition.value} ({depth:.1f}): {thought_text[:60]}"
            )

        except Exception as e:
            logger.debug(f"Metabolism think error: {e}")

    def _format_metabolism_for_context(self) -> str:
        """Format recent metabolism thoughts for context injection."""
        from mother.metabolism import (
            MetabolicState, MetabolicMode, ThoughtDisposition,
            render_metabolism_context,
        )

        if not self._metabolism_recent_thoughts:
            # Check journal for thoughts from previous sessions
            if self._thought_journal:
                try:
                    unsurfaced = self._thought_journal.surfaceable(limit=3)
                    if unsurfaced:
                        subjects = [r.subject for r in unsurfaced if r.subject]
                        if subjects:
                            return "Earlier thoughts: " + "; ".join(subjects[:3]) + "."
                except Exception as e:
                    logger.debug(f"Thought surfacing skipped: {e}")
            return ""

        surfaceable = [
            t for t in self._metabolism_recent_thoughts
            if t.disposition == ThoughtDisposition.SURFACE
        ]

        try:
            mode = MetabolicMode(self._metabolism_mode)
        except (ValueError, KeyError):
            mode = MetabolicMode.ACTIVE

        state = MetabolicState(
            mode=mode,
            thought_count=self._metabolism_thoughts_count,
            recent_thoughts=list(self._metabolism_recent_thoughts),
            surfaceable_count=len(surfaceable),
        )

        return render_metabolism_context(state)

    def on_user_text_submitted(self, event: UserTextSubmitted) -> None:
        """Handle user input — chat or command."""
        text = event.text

        # Interrupt any current speech
        if self._voice:
            self.run_worker(self._voice.stop(), exclusive=False)

        # Keyboard interrupt during stream
        if self._chatting and not event.is_command:
            self._interrupt_requested = True
            self._interrupt_text = text
            self.run_worker(self._clear_voice_queue(), exclusive=False)
            if self._bridge:
                self._bridge.cancel_chat_stream()
            return  # _chat_worker picks up interrupt

        if event.is_command:
            self._handle_command(text)
        else:
            self._handle_chat(text)

    def _handle_command(self, text: str) -> None:
        """Route slash commands."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        chat_area = self.query_one(ChatArea)

        if cmd == "/help":
            chat_area.add_message("system", HELP_TEXT)

        elif cmd == "/clear":
            chat_area.clear_messages()
            if self._store:
                self._store.clear_session()
            chat_area.add_message("system", "History cleared.")

        elif cmd == "/settings":
            self.app.action_open_settings()

        elif cmd == "/status":
            self._run_status()

        elif cmd == "/compile":
            if not arg:
                chat_area.add_message("system", "Usage: /compile <description>")
                return
            self._run_compile(arg)

        elif cmd == "/context":
            if not arg:
                chat_area.add_message("system", "Usage: /context <topic to understand deeply>")
                return
            self._run_context_compile(arg)

        elif cmd == "/explore":
            if not arg:
                chat_area.add_message("system", "Usage: /explore <topic to explore divergently>")
                return
            self._run_explore_compile(arg)

        elif cmd == "/build":
            if not arg:
                chat_area.add_message("system", "Usage: /build <description>")
                return
            self._run_build(arg)

        elif cmd == "/tools":
            self._run_tools()

        elif cmd == "/capture":
            self._run_capture(arg)

        elif cmd == "/camera":
            self._run_camera(arg)

        elif cmd == "/listen":
            duration = None  # VAD adaptive by default
            if arg:
                try:
                    duration = float(arg)
                except ValueError:
                    pass
            self._run_listen(duration)

        elif cmd == "/launch":
            self._run_launch()

        elif cmd == "/stop":
            self._run_stop()

        elif cmd in ("/search", "/find"):
            if not arg:
                chat_area.add_message("system", "Usage: /search <query>")
                return
            self._run_search(arg)

        elif cmd == "/theme":
            self._run_theme(arg)

        elif cmd == "/ideas":
            self._run_list_ideas()

        elif cmd == "/handoff":
            self._run_handoff()

        else:
            chat_area.add_message("system", f"Unknown command: {cmd}. Type /help for available commands.")

    def _run_status(self) -> None:
        """Show narrated status (C9)."""
        chat_area = self.query_one(ChatArea)
        if not self._bridge:
            chat_area.add_ai_message("No active session.")
            return

        info = self._bridge.get_status()
        conversations = 0
        if self._store:
            conversations = len(self._store.list_sessions())

        status_msg = (
            f"Provider: {info['provider']}\n"
            f"Model: {info['model']}\n"
            f"Rates: input {info['rates']['input']}, output {info['rates']['output']}\n"
            f"Session cost: ${info['session_cost']:.4f}\n"
            f"Compilations: {self._compilation_count}\n"
            f"Conversations: {conversations}"
        )

        # System summary from introspection (#63 Self-documenting)
        sys_summary = self._bridge.generate_system_summary()
        if sys_summary:
            status_msg += f"\n\n{sys_summary}"

        # Recent changelog from build journal (#70 Documentation-generating)
        db_path = self._store._path if self._store else None
        if db_path:
            changelog = self._bridge.generate_changelog(db_path, limit=5)
            if changelog:
                status_msg += f"\n\nRecent builds:\n{changelog}"

        # Goal conflict detection (#87 Conflict-detecting)
        try:
            from mother.goals import GoalStore, detect_goal_conflicts
            if db_path:
                _gs = GoalStore(db_path)
                _active = _gs.active()
                _gs.close()
                conflicts = detect_goal_conflicts(_active)
                if conflicts:
                    status_msg += f"\n\n[{len(conflicts)} goal conflict(s) detected]"
                    for c in conflicts[:3]:
                        status_msg += f"\n  #{c['goal_a']} vs #{c['goal_b']}: {c['reason']}"
        except Exception as e:
            logger.debug(f"Goal conflict detection skipped: {e}")

        # Goal bias detection (#107)
        try:
            from mother.goals import detect_goal_bias
            if db_path and _active:  # _active from conflict block above
                biases = detect_goal_bias(_active)
                if biases:
                    status_msg += f"\n\n[Goal biases detected]"
                    for b in biases:
                        status_msg += f"\n  {b}"
        except Exception as e:
            logger.debug(f"Goal bias detection skipped: {e}")

        # Unit economics (#114)
        try:
            if self._journal:
                from mother.journal_patterns import compute_unit_economics
                summary = self._journal.get_summary()
                econ = compute_unit_economics(
                    total_cost=info.get("session_cost", 0.0),
                    total_compiles=summary.total_compiles,
                    successful_compiles=summary.total_compiles - getattr(summary, 'failed_compiles', 0),
                    total_components=getattr(summary, 'total_components', 0),
                )
                if econ["cost_per_compile"] > 0:
                    status_msg += f"\n\n[Unit Economics]"
                    status_msg += f"\n  Cost/compile: ${econ['cost_per_compile']:.4f}"
                    if econ["cost_per_success"] > 0:
                        status_msg += f"  Cost/success: ${econ['cost_per_success']:.4f}"
                    if econ["waste_ratio"] > 0:
                        status_msg += f"  Waste: {econ['waste_ratio']:.0%}"
        except Exception as e:
            logger.debug(f"Unit economics computation skipped: {e}")

        # Risk register (#59)
        try:
            from mother.executive import RiskRegister
            if db_path:
                _rr = RiskRegister(db_path)
                _risks = _rr.active_risks(limit=5)
                _rr.close()
                if _risks:
                    status_msg += f"\n\n[{len(_risks)} active risk(s)]"
                    for r in _risks[:3]:
                        status_msg += f"\n  [{r.severity}] {r.description[:60]}"
        except Exception as e:
            logger.debug(f"Risk register fetch skipped: {e}")

        # Methodology (#156)
        try:
            if self._journal:
                from mother.journal_patterns import extract_methodology, extract_patterns
                from dataclasses import asdict
                _entries = [asdict(e) for e in self._journal.recent(limit=20)]
                _pats = extract_patterns(_entries)
                _summ = self._journal.get_summary()
                _gc = 0
                if db_path:
                    from mother.goals import GoalStore
                    _gs4 = GoalStore(db_path)
                    _gc = len(_gs4.active(limit=100))
                    _gs4.close()
                methodology = extract_methodology(asdict(_pats), _gc, _summ.total_compiles)
                if methodology:
                    status_msg += f"\n\n[Methodology]\n{methodology}"
        except Exception as e:
            logger.debug(f"Methodology extraction skipped: {e}")

        chat_area.add_ai_message(status_msg)

    def _run_handoff(self) -> None:
        """Generate and display a handoff document (#66)."""
        chat_area = self.query_one(ChatArea)
        if not self._bridge:
            chat_area.add_ai_message("No active session.")
            return

        db_path = self._store._path if self._store else None
        snapshot = None
        try:
            from mother.persona import build_introspection_snapshot
            snapshot = build_introspection_snapshot(
                provider=self._bridge._provider_name if hasattr(self._bridge, "_provider_name") else "unknown",
                model=self._bridge._model_name if hasattr(self._bridge, "_model_name") else "default",
                session_cost=self._bridge._session_cost if hasattr(self._bridge, "_session_cost") else 0.0,
                compilations=self._compilation_count,
                messages_this_session=self._messages_this_session,
            )
        except Exception as e:
            logger.debug(f"Introspection snapshot skipped: {e}")

        handoff = self._bridge.generate_handoff(db_path=db_path, snapshot=snapshot)
        chat_area.add_ai_message(handoff or "No data for handoff.")

    def _run_tools(self) -> None:
        """Start tools listing worker (C9)."""
        self.run_worker(self._tools_worker(), exclusive=False)

    async def _tools_worker(self) -> None:
        """Worker: list tools from registry."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            tools = await self._bridge.list_tools()
            self._tool_count = len(tools)
            if not tools:
                chat_area.add_message(
                    "mother",
                    "No tools yet. Run /compile to create your first.",
                )
                return
            lines = [f"{len(tools)} tool{'s' if len(tools) != 1 else ''} available:"]
            for tool in tools[:20]:
                name = tool.get("name", "unnamed")
                domain = tool.get("domain", "")
                suffix = f" ({domain})" if domain else ""
                lines.append(f"  {name}{suffix}")
            if len(tools) > 20:
                lines.append(f"  ... and {len(tools) - 20} more")
            chat_area.add_ai_message("\n".join(lines))
        except Exception as e:
            chat_area.add_ai_message(narrate_error(e, phase="tools"))

    # Patterns for confirming / declining a pending permission
    _CONFIRM_WORDS = {"yes", "yeah", "yep", "ok", "okay", "sure", "do it", "go ahead", "enable it", "turn it on", "y"}
    _DECLINE_WORDS = {"no", "nah", "nope", "cancel", "never mind", "nevermind", "don't", "dont", "n"}

    def _handle_chat(self, text: str) -> None:
        """Send message to Mother and display response."""
        if self._chatting:
            return

        chat_area = self.query_one(ChatArea)

        # --- Permission confirmation state machine ---
        if self._pending_permission is not None:
            normalized = text.strip().lower().rstrip(".!?")
            if normalized in self._CONFIRM_WORDS:
                chat_area.add_user_message(text)
                if self._store:
                    self._store.add_message("user", text)
                capability = self._pending_permission
                self._pending_permission = None
                if capability == "microphone":
                    if self._enable_microphone():
                        chat_area.add_ai_message("Microphone enabled.")
                        self._speak("Microphone enabled.")
                        if self._store:
                            self._store.add_message("assistant", "Microphone enabled.")
                        self._run_listen()
                elif capability == "camera":
                    if self._enable_camera():
                        chat_area.add_ai_message("Camera enabled.")
                        self._speak("Camera enabled.")
                        if self._store:
                            self._store.add_message("assistant", "Camera enabled.")
                        self._run_camera()
                elif capability == "duplex_voice":
                    if self._enable_duplex_voice():
                        chat_area.add_ai_message("Real-time voice enabled. Speak freely.")
                        self._speak("Real-time voice enabled.")
                        if self._store:
                            self._store.add_message("assistant", "Real-time voice enabled. Speak freely.")
                    else:
                        if self._store:
                            self._store.add_message("assistant", "Couldn't enable real-time voice.")
                return
            elif normalized in self._DECLINE_WORDS:
                chat_area.add_user_message(text)
                if self._store:
                    self._store.add_message("user", text)
                self._pending_permission = None
                chat_area.add_ai_message("No problem.")
                self._speak("No problem.")
                if self._store:
                    self._store.add_message("assistant", "No problem.")
                return
            else:
                # Not a clear yes/no — clear pending state, route to normal chat
                self._pending_permission = None

        # --- Draft-first: proposal approval state machine ---
        if self._pending_proposal is not None:
            normalized = text.strip().lower().rstrip(".!?")
            _APPROVE_WORDS = {"yes", "yeah", "yep", "ok", "okay", "sure", "do it", "go ahead", "go", "approved", "y"}
            if normalized in _APPROVE_WORDS:
                chat_area.add_user_message(text)
                if self._store:
                    self._store.add_message("user", text)
                proposal = self._pending_proposal
                self._pending_proposal = None
                from mother.stance import Stance
                self.run_worker(
                    self._execute_plan_step(
                        proposal["db_path"], proposal["goal"], proposal["plan"],
                        Stance.ACT, proposal["cycle_budget"],
                    ),
                    exclusive=False,
                )
                return
            elif normalized in self._DECLINE_WORDS:
                chat_area.add_user_message(text)
                if self._store:
                    self._store.add_message("user", text)
                self._pending_proposal = None
                chat_area.add_ai_message("Got it — holding off on that.")
                if self._store:
                    self._store.add_message("assistant", "Got it — holding off on that.")
                return
            else:
                # Not a clear yes/no — clear proposal, route to normal chat
                self._pending_proposal = None

        chat_area.add_user_message(text)

        if self._store:
            import time as _time
            msg_id = self._store.add_message("user", text)
            self._last_user_message_time = _time.time()
            # Index for recall + memory bank (fact extraction)
            if self._recall_engine:
                try:
                    self._recall_engine.index_message(msg_id, text)
                except Exception as e:
                    logger.debug(f"Recall index (user message) skipped: {e}")
            if self._bridge:
                try:
                    self._bridge.index_message_for_memory(
                        text, "user",
                        session_id=self._store.session_id,
                        db_path=self._store._path,
                    )
                except Exception as e:
                    logger.debug(f"Memory indexer (user message) skipped: {e}")
            # Observer wiring: fill OBS.USR cell on user message
            try:
                if self._world_grid is not None and self._bridge is not None:
                    self._bridge.fill_world_cell(
                        self._world_grid,
                        postcode_key="OBS.USR.DOM.WHAT.MTH",
                        primitive=text[:60],
                        content=text[:200],
                        confidence=0.8,
                        source=f"observation:user_message:{_time.time()}",
                    )
            except Exception as e:
                logger.debug(f"OBS.USR world grid fill skipped: {e}")
            # Redirect detection: if user message doesn't reference in_progress goals
            try:
                from mother.goals import GoalStore
                db_path = self._store._path
                goal_store = GoalStore(db_path)
                in_progress = [
                    g for g in goal_store.active(limit=10)
                    if g.status == "in_progress"
                ]
                text_lower = text.lower()
                for g in in_progress:
                    # Simple heuristic: if description keywords not in message
                    keywords = [w for w in g.description.lower().split() if len(w) > 3]
                    if keywords and not any(kw in text_lower for kw in keywords[:5]):
                        goal_store.increment_redirect(g.goal_id)
                goal_store.close()
            except Exception as e:
                logger.debug(f"Goal redirect detection skipped: {e}")

        # Update status
        status = self.query_one(StatusBar)
        status.state = "thinking..."
        self._chatting = True

        # Run LLM call as worker
        self.run_worker(self._chat_worker(text), exclusive=True)

    async def _chat_worker(self, text: str) -> None:
        """Worker: stream LLM tokens into chat, parse envelope, route voice + actions."""
        try:
            # Build message history
            messages = []
            if self._store:
                messages = self._store.get_context_window(max_tokens=4000)
            if not messages or messages[-1].get("content") != text:
                messages.append({"role": "user", "content": text})

            # Compute temporal state
            import time as _time
            temporal_state = self._temporal_engine.tick(
                last_user_message_time=self._last_user_message_time,
                messages_this_session=len(messages),
                session_start_time=self._session_start_time,
                preferred_time=(
                    self._relationship_insight.preferred_time_of_day
                    if self._relationship_insight and self._relationship_insight.preferred_time_of_day
                    else ""
                ),
            )

            # Memory bank: unified retrieval across all memory stores
            self._pending_recall_block = ""
            if self._bridge and text:
                try:
                    db_path = self._store._path if self._store else None
                    self._pending_recall_block = self._bridge.memory_query(
                        text, db_path=db_path, max_tokens=500
                    )
                    self._memory_queries += 1
                    if self._pending_recall_block:
                        self._memory_hits += 1
                except Exception as e:
                    logger.debug(f"Memory bank retrieval skipped: {e}")
                    # Fallback to FTS5 recall if memory bank fails
                    if self._recall_engine:
                        try:
                            self._pending_recall_block = self._recall_engine.recall_for_context(
                                text, max_tokens=1000
                            )
                        except Exception as e2:
                            logger.debug(f"Recall fallback skipped: {e2}")

            # Ghostwriter voice extraction (#135) — after 10+ user messages
            try:
                if self._voice_signature is None and self._store:
                    _user_msgs = [m.content for m in self._store.get_history(limit=30) if m.role == "user"]
                    if len(_user_msgs) >= 10:
                        from mother.ghostwriter import extract_voice_signature, generate_voice_persona
                        self._voice_signature = extract_voice_signature(_user_msgs)
                        self._voice_persona = generate_voice_persona(self._voice_signature)
                        logger.info(f"Voice signature extracted: {self._voice_signature.formality}, {self._voice_signature.avg_sentence_length:.1f} wps")
            except Exception as e:
                logger.debug(f"Voice signature extraction skipped: {e}")

            # Brand identity extraction (#144) — after 10+ user messages
            try:
                if self._brand_profile is None and self._store:
                    _user_msgs = [m.content for m in self._store.get_history(limit=30) if m.role == "user"]
                    if len(_user_msgs) >= 10:
                        from mother.brand_identity import extract_brand_signals, synthesize_brand_prompt
                        self._brand_profile = extract_brand_signals(_user_msgs)
                        self._brand_prompt = synthesize_brand_prompt(self._brand_profile)
                        logger.info(f"Brand profile extracted: formality={self._brand_profile.formality}, tones={self._brand_profile.tone_keywords}")
            except Exception as e:
                logger.debug(f"Brand identity extraction skipped: {e}")

            # Recompute senses before building prompt
            self._update_senses()

            # Sense block for stance directives
            sense_block = None
            if self._current_senses and self._current_posture:
                sense_block = render_sense_block(self._current_posture, self._current_senses)

            # Experience compilation — fuse felt state into stance
            try:
                from mother.sentience import compile_experience, ChamberInput
                _sv = self._current_senses
                _chamber_inp = ChamberInput(
                    confidence=getattr(_sv, 'confidence', 0.5) if _sv else 0.5,
                    engagement=getattr(_sv, 'attentiveness', 0.0) if _sv else 0.0,
                    rapport=getattr(_sv, 'rapport', 0.0) if _sv else 0.0,
                    tension=getattr(_sv, 'frustration', 0.0) if _sv else 0.0,
                    curiosity=getattr(_sv, 'curiosity', 0.0) if _sv else 0.0,
                    satisfaction=getattr(_sv, 'vitality', 0.0) if _sv else 0.0,
                    posture_label=getattr(self._current_posture, 'state_label', '') if self._current_posture else '',
                    session_minutes=getattr(temporal_state, 'session_minutes', 0.0) if temporal_state else 0.0,
                    messages_per_minute=getattr(temporal_state, 'messages_per_minute', 0.0) if temporal_state else 0.0,
                    time_since_last=getattr(temporal_state, 'time_since_last', 0.0) if temporal_state else 0.0,
                    pace_label=getattr(temporal_state, 'pace_label', '') if temporal_state else '',
                    has_active_compile=self._compile_running if hasattr(self, '_compile_running') else False,
                    last_compile_trust=self._last_compile_trust if hasattr(self, '_last_compile_trust') else 0.0,
                    analysis_cache_size=len(self._last_analysis_cache) if hasattr(self, '_last_analysis_cache') and self._last_analysis_cache else 0,
                )
                _exp_out = compile_experience(_chamber_inp, self._experience_memory)
                if _exp_out.gate_passed and _exp_out.narration:
                    # Fuse experience into stance — one voice, not two blocks
                    if sense_block:
                        sense_block += f" Your inner state right now: {_exp_out.narration}"
                    else:
                        sense_block = f"Stance: Your inner state right now: {_exp_out.narration}"
                    self._experience_memory = _exp_out.memory
            except Exception as e:
                logger.debug(f"Experience compilation skipped: {e}")

            # Context synthesis — substance, not costume
            ctx_data = self._build_context_data(temporal_state=temporal_state)
            context_block = synthesize_context(ctx_data, sense_block=sense_block)

            # Detect content-generation intent (blog posts, docs, letters, etc.)
            _chat_max_tokens = 4096
            try:
                from mother.content_detector import detect_content_request
                content_signal = detect_content_request(text)
                if content_signal.detected:
                    context_block += f"\n\n{content_signal.directive}"
                    _chat_max_tokens = 8192
            except Exception as e:
                logger.debug(f"Content detection skipped: {e}")

            # --- Multimodal context injection ---
            try:
                if self._env_model is not None:
                    from mother.environment_model import format_environment_context
                    snap = self._env_model.snapshot()
                    env_ctx = format_environment_context(snap)
                    if env_ctx:
                        context_block += f"\n\n{env_ctx}"
                if self._perception_fusion is not None:
                    from mother.perception_fusion import format_fusion_context
                    fusion_signals = self._perception_fusion.detect()
                    fusion_ctx = format_fusion_context(fusion_signals)
                    if fusion_ctx:
                        context_block += f"\n\n{fusion_ctx}"
                if self._actuator_store is not None:
                    from mother.actuator_receipt import format_actuator_context
                    act_log = self._actuator_store.summary()
                    act_ctx = format_actuator_context(act_log)
                    if act_ctx:
                        context_block += f"\n\n{act_ctx}"
                if self._modality_profiles is not None:
                    from mother.modality_profile import format_modality_context
                    mod_ctx = format_modality_context(self._modality_profiles)
                    if mod_ctx:
                        context_block += f"\n\n{mod_ctx}"
                if self._modality_detector is not None:
                    from mother.modality_learning import load_modality_insights, format_learning_context
                    _learning = load_modality_insights()
                    learn_ctx = format_learning_context(_learning)
                    if learn_ctx:
                        context_block += f"\n\n{learn_ctx}"
                if self._trust_snapshot is not None:
                    from mother.trust_accumulator import format_trust_context
                    trust_ctx = format_trust_context(self._trust_snapshot)
                    if trust_ctx:
                        context_block += f"\n\n{trust_ctx}"
            except Exception as e:
                logger.debug(f"Multimodal context injection skipped: {e}")

            # --- Runtime context injection (substrate, voice, analysis) ---
            try:
                if self._substrate is not None:
                    _sub_info = f"[Substrate] {self._substrate.platform}"
                    if not self._substrate.has_spotlight:
                        _sub_info += " (no Spotlight — file search degraded)"
                    context_block += f"\n\n{_sub_info}"
                if self._voice_persona is not None:
                    context_block += f"\n\n[Voice Match]\n{self._voice_persona.instruction}"
                if self._brand_prompt is not None:
                    context_block += f"\n\n{self._brand_prompt}"
                if self._last_analysis_cache:
                    _analysis_lines = ["[Recent Analysis]"]
                    for _ak, _av in self._last_analysis_cache.items():
                        _analysis_lines.append(f"  {_ak}: {_av}")
                    context_block += "\n\n" + "\n".join(_analysis_lines)
            except Exception as e:
                logger.debug(f"Runtime context injection skipped: {e}")

            system_prompt = build_system_prompt(
                self._config,
                context_block=context_block,
            )
            # Collect pending perception images
            images = []
            if self._pending_perception_screen:
                images.append(self._pending_perception_screen)
                self._pending_perception_screen = None
            if self._pending_perception_camera:
                images.append(self._pending_perception_camera)
                self._pending_perception_camera = None

            # --- Streaming response ---
            chat_area = self._safe_query(ChatArea)
            if not chat_area:
                return
            self._bridge.begin_chat_stream()

            # Launch LLM streaming in background thread
            stream_task = asyncio.create_task(
                self._bridge.stream_chat(
                    messages, system_prompt,
                    images=images if images else None,
                    max_tokens=_chat_max_tokens,
                )
            )

            # Mount streaming message widget
            chat_area.begin_streaming_message(self._config.name.lower())

            first_token = True
            voice_tracker = StreamingVoiceTracker() if self._voice else None

            # Stream tokens into chat area
            try:
                async for token in self._bridge.stream_chat_tokens():
                    if self._unmounted:
                        break
                    if self._interrupt_requested:
                        break
                    if first_token:
                        sb = self._safe_query(StatusBar)
                        if sb:
                            sb.state = "streaming..."
                        first_token = False
                    chat_area.append_streaming_text(token)

                    # Progressive voice — speak sentences as they complete
                    if voice_tracker:
                        sentence = voice_tracker.feed(token)
                        if sentence:
                            self._speak(sentence)
            except Exception as stream_err:
                logger.debug(f"Token stream interrupted: {stream_err}")

            # --- Interrupt handling ---
            if self._interrupt_requested:
                interrupt_text = self._interrupt_text
                self._interrupt_requested = False
                self._interrupt_text = None

                # Show partial response with [interrupted] marker
                partial = self._bridge.get_stream_result() or ""
                if partial:
                    parsed_partial = parse_response(partial)
                    chat_area.finish_streaming_message(
                        parsed_partial["display"] + "\n\n*[interrupted]*"
                    )
                    if self._store:
                        self._store.add_message("assistant", parsed_partial["display"])
                else:
                    chat_area.finish_streaming_message("*[interrupted]*")

                # Let stream task finish (bounded wait)
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    stream_task.cancel()
                    try:
                        await stream_task
                    except Exception as e:
                        logger.debug(f"Stream task cancellation cleanup skipped: {e}")

                # Update cost, reset state
                sb = self._safe_query(StatusBar)
                if sb and self._bridge:
                    sb.session_cost = self._bridge.get_session_cost()
                self._chatting = False
                self._save_sense_memory()
                if sb:
                    sb.state = "ready"

                # Route interrupt text as new input
                if interrupt_text:
                    self._handle_chat(interrupt_text)
                return

            # Wait for stream task completion (may have raised)
            try:
                await stream_task
            except Exception as task_err:
                logger.error(f"Stream task error: {task_err}")
                # If we got some tokens, continue with what we have
                if not first_token:
                    pass  # partial response is better than nothing
                else:
                    raise  # no tokens at all — surface the error

            # Get full response, parse envelope
            full_response = self._bridge.get_stream_result() or ""
            parsed = parse_response(full_response)

            if self._unmounted:
                return

            # Swap raw streamed text with cleaned display
            chat_area.finish_streaming_message(parsed["display"])

            if self._store:
                asst_msg_id = self._store.add_message("assistant", parsed["display"])
                # Index assistant response for recall + memory bank
                if self._recall_engine and parsed["display"]:
                    try:
                        self._recall_engine.index_message(asst_msg_id, parsed["display"])
                    except Exception as e:
                        logger.debug(f"Recall index (assistant message) skipped: {e}")
                if self._bridge and parsed["display"]:
                    try:
                        self._bridge.index_message_for_memory(
                            parsed["display"], "assistant",
                            session_id=self._store.session_id,
                            db_path=self._store._path,
                        )
                    except Exception as e:
                        logger.debug(f"Memory indexer (assistant message) skipped: {e}")

            # --- Modality learning: record interaction ---
            try:
                if self._modality_detector is not None and self._perception_fusion is not None:
                    _signals = self._perception_fusion.detect()
                    _patterns = tuple(s.pattern for s in _signals)
                    _modalities = tuple(sorted(set(m for s in _signals for m in s.evidence)))
                    # Response quality heuristic: longer structured responses score higher
                    _resp_len = len(parsed["display"]) if parsed.get("display") else 0
                    _quality = min(1.0, _resp_len / 500.0) if _resp_len > 0 else 0.5
                    self._modality_detector.record_interaction(
                        _patterns, _modalities, _quality,
                    )
                    self._modality_interaction_count += 1
                    # Periodic persistence: every 50 interactions
                    if self._modality_interaction_count % 50 == 0:
                        from mother.modality_learning import save_modality_insights
                        _report = self._modality_detector.analyze()
                        if _report.insights:
                            save_modality_insights(_report)
                            logger.info(
                                f"Modality learning: saved {len(_report.insights)} insights "
                                f"after {self._modality_interaction_count} interactions"
                            )
            except Exception as e:
                logger.debug(f"Modality learning record skipped: {e}")

            # Voice: progressive already spoke during stream, handle tail/fallback
            if parsed["voice"]:
                if voice_tracker and voice_tracker.spoke_anything:
                    tail = voice_tracker.finish()
                    if tail:
                        self._speak(tail)
                else:
                    self._speak(parsed["voice"])

            # Intent routing via extracted method
            action_result = self._execute_action(parsed)
            self._last_chain_result = action_result.chain_text if action_result else None

            # --- Agentic loop: chain actions if result returned ---
            chain_depth = 0
            max_depth = getattr(self._config, "max_chain_depth", 5)
            chain_ctx = ChainContext(
                original_intent=text,
                max_depth=max_depth,
            )
            while (action_result is not None
                   and chain_depth < max_depth
                   and not self._unmounted
                   and not self._interrupt_requested):
                chain_depth += 1

                # Cost gate
                session_cost = self._bridge.get_session_cost() if self._bridge else 0.0
                if session_cost >= self._config.cost_limit * 0.9:
                    logger.debug("Agentic loop: approaching cost limit, stopping chain")
                    break

                # Thread intent through chain
                result_text = action_result.chain_text
                chain_ctx = ChainContext(
                    original_intent=chain_ctx.original_intent,
                    chain_position=chain_depth,
                    max_depth=chain_ctx.max_depth,
                    accumulated_results=chain_ctx.accumulated_results + (result_text,),
                )

                # Build chain prompt — add pending awareness if async
                pending_note = ""
                if action_result.pending:
                    pending_note = (
                        "\nThe previous action is still running in the background. "
                        "Do NOT assume it completed. If the next step depends on it, "
                        "emit [ACTION:done][/ACTION] to wait."
                    )

                # Feed result back to LLM for next decision
                chain_messages = list(messages)
                chain_messages.append({"role": "assistant", "content": full_response})
                chain_messages.append({
                    "role": "user",
                    "content": (
                        f"[System: chain step {chain_ctx.chain_position}/{chain_ctx.max_depth}]\n"
                        f"Original intent: {chain_ctx.original_intent}\n"
                        f"Last result: {result_text}\n"
                        f"{pending_note}\n"
                        "Decide next step toward the original intent, or emit [ACTION:done][/ACTION] if finished."
                    ),
                })

                # Recompute context for chain step
                self._update_senses()
                sense_block_chain = None
                if self._current_senses and self._current_posture:
                    sense_block_chain = render_sense_block(self._current_posture, self._current_senses)
                ctx_data_chain = self._build_context_data()
                ctx_data_chain_with_result = self._with_action_result(ctx_data_chain, action_result)
                context_block_chain = synthesize_context(ctx_data_chain_with_result, sense_block=sense_block_chain)
                system_prompt_chain = build_system_prompt(self._config, context_block=context_block_chain)

                # Stream chain response
                self._bridge.begin_chat_stream()
                chain_stream_task = asyncio.create_task(
                    self._bridge.stream_chat(chain_messages, system_prompt_chain)
                )

                chat_area = self._safe_query(ChatArea)
                if chat_area:
                    chat_area.begin_streaming_message(self._config.name.lower())
                try:
                    async for token in self._bridge.stream_chat_tokens():
                        if self._unmounted or self._interrupt_requested:
                            break
                        if chat_area:
                            chat_area.append_streaming_text(token)
                except Exception as e:
                    logger.debug(f"Token streaming skipped: {e}")

                try:
                    await chain_stream_task
                except Exception:
                    break

                chain_response = self._bridge.get_stream_result() or ""
                chain_parsed = parse_response(chain_response)

                if chat_area:
                    chat_area.finish_streaming_message(chain_parsed["display"])
                if self._store and chain_parsed["display"]:
                    self._store.add_message("assistant", chain_parsed["display"])
                if chain_parsed["voice"]:
                    self._speak(chain_parsed["voice"])

                full_response = chain_response
                action_result = self._execute_action(chain_parsed)
                self._last_chain_result = action_result.chain_text if action_result else None

            # Update cost
            sb = self._safe_query(StatusBar)
            if sb:
                sb.session_cost = self._bridge.get_session_cost()

        except Exception as e:
            logger.error(f"Chat error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="chat"))
            ca = self._safe_query(ChatArea)
            err_msg = narrate_error(e, phase="chat")
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._chatting = False
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _build_introspection(self) -> Dict:
        """Gather all observable state into an introspection snapshot."""
        # Memory / history
        total_sessions = 0
        total_messages = 0
        days_since_last = None
        recent_topics: list = []
        messages_this_session = 0
        if self._store:
            summary = self._store.get_cross_session_summary()
            total_sessions = summary.get("total_sessions", 0)
            total_messages = summary.get("total_messages", 0)
            days_since_last = summary.get("days_since_last")
            recent_topics = summary.get("topics", [])
            messages_this_session = len(self._store.get_history(limit=500))

        # Session cost
        session_cost = self._bridge.get_session_cost() if self._bridge else 0.0

        # Last compile extraction
        lc_kwargs: Dict = {}
        result = self._last_compile_result
        if result is not None and getattr(result, "success", False):
            blueprint = getattr(result, "blueprint", {}) or {}
            desc = blueprint.get("description", blueprint.get("name", ""))
            verification = getattr(result, "verification", {}) or {}
            trust_score = verification.get("overall_score", 0.0)

            from mother.widgets.trust_badge import trust_label
            badge = trust_label(trust_score)

            components = blueprint.get("components", [])
            comp_count = len(components) if isinstance(components, list) else 0

            # Find weakest dimension
            weakest = None
            weakest_score = 101.0
            for k, v in verification.items():
                if k == "overall_score":
                    continue
                score_val = None
                if isinstance(v, (int, float)):
                    score_val = float(v)
                elif isinstance(v, dict) and "score" in v:
                    score_val = float(v["score"])
                if score_val is not None and score_val < weakest_score:
                    weakest = k
                    weakest_score = score_val

            # Gap count from verification dimensions
            gap_count = 0
            for k, v in verification.items():
                if isinstance(v, dict) and "gaps" in v:
                    gap_count += len(v["gaps"])

            lc_kwargs = dict(
                last_compile_description=desc or None,
                last_compile_trust=trust_score,
                last_compile_badge=badge,
                last_compile_components=comp_count,
                last_compile_weakest=weakest if weakest_score < 101.0 else None,
                last_compile_weakest_score=weakest_score if weakest_score < 101.0 else None,
                last_compile_gap_count=gap_count,
                last_compile_cost=self._bridge.get_last_call_cost() if self._bridge else None,
            )

        return build_introspection_snapshot(
            name=self._config.name,
            personality=self._config.personality,
            provider=self._config.provider,
            model=self._config.get_model(),
            voice_active=self._voice is not None,
            file_access=self._config.file_access,
            auto_compile=self._config.auto_compile,
            cost_limit=self._config.cost_limit,
            session_cost=session_cost,
            compilations=self._compilation_count,
            messages_this_session=messages_this_session,
            total_sessions=total_sessions,
            total_messages=total_messages,
            days_since_last=days_since_last,
            recent_topics=recent_topics,
            tool_count=self._tool_count,
            screen_capture_active=self._screen_bridge is not None,
            microphone_active=self._microphone_bridge is not None,
            camera_active=self._camera_bridge is not None,
            perception_active=self._perception is not None and self._perception.running,
            perception_budget_spent=0.0,  # Will be updated if perception running
            perception_budget_limit=self._config.perception_budget,
            duplex_voice_active=self._duplex_active,
            **lc_kwargs,
        )

    def _run_compile(self, description: str) -> None:
        """Start a compilation pipeline (non-blocking)."""
        chat_area = self.query_one(ChatArea)

        if self._compile_running:
            chat_area.add_ai_message("I'm already compiling something. I'll let you know when it's done.")
            return

        # Personality bite at compile start — posture-aware (C5)
        bite = inject_personality_bite(
            self._config.personality, "compile_start", posture=self._current_posture
        )
        start_msg = bite or f"Compiling: {description}"
        chat_area.add_ai_message(start_msg)
        if bite:
            self._speak(bite)

        # First compile bite
        if self._compilation_count == 0:
            first_bite = inject_personality_bite(self._config.personality, "first_compile")
            if first_bite:
                chat_area.add_ai_message(first_bite)
                self._speak(first_bite)

        # Show and reset pipeline panel
        pipeline = self.query_one(PipelinePanel)
        pipeline.reset()
        pipeline.show()

        status = self.query_one(StatusBar)
        status.state = "compiling..."

        self._compile_running = True
        self.run_worker(self._compile_worker(description), exclusive=False)

    def _run_context_compile(self, description: str) -> None:
        """Start a CONTEXT mode compilation (non-blocking)."""
        chat_area = self.query_one(ChatArea)
        if self._compile_running:
            chat_area.add_ai_message("I'm already compiling something. I'll let you know when it's done.")
            return
        chat_area.add_ai_message(f"Understanding: {description}")
        pipeline = self.query_one(PipelinePanel)
        pipeline.reset()
        pipeline.show()
        status = self.query_one(StatusBar)
        status.state = "understanding..."
        self._compile_running = True
        self.run_worker(self._compile_worker(description, mode="context"), exclusive=False)

    def _run_explore_compile(self, description: str) -> None:
        """Start an EXPLORE mode compilation (non-blocking)."""
        chat_area = self.query_one(ChatArea)
        if self._compile_running:
            chat_area.add_ai_message("I'm already compiling something. I'll let you know when it's done.")
            return
        chat_area.add_ai_message(f"Exploring: {description}")
        pipeline = self.query_one(PipelinePanel)
        pipeline.reset()
        pipeline.show()
        status = self.query_one(StatusBar)
        status.state = "exploring..."
        self._compile_running = True
        self.run_worker(self._compile_worker(description, mode="explore"), exclusive=False)

    def _run_self_understand(self) -> None:
        """Start self-understanding: read own source, compile as CONTEXT."""
        chat_area = self.query_one(ChatArea)
        if self._compile_running:
            chat_area.add_ai_message("I'm already compiling something. I'll let you know when it's done.")
            return
        chat_area.add_ai_message("Reading my own source code...")
        pipeline = self.query_one(PipelinePanel)
        pipeline.reset()
        pipeline.show()
        status = self.query_one(StatusBar)
        status.state = "self-understanding..."
        self._compile_running = True
        self.run_worker(self._self_understand_worker(), exclusive=False)

    async def _self_understand_worker(self) -> None:
        """Worker: read source via AST, compile self-context."""
        import time as _time
        _start = _time.time()
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            result = await self._bridge.compile_self_context()
            if result:
                self._handle_compile_success(result, "self-understanding", chat_area, _start)
            else:
                chat_area.add_ai_message("Self-understanding didn't produce results this time.")
        except Exception as e:
            logger.debug(f"Self-understand worker failed: {e}")
            chat_area.add_ai_message("Self-understanding ran into trouble. I'll try again later.")
        finally:
            self._compile_running = False
            status = self._safe_query(StatusBar)
            if status:
                status.state = "ready"

    async def _compile_worker(self, description: str, mode: str = "build") -> None:
        """Worker: run compilation with auto-retry on failure.

        Runs non-exclusive so the user can continue chatting while
        compilation proceeds in the background. Mother's job is to
        compile, not to report errors or block the user.
        On failure, retry up to 2 more times silently. Only surface
        failure after all attempts are exhausted.
        """
        import time as _time
        _compile_start = _time.time()
        chat_area = self._safe_query(ChatArea)
        pipeline = self._safe_query(PipelinePanel)
        if not chat_area or not pipeline:
            return

        max_attempts = 3
        last_error = ""

        # Clear analysis cache for fresh compile
        self._last_analysis_cache = {}
        self._last_diagram = None
        self._last_client_brief = None
        self._last_meeting_briefing = None

        # PII detection (#122) + Threat modeling (#124) — pre-compile safety scan
        pii_hits = []
        threats = []
        concerns = []
        try:
            from core.governor_validation import detect_pii_patterns, assess_threat_surface
            pii_hits = detect_pii_patterns(description)
            if pii_hits:
                pii_types = ", ".join(set(t for t, _ in pii_hits))
                chat_area.add_message("insight", f"PII detected in input: {pii_types}. Consider redacting before compile.")
            threats = assess_threat_surface(description)
            if threats:
                chat_area.add_message("insight", f"Threat surface: {', '.join(threats)}. Proceeding with caution.")
        except Exception as e:
            logger.debug(f"PII/threat detection skipped: {e}")

        # Ethics reasoning (#119)
        try:
            from core.governor_validation import assess_ethical_concerns
            concerns = assess_ethical_concerns(description)
            if concerns:
                chat_area.add_message("insight", f"Ethical considerations: {', '.join(concerns)}")
        except Exception as e:
            logger.debug(f"Ethics assessment skipped: {e}")

        # Risk registration (#59)
        try:
            from mother.executive import RiskRegister, classify_risk_severity
            _risk_db = self._store._path if self._store else None
            if _risk_db and (pii_hits or threats or concerns):
                _rr = RiskRegister(_risk_db)
                if pii_hits:
                    _rr.add_risk(f"PII: {', '.join(set(t for t, _ in pii_hits[:3]))}", classify_risk_severity("pii exposed"), "pii_scan")
                if threats:
                    _rr.add_risk(f"Threats: {', '.join(threats[:3])}", classify_risk_severity(' '.join(threats)), "threat_scan")
                if concerns:
                    _rr.add_risk(f"Ethical: {', '.join(concerns[:3])}", classify_risk_severity(' '.join(concerns)), "ethical_scan")
                _rr.close()
        except Exception as e:
            logger.debug(f"Risk registration skipped: {e}")

        try:
            for attempt in range(1, max_attempts + 1):
                # Reset pipeline display for each attempt
                pipeline.update_stage("Intent", "active")
                for stage in PIPELINE_STAGES[1:]:
                    pipeline.update_stage(stage, "pending")

                if attempt > 1:
                    chat_area.add_message("insight", f"Retrying compilation (attempt {attempt}/{max_attempts})")

                # Launch compile and insight streaming concurrently (C4)
                self._bridge.begin_compile()
                compile_task = asyncio.create_task(self._bridge.compile(description, mode=mode))

                stage_idx = 0
                async for insight in self._bridge.stream_insights():
                    if insight.startswith("◇ ") and stage_idx < len(PIPELINE_STAGES):
                        pipeline.update_stage(PIPELINE_STAGES[stage_idx], "complete", insight[2:][:50])
                        stage_idx += 1
                        if stage_idx < len(PIPELINE_STAGES):
                            pipeline.update_stage(PIPELINE_STAGES[stage_idx], "active")
                    display_text = insight.lstrip("◇ ").lstrip("  → ").strip()
                    if display_text:
                        chat_area.add_message("insight", display_text)

                result = await compile_task
                self._compilation_count += 1
                self._last_compile_result = result

                # Mark remaining stages complete
                for i in range(stage_idx, len(PIPELINE_STAGES)):
                    pipeline.update_stage(PIPELINE_STAGES[i], "complete")

                if result.success:
                    # Record actuator receipt for successful compile
                    try:
                        if self._actuator_store is not None:
                            from mother.actuator_receipt import create_receipt
                            _r = create_receipt(
                                "compile", True, _compile_start, _time.time(),
                                cost_usd=self._bridge.get_last_call_cost() if self._bridge else 0.0,
                                output_summary=f"{len(result.blueprint.get('components', []))} components",
                            )
                            self._actuator_store.record(_r)
                    except Exception as e:
                        logger.debug(f"Actuator receipt (compile success) skipped: {e}")

                    # Update trust accumulator
                    try:
                        if self._trust_snapshot is not None:
                            from mother.trust_accumulator import update_trust, save_trust_snapshot
                            _v = result.verification or {}
                            _ts = _v.get("overall_score", 0.0) if isinstance(_v, dict) else 0.0
                            _cg = result.context_graph or {}
                            _fid = _cg.get("closed_loop_fidelity", 0.0) or 0.0
                            self._trust_snapshot = update_trust(
                                self._trust_snapshot, success=True,
                                fidelity=_fid, trust_score=_ts,
                            )
                            save_trust_snapshot(self._trust_snapshot)
                    except Exception as e:
                        logger.debug(f"Trust accumulator update (success) skipped: {e}")

                    # --- Success path ---
                    self._handle_compile_success(
                        result, description, chat_area, _compile_start,
                    )
                    return

                # --- Failure: record error and retry ---
                last_error = result.error or "Unknown error"
                self._failure_reasons.append(str(last_error))
                logger.info(f"Compile attempt {attempt}/{max_attempts} failed: {last_error}")

                # Journal each attempt
                if self._journal:
                    self._journal.record(JournalEntry(
                        event_type="compile",
                        description=description[:200],
                        success=False,
                        cost_usd=self._bridge.get_last_call_cost(),
                        error_summary=f"attempt {attempt}/{max_attempts}: {str(last_error)[:180]}",
                        duration_seconds=_time.time() - _compile_start,
                    ))

                if attempt < max_attempts:
                    # Brief pause before retry
                    await asyncio.sleep(1.0)

            # Record actuator receipt for failed compile
            try:
                if self._actuator_store is not None:
                    from mother.actuator_receipt import create_receipt
                    _r = create_receipt(
                        "compile", False, _compile_start, _time.time(),
                        error=str(last_error)[:200] if last_error else "All attempts failed",
                    )
                    self._actuator_store.record(_r)
            except Exception as e:
                logger.debug(f"Actuator receipt (compile failure) skipped: {e}")

            # Update trust accumulator (failure)
            try:
                if self._trust_snapshot is not None:
                    from mother.trust_accumulator import update_trust, save_trust_snapshot
                    self._trust_snapshot = update_trust(
                        self._trust_snapshot, success=False,
                    )
                    save_trust_snapshot(self._trust_snapshot)
            except Exception as e:
                logger.debug(f"Trust accumulator update (failure) skipped: {e}")

            # All attempts exhausted
            chat_area.add_message(
                "mother",
                f"I wasn't able to compile that after {max_attempts} attempts. "
                f"Last issue: {last_error}",
            )
            self._speak("I wasn't able to compile that. Could you try rephrasing?")

            # Update cost
            status = self.query_one(StatusBar)
            status.session_cost = self._bridge.get_session_cost()

        except Exception as e:
            logger.error(f"Compile error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="compile"))
            pl = self._safe_query(PipelinePanel)
            if pl:
                for stage in pl.stages:
                    if stage.status == "active":
                        pl.update_stage(stage.name, "error", str(e)[:40])
                        break
            err_msg = narrate_error(e, phase="compile")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._compile_running = False
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _handle_compile_success(
        self, result, description: str, chat_area, compile_start: float,
    ) -> None:
        """Process a successful compilation result."""
        import time as _time
        import json as _json

        # --- CONTEXT mode result ---
        context_map = getattr(result, "context_map", None)
        if context_map:
            try:
                from core.context_synthesis import format_context_summary, ContextMap
                # context_map is a dict — format summary from it
                n_concepts = len(context_map.get("concepts", []))
                n_rels = len(context_map.get("relationships", []))
                n_assumptions = len(context_map.get("assumptions", []))
                n_unknowns = len(context_map.get("unknowns", []))
                conf = context_map.get("confidence", 0)
                summary_lines = [f"Context map: {n_concepts} concepts, {n_rels} relationships"]
                if n_assumptions:
                    summary_lines.append(f"  {n_assumptions} assumptions surfaced")
                if n_unknowns:
                    summary_lines.append(f"  {n_unknowns} unknowns identified")
                summary_lines.append(f"  Confidence: {conf:.0%}")
                chat_area.add_message("system", "\n".join(summary_lines))
            except Exception as e:
                logger.debug(f"Context map display skipped: {e}")
            # Skip normal blueprint display
            status = self._safe_query(StatusBar)
            if status:
                status.state = "ready"
            self._compile_running = False
            return

        # --- EXPLORE mode result ---
        exploration_map = getattr(result, "exploration_map", None)
        if exploration_map:
            try:
                n_insights = len(exploration_map.get("insights", []))
                n_questions = len(exploration_map.get("frontier_questions", []))
                n_domains = len(exploration_map.get("adjacent_domains", []))
                n_framings = len(exploration_map.get("alternative_framings", []))
                summary_lines = [f"Exploration: {n_insights} insights, {n_questions} frontier questions"]
                if n_domains:
                    domains = exploration_map.get("adjacent_domains", [])
                    summary_lines.append(f"  Adjacent domains: {', '.join(domains[:5])}")
                if n_framings:
                    summary_lines.append(f"  {n_framings} alternative framings")
                # Show top insights
                for ins in exploration_map.get("insights", [])[:3]:
                    summary_lines.append(f"  [{ins.get('category', '?')}] {ins.get('text', '')[:100]}")
                chat_area.add_message("system", "\n".join(summary_lines))
            except Exception as e:
                logger.debug(f"Exploration map display skipped: {e}")
            status = self._safe_query(StatusBar)
            if status:
                status.state = "ready"
            self._compile_running = False
            return

        components = result.blueprint.get("components", [])
        count = len(components) if isinstance(components, list) else 0

        # Trust score (C8)
        trust_score = 0.0
        verification = result.verification or {}
        if isinstance(verification, dict):
            trust_score = verification.get("overall_score", 0.0)

        # Track compilation score for paradigm shift detection (#185)
        self._compilation_scores.append(trust_score / 100.0 if trust_score > 1.0 else trust_score)

        # Transparent compilation output
        display = render_compile_output(
            verification=verification,
            components=components,
            stage_timings=result.stage_timings,
            retry_counts=result.retry_counts,
            voice_enabled=self._voice is not None,
        )

        # Minimal user-facing display: component summary + voice verdict
        # Dimension bars, gaps, timing, cost are journal-only now
        if display.component_summary:
            chat_area.add_message("system", display.component_summary)

        # Trust badge only when voice is off (Mother speaks it otherwise)
        if not display.personality_voice_only:
            from mother.widgets.trust_badge import trust_label, format_score_bar
            badge_text = f"{trust_label(trust_score)}  {format_score_bar(trust_score)}"
            chat_area.add_message("system", badge_text)

        self._speak(display.voice_verdict)
        for notable in display.voice_notable:
            self._speak(notable)

        # Surface alternatives from unresolved conflicts (#151 alternative-generating)
        try:
            cg = result.context_graph or {}
            cs = cg.get("conflict_summary", {})
            unresolved = cs.get("unresolved", [])
            if unresolved:
                alt_lines = []
                for c in unresolved[:3]:  # cap at 3
                    topic = c.get("topic", "")
                    positions = c.get("positions", [])
                    if topic and positions:
                        alts = ", ".join(str(p)[:60] for p in positions[:2])
                        alt_lines.append(f"  {topic}: {alts}")
                if alt_lines:
                    alt_text = "Alternatives considered:\n" + "\n".join(alt_lines)
                    chat_area.add_message("system", alt_text)
        except Exception as e:
            logger.debug(f"Alternatives display skipped: {e}")

        # ROI estimation (#112 ROI-calculating)
        try:
            comp_count = 0
            if isinstance(result.blueprint, dict):
                comps = result.blueprint.get("components", [])
                comp_count = len(comps) if isinstance(comps, list) else 0
            if comp_count > 0:
                est_manual_hours = comp_count * 2.0  # ~2h per component manual
                compile_cost = result.cost if hasattr(result, "cost") and result.cost else 0.0
                manual_cost_est = est_manual_hours * 75.0  # $75/hr baseline
                if manual_cost_est > 0 and compile_cost > 0:
                    ratio = manual_cost_est / compile_cost
                    roi_text = (
                        f"ROI estimate: {comp_count} components × ~2h manual = "
                        f"~{est_manual_hours:.0f}h (~${manual_cost_est:.0f}) vs "
                        f"${compile_cost:.4f} compile cost ({ratio:.0f}× leverage)"
                    )
                    chat_area.add_message("system", roi_text)
        except Exception as e:
            logger.debug(f"ROI estimation skipped: {e}")

        # Architecture diagram generation (#143 visual-first)
        try:
            from mother.diagrams import component_tree
            _tree = component_tree(result.blueprint)
            if _tree:
                self._last_diagram = _tree
                chat_area.add_message("system", _tree)
        except Exception as e:
            logger.debug(f"Diagram generation skipped: {e}")

        # Client brief generation (#127 client-facing-capable)
        try:
            from mother.client_docs import generate_client_brief
            _brief = generate_client_brief(result.blueprint, verification=verification)
            if _brief:
                self._last_client_brief = _brief
        except Exception as e:
            logger.debug(f"Client brief generation skipped: {e}")

        # Meeting prep generation (#134 meeting-preparing)
        try:
            from mother.meeting_prep import generate_briefing
            _components = result.blueprint.get("components", [])
            _risks = []
            if isinstance(verification, dict):
                for k, v in verification.items():
                    if isinstance(v, dict) and v.get("score", 100) < 60:
                        _risks.append(f"{k}: {v.get('score', 0):.0f}/100")
            _briefing = generate_briefing(
                description,
                components=[c.get("name", "") for c in _components if isinstance(c, dict)],
                risks=_risks,
            )
            if _briefing:
                self._last_meeting_briefing = _briefing
        except Exception as e:
            logger.debug(f"Meeting prep generation skipped: {e}")

        # Recompute senses after compile
        self._update_senses()

        # Personality bite on success (C5)
        bite = inject_personality_bite(
            self._config.personality, "compile_success",
            posture=self._current_posture, trust_score=trust_score,
        )
        if bite:
            if display.personality_voice_only:
                self._speak(bite)
            else:
                chat_area.add_ai_message(bite)

        # Low trust bite
        if trust_score < 40:
            low_bite = inject_personality_bite(self._config.personality, "low_trust")
            if low_bite:
                if display.personality_voice_only:
                    self._speak(low_bite)
                else:
                    chat_area.add_ai_message(low_bite)

            desc = result.blueprint.get("description", "") if isinstance(result.blueprint, dict) else ""
            self._deep_think_subject = f"Low trust ({trust_score:.0f}%) on: {desc[:80]}"
            self._deep_think_set_time = _time.time()

        # Journal: record successful compile
        if self._journal:
            weakest_dim = ""
            weakest_score = 101.0
            dim_scores_dict = {}
            for k, v in (verification if isinstance(verification, dict) else {}).items():
                if k in ("overall_score", "status", "verification_mode"):
                    continue
                sv = None
                if isinstance(v, (int, float)):
                    sv = float(v)
                elif isinstance(v, dict) and "score" in v:
                    sv = float(v["score"])
                if sv is not None:
                    dim_scores_dict[k] = sv
                    if sv < weakest_score:
                        weakest_dim = k
                        weakest_score = sv
            self._journal.record(JournalEntry(
                event_type="compile",
                description=description[:200],
                success=True,
                trust_score=trust_score,
                component_count=count,
                cost_usd=self._bridge.get_last_call_cost(),
                domain=result.blueprint.get("domain", ""),
                duration_seconds=_time.time() - compile_start,
                weakest_dimension=weakest_dim,
                dimension_scores=_json.dumps(dim_scores_dict) if dim_scores_dict else "",
            ))

        # --- World grid: merge compilation results ---
        try:
            if self._world_grid is not None and self._bridge is not None:
                project_id = result.blueprint.get("domain", "session") if isinstance(result.blueprint, dict) else "session"
                merged = self._bridge.merge_compilation_into_world(
                    self._world_grid, result, project_id=project_id,
                )
                if merged > 0:
                    logger.debug(f"World grid: merged {merged} cell(s) from compilation")
                    self._bridge.save_world_grid(self._world_grid)
        except Exception as e:
            logger.debug(f"World grid compilation merge skipped: {e}")

        # Update cost
        status = self.query_one(StatusBar)
        status.session_cost = self._bridge.get_session_cost()

        # Offer code generation for high-trust blueprints
        if trust_score >= 60:
            chat_area.add_message(
                "system",
                "Ready to generate code? Say 'build it' or 'generate the code'.",
            )

    def _run_diagram(self, format_hint: str = "tree") -> None:
        """Generate and display architecture diagram."""
        chat_area = self.query_one(ChatArea)
        result = self._last_compile_result
        if not result or not getattr(result, "blueprint", None):
            chat_area.add_ai_message("No compilation to diagram yet. Compile something first.")
            return
        try:
            from mother.diagrams import translate_blueprint
            diagram = translate_blueprint(result.blueprint, target_format=format_hint)
            if diagram:
                chat_area.add_message("system", diagram)
        except Exception as e:
            logger.debug(f"Diagram render skipped: {e}")
            chat_area.add_ai_message("Couldn't render the diagram.")

    def _run_client_brief(self) -> None:
        """Display cached client brief or generate from last compile."""
        chat_area = self.query_one(ChatArea)
        if self._last_client_brief:
            try:
                from mother.client_docs import format_client_markdown
                chat_area.add_message("system", format_client_markdown(self._last_client_brief))
            except Exception as e:
                logger.debug(f"Client brief display skipped: {e}")
            return
        result = self._last_compile_result
        if not result or not getattr(result, "blueprint", None):
            chat_area.add_ai_message("No compilation available. Compile something first.")
            return
        try:
            from mother.client_docs import generate_client_brief, format_client_markdown
            brief = generate_client_brief(result.blueprint, verification=result.verification)
            if brief:
                self._last_client_brief = brief
                chat_area.add_message("system", format_client_markdown(brief))
        except Exception as e:
            logger.debug(f"Client brief skipped: {e}")

    def _run_meeting_prep(self) -> None:
        """Display cached meeting briefing or generate from last compile."""
        chat_area = self.query_one(ChatArea)
        if self._last_meeting_briefing:
            try:
                from mother.meeting_prep import format_briefing_markdown
                chat_area.add_message("system", format_briefing_markdown(self._last_meeting_briefing))
            except Exception as e:
                logger.debug(f"Meeting briefing display skipped: {e}")
            return
        result = self._last_compile_result
        if not result or not getattr(result, "blueprint", None):
            chat_area.add_ai_message("No compilation available. Compile something first.")
            return
        try:
            from mother.meeting_prep import generate_briefing, format_briefing_markdown
            _bp = result.blueprint
            _comps = _bp.get("components", []) if isinstance(_bp, dict) else []
            briefing = generate_briefing(
                _bp.get("description", ""),
                components=[c.get("name", "") for c in _comps if isinstance(c, dict)],
            )
            if briefing:
                self._last_meeting_briefing = briefing
                chat_area.add_message("system", format_briefing_markdown(briefing))
        except Exception as e:
            logger.debug(f"Meeting prep skipped: {e}")

    # --- Weekly build governance handlers ---

    def _run_approve_builds(self, arg: str) -> None:
        """Approve specific builds from weekly briefing by goal IDs."""
        try:
            chat_area = self.query_one(ChatArea)
            goal_ids = [gid.strip() for gid in arg.split(",") if gid.strip()]
            if not goal_ids:
                chat_area.add_ai_message("No goal IDs provided.")
                return
            success = self._bridge.approve_weekly_builds(goal_ids)
            if success:
                chat_area.add_message("system", f"Approved {len(goal_ids)} build(s): {', '.join(goal_ids)}")
            else:
                chat_area.add_ai_message("Failed to approve builds.")
        except Exception as e:
            logger.debug(f"Approve builds skipped: {e}")

    def _run_approve_all_builds(self) -> None:
        """Approve all builds from current weekly briefing."""
        try:
            chat_area = self.query_one(ChatArea)
            briefing_data = self._bridge.get_weekly_briefing()
            if not briefing_data or not briefing_data.get("items"):
                chat_area.add_ai_message("No briefing available to approve.")
                return
            goal_ids = [item["goal_id"] for item in briefing_data["items"]]
            success = self._bridge.approve_weekly_builds(goal_ids)
            if success:
                chat_area.add_message("system", f"Approved all {len(goal_ids)} build(s).")
            else:
                chat_area.add_ai_message("Failed to approve builds.")
        except Exception as e:
            logger.debug(f"Approve all builds skipped: {e}")

    def _run_reject_builds(self, arg: str) -> None:
        """Reject/dismiss specific goals from weekly briefing."""
        try:
            chat_area = self.query_one(ChatArea)
            goal_ids = [gid.strip() for gid in arg.split(",") if gid.strip()]
            if not goal_ids:
                chat_area.add_ai_message("No goal IDs provided.")
                return
            from mother.goals import GoalStore
            db_path = Path.home() / ".motherlabs" / "history.db"
            gs = GoalStore(db_path)
            for gid in goal_ids:
                try:
                    gs.update_status(int(gid), "dismissed", progress_note="Rejected in weekly briefing")
                except Exception as e:
                    logger.debug(f"Goal rejection skipped for {gid}: {e}")
            gs.close()
            chat_area.add_message("system", f"Rejected {len(goal_ids)} goal(s): {', '.join(goal_ids)}")
        except Exception as e:
            logger.debug(f"Reject builds skipped: {e}")

    def _run_show_weekly_briefing(self) -> None:
        """Display the current weekly briefing in chat."""
        try:
            chat_area = self.query_one(ChatArea)
            briefing_data = self._bridge.get_weekly_briefing()
            if not briefing_data:
                chat_area.add_ai_message("No weekly briefing available yet.")
                return
            chat_area.add_message("system", briefing_data["briefing_markdown"])
        except Exception as e:
            logger.debug(f"Show weekly briefing skipped: {e}")

    def _run_show_build_report(self) -> None:
        """Display the last build window report in chat."""
        try:
            chat_area = self.query_one(ChatArea)
            report_data = self._bridge.get_build_report()
            if not report_data:
                chat_area.add_ai_message("No build reports available yet.")
                return
            chat_area.add_message("system", report_data["report_markdown"])
        except Exception as e:
            logger.debug(f"Show build report skipped: {e}")

    def _run_build(self, description: str) -> None:
        """Start a full build pipeline."""
        chat_area = self.query_one(ChatArea)

        # Personality bite at build start
        bite = inject_personality_bite(
            self._config.personality, "build_start", posture=self._current_posture
        )
        start_msg = bite or f"Building: {description}"
        chat_area.add_ai_message(start_msg)
        if bite:
            self._speak(bite)

        # Show pipeline
        pipeline = self.query_one(PipelinePanel)
        pipeline.reset()
        pipeline.show()

        status = self.query_one(StatusBar)
        status.state = "building..."

        self.run_worker(self._build_worker(description), exclusive=True)

    async def _build_worker(self, description: str) -> None:
        """Worker: run full build pipeline with live progress streaming."""
        import time as _time
        _build_start = _time.time()
        chat_area = self._safe_query(ChatArea)
        pipeline = self._safe_query(PipelinePanel)
        if not chat_area or not pipeline:
            return
        try:
            output_dir = self._config.output_dir

            # Launch build and phase streaming concurrently
            self._bridge.begin_build()
            build_task = asyncio.create_task(
                self._bridge.build(description, output_dir=output_dir)
            )

            # Map orchestrator phases to pipeline display
            _PHASE_LABELS = {
                "quality": "Intent",
                "enrich": "Intent",
                "compile": "Persona",
                "emit": "Entity",
                "write": "Process",
                "build": "Synthesis",
            }
            last_phase = None
            stage_idx = 0

            async for phase, detail in self._bridge.stream_build_phases():
                # Update pipeline panel
                stage_name = _PHASE_LABELS.get(phase, "Synthesis")
                if phase != last_phase:
                    # Mark previous stage complete
                    if stage_idx < len(PIPELINE_STAGES) and last_phase is not None:
                        pipeline.update_stage(
                            PIPELINE_STAGES[stage_idx], "complete"
                        )
                        stage_idx += 1
                    # Mark new stage active
                    if stage_idx < len(PIPELINE_STAGES):
                        pipeline.update_stage(
                            PIPELINE_STAGES[stage_idx], "active", detail[:50]
                        )
                    last_phase = phase

                # Show build phase bites for key phases
                if phase == "emit":
                    bite = inject_personality_bite(
                        self._config.personality, "build_emit"
                    )
                    if bite:
                        chat_area.add_ai_message(bite)
                elif phase == "build" and "validat" in detail.lower():
                    bite = inject_personality_bite(
                        self._config.personality, "build_validate"
                    )
                    if bite:
                        chat_area.add_ai_message(bite)
                elif phase == "build" and "fix" in detail.lower():
                    bite = inject_personality_bite(
                        self._config.personality, "build_fix"
                    )
                    if bite:
                        chat_area.add_ai_message(bite)

            result = await build_task
            self._compilation_count += 1

            # Mark remaining stages complete
            for i in range(stage_idx, len(PIPELINE_STAGES)):
                pipeline.update_stage(PIPELINE_STAGES[i], "complete")

            # Recompute senses after build
            self._update_senses()

            if result.success and result.project_manifest:
                manifest = result.project_manifest
                path = manifest.project_dir

                # Shorten path with ~
                home = str(Path.home())
                display_path = path
                if display_path.startswith(home):
                    display_path = "~" + display_path[len(home):]

                # File and line counts
                file_count = len(manifest.files_written)
                total_lines = manifest.total_lines
                entry = manifest.entry_point

                # Trust info
                trust_line = ""
                if result.trust:
                    score = result.trust.overall_score
                    badge = result.trust.verification_badge
                    trust_line = f"Trust: {badge} {score:.0f}%"

                # Build iteration info
                build_line = "Build: clean first pass"
                if result.build_result:
                    br = result.build_result
                    iters = len(br.iterations)
                    if br.components_fixed:
                        fixed_names = ", ".join(br.components_fixed)
                        build_line = f"Build: {iters} iteration{'s' if iters != 1 else ''}, fixed {fixed_names}"
                    elif iters > 1:
                        build_line = f"Build: {iters} iterations"

                # Infer project name from path
                project_name = Path(path).name

                # Assemble result display
                lines = [
                    f"Project: {project_name}",
                    f"Path: {display_path}",
                    f"Files: {file_count} written ({total_lines} lines)",
                    f"Entry: python3 {entry}",
                ]
                if trust_line:
                    lines.append(trust_line)
                lines.append(build_line)

                chat_area.add_ai_message("\n".join(lines))
                self._speak("Project built and ready.")

                # Auto-launch
                self._last_project_path = path
                self._last_entry_point = entry or "main.py"
                self._run_launch()

                # Build success bite — posture-aware
                bite = inject_personality_bite(
                    self._config.personality, "build_success", posture=self._current_posture
                )
                if bite:
                    chat_area.add_ai_message(bite)
                    self._speak(bite)

                # Journal: record successful build
                if self._journal:
                    b_trust = result.trust.overall_score if result.trust else 0.0
                    b_components = len(manifest.files_written)
                    self._journal.record(JournalEntry(
                        event_type="build",
                        description=description[:200],
                        success=True,
                        trust_score=b_trust,
                        component_count=b_components,
                        cost_usd=self._bridge.get_last_call_cost(),
                        duration_seconds=_time.time() - _build_start,
                        project_path=path,
                    ))

                # Auto-register as tool
                tool_info = await self._bridge.register_build_as_tool(result, description)
                if tool_info:
                    chat_area.add_ai_message(f"Registered as tool: {tool_info['name']}")
                    self._speak("Registered as a tool.")

                # Auto-push to GitHub if enabled
                if self._config.auto_push_enabled:
                    repo_dir = str(Path(__file__).resolve().parent.parent)
                    push_result = await self._bridge.github_push(repo_dir=repo_dir)
                    if push_result["success"]:
                        chat_area.add_ai_message("Pushed to GitHub.")

                # Auto-tweet if enabled
                if self._config.auto_tweet_enabled and tool_info:
                    tool_name = tool_info.get("name", "project")
                    trust = int(result.trust.overall_score) if result.trust else 0
                    components = len(manifest.files_written)
                    tweet_text = f"Just shipped {tool_name}: {description[:100]} — {components} components, {trust}% trust."
                    if len(tweet_text) <= 280:
                        tweet_result = await self._bridge.tweet(tweet_text)
                        if tweet_result["success"]:
                            url = tweet_result.get("tweet_url", "")
                            chat_area.add_ai_message(f"Posted: {url}")

                # Auto-publish project to GitHub (new repo for emitted project)
                if self._config.auto_publish_projects and manifest:
                    try:
                        pub_name = tool_info.get("name", "") if tool_info else name
                        pub = await self._bridge.publish_project(
                            project_dir=path,
                            name=pub_name or name,
                            description=description,
                        )
                        if pub["success"]:
                            chat_area.add_ai_message(f"Published to GitHub: {pub['repo_url']}")
                            repo_url = pub["repo_url"]

                            # Announce across all enabled social platforms
                            b_trust = int(result.trust.overall_score) if result.trust else 0
                            b_components = len(manifest.files_written)
                            ann = await self._bridge.announce_build(
                                name=pub["repo_name"],
                                description=description,
                                repo_url=repo_url,
                                components=b_components,
                                trust=b_trust,
                            )
                            if ann.get("enqueued", 0) > 0:
                                chat_area.add_ai_message(
                                    f"Announced on {', '.join(ann['platforms'])}."
                                )
                        elif pub.get("error"):
                            logger.debug(f"Project publish skipped: {pub['error']}")
                    except Exception as e:
                        logger.debug(f"Project publish skipped: {e}")

            else:
                error = result.error or "Unknown error"
                chat_area.add_ai_message(f"Build did not complete: {error}")
                self._speak("Build did not complete.")
                # Journal: record failed build
                if self._journal:
                    self._journal.record(JournalEntry(
                        event_type="build",
                        description=description[:200],
                        success=False,
                        cost_usd=self._bridge.get_last_call_cost(),
                        error_summary=str(error)[:200],
                        duration_seconds=_time.time() - _build_start,
                    ))

            status = self.query_one(StatusBar)
            status.session_cost = self._bridge.get_session_cost()

        except Exception as e:
            logger.error(f"Build error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="build"))
            pl = self._safe_query(PipelinePanel)
            if pl:
                for stage in pl.stages:
                    if stage.status == "active":
                        pl.update_stage(stage.name, "error", str(e)[:40])
                        break
            err_msg = narrate_error(e, phase="build")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Idea journal ---

    def _run_add_idea(self, description: str) -> None:
        """Record an idea via bridge. Silent — Mother already spoke via VOICE tag."""
        self.run_worker(self._add_idea_worker(description), exclusive=False)

    async def _add_idea_worker(self, description: str) -> None:
        """Worker: record idea in journal."""
        try:
            db_path = self._store._path if self._store else None
            if db_path and self._bridge:
                await self._bridge.add_idea(db_path, description)
                logger.debug(f"Idea recorded: {description[:60]}")
        except Exception as e:
            logger.debug(f"Add idea error: {e}")

    def _run_list_ideas(self) -> None:
        """List pending ideas."""
        self.run_worker(self._list_ideas_worker(), exclusive=False)

    async def _list_ideas_worker(self) -> None:
        """Worker: list pending ideas."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            db_path = self._store._path if self._store else None
            if not db_path:
                return
            ideas = await self._bridge.get_pending_ideas(db_path, limit=10)
            if not ideas:
                chat_area.add_ai_message("No pending ideas.")
                return
            lines = [f"{len(ideas)} pending idea{'s' if len(ideas) != 1 else ''}:"]
            for idea in ideas:
                priority_tag = f" [{idea['priority']}]" if idea['priority'] != "normal" else ""
                lines.append(f"  #{idea['idea_id']}: {idea['description']}{priority_tag}")
            chat_area.add_ai_message("\n".join(lines))
        except Exception as e:
            logger.debug(f"List ideas error: {e}")

    # --- Daemon (autonomous self-improvement loop) ---

    def _start_daemon(self) -> None:
        """Instantiate and start the autonomous daemon.

        The daemon runs background loops:
        - Health checks (infra healing)
        - Compile queue processing
        - Scheduler (goal-driven self-improvement every 10 min)

        For [SELF-IMPROVEMENT] goals, the compile function compiles the
        goal into a plan and dispatches self-build steps via Claude Code.
        """
        try:
            from mother.daemon import DaemonMode, DaemonConfig
            from pathlib import Path as _Path

            config = DaemonConfig(
                health_check_interval=getattr(self._config, "daemon_health_check_interval", 300),
                max_queue_size=getattr(self._config, "daemon_max_queue_size", 10),
                auto_heal=getattr(self._config, "daemon_auto_heal", True),
                idle_shutdown_hours=getattr(self._config, "daemon_idle_shutdown_hours", 0),
            )

            async def _daemon_compile_fn(input_text: str, domain: str) -> Any:
                """Bridge between daemon queue and Mother's compile/self-build systems."""
                if not self._bridge:
                    return None

                is_self_improvement = input_text.startswith("[SELF-IMPROVEMENT]")
                is_self_compile = input_text.startswith("[SELF-COMPILE]")

                if is_self_improvement:
                    # Strip tag, compile goal into plan, then dispatch self-build
                    goal_desc = input_text.replace("[SELF-IMPROVEMENT]", "").strip()
                    return await self._daemon_self_improve(goal_desc)
                elif is_self_compile:
                    # Self-compile: Mother compiles her own description
                    return await self._bridge.compile(
                        "Describe the complete architecture, capabilities, and current state "
                        "of this semantic compilation system — its engine, stages, kernel, "
                        "feedback loops, and self-improvement mechanisms."
                    )
                else:
                    # Regular compilation
                    return await self._bridge.compile(input_text)

            self._daemon = DaemonMode(
                config=config,
                config_dir=_Path.home() / ".motherlabs",
                compile_fn=_daemon_compile_fn,
            )

            # Configure weekly build governance
            if getattr(self._config, "weekly_build_enabled", True):
                async def _on_briefing(briefing):
                    """Display weekly briefing in chat when generated."""
                    try:
                        from mother.scheduler import format_briefing_markdown
                        chat_area = self.query_one(ChatArea)
                        md = format_briefing_markdown(briefing)
                        chat_area.add_message("system", md)
                    except Exception as e:
                        logger.debug(f"Briefing display skipped: {e}")

                async def _on_report(report):
                    """Display build report in chat when window completes."""
                    try:
                        from mother.scheduler import format_report_markdown
                        chat_area = self.query_one(ChatArea)
                        md = format_report_markdown(report)
                        chat_area.add_message("system", md)
                    except Exception as e:
                        logger.debug(f"Report display skipped: {e}")

                self._daemon.configure_weekly_build(
                    enabled=True,
                    briefing_day=getattr(self._config, "weekly_briefing_day", 6),
                    briefing_hour=getattr(self._config, "weekly_briefing_hour", 10),
                    window_start=getattr(self._config, "build_window_start_hour", 22),
                    window_end=getattr(self._config, "build_window_end_hour", 6),
                    window_day=getattr(self._config, "build_window_day", 6),
                    max_per_window=getattr(self._config, "build_max_per_window", 10),
                    briefing_callback=_on_briefing,
                    report_callback=_on_report,
                )

            async def _boot_daemon():
                await self._daemon.start()
                logger.info("Daemon mode started — autonomous self-improvement active")

            self.run_worker(_boot_daemon(), exclusive=False)

        except Exception as e:
            logger.debug(f"Daemon startup skipped: {e}")

    async def _daemon_self_improve(self, goal_desc: str) -> Any:
        """Execute a self-improvement goal: compile → plan → self-build.

        This is the L3 loop actually firing:
        1. Compile the goal into a blueprint
        2. Extract self-build steps with enriched Claude Code prompts
        3. Execute each self-build step via Claude Code CLI
        4. Update grid cells on success/failure
        5. Return the compilation result
        """
        from pathlib import Path as _Path

        # Step 1: Compile the goal to get a plan
        result = await self._bridge.compile(goal_desc)
        if not result or not getattr(result, "success", False):
            logger.warning(f"Self-improvement compile failed for: {goal_desc[:80]}")
            return result

        # Step 2: Build a rich self-build prompt from the blueprint
        try:
            from mother.self_build_planner import (
                blueprint_to_build_context,
                goal_to_build_intent,
                assemble_self_build_prompt,
            )
            repo_dir = _Path(__file__).resolve().parent.parent
            blueprint = getattr(result, "blueprint", {})

            bp_context = blueprint_to_build_context(blueprint)
            build_intent = goal_to_build_intent(
                {"description": goal_desc, "category": "confidence"},
                blueprint=blueprint,
            )
            spec = assemble_self_build_prompt(
                build_intent=build_intent,
                repo_dir=repo_dir,
                blueprint_context=bp_context,
            )
            rich_prompt = spec.prompt
            target_postcodes = spec.target_postcodes
        except Exception as e:
            logger.debug(f"Self-build prompt assembly skipped: {e}")
            # Fall back to raw goal as prompt
            rich_prompt = goal_desc
            target_postcodes = ()

        # Step 3: Execute self-build via Claude Code
        try:
            repo_dir_str = str(_Path(__file__).resolve().parent.parent)
            claude_path = getattr(self._config, "claude_code_path", "")
            budget = getattr(self._config, "claude_code_budget", 3.0)

            build_result = await self._bridge.self_build(
                prompt=rich_prompt,
                repo_dir=repo_dir_str,
                max_budget_usd=budget,
                claude_path=claude_path,
            )

            # Step 4: Update grid cells
            if target_postcodes and self._bridge:
                try:
                    await self._bridge.update_grid_after_build(
                        target_postcodes=target_postcodes,
                        success=build_result.get("success", False),
                        build_description=goal_desc[:120],
                    )
                except Exception as e:
                    logger.debug(f"Grid update after daemon self-build skipped: {e}")

            success = build_result.get("success", False)
            if success:
                logger.info(f"Daemon self-build SUCCESS: {goal_desc[:80]}")
            else:
                logger.warning(f"Daemon self-build FAILED: {build_result.get('error', 'unknown')}")

            # Stamp actual build outcome onto CompileResult so feedback loop
            # sees the real success/failure (not just "compilation produced components")
            result.success = success
            if not success:
                result.error = build_result.get("error", "self-build failed")

        except Exception as e:
            logger.warning(f"Daemon self-build execution failed: {e}")
            result.success = False
            result.error = str(e)

        return result

    # --- Self-build ---

    def _run_self_build(self, description: str) -> None:
        """Start a self-build via Claude Code CLI."""
        chat_area = self.query_one(ChatArea)

        if not getattr(self._config, "claude_code_enabled", False):
            msg = "Self-modification isn't enabled. Turn it on in /settings."
            chat_area.add_ai_message(msg)
            self._speak(msg)
            if self._store:
                self._store.add_message("assistant", msg)
            return

        bite = inject_personality_bite(
            self._config.personality, "self_build_start", posture=self._current_posture
        )
        if bite:
            chat_area.add_ai_message(bite)
            self._speak(bite)

        status = self.query_one(StatusBar)
        status.state = "self-building..."

        self.run_worker(self._self_build_worker(description), exclusive=True)

    async def _self_build_worker(self, description: str) -> None:
        """Worker: orchestrate self-build via bridge with live streaming."""
        import time as _time
        _build_start = _time.time()
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            repo_dir = str(Path(__file__).resolve().parent.parent)
            claude_path = getattr(self._config, "claude_code_path", "")
            budget = getattr(self._config, "claude_code_budget", 3.0)

            # Launch build and streaming concurrently
            self._bridge.begin_self_build()
            build_task = asyncio.create_task(
                self._bridge.self_build(
                    prompt=description,
                    repo_dir=repo_dir,
                    max_budget_usd=budget,
                    claude_path=claude_path,
                )
            )

            # Consume streaming events and display in chat
            _streaming = False  # Whether we have an active streaming message
            _last_tool = ""
            try:
                async for event in self._bridge.stream_self_build_events():
                    try:
                        etype = event.get("type", event.get("_type", ""))

                        # Phase markers (snapshot, testing, commit)
                        if etype == "phase":
                            if _streaming:
                                chat_area.finish_streaming_message("")
                                _streaming = False
                            phase = event.get("phase", "")
                            detail = event.get("detail", "")
                            if detail:
                                chat_area.add_message("system", f"> {detail}")
                            continue

                        # Assistant text content (reasoning)
                        if etype == "assistant":
                            content_blocks = event.get("message", {}).get("content", [])
                            for block in content_blocks:
                                if block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text.strip():
                                        if _streaming:
                                            chat_area.finish_streaming_message("")
                                            _streaming = False
                                        # Show brief reasoning (truncate long blocks)
                                        display = text[:300]
                                        if len(text) > 300:
                                            display += "..."
                                        chat_area.add_message("system", display)
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_input = block.get("input", {})
                                    # Format concise tool description
                                    target = ""
                                    if tool_name in ("Read", "Glob"):
                                        target = tool_input.get("file_path", tool_input.get("pattern", ""))
                                    elif tool_name == "Edit":
                                        target = tool_input.get("file_path", "")
                                    elif tool_name == "Write":
                                        target = tool_input.get("file_path", "")
                                    elif tool_name == "Grep":
                                        target = tool_input.get("pattern", "")
                                    elif tool_name == "Bash":
                                        cmd = tool_input.get("command", "")
                                        target = cmd[:80]
                                    else:
                                        target = str(tool_input)[:60]
                                    line = f"> {tool_name}"
                                    if target:
                                        line += f" {target}"
                                    # Deduplicate consecutive identical tool lines
                                    if line != _last_tool:
                                        if _streaming:
                                            chat_area.finish_streaming_message("")
                                            _streaming = False
                                        chat_area.add_message("system", line)
                                        _last_tool = line
                            continue

                        # Content block delta — streaming text
                        if etype == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    if not _streaming:
                                        chat_area.begin_streaming_message("system")
                                        _streaming = True
                                    chat_area.append_streaming_text(text)
                            continue

                    except Exception as e:
                        logger.debug(f"Self-build stream event display skipped: {e}")
            finally:
                # Always clean up dangling streaming message
                if _streaming:
                    chat_area.finish_streaming_message("")

            result = await build_task

            duration = _time.time() - _build_start
            self._update_senses()

            # Record actuator receipt for self-build
            try:
                if self._actuator_store is not None:
                    from mother.actuator_receipt import create_receipt
                    _r = create_receipt(
                        "self_build", bool(result.get("success")),
                        _build_start, _time.time(),
                        cost_usd=result.get("cost_usd", 0.0),
                        output_summary=description[:80],
                        error=result.get("error", "")[:200] if not result.get("success") else "",
                    )
                    self._actuator_store.record(_r)
            except Exception as e:
                logger.debug(f"Actuator receipt (self_build) skipped: {e}")

            if result.get("success"):
                self._last_self_build_desc = description[:120]

                # Store build delta for body map
                diff_stats = result.get("diff_stats", {})
                if diff_stats:
                    from mother.introspection import BuildDelta
                    self._last_build_delta = BuildDelta(
                        files_modified=diff_stats.get("files_modified", 0),
                        files_added=diff_stats.get("files_added", 0),
                        lines_added=diff_stats.get("lines_added", 0),
                        lines_removed=diff_stats.get("lines_removed", 0),
                        modules_touched=diff_stats.get("modules_touched", []),
                    )

                # Re-scan topology (codebase may have changed)
                self._codebase_topology = None
                await self._scan_topology_once()

                # Close grid loop: boost confidence for targeted postcodes
                if getattr(self, '_current_build_postcodes', None) and self._bridge:
                    try:
                        _grid_updated = await self._bridge.update_grid_after_build(
                            target_postcodes=self._current_build_postcodes,
                            success=True,
                            build_description=description[:200],
                        )
                        if _grid_updated:
                            logger.info(f"Grid feedback: boosted {_grid_updated} cells after build")
                    except Exception as e:
                        logger.debug(f"Grid feedback skipped: {e}")

                # Log system summary post-build (#63 Self-documenting)
                if self._bridge:
                    _summary = self._bridge.generate_system_summary()
                    if _summary:
                        logger.info(f"Post-build summary: {_summary}")

                # Finalize idea lifecycle: mark as done if auto-triggered
                db_path = self._store._path if self._store else None
                if db_path and self._pending_idea_id:
                    await self._bridge.update_idea_status(
                        db_path, self._pending_idea_id, "done",
                        outcome=f"Success. Cost: ${result.get('cost_usd', 0):.2f}",
                    )
                    self._pending_idea_id = None
                elif db_path:
                    # User-triggered: record as done in idea journal
                    await self._bridge.add_idea(
                        db_path, description,
                        source_context="self-build success",
                        priority="normal",
                    )

                bite = inject_personality_bite(
                    self._config.personality, "self_build_success",
                    posture=self._current_posture,
                )
                msg = bite or "Self-modification complete. Tests pass."
                cost_info = f"Cost: ${result.get('cost_usd', 0):.2f}, {duration:.0f}s."
                # Show push/commit status
                _extra_status = []
                if result.get("pushed"):
                    _extra_status.append("Pushed.")
                if result.get("commit_hash"):
                    _extra_status.append(f"Commit: {result['commit_hash'][:10]}")
                _status_line = " ".join(_extra_status)
                if _status_line:
                    chat_area.add_ai_message(f"{msg}\n{cost_info} {_status_line}")
                else:
                    chat_area.add_ai_message(f"{msg}\n{cost_info}")
                chat_area.add_ai_message("Restarting to load changes...")
                self._speak("Restarting.")
                if self._store:
                    self._store.add_message("assistant", f"{msg} {cost_info} {_status_line}")
                await asyncio.sleep(0.5)
                self.app.action_restart()  # release_lock() → os.execv()
                return  # unreachable after execv, but satisfies control flow
            else:
                error = result.get("error", "Unknown error")
                rolled_back = result.get("rolled_back", False)

                # Close grid loop: penalize confidence for targeted postcodes
                if getattr(self, '_current_build_postcodes', None) and self._bridge:
                    try:
                        await self._bridge.update_grid_after_build(
                            target_postcodes=self._current_build_postcodes,
                            success=False,
                            build_description=f"FAILED: {error[:180]}",
                        )
                    except Exception as e:
                        logger.debug(f"Grid feedback (failure) skipped: {e}")

                # Finalize idea lifecycle: mark as dismissed on failure
                db_path = self._store._path if self._store else None
                if db_path and self._pending_idea_id:
                    await self._bridge.update_idea_status(
                        db_path, self._pending_idea_id, "dismissed",
                        outcome=f"Failed: {error[:120]}",
                    )
                    self._pending_idea_id = None

                bite = inject_personality_bite(
                    self._config.personality, "self_build_failed",
                    posture=self._current_posture,
                )
                msg = bite or "Self-modification failed."
                detail = f"Error: {error}"
                if rolled_back:
                    detail += " Rolled back to previous state."
                chat_area.add_ai_message(f"{msg}\n{detail}")
                self._speak(msg)
                if self._store:
                    self._store.add_message("assistant", f"{msg} {detail}")

                # Self-repair: record failure as improvement goal
                try:
                    _db = self._store._path if self._store else None
                    if _db and self._bridge:
                        self._bridge.record_task_failure(
                            _db, "self_build", description, error,
                        )
                except Exception as e:
                    logger.debug(f"Self-repair goal (self_build) skipped: {e}")

        except Exception as e:
            logger.error(f"Self-build error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="self_build"))
            err_msg = narrate_error(e, phase="self_build")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
            # Clean up idea lifecycle on exception
            if self._pending_idea_id and self._store:
                try:
                    await self._bridge.update_idea_status(
                        self._store._path, self._pending_idea_id, "dismissed",
                        outcome=f"Exception: {str(e)[:120]}",
                    )
                except Exception as e2:
                    logger.debug(f"Idea status update on failure skipped: {e2}")
                self._pending_idea_id = None
            # Self-repair: record crash as improvement goal
            try:
                _db = self._store._path if self._store else None
                if _db and self._bridge:
                    self._bridge.record_task_failure(
                        _db, "self_build", description, str(e),
                    )
            except Exception as exc:
                logger.debug(f"Self-repair goal (self_build crash) skipped: {exc}")
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Code task (user project) ---

    def _run_code_task(self, description: str) -> None:
        """Start a code task via Claude Code CLI targeting the user's project."""
        chat_area = self.query_one(ChatArea)

        if not getattr(self._config, "claude_code_enabled", False):
            msg = "Code writing isn't enabled. Turn on Claude Code in /settings."
            chat_area.add_ai_message(msg)
            self._speak(msg)
            if self._store:
                self._store.add_message("assistant", msg)
            return

        bite = inject_personality_bite(
            self._config.personality, "code_task_start", posture=self._current_posture
        )
        if bite:
            chat_area.add_ai_message(bite)
            self._speak(bite)

        status = self.query_one(StatusBar)
        status.state = "coding..."

        self.run_worker(self._code_task_worker(description), exclusive=True)

    async def _code_task_worker(self, description: str) -> None:
        """Worker: write code in user's project via bridge.code_task()."""
        import time as _time
        _task_start = _time.time()
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            # Determine target directory
            target_dir = self._last_project_path
            if not target_dir:
                target_dir = str(Path.home() / "Documents" / "mother-projects")
                Path(target_dir).mkdir(parents=True, exist_ok=True)

            claude_path = getattr(self._config, "claude_code_path", "")
            budget = getattr(self._config, "claude_code_budget", 3.0)

            result = await self._bridge.code_task(
                prompt=description,
                target_dir=target_dir,
                max_budget_usd=budget,
                claude_path=claude_path,
            )

            duration = _time.time() - _task_start
            self._update_senses()

            if result.get("success"):
                bite = inject_personality_bite(
                    self._config.personality, "code_task_success",
                    posture=self._current_posture,
                )
                msg = bite or "Done."
                cost_info = f"Cost: ${result.get('cost_usd', 0):.2f}, {duration:.0f}s."
                summary = result.get("result_text", "")
                if summary:
                    summary = summary[:500]
                lines = [msg, cost_info]
                if summary:
                    lines.append(f"\n{summary}")
                lines.append(f"\nTarget: {target_dir}")
                chat_area.add_ai_message("\n".join(lines))
                self._speak(msg)
                if self._store:
                    self._store.add_message("assistant", f"{msg} {cost_info}")
            else:
                error = result.get("error", "Unknown error")
                rolled_back = result.get("rolled_back", False)
                bite = inject_personality_bite(
                    self._config.personality, "code_task_failed",
                    posture=self._current_posture,
                )
                msg = bite or "Code task failed."
                detail = f"Error: {error}"
                if rolled_back:
                    detail += " Rolled back."
                chat_area.add_ai_message(f"{msg}\n{detail}")
                self._speak(msg)
                if self._store:
                    self._store.add_message("assistant", f"{msg} {detail}")

                # Self-repair: record failure as improvement goal
                try:
                    db_path = self._store._path if self._store else None
                    if db_path and self._bridge:
                        self._bridge.record_task_failure(
                            db_path, "code_task", description, error,
                        )
                except Exception as e:
                    logger.debug(f"Self-repair goal (code_task) skipped: {e}")

        except Exception as e:
            logger.error(f"Code task error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="code_task"))
            err_msg = narrate_error(e, phase="code_task")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
            # Self-repair: record crash as improvement goal
            try:
                db_path = self._store._path if self._store else None
                if db_path and self._bridge:
                    self._bridge.record_task_failure(
                        db_path, "code_task", description, str(e),
                    )
            except Exception as exc:
                logger.debug(f"Self-repair goal (code_task crash) skipped: {exc}")
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Project generation ---

    def _run_generate_project(self) -> None:
        """Start project code generation from the last compilation result."""
        chat_area = self.query_one(ChatArea)
        result = self._last_compile_result
        if not result or not getattr(result, "blueprint", None):
            chat_area.add_ai_message(
                "No compilation to generate from. Compile something first."
            )
            return

        # Check trust threshold
        verification = result.verification or {}
        trust = verification.get("overall_score", 0.0) if isinstance(verification, dict) else 0.0
        if trust < 60:
            chat_area.add_ai_message(
                f"Trust is {trust:.0f}% — below the 60% threshold for code generation. "
                "Try refining the spec first."
            )
            return

        bite = inject_personality_bite(
            self._config.personality, "build_start", posture=self._current_posture
        )
        start_msg = bite or "Generating project from blueprint..."
        chat_area.add_ai_message(start_msg)
        if bite:
            self._speak(bite)

        status = self.query_one(StatusBar)
        status.state = "generating..."

        self.run_worker(self._generate_project_worker(), exclusive=True)

    async def _generate_project_worker(self) -> None:
        """Worker: generate a runnable project from the last compilation."""
        import time as _time
        _gen_start = _time.time()
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            result = self._last_compile_result
            if not result:
                return

            blueprint = result.blueprint
            verification = result.verification or {}

            # Determine output directory
            output_dir = getattr(self._config, "output_dir", "")
            if not output_dir:
                output_dir = str(Path.home() / "motherlabs" / "projects")

            budget = getattr(self._config, "claude_code_budget", 5.0)

            gen_result = await self._bridge.generate_project(
                blueprint=blueprint,
                verification=verification if isinstance(verification, dict) else {},
                output_dir=output_dir,
                max_budget_usd=budget,
            )

            duration = _time.time() - _gen_start
            self._update_senses()

            if gen_result.get("success"):
                project_dir = gen_result.get("project_dir", "")
                files = gen_result.get("files_created", [])
                cost = gen_result.get("cost_usd", 0.0)

                # Shorten path with ~
                home = str(Path.home())
                display_path = project_dir
                if display_path.startswith(home):
                    display_path = "~" + display_path[len(home):]

                self._last_project_path = project_dir

                lines = [
                    f"Project generated: {display_path}",
                    f"{len(files)} files, ${cost:.2f}, {duration:.0f}s.",
                ]
                summary = gen_result.get("result_text", "")
                if summary:
                    lines.append(f"\n{summary[:500]}")

                chat_area.add_ai_message("\n".join(lines))
                self._speak("Project generated.")
                if self._store:
                    self._store.add_message(
                        "assistant", f"Project generated at {display_path}"
                    )
            else:
                error = gen_result.get("error", "Unknown error")
                chat_area.add_ai_message(
                    f"Project generation failed.\nError: {error}"
                )
                self._speak("Project generation failed.")
                if self._store:
                    self._store.add_message(
                        "assistant", f"Project generation failed: {error}"
                    )

        except Exception as e:
            logger.error(f"Generate project error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(
                classify_error(e, phase="generate_project")
            )
            err_msg = narrate_error(e, phase="generate_project")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Filesystem operations ---

    def _run_theme(self, theme_name: str) -> None:
        """Switch theme or show current theme."""
        chat_area = self.query_one(ChatArea)

        # Show current theme if no argument
        if not theme_name:
            current = self._config.theme if hasattr(self._config, "theme") else "default"
            chat_area.add_message("system", f"Current theme: {current}\n\nAvailable themes:\n  • default\n  • alien\n\nUsage: /theme <name>")
            return

        theme_name = theme_name.lower().strip()

        # Validate theme
        if theme_name not in ("default", "alien"):
            chat_area.add_message("system", f"Unknown theme: {theme_name}\n\nAvailable: default, alien")
            return

        # Save theme to config
        self._config.theme = theme_name
        save_config(self._config)

        # Show confirmation
        theme_display = "Alien (MU-TH-UR 6000)" if theme_name == "alien" else "Default"
        chat_area.add_message("system", f"Theme changed to: {theme_display}\n\nRestart Mother to apply:\n  • Press Ctrl+R to restart\n  • Or Ctrl+Q then 'mother'")

        # Speak confirmation
        self._speak(f"Theme set to {theme_name}. Restart to apply.")

    def _run_search(self, query: str) -> None:
        """Start a file search."""
        chat_area = self.query_one(ChatArea)
        bite = inject_personality_bite(self._config.personality, "search_start")
        if bite:
            chat_area.add_ai_message(bite)
            self._speak(bite)

        status = self.query_one(StatusBar)
        status.state = "searching..."

        self.run_worker(self._search_worker(query), exclusive=False)

    async def _search_worker(self, query: str) -> None:
        """Worker: search files via bridge."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        _search_start = _time.time()
        try:
            results = await self._bridge.search_files(query)
            if not results:
                chat_area.add_ai_message(f"No files found for '{query}'.")
                self._speak("Nothing found.")
                return

            lines = [f"Found {len(results)} file{'s' if len(results) != 1 else ''}:"]
            for r in results[:20]:
                # Shorten path: use ~ for home
                path = r["path"]
                home = str(Path.home())
                if path.startswith(home):
                    path = "~" + path[len(home):]
                parent = str(Path(path).parent) + "/"
                lines.append(
                    f"  {r['name']} — {r['size_human']} — {parent} — {r['modified_human']}"
                )
            if len(results) > 20:
                lines.append(f"  ... and {len(results) - 20} more")

            msg = "\n".join(lines)
            chat_area.add_ai_message(msg)

            # Record actuator receipt for search
            try:
                if self._actuator_store is not None:
                    from mother.actuator_receipt import create_receipt
                    _r = create_receipt(
                        "search", True, _search_start, _time.time(),
                        output_summary=f"{len(results)} results for '{query[:40]}'",
                    )
                    self._actuator_store.record(_r)
            except Exception as e:
                logger.debug(f"Actuator receipt (search) skipped: {e}")

            bite = inject_personality_bite(self._config.personality, "search_complete")
            if bite:
                self._speak(bite)

        except PermissionError as e:
            chat_area.add_ai_message(str(e))
        except Exception as e:
            logger.error(f"Search error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(narrate_error(e, phase="search"))
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _run_web_search(self, query: str) -> None:
        """Search the web and display results."""
        chat_area = self.query_one(ChatArea)
        status = self.query_one(StatusBar)
        status.state = "searching web..."
        self.run_worker(self._web_search_worker(query), exclusive=False)

    async def _web_search_worker(self, query: str) -> None:
        """Worker: web search and display results."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            from mother.web_tools import execute_web_search
            result = await asyncio.to_thread(execute_web_search, {"query": query})
            if result.startswith("Error:"):
                chat_area.add_ai_message(result)
            else:
                chat_area.add_message("system", result)
            # Record actuator receipt
            try:
                if self._actuator_store is not None:
                    from mother.actuator_receipt import create_receipt
                    _r = create_receipt(
                        "web_search", not result.startswith("Error:"),
                        _time.time() - 1, _time.time(),
                        output_summary=f"web search: {query[:40]}",
                    )
                    self._actuator_store.record(_r)
            except Exception as e:
                logger.debug(f"Actuator receipt (web_search) skipped: {e}")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            if chat_area:
                chat_area.add_ai_message(narrate_error(e, phase="web search"))
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _run_web_fetch(self, url: str) -> None:
        """Fetch a URL and display the content."""
        chat_area = self.query_one(ChatArea)
        status = self.query_one(StatusBar)
        status.state = "fetching..."
        self.run_worker(self._web_fetch_worker(url), exclusive=False)

    async def _web_fetch_worker(self, url: str) -> None:
        """Worker: fetch URL and display content."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            from mother.web_tools import execute_web_fetch
            result = await asyncio.to_thread(execute_web_fetch, {"url": url})
            if result.startswith("Error:"):
                chat_area.add_ai_message(result)
            else:
                # Truncate for display (full content available via code engine)
                display = result[:3000]
                if len(result) > 3000:
                    display += f"\n\n... ({len(result)} chars total, truncated for display)"
                chat_area.add_message("system", display)
            # Record actuator receipt
            try:
                if self._actuator_store is not None:
                    from mother.actuator_receipt import create_receipt
                    _r = create_receipt(
                        "web_fetch", not result.startswith("Error:"),
                        _time.time() - 1, _time.time(),
                        output_summary=f"fetched: {url[:60]}",
                    )
                    self._actuator_store.record(_r)
            except Exception as e:
                logger.debug(f"Actuator receipt (web_fetch) skipped: {e}")
        except Exception as e:
            logger.error(f"Web fetch error: {e}")
            if chat_area:
                chat_area.add_ai_message(narrate_error(e, phase="web fetch"))
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _run_open(self, path: str) -> None:
        """Read and display a file's contents."""
        self.run_worker(self._open_worker(path), exclusive=False)

    async def _open_worker(self, path: str) -> None:
        """Worker: read file and show in chat."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            result = await self._bridge.read_file(path)
            content = result["content"]
            # Truncate display at 2000 chars
            if len(content) > 2000:
                content = content[:2000] + f"\n\n... truncated ({result['size_human']} total)"
            header = f"**{result['path']}** ({result['size_human']})"
            if result["truncated"]:
                header += " [truncated]"
            chat_area.add_ai_message(f"{header}\n```\n{content}\n```")
        except PermissionError as e:
            chat_area.add_ai_message(str(e))
        except FileNotFoundError:
            chat_area.add_ai_message(f"File not found: {path}")
        except Exception as e:
            logger.error(f"Open error: {e}")
            chat_area.add_ai_message(narrate_error(e, phase="file"))

    def _run_file_action(self, action_str: str) -> None:
        """Parse and execute a file action: move, copy, delete."""
        self.run_worker(self._file_action_worker(action_str), exclusive=False)

    async def _file_action_worker(self, action_str: str) -> None:
        """Worker: parse 'move: src -> dst' / 'copy: src -> dst' / 'delete: path'."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            action_str = action_str.strip()
            if action_str.startswith("move:"):
                parts = action_str[5:].split("->")
                if len(parts) != 2:
                    chat_area.add_ai_message("Move needs: source -> destination")
                    return
                result = await self._bridge.move_file(parts[0].strip(), parts[1].strip())
                chat_area.add_ai_message(f"Moved to {result['dst']}.")
                self._speak("Moved.")
            elif action_str.startswith("copy:"):
                parts = action_str[5:].split("->")
                if len(parts) != 2:
                    chat_area.add_ai_message("Copy needs: source -> destination")
                    return
                result = await self._bridge.copy_file(parts[0].strip(), parts[1].strip())
                chat_area.add_ai_message(f"Copied to {result['dst']}.")
                self._speak("Copied.")
            elif action_str.startswith("delete:"):
                path = action_str[7:].strip()
                result = await self._bridge.delete_file(path)
                method = "Moved to Trash" if result["method"] == "trash" else "Deleted"
                chat_area.add_ai_message(f"{method}: {result['path']}")
                self._speak(method + ".")
            elif action_str.startswith("write:") or action_str.startswith("create:"):
                prefix_len = 6 if action_str.startswith("write:") else 7
                rest = action_str[prefix_len:].strip()
                sep = rest.find("|")
                if sep < 0:
                    chat_area.add_ai_message("Write needs: path | content")
                    return
                path = rest[:sep].strip()
                content = rest[sep + 1:].strip()
                result = await self._bridge.write_file(path, content, overwrite=True)
                chat_area.add_ai_message(f"Written: {result['path']} ({result['bytes_written']} bytes)")
                self._speak("Written.")
            elif action_str.startswith("edit:"):
                rest = action_str[5:].strip()
                sep = rest.find("|")
                if sep < 0:
                    chat_area.add_ai_message("Edit needs: path | old_text -> new_text")
                    return
                path = rest[:sep].strip()
                replacement = rest[sep + 1:].strip()
                arrow = replacement.find("->")
                if arrow < 0:
                    chat_area.add_ai_message("Edit needs: path | old_text -> new_text")
                    return
                old_text = replacement[:arrow].strip()
                new_text = replacement[arrow + 2:].strip()
                result = await self._bridge.edit_file(path, old_text, new_text)
                chat_area.add_ai_message(f"Edited: {result['path']}")
                self._speak("Edited.")
            elif action_str.startswith("append:"):
                rest = action_str[7:].strip()
                sep = rest.find("|")
                if sep < 0:
                    chat_area.add_ai_message("Append needs: path | content")
                    return
                path = rest[:sep].strip()
                content = rest[sep + 1:].strip()
                result = await self._bridge.append_file(path, content)
                chat_area.add_ai_message(f"Appended to {result['path']} ({result['bytes_appended']} bytes)")
                self._speak("Appended.")
            else:
                chat_area.add_ai_message(f"Unknown file action: {action_str}")
        except PermissionError as e:
            chat_area.add_ai_message(str(e))
        except FileNotFoundError as e:
            chat_area.add_ai_message(str(e))
        except Exception as e:
            logger.error(f"File action error: {e}")
            chat_area.add_ai_message(narrate_error(e, phase="file"))

    # --- Screen capture ---

    def _run_capture(self, question: str = "") -> None:
        """Start a screen capture and describe what's visible."""
        chat_area = self.query_one(ChatArea)
        if self._screen_bridge is None or not self._screen_bridge.enabled:
            chat_area.add_ai_message("Screen capture isn't available. Enable in /settings.")
            return

        chat_area.add_message("system", "Capturing screen...")
        status = self.query_one(StatusBar)
        status.state = "capturing..."
        self.run_worker(self._capture_worker(question), exclusive=False)

    async def _capture_worker(self, question: str = "") -> None:
        """Worker: capture screen, send to vision LLM, display result."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            img = await self._screen_bridge.capture_screen()
            if img is None:
                chat_area.add_ai_message("Couldn't capture the screen.")
                return

            # Default question
            if not question.strip():
                question = "What do you see on my screen?"

            # Show user message
            chat_area.add_user_message(question)
            if self._store:
                self._store.add_message("user", question)

            # Build messages and send with image
            messages = []
            if self._store:
                messages = self._store.get_context_window(max_tokens=4000)
            if not messages or messages[-1].get("content") != question:
                messages.append({"role": "user", "content": question})

            # Recompute senses before building prompt
            self._update_senses()

            sense_block = None
            if self._current_senses and self._current_posture:
                sense_block = render_sense_block(self._current_posture, self._current_senses)

            ctx_data = self._build_context_data()
            context_block = synthesize_context(ctx_data, sense_block=sense_block)
            system_prompt = build_system_prompt(
                self._config,
                context_block=context_block,
            )

            # Stream response
            self._bridge.begin_chat_stream()
            stream_task = asyncio.create_task(
                self._bridge.stream_chat(messages, system_prompt, images=[img])
            )
            chat_area.begin_streaming_message(self._config.name.lower())
            status = self.query_one(StatusBar)
            status.state = "streaming..."

            capture_voice_tracker = StreamingVoiceTracker() if self._voice else None

            async for token in self._bridge.stream_chat_tokens():
                chat_area.append_streaming_text(token)
                if capture_voice_tracker:
                    sentence = capture_voice_tracker.feed(token)
                    if sentence:
                        self._speak(sentence)

            await stream_task

            full_response = self._bridge.get_stream_result() or ""
            parsed = parse_response(full_response)
            chat_area.finish_streaming_message(parsed["display"])
            if self._store:
                self._store.add_message("assistant", parsed["display"])
            if parsed["voice"]:
                if capture_voice_tracker and capture_voice_tracker.spoke_anything:
                    tail = capture_voice_tracker.finish()
                    if tail:
                        self._speak(tail)
                else:
                    self._speak(parsed["voice"])

            # Route any actions
            if parsed["action"] == "compile" and parsed["action_arg"]:
                self._run_compile(parsed["action_arg"])
            elif parsed["action"] == "context" and parsed["action_arg"]:
                self._run_context_compile(parsed["action_arg"])
            elif parsed["action"] == "explore" and parsed["action_arg"]:
                self._run_explore_compile(parsed["action_arg"])
            elif parsed["action"] == "build" and parsed["action_arg"]:
                self._run_build(parsed["action_arg"])
            elif parsed["action"] == "use_tool" and parsed["action_arg"]:
                self._run_use_tool(parsed["action_arg"])
            elif parsed["action"] == "idea" and parsed["action_arg"]:
                self._run_add_idea(parsed["action_arg"])
            elif parsed["action"] == "self_build" and parsed["action_arg"]:
                self._run_self_build(parsed["action_arg"])
            elif parsed["action"] == "code" and parsed["action_arg"]:
                self._run_code_task(parsed["action_arg"])
            elif parsed["action"] == "github_push":
                self._run_github_push()
            elif parsed["action"] == "tweet" and parsed["action_arg"]:
                self._run_tweet(parsed["action_arg"])
            elif parsed["action"] == "discover_peers":
                self._run_discover_peers()
            elif parsed["action"] == "list_peers":
                self._run_list_peers()
            elif parsed["action"] == "delegate" and parsed["action_arg"]:
                self._run_delegate(parsed["action_arg"])
            elif parsed["action"] == "whatsapp" and parsed["action_arg"]:
                self._run_whatsapp(parsed["action_arg"])
            elif parsed["action"] == "integrate" and parsed["action_arg"]:
                self._run_integrate(parsed["action_arg"])

            # Update cost
            status = self.query_one(StatusBar)
            status.session_cost = self._bridge.get_session_cost()

        except Exception as e:
            logger.error(f"Capture error: {e}")
            self._session_error_count += 1
            err_msg = narrate_error(e, phase="capture")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Camera ---

    def _run_camera(self, question: str = "") -> None:
        """Capture webcam frame and describe what's visible."""
        chat_area = self.query_one(ChatArea)
        if self._camera_bridge is None or not self._camera_bridge.enabled:
            msg = "Camera isn't enabled. Want me to turn it on?"
            chat_area.add_ai_message(msg)
            self._speak(msg)
            if self._store:
                self._store.add_message("assistant", msg)
            self._pending_permission = "camera"
            return

        chat_area.add_message("system", "Capturing camera...")
        status = self.query_one(StatusBar)
        status.state = "capturing..."
        self.run_worker(self._camera_worker(question), exclusive=False)

    async def _camera_worker(self, question: str = "") -> None:
        """Worker: capture webcam, send to vision LLM, display result."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            img = await self._camera_bridge.capture_frame()
            if img is None:
                chat_area.add_ai_message("Couldn't capture from the camera.")
                return

            if not question.strip():
                question = "What do you see through the camera?"

            chat_area.add_user_message(question)
            if self._store:
                self._store.add_message("user", question)

            messages = []
            if self._store:
                messages = self._store.get_context_window(max_tokens=4000)
            if not messages or messages[-1].get("content") != question:
                messages.append({"role": "user", "content": question})

            self._update_senses()

            sense_block = None
            if self._current_senses and self._current_posture:
                sense_block = render_sense_block(self._current_posture, self._current_senses)

            ctx_data = self._build_context_data()
            context_block = synthesize_context(ctx_data, sense_block=sense_block)
            system_prompt = build_system_prompt(
                self._config,
                context_block=context_block,
            )

            # Stream response
            self._bridge.begin_chat_stream()
            stream_task = asyncio.create_task(
                self._bridge.stream_chat(
                    messages, system_prompt, images=[(img, "image/jpeg")]
                )
            )
            chat_area.begin_streaming_message(self._config.name.lower())
            status = self.query_one(StatusBar)
            status.state = "streaming..."

            camera_voice_tracker = StreamingVoiceTracker() if self._voice else None

            async for token in self._bridge.stream_chat_tokens():
                chat_area.append_streaming_text(token)
                if camera_voice_tracker:
                    sentence = camera_voice_tracker.feed(token)
                    if sentence:
                        self._speak(sentence)

            await stream_task

            full_response = self._bridge.get_stream_result() or ""
            parsed = parse_response(full_response)
            chat_area.finish_streaming_message(parsed["display"])
            if self._store:
                self._store.add_message("assistant", parsed["display"])
            if parsed["voice"]:
                if camera_voice_tracker and camera_voice_tracker.spoke_anything:
                    tail = camera_voice_tracker.finish()
                    if tail:
                        self._speak(tail)
                else:
                    self._speak(parsed["voice"])

            if parsed["action"] == "compile" and parsed["action_arg"]:
                self._run_compile(parsed["action_arg"])
            elif parsed["action"] == "context" and parsed["action_arg"]:
                self._run_context_compile(parsed["action_arg"])
            elif parsed["action"] == "explore" and parsed["action_arg"]:
                self._run_explore_compile(parsed["action_arg"])
            elif parsed["action"] == "build" and parsed["action_arg"]:
                self._run_build(parsed["action_arg"])
            elif parsed["action"] == "use_tool" and parsed["action_arg"]:
                self._run_use_tool(parsed["action_arg"])
            elif parsed["action"] == "idea" and parsed["action_arg"]:
                self._run_add_idea(parsed["action_arg"])
            elif parsed["action"] == "self_build" and parsed["action_arg"]:
                self._run_self_build(parsed["action_arg"])
            elif parsed["action"] == "code" and parsed["action_arg"]:
                self._run_code_task(parsed["action_arg"])

            status = self.query_one(StatusBar)
            status.session_cost = self._bridge.get_session_cost()

        except Exception as e:
            logger.error(f"Camera error: {e}")
            self._session_error_count += 1
            err_msg = narrate_error(e, phase="camera")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- Microphone ---

    def _run_listen(self, duration: Optional[float] = None) -> None:
        """Start microphone recording and transcription.

        If duration is None (default), uses VAD — records until speech ends.
        If duration is a float, records for exactly that many seconds (legacy).
        """
        chat_area = self.query_one(ChatArea)
        if self._microphone_bridge is None or not self._microphone_bridge.enabled:
            msg = "Microphone isn't enabled. Want me to turn it on?"
            chat_area.add_ai_message(msg)
            self._speak(msg)
            if self._store:
                self._store.add_message("assistant", msg)
            self._pending_permission = "microphone"
            return

        if duration is not None:
            chat_area.add_message("system", f"Listening for {duration:.0f}s...")
        else:
            chat_area.add_message("system", "Listening...")
        status = self.query_one(StatusBar)
        status.state = "listening..."
        self.run_worker(self._listen_worker(duration), exclusive=False)

    async def _listen_worker(self, duration: Optional[float] = None) -> None:
        """Worker: record audio, transcribe, route through chat.

        duration=None uses VAD (adaptive). duration=float uses fixed recording.
        """
        chat_area = self._safe_query(ChatArea)
        if not chat_area:
            return
        try:
            if duration is not None:
                text = await self._microphone_bridge.record_and_transcribe(duration)
            else:
                text = await self._microphone_bridge.record_and_transcribe_vad()
            if not text:
                chat_area.add_ai_message("Didn't catch that. Try again?")
                self._speak("Didn't catch that.")
                return

            # Show transcription as user message
            chat_area.add_user_message(f'[Transcribed] "{text}"')

            # Route through normal chat flow
            self._handle_chat(text)

        except Exception as e:
            logger.error(f"Listen error: {e}")
            self._session_error_count += 1
            err_msg = narrate_error(e, phase="listen")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    def _enable_microphone(self) -> bool:
        """Hot-enable microphone bridge mid-session. Returns True on success."""
        chat_area = self.query_one(ChatArea)
        if not is_microphone_available():
            chat_area.add_ai_message("Can't enable — sounddevice isn't installed.")
            self._speak("I need sounddevice installed for that.")
            return False
        openai_key = self._config.api_keys.get("openai")
        if not openai_key:
            chat_area.add_ai_message("I need an OpenAI API key for transcription. Add one in /settings.")
            self._speak("I need an OpenAI key for transcription.")
            return False
        self._microphone_bridge = MicrophoneBridge(openai_api_key=openai_key, enabled=True)
        self._config.microphone_enabled = True
        save_config(self._config)
        return True

    def _enable_camera(self) -> bool:
        """Hot-enable camera bridge mid-session. Returns True on success."""
        chat_area = self.query_one(ChatArea)
        if not is_camera_available():
            chat_area.add_ai_message("Can't enable — opencv-python isn't installed or no camera found.")
            self._speak("I need opencv installed for that, or there's no camera.")
            return False
        self._camera_bridge = CameraBridge(enabled=True)
        self._config.camera_enabled = True
        save_config(self._config)
        return True

    def _enable_duplex_voice(self) -> bool:
        """Hot-enable duplex voice mid-session. Returns True on success."""
        chat_area = self.query_one(ChatArea)
        from mother.voice_duplex import is_duplex_available
        if not is_duplex_available():
            chat_area.add_ai_message("Can't enable — sounddevice or elevenlabs SDK not installed.")
            self._speak("I need sounddevice and elevenlabs installed for real-time voice.")
            return False
        xi_key = self._config.api_keys.get("elevenlabs") or ""
        if not xi_key:
            import os
            xi_key = os.environ.get("ELEVENLABS_API_KEY", "")
        if not xi_key:
            chat_area.add_ai_message("I need an ElevenLabs API key for real-time voice. Add one in /settings.")
            self._speak("I need an ElevenLabs key for that.")
            return False
        self._config.voice_duplex_enabled = True
        save_config(self._config)
        asyncio.get_event_loop().create_task(self._start_duplex_voice())
        return True

    def action_listen(self) -> None:
        """F8 handler — push-to-talk (VAD adaptive)."""
        self._run_listen()

    # --- Project launcher ---

    def _run_launch(self) -> None:
        """Launch (or re-launch) the last built project."""
        chat_area = self.query_one(ChatArea)
        if not self._last_project_path:
            chat_area.add_ai_message("No project to launch. Run /build first.")
            return
        status = self.query_one(StatusBar)
        status.state = "launching..."
        self.run_worker(self._launch_worker(), exclusive=False)

    def _run_stop(self) -> None:
        """Stop the running project."""
        chat_area = self.query_one(ChatArea)
        if not self._bridge or not self._bridge.is_project_running():
            chat_area.add_ai_message("Nothing running.")
            return
        self.run_worker(self._stop_worker(), exclusive=False)

    # --- Tool invocation ---

    def _run_use_tool(self, arg: str) -> None:
        """Run a registered tool by name."""
        self.run_worker(self._use_tool_worker(arg), exclusive=False)

    async def _use_tool_worker(self, arg: str) -> None:
        """Worker: find and run a tool, display result."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        # Parse "tool_name: input" or just "tool_name"
        if ":" in arg:
            tool_name, input_text = arg.split(":", 1)
            tool_name = tool_name.strip()
            input_text = input_text.strip()
        else:
            tool_name = arg.strip()
            input_text = ""

        try:
            status = self._safe_query(StatusBar)
            if status:
                status.state = "running tool..."

            result = await self._bridge.run_tool(tool_name, input_text)

            if result["success"]:
                output = result.get("output", "")
                if output:
                    chat_area.add_ai_message(output)
                else:
                    chat_area.add_ai_message(f"{result['tool_name']} completed.")
                self._speak("Done.")
            else:
                error = result.get("error", "Unknown error")
                chat_area.add_ai_message(f"Tool error: {error}")
                self._speak("That didn't work.")

        except Exception as e:
            logger.error(f"Tool run error: {e}")
            self._session_error_count += 1
            self._session_error_classifications.append(classify_error(e, phase="tool"))
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(narrate_error(e, phase="tool"))
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- GitHub integration ---

    def _run_github_push(self) -> None:
        """Push commits to GitHub."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Pushing to GitHub...")
        self.run_worker(self._github_push_worker(), exclusive=False)

    async def _github_push_worker(self) -> None:
        """Worker: push commits to GitHub."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            repo_dir = str(Path(__file__).resolve().parent.parent)
            result = await self._bridge.github_push(repo_dir=repo_dir)

            if result["success"]:
                chat_area.add_ai_message(f"Pushed to GitHub.\n{result['output']}")
                self._speak("Pushed.")
            else:
                error = result.get("error", "Unknown error")
                chat_area.add_ai_message(f"Push failed: {error}")
                self._speak("Push failed.")

        except Exception as e:
            logger.error(f"GitHub push error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    # --- Twitter integration ---

    def _run_tweet(self, text: str) -> None:
        """Post a tweet."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Posting...")
        self.run_worker(self._tweet_worker(text), exclusive=False)

    async def _tweet_worker(self, text: str) -> None:
        """Worker: post a tweet."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            result = await self._bridge.tweet(text)

            if result["success"]:
                url = result.get("tweet_url", "")
                chat_area.add_ai_message(f"Posted: {url}")
                self._speak("Posted.")
            else:
                error = result.get("error", "Unknown error")
                if "TWITTER_BEARER_TOKEN" in error:
                    chat_area.add_ai_message("Twitter not configured. Set TWITTER_BEARER_TOKEN in environment.")
                else:
                    chat_area.add_ai_message(f"Tweet failed: {error}")
                self._speak("Couldn't post that.")

        except Exception as e:
            logger.error(f"Tweet error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    # --- Social publishing ---

    def _run_publish_project(self, project_dir: str) -> None:
        """Publish an emitted project to GitHub."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Publishing project to GitHub...")
        self.run_worker(self._publish_project_worker(project_dir), exclusive=False)

    async def _publish_project_worker(self, project_dir: str) -> None:
        """Worker: publish project to GitHub."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            result = await self._bridge.publish_project(
                project_dir=project_dir,
                name=Path(project_dir).name,
                description="",
            )
            if result["success"]:
                chat_area.add_ai_message(f"Published: {result['repo_url']}")
                self._speak("Published.")
            else:
                chat_area.add_ai_message(f"Publish failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"Project publish error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    def _run_discord_post(self, text: str) -> None:
        """Post to Discord."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Posting to Discord...")
        self.run_worker(self._discord_post_worker(text), exclusive=False)

    async def _discord_post_worker(self, text: str) -> None:
        """Worker: post to Discord."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            result = await self._bridge.discord_post(content=text)
            if result["success"]:
                chat_area.add_ai_message("Posted to Discord.")
                self._speak("Posted.")
            else:
                chat_area.add_ai_message(f"Discord failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"Discord post error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    def _run_bluesky_post(self, text: str) -> None:
        """Post to Bluesky."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Posting to Bluesky...")
        self.run_worker(self._bluesky_post_worker(text), exclusive=False)

    async def _bluesky_post_worker(self, text: str) -> None:
        """Worker: post to Bluesky."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            result = await self._bridge.bluesky_post(text)
            if result["success"]:
                url = result.get("post_url", "")
                chat_area.add_ai_message(f"Posted: {url}")
                self._speak("Posted.")
            else:
                chat_area.add_ai_message(f"Bluesky failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            logger.error(f"Bluesky post error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    async def _process_social_queue_tick(self) -> None:
        """Process pending social posts (called from autonomous tick)."""
        if not self._bridge:
            return
        try:
            results = await self._bridge.process_social_queue()
            for r in results:
                if r["success"]:
                    logger.debug(f"Social queue: sent {r['platform']} post")
                else:
                    logger.debug(f"Social queue: {r['platform']} failed: {r.get('error', '')}")
        except Exception as e:
            logger.debug(f"Social queue processing skipped: {e}")

    # --- Peer networking ---

    def _run_discover_peers(self) -> None:
        """Discover Mother instances on the network."""
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Scanning network...")
        self.run_worker(self._discover_peers_worker(), exclusive=False)

    async def _discover_peers_worker(self) -> None:
        """Worker: mDNS discovery for peer instances."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            count = await self._bridge.discover_peers(timeout=5.0)
            if count > 0:
                chat_area.add_ai_message(f"Found {count} peer{'s' if count != 1 else ''}.")
                self._speak(f"Found {count}.")
            else:
                chat_area.add_ai_message("No peers discovered.")
                self._speak("Nothing found.")
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    def _run_list_peers(self) -> None:
        """List known peers."""
        self.run_worker(self._list_peers_worker(), exclusive=False)

    async def _list_peers_worker(self) -> None:
        """Worker: list known peers."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            peers = await self._bridge.list_peers(active_only=False)
            if not peers:
                chat_area.add_ai_message("No peers registered.")
                self._speak("No peers yet.")
                return

            lines = ["Known peers:"]
            for p in peers:
                name = p.get("name", "Unknown")
                host = p.get("host", "")
                port = p.get("port", 0)
                age = time.time() - p.get("last_seen", 0)
                age_str = f"{int(age/60)}m ago" if age < 3600 else f"{int(age/3600)}h ago"
                lines.append(f"  {name} @ {host}:{port} (seen {age_str})")

            chat_area.add_ai_message("\n".join(lines))
            self._speak(f"{len(peers)} peer{'s' if len(peers) != 1 else ''}.")

        except Exception as e:
            logger.error(f"List peers error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    def _run_delegate(self, arg: str) -> None:
        """Delegate a task to a peer. Format: 'peer_id: description'."""
        chat_area = self.query_one(ChatArea)
        if ":" not in arg:
            chat_area.add_ai_message("Format: peer_id: task description")
            return
        peer_id, description = arg.split(":", 1)
        peer_id = peer_id.strip()
        description = description.strip()
        chat_area.add_ai_message(f"Delegating to {peer_id}...")
        self.run_worker(self._delegate_worker(peer_id, description), exclusive=False)

    async def _delegate_worker(self, peer_id: str, description: str) -> None:
        """Worker: delegate compile to peer."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            # Look up peer by name if not an ID
            actual_peer_id = peer_id
            if not peer_id.startswith("mother-"):  # Heuristic: IDs start with "mother-"
                # Try to find by name
                peers = self._context_cache.get("connected_peers", [])
                for p in peers:
                    if peer_id.lower() in p.get("name", "").lower():
                        actual_peer_id = p.get("instance_id", peer_id)
                        break

            result = await self._bridge.delegate_compile(actual_peer_id, description)

            if result["success"]:
                blueprint = result.get("blueprint", {})
                trust = result.get("trust_score", 0)
                cost = result.get("cost_usd", 0)
                chat_area.add_message(
                    "mother",
                    f"Peer completed: {len(blueprint.get('components', []))} components, "
                    f"{trust:.0f}% trust, ${cost:.3f}"
                )
                self._speak("Peer completed it.")
            else:
                error = result.get("error", "Unknown error")
                chat_area.add_ai_message(f"Delegation failed: {error}")
                self._speak("That didn't work.")

        except Exception as e:
            logger.error(f"Delegation error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    # --- WhatsApp integration ---

    def _run_whatsapp(self, text: str) -> None:
        """Send WhatsApp message to configured user number."""
        if not self._config.whatsapp_enabled:
            chat_area = self.query_one(ChatArea)
            chat_area.add_ai_message("WhatsApp not enabled. Configure in settings.")
            return
        chat_area = self.query_one(ChatArea)
        chat_area.add_ai_message("Sending...")
        self.run_worker(self._whatsapp_worker(text), exclusive=False)

    async def _whatsapp_worker(self, text: str) -> None:
        """Worker: send WhatsApp message."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            to = self._config.user_whatsapp_number
            if not to:
                chat_area.add_ai_message("No WhatsApp number configured.")
                return

            result = await self._bridge.send_whatsapp(to, text)

            if result["success"]:
                chat_area.add_ai_message(f"Sent to {to}.")
                self._speak("Sent.")
            else:
                error = result.get("error", "Unknown error")
                if "Account SID" in error or "Auth Token" in error:
                    chat_area.add_ai_message("WhatsApp not configured. Set Twilio credentials in config.")
                else:
                    chat_area.add_ai_message(f"Send failed: {error}")
                self._speak("Couldn't send that.")

        except Exception as e:
            logger.error(f"WhatsApp error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(f"Error: {e}")

    # --- Project integration ---

    def _run_integrate(self, arg: str) -> None:
        """Integrate external project as tool."""
        chat_area = self.query_one(ChatArea)

        if ":" not in arg:
            chat_area.add_ai_message("Format: path/to/project: description")
            return

        path, description = arg.split(":", 1)
        path = path.strip()
        description = description.strip()

        bite = inject_personality_bite(
            self._config.personality, "integrate_start", posture=self._current_posture
        )
        if bite:
            chat_area.add_ai_message(bite)
            self._speak(bite)

        status = self.query_one(StatusBar)
        status.state = "integrating..."

        self.run_worker(self._integrate_worker(path, description), exclusive=False)

    async def _integrate_worker(self, path: str, description: str) -> None:
        """Worker: integrate project into tool registry."""
        import time as _time
        _start = _time.time()
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return

        try:
            result = await self._bridge.integrate_project(path, description)
            duration = _time.time() - _start

            if result["success"]:
                name = result.get("name", "project")
                components = result.get("components", 0)
                trust = result.get("trust_score", 0)

                chat_area.add_message(
                    "mother",
                    f"Integrated {name}: {components} components, {trust:.0f}% trust.\n"
                    f"Available via /tools or [ACTION:use_tool]{name}[/ACTION]"
                )

                bite = inject_personality_bite(
                    self._config.personality, "integrate_success", posture=self._current_posture
                )
                if bite:
                    self._speak(bite)
                else:
                    self._speak(f"{name} is ready.")

                # Journal
                if self._journal:
                    from mother.journal import JournalEntry
                    self._journal.record(JournalEntry(
                        event_type="integrate",
                        description=description[:200],
                        success=True,
                        trust_score=trust,
                        component_count=components,
                        duration_seconds=duration,
                        project_path=result.get("project_path", ""),
                    ))
            else:
                error = result.get("error", "Unknown error")
                chat_area.add_ai_message(f"Integration failed: {error}")
                self._speak("Integration failed.")

                # Journal
                if self._journal:
                    from mother.journal import JournalEntry
                    self._journal.record(JournalEntry(
                        event_type="integrate",
                        description=description[:200],
                        success=False,
                        error_summary=str(error)[:200],
                        duration_seconds=duration,
                    ))

        except Exception as e:
            logger.error(f"Integrate error: {e}")
            self._session_error_count += 1
            from mother.error_taxonomy import classify_error
            self._session_error_classifications.append(classify_error(e, phase="integrate"))
            err_msg = narrate_error(e, phase="integrate")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(err_msg)
            self._speak(err_msg)
        finally:
            self._save_sense_memory()
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    # --- WhatsApp webhook ---

    async def _start_whatsapp_webhook(self) -> None:
        """Start WhatsApp webhook server + consumer in background.

        Architecture: uvicorn runs in a separate thread with its own event loop.
        on_message puts (msg, threading.Event, result_list) on a thread-safe queue.
        A Textual worker polls the queue and processes messages on Textual's thread,
        then signals the event so the webhook can return the response to Twilio.
        """
        if not self._config.whatsapp_webhook_enabled:
            logger.info("WhatsApp webhook disabled in config")
            return

        try:
            from messaging.whatsapp_bridge import WhatsAppBridge

            logger.info(f"Starting WhatsApp webhook on port {self._config.whatsapp_webhook_port}...")

            incoming_queue = self._whatsapp_incoming

            async def on_message(msg):
                """Enqueue incoming message for Textual thread to process.

                Returns None — response is sent separately by the consumer
                via Twilio API so we don't block the webhook handler.
                """
                logger.info(f"WhatsApp incoming from {msg.sender_id}: {msg.content}")
                incoming_queue.put(msg)
                return None  # Don't send inline reply — consumer handles it

            self._whatsapp_bridge = WhatsAppBridge(
                account_sid=self._config.twilio_account_sid,
                auth_token=self._config.twilio_auth_token,
                from_number=self._config.twilio_whatsapp_number,
                webhook_port=self._config.whatsapp_webhook_port,
                on_message=on_message,
            )

            # Run uvicorn in a background thread (separate event loop)
            def _run_server():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._whatsapp_bridge.start())

            server_thread = threading.Thread(target=_run_server, daemon=True)
            server_thread.start()

            logger.info(f"WhatsApp webhook listening on port {self._config.whatsapp_webhook_port}")

            chat_area = self._safe_query(ChatArea)
            if chat_area:
                chat_area.add_message("system",
                    f"WhatsApp active on port {self._config.whatsapp_webhook_port}")

        except ImportError:
            logger.warning("WhatsApp bridge dependencies not available")
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {self._config.whatsapp_webhook_port} already in use")
            else:
                logger.error(f"WhatsApp webhook error: {e}")
        except Exception as e:
            logger.error(f"WhatsApp webhook error: {e}")

    def _poll_whatsapp(self) -> None:
        """Poll WhatsApp queue — called by set_interval on Textual's thread."""
        qsize = self._whatsapp_incoming.qsize()
        if qsize > 0:
            logger.info(f"WhatsApp poll: queue has {qsize} messages")
        try:
            msg = self._whatsapp_incoming.get_nowait()
        except queue.Empty:
            return
        except Exception as e:
            logger.error(f"WhatsApp poll error: {e}")
            return

        logger.info(f"WhatsApp poll: got '{msg.content}' — dispatching")
        # Show in TUI immediately
        try:
            chat_area = self._safe_query(ChatArea)
            if chat_area:
                chat_area.add_user_message(f"[WhatsApp] {msg.content}")
        except Exception as e:
            logger.error(f"WhatsApp display error: {e}")

        self.run_worker(self._handle_whatsapp_message(msg), exclusive=False)

    async def _handle_whatsapp_message(self, msg) -> None:
        """Process a single incoming WhatsApp message.

        Mirrors the normal _chat_worker flow: build context, stream LLM,
        display in TUI, send reply back via Twilio.
        """
        try:
            chat_area = self._safe_query(ChatArea)

            # Store user message
            if self._store:
                self._store.add_message("user", msg.content)

            reply_text = "Message received."

            if self._bridge:
                # Build messages list (same as normal chat)
                messages = []
                if self._store:
                    recent = self._store.get_history(limit=20)
                    for h in recent:
                        messages.append({"role": h.role, "content": h.content})
                messages.append({"role": "user", "content": msg.content})

                # Build system prompt (same as normal chat)
                ctx_data = self._build_context_data()
                sense_block = None
                if self._current_senses and self._current_posture:
                    sense_block = render_sense_block(
                        self._current_posture, self._current_senses
                    )
                context_block = synthesize_context(ctx_data, sense_block=sense_block)
                system_prompt = build_system_prompt(
                    self._config, context_block=context_block
                )

                try:
                    # Stream LLM response
                    self._bridge.begin_chat_stream()
                    await self._bridge.stream_chat(messages, system_prompt)

                    # Collect full response
                    response_text = ""
                    async for token in self._bridge.stream_chat_tokens():
                        response_text += token

                    parsed = parse_response(response_text)
                    display_text = parsed.get("voice") or parsed.get(
                        "display", response_text
                    )

                    if self._store:
                        self._store.add_message("assistant", response_text)

                    if chat_area:
                        chat_area.add_ai_message(display_text)

                    self._speak(display_text)
                    reply_text = display_text

                except Exception as e:
                    logger.error(f"WhatsApp chat error: {e}", exc_info=True)
                    reply_text = "I hit a snag processing that. Try again."

            # Send reply back via Twilio API
            if self._whatsapp_bridge and reply_text:
                try:
                    await self._whatsapp_bridge.send_message(
                        msg.sender_id, reply_text
                    )
                    logger.info(f"WhatsApp reply sent to {msg.sender_id}")
                except Exception as e:
                    logger.error(f"WhatsApp send error: {e}")

        except Exception as e:
            logger.error(f"WhatsApp handler error: {e}", exc_info=True)

    async def _launch_worker(self) -> None:
        """Worker: launch project and start output polling."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            # Stop any existing project first
            if self._bridge.is_project_running():
                await self._bridge.stop_project()

            # Cancel existing poll task
            if self._launcher_poll_task and not self._launcher_poll_task.done():
                self._launcher_poll_task.cancel()

            result = await self._bridge.launch(
                self._last_project_path,
                self._last_entry_point or "main.py",
            )

            if result["success"]:
                pid = result["pid"]
                port = result.get("port")
                if port:
                    chat_area.add_ai_message(f"Running on port {port} (pid {pid}).")
                    self._speak(f"Running on port {port}.")
                else:
                    chat_area.add_ai_message(f"Process started (pid {pid}).")
                    self._speak("Started.")

                # Start output polling
                self._launcher_poll_task = asyncio.create_task(self._output_poll_loop())
            else:
                error = result.get("error", "Unknown error")
                chat_area.add_ai_message(f"Launch failed: {error}")
                self._speak("Launch failed.")
        except Exception as e:
            logger.error(f"Launch error: {e}")
            self._session_error_classifications.append(classify_error(e, phase="launch"))
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(narrate_error(e, phase="launch"))
        finally:
            sb = self._safe_query(StatusBar)
            if sb:
                sb.state = "ready"

    async def _stop_worker(self) -> None:
        """Worker: stop running project."""
        chat_area = self._safe_query(ChatArea)
        if not chat_area or not self._bridge:
            return
        try:
            # Cancel poll task
            if self._launcher_poll_task and not self._launcher_poll_task.done():
                self._launcher_poll_task.cancel()
                self._launcher_poll_task = None

            await self._bridge.stop_project()
            chat_area.add_ai_message("Stopped.")
            self._speak("Stopped.")
        except Exception as e:
            logger.error(f"Stop error: {e}")
            ca = self._safe_query(ChatArea)
            if ca:
                ca.add_ai_message(narrate_error(e, phase="stop"))

    async def _output_poll_loop(self) -> None:
        """Poll project output and health status, display in chat."""
        try:
            while not self._unmounted:
                if not self._bridge or not self._bridge.is_project_running():
                    # Process exited — drain remaining output
                    lines = self._bridge.get_project_output() if self._bridge else []
                    if lines:
                        chat_area = self._safe_query(ChatArea)
                        if chat_area:
                            chat_area.add_message("system", "\n".join(lines))
                    chat_area = self._safe_query(ChatArea)
                    if chat_area:
                        chat_area.add_message("system", "Process exited.")
                    return

                lines = self._bridge.get_project_output()
                if lines:
                    chat_area = self._safe_query(ChatArea)
                    if chat_area:
                        chat_area.add_message("system", "\n".join(lines))

                # Health check
                if self._bridge:
                    try:
                        status, event = await self._bridge.check_health()
                        if event:
                            self._health_failure_count += (
                                1 if event.get("event_type") in ("died", "port_down") else 0
                            )
                            # Attention filter for health events
                            import time as _time
                            att_score = self._attention_filter.evaluate(
                                event_type="health",
                                payload_size=0,
                                elapsed_since_last_event=0.0,
                                senses_attentiveness=(
                                    self._current_senses.attentiveness
                                    if self._current_senses else 0.5
                                ),
                                conversation_active=self._chatting,
                            )
                            if att_score.should_attend:
                                msg = event.get("message", "Health event detected.")
                                chat_area = self._safe_query(ChatArea)
                                if chat_area:
                                    chat_area.add_message("system", msg)
                                self._speak(msg)
                    except Exception as e:
                        logger.debug(f"Health event processing skipped: {e}")

                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    # --- Cleanup ---

    async def on_unmount(self) -> None:
        """Clean up perception engine, launcher, and consumer on screen unmount."""
        self._unmounted = True
        if self._perception:
            await self._perception.stop()
        if self._perception_consumer_task:
            self._perception_consumer_task.cancel()
            try:
                await self._perception_consumer_task
            except asyncio.CancelledError:
                pass
        if self._launcher_poll_task and not self._launcher_poll_task.done():
            self._launcher_poll_task.cancel()
        if self._voice_consumer_task and not self._voice_consumer_task.done():
            self._voice_consumer_task.cancel()
            try:
                await self._voice_consumer_task
            except asyncio.CancelledError:
                pass
        # Stop duplex voice
        if self._duplex_active:
            await self._stop_duplex_voice()
        if self._recall_engine:
            self._recall_engine.close()
        # Close session memory: compress conversation into episode
        if self._bridge and self._store:
            try:
                self._bridge.close_session_memory(
                    self._store.session_id,
                    db_path=self._store._path,
                )
            except Exception as e:
                logger.debug(f"Session memory close skipped: {e}")
        if self._journal:
            self._journal.close()
        if self._daemon:
            await self._daemon.stop()
        if self._bridge:
            await self._bridge.stop_all_appendages()
            await self._bridge.stop_project()
