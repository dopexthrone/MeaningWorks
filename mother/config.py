"""
Mother configuration — persistent settings at ~/.motherlabs/mother.json.

Separate from engine config. Coexists without conflict.
"""

import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Dict, Optional


DEFAULT_CONFIG_DIR = Path.home() / ".motherlabs"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "mother.json"

PROVIDERS = ("claude", "openai", "grok", "gemini", "local")

DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "openai": "gpt-5.1",
    "grok": "grok-4-1-fast-reasoning",
    "gemini": "gemini-2.0-flash",
    "local": "llama3:8b",
}

ENV_VARS = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "grok": "XAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "local": "LOCAL_MODEL_URL",
}


@dataclass
class MotherConfig:
    """Persistent Mother configuration."""

    name: str = "Mother"
    user_name: str = "User"
    personality: str = "composed"
    theme: str = "default"  # "default" or "alien"
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    chat_model: str = ""  # model for conversation (cheaper). Empty = use main model.
    api_keys: Dict[str, str] = field(default_factory=dict)
    file_access: bool = True
    auto_compile: bool = False
    cost_limit: float = 100.0
    setup_complete: bool = False
    output_dir: str = str(Path.home() / "motherlabs" / "projects")
    voice_enabled: bool = False
    voice_id: str = "2obv5y63xKRNiEZAPxGD"
    voice_model: str = "eleven_v3"
    # Duplex voice (real-time conversational AI via ElevenLabs WebSocket)
    voice_duplex_enabled: bool = False
    voice_duplex_port: int = 11411
    voice_duplex_agent_id: str = ""  # auto-managed, cached after first creation
    voice_duplex_language: str = "en"
    screen_capture_enabled: bool = False
    microphone_enabled: bool = False
    camera_enabled: bool = False
    ambient_listening: bool = False
    screen_monitoring: bool = False
    perception_budget: float = 5.00
    screen_poll_interval: int = 10
    camera_poll_interval: int = 30
    claude_code_enabled: bool = False
    claude_code_path: str = str(Path.home() / ".local" / "bin" / "claude")
    claude_code_budget: float = 20.0
    # Local LLM
    local_base_url: str = "http://localhost:11434"
    local_model: str = "llama3:8b"
    # Autonomic operating mode
    autonomous_enabled: bool = True
    autonomous_tick_seconds: int = 60
    autonomous_budget_per_cycle: float = 1.00
    autonomous_budget_per_session: float = 10.00
    max_chain_depth: int = 20
    max_goal_stalls: int = 10      # consecutive no-ops before "stuck"
    max_goal_attempts: int = 10    # compilation failures before "stuck"
    # Dialogue initiative (impulse system)
    dialogue_initiative_enabled: bool = False
    impulse_tick_seconds: int = 90
    impulse_budget_per_session: float = 0.50
    # Appendage spawning
    appendage_enabled: bool = False
    appendage_build_budget: float = 3.0
    appendage_max_concurrent: int = 5
    appendage_base_dir: str = str(Path.home() / "motherlabs" / "appendages")
    appendage_auto_dissolve_hours: int = 48
    appendage_min_uses_to_solidify: int = 5
    # Inner dialogue / metabolism
    metabolism_enabled: bool = False
    metabolism_tick_seconds: int = 120
    metabolism_budget_per_session: float = 0.30
    metabolism_sleep_start_hour: int = 2
    metabolism_sleep_end_hour: int = 7
    metabolism_max_thoughts_per_session: int = 20
    metabolism_deep_think_timeout_minutes: int = 15
    # Pipeline mode
    pipeline_mode: str = "staged"
    # Panel server
    panel_server_enabled: bool = False
    panel_server_port: int = 7770
    panel_senses_push_interval: float = 2.0
    # Self-marketing
    auto_push_enabled: bool = False
    auto_push_after_build: bool = True  # push to remote after every successful self-build
    auto_tweet_enabled: bool = False
    # Social publishing
    discord_webhook_url: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    auto_discord_enabled: bool = False
    auto_bluesky_enabled: bool = False
    auto_publish_projects: bool = False    # auto-create GitHub repo for emitted projects
    publish_projects_public: bool = False  # default: private repos
    mother_git_name: str = "Mother"
    mother_git_email: str = "mother@motherlabs.ai"
    # WhatsApp integration
    whatsapp_enabled: bool = True
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    user_whatsapp_number: str = ""
    whatsapp_webhook_enabled: bool = True
    whatsapp_webhook_port: int = 8080
    ngrok_auth_token: str = ""
    # Output routing
    routing_enabled: bool = True
    routing_whatsapp_daily_limit: int = 50
    routing_whatsapp_night_digest: bool = True
    routing_night_start_hour: int = 23
    routing_night_end_hour: int = 7
    # Weekly build governance
    weekly_build_enabled: bool = True          # master toggle for weekly governance
    weekly_briefing_day: int = 6               # 0=Mon..6=Sun
    weekly_briefing_hour: int = 10             # hour to present briefing (local time)
    build_window_start_hour: int = 22          # start of build window
    build_window_end_hour: int = 6             # end of build window
    build_window_day: int = 6                  # day the window opens (Sunday night)
    build_max_per_window: int = 10             # max builds per window
    # Daemon mode
    daemon_health_check_interval: int = 300
    daemon_max_queue_size: int = 10
    daemon_auto_heal: bool = True
    daemon_idle_shutdown_hours: float = 0
    daemon_log_file: str = ""

    def get_model(self) -> str:
        """Return model, falling back to provider default."""
        return self.model or DEFAULT_MODELS.get(self.provider, "claude-sonnet-4-20250514")

    def get_env_var(self) -> str:
        """Return the env var name for the current provider."""
        return ENV_VARS.get(self.provider, "ANTHROPIC_API_KEY")


def _validate_config_types(data: dict) -> dict:
    """Validate and coerce config values to expected types.

    Drops fields with wrong types rather than crashing.
    Returns a cleaned dict safe to pass to MotherConfig().
    """
    field_types = {f.name: f.type for f in fields(MotherConfig)}
    cleaned = {}
    for key, val in data.items():
        if key not in field_types:
            continue
        expected = field_types[key]
        # bool must be checked before int (bool is subclass of int in Python)
        if expected is bool:
            if not isinstance(val, bool):
                continue
        elif expected is int:
            if isinstance(val, bool):
                continue  # reject bool for int fields
            if not isinstance(val, int):
                if isinstance(val, float) and val == int(val):
                    val = int(val)
                else:
                    continue
        elif expected is float:
            if not isinstance(val, (int, float)):
                continue
        elif expected is str:
            if not isinstance(val, str):
                continue
        # Complex types (Dict, etc.) pass through without validation
        cleaned[key] = val
    return cleaned


def load_config(path: Optional[str] = None) -> MotherConfig:
    """Load config from disk. Returns defaults if file missing.

    Validates field types — drops invalid values rather than crashing.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return MotherConfig()

    try:
        data = json.loads(config_path.read_text())
        filtered = _validate_config_types(data)
        return MotherConfig(**filtered)
    except (json.JSONDecodeError, TypeError):
        return MotherConfig()


def save_config(config: MotherConfig, path: Optional[str] = None) -> Path:
    """Save config to disk with restricted permissions (0600).

    Creates directory if needed. Returns path.
    Config may contain API keys — must not be world-readable.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(asdict(config), indent=2))
    # Restrict to owner-only read/write (contains credentials)
    try:
        os.chmod(str(config_path), 0o600)
    except OSError:
        pass  # Windows or restricted filesystem
    return config_path


def detect_first_run(path: Optional[str] = None) -> bool:
    """True if setup has never been completed."""
    config = load_config(path)
    return not config.setup_complete
