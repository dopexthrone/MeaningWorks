"""
Tests for test-readiness wiring (WI-1 through WI-5).

Covers:
- Pipeline mode default and threading
- Appendage config wiring (max_concurrent, auto-dissolve, solidify-on-use)
- Panel server integration
- /ideas command and idea surfacing
- Perception → goal generation
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mother.config import MotherConfig
from mother.bridge import EngineBridge
from mother.screens.chat import (
    ChatScreen,
    HELP_TEXT,
    SLASH_COMMANDS,
    parse_response,
)


# --- WI-1: Pipeline mode ---


class TestPipelineModeDefault:
    """Pipeline mode defaults to staged and threads through to engine."""

    def test_config_defaults_to_staged(self):
        config = MotherConfig()
        assert config.pipeline_mode == "staged"

    def test_bridge_accepts_pipeline_mode(self):
        bridge = EngineBridge(pipeline_mode="staged")
        assert bridge._pipeline_mode == "staged"

    def test_bridge_defaults_to_staged(self):
        bridge = EngineBridge()
        assert bridge._pipeline_mode == "staged"

    def test_bridge_passes_pipeline_mode_to_engine(self):
        bridge = EngineBridge(pipeline_mode="staged")
        with patch("core.engine.MotherlabsEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            bridge._get_engine()
            MockEngine.assert_called_once()
            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs["pipeline_mode"] == "staged"

    def test_bridge_legacy_mode_passes_through(self):
        bridge = EngineBridge(pipeline_mode="legacy")
        with patch("core.engine.MotherlabsEngine") as MockEngine:
            MockEngine.return_value = MagicMock()
            bridge._get_engine()
            call_kwargs = MockEngine.call_args[1]
            assert call_kwargs["pipeline_mode"] == "legacy"


# --- WI-2a: appendage_max_concurrent ---


class TestAppendageMaxConcurrent:
    """spawn_appendage rejects when at max_concurrent."""

    def test_bridge_accepts_max_concurrent(self):
        bridge = EngineBridge(max_concurrent_appendages=3)
        assert bridge._max_concurrent_appendages == 3

    def test_spawn_rejects_at_limit(self):
        bridge = EngineBridge(max_concurrent_appendages=2)
        # Simulate 2 already running
        bridge._appendage_processes = {1: MagicMock(), 2: MagicMock()}
        result = asyncio.run(bridge.spawn_appendage("/tmp/test.db", 3))
        assert result["success"] is False
        assert "Max concurrent" in result["error"]

    def test_spawn_allowed_below_limit(self):
        bridge = EngineBridge(max_concurrent_appendages=5)
        bridge._appendage_processes = {1: MagicMock()}
        # Will fail for other reasons (no db), but won't hit the limit gate
        with patch("mother.appendage.AppendageStore") as MockStore:
            MockStore.return_value.get.return_value = None
            result = asyncio.run(bridge.spawn_appendage("/tmp/test.db", 2))
            assert "Max concurrent" not in result.get("error", "")


# --- WI-2b: appendage_auto_dissolve ---


class TestAppendageAutoDissolve:
    """Auto-dissolve removes stale appendages."""

    def test_auto_dissolve_stale(self):
        """Stale appendages get dissolved."""
        config = MotherConfig(appendage_auto_dissolve_hours=24)
        screen = ChatScreen(config=config)

        mock_store = MagicMock()
        screen._store = MagicMock()
        screen._store._path = "/tmp/test.db"
        screen._bridge = MagicMock()
        screen._bridge.dissolve_appendage = AsyncMock()

        stale_spec = MagicMock()
        stale_spec.status = "built"
        stale_spec.created_at = time.time() - (48 * 3600)  # 48 hours ago
        stale_spec.appendage_id = 1
        stale_spec.name = "old-app"

        fresh_spec = MagicMock()
        fresh_spec.status = "built"
        fresh_spec.created_at = time.time() - 3600  # 1 hour ago
        fresh_spec.appendage_id = 2
        fresh_spec.name = "new-app"

        with patch("mother.appendage.AppendageStore") as MockStore:
            MockStore.return_value.all.return_value = [stale_spec, fresh_spec]
            asyncio.run(screen._auto_dissolve_stale())

        screen._bridge.dissolve_appendage.assert_called_once_with("/tmp/test.db", 1)


# --- WI-2c: appendage solidify-on-use ---


class TestAppendageSolidifyOnUse:
    """Auto-solidify triggers at min_uses_to_solidify threshold."""

    def test_bridge_accepts_min_uses(self):
        bridge = EngineBridge(min_uses_to_solidify=10)
        assert bridge._min_uses_to_solidify == 10

    def test_solidify_on_threshold(self):
        """Appendage solidifies after reaching use threshold."""
        bridge = EngineBridge(min_uses_to_solidify=3)

        mock_store = MagicMock()
        mock_spec = MagicMock()
        mock_spec.status = "active"
        mock_spec.appendage_id = 1
        mock_spec.pid = 123

        # After invoke, refreshed spec has use_count=3 (meets threshold)
        refreshed_spec = MagicMock()
        refreshed_spec.status = "active"
        refreshed_spec.use_count = 3
        refreshed_spec.pid = 123

        mock_store.get.side_effect = [mock_spec, refreshed_spec]
        mock_store.get_by_name.return_value = None

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_proc.invoke.return_value = MagicMock(
            success=True, output="ok", error="", duration_seconds=0.1,
        )

        bridge._appendage_processes = {1: mock_proc}

        with patch("mother.appendage.AppendageStore", return_value=mock_store):
            result = asyncio.run(bridge.invoke_appendage("/tmp/test.db", 1, {}))

        assert result["success"] is True
        mock_store.update_status.assert_any_call(1, "solidified", pid=123)


# --- WI-3: Panel server ---


class TestPanelServer:
    """Panel server starts when enabled in config."""

    def test_panel_server_not_started_by_default(self):
        config = MotherConfig()
        assert config.panel_server_enabled is False
        screen = ChatScreen(config=config)
        assert screen._panel_server_thread is None

    def test_start_panel_server_creates_thread(self):
        config = MotherConfig(panel_server_enabled=True, panel_server_port=17770)
        screen = ChatScreen(config=config)
        screen._bridge = MagicMock()

        with patch("mother.screens.chat.threading.Thread") as MockThread:
            mock_thread = MagicMock()
            MockThread.return_value = mock_thread
            with patch("mother.panel_server.create_app") as MockCreate:
                MockCreate.return_value = MagicMock()
                screen._start_panel_server()

            MockThread.assert_called_once()
            mock_thread.start.assert_called_once()
            assert screen._panel_server_thread is mock_thread


# --- WI-4: Idea journal ---


class TestIdeasCommand:
    """/ideas command is registered and lists pending ideas."""

    def test_ideas_in_slash_commands(self):
        assert "/ideas" in SLASH_COMMANDS

    def test_ideas_in_help_text(self):
        assert "/ideas" in HELP_TEXT

    def test_list_ideas_no_pending(self):
        """Shows 'No pending ideas' when empty."""
        screen = ChatScreen()
        screen._store = MagicMock()
        screen._store._path = "/tmp/test.db"
        screen._bridge = MagicMock()
        screen._bridge.get_pending_ideas = AsyncMock(return_value=[])

        mock_chat_area = MagicMock()
        screen._safe_query = MagicMock(return_value=mock_chat_area)

        asyncio.run(screen._list_ideas_worker())

        mock_chat_area.add_ai_message.assert_called_once_with("No pending ideas.")

    def test_list_ideas_with_pending(self):
        """Shows pending ideas when they exist."""
        screen = ChatScreen()
        screen._store = MagicMock()
        screen._store._path = "/tmp/test.db"
        screen._bridge = MagicMock()
        screen._bridge.get_pending_ideas = AsyncMock(return_value=[
            {"idea_id": 1, "description": "Build a dashboard", "priority": "high", "timestamp": 0},
            {"idea_id": 2, "description": "Add dark mode", "priority": "normal", "timestamp": 0},
        ])

        mock_chat_area = MagicMock()
        screen._safe_query = MagicMock(return_value=mock_chat_area)

        asyncio.run(screen._list_ideas_worker())

        msg = mock_chat_area.add_ai_message.call_args[0][0]
        assert "2 pending ideas" in msg
        assert "Build a dashboard" in msg
        assert "[high]" in msg


class TestIdeasSurfacing:
    """Ideas surfaced during autonomous idle with cooldown."""

    def test_surface_ideas_with_cooldown(self):
        screen = ChatScreen()
        screen._store = MagicMock()
        screen._store._path = "/tmp/test.db"
        screen._bridge = MagicMock()
        screen._bridge.count_pending_ideas = AsyncMock(return_value=2)
        screen._bridge.get_pending_ideas = AsyncMock(return_value=[
            {"idea_id": 1, "description": "Idea one", "priority": "normal", "timestamp": 0},
        ])
        mock_chat_area = MagicMock()
        screen._safe_query = MagicMock(return_value=mock_chat_area)

        # First call surfaces ideas
        asyncio.run(screen._surface_pending_ideas())
        assert mock_chat_area.add_ai_message.called
        assert screen._last_ideas_surfaced > 0

        # Second call within cooldown does nothing
        mock_chat_area.reset_mock()
        asyncio.run(screen._surface_pending_ideas())
        assert not mock_chat_area.add_ai_message.called

    def test_surface_ideas_skips_when_no_ideas(self):
        screen = ChatScreen()
        screen._store = MagicMock()
        screen._store._path = "/tmp/test.db"
        screen._bridge = MagicMock()
        screen._bridge.count_pending_ideas = AsyncMock(return_value=0)
        mock_chat_area = MagicMock()
        screen._safe_query = MagicMock(return_value=mock_chat_area)

        asyncio.run(screen._surface_pending_ideas())
        assert not mock_chat_area.add_ai_message.called


# --- WI-5: Perception → goal ---


class TestPerceptionGoalGeneration:
    """Perception prompt allows goal/idea creation."""

    def test_perception_prompt_includes_goal_action(self):
        """The proactive perception prompt should mention goal and idea actions."""
        # We verify the prompt text by reading the source — this test just
        # confirms the ACTION:goal pattern is present in the prompt string
        # used by _proactive_perception.
        import inspect
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._proactive_perception)
        assert "[ACTION:goal]" in source
        assert "[ACTION:idea]" in source

    def test_parse_response_goal_action(self):
        """parse_response correctly extracts goal actions."""
        raw = "I noticed something. [ACTION:goal]Monitor disk usage[/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "goal"
        assert parsed["action_arg"] == "Monitor disk usage"

    def test_parse_response_idea_action(self):
        """parse_response correctly extracts idea actions."""
        raw = "Interesting pattern. [ACTION:idea]Build a time tracker[/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "idea"
        assert parsed["action_arg"] == "Build a time tracker"
