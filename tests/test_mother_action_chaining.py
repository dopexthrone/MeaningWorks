"""Tests for action chaining (agentic loop) in chat.py."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from mother.screens.chat import parse_response, ChatScreen, ActionResult
from mother.config import MotherConfig


# --- _execute_action extraction ---

class TestExecuteAction:
    """Test that _execute_action routes correctly and returns expected values."""

    @pytest.fixture
    def screen(self):
        """Create a minimal ChatScreen for testing _execute_action."""
        config = MotherConfig()
        s = ChatScreen(config=config)
        # Stub out all _run_* methods so they don't interact with TUI
        s._run_compile = MagicMock()
        s._run_build = MagicMock()
        s._run_tools = MagicMock()
        s._run_status = MagicMock()
        s._run_search = MagicMock()
        s._run_open = MagicMock()
        s._run_file_action = MagicMock()
        s._run_launch = MagicMock()
        s._run_stop = MagicMock()
        s._run_capture = MagicMock()
        s._run_camera = MagicMock()
        s._run_use_tool = MagicMock()
        s._run_add_idea = MagicMock()
        s._run_self_build = MagicMock()
        s._run_github_push = MagicMock()
        s._run_tweet = MagicMock()
        s._run_discover_peers = MagicMock()
        s._run_list_peers = MagicMock()
        s._run_delegate = MagicMock()
        s._run_whatsapp = MagicMock()
        s._run_add_goal = MagicMock()
        s._run_list_goals = MagicMock()
        s._run_complete_goal = MagicMock()
        s._run_listen = MagicMock()
        s._run_integrate = MagicMock()
        s._pending_permission = None
        s._microphone_bridge = None
        s._camera_bridge = None
        return s

    # Chainable actions return ActionResult
    def test_compile_returns_action_result(self, screen):
        parsed = {"action": "compile", "action_arg": "booking system"}
        result = screen._execute_action(parsed)
        assert isinstance(result, ActionResult)
        assert "booking system" in result.message
        assert result.pending is True
        screen._run_compile.assert_called_once_with("booking system")

    def test_build_returns_action_result(self, screen):
        parsed = {"action": "build", "action_arg": "booking system"}
        result = screen._execute_action(parsed)
        assert isinstance(result, ActionResult)
        assert result.pending is True
        screen._run_build.assert_called_once_with("booking system")

    def test_tools_returns_action_result(self, screen):
        result = screen._execute_action({"action": "tools", "action_arg": ""})
        assert isinstance(result, ActionResult)
        assert result.pending is False
        screen._run_tools.assert_called_once()

    def test_status_returns_action_result(self, screen):
        result = screen._execute_action({"action": "status", "action_arg": ""})
        assert isinstance(result, ActionResult)
        assert result.pending is False
        screen._run_status.assert_called_once()

    def test_search_returns_action_result(self, screen):
        result = screen._execute_action({"action": "search", "action_arg": "resume"})
        assert isinstance(result, ActionResult)
        assert result.pending is True
        screen._run_search.assert_called_once_with("resume")

    def test_open_returns_action_result(self, screen):
        result = screen._execute_action({"action": "open", "action_arg": "/tmp/f.txt"})
        assert isinstance(result, ActionResult)
        assert result.pending is False

    def test_file_returns_action_result(self, screen):
        result = screen._execute_action({"action": "file", "action_arg": "write: /tmp/f.txt"})
        assert isinstance(result, ActionResult)
        assert result.pending is False

    def test_goal_returns_action_result(self, screen):
        result = screen._execute_action({"action": "goal", "action_arg": "build api"})
        assert isinstance(result, ActionResult)
        assert "build api" in result.message
        assert result.pending is False
        screen._run_add_goal.assert_called_once_with("build api")

    def test_goals_returns_action_result(self, screen):
        result = screen._execute_action({"action": "goals", "action_arg": ""})
        assert isinstance(result, ActionResult)
        screen._run_list_goals.assert_called_once()

    def test_goal_done_returns_action_result(self, screen):
        result = screen._execute_action({"action": "goal_done", "action_arg": "1"})
        assert isinstance(result, ActionResult)
        screen._run_complete_goal.assert_called_once_with("1")

    # Terminal actions return None
    def test_launch_returns_none(self, screen):
        result = screen._execute_action({"action": "launch", "action_arg": ""})
        assert result is None
        screen._run_launch.assert_called_once()

    def test_stop_returns_none(self, screen):
        result = screen._execute_action({"action": "stop", "action_arg": ""})
        assert result is None
        screen._run_stop.assert_called_once()

    def test_idea_returns_none(self, screen):
        result = screen._execute_action({"action": "idea", "action_arg": "dark mode"})
        assert result is None
        screen._run_add_idea.assert_called_once()

    def test_self_build_returns_none(self, screen):
        result = screen._execute_action({"action": "self_build", "action_arg": "webhooks"})
        assert result is None
        screen._run_self_build.assert_called_once()

    def test_github_push_returns_none(self, screen):
        result = screen._execute_action({"action": "github_push", "action_arg": ""})
        assert result is None
        screen._run_github_push.assert_called_once()

    def test_tweet_returns_none(self, screen):
        result = screen._execute_action({"action": "tweet", "action_arg": "hello"})
        assert result is None
        screen._run_tweet.assert_called_once()

    # No action / done
    def test_no_action_returns_none(self, screen):
        result = screen._execute_action({"action": None, "action_arg": ""})
        assert result is None

    def test_done_action_returns_none(self, screen):
        result = screen._execute_action({"action": "done", "action_arg": ""})
        assert result is None

    def test_empty_action_returns_none(self, screen):
        result = screen._execute_action({})
        assert result is None

    # Enable mic/camera terminal
    def test_enable_mic_returns_none(self, screen):
        result = screen._execute_action({"action": "enable_mic", "action_arg": ""})
        assert result is None
        assert screen._pending_permission == "microphone"

    def test_enable_camera_returns_none(self, screen):
        result = screen._execute_action({"action": "enable_camera", "action_arg": ""})
        assert result is None
        assert screen._pending_permission == "camera"


# --- Chain depth limiting ---

class TestChainDepth:
    def test_max_chain_depth_default(self):
        config = MotherConfig()
        assert config.max_chain_depth == 20

    def test_max_chain_depth_custom(self):
        config = MotherConfig(max_chain_depth=3)
        assert config.max_chain_depth == 3


# --- parse_response for done action ---

class TestParseResponseDone:
    def test_done_action_parsed(self):
        raw = "[ACTION:done][/ACTION][VOICE]All set.[/VOICE]"
        parsed = parse_response(raw)
        assert parsed["action"] == "done"
        assert parsed["voice"] == "All set."

    def test_done_action_no_arg(self):
        raw = "[ACTION:done][/ACTION]"
        parsed = parse_response(raw)
        assert parsed["action"] == "done"
        assert parsed["action_arg"] == ""


# --- _with_action_result ---

class TestWithActionResult:
    def test_creates_new_context_data_with_string(self):
        from mother.context import ContextData
        config = MotherConfig()
        s = ChatScreen(config=config)
        ctx = ContextData(name="Mother")
        result = s._with_action_result(ctx, "Compiled 12 components")
        assert result.pending_action_result == "Compiled 12 components"
        assert result.name == "Mother"

    def test_creates_new_context_data_with_action_result(self):
        from mother.context import ContextData
        config = MotherConfig()
        s = ChatScreen(config=config)
        ctx = ContextData(name="Mother")
        ar = ActionResult(message="Status displayed")
        result = s._with_action_result(ctx, ar)
        assert result.pending_action_result == "Status displayed"

    def test_creates_context_with_pending_action_result(self):
        from mother.context import ContextData
        config = MotherConfig()
        s = ChatScreen(config=config)
        ctx = ContextData(name="Mother")
        ar = ActionResult(message="Compilation started", pending=True)
        result = s._with_action_result(ctx, ar)
        assert result.pending_action_result == "[PENDING] Compilation started"

    def test_truncates_long_result(self):
        from mother.context import ContextData
        config = MotherConfig()
        s = ChatScreen(config=config)
        ctx = ContextData()
        result = s._with_action_result(ctx, "x" * 500)
        assert len(result.pending_action_result) == 200
