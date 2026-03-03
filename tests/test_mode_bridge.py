"""Tests for compilation mode wiring in bridge.py and chat.py."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================
# Bridge compile mode parameter
# ============================================================

class TestBridgeCompileMode:
    """bridge.compile() accepts mode parameter."""

    def test_compile_signature_has_mode(self):
        """compile() should accept a mode kwarg."""
        import inspect
        from mother.bridge import EngineBridge
        sig = inspect.signature(EngineBridge.compile)
        assert "mode" in sig.parameters

    def test_compile_mode_default_is_build(self):
        import inspect
        from mother.bridge import EngineBridge
        sig = inspect.signature(EngineBridge.compile)
        assert sig.parameters["mode"].default == "build"


# ============================================================
# Bridge static accessors
# ============================================================

class TestBridgeAccessors:
    """Static accessor methods for mode-specific results."""

    def test_get_context_map_from_result(self):
        from mother.bridge import EngineBridge
        mock_result = MagicMock()
        mock_result.context_map = {"concepts": [{"name": "X"}]}
        assert EngineBridge.get_context_map(mock_result) == {"concepts": [{"name": "X"}]}

    def test_get_context_map_none_when_missing(self):
        from mother.bridge import EngineBridge
        mock_result = MagicMock(spec=[])  # no attributes
        assert EngineBridge.get_context_map(mock_result) is None

    def test_get_exploration_map_from_result(self):
        from mother.bridge import EngineBridge
        mock_result = MagicMock()
        mock_result.exploration_map = {"insights": []}
        assert EngineBridge.get_exploration_map(mock_result) == {"insights": []}

    def test_get_exploration_map_none_when_missing(self):
        from mother.bridge import EngineBridge
        mock_result = MagicMock(spec=[])
        assert EngineBridge.get_exploration_map(mock_result) is None

    def test_get_context_map_is_static(self):
        from mother.bridge import EngineBridge
        assert isinstance(
            inspect.getattr_static(EngineBridge, "get_context_map"),
            staticmethod,
        )

    def test_get_exploration_map_is_static(self):
        from mother.bridge import EngineBridge
        assert isinstance(
            inspect.getattr_static(EngineBridge, "get_exploration_map"),
            staticmethod,
        )


import inspect


# ============================================================
# Compile worker mode threading
# ============================================================

class TestCompileWorkerMode:
    """_compile_worker passes mode to bridge.compile()."""

    def test_compile_worker_signature_has_mode(self):
        """_compile_worker should accept a mode parameter."""
        from mother.screens.chat import ChatScreen
        sig = inspect.signature(ChatScreen._compile_worker)
        assert "mode" in sig.parameters

    def test_compile_worker_mode_default_build(self):
        from mother.screens.chat import ChatScreen
        sig = inspect.signature(ChatScreen._compile_worker)
        assert sig.parameters["mode"].default == "build"


# ============================================================
# Run methods exist
# ============================================================

class TestModeRunMethods:
    """chat.py has _run_context_compile and _run_explore_compile."""

    def test_run_context_compile_exists(self):
        from mother.screens.chat import ChatScreen
        assert hasattr(ChatScreen, "_run_context_compile")
        assert callable(ChatScreen._run_context_compile)

    def test_run_explore_compile_exists(self):
        from mother.screens.chat import ChatScreen
        assert hasattr(ChatScreen, "_run_explore_compile")
        assert callable(ChatScreen._run_explore_compile)


# ============================================================
# Action dispatch routing
# ============================================================

class TestActionDispatch:
    """_execute_action routes context and explore actions."""

    def test_execute_action_handles_context(self):
        """_execute_action should recognize 'context' action."""
        from mother.screens.chat import ChatScreen
        # Verify the method exists and would handle context action
        # (full integration requires TUI app context, so we verify structurally)
        source = inspect.getsource(ChatScreen._execute_action)
        assert '"context"' in source

    def test_execute_action_handles_explore(self):
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._execute_action)
        assert '"explore"' in source


# ============================================================
# Slash command routing
# ============================================================

class TestSlashCommands:
    """Slash commands /context and /explore are routed."""

    def test_context_slash_command_routed(self):
        """Verify /context routing exists in command handler."""
        from mother.screens.chat import ChatScreen
        # Structural check — the handler code references /context
        source = inspect.getsource(ChatScreen)
        assert '"/context"' in source

    def test_explore_slash_command_routed(self):
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen)
        assert '"/explore"' in source


# ============================================================
# Handle compile success — mode-specific display
# ============================================================

class TestHandleCompileSuccessModes:
    """_handle_compile_success handles context_map and exploration_map."""

    def test_context_map_handling_in_source(self):
        """_handle_compile_success should check for context_map."""
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._handle_compile_success)
        assert "context_map" in source

    def test_exploration_map_handling_in_source(self):
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._handle_compile_success)
        assert "exploration_map" in source

    def test_context_map_early_return(self):
        """Context map display should return early (skip blueprint display)."""
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._handle_compile_success)
        # context_map block should return before reaching component display
        ctx_idx = source.index("context_map")
        return_idx = source.index("return", ctx_idx)
        components_idx = source.index("components = result.blueprint")
        assert return_idx < components_idx

    def test_exploration_map_early_return(self):
        from mother.screens.chat import ChatScreen
        source = inspect.getsource(ChatScreen._handle_compile_success)
        exp_idx = source.index("exploration_map")
        return_idx = source.index("return", exp_idx)
        components_idx = source.index("components = result.blueprint")
        assert return_idx < components_idx
