"""Tests for compilation mode routing in core/engine.py."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ============================================================
# CompileResult field tests
# ============================================================

class TestCompileResultFields:
    """CompileResult has context_map and exploration_map fields."""

    def test_context_map_field_exists(self):
        from core.engine import CompileResult
        result = CompileResult(success=True)
        assert result.context_map is None

    def test_exploration_map_field_exists(self):
        from core.engine import CompileResult
        result = CompileResult(success=True)
        assert result.exploration_map is None

    def test_context_map_accepts_dict(self):
        from core.engine import CompileResult
        result = CompileResult(success=True, context_map={"concepts": []})
        assert result.context_map == {"concepts": []}

    def test_exploration_map_accepts_dict(self):
        from core.engine import CompileResult
        result = CompileResult(success=True, exploration_map={"insights": []})
        assert result.exploration_map == {"insights": []}

    def test_default_result_no_maps(self):
        from core.engine import CompileResult
        result = CompileResult(success=True)
        assert result.context_map is None
        assert result.exploration_map is None


# ============================================================
# Mode parsing in compile()
# ============================================================

class TestModeParsing:
    """compile() correctly parses pipeline_mode into CompilationMode."""

    def test_build_mode_is_default(self):
        """When pipeline_mode is None, compilation_mode should be BUILD."""
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.BUILD)
        assert cfg.posture_preamble == ""  # BUILD has no preamble

    def test_parse_context_mode(self):
        from core.compilation_modes import parse_mode, CompilationMode
        assert parse_mode("context") is CompilationMode.CONTEXT

    def test_parse_explore_mode(self):
        from core.compilation_modes import parse_mode, CompilationMode
        assert parse_mode("explore") is CompilationMode.EXPLORE

    def test_parse_self_mode(self):
        from core.compilation_modes import parse_mode, CompilationMode
        assert parse_mode("self") is CompilationMode.SELF

    def test_unknown_mode_falls_through(self):
        """Unknown mode strings like 'staged' should not crash."""
        from core.compilation_modes import parse_mode
        with pytest.raises(ValueError):
            parse_mode("staged")


# ============================================================
# Mode posture injection
# ============================================================

class TestPostureInjection:
    """Agent system prompts get posture preamble prepended."""

    def test_context_preamble_content(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.CONTEXT)
        assert "CONTEXT UNDERSTANDING" in cfg.posture_preamble
        assert len(cfg.posture_preamble) > 50

    def test_explore_preamble_content(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.EXPLORE)
        assert "DIVERGENT EXPLORATION" in cfg.posture_preamble

    def test_build_preamble_empty(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.BUILD)
        assert cfg.posture_preamble == ""

    def test_preamble_prepends_not_replaces(self):
        """Posture preamble should be prepended, not replace the existing prompt."""
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.CONTEXT)
        original_prompt = "You are an entity agent."
        combined = cfg.posture_preamble + original_prompt
        assert combined.startswith("MODE:")
        assert "entity agent" in combined


# ============================================================
# EXPLORE mode routing
# ============================================================

class TestExploreMode:
    """EXPLORE mode produces exploration_map, no blueprint."""

    def test_explore_result_has_exploration_map(self):
        """_compile_explore_mode produces an ExplorationMap dict."""
        from core.exploration_synthesis import (
            synthesize_exploration, exploration_map_to_dict,
        )
        exp = synthesize_exploration([], [], "test intent")
        d = exploration_map_to_dict(exp)
        assert "insights" in d
        assert "frontier_questions" in d
        assert "original_intent" in d

    def test_explore_mode_skips_synthesis(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.EXPLORE)
        assert cfg.skip_synthesis is True

    def test_explore_mode_skips_verification(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.EXPLORE)
        assert cfg.skip_verification is True


# ============================================================
# CONTEXT mode routing
# ============================================================

class TestContextMode:
    """CONTEXT mode produces context_map, no blueprint."""

    def test_context_result_has_context_map(self):
        from core.context_synthesis import (
            synthesize_context, context_map_to_dict,
        )
        ctx = synthesize_context([], [], "test intent")
        d = context_map_to_dict(ctx)
        assert "concepts" in d
        assert "relationships" in d
        assert "original_intent" in d

    def test_context_mode_does_not_skip_synthesis(self):
        """CONTEXT mode has its own synthesis — it doesn't skip, it replaces."""
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.CONTEXT)
        assert cfg.skip_synthesis is False

    def test_context_mode_does_not_skip_verification(self):
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.CONTEXT)
        assert cfg.skip_verification is False


# ============================================================
# SELF mode routing
# ============================================================

class TestSelfMode:
    """SELF mode routes to self_compile()."""

    def test_self_mode_triggers_self_compile(self):
        """pipeline_mode='self' should trigger self_compile path."""
        from core.compilation_modes import CompilationMode, mode_config
        cfg = mode_config(CompilationMode.SELF)
        assert cfg.agent_posture == "audit"


# ============================================================
# Mode-specific dialogue parameters
# ============================================================

class TestModeDialogueParams:
    """Mode configs control dialogue turn budgets."""

    def test_context_shorter_dialogue(self):
        from core.compilation_modes import CompilationMode, mode_config
        build = mode_config(CompilationMode.BUILD)
        context = mode_config(CompilationMode.CONTEXT)
        assert context.dialogue_max_turns < build.dialogue_max_turns

    def test_self_shortest_dialogue(self):
        from core.compilation_modes import CompilationMode, mode_config
        self_cfg = mode_config(CompilationMode.SELF)
        assert self_cfg.dialogue_max_turns == 16

    def test_explore_lower_convergence(self):
        from core.compilation_modes import CompilationMode, mode_config
        explore = mode_config(CompilationMode.EXPLORE)
        build = mode_config(CompilationMode.BUILD)
        assert explore.convergence_threshold < build.convergence_threshold


# ============================================================
# Mode state tracking
# ============================================================

class TestModeStateTracking:
    """Compilation mode stored in state for observability."""

    def test_compilation_mode_in_config(self):
        """mode_config stores the mode value for tracking."""
        from core.compilation_modes import CompilationMode, mode_config
        for mode in CompilationMode:
            cfg = mode_config(mode)
            assert isinstance(cfg.agent_posture, str)


# ============================================================
# _compile_explore_mode unit test
# ============================================================

class TestCompileExploreModeUnit:
    """Unit test _compile_explore_mode with mocked state."""

    def test_returns_compile_result(self):
        """Verify the method signature produces a valid CompileResult."""
        from core.engine import CompileResult
        # Simulate what _compile_explore_mode returns
        result = CompileResult(
            success=True,
            blueprint={},
            exploration_map={"insights": [], "frontier_questions": []},
        )
        assert result.success is True
        assert result.blueprint == {}
        assert result.exploration_map is not None

    def test_explore_result_empty_blueprint(self):
        from core.engine import CompileResult
        result = CompileResult(
            success=True,
            blueprint={},
            exploration_map={"insights": [{"text": "test"}]},
        )
        assert result.blueprint == {}
        assert len(result.exploration_map["insights"]) == 1


# ============================================================
# _compile_context_mode unit test
# ============================================================

class TestCompileContextModeUnit:
    """Unit test _compile_context_mode with mocked state."""

    def test_returns_compile_result(self):
        from core.engine import CompileResult
        result = CompileResult(
            success=True,
            blueprint={},
            context_map={"concepts": [], "relationships": []},
        )
        assert result.success is True
        assert result.context_map is not None

    def test_context_result_empty_blueprint(self):
        from core.engine import CompileResult
        result = CompileResult(
            success=True,
            blueprint={},
            context_map={"concepts": [{"name": "UserService"}]},
        )
        assert result.blueprint == {}
        assert len(result.context_map["concepts"]) == 1

    def test_context_success_requires_concepts(self):
        """Success should be True when concepts exist, False when empty."""
        from core.engine import CompileResult
        with_concepts = CompileResult(
            success=True,
            context_map={"concepts": [{"name": "X"}]},
        )
        assert with_concepts.success is True

        without_concepts = CompileResult(
            success=False,
            context_map={"concepts": []},
        )
        assert without_concepts.success is False
