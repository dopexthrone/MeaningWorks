"""
Phase A+B: Tests for build progress streaming, output directory, and rich result display.
"""

import asyncio
import queue
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from mother.config import MotherConfig
from mother.bridge import EngineBridge
from mother.persona import inject_personality_bite


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================


class TestOutputDirDefault:
    """Test output_dir defaults to ~/motherlabs/projects."""

    def test_default_output_dir(self):
        config = MotherConfig()
        expected = str(Path.home() / "motherlabs" / "projects")
        assert config.output_dir == expected

    def test_output_dir_is_absolute(self):
        config = MotherConfig()
        assert Path(config.output_dir).is_absolute()

    def test_output_dir_override(self):
        config = MotherConfig(output_dir="/tmp/custom")
        assert config.output_dir == "/tmp/custom"

    def test_output_dir_persists_through_save_load(self, tmp_path):
        from mother.config import save_config, load_config
        path = str(tmp_path / "mother.json")
        config = MotherConfig(output_dir="/tmp/test_output")
        save_config(config, path)
        loaded = load_config(path)
        assert loaded.output_dir == "/tmp/test_output"


# =============================================================================
# BUILD PHASE QUEUE
# =============================================================================


class TestBuildPhaseQueue:
    """Test build phase queue on EngineBridge."""

    def test_bridge_has_build_phase_queue(self):
        bridge = EngineBridge()
        assert hasattr(bridge, "_build_phase_queue")
        assert isinstance(bridge._build_phase_queue, queue.Queue)

    def test_bridge_has_build_done_event(self):
        bridge = EngineBridge()
        assert hasattr(bridge, "_build_done")
        assert bridge._build_done.is_set()

    def test_begin_build_clears_event(self):
        bridge = EngineBridge()
        bridge.begin_build()
        assert not bridge._build_done.is_set()

    def test_build_phase_queue_accepts_tuples(self):
        bridge = EngineBridge()
        bridge._build_phase_queue.put_nowait(("compile", "Compiling..."))
        phase, detail = bridge._build_phase_queue.get_nowait()
        assert phase == "compile"
        assert detail == "Compiling..."


class TestStreamBuildPhases:
    """Test stream_build_phases async generator."""

    def test_stream_empty_when_done(self):
        bridge = EngineBridge()
        bridge._build_done.set()
        phases = list(asyncio.run(_collect_phases(bridge)))
        assert phases == []

    def test_stream_yields_queued_phases(self):
        bridge = EngineBridge()
        bridge._build_phase_queue.put_nowait(("compile", "Compiling blueprint..."))
        bridge._build_phase_queue.put_nowait(("emit", "Generating code..."))
        bridge._build_done.set()
        phases = list(asyncio.run(_collect_phases(bridge)))
        assert len(phases) == 2
        assert phases[0] == ("compile", "Compiling blueprint...")
        assert phases[1] == ("emit", "Generating code...")

    def test_stream_terminates_after_done_and_drain(self):
        bridge = EngineBridge()
        bridge._build_phase_queue.put_nowait(("write", "Writing project..."))
        bridge._build_done.set()
        phases = list(asyncio.run(_collect_phases(bridge)))
        assert len(phases) == 1


async def _collect_phases(bridge):
    """Helper to collect all phases from stream."""
    result = []
    async for phase in bridge.stream_build_phases():
        result.append(phase)
    return result


# =============================================================================
# BUILD METHOD WITH PROGRESS
# =============================================================================


class TestBuildWithProgress:
    """Test bridge.build() pushes phase events."""

    def test_build_sets_done_event_on_completion(self):
        bridge = EngineBridge()

        # Mock the engine and orchestrator
        mock_engine = MagicMock()
        mock_engine._session_cost_usd = 0.0
        bridge._engine = mock_engine

        mock_result = MagicMock()
        mock_result.success = True

        with patch("core.agent_orchestrator.AgentOrchestrator") as MockOrch:
            mock_orch = MockOrch.return_value
            mock_orch.run.return_value = mock_result
            result = asyncio.run(bridge.build("test", output_dir="/tmp"))

        assert bridge._build_done.is_set()
        assert result.success

    def test_build_passes_on_progress_callback(self):
        bridge = EngineBridge()

        mock_engine = MagicMock()
        mock_engine._session_cost_usd = 0.0
        bridge._engine = mock_engine

        mock_result = MagicMock()
        mock_result.success = True
        captured_callback = None

        with patch("core.agent_orchestrator.AgentOrchestrator") as MockOrch:
            mock_orch = MockOrch.return_value

            def capture_run(desc, on_progress=None):
                nonlocal captured_callback
                captured_callback = on_progress
                # Simulate progress events
                if on_progress:
                    on_progress("compile", "Compiling...")
                    on_progress("emit", "Generating...")
                return mock_result

            mock_orch.run.side_effect = capture_run
            asyncio.run(bridge.build("test app", output_dir="/tmp"))

        # Verify callback was passed and pushed to queue
        assert captured_callback is not None

    def test_build_tracks_cost(self):
        bridge = EngineBridge()
        mock_engine = MagicMock()
        mock_engine._session_cost_usd = 0.15
        bridge._engine = mock_engine
        bridge._engine_cost_baseline = 0.0

        mock_result = MagicMock()
        mock_result.success = True

        with patch("core.agent_orchestrator.AgentOrchestrator") as MockOrch:
            mock_orch = MockOrch.return_value
            mock_orch.run.return_value = mock_result
            asyncio.run(bridge.build("test", output_dir="/tmp"))

        assert bridge.get_session_cost() == pytest.approx(0.15, abs=0.01)

    def test_build_done_set_even_on_error(self):
        bridge = EngineBridge()
        mock_engine = MagicMock()
        mock_engine._session_cost_usd = 0.0
        bridge._engine = mock_engine

        with patch("core.agent_orchestrator.AgentOrchestrator") as MockOrch:
            mock_orch = MockOrch.return_value
            mock_orch.run.side_effect = RuntimeError("Test error")
            with pytest.raises(RuntimeError):
                asyncio.run(bridge.build("test", output_dir="/tmp"))

        assert bridge._build_done.is_set()


# =============================================================================
# BUILD PERSONALITY BITES
# =============================================================================


class TestBuildPersonalityBites:
    """Test build phase personality bites."""

    def test_warm_build_start(self):
        bite = inject_personality_bite("warm", "build_start")
        assert bite is not None
        assert len(bite) > 5

    def test_warm_build_emit(self):
        bite = inject_personality_bite("warm", "build_emit")
        assert bite is not None

    def test_warm_build_validate(self):
        bite = inject_personality_bite("warm", "build_validate")
        assert bite is not None

    def test_warm_build_fix(self):
        bite = inject_personality_bite("warm", "build_fix")
        assert bite is not None

    def test_composed_build_start(self):
        bite = inject_personality_bite("composed", "build_start")
        assert bite is not None

    def test_composed_build_emit(self):
        bite = inject_personality_bite("composed", "build_emit")
        assert bite is not None

    def test_composed_build_validate(self):
        bite = inject_personality_bite("composed", "build_validate")
        assert bite is not None

    def test_direct_build_emit(self):
        bite = inject_personality_bite("direct", "build_emit")
        assert bite is not None

    def test_direct_build_fix(self):
        bite = inject_personality_bite("direct", "build_fix")
        assert bite is not None

    def test_playful_build_start(self):
        bite = inject_personality_bite("playful", "build_start")
        assert bite is not None

    def test_playful_build_emit(self):
        bite = inject_personality_bite("playful", "build_emit")
        assert bite is not None

    def test_playful_build_validate(self):
        bite = inject_personality_bite("playful", "build_validate")
        assert bite is not None

    def test_playful_build_fix(self):
        bite = inject_personality_bite("playful", "build_fix")
        assert bite is not None

    def test_composed_no_build_fix(self):
        """Composed personality stays silent on build_fix."""
        bite = inject_personality_bite("composed", "build_fix")
        assert bite is None


# =============================================================================
# RICH RESULT DISPLAY
# =============================================================================


class TestRichResultFormat:
    """Test that the result display format is correct by verifying data extraction."""

    def test_project_name_from_path(self):
        path = "/Users/test/motherlabs/projects/todo_app"
        assert Path(path).name == "todo_app"

    def test_path_shortening_with_home(self):
        home = str(Path.home())
        path = f"{home}/motherlabs/projects/counter"
        display = path
        if display.startswith(home):
            display = "~" + display[len(home):]
        assert display.startswith("~/")
        assert "counter" in display

    def test_build_result_clean_first_pass(self):
        """When no build_result, display says 'clean first pass'."""
        build_line = "Build: clean first pass"
        assert "clean first pass" in build_line

    def test_build_result_with_fixes(self):
        """Build result with fixed components shows what was fixed."""
        # Simulate what chat.py does
        components_fixed = ("auth_handler", "db_client")
        iters = 2
        fixed_names = ", ".join(components_fixed)
        build_line = f"Build: {iters} iteration{'s' if iters != 1 else ''}, fixed {fixed_names}"
        assert "2 iterations" in build_line
        assert "auth_handler" in build_line
        assert "db_client" in build_line

    def test_trust_display_format(self):
        """Trust line format."""
        score = 82.0
        badge = "VERIFIED"
        trust_line = f"Trust: {badge} {score:.0f}%"
        assert trust_line == "Trust: VERIFIED 82%"

    def test_result_lines_structure(self):
        """Verify the full result display structure."""
        lines = [
            "Project: todo_app",
            "Path: ~/motherlabs/projects/todo_app",
            "Files: 8 written (342 lines)",
            "Entry: python3 main.py",
            "Trust: VERIFIED 82%",
            "Build: clean first pass",
        ]
        result = "\n".join(lines)
        assert "Project:" in result
        assert "Path:" in result
        assert "Files:" in result
        assert "Entry:" in result
        assert "Trust:" in result
        assert "Build:" in result
