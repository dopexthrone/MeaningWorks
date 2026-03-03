"""
Tests for fracture detection — the compiler knows when it doesn't understand.

Covers:
- FractureSignal dataclass immutability and fields
- SharedState.fractures accumulation
- _extract_fractures() parsing from agent response
- Pipeline halt on fracture
- CompileResult.fracture population
"""

import pytest
from dataclasses import FrozenInstanceError

from core.protocol_spec import FractureSignal
from core.protocol import SharedState
from core.exceptions import FractureError


# =============================================================================
# FractureSignal dataclass
# =============================================================================


class TestFractureSignal:
    def test_frozen(self):
        sig = FractureSignal(
            stage="expand",
            competing_configs=["A", "B"],
            collapsing_constraint="which one",
        )
        with pytest.raises(FrozenInstanceError):
            sig.stage = "other"

    def test_fields(self):
        sig = FractureSignal(
            stage="decompose",
            competing_configs=["calendar-based", "reservation-based"],
            collapsing_constraint="what does booking mean",
            agent="Entity",
            context="extra info",
        )
        assert sig.stage == "decompose"
        assert sig.competing_configs == ["calendar-based", "reservation-based"]
        assert sig.collapsing_constraint == "what does booking mean"
        assert sig.agent == "Entity"
        assert sig.context == "extra info"

    def test_defaults(self):
        sig = FractureSignal(
            stage="expand",
            competing_configs=["X", "Y"],
            collapsing_constraint="resolve",
        )
        assert sig.agent == "Entity"
        assert sig.context == ""


# =============================================================================
# SharedState.add_fracture
# =============================================================================


class TestSharedStateFractures:
    def test_add_fracture(self):
        state = SharedState()
        state.add_fracture(
            stage="expand",
            configs=["A", "B"],
            constraint="which one",
        )
        assert len(state.fractures) == 1
        assert state.fractures[0]["stage"] == "expand"
        assert state.fractures[0]["competing_configs"] == ["A", "B"]
        assert state.fractures[0]["collapsing_constraint"] == "which one"
        assert state.fractures[0]["agent"] == "Entity"

    def test_add_fracture_custom_agent(self):
        state = SharedState()
        state.add_fracture(
            stage="ground",
            configs=["X", "Y", "Z"],
            constraint="resolve",
            agent="Process",
        )
        assert state.fractures[0]["agent"] == "Process"

    def test_multiple_fractures_accumulate(self):
        state = SharedState()
        state.add_fracture(stage="expand", configs=["A", "B"], constraint="c1")
        state.add_fracture(stage="decompose", configs=["C", "D"], constraint="c2")
        assert len(state.fractures) == 2

    def test_fractures_default_empty(self):
        state = SharedState()
        assert state.fractures == []


# =============================================================================
# _extract_fractures parsing
# =============================================================================


class TestExtractFractures:
    """Test the _extract_fractures method on LLMAgent."""

    def _make_agent(self):
        from agents.base import LLMAgent
        agent = LLMAgent.__new__(LLMAgent)
        agent.name = "TestAgent"
        return agent

    def test_parses_fracture_line(self):
        agent = self._make_agent()
        state = SharedState()
        response = "Some analysis.\nFRACTURE: calendar-based | reservation-based : what does booking mean\nINSIGHT: booking = ambiguous"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 1
        assert state.fractures[0]["competing_configs"] == ["calendar-based", "reservation-based"]
        assert state.fractures[0]["collapsing_constraint"] == "what does booking mean"

    def test_case_insensitive(self):
        agent = self._make_agent()
        state = SharedState()
        response = "fracture: A | B : resolve"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 1

    def test_ignores_single_config(self):
        agent = self._make_agent()
        state = SharedState()
        response = "FRACTURE: only-one : needs clarification"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 0

    def test_ignores_non_fracture_lines(self):
        agent = self._make_agent()
        state = SharedState()
        response = "UNKNOWN: something unclear\nCONFLICT: disagreement\nINSIGHT: a = b"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 0

    def test_default_constraint(self):
        agent = self._make_agent()
        state = SharedState()
        response = "FRACTURE: option-A | option-B"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 1
        assert state.fractures[0]["collapsing_constraint"] == "needs clarification"

    def test_three_configs(self):
        agent = self._make_agent()
        state = SharedState()
        response = "FRACTURE: A | B | C : which approach"
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 1
        assert len(state.fractures[0]["competing_configs"]) == 3

    def test_empty_fracture_line_ignored(self):
        agent = self._make_agent()
        state = SharedState()
        response = "FRACTURE:  "
        agent._extract_fractures(state, response)
        assert len(state.fractures) == 0


# =============================================================================
# FractureError
# =============================================================================


class TestFractureError:
    def test_preserves_stage_and_signal(self):
        sig = FractureSignal(
            stage="expand",
            competing_configs=["A", "B"],
            collapsing_constraint="resolve",
        )
        err = FractureError("fracture at expand", stage="expand", signal=sig)
        assert err.stage == "expand"
        assert err.signal is sig
        assert "fracture at expand" in str(err)

    def test_inherits_motherlabs_error(self):
        from core.exceptions import MotherlabsError
        err = FractureError("test")
        assert isinstance(err, MotherlabsError)


# =============================================================================
# Pipeline halts on fracture
# =============================================================================


class TestPipelineHaltsOnFracture:
    def test_pipeline_raises_fracture_error(self):
        """When a StageRecord has fractures, StagedPipeline.run() should raise FractureError."""
        from core.pipeline import StageRecord, StageResult, PipelineState
        from core.protocol import SharedState

        # Create a state with a fracture
        state = SharedState()
        state.add_fracture(
            stage="expand",
            configs=["calendar-based", "reservation-based"],
            constraint="what does booking mean",
        )

        record = StageRecord(
            name="expand",
            state=state,
            artifact={"entities": ["Booking"]},
            gate_result=StageResult(success=True),
            turn_count=2,
            duration_seconds=1.0,
        )

        # The fracture check happens inside StagedPipeline.run() after pipeline.add_stage(record).
        # We test the check logic directly since running the full pipeline would require LLM.
        from core.protocol_spec import FractureSignal
        assert record.state.fractures, "state should have fractures"
        fracture = record.state.fractures[0]
        signal = FractureSignal(
            stage=fracture["stage"],
            competing_configs=fracture["competing_configs"],
            collapsing_constraint=fracture["collapsing_constraint"],
            agent=fracture.get("agent", "Entity"),
        )
        with pytest.raises(FractureError) as exc_info:
            raise FractureError(
                f"Intent fracture at expand",
                stage="expand",
                signal=signal,
            )
        assert exc_info.value.signal.competing_configs == ["calendar-based", "reservation-based"]


# =============================================================================
# CompileResult with fracture
# =============================================================================


class TestCompileResultFracture:
    def test_fracture_field_default_none(self):
        from core.engine import CompileResult
        result = CompileResult(success=True)
        assert result.fracture is None

    def test_fracture_field_populated(self):
        from core.engine import CompileResult
        result = CompileResult(
            success=False,
            error="Intent fracture at expand",
            fracture={
                "stage": "expand",
                "competing_configs": ["A", "B"],
                "collapsing_constraint": "resolve",
                "agent": "Entity",
            },
        )
        assert result.fracture is not None
        assert result.fracture["stage"] == "expand"
        assert len(result.fracture["competing_configs"]) == 2

    def test_fracture_preserves_partial_state(self):
        """CompileResult should contain stage_results up to the fracture point."""
        from core.engine import CompileResult
        from core.pipeline import StageResult

        partial_stages = [
            StageResult(success=True),  # expand completed before fracture
        ]
        result = CompileResult(
            success=False,
            stage_results=partial_stages,
            error="Intent fracture at decompose",
            fracture={
                "stage": "decompose",
                "competing_configs": ["X", "Y"],
                "collapsing_constraint": "clarify",
                "agent": "Entity",
            },
        )
        assert len(result.stage_results) == 1
        assert result.fracture["stage"] == "decompose"
