"""
Phase 4: Tests for pipeline visualization.
"""

import time
import pytest

from mother.widgets.pipeline import (
    PipelinePanel,
    PipelineNode,
    StageState,
    PIPELINE_STAGES,
    ICONS,
)


class TestStageState:
    """Test StageState dataclass."""

    def test_default_pending(self):
        stage = StageState(name="Intent")
        assert stage.status == "pending"
        assert stage.detail == ""

    def test_elapsed_none_when_not_started(self):
        stage = StageState(name="Intent")
        assert stage.elapsed is None

    def test_elapsed_computed_when_started(self):
        stage = StageState(name="Intent", start_time=time.monotonic() - 1.0)
        assert stage.elapsed >= 0.9

    def test_elapsed_fixed_when_complete(self):
        start = time.monotonic() - 2.0
        end = start + 1.5
        stage = StageState(name="Intent", start_time=start, end_time=end)
        assert abs(stage.elapsed - 1.5) < 0.01

    def test_display_line_pending(self):
        stage = StageState(name="Intent")
        line = stage.display_line
        assert "o" in line
        assert "Intent" in line

    def test_display_line_complete(self):
        stage = StageState(name="Verify", status="complete", start_time=1.0, end_time=2.5)
        line = stage.display_line
        assert "v" in line
        assert "Verify" in line

    def test_detail_line_empty_when_no_detail(self):
        stage = StageState(name="Intent")
        assert stage.detail_line == ""

    def test_detail_line_shows_detail(self):
        stage = StageState(name="Intent", detail="Extracting core concepts")
        assert "Extracting" in stage.detail_line


class TestPipelineStages:
    """Test pipeline stage constants."""

    def test_seven_stages(self):
        assert len(PIPELINE_STAGES) == 7

    def test_starts_with_intent(self):
        assert PIPELINE_STAGES[0] == "Intent"

    def test_ends_with_governor(self):
        assert PIPELINE_STAGES[-1] == "Governor"

    def test_all_icons_defined(self):
        for status in ("pending", "active", "complete", "error"):
            assert status in ICONS


class TestPipelinePanel:
    """Test PipelinePanel widget."""

    def test_creates_panel(self):
        panel = PipelinePanel()
        assert panel.id == "pipeline-panel"

    def test_has_seven_stages(self):
        panel = PipelinePanel()
        assert len(panel.stages) == 7

    def test_all_stages_start_pending(self):
        panel = PipelinePanel()
        for stage in panel.stages:
            assert stage.status == "pending"

    def test_update_stage_changes_status(self):
        panel = PipelinePanel()
        panel.update_stage("Intent", "active", "Processing...")
        assert panel.stages[0].status == "active"
        assert panel.stages[0].detail == "Processing..."

    def test_update_stage_case_insensitive(self):
        panel = PipelinePanel()
        panel.update_stage("intent", "complete")
        assert panel.stages[0].status == "complete"

    def test_reset_clears_all(self):
        panel = PipelinePanel()
        panel.update_stage("Intent", "complete")
        panel.update_stage("Persona", "active")
        panel.reset()
        for stage in panel.stages:
            assert stage.status == "pending"

    def test_update_sets_start_time(self):
        panel = PipelinePanel()
        panel.update_stage("Intent", "active")
        assert panel.stages[0].start_time is not None

    def test_update_sets_end_time_on_complete(self):
        panel = PipelinePanel()
        panel.update_stage("Intent", "active")
        panel.update_stage("Intent", "complete")
        assert panel.stages[0].end_time is not None
