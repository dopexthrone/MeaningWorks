"""
Pipeline panel — 7-stage vertical visualization during compilation.

Each node shows: icon + name + timing + insight detail.
States: pending (dim), active (cyan + spinner), complete (green), error (red).
"""

import time
from typing import Optional, List
from dataclasses import dataclass, field

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Static


# The 7 pipeline stages matching the engine
PIPELINE_STAGES = [
    "Intent",
    "Persona",
    "Entity",
    "Process",
    "Synthesis",
    "Verify",
    "Governor",
]

# Status icons
ICONS = {
    "pending": "  o",
    "active": "  >",
    "complete": "  v",
    "error": "  x",
}


@dataclass
class StageState:
    """State of a single pipeline stage."""

    name: str
    status: str = "pending"  # pending, active, complete, error
    detail: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.monotonic() - self.start_time
        return None

    @property
    def elapsed_str(self) -> str:
        e = self.elapsed
        if e is None:
            return ""
        return f"{e:.1f}s"

    @property
    def display_line(self) -> str:
        icon = ICONS.get(self.status, "  ?")
        timing = f"  {self.elapsed_str}" if self.elapsed_str else ""
        return f"{icon} {self.name}{timing}"

    @property
    def detail_line(self) -> str:
        if self.detail:
            return f"    {self.detail}"
        return ""


class PipelineNode(Static):
    """Individual pipeline stage display."""

    def __init__(self, stage: StageState, **kwargs):
        self._stage = stage
        super().__init__(**kwargs)
        self._refresh_display()

    def _refresh_display(self) -> None:
        lines = self._stage.display_line
        if self._stage.detail_line:
            lines += f"\n{self._stage.detail_line}"
        self.update(escape_markup(lines))

    def set_state(self, stage: StageState) -> None:
        self._stage = stage
        css_class = f"pipeline-node node-{stage.status}"
        self.set_classes(css_class)
        self._refresh_display()


class PipelinePanel(Container):
    """7-stage vertical pipeline display."""

    def __init__(self, **kwargs):
        super().__init__(id="pipeline-panel", **kwargs)
        self._stages: List[StageState] = [StageState(name=s) for s in PIPELINE_STAGES]
        self._pipeline_nodes: List[PipelineNode] = []

    def compose(self) -> ComposeResult:
        yield Static("Compilation Pipeline", classes="pipeline-header")
        for stage in self._stages:
            node = PipelineNode(stage, classes=f"pipeline-node node-{stage.status}")
            self._pipeline_nodes.append(node)
            yield node

    def update_stage(self, name: str, status: str, detail: str = "") -> None:
        """Update a named stage's status and detail."""
        for i, stage in enumerate(self._stages):
            if stage.name.lower() == name.lower():
                if status == "active" and stage.start_time is None:
                    stage.start_time = time.monotonic()
                elif status in ("complete", "error"):
                    stage.end_time = time.monotonic()
                stage.status = status
                stage.detail = detail
                if i < len(self._pipeline_nodes):
                    self._pipeline_nodes[i].set_state(stage)
                break

    def reset(self) -> None:
        """Reset all stages to pending."""
        self._stages = [StageState(name=s) for s in PIPELINE_STAGES]
        for i, node in enumerate(self._pipeline_nodes):
            if i < len(self._stages):
                node.set_state(self._stages[i])

    def show(self) -> None:
        self.add_class("visible")

    def hide(self) -> None:
        self.remove_class("visible")

    @property
    def stages(self) -> List[StageState]:
        return self._stages
