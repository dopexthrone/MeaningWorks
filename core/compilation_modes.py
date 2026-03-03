"""
Compilation modes — controls what the compiler produces.

LEAF module. Stdlib only. No imports from core/ or mother/.

The semantic reduction engine is domain-agnostic. The "software" assumption
is injected at two points: agent prompt postures and synthesis output schema.
Changing the mode changes what the compiler produces while reusing the same
Phase 1-3 pipeline.

Four modes:
- BUILD   — current behavior, blueprint → code (default)
- CONTEXT — deep understanding, populates grid + memory permanently
- EXPLORE — divergent exploration, surfaces what the user hasn't considered
- SELF    — compiler audits itself, produces improvement goals
"""

import enum
from dataclasses import dataclass
from typing import Tuple


class CompilationMode(enum.Enum):
    """What the compiler should produce."""
    BUILD = "build"
    CONTEXT = "context"
    EXPLORE = "explore"
    SELF = "self"


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode behavioral configuration.

    Controls how the pipeline behaves at each phase without changing
    the pipeline structure. Phases 1-3 are identical across modes —
    only posture preambles differ. Phase 4+ diverges by mode.
    """
    # Agent behavior directive — prepended to system prompts
    agent_posture: str

    # Dialogue parameters
    dialogue_max_turns: int
    convergence_threshold: float

    # Phase 4-5 switches
    skip_synthesis: bool
    skip_verification: bool

    # Verification config
    verification_label: str
    actionability_checks: Tuple[str, ...]

    # Grid domain hint for kernel bootstrap
    grid_domain_hint: str

    # Post-compile persistence
    persist_to_world_grid: bool
    persist_to_memory: bool

    # Posture preamble — prepended to every agent system prompt
    posture_preamble: str


# --- Mode configurations ---

_BUILD_CONFIG = ModeConfig(
    agent_posture="converge",
    dialogue_max_turns=64,
    convergence_threshold=0.80,
    skip_synthesis=False,
    skip_verification=False,
    verification_label="codegen_readiness",
    actionability_checks=("methods",),
    grid_domain_hint="SFT",
    persist_to_world_grid=False,
    persist_to_memory=False,
    posture_preamble="",  # BUILD = default behavior, no preamble
)

_CONTEXT_CONFIG = ModeConfig(
    agent_posture="excavate",
    dialogue_max_turns=32,
    convergence_threshold=0.70,
    skip_synthesis=False,
    skip_verification=False,
    verification_label="analytical_completeness",
    actionability_checks=("description", "relationships"),
    grid_domain_hint="ORG",
    persist_to_world_grid=True,
    persist_to_memory=True,
    posture_preamble=(
        "MODE: CONTEXT UNDERSTANDING\n"
        "Your goal is deep comprehension, not implementation planning. "
        "Extract concepts, relationships, assumptions, vocabulary, and unknowns. "
        "Prioritize surfacing what is implicit or assumed over what is explicit. "
        "Do NOT propose components, methods, or architecture — map the semantic territory.\n"
    ),
)

_EXPLORE_CONFIG = ModeConfig(
    agent_posture="diverge",
    dialogue_max_turns=48,
    convergence_threshold=0.50,
    skip_synthesis=True,
    skip_verification=True,
    verification_label="exploration_breadth",
    actionability_checks=(),
    grid_domain_hint="ORG",
    persist_to_world_grid=False,
    persist_to_memory=False,
    posture_preamble=(
        "MODE: DIVERGENT EXPLORATION\n"
        "Your goal is to surface what the user hasn't considered. "
        "Challenge assumptions, propose alternative framings, identify adjacent domains, "
        "and ask frontier questions. Breadth over depth. "
        "Do NOT converge toward a solution — expand the problem space.\n"
    ),
)

_SELF_CONFIG = ModeConfig(
    agent_posture="audit",
    dialogue_max_turns=16,
    convergence_threshold=0.90,
    skip_synthesis=False,
    skip_verification=False,
    verification_label="audit_completeness",
    actionability_checks=("description",),
    grid_domain_hint="SFT",
    persist_to_world_grid=True,
    persist_to_memory=False,
    posture_preamble=(
        "MODE: SELF-AUDIT\n"
        "You are auditing the compiler itself. Identify weaknesses, "
        "missing capabilities, recurring failure patterns, and improvement goals. "
        "Be ruthlessly honest about limitations.\n"
    ),
)

_MODE_CONFIGS = {
    CompilationMode.BUILD: _BUILD_CONFIG,
    CompilationMode.CONTEXT: _CONTEXT_CONFIG,
    CompilationMode.EXPLORE: _EXPLORE_CONFIG,
    CompilationMode.SELF: _SELF_CONFIG,
}


def mode_config(mode: CompilationMode) -> ModeConfig:
    """Get the configuration for a compilation mode.

    Args:
        mode: The compilation mode.

    Returns:
        Frozen ModeConfig for the given mode.

    Raises:
        ValueError: If mode is not a valid CompilationMode.
    """
    if not isinstance(mode, CompilationMode):
        raise ValueError(f"Expected CompilationMode, got {type(mode).__name__}: {mode!r}")
    return _MODE_CONFIGS[mode]


def parse_mode(value: str) -> CompilationMode:
    """Parse a string into a CompilationMode.

    Accepts: "build", "context", "explore", "self" (case-insensitive).

    Args:
        value: String mode name.

    Returns:
        CompilationMode enum member.

    Raises:
        ValueError: If value is not a recognized mode name.
    """
    normalized = value.strip().lower()
    try:
        return CompilationMode(normalized)
    except ValueError:
        valid = ", ".join(m.value for m in CompilationMode)
        raise ValueError(f"Unknown compilation mode: {value!r}. Valid modes: {valid}")
