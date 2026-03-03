"""
kernel/dispatch.py — Cell → action type dispatch.

LEAF module. Imports only kernel/cell.py for parse_postcode.

Deterministic mapping from grid cell location (layer + concern) to
the type of action Mother should take. The navigator picks the cell;
this module says what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel.cell import parse_postcode

__all__ = [
    "ActionDispatch",
    "ACTION_MAP",
    "dispatch_from_cell",
    "action_requires_llm",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionDispatch:
    """A dispatched action from a world grid cell."""
    action_type: str    # "perceive"|"compile"|"reflect"|"observe_user"|"execute_task"|...
    postcode: str       # cell that triggered this
    priority: float     # from navigator score
    context: str        # cell.primitive for action context
    requires_llm: bool  # cost gate


# ---------------------------------------------------------------------------
# Action map: (layer, concern) → action_type
# ---------------------------------------------------------------------------

ACTION_MAP: dict[tuple[str, str], str] = {
    ("OBS", "ENV"): "perceive",
    ("OBS", "USR"): "observe_user",
    ("INT", "PRJ"): "compile",
    ("INT", "TSK"): "execute_task",
    ("INT", "SEM"): "compile",
    ("MET", "MEM"): "reflect",
    ("MET", "GOL"): "self_improve",
    ("NET", "STA"): "check_external",
    ("TME", "SCH"): "check_schedule",
    ("AGN", "TSK"): "manage_agent",
}

# Actions that require an LLM call (cost gate)
_LLM_ACTIONS: frozenset[str] = frozenset({
    "compile",
    "self_improve",
    "reflect",
    "execute_task",
})

# Default action for unmapped (layer, concern) pairs
_DEFAULT_ACTION = "observe"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_from_cell(
    postcode_key: str,
    score: float,
    primitive: str = "",
) -> ActionDispatch:
    """Map cell location to action type.

    Args:
        postcode_key: The 5-axis postcode of the cell.
        score: Navigator score for priority.
        primitive: Cell primitive for action context.

    Returns:
        ActionDispatch with the resolved action type.

    Raises:
        ValueError: If postcode_key is malformed.
    """
    pc = parse_postcode(postcode_key)
    action_type = ACTION_MAP.get((pc.layer, pc.concern), _DEFAULT_ACTION)

    return ActionDispatch(
        action_type=action_type,
        postcode=postcode_key,
        priority=score,
        context=primitive,
        requires_llm=action_type in _LLM_ACTIONS,
    )


def action_requires_llm(action_type: str) -> bool:
    """Cost gate: does this action need an LLM call?"""
    return action_type in _LLM_ACTIONS
