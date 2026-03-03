"""Tests for dense extraction — progressive scope scheduling in kernel compile.

Validates:
- scope_schedule overrides target_scopes per iteration
- progressive depth coverage across scope hierarchy
- more cells produced with dense config vs baseline
- depth stats appear in author prompt
- deep scope extraction count ("10-25")
- schedule clamps to last entry for excess iterations
- empty schedule falls back to target_scopes
- AX1 provenance enforced at all depth levels
"""
from __future__ import annotations

import pytest

from kernel.agents import (
    CompileConfig,
    compile,
    _build_author_prompt,
    _depth_coverage_block,
    _extraction_range,
    author,
)
from kernel.cell import SCOPES, parse_postcode
from kernel.grid import Grid, INTENT_CONTRACT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CALL_LOG: list[str] = []


def _make_mock_llm(scope_extractions: dict[str, list[dict]] | None = None):
    """Create a mock LLM that returns scope-appropriate extractions.

    If scope_extractions is provided, returns extractions whose postcodes
    match the target scopes requested in the prompt. Otherwise returns
    a default set at ECO/APP scope.
    """
    call_count = [0]

    def _mock(prompt: str) -> list[dict]:
        call_count[0] += 1
        _CALL_LOG.append(prompt)

        if scope_extractions:
            # Check which scopes are requested in the prompt
            for scope_key, extractions in scope_extractions.items():
                if f"SCOPE FOCUS" in prompt and scope_key in prompt:
                    return extractions
            # Fall through to default if no scope match
            return scope_extractions.get("default", _default_extractions())

        return _default_extractions()

    return _mock, call_count


def _default_extractions() -> list[dict]:
    return [
        {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "system", "content": "The overall system", "confidence": 0.9, "connections": []},
        {"postcode": "ORG.FNC.APP.HOW.SFT", "primitive": "pipeline", "content": "Processing pipeline", "confidence": 0.85, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
        {"postcode": "STR.REL.APP.WHAT.SFT", "primitive": "data-flow", "content": "Data flow between components", "confidence": 0.8, "connections": ["ORG.FNC.APP.HOW.SFT"]},
    ]


def _scope_aware_extractions() -> dict[str, list[dict]]:
    """Return extractions keyed by scope focus strings."""
    return {
        "ECO": [
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "ecosystem", "content": "Ecosystem-level concept", "confidence": 0.9, "connections": []},
            {"postcode": "ORG.FNC.APP.HOW.SFT", "primitive": "app-pipeline", "content": "Application pipeline", "confidence": 0.85, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
        ],
        "DOM": [
            {"postcode": "STR.ENT.DOM.WHAT.SFT", "primitive": "domain-model", "content": "Domain model", "confidence": 0.88, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
            {"postcode": "COG.BHV.FET.HOW.SFT", "primitive": "feature-logic", "content": "Feature behavior", "confidence": 0.82, "connections": ["STR.ENT.DOM.WHAT.SFT"]},
        ],
        "CMP": [
            {"postcode": "STR.FNC.CMP.HOW.SFT", "primitive": "component-fn", "content": "Component function", "confidence": 0.85, "connections": ["STR.ENT.DOM.WHAT.SFT"]},
            {"postcode": "AGN.ACT.FNC.WHO.SFT", "primitive": "agent-actor", "content": "Agent that acts", "confidence": 0.80, "connections": ["STR.FNC.CMP.HOW.SFT"]},
            {"postcode": "CTR.GTE.CMP.IF.SFT", "primitive": "gate-check", "content": "Gate control", "confidence": 0.78, "connections": ["STR.FNC.CMP.HOW.SFT"]},
        ],
        "STP": [
            {"postcode": "STR.FNC.STP.HOW.SFT", "primitive": "step-fn", "content": "Step function", "confidence": 0.82, "connections": ["STR.FNC.CMP.HOW.SFT"]},
            {"postcode": "CTR.FLW.OPR.HOW.SFT", "primitive": "op-flow", "content": "Operation flow", "confidence": 0.79, "connections": ["STR.FNC.STP.HOW.SFT"]},
        ],
        "default": _default_extractions(),
    }


def _counting_mock_llm():
    """Mock LLM that returns distinct extractions per call, accumulating cells."""
    call_idx = [0]
    batches = [
        # Call 0: ECO/APP
        [
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "sys-a", "content": "System A", "confidence": 0.9, "connections": []},
            {"postcode": "ORG.FNC.APP.HOW.SFT", "primitive": "pipe-a", "content": "Pipeline A", "confidence": 0.85, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
        ],
        # Call 1: DOM/FET
        [
            {"postcode": "STR.ENT.DOM.WHAT.SFT", "primitive": "dom-b", "content": "Domain B", "confidence": 0.88, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
            {"postcode": "COG.BHV.FET.HOW.SFT", "primitive": "feat-b", "content": "Feature B", "confidence": 0.82, "connections": ["STR.ENT.DOM.WHAT.SFT"]},
        ],
        # Call 2: CMP/FNC
        [
            {"postcode": "STR.FNC.CMP.HOW.SFT", "primitive": "comp-c", "content": "Component C", "confidence": 0.85, "connections": ["STR.ENT.DOM.WHAT.SFT"]},
            {"postcode": "AGN.ACT.FNC.WHO.SFT", "primitive": "agent-c", "content": "Agent C", "confidence": 0.80, "connections": ["STR.FNC.CMP.HOW.SFT"]},
            {"postcode": "CTR.GTE.CMP.IF.SFT", "primitive": "gate-c", "content": "Gate C", "confidence": 0.78, "connections": ["STR.FNC.CMP.HOW.SFT"]},
        ],
        # Call 3: CMP/FNC/STP
        [
            {"postcode": "TME.SCH.STP.WHEN.SFT", "primitive": "sched-d", "content": "Schedule D", "confidence": 0.80, "connections": ["STR.FNC.CMP.HOW.SFT"]},
            {"postcode": "RES.LMT.FNC.HOW_MUCH.SFT", "primitive": "limit-d", "content": "Limit D", "confidence": 0.77, "connections": ["AGN.ACT.FNC.WHO.SFT"]},
        ],
        # Call 4: STP/OPR
        [
            {"postcode": "STR.FNC.STP.HOW.SFT", "primitive": "step-e", "content": "Step E", "confidence": 0.82, "connections": ["STR.FNC.CMP.HOW.SFT"]},
            {"postcode": "CTR.FLW.OPR.HOW.SFT", "primitive": "op-e", "content": "Operation E", "confidence": 0.79, "connections": ["STR.FNC.STP.HOW.SFT"]},
        ],
    ]

    def _mock(prompt: str) -> list[dict]:
        idx = min(call_idx[0], len(batches) - 1)
        call_idx[0] += 1
        return batches[idx]

    return _mock, call_idx


INPUT_TEXT = "Build a task management system with boards, lists, cards, drag-and-drop, and real-time collaboration."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScopeSchedule:
    """scope_schedule field and compile loop wiring."""

    def test_scope_schedule_overrides_target_scopes(self):
        """When scope_schedule is set, each iteration uses schedule[i] not target_scopes."""
        prompts_seen: list[str] = []

        def _capture_mock(prompt: str) -> list[dict]:
            prompts_seen.append(prompt)
            return _default_extractions()

        config = CompileConfig(
            max_iterations=3,
            min_fill_rate=0.99,  # Prevent early exit so all 3 iterations run
            target_scopes=("ECO",),  # Should be overridden
            scope_schedule=(
                ("DOM", "FET"),
                ("CMP", "FNC"),
                ("STP", "OPR"),
            ),
        )
        result = compile(INPUT_TEXT, _capture_mock, config=config)

        assert result.author_calls == 3
        # Iteration 0 should mention DOM/FET, not ECO
        assert "DOM" in prompts_seen[0]
        assert "FET" in prompts_seen[0]
        # Iteration 1 should mention CMP/FNC
        assert "CMP" in prompts_seen[1]
        assert "FNC" in prompts_seen[1]
        # Iteration 2 should mention STP/OPR
        assert "STP" in prompts_seen[2]
        assert "OPR" in prompts_seen[2]

    def test_schedule_clamps_to_last(self):
        """If iterations > len(schedule), last schedule entry is reused."""
        prompts_seen: list[str] = []

        def _capture_mock(prompt: str) -> list[dict]:
            prompts_seen.append(prompt)
            return _default_extractions()

        config = CompileConfig(
            max_iterations=4,
            min_fill_rate=0.99,  # Prevent early exit
            scope_schedule=(
                ("ECO", "APP"),
                ("CMP", "FNC"),
            ),
        )
        result = compile(INPUT_TEXT, _capture_mock, config=config)

        # Iterations 2 and 3 should reuse schedule[-1] = ("CMP", "FNC")
        assert result.author_calls >= 3
        for p in prompts_seen[2:]:
            assert "CMP" in p
            assert "FNC" in p

    def test_empty_schedule_falls_back(self):
        """Empty scope_schedule uses target_scopes (backward compatible)."""
        prompts_seen: list[str] = []

        def _capture_mock(prompt: str) -> list[dict]:
            prompts_seen.append(prompt)
            return _default_extractions()

        config = CompileConfig(
            max_iterations=2,
            min_fill_rate=0.99,  # Prevent early exit
            target_scopes=("DOM", "FET"),
            scope_schedule=(),  # Empty — should fall back
        )
        result = compile(INPUT_TEXT, _capture_mock, config=config)

        # All prompts should mention DOM/FET from target_scopes
        assert len(prompts_seen) == 2
        for p in prompts_seen:
            assert "DOM" in p


class TestProgressiveDepth:
    """Progressive depth coverage across scope hierarchy."""

    def test_progressive_depth_coverage(self):
        """Compile with progressive schedule produces cells at deep scopes."""
        mock_fn, call_count = _counting_mock_llm()

        config = CompileConfig(
            max_iterations=5,
            min_fill_rate=0.99,  # Force all 5 iterations
            scope_schedule=(
                ("ECO", "APP"),
                ("DOM", "FET"),
                ("CMP", "FNC"),
                ("CMP", "FNC", "STP"),
                ("STP", "OPR"),
            ),
        )
        result = compile(INPUT_TEXT, mock_fn, config=config)

        # Should have cells at deep scopes (CMP, FNC, STP, OPR)
        scopes_present = {c.postcode.scope for c in result.grid.filled_cells()}
        assert "CMP" in scopes_present, f"Missing CMP scope. Present: {scopes_present}"
        assert "STP" in scopes_present, f"Missing STP scope. Present: {scopes_present}"
        assert "OPR" in scopes_present, f"Missing OPR scope. Present: {scopes_present}"

    def test_more_cells_with_dense_config(self):
        """Dense config (5 iter + schedule) produces more cells than baseline (3 iter)."""
        mock_fn_base, _ = _counting_mock_llm()
        mock_fn_dense, _ = _counting_mock_llm()

        baseline_config = CompileConfig(max_iterations=3, min_fill_rate=0.99)
        dense_config = CompileConfig(
            max_iterations=5,
            min_fill_rate=0.99,  # Force all 5 iterations
            scope_schedule=(
                ("ECO", "APP"),
                ("DOM", "FET"),
                ("CMP", "FNC"),
                ("CMP", "FNC", "STP"),
                ("STP", "OPR"),
            ),
        )

        baseline = compile(INPUT_TEXT, mock_fn_base, config=baseline_config)
        dense = compile(INPUT_TEXT, mock_fn_dense, config=dense_config)

        baseline_count = len(baseline.grid.filled_cells())
        dense_count = len(dense.grid.filled_cells())

        assert dense_count > baseline_count, (
            f"Dense ({dense_count}) should produce more cells than baseline ({baseline_count})"
        )


class TestPromptEnrichment:
    """Depth stats and dynamic extraction count in author prompt."""

    def test_depth_stats_in_prompt(self):
        """_build_author_prompt includes DEPTH COVERAGE section."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")

        # Add some cells at different scopes
        from kernel.ops import fill
        fill(grid, "SEM.ENT.ECO.WHAT.SFT", "sys", "system", 0.9, source=("intent_contract",))
        fill(grid, "STR.FNC.CMP.HOW.SFT", "comp", "component", 0.85, source=("SEM.ENT.ECO.WHAT.SFT",))

        prompt = _build_author_prompt(grid, "test input")
        assert "DEPTH COVERAGE" in prompt
        assert "ECO:" in prompt
        assert "CMP:" in prompt
        # Scopes with 0 cells should show 0
        assert "FNC: 0 cells" in prompt

    def test_depth_coverage_block_counts(self):
        """_depth_coverage_block returns correct per-scope counts."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")

        from kernel.ops import fill
        fill(grid, "SEM.ENT.ECO.WHAT.SFT", "a", "a", 0.9, source=("intent_contract",))
        fill(grid, "ORG.FNC.ECO.HOW.SFT", "b", "b", 0.85, source=("intent_contract",))
        fill(grid, "STR.FNC.CMP.HOW.SFT", "c", "c", 0.8, source=("SEM.ENT.ECO.WHAT.SFT",))

        block = _depth_coverage_block(grid)
        # ECO should have at least 3 cells (intent + 2 fills at ECO scope)
        assert "ECO:" in block
        assert "CMP:" in block

    def test_deep_scope_extraction_count(self):
        """Prompt says '10-25' when target_scopes include CMP/FNC."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")

        prompt = _build_author_prompt(grid, "test input", target_scopes=("CMP", "FNC"))
        assert "10-25" in prompt

    def test_shallow_scope_extraction_count(self):
        """Prompt says '5-15' for ECO/APP scopes."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")

        prompt = _build_author_prompt(grid, "test input", target_scopes=("ECO", "APP"))
        assert "5-15" in prompt

    def test_no_scope_extraction_count(self):
        """Prompt says '5-15' when no target_scopes."""
        grid = Grid()
        grid.set_intent("test", "INT.SEM.ECO.WHY.SFT", "intent")

        prompt = _build_author_prompt(grid, "test input")
        assert "5-15" in prompt


class TestExtractionRange:
    """_extraction_range helper returns correct ranges."""

    def test_deep_scopes(self):
        for scope in ("CMP", "FNC", "STP", "OPR", "EXP", "VAL"):
            assert _extraction_range((scope,)) == "10-25"

    def test_shallow_scopes(self):
        for scope in ("ECO", "APP", "DOM", "FET"):
            assert _extraction_range((scope,)) == "5-15"

    def test_mixed_deep_and_shallow(self):
        """If any scope is deep, use the larger range."""
        assert _extraction_range(("DOM", "CMP")) == "10-25"

    def test_empty_scopes(self):
        assert _extraction_range(()) == "5-15"


class TestAxiomEnforcement:
    """AX1 provenance enforced at all depth levels."""

    def test_axiom_enforcement_at_depth(self):
        """Fills at STP/OPR scope still enforce AX1 provenance."""
        mock_fn, _ = _counting_mock_llm()

        config = CompileConfig(
            max_iterations=5,
            min_fill_rate=0.99,  # Force all iterations
            scope_schedule=(
                ("ECO", "APP"),
                ("DOM", "FET"),
                ("CMP", "FNC"),
                ("CMP", "FNC", "STP"),
                ("STP", "OPR"),
            ),
        )
        result = compile(INPUT_TEXT, mock_fn, config=config)

        # Every filled cell must have a non-empty source tuple
        for cell in result.grid.filled_cells():
            assert cell.source, f"Cell {cell.postcode.key} has empty source (AX1 violation)"
            # Source must trace to intent_contract or another cell
            has_valid_source = False
            for s in cell.source:
                if s == INTENT_CONTRACT:
                    has_valid_source = True
                    break
                if s.startswith(("human:", "contract:", "memory:")):
                    has_valid_source = True
                    break
                target = result.grid.get(s)
                if target and target.is_filled:
                    has_valid_source = True
                    break
            # Quarantined cells get their source zeroed out but we check filled only
            if cell.fill.name == "F":
                assert has_valid_source, (
                    f"Cell {cell.postcode.key} has broken provenance: source={cell.source}"
                )
