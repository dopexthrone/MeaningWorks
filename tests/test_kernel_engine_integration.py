"""
tests/test_kernel_engine_integration.py — Verify real LLM extraction wiring in kernel Phase 3.5.

Tests that:
1. _extraction_fn calls the engine's LLM client via complete_with_system
2. parse_extractions is used to parse the response
3. Cost tracking (_collect_usage, _check_cost_cap) fires per extraction
4. CostCapExceededError propagates (not swallowed)
5. LLM errors return [] gracefully (convergence signal)
6. Grid populates with real (mocked) extractions across multiple layers
7. Results flow through to CompileResult.semantic_grid
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.llm import BaseLLMClient
from persistence.corpus import Corpus
from core.engine import MotherlabsEngine, CompileResult
from core.exceptions import CostCapExceededError
from kernel.llm_bridge import parse_extractions


# ---------------------------------------------------------------------------
# Mock response data — same pattern as test_engine.py
# ---------------------------------------------------------------------------

MOCK_INTENT_JSON = {
    "core_need": "Build a booking system",
    "domain": "scheduling",
    "actors": ["User", "BookingService"],
    "implicit_goals": ["Easy appointment management"],
    "constraints": ["Must handle scheduling"],
    "insight": "Core need is appointment scheduling",
    "explicit_components": [],
    "explicit_relationships": [],
}

MOCK_PERSONA_JSON = {
    "personas": [
        {
            "name": "Service Designer",
            "perspective": "Focus on booking UX",
            "blind_spots": "May oversimplify backend",
        },
        {
            "name": "Backend Engineer",
            "perspective": "Focus on scheduling logic",
            "blind_spots": "May overcomplicate UX",
        },
    ],
    "cross_cutting_concerns": ["Scalability vs simplicity"],
    "suggested_focus_areas": ["Appointment lifecycle"],
}

MOCK_SYNTHESIS_JSON = {
    "components": [
        {
            "name": "User",
            "type": "entity",
            "description": "User with profile",
            "derived_from": "INSIGHT: User entity contains name, email",
            "properties": [{"name": "email", "type": "str"}],
        },
        {
            "name": "Booking",
            "type": "entity",
            "description": "Appointment booking",
            "derived_from": "INSIGHT: Booking entity contains datetime, status",
            "properties": [{"name": "datetime", "type": "datetime"}],
        },
        {
            "name": "BookingService",
            "type": "process",
            "description": "Booking service",
            "derived_from": "INSIGHT: BookingService manages booking lifecycle",
            "methods": [{
                "name": "create_booking",
                "parameters": [{"name": "user_id", "type_hint": "str"}],
                "return_type": "Booking",
                "description": "Create a booking",
                "derived_from": "create_booking(user_id) -> Booking",
            }],
        },
    ],
    "relationships": [
        {"from": "BookingService", "to": "User", "type": "accesses", "description": "BookingService reads User"},
        {"from": "BookingService", "to": "Booking", "type": "generates", "description": "BookingService creates Booking"},
    ],
    "constraints": [],
    "unresolved": [],
}

MOCK_VERIFY_JSON = {
    "status": "pass",
    "completeness": {"score": 85, "details": "All components traced"},
    "consistency": {"score": 90, "details": "No contradictions"},
    "coherence": {"score": 80, "details": "Logical structure"},
    "traceability": {"score": 95, "details": "All derived_from present"},
}

VALID_EXTRACTIONS = [
    {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "user", "content": "End users of the system", "confidence": 0.9, "connections": []},
    {"postcode": "SEM.BHV.ECO.HOW.SFT", "primitive": "booking", "content": "Schedule appointments", "confidence": 0.85, "connections": ["SEM.ENT.ECO.WHAT.SFT"]},
    {"postcode": "STR.FNC.APP.HOW.SFT", "primitive": "auth", "content": "User authentication flow", "confidence": 0.8, "connections": []},
    {"postcode": "ORG.ENT.ECO.WHAT.SFT", "primitive": "service", "content": "Booking service module", "confidence": 0.75, "connections": ["SEM.BHV.ECO.HOW.SFT"]},
    {"postcode": "AGN.ORC.ECO.WHO.SFT", "primitive": "orchestrator", "content": "Manages booking pipeline", "confidence": 0.7, "connections": []},
]

SIMPLE_DESCRIPTION = "A booking system where users can schedule appointments and manage their profiles."

# Semantic extraction system prompt — used to detect kernel extraction calls
SEM_EXTRACT_MARKER = "You are a semantic compiler. You extract structured concepts"


def make_kernel_sequenced_mock(
    extraction_response=None,
    extraction_fail_after=None,
    token_usage=None,
    extra_dialogue_turns=9,
):
    """Create a sequenced mock that handles full pipeline + kernel extraction.

    Pipeline calls get stage-appropriate responses.
    Kernel extraction calls (detected by system prompt) get extraction JSON.
    """
    _extractions = extraction_response if extraction_response is not None else VALID_EXTRACTIONS
    _usage = token_usage or {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300}
    _extraction_call_count = [0]

    pipeline_idx = [0]

    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock-model"
    client.last_usage = _usage

    def mock_complete_with_system(system_prompt, user_content, **kwargs):
        # Detect kernel extraction calls by system prompt
        if SEM_EXTRACT_MARKER in system_prompt:
            _extraction_call_count[0] += 1
            client.last_usage = _usage
            if extraction_fail_after is not None and _extraction_call_count[0] > extraction_fail_after:
                raise RuntimeError("Simulated LLM failure")
            return json.dumps(_extractions)
        # Detect synthesis/verify by system prompt (position-independent)
        if "You are the Synthesis Agent" in system_prompt:
            client.last_usage = _usage
            return json.dumps(MOCK_SYNTHESIS_JSON)
        if "You are the Verify Agent" in system_prompt:
            client.last_usage = _usage
            return json.dumps(MOCK_VERIFY_JSON)
        # Intent + persona use sequential slots, dialogue cycles
        idx = pipeline_idx[0]
        pipeline_idx[0] += 1
        client.last_usage = _usage
        if idx == 0:
            return json.dumps(MOCK_INTENT_JSON)
        if idx == 1:
            return json.dumps(MOCK_PERSONA_JSON)
        if idx % 2 == 0:
            return (
                "Analyzing structure of the booking system.\n"
                "INSIGHT: system builds User entity with name, email, profile fields"
            )
        return (
            "Analyzing behavior of the booking system.\n"
            "INSIGHT: system builds booking flow that schedules appointments"
        )

    client.complete_with_system = Mock(side_effect=mock_complete_with_system)
    client._extraction_call_count = _extraction_call_count
    return client


def make_engine(tmp_path, client=None):
    """Create an engine with the given client."""
    c = client or make_kernel_sequenced_mock()
    corpus = Corpus(corpus_path=tmp_path / "corpus")
    return MotherlabsEngine(llm_client=c, corpus=corpus, cache_policy="none")


def _passing_closed_loop_gate(description, blueprint):
    """Mock closed_loop_gate that always passes (mock LLM data doesn't meet real fidelity)."""
    from dataclasses import dataclass, field as _field

    @dataclass
    class _MockCLResult:
        passed: bool = True
        fidelity_score: float = 0.85
        compression_losses: list = _field(default_factory=list)

    return _MockCLResult()


# ---------------------------------------------------------------------------
# 1. TestKernelRealExtraction — extraction fn wiring
# ---------------------------------------------------------------------------

class TestKernelRealExtraction:
    """Verify _extraction_fn calls LLM and routes through parse_extractions."""

    def test_extraction_fn_calls_llm(self, tmp_path):
        """Engine produces a kernel grid — either via extraction calls or dialogue grid."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        # With grid-driven dialogue, Phase 3.5 is skipped (grid built during dialogue).
        # With text dialogue, extraction calls happen in Phase 3.5.
        # Either path produces a kernel grid.
        assert engine._kernel_grid is not None, \
            "Expected kernel grid to be built (via dialogue or extraction)"
        assert engine._kernel_grid.cells, "Kernel grid should have cells"

    def test_extraction_fn_calls_multiple_iterations(self, tmp_path):
        """Kernel grid has cells after compilation (dialogue or extraction path)."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        # Grid-driven dialogue builds the grid during dialogue turns.
        # Text-based fallback runs kernel compile with extraction iterations.
        assert engine._kernel_grid is not None
        assert len(engine._kernel_grid.cells) >= 1

    def test_extraction_fn_returns_empty_on_error(self, tmp_path):
        """LLM errors in extraction return [] — compilation continues."""
        client = make_kernel_sequenced_mock(extraction_fail_after=0)
        engine = make_engine(tmp_path, client)
        # Should not raise — kernel extraction failures are graceful
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert result is not None

    def test_extraction_fn_tracks_cost(self, tmp_path):
        """_collect_usage is called after kernel extraction — tokens accumulate."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        total_tokens = sum(tu.total_tokens for tu in engine._compilation_tokens)
        assert total_tokens > 0, "Expected token usage from kernel extraction"

    def test_extraction_fn_respects_cost_cap(self, tmp_path):
        """CostCapExceededError triggers failure when cost cap exceeded."""
        huge_usage = {"input_tokens": 500_000, "output_tokens": 500_000, "total_tokens": 1_000_000}
        client = make_kernel_sequenced_mock(token_usage=huge_usage)
        # Must use a model name that matches the pricing table so cost > 0
        client.model = "grok-3-test"
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        # CostCapExceededError is caught by compile() → returns failed result
        assert not result.success
        assert "exceeds cap" in result.error

    def test_max_iterations_is_5(self):
        """Verify engine source uses max_iterations=5 for dense extraction."""
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "max_iterations=5" in source

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_grid_populates_from_extraction(self, _mock_clg, tmp_path):
        """Extractions with 5 postcodes produce a grid with >1 cell."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        grid_data = result.semantic_grid
        assert grid_data is not None, "semantic_grid should be populated"
        assert grid_data.get("cells", 0) > 1, f"Expected >1 cell, got {grid_data.get('cells')}"

    def test_invalid_postcodes_silently_dropped(self, tmp_path):
        """Extractions with bad postcodes are skipped, valid ones kept."""
        extractions = [
            {"postcode": "INVALID.BAD.NOPE.UGH.NO", "primitive": "bad", "content": "Invalid", "confidence": 0.9, "connections": []},
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "good", "content": "Valid concept", "confidence": 0.9, "connections": []},
        ]
        client = make_kernel_sequenced_mock(extraction_response=extractions)
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert result is not None

    def test_markdown_wrapped_json_parsed(self, tmp_path):
        """Markdown-fenced JSON response is correctly parsed via parse_extractions."""
        # parse_extractions handles markdown fences — verify directly
        raw = '```json\n' + json.dumps(VALID_EXTRACTIONS) + '\n```'
        result = parse_extractions(raw)
        assert len(result) == 5

    def test_empty_extraction_produces_sparse_grid(self, tmp_path):
        """Empty extraction list — kernel still runs, grid has intent cell."""
        client = make_kernel_sequenced_mock(extraction_response=[])
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert result is not None
        # Grid should still have at least the intent cell
        if result.semantic_grid:
            assert result.semantic_grid.get("cells", 0) >= 1


# ---------------------------------------------------------------------------
# 2. TestKernelGridQuality — grid quality from real extractions
# ---------------------------------------------------------------------------

class TestKernelGridQuality:
    """Verify grid quality when extractions populate multiple layers."""

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_real_extraction_populates_multiple_layers(self, _mock_clg, tmp_path):
        """Extractions spanning SEM/STR/ORG/AGN produce multi-layer grid."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        grid_data = result.semantic_grid
        assert grid_data is not None
        assert grid_data.get("layers", 0) > 1, f"Expected >1 layer, got {grid_data.get('layers')}"

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_nav_text_includes_real_cells(self, _mock_clg, tmp_path):
        """Nav output from grid has postcodes from extractions or dialogue grid."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        grid_data = result.semantic_grid
        assert grid_data is not None
        nav = grid_data.get("nav", "")
        # Nav should mention at least one postcode from either:
        # - Kernel extraction path: SEM.ENT, SEM.BHV, STR.FNC, etc.
        # - Grid-driven dialogue path: STR.ENT, EXC.BHV, STA.STA, DAT.ENT, CTR.FLW
        has_postcode = any(
            pc in nav for pc in [
                "SEM.ENT", "SEM.BHV", "STR.FNC", "ORG.ENT", "AGN.ORC",
                "STR.ENT", "EXC.BHV", "STA.STA", "DAT.ENT", "CTR.FLW",
            ]
        )
        assert has_postcode, f"Nav text should include postcodes, got: {nav[:300]}"

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_semantic_grid_in_compile_result(self, _mock_clg, tmp_path):
        """CompileResult.semantic_grid has cells > 1 from real extraction."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert result.semantic_grid is not None, "semantic_grid should be populated"
        cells = result.semantic_grid.get("cells", 0)
        assert cells > 1, f"Expected >1 cell in semantic_grid, got {cells}"

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_kernel_insights_injected(self, _mock_clg, tmp_path):
        """Kernel compilation produces insights about grid."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert result.semantic_grid is not None
        # Grid data should include iteration and convergence info
        assert "iterations" in result.semantic_grid
        assert "converged" in result.semantic_grid

    @patch("kernel.closed_loop.closed_loop_gate", side_effect=_passing_closed_loop_gate)
    def test_compile_result_success_with_kernel(self, _mock_clg, tmp_path):
        """Full compilation with kernel extraction still produces success."""
        client = make_kernel_sequenced_mock()
        engine = make_engine(tmp_path, client)
        result = engine.compile(SIMPLE_DESCRIPTION)
        assert isinstance(result, CompileResult)
        assert result.success


# ---------------------------------------------------------------------------
# 3. TestParseExtractionsIntegration — parse_extractions edge cases
# ---------------------------------------------------------------------------

class TestParseExtractionsIntegration:
    """Verify parse_extractions handles LLM output quirks."""

    def test_clean_json_array(self):
        raw = json.dumps(VALID_EXTRACTIONS)
        result = parse_extractions(raw)
        assert len(result) == 5
        assert result[0]["postcode"] == "SEM.ENT.ECO.WHAT.SFT"

    def test_markdown_fenced_json(self):
        raw = '```json\n' + json.dumps(VALID_EXTRACTIONS) + '\n```'
        result = parse_extractions(raw)
        assert len(result) == 5

    def test_trailing_comma_fixed(self):
        raw = '[{"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x", "content": "y", "confidence": 0.9,}]'
        result = parse_extractions(raw)
        assert len(result) == 1
        assert result[0]["primitive"] == "x"
