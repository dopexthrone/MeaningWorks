"""
tests/test_kernel_llm_bridge.py — Tests for kernel LLM bridge.

Tests parse_extractions (always) and real xAI integration (when XAI_API_KEY set).
"""

import json
import os
import pytest

from kernel.llm_bridge import parse_extractions, _validate_extractions


# ---------------------------------------------------------------------------
# parse_extractions — pure function, no API needed
# ---------------------------------------------------------------------------

class TestParseExtractions:
    """Test JSON parsing from LLM responses."""

    def test_clean_json_array(self):
        raw = json.dumps([
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "concept", "content": "A thing", "confidence": 0.9, "connections": []},
        ])
        result = parse_extractions(raw)
        assert len(result) == 1
        assert result[0]["postcode"] == "SEM.ENT.ECO.WHAT.SFT"
        assert result[0]["confidence"] == 0.9

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"postcode": "COG.BHV.ECO.HOW.COG", "primitive": "fill", "content": "Fill op", "confidence": 0.8}]\n```'
        result = parse_extractions(raw)
        assert len(result) == 1
        assert result[0]["primitive"] == "fill"

    def test_markdown_fenced_no_lang(self):
        raw = '```\n[{"postcode": "ORG.ENT.ECO.WHAT.SFT", "primitive": "grid", "content": "Grid structure"}]\n```'
        result = parse_extractions(raw)
        assert len(result) == 1

    def test_json_with_surrounding_text(self):
        raw = 'Here are the extractions:\n[{"postcode": "STR.FNC.ECO.HOW.SFT", "primitive": "nav", "content": "Navigator"}]\nThat covers the main concepts.'
        result = parse_extractions(raw)
        assert len(result) == 1
        assert result[0]["primitive"] == "nav"

    def test_trailing_comma(self):
        raw = '[{"postcode": "AGN.BHV.ECO.WHO.SFT", "primitive": "agent", "content": "Agent pipeline", "confidence": 0.85,}]'
        result = parse_extractions(raw)
        assert len(result) == 1

    def test_multiple_extractions(self):
        raw = json.dumps([
            {"postcode": "SEM.ENT.ECO.WHAT.COG", "primitive": "a", "content": "A"},
            {"postcode": "ORG.ENT.ECO.WHAT.SFT", "primitive": "b", "content": "B"},
            {"postcode": "COG.BHV.ECO.HOW.COG", "primitive": "c", "content": "C"},
        ])
        result = parse_extractions(raw)
        assert len(result) == 3

    def test_missing_postcode_dropped(self):
        raw = json.dumps([
            {"primitive": "no-postcode", "content": "Missing postcode"},
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "valid", "content": "Has postcode"},
        ])
        result = parse_extractions(raw)
        assert len(result) == 1
        assert result[0]["primitive"] == "valid"

    def test_missing_primitive_dropped(self):
        raw = json.dumps([
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "content": "No primitive"},
        ])
        result = parse_extractions(raw)
        assert len(result) == 0

    def test_empty_response(self):
        assert parse_extractions("") == []
        assert parse_extractions("[]") == []

    def test_garbage_response(self):
        assert parse_extractions("This is not JSON at all.") == []

    def test_confidence_defaults_to_half(self):
        raw = json.dumps([{"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x", "content": "y"}])
        result = parse_extractions(raw)
        assert result[0]["confidence"] == 0.5

    def test_confidence_clamped(self):
        raw = json.dumps([
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x", "content": "y", "confidence": 1.5},
            {"postcode": "ORG.ENT.ECO.WHAT.SFT", "primitive": "z", "content": "w", "confidence": -0.3},
        ])
        result = parse_extractions(raw)
        assert result[0]["confidence"] == 1.0
        assert result[1]["confidence"] == 0.0

    def test_connections_default_to_empty(self):
        raw = json.dumps([{"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x", "content": "y"}])
        result = parse_extractions(raw)
        assert result[0]["connections"] == []

    def test_partial_objects_extracted(self):
        """Even malformed JSON with some valid objects should extract what it can."""
        raw = '{"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "a", "content": "A"} garbage {"postcode": "ORG.ENT.ECO.WHAT.SFT", "primitive": "b", "content": "B"}'
        result = parse_extractions(raw)
        assert len(result) == 2

    def test_nested_json_ignored(self):
        """Only top-level objects in the array matter."""
        raw = json.dumps([
            {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x", "content": "y", "extra": {"nested": True}},
        ])
        result = parse_extractions(raw)
        assert len(result) == 1


class TestValidateExtractions:
    """Test the validation/filtering logic."""

    def test_non_dict_items_dropped(self):
        result = _validate_extractions(["string", 42, None, {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "x"}])
        assert len(result) == 1

    def test_empty_postcode_dropped(self):
        result = _validate_extractions([{"postcode": "", "primitive": "x"}])
        assert len(result) == 0

    def test_strips_whitespace(self):
        result = _validate_extractions([{
            "postcode": "  SEM.ENT.ECO.WHAT.SFT  ",
            "primitive": "  concept  ",
            "content": "  something  ",
        }])
        assert result[0]["postcode"] == "SEM.ENT.ECO.WHAT.SFT"
        assert result[0]["primitive"] == "concept"
        assert result[0]["content"] == "something"


# ---------------------------------------------------------------------------
# Real xAI integration — only runs when XAI_API_KEY is set
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY"),
    reason="XAI_API_KEY not set — skipping real LLM test",
)
class TestXAIIntegration:
    """Real integration tests hitting xAI Grok API."""

    def test_make_llm_function_returns_callable(self):
        from kernel.llm_bridge import make_llm_function
        fn = make_llm_function(provider="grok")
        assert callable(fn)

    def test_simple_extraction(self):
        """Send a simple input and verify we get structured extractions back."""
        from kernel.llm_bridge import make_llm_function
        fn = make_llm_function(provider="grok", temperature=0.0)

        prompt = """You are a semantic compiler. Extract structured concepts from the input text.

INPUT TEXT:
A tattoo booking system where customers can browse artist portfolios, pick a style, and schedule appointments online.

CURRENT MAP STATE (0 filled cells):
(empty — this is the first pass)

COORDINATE SCHEMA:
Postcode format: LAYER.CONCERN.SCOPE.DIMENSION.DOMAIN
Layers: INT(Intent) SEM(Semantic) ORG(Organization) COG(Cognitive) AGN(Agency) STR(Structure) STA(State) IDN(Identity) TME(Time) EXC(Execution) CTR(Control) RES(Resource) OBS(Observability) NET(Network) EMG(Emergence) MET(Meta)
Concerns: SEM ENT BHV FNC REL PLN MEM ORC AGT ACT SCO STA TRN SNP VRS SCH GTE PLY MET LOG LMT FLW CND INT PRV CNS
Scopes: ECO(Ecosystem) APP(Application) DOM(Domain) FET(Feature) CMP(Component) FNC(Function) STP(Step) OPR(Operation) EXP(Expression) VAL(Value)
Dimensions: WHAT HOW WHY WHO WHEN WHERE IF HOW_MUCH
Domains: SFT ORG COG NET ECN PHY SOC EDU MED LGL

INSTRUCTIONS:
1. Extract concepts from the input text
2. Assign each a 5-axis postcode
3. Rate confidence 0.0-1.0 (how certain is this extraction)
4. List connections to other postcodes (existing or new)
5. Every extraction MUST trace to the input text — do not hallucinate

Return a JSON array of objects:
[{"postcode": "...", "primitive": "...", "content": "...", "confidence": 0.0-1.0, "connections": ["..."]}]

Extract 5-15 concepts."""

        result = fn(prompt)

        # Should get structured extractions
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) >= 3, f"Expected >=3 extractions, got {len(result)}"

        for ext in result:
            assert "postcode" in ext
            assert "primitive" in ext
            assert "content" in ext
            assert "confidence" in ext
            # Postcode should look like 5 dot-separated parts
            parts = ext["postcode"].split(".")
            assert len(parts) == 5, f"Bad postcode: {ext['postcode']}"

    def test_full_compile_with_grok(self):
        """Run the full kernel compile() with a real LLM."""
        from kernel.llm_bridge import make_llm_function
        from kernel.agents import compile, CompileConfig

        fn = make_llm_function(provider="grok", temperature=0.0)
        config = CompileConfig(max_iterations=2)  # Limit to 2 iterations for cost

        result = compile(
            input_text="A tattoo booking system where customers browse portfolios, pick styles, and schedule appointments.",
            llm_fn=fn,
            config=config,
        )

        assert result.grid.total_cells > 1
        filled = result.grid.filled_cells()
        assert len(filled) >= 1, "Should have at least 1 filled cell from LLM"

        # Print summary for manual inspection
        print(f"\n  Iterations: {result.iterations}")
        print(f"  Total cells: {result.grid.total_cells}")
        print(f"  Filled: {len(result.grid.filled_cells())}")
        print(f"  Layers: {sorted(result.grid.activated_layers)}")
        print(f"  Converged: {result.converged}")
        for cell in filled[:10]:
            print(f"    {cell.postcode.key} | {cell.fill.name} {cell.confidence:.2f} | {cell.primitive}")
