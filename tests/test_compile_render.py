"""Tests for mother/compile_render.py — transparent compilation output formatting."""

import pytest

from mother.compile_render import (
    DIMENSION_ORDER,
    CompileDisplay,
    collect_notable_events,
    format_component_summary,
    format_dimension_bar,
    format_dimension_breakdown,
    format_gaps,
    format_timing_line,
    format_voice_verdict,
    render_compile_output,
)


# --- Fixtures ---

def _make_verification(scores=None, gaps=None):
    """Build a realistic verification dict."""
    defaults = {
        "completeness": 67,
        "consistency": 92,
        "coherence": 75,
        "traceability": 58,
        "actionability": 41,
        "specificity": 49,
        "codegen_readiness": 52,
    }
    if scores:
        defaults.update(scores)

    result = {"overall_score": 0.0, "status": "needs_work"}
    for dim, score in defaults.items():
        entry = {"score": score, "details": f"{dim} details"}
        if gaps and dim in gaps:
            entry["gaps"] = gaps[dim]
        result[dim] = entry

    # Compute overall
    dim_scores = [defaults[d] for d in DIMENSION_ORDER if d in defaults]
    result["overall_score"] = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0
    return result


def _make_components(n=5):
    """Build a list of component dicts."""
    types = ["service", "entity", "process", "util", "controller"]
    return [{"name": f"Comp{i}", "type": types[i % len(types)]} for i in range(n)]


# --- format_dimension_bar ---

class TestFormatDimensionBar:
    def test_basic_bar(self):
        line = format_dimension_bar("completeness", 67)
        assert "completeness" in line
        assert "67%" in line
        assert "[" in line and "]" in line

    def test_zero_score(self):
        line = format_dimension_bar("specificity", 0)
        assert "0%" in line
        assert "[" + " " * 20 + "]" in line

    def test_full_score(self):
        line = format_dimension_bar("consistency", 100)
        assert "100%" in line
        assert "[" + "=" * 20 + "]" in line

    def test_alignment_padding(self):
        short = format_dimension_bar("coherence", 50)
        long = format_dimension_bar("codegen_readiness", 50)
        # Both bars should start at the same column
        assert short.index("[") == long.index("[")

    def test_custom_bar_width(self):
        line = format_dimension_bar("coherence", 50, bar_width=20)
        bar_start = line.index("[")
        bar_end = line.index("]")
        assert bar_end - bar_start - 1 == 20  # 20 chars inside brackets


# --- format_dimension_breakdown ---

class TestFormatDimensionBreakdown:
    def test_all_seven_dimensions(self):
        v = _make_verification()
        block = format_dimension_breakdown(v)
        lines = block.strip().split("\n")
        assert len(lines) == 7

    def test_dimension_order_preserved(self):
        v = _make_verification()
        block = format_dimension_breakdown(v)
        lines = block.strip().split("\n")
        for i, dim in enumerate(DIMENSION_ORDER):
            assert dim in lines[i]

    def test_empty_verification(self):
        assert format_dimension_breakdown({}) == ""

    def test_flat_scores(self):
        """Handles flat format: {"completeness": 80} (not nested dict)."""
        v = {"completeness": 80, "consistency": 90}
        block = format_dimension_breakdown(v)
        assert "80%" in block
        assert "90%" in block

    def test_nested_scores(self):
        """Handles nested format: {"completeness": {"score": 80}}."""
        v = _make_verification()
        block = format_dimension_breakdown(v)
        assert "67%" in block  # completeness default


# --- format_gaps ---

class TestFormatGaps:
    def test_gaps_below_threshold(self):
        v = _make_verification(
            gaps={"actionability": ["no methods", "missing types"]}
        )
        gaps = format_gaps(v, threshold=60)
        assert len(gaps) >= 1
        actionability_gap = [g for g in gaps if "actionability" in g]
        assert len(actionability_gap) == 1
        assert "no methods" in actionability_gap[0]
        assert "missing types" in actionability_gap[0]

    def test_no_gaps_when_all_above_threshold(self):
        v = _make_verification(scores={d: 80 for d in DIMENSION_ORDER})
        gaps = format_gaps(v, threshold=60)
        assert gaps == []

    def test_gap_line_without_details(self):
        """Dimension below threshold but no gap list — still shows score."""
        v = _make_verification(scores={"actionability": 30})
        gaps = format_gaps(v, threshold=60)
        actionability_gaps = [g for g in gaps if "actionability" in g]
        assert len(actionability_gaps) >= 1
        assert "30%" in actionability_gaps[0]

    def test_gap_cap_at_four(self):
        """At most 4 gap items per dimension."""
        v = _make_verification(
            scores={"specificity": 20},
            gaps={"specificity": [f"gap{i}" for i in range(10)]},
        )
        gaps = format_gaps(v, threshold=60)
        spec_gap = [g for g in gaps if "specificity" in g][0]
        # Should have at most 4 gap items joined by semicolons
        assert spec_gap.count(";") <= 3


# --- format_component_summary ---

class TestFormatComponentSummary:
    def test_basic_summary(self):
        comps = _make_components(3)
        s = format_component_summary(comps)
        assert s.startswith("Components: ")
        assert "Comp0 (service)" in s
        assert "Comp1 (entity)" in s

    def test_overflow_with_more(self):
        comps = _make_components(12)
        s = format_component_summary(comps, max_show=5)
        assert "+7 more" in s

    def test_empty_components(self):
        assert format_component_summary([]) == ""

    def test_no_type(self):
        comps = [{"name": "Widget"}]
        s = format_component_summary(comps)
        assert "Widget" in s
        assert "()" not in s  # no empty parens

    def test_non_dict_components(self):
        """Handles string components gracefully."""
        comps = ["ServiceA", "ServiceB"]
        s = format_component_summary(comps)
        assert "ServiceA" in s


# --- format_timing_line ---

class TestFormatTimingLine:
    def test_basic_timing(self):
        timings = {"intent": 2.1, "synthesis": 3.2, "verification": 1.1}
        line = format_timing_line(timings)
        assert "intent 2.1s" in line
        assert "synth 3.2s" in line
        assert "verify 1.1s" in line
        assert "total 6.4s" in line

    def test_empty_timings(self):
        assert format_timing_line({}) == ""

    def test_stage_aliases(self):
        timings = {"personas": 1.0, "dialogue": 2.0}
        line = format_timing_line(timings)
        assert "persona 1.0s" in line
        assert "dialogue 2.0s" in line

    def test_unknown_stage_passthrough(self):
        timings = {"custom_stage": 5.0}
        line = format_timing_line(timings)
        assert "custom_stage 5.0s" in line


# --- format_voice_verdict ---

class TestFormatVoiceVerdict:
    def test_high_trust(self):
        v = _make_verification(scores={d: 90 for d in DIMENSION_ORDER})
        verdict = format_voice_verdict(85, v, 10, {})
        assert "10 components" in verdict
        assert "solid" in verdict

    def test_medium_trust_with_weakest(self):
        v = _make_verification(scores={"actionability": 41})
        verdict = format_voice_verdict(55, v, 17, {})
        assert "17 components" in verdict
        assert "actionability" in verdict
        assert "41%" in verdict

    def test_low_trust(self):
        v = _make_verification(scores={d: 30 for d in DIMENSION_ORDER})
        verdict = format_voice_verdict(30, v, 5, {})
        assert "5 components" in verdict
        assert "refinement" in verdict

    def test_moderate_trust_no_dimensions(self):
        verdict = format_voice_verdict(55, {}, 8, {})
        assert "8 components" in verdict


# --- collect_notable_events ---

class TestCollectNotableEvents:
    def test_retries(self):
        events = collect_notable_events({"synthesis": 2}, {})
        assert len(events) == 1
        assert "Retried 2 times" in events[0]
        assert "synthesis" in events[0]

    def test_slow_stage(self):
        events = collect_notable_events({}, {"dialogue": 15.3}, slow_threshold=10.0)
        assert len(events) == 1
        assert "dialogue" in events[0]
        assert "15.3s" in events[0]

    def test_no_events(self):
        events = collect_notable_events({}, {"intent": 2.0})
        assert events == []

    def test_multiple_events(self):
        events = collect_notable_events(
            {"synthesis": 1},
            {"dialogue": 12.0, "verification": 11.0},
            slow_threshold=10.0,
        )
        assert len(events) == 3  # 1 retry + 2 slow


# --- render_compile_output ---

class TestRenderCompileOutput:
    def test_returns_compile_display(self):
        v = _make_verification()
        d = render_compile_output(
            verification=v,
            components=_make_components(5),
            stage_timings={"intent": 1.0, "synthesis": 2.0},
            retry_counts={},
        )
        assert isinstance(d, CompileDisplay)
        assert d.dimension_block != ""
        assert d.component_summary != ""
        assert d.timing_line != ""
        assert d.voice_verdict != ""
        assert d.personality_voice_only is False

    def test_voice_enabled_flag(self):
        v = _make_verification()
        d = render_compile_output(
            verification=v,
            components=[],
            stage_timings={},
            retry_counts={},
            voice_enabled=True,
        )
        assert d.personality_voice_only is True

    def test_empty_verification(self):
        d = render_compile_output(
            verification={},
            components=[],
            stage_timings={},
            retry_counts={},
        )
        assert d.dimension_block == ""
        assert d.gap_lines == []
        assert d.component_summary == ""
        assert d.timing_line == ""

    def test_gap_lines_populated(self):
        v = _make_verification(
            scores={"actionability": 30},
            gaps={"actionability": ["no method sigs"]},
        )
        d = render_compile_output(
            verification=v,
            components=_make_components(3),
            stage_timings={"intent": 1.0},
            retry_counts={},
        )
        assert len(d.gap_lines) >= 1
        assert any("actionability" in g for g in d.gap_lines)

    def test_notable_events_from_retries(self):
        v = _make_verification()
        d = render_compile_output(
            verification=v,
            components=_make_components(3),
            stage_timings={"dialogue": 15.0},
            retry_counts={"synthesis": 2},
        )
        assert len(d.voice_notable) >= 1
