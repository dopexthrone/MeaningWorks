"""Tests for core/exploration_synthesis.py — exploration map extraction."""

import pytest
from core.exploration_synthesis import (
    Insight,
    FrontierQuestion,
    AlternativeFraming,
    ExplorationMap,
    synthesize_exploration,
    exploration_map_to_dict,
    format_exploration_summary,
)


# ============================================================
# Test fixtures
# ============================================================

def _make_cell(postcode, primitive, content="", fill="F", confidence=0.8, connections=(), parent=None):
    return {
        "postcode": postcode,
        "primitive": primitive,
        "content": content,
        "fill": fill,
        "confidence": confidence,
        "connections": connections,
        "parent": parent,
    }


def _make_msg(sender, content):
    return {"sender": sender, "content": content}


SAMPLE_CELLS = [
    _make_cell("ORG.ENT.GLB.STR.SFT", "UserService", "User management", "F", 0.9),
    _make_cell("ORG.BHV.GLB.STR.SFT", "Authentication", "Login flow", "F", 0.85,
               connections=("ORG.ENT.GLB.STR.SFT",)),
    _make_cell("ORG.FNC.GLB.STR.SFT", "validate", "Input validation", "P", 0.4),
    _make_cell("APP.ENT.GLB.STR.SFT", "Dashboard", "Main UI", "E", 0.0,
               connections=("ORG.ENT.GLB.STR.SFT",)),
    _make_cell("APP.BHV.GLB.STR.SFT", "Routing", "Page routing", "E", 0.0),
    _make_cell("SFT.ENT.GLB.STR.SFT", "Database", "Data layer", "F", 0.7),
]

SAMPLE_MESSAGES = [
    _make_msg("Entity", "The system needs user authentication."),
    _make_msg("Process", "INSIGHT: Session management is the core complexity."),
    _make_msg("Entity", "Alternatively, we could use a stateless JWT approach."),
    _make_msg("Process", "What about compliance requirements for the EU market?"),
]


# ============================================================
# Insight extraction
# ============================================================

class TestInsightExtraction:
    """Insights from grid patterns and dialogue."""

    def test_dominant_layer_detected(self):
        # ORG has 3 cells (filled+partial) — should be dominant
        exp = synthesize_exploration(SAMPLE_CELLS, [], "test")
        pattern_insights = [i for i in exp.insights if i.category == "pattern"]
        assert any("ORG" in i.text for i in pattern_insights)

    def test_low_confidence_gap(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "A", fill="F", confidence=0.3),
            _make_cell("ORG.BHV.GLB.STR.SFT", "B", fill="P", confidence=0.4),
        ]
        exp = synthesize_exploration(cells, [], "test")
        gap_insights = [i for i in exp.insights if i.category == "gap"]
        assert len(gap_insights) >= 1

    def test_confidence_spread_contradiction(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "Solid", fill="F", confidence=0.95),
            _make_cell("ORG.BHV.GLB.STR.SFT", "Shaky", fill="P", confidence=0.3),
        ]
        exp = synthesize_exploration(cells, [], "test")
        contradictions = [i for i in exp.insights if i.category == "contradiction"]
        assert len(contradictions) >= 1

    def test_dialogue_insight_markers(self):
        exp = synthesize_exploration([], SAMPLE_MESSAGES, "test")
        insights = [i for i in exp.insights if i.source.startswith("dialogue:")]
        assert any("Session management" in i.text for i in insights)

    def test_opportunity_from_frontier(self):
        # Dashboard is empty but connected to UserService (filled, high conf)
        exp = synthesize_exploration(SAMPLE_CELLS, [], "test")
        opps = [i for i in exp.insights if i.category == "opportunity"]
        assert any("Dashboard" in i.text for i in opps)

    def test_capped_at_20(self):
        cells = [_make_cell(f"L{i}.ENT.GLB.STR.SFT", f"C{i}", fill="F", confidence=0.3)
                 for i in range(30)]
        exp = synthesize_exploration(cells, [], "test")
        assert len(exp.insights) <= 20

    def test_sorted_by_confidence(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        if exp.insights:
            confs = [i.confidence for i in exp.insights]
            assert confs == sorted(confs, reverse=True)


# ============================================================
# Frontier questions
# ============================================================

class TestFrontierQuestions:
    """Frontier questions from grid boundaries."""

    def test_underfilled_layer(self):
        # APP has 2 cells, 0 filled → should generate question
        exp = synthesize_exploration(SAMPLE_CELLS, [], "test")
        app_questions = [q for q in exp.frontier_questions if q.domain == "APP"]
        assert len(app_questions) >= 1

    def test_referenced_but_not_activated(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "X", fill="F", confidence=0.9,
                       connections=("NEW.ENT.GLB.STR.SFT",)),
        ]
        exp = synthesize_exploration(cells, [], "test")
        questions = [q for q in exp.frontier_questions if q.domain == "NEW"]
        assert len(questions) >= 1

    def test_user_keyword_generates_question(self):
        exp = synthesize_exploration([], [], "Build a user management system")
        questions = [q for q in exp.frontier_questions if q.domain == "USR"]
        assert len(questions) >= 1

    def test_scale_keyword_generates_question(self):
        exp = synthesize_exploration([], [], "Build a large-scale data pipeline")
        questions = [q for q in exp.frontier_questions if q.domain == "RES"]
        assert len(questions) >= 1

    def test_sorted_by_priority(self):
        exp = synthesize_exploration(SAMPLE_CELLS, [], "test")
        if exp.frontier_questions:
            priorities = [q.priority for q in exp.frontier_questions]
            assert priorities == sorted(priorities, reverse=True)

    def test_capped_at_10(self):
        # Many layers with empty cells
        cells = [_make_cell(f"L{i}.ENT.GLB.STR.SFT", f"C{i}", fill="E")
                 for i in range(20)]
        exp = synthesize_exploration(cells, [], "test with user and team and scale")
        assert len(exp.frontier_questions) <= 10


# ============================================================
# Adjacent domains
# ============================================================

class TestAdjacentDomains:
    """Adjacent domain discovery."""

    def test_sft_layer_adjacents(self):
        cells = [_make_cell("SFT.ENT.GLB.STR.SFT", "X", fill="F", confidence=0.9)]
        exp = synthesize_exploration(cells, [], "test")
        assert "security" in exp.adjacent_domains

    def test_dialogue_domain_hints(self):
        msgs = [_make_msg("Entity", "We need to consider compliance with GDPR.")]
        exp = synthesize_exploration([], msgs, "test")
        assert "compliance" in exp.adjacent_domains

    def test_capped_at_8(self):
        # Activate many layers
        cells = [_make_cell(f"{layer}.ENT.GLB.STR.SFT", f"C", fill="F")
                 for layer in ["SFT", "ORG", "DOM", "APP", "ECO"]]
        exp = synthesize_exploration(cells, [], "test")
        assert len(exp.adjacent_domains) <= 8

    def test_sorted(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        assert list(exp.adjacent_domains) == sorted(exp.adjacent_domains)


# ============================================================
# Alternative framings
# ============================================================

class TestAlternativeFramings:
    """Alternative framings of the original intent."""

    def test_generates_base_perspectives(self):
        exp = synthesize_exploration([], [], "Build a task manager")
        perspectives = [f.perspective for f in exp.alternative_framings]
        assert "user" in perspectives
        assert "economic" in perspectives
        assert "temporal" in perspectives
        assert "failure" in perspectives
        assert "inverse" in perspectives

    def test_includes_intent_text(self):
        exp = synthesize_exploration([], [], "Build a task manager")
        for f in exp.alternative_framings:
            if f.perspective != "dialogue":
                assert "Build a task manager" in f.framing

    def test_divergence_ranges(self):
        exp = synthesize_exploration([], [], "test")
        for f in exp.alternative_framings:
            assert 0.0 <= f.divergence <= 1.0

    def test_inverse_has_high_divergence(self):
        exp = synthesize_exploration([], [], "test")
        inverse = [f for f in exp.alternative_framings if f.perspective == "inverse"]
        assert inverse[0].divergence >= 0.8

    def test_dialogue_reframings(self):
        exp = synthesize_exploration([], SAMPLE_MESSAGES, "test")
        dialogue_framings = [f for f in exp.alternative_framings if f.perspective == "dialogue"]
        assert any("alternatively" in f.framing.lower() or "stateless" in f.framing.lower()
                    for f in dialogue_framings)

    def test_capped_at_8(self):
        msgs = [_make_msg("Entity", f"Alternatively, approach {i} could work.")
                for i in range(10)]
        exp = synthesize_exploration([], msgs, "test")
        assert len(exp.alternative_framings) <= 8

    def test_empty_intent_no_framings(self):
        exp = synthesize_exploration([], [], "")
        assert len(exp.alternative_framings) == 0


# ============================================================
# Depth chains
# ============================================================

class TestDepthChains:
    """Depth chain passthrough."""

    def test_passes_through_chains(self):
        chains = [
            {"chain_type": "frontier", "intent_text": "Explore X", "priority": 0.8},
            {"chain_type": "low_conf", "intent_text": "Deepen Y", "priority": 0.6},
        ]
        exp = synthesize_exploration([], [], "test", endpoint_chains=chains)
        assert len(exp.depth_chains) == 2

    def test_none_chains_empty(self):
        exp = synthesize_exploration([], [], "test", endpoint_chains=None)
        assert len(exp.depth_chains) == 0


# ============================================================
# ExplorationMap frozen
# ============================================================

class TestExplorationMapFrozen:
    """ExplorationMap is a frozen dataclass."""

    def test_frozen(self):
        exp = synthesize_exploration([], [], "test")
        with pytest.raises(AttributeError):
            exp.original_intent = "changed"

    def test_original_intent_preserved(self):
        exp = synthesize_exploration([], [], "Build a task manager")
        assert exp.original_intent == "Build a task manager"


# ============================================================
# Serialization
# ============================================================

class TestExplorationMapSerialization:
    """exploration_map_to_dict() JSON serialization."""

    def test_round_trips(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        d = exploration_map_to_dict(exp)
        assert isinstance(d, dict)
        assert d["original_intent"] == "test"
        assert isinstance(d["insights"], list)
        assert isinstance(d["frontier_questions"], list)
        assert isinstance(d["adjacent_domains"], list)
        assert isinstance(d["alternative_framings"], list)
        assert isinstance(d["depth_chains"], list)

    def test_insight_keys(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        d = exploration_map_to_dict(exp)
        if d["insights"]:
            i = d["insights"][0]
            assert "text" in i
            assert "category" in i
            assert "confidence" in i
            assert "source" in i

    def test_empty_serializes(self):
        exp = synthesize_exploration([], [], "test")
        d = exploration_map_to_dict(exp)
        assert d["insights"] == []
        assert d["frontier_questions"] == []


# ============================================================
# Format summary
# ============================================================

class TestFormatExplorationSummary:
    """format_exploration_summary() human-readable output."""

    def test_includes_intent(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "Build auth")
        summary = format_exploration_summary(exp)
        assert "Build auth" in summary

    def test_includes_insights_section(self):
        exp = synthesize_exploration(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        summary = format_exploration_summary(exp)
        assert "Insights" in summary

    def test_includes_frontier_section(self):
        exp = synthesize_exploration(SAMPLE_CELLS, [], "test")
        summary = format_exploration_summary(exp)
        assert "Frontier" in summary

    def test_empty_still_formats(self):
        exp = synthesize_exploration([], [], "test")
        summary = format_exploration_summary(exp)
        assert "Exploration:" in summary
