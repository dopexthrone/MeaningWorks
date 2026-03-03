"""Tests for core/context_synthesis.py — context map extraction."""

import pytest
from core.context_synthesis import (
    Concept,
    Relationship,
    Assumption,
    Unknown,
    ContextMap,
    synthesize_context,
    context_map_to_dict,
    format_context_summary,
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
    _make_cell("ORG.ENT.GLB.STR.SFT", "UserService", "Manages user accounts", "F", 0.9),
    _make_cell("ORG.BHV.GLB.STR.SFT", "Authentication", "Login and session management", "F", 0.85,
               connections=("ORG.ENT.GLB.STR.SFT",)),
    _make_cell("ORG.FNC.GLB.STR.SFT", "hashPassword", "Bcrypt hash", "P", 0.6),
    _make_cell("APP.ENT.GLB.STR.SFT", "Dashboard", "Main UI", "E", 0.0),
    _make_cell("APP.BHV.GLB.STR.SFT", "Routing", "Page navigation", "Q", 0.3),
]

SAMPLE_MESSAGES = [
    _make_msg("Entity", "The system assumes all users have email addresses."),
    _make_msg("Process", "Authentication typically uses JWT tokens."),
    _make_msg("Entity", "What about SSO integration?"),
    _make_msg("Process", "INSIGHT: Password hashing should use bcrypt with cost 12."),
]


# ============================================================
# Concept extraction
# ============================================================

class TestConceptExtraction:
    """Concepts extracted from filled/partial cells."""

    def test_extracts_filled_cells(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        names = [c.name for c in ctx.concepts]
        assert "UserService" in names
        assert "Authentication" in names

    def test_extracts_partial_cells(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        names = [c.name for c in ctx.concepts]
        assert "hashPassword" in names

    def test_skips_empty_cells(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        names = [c.name for c in ctx.concepts]
        assert "Dashboard" not in names

    def test_skips_questioned_cells(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        names = [c.name for c in ctx.concepts]
        assert "Routing" not in names

    def test_sorted_by_confidence(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        confs = [c.confidence for c in ctx.concepts]
        assert confs == sorted(confs, reverse=True)

    def test_concept_has_layer(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        user_svc = next(c for c in ctx.concepts if c.name == "UserService")
        assert user_svc.layer == "ORG"

    def test_concept_has_concern(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        user_svc = next(c for c in ctx.concepts if c.name == "UserService")
        assert user_svc.concern == "ENT"

    def test_concept_has_source_postcode(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        user_svc = next(c for c in ctx.concepts if c.name == "UserService")
        assert user_svc.source_postcode == "ORG.ENT.GLB.STR.SFT"

    def test_deduplicates_by_name(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "Foo", fill="F", confidence=0.9),
            _make_cell("APP.ENT.GLB.STR.SFT", "Foo", fill="F", confidence=0.7),
        ]
        ctx = synthesize_context(cells, [], "test")
        assert len([c for c in ctx.concepts if c.name == "Foo"]) == 1

    def test_empty_cells_no_concepts(self):
        ctx = synthesize_context([], [], "test")
        assert len(ctx.concepts) == 0


# ============================================================
# Relationship extraction
# ============================================================

class TestRelationshipExtraction:
    """Relationships from connections and parent refs."""

    def test_connection_relationships(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        rels = [r for r in ctx.relationships if r.relation_type == "connected"]
        assert len(rels) >= 1
        # Authentication → UserService connection
        rel_pairs = [(r.source, r.target) for r in rels]
        assert any(
            ("Authentication" in pair and "UserService" in pair)
            for pair in [(s, t) for s, t in rel_pairs]
        )

    def test_parent_child_relationships(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "Parent", fill="F", confidence=0.9),
            _make_cell("ORG.FNC.GLB.STR.SFT", "Child", fill="F", confidence=0.8,
                       parent="ORG.ENT.GLB.STR.SFT"),
        ]
        ctx = synthesize_context(cells, [], "test")
        parent_child = [r for r in ctx.relationships if r.relation_type == "parent-child"]
        assert len(parent_child) == 1
        assert parent_child[0].source == "Parent"
        assert parent_child[0].target == "Child"

    def test_no_self_relationships(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "Self", fill="F", confidence=0.9,
                       connections=("ORG.ENT.GLB.STR.SFT",)),
        ]
        ctx = synthesize_context(cells, [], "test")
        assert len(ctx.relationships) == 0

    def test_deduplicates_relationships(self):
        cells = [
            _make_cell("A.ENT.GLB.STR.SFT", "X", fill="F", confidence=0.9,
                       connections=("B.ENT.GLB.STR.SFT",)),
            _make_cell("B.ENT.GLB.STR.SFT", "Y", fill="F", confidence=0.9,
                       connections=("A.ENT.GLB.STR.SFT",)),
        ]
        ctx = synthesize_context(cells, [], "test")
        connected = [r for r in ctx.relationships if r.relation_type == "connected"]
        assert len(connected) == 1


# ============================================================
# Assumption extraction
# ============================================================

class TestAssumptionExtraction:
    """Assumptions from dialogue and intent text."""

    def test_extracts_from_dialogue(self):
        ctx = synthesize_context([], SAMPLE_MESSAGES, "test")
        texts = [a.text for a in ctx.assumptions]
        assert any("assumes" in t.lower() for t in texts)

    def test_extracts_typically_pattern(self):
        ctx = synthesize_context([], SAMPLE_MESSAGES, "test")
        texts = [a.text for a in ctx.assumptions]
        assert any("typically" in t.lower() for t in texts)

    def test_extracts_from_intent(self):
        ctx = synthesize_context([], [], "The system should always validate inputs")
        assert len(ctx.assumptions) >= 1

    def test_categories_populated(self):
        ctx = synthesize_context([], SAMPLE_MESSAGES, "All users must have accounts")
        for a in ctx.assumptions:
            assert a.category in ("structural", "behavioral", "domain", "constraint")

    def test_capped_at_20(self):
        # Generate lots of assumption-triggering messages
        msgs = [_make_msg("Entity", f"The system assumes behavior {i}.") for i in range(30)]
        ctx = synthesize_context([], msgs, "test")
        assert len(ctx.assumptions) <= 20

    def test_no_assumptions_from_empty(self):
        ctx = synthesize_context([], [], "test")
        assert len(ctx.assumptions) == 0


# ============================================================
# Unknown extraction
# ============================================================

class TestUnknownExtraction:
    """Unknowns from empty cells and dialogue questions."""

    def test_extracts_dialogue_questions(self):
        ctx = synthesize_context([], SAMPLE_MESSAGES, "test")
        questions = [u.question for u in ctx.unknowns]
        assert any("SSO" in q for q in questions)

    def test_extracts_from_questioned_cells(self):
        cells = [_make_cell("APP.BHV.GLB.STR.SFT", "Routing", fill="Q", confidence=0.3)]
        ctx = synthesize_context(cells, [], "test")
        assert len(ctx.unknowns) >= 1

    def test_extracts_empty_cells_with_connections(self):
        cells = [
            _make_cell("ORG.ENT.GLB.STR.SFT", "Known", fill="F", confidence=0.9),
            _make_cell("APP.ENT.GLB.STR.SFT", "Unknown", fill="E", confidence=0.0,
                       connections=("ORG.ENT.GLB.STR.SFT",)),
        ]
        ctx = synthesize_context(cells, [], "test")
        questions = [u.question for u in ctx.unknowns]
        assert any("Unknown" in q for q in questions)

    def test_sorted_by_priority(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        if ctx.unknowns:
            priorities = [u.priority for u in ctx.unknowns]
            assert priorities == sorted(priorities, reverse=True)

    def test_capped_at_15(self):
        msgs = [_make_msg("Entity", f"What about scenario {i}?") for i in range(20)]
        ctx = synthesize_context([], msgs, "test")
        assert len(ctx.unknowns) <= 15


# ============================================================
# Vocabulary extraction
# ============================================================

class TestVocabularyExtraction:
    """Domain vocabulary from cells and dialogue."""

    def test_cell_primitives_are_vocabulary(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        assert "UserService" in ctx.vocabulary
        assert "Authentication" in ctx.vocabulary

    def test_camelcase_from_dialogue(self):
        msgs = [_make_msg("Entity", "The AuthController handles login via TokenProvider.")]
        ctx = synthesize_context([], msgs, "test")
        assert "AuthController" in ctx.vocabulary
        assert "TokenProvider" in ctx.vocabulary

    def test_sorted_alphabetically(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        assert list(ctx.vocabulary) == sorted(ctx.vocabulary)

    def test_empty_gives_empty(self):
        ctx = synthesize_context([], [], "test")
        assert len(ctx.vocabulary) == 0


# ============================================================
# Memory connections
# ============================================================

class TestMemoryConnections:
    """Memory connection extraction from prior results."""

    def test_extracts_from_memory(self):
        memories = [
            {"source": "recall", "text": "Previous discussion about auth patterns"},
            {"source": "episode", "content": "Built user management last week"},
        ]
        ctx = synthesize_context([], [], "test", memory_results=memories)
        assert len(ctx.memory_connections) == 2

    def test_none_memories_gives_empty(self):
        ctx = synthesize_context([], [], "test", memory_results=None)
        assert len(ctx.memory_connections) == 0

    def test_truncates_long_texts(self):
        memories = [{"source": "x", "text": "a" * 200}]
        ctx = synthesize_context([], [], "test", memory_results=memories)
        assert len(ctx.memory_connections[0]) <= 110  # "[x] " + 100 chars

    def test_capped_at_10(self):
        memories = [{"source": "x", "text": f"memory {i}"} for i in range(20)]
        ctx = synthesize_context([], [], "test", memory_results=memories)
        assert len(ctx.memory_connections) <= 10


# ============================================================
# Overall confidence
# ============================================================

class TestOverallConfidence:
    """ContextMap.confidence calculation."""

    def test_average_of_concepts(self):
        cells = [
            _make_cell("A.ENT.GLB.STR.SFT", "X", fill="F", confidence=0.8),
            _make_cell("B.ENT.GLB.STR.SFT", "Y", fill="F", confidence=0.6),
        ]
        ctx = synthesize_context(cells, [], "test")
        assert ctx.confidence == pytest.approx(0.7, abs=0.01)

    def test_zero_when_no_concepts(self):
        ctx = synthesize_context([], [], "test")
        assert ctx.confidence == 0.0


# ============================================================
# ContextMap frozen
# ============================================================

class TestContextMapFrozen:
    """ContextMap is a frozen dataclass."""

    def test_frozen(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        with pytest.raises(AttributeError):
            ctx.confidence = 0.5

    def test_original_intent_preserved(self):
        ctx = synthesize_context([], [], "Build a task manager")
        assert ctx.original_intent == "Build a task manager"


# ============================================================
# Serialization
# ============================================================

class TestContextMapSerialization:
    """context_map_to_dict() JSON serialization."""

    def test_round_trips(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        d = context_map_to_dict(ctx)
        assert isinstance(d, dict)
        assert d["original_intent"] == "test"
        assert isinstance(d["concepts"], list)
        assert isinstance(d["relationships"], list)
        assert isinstance(d["assumptions"], list)
        assert isinstance(d["unknowns"], list)
        assert isinstance(d["vocabulary"], list)
        assert isinstance(d["memory_connections"], list)
        assert isinstance(d["confidence"], float)

    def test_concept_keys(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        d = context_map_to_dict(ctx)
        if d["concepts"]:
            c = d["concepts"][0]
            assert "name" in c
            assert "description" in c
            assert "layer" in c
            assert "concern" in c
            assert "confidence" in c
            assert "source_postcode" in c

    def test_empty_serializes(self):
        ctx = synthesize_context([], [], "test")
        d = context_map_to_dict(ctx)
        assert d["concepts"] == []
        assert d["relationships"] == []


# ============================================================
# Format summary
# ============================================================

class TestFormatContextSummary:
    """format_context_summary() human-readable output."""

    def test_includes_intent(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "Build auth system")
        summary = format_context_summary(ctx)
        assert "Build auth system" in summary

    def test_includes_confidence(self):
        ctx = synthesize_context(SAMPLE_CELLS, SAMPLE_MESSAGES, "test")
        summary = format_context_summary(ctx)
        assert "Confidence:" in summary

    def test_includes_concepts_section(self):
        ctx = synthesize_context(SAMPLE_CELLS, [], "test")
        summary = format_context_summary(ctx)
        assert "Concepts" in summary

    def test_empty_context_still_formats(self):
        ctx = synthesize_context([], [], "test")
        summary = format_context_summary(ctx)
        assert "Context Map:" in summary
