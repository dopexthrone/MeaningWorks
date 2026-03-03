"""
Tests for Corpus Analysis - Phase 22: Corpus Pattern Synthesis.

Tests CorpusAnalyzer: component archetypes, relationship patterns,
constraint templates, vocabulary extraction, anti-pattern detection,
domain model building, cross-domain isomorphisms, and synthesis formatting.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List

from persistence.sqlite_corpus import SQLiteCorpus
from persistence.corpus_analysis import (
    CorpusAnalyzer,
    ComponentArchetype,
    RelationshipPattern,
    DomainModel,
)


# =============================================================================
# HELPERS
# =============================================================================


def _make_context_graph(
    domain: str = "booking",
    core_need: str = "booking system",
    verification_scores: Dict[str, int] = None,
    insights: List[str] = None,
) -> Dict[str, Any]:
    """Build a context graph for testing."""
    cg: Dict[str, Any] = {
        "known": {
            "intent": {
                "domain": domain,
                "core_need": core_need,
            }
        },
        "insights": insights or [
            f"Insight about {domain}",
            f"Reservation is the core entity for {domain}",
        ],
    }
    if verification_scores:
        cg["known"]["verification"] = {"scores": verification_scores}
    return cg


def _make_blueprint(
    components: List[Dict[str, Any]] = None,
    relationships: List[Dict[str, Any]] = None,
    constraints: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a blueprint for testing with full component dicts."""
    if components is None:
        components = [
            {"name": "Reservation", "type": "entity", "description": "A booking reservation", "derived_from": "input",
             "methods": [{"name": "create"}, {"name": "cancel"}]},
            {"name": "Scheduler", "type": "process", "description": "Handles scheduling logic", "derived_from": "input",
             "methods": [{"name": "find_available"}, {"name": "book"}]},
            {"name": "Client", "type": "entity", "description": "The booking client", "derived_from": "input",
             "methods": [{"name": "register"}]},
        ]
    if relationships is None:
        relationships = [
            {"from": "Reservation", "to": "Client", "type": "belongs_to", "description": "reservation belongs to client"},
            {"from": "Scheduler", "to": "Reservation", "type": "creates", "description": "scheduler creates reservation"},
        ]
    return {
        "components": components,
        "relationships": relationships,
        "constraints": constraints or [],
        "unresolved": [],
    }


def _populated_corpus(
    corpus: SQLiteCorpus,
    domain: str = "booking",
    n: int = 4,
    components: List[Dict[str, Any]] = None,
    relationships: List[Dict[str, Any]] = None,
    constraints: List[Dict[str, Any]] = None,
    verification_scores: Dict[str, int] = None,
    insights: List[str] = None,
    success: bool = True,
) -> SQLiteCorpus:
    """Store n compilations with overlapping components."""
    for i in range(n):
        corpus.store(
            input_text=f"Build a {domain} system variant {i}",
            context_graph=_make_context_graph(
                domain=domain,
                core_need=f"{domain} need {i}",
                verification_scores=verification_scores,
                insights=insights,
            ),
            blueprint=_make_blueprint(
                components=components,
                relationships=relationships,
                constraints=constraints,
            ),
            insights=[f"{domain} insight {i}"],
            success=success,
        )
    return corpus


@pytest.fixture
def corpus(tmp_path):
    """Fresh SQLiteCorpus instance."""
    return SQLiteCorpus(corpus_path=tmp_path / "test_corpus")


@pytest.fixture
def analyzer(corpus):
    """CorpusAnalyzer with fresh corpus."""
    return CorpusAnalyzer(corpus)


@pytest.fixture
def booking_corpus(corpus):
    """Corpus with 4 booking compilations."""
    return _populated_corpus(corpus, domain="booking", n=4)


@pytest.fixture
def booking_analyzer(booking_corpus):
    """Analyzer with 4 booking compilations."""
    return CorpusAnalyzer(booking_corpus)


# =============================================================================
# NORMALIZATION TESTS
# =============================================================================


class TestNormalization:
    """Tests for CorpusAnalyzer.normalize_name()."""

    def test_spaces_stripped(self):
        assert CorpusAnalyzer.normalize_name("Auth Service") == "authservice"

    def test_underscores_stripped(self):
        assert CorpusAnalyzer.normalize_name("auth_service") == "authservice"

    def test_hyphens_stripped(self):
        assert CorpusAnalyzer.normalize_name("auth-service") == "authservice"

    def test_case_insensitive(self):
        assert CorpusAnalyzer.normalize_name("AuthService") == "authservice"
        assert CorpusAnalyzer.normalize_name("AUTHSERVICE") == "authservice"

    def test_all_variants_normalize_same(self):
        variants = ["Auth Service", "AuthService", "auth_service", "auth-service", "authService"]
        normalized = {CorpusAnalyzer.normalize_name(v) for v in variants}
        assert len(normalized) == 1
        assert normalized == {"authservice"}

    def test_empty_string(self):
        assert CorpusAnalyzer.normalize_name("") == ""

    def test_single_char(self):
        assert CorpusAnalyzer.normalize_name("A") == "a"


# =============================================================================
# ARCHETYPE EXTRACTION TESTS
# =============================================================================


class TestArchetypeExtraction:
    """Tests for extract_archetypes()."""

    def test_archetypes_from_sufficient_samples(self, booking_analyzer):
        archetypes = booking_analyzer.extract_archetypes("booking", min_samples=3)
        assert len(archetypes) > 0
        names = [a.canonical_name for a in archetypes]
        assert "Reservation" in names
        assert "Scheduler" in names

    def test_correct_frequencies(self, booking_analyzer):
        archetypes = booking_analyzer.extract_archetypes("booking", min_frequency=0.0, min_samples=3)
        for arch in archetypes:
            # All 4 compilations have the same components, so frequency=1.0
            assert arch.frequency == 1.0

    def test_common_methods_extracted(self, booking_analyzer):
        archetypes = booking_analyzer.extract_archetypes("booking", min_samples=3)
        reservation = next(a for a in archetypes if a.canonical_name == "Reservation")
        assert "create" in reservation.common_methods
        assert "cancel" in reservation.common_methods

    def test_common_relationships_extracted(self, booking_analyzer):
        archetypes = booking_analyzer.extract_archetypes("booking", min_samples=3)
        reservation = next(a for a in archetypes if a.canonical_name == "Reservation")
        rel_targets = [r["to"] for r in reservation.common_relationships]
        assert "Client" in rel_targets

    def test_below_min_samples_returns_empty(self, analyzer, corpus):
        # Only 2 compilations — below default min_samples=3
        _populated_corpus(corpus, domain="tiny", n=2)
        result = analyzer.extract_archetypes("tiny", min_samples=3)
        assert result == []

    def test_mixed_domains_stay_separated(self, corpus):
        _populated_corpus(corpus, domain="booking", n=4)
        _populated_corpus(corpus, domain="healthcare", n=4, components=[
            {"name": "Patient", "type": "entity", "description": "A patient", "derived_from": "input", "methods": []},
            {"name": "Doctor", "type": "entity", "description": "A doctor", "derived_from": "input", "methods": []},
        ], relationships=[])
        analyzer = CorpusAnalyzer(corpus)
        booking_archetypes = analyzer.extract_archetypes("booking", min_samples=3)
        healthcare_archetypes = analyzer.extract_archetypes("healthcare", min_samples=3)
        booking_names = {a.canonical_name for a in booking_archetypes}
        healthcare_names = {a.canonical_name for a in healthcare_archetypes}
        assert "Reservation" in booking_names
        assert "Patient" in healthcare_names
        assert "Patient" not in booking_names
        assert "Reservation" not in healthcare_names

    def test_canonical_name_most_frequent_variant(self, corpus):
        """When different compilations use different names for the same concept,
        the most frequent variant becomes canonical."""
        for i in range(2):
            corpus.store(
                input_text=f"Build booking v{i}",
                context_graph=_make_context_graph("booking"),
                blueprint=_make_blueprint(components=[
                    {"name": "Auth Service", "type": "entity", "description": "auth", "derived_from": "input", "methods": []},
                ], relationships=[]),
                insights=["test"],
                success=True,
            )
        # One compilation uses different variant
        corpus.store(
            input_text="Build booking v2",
            context_graph=_make_context_graph("booking"),
            blueprint=_make_blueprint(components=[
                {"name": "AuthService", "type": "entity", "description": "auth", "derived_from": "input", "methods": []},
            ], relationships=[]),
            insights=["test"],
            success=True,
        )
        analyzer = CorpusAnalyzer(corpus)
        archetypes = analyzer.extract_archetypes("booking", min_frequency=0.0, min_samples=3)
        auth = next(a for a in archetypes if CorpusAnalyzer.normalize_name(a.canonical_name) == "authservice")
        # "Auth Service" appeared 2x, "AuthService" 1x → canonical should be "Auth Service"
        assert auth.canonical_name == "Auth Service"
        assert set(auth.variants) == {"Auth Service", "AuthService"}

    def test_common_constraints_extracted(self, corpus):
        constraints = [
            {"type": "LENGTH", "target": "Reservation", "description": "max 500 chars"},
        ]
        _populated_corpus(corpus, domain="booking", n=4, constraints=constraints)
        analyzer = CorpusAnalyzer(corpus)
        archetypes = analyzer.extract_archetypes("booking", min_samples=3)
        reservation = next(a for a in archetypes if a.canonical_name == "Reservation")
        assert len(reservation.common_constraints) > 0
        assert reservation.common_constraints[0]["type"] == "LENGTH"

    def test_source_ids_tracked(self, booking_analyzer):
        archetypes = booking_analyzer.extract_archetypes("booking", min_samples=3)
        for arch in archetypes:
            assert len(arch.source_ids) > 0
            # Each source_id should be a non-empty string
            for sid in arch.source_ids:
                assert isinstance(sid, str)
                assert len(sid) > 0


# =============================================================================
# RELATIONSHIP PATTERN TESTS
# =============================================================================


class TestRelationshipPatterns:
    """Tests for extract_relationship_patterns()."""

    def test_recurring_triplets_detected(self, booking_analyzer):
        patterns = booking_analyzer.extract_relationship_patterns("booking", min_frequency=0.0, min_samples=3)
        # Should find at least the two relationships in our blueprint
        pair_patterns = [p for p in patterns if len(p.components) == 2]
        assert len(pair_patterns) >= 2

    def test_correct_frequency(self, booking_analyzer):
        patterns = booking_analyzer.extract_relationship_patterns("booking", min_frequency=0.0, min_samples=3)
        pair_patterns = [p for p in patterns if len(p.components) == 2]
        for p in pair_patterns:
            # All 4 compilations have same relationships → frequency=1.0
            assert p.frequency == 1.0

    def test_chains_detected(self, booking_analyzer):
        """Scheduler→Reservation and Reservation→Client should form a chain."""
        patterns = booking_analyzer.extract_relationship_patterns("booking", min_frequency=0.0, min_samples=3)
        chain_patterns = [p for p in patterns if len(p.components) == 3]
        assert len(chain_patterns) > 0
        # Should find Scheduler→Reservation→Client chain
        chain_names = [set(p.components) for p in chain_patterns]
        assert any({"Scheduler", "Reservation", "Client"} == s for s in chain_names)

    def test_source_ids_tracked(self, booking_analyzer):
        patterns = booking_analyzer.extract_relationship_patterns("booking", min_frequency=0.0, min_samples=3)
        for p in patterns:
            assert len(p.source_ids) > 0

    def test_below_threshold_filtered(self, corpus):
        """Relationships below min_frequency should be filtered out."""
        # 3 compilations with common rel, 1 without
        for i in range(3):
            corpus.store(
                input_text=f"Build rel test v{i}",
                context_graph=_make_context_graph("reltest"),
                blueprint=_make_blueprint(
                    components=[
                        {"name": "A", "type": "entity", "description": "a", "derived_from": "input", "methods": []},
                        {"name": "B", "type": "entity", "description": "b", "derived_from": "input", "methods": []},
                    ],
                    relationships=[
                        {"from": "A", "to": "B", "type": "depends_on", "description": "test"},
                    ],
                ),
                insights=["test"],
                success=True,
            )
        # 4th compilation with different relationship
        corpus.store(
            input_text="Build rel test v3",
            context_graph=_make_context_graph("reltest"),
            blueprint=_make_blueprint(
                components=[
                    {"name": "C", "type": "entity", "description": "c", "derived_from": "input", "methods": []},
                    {"name": "D", "type": "entity", "description": "d", "derived_from": "input", "methods": []},
                ],
                relationships=[
                    {"from": "C", "to": "D", "type": "triggers", "description": "test"},
                ],
            ),
            insights=["test"],
            success=True,
        )
        analyzer = CorpusAnalyzer(corpus)
        # With high threshold, C→D (25%) should be filtered
        patterns = analyzer.extract_relationship_patterns("reltest", min_frequency=0.5, min_samples=3)
        pair_patterns = [p for p in patterns if len(p.components) == 2]
        pair_names = [(p.components[0], p.components[1]) for p in pair_patterns]
        assert ("A", "B") in pair_names
        assert ("C", "D") not in pair_names


# =============================================================================
# CONSTRAINT TEMPLATE TESTS
# =============================================================================


class TestConstraintTemplates:
    """Tests for extract_constraint_templates()."""

    def test_recurring_constraints_grouped(self, corpus):
        constraints = [
            {"type": "LENGTH", "target": "Reservation", "description": "max 500 chars"},
            {"type": "RANGE", "target": "Client", "description": "age 18-120"},
        ]
        _populated_corpus(corpus, domain="booking", n=4, constraints=constraints)
        analyzer = CorpusAnalyzer(corpus)
        templates = analyzer.extract_constraint_templates("booking", min_samples=3)
        assert len(templates) >= 2
        types = [t["type"] for t in templates]
        assert "LENGTH" in types
        assert "RANGE" in types

    def test_frequency_correct(self, corpus):
        constraints = [
            {"type": "LENGTH", "target": "Reservation", "description": "max 500"},
        ]
        _populated_corpus(corpus, domain="booking", n=4, constraints=constraints)
        analyzer = CorpusAnalyzer(corpus)
        templates = analyzer.extract_constraint_templates("booking", min_samples=3)
        length_template = next(t for t in templates if t["type"] == "LENGTH")
        assert length_template["frequency"] == 1.0

    def test_source_ids_present(self, corpus):
        constraints = [
            {"type": "LENGTH", "target": "Reservation", "description": "max 500"},
        ]
        _populated_corpus(corpus, domain="booking", n=4, constraints=constraints)
        analyzer = CorpusAnalyzer(corpus)
        templates = analyzer.extract_constraint_templates("booking", min_samples=3)
        for t in templates:
            assert "source_ids" in t
            assert len(t["source_ids"]) > 0


# =============================================================================
# VOCABULARY TESTS
# =============================================================================


class TestVocabulary:
    """Tests for extract_vocabulary()."""

    def test_terms_extracted_from_insights(self, corpus):
        insights = [
            "Reservation is the core entity",
            "Scheduling handles availability",
        ]
        _populated_corpus(corpus, domain="booking", n=4, insights=insights)
        analyzer = CorpusAnalyzer(corpus)
        vocab = analyzer.extract_vocabulary("booking", min_samples=3)
        # "Reservation" and "Scheduling" are capitalized words ≥4 chars
        assert "reservation" in vocab or "scheduling" in vocab

    def test_frequency_tracked(self, corpus):
        insights = ["Reservation is core"]
        _populated_corpus(corpus, domain="booking", n=4, insights=insights)
        analyzer = CorpusAnalyzer(corpus)
        vocab = analyzer.extract_vocabulary("booking", min_samples=3)
        # Component names should appear in vocabulary from descriptions
        if "reservation" in vocab:
            assert vocab["reservation"]["frequency"] >= 2

    def test_domain_scoped(self, corpus):
        _populated_corpus(corpus, domain="booking", n=4)
        _populated_corpus(corpus, domain="healthcare", n=4, components=[
            {"name": "Patient", "type": "entity", "description": "A healthcare patient record", "derived_from": "input", "methods": []},
        ], relationships=[], insights=["Patient records must be HIPAA compliant"])
        analyzer = CorpusAnalyzer(corpus)
        booking_vocab = analyzer.extract_vocabulary("booking", min_samples=3)
        healthcare_vocab = analyzer.extract_vocabulary("healthcare", min_samples=3)
        # "patient" should only appear in healthcare domain
        if "patient" in healthcare_vocab:
            assert "patient" not in booking_vocab


# =============================================================================
# ANTI-PATTERN TESTS
# =============================================================================


class TestAntiPatterns:
    """Tests for detect_anti_patterns()."""

    def test_low_verification_components_detected(self, corpus):
        """Hollow components in low-verification compilations should be flagged."""
        hollow_components = [
            {"name": "EmptyThing", "type": "entity", "description": "", "derived_from": "input", "methods": []},
        ]
        for i in range(5):
            corpus.store(
                input_text=f"Build hollow system v{i}",
                context_graph=_make_context_graph(
                    "hollow",
                    verification_scores={"completeness": 50},
                ),
                blueprint=_make_blueprint(components=hollow_components, relationships=[]),
                insights=["test"],
                success=True,
            )
        analyzer = CorpusAnalyzer(corpus)
        anti_patterns = analyzer.detect_anti_patterns(min_samples=5)
        descriptions = [ap["description"] for ap in anti_patterns]
        assert "hollow_component" in descriptions

    def test_missing_description_flagged(self, corpus):
        components = [
            {"name": "NoDesc", "type": "entity", "description": "", "derived_from": "input", "methods": [{"name": "do"}]},
        ]
        for i in range(5):
            corpus.store(
                input_text=f"Build nodesc system v{i}",
                context_graph=_make_context_graph(
                    "nodesc",
                    verification_scores={"completeness": 40},
                ),
                blueprint=_make_blueprint(components=components, relationships=[]),
                insights=["test"],
                success=True,
            )
        analyzer = CorpusAnalyzer(corpus)
        anti_patterns = analyzer.detect_anti_patterns(min_samples=5)
        descriptions = [ap["description"] for ap in anti_patterns]
        assert "missing_description" in descriptions

    def test_minimum_samples_respected(self, corpus):
        """With fewer than min_samples, should return empty."""
        corpus.store(
            input_text="Single hollow",
            context_graph=_make_context_graph("hollow", verification_scores={"completeness": 30}),
            blueprint=_make_blueprint(components=[
                {"name": "X", "type": "entity", "description": "", "derived_from": "input", "methods": []},
            ], relationships=[]),
            insights=["test"],
            success=True,
        )
        analyzer = CorpusAnalyzer(corpus)
        assert analyzer.detect_anti_patterns(min_samples=5) == []


# =============================================================================
# DOMAIN MODEL TESTS
# =============================================================================


class TestDomainModel:
    """Tests for build_domain_model()."""

    def test_full_build_returns_all_sub_analyses(self, corpus):
        constraints = [
            {"type": "LENGTH", "target": "Reservation", "description": "max 500"},
        ]
        _populated_corpus(corpus, domain="booking", n=5, constraints=constraints)
        analyzer = CorpusAnalyzer(corpus)
        model = analyzer.build_domain_model("booking", min_samples=3)
        assert model is not None
        assert model.domain == "booking"
        assert model.sample_size >= 3
        assert len(model.archetypes) > 0
        assert len(model.constraint_templates) > 0
        assert isinstance(model.vocabulary, dict)
        assert isinstance(model.anti_patterns, list)

    def test_provenance_has_stratum_2(self, booking_corpus):
        analyzer = CorpusAnalyzer(booking_corpus)
        model = analyzer.build_domain_model("booking", min_samples=3)
        assert model is not None
        assert model.provenance["stratum"] == 2
        assert len(model.provenance["source_ids"]) > 0

    def test_returns_none_below_min_samples(self, corpus):
        _populated_corpus(corpus, domain="tiny", n=2)
        analyzer = CorpusAnalyzer(corpus)
        result = analyzer.build_domain_model("tiny", min_samples=3)
        assert result is None


# =============================================================================
# FORMAT SECTION TESTS
# =============================================================================


class TestFormatSection:
    """Tests for format_corpus_patterns_section()."""

    def test_non_empty_model_produces_formatted_string(self, booking_corpus):
        analyzer = CorpusAnalyzer(booking_corpus)
        model = analyzer.build_domain_model("booking", min_samples=3)
        assert model is not None
        formatted = analyzer.format_corpus_patterns_section(model)
        assert formatted is not None
        assert "DOMAIN: booking" in formatted
        assert "COMPONENT ARCHETYPES" in formatted
        assert "Reservation" in formatted

    def test_empty_model_returns_none(self, analyzer):
        empty_model = DomainModel(domain="empty", sample_size=0)
        result = analyzer.format_corpus_patterns_section(empty_model)
        assert result is None

    def test_includes_methods_and_relationships(self, booking_corpus):
        analyzer = CorpusAnalyzer(booking_corpus)
        model = analyzer.build_domain_model("booking", min_samples=3)
        formatted = analyzer.format_corpus_patterns_section(model)
        assert "common methods:" in formatted
        assert "create()" in formatted
        assert "common relationships:" in formatted

    def test_includes_chain_patterns(self, booking_corpus):
        analyzer = CorpusAnalyzer(booking_corpus)
        model = analyzer.build_domain_model("booking", min_samples=3)
        formatted = analyzer.format_corpus_patterns_section(model)
        assert "RELATIONSHIP PATTERNS:" in formatted
        assert "->" in formatted


# =============================================================================
# ISOMORPHISM TESTS
# =============================================================================


class TestIsomorphisms:
    """Tests for find_isomorphisms()."""

    def test_shared_structure_detected(self, corpus):
        """Two domains with an overlapping component should show overlap."""
        # Both domains have a "Scheduler" component
        _populated_corpus(corpus, domain="booking", n=4)
        _populated_corpus(corpus, domain="logistics", n=4, components=[
            {"name": "Scheduler", "type": "process", "description": "Logistics scheduling", "derived_from": "input", "methods": [{"name": "dispatch"}]},
            {"name": "Package", "type": "entity", "description": "A package to ship", "derived_from": "input", "methods": []},
        ], relationships=[
            {"from": "Scheduler", "to": "Package", "type": "manages", "description": "scheduler manages packages"},
        ])
        analyzer = CorpusAnalyzer(corpus)
        result = analyzer.find_isomorphisms("booking", "logistics")
        assert len(result["shared_archetypes"]) > 0
        assert result["overlap_score"] > 0.0
        shared_names = [s["name"] for s in result["shared_archetypes"]]
        assert "Scheduler" in shared_names

    def test_no_overlap_returns_empty(self, corpus):
        _populated_corpus(corpus, domain="booking", n=4)
        _populated_corpus(corpus, domain="completely_different", n=4, components=[
            {"name": "Zygote", "type": "entity", "description": "unique", "derived_from": "input", "methods": []},
            {"name": "Xenomorph", "type": "entity", "description": "unique", "derived_from": "input", "methods": []},
        ], relationships=[])
        analyzer = CorpusAnalyzer(corpus)
        result = analyzer.find_isomorphisms("booking", "completely_different")
        assert len(result["shared_archetypes"]) == 0
        assert result["overlap_score"] == 0.0


# =============================================================================
# INTEGRATION: FAILED COMPILATIONS EXCLUDED
# =============================================================================


class TestFailedCompilationsExcluded:
    """Verify that failed compilations are excluded from pattern extraction."""

    def test_failed_compilations_not_in_archetypes(self, corpus):
        # 3 successful + 1 failed
        _populated_corpus(corpus, domain="booking", n=3)
        corpus.store(
            input_text="Build a broken booking system",
            context_graph=_make_context_graph("booking"),
            blueprint=_make_blueprint(components=[
                {"name": "BrokenThing", "type": "entity", "description": "broken", "derived_from": "input", "methods": []},
            ], relationships=[]),
            insights=["broken"],
            success=False,
        )
        analyzer = CorpusAnalyzer(corpus)
        archetypes = analyzer.extract_archetypes("booking", min_frequency=0.0, min_samples=3)
        names = [a.canonical_name for a in archetypes]
        assert "BrokenThing" not in names
