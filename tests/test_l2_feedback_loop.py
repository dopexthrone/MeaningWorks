"""
Tests for L2 Corpus Feedback Loop.

Covers:
- corpus_feedback persistence in SQLite (migration, store, query)
- Adoption-weighted pattern extraction
- Pattern health summarization
- Context synthesis wiring
"""

import json
import tempfile
from pathlib import Path

import pytest

from persistence.sqlite_corpus import SQLiteCorpus
from persistence.corpus_analysis import CorpusAnalyzer
from mother.context import ContextData, synthesize_situation


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def corpus(tmp_path):
    """Fresh SQLiteCorpus in a temp directory."""
    return SQLiteCorpus(corpus_path=tmp_path)


def _store_compilation(
    corpus,
    input_text,
    domain="software",
    components=None,
    relationships=None,
    constraints=None,
    verification_score=None,
    corpus_feedback=None,
    success=True,
):
    """Helper to store a compilation with optional corpus_feedback."""
    if components is None:
        components = [{"name": "Auth", "type": "entity", "methods": [{"name": "login"}]}]
    blueprint = {
        "components": components,
        "relationships": relationships or [],
        "constraints": constraints or [],
    }
    context_graph = {
        "known": {"intent": {"domain": domain}},
        "insights": [],
    }
    feedback_json = json.dumps(corpus_feedback) if corpus_feedback else None
    return corpus.store(
        input_text=input_text,
        context_graph=context_graph,
        blueprint=blueprint,
        insights=["test insight"],
        success=success,
        verification_score=verification_score,
        corpus_feedback=feedback_json,
    )


# =============================================================================
# LAYER 1: PERSISTENCE
# =============================================================================


class TestCorpusFeedbackPersistence:
    """Test that corpus_feedback is stored and queryable."""

    def test_migration_adds_column(self, corpus):
        """corpus_feedback column exists after init."""
        conn = corpus._get_connection()
        try:
            row = conn.execute(
                "PRAGMA table_info(compilations)"
            ).fetchall()
            col_names = [r["name"] for r in row]
            assert "corpus_feedback" in col_names
        finally:
            conn.close()

    def test_store_without_feedback(self, corpus):
        """Storing without corpus_feedback works (backward compat)."""
        record = _store_compilation(corpus, "test no feedback")
        assert record.id
        assert record.success

    def test_store_with_feedback(self, corpus):
        """Storing with corpus_feedback persists it."""
        feedback = {
            "suggestion_hit_rate": 0.75,
            "suggestions_used": ["Auth", "Database"],
            "suggestions_ignored": ["Logger"],
            "corpus_influence": "strong",
        }
        _store_compilation(
            corpus, "test with feedback",
            corpus_feedback=feedback,
            verification_score=85.0,
        )

        # Verify via direct SQL
        conn = corpus._get_connection()
        try:
            row = conn.execute(
                "SELECT corpus_feedback FROM compilations"
            ).fetchone()
            assert row["corpus_feedback"] is not None
            parsed = json.loads(row["corpus_feedback"])
            assert parsed["suggestion_hit_rate"] == 0.75
            assert "Auth" in parsed["suggestions_used"]
        finally:
            conn.close()

    def test_get_adoption_data_empty(self, corpus):
        """get_adoption_data returns empty list when no feedback stored."""
        result = corpus.get_adoption_data("software")
        assert result == []

    def test_get_adoption_data_filters_by_domain(self, corpus):
        """get_adoption_data only returns data for the requested domain."""
        feedback = {
            "suggestions_used": ["Auth"],
            "suggestions_ignored": [],
            "corpus_influence": "strong",
        }
        _store_compilation(
            corpus, "software comp", domain="software",
            corpus_feedback=feedback, verification_score=80.0,
        )
        _store_compilation(
            corpus, "process comp", domain="process",
            corpus_feedback={"suggestions_used": ["Workflow"], "suggestions_ignored": [], "corpus_influence": "strong"},
            verification_score=70.0,
        )

        software_data = corpus.get_adoption_data("software")
        assert len(software_data) == 1
        assert "Auth" in software_data[0]["feedback"]["suggestions_used"]

        process_data = corpus.get_adoption_data("process")
        assert len(process_data) == 1
        assert "Workflow" in process_data[0]["feedback"]["suggestions_used"]

    def test_get_adoption_data_returns_verification_score(self, corpus):
        """get_adoption_data includes verification_score."""
        feedback = {
            "suggestions_used": ["Auth"],
            "suggestions_ignored": [],
            "corpus_influence": "strong",
        }
        _store_compilation(
            corpus, "scored comp", domain="software",
            corpus_feedback=feedback, verification_score=92.5,
        )
        data = corpus.get_adoption_data("software")
        assert len(data) == 1
        assert data[0]["verification_score"] == 92.5

    def test_get_adoption_data_skips_null_feedback(self, corpus):
        """Compilations without corpus_feedback are excluded."""
        _store_compilation(corpus, "no feedback comp", domain="software")
        feedback = {
            "suggestions_used": ["Auth"],
            "suggestions_ignored": [],
            "corpus_influence": "partial",
        }
        _store_compilation(
            corpus, "with feedback comp", domain="software",
            corpus_feedback=feedback, verification_score=75.0,
        )
        data = corpus.get_adoption_data("software")
        assert len(data) == 1

    def test_get_adoption_data_handles_malformed_json(self, corpus):
        """Malformed JSON in corpus_feedback is skipped gracefully."""
        # Store valid feedback
        _store_compilation(
            corpus, "valid comp", domain="software",
            corpus_feedback={"suggestions_used": ["X"], "suggestions_ignored": [], "corpus_influence": "partial"},
            verification_score=70.0,
        )
        # Manually insert malformed JSON
        conn = corpus._get_connection()
        try:
            conn.execute(
                "UPDATE compilations SET corpus_feedback = '{bad json' WHERE id = (SELECT id FROM compilations LIMIT 1)"
            )
            conn.commit()
        finally:
            conn.close()
        # Should not raise, returns empty since the only record is malformed
        data = corpus.get_adoption_data("software")
        assert isinstance(data, list)


# =============================================================================
# LAYER 2: ADOPTION WEIGHTS
# =============================================================================


class TestAdoptionWeights:
    """Test adoption-weighted pattern extraction."""

    def test_compute_adoption_weights_empty(self, corpus):
        """No feedback data → empty weights."""
        analyzer = CorpusAnalyzer(corpus)
        weights = analyzer.compute_adoption_weights("software")
        assert weights == {}

    def test_compute_adoption_weights_strong_signal(self, corpus):
        """Archetype adopted >= 70% with high trust → 1.5x."""
        analyzer = CorpusAnalyzer(corpus)
        # Store 4 compilations where Auth is used 3/4 times with high trust
        for i in range(3):
            _store_compilation(
                corpus, f"comp {i} auth used",
                domain="software",
                corpus_feedback={
                    "suggestions_used": ["Auth"],
                    "suggestions_ignored": [],
                    "corpus_influence": "strong",
                },
                verification_score=85.0,
            )
        _store_compilation(
            corpus, "comp 3 auth ignored",
            domain="software",
            corpus_feedback={
                "suggestions_used": [],
                "suggestions_ignored": ["Auth"],
                "corpus_influence": "none",
            },
            verification_score=60.0,
        )
        weights = analyzer.compute_adoption_weights("software")
        assert analyzer.normalize_name("Auth") in weights
        assert weights[analyzer.normalize_name("Auth")] == 1.5

    def test_compute_adoption_weights_ignored_pattern(self, corpus):
        """Archetype ignored >= 70% with 3+ suggestions → 0.5x."""
        analyzer = CorpusAnalyzer(corpus)
        # Store 4 compilations where Logger is ignored 3/4 times
        for i in range(3):
            _store_compilation(
                corpus, f"comp {i} logger ignored",
                domain="software",
                corpus_feedback={
                    "suggestions_used": [],
                    "suggestions_ignored": ["Logger"],
                    "corpus_influence": "none",
                },
                verification_score=70.0,
            )
        _store_compilation(
            corpus, "comp 3 logger used",
            domain="software",
            corpus_feedback={
                "suggestions_used": ["Logger"],
                "suggestions_ignored": [],
                "corpus_influence": "partial",
            },
            verification_score=50.0,
        )
        weights = analyzer.compute_adoption_weights("software")
        norm_logger = analyzer.normalize_name("Logger")
        assert norm_logger in weights
        assert weights[norm_logger] == 0.5

    def test_compute_adoption_weights_neutral(self, corpus):
        """Archetype adopted 40-69% → 1.0x."""
        analyzer = CorpusAnalyzer(corpus)
        # 2 used, 2 ignored → 50% adoption
        for i in range(2):
            _store_compilation(
                corpus, f"comp {i} used",
                domain="software",
                corpus_feedback={
                    "suggestions_used": ["Cache"],
                    "suggestions_ignored": [],
                    "corpus_influence": "partial",
                },
                verification_score=60.0,
            )
        for i in range(2):
            _store_compilation(
                corpus, f"comp {i} ignored",
                domain="software",
                corpus_feedback={
                    "suggestions_used": [],
                    "suggestions_ignored": ["Cache"],
                    "corpus_influence": "none",
                },
                verification_score=60.0,
            )
        weights = analyzer.compute_adoption_weights("software")
        norm_cache = analyzer.normalize_name("Cache")
        assert norm_cache in weights
        assert weights[norm_cache] == 1.0

    def test_adoption_weights_applied_to_archetypes(self, corpus):
        """Adoption weights modify archetype frequencies in extract_archetypes."""
        analyzer = CorpusAnalyzer(corpus)
        components = [
            {"name": "Auth", "type": "entity", "methods": [{"name": "login"}]},
            {"name": "Logger", "type": "entity", "methods": [{"name": "log"}]},
        ]
        # Store 3 compilations (min_samples) with both components
        for i in range(3):
            _store_compilation(
                corpus, f"archetype comp {i}",
                domain="software",
                components=components,
                verification_score=80.0,
            )
        # Without adoption weights: both at freq ~1.0
        archetypes_neutral = analyzer.extract_archetypes("software", min_frequency=0.3)
        names_neutral = {a.canonical_name for a in archetypes_neutral}
        assert "Auth" in names_neutral
        assert "Logger" in names_neutral

        # With adoption weights: Logger at 0.5x should drop below threshold at 0.6
        custom_weights = {analyzer.normalize_name("Logger"): 0.5}
        archetypes_weighted = analyzer.extract_archetypes(
            "software", min_frequency=0.6, adoption_weights=custom_weights,
        )
        names_weighted = {a.canonical_name for a in archetypes_weighted}
        assert "Auth" in names_weighted
        assert "Logger" not in names_weighted

    def test_build_domain_model_uses_adoption_weights(self, corpus):
        """build_domain_model integrates adoption weights transparently."""
        analyzer = CorpusAnalyzer(corpus)
        components = [
            {"name": "Auth", "type": "entity", "methods": [{"name": "login"}]},
        ]
        for i in range(3):
            _store_compilation(
                corpus, f"domain model comp {i}",
                domain="software",
                components=components,
                verification_score=80.0,
            )
        model = analyzer.build_domain_model("software")
        assert model is not None
        assert model.domain == "software"


# =============================================================================
# LAYER 3: PATTERN HEALTH
# =============================================================================


class TestPatternHealth:
    """Test pattern health summarization."""

    def test_summarize_pattern_health_insufficient_data(self, corpus):
        """< 3 feedback records → empty string."""
        analyzer = CorpusAnalyzer(corpus)
        result = analyzer.summarize_pattern_health("software")
        assert result == ""

    def test_summarize_pattern_health_insufficient_per_archetype(self, corpus):
        """Archetypes with < 3 suggestions → excluded from summary."""
        analyzer = CorpusAnalyzer(corpus)
        # 3 records but each archetype only suggested once
        for i in range(3):
            _store_compilation(
                corpus, f"sparse comp {i}",
                domain="software",
                corpus_feedback={
                    "suggestions_used": [f"Unique{i}"],
                    "suggestions_ignored": [],
                    "corpus_influence": "partial",
                },
                verification_score=80.0,
            )
        result = analyzer.summarize_pattern_health("software")
        assert result == ""

    def test_summarize_pattern_health_strong_pattern(self, corpus):
        """Archetype adopted >= 60% with high trust → labeled Strong."""
        analyzer = CorpusAnalyzer(corpus)
        for i in range(4):
            _store_compilation(
                corpus, f"strong comp {i}",
                domain="software",
                corpus_feedback={
                    "suggestions_used": ["Auth"],
                    "suggestions_ignored": [],
                    "corpus_influence": "strong",
                },
                verification_score=85.0,
            )
        result = analyzer.summarize_pattern_health("software")
        assert "Strong:" in result
        assert "Auth" in result

    def test_summarize_pattern_health_declining_pattern(self, corpus):
        """Archetype adopted < 30% with 3+ suggestions → labeled Declining."""
        analyzer = CorpusAnalyzer(corpus)
        for i in range(4):
            _store_compilation(
                corpus, f"declining comp {i}",
                domain="software",
                corpus_feedback={
                    "suggestions_used": [],
                    "suggestions_ignored": ["Logger"],
                    "corpus_influence": "none",
                },
                verification_score=70.0,
            )
        result = analyzer.summarize_pattern_health("software")
        assert "Declining:" in result
        assert "Logger" in result

    def test_summarize_pattern_health_mixed(self, corpus):
        """Both strong and declining patterns in same summary."""
        analyzer = CorpusAnalyzer(corpus)
        for i in range(4):
            _store_compilation(
                corpus, f"mixed comp {i}",
                domain="software",
                corpus_feedback={
                    "suggestions_used": ["Auth"],
                    "suggestions_ignored": ["Logger"],
                    "corpus_influence": "partial",
                },
                verification_score=80.0,
            )
        result = analyzer.summarize_pattern_health("software")
        assert "Strong:" in result
        assert "Auth" in result
        assert "Declining:" in result
        assert "Logger" in result


# =============================================================================
# CONTEXT SYNTHESIS WIRING
# =============================================================================


class TestContextWiring:
    """Test that pattern health flows through to context synthesis."""

    def test_context_data_has_field(self):
        """ContextData accepts corpus_pattern_health."""
        data = ContextData(corpus_pattern_health="Strong: Auth. Declining: Logger.")
        assert data.corpus_pattern_health == "Strong: Auth. Declining: Logger."

    def test_context_data_default_empty(self):
        """corpus_pattern_health defaults to empty string."""
        data = ContextData()
        assert data.corpus_pattern_health == ""

    def test_synthesize_situation_includes_pattern_health(self):
        """Pattern health renders in situation when corpus has data."""
        data = ContextData(
            corpus_total=10,
            corpus_success_rate=0.9,
            corpus_pattern_health="Strong: Auth, Gateway. Declining: Logger.",
        )
        situation = synthesize_situation(data)
        assert "Patterns: Strong: Auth, Gateway. Declining: Logger." in situation

    def test_synthesize_situation_omits_empty_pattern_health(self):
        """Empty pattern health produces no extra line."""
        data = ContextData(
            corpus_total=5,
            corpus_success_rate=0.8,
            corpus_pattern_health="",
        )
        situation = synthesize_situation(data)
        assert "Patterns:" not in situation

    def test_synthesize_situation_no_corpus_no_patterns(self):
        """Zero compilations → no corpus section → no patterns line."""
        data = ContextData(corpus_total=0)
        situation = synthesize_situation(data)
        assert "Patterns:" not in situation


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompatibility:
    """Ensure existing code paths are unaffected."""

    def test_store_existing_api_unchanged(self, corpus):
        """Existing store() calls without corpus_feedback still work."""
        record = corpus.store(
            input_text="legacy test",
            context_graph={"known": {"intent": {"domain": "software"}}},
            blueprint={"components": [{"name": "A"}]},
            insights=["i1"],
            success=True,
            provider="claude",
            model="sonnet",
            verification_score=75.0,
        )
        assert record.success
        # get_adoption_data returns empty since no feedback stored
        data = corpus.get_adoption_data("software")
        assert data == []

    def test_extract_archetypes_without_weights(self, corpus):
        """extract_archetypes works without adoption_weights param."""
        analyzer = CorpusAnalyzer(corpus)
        components = [{"name": "X", "type": "entity", "methods": []}]
        for i in range(3):
            _store_compilation(
                corpus, f"compat comp {i}",
                domain="software",
                components=components,
                verification_score=70.0,
            )
        archetypes = analyzer.extract_archetypes("software")
        assert isinstance(archetypes, list)

    def test_no_feedback_neutral_weight(self, corpus):
        """Compilations without feedback get neutral 1.0x weight (not penalized)."""
        analyzer = CorpusAnalyzer(corpus)
        weights = analyzer.compute_adoption_weights("software")
        # No data → empty dict → default 1.0 in extract_archetypes
        assert weights == {}
