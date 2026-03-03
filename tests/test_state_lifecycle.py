"""
Tests for State Lifecycle, Corpus Quality & Stage Handoff.

Covers:
- Part 1: Context Graph Compaction (CompactionSpec, compact_known, to_context_graph(compact=True/False))
- Part 2: Corpus Quality Weighting (verification_score storage, weighted archetype extraction)
- Part 3: Structured Stage Handoff (StageHandoff, _extract_handoff, injection, enhanced prime builders)
"""

import json
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.protocol import SharedState, Message, MessageType
from core.protocol_spec import PROTOCOL, CompactionSpec
from core.pipeline import (
    StageHandoff,
    _extract_handoff,
    PipelineState,
    StageRecord,
    StageResult,
    _build_ground_prime,
    _build_constrain_prime,
    _build_architect_prime,
)


# =============================================================================
# PART 1: Context Graph Compaction
# =============================================================================


class TestCompactionSpec:
    """Tests for CompactionSpec defaults and customization."""

    def test_compaction_spec_defaults(self):
        """CompactionSpec has sensible defaults."""
        spec = PROTOCOL.compaction
        assert spec.max_known_value_length == 500
        assert "pipeline_state" in spec.exclude_keys
        assert "_domain_adapter" in spec.exclude_keys
        assert "_stage_config" in spec.exclude_keys
        assert spec.max_insights == 50
        assert spec.max_decision_trace == 20

    def test_compaction_spec_custom(self):
        """CompactionSpec can be customized."""
        spec = CompactionSpec(
            max_known_value_length=100,
            exclude_keys=("custom_key",),
            max_insights=10,
            max_decision_trace=5,
        )
        assert spec.max_known_value_length == 100
        assert spec.exclude_keys == ("custom_key",)
        assert spec.max_insights == 10
        assert spec.max_decision_trace == 5

    def test_compaction_spec_frozen(self):
        """CompactionSpec is frozen."""
        spec = CompactionSpec()
        with pytest.raises(AttributeError):
            spec.max_known_value_length = 999


class TestCompactKnown:
    """Tests for SharedState.compact_known()."""

    def test_compact_known_excludes_keys(self):
        """compact_known removes excluded keys."""
        state = SharedState()
        state.known["input"] = "hello"
        state.known["pipeline_state"] = {"big": "data"}
        state.known["_domain_adapter"] = object()
        state.known["_stage_config"] = {"config": True}

        result = state.compact_known()
        assert "input" in result
        assert "pipeline_state" not in result
        assert "_domain_adapter" not in result
        assert "_stage_config" not in result

    def test_compact_known_truncates_long_strings(self):
        """compact_known truncates string values exceeding max length."""
        state = SharedState()
        long_string = "x" * 1000
        state.known["long_value"] = long_string
        state.known["short_value"] = "hi"

        result = state.compact_known()
        assert len(result["long_value"]) == 503  # 500 + "..."
        assert result["long_value"].endswith("...")
        assert result["short_value"] == "hi"

    def test_compact_known_preserves_non_strings(self):
        """compact_known passes through non-string values unchanged."""
        state = SharedState()
        state.known["a_dict"] = {"nested": True}
        state.known["a_list"] = [1, 2, 3]
        state.known["an_int"] = 42

        result = state.compact_known()
        assert result["a_dict"] == {"nested": True}
        assert result["a_list"] == [1, 2, 3]
        assert result["an_int"] == 42

    def test_compact_known_custom_spec(self):
        """compact_known respects custom CompactionSpec."""
        spec = CompactionSpec(
            max_known_value_length=10,
            exclude_keys=("secret",),
        )
        state = SharedState()
        state.known["secret"] = "hidden"
        state.known["value"] = "a" * 50

        result = state.compact_known(spec)
        assert "secret" not in result
        assert len(result["value"]) == 13  # 10 + "..."


class TestToContextGraphCompaction:
    """Tests for to_context_graph(compact=True/False)."""

    def _build_state_with_history(self, n_messages=30, n_insights=60):
        """Helper to build a state with lots of history/insights."""
        state = SharedState()
        state.known["input"] = "test description"
        state.known["pipeline_state"] = {"should_be_excluded": True}
        state.known["long_key"] = "z" * 1000

        for i in range(n_messages):
            msg = Message(
                sender="Entity",
                content=f"Message {i}",
                message_type=MessageType.PROPOSITION,
                insight=f"Insight {i}" if i < n_insights else None,
            )
            state.add_message(msg)

        return state

    def test_compact_false_preserves_all(self):
        """compact=False (default) preserves full known, all history, all insights."""
        state = self._build_state_with_history(30, 30)
        result = state.to_context_graph(compact=False)

        assert "pipeline_state" in result["known"]
        assert result["known"]["long_key"] == "z" * 1000
        assert len(result["decision_trace"]) == 30
        assert len(result["insights"]) == 30

    def test_compact_true_strips_and_caps(self):
        """compact=True strips excluded keys, truncates, and caps lists."""
        state = self._build_state_with_history(30, 60)
        result = state.to_context_graph(compact=True)

        # Excluded keys removed
        assert "pipeline_state" not in result["known"]
        # Long strings truncated
        assert len(result["known"]["long_key"]) == 503
        # Decision trace capped
        assert len(result["decision_trace"]) <= PROTOCOL.compaction.max_decision_trace
        # Insights capped
        assert len(result["insights"]) <= PROTOCOL.compaction.max_insights

    def test_compact_default_is_false(self):
        """Default behavior (no argument) is compact=False."""
        state = SharedState()
        state.known["pipeline_state"] = "keep"
        result = state.to_context_graph()
        assert "pipeline_state" in result["known"]

    def test_compact_version_bump(self):
        """Context graph version is 0.3.0."""
        state = SharedState()
        result = state.to_context_graph()
        assert result["meta"]["version"] == "0.3.0"


# =============================================================================
# PART 2: Corpus Quality Weighting
# =============================================================================


class TestCorpusVerificationScore:
    """Tests for verification_score storage and retrieval."""

    @pytest.fixture
    def corpus(self, tmp_path):
        """Create a temporary SQLiteCorpus."""
        from persistence.sqlite_corpus import SQLiteCorpus
        return SQLiteCorpus(corpus_path=tmp_path / "corpus")

    def test_store_with_verification_score(self, corpus):
        """Store accepts and persists verification_score."""
        record = corpus.store(
            input_text="test input",
            context_graph={"known": {"intent": {"domain": "test"}}},
            blueprint={"components": [{"name": "A", "type": "entity"}]},
            insights=["insight1"],
            success=True,
            verification_score=75.5,
        )
        assert record.verification_score == 75.5

        # Verify it's in the database
        retrieved = corpus.get(record.id)
        assert retrieved.verification_score == 75.5

    def test_store_without_verification_score(self, corpus):
        """Store works without verification_score (None default)."""
        record = corpus.store(
            input_text="test input no score",
            context_graph={"known": {"intent": {"domain": "test"}}},
            blueprint={"components": []},
            insights=[],
            success=True,
        )
        assert record.verification_score is None

    def test_migration_idempotent(self, corpus):
        """Re-initializing DB doesn't fail on existing verification_score column."""
        # Call _init_db again — should not raise
        corpus._init_db()
        # Store should still work
        record = corpus.store(
            input_text="after reinit",
            context_graph={"known": {"intent": {"domain": "test"}}},
            blueprint={"components": []},
            insights=[],
            success=True,
            verification_score=50.0,
        )
        assert record.verification_score == 50.0


class TestQualityWeightedArchetypes:
    """Tests for quality-weighted archetype extraction."""

    @pytest.fixture
    def corpus_with_data(self, tmp_path):
        """Create a corpus with 4 compilations of varying quality."""
        from persistence.sqlite_corpus import SQLiteCorpus
        corpus = SQLiteCorpus(corpus_path=tmp_path / "corpus")

        # High-quality compilation (score 90)
        corpus.store(
            input_text="high quality app 1",
            context_graph={"known": {"intent": {"domain": "webapp"}}},
            blueprint={
                "components": [
                    {"name": "AuthService", "type": "entity"},
                    {"name": "UserStore", "type": "entity"},
                ],
                "relationships": [],
                "constraints": [],
            },
            insights=["insight1"],
            success=True,
            verification_score=90.0,
        )

        # High-quality compilation (score 80)
        corpus.store(
            input_text="high quality app 2",
            context_graph={"known": {"intent": {"domain": "webapp"}}},
            blueprint={
                "components": [
                    {"name": "AuthService", "type": "entity"},
                    {"name": "UserStore", "type": "entity"},
                    {"name": "Logger", "type": "entity"},
                ],
                "relationships": [],
                "constraints": [],
            },
            insights=["insight2"],
            success=True,
            verification_score=80.0,
        )

        # Low-quality compilation (score 20)
        corpus.store(
            input_text="low quality app 3",
            context_graph={"known": {"intent": {"domain": "webapp"}}},
            blueprint={
                "components": [
                    {"name": "JunkComponent", "type": "entity"},
                    {"name": "AuthService", "type": "entity"},
                ],
                "relationships": [],
                "constraints": [],
            },
            insights=["insight3"],
            success=True,
            verification_score=20.0,
        )

        # No-score compilation (NULL → default 0.5)
        corpus.store(
            input_text="no score app 4",
            context_graph={"known": {"intent": {"domain": "webapp"}}},
            blueprint={
                "components": [
                    {"name": "AuthService", "type": "entity"},
                    {"name": "RandomThing", "type": "entity"},
                ],
                "relationships": [],
                "constraints": [],
            },
            insights=["insight4"],
            success=True,
            verification_score=None,
        )

        return corpus

    def test_weighted_extraction_favors_high_quality(self, corpus_with_data):
        """High-verification components get higher weighted frequency."""
        from persistence.corpus_analysis import CorpusAnalyzer
        analyzer = CorpusAnalyzer(corpus_with_data)
        archetypes = analyzer.extract_archetypes("webapp", min_frequency=0.1, min_samples=3)

        # AuthService appears in all 4 with high weights → should be top
        arch_names = {a.canonical_name for a in archetypes}
        assert "AuthService" in arch_names

        # Find AuthService archetype
        auth_arch = next(a for a in archetypes if a.canonical_name == "AuthService")
        # JunkComponent only in the low-quality compilation → lower weighted freq
        junk_archs = [a for a in archetypes if a.canonical_name == "JunkComponent"]
        if junk_archs:
            assert junk_archs[0].frequency < auth_arch.frequency

    def test_null_score_uses_default_weight(self, corpus_with_data):
        """Compilations with NULL verification_score use 0.5 default weight."""
        from persistence.corpus_analysis import CorpusAnalyzer
        analyzer = CorpusAnalyzer(corpus_with_data)
        archetypes = analyzer.extract_archetypes("webapp", min_frequency=0.01, min_samples=3)

        # RandomThing only appears in the NULL-score compilation
        random_archs = [a for a in archetypes if a.canonical_name == "RandomThing"]
        if random_archs:
            # Its weighted freq should be 0.5 / total_weight
            assert random_archs[0].frequency > 0

    def test_anti_patterns_unaffected(self, corpus_with_data):
        """Anti-pattern detection doesn't use quality weighting (unchanged)."""
        from persistence.corpus_analysis import CorpusAnalyzer
        analyzer = CorpusAnalyzer(corpus_with_data)
        # Should not raise
        anti = analyzer.detect_anti_patterns(min_samples=1)
        assert isinstance(anti, list)


# =============================================================================
# PART 3: Structured Stage Handoff
# =============================================================================


class TestStageHandoff:
    """Tests for StageHandoff dataclass."""

    def test_frozen(self):
        """StageHandoff is frozen."""
        h = StageHandoff()
        with pytest.raises(AttributeError):
            h.stage_source = "modified"

    def test_defaults_are_empty(self):
        """Default StageHandoff has empty tuples and empty string."""
        h = StageHandoff()
        assert h.type_assignments == ()
        assert h.relationship_hints == ()
        assert h.constraint_context == ()
        assert h.key_entities == ()
        assert h.stage_source == ""

    def test_asdict(self):
        """StageHandoff can be serialized via asdict."""
        h = StageHandoff(
            type_assignments=(("Foo", "entity"),),
            key_entities=("Foo",),
            stage_source="decompose",
        )
        d = asdict(h)
        # asdict preserves tuples for non-dataclass fields
        assert d["type_assignments"] == (("Foo", "entity"),)
        assert d["key_entities"] == ("Foo",)
        assert d["stage_source"] == "decompose"


class TestExtractHandoff:
    """Tests for _extract_handoff()."""

    def test_decompose_handoff(self):
        """Extracts type_assignments and key_entities from DECOMPOSE."""
        artifact = {
            "type_assignments": {"AuthService": "entity", "LoginFlow": "process"},
        }
        h = _extract_handoff("decompose", artifact)
        assert h.stage_source == "decompose"
        assert ("AuthService", "entity") in h.type_assignments
        assert ("LoginFlow", "process") in h.type_assignments
        assert "AuthService" in h.key_entities
        assert "LoginFlow" in h.key_entities

    def test_ground_handoff(self):
        """Extracts relationship_hints and key_entities from GROUND."""
        artifact = {
            "relationships": [
                {"from": "A", "to": "B", "type": "contains"},
                {"from": "B", "to": "C", "type": "triggers"},
            ],
        }
        h = _extract_handoff("ground", artifact)
        assert h.stage_source == "ground"
        assert ("A->B", 1.0) in h.relationship_hints
        assert ("B->C", 1.0) in h.relationship_hints
        assert "A" in h.key_entities
        assert "B" in h.key_entities
        assert "C" in h.key_entities

    def test_constrain_handoff(self):
        """Extracts constraint_context from CONSTRAIN."""
        artifact = {
            "constraints": [
                {"description": "Must be idempotent"},
                {"description": "Max 100ms latency"},
                {"description": ""},  # Empty — should be filtered
            ],
        }
        h = _extract_handoff("constrain", artifact)
        assert h.stage_source == "constrain"
        assert "Must be idempotent" in h.constraint_context
        assert "Max 100ms latency" in h.constraint_context
        assert "" not in h.constraint_context

    def test_unknown_stage_returns_empty(self):
        """Unknown stage returns empty handoff with stage_source set."""
        h = _extract_handoff("expand", {})
        assert h.stage_source == "expand"
        assert h.type_assignments == ()

    def test_empty_artifact_graceful(self):
        """Empty artifact produces empty handoff."""
        h = _extract_handoff("decompose", {})
        assert h.type_assignments == ()
        assert h.key_entities == ()


class TestPipelineStateHandoff:
    """Tests for handoff integration in PipelineState."""

    def _make_stage_record(self, name, artifact):
        return StageRecord(
            name=name,
            state=SharedState(),
            artifact=artifact,
            gate_result=StageResult(success=True),
            turn_count=4,
            duration_seconds=1.0,
        )

    def test_add_stage_sets_handoff(self):
        """add_stage() extracts and stores handoff from record."""
        pipeline = PipelineState(
            original_input="test",
            intent={"core_need": "test"},
            personas=[],
        )
        record = self._make_stage_record("decompose", {
            "type_assignments": {"Foo": "entity"},
        })
        pipeline.add_stage(record)
        assert pipeline.current_handoff is not None
        assert pipeline.current_handoff.stage_source == "decompose"
        assert ("Foo", "entity") in pipeline.current_handoff.type_assignments

    def test_handoff_updates_each_stage(self):
        """Each add_stage replaces the handoff."""
        pipeline = PipelineState(
            original_input="test",
            intent={"core_need": "test"},
            personas=[],
        )
        pipeline.add_stage(self._make_stage_record("decompose", {
            "type_assignments": {"A": "entity"},
        }))
        assert pipeline.current_handoff.stage_source == "decompose"

        pipeline.add_stage(self._make_stage_record("ground", {
            "relationships": [{"from": "A", "to": "B", "type": "triggers"}],
        }))
        assert pipeline.current_handoff.stage_source == "ground"


class TestEnhancedPrimeBuilders:
    """Tests for handoff-enhanced prime content builders."""

    def _make_pipeline_with_handoff(self, handoff):
        pipeline = PipelineState(
            original_input="Build a web app with auth and logging",
            intent={
                "core_need": "web app",
                "domain": "webapp",
                "actors": [],
                "explicit_components": [],
            },
            personas=[],
        )
        pipeline.current_handoff = handoff
        return pipeline

    def test_ground_prime_includes_entities(self):
        """GROUND prime includes previously identified entities from DECOMPOSE handoff."""
        handoff = StageHandoff(
            key_entities=("AuthService", "UserStore"),
            stage_source="decompose",
        )
        pipeline = self._make_pipeline_with_handoff(handoff)
        # Need DECOMPOSE artifact for ground prime
        pipeline.add_stage(StageRecord(
            name="decompose",
            state=SharedState(),
            artifact={
                "components": [
                    {"name": "AuthService", "type": "entity"},
                    {"name": "UserStore", "type": "entity"},
                ],
                "type_assignments": {"AuthService": "entity", "UserStore": "entity"},
            },
            gate_result=StageResult(success=True),
            turn_count=4,
            duration_seconds=1.0,
        ))
        prime = _build_ground_prime(pipeline)
        assert "Previously identified entities" in prime
        assert "AuthService" in prime

    def test_constrain_prime_includes_relationships(self):
        """CONSTRAIN prime includes known relationships from GROUND handoff."""
        handoff = StageHandoff(
            relationship_hints=(("A->B", 1.0), ("B->C", 1.0)),
            stage_source="ground",
        )
        pipeline = self._make_pipeline_with_handoff(handoff)
        # Need DECOMPOSE and GROUND artifacts
        pipeline.stages = [
            StageRecord("decompose", SharedState(),
                        {"components": [{"name": "A", "type": "entity"}]},
                        StageResult(success=True), 4, 1.0),
            StageRecord("ground", SharedState(),
                        {"relationships": [{"from": "A", "to": "B", "type": "triggers"}]},
                        StageResult(success=True), 4, 1.0),
        ]
        prime = _build_constrain_prime(pipeline)
        assert "Known relationships" in prime
        assert "A->B" in prime

    def test_architect_prime_includes_constraints(self):
        """ARCHITECT prime includes active constraints from CONSTRAIN handoff."""
        handoff = StageHandoff(
            constraint_context=("Must be idempotent", "Max 100ms"),
            stage_source="constrain",
        )
        pipeline = self._make_pipeline_with_handoff(handoff)
        pipeline.stages = [
            StageRecord("decompose", SharedState(),
                        {"components": [{"name": "A", "type": "entity"}]},
                        StageResult(success=True), 4, 1.0),
            StageRecord("ground", SharedState(),
                        {"relationships": []},
                        StageResult(success=True), 4, 1.0),
            StageRecord("constrain", SharedState(),
                        {"constraints": [{"description": "Must be idempotent"}]},
                        StageResult(success=True), 4, 1.0),
        ]
        prime = _build_architect_prime(pipeline)
        assert "Active constraints" in prime
        assert "Must be idempotent" in prime

    def test_no_handoff_graceful(self):
        """Prime builders work gracefully with no handoff."""
        pipeline = PipelineState(
            original_input="Build a simple app",
            intent={"core_need": "app", "domain": "test", "actors": [],
                    "explicit_components": []},
            personas=[],
        )
        pipeline.stages = [
            StageRecord("decompose", SharedState(),
                        {"components": [{"name": "X", "type": "entity"}],
                         "type_assignments": {"X": "entity"}},
                        StageResult(success=True), 4, 1.0),
        ]
        # current_handoff is set by add_stage but we use stages directly
        pipeline.current_handoff = None
        prime = _build_ground_prime(pipeline)
        assert "Previously identified" not in prime
        # Should still work
        assert "USER INPUT" in prime
