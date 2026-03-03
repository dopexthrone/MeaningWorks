"""
Tests for core.protocol_spec — Protocol as Data.

Verifies:
1. Immutability: frozen dataclasses reject assignment
2. Value cross-reference: spec values match original hardcoded values
3. Pipeline derivation: STAGE_CONFIGS from spec matches expected structure
4. Singleton: PROTOCOL is a ProtocolSpec instance
"""

import dataclasses
import pytest

from core.protocol_spec import (
    PROTOCOL,
    ProtocolSpec,
    ConfidenceSpec,
    DialogueSpec,
    GovernorSpec,
    PipelineSpec,
    PipelineStageSpec,
    EngineSpec,
    EngineStageSpec,
    ContextSpec,
    MessageDetectionSpec,
    ProvenanceSpec,
)


# =============================================================================
# 1. IMMUTABILITY
# =============================================================================


class TestImmutability:
    """Frozen dataclasses must reject assignment."""

    def test_protocol_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.confidence = ConfidenceSpec()

    def test_confidence_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.confidence.sufficient = 0.9

    def test_dialogue_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.dialogue.default_max_turns = 100

    def test_governor_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.governor.spread_threshold = 0.9

    def test_pipeline_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.pipeline.early_termination_agreements = 10

    def test_pipeline_stage_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.pipeline.stages[0].max_turns = 100

    def test_engine_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.engine.resynth_min_completeness = 0

    def test_context_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.context.recent_messages = 100

    def test_message_detection_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.message_detection.agreement_markers = ()

    def test_provenance_spec_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROTOCOL.provenance.stem_length = 99


# =============================================================================
# 2. VALUE CROSS-REFERENCE (match original hardcoded values)
# =============================================================================


class TestConfidenceValues:
    """Confidence spec values must match original constants."""

    def test_sufficient_threshold(self):
        assert PROTOCOL.confidence.sufficient == 0.7

    def test_convergence_threshold(self):
        assert PROTOCOL.confidence.convergence == 0.6

    def test_warning_threshold(self):
        assert PROTOCOL.confidence.warning == 0.4

    def test_boost_agreement(self):
        assert PROTOCOL.confidence.boost_agreement == 0.15

    def test_boost_accommodation(self):
        assert PROTOCOL.confidence.boost_accommodation == 0.08

    def test_boost_proposition_with_insight(self):
        assert PROTOCOL.confidence.boost_proposition_with_insight == 0.06

    def test_boost_proposition_without_insight(self):
        assert PROTOCOL.confidence.boost_proposition_without_insight == 0.02

    def test_boost_challenge_with_insight(self):
        assert PROTOCOL.confidence.boost_challenge_with_insight == 0.03

    def test_boost_challenge_without_insight(self):
        assert PROTOCOL.confidence.boost_challenge_without_insight == -0.02

    def test_boost_positive_marker(self):
        assert PROTOCOL.confidence.boost_positive_marker == 0.03

    def test_penalty_self_negative(self):
        assert PROTOCOL.confidence.penalty_self_negative == -0.05

    def test_boost_discovery_negative(self):
        assert PROTOCOL.confidence.boost_discovery_negative == 0.02

    def test_coverage_per_insight(self):
        assert PROTOCOL.confidence.coverage_per_insight == 0.08

    def test_unknown_penalty(self):
        assert PROTOCOL.confidence.unknown_penalty_per == 0.05
        assert PROTOCOL.confidence.unknown_penalty_cap == 0.3

    def test_blending_weights(self):
        assert PROTOCOL.confidence.insight_grounding_factor == 0.08
        assert PROTOCOL.confidence.blending_grounded_weight == 0.5
        assert PROTOCOL.confidence.blending_accumulated_weight == 0.5

    def test_consistency_window(self):
        assert PROTOCOL.confidence.consistency_window == 6

    def test_plateau_defaults(self):
        assert PROTOCOL.confidence.plateau_window == 3
        assert PROTOCOL.confidence.plateau_threshold == 0.02


class TestDialogueValues:
    """Dialogue spec values must match original constants."""

    def test_defaults(self):
        assert PROTOCOL.dialogue.default_max_turns == 64
        assert PROTOCOL.dialogue.default_min_turns == 6
        assert PROTOCOL.dialogue.default_min_insights == 8

    def test_depth_bonuses(self):
        assert PROTOCOL.dialogue.component_divisor == 3
        assert PROTOCOL.dialogue.component_bonus_cap == 3
        assert PROTOCOL.dialogue.actor_divisor == 2
        assert PROTOCOL.dialogue.actor_bonus_cap == 2
        assert PROTOCOL.dialogue.relationship_divisor == 4
        assert PROTOCOL.dialogue.relationship_bonus_cap == 2

    def test_depth_thresholds(self):
        assert PROTOCOL.dialogue.description_length_threshold == 2000
        assert PROTOCOL.dialogue.constraint_count_threshold == 5

    def test_depth_caps(self):
        assert PROTOCOL.dialogue.min_turns_cap == 48
        assert PROTOCOL.dialogue.max_turns_offset == 16
        assert PROTOCOL.dialogue.max_turns_ceiling == 64

    def test_convergence(self):
        assert PROTOCOL.dialogue.convergence_window == 4
        assert PROTOCOL.dialogue.convergence_agreement_threshold == 2


class TestGovernorValues:
    """Governor spec values must match original constants."""

    def test_low_dim(self):
        assert PROTOCOL.governor.low_dim_threshold == 0.4
        assert PROTOCOL.governor.low_dim_extra_turns == 2

    def test_spread(self):
        assert PROTOCOL.governor.spread_threshold == 0.4
        assert PROTOCOL.governor.spread_extra_turns == 3


class TestContextValues:
    """Context spec values must match original constants."""

    def test_recent_messages(self):
        assert PROTOCOL.context.recent_messages == 3

    def test_truncation(self):
        assert PROTOCOL.context.truncation_length == 150
        assert PROTOCOL.context.core_need_truncation == 100

    def test_phase_thresholds(self):
        assert PROTOCOL.context.explore_threshold == 4
        assert PROTOCOL.context.challenge_threshold == 8

    def test_persona_limits(self):
        assert PROTOCOL.context.max_personas == 3
        assert PROTOCOL.context.max_priorities == 3
        assert PROTOCOL.context.max_blind_spot_length == 80
        assert PROTOCOL.context.max_key_questions == 2


class TestMessageDetectionValues:
    """Message detection markers must match original inline lists."""

    def test_agreement_markers_count(self):
        assert len(PROTOCOL.message_detection.agreement_markers) == 12

    def test_agreement_markers_content(self):
        markers = PROTOCOL.message_detection.agreement_markers
        assert "sufficient" in markers
        assert "i agree" in markers
        assert "aligned" in markers

    def test_strong_challenge_markers_count(self):
        assert len(PROTOCOL.message_detection.strong_challenge_markers) == 9

    def test_weak_challenge_markers_count(self):
        assert len(PROTOCOL.message_detection.weak_challenge_markers) == 3

    def test_accommodation_markers_count(self):
        assert len(PROTOCOL.message_detection.accommodation_markers) == 14

    def test_positive_markers_count(self):
        assert len(PROTOCOL.message_detection.positive_markers) == 12

    def test_negative_markers_content(self):
        markers = PROTOCOL.message_detection.negative_markers
        assert "missing" in markers
        assert "missed" in markers
        assert "gap" in markers

    def test_self_markers_content(self):
        markers = PROTOCOL.message_detection.self_markers
        assert "i missed" in markers
        assert "i failed" in markers


class TestProvenanceValues:
    """Provenance spec values must match original constants."""

    def test_stem_length(self):
        assert PROTOCOL.provenance.stem_length == 5

    def test_min_matches(self):
        assert PROTOCOL.provenance.min_matches == 1

    def test_common_words_type(self):
        assert isinstance(PROTOCOL.provenance.common_words, frozenset)

    def test_common_words_content(self):
        cw = PROTOCOL.provenance.common_words
        # Meta-vocabulary
        assert "entity" in cw
        assert "component" in cw
        assert "process" in cw
        # Function words
        assert "needs" in cw
        assert "needed" in cw
        assert "everything" in cw


# =============================================================================
# 3. PIPELINE DERIVATION
# =============================================================================


class TestPipelineDerivation:
    """Pipeline spec produces correct STAGE_CONFIGS structure."""

    def test_stage_count(self):
        assert len(PROTOCOL.pipeline.stages) == 5

    def test_stage_names(self):
        names = [s.name for s in PROTOCOL.pipeline.stages]
        assert names == ["expand", "decompose", "ground", "constrain", "architect"]

    def test_stage_configs_match(self):
        """Re-derived STAGE_CONFIGS must match original hardcoded values."""
        expected = [
            ("expand", 4, 2, 300),
            ("decompose", 8, 3, 600),
            ("ground", 8, 3, 600),
            ("constrain", 6, 2, 300),
            ("architect", 6, 2, 300),
        ]
        derived = [
            (s.name, s.max_turns, s.min_turns, s.timeout_seconds)
            for s in PROTOCOL.pipeline.stages
        ]
        assert derived == expected

    def test_total_pipeline_timeout(self):
        """Sum of stage timeouts = 2100."""
        total = sum(s.timeout_seconds for s in PROTOCOL.pipeline.stages)
        assert total == 2100

    def test_early_termination(self):
        assert PROTOCOL.pipeline.early_termination_agreements == 2


class TestEngineDerivation:
    """Engine spec stage values match original STAGE_GATES."""

    def test_engine_stage_count(self):
        assert len(PROTOCOL.engine.stages) == 5

    def test_engine_stage_names(self):
        names = [s.name for s in PROTOCOL.engine.stages]
        assert names == ["intent", "personas", "dialogue", "synthesis", "verification"]

    def test_engine_timeouts(self):
        timeouts = {s.name: s.timeout_seconds for s in PROTOCOL.engine.stages}
        assert timeouts["intent"] == 300
        assert timeouts["personas"] == 300
        assert timeouts["dialogue"] == 1800
        assert timeouts["synthesis"] == 600
        assert timeouts["verification"] == 300

    def test_engine_retries(self):
        retries = {s.name: s.max_retries for s in PROTOCOL.engine.stages}
        assert retries["intent"] == 2
        assert retries["personas"] == 2
        assert retries["dialogue"] == 1
        assert retries["synthesis"] == 3
        assert retries["verification"] == 1

    def test_quality_thresholds(self):
        assert PROTOCOL.engine.quality_reject_threshold == 0.15
        assert PROTOCOL.engine.quality_warn_threshold == 0.35

    def test_resynth_threshold(self):
        assert PROTOCOL.engine.resynth_min_completeness == 30


# =============================================================================
# 4. SINGLETON & STRUCTURE
# =============================================================================


class TestSingleton:
    """PROTOCOL is a proper ProtocolSpec instance."""

    def test_type(self):
        assert isinstance(PROTOCOL, ProtocolSpec)

    def test_section_count(self):
        assert len(dataclasses.fields(PROTOCOL)) == 13

    def test_all_sections_present(self):
        assert isinstance(PROTOCOL.confidence, ConfidenceSpec)
        assert isinstance(PROTOCOL.dialogue, DialogueSpec)
        assert isinstance(PROTOCOL.governor, GovernorSpec)
        from core.protocol_spec import DialecticSpec
        assert isinstance(PROTOCOL.dialectic, DialecticSpec)
        assert isinstance(PROTOCOL.pipeline, PipelineSpec)
        assert isinstance(PROTOCOL.engine, EngineSpec)
        assert isinstance(PROTOCOL.context, ContextSpec)
        assert isinstance(PROTOCOL.message_detection, MessageDetectionSpec)
        assert isinstance(PROTOCOL.provenance, ProvenanceSpec)
        from core.protocol_spec import BuildSpec
        assert isinstance(PROTOCOL.build, BuildSpec)
