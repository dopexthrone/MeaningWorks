"""Tests for Build 2: Verification Quality.

Covers:
- score_coherence() with component_count scaling
- score_traceability() with specificity threshold + generic phrase rejection
- Synthesis prompt SECTION 3b presence
- Phase 22e persistent feedback fallback
"""

import pytest

from core.verification import (
    score_coherence,
    score_traceability,
    DimensionScore,
)


# --- score_coherence with component_count ---


class TestCoherenceFormula:
    def test_small_system_lower_orphan_penalty(self):
        """≤5 components should use orphan_weight=25 (less harsh)."""
        # Same orphan_ratio, different component counts
        small = score_coherence(
            orphan_ratio=0.5,
            relationship_density=0.5,
            health_score=0.5,
            dangling_count=0,
            component_count=3,
        )
        large = score_coherence(
            orphan_ratio=0.5,
            relationship_density=0.5,
            health_score=0.5,
            dangling_count=0,
            component_count=15,
        )
        # Small system should score higher (less penalty)
        assert small.score > large.score

    def test_medium_system_uses_middle_weight(self):
        """6-10 components should use orphan_weight=32."""
        medium = score_coherence(
            orphan_ratio=0.5,
            relationship_density=0.5,
            health_score=0.5,
            dangling_count=0,
            component_count=8,
        )
        small = score_coherence(
            orphan_ratio=0.5,
            relationship_density=0.5,
            health_score=0.5,
            dangling_count=0,
            component_count=3,
        )
        large = score_coherence(
            orphan_ratio=0.5,
            relationship_density=0.5,
            health_score=0.5,
            dangling_count=0,
            component_count=15,
        )
        assert small.score > medium.score > large.score

    def test_zero_orphans_scores_high(self):
        result = score_coherence(
            orphan_ratio=0.0,
            relationship_density=1.0,
            health_score=0.8,
            dangling_count=0,
            component_count=10,
        )
        assert result.score >= 65

    def test_density_bonus_caps_at_1_5(self):
        """Density > 1.5 should not increase score further."""
        at_cap = score_coherence(
            orphan_ratio=0.0,
            relationship_density=1.5,
            health_score=0.8,
            dangling_count=0,
            component_count=10,
        )
        over_cap = score_coherence(
            orphan_ratio=0.0,
            relationship_density=3.0,
            health_score=0.8,
            dangling_count=0,
            component_count=10,
        )
        assert at_cap.score == over_cap.score

    def test_backward_compat_default_component_count(self):
        """component_count=0 should use orphan_weight=25 (≤5 path)."""
        result = score_coherence(
            orphan_ratio=0.3,
            relationship_density=0.5,
            health_score=0.7,
            dangling_count=0,
        )
        assert isinstance(result, DimensionScore)
        assert 0 <= result.score <= 100

    def test_typical_10_component_blueprint_scores_above_60(self):
        """A typical blueprint (10 comps, 0.3 orphans, density 0.8) should score ≥60."""
        result = score_coherence(
            orphan_ratio=0.3,
            relationship_density=0.8,
            health_score=0.7,
            dangling_count=1,
            component_count=10,
        )
        assert result.score >= 60

    def test_coherence_name(self):
        result = score_coherence(0.0, 1.0, 1.0, 0)
        assert result.name == "coherence"


# --- score_traceability with specificity ---


class TestTraceabilitySpecificity:
    def test_generic_phrase_not_counted_as_specific(self):
        components = [
            {"name": "A", "derived_from": "user input"},
            {"name": "B", "derived_from": "from dialogue"},
            {"name": "C", "derived_from": "inferred from input"},
        ]
        result = score_traceability(components)
        # All have derived_from (ratio_derived=1.0, worth 60)
        # None are specific (all generic), so ratio_specific=0.0 (worth 0)
        assert result.score == 60

    def test_short_derived_from_not_specific(self):
        """derived_from ≤20 chars should not count as specific."""
        components = [
            {"name": "A", "derived_from": "short text here"},  # 15 chars
        ]
        result = score_traceability(components)
        assert result.score == 60  # has derived_from but not specific

    def test_long_specific_derived_from_scores_full(self):
        components = [
            {"name": "A", "derived_from": "User described a notification system that alerts on price changes (Section 1)"},
        ]
        result = score_traceability(components)
        assert result.score == 100  # ratio_derived=1.0 * 60 + ratio_specific=1.0 * 40

    def test_mixed_specificity(self):
        components = [
            {"name": "A", "derived_from": "User described a notification system that alerts on price changes"},
            {"name": "B", "derived_from": "user input"},
            {"name": "C"},  # missing
        ]
        result = score_traceability(components)
        # ratio_derived = 2/3 * 60 = 40
        # ratio_specific = 1/3 * 40 ≈ 13
        # Total ≈ 53
        assert 50 <= result.score <= 60

    def test_dialogue_context_is_generic(self):
        components = [
            {"name": "A", "derived_from": "dialogue context"},
        ]
        result = score_traceability(components)
        assert result.score == 60  # has derived but not specific

    def test_grid_ref_is_specific(self):
        components = [
            {"name": "A", "derived_from": "grid:STR.ENT.CMP.WHAT.SFT — entity extracted from dialogue round 2"},
        ]
        result = score_traceability(components)
        assert result.score == 100

    def test_empty_components(self):
        result = score_traceability([])
        assert result.score == 0

    def test_all_missing_derived_from(self):
        components = [{"name": "A"}, {"name": "B"}]
        result = score_traceability(components)
        assert result.score == 0


# --- Synthesis prompt quality section ---


class TestSynthesisQualityPrompt:
    def test_section_3b_present_in_synthesis(self):
        """The synthesis prompt should include SECTION 3b: QUALITY REQUIREMENTS."""
        # We test by building the sections list from _synthesize
        # Since _synthesize is complex, we test the section content directly
        from core.engine import MotherlabsEngine
        import inspect
        source = inspect.getsource(MotherlabsEngine._synthesize)
        assert "SECTION 3b: QUALITY REQUIREMENTS" in source
        assert "RELATIONSHIP DENSITY" in source
        assert "DERIVED_FROM SPECIFICITY" in source
        assert "user input" in source  # Listed as a BAD example
        assert "dialogue context" in source

    def test_banned_phrases_match_traceability(self):
        """Ensure the banned phrases in synthesis prompt match verification logic."""
        from core.verification import score_traceability
        import inspect
        verification_source = inspect.getsource(score_traceability)
        # These phrases should appear in both the synthesis prompt and verification
        for phrase in ["user input", "inferred from input", "dialogue context"]:
            assert phrase in verification_source


# --- Phase 22e persistent feedback ---


class TestPersistentFeedback:
    def test_phase_22e_loads_from_outcome_store(self):
        """Phase 22e should fall back to persistent outcome store."""
        from core.engine import MotherlabsEngine
        import inspect
        source = inspect.getsource(MotherlabsEngine.compile)
        assert "self._outcome_store" in source
        # The fallback should use .recent()
        assert "outcome_store.recent" in source or "_outcome_store.recent" in source
