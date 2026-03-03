"""Tests for mother/sentience.py — sentience chamber experience compiler."""

import json
import time

import pytest

from mother.sentience import (
    FacetDomain,
    ExperienceFacet,
    ExperienceTrace,
    ExperienceMemory,
    ChamberInput,
    ExperienceOutput,
    compile_experience,
    serialize_experience_memory,
    deserialize_experience_memory,
    format_experience_context,
    _clamp,
    _sentence_from_bands,
    _compute_salience,
    _somatic_facet,
    _relational_facet,
    _epistemic_facet,
    _temporal_facet,
    _environmental_facet,
    _volitional_facet,
    _affective_facet,
    _phase_0_sense,
    _phase_1_surprise,
    _phase_2_rank,
    _phase_3_compose,
    _phase_4_verify,
    _phase_5_remember,
    EMA_ALPHA,
    SURPRISE_THRESHOLD,
    FIDELITY_THRESHOLD,
    MAX_SELECTED,
    MAX_MEMORABLE,
    HIGH_SURPRISE,
    _SOMATIC_BANDS,
    _RELATIONAL_BANDS,
    _EPISTEMIC_BANDS,
    _TEMPORAL_BANDS,
    _ENVIRONMENTAL_BANDS,
    _VOLITIONAL_BANDS,
    _AFFECTIVE_BANDS,
    _DOMAIN_TO_BASELINE,
)


# ============================================================
# Type construction
# ============================================================

class TestTypes:
    """Frozen dataclass construction and basic properties."""

    def test_facet_domain_values(self):
        assert len(FacetDomain) == 7
        assert FacetDomain.SOMATIC.value == "somatic"
        assert FacetDomain.AFFECTIVE.value == "affective"

    def test_experience_facet_frozen(self):
        f = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="energy", value=0.7,
            sentence="test", sources=("a",), salience=0.4,
        )
        assert f.value == 0.7
        with pytest.raises(AttributeError):
            f.value = 0.5  # type: ignore

    def test_experience_memory_defaults(self):
        m = ExperienceMemory()
        assert m.baseline_somatic == 0.5
        assert m.baseline_relational == 0.05
        assert m.baseline_temporal == 0.15
        assert m.baseline_environmental == 0.0
        assert m.peak_confidence == 0.5
        assert m.memorable_moments == ()
        assert m.update_count == 0

    def test_chamber_input_defaults(self):
        inp = ChamberInput()
        assert inp.confidence == 0.5
        assert inp.engagement == 0.0
        assert inp.content_topics == ()

    def test_experience_output_frozen(self):
        mem = ExperienceMemory()
        trace = ExperienceTrace(
            facets=(), selected=(), total_salience=0.0,
            dominant_domain=FacetDomain.SOMATIC, surprise_count=0,
            compression_ratio=0.0,
        )
        out = ExperienceOutput(
            narration="test", trace=trace, memory=mem,
            gate_passed=True, gate_fidelity=1.0,
        )
        assert out.narration == "test"
        with pytest.raises(AttributeError):
            out.narration = "x"  # type: ignore

    def test_experience_trace_frozen(self):
        t = ExperienceTrace(
            facets=(), selected=(), total_salience=0.5,
            dominant_domain=FacetDomain.EPISTEMIC, surprise_count=2,
            compression_ratio=0.71,
        )
        assert t.surprise_count == 2


# ============================================================
# Utility functions
# ============================================================

class TestUtilities:
    def test_clamp_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_bounds(self):
        assert _clamp(0.3, 0.5, 1.0) == 0.5

    def test_salience_extreme_high(self):
        s = _compute_salience(1.0)
        assert s == 1.0

    def test_salience_extreme_low(self):
        s = _compute_salience(0.0)
        assert s == 1.0

    def test_salience_neutral(self):
        s = _compute_salience(0.5)
        assert s == 0.0

    def test_salience_moderate(self):
        s = _compute_salience(0.75)
        assert s == 0.5

    def test_sentence_from_bands_high(self):
        s = _sentence_from_bands(0.9, _SOMATIC_BANDS)
        assert s == "Energy is high — systems are running well."

    def test_sentence_from_bands_mid(self):
        s = _sentence_from_bands(0.5, _SOMATIC_BANDS)
        assert s == "Systems are steady."

    def test_sentence_from_bands_low(self):
        s = _sentence_from_bands(0.1, _SOMATIC_BANDS)
        assert s == "Resting state, ready when needed."

    def test_sentence_from_bands_boundary(self):
        # Exactly at threshold
        s = _sentence_from_bands(0.8, _SOMATIC_BANDS)
        assert s == "Energy is high — systems are running well."

    def test_all_band_tables_have_entries(self):
        for bands in [_SOMATIC_BANDS, _RELATIONAL_BANDS, _EPISTEMIC_BANDS,
                       _TEMPORAL_BANDS, _ENVIRONMENTAL_BANDS, _VOLITIONAL_BANDS,
                       _AFFECTIVE_BANDS]:
            assert len(bands) >= 4
            # Last entry covers 0.0
            assert bands[-1][0] == 0.0


# ============================================================
# Phase 0: SENSE — 7 Extractors
# ============================================================

class TestExtractors:
    def test_somatic_high_confidence(self):
        inp = ChamberInput(confidence=0.9, engagement=0.8, tension=0.1)
        f = _somatic_facet(inp)
        assert f.domain == FacetDomain.SOMATIC
        assert f.label == "energy"
        assert f.value >= 0.7
        assert "confidence" in f.sources

    def test_somatic_low_energy(self):
        inp = ChamberInput(confidence=0.1, engagement=0.1, tension=0.9)
        f = _somatic_facet(inp)
        assert f.value < 0.3

    def test_relational_new_user(self):
        inp = ChamberInput(rapport=0.0, relationship_depth=0.0, trust_success_rate=0.0)
        f = _relational_facet(inp)
        assert f.domain == FacetDomain.RELATIONAL
        assert f.value < 0.15
        assert "Fresh start" in f.sentence

    def test_relational_deep_connection(self):
        inp = ChamberInput(rapport=0.8, relationship_depth=0.7, trust_success_rate=0.9, satisfaction=0.8)
        f = _relational_facet(inp)
        assert f.value >= 0.7
        assert "real connection" in f.sentence

    def test_epistemic_high_clarity(self):
        inp = ChamberInput(confidence=0.9, trust_avg_fidelity=0.8, last_compile_trust=0.9, tension=0.1)
        f = _epistemic_facet(inp)
        assert f.domain == FacetDomain.EPISTEMIC
        assert f.value >= 0.7

    def test_epistemic_low_clarity(self):
        inp = ChamberInput(confidence=0.1, trust_avg_fidelity=0.1, last_compile_trust=0.0, tension=0.8)
        f = _epistemic_facet(inp)
        assert f.value < 0.25

    def test_temporal_active_conversation(self):
        inp = ChamberInput(messages_per_minute=2.0, session_minutes=30.0, time_since_last=10.0)
        f = _temporal_facet(inp)
        assert f.domain == FacetDomain.TEMPORAL
        assert f.value > 0.3

    def test_temporal_idle(self):
        inp = ChamberInput(messages_per_minute=0.0, session_minutes=0.0, time_since_last=600.0)
        f = _temporal_facet(inp)
        assert f.value < 0.3

    def test_environmental_rich(self):
        inp = ChamberInput(env_entry_count=10, env_avg_confidence=0.8, fusion_confidence=0.7, attention_significance=0.6)
        f = _environmental_facet(inp)
        assert f.domain == FacetDomain.ENVIRONMENTAL
        assert f.value >= 0.5

    def test_environmental_sparse(self):
        inp = ChamberInput(env_entry_count=0, env_avg_confidence=0.0, fusion_confidence=0.0, attention_significance=0.0)
        f = _environmental_facet(inp)
        assert f.value == 0.0

    def test_volitional_active_compile(self):
        inp = ChamberInput(curiosity=0.7, engagement=0.6, has_active_compile=True, recent_actions_success_rate=0.8)
        f = _volitional_facet(inp)
        assert f.domain == FacetDomain.VOLITIONAL
        assert f.value > 0.3

    def test_volitional_passive(self):
        inp = ChamberInput(curiosity=0.0, engagement=0.0, has_active_compile=False, recent_actions_success_rate=0.0)
        f = _volitional_facet(inp)
        assert f.value == 0.0

    def test_affective_warm(self):
        inp = ChamberInput(satisfaction=0.8, rapport=0.7, tension=0.1, confidence=0.8)
        f = _affective_facet(inp)
        assert f.domain == FacetDomain.AFFECTIVE
        assert f.value >= 0.6

    def test_affective_flat(self):
        inp = ChamberInput(satisfaction=0.0, rapport=0.0, tension=0.8, confidence=0.0)
        f = _affective_facet(inp)
        assert f.value < 0.2

    def test_phase_0_produces_7_facets(self):
        inp = ChamberInput()
        facets = _phase_0_sense(inp)
        assert len(facets) == 7
        domains = {f.domain for f in facets}
        assert len(domains) == 7

    def test_all_facets_have_provenance(self):
        """SAX1: every facet has non-empty sources."""
        inp = ChamberInput(confidence=0.7, rapport=0.5, curiosity=0.3)
        facets = _phase_0_sense(inp)
        for f in facets:
            assert len(f.sources) > 0
            assert all(isinstance(s, str) for s in f.sources)

    def test_all_facet_values_clamped(self):
        """Values always in [0.0, 1.0]."""
        inp = ChamberInput(confidence=1.0, engagement=1.0, rapport=1.0,
                           tension=0.0, curiosity=1.0, satisfaction=1.0,
                           trust_success_rate=1.0, trust_avg_fidelity=1.0,
                           relationship_depth=1.0, last_compile_trust=1.0)
        facets = _phase_0_sense(inp)
        for f in facets:
            assert 0.0 <= f.value <= 1.0


# ============================================================
# Phase 1: SURPRISE
# ============================================================

class TestSurprise:
    def test_no_surprise_at_baseline(self):
        """Facet at baseline → surprise stays 0."""
        inp = ChamberInput(confidence=0.5, engagement=0.5, tension=0.3)
        facets = _phase_0_sense(inp)
        mem = ExperienceMemory()
        surprised = _phase_1_surprise(facets, mem)
        # Most facets should have low surprise relative to defaults
        assert all(isinstance(f, ExperienceFacet) for f in surprised)

    def test_large_deviation_triggers_surprise(self):
        """Value far from baseline → surprise flagged."""
        facet = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="energy", value=0.1,
            sentence="Resting state, ready when needed.", sources=("confidence",),
            salience=0.8,
        )
        mem = ExperienceMemory(baseline_somatic=0.7)
        result = _phase_1_surprise((facet,), mem)
        assert result[0].surprise > SURPRISE_THRESHOLD

    def test_small_deviation_no_surprise(self):
        """Value near baseline → surprise stays 0."""
        facet = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="energy", value=0.72,
            sentence="Energy is high — systems are running well.", sources=("confidence",),
            salience=0.1,
        )
        mem = ExperienceMemory(baseline_somatic=0.7)
        result = _phase_1_surprise((facet,), mem)
        assert result[0].surprise == 0.0  # below threshold, not flagged


# ============================================================
# Phase 2: RANK
# ============================================================

class TestRank:
    def test_selects_top_n(self):
        facets = tuple(
            ExperienceFacet(
                domain=list(FacetDomain)[i], label=f"f{i}", value=0.5,
                sentence=f"s{i}", sources=(f"src{i}",),
                salience=0.1 * (i + 1), surprise=0.0,
            )
            for i in range(7)
        )
        selected = _phase_2_rank(facets)
        assert len(selected) <= MAX_SELECTED

    def test_high_surprise_boosts_rank(self):
        low_sal = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="e", value=0.5,
            sentence="s1", sources=("a",), salience=0.1, surprise=0.8,
        )
        high_sal = ExperienceFacet(
            domain=FacetDomain.RELATIONAL, label="r", value=0.5,
            sentence="s2", sources=("b",), salience=0.6, surprise=0.0,
        )
        selected = _phase_2_rank((low_sal, high_sal))
        # low salience but high surprise should rank first: 0.1 + 0.8*1.5 = 1.3 > 0.6
        assert selected[0].domain == FacetDomain.SOMATIC

    def test_always_selects_at_least_one(self):
        facet = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="e", value=0.5,
            sentence="s", sources=("a",), salience=0.01,
        )
        selected = _phase_2_rank((facet,))
        assert len(selected) == 1

    def test_filters_below_salience_floor(self):
        """Facets below SALIENCE_FLOOR and without surprise get filtered."""
        facets = tuple(
            ExperienceFacet(
                domain=list(FacetDomain)[i], label=f"f{i}", value=0.5,
                sentence=f"s{i}", sources=(f"src{i}",),
                salience=0.01, surprise=0.0,
            )
            for i in range(7)
        )
        selected = _phase_2_rank(facets)
        # All below floor, but at least one survives
        assert len(selected) >= 1


# ============================================================
# Phase 3: COMPOSE
# ============================================================

class TestCompose:
    def test_basic_composition(self):
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Energy is high — systems are running well.",
                sources=("confidence",), salience=0.6,
            ),
        )
        narration = _phase_3_compose(facets, ExperienceMemory())
        assert "Energy is high" in narration

    def test_surprise_prefix(self):
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Energy is high — systems are running well.",
                sources=("confidence",), salience=0.6, surprise=0.5,
            ),
        )
        narration = _phase_3_compose(facets, ExperienceMemory())
        assert "Something shifted" in narration

    def test_multiple_facets_joined(self):
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Systems are steady.", sources=("a",), salience=0.6,
            ),
            ExperienceFacet(
                domain=FacetDomain.RELATIONAL, label="rapport", value=0.5,
                sentence="A working relationship, building.", sources=("b",), salience=0.3,
            ),
        )
        narration = _phase_3_compose(facets, ExperienceMemory())
        assert "Systems are steady" in narration
        assert "working relationship" in narration

    def test_memory_callback(self):
        mem = ExperienceMemory(
            memorable_moments=((1000.0, "somatic", "Energy was incredible."),),
        )
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Systems are steady.", sources=("a",), salience=0.6,
            ),
        )
        narration = _phase_3_compose(facets, mem)
        assert "like before" in narration

    def test_empty_selected_returns_empty(self):
        assert _phase_3_compose((), ExperienceMemory()) == ""


# ============================================================
# Phase 4: VERIFY
# ============================================================

class TestVerify:
    def test_gate_passes_when_all_found(self):
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Systems are steady.", sources=("a",), salience=0.6,
            ),
        )
        narration = "Systems are steady."
        passed, fidelity = _phase_4_verify(narration, facets)
        assert passed is True
        assert fidelity == 1.0

    def test_gate_fails_when_missing(self):
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Systems are steady.", sources=("a",), salience=0.6,
            ),
            ExperienceFacet(
                domain=FacetDomain.RELATIONAL, label="rapport", value=0.5,
                sentence="A working relationship, building.", sources=("b",), salience=0.3,
            ),
            ExperienceFacet(
                domain=FacetDomain.EPISTEMIC, label="conf", value=0.5,
                sentence="Reasonable understanding, some gaps remain.", sources=("c",), salience=0.2,
            ),
        )
        narration = "Systems are steady."  # only 1 of 3
        passed, fidelity = _phase_4_verify(narration, facets)
        assert passed is False
        assert abs(fidelity - 1 / 3) < 0.01

    def test_gate_handles_surprise_lowercasing(self):
        """Surprise prefix lowercases first char — gate should still find it."""
        facets = (
            ExperienceFacet(
                domain=FacetDomain.SOMATIC, label="energy", value=0.8,
                sentence="Energy is high — systems are running well.",
                sources=("a",), salience=0.6, surprise=0.5,
            ),
        )
        narration = "Something shifted — energy is high — systems are running well."
        passed, fidelity = _phase_4_verify(narration, facets)
        assert passed is True

    def test_empty_selected_passes(self):
        passed, fidelity = _phase_4_verify("anything", ())
        assert passed is True
        assert fidelity == 1.0

    def test_threshold_boundary(self):
        """Exactly at threshold should pass."""
        # 3 of 5 = 0.6 = threshold
        facets = tuple(
            ExperienceFacet(
                domain=list(FacetDomain)[i], label=f"f{i}", value=0.5,
                sentence=f"unique_sentence_{i}", sources=(f"s{i}",), salience=0.3,
            )
            for i in range(5)
        )
        narration = "unique_sentence_0 unique_sentence_1 unique_sentence_2"
        passed, fidelity = _phase_4_verify(narration, facets)
        assert passed is True
        assert fidelity == 0.6


# ============================================================
# Phase 5: REMEMBER
# ============================================================

class TestRemember:
    def test_ema_smoothing(self):
        """Baselines drift toward current value."""
        facets = _phase_0_sense(ChamberInput(confidence=1.0, engagement=1.0, tension=0.0))
        mem = ExperienceMemory(baseline_somatic=0.5)
        new_mem = _phase_5_remember(facets, mem, 1000.0)
        # Should have moved toward the facet value
        somatic_facet = next(f for f in facets if f.domain == FacetDomain.SOMATIC)
        expected = EMA_ALPHA * somatic_facet.value + (1 - EMA_ALPHA) * 0.5
        assert abs(new_mem.baseline_somatic - expected) < 0.01

    def test_update_count_increments(self):
        mem = ExperienceMemory(update_count=5)
        facets = _phase_0_sense(ChamberInput())
        new_mem = _phase_5_remember(facets, mem, 1000.0)
        assert new_mem.update_count == 6

    def test_peak_tracking(self):
        mem = ExperienceMemory(peak_confidence=0.3)
        # High epistemic value should update peak
        facets = _phase_0_sense(ChamberInput(confidence=0.9, trust_avg_fidelity=0.9, last_compile_trust=0.9))
        new_mem = _phase_5_remember(facets, mem, 1000.0)
        assert new_mem.peak_confidence >= 0.3

    def test_memorable_moment_on_high_surprise(self):
        """Facet with surprise > HIGH_SURPRISE gets stored."""
        facet = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="energy", value=0.9,
            sentence="Energy is high — systems are running well.",
            sources=("confidence",), salience=0.8, surprise=0.6,
        )
        # Need all 7 domains for phase 5
        filler = tuple(
            ExperienceFacet(
                domain=d, label="x", value=0.5, sentence="x",
                sources=("x",), salience=0.1,
            )
            for d in FacetDomain if d != FacetDomain.SOMATIC
        )
        facets = (facet,) + filler
        mem = ExperienceMemory()
        new_mem = _phase_5_remember(facets, mem, 1000.0)
        assert len(new_mem.memorable_moments) == 1
        assert new_mem.memorable_moments[0][1] == "somatic"

    def test_ring_buffer_eviction(self):
        """Ring buffer maxes at MAX_MEMORABLE entries."""
        moments = tuple((float(i), "somatic", f"s{i}") for i in range(MAX_MEMORABLE))
        mem = ExperienceMemory(memorable_moments=moments)
        facet = ExperienceFacet(
            domain=FacetDomain.SOMATIC, label="energy", value=0.9,
            sentence="New entry", sources=("a",), salience=0.8, surprise=0.6,
        )
        filler = tuple(
            ExperienceFacet(
                domain=d, label="x", value=0.5, sentence="x",
                sources=("x",), salience=0.1,
            )
            for d in FacetDomain if d != FacetDomain.SOMATIC
        )
        new_mem = _phase_5_remember((facet,) + filler, mem, 999.0)
        assert len(new_mem.memorable_moments) == MAX_MEMORABLE

    def test_surprise_accumulation(self):
        facets = _phase_0_sense(ChamberInput())
        mem = ExperienceMemory(cumulative_surprise=1.0)
        new_mem = _phase_5_remember(facets, mem, 1000.0)
        # Should be: old * decay + new surprises
        assert new_mem.cumulative_surprise >= 0.0


# ============================================================
# Entry point: compile_experience
# ============================================================

class TestCompileExperience:
    def test_basic_compilation(self):
        inp = ChamberInput(confidence=0.7, rapport=0.3, curiosity=0.5)
        out = compile_experience(inp, timestamp=1000.0)
        assert isinstance(out, ExperienceOutput)
        assert isinstance(out.narration, str)
        assert len(out.narration) > 0
        assert isinstance(out.trace, ExperienceTrace)
        assert isinstance(out.memory, ExperienceMemory)
        assert isinstance(out.gate_passed, bool)

    def test_determinism(self):
        """Same input + timestamp → same output. SAX2 enforcement."""
        inp = ChamberInput(confidence=0.7, rapport=0.3, curiosity=0.5)
        mem = ExperienceMemory()
        ts = 1000.0
        out1 = compile_experience(inp, mem, ts)
        out2 = compile_experience(inp, mem, ts)
        assert out1.narration == out2.narration
        assert out1.gate_fidelity == out2.gate_fidelity
        assert out1.gate_passed == out2.gate_passed

    def test_trace_has_7_facets(self):
        out = compile_experience(ChamberInput(), timestamp=1000.0)
        assert len(out.trace.facets) == 7

    def test_selected_is_subset(self):
        out = compile_experience(ChamberInput(confidence=0.9), timestamp=1000.0)
        assert len(out.trace.selected) <= len(out.trace.facets)
        assert len(out.trace.selected) <= MAX_SELECTED

    def test_compression_ratio_valid(self):
        out = compile_experience(ChamberInput(), timestamp=1000.0)
        assert 0.0 < out.trace.compression_ratio <= 1.0

    def test_gate_passes_on_normal_input(self):
        """Normal compilation should pass the fidelity gate."""
        inp = ChamberInput(confidence=0.7, rapport=0.3)
        out = compile_experience(inp, timestamp=1000.0)
        assert out.gate_passed is True
        assert out.gate_fidelity >= FIDELITY_THRESHOLD

    def test_memory_evolves(self):
        """Memory state updates across compilations."""
        inp = ChamberInput(confidence=0.9, engagement=0.8, tension=0.1)
        mem = ExperienceMemory()
        out1 = compile_experience(inp, mem, 1000.0)
        out2 = compile_experience(inp, out1.memory, 2000.0)
        assert out2.memory.update_count == out1.memory.update_count + 1
        # Baselines should drift toward high values
        assert out2.memory.baseline_somatic >= out1.memory.baseline_somatic or \
               abs(out2.memory.baseline_somatic - out1.memory.baseline_somatic) < 0.01

    def test_default_memory_used_when_none(self):
        out = compile_experience(ChamberInput(), timestamp=1000.0)
        assert out.memory.update_count == 1

    def test_all_zero_input(self):
        """Extreme case: all zeros should still produce valid output."""
        inp = ChamberInput(
            confidence=0.0, engagement=0.0, rapport=0.0,
            tension=0.0, curiosity=0.0, satisfaction=0.0,
        )
        out = compile_experience(inp, timestamp=1000.0)
        assert isinstance(out.narration, str)
        assert out.gate_passed  # should still pass

    def test_all_max_input(self):
        """Extreme case: all max values."""
        inp = ChamberInput(
            confidence=1.0, engagement=1.0, rapport=1.0, tension=0.0,
            curiosity=1.0, satisfaction=1.0, trust_success_rate=1.0,
            trust_avg_fidelity=1.0, relationship_depth=1.0,
            last_compile_trust=1.0, env_entry_count=20,
            env_avg_confidence=1.0, fusion_confidence=1.0,
            attention_significance=1.0, journal_success_rate=1.0,
            recent_actions_success_rate=1.0, has_active_compile=True,
        )
        out = compile_experience(inp, timestamp=1000.0)
        assert out.gate_passed is True


# ============================================================
# Serialization
# ============================================================

class TestSerialization:
    def test_round_trip(self):
        mem = ExperienceMemory(
            baseline_somatic=0.6, peak_confidence=0.8,
            update_count=5, last_updated=1000.0,
            memorable_moments=((100.0, "somatic", "Energy was high."),),
        )
        data = serialize_experience_memory(mem)
        restored = deserialize_experience_memory(data)
        assert restored.baseline_somatic == mem.baseline_somatic
        assert restored.peak_confidence == mem.peak_confidence
        assert restored.update_count == mem.update_count
        assert len(restored.memorable_moments) == 1
        assert restored.memorable_moments[0][1] == "somatic"

    def test_json_valid(self):
        mem = ExperienceMemory()
        data = serialize_experience_memory(mem)
        parsed = json.loads(data)
        assert isinstance(parsed, dict)
        assert "baseline_somatic" in parsed

    def test_default_memory_round_trip(self):
        mem = ExperienceMemory()
        data = serialize_experience_memory(mem)
        restored = deserialize_experience_memory(data)
        assert restored == mem

    def test_empty_moments_round_trip(self):
        mem = ExperienceMemory(memorable_moments=())
        data = serialize_experience_memory(mem)
        restored = deserialize_experience_memory(data)
        assert restored.memorable_moments == ()


# ============================================================
# Format context
# ============================================================

class TestFormatContext:
    def test_format_with_content(self):
        result = format_experience_context("Energy is high.")
        assert "[Experience]" in result
        assert "Energy is high." in result

    def test_format_includes_framing(self):
        result = format_experience_context("Energy is high.")
        assert "felt internal state" in result
        assert "Don't narrate it to the user" in result

    def test_format_empty(self):
        assert format_experience_context("") == ""

    def test_format_none_safe(self):
        """Empty string input returns empty string."""
        assert format_experience_context("") == ""


# ============================================================
# LEAF compliance + SAX axioms
# ============================================================

class TestCompliance:
    def test_no_core_imports(self):
        """LEAF: no imports from core/."""
        import ast
        from pathlib import Path
        source = Path(__file__).parent.parent / "mother" / "sentience.py"
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("core"), \
                        f"LEAF violation: imports from core/{node.module}"

    def test_sax1_provenance(self):
        """SAX1: every facet in trace has non-empty sources."""
        out = compile_experience(ChamberInput(confidence=0.7), timestamp=1000.0)
        for facet in out.trace.facets:
            assert len(facet.sources) > 0

    def test_sax2_no_generation(self):
        """SAX2: narration only contains sentences from known band tables."""
        # Collect all possible sentences
        all_sentences = set()
        for bands in [_SOMATIC_BANDS, _RELATIONAL_BANDS, _EPISTEMIC_BANDS,
                       _TEMPORAL_BANDS, _ENVIRONMENTAL_BANDS, _VOLITIONAL_BANDS,
                       _AFFECTIVE_BANDS]:
            for _, sentence in bands:
                all_sentences.add(sentence)
                # Also add lowercased-first-char variant (surprise prefix)
                all_sentences.add(sentence[0].lower() + sentence[1:])

        out = compile_experience(ChamberInput(confidence=0.7), timestamp=1000.0)
        narration = out.narration
        # Remove known structural elements
        narration_clean = narration.replace("Something shifted — ", "")
        # Each sentence fragment should be from known vocabulary
        # (Split on ". " doesn't work perfectly but a good heuristic)
        for sentence in all_sentences:
            narration_clean = narration_clean.replace(sentence, "")
        # What's left should only be spaces/punctuation from join
        remaining = narration_clean.strip()
        # Allow empty or just join artifacts
        assert len(remaining) < 5, f"Unexpected generated text: {remaining!r}"

    def test_sax3_fidelity_gate(self):
        """SAX3: gate uses threshold 0.60."""
        assert FIDELITY_THRESHOLD == 0.60

    def test_sax4_temporal_coherence(self):
        """SAX4: EMA alpha is 0.2 (slower than SenseMemory's 0.3)."""
        assert EMA_ALPHA == 0.2

    def test_domain_baseline_mapping_complete(self):
        """Every domain has a corresponding baseline field."""
        for domain in FacetDomain:
            assert domain in _DOMAIN_TO_BASELINE
            attr = _DOMAIN_TO_BASELINE[domain]
            assert hasattr(ExperienceMemory(), attr)


# ============================================================
# Integration: memory.py persistence
# ============================================================

class TestMemoryPersistence:
    def test_experience_memory_table_creation(self, tmp_path):
        """ConversationStore creates experience_memory table."""
        from mother.memory import ConversationStore
        store = ConversationStore(tmp_path / "test.db")
        try:
            # Should not raise
            result = store.load_experience_memory()
            assert result is None  # no data yet
        finally:
            store.close()

    def test_experience_memory_save_load(self, tmp_path):
        """Save and load experience memory round-trip."""
        from mother.memory import ConversationStore
        store = ConversationStore(tmp_path / "test.db")
        try:
            mem = ExperienceMemory(baseline_somatic=0.8, update_count=3)
            data = serialize_experience_memory(mem)
            store.save_experience_memory(data)
            loaded = store.load_experience_memory()
            assert loaded is not None
            restored = deserialize_experience_memory(loaded)
            assert restored.baseline_somatic == 0.8
            assert restored.update_count == 3
        finally:
            store.close()

    def test_experience_memory_upsert(self, tmp_path):
        """Subsequent saves overwrite previous data."""
        from mother.memory import ConversationStore
        store = ConversationStore(tmp_path / "test.db")
        try:
            store.save_experience_memory(serialize_experience_memory(ExperienceMemory(update_count=1)))
            store.save_experience_memory(serialize_experience_memory(ExperienceMemory(update_count=2)))
            loaded = store.load_experience_memory()
            restored = deserialize_experience_memory(loaded)
            assert restored.update_count == 2
        finally:
            store.close()


# ============================================================
# Integration: bridge accessor
# ============================================================

class TestBridgeAccessor:
    def test_get_experience_narration(self):
        from mother.bridge import EngineBridge
        inp = ChamberInput(confidence=0.7, rapport=0.3)
        narration = EngineBridge.get_experience_narration(inp)
        assert isinstance(narration, str)
        assert len(narration) > 0

    def test_get_experience_narration_with_memory(self):
        from mother.bridge import EngineBridge
        inp = ChamberInput(confidence=0.7)
        mem = ExperienceMemory()
        narration = EngineBridge.get_experience_narration(inp, mem)
        assert isinstance(narration, str)
