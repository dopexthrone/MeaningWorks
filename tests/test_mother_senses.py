"""
Tests for Mother senses — intrinsic emotional state machine.

Covers: SenseVector computation, Posture derivation, Memory EMA smoothing,
serialization roundtrip, edge cases, integration with system prompt.
"""

import json
import time

import pytest

from mother.senses import (
    SenseObservations,
    SenseVector,
    SenseMemory,
    Posture,
    POSTURE_LABELS,
    EMA_ALPHA,
    compute_senses,
    compute_posture,
    update_memory,
    render_sense_block,
    select_personality_blend,
    serialize_memory,
    deserialize_memory,
    _clamp,
)


# ============================================================
# SenseVector basics
# ============================================================

class TestSenseVector:

    def test_default_values(self):
        v = SenseVector()
        assert v.confidence == 0.5
        assert v.rapport == 0.0
        assert v.curiosity == 0.3
        assert v.vitality == 1.0
        assert v.attentiveness == 0.5

    def test_frozen(self):
        v = SenseVector()
        with pytest.raises(AttributeError):
            v.confidence = 0.9

    def test_mean(self):
        v = SenseVector(confidence=1.0, rapport=1.0, curiosity=1.0, vitality=1.0, attentiveness=1.0, frustration=1.0)
        assert v.mean() == 1.0

    def test_mean_mixed(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.5, vitality=0.5, attentiveness=0.5, frustration=0.5)
        assert abs(v.mean() - 0.5) < 1e-9

    def test_mean_zero(self):
        v = SenseVector(confidence=0.0, rapport=0.0, curiosity=0.0, vitality=0.0, attentiveness=0.0, frustration=0.0)
        assert v.mean() == 0.0

    def test_frustration_default(self):
        v = SenseVector()
        assert v.frustration == 0.0


# ============================================================
# SenseObservations
# ============================================================

class TestSenseObservations:

    def test_default_values(self):
        obs = SenseObservations()
        assert obs.compile_count == 0
        assert obs.cost_limit == 5.0

    def test_frozen(self):
        obs = SenseObservations()
        with pytest.raises(AttributeError):
            obs.compile_count = 5


# ============================================================
# SenseMemory
# ============================================================

class TestSenseMemory:

    def test_default_values(self):
        m = SenseMemory()
        assert m.update_count == 0
        assert m.peak_confidence == 0.5

    def test_frozen(self):
        m = SenseMemory()
        with pytest.raises(AttributeError):
            m.update_count = 99


# ============================================================
# Posture
# ============================================================

class TestPosture:

    def test_default_values(self):
        p = Posture()
        assert p.state_label == "steady"
        assert p.voice_pace == 1.15

    def test_frozen(self):
        p = Posture()
        with pytest.raises(AttributeError):
            p.state_label = "focused"

    def test_posture_labels_constant(self):
        assert "focused" in POSTURE_LABELS
        assert "concerned" in POSTURE_LABELS
        assert "attentive" in POSTURE_LABELS
        assert "energized" in POSTURE_LABELS
        assert "steady" in POSTURE_LABELS


# ============================================================
# _clamp helper
# ============================================================

class TestClamp:

    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_min(self):
        assert _clamp(-0.5) == 0.0

    def test_above_max(self):
        assert _clamp(1.5) == 1.0

    def test_custom_range(self):
        assert _clamp(-2.0, -1.0, 1.0) == -1.0
        assert _clamp(2.0, -1.0, 1.0) == 1.0


# ============================================================
# compute_senses
# ============================================================

class TestComputeSenses:

    def test_zero_observations(self):
        """No data = neutral baseline."""
        obs = SenseObservations()
        v = compute_senses(obs)
        assert 0.0 <= v.confidence <= 1.0
        assert 0.0 <= v.rapport <= 1.0
        assert 0.0 <= v.curiosity <= 1.0
        assert 0.0 <= v.vitality <= 1.0
        assert 0.0 <= v.attentiveness <= 1.0

    def test_all_values_clamped(self):
        """All senses stay in 0.0–1.0 regardless of extreme input."""
        obs = SenseObservations(
            compile_count=1000,
            compile_success_count=1000,
            last_trust_score=100.0,
            session_error_count=0,
            total_sessions=100,
            total_messages=10000,
            messages_this_session=500,
            sessions_last_7_days=50,
            avg_user_message_length=5000.0,
            unique_topic_count=100,
            session_cost=0.0,
            cost_limit=100.0,
            session_duration_minutes=10.0,
        )
        v = compute_senses(obs)
        for field_name in ("confidence", "rapport", "curiosity", "vitality", "attentiveness"):
            val = getattr(v, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name} = {val} out of range"

    def test_high_success_rate_boosts_confidence(self):
        obs = SenseObservations(
            compile_count=10,
            compile_success_count=10,
            last_trust_score=90.0,
        )
        v = compute_senses(obs)
        assert v.confidence >= 0.7

    def test_low_success_rate_drops_confidence(self):
        obs = SenseObservations(
            compile_count=10,
            compile_success_count=2,
            last_trust_score=20.0,
            session_error_count=3,
        )
        v = compute_senses(obs)
        assert v.confidence < 0.5

    def test_errors_reduce_confidence(self):
        good = SenseObservations(session_error_count=0)
        bad = SenseObservations(session_error_count=5)
        v_good = compute_senses(good)
        v_bad = compute_senses(bad)
        assert v_good.confidence > v_bad.confidence

    def test_many_sessions_build_rapport(self):
        obs = SenseObservations(total_sessions=20, total_messages=500)
        v = compute_senses(obs)
        assert v.rapport >= 0.5

    def test_no_sessions_zero_rapport(self):
        obs = SenseObservations(total_sessions=0)
        v = compute_senses(obs)
        assert v.rapport == 0.0

    def test_recent_return_boosts_rapport(self):
        obs_recent = SenseObservations(total_sessions=5, days_since_last_session=0.5)
        obs_old = SenseObservations(total_sessions=5, days_since_last_session=10.0)
        v_recent = compute_senses(obs_recent)
        v_old = compute_senses(obs_old)
        assert v_recent.rapport > v_old.rapport

    def test_compiles_boost_curiosity(self):
        obs = SenseObservations(compile_count=5, unique_topic_count=5)
        v = compute_senses(obs)
        assert v.curiosity >= 0.5

    def test_cost_near_limit_drops_vitality(self):
        obs = SenseObservations(session_cost=4.5, cost_limit=5.0)
        v = compute_senses(obs)
        assert v.vitality < 0.3

    def test_cost_zero_full_vitality(self):
        obs = SenseObservations(session_cost=0.0, cost_limit=5.0)
        v = compute_senses(obs)
        assert v.vitality >= 0.8

    def test_errors_reduce_vitality(self):
        obs_clean = SenseObservations(session_error_count=0)
        obs_errors = SenseObservations(session_error_count=5)
        v_clean = compute_senses(obs_clean)
        v_errors = compute_senses(obs_errors)
        assert v_clean.vitality > v_errors.vitality

    def test_conversation_depth_boosts_attentiveness(self):
        obs = SenseObservations(messages_this_session=15, avg_user_message_length=150.0)
        v = compute_senses(obs)
        assert v.attentiveness >= 0.7

    def test_no_messages_moderate_attentiveness(self):
        """Mother always pays some attention even with no messages."""
        obs = SenseObservations(messages_this_session=0)
        v = compute_senses(obs)
        assert v.attentiveness >= 0.3

    def test_accepts_memory_param(self):
        """Memory param accepted (for future trajectory influence)."""
        obs = SenseObservations()
        mem = SenseMemory()
        v = compute_senses(obs, memory=mem)
        assert isinstance(v, SenseVector)

    def test_session_frequency_boosts_rapport(self):
        obs = SenseObservations(total_sessions=5, sessions_last_7_days=5)
        v = compute_senses(obs)
        obs_none = SenseObservations(total_sessions=5, sessions_last_7_days=0)
        v_none = compute_senses(obs_none)
        assert v.rapport > v_none.rapport

    def test_long_session_reduces_vitality(self):
        obs_short = SenseObservations(session_duration_minutes=10.0)
        obs_long = SenseObservations(session_duration_minutes=120.0)
        v_short = compute_senses(obs_short)
        v_long = compute_senses(obs_long)
        assert v_short.vitality > v_long.vitality

    def test_topic_diversity_drives_curiosity(self):
        obs_diverse = SenseObservations(unique_topic_count=10)
        obs_single = SenseObservations(unique_topic_count=1)
        v_diverse = compute_senses(obs_diverse)
        v_single = compute_senses(obs_single)
        assert v_diverse.curiosity > v_single.curiosity

    def test_zero_cost_limit(self):
        """cost_limit=0 should not crash."""
        obs = SenseObservations(cost_limit=0.0, session_cost=1.0)
        v = compute_senses(obs)
        assert v.vitality >= 0.1  # Clamped minimum


# ============================================================
# compute_posture
# ============================================================

class TestComputePosture:

    def test_high_confidence_vitality_gives_focused(self):
        v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.3, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.state_label == "focused"

    def test_high_curiosity_gives_energized(self):
        v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.7, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.state_label == "energized"

    def test_low_confidence_gives_concerned(self):
        v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.state_label == "concerned"

    def test_low_vitality_gives_concerned(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.5, vitality=0.2, attentiveness=0.6)
        p = compute_posture(v)
        assert p.state_label == "concerned"

    def test_new_user_gives_attentive(self):
        v = SenseVector(confidence=0.5, rapport=0.1, curiosity=0.3, vitality=0.7, attentiveness=0.5)
        p = compute_posture(v)
        assert p.state_label == "attentive"

    def test_default_gives_steady(self):
        v = SenseVector(confidence=0.5, rapport=0.3, curiosity=0.3, vitality=0.7, attentiveness=0.5)
        p = compute_posture(v)
        assert p.state_label == "steady"

    def test_weights_sum_to_one(self):
        v = SenseVector(confidence=0.7, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        total = p.weight_composed + p.weight_warm + p.weight_direct + p.weight_playful
        assert abs(total - 1.0) < 0.01

    def test_high_conf_rapport_boosts_playful(self):
        v = SenseVector(confidence=0.8, rapport=0.7, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.weight_playful > 0.25  # Above baseline

    def test_low_conf_boosts_composed(self):
        v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.weight_composed > p.weight_playful

    def test_new_user_boosts_direct(self):
        v = SenseVector(confidence=0.5, rapport=0.1, curiosity=0.3, vitality=0.7, attentiveness=0.5)
        p = compute_posture(v)
        assert p.weight_direct > 0.25

    def test_high_rapport_boosts_warm(self):
        v = SenseVector(confidence=0.5, rapport=0.7, curiosity=0.3, vitality=0.7, attentiveness=0.5)
        p = compute_posture(v)
        assert p.weight_warm > 0.25

    def test_low_vitality_boosts_direct(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.3, vitality=0.2, attentiveness=0.5)
        p = compute_posture(v)
        assert p.weight_direct > 0.25

    def test_energized_faster_voice(self):
        v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.7, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.voice_pace > 1.15

    def test_concerned_slower_voice(self):
        v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.voice_pace < 1.15

    def test_proactive_flag(self):
        v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.proactive is True

    def test_not_proactive_low_confidence(self):
        v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.proactive is False

    def test_encouraging_flag(self):
        v = SenseVector(confidence=0.4, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.encouraging is True

    def test_cautious_flag_low_confidence(self):
        v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.cautious is True

    def test_cautious_flag_low_vitality(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.5, vitality=0.2, attentiveness=0.6)
        p = compute_posture(v)
        assert p.cautious is True

    def test_abbreviated_flag_low_vitality(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.5, vitality=0.2, attentiveness=0.6)
        p = compute_posture(v)
        assert p.abbreviated is True

    def test_not_abbreviated_healthy(self):
        v = SenseVector(confidence=0.5, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        p = compute_posture(v)
        assert p.abbreviated is False

    def test_summary_matches_label(self):
        for label in POSTURE_LABELS:
            # Build a sense vector that triggers the label
            if label == "focused":
                v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.3, vitality=0.8, attentiveness=0.6)
            elif label == "energized":
                v = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.7, vitality=0.8, attentiveness=0.6)
            elif label == "concerned":
                v = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
            elif label == "attentive":
                v = SenseVector(confidence=0.5, rapport=0.1, curiosity=0.3, vitality=0.7, attentiveness=0.5)
            else:  # steady
                v = SenseVector(confidence=0.5, rapport=0.3, curiosity=0.3, vitality=0.7, attentiveness=0.5)
            p = compute_posture(v)
            assert p.summary  # Non-empty
            assert isinstance(p.summary, str)


# ============================================================
# update_memory
# ============================================================

class TestUpdateMemory:

    def test_first_update_no_previous(self):
        v = SenseVector(confidence=0.8, rapport=0.3, curiosity=0.5, vitality=0.9, attentiveness=0.6)
        m = update_memory(v, previous=None, timestamp=100.0)
        assert m.baseline_confidence == 0.8
        assert m.baseline_rapport == 0.3
        assert m.peak_confidence == 0.8
        assert m.update_count == 1
        assert m.last_updated == 100.0

    def test_ema_smoothing(self):
        prev = SenseMemory(baseline_confidence=0.5, update_count=1, last_updated=100.0)
        v = SenseVector(confidence=1.0)
        m = update_memory(v, previous=prev, timestamp=200.0)
        # EMA: 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        assert abs(m.baseline_confidence - 0.65) < 0.001

    def test_ema_smoothing_drops(self):
        prev = SenseMemory(baseline_confidence=0.8, update_count=5, last_updated=100.0)
        v = SenseVector(confidence=0.2)
        m = update_memory(v, previous=prev, timestamp=200.0)
        # EMA: 0.3 * 0.2 + 0.7 * 0.8 = 0.62
        assert abs(m.baseline_confidence - 0.62) < 0.001

    def test_single_failure_doesnt_crater(self):
        """A single bad compile shouldn't crater confidence if baseline is high."""
        prev = SenseMemory(baseline_confidence=0.85, update_count=20, last_updated=100.0)
        v = SenseVector(confidence=0.2)
        m = update_memory(v, previous=prev, timestamp=200.0)
        # Baseline should still be above 0.5
        assert m.baseline_confidence > 0.5

    def test_update_count_increments(self):
        prev = SenseMemory(update_count=5, last_updated=100.0)
        v = SenseVector()
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.update_count == 6

    def test_peak_confidence_tracked(self):
        prev = SenseMemory(peak_confidence=0.7, update_count=1, last_updated=100.0)
        v = SenseVector(confidence=0.9)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.peak_confidence == 0.9

    def test_peak_confidence_preserved(self):
        prev = SenseMemory(peak_confidence=0.9, update_count=1, last_updated=100.0)
        v = SenseVector(confidence=0.5)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.peak_confidence == 0.9

    def test_peak_rapport_tracked(self):
        prev = SenseMemory(peak_rapport=0.3, update_count=1, last_updated=100.0)
        v = SenseVector(rapport=0.6)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.peak_rapport == 0.6

    def test_confidence_trend_positive(self):
        prev = SenseMemory(baseline_confidence=0.5, update_count=5, last_updated=100.0)
        v = SenseVector(confidence=0.9)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.confidence_trend > 0

    def test_confidence_trend_negative(self):
        prev = SenseMemory(baseline_confidence=0.8, update_count=5, last_updated=100.0)
        v = SenseVector(confidence=0.3)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.confidence_trend < 0

    def test_rapport_trend(self):
        prev = SenseMemory(baseline_rapport=0.2, update_count=5, last_updated=100.0)
        v = SenseVector(rapport=0.7)
        m = update_memory(v, previous=prev, timestamp=200.0)
        assert m.rapport_trend > 0

    def test_multiple_updates_converge(self):
        """Repeated same observation converges baseline toward it."""
        mem = None
        target = SenseVector(confidence=0.9, rapport=0.8, curiosity=0.7, vitality=0.9, attentiveness=0.8)
        for i in range(20):
            mem = update_memory(target, previous=mem, timestamp=100.0 + i)
        assert abs(mem.baseline_confidence - 0.9) < 0.05

    def test_timestamp_default(self):
        v = SenseVector()
        m = update_memory(v, previous=None)
        assert m.last_updated > 0


# ============================================================
# render_sense_block
# ============================================================

class TestRenderSenseBlock:

    def test_starts_with_stance(self):
        v = SenseVector(confidence=0.82, rapport=0.61, curiosity=0.45, vitality=0.88, attentiveness=0.70)
        p = compute_posture(v)
        block = render_sense_block(p, v)
        assert block.startswith("Stance:")

    def test_contains_behavioral_directives_not_numbers(self):
        """Should contain behavioral text, not raw sense values."""
        v = SenseVector(confidence=0.82, rapport=0.61, curiosity=0.45, vitality=0.88, attentiveness=0.70)
        p = compute_posture(v)
        block = render_sense_block(p, v)
        # Should NOT contain dashboard-style numbers
        assert "confidence 0.82" not in block
        assert "rapport 0.61" not in block
        # Should contain behavioral language
        assert any(word in block.lower() for word in ["decisive", "locked", "trust", "competence", "momentum", "balanced", "careful", "precise", "attention"])

    def test_different_postures_produce_different_directives(self):
        """Focused vs concerned should produce clearly different text."""
        v_focused = SenseVector(confidence=0.82, rapport=0.61, curiosity=0.45, vitality=0.88, attentiveness=0.70)
        p_focused = compute_posture(v_focused)
        block_focused = render_sense_block(p_focused, v_focused)

        v_concerned = SenseVector(confidence=0.25, rapport=0.1, curiosity=0.2, vitality=0.2, attentiveness=0.5)
        p_concerned = compute_posture(v_concerned)
        block_concerned = render_sense_block(p_concerned, v_concerned)

        assert block_focused != block_concerned

    def test_new_user_gets_trust_earning_directive(self):
        """Rapport < 0.15 should include directive about earning trust."""
        v = SenseVector(confidence=0.5, rapport=0.05, curiosity=0.3, vitality=0.8, attentiveness=0.4)
        p = compute_posture(v)
        block = render_sense_block(p, v)
        assert "trust" in block.lower()

    def test_high_rapport_gets_partner_directive(self):
        """High rapport should include partner/challenge language."""
        v = SenseVector(confidence=0.7, rapport=0.8, curiosity=0.5, vitality=0.8, attentiveness=0.7)
        p = compute_posture(v)
        block = render_sense_block(p, v)
        assert any(word in block.lower() for word in ["partner", "challenge", "trust", "direct"])


# ============================================================
# select_personality_blend
# ============================================================

class TestSelectPersonalityBlend:

    def test_returns_valid_personality(self):
        p = Posture()
        result = select_personality_blend(p)
        assert result in ("composed", "warm", "direct", "playful")

    def test_highest_weight_wins(self):
        p = Posture(weight_composed=0.1, weight_warm=0.1, weight_direct=0.1, weight_playful=0.7)
        assert select_personality_blend(p) == "playful"

    def test_composed_dominant(self):
        p = Posture(weight_composed=0.7, weight_warm=0.1, weight_direct=0.1, weight_playful=0.1)
        assert select_personality_blend(p) == "composed"

    def test_warm_dominant(self):
        p = Posture(weight_composed=0.1, weight_warm=0.7, weight_direct=0.1, weight_playful=0.1)
        assert select_personality_blend(p) == "warm"

    def test_direct_dominant(self):
        p = Posture(weight_composed=0.1, weight_warm=0.1, weight_direct=0.7, weight_playful=0.1)
        assert select_personality_blend(p) == "direct"


# ============================================================
# Serialization roundtrip
# ============================================================

class TestSerialization:

    def test_roundtrip(self):
        m = SenseMemory(
            baseline_confidence=0.75,
            baseline_rapport=0.4,
            peak_confidence=0.9,
            update_count=10,
            last_updated=12345.0,
        )
        data = serialize_memory(m)
        restored = deserialize_memory(data)
        assert restored == m

    def test_json_valid(self):
        m = SenseMemory()
        data = serialize_memory(m)
        parsed = json.loads(data)
        assert isinstance(parsed, dict)
        assert "baseline_confidence" in parsed

    def test_all_fields_preserved(self):
        m = SenseMemory(
            baseline_confidence=0.1,
            baseline_rapport=0.2,
            baseline_curiosity=0.3,
            baseline_vitality=0.4,
            baseline_attentiveness=0.5,
            confidence_trend=0.1,
            rapport_trend=-0.2,
            peak_confidence=0.9,
            peak_rapport=0.8,
            last_updated=999.0,
            update_count=42,
        )
        restored = deserialize_memory(serialize_memory(m))
        assert restored.baseline_confidence == 0.1
        assert restored.rapport_trend == -0.2
        assert restored.peak_rapport == 0.8
        assert restored.update_count == 42

    def test_default_roundtrip(self):
        m = SenseMemory()
        assert deserialize_memory(serialize_memory(m)) == m


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_all_errors(self):
        obs = SenseObservations(
            compile_count=10,
            compile_success_count=0,
            last_trust_score=0.0,
            session_error_count=10,
            session_cost=4.9,
            cost_limit=5.0,
            session_duration_minutes=180.0,
        )
        v = compute_senses(obs)
        p = compute_posture(v)
        assert p.state_label == "concerned"
        assert p.cautious is True
        assert v.confidence < 0.3
        assert v.vitality < 0.3

    def test_max_observations(self):
        obs = SenseObservations(
            compile_count=100,
            compile_success_count=100,
            last_trust_score=100.0,
            session_error_count=0,
            total_sessions=50,
            total_messages=5000,
            messages_this_session=100,
            sessions_last_7_days=7,
            avg_user_message_length=300.0,
            unique_topic_count=30,
            session_cost=0.01,
            cost_limit=100.0,
            session_duration_minutes=30.0,
        )
        v = compute_senses(obs)
        p = compute_posture(v)
        assert v.confidence >= 0.8
        assert v.rapport >= 0.5
        assert p.state_label in ("focused", "energized")

    def test_first_session_profile(self):
        """Brand new user — zero everything."""
        obs = SenseObservations(
            total_sessions=0,
            total_messages=0,
            messages_this_session=1,
        )
        v = compute_senses(obs)
        p = compute_posture(v)
        assert p.state_label == "attentive"  # New user

    def test_returning_power_user(self):
        obs = SenseObservations(
            compile_count=20,
            compile_success_count=18,
            last_trust_score=85.0,
            total_sessions=30,
            total_messages=2000,
            messages_this_session=5,
            days_since_last_session=0.1,
            sessions_last_7_days=5,
            unique_topic_count=8,
            session_cost=0.5,
            cost_limit=10.0,
        )
        v = compute_senses(obs)
        assert v.confidence >= 0.7
        assert v.rapport >= 0.5

    # --- Emotional ramp tests ---

    def test_rapport_builds_within_first_session(self):
        """Rapport should start rising after a few messages in first session."""
        obs_start = SenseObservations(
            total_sessions=0, total_messages=0, messages_this_session=1,
        )
        obs_mid = SenseObservations(
            total_sessions=0, total_messages=0, messages_this_session=10,
        )
        v_start = compute_senses(obs_start)
        v_mid = compute_senses(obs_mid)
        assert v_mid.rapport > v_start.rapport
        # After 10 messages, should have crossed the "attentive" threshold
        assert v_mid.rapport >= 0.12

    def test_first_compile_boosts_curiosity(self):
        """First compile should visibly increase curiosity."""
        obs_before = SenseObservations(
            messages_this_session=5, compile_count=0,
        )
        obs_after = SenseObservations(
            messages_this_session=5, compile_count=1, compile_success_count=1,
            last_trust_score=70.0,
        )
        v_before = compute_senses(obs_before)
        v_after = compute_senses(obs_after)
        assert v_after.curiosity > v_before.curiosity
        # Should be a meaningful jump, not just noise
        assert v_after.curiosity - v_before.curiosity >= 0.1

    def test_posture_shifts_from_attentive_to_steady_in_first_session(self):
        """Within one session, posture should transition from attentive to steady."""
        obs_start = SenseObservations(
            total_sessions=0, messages_this_session=1,
        )
        obs_mid = SenseObservations(
            total_sessions=0, messages_this_session=15,
            unique_topic_count=2,
        )
        p_start = compute_posture(compute_senses(obs_start))
        p_mid = compute_posture(compute_senses(obs_mid))
        assert p_start.state_label == "attentive"
        assert p_mid.state_label != "attentive"  # Should have shifted

    def test_returning_next_day_gets_rapport_bonus(self):
        """Returning within a day should give a meaningful rapport boost."""
        obs_new = SenseObservations(
            total_sessions=1, total_messages=10, messages_this_session=1,
        )
        obs_returning = SenseObservations(
            total_sessions=1, total_messages=10, messages_this_session=1,
            days_since_last_session=0.5,
        )
        v_new = compute_senses(obs_new)
        v_ret = compute_senses(obs_returning)
        assert v_ret.rapport > v_new.rapport
        assert v_ret.rapport - v_new.rapport >= 0.1


# ============================================================
# Integration: sense block in system prompt
# ============================================================

class TestIntegration:

    def test_sense_block_can_append_to_introspection(self):
        """Verify sense block format is compatible with render_introspection_block."""
        obs = SenseObservations(
            compile_count=5,
            compile_success_count=4,
            last_trust_score=75.0,
            total_sessions=10,
            total_messages=200,
            messages_this_session=8,
        )
        v = compute_senses(obs)
        p = compute_posture(v)
        block = render_sense_block(p, v)

        # Simulating what persona.py would do
        introspection_lines = ["[Self-observation]", "Identity: Mother | composed"]
        introspection_lines.append(block)
        full_block = "\n".join(introspection_lines)

        assert "[Self-observation]" in full_block
        assert "Stance:" in full_block

    def test_personality_blend_matches_modifier_keys(self):
        """select_personality_blend returns keys that exist in PERSONALITY_MODIFIERS."""
        from mother.persona import PERSONALITY_MODIFIERS
        for conf in [0.2, 0.5, 0.8]:
            for rapp in [0.1, 0.5, 0.8]:
                v = SenseVector(confidence=conf, rapport=rapp, curiosity=0.5, vitality=0.7, attentiveness=0.5)
                p = compute_posture(v)
                blend = select_personality_blend(p)
                assert blend in PERSONALITY_MODIFIERS

    def test_full_pipeline_deterministic(self):
        """Same observations always produce same output."""
        obs = SenseObservations(
            compile_count=5, compile_success_count=4,
            last_trust_score=75.0, total_sessions=10,
        )
        v1 = compute_senses(obs)
        v2 = compute_senses(obs)
        assert v1 == v2

        p1 = compute_posture(v1)
        p2 = compute_posture(v2)
        assert p1 == p2

    def test_posture_to_greeting_mapping(self):
        """Different postures produce different summaries via compute_posture."""
        v_concerned = SenseVector(confidence=0.3, rapport=0.5, curiosity=0.5, vitality=0.8, attentiveness=0.6)
        v_energized = SenseVector(confidence=0.8, rapport=0.5, curiosity=0.7, vitality=0.8, attentiveness=0.6)
        p_concerned = compute_posture(v_concerned)
        p_energized = compute_posture(v_energized)
        assert p_concerned.summary != p_energized.summary


# ============================================================
# Neurologis Automatica — new field extensions
# ============================================================

class TestNeurologisFields:

    def test_idle_reduces_attentiveness(self):
        """idle_seconds > 300 should reduce attentiveness by 0.05."""
        base = SenseObservations(messages_this_session=5)
        idle = SenseObservations(messages_this_session=5, idle_seconds=400)
        v_base = compute_senses(base)
        v_idle = compute_senses(idle)
        assert v_idle.attentiveness < v_base.attentiveness
        assert abs(v_base.attentiveness - v_idle.attentiveness - 0.05) < 0.001

    def test_tempo_boosts_curiosity(self):
        """conversation_tempo > 3.0 should boost curiosity by 0.05."""
        base = SenseObservations()
        fast = SenseObservations(conversation_tempo=4.0)
        v_base = compute_senses(base)
        v_fast = compute_senses(fast)
        assert v_fast.curiosity > v_base.curiosity
        assert abs(v_fast.curiosity - v_base.curiosity - 0.05) < 0.001

    def test_attention_drains_vitality(self):
        """attention_load > 0.7 should reduce vitality by 0.05."""
        base = SenseObservations()
        loaded = SenseObservations(attention_load=0.9)
        v_base = compute_senses(base)
        v_loaded = compute_senses(loaded)
        assert v_loaded.vitality < v_base.vitality
        assert abs(v_base.vitality - v_loaded.vitality - 0.05) < 0.001

    def test_memory_boosts_confidence(self):
        """memory_hits > 0 should boost confidence (capped at 0.05)."""
        base = SenseObservations()
        recall = SenseObservations(memory_hits_this_session=5)
        v_base = compute_senses(base)
        v_recall = compute_senses(recall)
        assert v_recall.confidence > v_base.confidence
        # 5/20 = 0.25, min(0.25, 0.05) = 0.05 wait no — min(5/20, 0.05) = min(0.25, 0.05) = 0.05
        assert abs(v_recall.confidence - v_base.confidence - 0.05) < 0.001

    def test_backward_compatibility(self):
        """Existing code without new fields still works (all default to 0)."""
        obs = SenseObservations(
            compile_count=3,
            compile_success_count=2,
            last_trust_score=60.0,
            total_sessions=5,
        )
        v = compute_senses(obs)
        p = compute_posture(v)
        assert 0.0 <= v.confidence <= 1.0
        assert p.state_label in ("focused", "concerned", "attentive", "energized", "steady")


# ============================================================
# Operational awareness (Phase B) — new field extensions
# ============================================================

class TestOperationalAwarenessFields:

    def test_streak_boosts_confidence(self):
        """build_success_streak > 3 should boost confidence by 0.05."""
        base = SenseObservations()
        streak = SenseObservations(build_success_streak=5)
        v_base = compute_senses(base)
        v_streak = compute_senses(streak)
        assert v_streak.confidence > v_base.confidence
        assert abs(v_streak.confidence - v_base.confidence - 0.05) < 0.001

    def test_health_failures_drop_confidence(self):
        """project_health_failures > 0 should drop confidence by 0.05."""
        base = SenseObservations()
        failing = SenseObservations(project_health_failures=2)
        v_base = compute_senses(base)
        v_failing = compute_senses(failing)
        assert v_failing.confidence < v_base.confidence
        assert abs(v_base.confidence - v_failing.confidence - 0.05) < 0.001

    def test_error_severity_drags_vitality(self):
        """error_severity_sum > 1.0 should drag vitality by 0.05."""
        base = SenseObservations()
        errors = SenseObservations(error_severity_sum=1.5)
        v_base = compute_senses(base)
        v_errors = compute_senses(errors)
        assert v_errors.vitality < v_base.vitality
        assert abs(v_base.vitality - v_errors.vitality - 0.05) < 0.001


# ============================================================
# Frustration sense
# ============================================================

class TestFrustration:

    def test_frustration_rises_with_errors(self):
        """High error count + low confidence → high frustration."""
        obs = SenseObservations(
            session_error_count=8,
            messages_this_session=10,
            compile_count=5,
            compile_success_count=0,
            last_trust_score=10.0,
            session_cost=4.0,
            cost_limit=5.0,
            session_duration_minutes=60.0,
        )
        v = compute_senses(obs)
        assert v.frustration >= 0.5

    def test_frustration_low_when_healthy(self):
        """No errors, high confidence → low frustration."""
        obs = SenseObservations(
            session_error_count=0,
            messages_this_session=5,
            compile_count=5,
            compile_success_count=5,
            last_trust_score=90.0,
            session_cost=0.1,
            cost_limit=5.0,
        )
        v = compute_senses(obs)
        assert v.frustration < 0.3

    def test_frustration_zero_at_default(self):
        """Default observations → minimal frustration."""
        obs = SenseObservations()
        v = compute_senses(obs)
        # With defaults: error_density=0, confidence~0.5, msgs=0, vitality=1.0
        # 0.3*0 + 0.3*(1-0.5) + 0.2*0 + 0.2*0 = 0.15
        assert v.frustration < 0.2

    def test_high_frustration_triggers_concerned_posture(self):
        """Frustration >= 0.6 should force concerned posture."""
        v = SenseVector(
            confidence=0.8, rapport=0.5, curiosity=0.5,
            vitality=0.8, attentiveness=0.6, frustration=0.7,
        )
        p = compute_posture(v)
        assert p.state_label == "concerned"
        assert p.encouraging is True

    def test_frustration_in_render_sense_block(self):
        """High frustration should appear in behavioral directives."""
        v = SenseVector(
            confidence=0.5, rapport=0.3, curiosity=0.3,
            vitality=0.7, attentiveness=0.5, frustration=0.65,
        )
        p = compute_posture(v)
        block = render_sense_block(p, v)
        assert "frustrat" in block.lower()

    def test_moderate_frustration_in_render(self):
        """Moderate frustration (0.4-0.6) produces friction message."""
        v = SenseVector(
            confidence=0.5, rapport=0.3, curiosity=0.3,
            vitality=0.7, attentiveness=0.5, frustration=0.45,
        )
        p = compute_posture(v)
        block = render_sense_block(p, v)
        assert "friction" in block.lower()
