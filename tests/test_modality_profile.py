"""Tests for mother/modality_profile.py — per-modality configuration."""

import pytest

from mother.modality_profile import (
    ModalityProfile,
    ModalityBudget,
    default_profiles,
    should_process,
    allocate_budget,
    update_budget_after_event,
    adjust_threshold,
    format_modality_context,
)


# ---------------------------------------------------------------------------
# ModalityProfile dataclass
# ---------------------------------------------------------------------------

class TestModalityProfile:
    def test_frozen(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8)
        with pytest.raises(AttributeError):
            p.reliability = 0.5

    def test_defaults(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8)
        assert p.enabled is True
        assert p.attention_threshold == 0.3


# ---------------------------------------------------------------------------
# ModalityBudget dataclass
# ---------------------------------------------------------------------------

class TestModalityBudget:
    def test_frozen(self):
        b = ModalityBudget(modality="screen", hourly_limit=1.0, events_this_hour=0,
                           cost_this_hour=0.0, remaining=1.0)
        with pytest.raises(AttributeError):
            b.remaining = 0.5


# ---------------------------------------------------------------------------
# default_profiles
# ---------------------------------------------------------------------------

class TestDefaultProfiles:
    def test_returns_three_modalities(self):
        profiles = default_profiles()
        assert set(profiles.keys()) == {"screen", "speech", "camera"}

    def test_screen_profile_values(self):
        p = default_profiles()["screen"]
        assert p.name == "screen"
        assert p.reliability == pytest.approx(0.95)
        assert p.cost_per_event == pytest.approx(0.005)
        assert p.information_density == pytest.approx(0.8)

    def test_speech_profile_values(self):
        p = default_profiles()["speech"]
        assert p.reliability == pytest.approx(0.70)
        assert p.information_density == pytest.approx(0.9)

    def test_camera_profile_values(self):
        p = default_profiles()["camera"]
        assert p.reliability == pytest.approx(0.60)
        assert p.information_density == pytest.approx(0.5)

    def test_all_enabled_by_default(self):
        for p in default_profiles().values():
            assert p.enabled is True

    def test_returns_new_dict_each_call(self):
        a = default_profiles()
        b = default_profiles()
        assert a is not b


# ---------------------------------------------------------------------------
# should_process
# ---------------------------------------------------------------------------

class TestShouldProcess:
    def test_above_threshold_no_budget(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        assert should_process(p, attention_score=0.5) is True

    def test_below_threshold(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        assert should_process(p, attention_score=0.1) is False

    def test_exactly_at_threshold(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        assert should_process(p, attention_score=0.3) is True

    def test_disabled_modality(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, enabled=False)
        assert should_process(p, attention_score=1.0) is False

    def test_budget_exhausted(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8)
        b = ModalityBudget(modality="screen", hourly_limit=0.1, events_this_hour=20,
                           cost_this_hour=0.1, remaining=0.0)
        assert should_process(p, attention_score=0.5, budget=b) is False

    def test_budget_available(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8)
        b = ModalityBudget(modality="screen", hourly_limit=1.0, events_this_hour=5,
                           cost_this_hour=0.025, remaining=0.975)
        assert should_process(p, attention_score=0.5, budget=b) is True


# ---------------------------------------------------------------------------
# allocate_budget
# ---------------------------------------------------------------------------

class TestAllocateBudget:
    def test_proportional_to_density(self):
        profiles = default_profiles()
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        # Speech (0.9) should get more than camera (0.5)
        assert budgets["speech"].hourly_limit > budgets["camera"].hourly_limit

    def test_total_equals_budget(self):
        profiles = default_profiles()
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        total = sum(b.hourly_limit for b in budgets.values())
        assert total == pytest.approx(1.0)

    def test_disabled_gets_zero(self):
        profiles = default_profiles()
        # Disable camera
        profiles["camera"] = ModalityProfile(
            name="camera", reliability=0.6, cost_per_event=0.005,
            latency_ms=300.0, information_density=0.5, enabled=False,
        )
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        assert budgets["camera"].hourly_limit == 0.0
        assert budgets["camera"].remaining == 0.0

    def test_disabled_share_redistributed(self):
        profiles = default_profiles()
        profiles["camera"] = ModalityProfile(
            name="camera", reliability=0.6, cost_per_event=0.005,
            latency_ms=300.0, information_density=0.5, enabled=False,
        )
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        enabled_total = sum(b.hourly_limit for b in budgets.values() if b.hourly_limit > 0)
        assert enabled_total == pytest.approx(1.0)

    def test_all_disabled(self):
        profiles = {
            "screen": ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                                      latency_ms=200.0, information_density=0.8, enabled=False),
        }
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        assert budgets["screen"].hourly_limit == 0.0

    def test_initial_cost_and_events_zero(self):
        profiles = default_profiles()
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        for b in budgets.values():
            assert b.events_this_hour == 0
            assert b.cost_this_hour == 0.0

    def test_remaining_equals_limit(self):
        profiles = default_profiles()
        budgets = allocate_budget(profiles, total_hourly_budget=1.0)
        for b in budgets.values():
            if b.hourly_limit > 0:
                assert b.remaining == pytest.approx(b.hourly_limit)


# ---------------------------------------------------------------------------
# update_budget_after_event
# ---------------------------------------------------------------------------

class TestUpdateBudgetAfterEvent:
    def test_increments_count(self):
        b = ModalityBudget(modality="screen", hourly_limit=1.0, events_this_hour=5,
                           cost_this_hour=0.025, remaining=0.975)
        b2 = update_budget_after_event(b, cost=0.005)
        assert b2.events_this_hour == 6

    def test_adds_cost(self):
        b = ModalityBudget(modality="screen", hourly_limit=1.0, events_this_hour=0,
                           cost_this_hour=0.0, remaining=1.0)
        b2 = update_budget_after_event(b, cost=0.1)
        assert b2.cost_this_hour == pytest.approx(0.1)
        assert b2.remaining == pytest.approx(0.9)

    def test_remaining_floors_at_zero(self):
        b = ModalityBudget(modality="screen", hourly_limit=0.01, events_this_hour=1,
                           cost_this_hour=0.005, remaining=0.005)
        b2 = update_budget_after_event(b, cost=0.01)
        assert b2.remaining == 0.0

    def test_returns_new_budget(self):
        b = ModalityBudget(modality="screen", hourly_limit=1.0, events_this_hour=0,
                           cost_this_hour=0.0, remaining=1.0)
        b2 = update_budget_after_event(b, cost=0.005)
        assert b is not b2


# ---------------------------------------------------------------------------
# adjust_threshold
# ---------------------------------------------------------------------------

class TestAdjustThreshold:
    def test_noisy_raises_threshold(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        # Signal rate 10x target → too noisy
        adjusted = adjust_threshold(p, recent_signal_rate=50.0, target_rate=5.0)
        assert adjusted.attention_threshold > 0.3

    def test_quiet_lowers_threshold(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        # Signal rate 0.5x target → too quiet
        adjusted = adjust_threshold(p, recent_signal_rate=1.0, target_rate=5.0)
        assert adjusted.attention_threshold < 0.3

    def test_in_range_no_change(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        # Signal rate near target → no change
        adjusted = adjust_threshold(p, recent_signal_rate=5.0, target_rate=5.0)
        assert adjusted.attention_threshold == 0.3

    def test_max_threshold_cap(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.88)
        adjusted = adjust_threshold(p, recent_signal_rate=100.0, target_rate=5.0, max_threshold=0.9)
        assert adjusted.attention_threshold <= 0.9

    def test_min_threshold_cap(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.12)
        adjusted = adjust_threshold(p, recent_signal_rate=0.0, target_rate=5.0, min_threshold=0.1)
        assert adjusted.attention_threshold >= 0.1

    def test_zero_target_rate_no_change(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        adjusted = adjust_threshold(p, recent_signal_rate=10.0, target_rate=0.0)
        assert adjusted.attention_threshold == 0.3

    def test_preserves_other_fields(self):
        p = ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                            latency_ms=200.0, information_density=0.8, attention_threshold=0.3)
        adjusted = adjust_threshold(p, recent_signal_rate=50.0, target_rate=5.0)
        assert adjusted.name == "screen"
        assert adjusted.reliability == 0.95
        assert adjusted.cost_per_event == 0.005
        assert adjusted.enabled is True


# ---------------------------------------------------------------------------
# format_modality_context
# ---------------------------------------------------------------------------

class TestFormatModalityContext:
    def test_empty_returns_empty_string(self):
        assert format_modality_context({}) == ""

    def test_all_disabled_returns_empty(self):
        profiles = {
            "screen": ModalityProfile(name="screen", reliability=0.95, cost_per_event=0.005,
                                      latency_ms=200.0, information_density=0.8, enabled=False),
        }
        assert format_modality_context(profiles) == ""

    def test_contains_header(self):
        profiles = default_profiles()
        result = format_modality_context(profiles)
        assert "[Active Modalities]" in result

    def test_contains_modality_name(self):
        profiles = default_profiles()
        result = format_modality_context(profiles)
        assert "screen" in result
        assert "speech" in result
        assert "camera" in result

    def test_contains_reliability(self):
        profiles = default_profiles()
        result = format_modality_context(profiles)
        assert "95%" in result  # screen reliability

    def test_contains_threshold(self):
        profiles = default_profiles()
        result = format_modality_context(profiles)
        assert "threshold=" in result

    def test_disabled_not_shown(self):
        profiles = default_profiles()
        profiles["camera"] = ModalityProfile(
            name="camera", reliability=0.6, cost_per_event=0.005,
            latency_ms=300.0, information_density=0.5, enabled=False,
        )
        result = format_modality_context(profiles)
        # Camera line should not appear
        lines = result.split("\n")
        camera_lines = [l for l in lines if "camera" in l]
        assert len(camera_lines) == 0

    def test_sorted_by_density_descending(self):
        profiles = default_profiles()
        result = format_modality_context(profiles)
        lines = [l.strip() for l in result.split("\n") if l.strip().startswith(("screen", "speech", "camera"))]
        # Speech (0.9) > Screen (0.8) > Camera (0.5)
        assert lines[0].startswith("speech")
        assert lines[1].startswith("screen")
        assert lines[2].startswith("camera")
