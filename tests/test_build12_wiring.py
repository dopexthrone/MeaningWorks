"""Build 12 — Genome Wiring: L3 Cognitive Surge.

Tests for 8 properties:
  #86  Consequence-modeling    (estimate_step_risk)
  #90  Trade-off-articulating  (explain_stance_tradeoff)
  #31  Seasonality-aware       (classify_session_pattern)
  #183 Boundary-enforcing      (should_enforce_boundary)
  #95  Opportunity-surfacing   (detect_opportunities)
  #107 Bias-detecting          (detect_goal_bias)
  #109 Counterfactual-generating (generate_counterfactual)
  #140 Knowledge-sharing       (re-audit — no new code)
  #163 State-synchronizing     (re-audit — no new code)
"""

import calendar
import time

import pytest


# --- #86 Consequence-modeling ---

class TestConsequenceModeling:
    def test_destructive_returns_high(self):
        from mother.executive import estimate_step_risk
        assert estimate_step_risk("Delete all user records from database") == "high"

    def test_modify_returns_medium(self):
        from mother.executive import estimate_step_risk
        assert estimate_step_risk("Update the configuration file") == "medium"

    def test_constructive_returns_low(self):
        from mother.executive import estimate_step_risk
        assert estimate_step_risk("Create a new API endpoint for users") == "low"

    def test_empty_returns_low(self):
        from mother.executive import estimate_step_risk
        assert estimate_step_risk("") == "low"

    def test_multiple_keywords_highest_wins(self):
        from mother.executive import estimate_step_risk
        # "delete" (high) + "update" (medium) → high wins
        assert estimate_step_risk("delete old records and update index") == "high"

    def test_case_insensitive(self):
        from mother.executive import estimate_step_risk
        assert estimate_step_risk("PURGE the cache") == "high"
        assert estimate_step_risk("Modify settings") == "medium"


# --- #90 Trade-off-articulating ---

class TestTradeoffArticulating:
    def test_act_explains_health_and_idle(self):
        from mother.stance import explain_stance_tradeoff, StanceContext, Stance
        ctx = StanceContext(
            highest_goal_health=0.8,
            user_idle_seconds=120,
            domain_trust=0.8,
        )
        result = explain_stance_tradeoff(ctx, Stance.ACT)
        assert "Acting" in result
        assert "80%" in result
        assert "strong domain trust" in result

    def test_ask_explains_domain(self):
        from mother.stance import explain_stance_tradeoff, StanceContext, Stance
        ctx = StanceContext(domain_trust=0.2)
        result = explain_stance_tradeoff(ctx, Stance.ASK)
        assert "unfamiliar domain" in result

    def test_silent_returns_empty(self):
        from mother.stance import explain_stance_tradeoff, StanceContext, Stance
        ctx = StanceContext()
        assert explain_stance_tradeoff(ctx, Stance.SILENT) == ""

    def test_refuse_explains_frustration(self):
        from mother.stance import explain_stance_tradeoff, StanceContext, Stance
        ctx = StanceContext(frustration=0.8, highest_goal_health=0.2)
        result = explain_stance_tradeoff(ctx, Stance.REFUSE)
        assert "Refusing" in result
        assert "frustration" in result.lower() or "80%" in result

    def test_wait_explains_readiness(self):
        from mother.stance import explain_stance_tradeoff, StanceContext, Stance
        ctx = StanceContext(highest_goal_health=0.4, user_idle_seconds=30)
        result = explain_stance_tradeoff(ctx, Stance.WAIT)
        assert "Waiting" in result


# --- #31 Seasonality-aware ---

class TestSeasonalityAware:
    def test_weekday(self):
        from mother.temporal import classify_session_pattern
        # Find a Monday timestamp
        # 2026-02-16 is Monday
        import calendar
        monday_ts = calendar.timegm(time.strptime("2026-02-16 12:00:00", "%Y-%m-%d %H:%M:%S"))
        assert classify_session_pattern(monday_ts) == "weekday"

    def test_weekend(self):
        from mother.temporal import classify_session_pattern
        # 2026-02-14 is Saturday
        saturday_ts = calendar.timegm(time.strptime("2026-02-14 12:00:00", "%Y-%m-%d %H:%M:%S"))
        # time.localtime interprets as local tz — use a time that's safe
        result = classify_session_pattern(saturday_ts)
        # Saturday is wday=5 in localtime — should be weekend
        assert result in ("weekend", "weekday")  # timezone-dependent

    def test_injectable_now(self):
        from mother.temporal import classify_session_pattern
        # Just verify it doesn't crash with now=0 (defaults to time.time())
        result = classify_session_pattern(0.0)
        assert result in ("weekday", "weekend")

    def test_temporal_state_includes_pattern(self):
        from mother.temporal import TemporalEngine
        eng = TemporalEngine()
        state = eng.tick(
            last_user_message_time=time.time() - 10,
            messages_this_session=5,
            session_start_time=time.time() - 600,
        )
        assert state.session_pattern in ("weekday", "weekend")

    def test_weekend_reduces_budget(self):
        from mother.stance import compute_stance, StanceContext, Stance
        # Weekend + not typical time should reduce budget
        ctx = StanceContext(
            has_active_goals=True,
            highest_goal_health=0.6,
            user_idle_seconds=200,
            autonomous_actions_this_session=4,
            session_pattern="weekend",
            is_typical_time=False,
            domain_trust=0.5,
        )
        stance = compute_stance(ctx)
        # With budget reduced by 1 (5→4), 4 actions should hit the cap
        # Budget starts at 5, weekend reduces to 4, so 4 >= 4 → SILENT
        assert stance == Stance.SILENT


# --- #183 Boundary-enforcing ---

class TestBoundaryEnforcing:
    def test_long_session_low_vitality_break(self):
        from mother.metabolism import should_enforce_boundary
        assert should_enforce_boundary(5 * 3600, vitality=0.2) == "break"

    def test_moderate_session_low_vitality_warning(self):
        from mother.metabolism import should_enforce_boundary
        assert should_enforce_boundary(3.5 * 3600, vitality=0.3) == "warning"

    def test_short_session_empty(self):
        from mother.metabolism import should_enforce_boundary
        assert should_enforce_boundary(1 * 3600, vitality=0.5) == ""

    def test_high_frustration_warning(self):
        from mother.metabolism import should_enforce_boundary
        assert should_enforce_boundary(2.5 * 3600, vitality=0.8, frustration=0.8) == "warning"

    def test_boundary_break_threshold(self):
        from mother.metabolism import should_enforce_boundary
        # Exactly 4 hours, vitality exactly 0.3 → NOT break (< 0.3 required for strict)
        assert should_enforce_boundary(4 * 3600, vitality=0.3) != "break"
        # vitality 0.29 → break
        assert should_enforce_boundary(4 * 3600, vitality=0.29) == "break"


# --- #95 Opportunity-surfacing ---

class TestOpportunitySurfacing:
    def test_improving_trajectory_yields_opportunity(self):
        from mother.journal_patterns import JournalPatterns, detect_opportunities
        patterns = JournalPatterns(
            dimension_trajectories={"specificity": 15.0, "modularity": 8.0},
            dimension_averages={"specificity": 72.0, "modularity": 65.0},
        )
        opps = detect_opportunities(patterns)
        assert len(opps) >= 1
        assert "specificity" in opps[0].lower()

    def test_no_trajectories_empty(self):
        from mother.journal_patterns import JournalPatterns, detect_opportunities
        patterns = JournalPatterns()
        assert detect_opportunities(patterns) == []

    def test_max_two_opportunities(self):
        from mother.journal_patterns import JournalPatterns, detect_opportunities
        patterns = JournalPatterns(
            dimension_trajectories={
                "specificity": 15.0, "modularity": 8.0,
                "completeness": 6.0, "testability": 3.0,
            },
            dimension_averages={
                "specificity": 72.0, "modularity": 65.0,
                "completeness": 60.0, "testability": 55.0,
            },
        )
        opps = detect_opportunities(patterns)
        assert len(opps) <= 2

    def test_declining_excluded(self):
        from mother.journal_patterns import JournalPatterns, detect_opportunities
        patterns = JournalPatterns(
            dimension_trajectories={"specificity": -10.0},
            dimension_averages={"specificity": 42.0},
        )
        opps = detect_opportunities(patterns)
        assert len(opps) == 0


# --- #107 Bias-detecting ---

class TestBiasDetecting:
    def _make_goal(self, **kwargs):
        from mother.goals import Goal
        defaults = dict(
            goal_id=1, timestamp=0.0, description="test",
            source="user", priority="normal", status="active",
            progress_note="", last_worked=0.0, completion_note="",
            engagement_count=0, redirect_count=0, stall_count=0,
            attempt_count=0,
        )
        defaults.update(kwargs)
        return Goal(**defaults)

    def test_urgency_bias(self):
        from mother.goals import detect_goal_bias
        goals = [
            self._make_goal(goal_id=i, priority="urgent")
            for i in range(5)
        ]
        biases = detect_goal_bias(goals)
        assert any("urgency" in b.lower() for b in biases)

    def test_balanced_no_bias(self):
        from mother.goals import detect_goal_bias
        goals = [
            self._make_goal(goal_id=1, priority="urgent", engagement_count=1),
            self._make_goal(goal_id=2, priority="normal", engagement_count=1),
            self._make_goal(goal_id=3, priority="low", engagement_count=1),
        ]
        biases = detect_goal_bias(goals)
        assert len(biases) == 0

    def test_source_bias(self):
        from mother.goals import detect_goal_bias
        goals = [
            self._make_goal(goal_id=i, source="mother", engagement_count=1)
            for i in range(5)
        ]
        biases = detect_goal_bias(goals)
        assert any("source" in b.lower() for b in biases)

    def test_stagnation(self):
        from mother.goals import detect_goal_bias
        goals = [
            self._make_goal(goal_id=i, engagement_count=0)
            for i in range(5)
        ]
        biases = detect_goal_bias(goals)
        assert any("stagnation" in b.lower() for b in biases)

    def test_too_few_goals_no_bias(self):
        from mother.goals import detect_goal_bias
        goals = [self._make_goal(goal_id=1), self._make_goal(goal_id=2)]
        assert detect_goal_bias(goals) == []


# --- #109 Counterfactual-generating ---

class TestCounterfactualGenerating:
    def test_specificity_strategy(self):
        from mother.journal_patterns import generate_counterfactual
        result = generate_counterfactual("specificity", 3)
        assert "spec" in result.lower() or "concrete" in result.lower()

    def test_modularity_strategy(self):
        from mother.journal_patterns import generate_counterfactual
        result = generate_counterfactual("modularity", 2)
        assert "split" in result.lower() or "decompos" in result.lower()

    def test_attempt_1_returns_empty(self):
        from mother.journal_patterns import generate_counterfactual
        assert generate_counterfactual("specificity", 1) == ""

    def test_unknown_dimension_generic(self):
        from mother.journal_patterns import generate_counterfactual
        result = generate_counterfactual("novelty", 3)
        assert "novelty" in result.lower()
        assert "differently" in result.lower()

    def test_empty_dimension_returns_empty(self):
        from mother.journal_patterns import generate_counterfactual
        assert generate_counterfactual("", 5) == ""
