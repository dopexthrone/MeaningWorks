"""Build 13 — Genome Wiring: L3 Cognitive Completion + Infrastructure.

Tests for 10 properties:
  #91  Reversibility-assessing     (classify_reversibility)
  #104 Skill-gap-mapping           (detect_skill_gap)
  #102 Assumption-challenging      (challenge_assumptions)
  #92  Probability-estimating      (estimate_success_probability)
  #122 Privacy-protecting          (detect_pii_patterns)
  #124 Threat-modeling             (assess_threat_surface)
  #114 Unit-economics-literate     (compute_unit_economics)
  #6   Constraint-inheriting       (propagate_constraints)
  #182 Ritual-building             (suggest_ritual)
  #49  Batch-capable               (batch_compatible_goals)
"""

import pytest


# --- #91 Reversibility-assessing ---

class TestReversibilityAssessing:
    def test_delete_is_one_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Delete all user records") == "one-way"

    def test_deploy_is_one_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Deploy to production") == "one-way"

    def test_test_is_two_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Test the API endpoint") == "two-way"

    def test_review_is_two_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Review the code changes") == "two-way"

    def test_ambiguous_defaults_two_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Process the data") == "two-way"

    def test_publish_is_one_way(self):
        from mother.executive import classify_reversibility
        assert classify_reversibility("Publish the blog post") == "one-way"


# --- #104 Skill-gap-mapping ---

class TestSkillGapMapping:
    def test_specificity_gap(self):
        from mother.journal_patterns import detect_skill_gap
        result = detect_skill_gap(["specificity", "modularity"], {})
        assert "specific" in result.lower()

    def test_domain_amplification(self):
        from mother.journal_patterns import detect_skill_gap
        result = detect_skill_gap(
            ["traceability"],
            {"software": ["traceability", "testability"]},
        )
        assert "trace" in result.lower()
        assert "software" in result.lower()

    def test_empty_chronic_returns_empty(self):
        from mother.journal_patterns import detect_skill_gap
        assert detect_skill_gap([], {}) == ""

    def test_unknown_dimension_returns_empty(self):
        from mother.journal_patterns import detect_skill_gap
        assert detect_skill_gap(["novelty"], {}) == ""


# --- #102 Assumption-challenging ---

class TestAssumptionChallenging:
    def test_simple_keyword_triggers(self):
        from mother.journal_patterns import challenge_assumptions
        result = challenge_assumptions("Build a simple API", 3)
        assert "simple" in result.lower() or "complexity" in result.lower()

    def test_user_keyword_triggers(self):
        from mother.journal_patterns import challenge_assumptions
        result = challenge_assumptions("Create user dashboard", 2)
        assert "user" in result.lower()

    def test_attempt_1_returns_empty(self):
        from mother.journal_patterns import challenge_assumptions
        assert challenge_assumptions("Build something", 1) == ""

    def test_generic_fallback(self):
        from mother.journal_patterns import challenge_assumptions
        result = challenge_assumptions("Implement the widget system", 3)
        assert "assumption" in result.lower() or "baked" in result.lower()

    def test_empty_description_returns_empty(self):
        from mother.journal_patterns import challenge_assumptions
        assert challenge_assumptions("", 5) == ""


# --- #92 Probability-estimating ---

class TestProbabilityEstimating:
    def test_high_success_rate_high_probability(self):
        from mother.journal_patterns import estimate_success_probability
        prob = estimate_success_probability(0.9, 0, 80.0)
        assert prob >= 0.7

    def test_many_failures_reduces_probability(self):
        from mother.journal_patterns import estimate_success_probability
        prob = estimate_success_probability(0.8, 5, 50.0)
        assert prob < 0.5

    def test_zero_rate_low_probability(self):
        from mother.journal_patterns import estimate_success_probability
        prob = estimate_success_probability(0.0, 3, 30.0)
        assert prob <= 0.15

    def test_clamped_range(self):
        from mother.journal_patterns import estimate_success_probability
        prob = estimate_success_probability(1.0, 0, 100.0)
        assert 0.05 <= prob <= 0.95

    def test_no_attempts_uses_base_rate(self):
        from mother.journal_patterns import estimate_success_probability
        prob = estimate_success_probability(0.5, 0, 50.0)
        assert 0.3 <= prob <= 0.6


# --- #122 Privacy-protecting ---

class TestPrivacyProtecting:
    def test_email_detected(self):
        from core.governor_validation import detect_pii_patterns
        result = detect_pii_patterns("Send to user@example.com please")
        assert any("email" in t for t, _ in result)

    def test_ssn_detected(self):
        from core.governor_validation import detect_pii_patterns
        result = detect_pii_patterns("SSN: 123-45-6789")
        assert any("SSN" in t for t, _ in result)

    def test_credit_card_detected(self):
        from core.governor_validation import detect_pii_patterns
        result = detect_pii_patterns("Card: 4111-1111-1111-1111")
        assert any("credit" in t for t, _ in result)

    def test_hardcoded_credential_detected(self):
        from core.governor_validation import detect_pii_patterns
        result = detect_pii_patterns("api_key = sk-12345abc")
        assert any("credential" in t for t, _ in result)

    def test_clean_text_no_pii(self):
        from core.governor_validation import detect_pii_patterns
        assert detect_pii_patterns("Build a todo app with React") == []

    def test_empty_text(self):
        from core.governor_validation import detect_pii_patterns
        assert detect_pii_patterns("") == []


# --- #124 Threat-modeling ---

class TestThreatModeling:
    def test_path_traversal(self):
        from core.governor_validation import assess_threat_surface
        result = assess_threat_surface("Read file from ../../etc/passwd")
        assert "path traversal" in result

    def test_sql_injection(self):
        from core.governor_validation import assess_threat_surface
        result = assess_threat_surface('SELECT * FROM users WHERE id = " + user_input')
        assert "SQL injection" in result

    def test_pickle_deserialization(self):
        from core.governor_validation import assess_threat_surface
        result = assess_threat_surface("Use pickle.load to read cached data")
        assert "pickle deserialization" in result

    def test_clean_text_no_threats(self):
        from core.governor_validation import assess_threat_surface
        assert assess_threat_surface("Build a REST API with FastAPI") == []

    def test_xss_vector(self):
        from core.governor_validation import assess_threat_surface
        result = assess_threat_surface("Set innerHTML from user data")
        assert "XSS vector" in result


# --- #114 Unit-economics-literate ---

class TestUnitEconomics:
    def test_basic_economics(self):
        from mother.journal_patterns import compute_unit_economics
        result = compute_unit_economics(
            total_cost=1.0, total_compiles=10,
            successful_compiles=8, total_components=20,
        )
        assert result["cost_per_compile"] == 0.1
        assert result["cost_per_component"] == 0.05
        assert result["cost_per_success"] == 0.125
        assert result["waste_ratio"] == 0.2

    def test_zero_compiles(self):
        from mother.journal_patterns import compute_unit_economics
        result = compute_unit_economics(0.0, 0, 0, 0)
        assert result["cost_per_compile"] == 0.0

    def test_all_failures(self):
        from mother.journal_patterns import compute_unit_economics
        result = compute_unit_economics(5.0, 5, 0, 10)
        assert result["waste_ratio"] == 1.0
        assert result["cost_per_success"] == 0.0

    def test_perfect_success(self):
        from mother.journal_patterns import compute_unit_economics
        result = compute_unit_economics(2.0, 4, 4, 8)
        assert result["waste_ratio"] == 0.0


# --- #6 Constraint-inheriting ---

class TestConstraintInheriting:
    def test_propagates_constraints(self):
        from mother.appendage import propagate_constraints
        parent = "Build a tool.\nYou must use Python 3.14.\nNever access network."
        child = "Build a file watcher."
        result = propagate_constraints(parent, "[]", child)
        assert "must use Python" in result
        assert "Never access" in result
        assert "Build a file watcher" in result

    def test_propagates_capabilities(self):
        from mother.appendage import propagate_constraints
        result = propagate_constraints("", '["screen-capture", "audio"]', "Child desc")
        assert "screen-capture" in result
        assert "Child desc" in result

    def test_no_parent_passthrough(self):
        from mother.appendage import propagate_constraints
        result = propagate_constraints("", "", "Just the child")
        assert result == "Just the child"

    def test_caps_constraint_lines(self):
        from mother.appendage import propagate_constraints
        # More than 5 constraint lines — should cap at 5
        parent = "\n".join([f"You must do thing {i}" for i in range(10)])
        result = propagate_constraints(parent, "[]", "child")
        constraint_count = result.count("must do thing")
        assert constraint_count == 5


# --- #182 Ritual-building ---

class TestRitualBuilding:
    def test_daily_ritual(self):
        from mother.impulse import suggest_ritual
        result = suggest_ritual(1.0, "morning", 10)
        assert "daily" in result.lower()

    def test_weekly_ritual(self):
        from mother.impulse import suggest_ritual
        result = suggest_ritual(7.0, "evening", 8)
        assert "weekly" in result.lower()

    def test_too_few_sessions(self):
        from mother.impulse import suggest_ritual
        assert suggest_ritual(1.0, "morning", 3) == ""

    def test_too_infrequent(self):
        from mother.impulse import suggest_ritual
        assert suggest_ritual(15.0, "", 10) == ""

    def test_ritual_time_match(self):
        from mother.impulse import suggest_ritual
        result = suggest_ritual(1.0, "morning", 10, current_time_of_day="morning")
        assert "schedule" in result.lower() or "on schedule" in result.lower()

    def test_every_few_days(self):
        from mother.impulse import suggest_ritual
        result = suggest_ritual(3.0, "", 6)
        assert "every few days" in result.lower()


# --- #49 Batch-capable ---

class TestBatchCapable:
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

    def test_independent_goals_batch(self):
        from mother.goals import batch_compatible_goals
        goals = [
            self._make_goal(goal_id=1, description="Build user authentication"),
            self._make_goal(goal_id=2, description="Create payment gateway"),
            self._make_goal(goal_id=3, description="Deploy monitoring dashboard"),
        ]
        batches = batch_compatible_goals(goals)
        # All same priority, low overlap → should batch together
        assert any(len(b) >= 2 for b in batches)

    def test_different_priorities_separate(self):
        from mother.goals import batch_compatible_goals
        goals = [
            self._make_goal(goal_id=1, priority="urgent", description="Fix login bug"),
            self._make_goal(goal_id=2, priority="low", description="Add dark mode"),
        ]
        batches = batch_compatible_goals(goals)
        # Different priorities → separate batches
        assert all(len(b) == 1 for b in batches)

    def test_single_goal_no_batch(self):
        from mother.goals import batch_compatible_goals
        goals = [self._make_goal(goal_id=1)]
        batches = batch_compatible_goals(goals)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_empty_returns_empty(self):
        from mother.goals import batch_compatible_goals
        assert batch_compatible_goals([]) == []

    def test_max_batch_size_respected(self):
        from mother.goals import batch_compatible_goals
        goals = [
            self._make_goal(goal_id=i, description=f"unique task number {i}")
            for i in range(10)
        ]
        batches = batch_compatible_goals(goals, max_batch=2)
        assert all(len(b) <= 2 for b in batches)
