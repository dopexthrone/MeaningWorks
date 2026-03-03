"""Build 14 — Genome wiring tests.

Properties:
  #103 Reframe-capable (L3)
  #106 Serendipity-engineering (L3)
  #119 Ethics-reasoning (L3)
  #59  Risk-registering (L2)
  #154 Teaching-mode (L4-Creative)
  #156 Skill-transfer (L4-Creative)
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest


# ======================================================================
# #103 Reframe-capable
# ======================================================================

class TestReframeCapable:
    """generate_reframe() — structural problem reframing when stuck."""

    def test_build_trigger(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("build a dashboard", [], 3)
        assert "defining what it actually needs to do" in result

    def test_fix_trigger(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("fix the login bug", [], 2)
        assert "system around it changed" in result

    def test_optimize_trigger(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("optimize database queries", [], 2)
        assert "replacement" in result

    def test_attempt_1_returns_empty(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("build something", [], 1)
        assert result == ""

    def test_empty_description_returns_empty(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("", [], 3)
        assert result == ""

    def test_fallback_with_chronic(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("do something unusual", ["specificity"], 3)
        assert "specificity" in result
        assert "rethinking" in result

    def test_fallback_no_chronic(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("do something unusual", [], 3)
        assert "wrong problem" in result

    def test_scale_trigger(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("scale the infrastructure", [], 2)
        assert "simplification" in result

    def test_migrate_trigger(self):
        from mother.journal_patterns import generate_reframe
        result = generate_reframe("migrate to new platform", [], 2)
        assert "destination" in result


# ======================================================================
# #106 Serendipity-engineering
# ======================================================================

class TestSerendipityEngineering:
    """find_cross_topic_connections() — unexpected cross-session links."""

    def test_word_overlap_returns_connection(self):
        from mother.journal_patterns import find_cross_topic_connections
        topics = ["authentication system design"]
        subjects = ["recurring authentication failures in API"]
        result = find_cross_topic_connections(topics, subjects)
        assert "authentication" in result
        assert "Unexpected connection" in result

    def test_no_overlap_returns_empty(self):
        from mother.journal_patterns import find_cross_topic_connections
        topics = ["build a dashboard"]
        subjects = ["authentication failures"]
        result = find_cross_topic_connections(topics, subjects)
        assert result == ""

    def test_empty_topics_returns_empty(self):
        from mother.journal_patterns import find_cross_topic_connections
        result = find_cross_topic_connections([], ["something"])
        assert result == ""

    def test_empty_subjects_returns_empty(self):
        from mother.journal_patterns import find_cross_topic_connections
        result = find_cross_topic_connections(["something"], [])
        assert result == ""

    def test_picks_best_match(self):
        from mother.journal_patterns import find_cross_topic_connections
        topics = ["database optimization query"]
        subjects = [
            "frontend styling",
            "database query performance issues",
            "authentication flow",
        ]
        result = find_cross_topic_connections(topics, subjects)
        assert "database" in result or "query" in result

    def test_stopwords_excluded(self):
        from mother.journal_patterns import find_cross_topic_connections
        topics = ["the new system"]
        subjects = ["the old system"]
        result = find_cross_topic_connections(topics, subjects)
        # "system" is not a stopword, should match
        assert "system" in result


# ======================================================================
# #119 Ethics-reasoning
# ======================================================================

class TestEthicsReasoning:
    """assess_ethical_concerns() — intent-level ethical scanning."""

    def test_surveillance_detected(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("build a system to track user activity")
        assert "surveillance concern" in result

    def test_manipulation_detected(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("create dark patterns to increase signups")
        assert "manipulation concern" in result

    def test_discrimination_detected(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("filter by race and gender")
        assert "discrimination concern" in result

    def test_deception_detected(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("build a phishing page")
        assert "deception concern" in result

    def test_harm_detected(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("develop ransomware")
        assert "harm concern" in result

    def test_harmless_returns_empty(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("build a todo app")
        assert result == []

    def test_empty_returns_empty(self):
        from core.governor_validation import assess_ethical_concerns
        result = assess_ethical_concerns("")
        assert result == []

    def test_cap_at_5(self):
        from core.governor_validation import assess_ethical_concerns
        # All patterns at once
        text = "track user with dark patterns to discriminate via phishing and ransomware"
        result = assess_ethical_concerns(text)
        assert len(result) <= 5


# ======================================================================
# #59 Risk-registering
# ======================================================================

class TestRiskRegistering:
    """RiskRegister + classify_risk_severity — persistent risk store."""

    def test_add_and_retrieve(self):
        from mother.executive import RiskRegister
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            rr = RiskRegister(f.name)
            rid = rr.add_risk("test risk", "medium", "test")
            assert rid > 0
            risks = rr.active_risks()
            assert len(risks) == 1
            assert risks[0].description == "test risk"
            rr.close()

    def test_severity_critical(self):
        from mother.executive import classify_risk_severity
        assert classify_risk_severity("pii exposed in logs") == "critical"
        assert classify_risk_severity("data loss detected") == "critical"

    def test_severity_high(self):
        from mother.executive import classify_risk_severity
        assert classify_risk_severity("delete old records") == "high"
        assert classify_risk_severity("deploy to production") == "high"

    def test_severity_medium(self):
        from mother.executive import classify_risk_severity
        assert classify_risk_severity("update configuration") == "medium"

    def test_severity_low(self):
        from mother.executive import classify_risk_severity
        assert classify_risk_severity("read documentation") == "low"

    def test_update_status(self):
        from mother.executive import RiskRegister
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            rr = RiskRegister(f.name)
            rid = rr.add_risk("risky thing", "high", "test")
            rr.update_status(rid, "mitigated", "added safeguards")
            active = rr.active_risks()
            assert len(active) == 0  # no longer active
            rr.close()

    def test_ordering_by_severity(self):
        from mother.executive import RiskRegister
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            rr = RiskRegister(f.name)
            rr.add_risk("low risk", "low", "test")
            rr.add_risk("critical risk", "critical", "test")
            rr.add_risk("medium risk", "medium", "test")
            risks = rr.active_risks()
            assert risks[0].severity == "critical"
            assert risks[-1].severity == "low"
            rr.close()

    def test_empty_returns_empty(self):
        from mother.executive import classify_risk_severity
        assert classify_risk_severity("") == "low"


# ======================================================================
# #154 Teaching-mode
# ======================================================================

class TestTeachingMode:
    """generate_teaching_summary() — 'here's what happened' narrative."""

    def test_narrative_generation(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary(
            {"trust_score": 75, "component_count": 3, "step_names": ["compile", "build"]},
            {},
        )
        assert "What happened:" in result
        assert "3 components" in result

    def test_includes_trust(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary(
            {"trust_score": 80, "component_count": 1},
            {},
        )
        assert "80%" in result

    def test_includes_weakest_dim(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary(
            {"trust_score": 70, "component_count": 2,
             "dimension_scores": {"specificity": 40, "completeness": 90}},
            {},
        )
        assert "weakest in specificity" in result

    def test_empty_returns_empty(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary({}, {})
        assert result == ""

    def test_with_learning_context(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary(
            {"trust_score": 65, "component_count": 4},
            {"chronic_weak": ["traceability"]},
        )
        assert "traceability" in result

    def test_with_steps(self):
        from mother.journal_patterns import generate_teaching_summary
        result = generate_teaching_summary(
            {"trust_score": 70, "component_count": 1,
             "step_names": ["compile", "build", "verify"]},
            {},
        )
        assert "compile" in result


# ======================================================================
# #156 Skill-transfer
# ======================================================================

class TestSkillTransfer:
    """extract_methodology() — exportable working methodology."""

    def test_methodology_generation(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology(
            {
                "chronic_weak": ["specificity"],
                "dimension_trajectories": {"completeness": 5.0, "specificity": -3.0},
                "dimension_averages": {"completeness": 80.0, "specificity": 45.0},
            },
            goal_count=3,
            compile_count=5,
        )
        assert "Over 5 compiles" in result
        assert "3 active goals" in result

    def test_needs_min_3_compiles(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology({}, goal_count=1, compile_count=2)
        assert result == ""

    def test_includes_chronic(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology(
            {"chronic_weak": ["modularity"], "dimension_trajectories": {},
             "dimension_averages": {}},
            goal_count=2, compile_count=5,
        )
        assert "modularity" in result

    def test_includes_trends(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology(
            {"chronic_weak": [],
             "dimension_trajectories": {"completeness": 8.0, "specificity": -4.0},
             "dimension_averages": {}},
            goal_count=1, compile_count=4,
        )
        assert "Improving" in result
        assert "Declining" in result

    def test_includes_strengths(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology(
            {"chronic_weak": [], "dimension_trajectories": {},
             "dimension_averages": {"completeness": 90.0, "consistency": 85.0, "specificity": 40.0}},
            goal_count=0, compile_count=3,
        )
        assert "Strengths" in result
        assert "completeness" in result

    def test_empty_patterns(self):
        from mother.journal_patterns import extract_methodology
        result = extract_methodology(
            {"chronic_weak": [], "dimension_trajectories": {},
             "dimension_averages": {}},
            goal_count=0, compile_count=3,
        )
        assert "Over 3 compiles" in result
