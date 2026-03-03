"""
Tests for the three feedback loop modules:
- kernel/observer.py (tested separately in test_kernel_observer.py)
- mother/governor_feedback.py
- mother/goal_generator.py
"""

import pytest

from mother.governor_feedback import (
    CompilationOutcome,
    WeaknessSignal,
    FeedbackReport,
    analyze_outcomes,
    extract_rejection_patterns,
    score_compiler_health,
    generate_compiler_prompt_patch,
    _categorize_rejection,
    _compute_trend,
    _detect_weaknesses,
    _detect_strengths,
    _CRITICAL_THRESHOLD,
    _WARNING_THRESHOLD,
    _WATCH_THRESHOLD,
)
from mother.goal_generator import (
    ImprovementGoal,
    GoalSet,
    goals_from_grid,
    goals_from_feedback,
    goals_from_anomalies,
    generate_goal_set,
    _HEALTHY_CONFIDENCE,
    _CORE_LAYERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _outcome(
    trust=70.0, comp=70.0, cons=70.0, coh=70.0, trac=70.0,
    act=70.0, spec=70.0, cgen=70.0,
    rejected=False, reason="", domain="software", cid="c-1",
) -> CompilationOutcome:
    return CompilationOutcome(
        compile_id=cid,
        input_summary="test input",
        trust_score=trust,
        completeness=comp,
        consistency=cons,
        coherence=coh,
        traceability=trac,
        actionability=act,
        specificity=spec,
        codegen_readiness=cgen,
        component_count=5,
        rejected=rejected,
        rejection_reason=reason,
        domain=domain,
    )


def _outcomes_healthy(n=5) -> list[CompilationOutcome]:
    return [_outcome(trust=80+i, comp=85, cons=82, coh=88, trac=80, cid=f"c-{i}") for i in range(n)]


def _outcomes_struggling(n=5) -> list[CompilationOutcome]:
    return [_outcome(trust=35+i, comp=30, cons=40, coh=35, trac=25, cid=f"c-{i}") for i in range(n)]


def _outcomes_mixed(n=6) -> list[CompilationOutcome]:
    results = []
    for i in range(n):
        if i % 2 == 0:
            results.append(_outcome(trust=80, comp=85, cons=82, coh=88, trac=80, cid=f"c-{i}"))
        else:
            results.append(_outcome(trust=30, comp=25, cons=35, coh=30, trac=20,
                                   rejected=True, reason="trust score too low", cid=f"c-{i}"))
    return results


# ===================================================================
# GOVERNOR FEEDBACK
# ===================================================================

class TestCompilationOutcome:
    def test_frozen(self):
        o = _outcome()
        with pytest.raises(AttributeError):
            o.trust_score = 99.0

    def test_defaults(self):
        o = _outcome()
        assert o.rejected is False
        assert o.rejection_reason == ""
        assert o.domain == "software"


class TestAnalyzeOutcomes:
    def test_empty(self):
        report = analyze_outcomes([])
        assert report.outcomes_analyzed == 0
        assert report.trend == "insufficient_data"
        assert len(report.compiler_hints) == 1

    def test_healthy_outcomes(self):
        report = analyze_outcomes(_outcomes_healthy())
        assert report.outcomes_analyzed == 5
        assert report.rejection_rate == 0.0
        assert len(report.strengths) > 0
        assert report.trend in ("stable", "improving", "insufficient_data")

    def test_struggling_outcomes(self):
        report = analyze_outcomes(_outcomes_struggling())
        assert report.outcomes_analyzed == 5
        assert len(report.weaknesses) > 0
        # All dimensions are below warning threshold
        dims = {w.dimension for w in report.weaknesses}
        assert len(dims) >= 3  # at least 3 weak dimensions

    def test_mixed_outcomes(self):
        report = analyze_outcomes(_outcomes_mixed())
        assert report.rejection_rate == 0.5
        assert any("reject" in h.lower() or "critical" in h.lower()
                   for h in report.compiler_hints)

    def test_single_outcome(self):
        report = analyze_outcomes([_outcome()])
        assert report.outcomes_analyzed == 1
        assert report.trend == "insufficient_data"

    def test_all_rejected(self):
        outcomes = [_outcome(trust=20, rejected=True, reason="trust too low", cid=f"c-{i}")
                    for i in range(5)]
        report = analyze_outcomes(outcomes)
        assert report.rejection_rate == 1.0


class TestExtractRejectionPatterns:
    def test_no_rejections(self):
        patterns = extract_rejection_patterns(_outcomes_healthy())
        assert patterns == {}

    def test_trust_rejections(self):
        outcomes = [
            _outcome(rejected=True, reason="trust score below threshold", cid="c-1"),
            _outcome(rejected=True, reason="low trust", cid="c-2"),
            _outcome(rejected=False, cid="c-3"),
        ]
        patterns = extract_rejection_patterns(outcomes)
        assert patterns.get("trust", 0) == 2

    def test_mixed_categories(self):
        outcomes = [
            _outcome(rejected=True, reason="trust too low", cid="c-1"),
            _outcome(rejected=True, reason="cost budget exceeded", cid="c-2"),
            _outcome(rejected=True, reason="compilation timeout", cid="c-3"),
        ]
        patterns = extract_rejection_patterns(outcomes)
        assert "trust" in patterns
        assert "cost" in patterns
        assert "timeout" in patterns

    def test_empty_reason_skipped(self):
        outcomes = [_outcome(rejected=True, reason="", cid="c-1")]
        patterns = extract_rejection_patterns(outcomes)
        assert patterns == {}


class TestScoreCompilerHealth:
    def test_healthy(self):
        score = score_compiler_health(_outcomes_healthy())
        assert score > 70.0

    def test_struggling(self):
        score = score_compiler_health(_outcomes_struggling())
        assert score < 50.0

    def test_empty(self):
        score = score_compiler_health([])
        assert score == 50.0  # neutral

    def test_clamped(self):
        # Even terrible outcomes don't go below 0
        outcomes = [_outcome(trust=0, comp=0, cons=0, coh=0, trac=0,
                            rejected=True, reason="everything failed", cid=f"c-{i}")
                    for i in range(10)]
        score = score_compiler_health(outcomes)
        assert score >= 0.0

    def test_perfect(self):
        outcomes = [_outcome(trust=100, comp=100, cons=100, coh=100, trac=100, cid=f"c-{i}")
                    for i in range(5)]
        score = score_compiler_health(outcomes)
        assert score >= 85.0  # high but may have balance penalty


class TestCategorizeRejection:
    def test_trust(self):
        assert _categorize_rejection("Trust score below 60") == "trust"

    def test_quality(self):
        assert _categorize_rejection("Input quality too low") == "quality"

    def test_cost(self):
        assert _categorize_rejection("Cost budget exceeded") == "cost"

    def test_timeout(self):
        assert _categorize_rejection("Compilation timeout") == "timeout"

    def test_empty_blueprint(self):
        assert _categorize_rejection("Empty blueprint, no components") == "empty"

    def test_unknown(self):
        assert _categorize_rejection("Something weird happened") == "other"


class TestComputeTrend:
    def test_improving(self):
        # First half low, second half high
        outcomes = [_outcome(trust=40, cid=f"c-{i}") for i in range(3)]
        outcomes += [_outcome(trust=80, cid=f"c-{i+3}") for i in range(3)]
        assert _compute_trend(outcomes) == "improving"

    def test_degrading(self):
        # First half high, second half low
        outcomes = [_outcome(trust=80, cid=f"c-{i}") for i in range(3)]
        outcomes += [_outcome(trust=40, cid=f"c-{i+3}") for i in range(3)]
        assert _compute_trend(outcomes) == "degrading"

    def test_stable(self):
        outcomes = [_outcome(trust=70, cid=f"c-{i}") for i in range(6)]
        assert _compute_trend(outcomes) == "stable"

    def test_insufficient(self):
        outcomes = [_outcome(trust=70, cid="c-1"), _outcome(trust=80, cid="c-2")]
        assert _compute_trend(outcomes) == "insufficient_data"


class TestDetectWeaknesses:
    def test_no_weaknesses(self):
        dim_means = {"completeness": 85.0, "consistency": 90.0,
                     "coherence": 80.0, "traceability": 88.0,
                     "actionability": 82.0, "specificity": 85.0,
                     "codegen_readiness": 80.0}
        dim_scores = {d: [v] for d, v in dim_means.items()}
        weaknesses = _detect_weaknesses(dim_means, dim_scores)
        assert len(weaknesses) == 0

    def test_critical_weakness(self):
        dim_means = {"completeness": 25.0, "consistency": 80.0,
                     "coherence": 80.0, "traceability": 80.0,
                     "actionability": 80.0, "specificity": 80.0,
                     "codegen_readiness": 80.0}
        dim_scores = {d: [v] for d, v in dim_means.items()}
        weaknesses = _detect_weaknesses(dim_means, dim_scores)
        assert any(w.severity == "critical" for w in weaknesses)
        assert weaknesses[0].dimension == "completeness"

    def test_warning_weakness(self):
        dim_means = {"completeness": 55.0, "consistency": 80.0,
                     "coherence": 80.0, "traceability": 80.0}
        dim_scores = {d: [v] for d, v in dim_means.items()}
        weaknesses = _detect_weaknesses(dim_means, dim_scores)
        assert any(w.severity == "warning" for w in weaknesses)

    def test_watch_weakness(self):
        dim_means = {"completeness": 72.0, "consistency": 80.0,
                     "coherence": 80.0, "traceability": 80.0}
        dim_scores = {d: [v] for d, v in dim_means.items()}
        weaknesses = _detect_weaknesses(dim_means, dim_scores)
        assert any(w.severity == "watch" for w in weaknesses)


class TestDetectStrengths:
    def test_strong_dimensions(self):
        dim_means = {"completeness": 90.0, "consistency": 88.0,
                     "coherence": 92.0, "traceability": 85.0}
        strengths = _detect_strengths(dim_means)
        assert len(strengths) == 4
        assert all("strong" in s for s in strengths)

    def test_healthy_dimensions(self):
        dim_means = {"completeness": 78.0, "consistency": 76.0,
                     "coherence": 77.0, "traceability": 75.0}
        strengths = _detect_strengths(dim_means)
        assert len(strengths) == 4
        assert all("healthy" in s for s in strengths)

    def test_no_strengths(self):
        dim_means = {"completeness": 40.0, "consistency": 35.0,
                     "coherence": 30.0, "traceability": 25.0}
        strengths = _detect_strengths(dim_means)
        assert len(strengths) == 0


class TestGeneratePromptPatch:
    def test_empty_report(self):
        report = analyze_outcomes([])
        patch = generate_compiler_prompt_patch(report)
        assert "Compiler Self-Improvement" in patch

    def test_with_weaknesses(self):
        report = analyze_outcomes(_outcomes_struggling())
        patch = generate_compiler_prompt_patch(report)
        assert "Known Weaknesses" in patch
        assert "Active Directives" in patch

    def test_healthy_report(self):
        report = analyze_outcomes(_outcomes_healthy())
        patch = generate_compiler_prompt_patch(report)
        assert "Active Directives" in patch


# ===================================================================
# GOAL GENERATOR
# ===================================================================

class TestImprovementGoal:
    def test_frozen(self):
        g = ImprovementGoal(
            goal_id="G-001", priority="high", category="confidence",
            description="test", source="test",
        )
        with pytest.raises(AttributeError):
            g.priority = "low"

    def test_defaults(self):
        g = ImprovementGoal(
            goal_id="G-001", priority="high", category="confidence",
            description="test", source="test",
        )
        assert g.target_postcodes == ()
        assert g.estimated_effort == "unknown"
        assert g.success_metric == ""


class TestGoalsFromGrid:
    def test_no_issues(self):
        cells = [("INT.SEM.ECO.WHY.SFT", 0.90, "F", "intent")]
        goals = goals_from_grid(cells, _CORE_LAYERS, 1)
        # No low confidence, all core layers present
        confidence_goals = [g for g in goals if g.category == "confidence"]
        assert len(confidence_goals) == 0

    def test_critical_confidence(self):
        cells = [
            ("INT.SEM.ECO.WHY.SFT", 0.20, "P", "intent"),
            ("ORG.ENT.APP.WHAT.SFT", 0.15, "F", "entity"),
        ]
        goals = goals_from_grid(cells, _CORE_LAYERS, 2)
        critical = [g for g in goals if g.priority == "critical" and g.category == "confidence"]
        assert len(critical) == 1
        assert len(critical[0].target_postcodes) == 2

    def test_warning_confidence(self):
        cells = [("INT.SEM.ECO.WHY.SFT", 0.40, "P", "intent")]
        goals = goals_from_grid(cells, _CORE_LAYERS, 1)
        high = [g for g in goals if g.priority == "high" and g.category == "confidence"]
        assert len(high) == 1

    def test_watch_confidence(self):
        cells = [("INT.SEM.ECO.WHY.SFT", 0.60, "P", "intent")]
        goals = goals_from_grid(cells, _CORE_LAYERS, 1)
        medium = [g for g in goals if g.priority == "medium" and g.category == "confidence"]
        assert len(medium) == 1

    def test_missing_layers(self):
        cells = [("INT.SEM.ECO.WHY.SFT", 0.90, "F", "intent")]
        active = {"INT"}  # missing 6 core layers
        goals = goals_from_grid(cells, active, 1)
        coverage = [g for g in goals if g.category == "coverage"]
        assert len(coverage) >= 1
        assert "Core layers" in coverage[0].description

    def test_sparse_layers(self):
        cells = [
            ("INT.SEM.ECO.WHY.SFT", 0.90, "F", "intent"),
            # INT has only 1 cell → sparse
        ]
        active = _CORE_LAYERS
        goals = goals_from_grid(cells, active, 1)
        sparse = [g for g in goals if "Sparse" in g.description]
        assert len(sparse) == 1

    def test_quarantined_cells(self):
        cells = [
            ("INT.SEM.ECO.WHY.SFT", 0.10, "Q", "intent"),
            ("ORG.ENT.APP.WHAT.SFT", 0.05, "Q", "entity"),
        ]
        goals = goals_from_grid(cells, _CORE_LAYERS, 2)
        quarantined = [g for g in goals if g.category == "resilience"]
        assert len(quarantined) == 1
        assert "2 quarantined" in quarantined[0].description

    def test_empty_grid(self):
        goals = goals_from_grid([], set(), 0)
        coverage = [g for g in goals if g.category == "coverage"]
        assert len(coverage) >= 1  # missing core layers

    def test_ignores_empty_fill_state(self):
        cells = [("INT.SEM.ECO.WHY.SFT", 0.10, "E", "intent")]
        goals = goals_from_grid(cells, _CORE_LAYERS, 1)
        # E cells not counted as low confidence
        confidence = [g for g in goals if g.category == "confidence"]
        assert len(confidence) == 0


class TestGoalsFromFeedback:
    def test_no_issues(self):
        goals = goals_from_feedback([], 0.0, "stable")
        assert len(goals) == 0

    def test_high_rejection_rate(self):
        goals = goals_from_feedback([], 0.6, "stable")
        assert len(goals) == 1
        assert goals[0].priority == "critical"

    def test_moderate_rejection_rate(self):
        goals = goals_from_feedback([], 0.35, "stable")
        assert len(goals) == 1
        assert goals[0].priority == "high"

    def test_degrading_trend(self):
        goals = goals_from_feedback([], 0.0, "degrading")
        assert len(goals) == 1
        assert "trending downward" in goals[0].description

    def test_weakness_dimensions(self):
        weaknesses = [
            ("completeness", "critical", 30.0),
            ("consistency", "warning", 55.0),
        ]
        goals = goals_from_feedback(weaknesses, 0.0, "stable")
        assert len(goals) == 2

    def test_stable_trend_no_goals(self):
        goals = goals_from_feedback([], 0.1, "stable")
        assert len(goals) == 0

    def test_insufficient_data_no_goals(self):
        goals = goals_from_feedback([], 0.0, "insufficient_data")
        assert len(goals) == 0


class TestGoalsFromAnomalies:
    def test_no_anomalies(self):
        goals = goals_from_anomalies(0, [])
        assert len(goals) == 0

    def test_few_anomalies(self):
        goals = goals_from_anomalies(2, ["A.B.C.D.E", "F.G.H.I.J"])
        assert len(goals) == 1
        assert goals[0].priority == "medium"

    def test_moderate_anomalies(self):
        goals = goals_from_anomalies(4, ["A"] * 4)
        assert len(goals) == 1
        assert goals[0].priority == "high"

    def test_many_anomalies(self):
        goals = goals_from_anomalies(8, ["A"] * 8)
        assert len(goals) == 1
        assert goals[0].priority == "critical"

    def test_postcodes_capped(self):
        goals = goals_from_anomalies(15, [f"A{i}" for i in range(15)])
        assert len(goals[0].target_postcodes) <= 10


class TestGenerateGoalSet:
    def test_empty(self):
        gs = generate_goal_set([], [], [])
        assert gs.total_goals == 0
        assert "No improvement" in gs.analysis_summary

    def test_with_critical(self):
        grid = [ImprovementGoal(
            goal_id="G-001", priority="critical", category="confidence",
            description="Critical issue", source="test",
        )]
        gs = generate_goal_set(grid, [], [])
        assert gs.critical_count == 1
        assert "critical" in gs.analysis_summary.lower()

    def test_merges_sources(self):
        grid = [ImprovementGoal(
            goal_id="G-001", priority="medium", category="coverage",
            description="Coverage gap", source="grid",
        )]
        feedback = [ImprovementGoal(
            goal_id="G-101", priority="high", category="quality",
            description="Quality issue", source="feedback",
        )]
        anomaly = [ImprovementGoal(
            goal_id="G-200", priority="medium", category="resilience",
            description="Anomalies", source="observer",
        )]
        gs = generate_goal_set(grid, feedback, anomaly)
        assert gs.total_goals == 3
        # High priority first
        assert gs.goals[0].priority == "high"

    def test_priority_ordering(self):
        goals = [
            ImprovementGoal(goal_id="G-1", priority="low", category="confidence",
                          description="low", source="t"),
            ImprovementGoal(goal_id="G-2", priority="critical", category="quality",
                          description="crit", source="t"),
            ImprovementGoal(goal_id="G-3", priority="high", category="coverage",
                          description="high", source="t"),
            ImprovementGoal(goal_id="G-4", priority="medium", category="resilience",
                          description="med", source="t"),
        ]
        gs = generate_goal_set(goals, [], [])
        priorities = [g.priority for g in gs.goals]
        assert priorities[0] == "critical"
        assert priorities[-1] == "low"

    def test_counts(self):
        goals = [
            ImprovementGoal(goal_id="G-1", priority="critical", category="coverage",
                          description="a", source="t"),
            ImprovementGoal(goal_id="G-2", priority="high", category="confidence",
                          description="b", source="t"),
            ImprovementGoal(goal_id="G-3", priority="medium", category="confidence",
                          description="c", source="t"),
        ]
        gs = generate_goal_set(goals, [], [])
        assert gs.critical_count == 1
        assert gs.coverage_gaps == 1
        assert gs.confidence_issues == 2
