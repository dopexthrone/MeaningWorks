"""Tests for governor_feedback."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from governor_feedback import (
    analyze_outcomes,
    CompilationOutcome,
    score_compiler_health,
)


def test_analyze_outcomes_100_rejection():
    outcomes = [
        CompilationOutcome(
            compile_id="test1",
            input_summary="test input",
            trust_score=30.0,
            completeness=20.0,
            consistency=40.0,
            coherence=30.0,
            traceability=10.0,
            actionability=50.0,
            specificity=60.0,
            codegen_readiness=40.0,
            component_count=3,
            rejected=True,
            rejection_reason="low trust scores",
        )
    ]
    report = analyze_outcomes(outcomes)
    assert report.rejection_rate == 1.0
    assert report.outcomes_analyzed == 1
    assert any("EMERGENCY: 100% rejection rate" in hint for hint in report.compiler_hints)


def test_analyze_outcomes_mixed():
    outcomes = [
        CompilationOutcome(
            compile_id="pass",
            input_summary="test",
            trust_score=90.0,
            completeness=90.0,
            consistency=90.0,
            coherence=90.0,
            traceability=90.0,
            actionability=90.0,
            specificity=90.0,
            codegen_readiness=90.0,
            component_count=5,
            rejected=False,
        ),
        CompilationOutcome(
            compile_id="fail",
            input_summary="test",
            trust_score=20.0,
            completeness=20.0,
            consistency=20.0,
            coherence=20.0,
            traceability=20.0,
            actionability=20.0,
            specificity=20.0,
            codegen_readiness=20.0,
            component_count=2,
            rejected=True,
        ),
    ]
    report = analyze_outcomes(outcomes)
    assert report.rejection_rate == 0.5
    assert any("CRITICAL" in hint for hint in report.compiler_hints)


def test_score_compiler_health_100_rejection():
    outcomes = [
        CompilationOutcome(
            compile_id="1",
            input_summary="test",
            trust_score=20.0,
            completeness=20.0,
            consistency=20.0,
            coherence=20.0,
            traceability=20.0,
            actionability=20.0,
            specificity=20.0,
            codegen_readiness=20.0,
            component_count=1,
            rejected=True,
        )
    ]
    health = score_compiler_health(outcomes)
    assert health < 50.0  # rejection penalty dominates


def test_actionability_remediation():
    outcomes = [
        CompilationOutcome(
            compile_id="act1",
            input_summary="test",
            trust_score=80.0,
            completeness=90.0,
            consistency=90.0,
            coherence=90.0,
            traceability=90.0,
            actionability=50.0,
            specificity=90.0,
            codegen_readiness=90.0,
            component_count=5,
        )
    ]
    report = analyze_outcomes(outcomes)
    weaknesses = list(report.weaknesses)
    assert len(weaknesses) >= 1
    act_weak = next((w for w in weaknesses if w.dimension == "actionability"), None)
    assert act_weak is not None
    assert act_weak.severity == "warning"
    remediation = act_weak.remediation
    assert "ENSURE ACTIONABILITY" in remediation
    assert "verbs →" in remediation
    assert any("Focus area:" in h and "ENSURE ACTIONABILITY" in h for h in report.compiler_hints)