"""
mother/governor_feedback.py — Rejection analysis → compiler refinement.

LEAF module. Analyzes compilation outcomes (trust scores, verification failures,
governor rejections) and produces refinement signals that feed back into future
compilations. This is the L2 feedback loop: F({O}) → patterns.

The governor doesn't just gate — it teaches. Every rejection carries information
about what the compiler got wrong. This module extracts that information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompilationOutcome:
    """Record of a single compilation's quality signals."""

    compile_id: str
    input_summary: str  # first 200 chars of input
    trust_score: float  # 0-100 from verification
    completeness: float  # 0-100
    consistency: float  # 0-100
    coherence: float  # 0-100
    traceability: float  # 0-100
    actionability: float  # 0-100
    specificity: float  # 0-100
    codegen_readiness: float  # 0-100
    component_count: int
    rejected: bool = False
    rejection_reason: str = ""
    domain: str = "software"
    compression_loss_categories: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class WeaknessSignal:
    """A specific weakness detected across compilations."""

    dimension: str  # "completeness", "consistency", "coherence", "traceability"
    severity: str  # "critical", "warning", "watch"
    mean_score: float
    occurrences: int
    pattern: str  # human-readable description
    remediation: str  # actionable hint for the compiler


@dataclass(frozen=True)
class FeedbackReport:
    """Aggregate feedback from compilation history."""

    outcomes_analyzed: int
    rejection_rate: float  # 0.0-1.0
    weaknesses: tuple[WeaknessSignal, ...]
    strengths: tuple[str, ...]
    compiler_hints: tuple[str, ...]  # ordered, most important first
    trend: str  # "improving", "degrading", "stable", "insufficient_data"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Trust dimensions and their weights for overall quality
_TRUST_DIMENSIONS = ("completeness", "consistency", "coherence", "traceability",
                      "actionability", "specificity", "codegen_readiness")

# Severity thresholds (mean score across outcomes)
_CRITICAL_THRESHOLD = 40.0
_WARNING_THRESHOLD = 60.0
_WATCH_THRESHOLD = 75.0

# Minimum outcomes needed for trend analysis
_MIN_TREND_OUTCOMES = 3

# Remediation hints per dimension
_REMEDIATION = {
    "completeness": "PROTOCOL FOR COMPLETENESS 100%: 1. Extract ALL nouns/entities/verbs/phrases from input (count them). 2. MAP EACH 1:1 to components/methods/relationships (NO omissions). 3. Verify 100% coverage. 4. Document: 'Coverage: 100% (18 nouns/verbs -> 8 components, 25 methods)' in blueprint.description. Example: input 'real-time dashboard with charts, user auth' -> dashboard-service (fetchMetrics, renderChart), auth-service (login, refreshToken), realtime-hub (subscribeUser). ZERO unmapped elements.",
    "consistency": "ENSURE CONSISTENCY 98%: ZERO overlaps/duplicates across components. EXACT data contracts: types/names/constraints match 100% in relationships. Example: user-service.createUser RETURNS {'user_id': 'str(UUIDv4)', 'email': 'str'}, auth.validateUser INPUT {'user_id': 'str(UUIDv4)', 'email': 'str'}. Relationships MUST document ALL flows: {'from':'user.createUser','to':'auth.validateUser','payload':'user_id:str(UUIDv4),email:str'}. Data models vs logic services SEPARATE. Verify NO conflicts before finalize.",
    "coherence": "Align component responsibilities with domain semantics. Reduce scope overlap between components.",
    "traceability": "Strengthen provenance chains. Every blueprint element must trace to a specific input phrase.",
    "actionability": """ENSURE ACTIONABILITY 98%: 1. Extract ALL verbs/phrases from input (count V). 2. MAP to 3-6 imperative methods PER COMPONENT (avg 4/component). 95%+ components MUST have methods (NO data-only). 3. EVERY method: FULLY typed params/returns w/ EXACT constraints (use formats from specificity). 4. blueprint.description MUST state: 'Actionability: 98% (V=12 verbs → M=42 methods | C=14 components, P=100% have methods)'. Examples:
- {'name':'processOrder','params':{'order_id':'str(UUIDv4)','items':'list[dict(id:str(UUIDv4),qty:int(1-100),price:decimal(8,2)>0)]','shipping':'str(enum[standard,express,overnight])','coupon':'str(opt)"},"returns":{'order_id':'str(UUIDv4)','status':'enum[pending,shipped,delivered,cancelled]','total':decimal(10,2),'errors':'list[str(maxlen=100)]'}}
- {'name':'validatePayment','params':{'token':'str(PCI_DSS)','amount':decimal(10,2,min=0.01)},"returns":{'approved':'bool','transaction_id':'str(opt)','decline_reason':'str(enum[insufficient_funds,invalid_card,expired])'}}
Verify: COUNT verbs/methods/components/% BEFORE finalize. FAIL if <95% components have methods or missing types.""",
    "specificity": """CRITICAL BOOST SPECIFICITY TO 100%: COUNT ALL fields/params/returns (N=total). TYPE+CONSTRAINT 100%. FAIL <98%. desc += 'Specificity: 100% (N=62/62 ALL typed/constrained)'.

MANDATORY FORMATS (USE THESE):
user_id: str(UUIDv4,primary_key)
email: str(RFC5322_valid,unique,normalized)
amount: decimal(10,2,min=0,max=999999.99)
status: enum[pending,active,suspended,archived,deleted]
tags: list[str(nonempty,maxlen=10)]
created_at: datetime(ISO8601,UTC)
errors: list[str(maxlen=200)]

ZERO: 'data','info','str','any','dict','object'. BAD:'user:dict' GOOD:'user:dict(id:str(UUIDv4),email:str(RFC5322_unique),status:enum[active,suspended])'

VERIFY: EVERY param/return typed+constrained. COUNT in desc. Governor rejects vague blueprints 100%.""",
    "codegen_readiness": """MANDATORY 'codegen' BLOCK (REJECTED 100% WITHOUT):

STRUCTURE:
'codegen': {
  'language': 'python3.12',  # or typescript/rust
  'framework': 'fastapi/pydantic/sqlmodel',  # web:fastapi, data:pandas/sqlmodel, ML:pytorch, UI:nextjs, infra:terraform
  'dependencies': ['fastapi==0.112.0', 'pydantic==2.8.2', 'sqlmodel==0.0.8'],
  'files': [  # 3 files PER COMPONENT: service.py, models.py, tests.py
    {
      'path': 'services/user_service.py',
      'content': '''from typing import Dict
from pydantic import BaseModel

class CreateUserReq(BaseModel):
    email: str  # RFC5322_valid, unique
    password: str  # minlen=12

def create_user(req: CreateUserReq) -> Dict[str, Any]:
    \"\"\"Create new user.\"\"\"
    # TODO: validate, hash pwd, persist
    return {"user_id": "uuid-123", "status": "created"}

def test_create_user():
    result = create_user(CreateUserReq(email="test@example.com", password="pass123456"))
    assert result["status"] == "created"
'''
    }
  ]
}

RULES:
- EVERY component -> AT LEAST 3 files (service.py, models.py, tests.py)
- EVERY method -> typed stub + docstring + #TODO + unit test
- SYNTAX 100% VALID: copy-paste into IDE runs without errors
- NO DANGEROUS CODE: requests/pathlib/sqlite3 ONLY
- Domain-specific: web->FastAPI, data->Pandas/Numpy, ML->PyTorch
- desc += 'Codegen ready: 100% (N files, M methods stubbed/tests)'

VERIFY ALL before output. FAIL=REFINE.""",
}

# Compression loss remediation hints per category
_COMPRESSION_REMEDIATION = {
    "entity": """PROTOCOL FOR ENTITY RETENTION 100%: 1. Extract ALL nouns/entities/phrases from input (count N). 2. MAP EACH 1:1 to blueprint components/fields/method params/relationships (NO omissions). 3. Verify 100% coverage — fuzzy match names/attributes. 4. Document in blueprint.description: 'Entity coverage: 100% (N=18 nouns → 18 blueprint elements)'. Examples: 'user dashboard' → UserDashboard component; 'auth token' → auth_token:str(UUIDv4) field. ZERO unmapped nouns/entities.""",
    "constraint": "Capture all constraints explicitly. Check for implicit constraints in user language. MANDATORY: min/max/enum/unique/required in ALL fields/params. State: 'Constraints: 100% (12/12 explicit)' in desc.",
    "behavior": """PROTOCOL FOR BEHAVIOR RETENTION 100%: 1. Extract ALL verbs/actions/workflows from input (count V). 2. MAP to imperative methods in components (3-6 per component). 3. Verify 100% coverage. 4. Document: 'Behavior coverage: 100% (V=12 verbs → 42 methods)'. Example: 'handle login' → loginUser(params:email:str(RFC5322),password:str(minlen=12)) returns:{user_id:str(UUIDv4),token:str(JWT),errors:list[str]}. NO unmapped actions.""",
    "relationship": "Preserve ALL connections: 'user → auth → payment'. Document EVERY flow in relationships[]. State: 'Relationships: 100% (8 flows mapped)'. Example: {'from':'user.create','to':'auth.validate','payload':'user_id:str(UUIDv4)'}",
    "context": "Retain ALL domain context/background/assumptions. NO stripping qualifiers ('real-time', 'secure', 'scalable'). Include in component desc. Verify: input qualifiers → blueprint qualifiers.",
}

# Rejection reason patterns → compiler hints
_REJECTION_PATTERNS = {
    "trust": "Prior compilations failed trust checks. Focus on verification scores before synthesis.",
    "quality": "Input quality was too low. Consider interrogation phase for sparse inputs.",
    "cost": "Cost cap hit. Reduce dialogue rounds or component count.",
    "timeout": "Compilation timed out. Simplify decomposition or reduce agent rounds.",
    "empty": "Blueprint was empty. Ensure synthesis agent receives sufficient dialogue context.",
    "provenance": "CRITICAL PROVENANCE: EVERY blueprint element has 'provenance.instance_id' == FULL input instance_id (no short 'i1'). COPY EXACTLY from compile context. Governor rejects 100% suspicious short IDs. VERIFY lengths match before output.",
    "safety": "CRITICAL SAFETY: NO subprocess/exec/eval/sh/os.system/os.popen in ANY method/behavior/returns. Governor rejects 100%. Use safe: requests, sqlite3, pathlib. Example safe: {'name':'fetchData','params':{'url':'str'},'returns':{'data':'dict'}}.",
}


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def analyze_outcomes(outcomes: list[CompilationOutcome]) -> FeedbackReport:
    """Analyze a batch of compilation outcomes and produce feedback.

    This is the main entry point. Feed it all recent outcomes
    and it returns actionable refinement signals.
    """
    if not outcomes:
        return FeedbackReport(
            outcomes_analyzed=0,
            rejection_rate=0.0,
            weaknesses=(),
            strengths=(),
            compiler_hints=("No compilation history available.",),
            trend="insufficient_data",
        )

    n = len(outcomes)
    rejected_count = sum(1 for o in outcomes if o.rejected)
    rejection_rate = rejected_count / n

    # Compute per-dimension means
    dim_scores: dict[str, list[float]] = {d: [] for d in _TRUST_DIMENSIONS}
    for o in outcomes:
        dim_scores["completeness"].append(o.completeness)
        dim_scores["consistency"].append(o.consistency)
        dim_scores["coherence"].append(o.coherence)
        dim_scores["traceability"].append(o.traceability)
        dim_scores["actionability"].append(getattr(o, "actionability", 65.0))
        dim_scores["specificity"].append(getattr(o, "specificity", 65.0))
        dim_scores["codegen_readiness"].append(getattr(o, "codegen_readiness", 65.0))

    dim_means = {d: sum(s) / len(s) for d, s in dim_scores.items() if s}

    # Detect weaknesses
    weaknesses = _detect_weaknesses(dim_means, dim_scores)

    # Detect compression loss weaknesses
    compression_weaknesses = _detect_compression_weaknesses(outcomes)
    weaknesses.extend(compression_weaknesses)

    # Detect strengths
    strengths = _detect_strengths(dim_means)

    # Generate compiler hints
    hints = _generate_hints(outcomes, weaknesses, rejection_rate)

    # Compute trend
    trend = _compute_trend(outcomes)

    return FeedbackReport(
        outcomes_analyzed=n,
        rejection_rate=round(rejection_rate, 4),
        weaknesses=tuple(weaknesses),
        strengths=tuple(strengths),
        compiler_hints=tuple(hints),
        trend=trend,
    )


def extract_rejection_patterns(outcomes: list[CompilationOutcome]) -> dict[str, int]:
    """Extract rejection reason frequency from outcomes."""
    patterns: dict[str, int] = {}
    for o in outcomes:
        if o.rejected and o.rejection_reason:
            # Normalize reason to category
            category = _categorize_rejection(o.rejection_reason)
            patterns[category] = patterns.get(category, 0) + 1
    return patterns


def score_compiler_health(outcomes: list[CompilationOutcome]) -> float:
    """Single 0-100 health score for the compiler based on recent outcomes.

    Factors: trust scores, rejection rate, dimension balance.
    """
    if not outcomes:
        return 50.0  # neutral when no data

    trust_mean = sum(o.trust_score for o in outcomes) / len(outcomes)
    rejection_penalty = sum(1 for o in outcomes if o.rejected) / len(outcomes) * 30

    # Balance penalty: max - min dimension score
    dim_means = _dimension_means(outcomes)
    if dim_means:
        balance = max(dim_means.values()) - min(dim_means.values())
        balance_penalty = min(balance * 0.3, 15.0)  # cap at 15
    else:
        balance_penalty = 0.0

    health = trust_mean - rejection_penalty - balance_penalty
    return round(max(0.0, min(100.0, health)), 2)


def generate_compiler_prompt_patch(report: FeedbackReport) -> str:
    """Generate a prompt patch that can be injected into the compiler's system prompt.

    This is how the feedback loop closes: analysis becomes instruction.
    """
    if not report.compiler_hints:
        return ""

    lines = ["## Compiler Self-Improvement Directives", ""]
    lines.append(f"Based on {report.outcomes_analyzed} recent compilations:")
    lines.append(f"- Trend: {report.trend}")
    lines.append(f"- Rejection rate: {report.rejection_rate:.1%}")
    lines.append("")

    if report.weaknesses:
        lines.append("### Known Weaknesses")
        for w in report.weaknesses:
            lines.append(f"- [{w.severity.upper()}] {w.dimension}: {w.pattern}")
            lines.append(f"  Fix: {w.remediation}")
        lines.append("")

    if report.compiler_hints:
        lines.append("### Active Directives")
        for i, hint in enumerate(report.compiler_hints, 1):
            lines.append(f"{i}. {hint}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dimension_means(outcomes: list[CompilationOutcome]) -> dict[str, float]:
    """Compute mean score per trust dimension."""
    if not outcomes:
        return {}
    return {
        "completeness": sum(o.completeness for o in outcomes) / len(outcomes),
        "consistency": sum(o.consistency for o in outcomes) / len(outcomes),
        "coherence": sum(o.coherence for o in outcomes) / len(outcomes),
        "traceability": sum(o.traceability for o in outcomes) / len(outcomes),
    }


def _detect_weaknesses(
    dim_means: dict[str, float],
    dim_scores: dict[str, list[float]],
) -> list[WeaknessSignal]:
    """Detect weakness signals from dimension scores."""
    weaknesses: list[WeaknessSignal] = []

    for dim in _TRUST_DIMENSIONS:
        mean = dim_means.get(dim, 0.0)
        scores = dim_scores.get(dim, [])
        count = len(scores)

        if mean < _CRITICAL_THRESHOLD:
            severity = "critical"
            pattern = f"Mean {dim} score {mean:.1f} is critically low across {count} compilations."
        elif mean < _WARNING_THRESHOLD:
            severity = "warning"
            pattern = f"Mean {dim} score {mean:.1f} is below acceptable threshold."
        elif mean < _WATCH_THRESHOLD:
            severity = "watch"
            pattern = f"Mean {dim} score {mean:.1f} is marginal — monitor for degradation."
        else:
            continue  # no weakness

        weaknesses.append(WeaknessSignal(
            dimension=dim,
            severity=severity,
            mean_score=round(mean, 2),
            occurrences=count,
            pattern=pattern,
            remediation=_REMEDIATION.get(dim, "Review compilation output for this dimension."),
        ))

    # Sort: critical first, then warning, then watch
    severity_order = {"critical": 0, "warning": 1, "watch": 2}
    weaknesses.sort(key=lambda w: (severity_order.get(w.severity, 3), w.mean_score))
    return weaknesses


def _detect_compression_weaknesses(
    outcomes: list[CompilationOutcome],
) -> list[WeaknessSignal]:
    """Detect weakness signals from compression loss categories across outcomes."""
    if not outcomes:
        return []

    # Count how often each category appears across outcomes
    cat_count: dict[str, int] = {}
    cat_severity_sum: dict[str, float] = {}
    outcomes_with_cats = 0
    for o in outcomes:
        if o.compression_loss_categories:
            outcomes_with_cats += 1
            for cat, sev in o.compression_loss_categories:
                cat_count[cat] = cat_count.get(cat, 0) + 1
                cat_severity_sum[cat] = cat_severity_sum.get(cat, 0.0) + sev

    if outcomes_with_cats == 0:
        return []

    weaknesses: list[WeaknessSignal] = []
    for cat, count in cat_count.items():
        freq = count / outcomes_with_cats
        if freq <= 0.30:
            continue
        severity = "critical" if freq > 0.50 else "warning"
        avg_sev = cat_severity_sum[cat] / count
        weaknesses.append(WeaknessSignal(
            dimension=f"compression:{cat}",
            severity=severity,
            mean_score=round(avg_sev, 2),
            occurrences=count,
            pattern=f"{cat} compression losses in {freq:.0%} of compilations (avg severity {avg_sev:.2f}).",
            remediation=_COMPRESSION_REMEDIATION.get(cat, "Review compression losses for this category."),
        ))

    # Sort: critical first
    severity_order = {"critical": 0, "warning": 1}
    weaknesses.sort(key=lambda w: (severity_order.get(w.severity, 2), -w.occurrences))
    return weaknesses


def _detect_strengths(dim_means: dict[str, float]) -> list[str]:
    """Identify strong dimensions."""
    strengths = []
    for dim in _TRUST_DIMENSIONS:
        mean = dim_means.get(dim, 0.0)
        if mean >= 85.0:
            strengths.append(f"{dim} is strong at {mean:.1f}%")
        elif mean >= _WATCH_THRESHOLD:
            strengths.append(f"{dim} is healthy at {mean:.1f}%")
    return strengths


def _generate_hints(
    outcomes: list[CompilationOutcome],
    weaknesses: list[WeaknessSignal],
    rejection_rate: float,
) -> list[str]:
    """Generate ordered compiler hints from analysis."""
    hints: list[str] = []

    # Rejection rate hints
    if rejection_rate >= 1.0:
        hints.append("""EMERGENCY: 100% rejection rate across all recent compilations. Root cause analysis required: governor thresholds, prompts, agent chains.

CRITICAL VERIFICATION CHECKLIST (rejection_rate=100%): MANDATORY PRE-OUTPUT CHECK.
1. PROVENANCE: 100% coverage w/ FULL instance_id on EVERY elem (copy exact from context).
2. SAFETY: NO subprocess/exec/eval/sh/os.system.
3. CODEGEN: Root block w/ domain framework + stubs/tests EVERY component/method.
4. SPECIFICITY: 100% fields/params typed+constrained, COUNT in desc.
5. DIMS: All trust dims >=92%.
6. COVERAGE: State exact noun->component mapping count in desc.
FAIL=REFINE.""")
    elif rejection_rate >= 0.5:
        hints.append("CRITICAL: Over half of recent compilations were rejected. Review input quality and agent configuration.")
    elif rejection_rate > 0.2:
        hints.append("WARNING: Significant rejection rate. Focus on the weakest trust dimension.")

    # Weakness-specific hints
    for w in weaknesses:
        if w.severity == "critical":
            hints.append(f"CRITICAL: {w.remediation}")
        elif w.severity == "warning":
            hints.append(f"Focus area: {w.remediation}")

    # Rejection pattern hints
    patterns = extract_rejection_patterns(outcomes)
    for category, count in sorted(patterns.items(), key=lambda x: -x[1]):
        hint = _REJECTION_PATTERNS.get(category)
        if hint and count >= 2:
            hints.append(hint)

    # Domain concentration check
    domains = {o.domain for o in outcomes}
    if len(domains) == 1:
        hints.append(f"All compilations in '{next(iter(domains))}' domain. Cross-domain generalization untested.")

    if not hints:
        hints.append("No significant issues detected. Maintain current approach.")

    return hints


def _compute_trend(outcomes: list[CompilationOutcome]) -> str:
    """Compute trend direction from outcome sequence."""
    if len(outcomes) < _MIN_TREND_OUTCOMES:
        return "insufficient_data"

    # Split into halves and compare trust scores
    mid = len(outcomes) // 2
    first_half = outcomes[:mid]
    second_half = outcomes[mid:]

    first_mean = sum(o.trust_score for o in first_half) / len(first_half)
    second_mean = sum(o.trust_score for o in second_half) / len(second_half)

    delta = second_mean - first_mean
    if delta > 5.0:
        return "improving"
    elif delta < -5.0:
        return "degrading"
    return "stable"


def _categorize_rejection(reason: str) -> str:
    """Map a rejection reason string to a category."""
    lower = reason.lower()
    if "trust" in lower or "score" in lower:
        return "trust"
    if "quality" in lower or "input" in lower:
        return "quality"
    if "cost" in lower or "budget" in lower:
        return "cost"
    if "timeout" in lower or "time" in lower:
        return "timeout"
    if "empty" in lower or "no component" in lower:
        return "empty"
    return "other"
