"""
Financial operations — heuristic reasoning about costs, billing, margins, financial health.

LEAF module. Genome #116 (financial-operating).

All functions are pure — no external API calls, no LLM invocations.
Heuristic analysis over structured financial inputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_BILLING_CYCLE_PATTERNS: Dict[str, List[str]] = {
    "monthly": ["monthly", "per month", "/mo", "month-to-month", "30 days"],
    "annual": ["annual", "yearly", "per year", "/yr", "12 months", "365 days"],
    "quarterly": ["quarterly", "per quarter", "every 3 months", "q1", "q2", "q3", "q4"],
    "weekly": ["weekly", "per week", "/wk", "7 days"],
    "one-time": ["one-time", "once", "single payment", "lifetime", "perpetual"],
    "usage-based": ["per use", "pay-as-you-go", "metered", "per request", "per call", "per unit"],
}

_COST_CATEGORIES: Dict[str, List[str]] = {
    "infrastructure": ["hosting", "server", "cloud", "aws", "gcp", "azure", "cdn", "storage", "compute", "database"],
    "personnel": ["salary", "wages", "contractor", "freelance", "hire", "team", "developer", "designer"],
    "tooling": ["license", "subscription", "saas", "tool", "software", "ide", "ci/cd"],
    "marketing": ["ads", "advertising", "seo", "content", "campaign", "brand", "social media"],
    "operations": ["office", "legal", "accounting", "insurance", "compliance", "admin"],
    "api": ["api cost", "token", "openai", "anthropic", "llm", "model", "inference"],
}

_HEALTH_THRESHOLDS = {
    "margin_critical": 0.05,
    "margin_warning": 0.15,
    "margin_healthy": 0.30,
    "ltv_cac_critical": 1.0,
    "ltv_cac_warning": 3.0,
    "ltv_cac_healthy": 5.0,
}


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialSnapshot:
    """Point-in-time financial state summary."""
    total_revenue: float
    total_costs: float
    gross_margin: float  # (revenue - costs) / revenue, 0 if no revenue
    cost_breakdown: Tuple[Tuple[str, float], ...]  # (category, amount) pairs
    billing_model: str  # detected billing cycle
    health_status: str  # "critical", "warning", "moderate", "healthy"
    recommendations: Tuple[str, ...]


# ---------------------------------------------------------------------------
# #116 — Financial-operating: cost estimation
# ---------------------------------------------------------------------------

def estimate_project_cost(
    description: str,
    team_size: int = 1,
    duration_months: int = 3,
    hourly_rate: float = 75.0,
) -> Dict[str, object]:
    """Estimate project cost from description, team, and duration.

    Returns dict with: total_estimate, personnel_cost, infrastructure_estimate,
    tooling_estimate, contingency, breakdown, reasoning.
    """
    desc_lower = description.lower()
    desc_words = frozenset(desc_lower.split())

    # Personnel cost — primary driver
    hours_per_month = 160
    personnel = team_size * duration_months * hours_per_month * hourly_rate

    # Infrastructure estimate from keywords
    infra_keywords = frozenset(_COST_CATEGORIES["infrastructure"])
    infra_signal = len(desc_words & infra_keywords)
    if infra_signal >= 3:
        infrastructure = personnel * 0.25  # heavy infra
    elif infra_signal >= 1:
        infrastructure = personnel * 0.15  # moderate infra
    else:
        infrastructure = personnel * 0.08  # minimal infra

    # Tooling estimate
    tool_keywords = frozenset(_COST_CATEGORIES["tooling"])
    api_keywords = frozenset(_COST_CATEGORIES["api"])
    tool_signal = len(desc_words & tool_keywords) + len(desc_words & api_keywords)
    if tool_signal >= 3:
        tooling = personnel * 0.15
    elif tool_signal >= 1:
        tooling = personnel * 0.08
    else:
        tooling = personnel * 0.03

    # Contingency (15% buffer)
    subtotal = personnel + infrastructure + tooling
    contingency = subtotal * 0.15
    total = subtotal + contingency

    breakdown = {
        "personnel": round(personnel, 2),
        "infrastructure": round(infrastructure, 2),
        "tooling": round(tooling, 2),
        "contingency": round(contingency, 2),
    }

    reasoning = (
        f"{team_size} person(s) x {duration_months} months @ ${hourly_rate}/hr. "
        f"Infrastructure {'heavy' if infra_signal >= 3 else 'moderate' if infra_signal >= 1 else 'light'}. "
        f"Tooling {'significant' if tool_signal >= 3 else 'moderate' if tool_signal >= 1 else 'minimal'}. "
        f"15% contingency buffer applied."
    )

    return {
        "total_estimate": round(total, 2),
        "personnel_cost": round(personnel, 2),
        "infrastructure_estimate": round(infrastructure, 2),
        "tooling_estimate": round(tooling, 2),
        "contingency": round(contingency, 2),
        "breakdown": breakdown,
        "reasoning": reasoning,
    }


def detect_billing_cycle(description: str) -> Dict[str, str]:
    """Detect billing cycle from text description.

    Returns dict with: cycle, confidence, reasoning.
    """
    desc_lower = description.lower()

    matches: List[Tuple[str, int]] = []
    for cycle, patterns in _BILLING_CYCLE_PATTERNS.items():
        count = sum(1 for p in patterns if p in desc_lower)
        if count > 0:
            matches.append((cycle, count))

    if not matches:
        return {
            "cycle": "unknown",
            "confidence": "low",
            "reasoning": "No billing cycle indicators found in description",
        }

    matches.sort(key=lambda x: x[1], reverse=True)
    best_cycle, best_count = matches[0]

    confidence = "high" if best_count >= 2 else "medium"

    return {
        "cycle": best_cycle,
        "confidence": confidence,
        "reasoning": f"Detected {best_cycle} billing from {best_count} indicator(s)",
    }


def compute_margins(
    revenue: float,
    costs: float,
    cost_items: Optional[List[Tuple[str, float]]] = None,
) -> Dict[str, object]:
    """Compute margin analysis from revenue and cost data.

    Returns dict with: gross_margin, net_margin_estimate, cost_ratio,
    largest_cost_category, optimization_suggestions.
    """
    if revenue <= 0:
        return {
            "gross_margin": 0.0,
            "net_margin_estimate": 0.0,
            "cost_ratio": float("inf") if costs > 0 else 0.0,
            "largest_cost_category": "n/a",
            "optimization_suggestions": ["Generate revenue before optimizing margins"],
        }

    gross_margin = (revenue - costs) / revenue
    # Estimate net = gross - 15% overhead (taxes, admin, etc.)
    net_margin_estimate = gross_margin - 0.15

    cost_ratio = costs / revenue

    # Find largest cost category
    items = cost_items or []
    largest = "unknown"
    if items:
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        largest = items_sorted[0][0]

    suggestions: List[str] = []
    if gross_margin < _HEALTH_THRESHOLDS["margin_critical"]:
        suggestions.append("Margins critically low — reduce costs or increase pricing immediately")
    elif gross_margin < _HEALTH_THRESHOLDS["margin_warning"]:
        suggestions.append("Margins thin — review largest cost categories for optimization")

    if cost_ratio > 0.8:
        suggestions.append(f"Cost ratio {cost_ratio:.0%} — spending exceeds 80% of revenue")
    if largest != "unknown" and items:
        top_pct = items_sorted[0][1] / costs if costs > 0 else 0
        if top_pct > 0.5:
            suggestions.append(f"'{largest}' is {top_pct:.0%} of total costs — concentration risk")

    if not suggestions:
        suggestions.append("Margins healthy — maintain current cost structure")

    return {
        "gross_margin": round(gross_margin, 4),
        "net_margin_estimate": round(net_margin_estimate, 4),
        "cost_ratio": round(cost_ratio, 4),
        "largest_cost_category": largest,
        "optimization_suggestions": suggestions,
    }


def assess_financial_health(
    revenue: float,
    costs: float,
    runway_months: float = 0.0,
    ltv: float = 0.0,
    cac: float = 0.0,
    cost_items: Optional[List[Tuple[str, float]]] = None,
) -> FinancialSnapshot:
    """Comprehensive financial health assessment.

    Combines margin analysis, billing detection, and health scoring.

    Returns frozen FinancialSnapshot.
    """
    # Margin
    gross_margin = (revenue - costs) / revenue if revenue > 0 else 0.0

    # Cost breakdown
    items = cost_items or []
    breakdown = tuple((cat, amt) for cat, amt in items)

    # Billing model — infer from cost patterns
    has_recurring = any(
        cat in ("infrastructure", "tooling", "api", "personnel")
        for cat, _ in items
    )
    billing_model = "recurring" if has_recurring else "project-based" if items else "unknown"

    # Health scoring
    health_signals: List[str] = []
    score = 0

    if gross_margin >= _HEALTH_THRESHOLDS["margin_healthy"]:
        score += 3
    elif gross_margin >= _HEALTH_THRESHOLDS["margin_warning"]:
        score += 2
    elif gross_margin >= _HEALTH_THRESHOLDS["margin_critical"]:
        score += 1

    if runway_months >= 12:
        score += 3
    elif runway_months >= 6:
        score += 2
    elif runway_months >= 3:
        score += 1

    ltv_cac = ltv / cac if cac > 0 else 0.0
    if ltv_cac >= _HEALTH_THRESHOLDS["ltv_cac_healthy"]:
        score += 3
    elif ltv_cac >= _HEALTH_THRESHOLDS["ltv_cac_warning"]:
        score += 2
    elif ltv_cac >= _HEALTH_THRESHOLDS["ltv_cac_critical"]:
        score += 1

    if score >= 7:
        health_status = "healthy"
    elif score >= 4:
        health_status = "moderate"
    elif score >= 2:
        health_status = "warning"
    else:
        health_status = "critical"

    # Recommendations
    recs: List[str] = []
    if gross_margin < _HEALTH_THRESHOLDS["margin_warning"]:
        recs.append("Improve margins through pricing or cost reduction")
    if runway_months > 0 and runway_months < 6:
        recs.append("Extend runway — fundraise or cut burn")
    if cac > 0 and ltv_cac < _HEALTH_THRESHOLDS["ltv_cac_warning"]:
        recs.append("Improve LTV/CAC ratio — reduce acquisition cost or increase retention")
    if revenue <= 0:
        recs.append("Achieve first revenue milestone")
    if not recs:
        recs.append("Financial position strong — invest in growth")

    return FinancialSnapshot(
        total_revenue=revenue,
        total_costs=costs,
        gross_margin=round(gross_margin, 4),
        cost_breakdown=breakdown,
        billing_model=billing_model,
        health_status=health_status,
        recommendations=tuple(recs),
    )
