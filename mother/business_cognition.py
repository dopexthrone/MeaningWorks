"""
Business cognition — heuristic reasoning about viability, markets, competition, pricing.

LEAF module. Genome #111 (business-aware), #113 (market-positioning),
#115 (competitive-mapping), #117 (pricing-modeling).

All functions are pure — no external API calls, no LLM invocations.
Heuristic keyword analysis over structured inputs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Heuristic constants
# ---------------------------------------------------------------------------

_HIGH_RISK_KEYWORDS: frozenset = frozenset({
    "gambling", "crypto", "cannabis", "weapons", "adult",
    "pharmaceutical", "tobacco", "alcohol", "debt-collection",
})

_REGULATED_KEYWORDS: frozenset = frozenset({
    "healthcare", "insurance", "banking", "finance", "education",
    "legal", "government", "defense", "energy", "telecom",
})

_SAAS_KEYWORDS: frozenset = frozenset({
    "saas", "subscription", "recurring", "platform", "cloud",
    "api", "dashboard", "analytics", "crm", "erp",
})

_MARKETPLACE_KEYWORDS: frozenset = frozenset({
    "marketplace", "two-sided", "matching", "listing",
    "buyer", "seller", "commission", "escrow",
})

_ECOMMERCE_KEYWORDS: frozenset = frozenset({
    "ecommerce", "e-commerce", "shop", "store", "cart",
    "checkout", "inventory", "shipping", "retail",
})

_CONTENT_KEYWORDS: frozenset = frozenset({
    "content", "media", "publishing", "blog", "newsletter",
    "streaming", "video", "podcast", "creator",
})


# ---------------------------------------------------------------------------
# #111 — Business-aware: viability assessment
# ---------------------------------------------------------------------------

def assess_business_viability(
    runway_months: float,
    burn_rate: float,
    revenue_streams: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Assess business viability from runway, burn, and revenue diversification.

    Returns dict with keys: status, runway_assessment, burn_assessment,
    diversification, recommendation.
    """
    streams = revenue_streams or []

    # Runway classification
    if runway_months < 3:
        status = "CRITICAL"
        runway_assessment = "Less than 3 months runway — immediate action required"
    elif runway_months < 6:
        status = "WARNING"
        runway_assessment = "Under 6 months runway — fundraising or revenue acceleration needed"
    elif runway_months < 12:
        status = "MODERATE"
        runway_assessment = "6-12 months runway — plan next funding round or path to profitability"
    else:
        status = "HEALTHY"
        runway_assessment = "12+ months runway — focus on growth"

    # Burn rate classification
    if burn_rate <= 0:
        burn_assessment = "No burn — profitable or pre-launch"
    elif burn_rate < 5000:
        burn_assessment = "Low burn — lean operation"
    elif burn_rate < 25000:
        burn_assessment = "Moderate burn — typical early-stage"
    elif burn_rate < 100000:
        burn_assessment = "High burn — ensure growth justifies spend"
    else:
        burn_assessment = "Very high burn — requires strong revenue trajectory"

    # Revenue diversification
    stream_count = len(streams)
    if stream_count == 0:
        diversification = "No revenue streams — pre-revenue"
    elif stream_count == 1:
        diversification = "Single revenue stream — concentration risk"
    elif stream_count <= 3:
        diversification = "Moderate diversification — healthy for early stage"
    else:
        diversification = "Well diversified — multiple revenue sources"

    # Recommendation
    if status == "CRITICAL":
        recommendation = "Cut non-essential costs immediately. Pursue emergency funding or revenue."
    elif status == "WARNING" and stream_count == 0:
        recommendation = "Prioritize first revenue or fundraising over feature development."
    elif status == "WARNING":
        recommendation = "Accelerate revenue growth. Consider fundraising as backup."
    elif status == "MODERATE":
        recommendation = "Build sustainable growth engine. Plan next phase funding."
    else:
        recommendation = "Invest in growth and product expansion."

    return {
        "status": status,
        "runway_assessment": runway_assessment,
        "burn_assessment": burn_assessment,
        "diversification": diversification,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# #113 — Market-positioning: classify market type and position
# ---------------------------------------------------------------------------

def classify_market_position(
    market_description: str,
    known_competitors: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Classify market type, density, and recommend positioning.

    Returns dict with keys: market_type, density, positioning, reasoning.
    """
    competitors = known_competitors or []
    desc_lower = market_description.lower()
    desc_words = frozenset(desc_lower.split())

    # Market type from keyword overlap
    type_scores: List[Tuple[str, int]] = [
        ("saas", len(desc_words & _SAAS_KEYWORDS)),
        ("marketplace", len(desc_words & _MARKETPLACE_KEYWORDS)),
        ("ecommerce", len(desc_words & _ECOMMERCE_KEYWORDS)),
        ("content", len(desc_words & _CONTENT_KEYWORDS)),
    ]
    type_scores.sort(key=lambda x: x[1], reverse=True)
    market_type = type_scores[0][0] if type_scores[0][1] > 0 else "general"

    # Density from competitor count
    comp_count = len(competitors)
    if comp_count == 0:
        density = "blue-ocean"
        density_reasoning = "No known competitors — potential blue ocean opportunity"
    elif comp_count <= 3:
        density = "low"
        density_reasoning = f"{comp_count} competitors — room for differentiation"
    elif comp_count <= 8:
        density = "moderate"
        density_reasoning = f"{comp_count} competitors — established market, differentiation critical"
    else:
        density = "crowded"
        density_reasoning = f"{comp_count}+ competitors — highly competitive, niche or cost leadership needed"

    # Positioning recommendation
    if density == "blue-ocean":
        positioning = "first-mover"
        pos_reasoning = "Establish category definition and market education"
    elif density == "low":
        positioning = "differentiator"
        pos_reasoning = "Build unique value proposition while market is still forming"
    elif density == "moderate":
        positioning = "differentiator"
        pos_reasoning = "Focus on underserved segments or superior UX"
    else:
        positioning = "niche"
        pos_reasoning = "Target specific underserved vertical or use-case"

    # Check for regulated markets
    if desc_words & _REGULATED_KEYWORDS:
        pos_reasoning += ". Note: regulated market — compliance is a barrier-to-entry advantage"
    if desc_words & _HIGH_RISK_KEYWORDS:
        pos_reasoning += ". Warning: high-risk vertical — payment processing and legal exposure concerns"

    return {
        "market_type": market_type,
        "density": density,
        "density_reasoning": density_reasoning,
        "positioning": positioning,
        "reasoning": pos_reasoning,
    }


# ---------------------------------------------------------------------------
# #115 — Competitive-mapping: structured competitor analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompetitiveMap:
    """Structured competitive landscape analysis."""
    market_type: str
    competitors: Tuple[str, ...]
    differentiators: Tuple[str, ...]
    gaps: Tuple[str, ...]
    threat_level: str  # "low", "moderate", "high", "critical"
    summary: str


def build_competitive_map(
    market_description: str,
    competitors: Optional[List[str]] = None,
    differentiators: Optional[List[str]] = None,
) -> CompetitiveMap:
    """Build a competitive landscape map from market info.

    Analyzes competitor density, identifies gaps, classifies threat level.
    """
    comps = competitors or []
    diffs = differentiators or []
    desc_lower = market_description.lower()
    desc_words = frozenset(desc_lower.split())

    # Market type (reuse logic)
    type_scores = [
        ("saas", len(desc_words & _SAAS_KEYWORDS)),
        ("marketplace", len(desc_words & _MARKETPLACE_KEYWORDS)),
        ("ecommerce", len(desc_words & _ECOMMERCE_KEYWORDS)),
        ("content", len(desc_words & _CONTENT_KEYWORDS)),
    ]
    type_scores.sort(key=lambda x: x[1], reverse=True)
    market_type = type_scores[0][0] if type_scores[0][1] > 0 else "general"

    # Threat level from competitor count + differentiator coverage
    comp_count = len(comps)
    diff_count = len(diffs)
    if comp_count == 0:
        threat_level = "low"
    elif comp_count <= 3 and diff_count >= 2:
        threat_level = "low"
    elif comp_count <= 3:
        threat_level = "moderate"
    elif comp_count <= 8 and diff_count >= 3:
        threat_level = "moderate"
    elif comp_count <= 8:
        threat_level = "high"
    else:
        threat_level = "critical" if diff_count < 3 else "high"

    # Gap detection from market keywords not covered by differentiators
    diff_words = frozenset(" ".join(d.lower() for d in diffs).split())
    potential_gaps = []
    gap_areas = {
        "pricing": {"free", "affordable", "pricing", "cost"},
        "integration": {"integration", "api", "plugin", "connect"},
        "support": {"support", "onboarding", "training", "help"},
        "mobile": {"mobile", "app", "ios", "android"},
        "analytics": {"analytics", "reporting", "metrics", "data"},
    }
    for area, keywords in gap_areas.items():
        if desc_words & keywords and not diff_words & keywords:
            potential_gaps.append(area)

    # Summary
    if comp_count == 0:
        summary = f"Blue ocean {market_type} market. First-mover advantage available."
    elif threat_level in ("low", "moderate"):
        summary = (
            f"{market_type.title()} market with {comp_count} competitor(s). "
            f"Threat level {threat_level}. "
            f"{diff_count} differentiator(s) identified."
        )
    else:
        summary = (
            f"Competitive {market_type} market with {comp_count} players. "
            f"Threat level {threat_level}. "
            f"Strong differentiation or niche focus required."
        )

    if potential_gaps:
        summary += f" Gaps: {', '.join(potential_gaps)}."

    return CompetitiveMap(
        market_type=market_type,
        competitors=tuple(comps),
        differentiators=tuple(diffs),
        gaps=tuple(potential_gaps),
        threat_level=threat_level,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# #117 — Pricing-modeling: compute pricing strategy
# ---------------------------------------------------------------------------

def compute_pricing_strategy(
    cost_per_unit: float,
    market_position: str = "differentiator",
    competitor_pricing: Optional[List[float]] = None,
) -> Dict[str, object]:
    """Compute a pricing strategy from cost basis and market position.

    Args:
        cost_per_unit: Direct cost to deliver one unit of value.
        market_position: One of "cost-leader", "differentiator", "niche", "disruptor".
        competitor_pricing: Known competitor price points.

    Returns dict with: recommended_price, margin, strategy_name, reasoning, price_range.
    """
    comp_prices = competitor_pricing or []

    if not comp_prices:
        # No competitor data — cost-plus pricing
        recommended = cost_per_unit * 2.5
        strategy_name = "cost-plus"
        reasoning = "No competitor data available — using 2.5x cost-plus markup"
    else:
        avg_price = sum(comp_prices) / len(comp_prices)

        position_multipliers = {
            "cost-leader": 0.8,
            "differentiator": 1.2,
            "niche": 1.5,
            "disruptor": 0.5,
        }
        multiplier = position_multipliers.get(market_position, 1.0)
        recommended = avg_price * multiplier

        strategy_names = {
            "cost-leader": "competitive-undercut",
            "differentiator": "value-premium",
            "niche": "premium-niche",
            "disruptor": "penetration",
        }
        strategy_name = strategy_names.get(market_position, "market-aligned")

        reasoning = (
            f"Average competitor price ${avg_price:.2f}. "
            f"{market_position} position → {multiplier}x multiplier."
        )

    # Enforce 10% minimum margin floor
    min_price = cost_per_unit * 1.1
    if recommended < min_price:
        recommended = min_price
        reasoning += f" Price floored at 10% margin (${min_price:.2f})."

    margin = (recommended - cost_per_unit) / recommended if recommended > 0 else 0.0

    # Price range (±20%)
    price_range = (round(recommended * 0.8, 2), round(recommended * 1.2, 2))

    return {
        "recommended_price": round(recommended, 2),
        "margin": round(margin, 4),
        "strategy_name": strategy_name,
        "reasoning": reasoning,
        "price_range": price_range,
    }
