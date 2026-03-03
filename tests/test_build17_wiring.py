"""
Build 17 wiring tests — business cognition, compliance, brand, multi-instance.

58 tests covering 10 genome capabilities:
  #111 business-aware, #113 market-positioning, #115 competitive-mapping,
  #117 pricing-modeling, #120 legal-awareness, #121 compliance-checking,
  #125 data-sovereignty, #144 brand-consistent, #129 negotiation-preparing,
  #81 multi-instance.
"""

import pytest

# ---------------------------------------------------------------------------
# Group A: Business Cognition (#111, #113, #115, #117)
# ---------------------------------------------------------------------------

class TestBusinessAware:
    """Genome #111 — business-aware: viability assessment."""

    def test_critical_runway(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(2.0, 10000)
        assert result["status"] == "CRITICAL"
        assert "immediate" in result["recommendation"].lower()

    def test_warning_runway(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(5.0, 10000)
        assert result["status"] == "WARNING"

    def test_moderate_runway(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(9.0, 10000)
        assert result["status"] == "MODERATE"

    def test_healthy_runway(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(18.0, 5000)
        assert result["status"] == "HEALTHY"
        assert "growth" in result["recommendation"].lower()

    def test_no_revenue_streams(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(4.0, 8000, [])
        assert "pre-revenue" in result["diversification"].lower()

    def test_high_burn_rate(self):
        from mother.business_cognition import assess_business_viability
        result = assess_business_viability(12.0, 150000, ["subscriptions", "consulting"])
        assert "very high" in result["burn_assessment"].lower()
        assert result["status"] == "HEALTHY"


class TestMarketPositioning:
    """Genome #113 — market-positioning: classify market type and position."""

    def test_saas_market(self):
        from mother.business_cognition import classify_market_position
        result = classify_market_position("saas subscription cloud dashboard platform")
        assert result["market_type"] == "saas"

    def test_blue_ocean(self):
        from mother.business_cognition import classify_market_position
        result = classify_market_position("novel approach to widget optimization", [])
        assert result["density"] == "blue-ocean"
        assert result["positioning"] == "first-mover"

    def test_crowded_market(self):
        from mother.business_cognition import classify_market_position
        comps = [f"competitor_{i}" for i in range(12)]
        result = classify_market_position("marketplace listing buyer seller", comps)
        assert result["density"] == "crowded"

    def test_moderate_market(self):
        from mother.business_cognition import classify_market_position
        result = classify_market_position("ecommerce store checkout", ["shopify", "woocommerce", "bigcommerce", "magento"])
        assert result["density"] == "low" or result["density"] == "moderate"

    def test_empty_description(self):
        from mother.business_cognition import classify_market_position
        result = classify_market_position("")
        assert result["market_type"] == "general"


class TestCompetitiveMapping:
    """Genome #115 — competitive-mapping: structured competitor analysis."""

    def test_frozen_dataclass(self):
        from mother.business_cognition import CompetitiveMap
        m = CompetitiveMap("saas", ("A",), ("fast",), ("mobile",), "low", "summary")
        with pytest.raises(AttributeError):
            m.market_type = "other"

    def test_low_threat(self):
        from mother.business_cognition import build_competitive_map
        result = build_competitive_map("saas platform", ["comp1"], ["speed", "ux"])
        assert result.threat_level == "low"

    def test_high_threat(self):
        from mother.business_cognition import build_competitive_map
        comps = [f"c{i}" for i in range(7)]
        result = build_competitive_map("saas platform analytics", comps, ["speed"])
        assert result.threat_level in ("high", "critical")

    def test_no_differentiators(self):
        from mother.business_cognition import build_competitive_map
        comps = [f"c{i}" for i in range(5)]
        result = build_competitive_map("marketplace listing", comps, [])
        assert result.threat_level in ("high", "critical")

    def test_summary_nonempty(self):
        from mother.business_cognition import build_competitive_map
        result = build_competitive_map("ecommerce store", ["shopify"])
        assert len(result.summary) > 10

    def test_with_differentiators_and_gaps(self):
        from mother.business_cognition import build_competitive_map
        result = build_competitive_map(
            "saas analytics platform mobile",
            ["comp1", "comp2"],
            ["fast api integration"],
        )
        assert isinstance(result.gaps, tuple)
        assert isinstance(result.differentiators, tuple)


class TestPricingModeling:
    """Genome #117 — pricing-modeling: compute pricing strategy."""

    def test_cost_plus_no_competitors(self):
        from mother.business_cognition import compute_pricing_strategy
        result = compute_pricing_strategy(10.0)
        assert result["strategy_name"] == "cost-plus"
        assert result["recommended_price"] == 25.0  # 2.5x

    def test_cost_leader_undercut(self):
        from mother.business_cognition import compute_pricing_strategy
        result = compute_pricing_strategy(5.0, "cost-leader", [20.0, 30.0])
        # avg=25, cost-leader=0.8x → 20.0
        assert result["recommended_price"] == 20.0
        assert result["strategy_name"] == "competitive-undercut"

    def test_differentiator_premium(self):
        from mother.business_cognition import compute_pricing_strategy
        result = compute_pricing_strategy(5.0, "differentiator", [20.0, 30.0])
        # avg=25, 1.2x → 30.0
        assert result["recommended_price"] == 30.0

    def test_margin_floor_enforced(self):
        from mother.business_cognition import compute_pricing_strategy
        # cost=20, disruptor at avg=25 → 0.5x=12.5, but floor=22.0 (10% margin)
        result = compute_pricing_strategy(20.0, "disruptor", [25.0])
        assert result["recommended_price"] >= 20.0 * 1.1

    def test_niche_premium(self):
        from mother.business_cognition import compute_pricing_strategy
        result = compute_pricing_strategy(5.0, "niche", [20.0, 30.0])
        # avg=25, 1.5x → 37.5
        assert result["recommended_price"] == 37.5

    def test_all_keys_present(self):
        from mother.business_cognition import compute_pricing_strategy
        result = compute_pricing_strategy(10.0, "differentiator", [15.0])
        assert "recommended_price" in result
        assert "margin" in result
        assert "strategy_name" in result
        assert "reasoning" in result
        assert "price_range" in result


# ---------------------------------------------------------------------------
# Group B: Compliance Reasoning (#120, #121, #125)
# ---------------------------------------------------------------------------

class TestLegalAwareness:
    """Genome #120 — legal-awareness: regulatory exposure assessment."""

    def test_healthcare_us(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure(["health", "patient", "diagnosis"], "us")
        assert any("HIPAA" in r for r in result)
        assert any("SOC 2" in r or "CCPA" in r for r in result)

    def test_finance_eu(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure(["payment", "banking", "kyc"], "eu")
        assert any("PCI" in r for r in result)
        assert any("GDPR" in r for r in result)

    def test_privacy_global(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure(["personal", "tracking", "consent"], "global")
        assert any("Privacy" in r or "privacy" in r for r in result)

    def test_no_keywords(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure([], "us")
        assert len(result) == 1
        assert "unable" in result[0].lower()

    def test_unknown_geography(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure(["health"], "narnia")
        assert any("global baseline" in r.lower() for r in result)

    def test_multi_domain(self):
        from mother.compliance_reasoning import assess_regulatory_exposure
        result = assess_regulatory_exposure(["health", "payment", "ai", "personal"], "eu")
        # Should find healthcare, finance, privacy, and AI regulations
        domains_found = set()
        for r in result:
            if "[healthcare]" in r:
                domains_found.add("healthcare")
            if "[finance]" in r:
                domains_found.add("finance")
            if "[ai]" in r:
                domains_found.add("ai")
            if "[privacy]" in r:
                domains_found.add("privacy")
        assert len(domains_found) >= 3


class TestComplianceChecking:
    """Genome #121 — compliance-checking: blueprint component compliance."""

    def test_payment_pci(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance(["PaymentService", "checkout-handler"])
        assert any("PCI" in r for r in result)

    def test_auth_owasp(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance(["AuthenticationModule", "login-handler"])
        assert any("OWASP" in r or "password" in r.lower() for r in result)

    def test_user_data_privacy(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance(["UserProfileService", "account-settings"])
        assert any("privacy" in r.lower() or "consent" in r.lower() or "deletion" in r.lower() for r in result)

    def test_empty_components(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance([])
        assert result == []

    def test_healthcare_domain_storage(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance(
            ["PatientDatabase", "clinical-store"],
            domain="healthcare",
        )
        assert any("HIPAA" in r for r in result)
        assert any("encryption" in r.lower() or "Encryption" in r for r in result)

    def test_external_api(self):
        from mother.compliance_reasoning import check_blueprint_compliance
        result = check_blueprint_compliance(["ThirdPartyApiClient", "webhook-handler"])
        assert any("API" in r or "api" in r for r in result)


class TestDataSovereignty:
    """Genome #125 — data-sovereignty-respecting: data locality validation."""

    def test_eu_in_eu_compliant(self):
        from mother.compliance_reasoning import validate_data_locality
        ok, violations = validate_data_locality(["eu-west-1", "eu-central-1"], "eu")
        assert ok is True
        assert violations == []

    def test_eu_in_us_violation(self):
        from mother.compliance_reasoning import validate_data_locality
        ok, violations = validate_data_locality(["us-east-1"], "eu")
        assert ok is False
        assert len(violations) > 0
        assert "sovereignty" in violations[0].lower()

    def test_us_anywhere_ok(self):
        from mother.compliance_reasoning import validate_data_locality
        ok, violations = validate_data_locality(["eu-west-1", "ap-southeast-1"], "us")
        assert ok is True

    def test_china_violation(self):
        from mother.compliance_reasoning import validate_data_locality
        ok, violations = validate_data_locality(["us-east-1"], "china")
        assert ok is False

    def test_empty_locations_compliant(self):
        from mother.compliance_reasoning import validate_data_locality
        ok, violations = validate_data_locality([], "eu")
        assert ok is True


# ---------------------------------------------------------------------------
# Group C: Brand Identity (#144, #129)
# ---------------------------------------------------------------------------

class TestBrandConsistent:
    """Genome #144 — brand-consistent: brand signal extraction."""

    def test_frozen_dataclass(self):
        from mother.brand_identity import BrandProfile
        p = BrandProfile(("casual",), ("trust",), (), (), 0.5, 1)
        with pytest.raises(AttributeError):
            p.formality = 0.9

    def test_casual_tone(self):
        from mother.brand_identity import extract_brand_signals
        messages = [
            "Hey that's pretty cool stuff!",
            "Yeah gonna wanna try this thing out",
            "Super awesome, kinda like it",
        ]
        profile = extract_brand_signals(messages)
        assert "casual" in profile.tone_keywords
        assert profile.formality < 0.5

    def test_technical_tone(self):
        from mother.brand_identity import extract_brand_signals
        messages = [
            "The implementation uses a modular architecture with clean interfaces.",
            "Component pipeline handles protocol-level module integration.",
            "Infrastructure layer provides algorithm optimization.",
        ]
        profile = extract_brand_signals(messages)
        assert "technical" in profile.tone_keywords

    def test_values_detected(self):
        from mother.brand_identity import extract_brand_signals
        messages = [
            "We believe in trust and transparency above all.",
            "Quality and innovation drive our trust in the process.",
        ]
        profile = extract_brand_signals(messages)
        assert "trust" in profile.values

    def test_empty_messages(self):
        from mother.brand_identity import extract_brand_signals
        profile = extract_brand_signals([])
        assert profile.message_count == 0
        assert profile.formality == 0.5
        assert profile.tone_keywords == ()

    def test_synthesize_prompt(self):
        from mother.brand_identity import BrandProfile, synthesize_brand_prompt
        profile = BrandProfile(
            tone_keywords=("professional", "warm"),
            values=("trust", "quality"),
            vocabulary=("architecture", "pipeline"),
            avoids=("jargon",),
            formality=0.8,
            message_count=10,
        )
        prompt = synthesize_brand_prompt(profile)
        assert "Brand Voice" in prompt
        assert "professional" in prompt
        assert "trust" in prompt
        assert "High" in prompt  # high formality


class TestNegotiationPreparing:
    """Genome #129 — negotiation-preparing: structured negotiation briefs."""

    def test_frozen_dataclass(self):
        from mother.brand_identity import NegotiationBrief
        b = NegotiationBrief("g", "c", ("i",), ("ci",), "b", "o", ("s",), ("r",))
        with pytest.raises(AttributeError):
            b.goal = "other"

    def test_goal_in_opening(self):
        from mother.brand_identity import generate_negotiation_brief
        brief = generate_negotiation_brief("Reduce licensing cost by 20%")
        assert "Reduce" in brief.opening_position

    def test_parties_in_counterparty(self):
        from mother.brand_identity import generate_negotiation_brief
        brief = generate_negotiation_brief(
            "Negotiate SaaS contract renewal",
            parties=["Acme Corp"],
        )
        assert any("Acme" in ci for ci in brief.counterparty_interests)

    def test_concession_nonempty(self):
        from mother.brand_identity import generate_negotiation_brief
        brief = generate_negotiation_brief("Get better terms")
        assert len(brief.concession_strategy) > 0

    def test_risk_factors_from_context(self):
        from mother.brand_identity import generate_negotiation_brief
        brief = generate_negotiation_brief(
            "Extend partnership",
            context="There is a deadline pressure and penalty clause for late delivery",
        )
        assert any(r in ("deadline", "penalty") for r in brief.risk_factors)


# ---------------------------------------------------------------------------
# Group D: Multi-instance (#81)
# ---------------------------------------------------------------------------

class TestMultiInstance:
    """Genome #81 — multi-instance: capability profiling and peer selection."""

    def test_frozen_dataclass(self):
        from mother.delegation import CapabilityProfile
        p = CapabilityProfile("id1", ("compile",), ("software",), 0.5, 0.8, "sonnet")
        with pytest.raises(AttributeError):
            p.current_load = 0.9

    def test_clamp_values(self):
        from mother.delegation import build_capability_profile
        p = build_capability_profile("x", current_load=1.5, trust_score=-0.3)
        assert p.current_load == 1.0
        assert p.trust_score == 0.0

    def test_empty_profiles(self):
        from mother.delegation import select_best_peer
        assert select_best_peer([]) is None

    def test_trust_ranking(self):
        from mother.delegation import build_capability_profile, select_best_peer
        p1 = build_capability_profile("low", capabilities=["compile"], trust_score=0.3, model_tier="sonnet")
        p2 = build_capability_profile("high", capabilities=["compile"], trust_score=0.9, model_tier="sonnet")
        result = select_best_peer([p1, p2], "compile")
        assert result == "high"

    def test_domain_bonus(self):
        from mother.delegation import build_capability_profile, select_best_peer
        p1 = build_capability_profile("general", capabilities=["compile"], domain_strengths=["process"], trust_score=0.8, model_tier="sonnet")
        p2 = build_capability_profile("specialist", capabilities=["compile"], domain_strengths=["software"], trust_score=0.7, model_tier="sonnet")
        result = select_best_peer([p1, p2], "compile", required_domain="software")
        assert result == "specialist"  # 0.7 * 0.8 * 1.5 = 0.84 > 0.8 * 0.8 * 1.0 = 0.64

    def test_load_factor(self):
        from mother.delegation import build_capability_profile, select_best_peer
        p1 = build_capability_profile("idle", capabilities=["compile"], trust_score=0.7, current_load=0.1, model_tier="sonnet")
        p2 = build_capability_profile("busy", capabilities=["compile"], trust_score=0.9, current_load=0.9, model_tier="sonnet")
        result = select_best_peer([p1, p2], "compile")
        # idle: 0.7 * 0.8 * 0.9 = 0.504,  busy: 0.9 * 0.8 * 0.1 = 0.072
        assert result == "idle"

    def test_no_capability_match(self):
        from mother.delegation import build_capability_profile, select_best_peer
        p1 = build_capability_profile("voice_only", capabilities=["voice"], trust_score=0.9)
        result = select_best_peer([p1], "compile")
        assert result is None
