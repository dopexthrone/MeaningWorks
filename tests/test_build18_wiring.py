"""
Build 18 wiring tests — 64 tests across 8 capabilities.

Covers:
- #116 financial-operating (existing code, new tests)
- #139 conflict-mediating (existing code, new tests)
- #35 market-sensing (new module)
- #38 stakeholder-modeling (new module)
- #148 translation-native (diagrams.py extension)
- #162 instance-specializing (delegation.py extension)
- #168 access-control-enforcing (auth.py extension)
- #185 paradigm-shifting (new module)
"""

import os
import tempfile

import pytest


# ============================================================================
# #116 Financial Operating — mother/financial_ops.py
# ============================================================================

class TestFinancialOperating:
    """Genome #116: financial-operating."""

    def test_snapshot_frozen(self):
        from mother.financial_ops import FinancialSnapshot
        snap = FinancialSnapshot(
            total_revenue=100.0, total_costs=50.0, gross_margin=0.5,
            cost_breakdown=(("infra", 30.0),), billing_model="recurring",
            health_status="healthy", recommendations=("Keep going",),
        )
        with pytest.raises(AttributeError):
            snap.total_revenue = 200.0

    def test_estimate_project_cost_basic(self):
        from mother.financial_ops import estimate_project_cost
        result = estimate_project_cost("Build a web app", team_size=1, duration_months=1)
        assert "total_estimate" in result
        assert result["total_estimate"] > 0
        assert "breakdown" in result

    def test_estimate_project_cost_heavy_infra(self):
        from mother.financial_ops import estimate_project_cost
        result = estimate_project_cost(
            "hosting server cloud aws database storage compute cdn",
            team_size=2, duration_months=3,
        )
        assert result["infrastructure_estimate"] > 0
        # Heavy infra should be 25% of personnel
        assert result["infrastructure_estimate"] == result["personnel_cost"] * 0.25

    def test_detect_billing_monthly(self):
        from mother.financial_ops import detect_billing_cycle
        result = detect_billing_cycle("We charge monthly subscriptions per month")
        assert result["cycle"] == "monthly"
        assert result["confidence"] == "high"

    def test_detect_billing_unknown(self):
        from mother.financial_ops import detect_billing_cycle
        result = detect_billing_cycle("This is a random description")
        assert result["cycle"] == "unknown"
        assert result["confidence"] == "low"

    def test_margins_zero_revenue(self):
        from mother.financial_ops import compute_margins
        result = compute_margins(0.0, 100.0)
        assert result["gross_margin"] == 0.0
        assert "Generate revenue" in result["optimization_suggestions"][0]

    def test_margins_healthy(self):
        from mother.financial_ops import compute_margins
        result = compute_margins(1000.0, 500.0)
        assert result["gross_margin"] == 0.5
        assert result["cost_ratio"] == 0.5

    def test_health_critical(self):
        from mother.financial_ops import assess_financial_health
        snap = assess_financial_health(revenue=0, costs=1000, runway_months=1)
        assert snap.health_status == "critical"
        assert snap.gross_margin == 0.0


# ============================================================================
# #139 Conflict Mediating — mother/conflict_mediation.py
# ============================================================================

class TestConflictMediating:
    """Genome #139: conflict-mediating."""

    def test_analysis_frozen(self):
        from mother.conflict_mediation import ConflictAnalysis
        analysis = ConflictAnalysis(
            conflict_types=("priority",), severity="high", severity_score=4,
            parties_involved=("team-a",), resolution_strategies=(),
            recommended_strategy="structured-triage", summary="test",
        )
        with pytest.raises(AttributeError):
            analysis.severity = "low"

    def test_detect_priority_conflict(self):
        from mother.conflict_mediation import detect_conflicts
        result = detect_conflicts(["This is urgent and critical", "Non-negotiable deadline"])
        assert "priority" in result

    def test_detect_resource_conflict(self):
        from mother.conflict_mediation import detect_conflicts
        result = detect_conflicts(["Budget is stretched", "We need more headcount"])
        assert "resource" in result

    def test_detect_no_conflict(self):
        from mother.conflict_mediation import detect_conflicts
        result = detect_conflicts(["Everything is going well", "Good progress"])
        assert result == []

    def test_classify_primary(self):
        from mother.conflict_mediation import classify_conflict_type
        result = classify_conflict_type(["urgent critical blocker asap"])
        assert result["primary_type"] == "priority"
        assert result["signal_count"] >= 3

    def test_classify_empty(self):
        from mother.conflict_mediation import classify_conflict_type
        result = classify_conflict_type([])
        assert result["primary_type"] == "none"

    def test_severity_scoring(self):
        from mother.conflict_mediation import generate_resolution_strategy
        analysis = generate_resolution_strategy(
            ["urgent critical blocker", "budget stretched overloaded"],
            parties=["team-a", "team-b"],
        )
        assert analysis.severity_score >= 5
        assert analysis.severity in ("high", "critical")
        assert len(analysis.resolution_strategies) >= 1

    def test_no_conflict_analysis(self):
        from mother.conflict_mediation import generate_resolution_strategy
        analysis = generate_resolution_strategy(["All good here"])
        assert analysis.severity == "low"
        assert analysis.severity_score == 0
        assert analysis.recommended_strategy == "none"


# ============================================================================
# #35 Market Sensing — mother/market_sensing.py
# ============================================================================

class TestMarketSensing:
    """Genome #35: market-sensing."""

    def test_signal_frozen(self):
        from mother.market_sensing import MarketSignal
        signal = MarketSignal(
            trend_direction="up", market_phase="growth", demand_strength="strong",
            risk_level="low", signal_count=3, signals_detected=("growing",),
            timing_recommendation="Enter now", summary="test",
        )
        with pytest.raises(AttributeError):
            signal.trend_direction = "down"

    def test_detect_trend_signals(self):
        from mother.market_sensing import detect_market_signals
        signals = detect_market_signals("The market is growing and emerging fast")
        assert "growing" in signals
        assert "emerging" in signals

    def test_detect_demand_signals(self):
        from mother.market_sensing import detect_market_signals
        signals = detect_market_signals("There is high demand and a big gap in the market")
        assert "demand" in signals
        assert "gap" in signals

    def test_growing_market(self):
        from mother.market_sensing import assess_market_timing
        result = assess_market_timing("Growing expanding market with high demand need gap")
        assert result.trend_direction == "up"
        assert result.market_phase == "growth"
        assert result.demand_strength in ("strong", "moderate")
        assert result.signal_count >= 3

    def test_declining_market(self):
        from mother.market_sensing import assess_market_timing
        result = assess_market_timing("Declining shrinking obsolete market contracting")
        assert result.trend_direction == "down"
        assert result.market_phase == "decline"
        assert "avoid" in result.timing_recommendation.lower() or "declin" in result.timing_recommendation.lower()

    def test_no_signals(self):
        from mother.market_sensing import assess_market_timing
        result = assess_market_timing("Build a calculator app")
        assert result.signal_count == 0
        assert result.demand_strength == "weak"

    def test_high_risk(self):
        from mother.market_sensing import assess_market_timing
        result = assess_market_timing("risk volatile uncertain recession downturn threat")
        assert result.risk_level == "high"

    def test_timing_early(self):
        from mother.market_sensing import assess_market_timing
        result = assess_market_timing("emerging novel pioneering new first")
        assert result.market_phase == "early"
        assert "first-mover" in result.timing_recommendation.lower() or "enter" in result.timing_recommendation.lower()


# ============================================================================
# #38 Stakeholder Modeling — mother/stakeholder_modeling.py
# ============================================================================

class TestStakeholderModeling:
    """Genome #38: stakeholder-modeling."""

    def test_map_frozen(self):
        from mother.stakeholder_modeling import StakeholderMap
        smap = StakeholderMap(
            stakeholders=(("end-user", ("usability",)),),
            primary_stakeholder="end-user",
            concern_overlap=(), conflict_zones=(), summary="test",
        )
        with pytest.raises(AttributeError):
            smap.primary_stakeholder = "developer"

    def test_identify_user_stakeholder(self):
        from mother.stakeholder_modeling import identify_stakeholders
        roles = identify_stakeholders("The user customer needs a dashboard")
        assert "end-user" in roles

    def test_identify_developer_stakeholder(self):
        from mother.stakeholder_modeling import identify_stakeholders
        roles = identify_stakeholders("The developer engineer needs an API")
        assert "developer" in roles

    def test_identify_multiple(self):
        from mother.stakeholder_modeling import identify_stakeholders
        roles = identify_stakeholders("The user customer and developer engineer both need security compliance")
        assert "end-user" in roles
        assert "developer" in roles

    def test_empty_description(self):
        from mother.stakeholder_modeling import build_stakeholder_map
        result = build_stakeholder_map("")
        assert result.primary_stakeholder == "unknown"
        assert result.stakeholders == ()

    def test_concern_overlap(self):
        from mother.stakeholder_modeling import build_stakeholder_map
        result = build_stakeholder_map(
            "The user customer and developer engineer need security performance"
        )
        # Both roles should share security and/or performance concerns
        assert len(result.concern_overlap) >= 1

    def test_primary_stakeholder(self):
        from mother.stakeholder_modeling import build_stakeholder_map
        result = build_stakeholder_map(
            "The user customer consumer visitor subscriber client buyer needs an app"
        )
        assert result.primary_stakeholder == "end-user"


# ============================================================================
# #148 Translation Native — mother/diagrams.py extension
# ============================================================================

class TestTranslationNative:
    """Genome #148: translation-native."""

    def _sample_blueprint(self):
        return {
            "components": [
                {"name": "Frontend", "type": "interface"},
                {"name": "Backend", "type": "service"},
                {"name": "Database", "type": "entity"},
            ],
            "relationships": [
                {"from": "Frontend", "to": "Backend", "type": "sends"},
                {"from": "Backend", "to": "Database", "type": "queries"},
            ],
        }

    def test_sequence_basic(self):
        from mother.diagrams import blueprint_to_sequence_diagram
        result = blueprint_to_sequence_diagram(self._sample_blueprint(), title="Test")
        assert "sequenceDiagram" in result
        assert "Frontend" in result
        assert "Backend" in result

    def test_sequence_empty(self):
        from mother.diagrams import blueprint_to_sequence_diagram
        result = blueprint_to_sequence_diagram({"relationships": []})
        assert result == ""

    def test_wireframe_basic(self):
        from mother.diagrams import blueprint_to_wireframe
        result = blueprint_to_wireframe(self._sample_blueprint(), title="Test")
        assert "Frontend" in result
        assert "+" in result  # box border
        assert "|" in result  # box sides

    def test_wireframe_empty(self):
        from mother.diagrams import blueprint_to_wireframe
        result = blueprint_to_wireframe({"components": []})
        assert result == ""

    def test_translate_flowchart(self):
        from mother.diagrams import translate_blueprint
        result = translate_blueprint(self._sample_blueprint(), target_format="flowchart")
        assert "flowchart" in result

    def test_translate_sequence(self):
        from mother.diagrams import translate_blueprint
        result = translate_blueprint(self._sample_blueprint(), target_format="sequence")
        assert "sequenceDiagram" in result

    def test_translate_wireframe(self):
        from mother.diagrams import translate_blueprint
        result = translate_blueprint(self._sample_blueprint(), target_format="wireframe")
        assert "Frontend" in result
        assert "+" in result

    def test_translate_unknown_format(self):
        from mother.diagrams import translate_blueprint
        result = translate_blueprint(self._sample_blueprint(), target_format="unknown_format")
        # Should fallback to tree
        assert "Frontend" in result


# ============================================================================
# #162 Instance Specializing — mother/delegation.py extension
# ============================================================================

class TestInstanceSpecializing:
    """Genome #162: instance-specializing."""

    def _make_profile(self, **kwargs):
        from mother.delegation import build_capability_profile
        defaults = dict(
            instance_id="inst-1",
            capabilities=["compile", "build"],
            domain_strengths=["software"],
            current_load=0.2,
            trust_score=0.8,
            model_tier="opus",
        )
        defaults.update(kwargs)
        return build_capability_profile(**defaults)

    def test_specialization_profile_frozen(self):
        from mother.delegation import SpecializationProfile
        sp = SpecializationProfile(
            instance_id="x", specialization="generalist", domain_specialty="general",
            success_rate=0.5, avg_latency_seconds=10.0, tasks_completed=5,
            recommended_tasks=("compile",),
        )
        with pytest.raises(AttributeError):
            sp.specialization = "specialist"

    def test_infer_specialist(self):
        from mother.delegation import infer_specialization
        profile = self._make_profile()
        history = [{"success": True, "duration": 5.0} for _ in range(12)]
        result = infer_specialization(profile, task_history=history)
        assert result.specialization == "specialist"
        assert result.success_rate >= 0.75
        assert result.tasks_completed == 12

    def test_infer_generalist(self):
        from mother.delegation import infer_specialization
        profile = self._make_profile()
        history = [{"success": True, "duration": 5.0} for _ in range(3)]
        result = infer_specialization(profile, task_history=history)
        assert result.specialization == "generalist"

    def test_domain_strength(self):
        from mother.delegation import infer_specialization
        profile = self._make_profile(domain_strengths=["api"])
        result = infer_specialization(profile)
        assert result.domain_specialty == "api"

    def test_with_task_history(self):
        from mother.delegation import infer_specialization
        profile = self._make_profile(model_tier="haiku")
        history = [
            {"success": i % 3 != 0, "duration": 10.0 + i}
            for i in range(15)
        ]
        result = infer_specialization(profile, task_history=history)
        assert result.tasks_completed == 15
        assert result.avg_latency_seconds > 0

    def test_recommend_single(self):
        from mother.delegation import infer_specialization, recommend_instance_roles
        profile = self._make_profile()
        sp = infer_specialization(profile, task_history=[
            {"success": True, "duration": 5.0} for _ in range(12)
        ])
        assignments = recommend_instance_roles([sp], ["compile"])
        assert len(assignments) == 1
        assert assignments[0] == ("inst-1", "compile")

    def test_recommend_multiple(self):
        from mother.delegation import infer_specialization, recommend_instance_roles, build_capability_profile
        p1 = build_capability_profile("inst-1", ["compile"], ["software"], model_tier="opus")
        p2 = build_capability_profile("inst-2", ["build"], ["api"], model_tier="haiku")
        sp1 = infer_specialization(p1)
        sp2 = infer_specialization(p2)
        assignments = recommend_instance_roles([sp1, sp2], ["compile", "build"])
        assert len(assignments) == 2

    def test_recommend_empty(self):
        from mother.delegation import recommend_instance_roles
        assert recommend_instance_roles([], ["compile"]) == []


# ============================================================================
# #168 Access Control — motherlabs_platform/auth.py extension
# ============================================================================

class TestAccessControl:
    """Genome #168: access-control-enforcing."""

    @pytest.fixture
    def key_store(self, tmp_path):
        from motherlabs_platform.auth import KeyStore
        db_path = str(tmp_path / "test_auth.db")
        return KeyStore(db_path=db_path)

    def test_acl_result_frozen(self):
        from motherlabs_platform.auth import AccessControlResult
        acr = AccessControlResult(
            allowed=True, key_id="k1", action="read",
            roles=("reader",), reason="ok",
        )
        with pytest.raises(AttributeError):
            acr.allowed = False

    def test_grant_valid_role(self, key_store):
        key_id, _ = key_store.create_key("test")
        assert key_store.grant_role(key_id, "admin") is True
        roles = key_store.get_roles(key_id)
        assert "admin" in roles

    def test_grant_invalid_role(self, key_store):
        key_id, _ = key_store.create_key("test")
        assert key_store.grant_role(key_id, "superuser") is False

    def test_revoke_role(self, key_store):
        key_id, _ = key_store.create_key("test")
        key_store.grant_role(key_id, "compiler")
        assert key_store.revoke_role(key_id, "compiler") is True
        roles = key_store.get_roles(key_id)
        assert "compiler" not in roles

    def test_get_roles_empty(self, key_store):
        key_id, _ = key_store.create_key("test")
        roles = key_store.get_roles(key_id)
        assert roles == []

    def test_admin_access(self, key_store):
        key_id, _ = key_store.create_key("test")
        key_store.grant_role(key_id, "admin")
        result = key_store.check_access(key_id, "compile")
        assert result.allowed is True
        assert "admin" in result.roles

    def test_reader_denied_compile(self, key_store):
        key_id, _ = key_store.create_key("test")
        key_store.grant_role(key_id, "reader")
        result = key_store.check_access(key_id, "compile")
        assert result.allowed is False

    def test_no_roles_denied(self, key_store):
        key_id, _ = key_store.create_key("test")
        result = key_store.check_access(key_id, "read")
        assert result.allowed is False
        assert "No roles" in result.reason

    def test_multiple_roles_union(self, key_store):
        key_id, _ = key_store.create_key("test")
        key_store.grant_role(key_id, "reader")
        key_store.grant_role(key_id, "compiler")
        result = key_store.check_access(key_id, "compile")
        assert result.allowed is True
        assert len(result.roles) == 2


# ============================================================================
# #185 Paradigm Shifting — mother/paradigm_detector.py
# ============================================================================

class TestParadigmShifting:
    """Genome #185: paradigm-shifting."""

    def test_signal_frozen(self):
        from mother.paradigm_detector import ParadigmShiftSignal
        sig = ParadigmShiftSignal(
            signal_type="stagnation", category="general", severity="low",
            evidence="test", recommendation="test", summary="test",
        )
        with pytest.raises(AttributeError):
            sig.severity = "high"

    def test_assessment_frozen(self):
        from mother.paradigm_detector import ParadigmAssessment
        pa = ParadigmAssessment(
            signals=(), shift_recommended=False, urgency="none",
            assessment_summary="test",
        )
        with pytest.raises(AttributeError):
            pa.shift_recommended = True

    def test_stagnation_true(self):
        from mother.paradigm_detector import detect_stagnation
        scores = [0.65, 0.65, 0.65, 0.65, 0.65]
        assert detect_stagnation(scores) is True

    def test_stagnation_false(self):
        from mother.paradigm_detector import detect_stagnation
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        assert detect_stagnation(scores) is False

    def test_regression_true(self):
        from mother.paradigm_detector import detect_regression
        scores = [0.8, 0.7, 0.6]
        assert detect_regression(scores) is True

    def test_regression_false(self):
        from mother.paradigm_detector import detect_regression
        scores = [0.6, 0.7, 0.8]
        assert detect_regression(scores) is False

    def test_no_shift(self):
        from mother.paradigm_detector import assess_paradigm_shift
        result = assess_paradigm_shift(
            compilation_scores=[0.8, 0.85, 0.9],
            failure_reasons=[],
            goal_completion_rate=0.9,
            stuck_count=0,
        )
        assert result.shift_recommended is False
        assert result.urgency == "none"
        assert len(result.signals) == 0

    def test_shift_recommended(self):
        from mother.paradigm_detector import assess_paradigm_shift
        result = assess_paradigm_shift(
            compilation_scores=[0.5, 0.5, 0.5, 0.5, 0.5],  # stagnation
            failure_reasons=["architecture rewrite needed", "framework limitation"],
            goal_completion_rate=0.2,
            stuck_count=4,
        )
        assert result.shift_recommended is True
        assert result.urgency in ("moderate", "high")
        assert len(result.signals) >= 2
