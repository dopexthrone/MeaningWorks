"""
Tests for V2 API — models, routes, metering, middleware.

Phase D: V2 API + Platform Layer
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock
from starlette.requests import Request

# Import models directly (no FastAPI server needed)
from api.v2.models import (
    V2CompileRequest,
    V2CompileResponse,
    V2ValidateRequest,
    V2ValidateResponse,
    TrustResponse,
    UsageResponse,
    DomainInfoResponse,
    DomainListResponse,
    V2HealthResponse,
    V2MetricsResponse,
    V2CompileTreeRequest,
    V2MaterializeRequest,
    TaskDecisionRequest,
)
from motherlabs_platform.metering import MeteringTracker, DomainMetrics
from core.adapter_registry import get_adapter, list_adapters, clear_registry, register_adapter
from core.blueprint_protocol import make_node_ref
from core.trust import compute_trust_indicators, serialize_trust_indicators


@pytest.fixture(autouse=True)
def ensure_adapters():
    """Ensure adapters are registered."""
    import adapters  # noqa: F401
    yield


# =============================================================================
# V2 Request Models
# =============================================================================

class TestV2CompileRequest:
    def test_minimal(self):
        req = V2CompileRequest(description="Build an auth system")
        assert req.domain == "software"
        assert req.trust_level == "standard"
        assert req.materialize is True
        assert req.enrich is False

    def test_with_domain(self):
        req = V2CompileRequest(description="Onboarding process", domain="process")
        assert req.domain == "process"

    def test_with_all_fields(self):
        req = V2CompileRequest(
            description="Build a thing",
            domain="software",
            trust_level="thorough",
            materialize=False,
            output_format="yaml",
            provider="claude",
            enrich=True,
            canonical_components=["Auth", "DB"],
            canonical_relationships=[["Auth", "DB", "accesses"]],
        )
        assert req.trust_level == "thorough"
        assert req.materialize is False
        assert req.canonical_components == ["Auth", "DB"]

    def test_validation_empty_description(self):
        with pytest.raises(Exception):
            V2CompileRequest(description="")


class TestV2ValidateRequest:
    def test_minimal(self):
        req = V2ValidateRequest(description="Check this process")
        assert req.domain == "software"

    def test_process_domain(self):
        req = V2ValidateRequest(description="Onboarding", domain="process")
        assert req.domain == "process"


class TestV2CompileTreeRequest:
    def test_defaults(self):
        req = V2CompileTreeRequest(description="Build a platform")
        assert req.max_children == 8
        assert req.domain == "software"


class TestV2MaterializeRequest:
    def test_minimal(self):
        req = V2MaterializeRequest(blueprint={"components": []})
        assert req.domain == "software"
        assert req.max_tokens == 4096


# =============================================================================
# V2 Response Models
# =============================================================================

class TestTrustResponse:
    def test_defaults(self):
        t = TrustResponse()
        assert t.overall_score == 0.0
        assert t.verification_badge == "unverified"
        assert t.fidelity_scores == {}

    def test_with_data(self):
        t = TrustResponse(
            overall_score=82.3,
            provenance_depth=2,
            fidelity_scores={"completeness": 89, "consistency": 95},
            verification_badge="verified",
            gap_report=["Missing: error handling"],
        )
        assert t.overall_score == 82.3
        assert t.verification_badge == "verified"

    def test_json_serializable(self):
        t = TrustResponse(overall_score=75.0, fidelity_scores={"completeness": 80})
        data = t.model_dump()
        json_str = json.dumps(data)
        assert json_str


class TestV2CompileResponse:
    def test_defaults(self):
        r = V2CompileResponse(success=True)
        assert r.domain == "software"
        assert r.trust.verification_badge == "unverified"

    def test_with_trust(self):
        trust = TrustResponse(overall_score=90, verification_badge="verified")
        r = V2CompileResponse(
            success=True,
            blueprint={"components": [{"name": "Auth"}]},
            trust=trust,
            domain="process",
        )
        assert r.trust.overall_score == 90
        assert r.domain == "process"

    def test_error_response(self):
        r = V2CompileResponse(success=False, error="Provider error")
        assert not r.success
        assert r.error == "Provider error"

    def test_glass_box_fields(self):
        r = V2CompileResponse(
            success=True,
            structured_insights=[{"text": "Identified: Auth", "category": "discovery"}],
            difficulty={"unknown_count": 1},
            stage_results=[{"stage": "intent", "success": True, "errors": [], "warnings": [], "retries": 0}],
            stage_timings={"intent": 0.4},
            retry_counts={"intent": 0},
        )
        assert r.structured_insights[0]["text"] == "Identified: Auth"
        assert r.difficulty["unknown_count"] == 1
        assert r.stage_results[0].stage == "intent"
        assert r.stage_timings["intent"] == 0.4

    def test_semantic_nodes_field(self):
        r = V2CompileResponse(
            success=True,
            semantic_nodes=[
                {
                    "id": "node-1",
                    "postcode": "INT.SEM.APP.WHY.SFT",
                    "primitive": "purpose",
                    "description": "Compile intent into a buildable plan.",
                    "notes": [],
                    "fill_state": "F",
                    "confidence": 0.98,
                    "status": "promoted",
                    "version": 1,
                    "created_at": "2026-03-07T22:00:00Z",
                    "updated_at": "2026-03-07T22:00:00Z",
                    "last_verified": "2026-03-07T22:00:00Z",
                    "freshness": {"decay_rate": 0.001, "floor": 0.6, "stale_after": 90},
                    "parent": None,
                    "children": [],
                    "connections": [],
                    "references": {
                        "read_before": [],
                        "read_after": [],
                        "see_also": [],
                        "deep_dive": [],
                        "warns": [],
                    },
                    "provenance": {
                        "source_ref": ["user:intake"],
                        "agent_id": "Intent",
                        "run_id": "run-1",
                        "timestamp": "2026-03-07T22:00:00Z",
                        "human_input": True,
                    },
                    "token_cost": 0,
                    "constraints": [],
                    "constraint_source": [],
                }
            ],
        )
        assert r.semantic_nodes[0].primitive == "purpose"
        assert r.semantic_nodes[0].layer == "INT"

    def test_termination_condition_field(self):
        r = V2CompileResponse(
            success=True,
            termination_condition={
                "status": "stalled",
                "reason": "semantic_progress_stalled",
                "message": "No semantic progress after re-synthesis.",
                "next_action": "Narrow the scope.",
            },
        )
        assert r.termination_condition["reason"] == "semantic_progress_stalled"

    def test_governance_report_field(self):
        r = V2CompileResponse(
            success=True,
            governance_report={
                "total_nodes": 2,
                "promoted": 1,
                "quarantined": [],
                "escalated": [{"postcode": "INT.SEM.APP.WHY.SFT", "question": "Clarify scope?"}],
                "axiom_violations": [],
                "human_decisions": [],
                "coverage": 72.5,
                "anti_goals_checked": 0,
                "compilation_depth": {"label": "demo", "gaps_remaining": 1},
                "cost_report": {"actual_usd": 0.12, "halted": False},
            },
        )
        assert r.governance_report is not None
        assert r.governance_report.coverage == 72.5
        assert r.governance_report.compilation_depth.label == "demo"


class TestUsageResponse:
    def test_defaults(self):
        u = UsageResponse()
        assert u.tokens == 0
        assert u.cost_usd == 0.0
        assert u.domain == "software"


class TestDomainInfoResponse:
    def test_from_adapter(self):
        adapter = get_adapter("software")
        info = DomainInfoResponse(
            name=adapter.name,
            version=adapter.version,
            output_format=adapter.materialization.output_format,
            file_extension=adapter.materialization.file_extension,
            vocabulary_types=sorted(adapter.vocabulary.type_keywords.keys()),
        )
        assert info.name == "software"
        assert info.output_format == "python"
        assert "agent" in info.vocabulary_types

    def test_process_adapter(self):
        adapter = get_adapter("process")
        info = DomainInfoResponse(
            name=adapter.name,
            version=adapter.version,
            output_format=adapter.materialization.output_format,
        )
        assert info.name == "process"
        assert info.output_format == "yaml"


class TestDomainListResponse:
    def test_list(self):
        domains = []
        for name in list_adapters():
            adapter = get_adapter(name)
            domains.append(DomainInfoResponse(
                name=adapter.name, version=adapter.version,
            ))
        resp = DomainListResponse(domains=domains)
        assert len(resp.domains) >= 2
        names = [d.name for d in resp.domains]
        assert "software" in names
        assert "process" in names


class TestV2HealthResponse:
    def test_defaults(self):
        h = V2HealthResponse()
        assert h.status == "ok"
        assert h.version == "2.0.0"

    def test_with_domains(self):
        h = V2HealthResponse(domains_available=["software", "process"])
        assert len(h.domains_available) == 2


# =============================================================================
# Metering
# =============================================================================

class TestMeteringTracker:
    def test_empty(self):
        m = MeteringTracker()
        metrics = m.get_metrics()
        assert metrics["total_compilations"] == 0
        assert metrics["total_cost_usd"] == 0.0

    def test_record_compilation(self):
        m = MeteringTracker()
        m.record_compilation("software", 1.5, 0.05)
        metrics = m.get_metrics()
        assert metrics["total_compilations"] == 1
        assert metrics["per_domain"]["software"]["compilation_count"] == 1
        assert metrics["per_domain"]["software"]["total_cost_usd"] == 0.05

    def test_multiple_domains(self):
        m = MeteringTracker()
        m.record_compilation("software", 1.0, 0.05)
        m.record_compilation("process", 2.0, 0.03)
        m.record_compilation("software", 1.5, 0.04)
        metrics = m.get_metrics()
        assert metrics["total_compilations"] == 3
        assert metrics["per_domain"]["software"]["compilation_count"] == 2
        assert metrics["per_domain"]["process"]["compilation_count"] == 1

    def test_error_tracking(self):
        m = MeteringTracker()
        m.record_compilation("software", 0.5, 0.01, success=False)
        metrics = m.get_metrics()
        assert metrics["per_domain"]["software"]["errors"] == 1
        assert metrics["per_domain"]["software"]["success_rate"] == 0.0

    def test_success_rate(self):
        m = MeteringTracker()
        m.record_compilation("software", 1.0, 0.05, success=True)
        m.record_compilation("software", 1.0, 0.05, success=True)
        m.record_compilation("software", 0.5, 0.01, success=False)
        metrics = m.get_metrics()
        rate = metrics["per_domain"]["software"]["success_rate"]
        assert abs(rate - 0.667) < 0.01

    def test_avg_duration(self):
        m = MeteringTracker()
        m.record_compilation("software", 1.0, 0.05)
        m.record_compilation("software", 3.0, 0.05)
        metrics = m.get_metrics()
        assert metrics["per_domain"]["software"]["avg_duration_seconds"] == 2.0

    def test_get_domain_count(self):
        m = MeteringTracker()
        assert m.get_domain_count("software") == 0
        m.record_compilation("software", 1.0, 0.05)
        assert m.get_domain_count("software") == 1

    def test_reset(self):
        m = MeteringTracker()
        m.record_compilation("software", 1.0, 0.05)
        m.reset()
        assert m.get_metrics()["total_compilations"] == 0

    def test_uptime(self):
        m = MeteringTracker()
        import time
        time.sleep(0.15)
        metrics = m.get_metrics()
        assert metrics["uptime_seconds"] > 0


# =============================================================================
# Trust integration with V2 response
# =============================================================================

class TestTrustIntegration:
    def test_trust_indicators_to_response(self):
        """Verify trust module output maps to V2 TrustResponse model."""
        trust = compute_trust_indicators(
            blueprint={
                "components": [
                    {"name": "Auth", "type": "entity", "description": "Auth system",
                     "derived_from": "user wants authentication"},
                ],
                "relationships": [],
                "constraints": [],
            },
            verification={
                "completeness": {"score": 80},
                "consistency": {"score": 90},
                "coherence": {"score": 75},
                "traceability": {"score": 85},
                "actionability": {"score": 60},
                "specificity": {"score": 55},
                "codegen_readiness": {"score": 70},
            },
            context_graph={"input_hash": "abc", "insights": ["i1"]},
            dimensional_metadata={},
            intent_keywords=["auth"],
        )
        data = serialize_trust_indicators(trust)
        response = TrustResponse(**data)
        assert response.overall_score > 0
        assert response.verification_badge == "verified"
        assert "completeness" in response.fidelity_scores

    def test_full_v2_response_shape(self):
        """Verify complete V2 response has correct shape."""
        trust = TrustResponse(
            overall_score=82.3,
            provenance_depth=2,
            fidelity_scores={
                "completeness": 89, "consistency": 95,
                "coherence": 78, "traceability": 85,
                "actionability": 72, "specificity": 68,
                "codegen_readiness": 74,
            },
            gap_report=["No input covers error handling"],
            verification_badge="verified",
            silence_zones=["behavioral dimension under-explored"],
            confidence_trajectory=[0.1, 0.3, 0.5, 0.65, 0.78, 0.82],
            derivation_chain_length=3.0,
        )
        response = V2CompileResponse(
            success=True,
            blueprint={"components": [{"name": "Auth"}]},
            trust=trust,
            domain="software",
            adapter_version="1.0",
            usage=UsageResponse(tokens=12400, cost_usd=0.05),
        )

        data = response.model_dump()
        assert data["trust"]["overall_score"] == 82.3
        assert data["trust"]["verification_badge"] == "verified"
        assert data["usage"]["cost_usd"] == 0.05
        assert data["domain"] == "software"


# =============================================================================
# Route helpers (unit tests, no server)
# =============================================================================

class TestV2RecompileRequest:
    def test_minimal(self):
        from api.v2.models import V2RecompileRequest
        req = V2RecompileRequest(
            current_blueprint={"components": [], "core_need": "test"},
            enhancement="Add weather checking",
        )
        assert req.domain == "agent_system"
        assert req.enhancement == "Add weather checking"

    def test_custom_domain(self):
        from api.v2.models import V2RecompileRequest
        req = V2RecompileRequest(
            current_blueprint={"components": []},
            enhancement="Add API endpoint",
            domain="software",
        )
        assert req.domain == "software"


class TestV2RecompileResponse:
    def test_defaults(self):
        from api.v2.models import V2RecompileResponse
        r = V2RecompileResponse(success=True)
        assert r.materialized_output == {}
        assert r.domain == "agent_system"

    def test_with_materialized_output(self):
        from api.v2.models import V2RecompileResponse
        r = V2RecompileResponse(
            success=True,
            materialized_output={"ChatAgent": "class ChatAgent: pass"},
            enhancement_applied="Add weather",
        )
        assert "ChatAgent" in r.materialized_output
        assert r.enhancement_applied == "Add weather"

    def test_error_response(self):
        from api.v2.models import V2RecompileResponse
        r = V2RecompileResponse(success=False, error="Provider error")
        assert not r.success
        assert r.error == "Provider error"

    def test_with_scaffold_files(self):
        from api.v2.models import V2RecompileResponse
        r = V2RecompileResponse(
            success=True,
            materialized_output={
                "ChatAgent": "class ChatAgent: pass",
                "runtime.py": "class Runtime: pass",
                "config.py": "class Config: pass",
            },
        )
        assert len(r.materialized_output) == 3
        assert "runtime.py" in r.materialized_output

    def test_json_serializable(self):
        from api.v2.models import V2RecompileResponse
        r = V2RecompileResponse(
            success=True,
            blueprint={"components": [{"name": "A"}]},
            materialized_output={"A": "class A: pass"},
        )
        data = r.model_dump()
        json_str = json.dumps(data)
        assert json_str


class TestBuildTrustResponse:
    def test_builds_from_compilation_data(self):
        from api.v2.routes import _build_trust_response
        trust = _build_trust_response(
            blueprint={"components": [], "relationships": [], "constraints": []},
            verification={"completeness": {"score": 50}},
            context_graph={},
            dimensional_metadata={},
            intent_keywords=[],
        )
        assert isinstance(trust, TrustResponse)
        assert trust.overall_score >= 0


class TestSwarmResultNormalization:
    def test_preserves_glass_box_fields_and_yaml(self):
        from api.v2.routes import _normalize_swarm_compile_result

        raw = {
            "success": True,
            "total_duration_s": 3.2,
            "benchmark": {"composite_pct": 88.0},
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "core_need": "Build auth",
                    "unresolved": [],
                },
                "verification": {"completeness": {"score": 80}},
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 86.0, "verification_badge": "verified"},
                "generated_code": {"AuthService": "class AuthService: pass"},
                "project_manifest": {
                    "project_name": "build_auth",
                    "file_contents": {"auth_service.py": "class AuthService: pass"},
                },
                "compile_result": {
                    "dimensional_metadata": {"coverage": {"entity": 0.92}},
                    "interface_map": {"AuthService": {"methods": []}},
                    "structured_insights": [
                        {"text": "Identified: AuthService", "category": "discovery", "stage": "entity_extraction"},
                    ],
                    "difficulty": {"unknown_count": 1, "irritation_depth": 0.25},
                    "stage_results": [
                        {"stage": "intent", "success": True, "errors": [], "warnings": [], "retries": 0},
                        {"stage": "verification", "success": False, "errors": ["missing rollback"], "warnings": [], "retries": 1},
                    ],
                    "stage_timings": {"intent": 0.3, "verification": 1.7},
                    "retry_counts": {"verification": 1},
                },
                "stub_report": {"stub_count": 0},
            },
        }

        normalized = _normalize_swarm_compile_result(raw)

        assert normalized["structured_insights"][0]["text"] == "Identified: AuthService"
        assert normalized["difficulty"]["unknown_count"] == 1
        assert normalized["dimensional_metadata"]["coverage"]["entity"] == 0.92
        assert normalized["interface_map"]["AuthService"]["methods"] == []
        assert normalized["stage_results"][1]["stage"] == "verification"
        assert normalized["stage_results"][1]["errors"] == ["missing rollback"]
        assert normalized["stage_timings"]["verification"] == 1.7
        assert normalized["retry_counts"]["verification"] == 1
        assert normalized["project_name"] == "build_auth"
        assert normalized["semantic_nodes"][0]["primitive"] == "purpose"
        auth_ref = make_node_ref("STR.ENT.APP.WHAT.SFT", "AuthService")
        auth_node = next(node for node in normalized["semantic_nodes"] if node["primitive"] == "AuthService")
        assert auth_node["references"]["read_before"] == ["INT.SEM.APP.WHY.SFT/purpose"]
        assert make_node_ref(auth_node["postcode"], auth_node["primitive"]) == auth_ref
        assert normalized["governance_report"]["total_nodes"] == 2
        assert normalized["governance_report"]["promoted"] == 2
        assert normalized["governance_report"]["coverage"] > 0
        assert "MANIFEST.yaml" in normalized["yaml_output"]

    def test_normalize_swarm_compile_result_prefers_native_blueprint_semantic_nodes(self):
        from api.v2.routes import _normalize_swarm_compile_result

        raw = {
            "success": True,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "core_need": "Build auth",
                    "unresolved": [],
                    "semantic_nodes": [
                        {
                            "postcode": "EXC.FNC.APP.HOW.SFT",
                            "primitive": "authenticate",
                            "description": "Validate credentials and create a session.",
                            "fill_state": "F",
                            "confidence": 0.94,
                            "connections": ["STR.ENT.APP.WHAT.SFT/authservice"],
                            "source_ref": ["Build auth"],
                        }
                    ],
                },
                "verification": {"completeness": {"score": 80}},
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 86.0, "verification_badge": "verified"},
                "compile_result": {},
            },
        }

        normalized = _normalize_swarm_compile_result(raw)

        refs = {
            make_node_ref(node["postcode"], node["primitive"])
            for node in normalized["semantic_nodes"]
        }
        assert "EXC.FNC.APP.HOW.SFT/authenticate" in refs

    def test_normalize_swarm_compile_result_derives_termination_condition(self):
        from api.v2.routes import _normalize_swarm_compile_result

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "blueprint": {"components": []},
                "verification": {},
                "context_graph": {},
                "trust": {},
                "compile_result": {
                    "success": True,
                    "blocking_escalations": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "Which provider fallback should AuthService use?",
                            "options": ["Anthropic", "OpenAI"],
                        }
                    ],
                },
            },
        }

        normalized = _normalize_swarm_compile_result(raw)
        assert normalized["termination_condition"]["reason"] == "human_decision_required"


class TestTaskDecisionLedger:
    def test_record_task_decision_persists_progress(self, tmp_path, monkeypatch):
        from api.v2.routes import record_task_decision
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))
        raw_request = Request({"type": "http", "headers": []})

        response = asyncio.run(record_task_decision(
            "task-123",
            TaskDecisionRequest(
                postcode="INT.SEM.APP.WHY.SFT",
                question="Lock the scope?",
                answer="yes",
                timestamp="2026-03-07T22:30:00Z",
            ),
            raw_request,
        ))

        saved = progress_module.read_progress("task-123")

        assert response.saved is True
        assert saved is not None
        assert saved["human_decisions"][0]["answer"] == "yes"
        assert saved["human_decisions"][0]["postcode"] == "INT.SEM.APP.WHY.SFT"
        assert response.next_task_id is None

    def test_get_task_status_returns_ledger_on_complete(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": True,
            "total_duration_s": 2.1,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "core_need": "Build auth",
                    "unresolved": ["AuthService needs provider fallback"],
                },
                "verification": {"completeness": {"score": 80}},
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 86.0, "verification_badge": "verified"},
                "compile_result": {},
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-456"))

        assert response.status == "complete"
        assert response.progress is not None
        assert len(response.progress["escalations"]) >= 1

    def test_get_task_status_returns_awaiting_decision_for_fracture(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "request_type": "full_build",
                "blueprint": {},
                "verification": {},
                "context_graph": {},
                "trust": {},
                "compile_result": {
                    "success": False,
                    "error": "Clarification required before compilation can continue",
                    "fracture": {
                        "stage": "interrogation",
                        "competing_configs": ["Public app", "Internal tool"],
                        "collapsing_constraint": "Which direction should I take?",
                        "agent": "Interrogation",
                    },
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-await"))

        assert response.status == "awaiting_decision"
        assert response.progress is not None
        assert response.progress["escalations"][0]["question"] == "Which direction should I take?"
        assert response.progress["escalations"][0]["options"] == ["Public app", "Internal tool"]

    def test_record_task_decision_spawns_continuation_for_fracture(self, tmp_path, monkeypatch):
        from api.v2.routes import record_task_decision
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "provider": "claude",
                "request_type": "full_build",
                "cost_cap_usd": 5.0,
                "blueprint": {},
                "verification": {},
                "context_graph": {},
                "trust": {},
                "compile_result": {
                    "success": False,
                    "error": "Clarification required before compilation can continue",
                    "fracture": {
                        "stage": "interrogation",
                        "competing_configs": ["Public app", "Internal tool"],
                        "collapsing_constraint": "Which direction should I take?",
                        "agent": "Interrogation",
                    },
                },
            },
        }
        raw_request = Request({"type": "http", "headers": []})

        with patch("worker.config.huey.result", return_value=raw), patch(
            "worker.swarm_tasks.swarm_execute_task",
            return_value=MagicMock(id="task-next"),
        ) as mock_resume:
            response = asyncio.run(record_task_decision(
                "task-await",
                TaskDecisionRequest(
                    postcode="INT.SEM.APP.IF.SFT",
                    question="Which direction should I take?",
                    answer="Internal tool",
                ),
                raw_request,
            ))

        assert response.saved is True
        assert response.next_task_id == "task-next"
        resume_intent = mock_resume.call_args.kwargs["intent"]
        assert "Internal tool" in resume_intent
        assert "Continue compilation with this decision treated as locked context." in resume_intent

    def test_get_task_status_returns_awaiting_decision_for_blocking_semantic_gate(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "request_type": "full_build",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "unresolved": ["AuthService needs provider fallback"],
                },
                "verification": {},
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 78.0, "gap_report": ["AuthService needs provider fallback"]},
                "compile_result": {
                    "success": True,
                    "semantic_nodes": [
                        {
                            "id": "node-1-authservice",
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "primitive": "AuthService",
                            "description": "Handles auth",
                            "notes": [],
                            "fill_state": "B",
                            "confidence": 0.62,
                            "status": "authored",
                            "version": 1,
                            "created_at": "2026-03-07T22:00:00Z",
                            "updated_at": "2026-03-07T22:00:00Z",
                            "last_verified": "2026-03-07T22:00:00Z",
                            "freshness": {"decay_rate": 0.001, "floor": 0.6, "stale_after": 90},
                            "parent": None,
                            "children": [],
                            "connections": [],
                            "references": {"read_before": [], "read_after": [], "see_also": [], "deep_dive": [], "warns": []},
                            "provenance": {
                                "source_ref": ["Build auth"],
                                "agent_id": "Synthesis",
                                "run_id": "swarm",
                                "timestamp": "2026-03-07T22:00:00Z",
                                "human_input": False,
                            },
                            "token_cost": 0,
                            "constraints": [],
                            "constraint_source": [],
                        }
                    ],
                    "blocking_escalations": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "AuthService needs provider fallback",
                            "options": [],
                        }
                    ],
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-blocked"))

        assert response.status == "awaiting_decision"
        assert response.progress is not None
        assert response.progress["escalations"][0]["question"] == "AuthService needs provider fallback"

    def test_get_task_status_derives_awaiting_decision_for_conflict_gate(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "request_type": "full_build",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "unresolved": [],
                },
                "verification": {},
                "context_graph": {
                    "keywords": ["auth"],
                    "conflict_summary": {
                        "unresolved": [
                            {
                                "topic": "AuthService: storage strategy",
                                "category": "MISSING_INFO",
                                "positions": {
                                    "Entity": "Persist sessions in PostgreSQL",
                                    "Process": "Keep sessions stateless with JWT",
                                },
                            }
                        ]
                    },
                },
                "trust": {"overall_score": 82.0, "gap_report": []},
                "compile_result": {
                    "success": True,
                    "semantic_nodes": [
                        {
                            "id": "node-1-authservice",
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "primitive": "AuthService",
                            "description": "Handles auth",
                            "notes": [],
                            "fill_state": "P",
                            "confidence": 0.78,
                            "status": "authored",
                            "version": 1,
                            "created_at": "2026-03-07T22:00:00Z",
                            "updated_at": "2026-03-07T22:00:00Z",
                            "last_verified": "2026-03-07T22:00:00Z",
                            "freshness": {"decay_rate": 0.001, "floor": 0.6, "stale_after": 90},
                            "parent": None,
                            "children": [],
                            "connections": [],
                            "references": {"read_before": [], "read_after": [], "see_also": [], "deep_dive": [], "warns": []},
                            "provenance": {
                                "source_ref": ["Build auth"],
                                "agent_id": "Synthesis",
                                "run_id": "swarm",
                                "timestamp": "2026-03-07T22:00:00Z",
                                "human_input": False,
                            },
                            "token_cost": 0,
                            "constraints": [],
                            "constraint_source": [],
                        }
                    ],
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-conflict"))

        assert response.status == "awaiting_decision"
        assert response.progress is not None
        escalation = response.progress["escalations"][0]
        assert "storage strategy" in escalation["question"].lower()
        assert escalation["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert escalation["options"] == [
            "Persist sessions in PostgreSQL",
            "Keep sessions stateless with JWT",
        ]

    def test_get_task_status_prefers_native_blueprint_semantic_gates(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "request_type": "full_build",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "unresolved": [],
                    "semantic_gates": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "AuthService needs provider fallback",
                            "options": [],
                            "node_ref": "STR.ENT.APP.WHAT.SFT/authservice",
                            "kind": "gap",
                            "stage": "verification",
                        }
                    ],
                },
                "verification": {},
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 78.0, "gap_report": []},
                "compile_result": {
                    "success": True,
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-native-gate"))

        assert response.status == "awaiting_decision"
        escalation = response.progress["escalations"][0]
        assert escalation["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert escalation["kind"] == "gap"

    def test_get_task_status_hydrates_verification_semantic_gates(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "request_type": "full_build",
                "blueprint": {
                    "components": [
                        {
                            "name": "AuthService",
                            "type": "entity",
                            "description": "Handles auth",
                            "derived_from": "Build auth",
                            "attributes": {},
                            "methods": [],
                            "validation_rules": [],
                        }
                    ],
                    "relationships": [],
                    "constraints": [],
                    "unresolved": [],
                },
                "verification": {
                    "status": "needs_work",
                    "semantic_gates": [
                        {
                            "owner_component": "AuthService",
                            "question": "Which provider fallback should AuthService use?",
                            "kind": "semantic_conflict",
                            "options": ["Anthropic", "OpenAI"],
                            "stage": "verification",
                        }
                    ],
                },
                "context_graph": {"keywords": ["auth"]},
                "trust": {"overall_score": 78.0, "gap_report": []},
                "compile_result": {
                    "success": True,
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-verification-gate"))

        assert response.status == "awaiting_decision"
        escalation = response.progress["escalations"][0]
        assert escalation["node_ref"] == "STR.ENT.APP.WHAT.SFT/authservice"
        assert escalation["kind"] == "semantic_conflict"
        assert escalation["options"] == ["Anthropic", "OpenAI"]

    def test_record_task_decision_spawns_continuation_for_blocking_semantic_gate(self, tmp_path, monkeypatch):
        from api.v2.routes import record_task_decision
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "provider": "claude",
                "request_type": "full_build",
                "cost_cap_usd": 5.0,
                "compile_result": {
                    "success": True,
                    "blocking_escalations": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "AuthService needs provider fallback",
                            "options": [],
                        }
                    ],
                },
            },
        }
        raw_request = Request({"type": "http", "headers": []})

        with patch("worker.config.huey.result", return_value=raw), patch(
            "worker.swarm_tasks.swarm_execute_task",
            return_value=MagicMock(id="task-blocked-next"),
        ) as mock_resume:
            response = asyncio.run(record_task_decision(
                "task-blocked",
                TaskDecisionRequest(
                    postcode="STR.ENT.APP.WHAT.SFT",
                    question="AuthService needs provider fallback",
                    answer="Use Anthropic primary with OpenAI fallback",
                ),
                raw_request,
            ))

        assert response.saved is True
        assert response.next_task_id == "task-blocked-next"
        assert "Use Anthropic primary with OpenAI fallback" in mock_resume.call_args.kwargs["intent"]

    def test_record_task_decision_carries_conflict_options_into_continuation(self, tmp_path, monkeypatch):
        from api.v2.routes import record_task_decision
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "provider": "claude",
                "request_type": "full_build",
                "cost_cap_usd": 5.0,
                "compile_result": {
                    "success": True,
                    "blocking_escalations": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
                            "options": [
                                "Persist sessions in PostgreSQL",
                                "Keep sessions stateless with JWT",
                            ],
                        }
                    ],
                },
            },
        }
        raw_request = Request({"type": "http", "headers": []})

        with patch("worker.config.huey.result", return_value=raw), patch(
            "worker.swarm_tasks.swarm_execute_task",
            return_value=MagicMock(id="task-conflict-next"),
        ) as mock_resume:
            response = asyncio.run(record_task_decision(
                "task-conflict",
                TaskDecisionRequest(
                    postcode="STR.ENT.APP.WHAT.SFT",
                    question="Which direction should Motherlabs lock for AuthService: storage strategy?",
                    answer="Persist sessions in PostgreSQL",
                ),
                raw_request,
            ))

        assert response.saved is True
        assert response.next_task_id == "task-conflict-next"
        resume_intent = mock_resume.call_args.kwargs["intent"]
        assert "Persist sessions in PostgreSQL" in resume_intent
        assert "Keep sessions stateless with JWT" in resume_intent

    def test_record_task_decision_blocks_repeated_pause_cycle(self, tmp_path, monkeypatch):
        from api.v2.routes import record_task_decision
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        repeated_pause = {
            "postcode": "STR.ENT.APP.WHAT.SFT",
            "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
            "options": [
                "Persist sessions in PostgreSQL",
                "Keep sessions stateless with JWT",
            ],
        }
        lineage_result = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "provider": "claude",
                "request_type": "full_build",
                "cost_cap_usd": 5.0,
                "compile_result": {
                    "success": True,
                    "blocking_escalations": [repeated_pause],
                },
            },
        }
        current = {
            **lineage_result,
            "state": {
                **lineage_result["state"],
                "previous_task_id": "task-prev-1",
            },
        }
        previous_1 = {
            **lineage_result,
            "state": {
                **lineage_result["state"],
                "previous_task_id": "task-prev-2",
            },
        }
        previous_2 = {
            **lineage_result,
            "state": {
                **lineage_result["state"],
                "previous_task_id": None,
            },
        }
        raw_request = Request({"type": "http", "headers": []})

        def _result_for(task_id, preserve=True):
            return {
                "task-conflict": current,
                "task-prev-1": previous_1,
                "task-prev-2": previous_2,
            }.get(task_id)

        with patch("worker.config.huey.result", side_effect=_result_for), patch(
            "worker.swarm_tasks.swarm_execute_task",
            return_value=MagicMock(id="task-conflict-next"),
        ) as mock_resume:
            response = asyncio.run(record_task_decision(
                "task-conflict",
                TaskDecisionRequest(
                    postcode=repeated_pause["postcode"],
                    question=repeated_pause["question"],
                    answer="Persist sessions in PostgreSQL",
                ),
                raw_request,
            ))

        saved = progress_module.read_progress("task-conflict")

        assert response.saved is True
        assert response.next_task_id is None
        assert response.termination_condition is not None
        assert response.termination_condition["reason"] == "continuation_cycle_detected"
        assert saved is not None
        assert saved["termination_condition"]["reason"] == "continuation_cycle_detected"
        mock_resume.assert_not_called()

    def test_get_task_status_prefers_progress_termination_condition(self, tmp_path, monkeypatch):
        from api.v2.routes import get_task_status
        from worker import progress as progress_module

        monkeypatch.setattr(progress_module, "_PROGRESS_DB", str(tmp_path / "progress.db"))

        progress_module.write_task_termination(
            "task-guarded",
            {
                "status": "stalled",
                "reason": "continuation_cycle_detected",
                "message": "The same semantic pause has reappeared across continuation tasks.",
                "next_action": "Start a fresh compile.",
            },
        )

        raw = {
            "success": False,
            "state": {
                "intent": "Build auth",
                "domain": "software",
                "compile_result": {
                    "success": True,
                    "blocking_escalations": [
                        {
                            "postcode": "STR.ENT.APP.WHAT.SFT",
                            "question": "Which direction should Motherlabs lock for AuthService: storage strategy?",
                            "options": [
                                "Persist sessions in PostgreSQL",
                                "Keep sessions stateless with JWT",
                            ],
                        }
                    ],
                },
            },
        }

        with patch("worker.config.huey.result", return_value=raw):
            response = asyncio.run(get_task_status("task-guarded"))

        assert response.status == "complete"
        assert response.result is not None
        assert response.result["termination_condition"]["reason"] == "continuation_cycle_detected"
