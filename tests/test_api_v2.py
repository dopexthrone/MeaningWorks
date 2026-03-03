"""
Tests for V2 API — models, routes, metering, middleware.

Phase D: V2 API + Platform Layer
"""

import pytest
import json
from unittest.mock import patch, MagicMock

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
)
from motherlabs_platform.metering import MeteringTracker, DomainMetrics
from core.adapter_registry import get_adapter, list_adapters, clear_registry, register_adapter
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
