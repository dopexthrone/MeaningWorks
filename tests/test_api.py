"""
Phase 7.3: REST API Tests.

Tests FastAPI endpoints using TestClient.
"""

import json
import pytest
from unittest.mock import Mock
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app, set_engine, get_engine
from api.models import CompileRequest, CompileResponse, HealthResponse
from core.engine import MotherlabsEngine, CompileResult, StageResult
from core.llm import BaseLLMClient
from persistence.corpus import Corpus


# =============================================================================
# MOCK RESPONSES (reuse from test_engine)
# =============================================================================

MOCK_INTENT_JSON = {
    "core_need": "Build a user authentication system",
    "domain": "authentication",
    "actors": ["User", "AuthService"],
    "implicit_goals": ["Secure session management"],
    "constraints": ["Must handle sessions"],
    "insight": "Core need is secure identity verification",
    "explicit_components": [],
    "explicit_relationships": [],
}

MOCK_PERSONA_JSON = {
    "personas": [
        {
            "name": "Security Architect",
            "perspective": "Focus on auth security",
            "blind_spots": "May over-engineer",
        },
        {
            "name": "UX Designer",
            "perspective": "Focus on UX",
            "blind_spots": "May underestimate security",
        },
    ],
}

MOCK_SYNTHESIS_JSON = {
    "components": [
        {"name": "User", "type": "entity", "description": "User account with authentication credentials and email", "derived_from": "INSIGHT: User entity contains email, password_hash, created_at fields"},
        {"name": "Session", "type": "entity", "description": "Active user session with token and expiry for authentication", "derived_from": "INSIGHT: Session entity contains token, expiry for session management"},
        {"name": "AuthService", "type": "process", "description": "Authentication service handling login and session management", "derived_from": "INSIGHT: AuthService manages User and Session lifecycle"},
    ],
    "relationships": [
        {"from": "AuthService", "to": "User", "type": "accesses", "description": "AuthService validates User credentials"},
        {"from": "AuthService", "to": "Session", "type": "generates", "description": "AuthService creates Session on login"},
        {"from": "Session", "to": "User", "type": "depends_on", "description": "Session belongs to User"},
    ],
    "constraints": [],
    "unresolved": [],
}

MOCK_VERIFY_JSON = {
    "status": "pass",
    "completeness": {"score": 85},
    "consistency": {"score": 90},
    "coherence": {"score": 80},
    "traceability": {"score": 95},
}


MOCK_KERNEL_EXTRACTIONS = json.dumps([
    {"postcode": "SEM.ENT.ECO.WHAT.SFT", "primitive": "user", "content": "User entity", "confidence": 0.9, "connections": []},
    {"postcode": "SEM.BHV.ECO.HOW.SFT", "primitive": "auth", "content": "Auth flow", "confidence": 0.85, "connections": []},
])

SEM_EXTRACT_MARKER = "You are a semantic compiler. You extract structured concepts"


def make_api_mock():
    """Create mock LLM client for API testing."""
    client = Mock(spec=BaseLLMClient)
    client.deterministic = True
    client.model = "mock-model"
    call_count = [0]

    def mock_complete(system_prompt, user_content, **kwargs):
        # Kernel extraction calls get extraction JSON (not pipeline responses)
        if SEM_EXTRACT_MARKER in system_prompt:
            return MOCK_KERNEL_EXTRACTIONS
        # Detect synthesis/verify by system prompt (position-independent)
        if "You are the Synthesis Agent" in system_prompt:
            return json.dumps(MOCK_SYNTHESIS_JSON)
        if "You are the Verify Agent" in system_prompt:
            return json.dumps(MOCK_VERIFY_JSON)
        # Intent + persona use sequential slots, dialogue cycles
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return json.dumps(MOCK_INTENT_JSON)
        if idx == 1:
            return json.dumps(MOCK_PERSONA_JSON)
        if idx % 2 == 0:
            return "INSIGHT: system builds User entity with email, password_hash"
        return "INSIGHT: system builds login flow that validates user credentials"

    client.complete_with_system = Mock(side_effect=mock_complete)
    return client


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def api_engine(tmp_path):
    """Create engine for API testing and register it."""
    client = make_api_mock()
    corpus = Corpus(corpus_path=tmp_path / "corpus")
    engine = MotherlabsEngine(
        llm_client=client,
        corpus=corpus,
        auto_store=True,
        cache_policy="none",
    )
    set_engine(engine)
    yield engine
    set_engine(None)


@pytest.fixture
def test_client(api_engine):
    """Create FastAPI TestClient."""
    return TestClient(app)


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================


class TestHealthEndpoint:
    """Test GET /v1/health."""

    def test_health_returns_ok(self, test_client):
        """Health check returns a valid status."""
        response = test_client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy", "degraded", "unhealthy")

    def test_health_returns_version(self, test_client):
        """Health check returns version."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.2.0"

    def test_health_returns_corpus_size(self, test_client):
        """Health check returns corpus size."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert "corpus_size" in data
        assert isinstance(data["corpus_size"], int)


# =============================================================================
# COMPILE ENDPOINT
# =============================================================================


class TestCompileEndpoint:
    """Test POST /v1/compile."""

    def test_compile_success(self, test_client):
        """Compile endpoint returns successful result."""
        response = test_client.post(
            "/v1/compile",
            json={"description": "Build a user authentication system"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert len(data["blueprint"]["components"]) > 0

    def test_compile_returns_insights(self, test_client):
        """Compile endpoint returns insights."""
        response = test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        data = response.json()
        assert isinstance(data["insights"], list)

    def test_compile_returns_stage_results(self, test_client):
        """Compile endpoint returns stage results."""
        response = test_client.post(
            "/v1/compile",
            json={"description": "Build auth"},
        )
        data = response.json()
        assert isinstance(data["stage_results"], list)
        assert len(data["stage_results"]) == 5

    def test_compile_with_canonical_components(self, test_client):
        """Compile with canonical components."""
        response = test_client.post(
            "/v1/compile",
            json={
                "description": "Build auth",
                "canonical_components": ["User", "Session"],
            },
        )
        assert response.status_code == 200

    def test_compile_empty_description_fails(self, test_client):
        """Empty description returns 422."""
        response = test_client.post(
            "/v1/compile",
            json={"description": ""},
        )
        assert response.status_code == 422

    def test_compile_missing_description_fails(self, test_client):
        """Missing description returns 422."""
        response = test_client.post("/v1/compile", json={})
        assert response.status_code == 422

    def test_compile_returns_schema_validation(self, test_client):
        """Compile returns schema validation info."""
        response = test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        data = response.json()
        assert "schema_validation" in data

    def test_compile_returns_graph_validation(self, test_client):
        """Compile returns graph validation info."""
        response = test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        data = response.json()
        assert "graph_validation" in data

    def test_compile_with_relationships(self, test_client):
        """Compile with canonical relationships."""
        response = test_client.post(
            "/v1/compile",
            json={
                "description": "Build auth",
                "canonical_relationships": [
                    ["AuthService", "User", "accesses"],
                ],
            },
        )
        assert response.status_code == 200


# =============================================================================
# SELF-COMPILE ENDPOINT
# =============================================================================


class TestSelfCompileEndpoint:
    """Test POST /v1/self-compile."""

    def test_self_compile(self, test_client):
        """Self-compile endpoint works."""
        response = test_client.post("/v1/self-compile")
        assert response.status_code == 200
        data = response.json()
        assert "blueprint" in data
        assert "success" in data


# =============================================================================
# CORPUS ENDPOINTS
# =============================================================================


class TestCorpusEndpoints:
    """Test GET /v1/corpus and GET /v1/corpus/{id}."""

    def test_corpus_list_empty(self, test_client):
        """Empty corpus returns empty list."""
        response = test_client.get("/v1/corpus")
        assert response.status_code == 200
        data = response.json()
        assert data["records"] == []
        assert data["total"] == 0

    def test_corpus_list_after_compile(self, test_client):
        """Corpus lists compilations after compile."""
        # First compile something
        test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        # Then list
        response = test_client.get("/v1/corpus")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["records"]) >= 1

    def test_corpus_pagination(self, test_client):
        """Corpus supports pagination."""
        response = test_client.get("/v1/corpus?page=1&per_page=5")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["per_page"] == 5

    def test_corpus_domain_filter(self, test_client):
        """Corpus supports domain filtering."""
        test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        response = test_client.get("/v1/corpus?domain=authentication")
        assert response.status_code == 200

    def test_corpus_get_not_found(self, test_client):
        """Non-existent compilation returns 404."""
        response = test_client.get("/v1/corpus/nonexistent_id")
        assert response.status_code == 404

    def test_corpus_get_after_compile(self, test_client):
        """Can retrieve compilation by ID after compiling."""
        # Compile first
        test_client.post(
            "/v1/compile",
            json={"description": "Build auth system"},
        )
        # Get the list to find the ID
        list_response = test_client.get("/v1/corpus")
        records = list_response.json()["records"]
        if records:
            record_id = records[0]["id"]
            response = test_client.get(f"/v1/corpus/{record_id}")
            assert response.status_code == 200
            data = response.json()
            assert "record" in data
            assert "blueprint" in data


# =============================================================================
# API MODELS
# =============================================================================


class TestAPIModels:
    """Test Pydantic request/response models."""

    def test_compile_request_validation(self):
        """CompileRequest validates description."""
        req = CompileRequest(description="Build something")
        assert req.description == "Build something"

    def test_compile_request_optional_fields(self):
        """CompileRequest has optional fields."""
        req = CompileRequest(description="Build something")
        assert req.provider is None
        assert req.canonical_components is None

    def test_health_response_defaults(self):
        """HealthResponse has sensible defaults."""
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.2.0"


# =============================================================================
# CODEGEN ENDPOINTS - Phase 9.3
# =============================================================================


class TestCodegenEndpoint:
    """Test POST /v1/codegen."""

    def test_codegen_basic(self, test_client):
        """Basic codegen from a simple blueprint."""
        blueprint = {
            "components": [
                {"name": "User", "type": "entity", "description": "A user"},
                {"name": "Order", "type": "entity", "description": "An order"},
            ],
            "relationships": [
                {"from": "Order", "to": "User", "type": "depends_on"},
            ],
            "constraints": [],
        }
        response = test_client.post("/v1/codegen", json={
            "blueprint": blueprint,
            "include_tests": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "blueprint_code.py" in data["files"]
        assert "test_blueprint.py" in data["files"]
        assert data["component_count"] == 2

    def test_codegen_without_tests(self, test_client):
        """Codegen without tests generates only code file."""
        blueprint = {
            "components": [
                {"name": "Item", "type": "entity", "description": "An item"},
            ],
            "relationships": [],
            "constraints": [],
        }
        response = test_client.post("/v1/codegen", json={
            "blueprint": blueprint,
            "include_tests": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"]
        assert "blueprint_code.py" in data["files"]
        assert "test_blueprint.py" not in data["files"]

    def test_codegen_generated_code_is_python(self, test_client):
        """Generated code should be valid Python."""
        blueprint = {
            "components": [
                {"name": "Session", "type": "entity", "description": "A session"},
            ],
            "relationships": [],
            "constraints": [],
        }
        response = test_client.post("/v1/codegen", json={
            "blueprint": blueprint,
        })
        data = response.json()
        code = data["files"]["blueprint_code.py"]
        # Should be syntactically valid
        compile(code, "<test>", "exec")

    def test_codegen_with_constraints(self, test_client):
        """Codegen with constraints includes constraint comments."""
        blueprint = {
            "components": [
                {
                    "name": "Booking",
                    "type": "entity",
                    "description": "A booking",
                    "methods": [{
                        "name": "reserve",
                        "parameters": [],
                        "return_type": "bool",
                        "description": "Reserve a slot",
                    }],
                },
            ],
            "relationships": [],
            "constraints": [
                {"description": "duration must be positive", "applies_to": ["Booking"]},
            ],
        }
        response = test_client.post("/v1/codegen", json={
            "blueprint": blueprint,
        })
        data = response.json()
        code = data["files"]["blueprint_code.py"]
        assert "CONSTRAINT:" in code or "return False" in code

    def test_codegen_empty_blueprint(self, test_client):
        """Empty blueprint generates minimal code."""
        response = test_client.post("/v1/codegen", json={
            "blueprint": {"components": [], "relationships": [], "constraints": []},
        })
        data = response.json()
        assert data["success"]
        assert data["component_count"] == 0

    def test_codegen_component_count(self, test_client):
        """Component count matches blueprint."""
        blueprint = {
            "components": [
                {"name": "A", "type": "entity", "description": "A"},
                {"name": "B", "type": "entity", "description": "B"},
                {"name": "C", "type": "entity", "description": "C"},
            ],
            "relationships": [],
            "constraints": [],
        }
        response = test_client.post("/v1/codegen", json={"blueprint": blueprint})
        assert response.json()["component_count"] == 3


class TestCompileAndGenerateEndpoint:
    """Test POST /v1/compile-and-generate."""

    def test_compile_and_generate_basic(self, test_client):
        """Full pipeline produces both compilation and codegen."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={
                "description": "Build a booking system for a tattoo studio with artists, clients, and sessions",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "compilation" in data
        assert "codegen" in data
        assert data["codegen"]["success"]
        assert "blueprint_code.py" in data["codegen"]["files"]

    def test_compile_and_generate_without_tests(self, test_client):
        """include_tests=false skips test generation."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={
                "description": "Build a booking system for a tattoo studio with artists and clients",
                "include_tests": False,
            },
        )
        data = response.json()
        assert "test_blueprint.py" not in data["codegen"]["files"]

    def test_compile_and_generate_vague_input(self, test_client):
        """Vague input should be rejected by quality gate."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={"description": "Build a thing"},
        )
        # Should get a 422 from InputQualityError
        assert response.status_code == 422

    def test_compile_and_generate_compilation_included(self, test_client):
        """Compilation section has standard compile response fields."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={
                "description": "Build an inventory system for a restaurant with managers tracking ingredient stock levels",
            },
        )
        data = response.json()
        compilation = data["compilation"]
        assert "success" in compilation
        assert "blueprint" in compilation
        assert "insights" in compilation

    def test_compile_and_generate_with_custom_quality(self, test_client):
        """Custom min_quality_score is respected."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={
                "description": "A booking system",
                "min_quality_score": 0.95,
            },
        )
        # Should be rejected - score won't be 0.95
        assert response.status_code == 422

    def test_compile_and_generate_llm_mode(self, test_client):
        """codegen_mode='llm' uses agent emission instead of template."""
        response = test_client.post(
            "/v1/compile-and-generate",
            json={
                "description": "Build a booking system for a tattoo studio with artists, clients, and sessions",
                "codegen_mode": "llm",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "codegen" in data
        # LLM mode produces files (MockClient returns non-code so files may be empty)
        assert "success" in data["codegen"]


# =============================================================================
# MATERIALIZE ENDPOINT (Phase D)
# =============================================================================


class TestMaterializeEndpoint:
    """Test POST /v1/materialize."""

    def test_materialize_basic(self, test_client):
        """Basic materialization with a small blueprint."""
        response = test_client.post(
            "/v1/materialize",
            json={
                "blueprint": {
                    "components": [
                        {"name": "UserService", "type": "service", "description": "User management"},
                    ],
                    "relationships": [],
                    "constraints": [],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_nodes" in data
        assert "pass_rate" in data
        assert "files" in data

    def test_materialize_empty_blueprint(self, test_client):
        """Empty blueprint returns zero nodes."""
        response = test_client.post(
            "/v1/materialize",
            json={"blueprint": {"components": [], "relationships": []}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 0

    def test_materialize_with_interface_map(self, test_client):
        """Providing interface_map skips extraction."""
        response = test_client.post(
            "/v1/materialize",
            json={
                "blueprint": {
                    "components": [
                        {"name": "A", "type": "service", "description": "Service A"},
                        {"name": "B", "type": "service", "description": "Service B"},
                    ],
                    "relationships": [{"from": "A", "to": "B", "type": "depends_on"}],
                    "constraints": [],
                },
                "interface_map": {
                    "contracts": [{
                        "node_a": "A",
                        "node_b": "B",
                        "relationship_type": "depends_on",
                        "relationship_description": "A depends on B",
                        "data_flows": [],
                        "constraints": [],
                        "fragility": 0.5,
                        "confidence": 0.9,
                        "directionality": "A_depends_on_B",
                        "derived_from": "test",
                    }],
                    "unmatched_relationships": [],
                    "extraction_confidence": 1.0,
                    "derived_from": "test",
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 2

    def test_materialize_response_shape(self, test_client):
        """Response has all expected fields."""
        response = test_client.post(
            "/v1/materialize",
            json={
                "blueprint": {
                    "components": [
                        {"name": "Foo", "type": "entity", "description": "A foo"},
                    ],
                    "relationships": [],
                    "constraints": [],
                },
            },
        )
        data = response.json()
        assert "success" in data
        assert "files" in data
        assert "verification" in data
        assert "total_nodes" in data
        assert "success_count" in data
        assert "failure_count" in data
        assert "pass_rate" in data


# =============================================================================
# PHASE 19: METRICS AND ENHANCED HEALTH TESTS
# =============================================================================


class TestMetricsEndpoint:
    """Phase 19: Test GET /v1/metrics."""

    def test_metrics_endpoint_returns_200(self, test_client):
        """Metrics endpoint returns 200."""
        response = test_client.get("/v1/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_has_aggregate(self, test_client):
        """Metrics response includes aggregate field."""
        response = test_client.get("/v1/metrics")
        data = response.json()
        assert "aggregate" in data
        assert "window_size" in data["aggregate"]

    def test_metrics_endpoint_has_corpus_stats(self, test_client):
        """Metrics response includes corpus stats."""
        response = test_client.get("/v1/metrics")
        data = response.json()
        assert "corpus" in data
        assert "total_compilations" in data["corpus"]

    def test_metrics_endpoint_has_cache(self, test_client):
        """Metrics response includes cache stats."""
        response = test_client.get("/v1/metrics")
        data = response.json()
        assert "cache" in data


class TestEnhancedHealth:
    """Phase 19: Test enhanced GET /v1/health."""

    def test_health_returns_enhanced_fields(self, test_client):
        """Health endpoint now returns uptime and recent metrics."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert "compilations_recent" in data
        assert "recent_success_rate" in data
        assert "issues" in data

    def test_health_backwards_compatible(self, test_client):
        """Health endpoint still returns original fields."""
        response = test_client.get("/v1/health")
        data = response.json()
        assert data["status"] in ("ok", "healthy", "degraded", "unhealthy")
        assert "version" in data
        assert data["version"] == "0.2.0"


# =============================================================================
# AGENT ENDPOINT (Ship the Compiler as a Product)
# =============================================================================


class TestAgentEndpoint:
    """Test POST /v1/agent."""

    def test_agent_endpoint_exists(self, test_client):
        """Agent endpoint accepts POST requests."""
        response = test_client.post("/v1/agent", json={
            "description": "A task management system with teams and deadlines",
        })
        # Should return 200 (may succeed or fail depending on mock responses)
        assert response.status_code in (200, 422)

    def test_agent_endpoint_returns_structure(self, test_client):
        """Agent response has expected fields."""
        from unittest.mock import patch
        from core.agent_orchestrator import AgentResult

        mock_result = AgentResult(
            success=True,
            blueprint={"domain": "test", "components": []},
            generated_code={"Widget": "class Widget: pass"},
            quality_score=0.85,
            timing={"compile": 1.0},
        )

        with patch('core.agent_orchestrator.AgentOrchestrator') as MockOrch:
            mock_orch_instance = MockOrch.return_value
            mock_orch_instance.run.return_value = mock_result

            response = test_client.post("/v1/agent", json={
                "description": "A task management system with teams and deadlines",
            })

        data = response.json()
        assert "success" in data
        assert "blueprint" in data
        assert "generated_code" in data
        assert "quality_score" in data
        assert "timing" in data
        # Semantic data fields
        assert "verification" in data
        assert "dimensional_metadata" in data
        assert "interface_map" in data
        assert "stage_results" in data

    def test_agent_empty_description_rejected(self, test_client):
        """Empty description is rejected by validation."""
        response = test_client.post("/v1/agent", json={
            "description": "",
        })
        assert response.status_code == 422


class TestCompileResponseFields:
    """Test that /v1/compile returns dimensional_metadata, interface_map, corpus_suggestions."""

    def test_compile_returns_dimensional_metadata(self, test_client):
        response = test_client.post("/v1/compile", json={
            "description": "A task management system with teams and deadlines and notifications"
        })
        data = response.json()
        assert "dimensional_metadata" in data
        assert isinstance(data["dimensional_metadata"], dict)

    def test_compile_returns_interface_map(self, test_client):
        response = test_client.post("/v1/compile", json={
            "description": "A task management system with teams and deadlines and notifications"
        })
        data = response.json()
        assert "interface_map" in data
        assert isinstance(data["interface_map"], dict)

    def test_compile_returns_corpus_suggestions(self, test_client):
        response = test_client.post("/v1/compile", json={
            "description": "A task management system with teams and deadlines and notifications"
        })
        data = response.json()
        assert "corpus_suggestions" in data
        assert isinstance(data["corpus_suggestions"], dict)

    def test_compile_with_enrich_flag(self, test_client):
        """enrich=True is accepted without error."""
        response = test_client.post("/v1/compile", json={
            "description": "A task management system with teams and deadlines",
            "enrich": True,
        })
        assert response.status_code == 200


class TestSelfCompileLoopEndpoint:
    """Test POST /v1/self-compile-loop."""

    def test_self_compile_loop_endpoint_exists(self, test_client):
        response = test_client.post("/v1/self-compile-loop", json={"runs": 1})
        assert response.status_code == 200

    def test_self_compile_loop_response_shape(self, test_client):
        response = test_client.post("/v1/self-compile-loop", json={"runs": 1})
        data = response.json()
        assert "converged" in data
        assert "stability_score" in data
        assert "run_count" in data
        assert "patterns" in data
        assert "overall_health" in data


class TestCompileTreeEndpoint:
    """Test POST /v1/compile-tree."""

    def test_compile_tree_endpoint_exists(self, test_client):
        response = test_client.post("/v1/compile-tree", json={
            "description": "A large e-commerce platform with inventory, orders, payments, and shipping"
        })
        assert response.status_code == 200

    def test_compile_tree_response_shape(self, test_client):
        response = test_client.post("/v1/compile-tree", json={
            "description": "A large e-commerce platform with inventory, orders, payments, and shipping"
        })
        data = response.json()
        assert "success" in data
        assert "root_blueprint" in data
        assert "tree_health" in data
        assert "total_components" in data

    def test_compile_tree_empty_description_rejected(self, test_client):
        response = test_client.post("/v1/compile-tree", json={
            "description": "",
        })
        assert response.status_code == 422
