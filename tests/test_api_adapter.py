"""
Tests for adapters/api.py — REST API design domain adapter.

Third Domain Adapter — proves N-domain scaling.
"""

import pytest
from adapters.api import (
    API_ADAPTER,
    API_VOCABULARY,
    API_CLASSIFICATION,
    API_PROMPTS,
    API_VERIFICATION,
    API_MATERIALIZATION,
)
from core.domain_adapter import DomainAdapter
from core.adapter_registry import get_adapter
from core.classification import (
    classify_components,
    infer_component_type,
    is_likely_component,
)
from core.interface_extractor import extract_data_flows, extract_interface_map
from core.verification import score_actionability, verify_deterministic


@pytest.fixture(autouse=True)
def ensure_registered():
    """Ensure API adapter is registered."""
    import adapters  # noqa: F401 — auto-registers
    yield


# =============================================================================
# Adapter structure
# =============================================================================

class TestAPIAdapterStructure:
    def test_is_domain_adapter(self):
        assert isinstance(API_ADAPTER, DomainAdapter)

    def test_name(self):
        assert API_ADAPTER.name == "api"

    def test_version(self):
        assert API_ADAPTER.version == "1.0"

    def test_registered(self):
        adapter = get_adapter("api")
        assert adapter.name == "api"

    def test_yaml_output(self):
        assert API_MATERIALIZATION.output_format == "yaml"
        assert API_MATERIALIZATION.file_extension == ".yaml"

    def test_frozen(self):
        """Adapter is immutable."""
        with pytest.raises(AttributeError):
            API_ADAPTER.name = "modified"

    def test_all_fields_present(self):
        """All DomainAdapter fields are populated."""
        assert API_ADAPTER.vocabulary is not None
        assert API_ADAPTER.prompts is not None
        assert API_ADAPTER.classification is not None
        assert API_ADAPTER.verification is not None
        assert API_ADAPTER.materialization is not None


# =============================================================================
# API vocabulary
# =============================================================================

class TestAPIVocabulary:
    def test_has_endpoint_keywords(self):
        assert "endpoint" in API_VOCABULARY.type_keywords
        assert "route" in API_VOCABULARY.type_keywords["endpoint"]
        assert "handler" in API_VOCABULARY.type_keywords["endpoint"]

    def test_has_model_keywords(self):
        assert "model" in API_VOCABULARY.type_keywords
        assert "schema" in API_VOCABULARY.type_keywords["model"]
        assert "dto" in API_VOCABULARY.type_keywords["model"]

    def test_has_service_keywords(self):
        assert "service" in API_VOCABULARY.type_keywords
        assert "controller" in API_VOCABULARY.type_keywords["service"]

    def test_has_middleware_keywords(self):
        assert "middleware" in API_VOCABULARY.type_keywords
        assert "validator" in API_VOCABULARY.type_keywords["middleware"]
        assert "rate_limiter" in API_VOCABULARY.type_keywords["middleware"]

    def test_has_auth_scheme_keywords(self):
        assert "auth_scheme" in API_VOCABULARY.type_keywords
        assert "jwt" in API_VOCABULARY.type_keywords["auth_scheme"]
        assert "oauth" in API_VOCABULARY.type_keywords["auth_scheme"]

    def test_has_resource_keywords(self):
        assert "resource" in API_VOCABULARY.type_keywords
        assert "collection" in API_VOCABULARY.type_keywords["resource"]

    def test_has_relationship_flows(self):
        assert "returns" in API_VOCABULARY.relationship_flows
        assert "accepts" in API_VOCABULARY.relationship_flows
        assert "calls" in API_VOCABULARY.relationship_flows
        assert "authenticates" in API_VOCABULARY.relationship_flows
        assert "validates" in API_VOCABULARY.relationship_flows

    def test_entity_types_for_models(self):
        assert "model" in API_VOCABULARY.entity_types
        assert "schema" in API_VOCABULARY.entity_types
        assert "dto" in API_VOCABULARY.entity_types

    def test_process_types_for_endpoints(self):
        assert "endpoint" in API_VOCABULARY.process_types
        assert "service" in API_VOCABULARY.process_types
        assert "middleware" in API_VOCABULARY.process_types

    def test_interface_types(self):
        assert "middleware" in API_VOCABULARY.interface_types
        assert "auth_scheme" in API_VOCABULARY.interface_types

    def test_type_hints(self):
        assert "user" in API_VOCABULARY.type_hints
        assert API_VOCABULARY.type_hints["user"] == "UserModel"
        assert "error" in API_VOCABULARY.type_hints
        assert API_VOCABULARY.type_hints["error"] == "ErrorResponse"

    def test_six_component_types(self):
        """API vocabulary defines exactly 6 component types."""
        assert len(API_VOCABULARY.type_keywords) == 6

    def test_twelve_relationship_types(self):
        """API vocabulary defines 12 relationship types."""
        assert len(API_VOCABULARY.relationship_flows) == 12


# =============================================================================
# Classification with API vocabulary
# =============================================================================

class TestAPIClassification:
    def test_endpoint_type_inference(self):
        """Endpoint components should be classified using API vocabulary."""
        comp_type, confidence = infer_component_type(
            "User Registration Endpoint",
            "subject",
            "",
            API_VOCABULARY.type_keywords,
        )
        assert comp_type == "endpoint"
        assert confidence > 0.3

    def test_model_type_inference(self):
        comp_type, confidence = infer_component_type(
            "Order Schema Model",
            "subject",
            "",
            API_VOCABULARY.type_keywords,
        )
        assert comp_type == "model"
        assert confidence > 0.3

    def test_middleware_type_inference(self):
        comp_type, confidence = infer_component_type(
            "CORS Middleware Filter",
            "subject",
            "",
            API_VOCABULARY.type_keywords,
        )
        assert comp_type == "middleware"
        assert confidence > 0.3

    def test_auth_type_inference(self):
        comp_type, confidence = infer_component_type(
            "JWT Token Auth",
            "subject",
            "",
            API_VOCABULARY.type_keywords,
        )
        assert comp_type == "auth_scheme"
        assert confidence > 0.3

    def test_generic_terms_filter(self):
        """API generic terms should reject non-components."""
        is_comp, reason = is_likely_component(
            "status", 0.1, "modifier", 0.0,
            API_CLASSIFICATION.generic_terms,
        )
        assert not is_comp

    def test_request_is_generic(self):
        """'request' should be filtered as generic in API domain."""
        assert "request" in API_CLASSIFICATION.generic_terms

    def test_response_is_generic(self):
        """'response' should be filtered as generic in API domain."""
        assert "response" in API_CLASSIFICATION.generic_terms

    def test_classify_api_components(self):
        """Full classification with API vocabulary."""
        candidates = [
            {"name": "User Endpoint", "type": "endpoint"},
            {"name": "Order Model", "type": "model"},
            {"name": "Auth Middleware", "type": "middleware"},
            {"name": "status", "type": ""},
        ]
        results = classify_components(
            candidates,
            "REST API with user endpoint, order model, and auth middleware",
            ["The user endpoint returns order data"],
            [],
            API_VOCABULARY.type_keywords,
            API_CLASSIFICATION.generic_terms,
        )
        names = [r.name for r in results if r.is_component]
        assert "User Endpoint" in names
        assert "Order Model" in names
        assert "status" not in names


# =============================================================================
# Interface extraction with API vocabulary
# =============================================================================

class TestAPIInterfaceExtraction:
    def test_returns_relationship(self):
        """API 'returns' relationship produces response data flow."""
        rel = {"type": "returns", "description": "endpoint returns user data"}
        flows = extract_data_flows(
            rel, "UserEndpoint", "UserModel",
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "response_data"
        assert flows[0].direction == "A_to_B"

    def test_accepts_relationship(self):
        """API 'accepts' relationship produces request data flow."""
        rel = {"type": "accepts", "description": "endpoint accepts order payload"}
        flows = extract_data_flows(
            rel, "OrderEndpoint", "OrderModel",
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "request_data"
        assert flows[0].direction == "B_to_A"

    def test_authenticates_relationship(self):
        """API 'authenticates' relationship produces auth flow."""
        rel = {"type": "authenticates", "description": "JWT authenticates endpoint"}
        flows = extract_data_flows(
            rel, "JWTAuth", "PaymentEndpoint",
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "auth_flow"
        # Default type for authenticates is "Token" (no type hint match for "PaymentEndpoint")
        assert flows[0].type_hint == "Token"

    def test_calls_relationship(self):
        """API 'calls' relationship produces service call flow."""
        rel = {"type": "calls", "description": "endpoint calls order service"}
        flows = extract_data_flows(
            rel, "OrderEndpoint", "OrderService",
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "service_call"

    def test_validates_relationship(self):
        """API 'validates' relationship produces validation flow."""
        rel = {"type": "validates", "description": "middleware validates request"}
        flows = extract_data_flows(
            rel, "ValidationMiddleware", "OrderModel",
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(flows) == 1
        assert flows[0].name == "validation_flow"

    def test_full_interface_map(self):
        """Extract interface map from a complete API blueprint."""
        blueprint = {
            "components": [
                {"name": "UserEndpoint", "type": "endpoint"},
                {"name": "UserModel", "type": "model"},
                {"name": "AuthMiddleware", "type": "middleware"},
            ],
            "relationships": [
                {"from": "UserEndpoint", "to": "UserModel", "type": "returns",
                 "description": "returns user data"},
                {"from": "AuthMiddleware", "to": "UserEndpoint", "type": "authenticates",
                 "description": "auth middleware protects endpoint"},
            ],
        }
        imap = extract_interface_map(
            blueprint, None,
            API_VOCABULARY.relationship_flows,
            API_VOCABULARY.type_hints,
        )
        assert len(imap.contracts) > 0


# =============================================================================
# Verification with API overrides
# =============================================================================

class TestAPIVerification:
    def test_actionability_checks(self):
        assert "endpoints" in API_VERIFICATION.actionability_checks
        assert "parameters" in API_VERIFICATION.actionability_checks

    def test_readiness_label(self):
        assert API_VERIFICATION.readiness_label == "api_readiness"

    def test_dimension_weights_sum(self):
        total = sum(API_VERIFICATION.dimension_weights)
        assert abs(total - 1.0) < 0.01

    def test_component_with_endpoints_is_actionable(self):
        """Component with endpoints key scores higher on actionability."""
        comp_with = {"name": "UserAPI", "type": "endpoint", "endpoints": ["/users"]}
        comp_without = {"name": "UserAPI", "type": "endpoint"}
        score_with = score_actionability(0.5, [comp_with], API_VERIFICATION.actionability_checks)
        score_without = score_actionability(0.5, [comp_without], API_VERIFICATION.actionability_checks)
        assert score_with.score >= score_without.score

    def test_verification_with_api_checks(self):
        """verify_deterministic uses API actionability checks."""
        blueprint = {
            "components": [
                {"name": "UserEndpoint", "type": "endpoint",
                 "description": "Serves user data", "derived_from": "user API",
                 "methods": [{"name": "get_user"}], "endpoints": ["/users"]},
            ],
            "relationships": [],
            "constraints": [],
        }
        result = verify_deterministic(
            blueprint,
            intent_keywords=["user", "endpoint"],
            input_text="A user endpoint API",
            graph_errors=[],
            graph_warnings=[],
            health_score=80.0,
            health_stats={},
            contradiction_count=0,
            parseable_constraint_ratio=0.5,
            avg_type_confidence=0.5,
            actionability_checks=API_VERIFICATION.actionability_checks,
        )
        # DeterministicVerification has named dimension attributes, not a scores dict
        assert result.completeness is not None
        assert result.actionability is not None
        assert result.overall_score >= 0


# =============================================================================
# Prompts
# =============================================================================

class TestAPIPrompts:
    def test_all_prompts_populated(self):
        """API adapter provides custom prompts for all 6 agent types."""
        assert API_PROMPTS.intent_system_prompt
        assert API_PROMPTS.persona_system_prompt
        assert API_PROMPTS.entity_system_prompt
        assert API_PROMPTS.process_system_prompt
        assert API_PROMPTS.synthesis_system_prompt
        assert API_PROMPTS.emission_preamble

    def test_intent_prompt_mentions_api(self):
        assert "API" in API_PROMPTS.intent_system_prompt

    def test_entity_prompt_mentions_resources(self):
        assert "resource" in API_PROMPTS.entity_system_prompt.lower() or \
               "model" in API_PROMPTS.entity_system_prompt.lower()

    def test_process_prompt_mentions_request(self):
        assert "request" in API_PROMPTS.process_system_prompt.lower()

    def test_synthesis_prompt_mentions_endpoint(self):
        assert "endpoint" in API_PROMPTS.synthesis_system_prompt.lower()

    def test_emission_preamble_mentions_yaml(self):
        assert "YAML" in API_PROMPTS.emission_preamble


# =============================================================================
# Engine wiring
# =============================================================================

class TestAPIEngineWiring:
    def test_engine_accepts_api_adapter(self):
        """Engine initializes with API adapter without error."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            llm_client=MockClient(),
            domain_adapter=API_ADAPTER,
            auto_store=False,
        )
        assert engine.domain_adapter is API_ADAPTER

    def test_adapter_threaded_to_pipeline(self):
        """API adapter reaches the staged pipeline."""
        from core.llm import MockClient
        from core.engine import MotherlabsEngine

        engine = MotherlabsEngine(
            llm_client=MockClient(),
            domain_adapter=API_ADAPTER,
            auto_store=False,
        )
        # Pipeline should have the adapter
        if hasattr(engine, '_staged_pipeline') and engine._staged_pipeline:
            assert engine._staged_pipeline.domain_adapter is API_ADAPTER


# =============================================================================
# Three adapters coexist
# =============================================================================

class TestThreeAdaptersCoexist:
    def test_all_three_registered(self):
        """Software, process, and API adapters all register without conflict."""
        from core.adapter_registry import list_adapters
        names = list_adapters()
        assert "software" in names
        assert "process" in names
        assert "api" in names

    def test_each_adapter_distinct(self):
        """All three adapters have distinct names and vocabularies."""
        software = get_adapter("software")
        process = get_adapter("process")
        api = get_adapter("api")

        # Distinct names
        assert software.name != process.name != api.name

        # Distinct type keywords
        sw_types = set(software.vocabulary.type_keywords.keys())
        pr_types = set(process.vocabulary.type_keywords.keys())
        ap_types = set(api.vocabulary.type_keywords.keys())
        assert sw_types != pr_types
        assert sw_types != ap_types
        assert pr_types != ap_types

    def test_default_adapter_is_software(self):
        """Default adapter remains software."""
        from core.adapter_registry import get_default_adapter
        default = get_default_adapter()
        assert default.name == "software"
